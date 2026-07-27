"""分层回测的单组 long-only 变体：只买某一个分位组，每日调仓。

与 ``QuantileLongShortStrategy`` 的分组口径**逐字一致**
（``backtest.py:228-271``）::

    rank = 分数.rank("ordinal", descending=False)          # 升序，1=最低分
    group = (rank - 1) * n_groups // rank.max()            # 0..n_groups-1
    # group 0   = 分数最低组
    # group n-1 = 分数最高组

差别只有两点：不建空头腿，且只持有指定的那一组（组内等权，Σw=1）。

``--group`` 用**人类直觉编号**：``1`` = 分数最高组，``n_groups`` = 分数最低组。
内部映射 ``code_group = n_groups - group``。

调仓：每个交易日重排分位、重建持仓（分层回测原生口径）。
执行：t 日信号 → t+1 开盘，含涨跌停/ST/停牌/T+1 约束，``max_participation_rate=1.0``。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from factorzen.core.experiment import get_git_sha
from factorzen.daily.evaluation.backtest import (
    BacktestConfig,
    PrecomputedWeightsStrategy,
    run_strategy_backtest,
    trim_backtest_to_first_trade,
)
from factorzen.pipelines.combine_backtest import (
    _metrics_from_result,
    build_cost_model_from_bps,
    load_market_panel,
)
from factorzen.pipelines.daily_single import filter_frame_by_membership

SCORES = (
    "workspace/factors/_runs/20260721_library_eqw_alla_2016_2026/combined.parquet"
)
OUT_ROOT = Path("workspace/combine_backtests/quantile_group")


def build_group_weights(
    scores: pl.DataFrame, *, score_col: str, n_groups: int, group: int
) -> dict:
    """按分层回测的分组公式取指定组，组内等权。group=1 为分数最高组。"""
    code_group = n_groups - group  # 1 → n_groups-1（最高组）；n_groups → 0（最低组）
    valid = scores.filter(
        pl.col(score_col).is_not_null() & pl.col(score_col).is_finite()
    )
    grouped = (
        valid.with_columns(
            pl.col(score_col).rank("ordinal", descending=False).over("trade_date").alias("_rk")
        )
        .with_columns(pl.col("_rk").max().over("trade_date").alias("_rkmax"))
        # 与引擎一致：整数除法分组；截面股数 < n_groups 时该日不建仓
        .filter(pl.col("_rkmax") >= n_groups)
        .with_columns(
            (((pl.col("_rk") - 1) * n_groups) // pl.col("_rkmax")).cast(pl.Int32).alias("_g")
        )
        .filter(pl.col("_g") == code_group)
        .select(["trade_date", "ts_code"])
    )
    sizes = grouped.group_by("trade_date").agg(pl.len().alias("_n"))
    weighted = grouped.join(sizes, on="trade_date", how="left").with_columns(
        (1.0 / pl.col("_n")).alias("target_weight")
    )
    return {
        (k[0] if isinstance(k, tuple) else k): g.select(["ts_code", "target_weight"])
        for k, g in weighted.partition_by("trade_date", as_dict=True).items()
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default=SCORES)
    ap.add_argument("--score-col", required=True, dest="score_col")
    ap.add_argument("--universe", default="all_a")
    ap.add_argument("--n-groups", type=int, default=5, dest="n_groups")
    ap.add_argument("--group", type=int, required=True,
                    help="1=分数最高组 … n_groups=分数最低组")
    ap.add_argument("--start", default="20160104")
    ap.add_argument("--end", default="20260720")
    ap.add_argument("--cost-bps", type=float, default=None, dest="cost_bps",
                    help="缺省=LinearCostModel 真实费率；0=零成本")
    ap.add_argument("--run-id", default=None, dest="run_id")
    args = ap.parse_args()

    if not 1 <= args.group <= args.n_groups:
        raise ValueError(f"--group 必须在 1..{args.n_groups}，收到 {args.group}")

    market = load_market_panel(
        start=args.start, end=args.end, universe=args.universe, market="ashare"
    )
    price_df = market["price_df"]
    dates = sorted(price_df.select("trade_date").unique()["trade_date"].to_list())

    scores = pl.read_parquet(args.scores, columns=["trade_date", "ts_code", args.score_col])
    scores = scores.filter(pl.col("trade_date").is_between(dates[0], dates[-1]))
    scores = filter_frame_by_membership(scores, market["membership"])
    if scores.is_empty():
        raise ValueError("PIT membership 过滤后分数截面为空")

    weights = build_group_weights(
        scores, score_col=args.score_col, n_groups=args.n_groups, group=args.group
    )
    sizes = [w.height for w in weights.values()]
    print(
        f"[q{args.group}/{args.n_groups}] 调仓日={len(weights)} "
        f"组内股数 中位={sorted(sizes)[len(sizes) // 2]} min={min(sizes)} max={max(sizes)}"
    )

    strategy = PrecomputedWeightsStrategy(weights)
    strategy.name = f"q{args.group}_of_{args.n_groups}_long_only"
    cfg = BacktestConfig(
        factor_col="factor_clean",
        frequency="daily",
        max_abs_weight=1.0,
        max_participation_rate=1.0,
        strategy_type=strategy.name,
        strategy_params={"n_groups": args.n_groups, "group": args.group},
        cost_model="linear",
    )
    result = run_strategy_backtest(
        strategy,
        scores.rename({args.score_col: "factor_clean"}),
        price_df,
        config=cfg,
        cost_model=build_cost_model_from_bps(args.cost_bps),
        factor_name=f"q{args.group}_{args.score_col}",
        is_st_by_date=market["is_st_by_date"],
    )
    result = trim_backtest_to_first_trade(result)
    metrics = _metrics_from_result(result)

    rid = args.run_id or f"{args.score_col}_{args.universe}_q{args.group}of{args.n_groups}"
    out = OUT_ROOT / rid
    out.mkdir(parents=True, exist_ok=True)
    result.nav.write_parquet(str(out / "nav.parquet"))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": rid,
                "git_sha": get_git_sha(),
                "strategy": strategy.name,
                "definition": (
                    f"分层回测分组口径（与 QuantileLongShortStrategy 一致），只买第 "
                    f"{args.group} 组（1=分数最高组，共 {args.n_groups} 组），"
                    f"组内等权 long-only，每日调仓"
                ),
                "scores": args.scores,
                "score_col": args.score_col,
                "universe": args.universe,
                "n_groups": args.n_groups,
                "group": args.group,
                "start": args.start,
                "end": args.end,
                "cost_bps": "linear_default" if args.cost_bps is None else args.cost_bps,
                "exec": "t 日信号 → t+1 开盘执行；max_participation_rate=1.0",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[q{args.group}/{args.n_groups}] {rid} → {out}\n"
        f"  年化={metrics['ann_ret']:.2%}  SR={metrics['sharpe']:.2f}  "
        f"最大回撤={metrics['max_dd']:.2%}  日均换手={metrics['avg_turnover']:.2%}  "
        f"期末净值={float(result.nav['nav'][-1]):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

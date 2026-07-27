"""滚动分层建仓策略回测：每日买入全 A 因子前 200 名，持有 10 个交易日。

策略定义
--------
- 每个交易日 t：按分数在**当日 PIT 全 A 成分**内取前 ``top_n`` 名（默认 200）。
- 每天投入 ``1/hold_days`` 的资金等权买入这批票（**每天都买**，不是每 10 天买一次）。
- 每层持有 ``hold_days`` 个交易日后卖出。
- 稳态下同时持有 10 层、总仓位 100%；一只票若连续多日入选，权重自然累加。

目标权重（signal_date = t）::

    w_t(i) = Σ_{j=0}^{H-1}  1{i ∈ Top_N(t-j)} / (H · N_{t-j})

其中 H=hold_days，N_{t-j} 为该层实际选中的股票数（不足 N 时该层内部仍等权满仓 1/H）。
前 H-1 天层数未满，仓位从 1/H 线性爬坡到 1，属预热，不特殊处理。

执行口径
--------
权重经 ``PrecomputedWeightsStrategy`` 交给 ``run_strategy_backtest`` 统一日环引擎：
t 日信号 → **t+1 开盘执行**，含涨跌停/ST/停牌/T+1 约束与 LinearCostModel 真实成本，
``max_participation_rate=1.0``（满仓成交，与 ``fz combine backtest`` 同口径）。

用法::

    PYTHONPATH=<master_src>/src pixi run -- python workspace/configs/sleeve_top200_h10.py \\
        --score-col eq_all --universe all_a --top-n 200 --hold-days 10
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
OUT_ROOT = Path("workspace/combine_backtests/sleeve")


def build_sleeve_weights(
    scores: pl.DataFrame,
    *,
    score_col: str,
    top_n: int,
    hold_days: int,
    trade_dates: list,
    direction: str = "top",
) -> dict:
    """构造滚动分层目标权重 dict[signal_date] -> DataFrame[ts_code, target_weight]。

    ``direction="bottom"`` 取分数**最低**的 top_n 只（反向对照）。
    """
    date_idx = {d: i for i, d in enumerate(trade_dates)}
    n_dates = len(trade_dates)

    # 每日 top_n：分数降序（bottom 则升序），ordinal 打散并列（避免并列被整组纳入）
    top = (
        scores.filter(pl.col(score_col).is_not_null() & pl.col(score_col).is_finite())
        .with_columns(
            pl.col(score_col)
            .rank("ordinal", descending=(direction == "top"))
            .over("trade_date")
            .alias("_rk")
        )
        .filter(pl.col("_rk") <= top_n)
        .select(["trade_date", "ts_code"])
    )
    # 每层内部等权满仓 1/H：层权重 = 1 / (H · 该层实际股票数)
    layer_size = top.group_by("trade_date").agg(pl.len().alias("_n"))
    top = top.join(layer_size, on="trade_date", how="left").with_columns(
        (1.0 / (hold_days * pl.col("_n"))).alias("_w")
    )
    top = top.with_columns(
        pl.col("trade_date").replace_strict(date_idx, return_dtype=pl.Int32).alias("_i")
    )

    # 层 d 贡献到 t ∈ [d, d+H-1]：cross join 偏移量后按目标日聚合
    offsets = pl.DataFrame({"_off": list(range(hold_days))}, schema={"_off": pl.Int32})
    spread = (
        top.join(offsets, how="cross")
        .with_columns((pl.col("_i") + pl.col("_off")).alias("_t"))
        .filter(pl.col("_t") < n_dates)
        .group_by(["_t", "ts_code"])
        .agg(pl.col("_w").sum().alias("target_weight"))
    )

    weights: dict = {}
    for (t_idx,), grp in spread.partition_by("_t", as_dict=True).items():
        weights[trade_dates[int(t_idx)]] = grp.select(["ts_code", "target_weight"])
    return weights


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default=SCORES, help="分数面板 parquet（默认库等权面板）")
    ap.add_argument("--score-col", required=True, dest="score_col")
    ap.add_argument("--universe", default="all_a")
    ap.add_argument("--top-n", type=int, default=200, dest="top_n")
    ap.add_argument("--hold-days", type=int, default=10, dest="hold_days")
    ap.add_argument("--direction", choices=["top", "bottom"], default="top",
                    help="top=选分数最高的 N 只；bottom=选最低的 N 只（反向对照）")
    ap.add_argument("--start", default="20160104")
    ap.add_argument("--end", default="20260720")
    ap.add_argument("--cost-bps", type=float, default=None, dest="cost_bps")
    ap.add_argument("--run-id", default=None, dest="run_id")
    args = ap.parse_args()

    market = load_market_panel(
        start=args.start, end=args.end, universe=args.universe, market="ashare"
    )
    price_df = market["price_df"]
    trade_dates = sorted(price_df.select("trade_date").unique()["trade_date"].to_list())

    scores = pl.read_parquet(args.scores, columns=["trade_date", "ts_code", args.score_col])
    scores = scores.filter(
        pl.col("trade_date").is_between(trade_dates[0], trade_dates[-1])
    )
    scores = filter_frame_by_membership(scores, market["membership"])
    if scores.is_empty():
        raise ValueError("PIT membership 过滤后分数截面为空")

    weights = build_sleeve_weights(
        scores,
        score_col=args.score_col,
        top_n=args.top_n,
        hold_days=args.hold_days,
        trade_dates=trade_dates,
        direction=args.direction,
    )
    print(f"[sleeve] 目标权重日数={len(weights)}  首日={trade_dates[0]}  末日={trade_dates[-1]}")

    strategy = PrecomputedWeightsStrategy(weights)
    strategy.name = f"sleeve_top{args.top_n}_h{args.hold_days}"
    cfg = BacktestConfig(
        factor_col="factor_clean",
        frequency="daily",
        max_abs_weight=1.0,
        max_participation_rate=1.0,
        strategy_type=strategy.name,
        strategy_params={"top_n": args.top_n, "hold_days": args.hold_days},
        cost_model="linear",
    )
    # Precomputed 走 has_target_for；factor_df 只需提供日期×股票骨架
    factor_df = scores.rename({args.score_col: "factor_clean"})
    result = run_strategy_backtest(
        strategy,
        factor_df,
        price_df,
        config=cfg,
        cost_model=build_cost_model_from_bps(args.cost_bps),
        factor_name=f"sleeve_{args.score_col}",
        is_st_by_date=market["is_st_by_date"],
    )
    result = trim_backtest_to_first_trade(result)
    metrics = _metrics_from_result(result)

    rid = args.run_id or f"{args.score_col}_{args.universe}_top{args.top_n}_h{args.hold_days}"
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
                    f"每日选 {args.universe} 因子"
                    f"{'前' if args.direction == 'top' else '后'} {args.top_n} 名，每天投入 "
                    f"1/{args.hold_days} 资金等权买入，持有 {args.hold_days} 个交易日"
                ),
                "scores": args.scores,
                "score_col": args.score_col,
                "universe": args.universe,
                "start": args.start,
                "end": args.end,
                "top_n": args.top_n,
                "direction": args.direction,
                "hold_days": args.hold_days,
                "cost_bps": "linear_default" if args.cost_bps is None else args.cost_bps,
                "exec": "t 日信号 → t+1 开盘执行；max_participation_rate=1.0",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[sleeve] {rid} → {out}\n"
        f"  年化={metrics['ann_ret']:.2%}  SR={metrics['sharpe']:.2f}  "
        f"最大回撤={metrics['max_dd']:.2%}  日均换手={metrics['avg_turnover']:.2%}  "
        f"累计成本={metrics['total_cost']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""``fz combine backtest``：组合 OOS 分数面板 → 日环策略回测桥。

输入二选一：
- ``--scores``：任意 (trade_date, ts_code, <分数列>) parquet
- ``--run-dir`` + ``--method``：读 combine 产物 ``oos_scores/<method>.parquet``

数据装配与 ``daily_single`` 同口径（PIT membership、复权日线、is_st_by_date、
LinearCostModel 默认成本）；策略只走 ``strategy_registry`` 既有类，不发明新策略。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.config.settings import COMBINE_BACKTESTS_DIR
from factorzen.core.experiment import build_manifest_base, get_git_sha
from factorzen.daily.evaluation.backtest import (
    BacktestConfig,
    CostModel,
    run_strategy_backtest,
    trim_backtest_to_first_trade,
)
from factorzen.daily.evaluation.cost_models import LinearCostModel
from factorzen.daily.evaluation.strategy_registry import build_strategy

# 与 ``build_default_daily_research_config`` / daily_single 无 YAML 默认主策略对齐
_DEFAULT_STRATEGY = "quantile_ls_5"

# 命名策略 → (registry type, params)；只登记现有类，不发明
_NAMED_STRATEGIES: dict[str, tuple[str, dict[str, Any]]] = {
    "quantile_ls_5": ("quantile_long_short", {"quantiles": 5}),
    "quantile_long_short": ("quantile_long_short", {"quantiles": 5}),
    "topn_long_only": ("topn_long_only", {"top_n": 50}),
    "topn_50": ("topn_long_only", {"top_n": 50}),
    "factor_weighted": ("factor_weighted", {}),
}

_NUMERIC_DTYPES = {
    pl.Float32,
    pl.Float64,
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
}


def load_scores_panel(
    path: str | Path,
    score_col: str | None = None,
) -> pl.DataFrame:
    """解析外部/内部分数 parquet → ``(trade_date, ts_code, factor_clean)``。

    畸形输入只抛 ``ValueError``（契约）。
    - ``score_col`` 缺省：取除 trade_date/ts_code 外**唯一**数值列；多列则必填。
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"分数 parquet 不存在: {p}")
    try:
        df = pl.read_parquet(p)
    except Exception as exc:
        raise ValueError(f"无法读取分数 parquet: {p} ({exc})") from exc

    if not isinstance(df, pl.DataFrame) or df.is_empty():
        raise ValueError(f"分数面板为空: {p}")

    cols = set(df.columns)
    if "trade_date" not in cols or "ts_code" not in cols:
        raise ValueError(
            f"分数面板必须含 trade_date、ts_code 列；实际列={list(df.columns)}"
        )

    if score_col is not None:
        if score_col not in df.columns:
            raise ValueError(
                f"指定 score-col={score_col!r} 不在面板中；可用列={list(df.columns)}"
            )
        chosen = score_col
    else:
        candidates = [
            c
            for c in df.columns
            if c not in ("trade_date", "ts_code") and df.schema[c] in _NUMERIC_DTYPES
        ]
        if not candidates:
            raise ValueError(
                f"除 trade_date/ts_code 外无数值分数列；列={list(df.columns)}"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"分数面板含多列数值 {candidates}，请用 --score-col 指定其一"
                f"（multiple score columns require --score-col）"
            )
        chosen = candidates[0]

    out = df.select(
        [
            pl.col("trade_date"),
            pl.col("ts_code").cast(pl.Utf8),
            pl.col(chosen).cast(pl.Float64).alias("factor_clean"),
        ]
    ).filter(pl.col("factor_clean").is_not_null() & pl.col("factor_clean").is_finite())
    if out.is_empty():
        raise ValueError(f"分数面板过滤后为空（列={chosen}）: {p}")
    return out


def resolve_scores_source(
    *,
    scores: str | Path | None,
    run_dir: str | Path | None,
    method: str,
    score_col: str | None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """二选一解析分数来源，返回 factor 面板 + scores_source 元数据。"""
    if scores is not None and run_dir is not None:
        raise ValueError("--scores 与 --run-dir 互斥，请只给一个")
    if scores is None and run_dir is None:
        raise ValueError("必须提供 --scores 或 --run-dir")

    if scores is not None:
        panel = load_scores_panel(scores, score_col=score_col)
        meta = {
            "type": "scores",
            "path": str(Path(scores).resolve()),
            "score_col": score_col,
            "method": None,
        }
        return panel, meta

    rd = Path(run_dir)  # type: ignore[arg-type]
    oos_path = rd / "oos_scores" / f"{method}.parquet"
    if not oos_path.exists():
        raise ValueError(
            f"combine 产物缺 oos_scores/{method}.parquet: {oos_path}。"
            f"请先跑 fz combine（会写 oos_scores/），或改用 --scores。"
        )
    panel = load_scores_panel(oos_path, score_col=score_col or "score")
    meta = {
        "type": "run_dir",
        "path": str(oos_path.resolve()),
        "run_dir": str(rd.resolve()),
        "method": method,
        "score_col": score_col or "score",
    }
    return panel, meta


def resolve_strategy(name: str, *, factor_col: str = "factor_clean"):
    """按名构建既有策略类（registry / 命名别名），绝不发明新类。"""
    key = name.strip()
    if key in _NAMED_STRATEGIES:
        type_name, params = _NAMED_STRATEGIES[key]
        params = {**params, "factor_col": factor_col}
        strategy = build_strategy(type_name, params)
        strategy.name = key
        return strategy
    # 直接走 registry 类型名（如 optimizer_strategy）
    try:
        strategy = build_strategy(key, {"factor_col": factor_col})
    except Exception as exc:
        known = [
            *sorted(_NAMED_STRATEGIES),
            "quantile_long_short",
            "topn_long_only",
            "factor_weighted",
            "optimizer_strategy",
        ]
        raise ValueError(
            f"未知策略 {name!r}；支持命名别名/registry 类型: {sorted(set(known))}"
        ) from exc
    strategy.name = key
    return strategy


def build_cost_model_from_bps(cost_bps: float | None):
    """成本模型。

    - ``None``：与 ``daily_single`` 一致，``LinearCostModel()`` 默认费率。
    - 数值：单边成本 = cost_bps/1e4（commission；印花税/滑点/融券置 0），便于
      ``--cost-bps 0`` 零成本对照与成本传导数值验证。
    """
    if cost_bps is None:
        return LinearCostModel()
    if cost_bps < 0:
        raise ValueError(f"--cost-bps 不能为负，收到 {cost_bps}")
    rate = float(cost_bps) / 1e4
    return CostModel(
        commission=rate,
        stamp_tax=0.0,
        slippage=0.0,
        borrow_annual=0.0,
    )


def load_market_panel(
    *,
    start: str,
    end: str,
    universe: str,
    market: str = "ashare",
) -> dict[str, Any]:
    """装配回测行情：照抄 daily_single 的 membership + 日线（含复权列）+ ST 表。

    可被测试 monkeypatch，避免触碰 workspace/ 真库。
    """
    if market != "ashare":
        raise ValueError(
            f"combine backtest 当前仅支持 market=ashare，收到 {market!r}"
        )

    from factorzen.core.universe import build_is_st_by_date
    from factorzen.daily.data.context import FactorDataContext
    from factorzen.pipelines.daily_single import (
        filter_frame_by_membership,
        load_pit_membership,
    )

    membership, ts_codes, _universe_meta = load_pit_membership(start, end, universe)
    if not ts_codes and universe != "all_a":
        raise ValueError(f"股票池为空: universe={universe} [{start},{end}]")

    ctx = FactorDataContext(
        start=start,
        end=end,
        required_data=["daily"],
        lookback_days=5,
        universe=ts_codes if ts_codes else None,
        snapshot_mode="daily",
    )
    price_df = ctx.daily.collect()
    if price_df.is_empty():
        raise ValueError(f"日线行情为空: [{start},{end}] universe={universe}")

    price_df = filter_frame_by_membership(price_df, membership)
    if price_df.is_empty():
        raise ValueError("PIT membership 过滤后行情为空")

    codes = price_df.select("ts_code").unique()["ts_code"].to_list()
    trade_dates_list = price_df.select("trade_date").unique()["trade_date"].to_list()
    # trade_date 可能是 Date / Utf8；build_is_st_by_date 期望 date 或可解析串
    is_st_by_date = build_is_st_by_date(codes, trade_dates_list)
    return {
        "price_df": price_df,
        "membership": membership,
        "is_st_by_date": is_st_by_date,
        "ts_codes": codes,
    }


def _filter_factor_to_window(
    factor_df: pl.DataFrame,
    start: str,
    end: str,
) -> pl.DataFrame:
    """按 [start, end] 裁分数面板（trade_date 兼容 Date/Utf8）。"""
    td = factor_df["trade_date"]
    if td.dtype == pl.Date:
        s = datetime.strptime(start, "%Y%m%d").date()
        e = datetime.strptime(end, "%Y%m%d").date()
        return factor_df.filter(
            (pl.col("trade_date") >= s) & (pl.col("trade_date") <= e)
        )
    # Utf8 YYYYMMDD / YYYY-MM-DD
    td_str = pl.col("trade_date").cast(pl.Utf8).str.replace_all("-", "")
    return factor_df.filter((td_str >= start) & (td_str <= end))


def apply_rebalance_hold(
    factor_df: pl.DataFrame,
    rebalance_days: int,
    *,
    factor_col: str = "factor_clean",
) -> pl.DataFrame:
    """桥层实现 k 日调仓：分数面板降采样 + 按股票前向填充（不动引擎）。

    - ``rebalance_days <= 1``：原样返回（逐日）。
    - ``rebalance_days = k > 1``：在分数面板的交易日序列上取 ``dates[::k]`` 为调仓日，
      仅保留调仓日截面分数，再对每只股票 ``forward_fill`` 到全部交易日。
      非调仓日分数不变 → quantile/topN 目标权重不变 → 引擎日环换手≈0，等效 k 天调仓。
      行情仍逐日进入引擎，净值仍逐日更新。
    """
    if rebalance_days <= 1:
        return factor_df
    if factor_col not in factor_df.columns:
        raise ValueError(f"apply_rebalance_hold 缺列 {factor_col!r}")
    if factor_df.is_empty():
        return factor_df

    work = factor_df.sort(["ts_code", "trade_date"])
    all_dates = (
        work.select("trade_date").unique().sort("trade_date")["trade_date"].to_list()
    )
    if len(all_dates) < 2:
        return work

    rebalance_dates = all_dates[::rebalance_days]
    # 非调仓日分数置空，再按股票前向填充（保留原 (date, code) 网格，不发明新键）
    out = (
        work.with_columns(
            pl.when(pl.col("trade_date").is_in(rebalance_dates))
            .then(pl.col(factor_col))
            .otherwise(None)
            .alias(factor_col)
        )
        .sort(["ts_code", "trade_date"])
        .with_columns(pl.col(factor_col).forward_fill().over("ts_code"))
        .filter(pl.col(factor_col).is_not_null() & pl.col(factor_col).is_finite())
    )
    if out.is_empty():
        raise ValueError(
            f"rebalance_days={rebalance_days} 前向填充后分数面板为空"
            f"（调仓日数={len(rebalance_dates)}）"
        )
    return out


def _metrics_from_result(result) -> dict[str, float]:
    stats = result.summary_stats.get("long_short") or result.summary_stats.get(
        "portfolio", {}
    )
    total_cost = float(stats.get("total_cost", 0.0) or 0.0)
    # 成本占比：累计成本 / (1 + 累计毛收益近似)；用 |ann_ret| 不可比，改用 cost/nav 路径
    nav_end = 1.0
    if not result.nav.is_empty() and "nav" in result.nav.columns:
        nav_end = float(result.nav.sort("trade_date")["nav"][-1])
    cost_share = float(total_cost / nav_end) if nav_end > 1e-12 else 0.0
    return {
        "ann_ret": float(stats.get("ann_ret", 0.0) or 0.0),
        "sharpe": float(stats.get("sharpe", 0.0) or 0.0),
        "max_dd": float(stats.get("max_dd", 0.0) or 0.0),
        "avg_turnover": float(stats.get("avg_turnover", 0.0) or 0.0),
        "ann_turnover": float(stats.get("ann_turnover", 0.0) or 0.0),
        "total_cost": total_cost,
        "cost_share": cost_share,
    }


def _default_run_id() -> str:
    return datetime.now().strftime("combine_bt_%Y%m%d_%H%M%S")


def run_combine_backtest(
    *,
    scores: str | Path | None = None,
    run_dir: str | Path | None = None,
    method: str = "equal_weight",
    score_col: str | None = None,
    strategy: str = _DEFAULT_STRATEGY,
    start: str,
    end: str,
    universe: str = "csi300",
    market: str = "ashare",
    cost_bps: float | None = None,
    rebalance_days: int | None = None,
    out_dir: str | Path = COMBINE_BACKTESTS_DIR,
    run_id: str | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """分数面板 → 策略 → ``run_strategy_backtest`` → 落盘 manifest/metrics/nav。

    ``rebalance_days``（桥层，不动引擎）：
    - ``None`` 或 ``1``：逐日用当日分数生成目标权重。
    - ``k > 1``：分数面板降采样到每 k 个交易日 + 按股票前向填充，使非调仓日
      目标权重不变（换手≈0），行情仍逐日驱动净值。
    """
    if rebalance_days is not None and rebalance_days < 1:
        raise ValueError(f"--rebalance-days 必须 ≥ 1，收到 {rebalance_days}")

    factor_df, scores_source = resolve_scores_source(
        scores=scores, run_dir=run_dir, method=method, score_col=score_col,
    )
    factor_df = _filter_factor_to_window(factor_df, start, end)
    if factor_df.is_empty():
        raise ValueError(f"分数面板在 [{start},{end}] 内为空")

    market_data = load_market_panel(
        start=start, end=end, universe=universe, market=market,
    )
    price_df = market_data["price_df"]
    membership = market_data["membership"]
    is_st_by_date = market_data["is_st_by_date"]

    # 与 daily_single 一致：评估截面按 PIT membership 过滤
    from factorzen.pipelines.daily_single import filter_frame_by_membership

    factor_df = filter_frame_by_membership(factor_df, membership)
    if factor_df.is_empty():
        raise ValueError("PIT membership 过滤后分数截面为空")

    # 桥层 k 日调仓：分数 hold，引擎仍日环（净值逐日更新）
    rb = 1 if rebalance_days is None else int(rebalance_days)
    factor_df = apply_rebalance_hold(factor_df, rb, factor_col="factor_clean")

    strat = resolve_strategy(strategy, factor_col="factor_clean")
    cost_model = build_cost_model_from_bps(cost_bps)
    # max_abs_weight=1.0：与引擎默认一致。分位/TopN 单票权重=1/n_bucket，
    # 小票池时 0.1 会误杀合法等权（daily_single 的 0.1 面向 CSI 大票池单票风控）。
    # max_participation_rate=1.0：满仓成交，否则 0.05 参与率会在「分数 hold」的非调仓日
    # 仍因未完成成交而持续换手，破坏 rebalance_days 的「权重不变→换手≈0」语义。
    cfg = BacktestConfig(
        factor_col="factor_clean",
        frequency="daily",
        max_abs_weight=1.0,
        max_participation_rate=1.0,
        strategy_type=getattr(strat, "name", strategy),
        strategy_params={},
        cost_model="linear",
    )

    result = run_strategy_backtest(
        strat,
        factor_df,
        price_df,
        config=cfg,
        cost_model=cost_model,
        factor_name=f"combine_{method}" if scores_source.get("method") else "combine_scores",
        is_st_by_date=is_st_by_date,
    )
    result = trim_backtest_to_first_trade(result)
    metrics = _metrics_from_result(result)

    rid = run_id or _default_run_id()
    out_path = Path(out_dir) / rid
    out_path.mkdir(parents=True, exist_ok=True)

    # cost_bps 在 manifest 中记显式值；None 表示 LinearCostModel 默认
    cost_bps_recorded: float | str
    if cost_bps is None:
        cost_bps_recorded = "linear_default"
    else:
        cost_bps_recorded = float(cost_bps)

    cmd = list(command or ["combine", "backtest"])
    manifest = build_manifest_base(cmd, {
        "start": start,
        "end": end,
        "universe": universe,
        "market": market,
        "strategy": strategy,
        "method": method if scores_source.get("type") == "run_dir" else None,
        "cost_bps": cost_bps_recorded,
        "rebalance_days": rebalance_days,
        "score_col": score_col,
    })
    # 铁律字段提到顶层，方便审计扫一眼
    manifest.update(
        {
            "run_id": rid,
            "git_sha": get_git_sha(),
            "start": start,
            "end": end,
            "universe": universe,
            "market": market,
            "strategy": strategy,
            "cost_bps": cost_bps_recorded,
            "rebalance_days": rebalance_days,
            "scores_source": scores_source,
            "command": cmd,
            "note_rebalance": (
                "桥层实现：rebalance_days=k>1 时分数面板降采样到每 k 个交易日并按股票"
                "前向填充；引擎仍日环，非调仓日目标权重不变、净值仍逐日更新。"
                "k=1 或 None 为逐日。"
            ),
        }
    )

    (out_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_path / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result.nav.write_parquet(out_path / "nav.parquet")

    return {
        "run_dir": str(out_path),
        "run_id": rid,
        "metrics": metrics,
        "manifest": manifest,
        "nav": result.nav,
        "result": result,
    }


def print_backtest_summary(res: dict[str, Any]) -> None:
    m = res["metrics"]
    print(f"[combine backtest] 完成 → {res['run_dir']}")
    print(
        f"  年化={m['ann_ret']:.2%}  SR={m['sharpe']:.2f}  "
        f"最大回撤={m['max_dd']:.2%}  日均换手={m['avg_turnover']:.2%}  "
        f"累计成本={m['total_cost']:.4f}  成本占比≈{m['cost_share']:.2%}"
    )


def cmd_combine_backtest(args: Any) -> int:
    """CLI handler：ValueError → stderr + exit 2。"""
    try:
        res = run_combine_backtest(
            scores=getattr(args, "scores", None),
            run_dir=getattr(args, "run_dir", None),
            method=getattr(args, "method", "equal_weight"),
            score_col=getattr(args, "score_col", None),
            strategy=getattr(args, "strategy", _DEFAULT_STRATEGY),
            start=args.start,
            end=args.end,
            universe=getattr(args, "universe", "csi300"),
            market=getattr(args, "market", "ashare"),
            cost_bps=getattr(args, "cost_bps", None),
            rebalance_days=getattr(args, "rebalance_days", None),
            out_dir=getattr(args, "out_dir", COMBINE_BACKTESTS_DIR),
            run_id=getattr(args, "run_id", None),
            command=["combine", "backtest"],
        )
    except ValueError as exc:
        print(f"[combine backtest] {exc}", file=sys.stderr)
        return 2
    print_backtest_summary(res)
    return 0


__all__ = [
    "apply_rebalance_hold",
    "build_cost_model_from_bps",
    "cmd_combine_backtest",
    "load_market_panel",
    "load_scores_panel",
    "print_backtest_summary",
    "resolve_scores_source",
    "resolve_strategy",
    "run_combine_backtest",
]

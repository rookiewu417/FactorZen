"""模拟交易闭环：M4 目标组合 → backtest 回测 → 净值/绩效落盘。

真实依赖签名（以 backtest.py 代码为准）：

    run_strategy_backtest(
        strategy: Strategy,
        factor_df: pl.DataFrame,    # 第 2 参数，非 price_df！
        price_df: pl.DataFrame,
        config: BacktestConfig | None = None,
        cost_model: CostModel | CostModelBase | None = None,
        factor_name: str = "",
        *,
        collect_positions: bool = True,
        collect_trades: bool = True,
        include_context_positions: bool = True,
    ) -> StrategyBacktestResult

    StrategyBacktestResult.summary_stats:
        dict[int | str, dict[str, float]]
        keys: "portfolio" / "long_short"，每个子 dict 含:
            ann_ret, ann_vol, sharpe, max_dd, avg_turnover, total_cost, ann_turnover

    StrategyBacktestResult.nav:
        pl.DataFrame, columns: trade_date, gross_return, cost, borrow_cost, net_return, nav, cash_weight
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import date
from pathlib import Path

import polars as pl

from factorzen.daily.evaluation.backtest import (
    PrecomputedWeightsStrategy,
    run_strategy_backtest,
)

_logger = logging.getLogger(__name__)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _load_weights_by_date(
    portfolio_run_dirs: list[str],
) -> dict[date, pl.DataFrame]:
    """各 run_dir 的 weights.parquet + manifest.json → {signal_date: DataFrame[ts_code, target_weight]}。"""
    out: dict[date, pl.DataFrame] = {}
    for rd in portfolio_run_dirs:
        rd_p = Path(rd)
        manifest = json.loads((rd_p / "manifest.json").read_text())
        sig = manifest.get("signal_date")
        if sig is None:
            continue
        sig_date = date.fromisoformat(str(sig))
        w = pl.read_parquet(rd_p / "weights.parquet").select(
            ["ts_code", "target_weight"]
        )
        out[sig_date] = w
    return out


def _build_dummy_factor_df(weights_by_date: dict[date, pl.DataFrame]) -> pl.DataFrame:
    """构造最简 factor_df（每个信号日取第一只股票 + 哑元值），仅用于通过 _prepare_factor_df 校验。

    PrecomputedWeightsStrategy 走快速路径时 factor 数据不被访问，
    但 run_strategy_backtest 仍会对 factor_df 做列校验，故须提供合法 DataFrame。
    """
    rows = []
    for sig_date, weight_df in weights_by_date.items():
        if weight_df.height > 0:
            rows.append(
                {
                    "trade_date": sig_date,
                    "ts_code": weight_df["ts_code"][0],
                    "factor_clean": 0.0,
                }
            )
    if not rows:
        # 极端兜底
        rows = [
            {
                "trade_date": next(iter(weights_by_date)),
                "ts_code": "000001.SZ",
                "factor_clean": 0.0,
            }
        ]
    return pl.DataFrame(rows)


def run_portfolio_simulation(
    portfolio_run_dirs: list[str],
    daily: pl.DataFrame,
    *,
    out_dir: str = "workspace/sim",
    run_id: str | None = None,
) -> dict:
    """把 M4 目标组合喂给回测引擎，落盘净值与绩效指标。

    Args:
        portfolio_run_dirs: 各 run_dir 路径列表，每个目录含
            ``weights.parquet``（列 ts_code, target_weight）和
            ``manifest.json``（含 signal_date）。
        daily: 日线行情 DataFrame（列 trade_date, ts_code, open, close, …）。
        out_dir: 输出根目录。
        run_id: 本次模拟 ID；None 时默认 "sim"。

    Returns:
        dict 含 run_dir, sharpe, max_dd, ann_ret。
        同时落盘 <out_dir>/<run_id>/nav.parquet / metrics.json / manifest.json。
    """
    weights_by_date = _load_weights_by_date(portfolio_run_dirs)
    if not weights_by_date:
        raise ValueError("no portfolio weights with signal_date found in any run_dir")

    # PrecomputedWeightsStrategy 走快速路径：
    #   collect_positions=False, collect_trades=False, include_context_positions=False
    # factor_df 仍须满足列校验，但实际值不影响回测结果。
    factor_df = _build_dummy_factor_df(weights_by_date)
    strategy = PrecomputedWeightsStrategy(weights_by_date)

    bt = run_strategy_backtest(
        strategy,
        factor_df,
        daily,
        collect_positions=False,
        collect_trades=False,
        include_context_positions=False,
    )

    # 若 signal_date 晚于或等于回测末日（其后无执行日），权重永不生效 → 净值为空。
    if bt.nav.is_empty():
        max_trade_date = (
            daily.select("trade_date").max()["trade_date"][0]
            if not daily.is_empty()
            else None
        )
        _logger.warning(
            "signal_date 晚于或等于回测末日，未产生任何调仓"
            "（最早信号日=%s, 回测末日=%s）",
            min(weights_by_date, default=None),
            max_trade_date,
        )

    # summary_stats 结构：{"portfolio": {ann_ret, ann_vol, sharpe, max_dd, ...}, "long_short": ...}
    portfolio_stats: dict[str, float] = bt.summary_stats.get(
        "portfolio", bt.summary_stats.get("long_short", {})
    )
    metrics: dict[str, float] = dict(portfolio_stats)

    rid = run_id or "sim"
    run_dir = Path(out_dir) / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    bt.nav.write_parquet(run_dir / "nav.parquet")
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": rid,
                "n_signals": len(weights_by_date),
                "git_sha": _git_sha(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return {
        "run_dir": str(run_dir),
        "sharpe": metrics.get("sharpe"),
        "max_dd": metrics.get("max_dd"),
        "ann_ret": metrics.get("ann_ret"),
    }

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
    CostModel,
    PrecomputedWeightsStrategy,
    run_strategy_backtest,
)
from factorzen.daily.evaluation.cost_models import CostModelBase

_logger = logging.getLogger(__name__)

# 组合优化成功状态（参考 portfolio/optimizer.py::OptimizeResult.status，来源于
# cvxpy Problem.status）。其余状态（infeasible / infeasible_inaccurate /
# unbounded / unbounded_inaccurate / solver_error 等）意味着 pipelines/
# portfolio_build.py 把全零持仓兜底写盘，不能当作有效信号执行。
_SUCCESS_OPT_STATUSES = frozenset({"optimal", "optimal_inaccurate"})


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
    """各 run_dir 的 weights.parquet + manifest.json → {signal_date: DataFrame[ts_code, target_weight]}。

    跳过 manifest.status 为非成功状态的 run_dir：组合优化 infeasible/unbounded/
    solver_error 时，pipelines/portfolio_build.py 会把全零持仓兜底写盘（见该
    模块 ``w = opt.weights if opt.weights is not None else np.zeros(...)``）。
    这只是为了让 weights.parquet 始终可写，并不代表"清仓"是真实信号，sim 不应
    把它当作有效持仓执行。manifest 完全没有 status 字段时（历史产物/旧版
    pipeline）视为有效，保持向后兼容。
    """
    out: dict[date, pl.DataFrame] = {}
    for rd in portfolio_run_dirs:
        rd_p = Path(rd)
        manifest = json.loads((rd_p / "manifest.json").read_text())
        sig = manifest.get("signal_date")
        if sig is None:
            continue
        status = manifest.get("status")
        if status is not None and status not in _SUCCESS_OPT_STATUSES:
            _logger.warning(
                "跳过 run_dir=%s（signal_date=%s）：组合优化 status=%r 非成功状态，"
                "可能是 infeasible/unbounded 兜底的全零持仓，不作为有效信号执行",
                rd,
                sig,
                status,
            )
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
    cost_model: CostModel | CostModelBase | None = None,
) -> dict:
    """把 M4 目标组合喂给回测引擎，落盘净值与绩效指标。

    Args:
        portfolio_run_dirs: 各 run_dir 路径列表，每个目录含
            ``weights.parquet``（列 ts_code, target_weight）和
            ``manifest.json``（含 signal_date，可选 status）。
        daily: 日线行情 DataFrame（列 trade_date, ts_code, open, close, …）。
        out_dir: 输出根目录。
        run_id: 本次模拟 ID；None 时默认 "sim"。
        cost_model: 交易成本模型；None（默认）时使用项目默认费率的
            ``CostModel()``（佣金+滑点+印花税），**不会**得到零成本回测——
            如确实需要零成本对照，显式传入
            ``CostModel(commission=0, stamp_tax=0, slippage=0)``。

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
    # cost_model 必须是 None 或 CostModel（dataclass）才能走快路径，见
    # backtest.py::_should_use_precomputed_weights_fast_path；传 CostModelBase
    # 子类（LinearCostModel/SquareRootImpactCostModel）会被退回慢路径。
    factor_df = _build_dummy_factor_df(weights_by_date)
    strategy = PrecomputedWeightsStrategy(weights_by_date)
    effective_cost_model: CostModel | CostModelBase = (
        cost_model if cost_model is not None else CostModel()
    )

    bt = run_strategy_backtest(
        strategy,
        factor_df,
        daily,
        cost_model=effective_cost_model,
        collect_positions=False,
        collect_trades=False,
        include_context_positions=False,
    )

    # 逐个 signal_date 检查是否落在实际回测执行的日期范围内：signal_date 晚于
    # 或等于回测末日（其后无执行日）时，该信号永不生效。此前只看整体
    # bt.nav.is_empty() 会漏掉"N 个 run_dir 中只有最新一次过期、其余历史信号
    # 仍正常执行"这一更常见的真实场景（如每日/每周 build 后立即 sim，但行情
    # 数据还没更新到位）——此时 nav 整体非空，但过期的那个信号被悄悄忽略，
    # 不会有任何告警。
    max_trade_date = (
        daily.select("trade_date").max()["trade_date"][0] if not daily.is_empty() else None
    )
    for sig_date in sorted(weights_by_date):
        if max_trade_date is None or sig_date >= max_trade_date:
            _logger.warning(
                "signal_date=%s 晚于或等于回测末日（回测末日=%s），未产生任何调仓，"
                "该信号未被消费",
                sig_date,
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

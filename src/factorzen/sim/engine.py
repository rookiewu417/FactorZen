"""模拟交易闭环：M4 目标组合 → backtest 回测 → 净值/绩效落盘。

真实依赖签名（以 backtest.py 代码为准）：

    run_strategy_backtest(
        strategy: Strategy,
        factor_df: pl.DataFrame | None,  # 第 2 参数，非 price_df！
        price_df: pl.DataFrame,
        config: BacktestConfig | None = None,
        cost_model: CostModel | CostModelBase | None = None,
        factor_name: str = "",
        *,
        collect_positions: bool = True,
        collect_trades: bool = True,
        include_context_positions: bool = True,
        is_st_by_date: dict[date, set[str]] | None = None,
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
from dataclasses import asdict, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.config.settings import SIM_DIR
from factorzen.core.experiment import get_git_sha
from factorzen.core.universe import build_is_st_by_date
from factorzen.daily.evaluation.backtest import (
    BacktestConfig,
    CostModel,
    PrecomputedWeightsStrategy,
    run_strategy_backtest,
)
from factorzen.daily.evaluation.cost_models import CostModelBase

_logger = logging.getLogger(__name__)


def _jsonable(obj: Any) -> Any:
    """把 cost_model / BacktestConfig 等转为 JSON 可序列化结构。

    date → isoformat；dataclass → asdict 递归；不可序列化对象 → type 名 + 公开属性或 str。
    """
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    # CostModelBase 非 dataclass 子类等：记 type + 公开属性
    if hasattr(obj, "__dict__"):
        out: dict[str, Any] = {"_type": type(obj).__name__}
        for k, v in vars(obj).items():
            if k.startswith("_"):
                continue
            try:
                out[k] = _jsonable(v)
            except (TypeError, ValueError):
                out[k] = str(v)
        return out
    return str(obj)

# 组合优化成功状态（参考 portfolio/optimizer.py::OptimizeResult.status，来源于
# cvxpy Problem.status，或捕获 SolverError/DCPError 时固定的 "error"）。其余
# 状态（infeasible / infeasible_inaccurate / unbounded / unbounded_inaccurate /
# error 等）意味着 pipelines/portfolio_build.py 把全零持仓兜底写盘，不能当作
# 有效信号执行。
_SUCCESS_OPT_STATUSES = frozenset({"optimal", "optimal_inaccurate"})


def _load_weights_by_date(
    portfolio_run_dirs: list[str],
) -> dict[date, pl.DataFrame]:
    """各 run_dir 的 weights.parquet + manifest.json → {signal_date: DataFrame[ts_code, target_weight]}。

    跳过 manifest.status 为非成功状态的 run_dir：组合优化 infeasible/unbounded/
    error 时，pipelines/portfolio_build.py 会把全零持仓兜底写盘（见该模块
    ``w = opt.weights if opt.weights is not None else np.zeros(...)``）。
    这只是为了让 weights.parquet 始终可写，并不代表"清仓"是真实信号，sim 不应
    把它当作有效持仓执行。manifest 完全没有 status 字段时（历史产物/旧版
    pipeline）视为有效，保持向后兼容。

    manifest 缺 signal_date 字段（跳过）、多个 run_dir 撞同一 signal_date
    （按传入顺序后者覆盖前者）这两种情况均会 warning 说明，不再静默发生。
    """
    out: dict[date, pl.DataFrame] = {}
    source_by_date: dict[date, str] = {}
    for rd in portfolio_run_dirs:
        rd_p = Path(rd)
        mf_path = rd_p / "manifest.json"
        if not mf_path.exists():
            # 半成品目录（portfolio_build 写完 weights、未写 manifest 就崩）→ 跳过、不炸
            _logger.warning("跳过 run_dir=%s：无 manifest.json（疑似半成品目录）", rd)
            continue
        manifest = json.loads(mf_path.read_text())
        sig = manifest.get("signal_date")
        if sig is None:
            _logger.warning(
                "跳过 run_dir=%s：manifest.json 缺 signal_date 字段，无法作为有效信号执行",
                rd,
            )
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
        if sig_date in out:
            _logger.warning(
                "signal_date=%s 撞键：run_dir=%s 覆盖 run_dir=%s（按传入顺序，后者生效）",
                sig_date,
                rd,
                source_by_date[sig_date],
            )
        w = pl.read_parquet(rd_p / "weights.parquet").select(
            ["ts_code", "target_weight"]
        )
        out[sig_date] = w
        source_by_date[sig_date] = rd
    return out


def run_portfolio_simulation(
    portfolio_run_dirs: list[str],
    daily: pl.DataFrame,
    *,
    out_dir: str = str(SIM_DIR),
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

    # PrecomputedWeightsStrategy：统一日环内预填 target 矩阵，不依赖 factor 内容。
    # collect_*=False → 不写明细；factor_df=None（策略不消费因子）。
    # CostModel / CostModelBase 均走同一日环；CostModelBase 在日环内 per-name 计费。
    strategy = PrecomputedWeightsStrategy(weights_by_date)
    effective_cost_model: CostModel | CostModelBase = (
        cost_model if cost_model is not None else CostModel()
    )

    # PIT 收窄 ST 股票涨跌停阈值（4.8% 而非主板 9.8%，见
    # core/universe.py::_get_board_limit）；只构建一次，全程复用。
    codes = daily.select("ts_code").unique()["ts_code"].to_list()
    trade_dates_list = daily.select("trade_date").unique()["trade_date"].to_list()
    is_st_by_date = build_is_st_by_date(codes, trade_dates_list)

    # 组合流权重来自 portfolio/ 优化器，已受其自身 OptimizerConstraints（gross /
    # 单票上限）约束——sim 不应再拿 daily-research 的默认 BacktestConfig 上限
    # （max_gross_exposure=2.0 / max_abs_weight=1.0）二次校验，否则杠杆/多空/集中
    # 组合（gross>2.0 或单票>1.0）会触发 ValueError 崩掉整批模拟（portfolio/ 与
    # daily/optimization/ 命名空间的约束语义本就不同，不可互相套用）。这里把两个
    # exposure 上限放到 inf（信任优化器的暴露决策），但 _validate_target_weights
    # 仍保留 NaN/inf/重复 ts_code 的数据损坏防线。
    sim_config = BacktestConfig(
        max_gross_exposure=float("inf"),
        max_abs_weight=float("inf"),
    )
    bt = run_strategy_backtest(
        strategy,
        None,
        daily,
        config=sim_config,
        cost_model=effective_cost_model,
        collect_positions=False,
        collect_trades=False,
        include_context_positions=False,
        is_st_by_date=is_st_by_date,
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
    sig_dates = list(weights_by_date.keys())
    n_exec_dates = int(bt.nav.height) if not bt.nav.is_empty() else 0
    manifest: dict[str, Any] = {
        "run_id": rid,
        "n_signals": len(weights_by_date),
        "git_sha": get_git_sha(),
        "inputs": list(portfolio_run_dirs),
        "start": min(sig_dates).isoformat() if sig_dates else None,
        "end": max(sig_dates).isoformat() if sig_dates else None,
        "n_exec_dates": n_exec_dates,
        "cost_model": _jsonable(effective_cost_model),
        "config": _jsonable(sim_config),
        "command": "sim run",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )

    return {
        "run_dir": str(run_dir),
        "sharpe": metrics.get("sharpe"),
        "max_dd": metrics.get("max_dd"),
        "ann_ret": metrics.get("ann_ret"),
    }

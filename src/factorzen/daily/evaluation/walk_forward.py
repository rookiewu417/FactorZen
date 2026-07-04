"""策略级 Walk-Forward 验证。

滚动窗口切分（展开窗口模式），在每折的历史观察期上获取 IS Sharpe，
在未来验证期上获取 OOS Sharpe / 收益，最终拼接所有 OOS 收益并计算累计净值。
固定因子主流程不会在 IS 期拟合因子参数，IS 仅作为历史表现参照。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from factorzen.config.constants import TRADING_DAYS_PER_YEAR
from factorzen.core.universe import build_is_st_by_date
from factorzen.daily.evaluation.backtest import BacktestConfig, Strategy, run_strategy_backtest

logger = logging.getLogger(__name__)


# ── WalkForwardSplitter ──────────────────────────────────────────────────────


@dataclass
class WalkForwardSplitter:
    """滚动窗口切分器（展开窗口模式：历史观察期始终从第 0 天开始）。

    字段名保留 train/test 是为了兼容已有配置；语义上分别对应
    IS 历史观察期和 OOS 未来验证期。
    """

    train_days: int = 504  # IS 历史观察期长度（交易日）
    test_days: int = 63  # OOS 未来验证期长度（交易日）
    step_days: int = 63  # 每折步进（交易日），通常等于 test_days
    embargo_days: int = 5  # IS 期末到 OOS 期首的间隔（防时序泄漏）

    def split(self, dates: list[Any]) -> list[tuple[list[Any], list[Any]]]:
        """切分 dates 列表，返回 [(train_dates, test_dates), ...] 列表。

        展开窗口：每折历史观察期从 dates[0] 到 dates[train_end_idx]；
        未来验证期从 dates[test_start_idx] 到 dates[test_end_idx]，
        train_end_idx 和 test_start_idx 相差 embargo_days。

        若切分数量 < 1 则返回空列表。
        """
        results: list[tuple[list[Any], list[Any]]] = []
        n = len(dates)
        test_start_idx = self.train_days + self.embargo_days
        while test_start_idx + self.test_days <= n:
            train_end_idx = test_start_idx - self.embargo_days
            train_dates = dates[0:train_end_idx]
            test_end_idx = min(test_start_idx + self.test_days, n)
            test_dates = dates[test_start_idx:test_end_idx]
            if train_dates and test_dates:
                results.append((train_dates, test_dates))
            test_start_idx += self.step_days
        return results

    def n_splits(self, total_days: int) -> int:
        """预估切分折数（不生成实际日期列表）。"""
        first_test_start = self.train_days + self.embargo_days
        if first_test_start + self.test_days > total_days:
            return 0
        return max(0, (total_days - first_test_start - self.test_days) // self.step_days + 1)


# ── WalkForwardFoldResult ────────────────────────────────────────────────────


@dataclass
class WalkForwardFoldResult:
    """单折 walk-forward 结果。"""

    fold_id: int
    train_start: Any  # date or str
    train_end: Any
    test_start: Any
    test_end: Any
    is_sharpe: float
    oos_sharpe: float
    oos_ann_ret: float
    oos_max_dd: float
    params: dict[str, Any] = field(default_factory=dict)


# ── WalkForwardResult ────────────────────────────────────────────────────────


@dataclass
class WalkForwardResult:
    """Walk-forward 验证汇总结果。"""

    folds: list[WalkForwardFoldResult]
    oos_returns: pl.DataFrame  # trade_date, net_return, nav, fold_id — 拼接所有 OOS 期
    is_sharpe_mean: float
    oos_sharpe_mean: float
    oos_sharpe_std: float
    oos_max_dd: float  # 拼接 OOS 净值的最大回撤
    stability_ratio: float  # oos_sharpe_mean / is_sharpe_mean (>0.3 = 稳健)

    def summary(self) -> str:
        n = len(self.folds)
        return (
            f"WalkForward ({n} folds): "
            f"IS Sharpe={self.is_sharpe_mean:.2f} "
            f"OOS Sharpe={self.oos_sharpe_mean:.2f}±{self.oos_sharpe_std:.2f} "
            f"Stability={self.stability_ratio:.2f} "
            f"OOS MaxDD={self.oos_max_dd:.1%}"
        )


# ── 内部辅助函数 ──────────────────────────────────────────────────────────────


def _ensure_date_col(df: pl.DataFrame, col: str = "trade_date") -> pl.DataFrame:
    """将 trade_date 列统一转换为 pl.Date 类型。"""
    dtype = df.schema.get(col)
    if dtype is None:
        return df
    if dtype == pl.Date:
        return df
    if dtype == pl.Datetime:
        return df.with_columns(pl.col(col).dt.date().alias(col))
    if dtype == pl.Utf8:
        parsed_dash = pl.col(col).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        parsed_plain = pl.col(col).str.strptime(pl.Date, "%Y%m%d", strict=False)
        return df.with_columns(parsed_dash.fill_null(parsed_plain).alias(col))
    return df


def _extract_sharpe(result: Any) -> float:
    """从 StrategyBacktestResult.summary_stats 中提取 Sharpe 比率。"""
    stats: dict[Any, Any] = result.summary_stats
    for key in ("long_short", "portfolio"):
        if key in stats and isinstance(stats[key], dict):
            return float(stats[key].get("sharpe", 0.0))
    # 回退到第一个含 dict 的值
    for key in sorted(stats, key=str):
        if isinstance(stats[key], dict):
            return float(stats[key].get("sharpe", 0.0))
    return 0.0


def _is_expected_candidate_failure(exc: Exception) -> bool:
    message = str(exc)
    return message in {
        "target_weight exceeds max_abs_weight",
        "gross exposure exceeds max_gross_exposure",
    }


def _filter_date_range(df: pl.DataFrame, start: Any, end: Any) -> pl.DataFrame:
    return df.filter((pl.col("trade_date") >= start) & (pl.col("trade_date") <= end))


def _sharpe_from_returns(returns: pl.DataFrame) -> float:
    if returns.is_empty():
        return 0.0
    rets = returns["net_return"].to_numpy()
    valid = rets[np.isfinite(rets)]
    if len(valid) == 0:
        return 0.0
    ann_ret = float(np.mean(valid) * TRADING_DAYS_PER_YEAR)
    ann_vol = float(np.std(valid) * np.sqrt(TRADING_DAYS_PER_YEAR))
    return ann_ret / ann_vol if ann_vol > 0 else 0.0


def _sharpe_between(returns: pl.DataFrame, start: Any, end: Any) -> float:
    return _sharpe_from_returns(_filter_date_range(returns, start, end))


def _resolve_parallel_workers(
    parallel_workers: int | None,
    n_jobs: int,
    seed: int | None,
) -> int:
    if seed is not None or n_jobs <= 1:
        return 1
    if parallel_workers is None:
        return 1
    return max(1, min(int(parallel_workers), n_jobs))


def _extract_ann_ret(result: Any) -> float:
    """提取年化收益率。"""
    stats: dict[Any, Any] = result.summary_stats
    for key in ("long_short", "portfolio"):
        if key in stats and isinstance(stats[key], dict):
            return float(stats[key].get("ann_ret", 0.0))
    return 0.0


def _extract_max_dd(result: Any) -> float:
    """提取最大回撤。"""
    stats: dict[Any, Any] = result.summary_stats
    for key in ("long_short", "portfolio"):
        if key in stats and isinstance(stats[key], dict):
            return float(stats[key].get("max_dd", 0.0))
    return 0.0


def _compute_oos_max_dd(nav_series: list[float]) -> float:
    """从净值序列计算最大回撤（负数）。"""
    if not nav_series:
        return 0.0
    arr = np.concatenate([[1.0], np.array(nav_series, dtype=float)])
    running_max = np.maximum.accumulate(arr)
    dd = arr / running_max - 1.0
    return float(np.min(dd))


# ── 主函数 ────────────────────────────────────────────────────────────────────


def run_walk_forward(
    strategy_factory: Callable[[dict[str, Any]], Strategy],
    factor_df: pl.DataFrame,
    price_df: pl.DataFrame,
    splitter: WalkForwardSplitter,
    config: BacktestConfig | None = None,
    factor_name: str = "",
    params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> WalkForwardResult:
    """策略级 walk-forward 验证。

    对每折：
    1. 切出 IS 历史观察期 / OOS 未来验证期的因子和价格数据
    2. 用 strategy_factory(params) 生成策略实例
    3. IS 期：在历史观察期上运行回测，提取 IS Sharpe
    4. OOS 期：在未来验证期上运行回测，提取 OOS Sharpe / 收益
    5. 拼接所有 OOS returns 并计算累计净值

    Args:
        strategy_factory: 接受 params 字典返回 Strategy 实例。
        factor_df: 含 trade_date, ts_code, factor_clean（或配置中的因子列）的 DataFrame。
        price_df: 含 trade_date, ts_code, close 的 DataFrame。
        splitter: WalkForwardSplitter 实例。
        config: BacktestConfig，None 时使用默认值。
        factor_name: 因子名称。
        params: 传给 strategy_factory 的参数字典，None 时传空字典。
        seed: 随机种子，若指定则在每折开始时设置 seed + fold_id。

    Returns:
        WalkForwardResult
    """
    cfg = config or BacktestConfig()
    effective_params: dict[str, Any] = params or {}

    # 统一日期类型
    factor_df = _ensure_date_col(factor_df, "trade_date")
    price_df = _ensure_date_col(price_df, "trade_date")

    # 获取排序后的唯一日期列表
    dates: list[Any] = (
        price_df.select("trade_date").unique().sort("trade_date")["trade_date"].to_list()
    )
    # PIT 收窄 ST 股票涨跌停阈值（见 core/universe.py::_get_board_limit）；
    # 只构建一次，IS/OOS 每折均复用。
    codes = price_df.select("ts_code").unique()["ts_code"].to_list()
    is_st_by_date = build_is_st_by_date(codes, dates)

    folds_data = splitter.split(dates)
    if not folds_data:
        logger.warning("WalkForwardSplitter 未生成任何折，请检查数据长度和参数设置")
        empty_oos = pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "net_return": pl.Float64,
                "nav": pl.Float64,
                "fold_id": pl.Int32,
            }
        )
        return WalkForwardResult(
            folds=[],
            oos_returns=empty_oos,
            is_sharpe_mean=0.0,
            oos_sharpe_mean=0.0,
            oos_sharpe_std=0.0,
            oos_max_dd=0.0,
            stability_ratio=0.0,
        )

    fold_results: list[WalkForwardFoldResult] = []
    oos_return_parts: list[pl.DataFrame] = []

    for fold_id, (train_dates, test_dates) in enumerate(folds_data):
        if seed is not None:
            from factorzen.core.seed import set_global_seed

            set_global_seed(seed + fold_id)

        train_set = set(train_dates)
        test_set = set(test_dates)

        # 切出 IS 历史观察期 / OOS 未来验证期数据
        train_factor = factor_df.filter(pl.col("trade_date").is_in(train_set))
        train_price = price_df.filter(pl.col("trade_date").is_in(train_set))
        test_factor = factor_df.filter(pl.col("trade_date").is_in(test_set))
        test_price = price_df.filter(pl.col("trade_date").is_in(test_set))

        # IS 回测
        is_sharpe = 0.0
        try:
            strategy_is = strategy_factory(effective_params)
            is_result = run_strategy_backtest(
                strategy_is,
                train_factor,
                train_price,
                cfg,
                factor_name=factor_name,
                is_st_by_date=is_st_by_date,
            )
            is_sharpe = _extract_sharpe(is_result)
        except Exception as exc:
            logger.warning(f"Fold {fold_id} IS 回测失败，跳过该折: {exc}", exc_info=True)
            continue

        # OOS 回测
        oos_sharpe = 0.0
        oos_ann_ret = 0.0
        oos_max_dd_fold = 0.0
        oos_returns_df: pl.DataFrame | None = None
        try:
            strategy_oos = strategy_factory(effective_params)
            oos_result = run_strategy_backtest(
                strategy_oos,
                test_factor,
                test_price,
                cfg,
                factor_name=factor_name,
                is_st_by_date=is_st_by_date,
            )
            oos_sharpe = _extract_sharpe(oos_result)
            oos_ann_ret = _extract_ann_ret(oos_result)
            oos_max_dd_fold = _extract_max_dd(oos_result)
            if not oos_result.returns.is_empty():
                oos_returns_df = oos_result.returns.select(
                    ["trade_date", "net_return"]
                ).with_columns(pl.lit(fold_id).cast(pl.Int32).alias("fold_id"))
        except Exception as exc:
            logger.warning(f"Fold {fold_id} OOS 回测失败，跳过该折: {exc}", exc_info=True)
            continue

        fold_result = WalkForwardFoldResult(
            fold_id=fold_id,
            train_start=train_dates[0],
            train_end=train_dates[-1],
            test_start=test_dates[0],
            test_end=test_dates[-1],
            is_sharpe=is_sharpe,
            oos_sharpe=oos_sharpe,
            oos_ann_ret=oos_ann_ret,
            oos_max_dd=oos_max_dd_fold,
            params=effective_params,
        )
        fold_results.append(fold_result)

        if oos_returns_df is not None and not oos_returns_df.is_empty():
            oos_return_parts.append(oos_returns_df)

    # 拼接所有 OOS 日收益，计算累计净值
    if oos_return_parts:
        all_oos = pl.concat(oos_return_parts).sort("trade_date")
        # 计算跨所有折的连续累计净值
        rets = all_oos["net_return"].to_numpy()
        nav_values = np.cumprod(1.0 + rets)
        all_oos = all_oos.with_columns(pl.Series("nav", nav_values))
        oos_max_dd_total = _compute_oos_max_dd(nav_values.tolist())
    else:
        all_oos = pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "net_return": pl.Float64,
                "fold_id": pl.Int32,
                "nav": pl.Float64,
            }
        )
        oos_max_dd_total = 0.0

    # 汇总统计
    if fold_results:
        is_sharpes = [f.is_sharpe for f in fold_results]
        oos_sharpes = [f.oos_sharpe for f in fold_results]
        is_sharpe_mean = float(np.mean(is_sharpes))
        oos_sharpe_mean = float(np.mean(oos_sharpes))
        oos_sharpe_std = float(np.std(oos_sharpes))
        # 稳定性比率：防止 is_sharpe_mean 接近 0
        stability_ratio = oos_sharpe_mean / max(abs(is_sharpe_mean), 1e-8)
    else:
        is_sharpe_mean = 0.0
        oos_sharpe_mean = 0.0
        oos_sharpe_std = 0.0
        stability_ratio = 0.0

    return WalkForwardResult(
        folds=fold_results,
        oos_returns=all_oos,
        is_sharpe_mean=is_sharpe_mean,
        oos_sharpe_mean=oos_sharpe_mean,
        oos_sharpe_std=oos_sharpe_std,
        oos_max_dd=oos_max_dd_total,
        stability_ratio=stability_ratio,
    )


def run_walk_forward_search(
    *,
    strategy_factory: Callable[[dict[str, Any]], Strategy],
    factor_df: pl.DataFrame,
    price_df: pl.DataFrame,
    splitter: WalkForwardSplitter,
    param_candidates: list[dict[str, Any]],
    config: BacktestConfig | None = None,
    factor_name: str = "",
    seed: int | None = None,
    reuse_is_backtests: bool = False,
    parallel_workers: int | None = None,
    include_context_positions: bool = True,
) -> WalkForwardResult:
    """Walk-forward validation with per-fold IS parameter selection."""
    cfg = config or BacktestConfig()
    candidates = param_candidates or [{}]
    factor_df = _ensure_date_col(factor_df, "trade_date")
    price_df = _ensure_date_col(price_df, "trade_date")
    dates: list[Any] = (
        price_df.select("trade_date").unique().sort("trade_date")["trade_date"].to_list()
    )
    # PIT 收窄 ST 股票涨跌停阈值（见 core/universe.py::_get_board_limit）；只
    # 构建一次，IS 全量缓存 / 逐折 IS 搜索 / OOS 三处调用均复用同一份。
    codes = price_df.select("ts_code").unique()["ts_code"].to_list()
    is_st_by_date = build_is_st_by_date(codes, dates)

    folds_data = splitter.split(dates)
    if not folds_data:
        empty_oos = pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "net_return": pl.Float64,
                "nav": pl.Float64,
                "fold_id": pl.Int32,
            }
        )
        return WalkForwardResult(
            folds=[],
            oos_returns=empty_oos,
            is_sharpe_mean=0.0,
            oos_sharpe_mean=0.0,
            oos_sharpe_std=0.0,
            oos_max_dd=0.0,
            stability_ratio=0.0,
        )

    fold_results: list[WalkForwardFoldResult] = []
    oos_return_parts: list[pl.DataFrame] = []
    reuse_is = reuse_is_backtests and seed is None
    cached_is_returns: list[tuple[dict[str, Any], pl.DataFrame]] = []
    if reuse_is:
        for params in candidates:
            try:
                result = run_strategy_backtest(
                    strategy_factory(params),
                    factor_df,
                    price_df,
                    cfg,
                    factor_name=factor_name,
                    collect_positions=False,
                    collect_trades=False,
                    include_context_positions=include_context_positions,
                    is_st_by_date=is_st_by_date,
                )
                cached_is_returns.append((dict(params), result.returns))
            except Exception as exc:
                message = f"IS cache failed for params={params}: {exc}"
                if _is_expected_candidate_failure(exc):
                    logger.warning(message)
                else:
                    logger.warning(message, exc_info=True)

    oos_jobs: list[tuple[int, list[Any], list[Any], dict[str, Any], float]] = []

    for fold_id, (train_dates, test_dates) in enumerate(folds_data):
        if seed is not None:
            from factorzen.core.seed import set_global_seed

            set_global_seed(seed + fold_id)

        train_factor = None
        train_price = None
        if not reuse_is:
            train_factor = _filter_date_range(factor_df, train_dates[0], train_dates[-1])
            train_price = _filter_date_range(price_df, train_dates[0], train_dates[-1])

        best_params: dict[str, Any] | None = None
        best_is_sharpe: float | None = None
        candidate_iter = (
            cached_is_returns if reuse_is else [(dict(params), None) for params in candidates]
        )
        for params, cached_returns in candidate_iter:
            if cached_returns is not None:
                sharpe = _sharpe_between(cached_returns, train_dates[0], train_dates[-1])
            else:
                try:
                    assert train_factor is not None and train_price is not None
                    result = run_strategy_backtest(
                        strategy_factory(params),
                        train_factor,
                        train_price,
                        cfg,
                        factor_name=factor_name,
                        collect_positions=False,
                        collect_trades=False,
                        include_context_positions=include_context_positions,
                        is_st_by_date=is_st_by_date,
                    )
                    sharpe = _extract_sharpe(result)
                except Exception as exc:
                    message = f"Fold {fold_id} IS search failed for params={params}: {exc}"
                    if _is_expected_candidate_failure(exc):
                        logger.warning(message)
                    else:
                        logger.warning(message, exc_info=True)
                    continue
            if best_is_sharpe is None or sharpe > best_is_sharpe:
                best_is_sharpe = sharpe
                best_params = dict(params)

        if best_params is None:
            continue

        oos_jobs.append(
            (fold_id, train_dates, test_dates, best_params, float(best_is_sharpe or 0.0))
        )

    def _run_oos_job(
        job: tuple[int, list[Any], list[Any], dict[str, Any], float],
    ) -> tuple[WalkForwardFoldResult, pl.DataFrame | None] | None:
        fold_id, train_dates, test_dates, best_params, best_is_sharpe = job
        oos_sharpe = 0.0
        oos_ann_ret = 0.0
        oos_max_dd_fold = 0.0
        oos_returns_df: pl.DataFrame | None = None
        try:
            test_factor = _filter_date_range(factor_df, test_dates[0], test_dates[-1])
            test_price = _filter_date_range(price_df, test_dates[0], test_dates[-1])
            oos_result = run_strategy_backtest(
                strategy_factory(best_params),
                test_factor,
                test_price,
                cfg,
                factor_name=factor_name,
                collect_positions=False,
                collect_trades=False,
                include_context_positions=include_context_positions,
                is_st_by_date=is_st_by_date,
            )
            oos_sharpe = _extract_sharpe(oos_result)
            oos_ann_ret = _extract_ann_ret(oos_result)
            oos_max_dd_fold = _extract_max_dd(oos_result)
            if not oos_result.returns.is_empty():
                oos_returns_df = oos_result.returns.select(
                    ["trade_date", "net_return"]
                ).with_columns(pl.lit(fold_id).cast(pl.Int32).alias("fold_id"))
        except Exception as exc:
            logger.warning(f"Fold {fold_id} OOS 回测失败，跳过该折: {exc}", exc_info=True)
            return None

        return (
            WalkForwardFoldResult(
                fold_id=fold_id,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                test_start=test_dates[0],
                test_end=test_dates[-1],
                is_sharpe=float(best_is_sharpe or 0.0),
                oos_sharpe=oos_sharpe,
                oos_ann_ret=oos_ann_ret,
                oos_max_dd=oos_max_dd_fold,
                params=best_params,
            ),
            oos_returns_df,
        )

    workers = _resolve_parallel_workers(parallel_workers, len(oos_jobs), seed)
    completed: list[tuple[WalkForwardFoldResult, pl.DataFrame | None]] = []
    if workers <= 1:
        for job in oos_jobs:
            item = _run_oos_job(job)
            if item is not None:
                completed.append(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_oos_job, job) for job in oos_jobs]
            for future in as_completed(futures):
                item = future.result()
                if item is not None:
                    completed.append(item)

    for fold_result, oos_returns_df in sorted(completed, key=lambda item: item[0].fold_id):
        fold_results.append(fold_result)
        if oos_returns_df is not None and not oos_returns_df.is_empty():
            oos_return_parts.append(oos_returns_df)

    if oos_return_parts:
        all_oos = pl.concat(oos_return_parts).sort("trade_date")
        rets = all_oos["net_return"].to_numpy()
        nav_values = np.cumprod(1.0 + rets)
        all_oos = all_oos.with_columns(pl.Series("nav", nav_values))
        oos_max_dd_total = _compute_oos_max_dd(nav_values.tolist())
    else:
        all_oos = pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "net_return": pl.Float64,
                "fold_id": pl.Int32,
                "nav": pl.Float64,
            }
        )
        oos_max_dd_total = 0.0

    if fold_results:
        is_sharpes = [f.is_sharpe for f in fold_results]
        oos_sharpes = [f.oos_sharpe for f in fold_results]
        is_sharpe_mean = float(np.mean(is_sharpes))
        oos_sharpe_mean = float(np.mean(oos_sharpes))
        oos_sharpe_std = float(np.std(oos_sharpes))
        stability_ratio = oos_sharpe_mean / max(abs(is_sharpe_mean), 1e-8)
    else:
        is_sharpe_mean = 0.0
        oos_sharpe_mean = 0.0
        oos_sharpe_std = 0.0
        stability_ratio = 0.0

    return WalkForwardResult(
        folds=fold_results,
        oos_returns=all_oos,
        is_sharpe_mean=is_sharpe_mean,
        oos_sharpe_mean=oos_sharpe_mean,
        oos_sharpe_std=oos_sharpe_std,
        oos_max_dd=oos_max_dd_total,
        stability_ratio=stability_ratio,
    )

"""策略级 Walk-Forward 验证测试。"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from factorzen.daily.evaluation.walk_forward import (
    WalkForwardFoldResult,
    WalkForwardResult,
    WalkForwardSplitter,
    _compute_oos_max_dd,
    run_walk_forward,
)

# ── 测试夹具 ─────────────────────────────────────────────────────────────────


@pytest.fixture
def factor_df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n_dates, n_stocks = 300, 30
    start = date(2022, 1, 3)
    dates = [(start + timedelta(days=i)).isoformat() for i in range(n_dates)]
    records = []
    for d in dates:
        for s in range(n_stocks):
            records.append(
                {
                    "trade_date": d,
                    "ts_code": f"{s:06d}.SZ",
                    "factor_clean": float(rng.normal()),
                }
            )
    return pl.DataFrame(records)


@pytest.fixture
def price_df(factor_df: pl.DataFrame) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    codes = factor_df["ts_code"].unique().to_list()
    dates = factor_df["trade_date"].unique().sort().to_list()
    records = []
    for code in codes:
        price = 10.0
        for d in dates:
            price *= 1 + rng.normal(0.0005, 0.02)
            records.append(
                {
                    "trade_date": d,
                    "ts_code": code,
                    "close": price,
                    "open": price * 0.998,
                }
            )
    return pl.DataFrame(records)


# ── TestWalkForwardSplitter ──────────────────────────────────────────────────


class TestWalkForwardSplitter:
    def test_n_splits_formula(self):
        """n_splits(total_days) 应与 split(dates) 实际返回折数一致。"""
        splitter = WalkForwardSplitter(
            train_days=100, test_days=30, step_days=30, embargo_days=5
        )
        total_days = 250
        dates = [f"day_{i}" for i in range(total_days)]
        actual = len(splitter.split(dates))
        estimated = splitter.n_splits(total_days)
        assert estimated == actual, (
            f"n_splits({total_days})={estimated} 与 split 实际折数={actual} 不一致"
        )

    def test_embargo_prevents_leakage(self):
        """每折历史观察期末尾索引 + embargo_days <= 未来验证期首索引。"""
        splitter = WalkForwardSplitter(
            train_days=100, test_days=30, step_days=30, embargo_days=5
        )
        total_days = 250
        dates = list(range(total_days))
        folds = splitter.split(dates)
        assert len(folds) > 0, "应有至少一折"
        for train_dates, test_dates in folds:
            # 找最后一个历史观察日在 dates 中的索引
            train_end_val = train_dates[-1]
            test_start_val = test_dates[0]
            train_end_idx = dates.index(train_end_val)
            test_start_idx = dates.index(test_start_val)
            assert test_start_idx - train_end_idx >= splitter.embargo_days, (
                f"embargo 不足: train_end_idx={train_end_idx}, "
                f"test_start_idx={test_start_idx}"
            )

    def test_empty_when_too_short(self):
        """总日数不足时返回空列表，不崩溃。"""
        splitter = WalkForwardSplitter(
            train_days=200, test_days=50, step_days=50, embargo_days=10
        )
        # total_days < train_days + embargo_days + test_days
        dates = [f"day_{i}" for i in range(100)]
        result = splitter.split(dates)
        assert result == []

    def test_train_always_from_zero(self):
        """展开窗口：每折历史观察期从 dates[0] 开始。"""
        splitter = WalkForwardSplitter(
            train_days=80, test_days=20, step_days=20, embargo_days=5
        )
        dates = [f"day_{i}" for i in range(200)]
        folds = splitter.split(dates)
        assert len(folds) > 0
        for train_dates, _test_dates in folds:
            assert train_dates[0] == dates[0], (
                "展开窗口：历史观察期第一个日期应始终为 dates[0]"
            )


# ── TestRunWalkForward ───────────────────────────────────────────────────────


def test_oos_max_drawdown_includes_initial_nav():
    assert _compute_oos_max_dd([0.90]) == pytest.approx(-0.10)


class TestRunWalkForward:
    def _make_splitter(self) -> WalkForwardSplitter:
        return WalkForwardSplitter(
            train_days=100, test_days=30, step_days=30, embargo_days=5
        )

    def _strategy_factory(self, params: dict) -> object:
        from factorzen.daily.evaluation.backtest import QuantileLongShortStrategy

        return QuantileLongShortStrategy(n_groups=params.get("n_groups", 5))

    def test_oos_nav_starts_at_one(self, factor_df: pl.DataFrame, price_df: pl.DataFrame):
        """OOS 拼接净值序列的第一个值应接近 1.0（从初始净值 1.0 开始乘以 (1+ret)）。"""
        splitter = self._make_splitter()
        result = run_walk_forward(
            strategy_factory=self._strategy_factory,
            factor_df=factor_df,
            price_df=price_df,
            splitter=splitter,
            params={"n_groups": 5},
        )
        assert len(result.folds) > 0, "Expected at least one WF fold but got 0 — fixture may be too small"
        if result.folds and not result.oos_returns.is_empty():
            first_nav = result.oos_returns.sort("trade_date")["nav"][0]
            # 第一个 nav 应等于 1 + first_net_return，不必精确为 1.0
            first_ret = result.oos_returns.sort("trade_date")["net_return"][0]
            expected = 1.0 * (1.0 + first_ret)
            assert abs(float(first_nav) - float(expected)) < 1e-9

    def test_oos_returns_no_gaps(self, factor_df: pl.DataFrame, price_df: pl.DataFrame):
        """OOS 日期在不同折之间不应重叠（每个日期至多出现一次）。"""
        splitter = self._make_splitter()
        result = run_walk_forward(
            strategy_factory=self._strategy_factory,
            factor_df=factor_df,
            price_df=price_df,
            splitter=splitter,
            params={"n_groups": 5},
        )
        assert len(result.folds) > 0, "Expected at least one WF fold but got 0 — fixture may be too small"
        if result.oos_returns.is_empty():
            return
        dates = result.oos_returns["trade_date"].to_list()
        assert len(dates) == len(set(dates)), "OOS 日期存在重叠（跨折）"

    def test_stability_ratio_bounds(self, factor_df: pl.DataFrame, price_df: pl.DataFrame):
        """stability_ratio 不应为 NaN 或 Inf，即使 IS Sharpe 接近 0。"""
        splitter = self._make_splitter()
        result = run_walk_forward(
            strategy_factory=self._strategy_factory,
            factor_df=factor_df,
            price_df=price_df,
            splitter=splitter,
            params={"n_groups": 5},
        )
        assert np.isfinite(result.stability_ratio), (
            f"stability_ratio 应为有限数，got {result.stability_ratio}"
        )

    def test_result_structure(self, factor_df: pl.DataFrame, price_df: pl.DataFrame):
        """WalkForwardResult 应包含所有必需字段且类型正确。"""
        splitter = self._make_splitter()
        result = run_walk_forward(
            strategy_factory=self._strategy_factory,
            factor_df=factor_df,
            price_df=price_df,
            splitter=splitter,
            params={"n_groups": 5},
        )
        assert len(result.folds) > 0, "Expected at least one WF fold but got 0 — fixture may be too small"
        assert isinstance(result, WalkForwardResult)
        assert isinstance(result.folds, list)
        assert isinstance(result.oos_returns, pl.DataFrame)
        assert isinstance(result.is_sharpe_mean, float)
        assert isinstance(result.oos_sharpe_mean, float)
        assert isinstance(result.oos_sharpe_std, float)
        assert isinstance(result.oos_max_dd, float)
        assert isinstance(result.stability_ratio, float)

        # 检查必须列
        if not result.oos_returns.is_empty():
            cols = set(result.oos_returns.columns)
            assert "trade_date" in cols
            assert "net_return" in cols
            assert "fold_id" in cols
            assert "nav" in cols

        # 每折都是 WalkForwardFoldResult
        for fold in result.folds:
            assert isinstance(fold, WalkForwardFoldResult)
            assert isinstance(fold.fold_id, int)
            assert isinstance(fold.is_sharpe, float)
            assert isinstance(fold.oos_sharpe, float)
            assert isinstance(fold.oos_ann_ret, float)
            assert isinstance(fold.oos_max_dd, float)
            assert isinstance(fold.params, dict)

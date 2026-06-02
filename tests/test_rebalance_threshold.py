"""rebalance_threshold 功能测试：换手率低于阈值时跳过调仓。"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from factorzen.daily.evaluation.backtest import (
    BacktestConfig,
    TopNLongOnlyStrategy,
    run_strategy_backtest,
)

# ──────────────────────────────────────────────────────────
# 测试夹具
# ──────────────────────────────────────────────────────────


def _make_fixtures(
    n_days: int = 40,
    n_stocks: int = 20,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """构造最小因子+价格数据（无依赖外部存储）。"""
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 3)
    dates = [(start + timedelta(days=i)).isoformat() for i in range(n_days)]

    factor_rows = []
    price_rows = []
    last_close = {f"00{s:04d}.SZ": 10.0 + s for s in range(n_stocks)}

    for d in dates:
        for s in range(n_stocks):
            ts = f"00{s:04d}.SZ"
            factor_rows.append({
                "trade_date": d,
                "ts_code": ts,
                "factor_clean": float(rng.standard_normal()),
            })
            open_price = last_close[ts]
            close_price = open_price * (1.0 + float(rng.uniform(-0.05, 0.05)))
            price_rows.append({
                "trade_date": d,
                "ts_code": ts,
                "open": open_price,
                "close": close_price,
                "pre_close": last_close[ts],
                "pct_chg": (close_price / last_close[ts] - 1.0) * 100,
                "vol": float(rng.uniform(1e6, 1e8)),
                "amount": float(rng.uniform(1e7, 1e9)),
            })
            last_close[ts] = close_price

    return pl.DataFrame(factor_rows), pl.DataFrame(price_rows)


# ──────────────────────────────────────────────────────────
# BacktestConfig rebalance_threshold 字段测试
# ──────────────────────────────────────────────────────────


def test_backtest_config_has_rebalance_threshold():
    """BacktestConfig 包含 rebalance_threshold 字段，默认 None。"""
    cfg = BacktestConfig()
    assert hasattr(cfg, "rebalance_threshold")
    assert cfg.rebalance_threshold is None


def test_backtest_config_rebalance_threshold_custom():
    """BacktestConfig 可自定义 rebalance_threshold。"""
    cfg = BacktestConfig(rebalance_threshold=0.5)
    assert cfg.rebalance_threshold == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────
# 高阈值 → 换手率应降低（几乎不调仓）
# ──────────────────────────────────────────────────────────


def test_high_threshold_reduces_turnover():
    """rebalance_threshold 很大时，几乎每期都跳过调仓，换手率应显著低于无阈值。"""
    factor_df, price_df = _make_fixtures()
    strategy = TopNLongOnlyStrategy(n=5)

    cfg_no_threshold = BacktestConfig(
        rebalance_threshold=None,
        max_participation_rate=1.0,
    )
    cfg_high_threshold = BacktestConfig(
        rebalance_threshold=100.0,  # 极大阈值，几乎永远不触发调仓
        max_participation_rate=1.0,
    )

    result_no = run_strategy_backtest(strategy, factor_df, price_df, cfg_no_threshold)
    result_high = run_strategy_backtest(strategy, factor_df, price_df, cfg_high_threshold)

    turnover_no = result_no.summary_stats["portfolio"]["avg_turnover"]
    turnover_high = result_high.summary_stats["portfolio"]["avg_turnover"]

    assert turnover_high <= turnover_no + 1e-6, (
        f"高阈值换手率 {turnover_high:.4f} 应 ≤ 无阈值换手率 {turnover_no:.4f}"
    )


def test_zero_threshold_matches_no_threshold():
    """rebalance_threshold=0 时，每期换手率 > 0 → 永不跳过，结果应与 None 完全相同。"""
    factor_df, price_df = _make_fixtures()
    strategy = TopNLongOnlyStrategy(n=5)

    cfg_none = BacktestConfig(rebalance_threshold=None, max_participation_rate=1.0)
    cfg_zero = BacktestConfig(rebalance_threshold=0.0, max_participation_rate=1.0)

    result_none = run_strategy_backtest(strategy, factor_df, price_df, cfg_none)
    result_zero = run_strategy_backtest(strategy, factor_df, price_df, cfg_zero)

    nav_none = result_none.nav["nav"].to_list()
    nav_zero = result_zero.nav["nav"].to_list()

    assert len(nav_none) == len(nav_zero)
    for a, b in zip(nav_none, nav_zero, strict=True):
        assert abs(a - b) < 1e-10, f"threshold=0 与 threshold=None 结果应一致: {a} vs {b}"


def test_result_structure_with_threshold():
    """带 rebalance_threshold 的回测结果结构完整。"""
    from factorzen.daily.evaluation.backtest import StrategyBacktestResult

    factor_df, price_df = _make_fixtures()
    strategy = TopNLongOnlyStrategy(n=5)
    cfg = BacktestConfig(rebalance_threshold=0.3, max_participation_rate=1.0)

    result = run_strategy_backtest(strategy, factor_df, price_df, cfg)

    assert isinstance(result, StrategyBacktestResult)
    assert "net_return" in result.returns.columns
    assert "nav" in result.nav.columns
    assert "portfolio" in result.summary_stats


def test_moderate_threshold_reduces_turnover():
    """适中阈值（0.3）时换手率不超过无阈值版本。"""
    factor_df, price_df = _make_fixtures()
    strategy = TopNLongOnlyStrategy(n=5)

    cfg_no = BacktestConfig(rebalance_threshold=None, max_participation_rate=1.0)
    cfg_mod = BacktestConfig(rebalance_threshold=0.3, max_participation_rate=1.0)

    result_no = run_strategy_backtest(strategy, factor_df, price_df, cfg_no)
    result_mod = run_strategy_backtest(strategy, factor_df, price_df, cfg_mod)

    to_no = result_no.summary_stats["portfolio"]["avg_turnover"]
    to_mod = result_mod.summary_stats["portfolio"]["avg_turnover"]

    # 有阈值时换手率应 ≤ 无阈值
    assert to_mod <= to_no + 1e-6, f"moderate threshold turnover {to_mod:.4f} > no threshold {to_no:.4f}"

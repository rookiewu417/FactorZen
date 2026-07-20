"""S2 防回归：验证 CostModel 和成本扣除逻辑。

验证：
- cost_model=None 时结果与旧版相同（向后兼容）
- 启用 CostModel 后，多空年化收益应下降（成本>0）
- CostModel 计算方法正确
- 融券费率从多空收益中单独扣除
"""

import numpy as np
import polars as pl

from factorzen.config.constants import (
    BORROW_RATE_ANNUAL,
    COMMISSION_RATE,
    SLIPPAGE_RATE,
    STAMP_TAX_RATE,
    TRADING_DAYS_PER_YEAR,
)
from factorzen.daily.evaluation.backtest import CostModel, run_stratified_backtest
from factorzen.daily.evaluation.cost_models import SquareRootImpactCostModel


def _make_data(n_dates: int = 200, n_stocks: int = 100, seed: int = 0):
    """合成因子+价格数据。"""
    rng = np.random.default_rng(seed)
    dates = [f"2024-{(i // 28 + 1):02d}-{(i % 28 + 1):02d}" for i in range(n_dates)]
    stocks = [f"{i:06d}.SZ" for i in range(n_stocks)]

    factor_rows, price_rows = [], []
    last_close = {s: 10.0 + i for i, s in enumerate(stocks)}
    for day_idx, d in enumerate(dates):
        factor_vals = rng.standard_normal(n_stocks)
        intraday_rets = 0.05 * factor_vals / n_stocks + rng.normal(0, 0.02, n_stocks)
        for i, s in enumerate(stocks):
            if day_idx < n_dates - 1:
                factor_rows.append(
                    {"trade_date": d, "ts_code": s, "factor_clean": float(factor_vals[i])}
                )
            open_price = last_close[s]
            close_price = open_price * (1.0 + float(intraday_rets[i]))
            price_rows.append(
                {
                    "trade_date": d,
                    "ts_code": s,
                    "open": open_price,
                    "close": close_price,
                    "pre_close": last_close[s],
                    "pct_chg": (close_price / last_close[s] - 1.0) * 100,
                    "vol": 1000.0,
                    "amount": 1_000_000_000.0,
                }
            )
            last_close[s] = close_price

    return pl.DataFrame(factor_rows), pl.DataFrame(price_rows)


class TestCostModel:
    def test_default_values(self):
        """CostModel 默认值应与 constants.py 一致。"""
        cm = CostModel()
        assert cm.commission == COMMISSION_RATE
        assert cm.stamp_tax == STAMP_TAX_RATE
        assert cm.slippage == SLIPPAGE_RATE
        assert cm.borrow_annual == BORROW_RATE_ANNUAL

    def test_round_trip_cost(self):
        """往返成本 = 2×commission + 2×slippage + stamp_tax。"""
        cm = CostModel()
        expected = 2 * cm.commission + 2 * cm.slippage + cm.stamp_tax
        assert abs(cm.round_trip_cost() - expected) < 1e-10

    def test_borrow_rate_per_period_daily(self):
        """日频融券费率 = 年化 / 252。"""
        cm = CostModel()
        expected = cm.borrow_annual / TRADING_DAYS_PER_YEAR
        assert abs(cm.borrow_rate_per_period("daily") - expected) < 1e-12

    def test_borrow_rate_per_period_weekly(self):
        """周频融券费率 = 年化 × 5 / 252。"""
        cm = CostModel()
        assert (
            abs(cm.borrow_rate_per_period("weekly") - cm.borrow_annual * 5 / TRADING_DAYS_PER_YEAR)
            < 1e-12
        )

    def test_custom_values(self):
        """支持自定义成本参数。"""
        cm = CostModel(commission=0.0001, stamp_tax=0.0, slippage=0.0, borrow_annual=0.05)
        assert abs(cm.round_trip_cost() - 0.0002) < 1e-12  # 2×0.0001


class TestBacktestCostIntegration:
    def test_costs_reduce_returns(self):
        """启用 CostModel 后，多空年化收益应低于无成本版本。"""
        factor_df, price_df = _make_data()
        r_free = run_stratified_backtest(factor_df, price_df, n_groups=5)
        r_cost = run_stratified_backtest(factor_df, price_df, n_groups=5, cost_model=CostModel())
        ann_ret_free = r_free.summary_stats["long_short"]["ann_ret"]
        ann_ret_cost = r_cost.summary_stats["long_short"]["ann_ret"]
        assert ann_ret_free > ann_ret_cost, (
            f"加入成本后年化收益应下降: 无成本={ann_ret_free:.4f}, 含成本={ann_ret_cost:.4f}"
        )

    def test_zero_cost_model_matches_no_cost(self):
        """CostModel 全部费率设为 0 时，结果应与 cost_model=None 完全相同。"""
        factor_df, price_df = _make_data()
        r_none = run_stratified_backtest(factor_df, price_df, n_groups=5)
        r_zero = run_stratified_backtest(
            factor_df,
            price_df,
            n_groups=5,
            cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
        )
        # 零成本时结果应近似相同（因 borrow_rate=0 且 round_trip=0）
        s_none = r_none.summary_stats["long_short"]["ann_ret"]
        s_zero = r_zero.summary_stats["long_short"]["ann_ret"]
        # 允许极小浮点误差
        assert abs(s_none - s_zero) < 1e-6, f"零成本模型应与无成本相同: {s_none} vs {s_zero}"


class TestSquareRootImpactCostModel:
    def test_adv_missing_uses_fallback(self):
        """adv=None 时应使用 fallback_adv，成本有限且为正。"""
        m = SquareRootImpactCostModel(alpha=0.1, fallback_adv=1e7)
        cost = m.trade_cost(delta_weight=0.01, adv=None)
        cost_explicit = m.trade_cost(delta_weight=0.01, adv=1e7)
        assert cost == cost_explicit, "adv=None 应等价于 adv=fallback_adv"
        assert cost > 0

    def test_adv_zero_uses_fallback(self):
        """adv=0 时应回退到 fallback_adv，不触发除零。"""
        m = SquareRootImpactCostModel(alpha=0.1, fallback_adv=1e7)
        cost = m.trade_cost(delta_weight=0.01, adv=0.0)
        assert cost > 0
        assert np.isfinite(cost)

    def test_extreme_turnover_finite(self):
        """极端换手（delta_weight=1.0）成本应有限且不溢出。"""
        m = SquareRootImpactCostModel(alpha=0.1, fallback_adv=1e7)
        cost = m.trade_cost(delta_weight=1.0, adv=1e6)
        assert np.isfinite(cost)
        assert cost > 0

    def test_alpha_parameterization(self):
        """alpha 越大，冲击成本越大。"""
        m_low = SquareRootImpactCostModel(alpha=0.01)
        m_high = SquareRootImpactCostModel(alpha=1.0)
        assert m_high.trade_cost(delta_weight=0.05) > m_low.trade_cost(delta_weight=0.05)

    def test_fallback_adv_scales_impact_for_given_adv(self):
        """fallback_adv 是 ADV 归一化基准：基准越大，同等 adv 被视为流动性越差，成本越高。"""
        adv = 1e6  # 固定的实际成交额
        # fallback_adv=1e9：adv_normalized = 1e6/1e9 = 0.001，冲击高
        m_high_ref = SquareRootImpactCostModel(alpha=0.1, fallback_adv=1e9)
        # fallback_adv=1e5：adv_normalized = 1e6/1e5 = 10，冲击低
        m_low_ref = SquareRootImpactCostModel(alpha=0.1, fallback_adv=1e5)
        assert m_high_ref.trade_cost(delta_weight=0.01, adv=adv) > m_low_ref.trade_cost(
            delta_weight=0.01, adv=adv
        )

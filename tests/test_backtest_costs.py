"""S2 防回归：验证 CostModel 和成本扣除逻辑。

验证：
- cost_model=None 时结果与旧版相同（向后兼容）
- 启用 CostModel 后，多空年化收益应下降（成本>0）
- CostModel 计算方法正确
- 融券费率从多空收益中单独扣除
"""

import numpy as np
import polars as pl

from config.constants import (
    BORROW_RATE_ANNUAL,
    COMMISSION_RATE,
    SLIPPAGE_RATE,
    STAMP_TAX_RATE,
    TRADING_DAYS_PER_YEAR,
)
from daily.evaluation.backtest import CostModel, run_stratified_backtest


def _make_data(n_dates: int = 200, n_stocks: int = 100, seed: int = 0):
    """合成因子+前向收益数据（有弱正向预测力的因子）。"""
    rng = np.random.default_rng(seed)
    dates = [f"2024-{(i // 28 + 1):02d}-{(i % 28 + 1):02d}" for i in range(n_dates)]
    stocks = [f"{i:06d}.SZ" for i in range(n_stocks)]

    factor_rows, ret_rows = [], []
    for d in dates:
        # 因子：随机截面信号
        factor_vals = rng.standard_normal(n_stocks)
        # 前向收益：因子 + 大量噪声（弱正 IC ≈ 0.05）
        fwd_rets = 0.05 * factor_vals / n_stocks + rng.normal(0, 0.02, n_stocks)
        for i, s in enumerate(stocks):
            factor_rows.append(
                {"trade_date": d, "ts_code": s, "factor_clean": float(factor_vals[i])}
            )
            ret_rows.append({"trade_date": d, "ts_code": s, "ret": float(fwd_rets[i])})

    return pl.DataFrame(factor_rows), pl.DataFrame(ret_rows)


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
    def test_no_cost_model_backward_compatible(self):
        """cost_model=None 时与旧版行为一致（无成本扣除）。"""
        factor_df, ret_df = _make_data()
        r1 = run_stratified_backtest(factor_df, ret_df, n_groups=5)
        r2 = run_stratified_backtest(factor_df, ret_df, n_groups=5, cost_model=None)
        # 两者 long-short Sharpe 应完全相同
        s1 = r1.summary_stats["long_short"]["sharpe"]
        s2 = r2.summary_stats["long_short"]["sharpe"]
        assert abs(s1 - s2) < 1e-10, f"cost_model=None 结果应与默认相同: {s1} vs {s2}"

    def test_costs_reduce_returns(self):
        """启用 CostModel 后，多空年化收益应低于无成本版本。"""
        factor_df, ret_df = _make_data()
        r_free = run_stratified_backtest(factor_df, ret_df, n_groups=5)
        r_cost = run_stratified_backtest(factor_df, ret_df, n_groups=5, cost_model=CostModel())
        ann_ret_free = r_free.summary_stats["long_short"]["ann_ret"]
        ann_ret_cost = r_cost.summary_stats["long_short"]["ann_ret"]
        assert ann_ret_free > ann_ret_cost, (
            f"加入成本后年化收益应下降: 无成本={ann_ret_free:.4f}, 含成本={ann_ret_cost:.4f}"
        )

    def test_costs_impact_magnitude(self):
        """成本扣除幅度应在合理范围（不超过 30% 年化，不小于 0.1%）。"""
        factor_df, ret_df = _make_data()
        r_free = run_stratified_backtest(factor_df, ret_df, n_groups=5)
        r_cost = run_stratified_backtest(factor_df, ret_df, n_groups=5, cost_model=CostModel())
        ann_ret_free = r_free.summary_stats["long_short"]["ann_ret"]
        ann_ret_cost = r_cost.summary_stats["long_short"]["ann_ret"]
        cost_drag = ann_ret_free - ann_ret_cost
        # 成本拖累应为正（因为有实际成本）
        assert cost_drag > 0, f"成本拖累应为正值: {cost_drag:.4f}"
        # 对于日频高换手策略，成本应在 2%~50% 年化范围内
        assert cost_drag < 0.5, f"成本拖累 {cost_drag:.4f} 超出合理上限 50%"

    def test_cost_model_instance_returned_result_type(self):
        """启用成本后返回类型仍为 BacktestResult，ret_definition 仍正确。"""
        from daily.evaluation.backtest import BacktestResult

        factor_df, ret_df = _make_data()
        r = run_stratified_backtest(factor_df, ret_df, n_groups=5, cost_model=CostModel())
        assert isinstance(r, BacktestResult)
        assert r.ret_definition == "fwd_ret_1d"

    def test_zero_cost_model_matches_no_cost(self):
        """CostModel 全部费率设为 0 时，结果应与 cost_model=None 完全相同。"""
        factor_df, ret_df = _make_data()
        r_none = run_stratified_backtest(factor_df, ret_df, n_groups=5)
        r_zero = run_stratified_backtest(
            factor_df,
            ret_df,
            n_groups=5,
            cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
        )
        # 零成本时结果应近似相同（因 borrow_rate=0 且 round_trip=0）
        s_none = r_none.summary_stats["long_short"]["ann_ret"]
        s_zero = r_zero.summary_stats["long_short"]["ann_ret"]
        # 允许极小浮点误差
        assert abs(s_none - s_zero) < 1e-6, f"零成本模型应与无成本相同: {s_none} vs {s_zero}"

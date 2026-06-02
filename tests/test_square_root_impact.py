"""平方根冲击成本模型单元测试。"""


from factorzen.daily.evaluation.cost_models import LinearCostModel, SquareRootImpactCostModel


class TestLinearCostModel:
    def test_zero_delta_returns_zero(self):
        m = LinearCostModel()
        assert m.trade_cost(0.0) == 0.0

    def test_buy_cost_has_no_stamp_tax(self):
        """买入无印花税。"""
        m = LinearCostModel(commission_rate=0.0003, stamp_tax_rate=0.001, slippage_rate=0.001)
        buy_cost = m.trade_cost(0.1)
        # commission + slippage only
        expected = 0.1 * (0.0003 + 0.001)
        assert abs(buy_cost - expected) < 1e-12

    def test_sell_cost_includes_stamp_tax(self):
        """卖出包含印花税。"""
        m = LinearCostModel(commission_rate=0.0003, stamp_tax_rate=0.001, slippage_rate=0.001)
        sell_cost = m.trade_cost(-0.1)
        expected = 0.1 * (0.0003 + 0.001 + 0.001)
        assert abs(sell_cost - expected) < 1e-12

    def test_sell_cost_greater_than_buy_cost(self):
        """卖出成本（含印花税）> 买入成本。"""
        m = LinearCostModel()
        buy_cost = m.trade_cost(0.1)
        sell_cost = m.trade_cost(-0.1)
        assert sell_cost > buy_cost

    def test_linear_cost_symmetric_abs(self):
        """|cost(+delta)| ≠ |cost(-delta)| (asymmetric due to stamp tax)."""
        m = LinearCostModel()
        assert m.trade_cost(0.5) != m.trade_cost(-0.5)

    def test_borrow_rate_daily(self):
        m = LinearCostModel(annual_borrow_rate=0.08, trading_days_per_year=252)
        expected = 0.08 / 252
        assert abs(m.borrow_rate_per_period("daily") - expected) < 1e-12

    def test_borrow_rate_weekly(self):
        m = LinearCostModel(annual_borrow_rate=0.08, trading_days_per_year=252)
        expected = 0.08 * 5 / 252
        assert abs(m.borrow_rate_per_period("weekly") - expected) < 1e-12


class TestSquareRootImpactCostModel:
    def test_zero_delta_returns_zero(self):
        m = SquareRootImpactCostModel()
        assert m.trade_cost(0.0) == 0.0

    def test_sqroot_cost_greater_than_linear_for_large_trades(self):
        """大交易时，平方根冲击成本 > 纯线性成本（有冲击项加成）。"""
        linear = LinearCostModel()
        sqroot = SquareRootImpactCostModel(alpha=0.1)
        large_delta = 0.5
        assert sqroot.trade_cost(large_delta) > linear.trade_cost(large_delta)

    def test_sqroot_cost_buy_positive(self):
        """买入成本 > 0。"""
        m = SquareRootImpactCostModel(alpha=0.1)
        assert m.trade_cost(0.1) > 0.0

    def test_sqroot_cost_sell_positive(self):
        """卖出成本 > 0。"""
        m = SquareRootImpactCostModel(alpha=0.1)
        assert m.trade_cost(-0.1) > 0.0

    def test_sqroot_is_superlinear(self):
        """平方根冲击：大交易边际成本率 > 小交易边际成本率（超线性）。"""
        m = SquareRootImpactCostModel(alpha=0.1)
        cost_small = m.trade_cost(0.01)
        cost_large = m.trade_cost(0.1)
        # 如果是纯线性，ratio = 10；超线性时 ratio > 10
        ratio = cost_large / cost_small
        assert ratio > 10, f"期望超线性 (ratio > 10)，实际 ratio={ratio:.3f}"

    def test_alpha_zero_equals_linear(self):
        """alpha=0 时，平方根模型退化为纯线性（无冲击项）。"""
        lin = LinearCostModel(commission_rate=0.0003, stamp_tax_rate=0.001, slippage_rate=0.001)
        sqr = SquareRootImpactCostModel(
            alpha=0.0,
            commission_rate=0.0003,
            stamp_tax_rate=0.001,
            slippage_rate=0.001,
        )
        delta = 0.3
        assert abs(sqr.trade_cost(delta) - lin.trade_cost(delta)) < 1e-12

    def test_adv_provided_reduces_impact(self):
        """提供更大的 ADV 时，冲击成本应降低（流动性更好）。"""
        m = SquareRootImpactCostModel(alpha=0.1, fallback_adv=1e7)
        cost_low_adv = m.trade_cost(0.1, adv=1e6)   # ADV 小（1e7 fallback / 1e6 = 10x smaller → higher impact）
        cost_high_adv = m.trade_cost(0.1, adv=1e8)  # ADV 大
        assert cost_high_adv < cost_low_adv, "ADV 越大，冲击成本应越小"

    def test_borrow_rate_daily(self):
        m = SquareRootImpactCostModel(annual_borrow_rate=0.08, trading_days_per_year=252)
        expected = 0.08 / 252
        assert abs(m.borrow_rate_per_period("daily") - expected) < 1e-12

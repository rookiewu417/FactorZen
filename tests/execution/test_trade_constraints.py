from factorzen.daily.evaluation.backtest import BacktestConfig
from factorzen.daily.evaluation.trade_constraints import apply_trade_constraints

CFG = BacktestConfig()  # limit_up_pct=9.8, max_participation_rate=0.05, fallback_adv=None


def _pm(open_, pre_close, vol):
    return {"X.SZ": {"open": open_, "pre_close": pre_close, "vol": vol}}


def test_normal_fill_passes_through():
    # 开盘 +2%，非停牌，无 adv → delta 原样返回
    d, r = apply_trade_constraints(
        code="X.SZ", delta=0.10, price_map=_pm(10.2, 10.0, 1e6), portfolio_value=1e6, config=CFG
    )
    assert (d, r) == (0.10, "")


def test_suspended_returns_zero():
    d, r = apply_trade_constraints(
        code="X.SZ", delta=0.10, price_map=_pm(10.2, 10.0, 0.0), portfolio_value=1e6, config=CFG
    )
    assert (d, r) == (0.0, "suspended")


def test_limit_up_blocks_buy():
    # 开盘 +9.9% ≥ 主板 9.8% 阈值，买单被拦
    d, r = apply_trade_constraints(
        code="X.SZ", delta=0.10, price_map=_pm(10.99, 10.0, 1e6), portfolio_value=1e6, config=CFG
    )
    assert (d, r) == (0.0, "limit_up")


def test_limit_down_blocks_sell():
    d, r = apply_trade_constraints(
        code="X.SZ", delta=-0.10, price_map=_pm(9.01, 10.0, 1e6), portfolio_value=1e6, config=CFG
    )
    assert (d, r) == (0.0, "limit_down")


def test_missing_price_returns_zero():
    d, r = apply_trade_constraints(
        code="X.SZ",
        delta=0.10,
        price_map={"X.SZ": {"open": None, "pre_close": 10.0, "vol": 1e6}},
        portfolio_value=1e6,
        config=CFG,
    )
    assert (d, r) == (0.0, "missing_price")


def test_capacity_caps_delta():
    # adv=1000万, 参与率5% → 最大成交额50万; portfolio_value=1000万 → max_delta=0.05
    d, r = apply_trade_constraints(
        code="X.SZ",
        delta=0.20,
        price_map=_pm(10.2, 10.0, 1e6),
        portfolio_value=1e7,
        config=CFG,
        adv=1e7,
    )
    assert r == "capacity"
    assert abs(d - 0.05) < 1e-12


def test_invalid_portfolio_value():
    d, r = apply_trade_constraints(
        code="X.SZ",
        delta=0.10,
        price_map=_pm(10.2, 10.0, 1e6),
        portfolio_value=0.0,
        config=CFG,
        adv=1e7,
    )
    assert (d, r) == (0.0, "invalid_portfolio_value")

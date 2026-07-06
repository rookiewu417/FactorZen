"""PaperBroker 对停牌（当日无行情）持仓须按最近已知价估值，而非按 0（P0-3）。

根因：get_cash 只用当日 market 的 close 估值，停牌股当日 daily 无行 → close 缺失 →
市值记 0 → NAV 凭空塌陷；且 engine.step 用被低估的 nav_before 重算目标股数 → 误卖其他
正常持仓、复牌后再买回。修复：保留每只持仓的最近已知价，缺当日行情时按最近价估值。
"""
from datetime import date

from factorzen.execution.broker import Order
from factorzen.execution.brokers.paper import PaperBroker


def _mkt(code, open_, pre_close, close, vol, adv=1e12):
    return {code: {"open": open_, "pre_close": pre_close, "close": close, "vol": vol, "adv": adv}}


def test_suspended_holding_valued_at_last_known_price():
    b = PaperBroker(initial_cash=1_000_000.0)
    # day1：买入 X.SZ 10000 股 @10 元（市值 10万）
    b.advance_to(date(2026, 1, 5), _mkt("X.SZ", 10.0, 10.0, 10.0, 1e6))
    b.place_orders([Order("X.SZ", "buy", 10000, "market", None)])
    b.poll_fills()
    nav_day1 = b.get_cash().total_asset
    assert abs(nav_day1 - 1_000_000.0) < 5_000.0  # ~1M（扣少量成本）

    # day2：X.SZ 停牌（当日 market 无该股行情）
    b.advance_to(date(2026, 1, 6), {})  # 空 market = 停牌无行
    nav_day2 = b.get_cash().total_asset
    assert abs(nav_day2 - nav_day1) < 1e-6, (
        f"停牌日 NAV 应稳定在 {nav_day1:.0f}（持仓按最近价 10 估值），"
        f"实得 {nav_day2:.0f}（修复前按 0 估值→塌陷）"
    )
    # 市值应体现 10000*10=10万，而非 0
    assert abs(b.get_cash().market_value - 100_000.0) < 1e-6


def test_last_price_survives_state_roundtrip():
    """最近价须随 state 持久化，续跑(load_state)后停牌估值仍正确。"""
    b = PaperBroker(initial_cash=1_000_000.0)
    b.advance_to(date(2026, 1, 5), _mkt("X.SZ", 10.0, 10.0, 10.0, 1e6))
    b.place_orders([Order("X.SZ", "buy", 10000, "market", None)])
    b.poll_fills()
    st = b.state()

    b2 = PaperBroker(initial_cash=1_000_000.0)
    b2.load_state(st)
    b2.advance_to(date(2026, 1, 6), {})  # 停牌
    assert abs(b2.get_cash().market_value - 100_000.0) < 1e-6, (
        "续跑后停牌股应按最近价估值；最近价须随 state 持久化"
    )

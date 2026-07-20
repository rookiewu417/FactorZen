"""合并自: test_paper_frictionless.py, test_paper_suspended_valuation.py, test_paper_broker.py, test_broker.py, test_empty_target_liquidates.py
目标: test_paper_broker.py

--- 来源 test_paper_frictionless.py ---
(无原 docstring)

--- 来源 test_paper_suspended_valuation.py ---
PaperBroker 对停牌（当日无行情）持仓须按最近已知价估值，而非按 0（P0-3）。

根因：get_cash 只用当日 market 的 close 估值，停牌股当日 daily 无行 → close 缺失 →
市值记 0 → NAV 凭空塌陷；且 engine.step 用被低估的 nav_before 重算目标股数 → 误卖其他
正常持仓、复牌后再买回。修复：保留每只持仓的最近已知价，缺当日行情时按最近价估值。

--- 来源 test_paper_broker.py ---
(无原 docstring)

--- 来源 test_broker.py ---
(无原 docstring)

--- 来源 test_empty_target_liquidates.py ---
空目标信号 = 清仓语义：applicable 但为空 → 正常 step 清仓；真·无 applicable 才跳过。
"""

import json
from datetime import date

import polars as pl
import pytest

from factorzen.execution.broker import (
    BrokerAdapter,
    Order,
    round_lot,
)
from factorzen.execution.brokers.paper import PaperBroker
from factorzen.execution.drivers import run_replay
from factorzen.execution.store import SessionStore


# ==== 来自 test_paper_frictionless.py ====
def _mkt__paper_frictionless(open_, pre_close, close, vol, adv=1e12):
    return {"X.SZ": {"open": open_, "pre_close": pre_close, "close": close, "vol": vol, "adv": adv}}

def test_frictionless_and_mark_suite():
    """test_frictionless_fills_suspended_fully_at_close；test_frictionless_ignores_cash_and_lot；test_suspended_holding_valued_at_last_known_price；最近价须随 state 持久化，续跑(load_state)后停牌估值仍正确。"""
    # -- 原 test_frictionless_fills_suspended_fully_at_close --
    def _section_0_test_frictionless_fills_suspended_fully_at_close():
        b = PaperBroker(initial_cash=1_000_000.0, frictionless=True)
        b.advance_to(date(2026, 1, 5), _mkt__paper_frictionless(10.0, 10.0, 11.0, 0.0))  # vol=0 停牌
        b.place_orders([Order("X.SZ", "buy", 1000, "market", None)])
        f = b.poll_fills()[0]
        assert f.filled_volume == 1000        # 停牌也全额（frictionless）
        assert abs(f.price - 11.0) < 1e-9     # 按 close 成交
        assert f.cost == 0.0                  # 零成本

    _section_0_test_frictionless_fills_suspended_fully_at_close()

    # -- 原 test_frictionless_ignores_cash_and_lot --
    def _section_1_test_frictionless_ignores_cash_and_lot():
        b = PaperBroker(initial_cash=500.0, frictionless=True)  # 现金远不够
        b.advance_to(date(2026, 1, 5), _mkt__paper_frictionless(10.0, 10.0, 10.0, 1e6))
        b.place_orders([Order("X.SZ", "buy", 1550, "market", None)])  # 非整百
        assert b.poll_fills()[0].filled_volume == 1550   # 不整手、不受现金限
        assert b.get_cash().available < 0                # 现金可为负

    _section_1_test_frictionless_ignores_cash_and_lot()

    # -- 原 test_suspended_holding_valued_at_last_known_price --
    def _section_2_test_suspended_holding_valued_at_last_known_price():
        b = PaperBroker(initial_cash=1_000_000.0)
        # day1：买入 X.SZ 10000 股 @10 元（市值 10万）
        b.advance_to(date(2026, 1, 5), _mkt__paper_suspended_valuation("X.SZ", 10.0, 10.0, 10.0, 1e6))
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

    _section_2_test_suspended_holding_valued_at_last_known_price()

    # -- 原 test_last_price_survives_state_roundtrip --
    def _section_3_test_last_price_survives_state_roundtrip():
        b = PaperBroker(initial_cash=1_000_000.0)
        b.advance_to(date(2026, 1, 5), _mkt__paper_suspended_valuation("X.SZ", 10.0, 10.0, 10.0, 1e6))
        b.place_orders([Order("X.SZ", "buy", 10000, "market", None)])
        b.poll_fills()
        st = b.state()

        b2 = PaperBroker(initial_cash=1_000_000.0)
        b2.load_state(st)
        b2.advance_to(date(2026, 1, 6), {})  # 停牌
        assert abs(b2.get_cash().market_value - 100_000.0) < 1e-6, (
            "续跑后停牌股应按最近价估值；最近价须随 state 持久化"
        )

    _section_3_test_last_price_survives_state_roundtrip()


# ==== 来自 test_paper_suspended_valuation.py ====
def _mkt__paper_suspended_valuation(code, open_, pre_close, close, vol, adv=1e12):
    return {code: {"open": open_, "pre_close": pre_close, "close": close, "vol": vol, "adv": adv}}


# ==== 来自 test_paper_broker.py ====
def _mkt__paper_broker(open_, pre_close, close, vol, adv=1e12):
    # adv 极大 → 容量不绑定；聚焦其它摩擦
    return {"X.SZ": {"open": open_, "pre_close": pre_close, "close": close, "vol": vol, "adv": adv}}


def test_paper_broker_rules_suite():
    """test_buy_fill_updates_cash_and_position；test_suspended_rejects_order；test_limit_up_rejects_buy；test_lot_rounding_drops_remainder；test_insufficient_cash_caps_buy；test_t1_frozen_blocks_same_day_sell；test_total_asset_marks_to_close；test_round_lot_floors_to_hundred；权重空间往返(shares→delta_w→shares)的浮点 ulp 不应吃掉整手。"""
    # -- 原 test_buy_fill_updates_cash_and_position --
    def _section_0_test_buy_fill_updates_cash_and_position():
        b = PaperBroker(initial_cash=1_000_000.0)
        b.advance_to(date(2026, 1, 5), _mkt__paper_broker(10.0, 10.0, 10.0, 1e6))
        acks = b.place_orders([Order("X.SZ", "buy", 1000, "market", None)])
        fills = b.poll_fills()
        assert acks[0].accepted and fills[0].filled_volume == 1000
        pos = b.get_positions()["X.SZ"]
        assert pos.volume == 1000
        # 现金 = 100万 - 1000*10 - 成本
        assert b.get_cash().available < 1_000_000.0 - 10_000.0 + 1e-6

    _section_0_test_buy_fill_updates_cash_and_position()

    # -- 原 test_suspended_rejects_order --
    def _section_1_test_suspended_rejects_order():
        b = PaperBroker(initial_cash=1_000_000.0)
        b.advance_to(date(2026, 1, 5), _mkt__paper_broker(10.0, 10.0, 10.0, 0.0))  # vol=0 停牌
        acks = b.place_orders([Order("X.SZ", "buy", 1000, "market", None)])
        assert not acks[0].accepted and acks[0].reason == "suspended"
        assert b.poll_fills() == []
        assert "X.SZ" not in b.get_positions()

    _section_1_test_suspended_rejects_order()

    # -- 原 test_limit_up_rejects_buy --
    def _section_2_test_limit_up_rejects_buy():
        b = PaperBroker(initial_cash=1_000_000.0)
        b.advance_to(date(2026, 1, 5), _mkt__paper_broker(10.99, 10.0, 11.0, 1e6))  # 开盘+9.9%涨停
        acks = b.place_orders([Order("X.SZ", "buy", 1000, "market", None)])
        assert not acks[0].accepted and acks[0].reason == "limit_up"

    _section_2_test_limit_up_rejects_buy()

    # -- 原 test_lot_rounding_drops_remainder --
    def _section_3_test_lot_rounding_drops_remainder():
        b = PaperBroker(initial_cash=1_000_000.0)
        b.advance_to(date(2026, 1, 5), _mkt__paper_broker(10.0, 10.0, 10.0, 1e6))
        # 下 150 股 → 整手向零取整到 100
        acks = b.place_orders([Order("X.SZ", "buy", 150, "market", None)])
        assert b.poll_fills()[0].filled_volume == 100
        assert acks[0].reason == "lot_round"

    _section_3_test_lot_rounding_drops_remainder()

    # -- 原 test_insufficient_cash_caps_buy --
    def _section_4_test_insufficient_cash_caps_buy():
        b = PaperBroker(initial_cash=1_050.0)  # 只够 100 股(1000元)+成本
        b.advance_to(date(2026, 1, 5), _mkt__paper_broker(10.0, 10.0, 10.0, 1e6))
        b.place_orders([Order("X.SZ", "buy", 1000, "market", None)])
        fill = b.poll_fills()[0]
        assert fill.filled_volume == 100 and b.get_cash().available >= 0.0

    _section_4_test_insufficient_cash_caps_buy()

    # -- 原 test_t1_frozen_blocks_same_day_sell --
    def _section_5_test_t1_frozen_blocks_same_day_sell():
        b = PaperBroker(initial_cash=1_000_000.0)
        b.advance_to(date(2026, 1, 5), _mkt__paper_broker(10.0, 10.0, 10.0, 1e6))
        b.place_orders([Order("X.SZ", "buy", 1000, "market", None)])
        b.poll_fills()
        # 同日卖：can_use_volume 当日买入部分为 0（T+1）
        assert b.get_positions()["X.SZ"].can_use_volume == 0
        acks = b.place_orders([Order("X.SZ", "sell", 1000, "market", None)])
        assert not acks[0].accepted and acks[0].reason == "t1_frozen"

    _section_5_test_t1_frozen_blocks_same_day_sell()

    # -- 原 test_total_asset_marks_to_close --
    def _section_6_test_total_asset_marks_to_close():
        b = PaperBroker(initial_cash=1_000_000.0)
        b.advance_to(date(2026, 1, 5), _mkt__paper_broker(10.0, 10.0, 12.0, 1e6))  # close=12
        b.place_orders([Order("X.SZ", "buy", 1000, "market", None)])
        b.poll_fills()
        cash = b.get_cash()
        # 持仓市值按 close=12 标记 = 1000*12 = 12000
        assert abs(cash.market_value - 12_000.0) < 1e-6

    _section_6_test_total_asset_marks_to_close()

    # -- 原 test_round_lot_floors_to_hundred --
    def _section_7_test_round_lot_floors_to_hundred():
        assert round_lot(150) == 100
        assert round_lot(199.9) == 100
        assert round_lot(-150) == -100      # 卖单同向缩小
        assert round_lot(50) == 0
        assert round_lot(300) == 300

    _section_7_test_round_lot_floors_to_hundred()

    # -- 原 test_round_lot_absorbs_float_ulp --
    def _section_8_test_round_lot_absorbs_float_ulp():
        assert round_lot(12899.999999999998) == 12900
        assert round_lot(-12899.999999999998) == -12900
        assert round_lot(9999.99999999999) == 10000
        # 真实的非整手小数仍向零取整（ulp 容差远小于 1 股）
        assert round_lot(12950.4) == 12900
        assert round_lot(12899.5) == 12800

    _section_8_test_round_lot_absorbs_float_ulp()


# ==== 来自 test_broker.py ====


def test_paper_broker_signal_edge_suite(tmp_path):
    """test_broker_adapter_is_abstract；test_empty_applicable_signal_liquidates；test_no_applicable_signal_still_skips"""
    # -- 原 test_broker_adapter_is_abstract --
    def _section_0_test_broker_adapter_is_abstract():
        with pytest.raises(TypeError):
            BrokerAdapter()  # 抽象类不可实例化

    _section_0_test_broker_adapter_is_abstract()

    # -- 原 test_empty_applicable_signal_liquidates --
    def _section_1_test_empty_applicable_signal_liquidates(tmp_path):
        d1, d2, d3 = date(2026,1,5), date(2026,1,6), date(2026,1,7)
        daily = _daily([d1,d2,d3], "A.SZ")
        buy = _pf(tmp_path/"buy", d1, {"A.SZ": 0.9})          # d1 建仓
        empty = _pf(tmp_path/"empty", d2, {})                 # d2 空目标 → 应清仓
        run_replay(session_dir=tmp_path/"s", portfolio_run_dirs=[buy, empty], daily=daily,
                   initial_cash=1_000_000.0, from_date=d1, to_date=d3, seed=0)
        store = SessionStore(tmp_path/"s")
        ledger = store.ledger_records()
        # 次日执行(s<d)：d1 建仓信号于 d2 生效建仓，d2 空目标信号于 d3 生效清仓。
        d3rec = next(r for r in ledger if r["as_of_date"] == d3.isoformat())
        sells = [o for o in d3rec["orders"] if o.get("side") == "sell" and o.get("ts_code") == "A.SZ"]
        assert sells, f"d3 空目标应有清仓卖单, 实际 orders={d3rec['orders']}"
        # d3 起（持久化的 broker 续跑态）持仓为空
        bs = store.load_state()
        held = {c: p for c, p in bs.get("pos", bs.get("positions", {})).items()
                if (p.get("volume", 0) if isinstance(p, dict) else 0) > 0}
        assert held == {}, f"空目标后应清仓, 实际 {held}"

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_empty_applicable_signal_liquidates(_tp1)

    # -- 原 test_no_applicable_signal_still_skips --
    def _section_2_test_no_applicable_signal_still_skips(tmp_path):
        d1, d2 = date(2026,1,5), date(2026,1,6)
        daily = _daily([d1,d2], "A.SZ")
        late = _pf(tmp_path/"late", d2, {"A.SZ": 0.9})        # 信号日 d2
        run_replay(session_dir=tmp_path/"s", portfolio_run_dirs=[late], daily=daily,
                   initial_cash=1_000_000.0, from_date=d1, to_date=d1, seed=0)  # 只跑 d1
        # d1 无适用信号(信号 d2>d1) → 跳过, ledger 空
        assert SessionStore(tmp_path/"s").nav_frame().height == 0

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_no_applicable_signal_still_skips(_tp2)


# ==== 来自 test_empty_target_liquidates.py ====
def _pf(dir_, sig, weights: dict):
    dir_.mkdir(parents=True, exist_ok=True)
    codes = list(weights)
    ws = [weights[c] for c in codes]
    pl.DataFrame({"ts_code": codes, "target_weight": ws},
                 schema={"ts_code": pl.Utf8, "target_weight": pl.Float64}).write_parquet(dir_/"weights.parquet")
    (dir_/"manifest.json").write_text(json.dumps({"signal_date": sig.isoformat(), "status": "optimal"}))
    return str(dir_)

def _daily(dates, code):
    return pl.DataFrame([{"trade_date": d, "ts_code": code, "open": 10.0, "pre_close": 10.0,
                          "close": 10.0, "vol": 1e8, "amount": 1e9} for d in dates])



"""合并自: test_store.py, test_store_acks.py, test_engine.py, test_trade_constraints.py, test_daily_step_guards.py
目标: test_store_engine.py

--- 来源 test_store.py ---
(无原 docstring)

--- 来源 test_store_acks.py ---
(无原 docstring)

--- 来源 test_engine.py ---
(无原 docstring)

--- 来源 test_trade_constraints.py ---
(无原 docstring)

--- 来源 test_daily_step_guards.py ---
run_daily_step 的交易日历守卫(E3) + 日期单调性守卫(E2)。

E3：as_of 为非交易日(daily 无该日行)时 market 为空，若照常 step 会落一条纯现金塌陷 nav
   行且被 has_date 永久锁死、无法修复。须直接跳过不落盘。
E2：resume 无日期单调性守卫时，补跑早于已推进日期的 as_of 会用「未来的」broker 状态步进
   过去，ledger 乱序、state 被污染。须拒绝。
"""

import json
from datetime import date

import polars as pl

from factorzen.daily.evaluation.backtest import BacktestConfig
from factorzen.daily.evaluation.trade_constraints import apply_trade_constraints
from factorzen.execution.broker import BrokerAdapter, Cash, Fill, OrderAck, Position
from factorzen.execution.drivers import run_daily_step
from factorzen.execution.engine import build_orders, step
from factorzen.execution.store import SessionStore


# ==== 来自 test_store.py ====
def _rec__store(d, nav, bstate):
    return {
        "as_of_date": d.isoformat(),
        "nav_before": nav,
        "nav_after": nav,
        "broker_state": bstate,
        "orders": [],
        "fills": [],
    }

def test_execution_store_suite(tmp_path):
    """test_append_and_idempotency；test_resume_reads_latest_state；已有会话再 init（如 fz live replay 复用 session）不应覆盖原 config——；test_append_persists_acks_and_reads_back；test_ledger_records_backward_compat_no_acks"""
    # -- 原 test_append_and_idempotency --
    def _section_0_test_append_and_idempotency(tmp_path):
        s = SessionStore(tmp_path / "sess1")
        s.init({"broker": "paper", "initial_cash": 1e6})
        d = date(2026, 1, 5)
        assert not s.has_date(d)
        s.append(_rec__store(d, 1e6, {"cash": 1e6, "pos": {}, "order_seq": 0}))
        assert s.has_date(d)  # 幂等哨兵
        assert s.load_state()["cash"] == 1e6
        assert s.nav_frame().height == 1

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_append_and_idempotency(_tp0)

    # -- 原 test_resume_reads_latest_state --
    def _section_1_test_resume_reads_latest_state(tmp_path):
        s = SessionStore(tmp_path / "sess1")
        s.init({"broker": "paper", "initial_cash": 1e6})
        s.append(_rec__store(date(2026, 1, 5), 1e6, {"cash": 9e5, "pos": {}, "order_seq": 2}))
        s2 = SessionStore(tmp_path / "sess1")  # 新实例重载
        assert s2.load_state()["cash"] == 9e5
        assert s2.has_date(date(2026, 1, 5))

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_resume_reads_latest_state(_tp1)

    # -- 原 test_init_preserves_existing_manifest_config --
    def _section_2_test_init_preserves_existing_manifest_config(tmp_path):
        import json

        s = SessionStore(tmp_path / "sess")
        s.init({"broker": "paper", "initial_cash": 2_000_000.0, "slippage_bps": 5.0})
        # 模拟 replay 用默认 config 再 init 同一 session
        s.init({"broker": "paper", "initial_cash": 1_000_000.0})
        cfg = json.loads((tmp_path / "sess" / "manifest.json").read_text())["config"]
        assert cfg["slippage_bps"] == 5.0, "已有会话的 slippage_bps 不应被覆盖"
        assert cfg["initial_cash"] == 2_000_000.0

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_init_preserves_existing_manifest_config(_tp2)

    # -- 原 test_append_persists_acks_and_reads_back --
    def _section_3_test_append_persists_acks_and_reads_back(tmp_path):
        s = SessionStore(tmp_path / "sess")
        s.init({"broker": "paper", "initial_cash": 1e6})
        orders = [{"ts_code": "X.SZ", "side": "buy", "volume": 1000, "price_type": "market", "price": None}]
        acks = [{"order_id": "paper-1", "ts_code": "X.SZ", "accepted": False, "reason": "suspended"}]
        fills = []
        s.append(_rec__store_acks(date(2026, 1, 5), orders, acks, fills, {"cash": 1e6, "pos": {}, "order_seq": 1}))
        recs = s.ledger_records()
        assert len(recs) == 1
        assert recs[0]["acks"][0]["reason"] == "suspended"
        assert recs[0]["orders"][0]["ts_code"] == "X.SZ"

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_append_persists_acks_and_reads_back(_tp3)

    # -- 原 test_ledger_records_backward_compat_no_acks --
    def _section_4_test_ledger_records_backward_compat_no_acks(tmp_path):
        import json

        import polars as pl

        d = tmp_path / "sess"
        d.mkdir(parents=True)
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-01-05",
                    "nav_before": 1e6,
                    "nav_after": 1e6,
                    "payload": json.dumps({"orders": [], "fills": []}),
                }
            ]
        ).write_parquet(d / "ledger.parquet")
        recs = SessionStore(d).ledger_records()
        assert recs[0]["acks"] == []  # 旧无 acks → 空

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_ledger_records_backward_compat_no_acks(_tp4)


# ==== 来自 test_store_acks.py ====
def _rec__store_acks(d, orders, acks, fills, bstate):
    return {
        "as_of_date": d.isoformat(),
        "nav_before": 1e6,
        "nav_after": 1e6,
        "broker_state": bstate,
        "orders": orders,
        "acks": acks,
        "fills": fills,
    }


# ==== 来自 test_engine.py ====
def test_build_orders_suite():
    """test_build_orders_sell_before_buy；test_build_orders_skips_zero_delta"""
    # -- 原 test_build_orders_sell_before_buy --
    def _section_0_test_build_orders_sell_before_buy():
        positions = {"A.SZ": Position("A.SZ", 1000, 1000, 10.0)}  # 现持 A 1000
        target = {"A.SZ": 300, "B.SZ": 500}  # A 减到 300, B 建 500
        orders = build_orders(target, positions)
        sides = [(o.ts_code, o.side, o.volume) for o in orders]
        # 卖单必须排在买单前
        assert sides.index(("A.SZ", "sell", 700)) < sides.index(("B.SZ", "buy", 500))

    _section_0_test_build_orders_sell_before_buy()

    # -- 原 test_build_orders_skips_zero_delta --
    def _section_1_test_build_orders_skips_zero_delta():
        positions = {"A.SZ": Position("A.SZ", 300, 300, 10.0)}
        assert build_orders({"A.SZ": 300}, positions) == []

    _section_1_test_build_orders_skips_zero_delta()


class FakeBroker(BrokerAdapter):
    def __init__(self):
        self._acks = []
        self._orders = []
        self._positions: dict[str, Position] = {}
        self._cash = Cash(1_000_000.0, 1_000_000.0, 0.0)

    def get_positions(self):
        return self._positions

    def get_cash(self):
        return self._cash

    def place_orders(self, orders):
        self._orders = orders
        self._acks = [OrderAck(f"o{i}", o.ts_code, True, "") for i, o in enumerate(orders)]
        return self._acks

    def poll_fills(self):
        fills = [
            Fill(f"o{i}", o.ts_code, o.side, o.volume, 10.0, 0.0, date(2026, 1, 5))
            for i, o in enumerate(self._orders)
        ]
        # 模拟成交落地：更新持仓/现金，供 broker_state 断言用（不是 step() 自身的镜像计算）。
        for order, fill in zip(self._orders, fills, strict=True):
            cur = self._positions.get(order.ts_code)
            cur_vol = cur.volume if cur else 0
            signed = fill.filled_volume if order.side == "buy" else -fill.filled_volume
            new_vol = cur_vol + signed
            self._positions[order.ts_code] = Position(order.ts_code, new_vol, new_vol, fill.price)
            cost = fill.filled_volume * fill.price
            delta_available = -cost if order.side == "buy" else cost
            delta_market_value = cost if order.side == "buy" else -cost
            self._cash = Cash(
                self._cash.available + delta_available,
                self._cash.total_asset,
                self._cash.market_value + delta_market_value,
            )
        return fills


def test_engine_step_suite():
    """test_step_sizes_target_shares_from_nav；test_step_record_includes_broker_state"""
    # -- 原 test_step_sizes_target_shares_from_nav --
    def _section_0_test_step_sizes_target_shares_from_nav():
        b = FakeBroker()
        # NAV=100万, 目标权重 A=0.3, ref_price=10 → 目标股数 = round_lot(0.3*1e6/10)=30000
        rec = step(b, {"A.SZ": 0.3}, {"A.SZ": 10.0})
        buy = next(o for o in b._orders if o.ts_code == "A.SZ")
        assert buy.volume == 30000 and buy.side == "buy"
        assert rec["nav_before"] == 1_000_000.0
        assert len(rec["fills"]) == 1

    _section_0_test_step_sizes_target_shares_from_nav()

    # -- 原 test_step_record_includes_broker_state --
    def _section_1_test_step_record_includes_broker_state():
        b = FakeBroker()
        rec = step(b, {"A.SZ": 0.3}, {"A.SZ": 10.0})
        # broker_state 是 step() 专为满足 store.append() 契约而附带的字段，结构需锁死。
        assert set(rec["broker_state"]) == {"positions", "cash"}
        assert set(rec["broker_state"]["cash"]) == {"available", "total_asset", "market_value"}
        positions = rec["broker_state"]["positions"]
        assert set(positions) == {"A.SZ"}
        assert set(positions["A.SZ"]) == {"ts_code", "volume", "can_use_volume", "avg_cost"}
        # 买入 30000 股 @10 后，broker_state 须真实反映持仓与现金变化（非恒真占位）。
        assert positions["A.SZ"]["volume"] == 30000
        assert rec["broker_state"]["cash"]["available"] == 1_000_000.0 - 30000 * 10.0

    _section_1_test_step_record_includes_broker_state()


# ==== 来自 test_trade_constraints.py ====
CFG = BacktestConfig()  # limit_up_pct=9.8, max_participation_rate=0.05, fallback_adv=None


def _pm(open_, pre_close, vol):
    return {"X.SZ": {"open": open_, "pre_close": pre_close, "vol": vol}}


def test_fill_constraints_suite():
    """test_normal_fill_passes_through；test_suspended_returns_zero；test_limit_up_blocks_buy；test_limit_down_blocks_sell；test_missing_price_returns_zero；test_capacity_caps_delta；test_invalid_portfolio_value"""
    # -- 原 test_normal_fill_passes_through --
    def _section_0_test_normal_fill_passes_through():
        d, r = apply_trade_constraints(
            code="X.SZ", delta=0.10, price_map=_pm(10.2, 10.0, 1e6), portfolio_value=1e6, config=CFG
        )
        assert (d, r) == (0.10, "")

    _section_0_test_normal_fill_passes_through()

    # -- 原 test_suspended_returns_zero --
    def _section_1_test_suspended_returns_zero():
        d, r = apply_trade_constraints(
            code="X.SZ", delta=0.10, price_map=_pm(10.2, 10.0, 0.0), portfolio_value=1e6, config=CFG
        )
        assert (d, r) == (0.0, "suspended")

    _section_1_test_suspended_returns_zero()

    # -- 原 test_limit_up_blocks_buy --
    def _section_2_test_limit_up_blocks_buy():
        d, r = apply_trade_constraints(
            code="X.SZ", delta=0.10, price_map=_pm(10.99, 10.0, 1e6), portfolio_value=1e6, config=CFG
        )
        assert (d, r) == (0.0, "limit_up")

    _section_2_test_limit_up_blocks_buy()

    # -- 原 test_limit_down_blocks_sell --
    def _section_3_test_limit_down_blocks_sell():
        d, r = apply_trade_constraints(
            code="X.SZ", delta=-0.10, price_map=_pm(9.01, 10.0, 1e6), portfolio_value=1e6, config=CFG
        )
        assert (d, r) == (0.0, "limit_down")

    _section_3_test_limit_down_blocks_sell()

    # -- 原 test_missing_price_returns_zero --
    def _section_4_test_missing_price_returns_zero():
        d, r = apply_trade_constraints(
            code="X.SZ",
            delta=0.10,
            price_map={"X.SZ": {"open": None, "pre_close": 10.0, "vol": 1e6}},
            portfolio_value=1e6,
            config=CFG,
        )
        assert (d, r) == (0.0, "missing_price")

    _section_4_test_missing_price_returns_zero()

    # -- 原 test_capacity_caps_delta --
    def _section_5_test_capacity_caps_delta():
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

    _section_5_test_capacity_caps_delta()

    # -- 原 test_invalid_portfolio_value --
    def _section_6_test_invalid_portfolio_value():
        d, r = apply_trade_constraints(
            code="X.SZ",
            delta=0.10,
            price_map=_pm(10.2, 10.0, 1e6),
            portfolio_value=0.0,
            config=CFG,
            adv=1e7,
        )
        assert (d, r) == (0.0, "invalid_portfolio_value")

    _section_6_test_invalid_portfolio_value()


# ==== 来自 test_daily_step_guards.py ====
def _pf(dir_, sig, code, w):
    dir_.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"ts_code": [code], "target_weight": [w]}).write_parquet(dir_ / "weights.parquet")
    (dir_ / "manifest.json").write_text(json.dumps({"signal_date": sig.isoformat(), "status": "optimal"}))
    return str(dir_)


def _daily(dates, code):
    return pl.DataFrame([{"trade_date": d, "ts_code": code, "open": 10.0, "pre_close": 10.0,
                          "close": 10.0, "vol": 1e8, "amount": 1e9} for d in dates])


def test_engine_guards_suite(tmp_path):
    """test_non_trading_day_is_skipped_not_recorded；test_backwards_as_of_rejected；崩溃恢复：state._last_as_of 与 ledger 末行不一致（ledger 写完、state 未写完就崩）"""
    # -- 原 test_non_trading_day_is_skipped_not_recorded --
    def _section_0_test_non_trading_day_is_skipped_not_recorded(tmp_path):
        d1, holiday = date(2026, 1, 5), date(2026, 1, 10)  # 1/10 非交易日(daily 无行)
        daily = _daily([d1], "A.SZ")
        rd = _pf(tmp_path / "pf", date(2026, 1, 2), "A.SZ", 0.5)
        cfg = {"initial_cash": 1_000_000.0}
        sess = tmp_path / "sess"
        SessionStore(sess).init({"broker": "paper", **cfg})
        run_daily_step(sess, d1, [rd], daily, config=cfg)  # 正常执行日

        r = run_daily_step(sess, holiday, [rd], daily, config=cfg)
        assert r["skipped"], "非交易日应跳过"
        # 不应落塌陷 nav 行；ledger 只含 d1
        got = SessionStore(sess).nav_frame()["as_of_date"].to_list()
        assert holiday.isoformat() not in got, "非交易日不应落 ledger 行（否则被 has_date 永久锁死）"

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_non_trading_day_is_skipped_not_recorded(_tp0)

    # -- 原 test_backwards_as_of_rejected --
    def _section_1_test_backwards_as_of_rejected(tmp_path):
        d5, d6, d7 = date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)
        daily = _daily([d5, d6, d7], "A.SZ")
        rd = _pf(tmp_path / "pf", date(2026, 1, 2), "A.SZ", 0.5)
        cfg = {"initial_cash": 1_000_000.0}
        sess = tmp_path / "sess"
        SessionStore(sess).init({"broker": "paper", **cfg})

        run_daily_step(sess, d5, [rd], daily, config=cfg)
        run_daily_step(sess, d7, [rd], daily, config=cfg)  # 已推进到 d7
        nav_before = SessionStore(sess).nav_frame()["as_of_date"].to_list()

        # 补跑 d6（早于已推进的 d7）→ 应拒绝，不污染 ledger
        r = run_daily_step(sess, d6, [rd], daily, config=cfg)
        assert r["skipped"], "早于已推进日期的补跑应被拒绝"
        nav_after = SessionStore(sess).nav_frame()["as_of_date"].to_list()
        assert nav_after == nav_before, "乱序补跑不应改动 ledger"

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_backwards_as_of_rejected(_tp1)

    # -- 原 test_inconsistent_state_ledger_raises --
    def _section_2_test_inconsistent_state_ledger_raises(tmp_path):
        import pytest

        d5, d6, d7 = date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)
        daily = _daily([d5, d6, d7], "A.SZ")
        rd = _pf(tmp_path / "pf", date(2026, 1, 2), "A.SZ", 0.5)
        cfg = {"initial_cash": 1_000_000.0}
        sess = tmp_path / "sess"
        SessionStore(sess).init({"broker": "paper", **cfg})
        run_daily_step(sess, d5, [rd], daily, config=cfg)
        run_daily_step(sess, d6, [rd], daily, config=cfg)

        # 手动制造不一致：state 回退到 d5（模拟 d6 的 ledger 写完但 state 未更新就崩）
        st = SessionStore(sess).load_state()
        st["_last_as_of"] = d5.isoformat()
        (sess / "state.json").write_text(json.dumps(st))

        with pytest.raises(RuntimeError, match="不一致"):
            run_daily_step(sess, d7, [rd], daily, config=cfg)

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_inconsistent_state_ledger_raises(_tp2)



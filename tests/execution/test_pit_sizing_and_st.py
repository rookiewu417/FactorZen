"""PIT 修复：#2 定量用 pre_close（非执行日 close）；#6 逐日 ST 收窄涨跌停。

旧行为：
- drivers 用 m["close"] 作 ref_price → 前视（执行日收盘价决策时未知）
- PaperBroker 不传 is_st → ST 股用主板 9.8% 宽阈值
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl

from factorzen.execution.broker import Order, round_lot
from factorzen.execution.brokers.paper import PaperBroker
from factorzen.execution.drivers import run_daily_step
from factorzen.execution.store import SessionStore


def _pf(dir_: Path, sig: date, code: str, w: float) -> str:
    dir_.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"ts_code": [code], "target_weight": [w]}).write_parquet(dir_ / "weights.parquet")
    (dir_ / "manifest.json").write_text(
        json.dumps({"signal_date": sig.isoformat(), "status": "optimal"})
    )
    return str(dir_)


# ── #2 pre_close 定量 ──────────────────────────────────────────────


def test_daily_step_sizes_target_shares_with_pre_close_not_close(tmp_path: Path) -> None:
    """执行日 pre_close 与 close 明显不同时，目标股数须按 pre_close 算（非 close）。

    pre_close=10、close=13、w=0.5、nav=1e6：
      pre_close → round_lot(0.5*1e6/10)=50000
      close     → round_lot(0.5*1e6/13)=38400
    旧实现用 close → 红。
    """
    sig, exec_d = date(2026, 1, 5), date(2026, 1, 6)
    code = "A.SZ"
    pre_close, close_px, open_px = 10.0, 13.0, 10.0
    daily = pl.DataFrame(
        [
            {
                "trade_date": sig,
                "ts_code": code,
                "open": 10.0,
                "pre_close": 10.0,
                "close": 10.0,
                "vol": 1e8,
                "amount": 1e9,
            },
            {
                "trade_date": exec_d,
                "ts_code": code,
                "open": open_px,
                "pre_close": pre_close,
                "close": close_px,
                "vol": 1e8,
                "amount": 1e9,
            },
        ]
    )
    rd = _pf(tmp_path / "pf", sig, code, 0.5)
    cfg = {"initial_cash": 1_000_000.0, "slippage_bps": 0.0}
    sess = tmp_path / "sess"
    SessionStore(sess).init({"broker": "paper", **cfg})

    r = run_daily_step(sess, exec_d, [rd], daily, config=cfg)
    assert not r["skipped"] and r["n_fills"] >= 1

    recs = SessionStore(sess).ledger_records()
    assert len(recs) == 1
    orders = recs[0]["orders"]
    buy = next(o for o in orders if o["ts_code"] == code and o["side"] == "buy")

    expected_pre = round_lot(0.5 * 1_000_000.0 / pre_close)  # 50000
    expected_close = round_lot(0.5 * 1_000_000.0 / close_px)  # 38400
    assert expected_pre != expected_close  # 判别力：两价必须拉开
    assert buy["volume"] == expected_pre, (
        f"定量须用 pre_close={pre_close} → {expected_pre}，"
        f"勿用 close={close_px} → {expected_close}；实际 volume={buy['volume']}"
    )


# ── #6 ST 收窄涨跌停 ──────────────────────────────────────────────


def test_paper_broker_st_limit_up_rejects_buy_when_is_st() -> None:
    """主板 ST 阈值 4.8%：开盘 +6% 对 ST 已涨停拒买；非 ST 未涨停应成交。

    旧实现 PaperBroker 不传 is_st → 恒用 9.8% → 6% 不拒 → 红。
    """
    # open/pre_close = 10.6/10 → +6%，落在 (4.8%, 9.8%) 之间
    open_px, pre_close = 10.6, 10.0
    code = "600001.SH"  # 主板

    # ST：应收窄并拒
    b_st = PaperBroker(initial_cash=1_000_000.0)
    b_st.advance_to(
        date(2026, 1, 5),
        {
            code: {
                "open": open_px,
                "pre_close": pre_close,
                "close": open_px,
                "vol": 1e6,
                "adv": 1e12,
                "is_st": True,
            }
        },
    )
    acks_st = b_st.place_orders([Order(code, "buy", 1000, "market", None)])
    assert not acks_st[0].accepted and acks_st[0].reason == "limit_up", (
        f"ST 股 +6% 应判涨停拒买，实际 accepted={acks_st[0].accepted} reason={acks_st[0].reason}"
    )

    # 非 ST（缺 is_st 或 False）：宽阈值，应成交
    b_ns = PaperBroker(initial_cash=1_000_000.0)
    b_ns.advance_to(
        date(2026, 1, 5),
        {
            code: {
                "open": open_px,
                "pre_close": pre_close,
                "close": open_px,
                "vol": 1e6,
                "adv": 1e12,
                "is_st": False,
            }
        },
    )
    acks_ns = b_ns.place_orders([Order(code, "buy", 1000, "market", None)])
    assert acks_ns[0].accepted, (
        f"非 ST +6% 不应拒，实际 reason={acks_ns[0].reason}"
    )


def test_daily_step_passes_is_st_from_build_is_st_by_date(
    tmp_path: Path, monkeypatch
) -> None:
    """drivers 须构造 is_st_by_date 并写入 market entry，使 broker 对 ST 收窄阈值。

    monkeypatch build_is_st_by_date 把执行日标为 ST；开盘 +6% → limit_up 拒买。
    旧实现不构造/不传 is_st → 买单成交 → 红。
    """
    sig, exec_d = date(2026, 1, 5), date(2026, 1, 6)
    code = "600001.SH"
    open_px, pre_close = 10.6, 10.0  # +6%
    daily = pl.DataFrame(
        [
            {
                "trade_date": sig,
                "ts_code": code,
                "open": 10.0,
                "pre_close": 10.0,
                "close": 10.0,
                "vol": 1e8,
                "amount": 1e9,
            },
            {
                "trade_date": exec_d,
                "ts_code": code,
                "open": open_px,
                "pre_close": pre_close,
                "close": open_px,
                "vol": 1e8,
                "amount": 1e9,
            },
        ]
    )
    # 仅执行日将该股标为 ST
    monkeypatch.setattr(
        "factorzen.execution.drivers.build_is_st_by_date",
        lambda codes, dates: {exec_d: {code}},
    )
    rd = _pf(tmp_path / "pf", sig, code, 0.5)
    cfg = {"initial_cash": 1_000_000.0, "slippage_bps": 0.0}
    sess = tmp_path / "sess"
    SessionStore(sess).init({"broker": "paper", **cfg})

    r = run_daily_step(sess, exec_d, [rd], daily, config=cfg)
    assert not r["skipped"]
    # 买单被 ST 涨停拒 → 无成交（或 acks 含 limit_up）
    recs = SessionStore(sess).ledger_records()
    assert len(recs) == 1
    acks = recs[0]["acks"]
    assert any(not a["accepted"] and a["reason"] == "limit_up" for a in acks), (
        f"ST +6% 应 limit_up 拒买；acks={acks}, fills={recs[0]['fills']}"
    )
    assert r["n_fills"] == 0

"""美股市场 adapter 离线单测（注入 fetch canned JSON，CI 无网络可跑）。

覆盖：factors 派生列(vwap 后复权典型价/ret_1d)、leaf_map parity(价量叶子可求值 + 期货 oi 拒之)、
run_us_mining 端到端 pipe、registry 注册、prompt 市场化(us caveats/signal_families/leaf_guidance)、
universe 静态快照(幸存者偏差 MVP)。
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from factorzen.markets import registry
from factorzen.markets.us.profile import build_us_profile


def _ts(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day, 14, 30, tzinfo=dt.timezone.utc).timestamp())


def _chart_json(sym, ts, o, h, l, c, v, adj) -> bytes:  # noqa: E741
    return json.dumps({"chart": {"result": [{
        "meta": {"symbol": sym, "currency": "USD"},
        "timestamp": ts,
        "indicators": {"quote": [{"open": o, "high": h, "low": l, "close": c, "volume": v}],
                       "adjclose": [{"adjclose": adj}]},
    }], "error": None}}).encode()


def _bdays(n: int, start: dt.date = dt.date(2023, 1, 2)) -> list[dt.date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def _make_fetch(symbols, dates, seed=0):
    rng = np.random.default_rng(seed)
    closes = {}
    for i, sym in enumerate(symbols):
        base = 50.0 + i
        closes[sym] = base * np.cumprod(1 + rng.normal(0, 0.02, len(dates)))

    def fetch(url: str) -> bytes:
        sym = url.split("/chart/")[1].split("?")[0]
        c = closes[sym]
        ts = [_ts(d) for d in dates]
        v = [float(rng.integers(100000, 1000000)) for _ in dates]
        return _chart_json(sym, ts, (c * 0.999).tolist(), (c * 1.01).tolist(),
                           (c * 0.99).tolist(), c.tolist(), v, c.tolist())
    return fetch


def _profile(symbols, dates, tmp_path, top_n=None):
    return build_us_profile(
        top_n=top_n, cache_root=str(tmp_path), fetch=_make_fetch(symbols, dates),
        request_interval=0.0, symbols=symbols,
    )


# ── factors 派生列 ───────────────────────────────────────
def test_factors_derived_vwap_typical_ret(tmp_path) -> None:
    dates = _bdays(10)
    prof = _profile(["AAA", "BBB"], dates, tmp_path)
    bars = prof.provider.fetch_bars(None, dates[0].strftime("%Y%m%d"), dates[-1].strftime("%Y%m%d"))
    der = prof.factors.derived_columns(bars).sort(["ts_code", "trade_date"])
    assert {"vwap", "log_vol", "ret_1d"}.issubset(der.columns)
    row = der.filter(pl.col("ts_code") == "AAA").sort("trade_date").row(2, named=True)
    # vwap = (high+low+close)/3（后复权典型价）
    assert row["vwap"] == pytest.approx((row["high"] + row["low"] + row["close"]) / 3.0)


# ── leaf_map parity ──────────────────────────────────────
def test_leaf_map_parity_us_parses_futures_rejects(tmp_path) -> None:
    from factorzen.discovery.expression import evaluate_materialized, parse_expr

    dates = _bdays(10)
    prof = _profile(["AAA", "BBB"], dates, tmp_path)
    leaf_map = prof.factors.leaf_features()
    bars = prof.provider.fetch_bars(None, dates[0].strftime("%Y%m%d"), dates[-1].strftime("%Y%m%d"))
    der = prof.factors.derived_columns(bars).sort(["ts_code", "trade_date"])
    node = parse_expr("ts_mean(vwap, 3)", leaf_map)  # 价量叶子
    vals = evaluate_materialized(node, der, leaf_map)
    assert vals.len() == der.height
    # 美股 leaf_map 无期货 oi 叶子 → 解析失败（异常契约：只抛 ValueError，陷阱#7）
    with pytest.raises(ValueError, match="oi"):
        parse_expr("ts_mean(oi, 3)", leaf_map)


# ── run_us_mining 端到端 pipe ────────────────────────────
def test_run_us_mining_pipe(tmp_path) -> None:
    from factorzen.markets.us.mining import run_us_mining

    dates = _bdays(50)
    symbols = [f"S{i:02d}" for i in range(35)]  # 35 > _MIN_CROSS_SAMPLES=30
    prof = _profile(symbols, dates, tmp_path)
    start, end = dates[0].strftime("%Y%m%d"), dates[-1].strftime("%Y%m%d")
    res = run_us_mining(
        prof, prof.universe.snapshot(end), start, end,
        n_trials=30, top_k=5, seed=7, out_dir=str(tmp_path / "sessions"),
    )
    assert "candidates" in res and "session_dir" in res  # 管道贯通不崩


# ── universe 静态快照 ─────────────────────────────────────
def test_universe_static_snapshot_survivorship(tmp_path) -> None:
    dates = _bdays(5)
    symbols = [f"S{i:02d}" for i in range(10)]
    prof = _profile(symbols, dates, tmp_path)
    # snapshot 对任意 d 返回同一静态池（不做 PIT 历史成分，幸存者偏差 MVP）
    a = prof.universe.snapshot("20230601")
    b = prof.universe.snapshot("20200101")
    assert a == b == symbols


# ── registry ─────────────────────────────────────────────
def test_registry_get_us() -> None:
    prof = registry.get("us")
    assert prof.name == "us"
    assert prof.risk is None
    lf = prof.factors.leaf_features()
    assert "vwap" in lf and "close" in lf
    assert "oi" not in lf and "funding_rate" not in lf  # 不广告不存在的叶子


# ── prompt 市场化 ────────────────────────────────────────
def test_prompt_market_us() -> None:
    from factorzen.agents.roles.hypothesis import signal_families
    from factorzen.llm.generation import build_agent_messages
    from factorzen.llm.prompt_fragments import market_caveats

    cav = market_caveats("us")
    assert "涨跌停" in cav and "拆股" in cav and "幸存者偏差" in cav
    # 无 A 股/crypto/期货专属叶子泄漏
    assert "north_ratio" not in cav and "funding" not in cav and "持仓量" not in cav
    sig = signal_families("us")
    assert "量价" in sig
    leaves = list(registry.get("us").factors.leaf_features().keys())
    msgs = build_agent_messages(["add", "ts_mean"], leaves, market="us")
    joined = "\n".join(m["content"] for m in msgs)
    assert "vwap" in joined and "复权" in joined
    assert "north_ratio" not in joined and "oi" not in joined  # 无他市场叶子泄漏

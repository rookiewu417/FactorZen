"""
test_markets.py：合并自 agents 相关碎片测试（test_markets.py）。
test_mining_multimarket.py：Phase 1：M5/M6 LLM 挖掘多市场化（crypto）+ A 股逐字节零回归。
"""

from __future__ import annotations

import datetime as dt
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factorzen.markets import registry
from factorzen.markets.futures.calendar import FuturesCalendar
from factorzen.markets.futures.profile import build_futures_profile
from factorzen.markets.us.profile import build_us_profile

# ==== 来自 test_markets.py ====
# ==== 来自 test_futures_market.py ====
_SUF = "SHF"


def _trade_dates(n: int, start: date = date(2024, 1, 2)) -> list[date]:
    # 简化：连续自然日当交易日（fake 日历同集），够单测用
    return [start + timedelta(days=i) for i in range(n)]


class FakePro:
    """最小 Tushare pro 桩：fut_daily/fut_mapping/fut_basic（单交易所 SHFE）。

    每品种 2 合约 c1/c2（contango: c2 略高），主力前半段 c1、后半段 c2（中点展期）。
    两合约每日都有行情（保证展期日 new_prev_close 可得）。
    """

    def __init__(self, varieties: list[str], dates: list[date], seed: int = 0) -> None:
        self.varieties = varieties
        self.dates = dates
        rng = np.random.default_rng(seed)
        self.roll_idx = len(dates) // 2
        rows = []
        map_rows = []
        for vi, v in enumerate(varieties):
            base = 1000.0 + vi * 50.0
            # 两合约各自的随机游走价格（c2 系统性高 8%，制造非 1 的 roll 因子）
            p1 = base * np.cumprod(1 + rng.normal(0, 0.01, len(dates)))
            p2 = base * 1.08 * np.cumprod(1 + rng.normal(0, 0.01, len(dates)))
            for di, d in enumerate(dates):
                c1 = f"{v}2401.{_SUF}"
                c2 = f"{v}2402.{_SUF}"
                for code, px in ((c1, p1[di]), (c2, p2[di])):
                    vol = float(rng.integers(1000, 50000))
                    rows.append((code, d.strftime("%Y%m%d"), px * 0.999, px * 1.01,
                                 px * 0.99, px, px, vol, px * vol * 5 / 1e4,
                                 float(rng.integers(5000, 80000))))
                main = c1 if di < self.roll_idx else c2
                map_rows.append((f"{v}.{_SUF}", d.strftime("%Y%m%d"), main))
        self._daily = pd.DataFrame(rows, columns=[
            "ts_code", "trade_date", "open", "high", "low", "close",
            "settle", "vol", "amount", "oi"])
        self._mapping = pd.DataFrame(map_rows, columns=["ts_code", "trade_date", "mapping_ts_code"])

    def fut_daily(self, trade_date: str):
        return self._daily[self._daily["trade_date"] == trade_date].copy()

    def fut_mapping(self, trade_date: str):
        return self._mapping[self._mapping["trade_date"] == trade_date].copy()

    def fut_basic(self, exchange: str, fut_type: str = "1"):
        if exchange != "SHFE":
            return pd.DataFrame(columns=["ts_code", "symbol", "name", "fut_code", "exchange", "list_date"])
        rows = []
        for v in self.varieties:
            rows.append((f"{v}2401.{_SUF}", f"{v}2401", f"{v}主力", v, exchange, "20200101"))
            rows.append((f"{v}2402.{_SUF}", f"{v}2402", f"{v}次主力", v, exchange, "20200101"))
        return pd.DataFrame(rows, columns=["ts_code", "symbol", "name", "fut_code", "exchange", "list_date"])


def _cal(dates: list[date]) -> FuturesCalendar:
    cal_df = pl.DataFrame({"cal_date": dates, "is_open": [1] * len(dates)}).with_columns(
        pl.col("is_open").cast(pl.Int8)
    )
    return FuturesCalendar(cal_df=cal_df)


def _profile__futures(varieties, dates, tmp_path, top_n=40):
    pro = FakePro(varieties, dates)
    return build_futures_profile(
        pro=pro, exchanges=("SHFE",), top_n=top_n,
        calendar=_cal(dates), cache_root=str(tmp_path),
    )


# ── provider → 主力连续 ──────────────────────────────────
def test_provider_fetch_bars_continuous(tmp_path) -> None:
    dates = _trade_dates(10)
    prof = _profile__futures(["CU", "RB"], dates, tmp_path)
    start, end = dates[0].strftime("%Y%m%d"), dates[-1].strftime("%Y%m%d")
    bars = prof.provider.fetch_bars(None, start, end)
    assert set(bars["ts_code"].unique().to_list()) == {"CU.SHF", "RB.SHF"}
    assert {"open", "high", "low", "close", "vol", "amount", "oi", "adj_factor",
            "mapping_ts_code"}.issubset(bars.columns)
    # 展期日 adj_factor 应变化（contango → roll_step≠1），首段=1
    cu = bars.filter(pl.col("ts_code") == "CU.SHF").sort("trade_date")
    adj = cu["adj_factor"].to_list()
    assert adj[0] == 1.0
    assert adj[-1] != 1.0  # 后半段被复权
    # 复权后 close 的 ret 无巨幅展期跳变（|ret|<0.1，contango 8% 跳变已被消除）
    ret = (cu["close"] / cu["close"].shift(1) - 1.0).drop_nulls().to_list()
    assert max(abs(r) for r in ret) < 0.1


def test_provider_cache_audit_incremental(tmp_path) -> None:
    """缓存审计：第二次 fetch 命中缓存不重复拉取（用 fetch 计数验证）。"""
    dates = _trade_dates(8)
    pro = FakePro(["CU"], dates)
    calls = {"daily": 0}
    orig = pro.fut_daily

    def counting(trade_date):
        calls["daily"] += 1
        return orig(trade_date)

    pro.fut_daily = counting  # type: ignore[method-assign]
    prof = build_futures_profile(pro=pro, exchanges=("SHFE",),
                                 calendar=_cal(dates), cache_root=str(tmp_path))
    start, end = dates[0].strftime("%Y%m%d"), dates[-1].strftime("%Y%m%d")
    prof.provider.fetch_bars(None, start, end)
    first = calls["daily"]
    assert first == len(dates)  # 冷缓存逐日拉
    # 新 provider（清进程内 meta 缓存）同窗口再拉 → 命中盘缓存，0 次 API
    calls["daily"] = 0
    prof2 = build_futures_profile(pro=pro, exchanges=("SHFE",),
                                  calendar=_cal(dates), cache_root=str(tmp_path))
    prof2.provider.fetch_bars(None, start, end)
    assert calls["daily"] == 0  # 全命中缓存


# ── factors 派生列 ───────────────────────────────────────
def test_factors_derived_vwap_adjusted_oi_chg_roll_null(tmp_path) -> None:
    dates = _trade_dates(10)
    prof = _profile__futures(["CU"], dates, tmp_path)
    start, end = dates[0].strftime("%Y%m%d"), dates[-1].strftime("%Y%m%d")
    bars = prof.provider.fetch_bars(None, start, end)
    der = prof.factors.derived_columns(bars).sort("trade_date")
    assert {"vwap", "log_vol", "ret_1d", "oi_chg"}.issubset(der.columns)
    # 展期日 oi_chg 置 null（换合约机械跳变被消除）
    roll_i = len(dates) // 2
    oi_chg = der["oi_chg"].to_list()
    assert oi_chg[roll_i] is None
    # vwap 随 adj_factor 复权：后半段 vwap = amount/vol * adj_factor
    row = der.row(roll_i + 1, named=True)
    expected_vwap = row["amount"] / row["vol"] * row["adj_factor"]
    assert abs(row["vwap"] - expected_vwap) < 1e-6


# ── leaf_map parity ──────────────────────────────────────
def test_leaf_map_parity_oi_parses_ashare_rejects(tmp_path) -> None:
    from factorzen.discovery.expression import evaluate_materialized, parse_expr
    from factorzen.discovery.operators import LEAF_FEATURES as ASHARE_LEAVES

    dates = _trade_dates(10)
    prof = _profile__futures(["CU", "RB"], dates, tmp_path)
    leaf_map = prof.factors.leaf_features()
    bars = prof.provider.fetch_bars(None, dates[0].strftime("%Y%m%d"), dates[-1].strftime("%Y%m%d"))
    der = prof.factors.derived_columns(bars).sort(["ts_code", "trade_date"])
    node = parse_expr("ts_mean(oi_chg, 3)", leaf_map)  # oi_chg 是期货特有派生叶子
    vals = evaluate_materialized(node, der, leaf_map)
    assert vals.len() == der.height  # 可求值
    # A 股默认 leaf_map 无 oi/oi_chg → 解析失败（异常契约：解析只抛 ValueError，陷阱#7）
    with pytest.raises(ValueError, match="oi_chg"):
        parse_expr("ts_mean(oi_chg, 3)", ASHARE_LEAVES)


# ── run_futures_mining 端到端 pipe ───────────────────────
def test_run_futures_mining_pipe(tmp_path) -> None:
    from factorzen.markets.futures.mining import run_futures_mining

    dates = _trade_dates(50)  # 够 holdout 切分 + 若干评估
    varieties = [f"V{i:02d}" for i in range(35)]  # 35 品种 > _MIN_CROSS_SAMPLES=30
    prof = _profile__futures(varieties, dates, tmp_path)
    start, end = dates[0].strftime("%Y%m%d"), dates[-1].strftime("%Y%m%d")
    res = run_futures_mining(
        prof, prof.universe.snapshot(end), start, end,
        n_trials=30, top_k=5, seed=7, out_dir=str(tmp_path / "sessions"),
    )
    assert "candidates" in res and "session_dir" in res  # 管道贯通，不崩


# ── registry ─────────────────────────────────────────────
def test_registry_get_futures() -> None:
    prof = registry.get("futures")
    assert prof.name == "futures"
    assert prof.risk is None
    assert "oi" in prof.factors.leaf_features()


# ── prompt 市场化 ────────────────────────────────────────
def test_prompt_market_futures() -> None:
    from factorzen.agents.roles.hypothesis import signal_families
    from factorzen.llm.generation import build_agent_messages
    from factorzen.llm.prompt_fragments import market_caveats

    cav = market_caveats("futures")
    assert "主力连续" in cav and "持仓量" in cav
    assert "north_ratio" not in cav and "roe" not in cav and "funding" not in cav
    sig = signal_families("futures")
    assert "持仓量" in sig
    leaves = list(registry.get("futures").factors.leaf_features().keys())
    msgs = build_agent_messages(["add", "ts_mean"], leaves, market="futures")
    joined = "\n".join(m["content"] for m in msgs)
    assert "oi" in joined and "后复权" in joined
    assert "north_ratio" not in joined  # 无 A 股叶子泄漏

# ==== 来自 test_us_market.py ====
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


def _profile__us(symbols, dates, tmp_path, top_n=None):
    return build_us_profile(
        top_n=top_n, cache_root=str(tmp_path), fetch=_make_fetch(symbols, dates),
        request_interval=0.0, symbols=symbols,
    )


# ── factors 派生列 ───────────────────────────────────────
def test_factors_derived_vwap_typical_ret(tmp_path) -> None:
    dates = _bdays(10)
    prof = _profile__us(["AAA", "BBB"], dates, tmp_path)
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
    prof = _profile__us(["AAA", "BBB"], dates, tmp_path)
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
    prof = _profile__us(symbols, dates, tmp_path)
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
    prof = _profile__us(symbols, dates, tmp_path)
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

# ==== 来自 test_mining_multimarket.py ====
_GOLDEN = json.loads((Path(__file__).resolve().parent.parent / "golden_ashare_prompts.json").read_text())


# ── 1.1 A 股逐字节零回归（golden 对照，改前捕获） ──────────────────────────────
def test_build_agent_messages_ashare_byte_identical():
    from factorzen.llm.generation import build_agent_messages
    m = build_agent_messages(["ts_mean", "ts_std"], ["close", "vol"], "FB", ["neg1"])
    assert m[0]["content"] == _GOLDEN["bam_sys"]
    assert m[1]["content"] == _GOLDEN["bam_user"]
    # market="ashare" 显式 == 默认（证明 default 分支等价）
    m2 = build_agent_messages(["ts_mean", "ts_std"], ["close", "vol"], "FB", ["neg1"],
                              market="ashare")
    assert m2[0]["content"] == _GOLDEN["bam_sys"]


def test_build_agent_messages_ashare_budget_byte_identical():
    from factorzen.llm.generation import build_agent_messages
    m = build_agent_messages(["ts_mean"], ["close"], "", [],
                             leaf_budgets={"north_ratio": 238})
    assert m[0]["content"] == _GOLDEN["bam_budget_sys"]


def test_coder_syntax_prompt_ashare_byte_identical():
    from factorzen.agents.roles.coder import _syntax_prompt
    assert _syntax_prompt() == _GOLDEN["coder_syntax"]
    assert _syntax_prompt({"north_ratio": 238}) == _GOLDEN["coder_syntax_budget"]
    # 显式 market/leaf_names=None 亦等价
    assert _syntax_prompt(market="ashare", leaf_names=None) == _GOLDEN["coder_syntax"]


def test_hypothesis_prompts_ashare_byte_identical():
    from factorzen.agents.roles.hypothesis import propose_hypotheses, propose_structured
    cap: dict = {}

    def fake(msgs):
        cap["sys"] = msgs[0]["content"]
        cap["user"] = msgs[1]["content"]
        return '{"hypotheses":["x"]}'
    propose_hypotheses(fake, known_invalid=["a"], known_valid=["b"], feedback="fb", n=2)
    assert cap["sys"] == _GOLDEN["hyp_sys"]
    assert cap["user"] == _GOLDEN["hyp_user"]

    def fake2(msgs):
        cap["s2"] = msgs[0]["content"]
        return '{"hypotheses":[{"direction":"d"}]}'
    propose_structured(fake2, known_invalid=[], known_valid=[])
    assert cap["s2"] == _GOLDEN["struct_sys"]


def test_signal_families_ashare_byte_identical():
    from factorzen.agents.roles.hypothesis import signal_families
    assert signal_families() == _GOLDEN["signal_families"]
    assert signal_families("ashare") == _GOLDEN["signal_families"]


# ── 1.1 crypto prompt 市场化 ───────────────────────────────────────────────────
def test_market_caveats_crypto_vs_ashare():
    from factorzen.llm.prompt_fragments import ASHARE_CAVEATS, market_caveats
    cr = market_caveats("crypto")
    for kw in ["funding", "open_interest", "taker_buy_ratio", "T+0", "24/7", "PIT"]:
        assert kw in cr, f"crypto caveats 缺 {kw}"
    # crypto caveats 自包含：不引用 A 股规则口径（T+1/停牌），也不广告 A 股专有叶子
    assert "T+1" not in cr and "north_ratio" not in cr and "roe" not in cr
    assert market_caveats("ashare") == ASHARE_CAVEATS  # ashare 逐字节
    # 未知市场 → 通用兜底（含 PIT），不抛
    assert "PIT" in market_caveats("does_not_exist")


def test_build_agent_messages_crypto_leaves_and_caveats():
    from factorzen.llm.generation import build_agent_messages
    sys = build_agent_messages(
        ["ts_mean"], ["close", "funding_rate", "open_interest", "taker_buy_ratio"],
        market="crypto")[0]["content"]
    assert "funding" in sys and "open_interest" in sys and "T+0" in sys
    # A 股专有叶子不得泄漏进 crypto prompt（不广告不存在的叶子——能力层↔接线层漂移）
    assert "north_ratio" not in sys and "roe" not in sys and "T+1" not in sys


def test_coder_syntax_prompt_crypto():
    from factorzen.agents.roles.coder import _syntax_prompt
    sys = _syntax_prompt(market="crypto",
                         leaf_names=["close", "funding_rate", "open_interest"])
    assert "funding_rate" in sys and "T+0" in sys
    assert "north_ratio" not in sys and "roe" not in sys


def test_signal_families_crypto():
    from factorzen.agents.roles.hypothesis import signal_families
    fam = signal_families("crypto")
    assert "funding" in fam or "资金费率" in fam
    assert "open_interest" in fam or "持仓量" in fam
    assert "北向" not in fam and "roe" not in fam


def test_propose_structured_crypto_injects_crypto_market():
    from factorzen.agents.roles.hypothesis import propose_structured
    cap: dict = {}

    def fake(msgs):
        cap["sys"] = msgs[0]["content"]
        return '{"hypotheses":[{"direction":"d"}]}'
    propose_structured(fake, known_invalid=[], known_valid=[], market="crypto")
    assert "funding" in cap["sys"] and "T+0" in cap["sys"]
    assert "涨跌停" not in cap["sys"]


# ── 1.2 生成/评估层吃 profile ──────────────────────────────────────────────────
class _CryptoProfileStub:
    """轻量 crypto profile：evaluation/AgentContext 只用 .name + .factors。"""
    name = "crypto"

    def __init__(self):
        from factorzen.markets.crypto.factors import CryptoFactorSet
        self.factors = CryptoFactorSet()


def _crypto_daily(n_syms: int = 40, n_days: int = 90) -> pl.DataFrame:
    """合成 crypto 挖掘帧：多标的截面 + funding_rate/open_interest 叶子（≥30 只满足 IC 截面门）。"""
    import datetime as dt

    import numpy as np
    rng = np.random.default_rng(7)
    base = dt.date(2024, 1, 1)
    rows = []
    for s in range(n_syms):
        price = 100.0 + s * 10
        for d in range(n_days):
            price *= 1.0 + float(rng.normal(0, 0.02))
            vol = float(rng.uniform(1e3, 1e5))
            rows.append({
                "ts_code": f"SYM{s}USDT",
                "trade_date": base + dt.timedelta(days=d),
                "open": price * 0.99, "high": price * 1.01, "low": price * 0.98,
                "close": price, "vol": vol, "amount": price * vol,
                "funding_rate": float(rng.normal(0.0001, 0.0002)),
                "open_interest": float(rng.uniform(1e6, 1e7)),
                "taker_buy_volume": vol * float(rng.uniform(0.4, 0.6)),
            })
    return pl.DataFrame(rows)


def test_agent_context_from_profile_crypto_vs_default():
    from factorzen.agents.nodes import AgentContext
    from factorzen.discovery.operators import LEAF_FEATURES, OPERATORS
    # 默认（None）= A 股，零回归
    d = AgentContext.from_profile(None)
    assert d.market == "ashare" and d.leaf_map is None
    assert d.leaf_names == list(LEAF_FEATURES.keys())
    assert d.op_names == list(OPERATORS.keys())
    # crypto
    c = AgentContext.from_profile(_CryptoProfileStub())
    assert c.market == "crypto"
    assert "funding_rate" in c.leaf_names and "open_interest" in c.leaf_names
    assert c.leaf_map is not None and c.leaf_map["funding_rate"] == "funding_rate"
    # 算子集市场无关
    assert c.op_names == list(OPERATORS.keys())


def test_evaluate_expressions_crypto_profile_parses_funding():
    from factorzen.discovery.evaluation import evaluate_expressions
    from factorzen.discovery.scoring import DataBundle
    daily = _crypto_daily()
    bundle = DataBundle.build(daily)
    prof = _CryptoProfileStub()
    # crypto profile 下 funding_rate 表达式可解析可求值
    res = evaluate_expressions(["ts_mean(funding_rate, 5)", "ts_zscore(open_interest, 10)"],
                               daily, bundle, profile=prof)
    assert all(r["compile_ok"] for r in res), res
    assert any(r["ic_train"] is not None for r in res)
    # 无 profile（A 股默认）→ funding_rate 未知叶子 → compile_ok False（零回归的排斥面）
    res_a = evaluate_expressions(["ts_mean(funding_rate, 5)"], daily, bundle)
    assert res_a[0]["compile_ok"] is False


def test_crypto_leaf_map_parity_across_parse_warmup_eval():
    """crypto 表达式在评估/预热门/预算三条路径 leaf_map 一致（parse_expr / warmup_shortfall /
    leaf_warmup_budgets 都吃同一 crypto leaf_map）。"""
    import datetime as dt

    from factorzen.discovery.evaluation import _preprocess_daily
    from factorzen.discovery.expression import (
        leaf_warmup_budgets,
        parse_expr,
        warmup_shortfall,
    )
    prof = _CryptoProfileStub()
    leaf_map = prof.factors.leaf_features()
    daily = _crypto_daily(n_days=60)
    prepped = _preprocess_daily(daily, prof)
    eval_start = dt.date(2024, 2, 1)
    # parse
    node = parse_expr("ts_mean(funding_rate, 20)", leaf_map)
    # 预热门 have 与预算表逐值一致（同 leaf_map）
    budgets = leaf_warmup_budgets(prepped, eval_start, ["funding_rate"], leaf_map=leaf_map)
    from factorzen.discovery.expression import warmup_bars_by_leaf
    have = warmup_bars_by_leaf(node, prepped, eval_start, leaf_map)["funding_rate"]
    assert have == budgets["funding_rate"]
    # warmup_shortfall 用同一 leaf_map（窗口 20 < have → 不欠预热）
    assert warmup_shortfall(node, prepped, eval_start, leaf_map) is None


def test_make_health_check_crypto_profile_funding_healthy():
    from factorzen.discovery.evaluation import make_health_check
    prof = _CryptoProfileStub()
    daily = _crypto_daily(n_days=60)
    check = make_health_check(daily, profile=prof, leaf_map=prof.factors.leaf_features())
    # crypto 叶子表达式健康（None），不被误判解析失败
    assert check("ts_mean(funding_rate, 10)") is None
    # A 股默认（无 leaf_map）→ funding_rate 判解析失败
    check_a = make_health_check(daily)
    assert check_a("ts_mean(funding_rate, 10)") is not None


# ── 1.3 experiment_index 按 market 分族（A 股 known_invalid 不得泄漏进 crypto recall） ──
def test_experiment_index_scoped_by_market(tmp_path):
    from factorzen.agents.experiment_index import ExperimentIndex
    from factorzen.agents.roles.librarian import recall
    idx = ExperimentIndex(str(tmp_path / "idx.jsonl"))
    aw = {"start": "20240101", "end": "20241231", "universe": "csi800", "market": "ashare"}
    cw = {"start": "20240101", "end": "20241231", "universe": None, "market": "crypto"}
    # A 股一条「已验证无效」记录 + crypto 一条「已验证有效」记录
    idx.append([
        {"expression": "ts_mean(north_ratio, 20)", "hypothesis": "h", "ic_train": 0.001,
         "ir_train": 0.01, "n_train": 100, "passed": False, "verdict": None,
         "decorrelated": False, "compile_ok": True, "error": None, "data_window": aw,
         "run_id": "a"},
        {"expression": "ts_mean(funding_rate, 20)", "hypothesis": "h", "ic_train": 0.05,
         "ir_train": 0.3, "n_train": 100, "passed": True, "verdict": "keep", "holdout_ic": 0.04,
         "decorrelated": False, "compile_ok": True, "error": None, "data_window": cw,
         "run_id": "c"},
    ])
    a_rec = recall(idx, data_window=aw)
    c_rec = recall(idx, data_window=cw)
    # A 股的 north_ratio 负例不得出现在 crypto 族的任何召回里
    assert any("north_ratio" in e for e in a_rec.known_invalid)
    assert all("north_ratio" not in e for e in c_rec.known_invalid + c_rec.known_valid)
    assert all("north_ratio" not in e for e in c_rec.seen)
    # crypto 的有效因子只在 crypto 族可见
    assert any("funding_rate" in e for e in c_rec.known_valid)
    assert all("funding_rate" not in e for e in a_rec.known_valid + a_rec.known_invalid)


# ── 裸 JSON 数组容错（crypto smoke 实测：DeepSeek 常返回顶层数组而非包装对象，
#    旧解析直接丢整轮假设——4/6 与 4/4 轮「Hypothesis 未产出假设」的根因） ──────────
_BARE_STRUCT_ARRAY = (
    '[\n  {"direction": "d1", "mechanism": "m1", "expected_sign": 1, "falsification": "f1"},\n'
    '  {"direction": "d2", "mechanism": "m2", "expected_sign": -1, "falsification": "f2"}\n]'
)


def test_propose_structured_accepts_bare_json_array():
    from factorzen.agents.roles.hypothesis import propose_structured
    out = propose_structured(lambda _m: _BARE_STRUCT_ARRAY,
                             known_invalid=[], known_valid=[], n=2, market="crypto")
    assert [h["direction"] for h in out] == ["d1", "d2"]


def test_propose_structured_accepts_fenced_bare_array():
    from factorzen.agents.roles.hypothesis import propose_structured
    out = propose_structured(lambda _m: "```json\n" + _BARE_STRUCT_ARRAY + "\n```",
                             known_invalid=[], known_valid=[])
    assert len(out) == 2 and out[1]["expected_sign"] == -1


def test_propose_hypotheses_accepts_bare_string_array():
    from factorzen.agents.roles.hypothesis import propose_hypotheses
    out = propose_hypotheses(lambda _m: '["方向1", "方向2"]',
                             known_invalid=[], known_valid=[], n=2)
    assert out == ["方向1", "方向2"]


def test_write_expressions_accepts_bare_string_array():
    from factorzen.agents.roles.coder import write_expressions
    out = write_expressions("h", lambda _m: '["ts_mean(close,5)", "rank(vol)"]')
    assert out == ["ts_mean(close,5)", "rank(vol)"]


def test_decompose_tasks_accepts_bare_dict_array():
    from factorzen.agents.roles.coder import decompose_tasks
    raw = '[{"name": "n1", "description": "d1", "rationale": "r1"}]'
    out = decompose_tasks("h", lambda _m: raw)
    assert out == [{"name": "n1", "description": "d1", "rationale": "r1"}]


def test_wrapped_object_still_wins_over_array_fallback():
    """包装对象路径零回归：正常 {"hypotheses": [...]} 响应不受数组回退影响。"""
    from factorzen.agents.roles.hypothesis import propose_hypotheses
    out = propose_hypotheses(lambda _m: '{"hypotheses": ["a"]}',
                             known_invalid=[], known_valid=[])
    assert out == ["a"]


# ── 1.3 CLI 接线：--market crypto 装配帧 + 透传 profile；ashare 保持 profile=None ──
def _team_args(**over):
    import argparse
    base = dict(start="20240301", end="20241231", universe=None, market="ashare",
                symbols=None, top_n=50, iterations=2, top_k=5, seed=42,
                index_path="/tmp/e.jsonl", structured=True, patience=None, heal_rounds=0,
                hypotheses_per_round=1, freq="daily", command_line="mine team")
    base.update(over)
    return argparse.Namespace(**base)


def test_cmd_mine_team_ashare_passes_profile_none(monkeypatch):
    from factorzen.cli import main as cli
    cap: dict = {}
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily",
                        lambda start, end, universe=None, lookback_days=None, **kw: _mock_ashare_daily())

    def fake_team_mine(daily, **kw):
        cap.update(kw)
        return {"n_candidates": 0, "n_trials": 0, "run_dir": "x"}
    monkeypatch.setattr("factorzen.pipelines.factor_mine_team.run_team_mine", fake_team_mine)
    rc = cli._cmd_mine_team(_team_args(market="ashare"))
    assert rc == 0
    assert cap["profile"] is None            # A 股零回归：不带 profile
    assert cap["eval_start"] == "20240301"


def test_cmd_mine_team_crypto_assembles_and_threads_profile(monkeypatch):
    from factorzen.cli import main as cli
    cap: dict = {}
    fake_profile = _CryptoProfileStub()
    fake_profile.base_freq = "daily"
    fake_profile.provider = object()

    monkeypatch.setattr("factorzen.markets.crypto.profile.build_crypto_profile",
                        lambda **_k: fake_profile)

    def fake_build(provider, symbols, start, end, freq):
        cap["build"] = dict(symbols=symbols, start=start, end=end, freq=freq)
        return _crypto_daily(n_days=40)
    monkeypatch.setattr("factorzen.markets.crypto.mining.build_crypto_daily", fake_build)

    def fake_team_mine(daily, **kw):
        cap.update(kw)
        return {"n_candidates": 0, "n_trials": 0, "run_dir": "x"}
    monkeypatch.setattr("factorzen.pipelines.factor_mine_team.run_team_mine", fake_team_mine)

    rc = cli._cmd_mine_team(_team_args(market="crypto", symbols="BTCUSDT,ETHUSDT"))
    assert rc == 0
    # profile 透传（非 None）+ eval_start=挖掘窗口 start（预热前缀边界）
    assert cap["profile"] is fake_profile
    assert cap["eval_start"] == "20240301"
    # 预热前缀：build_crypto_daily 的 start 明显早于挖掘窗口 start（AGENT_WARMUP_LOOKBACK 自然日）
    assert cap["build"]["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert cap["build"]["start"] < "20240301"
    # data_window.market 如实记录 crypto
    assert cap["data_window"]["market"] == "crypto"


def _mock_ashare_daily() -> pl.DataFrame:
    import datetime as dt
    rows = []
    base = dt.date(2024, 1, 1)
    for s in range(35):
        for d in range(40):
            rows.append({"ts_code": f"{s:06d}.SZ", "trade_date": base + dt.timedelta(days=d),
                         "close": 10.0 + d, "open": 10.0, "high": 11.0, "low": 9.0,
                         "vol": 1e5, "amount": 1e6})
    return pl.DataFrame(rows)


# ── Phase 3 US CLI 接线：--market us 装配后复权帧 + 透传 profile（价量族，无 A 股叶子泄漏） ──
class _USProfileStub:
    name = "us"

    def __init__(self):
        from factorzen.markets.us.factors import USFactorSet
        self.factors = USFactorSet()


def _us_daily(n_syms: int = 35, n_days: int = 40) -> pl.DataFrame:
    import datetime as dt
    base = dt.date(2024, 1, 1)
    rows = []
    for s in range(n_syms):
        for d in range(n_days):
            rows.append({"ts_code": f"US{s:03d}", "trade_date": base + dt.timedelta(days=d),
                         "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0 + d,
                         "vol": 1e5, "amount": 1e6})
    return pl.DataFrame(rows)


def test_cmd_mine_team_us_assembles_and_threads_profile(monkeypatch):
    from factorzen.cli import main as cli
    cap: dict = {}
    fake_profile = _USProfileStub()
    fake_profile.base_freq = "daily"
    fake_profile.provider = object()

    class _U:
        def snapshot(self, d):
            return ["AAPL", "MSFT"]
    fake_profile.universe = _U()
    monkeypatch.setattr("factorzen.markets.us.profile.build_us_profile", lambda **_k: fake_profile)

    def fake_build(provider, symbols, start, end, freq="daily"):
        cap["build"] = dict(symbols=symbols, start=start, end=end, freq=freq)
        return _us_daily()
    monkeypatch.setattr("factorzen.markets.us.mining.build_us_daily", fake_build)

    def fake_team_mine(daily, **kw):
        cap.update(kw)
        return {"n_candidates": 0, "n_trials": 0, "run_dir": "x"}
    monkeypatch.setattr("factorzen.pipelines.factor_mine_team.run_team_mine", fake_team_mine)

    rc = cli._cmd_mine_team(_team_args(market="us", symbols=None))
    assert rc == 0
    assert cap["profile"] is fake_profile          # profile 透传（非 None）
    assert cap["eval_start"] == "20240301"          # eval_start=挖掘窗口 start（预热边界）
    assert cap["build"]["symbols"] == ["AAPL", "MSFT"]  # 缺 --symbols → universe 静态快照
    assert cap["build"]["start"] < "20240301"       # 预热前缀：早于挖掘窗口 start
    assert cap["data_window"]["market"] == "us"     # manifest 如实记录 us


def test_us_leaf_map_has_no_ashare_leaves():
    # us 叶子仅价量族，A 股专有叶子（north_ratio/roe/net_mf_amount）零泄漏
    from factorzen.markets.us.factors import USFactorSet
    leaves = set(USFactorSet().leaf_features())
    assert {"north_ratio", "roe", "net_mf_amount", "funding_rate", "oi"}.isdisjoint(leaves)
    assert {"close", "vwap", "log_vol", "ret_1d", "amount"}.issubset(leaves)

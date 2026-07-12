"""商品期货市场 adapter 离线单测（fake Tushare pro，CI 无 token/网络可跑）。

覆盖：provider 拉数+缓存审计→主力连续、factors 派生列(vwap 复权/oi_chg 展期置 null)、
leaf_map parity(oi/oi_chg 可解析求值 + A 股默认拒之)、run_futures_mining 端到端 pipe、
registry 注册、prompt 市场化(futures caveats/signal_families/leaf_guidance)。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factorzen.markets import registry
from factorzen.markets.futures.calendar import FuturesCalendar
from factorzen.markets.futures.profile import build_futures_profile

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


def _profile(varieties, dates, tmp_path, top_n=40):
    pro = FakePro(varieties, dates)
    return build_futures_profile(
        pro=pro, exchanges=("SHFE",), top_n=top_n,
        calendar=_cal(dates), cache_root=str(tmp_path),
    )


# ── provider → 主力连续 ──────────────────────────────────
def test_provider_fetch_bars_continuous(tmp_path) -> None:
    dates = _trade_dates(10)
    prof = _profile(["CU", "RB"], dates, tmp_path)
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
    prof = _profile(["CU"], dates, tmp_path)
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
    prof = _profile(["CU", "RB"], dates, tmp_path)
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
    prof = _profile(varieties, dates, tmp_path)
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

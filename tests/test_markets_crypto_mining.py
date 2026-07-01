"""MC1 T4/T5/T6: crypto 挖掘入口 —— 数据装配 + export-alpha + 端到端(离线)。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from factorzen.markets.crypto.mining import (
    build_crypto_daily,
    export_crypto_alpha,
    run_crypto_mining,
)
from factorzen.markets.crypto.profile import build_crypto_profile

_N_SYM = 40
_N_DAYS = 55
_START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


class FakeCCXTBulk:
    """生成 _N_SYM 个标的的合成 OHLCV/funding/OI（截面 ≥30，够挖掘）。"""

    def __init__(self, n_sym: int = _N_SYM, n_days: int = _N_DAYS, seed: int = 11):
        rng = np.random.default_rng(seed)
        self._ohlcv: dict[str, list] = {}
        self._funding: dict[str, list] = {}
        self._oi: dict[str, list] = {}
        self._symbols = [f"SYM{i:02d}USDT" for i in range(n_sym)]
        for i in range(n_sym):
            unified = f"SYM{i:02d}/USDT:USDT"
            price = 100.0 + i
            bars, fund, oi = [], [], []
            for d in range(n_days):
                day = _START + timedelta(days=d)
                price = max(1.0, price * (1 + rng.normal(0, 0.02)))
                vol = float(rng.uniform(50, 500))
                bars.append([_ms(day), price, price * 1.01, price * 0.99, price, vol])
                for h in (0, 8, 16):
                    fund.append({"timestamp": _ms(day + timedelta(hours=h)),
                                 "fundingRate": float(rng.normal(0.0001, 0.0002))})
                oi.append({"timestamp": _ms(day), "openInterestAmount": float(rng.uniform(1e3, 5e3))})
            self._ohlcv[unified] = bars
            self._funding[unified] = fund
            self._oi[unified] = oi

    @property
    def symbols(self):
        return list(self._symbols)

    def fetch_ohlcv(self, symbol, timeframe="1d", since=None, limit=1000):
        data = self._ohlcv.get(symbol, [])
        if since is not None:
            data = [r for r in data if r[0] >= since]
        return data[:limit]

    def fetch_funding_rate_history(self, symbol, since=None, limit=1000):
        data = self._funding.get(symbol, [])
        if since is not None:
            data = [r for r in data if r["timestamp"] >= since]
        return data[:limit]

    def fetch_open_interest_history(self, symbol, timeframe="1d", since=None, limit=1000):
        data = self._oi.get(symbol, [])
        if since is not None:
            data = [r for r in data if r["timestamp"] >= since]
        return data[:limit]

    def load_markets(self):
        return {
            f"SYM{i:02d}/USDT:USDT": {"base": f"SYM{i:02d}", "quote": "USDT", "swap": True,
                                      "info": {}}
            for i in range(len(self._symbols))
        }


def _profile_and_syms():
    fake = FakeCCXTBulk()
    return build_crypto_profile(client=fake), fake.symbols


# ── T4: 数据装配 ──────────────────────────────────────────────────────────────
def test_build_crypto_daily_joins_funding_oi():
    profile, syms = _profile_and_syms()
    daily = build_crypto_daily(profile.provider, syms[:3], "20240101", "20240110")
    assert {"ts_code", "trade_date", "close", "vol", "amount",
            "funding_rate", "open_interest"} <= set(daily.columns)
    # funding/OI 已 join 且无 null
    assert daily["funding_rate"].null_count() == 0
    assert daily["open_interest"].null_count() == 0
    # 每标的 10 天
    assert daily.filter(pl.col("ts_code") == syms[0]).height == 10


# ── T5: export-alpha ─────────────────────────────────────────────────────────
def test_export_crypto_alpha_cross_section():
    profile, syms = _profile_and_syms()
    cross = export_crypto_alpha(profile, "ts_mean(close, 5)", syms, "20240101", "20240220",
                                date="20240220")
    assert cross.columns == ["ts_code", "alpha"]
    assert cross["alpha"].is_finite().all()
    assert cross.height >= 30  # 大部分标的当日有值


# ── T6: 端到端 ────────────────────────────────────────────────────────────────
def test_end_to_end_crypto_mining(tmp_path):
    """crypto perps: 装配→挖掘→带 OOS/holdout/PBO 的 candidates→export rank1 alpha。"""
    profile, syms = _profile_and_syms()
    result = run_crypto_mining(
        profile, syms, "20240101", "20240224",
        n_trials=40, top_k=5, seed=3, out_dir=str(tmp_path),
    )
    assert result["candidates"], "端到端挖掘应产出候选(验证 crypto 上可挖 alpha)"
    cand_csv = tmp_path / "session_3_random" / "candidates.csv"
    assert cand_csv.exists()
    cand = pl.read_csv(cand_csv)
    assert {"holdout_ic", "dsr_pvalue", "pbo"} <= set(cand.columns)  # OOS+防过拟合验证
    # 用 rank1 表达式导出当日 α 截面
    rank1_expr = cand.sort("rank")["expression"][0]
    cross = export_crypto_alpha(profile, rank1_expr, syms, "20240101", "20240224",
                                date="20240224")
    assert cross.columns == ["ts_code", "alpha"]
    assert cross.height >= 30

"""test_markets_crypto_mining.py：MC1 T4/T5/T6: crypto 挖掘入口 —— 数据装配 + export-alpha + 端到端(离线)。
test_markets_crypto_factors.py：MC0 Task 6: crypto FactorSet（叶子字典 + 派生列）。
test_markets_crypto_universe.py：MC0 Task 5: crypto Universe（成交额 Top-N + 流动性/新币过滤）。
test_markets_crypto_rules_costs.py：MC0 Task 4: crypto TradingRules + CostModel（T+0/可空 + maker/taker/funding）。
test_markets_crypto_profile_smoke.py：MC0 Task 7: crypto MarketProfile 注册 + 离线端到端 smoke。
"""

from __future__ import annotations

import math
from datetime import (
    date,
    datetime,
    timedelta,
    timezone,
)

import numpy as np
import polars as pl

from factorzen.markets import registry
from factorzen.markets.base import (
    CostModel,
    DataProvider,
    FactorSet,
    MarketProfile,
    TradingRules,
    Universe,
)
from factorzen.markets.crypto.costs import CryptoCostModel
from factorzen.markets.crypto.factors import CryptoFactorSet
from factorzen.markets.crypto.lake_provider import CryptoLakeProvider
from factorzen.markets.crypto.mining import (
    build_crypto_daily,
    export_crypto_alpha,
    run_crypto_mining,
)
from factorzen.markets.crypto.profile import build_crypto_profile
from factorzen.markets.crypto.provider import CryptoDataProvider
from factorzen.markets.crypto.rules import CryptoTradingRules
from factorzen.markets.crypto.universe import CryptoUniverse
from tests.markets.test_providers import FakeCCXT

# ==== 来自 test_markets_crypto_mining.py ====
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

# ==== 来自 test_markets_crypto_factors.py ====
def test_is_a_factorset():
    assert isinstance(CryptoFactorSet(), FactorSet)


def test_leaf_features_include_crypto_specific():
    fs = CryptoFactorSet()
    leaves = fs.leaf_features()
    # 价量叶子
    for name in ["close", "open", "high", "low", "vol", "amount", "vwap", "log_vol", "ret_1d"]:
        assert name in leaves
    # crypto 无复权：close 直接映射 close（非 close_adj）
    assert leaves["close"] == "close"
    # crypto 特有叶子
    assert "funding_rate" in leaves and "open_interest" in leaves


def test_basic_features():
    fs = CryptoFactorSet()
    assert fs.basic_features() == {"funding_rate", "open_interest"}


def test_derived_columns_ground_truth():
    fs = CryptoFactorSet()
    bars = pl.DataFrame({
        "ts_code": ["BTCUSDT"] * 3 + ["ETHUSDT"] * 3,
        "trade_date": [date(2024, 1, i) for i in (1, 2, 3)] * 2,
        "close": [100.0, 110.0, 105.0, 50.0, 55.0, 60.0],
        "vol": [10.0, 20.0, 5.0, 100.0, 50.0, 80.0],
        "amount": [1000.0, 2200.0, 525.0, 5000.0, 2750.0, 4800.0],
    })
    out = fs.derived_columns(bars).sort(["ts_code", "trade_date"])
    btc = out.filter(pl.col("ts_code") == "BTCUSDT").sort("trade_date")
    # vwap = amount / vol
    assert btc["vwap"].to_list() == [100.0, 110.0, 105.0]
    # ret_1d = close.pct_change（首行 null）
    assert btc["ret_1d"][0] is None
    assert abs(btc["ret_1d"][1] - 0.1) < 1e-12
    assert abs(btc["ret_1d"][2] - (105.0 / 110.0 - 1)) < 1e-12
    # log_vol = ln(vol)，对拍 numpy
    np.testing.assert_allclose(btc["log_vol"].to_list(), np.log([10.0, 20.0, 5.0]), rtol=1e-12)
    # ret_1d 不跨标的泄漏：ETH 首行也是 null
    eth = out.filter(pl.col("ts_code") == "ETHUSDT").sort("trade_date")
    assert eth["ret_1d"][0] is None
    assert math.isclose(eth["ret_1d"][1], 0.1, rel_tol=1e-12)


def test_taker_buy_ratio_derived_and_guarded():
    fs = CryptoFactorSet()
    assert "taker_buy_ratio" in fs.leaf_features()
    bars = pl.DataFrame({
        "ts_code": ["BTCUSDT"] * 2, "trade_date": [1, 2],
        "close": [1.0, 2.0], "vol": [10.0, 0.0], "amount": [10.0, 0.0],
        "taker_buy_volume": [6.0, 3.0],
    })
    out = fs.derived_columns(bars)
    assert out["taker_buy_ratio"].to_list()[0] == 0.6
    assert out["taker_buy_ratio"][1] is None  # vol=0 → null 不除零


def test_taker_buy_ratio_null_when_source_missing():
    bars = pl.DataFrame({"ts_code": ["BTCUSDT"], "trade_date": [1],
                         "close": [1.0], "vol": [1.0], "amount": [1.0]})
    out = CryptoFactorSet().derived_columns(bars)  # ccxt 旧路径无 taker_buy_volume
    assert out["taker_buy_ratio"].to_list() == [None]

# ==== 来自 test_markets_crypto_universe.py ====
class _FakeProvider(DataProvider):
    """返回受控 fixture bars + meta 的假 provider。"""

    def __init__(self, bars: pl.DataFrame, meta: pl.DataFrame):
        self._bars = bars
        self._meta = meta

    def fetch_bars(self, symbols, start, end, freq="daily"):
        df = self._bars
        s = date(int(start[:4]), int(start[4:6]), int(start[6:8]))
        e = date(int(end[:4]), int(end[4:6]), int(end[6:8]))
        df = df.filter((pl.col("trade_date") >= s) & (pl.col("trade_date") <= e))
        if symbols is not None:
            df = df.filter(pl.col("ts_code").is_in(symbols))
        return df

    def fetch_symbol_meta(self):
        return self._meta


def _fixture() -> _FakeProvider:
    # 窗口 [2024-01-11, 2024-02-10]，d=2024-02-10，lookback=30，min_list_days=30
    rows = []
    def add(code, d, amount):
        rows.append({"ts_code": code, "trade_date": d, "close": 100.0,
                     "vol": amount / 100.0, "amount": amount})
    # BTC/ETH：老币、窗口内高成交额
    for dd in [date(2024, 1, 11), date(2024, 2, 1), date(2024, 2, 10)]:
        add("BTCUSDT", dd, 5000.0)
        add("ETHUSDT", dd, 3000.0)
        add("LOWUSDT", dd, 10.0)  # 老币但成交额极低 → min_amount 剔除
    # NEW：新币（2024-02-05 才上市），成交额极高 → 应被 age 过滤剔除
    for dd in [date(2024, 2, 5), date(2024, 2, 10)]:
        add("NEWUSDT", dd, 99999.0)
    bars = pl.DataFrame(rows)
    meta = pl.DataFrame({
        "ts_code": ["BTCUSDT", "ETHUSDT", "LOWUSDT", "NEWUSDT"],
        "name": ["BTC", "ETH", "LOW", "NEW"],
        "list_date": [date(2020, 1, 1), date(2020, 1, 1), date(2020, 1, 1), date(2024, 2, 5)],
    })
    return _FakeProvider(bars, meta)


def test_is_a_universe():
    u = CryptoUniverse(provider=_fixture())
    assert isinstance(u, Universe)


def test_snapshot_topn_liquidity_age_filter():
    u = CryptoUniverse(provider=_fixture(), top_n=3, lookback_days=30,
                       min_amount=100.0, min_list_days=30)
    snap = u.snapshot("20240210")
    # NEW 被 age 剔除，LOW 被 min_amount 剔除 → 剩 BTC/ETH，按成交额降序
    assert snap == ["BTCUSDT", "ETHUSDT"]


def test_snapshot_topn_caps_count():
    u = CryptoUniverse(provider=_fixture(), top_n=1, lookback_days=30,
                       min_amount=0.0, min_list_days=30)
    snap = u.snapshot("20240210")
    assert snap == ["BTCUSDT"]  # 成交额第一


def test_benchmark_returns_btc_close_series():
    u = CryptoUniverse(provider=_fixture(), benchmark_symbol="BTCUSDT")
    bench = u.benchmark("20240101", "20240210")
    assert set(bench.columns) >= {"trade_date", "close"}
    assert bench.height == 3  # BTC 3 根

# ==== 来自 test_markets_crypto_rules_costs.py ====
def test_rules_are_tradingrules():
    assert isinstance(CryptoTradingRules(), TradingRules)


def test_rules_t0_short_execution():
    r = CryptoTradingRules()
    assert r.allow_short is True
    assert r.settlement_lag == 0  # T+0
    assert r.execution_price_col == "close"  # next-bar close 撮合


def test_tradable_mask_blocks_zero_volume():
    r = CryptoTradingRules()
    bars = pl.DataFrame({"ts_code": ["A", "B", "C"], "vol": [10.0, 0.0, 5.0]})
    buy = r.tradable_mask(bars, "buy")
    sell = r.tradable_mask(bars, "sell")
    assert buy.to_list() == [True, False, True]
    # crypto 买卖对称（无涨跌停不对称）
    assert sell.to_list() == [True, False, True]


def test_costs_are_costmodel():
    assert isinstance(CryptoCostModel(), CostModel)


def test_trade_cost_symmetric_maker_taker():
    c = CryptoCostModel(maker=0.0002, taker=0.0005, slippage=0.0005)
    # taker 卖：10000*(0.0005+0.0005)=10.0
    assert abs(c.trade_cost("sell", 10000.0, is_maker=False) - 10.0) < 1e-9
    # maker 买：10000*(0.0002+0.0005)=7.0
    assert abs(c.trade_cost("buy", 10000.0, is_maker=True) - 7.0) < 1e-9
    # 买卖对称：同 side 参数下成本相等（无印花税不对称）
    assert c.trade_cost("buy", 10000.0) == c.trade_cost("sell", 10000.0)


def test_carry_cost_funding_sign():
    c = CryptoCostModel()
    # 多头 pos=+10000, funding=0.0001, 3 期 → 付费 +3.0
    assert abs(c.carry_cost(10000.0, 3, funding_rate=0.0001) - 3.0) < 1e-9
    # 空头 pos=-10000 同 funding → 收费 -3.0
    assert abs(c.carry_cost(-10000.0, 3, funding_rate=0.0001) - (-3.0)) < 1e-9
    # funding=0 → 无 carry
    assert c.carry_cost(10000.0, 5, funding_rate=0.0) == 0.0

# ==== 来自 test_markets_crypto_profile_smoke.py ====
def test_registry_get_crypto():
    p = registry.get("crypto")
    assert isinstance(p, MarketProfile)
    assert p.name == "crypto"
    assert p.quote_currency == "USDT"
    assert p.base_freq == "daily"
    from factorzen.markets.crypto.risk import CryptoRiskModel
    assert isinstance(p.risk, CryptoRiskModel)  # MC3 填入 crypto 风险模型
    assert p.calendar.periods_per_year() == 365.0


def test_offline_end_to_end_pipeline():
    """注入 FakeCCXT，走 provider→factors→universe 端到端，不联网。"""
    p = build_crypto_profile(client=FakeCCXT())
    bars = p.provider.fetch_bars(["BTCUSDT", "ETHUSDT"], "20240101", "20240103")
    assert bars.height == 5
    enriched = p.factors.derived_columns(bars)
    assert {"vwap", "log_vol", "ret_1d"} <= set(enriched.columns)
    snap = p.universe.snapshot("20240103")
    assert "BTCUSDT" in snap  # 非空、schema 正确


def test_profile_defaults_to_lake_without_client(tmp_path):
    profile = build_crypto_profile(lake_root=tmp_path)
    assert isinstance(profile.provider, CryptoLakeProvider)


def test_profile_uses_ccxt_when_client_injected():
    profile = build_crypto_profile(client=object())
    assert isinstance(profile.provider, CryptoDataProvider)


def test_profile_explicit_source_wins(tmp_path):
    profile = build_crypto_profile(client=object(), source="lake", lake_root=tmp_path)
    assert isinstance(profile.provider, CryptoLakeProvider)

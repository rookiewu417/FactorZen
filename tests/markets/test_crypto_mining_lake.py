"""合并自: test_crypto_mining.py, test_crypto_lake.py
目标: test_crypto_mining_lake.py

--- 来源 test_crypto_mining.py ---
test_markets_crypto_mining.py：MC1 T4/T5/T6: crypto 挖掘入口 —— 数据装配 + export-alpha + 端到端(离线)。
test_markets_crypto_factors.py：MC0 Task 6: crypto FactorSet（叶子字典 + 派生列）。
test_markets_crypto_universe.py：MC0 Task 5: crypto Universe（成交额 Top-N + 流动性/新币过滤）。
test_markets_crypto_rules_costs.py：MC0 Task 4: crypto TradingRules + CostModel（T+0/可空 + maker/taker/funding）。
test_markets_crypto_profile_smoke.py：MC0 Task 7: crypto MarketProfile 注册 + 离线端到端 smoke。

--- 来源 test_crypto_lake.py ---
test_markets_crypto_vision.py：vision 下载器离线单测:canned XML/CSV/zip,fetch 全注入。
test_markets_crypto_lake_provider.py：湖 provider:读湖+重采样+freq 分派;mini-lake fixture 供全链路测试复用。
test_markets_crypto_lake.py：数据湖读写 roundtrip + 区间过滤 + 缺标的空帧。
test_crypto_backfill_incremental_meta.py：vision backfill 当月增量补拉(M3) + 写 meta 打通 universe(M4)。
test_markets_crypto_calendar.py：MC0 Task 2: crypto 24/7 连续交易日历。
test_markets_crypto_resample.py：重采样/对齐 ground truth 单测:全部手算期望值,不用被测函数自导自演。
test_markets_crypto_frequency.py：频率表单测:别名/年化/未知 raise(全链路唯一事实源)。
"""

from __future__ import annotations

import io
import math
import zipfile
from datetime import (
    date,
    datetime,
    timedelta,
    timezone,
)

import numpy as np
import polars as pl
import pytest

from factorzen.markets import registry
from factorzen.markets.base import (
    Calendar,
    CostModel,
    DataProvider,
    FactorSet,
    MarketProfile,
    TradingRules,
    Universe,
)
from factorzen.markets.crypto.calendar import CryptoCalendar
from factorzen.markets.crypto.costs import CryptoCostModel
from factorzen.markets.crypto.factors import CryptoFactorSet
from factorzen.markets.crypto.frequency import (
    BAR_FREQS,
    normalize_freq,
    periods_per_year,
)
from factorzen.markets.crypto.lake import (
    CryptoLake,
    day_range,
    month_range,
)
from factorzen.markets.crypto.lake_provider import CryptoLakeProvider
from factorzen.markets.crypto.mining import (
    build_crypto_daily,
    export_crypto_alpha,
    run_crypto_mining,
)
from factorzen.markets.crypto.profile import build_crypto_profile
from factorzen.markets.crypto.provider import CryptoDataProvider
from factorzen.markets.crypto.resample import (
    align_funding,
    align_open_interest,
    resample_bars,
)
from factorzen.markets.crypto.rules import CryptoTradingRules
from factorzen.markets.crypto.universe import CryptoUniverse
from factorzen.markets.crypto.vision import (
    backfill,
    fetch_zip_csv,
    list_um_symbols,
    parse_funding_csv,
    parse_kline_csv,
    parse_metrics_csv,
    rank_symbols_by_amount,
)
from tests.markets.test_providers import FakeCCXT

# ==== 来自 test_crypto_mining.py ====
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
def test_crypto_mining_features_suite(tmp_path):
    """test_build_crypto_daily_joins_funding_oi；test_export_crypto_alpha_cross_section；crypto perps: 装配→挖掘→带 OOS/holdout/PBO 的 candidates→export rank1 alpha。；test_is_a_factorset；test_leaf_features_include_crypto_specific；test_basic_features；test_derived_columns_ground_truth；test_taker_buy_ratio_derived_and_guarded；test_taker_buy_ratio_null_when_source_missing"""
    # -- 原 test_build_crypto_daily_joins_funding_oi --
    def _section_0_test_build_crypto_daily_joins_funding_oi():
        profile, syms = _profile_and_syms()
        daily = build_crypto_daily(profile.provider, syms[:3], "20240101", "20240110")
        assert {"ts_code", "trade_date", "close", "vol", "amount",
                "funding_rate", "open_interest"} <= set(daily.columns)
        # funding/OI 已 join 且无 null
        assert daily["funding_rate"].null_count() == 0
        assert daily["open_interest"].null_count() == 0
        # 每标的 10 天
        assert daily.filter(pl.col("ts_code") == syms[0]).height == 10

    _section_0_test_build_crypto_daily_joins_funding_oi()

    # -- 原 test_export_crypto_alpha_cross_section --
    def _section_1_test_export_crypto_alpha_cross_section():
        profile, syms = _profile_and_syms()
        cross = export_crypto_alpha(profile, "ts_mean(close, 5)", syms, "20240101", "20240220",
                                    date="20240220")
        assert cross.columns == ["ts_code", "alpha"]
        assert cross["alpha"].is_finite().all()
        assert cross.height >= 30  # 大部分标的当日有值

    _section_1_test_export_crypto_alpha_cross_section()

    # -- 原 test_end_to_end_crypto_mining --
    def _section_2_test_end_to_end_crypto_mining(tmp_path):
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

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_end_to_end_crypto_mining(_tp2)

    # -- 原 test_is_a_factorset --
    def _section_3_test_is_a_factorset():
        assert isinstance(CryptoFactorSet(), FactorSet)

    _section_3_test_is_a_factorset()

    # -- 原 test_leaf_features_include_crypto_specific --
    def _section_4_test_leaf_features_include_crypto_specific():
        fs = CryptoFactorSet()
        leaves = fs.leaf_features()
        # 价量叶子
        for name in ["close", "open", "high", "low", "vol", "amount", "vwap", "log_vol", "ret_1d"]:
            assert name in leaves
        # crypto 无复权：close 直接映射 close（非 close_adj）
        assert leaves["close"] == "close"
        # crypto 特有叶子
        assert "funding_rate" in leaves and "open_interest" in leaves

    _section_4_test_leaf_features_include_crypto_specific()

    # -- 原 test_basic_features --
    def _section_5_test_basic_features():
        fs = CryptoFactorSet()
        assert fs.basic_features() == {"funding_rate", "open_interest"}

    _section_5_test_basic_features()

    # -- 原 test_derived_columns_ground_truth --
    def _section_6_test_derived_columns_ground_truth():
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

    _section_6_test_derived_columns_ground_truth()

    # -- 原 test_taker_buy_ratio_derived_and_guarded --
    def _section_7_test_taker_buy_ratio_derived_and_guarded():
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

    _section_7_test_taker_buy_ratio_derived_and_guarded()

    # -- 原 test_taker_buy_ratio_null_when_source_missing --
    def _section_8_test_taker_buy_ratio_null_when_source_missing():
        bars = pl.DataFrame({"ts_code": ["BTCUSDT"], "trade_date": [1],
                             "close": [1.0], "vol": [1.0], "amount": [1.0]})
        out = CryptoFactorSet().derived_columns(bars)  # ccxt 旧路径无 taker_buy_volume
        assert out["taker_buy_ratio"].to_list() == [None]

    _section_8_test_taker_buy_ratio_null_when_source_missing()


# ── T5: export-alpha ─────────────────────────────────────────────────────────


# ── T6: 端到端 ────────────────────────────────────────────────────────────────

# ==== 来自 test_markets_crypto_factors.py ====


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


def test_crypto_universe_rules_costs_suite():
    """test_is_a_universe；test_snapshot_topn_liquidity_age_filter；test_snapshot_topn_caps_count；test_benchmark_returns_btc_close_series；test_rules_are_tradingrules；test_rules_t0_short_execution；test_tradable_mask_blocks_zero_volume；test_costs_are_costmodel；test_trade_cost_symmetric_maker_taker；test_carry_cost_funding_sign；test_registry_get_crypto；注入 FakeCCXT，走 provider→factors→universe 端到端，不联网。"""
    # -- 原 test_is_a_universe --
    def _section_0_test_is_a_universe():
        u = CryptoUniverse(provider=_fixture())
        assert isinstance(u, Universe)

    _section_0_test_is_a_universe()

    # -- 原 test_snapshot_topn_liquidity_age_filter --
    def _section_1_test_snapshot_topn_liquidity_age_filter():
        u = CryptoUniverse(provider=_fixture(), top_n=3, lookback_days=30,
                           min_amount=100.0, min_list_days=30)
        snap = u.snapshot("20240210")
        # NEW 被 age 剔除，LOW 被 min_amount 剔除 → 剩 BTC/ETH，按成交额降序
        assert snap == ["BTCUSDT", "ETHUSDT"]

    _section_1_test_snapshot_topn_liquidity_age_filter()

    # -- 原 test_snapshot_topn_caps_count --
    def _section_2_test_snapshot_topn_caps_count():
        u = CryptoUniverse(provider=_fixture(), top_n=1, lookback_days=30,
                           min_amount=0.0, min_list_days=30)
        snap = u.snapshot("20240210")
        assert snap == ["BTCUSDT"]  # 成交额第一

    _section_2_test_snapshot_topn_caps_count()

    # -- 新增:零成交僵尸标的必须出池(实测 FTMUSDT/MKRUSDT 合约迁移后价格冻结) --
    def _section_2b_test_snapshot_drops_zero_amount_symbols():
        prov = _fixture()
        dead = pl.DataFrame([
            {"ts_code": "DEADUSDT", "trade_date": dd, "close": 100.0, "vol": 0.0, "amount": 0.0}
            for dd in (date(2024, 1, 11), date(2024, 2, 1), date(2024, 2, 10))
        ])
        prov._bars = pl.concat([prov._bars, dead], how="vertical")
        prov._meta = pl.concat([prov._meta, pl.DataFrame(
            {"ts_code": ["DEADUSDT"], "name": ["DEAD"], "list_date": [date(2020, 1, 1)]}
        )], how="vertical")
        # top_n 大于池子规模时,零成交标的过去会因 `0 >= min_amount(0.0)` 混进来
        snap = CryptoUniverse(provider=prov, top_n=99, lookback_days=30,
                              min_amount=0.0, min_list_days=30).snapshot("20240210")
        assert "DEADUSDT" not in snap
        assert snap[:2] == ["BTCUSDT", "ETHUSDT"]  # 健康标的不受影响

    _section_2b_test_snapshot_drops_zero_amount_symbols()

    # -- 原 test_benchmark_returns_btc_close_series --
    def _section_3_test_benchmark_returns_btc_close_series():
        u = CryptoUniverse(provider=_fixture(), benchmark_symbol="BTCUSDT")
        bench = u.benchmark("20240101", "20240210")
        assert set(bench.columns) >= {"trade_date", "close"}
        assert bench.height == 3  # BTC 3 根

    _section_3_test_benchmark_returns_btc_close_series()

    # -- 原 test_rules_are_tradingrules --
    def _section_4_test_rules_are_tradingrules():
        assert isinstance(CryptoTradingRules(), TradingRules)

    _section_4_test_rules_are_tradingrules()

    # -- 原 test_rules_t0_short_execution --
    def _section_5_test_rules_t0_short_execution():
        r = CryptoTradingRules()
        assert r.allow_short is True
        assert r.settlement_lag == 0  # T+0
        assert r.execution_price_col == "close"  # next-bar close 撮合

    _section_5_test_rules_t0_short_execution()

    # -- 原 test_tradable_mask_blocks_zero_volume --
    def _section_6_test_tradable_mask_blocks_zero_volume():
        r = CryptoTradingRules()
        bars = pl.DataFrame({"ts_code": ["A", "B", "C"], "vol": [10.0, 0.0, 5.0]})
        buy = r.tradable_mask(bars, "buy")
        sell = r.tradable_mask(bars, "sell")
        assert buy.to_list() == [True, False, True]
        # crypto 买卖对称（无涨跌停不对称）
        assert sell.to_list() == [True, False, True]

    _section_6_test_tradable_mask_blocks_zero_volume()

    # -- 原 test_costs_are_costmodel --
    def _section_7_test_costs_are_costmodel():
        assert isinstance(CryptoCostModel(), CostModel)

    _section_7_test_costs_are_costmodel()

    # -- 原 test_trade_cost_symmetric_maker_taker --
    def _section_8_test_trade_cost_symmetric_maker_taker():
        c = CryptoCostModel(maker=0.0002, taker=0.0005, slippage=0.0005)
        # taker 卖：10000*(0.0005+0.0005)=10.0
        assert abs(c.trade_cost("sell", 10000.0, is_maker=False) - 10.0) < 1e-9
        # maker 买：10000*(0.0002+0.0005)=7.0
        assert abs(c.trade_cost("buy", 10000.0, is_maker=True) - 7.0) < 1e-9
        # 买卖对称：同 side 参数下成本相等（无印花税不对称）
        assert c.trade_cost("buy", 10000.0) == c.trade_cost("sell", 10000.0)

    _section_8_test_trade_cost_symmetric_maker_taker()

    # -- 原 test_carry_cost_funding_sign --
    def _section_9_test_carry_cost_funding_sign():
        c = CryptoCostModel()
        # 多头 pos=+10000, funding=0.0001, 3 期 → 付费 +3.0
        assert abs(c.carry_cost(10000.0, 3, funding_rate=0.0001) - 3.0) < 1e-9
        # 空头 pos=-10000 同 funding → 收费 -3.0
        assert abs(c.carry_cost(-10000.0, 3, funding_rate=0.0001) - (-3.0)) < 1e-9
        # funding=0 → 无 carry
        assert c.carry_cost(10000.0, 5, funding_rate=0.0) == 0.0

    _section_9_test_carry_cost_funding_sign()

    # -- 原 test_registry_get_crypto --
    def _section_10_test_registry_get_crypto():
        p = registry.get("crypto")
        assert isinstance(p, MarketProfile)
        assert p.name == "crypto"
        assert p.quote_currency == "USDT"
        assert p.base_freq == "daily"
        from factorzen.markets.crypto.risk import CryptoRiskModel
        assert isinstance(p.risk, CryptoRiskModel)  # MC3 填入 crypto 风险模型
        assert p.calendar.periods_per_year() == 365.0

    _section_10_test_registry_get_crypto()

    # -- 原 test_offline_end_to_end_pipeline --
    def _section_11_test_offline_end_to_end_pipeline():
        p = build_crypto_profile(client=FakeCCXT())
        bars = p.provider.fetch_bars(["BTCUSDT", "ETHUSDT"], "20240101", "20240103")
        assert bars.height == 5
        enriched = p.factors.derived_columns(bars)
        assert {"vwap", "log_vol", "ret_1d"} <= set(enriched.columns)
        snap = p.universe.snapshot("20240103")
        assert "BTCUSDT" in snap  # 非空、schema 正确

    _section_11_test_offline_end_to_end_pipeline()


# ==== 来自 test_markets_crypto_rules_costs.py ====


# ==== 来自 test_markets_crypto_profile_smoke.py ====


def test_crypto_profile_source_suite(tmp_path):
    """test_profile_defaults_to_lake_without_client；test_profile_uses_ccxt_when_client_injected；test_profile_explicit_source_wins"""
    # -- 原 test_profile_defaults_to_lake_without_client --
    def _section_0_test_profile_defaults_to_lake_without_client(tmp_path):
        profile = build_crypto_profile(lake_root=tmp_path)
        assert isinstance(profile.provider, CryptoLakeProvider)

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_profile_defaults_to_lake_without_client(_tp0)

    # -- 原 test_profile_uses_ccxt_when_client_injected --
    def _section_1_test_profile_uses_ccxt_when_client_injected():
        profile = build_crypto_profile(client=object())
        assert isinstance(profile.provider, CryptoDataProvider)

    _section_1_test_profile_uses_ccxt_when_client_injected()

    # -- 原 test_profile_explicit_source_wins --
    def _section_2_test_profile_explicit_source_wins(tmp_path):
        profile = build_crypto_profile(client=object(), source="lake", lake_root=tmp_path)
        assert isinstance(profile.provider, CryptoLakeProvider)

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_profile_explicit_source_wins(_tp2)


# ==== 来自 test_crypto_lake.py ====
# ==== 来自 test_markets_crypto_vision.py ====
_KLINE_HEADER = (b"open_time,open,high,low,close,volume,close_time,quote_volume,"
                 b"count,taker_buy_volume,taker_buy_quote_volume,ignore\n")
_KLINE_ROW = b"1782604800000,60000.4,60018.7,60000.3,60018.6,37.652,1782604859999,2259400.8,1187,12.342,740568.6,0\n"


def _zip_bytes(name: str, payload: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, payload)
    return buf.getvalue()


def test_vision_parse_and_s3_suite(tmp_path):
    """test_parse_kline_csv_with_and_without_header；test_parse_funding_and_metrics；test_list_um_symbols_from_s3_xml；test_fetch_zip_csv_retries_then_none；test_backfill_writes_lake_and_records_gaps；test_rank_symbols_by_amount"""
    # -- 原 test_parse_kline_csv_with_and_without_header --
    def _section_0_test_parse_kline_csv_with_and_without_header():
        for raw in (_KLINE_HEADER + _KLINE_ROW, _KLINE_ROW):  # vision 老文件无表头
            df = parse_kline_csv(raw)
            assert df.columns == ["trade_date", "open", "high", "low", "close",
                                  "vol", "amount", "taker_buy_volume"]
            assert df.schema["trade_date"] == pl.Datetime("us")
            assert df["amount"][0] == pytest.approx(2259400.8)   # 真 quote_volume,非 close*vol
            assert df["taker_buy_volume"][0] == pytest.approx(12.342)

    _section_0_test_parse_kline_csv_with_and_without_header()

    # -- 新增:volume 前万行全整数、之后出现小数(实测 SOLUSDT 2025-04) --
    def _section_0b_test_parse_kline_csv_late_float_volume():
        # 按前 N 行推断 dtype 会把 volume 定成 i64,之后的 "6701.8" 解析即崩;
        # 行数须超过推断窗口才有判别力。
        head = b"1782604800000,60000.4,60018.7,60000.3,60018.6,"
        tail = b",1782604859999,2259400.8,1187,12.342,740568.6,0\n"
        rows = [head + (b"37" if i < 10_000 else b"6701.8") + tail for i in range(10_001)]
        df = parse_kline_csv(b"".join(rows))
        assert df.height == 10_001
        assert df.schema["vol"] == pl.Float64
        assert df["vol"].max() == pytest.approx(6701.8)

    _section_0b_test_parse_kline_csv_late_float_volume()

    # -- 原 test_parse_funding_and_metrics --
    def _section_1_test_parse_funding_and_metrics():
        fr = parse_funding_csv(b"calc_time,funding_interval_hours,last_funding_rate\n"
                               b"1777593600000,8,-0.00003746\n")
        assert fr.columns == ["event_time", "funding_rate"]
        assert fr["funding_rate"][0] == pytest.approx(-0.00003746)
        mt = parse_metrics_csv(
            b"create_time,symbol,sum_open_interest,sum_open_interest_value,"
            b"count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
            b"count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
            b"2026-06-27 00:05:00,BTCUSDT,103630.42,6225989541.67,2.2,1.2,2.1,0.7\n")
        assert mt.columns == ["event_time", "open_interest"]
        assert mt["open_interest"][0] == pytest.approx(103630.42)

    _section_1_test_parse_funding_and_metrics()

    # -- 原 test_list_um_symbols_from_s3_xml --
    def _section_2_test_list_um_symbols_from_s3_xml():
        xml = (b"<?xml version='1.0'?><ListBucketResult>"
               b"<Prefix>data/futures/um/monthly/klines/</Prefix>"
               b"<CommonPrefixes><Prefix>data/futures/um/monthly/klines/BTCUSDT/</Prefix></CommonPrefixes>"
               b"<CommonPrefixes><Prefix>data/futures/um/monthly/klines/ETHUSDT/</Prefix></CommonPrefixes>"
               b"<CommonPrefixes><Prefix>data/futures/um/monthly/klines/BTCUSD_PERP/</Prefix></CommonPrefixes>"
               b"<IsTruncated>false</IsTruncated></ListBucketResult>")
        seen = {}

        def _fetch(url):
            seen["url"] = url
            return xml

        syms = list_um_symbols(fetch=_fetch)
        assert syms == ["BTCUSDT", "ETHUSDT"]  # 非 USDT 结尾剔除
        assert "s3" in seen["url"]  # listing 必须走 S3 endpoint(CDN 前端只返回 HTML)

    _section_2_test_list_um_symbols_from_s3_xml()

    # -- 原 test_fetch_zip_csv_retries_then_none --
    def _section_3_test_fetch_zip_csv_retries_then_none():
        calls = {"n": 0}
        def bad_fetch(url):
            calls["n"] += 1
            raise OSError("404")
        assert fetch_zip_csv("http://x/a.zip", fetch=bad_fetch, retries=2) is None
        assert calls["n"] == 3  # 1 次 + 2 重试

    _section_3_test_fetch_zip_csv_retries_then_none()

    # -- 原 test_backfill_writes_lake_and_records_gaps --
    def _section_4_test_backfill_writes_lake_and_records_gaps(tmp_path):
        lake = CryptoLake(tmp_path)
        kzip = _zip_bytes("k.csv", _KLINE_HEADER + _KLINE_ROW)
        fzip = _zip_bytes("f.csv", b"calc_time,funding_interval_hours,last_funding_rate\n"
                                   b"1782604800000,8,0.0001\n")
        def fetch(url):
            if "fundingRate" in url:
                return fzip
            if "/klines/" in url and "1m" in url:
                return kzip
            raise OSError("404")  # metrics 全 404 → 进 gaps
        manifest = backfill(lake, ["BTCUSDT"], "20260628", "20260628", fetch=fetch, log=lambda *a: None)
        assert lake.read_klines(["BTCUSDT"], "20260628", "20260628").height == 1
        assert lake.read_funding(["BTCUSDT"], "20260628", "20260628").height == 1
        assert any("metrics" in g for g in manifest["gaps"])  # 缺口不静默
        assert (tmp_path / "manifest.json").exists()
        # 增量:重跑不重复下载已有分区
        counts = {"n": 0}
        def counting_fetch(url):
            counts["n"] += 1
            return fetch(url)
        backfill(lake, ["BTCUSDT"], "20260628", "20260628", fetch=counting_fetch, log=lambda *a: None)
        assert all("/klines/" not in u for u in []) or counts["n"] < 3  # kline/funding 已存在被跳过

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_backfill_writes_lake_and_records_gaps(_tp4)

    # -- 原 test_rank_symbols_by_amount --
    def _section_5_test_rank_symbols_by_amount(tmp_path):
        def _kd(amount_row: bytes) -> bytes:
            return _zip_bytes("d.csv", _KLINE_HEADER + amount_row)
        big = b"1782604800000,1,1,1,1,1,1782604859999,9999999,1,1,1,0\n"
        small = b"1782604800000,1,1,1,1,1,1782604859999,1000,1,1,1,0\n"
        def fetch(url):
            if "BTCUSDT-1d" in url:
                return _kd(big)
            if "ETHUSDT-1d" in url:
                return _kd(small)
            raise OSError("404")
        top = rank_symbols_by_amount(["BTCUSDT", "ETHUSDT"], "2026-05", top_n=1, fetch=fetch)
        assert top == ["BTCUSDT"]

    _tp5 = tmp_path / "_s5"
    _tp5.mkdir(exist_ok=True)
    _section_5_test_rank_symbols_by_amount(_tp5)


# ==== 来自 test_markets_crypto_lake_provider.py ====
def make_mini_lake(root, symbols=("BTCUSDT", "ETHUSDT"), days=(1, 2)) -> CryptoLake:
    """2 标的 × N 日、每日 00:00-01:59 共 120 根 1m bar 的最小湖。"""
    lake = CryptoLake(root)
    for si, sym in enumerate(symbols):
        frames = []
        for d in days:
            ts = [datetime(2026, 5, d, h, m) for h in (0, 1) for m in range(60)]
            base = 100.0 * (si + 1)
            px = [base + i * 0.1 for i in range(len(ts))]
            frames.append(pl.DataFrame({
                "trade_date": ts, "open": px, "high": [p + 0.5 for p in px],
                "low": [p - 0.5 for p in px], "close": [p + 0.2 for p in px],
                "vol": [1.0] * len(ts), "amount": [p * 1.0 for p in px],
                "taker_buy_volume": [0.6] * len(ts),
            }).with_columns(pl.col("trade_date").cast(pl.Datetime("us"))))
        lake.write_klines(sym, "2026-05", pl.concat(frames))
        lake.write_funding(sym, "2026-05", pl.DataFrame({
            "event_time": [datetime(2026, 5, d, 0, 0) for d in days],
            "funding_rate": [0.0001] * len(days),
        }).with_columns(pl.col("event_time").cast(pl.Datetime("us"))))
        for d in days:
            lake.write_metrics(sym, f"202605{d:02d}", pl.DataFrame({
                "event_time": [datetime(2026, 5, d, 0, 5)], "open_interest": [1000.0 + d],
            }).with_columns(pl.col("event_time").cast(pl.Datetime("us"))))
    lake.write_meta(pl.DataFrame({
        "ts_code": list(symbols), "name": [s[:-4] for s in symbols],
        "list_date": [date(2024, 1, 1)] * len(symbols)}))
    return lake


def test_lake_provider_fetch_suite(tmp_path):
    """test_fetch_bars_daily_date_key；test_fetch_bars_15m_datetime_key；test_fetch_funding_and_oi_freq；test_empty_lake_raises；test_meta_roundtrip；test_month_and_day_range；test_kline_roundtrip_and_filter；test_funding_meta_manifest_roundtrip；test_backfill_current_month_increment_tops_up；test_backfill_writes_meta_for_universe"""
    # -- 原 test_fetch_bars_daily_date_key --
    def _section_0_test_fetch_bars_daily_date_key(tmp_path):
        make_mini_lake(tmp_path)
        p = CryptoLakeProvider(lake_root=tmp_path)
        bars = p.fetch_bars(["BTCUSDT"], "20260501", "20260502", "daily")
        assert bars.schema["trade_date"] == pl.Date
        assert bars.height == 2 and bars["vol"].to_list() == [120.0, 120.0]

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_fetch_bars_daily_date_key(_tp0)

    # -- 原 test_fetch_bars_15m_datetime_key --
    def _section_1_test_fetch_bars_15m_datetime_key(tmp_path):
        make_mini_lake(tmp_path)
        p = CryptoLakeProvider(lake_root=tmp_path)
        bars = p.fetch_bars(["BTCUSDT", "ETHUSDT"], "20260501", "20260501", "15m")
        assert bars.schema["trade_date"] == pl.Datetime("us")
        assert bars.filter(pl.col("ts_code") == "BTCUSDT").height == 8  # 2h → 8 根 15m
        assert bars.filter(pl.col("ts_code") == "BTCUSDT")["vol"].to_list() == [15.0] * 8

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_fetch_bars_15m_datetime_key(_tp1)

    # -- 原 test_fetch_funding_and_oi_freq --
    def _section_2_test_fetch_funding_and_oi_freq(tmp_path):
        make_mini_lake(tmp_path)
        p = CryptoLakeProvider(lake_root=tmp_path)
        fd = p.fetch_funding(["BTCUSDT"], "20260501", "20260502", "daily")
        assert fd.schema["trade_date"] == pl.Date and fd.height == 2
        f15 = p.fetch_funding(["BTCUSDT"], "20260501", "20260501", "15m")
        assert f15["trade_date"].to_list() == [datetime(2026, 5, 1, 0, 0)]
        oi = p.fetch_open_interest(["BTCUSDT"], "20260501", "20260501", "15m")
        assert oi["open_interest"].to_list() == [1001.0]

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_fetch_funding_and_oi_freq(_tp2)

    # -- 原 test_empty_lake_raises --
    def _section_3_test_empty_lake_raises(tmp_path):
        p = CryptoLakeProvider(lake_root=tmp_path / "nope")
        with pytest.raises(RuntimeError, match="backfill"):
            p.fetch_bars(["BTCUSDT"], "20260501", "20260502", "daily")

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_empty_lake_raises(_tp3)

    # -- 原 test_meta_roundtrip --
    def _section_4_test_meta_roundtrip(tmp_path):
        make_mini_lake(tmp_path)
        meta = CryptoLakeProvider(lake_root=tmp_path).fetch_symbol_meta()
        assert set(meta["ts_code"].to_list()) == {"BTCUSDT", "ETHUSDT"}

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_meta_roundtrip(_tp4)

    # -- 原 test_month_and_day_range --
    def _section_5_test_month_and_day_range():
        assert month_range("20250715", "20251003") == ["2025-07", "2025-08", "2025-09", "2025-10"]
        assert day_range("2026-05", "20260530", "20260602") == ["20260530", "20260531"]

    _section_5_test_month_and_day_range()

    # -- 原 test_kline_roundtrip_and_filter --
    def _section_6_test_kline_roundtrip_and_filter(tmp_path):
        lake = CryptoLake(tmp_path)
        lake.write_klines("BTCUSDT", "2026-05", pl.concat([_k(1), _k(2)]))
        lake.write_klines("ETHUSDT", "2026-05", _k(1))
        out = lake.read_klines(["BTCUSDT"], "20260502", "20260502")
        assert out["ts_code"].unique().to_list() == ["BTCUSDT"]
        assert out.height == 2  # 只有 5/2 的两根
        assert lake.read_klines(["XRPUSDT"], "20260501", "20260502").is_empty()  # 缺标的→空帧
        assert sorted(lake.symbols()) == ["BTCUSDT", "ETHUSDT"]

    _tp6 = tmp_path / "_s6"
    _tp6.mkdir(exist_ok=True)
    _section_6_test_kline_roundtrip_and_filter(_tp6)

    # -- 原 test_funding_meta_manifest_roundtrip --
    def _section_7_test_funding_meta_manifest_roundtrip(tmp_path):
        lake = CryptoLake(tmp_path)
        ev = pl.DataFrame({"event_time": [datetime(2026, 5, 1, 8)], "funding_rate": [0.0001]}
                          ).with_columns(pl.col("event_time").cast(pl.Datetime("us")))
        lake.write_funding("BTCUSDT", "2026-05", ev)
        got = lake.read_funding(["BTCUSDT"], "20260501", "20260501")
        assert got["funding_rate"].to_list() == [0.0001] and got["ts_code"][0] == "BTCUSDT"
        meta = pl.DataFrame({"ts_code": ["BTCUSDT"], "name": ["BTC"],
                             "list_date": [datetime(2020, 1, 1).date()]})
        lake.write_meta(meta)
        assert lake.read_meta()["ts_code"].to_list() == ["BTCUSDT"]
        lake.write_manifest({"gaps": []})
        assert lake.read_manifest() == {"gaps": []}

    _tp7 = tmp_path / "_s7"
    _tp7.mkdir(exist_ok=True)
    _section_7_test_funding_meta_manifest_roundtrip(_tp7)

    # -- 原 test_backfill_current_month_increment_tops_up --
    def _section_8_test_backfill_current_month_increment_tops_up(tmp_path):
        lake = CryptoLake(tmp_path)
        # 第一次：只回填 6-28（日包，无月包）
        backfill(lake, ["BTCUSDT"], "20260628", "20260628",
                 fetch=_fetch_daypacks_only, log=lambda *a: None)
        assert lake.read_klines(["BTCUSDT"], "20260628", "20260628").height == 1

        # 第二次：扩到 6-29 —— 修复前当月分区已存在被整月跳过，6-29 永久缺失
        backfill(lake, ["BTCUSDT"], "20260628", "20260629",
                 fetch=_fetch_daypacks_only, log=lambda *a: None)
        got = lake.read_klines(["BTCUSDT"], "20260628", "20260629")
        dates = set(got.select(pl.col("trade_date").dt.strftime("%Y%m%d")).to_series().to_list())
        assert dates == {"20260628", "20260629"}, f"当月增量应补拉 6-29，实得 {dates}"

    _tp8 = tmp_path / "_s8"
    _tp8.mkdir(exist_ok=True)
    _section_8_test_backfill_current_month_increment_tops_up(_tp8)

    # -- 原 test_backfill_writes_meta_for_universe --
    def _section_9_test_backfill_writes_meta_for_universe(tmp_path):
        lake = CryptoLake(tmp_path)
        backfill(lake, ["BTCUSDT"], "20260628", "20260628",
                 fetch=_fetch_daypacks_only, log=lambda *a: None)
        meta = lake.read_meta()
        assert meta.height == 1
        assert meta["ts_code"][0] == "BTCUSDT"
        # universe.snapshot 要求 list_date 非空，否则过滤后为空
        assert meta["list_date"][0] is not None

    _tp9 = tmp_path / "_s9"
    _tp9.mkdir(exist_ok=True)
    _section_9_test_backfill_writes_meta_for_universe(_tp9)


# ==== 来自 test_markets_crypto_lake.py ====
def _k(day: int) -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": [datetime(2026, 5, day, 0, 0), datetime(2026, 5, day, 0, 1)],
        "open": [1.0, 2.0], "high": [2.0, 3.0], "low": [0.5, 1.5],
        "close": [1.5, 2.5], "vol": [10.0, 20.0], "amount": [15.0, 50.0],
        "taker_buy_volume": [4.0, 9.0],
    }).with_columns(pl.col("trade_date").cast(pl.Datetime("us")))


# ==== 来自 test_crypto_backfill_incremental_meta.py ====
_HEADER = (b"open_time,open,high,low,close,volume,close_time,quote_volume,"
           b"count,taker_buy_volume,taker_buy_quote_volume,ignore\n")
_D28 = 1782604800000  # 2026-06-28 00:00 UTC
_DAY = 86_400_000


def _zip(payload: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("d.csv", payload)
    return buf.getvalue()


def _day_zip(open_ms: int) -> bytes:
    row = f"{open_ms},1,1,1,1,1,{open_ms + 59999},100,1,1,1,0\n".encode()
    return _zip(_HEADER + row)


def _fetch_daypacks_only(url: str) -> bytes:
    # 当月无月包(/monthly/ 全 404)；日包按 URL 里的日期返回对应时间戳
    if "/daily/klines/" in url and "2026-06-28" in url:
        return _day_zip(_D28)
    if "/daily/klines/" in url and "2026-06-29" in url:
        return _day_zip(_D28 + _DAY)
    raise OSError("404")


# ==== 来自 test_markets_crypto_calendar.py ====
def test_crypto_calendar_suite():
    """test_is_a_calendar；24/7：区间内每个自然日都是交易日（含周末）。；test_is_session_always_true；test_next_prev_session_are_natural_days；test_periods_per_year"""
    # -- 原 test_is_a_calendar --
    def _section_0_test_is_a_calendar():
        assert isinstance(CryptoCalendar(), Calendar)

    _section_0_test_is_a_calendar()

    # -- 原 test_sessions_are_continuous_including_weekends --
    def _section_1_test_sessions_are_continuous_including_weekends():
        cal = CryptoCalendar()
        days = cal.sessions("20240101", "20240107")  # 2024-01-06=周六, 01-07=周日
        assert days == [date(2024, 1, d) for d in range(1, 8)]
        assert len(days) == 7

    _section_1_test_sessions_are_continuous_including_weekends()

    # -- 原 test_is_session_always_true --
    def _section_2_test_is_session_always_true():
        cal = CryptoCalendar()
        assert cal.is_session(date(2024, 1, 6)) is True  # 周六
        assert cal.is_session("20240107") is True  # 周日

    _section_2_test_is_session_always_true()

    # -- 原 test_next_prev_session_are_natural_days --
    def _section_3_test_next_prev_session_are_natural_days():
        cal = CryptoCalendar()
        assert cal.next_session(date(2024, 1, 1)) == date(2024, 1, 2)
        assert cal.next_session("20240105", n=2) == date(2024, 1, 7)  # 跨周末
        assert cal.prev_session("20240101", n=2) == date(2023, 12, 30)

    _section_3_test_next_prev_session_are_natural_days()

    # -- 原 test_periods_per_year --
    def _section_4_test_periods_per_year():
        cal = CryptoCalendar()
        assert cal.periods_per_year() == 365.0
        assert cal.periods_per_year("daily") == 365.0
        assert cal.periods_per_year("hourly") == 365.0 * 24
        assert cal.periods_per_year("weekly") == 52.0
        assert cal.periods_per_year("monthly") == 12.0

    _section_4_test_periods_per_year()


# ==== 来自 test_markets_crypto_resample.py ====
def _bars_1m() -> pl.DataFrame:
    # BTCUSDT 4 根 1m bar:00:00/00:01 属 15m bar0,00:15/00:16 属 15m bar1
    return pl.DataFrame({
        "ts_code": ["BTCUSDT"] * 4,
        "trade_date": [datetime(2026, 5, 1, 0, 0), datetime(2026, 5, 1, 0, 1),
                       datetime(2026, 5, 1, 0, 15), datetime(2026, 5, 1, 0, 16)],
        "open":  [100.0, 101.0, 103.0, 102.0],
        "high":  [102.0, 104.0, 103.5, 105.0],
        "low":   [ 99.0, 100.5, 101.0, 101.5],
        "close": [101.0, 103.0, 102.0, 104.0],
        "vol":   [10.0, 20.0, 5.0, 15.0],
        "amount": [1000.0, 2000.0, 500.0, 1500.0],
        "taker_buy_volume": [6.0, 8.0, 2.0, 9.0],
    }).with_columns(pl.col("trade_date").cast(pl.Datetime("us")))


def test_crypto_resample_align_suite():
    """test_resample_15m_ground_truth；test_resample_daily_casts_date；test_resample_empty_passthrough；test_align_funding_daily_sums_three_legs；test_align_funding_1h_lands_on_settlement_bars；test_align_open_interest_last_in_bar"""
    # -- 原 test_resample_15m_ground_truth --
    def _section_0_test_resample_15m_ground_truth():
        out = resample_bars(_bars_1m(), "15m")
        assert out["trade_date"].to_list() == [datetime(2026, 5, 1, 0, 0), datetime(2026, 5, 1, 0, 15)]
        # bar0: open=首根 open,close=末根 close,high=max,low=min,量额=sum
        assert out["open"].to_list() == [100.0, 103.0]
        assert out["close"].to_list() == [103.0, 104.0]
        assert out["high"].to_list() == [104.0, 105.0]
        assert out["low"].to_list() == [99.0, 101.0]
        assert out["vol"].to_list() == [30.0, 20.0]
        assert out["amount"].to_list() == [3000.0, 2000.0]
        assert out["taker_buy_volume"].to_list() == [14.0, 11.0]

    _section_0_test_resample_15m_ground_truth()

    # -- 原 test_resample_daily_casts_date --
    def _section_1_test_resample_daily_casts_date():
        out = resample_bars(_bars_1m(), "daily")
        assert out.schema["trade_date"] == pl.Date
        assert out["trade_date"].to_list() == [date(2026, 5, 1)]
        assert out["open"].to_list() == [100.0]
        assert out["close"].to_list() == [104.0]
        assert out["vol"].to_list() == [50.0]

    _section_1_test_resample_daily_casts_date()

    # -- 原 test_resample_empty_passthrough --
    def _section_2_test_resample_empty_passthrough():
        assert resample_bars(_bars_1m().head(0), "1h").is_empty()

    _section_2_test_resample_empty_passthrough()

    # -- 原 test_align_funding_daily_sums_three_legs --
    def _section_3_test_align_funding_daily_sums_three_legs():
        out = align_funding(_funding_events(), "daily")
        assert out.schema["trade_date"] == pl.Date
        assert out["trade_date"].to_list() == [date(2026, 5, 1)]
        assert abs(out["funding_rate"][0] - 0.0006) < 1e-12  # 现日频行为:三档和

    _section_3_test_align_funding_daily_sums_three_legs()

    # -- 原 test_align_funding_1h_lands_on_settlement_bars --
    def _section_4_test_align_funding_1h_lands_on_settlement_bars():
        out = align_funding(_funding_events(), "1h").sort("trade_date")
        assert out["trade_date"].to_list() == [
            datetime(2026, 5, 1, 0, 0), datetime(2026, 5, 1, 8, 0), datetime(2026, 5, 1, 16, 0)]
        assert out["funding_rate"].to_list() == [0.0001, 0.0002, 0.0003]

    _section_4_test_align_funding_1h_lands_on_settlement_bars()

    # -- 原 test_align_open_interest_last_in_bar --
    def _section_5_test_align_open_interest_last_in_bar():
        metrics = pl.DataFrame({
            "ts_code": ["BTCUSDT"] * 3,
            "event_time": [datetime(2026, 5, 1, 0, 0), datetime(2026, 5, 1, 0, 5),
                           datetime(2026, 5, 1, 0, 20)],
            "open_interest": [10.0, 20.0, 30.0],
        }).with_columns(pl.col("event_time").cast(pl.Datetime("us")))
        out15 = align_open_interest(metrics, "15m").sort("trade_date")
        assert out15["open_interest"].to_list() == [20.0, 30.0]  # bar 内最后一笔
        outd = align_open_interest(metrics, "daily")
        assert outd["open_interest"].to_list() == [30.0]  # 当日最后值

    _section_5_test_align_open_interest_last_in_bar()


def _funding_events() -> pl.DataFrame:
    return pl.DataFrame({
        "ts_code": ["BTCUSDT"] * 3,
        "event_time": [datetime(2026, 5, 1, 0, 0), datetime(2026, 5, 1, 8, 0),
                       datetime(2026, 5, 1, 16, 0)],
        "funding_rate": [0.0001, 0.0002, 0.0003],
    }).with_columns(pl.col("event_time").cast(pl.Datetime("us")))


# ==== 来自 test_markets_crypto_frequency.py ====
def test_crypto_freq_normalize_suite():
    """test_normalize_known_and_alias；test_normalize_unknown_raises；test_periods_per_year_values；test_bar_freqs_polars_every"""
    # -- 原 test_normalize_known_and_alias --
    def _section_0_test_normalize_known_and_alias():
        assert normalize_freq("daily") == "daily"
        assert normalize_freq("1h") == "1h"
        assert normalize_freq("hourly") == "1h"  # 别名
        assert normalize_freq("15m") == "15m"

    _section_0_test_normalize_known_and_alias()

    # -- 原 test_normalize_unknown_raises --
    def _section_1_test_normalize_unknown_raises():
        with pytest.raises(ValueError, match="未知频率"):
            normalize_freq("3m")

    _section_1_test_normalize_unknown_raises()

    # -- 原 test_periods_per_year_values --
    def _section_2_test_periods_per_year_values():
        assert periods_per_year("1m") == 365.0 * 24 * 60
        assert periods_per_year("5m") == 365.0 * 24 * 12
        assert periods_per_year("15m") == 365.0 * 24 * 4
        assert periods_per_year("1h") == 365.0 * 24
        assert periods_per_year("daily") == 365.0
        assert periods_per_year("hourly") == 365.0 * 24  # 别名走 1h
        assert periods_per_year("weekly") == 52.0  # calendar 兼容
        assert periods_per_year("monthly") == 12.0

    _section_2_test_periods_per_year_values()

    # -- 原 test_bar_freqs_polars_every --
    def _section_3_test_bar_freqs_polars_every():
        assert BAR_FREQS["daily"].every == "1d"
        assert BAR_FREQS["15m"].every == "15m"
        assert BAR_FREQS["daily"].timeframe == "1d"  # ccxt timeframe

    _section_3_test_bar_freqs_polars_every()



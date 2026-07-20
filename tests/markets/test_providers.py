"""合并自: test_providers.py, test_ashare_wrap.py
目标: test_providers.py

--- 来源 test_providers.py ---
test_markets_base.py：MC0 Task 1: 市场抽象地基 —— Port 接口 + MarketProfile + registry。
test_us_provider.py：美股 Yahoo provider 复权 ground-truth + 缓存审计（离线，注入 fetch，无网络）。
test_markets_crypto_provider.py：MC0 Task 3: crypto CCXT DataProvider（离线 fake，无网络）。
test_crypto_provider_pagination.py：crypto provider funding 分页(M1) + OI timeframe/日聚合(M2)。
test_futures_continuous.py：主力连续合约后复权 ground-truth 测试（不写恒真断言，逐值对手工计算）。

--- 来源 test_ashare_wrap.py ---
MC0 Task 8: A 股 adapter wrap parity（离线可验证部分）。
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import FrozenInstanceError
from datetime import (
    date,
    datetime,
    timezone,
)

import numpy as np
import polars as pl
import pytest

from factorzen.config.constants import (
    COMMISSION_RATE,
    SLIPPAGE_RATE,
    STAMP_TAX_RATE,
)
from factorzen.discovery.operators import BASIC_FEATURES, LEAF_FEATURES
from factorzen.markets import registry
from factorzen.markets.ashare.calendar import AShareCalendar
from factorzen.markets.ashare.costs import AShareCostModel
from factorzen.markets.ashare.factors import AShareFactorSet
from factorzen.markets.ashare.rules import AShareTradingRules
from factorzen.markets.base import (
    Calendar,
    CostModel,
    DataProvider,
    FactorSet,
    MarketProfile,
    RiskModel,
    TradingRules,
    Universe,
)
from factorzen.markets.crypto.provider import CryptoDataProvider
from factorzen.markets.futures.continuous import build_continuous
from factorzen.markets.us.provider import (
    USDataProvider,
    parse_chart_json,
)


# ==== 来自 test_providers.py ====
# ==== 来自 test_markets_base.py ====
# ── 最小 concrete 子类（用于构造 DummyProfile）────────────────────────────────
class _DP(DataProvider):
    def fetch_bars(self, symbols, start, end, freq="daily"):
        return pl.DataFrame({"ts_code": [], "trade_date": []})

    def fetch_symbol_meta(self):
        return pl.DataFrame({"ts_code": []})


class _CAL(Calendar):
    def sessions(self, start, end):
        return [date(2024, 1, 1)]

    def is_session(self, d):
        return True

    def next_session(self, d, n=1):
        return date(2024, 1, 2)

    def prev_session(self, d, n=1):
        return date(2023, 12, 31)

    def periods_per_year(self, freq="daily"):
        return 365.0


class _RULES(TradingRules):
    @property
    def allow_short(self):
        return True

    @property
    def settlement_lag(self):
        return 0

    @property
    def execution_price_col(self):
        return "close"

    def tradable_mask(self, bars, side):
        return pl.Series([True] * bars.height)


class _COST(CostModel):
    def trade_cost(self, side, notional, is_maker=False):
        return 0.0

    def carry_cost(self, position_value, periods, funding_rate=0.0):
        return 0.0


class _UNI(Universe):
    def snapshot(self, d):
        return ["BTCUSDT"]

    def benchmark(self, start, end):
        return pl.DataFrame({"trade_date": [], "close": []})


class _FS(FactorSet):
    def leaf_features(self):
        return {"close": "close"}

    def basic_features(self):
        return set()

    def derived_columns(self, bars):
        return bars


class _RISK(RiskModel):
    def style_factors(self):
        return {}

    def sector_classification(self, symbols, d):
        return pl.DataFrame()


def _dummy_profile() -> MarketProfile:
    return MarketProfile(
        name="dummy",
        quote_currency="XXX",
        base_freq="daily",
        provider=_DP(),
        calendar=_CAL(),
        rules=_RULES(),
        costs=_COST(),
        universe=_UNI(),
        factors=_FS(),
        risk=_RISK(),
    )


# ── 测试 ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "cls",
    [DataProvider, Calendar, TradingRules, CostModel, Universe, FactorSet, RiskModel],
)
def test_ports_are_abstract(cls):
    """7 个 Port 均为抽象类，不可直接实例化。"""
    with pytest.raises(TypeError):
        cls()  # type: ignore[abstract]


def test_market_profile_bundles_ports():
    """MarketProfile 打包 7 个 port + 元数据，且 frozen。"""
    p = _dummy_profile()
    assert p.name == "dummy"
    assert p.quote_currency == "XXX"
    assert p.base_freq == "daily"
    assert isinstance(p.provider, DataProvider)
    assert isinstance(p.calendar, Calendar)
    assert p.calendar.periods_per_year() == 365.0
    with pytest.raises(FrozenInstanceError):
        p.name = "changed"  # type: ignore[misc]  # frozen


def test_risk_is_optional():
    """RiskModel 可为 None（crypto 本期延后到 MC3 填）。"""
    p = _dummy_profile()
    p2 = MarketProfile(
        name="norisk",
        quote_currency="XXX",
        base_freq="daily",
        provider=p.provider,
        calendar=p.calendar,
        rules=p.rules,
        costs=p.costs,
        universe=p.universe,
        factors=p.factors,
    )
    assert p2.risk is None


def test_registry_register_get_list():
    """registry：register→get 返回同一 profile（缓存），list 含其名。"""
    registry.register("dummy", _dummy_profile)
    got = registry.get("dummy")
    assert got.name == "dummy"
    assert registry.get("dummy") is got  # 缓存：同一实例
    assert "dummy" in registry.list_markets()


def test_registry_unknown_raises():
    """未注册市场 get 抛 KeyError。"""
    with pytest.raises(KeyError):
        registry.get("__nonexistent_market__")

# ==== 来自 test_us_provider.py ====
def _ts(y: int, m: int, d: int) -> int:
    # 美股开盘 ~14:30 UTC（EST）→ UTC 日期与 ET 交易日同日
    return int(dt.datetime(y, m, d, 14, 30, tzinfo=dt.timezone.utc).timestamp())


def _chart_json(
    symbol: str,
    timestamps: list[int],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[int],
    adjcloses: list[float],
) -> bytes:
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"symbol": symbol, "currency": "USD"},
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": opens,
                                "high": highs,
                                "low": lows,
                                "close": closes,
                                "volume": volumes,
                            }
                        ],
                        "adjclose": [{"adjclose": adjcloses}],
                    },
                }
            ],
            "error": None,
        }
    }
    return json.dumps(payload).encode()


# ── 拆股复权 ground-truth（逐值）───────────────────────────
def _split_json() -> bytes:
    # 2:1 拆股：day2→day3 之间价格腰斩，adjclose 把拆股前两日 ×0.5 回填
    ts = [_ts(2024, 1, d) for d in (2, 3, 4, 5)]
    opens = [99.0, 100.0, 50.5, 50.0]
    highs = [101.0, 103.0, 52.0, 52.0]
    lows = [98.0, 99.0, 49.0, 50.0]
    closes = [100.0, 102.0, 50.0, 51.0]
    volumes = [1000, 1200, 2400, 2000]
    adjcloses = [50.0, 51.0, 50.0, 51.0]  # 拆股前两日被 ×0.5
    return _chart_json("TEST", ts, opens, highs, lows, closes, volumes, adjcloses)


def test_split_adjustment_ground_truth() -> None:
    df = parse_chart_json(_split_json(), "TEST").sort("trade_date")
    assert df["ts_code"].unique().to_list() == ["TEST"]
    assert df["trade_date"].to_list() == [dt.date(2024, 1, d) for d in (2, 3, 4, 5)]

    # adj_factor = adjclose / close_raw，逐值
    assert df["adj_factor"].to_list() == pytest.approx([0.5, 0.5, 1.0, 1.0])
    # 复权 close == adjclose（手工）
    assert df["close"].to_list() == pytest.approx([50.0, 51.0, 50.0, 51.0])
    # 复权 open/high/low = raw × adj_factor（逐值手工）
    assert df["open"].to_list() == pytest.approx([49.5, 50.0, 50.5, 50.0])
    assert df["high"].to_list() == pytest.approx([50.5, 51.5, 52.0, 52.0])
    assert df["low"].to_list() == pytest.approx([49.0, 49.5, 49.0, 50.0])
    # amount = 未复权美元成交额 = close_raw × vol_raw（拆股不变量）
    assert df["amount"].to_list() == pytest.approx([100000.0, 122400.0, 120000.0, 102000.0])
    # vol = 原始股数（未复权）
    assert df["vol"].to_list() == pytest.approx([1000.0, 1200.0, 2400.0, 2000.0])

    # 复权后 ret_1d 无拆股跳变（|ret|<0.1）；未复权 close 会有 ~-51% 假跳变
    ret = (df["close"] / df["close"].shift(1) - 1.0).drop_nulls().to_list()
    assert max(abs(r) for r in ret) < 0.1
    raw_ret = (df["adj_factor"].pow(-1) * df["close"])  # = close_raw
    raw_close = raw_ret.to_list()
    raw_jump = raw_close[2] / raw_close[1] - 1.0
    assert raw_jump < -0.4  # 反例：未复权确有拆股假崩（判别力，非恒真）


def test_no_corporate_action_adjusted_equals_raw() -> None:
    # adjclose == close：adj_factor 全 1，复权 == 原始
    ts = [_ts(2024, 2, d) for d in (1, 2, 5)]
    df = parse_chart_json(
        _chart_json("NOCA", ts, [10.0, 11.0, 12.0], [10.5, 11.5, 12.5],
                    [9.5, 10.5, 11.5], [10.0, 11.0, 12.0], [100, 200, 300],
                    [10.0, 11.0, 12.0]),
        "NOCA",
    ).sort("trade_date")
    assert df["adj_factor"].to_list() == pytest.approx([1.0, 1.0, 1.0])
    assert df["close"].to_list() == pytest.approx([10.0, 11.0, 12.0])
    assert df["open"].to_list() == pytest.approx([10.0, 11.0, 12.0])


def test_null_rows_dropped() -> None:
    # Yahoo 有时在区间里塞 null（休市/停牌行）→ 丢弃，不产 NaN 穿透
    ts = [_ts(2024, 3, d) for d in (1, 4, 5)]
    df = parse_chart_json(
        _chart_json("NUL", ts, [10.0, None, 12.0], [10.5, None, 12.5],
                    [9.5, None, 11.5], [10.0, None, 12.0], [100, None, 300],
                    [10.0, None, 12.0]),
        "NUL",
    ).sort("trade_date")
    assert df.height == 2
    assert df["trade_date"].to_list() == [dt.date(2024, 3, 1), dt.date(2024, 3, 5)]


# ── provider fetch_bars + 缓存审计 ─────────────────────────
def _fake_fetch_factory(counter: dict) -> object:
    def fake(url: str) -> bytes:
        counter["n"] = counter.get("n", 0) + 1
        # 从 URL 取 symbol（.../chart/{SYM}?...）
        sym = url.split("/chart/")[1].split("?")[0]
        ts = [_ts(2023, 1, 2), _ts(2023, 1, 3), _ts(2023, 1, 4)]
        return _chart_json(sym, ts, [10.0, 11.0, 12.0], [10.5, 11.5, 12.5],
                           [9.5, 10.5, 11.5], [10.0, 11.0, 12.0], [100, 200, 300],
                           [10.0, 11.0, 12.0])
    return fake


def test_fetch_bars_and_cache_audit(tmp_path) -> None:
    counter: dict = {}
    prov = USDataProvider(cache_root=str(tmp_path), fetch=_fake_fetch_factory(counter),
                          request_interval=0.0)
    bars = prov.fetch_bars(["AAA", "BBB"], "20230101", "20230131")
    assert set(bars["ts_code"].unique().to_list()) == {"AAA", "BBB"}
    assert {"open", "high", "low", "close", "vol", "amount", "adj_factor"}.issubset(bars.columns)
    first = counter["n"]
    assert first == 2  # 两标的各一次

    # 同窗口再取 → 命中缓存，0 次网络
    counter["n"] = 0
    prov2 = USDataProvider(cache_root=str(tmp_path), fetch=_fake_fetch_factory(counter),
                           request_interval=0.0)
    bars2 = prov2.fetch_bars(["AAA", "BBB"], "20230101", "20230131")
    assert counter["n"] == 0
    assert bars2.height == bars.height


def test_fetch_bars_filters_window(tmp_path) -> None:
    counter: dict = {}
    prov = USDataProvider(cache_root=str(tmp_path), fetch=_fake_fetch_factory(counter),
                          request_interval=0.0)
    # 缓存含 1/2–1/4，请求只要 1/3–1/4 → 只返 2 行
    bars = prov.fetch_bars(["AAA"], "20230103", "20230104")
    assert bars.height == 2
    assert bars["trade_date"].min() == dt.date(2023, 1, 3)


def test_fetch_symbol_meta_from_snapshot(tmp_path) -> None:
    prov = USDataProvider(cache_root=str(tmp_path), fetch=_fake_fetch_factory({}),
                          request_interval=0.0)
    meta = prov.fetch_symbol_meta()
    assert "ts_code" in meta.columns
    assert "AAPL" in meta["ts_code"].to_list()
    assert meta.height > 400  # ~490 静态快照

# ==== 来自 test_markets_crypto_provider.py ====
def _ms(y: int, m: int, d: int) -> int:
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1000)


class FakeCCXT:
    """模拟 ccxt binanceusdm 的最小子集（结构对齐官方 unified API）。"""

    def __init__(self):
        # unified symbol -> 日线 [ms, o,h,l,c,v]
        self._ohlcv = {
            "BTC/USDT:USDT": [
                [_ms(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 10.0],
                [_ms(2024, 1, 2), 105.0, 120.0, 104.0, 118.0, 12.0],
                [_ms(2024, 1, 3), 118.0, 119.0, 108.0, 110.0, 8.0],
            ],
            "ETH/USDT:USDT": [
                [_ms(2024, 1, 1), 50.0, 55.0, 48.0, 52.0, 20.0],
                [_ms(2024, 1, 2), 52.0, 53.0, 49.0, 50.0, 22.0],
            ],
        }
        # unified symbol -> funding 事件（每日 3 次，8h 一档）
        self._funding = {
            "BTC/USDT:USDT": [
                {"timestamp": _ms(2024, 1, 1), "fundingRate": 0.0001},
                {"timestamp": _ms(2024, 1, 1) + 8 * 3600_000, "fundingRate": 0.0002},
                {"timestamp": _ms(2024, 1, 1) + 16 * 3600_000, "fundingRate": -0.0001},
                {"timestamp": _ms(2024, 1, 2), "fundingRate": 0.0003},
            ],
        }
        self._oi = {
            "BTC/USDT:USDT": [
                {"timestamp": _ms(2024, 1, 1), "openInterestAmount": 1000.0},
                {"timestamp": _ms(2024, 1, 2), "openInterestAmount": 1100.0},
            ],
        }

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
            "BTC/USDT:USDT": {
                "base": "BTC", "quote": "USDT", "swap": True,
                "info": {"onboardDate": str(_ms(2019, 9, 8))},
            },
            "ETH/USDT:USDT": {
                "base": "ETH", "quote": "USDT", "swap": True, "info": {},
            },
            "BTC/USDT": {"base": "BTC", "quote": "USDT", "swap": False, "info": {}},  # 现货，应剔除
        }


def _provider():
    return CryptoDataProvider(client=FakeCCXT())


def test_is_a_dataprovider():
    assert isinstance(_provider(), DataProvider)


def test_symbol_mapping_roundtrip():
    p = _provider()
    assert p._to_unified("BTCUSDT") == "BTC/USDT:USDT"
    assert p._to_ts_code("BTC/USDT:USDT") == "BTCUSDT"


def test_fetch_bars_schema_and_values():
    p = _provider()
    df = p.fetch_bars(["BTCUSDT", "ETHUSDT"], "20240101", "20240103")
    assert set(df.columns) >= {
        "ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount",
    }
    # BTC 3 天 + ETH 2 天 = 5 行
    assert df.height == 5
    btc = df.filter(pl.col("ts_code") == "BTCUSDT").sort("trade_date")
    # amount = close * vol
    assert btc["amount"][0] == 105.0 * 10.0
    assert str(btc["trade_date"][0]) == "2024-01-01"


def test_fetch_bars_respects_end_date():
    p = _provider()
    df = p.fetch_bars(["BTCUSDT"], "20240101", "20240102")  # 只到 1/2
    assert df.height == 2


def test_fetch_funding_daily_sum():
    """日频 funding = 当日多档 funding 之和（Binance 每 8h 一档）。"""
    p = _provider()
    fd = p.fetch_funding(["BTCUSDT"], "20240101", "20240103")
    assert set(fd.columns) >= {"ts_code", "trade_date", "funding_rate"}
    d1 = fd.filter(pl.col("trade_date") == pl.date(2024, 1, 1))
    # 0.0001 + 0.0002 - 0.0001 = 0.0002
    assert abs(d1["funding_rate"][0] - 0.0002) < 1e-12


def test_fetch_open_interest():
    p = _provider()
    oi = p.fetch_open_interest(["BTCUSDT"], "20240101", "20240103")
    assert set(oi.columns) >= {"ts_code", "trade_date", "open_interest"}
    assert oi.height == 2


def test_fetch_symbol_meta_only_swap_quote():
    p = _provider()
    meta = p.fetch_symbol_meta()
    codes = set(meta["ts_code"].to_list())
    assert "BTCUSDT" in codes and "ETHUSDT" in codes
    # 现货 BTC/USDT(swap=False) 应被剔除 —— 只有 2 个永续
    assert meta.height == 2

# ==== 来自 test_crypto_provider_pagination.py ====
_DAY_MS = 86_400_000
_H8 = 8 * 3600_000
_BASE = 1_704_067_200_000  # 2024-01-01 00:00 UTC


class _PagedCCXT:
    def __init__(self, funding, oi):
        self._f = funding
        self._oi = oi
        self.oi_timeframes: list[str] = []

    def _to_unified(self, s):  # 不用；provider 自己映射
        return s

    def fetch_funding_rate_history(self, symbol, since=None, limit=1000):
        data = [r for r in self._f if since is None or r["timestamp"] >= since]
        return data[:limit]

    def fetch_open_interest_history(self, symbol, timeframe="1h", since=None, limit=1000):
        # 默认 '1h' 模拟真实 ccxt；provider 须显式传 '1d'
        self.oi_timeframes.append(timeframe)
        data = [r for r in self._oi if since is None or r["timestamp"] >= since]
        return data[:limit]

    def load_markets(self):
        return {}


def test_fetch_funding_paginates_beyond_1000():
    # 1200 档 funding（8h 一档）= 400 天，超过单页 1000 档(~333 天)
    funding = [{"timestamp": _BASE + i * _H8, "fundingRate": 0.0001} for i in range(1200)]
    client = _PagedCCXT(funding, [])
    p = CryptoDataProvider(client=client)
    end = "20250204"  # 2024-01-01 + 400 天 ≈ 2025-02-04
    fd = p.fetch_funding(["BTCUSDT"], "20240101", end)
    # 400 天全部拉到（每天 3 档聚合成 1 行）；修复前只 ~333 天
    assert fd.height >= 399, f"长区间 funding 应分页拉全，实得 {fd.height} 天（修复前截断到 ~333）"


def test_fetch_open_interest_daily_timeframe_and_aggregation():
    # 每天 24 个小时级 OI 点，共 2 天 = 48 条
    oi = [{"timestamp": _BASE + d * _DAY_MS + h * 3600_000, "openInterestAmount": 1000.0 + h}
          for d in range(2) for h in range(24)]
    client = _PagedCCXT([], oi)
    p = CryptoDataProvider(client=client)
    result = p.fetch_open_interest(["BTCUSDT"], "20240101", "20240102")
    # 按日聚合 → 每天 1 行（修复前 24 行/日 → 48 行，join 后日频帧爆炸 24 倍）
    assert result.height == 2, f"OI 应按日聚合成 2 行，实得 {result.height}"
    # 每个 (ts_code, trade_date) 唯一
    assert result.select(["ts_code", "trade_date"]).n_unique() == result.height
    # provider 须显式请求 daily timeframe，而非用 ccxt 默认 '1h'
    assert "1d" in client.oi_timeframes, f"应请求 timeframe='1d'，实得 {client.oi_timeframes}"

# ==== 来自 test_futures_continuous.py ====
def _daily(rows: list[tuple]) -> pl.DataFrame:
    # (ts_code, trade_date, open, high, low, close, vol, amount, oi)
    return pl.DataFrame(
        rows,
        schema=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "oi"],
        orient="row",
    )


def _mapping(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema=["ts_code", "trade_date", "mapping_ts_code"], orient="row"
    )


def test_single_roll_ground_truth() -> None:
    """两合约一次展期，逐值对手工计算的后复权 close。

    品种 CU.SHF：d1-d3 主力 A(CU2401)，d4-d5 主力 B(CU2402)。
    A close: 100,102,101（d1-d3）。B close: d3=110, d4=112, d5=111（B 在 d3 已上市交易）。
    展期日 d4：roll_step = A_{d3}/B_{d3} = 101/110。adj_factor: d1-3=1，d4-5=101/110=0.9181818…
    手工后复权 close: [100, 102, 101, 112*0.9181818=102.8363636, 111*0.9181818=101.9181818]
    """
    d = [date(2024, 1, i) for i in range(1, 6)]
    daily = _daily([
        ("CU2401.SHF", d[0], 100, 100, 100, 100.0, 10, 1.0, 5),
        ("CU2401.SHF", d[1], 102, 102, 102, 102.0, 10, 1.0, 5),
        ("CU2401.SHF", d[2], 101, 101, 101, 101.0, 10, 1.0, 5),
        # B 在 d3 已交易（非主力），提供 new_prev_close
        ("CU2402.SHF", d[2], 110, 110, 110, 110.0, 20, 2.0, 8),
        ("CU2402.SHF", d[3], 112, 112, 112, 112.0, 20, 2.0, 8),
        ("CU2402.SHF", d[4], 111, 111, 111, 111.0, 20, 2.0, 8),
    ])
    mapping = _mapping([
        ("CU.SHF", d[0], "CU2401.SHF"),
        ("CU.SHF", d[1], "CU2401.SHF"),
        ("CU.SHF", d[2], "CU2401.SHF"),
        ("CU.SHF", d[3], "CU2402.SHF"),
        ("CU.SHF", d[4], "CU2402.SHF"),
    ])
    out = build_continuous(mapping, daily, fut_codes={"CU"}).sort("trade_date")
    assert out["ts_code"].unique().to_list() == ["CU.SHF"]

    f = 101.0 / 110.0
    expected_close = [100.0, 102.0, 101.0, 112.0 * f, 111.0 * f]
    got = out["close"].to_list()
    for g, e in zip(got, expected_close, strict=True):
        assert abs(g - e) < 1e-9, f"close {g} != {e}"

    expected_adj = [1.0, 1.0, 1.0, f, f]
    for g, e in zip(out["adj_factor"].to_list(), expected_adj, strict=True):
        assert abs(g - e) < 1e-12

    # 展期日 ret = 新主力自身收益（非跨合约跳变）
    ret = (out["close"] / out["close"].shift(1) - 1.0).to_list()
    assert abs(ret[3] - (112.0 / 110.0 - 1.0)) < 1e-9  # roll day = B 自身 d4 收益
    assert abs(ret[4] - (111.0 / 112.0 - 1.0)) < 1e-9
    assert abs(ret[1] - (102.0 / 100.0 - 1.0)) < 1e-9  # 非展期 = A 自身收益

    # 量列不复权（原始）
    assert out.sort("trade_date")["vol"].to_list() == [10, 10, 10, 20, 20]
    assert out.sort("trade_date")["oi"].to_list() == [5, 5, 5, 8, 8]


def test_two_rolls_cumulative_ground_truth() -> None:
    """三合约两次展期，累乘 adj_factor 逐值校验（防单展期恰好巧合）。"""
    d = [date(2024, 1, i) for i in range(1, 7)]
    daily = _daily([
        ("A.C1", d[0], 100, 100, 100, 100.0, 10, 1.0, 5),
        ("A.C1", d[1], 101, 101, 101, 101.0, 10, 1.0, 5),
        ("A.C2", d[1], 200, 200, 200, 200.0, 10, 1.0, 5),  # B d2 (new_prev for roll@d3)
        ("A.C2", d[2], 202, 202, 202, 202.0, 10, 1.0, 5),
        ("A.C2", d[3], 203, 203, 203, 203.0, 10, 1.0, 5),
        ("A.C3", d[3], 50, 50, 50, 50.0, 10, 1.0, 5),      # C d4 (new_prev for roll@d5)
        ("A.C3", d[4], 51, 51, 51, 51.0, 10, 1.0, 5),
        ("A.C3", d[5], 52, 52, 52, 52.0, 10, 1.0, 5),
    ])
    mapping = _mapping([
        ("A.DCE", d[0], "A.C1"), ("A.DCE", d[1], "A.C1"),
        ("A.DCE", d[2], "A.C2"), ("A.DCE", d[3], "A.C2"),
        ("A.DCE", d[4], "A.C3"), ("A.DCE", d[5], "A.C3"),
    ])
    out = build_continuous(mapping, daily, fut_codes={"A"}).sort("trade_date")

    step3 = 101.0 / 200.0   # A_{d2}/B_{d2}
    step5 = 203.0 / 50.0    # B_{d4}/C_{d4}
    c = [1.0, 1.0, step3, step3, step3 * step5, step3 * step5]
    raw_close = [100.0, 101.0, 202.0, 203.0, 51.0, 52.0]
    expected = [r * cc for r, cc in zip(raw_close, c, strict=True)]
    for g, e in zip(out["close"].to_list(), expected, strict=True):
        assert abs(g - e) < 1e-9, f"{g} != {e}"

    ret = (out["close"] / out["close"].shift(1) - 1.0).to_list()
    assert abs(ret[2] - (202.0 / 200.0 - 1.0)) < 1e-9  # roll@d3 = B own
    assert abs(ret[4] - (51.0 / 50.0 - 1.0)) < 1e-9    # roll@d5 = C own
    assert abs(ret[3] - (203.0 / 202.0 - 1.0)) < 1e-9
    assert abs(ret[5] - (52.0 / 51.0 - 1.0)) < 1e-9


def test_secondary_L_code_filtered() -> None:
    """次主力连续（L 后缀）经 fut_codes 过滤，只留主力连续。"""
    d = [date(2024, 1, 1), date(2024, 1, 2)]
    daily = _daily([
        ("CU2401.SHF", d[0], 100, 100, 100, 100.0, 10, 1.0, 5),
        ("CU2401.SHF", d[1], 101, 101, 101, 101.0, 10, 1.0, 5),
        ("CU2402.SHF", d[0], 110, 110, 110, 110.0, 20, 2.0, 8),
        ("CU2402.SHF", d[1], 111, 111, 111, 111.0, 20, 2.0, 8),
    ])
    mapping = _mapping([
        ("CU.SHF", d[0], "CU2401.SHF"), ("CU.SHF", d[1], "CU2401.SHF"),
        ("CUL.SHF", d[0], "CU2402.SHF"), ("CUL.SHF", d[1], "CU2402.SHF"),
    ])
    out = build_continuous(mapping, daily, fut_codes={"CU"})
    assert out["ts_code"].unique().to_list() == ["CU.SHF"]  # CUL.SHF 被过滤


def test_missing_new_prev_close_no_adjust() -> None:
    """新主力在展期前一日无报价 → 该展期不复权（roll_step=1），诚实退化不崩。"""
    d = [date(2024, 1, i) for i in range(1, 4)]
    daily = _daily([
        ("X2401.SHF", d[0], 100, 100, 100, 100.0, 10, 1.0, 5),
        ("X2401.SHF", d[1], 101, 101, 101, 101.0, 10, 1.0, 5),
        # 新主力 X2402 在 d2（前一日）无行情，只在 d3 出现
        ("X2402.SHF", d[2], 111, 111, 111, 111.0, 20, 2.0, 8),
    ])
    mapping = _mapping([
        ("X.SHF", d[0], "X2401.SHF"),
        ("X.SHF", d[1], "X2401.SHF"),
        ("X.SHF", d[2], "X2402.SHF"),  # 展期日，但 new_prev 缺失
    ])
    out = build_continuous(mapping, daily, fut_codes={"X"}).sort("trade_date")
    # 无法算 roll_step → 保持 1.0，close 原样（接受原始跳变，不 NaN 不崩）
    assert out["adj_factor"].to_list() == [1.0, 1.0, 1.0]
    assert out["close"].to_list() == [100.0, 101.0, 111.0]


def test_empty_inputs() -> None:
    empty_m = _mapping([]).clear()
    empty_d = _daily([]).clear()
    out = build_continuous(empty_m, empty_d, fut_codes={"CU"})
    assert out.is_empty()
    assert set(["ts_code", "trade_date", "close", "adj_factor"]).issubset(out.columns)

# ==== 来自 test_ashare_wrap.py ====
def test_calendar_periods_per_year_252():
    cal = AShareCalendar()
    assert cal.periods_per_year() == 252.0
    assert cal.periods_per_year("daily") == 252.0
    assert abs(cal.periods_per_year("monthly") - 12.0) < 1e-9


def test_factorset_leaves_match_operators():
    """叶子字典与 discovery.operators 同源（避免漂移）。"""
    fs = AShareFactorSet()
    assert fs.leaf_features() == LEAF_FEATURES
    assert fs.basic_features() == BASIC_FEATURES


def test_factorset_derived_columns_ashare_convention():
    """A 股派生：vwap=amount/vol, log_vol=ln(vol+1), ret_1d 用 close_adj。"""
    fs = AShareFactorSet()
    bars = pl.DataFrame({
        "ts_code": ["000001.SZ"] * 3,
        "trade_date": [date(2024, 1, i) for i in (2, 3, 4)],
        "close_adj": [10.0, 11.0, 10.5],
        "vol": [100.0, 200.0, 50.0],
        "amount": [1000.0, 2200.0, 525.0],
    })
    out = fs.derived_columns(bars).sort("trade_date")
    assert out["vwap"].to_list() == [10.0, 11.0, 10.5]
    np.testing.assert_allclose(out["log_vol"].to_list(), np.log(np.array([100.0, 200.0, 50.0]) + 1))
    assert out["ret_1d"][0] is None
    assert abs(out["ret_1d"][1] - 0.1) < 1e-12


def test_cost_stamp_tax_asymmetry():
    """卖出含印花税、买入不含（A 股关键差异）。"""
    c = AShareCostModel()
    buy = c.trade_cost("buy", 10000.0)
    sell = c.trade_cost("sell", 10000.0)
    assert abs(buy - 10000.0 * (COMMISSION_RATE + SLIPPAGE_RATE)) < 1e-6
    assert abs(sell - 10000.0 * (COMMISSION_RATE + SLIPPAGE_RATE + STAMP_TAX_RATE)) < 1e-6
    # 卖出比买入贵一个印花税
    assert abs((sell - buy) - 10000.0 * STAMP_TAX_RATE) < 1e-6


def test_cost_carry_long_only():
    """long-only：多头无持有成本；空头计融券利息。"""
    c = AShareCostModel()
    assert c.carry_cost(10000.0, 5) == 0.0  # 多头无成本
    assert c.carry_cost(-10000.0, 5) > 0.0  # 空头有融券利息


def test_rules_long_only_t1():
    r = AShareTradingRules()
    assert r.allow_short is False
    assert r.settlement_lag == 1
    assert r.execution_price_col == "open"
    bars = pl.DataFrame({"ts_code": ["A", "B"], "vol": [10.0, 0.0]})
    assert r.tradable_mask(bars, "buy").to_list() == [True, False]


def test_ashare_provider_fetch_bars_rejects_non_daily():
    """AShareDataProvider 仅经 fetch_daily 取日频；非 daily freq 须显式报错，
    而非静默返回日频数据（与 CryptoDataProvider.fetch_funding 的守卫一致）。
    """
    from factorzen.markets.ashare.provider import AShareDataProvider

    p = AShareDataProvider()
    for bad in ["weekly", "monthly", "1min", "60min"]:
        with pytest.raises(ValueError):
            p.fetch_bars(None, "20240101", "20240131", freq=bad)


def test_ashare_provider_fetch_bars_daily_delegates(monkeypatch):
    """freq='daily'（默认）委托 core.loader.fetch_daily，参数透传。"""
    import factorzen.core.loader as loader
    from factorzen.markets.ashare.provider import AShareDataProvider

    sentinel = pl.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [date(2024, 1, 2)]})
    called: dict = {}

    def _fake_fetch_daily(start, end, ts_codes=None):
        called["args"] = (start, end, ts_codes)
        return sentinel

    monkeypatch.setattr(loader, "fetch_daily", _fake_fetch_daily)
    out = AShareDataProvider().fetch_bars(["000001.SZ"], "20240101", "20240131")
    assert out.equals(sentinel)
    assert called["args"] == ("20240101", "20240131", ["000001.SZ"])


def test_registry_get_ashare():
    p = registry.get("ashare")
    assert isinstance(p, MarketProfile)
    assert p.name == "ashare"
    assert p.quote_currency == "CNY"
    assert p.risk is None
    assert p.calendar.periods_per_year() == 252.0


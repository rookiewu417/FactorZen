"""美股 Yahoo provider 复权 ground-truth + 缓存审计（离线，注入 fetch，无网络）。

复权硬契约（见 markets/us/provider.py）：Yahoo OHLC 未复权、adjclose 已复权，
``adj_factor = adjclose / close_raw`` 比率复权 OHLC，使 ret_1d 无拆股跳变。
本测试**构造含 2:1 拆股的 canned JSON，逐值对手工计算**（非恒真断言，陷阱#1）。
"""
from __future__ import annotations

import datetime as dt
import json

import pytest

from factorzen.markets.us.provider import USDataProvider, parse_chart_json


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

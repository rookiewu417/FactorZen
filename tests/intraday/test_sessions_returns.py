"""test_intraday_sessions.py：A 股 session/频率单一真源：canonicalize/resample/bar_index
test_intraday_returns.py：分钟前向收益列、跨日边界与跨股票不泄漏
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from factorzen.intraday.evaluation.returns import compute_intraday_fwd_returns
from factorzen.intraday.sessions import (
    ASHARE_BAR_FREQS,
    BAR_LABEL_CONVENTION,
    canonicalize_minute,
    normalize_freq,
    resample_intraday,
    session_bar_index,
)


# ==== 来自 test_intraday_sessions.py ====
def _dt(h: int, m: int, day: int = 2) -> datetime:
    return datetime(2024, 1, day, h, m, 0)

def _mini_frame() -> pl.DataFrame:
    """含竞价、午休边界、盘后、vol=0 的微型帧。"""
    rows = [
        # 09:30 竞价
        ("000001.SZ", _dt(9, 30), 10.0, 10.0, 10.0, 10.0, 100, 1000.0),
        # 上午连续
        ("000001.SZ", _dt(9, 31), 10.1, 10.2, 10.0, 10.15, 200, 2000.0),
        ("000001.SZ", _dt(9, 32), 10.15, 10.3, 10.1, 10.2, 150, 1500.0),
        ("000001.SZ", _dt(9, 33), 10.2, 10.25, 10.15, 10.2, 100, 1000.0),
        ("000001.SZ", _dt(9, 34), 10.2, 10.4, 10.2, 10.35, 120, 1200.0),
        ("000001.SZ", _dt(9, 35), 10.35, 10.4, 10.3, 10.3, 80, 800.0),
        # 11:26..11:30 边界
        ("000001.SZ", _dt(11, 26), 11.0, 11.1, 10.9, 11.0, 50, 500.0),
        ("000001.SZ", _dt(11, 27), 11.0, 11.05, 10.95, 11.0, 40, 400.0),
        ("000001.SZ", _dt(11, 28), 11.0, 11.0, 10.9, 10.95, 30, 300.0),
        ("000001.SZ", _dt(11, 29), 10.95, 11.0, 10.9, 10.9, 20, 200.0),
        ("000001.SZ", _dt(11, 30), 10.9, 10.95, 10.85, 10.9, 60, 600.0),
        # 下午开盘
        ("000001.SZ", _dt(13, 1), 10.9, 11.0, 10.85, 10.95, 70, 700.0),
        ("000001.SZ", _dt(13, 2), 10.95, 11.0, 10.9, 11.0, 55, 550.0),
        ("000001.SZ", _dt(13, 3), 11.0, 11.1, 10.95, 11.05, 45, 450.0),
        ("000001.SZ", _dt(13, 4), 11.05, 11.1, 11.0, 11.0, 35, 350.0),
        ("000001.SZ", _dt(13, 5), 11.0, 11.05, 10.95, 11.0, 25, 250.0),
        # 收盘 + vol=0 价格延续
        ("000001.SZ", _dt(14, 59), 11.2, 11.2, 11.2, 11.2, 0, 0.0),
        ("000001.SZ", _dt(15, 0), 11.2, 11.3, 11.15, 11.25, 500, 5500.0),
        # 盘后应被剔除
        ("000001.SZ", _dt(15, 5), 11.25, 11.25, 11.25, 11.25, 10, 100.0),
    ]
    return pl.DataFrame(
        {
            "ts_code": [r[0] for r in rows],
            "trade_time": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us")),
            "open": [r[2] for r in rows],
            "high": [r[3] for r in rows],
            "low": [r[4] for r in rows],
            "close": [r[5] for r in rows],
            "vol": pl.Series([r[6] for r in rows], dtype=pl.Int64),
            "amount": [r[7] for r in rows],
        }
    )

class TestConstants:
    def test_label_convention_is_end(self) -> None:
        assert BAR_LABEL_CONVENTION == "end"

    def test_freq_table(self) -> None:
        assert ASHARE_BAR_FREQS["1min"].minutes == 1
        assert ASHARE_BAR_FREQS["1min"].bars_per_day == 240
        assert ASHARE_BAR_FREQS["5min"].bars_per_day == 48
        assert ASHARE_BAR_FREQS["60min"].bars_per_day == 4

class TestNormalizeFreq:
    def test_valid(self) -> None:
        assert normalize_freq("5min") == "5min"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="未知频率"):
            normalize_freq("7min")

class TestCanonicalize:
    def test_drops_after_hours_keeps_boundaries(self) -> None:
        df = _mini_frame()
        out = canonicalize_minute(df.lazy()).collect()
        times = set(
            (t.hour, t.minute) for t in out["trade_time"].to_list()  # type: ignore[union-attr]
        )
        assert (15, 5) not in times
        assert (9, 30) in times
        assert (11, 30) in times
        assert (13, 1) in times
        assert (15, 0) in times
        assert out.height == df.height - 1

class TestSessionBarIndex:
    def test_key_indices(self) -> None:
        times = [
            _dt(9, 30),
            _dt(9, 31),
            _dt(11, 30),
            _dt(13, 1),
            _dt(15, 0),
            _dt(15, 5),  # non-canonical → null
        ]
        df = pl.DataFrame(
            {"trade_time": pl.Series(times, dtype=pl.Datetime("us"))}
        ).with_columns(session_bar_index().alias("idx"))
        idxs = df["idx"].to_list()
        assert idxs == [0, 1, 120, 121, 240, None]

class TestResample5min:
    def test_auction_merges_into_0935(self) -> None:
        """09:30+09:31..09:35 → 标签 09:35；open=竞价 open，vol=六根之和。"""
        df = _mini_frame()
        out = resample_intraday(df, "5min")
        row = out.filter(
            (pl.col("trade_time").dt.hour() == 9)
            & (pl.col("trade_time").dt.minute() == 35)
        )
        assert row.height == 1
        assert row["open"][0] == pytest.approx(10.0)
        # vol = 100+200+150+100+120+80 = 750
        assert row["vol"][0] == 750
        # amount = 1000+2000+1500+1000+1200+800 = 7500
        assert row["amount"][0] == pytest.approx(7500.0)
        assert row["close"][0] == pytest.approx(10.3)
        assert row["high"][0] == pytest.approx(10.4)
        assert row["low"][0] == pytest.approx(10.0)

    def test_morning_close_bucket_1130(self) -> None:
        """11:26..11:30 → 标签 11:30。"""
        df = _mini_frame()
        out = resample_intraday(df, "5min")
        row = out.filter(
            (pl.col("trade_time").dt.hour() == 11)
            & (pl.col("trade_time").dt.minute() == 30)
        )
        assert row.height == 1
        assert row["open"][0] == pytest.approx(11.0)
        assert row["close"][0] == pytest.approx(10.9)
        # vol = 50+40+30+20+60 = 200
        assert row["vol"][0] == 200

    def test_afternoon_open_no_lunch_swallow(self) -> None:
        """13:01..13:05 → 标签 13:05；不吞午休。"""
        df = _mini_frame()
        out = resample_intraday(df, "5min")
        row = out.filter(
            (pl.col("trade_time").dt.hour() == 13)
            & (pl.col("trade_time").dt.minute() == 5)
        )
        assert row.height == 1
        assert row["open"][0] == pytest.approx(10.9)
        assert row["close"][0] == pytest.approx(11.0)
        # vol = 70+55+45+35+25 = 230
        assert row["vol"][0] == 230
        # 不应出现 12:xx 或 13:00 标签
        labels = [
            (t.hour, t.minute) for t in out["trade_time"].to_list()  # type: ignore[union-attr]
        ]
        assert (13, 0) not in labels
        assert all(h != 12 for h, _ in labels)

    def test_close_1500_bucket(self) -> None:
        """15:00 归入标签 15:00 桶（与 14:59 等同桶时合并）。"""
        df = _mini_frame()
        out = resample_intraday(df, "5min")
        row = out.filter(
            (pl.col("trade_time").dt.hour() == 15)
            & (pl.col("trade_time").dt.minute() == 0)
        )
        assert row.height == 1
        # 14:56..15:00 桶；帧内只有 14:59 + 15:00
        assert row["vol"][0] == 500  # 0 + 500
        assert row["close"][0] == pytest.approx(11.25)
        assert row["open"][0] == pytest.approx(11.2)

    def test_after_hours_dropped(self) -> None:
        df = _mini_frame()
        out = resample_intraday(df, "5min")
        times = [
            (t.hour, t.minute) for t in out["trade_time"].to_list()  # type: ignore[union-attr]
        ]
        assert (15, 5) not in times

class TestResample30min:
    def test_morning_buckets(self) -> None:
        """30min：上午末桶标签 11:30。"""
        # 构造完整 09:30 + 若干 bar 以触发 11:30 桶
        rows_t = [_dt(9, 30)] + [_dt(11, m) for m in range(1, 31)]
        # 11:01..11:30 → idx 91..120 → bucket 3 → end 11:30
        n = len(rows_t)
        df = pl.DataFrame(
            {
                "ts_code": ["000001.SZ"] * n,
                "trade_time": pl.Series(rows_t, dtype=pl.Datetime("us")),
                "open": [10.0] * n,
                "high": [10.5] * n,
                "low": [9.5] * n,
                "close": [10.2] * n,
                "vol": pl.Series([1] * n, dtype=pl.Int64),
                "amount": [1.0] * n,
            }
        )
        out = resample_intraday(df, "30min")
        labels = [
            (t.hour, t.minute) for t in out["trade_time"].to_list()  # type: ignore[union-attr]
        ]
        assert (11, 30) in labels
        # 竞价进第一桶 end=10:00
        assert (10, 0) in labels

class TestResample60min:
    def test_four_session_labels(self) -> None:
        """60min：上午 10:30/11:30，下午 14:00/15:00。"""
        # 每小时取代表性 bar + 竞价
        times = [
            _dt(9, 30),
            _dt(9, 31),
            _dt(10, 30),
            _dt(10, 31),
            _dt(11, 30),
            _dt(13, 1),
            _dt(14, 0),
            _dt(14, 1),
            _dt(15, 0),
        ]
        n = len(times)
        df = pl.DataFrame(
            {
                "ts_code": ["000001.SZ"] * n,
                "trade_time": pl.Series(times, dtype=pl.Datetime("us")),
                "open": [float(i) for i in range(n)],
                "high": [float(i) + 0.5 for i in range(n)],
                "low": [float(i) - 0.5 for i in range(n)],
                "close": [float(i) + 0.1 for i in range(n)],
                "vol": pl.Series([10] * n, dtype=pl.Int64),
                "amount": [100.0] * n,
            }
        )
        out = resample_intraday(df, "60min")
        labels = sorted(
            (t.hour, t.minute) for t in out["trade_time"].to_list()  # type: ignore[union-attr]
        )
        assert labels == [(10, 30), (11, 30), (14, 0), (15, 0)]

class TestResample1min:
    def test_auction_into_0931_and_240_bars(self) -> None:
        """freq=1min：竞价并入 09:31；完整日恰 240 根。"""
        # 构造完整 241 根（含 09:30）
        times = [_dt(9, 30)]
        # 09:31..11:30
        for mins in range(9 * 60 + 31, 11 * 60 + 30 + 1):
            times.append(_dt(mins // 60, mins % 60))
        # 13:01..15:00
        for mins in range(13 * 60 + 1, 15 * 60 + 0 + 1):
            times.append(_dt(mins // 60, mins % 60))
        assert len(times) == 241  # 1 auction + 240 regular

        n = len(times)
        vols = [1000] + [1] * (n - 1)  # 竞价大量
        df = pl.DataFrame(
            {
                "ts_code": ["000001.SZ"] * n,
                "trade_time": pl.Series(times, dtype=pl.Datetime("us")),
                "open": [10.0] * n,
                "high": [10.5] * n,
                "low": [9.5] * n,
                "close": [10.1] * n,
                "vol": pl.Series(vols, dtype=pl.Int64),
                "amount": [float(v) for v in vols],
            }
        )
        out = resample_intraday(df, "1min").sort(["ts_code", "trade_time"])
        assert out.height == 240
        # 首 bar 应为 09:31，vol = 竞价 1000 + 09:31 的 1 = 1001
        first = out.row(0, named=True)
        assert first["trade_time"].hour == 9
        assert first["trade_time"].minute == 31
        assert first["vol"] == 1001
        # 末 bar 15:00
        last = out.row(-1, named=True)
        assert last["trade_time"].hour == 15
        assert last["trade_time"].minute == 0

class TestEdgeCases:
    def test_empty_frame(self) -> None:
        empty = pl.DataFrame(
            schema={
                "ts_code": pl.String,
                "trade_time": pl.Datetime("us"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "vol": pl.Int64,
                "amount": pl.Float64,
            }
        )
        out = resample_intraday(empty, "5min")
        assert out.is_empty()
        assert out.schema["vol"] == pl.Int64
        assert out.schema["trade_time"] == pl.Datetime("us")

    def test_partial_bars_halt(self) -> None:
        """临停：某 5min 桶只有 2 根 bar，正常聚合不崩。"""
        times = [_dt(9, 31), _dt(9, 32)]  # 不完整 09:35 桶
        df = pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ"],
                "trade_time": pl.Series(times, dtype=pl.Datetime("us")),
                "open": [10.0, 10.1],
                "high": [10.2, 10.3],
                "low": [9.9, 10.0],
                "close": [10.1, 10.2],
                "vol": pl.Series([10, 20], dtype=pl.Int64),
                "amount": [100.0, 200.0],
            }
        )
        out = resample_intraday(df, "5min")
        assert out.height == 1
        assert out["trade_time"][0].minute == 35  # type: ignore[union-attr]
        assert out["vol"][0] == 30
        assert out["open"][0] == pytest.approx(10.0)
        assert out["close"][0] == pytest.approx(10.2)

# ==== 来自 test_intraday_returns.py ====
def _make_minute_df() -> pl.DataFrame:
    """3 只股票，每只 10 根 bar。"""
    rows = []
    base_time = datetime(2026, 5, 16, 9, 30, 0)
    for ts in ["000001.SZ", "000002.SZ", "000003.SZ"]:
        for i in range(10):
            rows.append(
                {
                    "ts_code": ts,
                    "trade_time": base_time + timedelta(minutes=i),
                    "close": 10.0 + i * 0.1,
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("trade_time").cast(pl.Datetime))

def test_fwd_ret_1bar_value():
    df = compute_intraday_fwd_returns(_make_minute_df(), periods=[1])
    row = df.filter((pl.col("ts_code") == "000001.SZ") & (pl.col("trade_time").dt.minute() == 30))
    expected = (10.1 - 10.0) / 10.0
    assert abs(row["fwd_ret_1bar"][0] - expected) < 1e-9

def test_no_cross_stock_leakage():
    df = compute_intraday_fwd_returns(_make_minute_df(), periods=[1])
    for ts in ["000001.SZ", "000002.SZ", "000003.SZ"]:
        last_val = df.filter(pl.col("ts_code") == ts).sort("trade_time").tail(1)["fwd_ret_1bar"][0]
        assert last_val is None, f"{ts} 最后一行应为 null，实为 {last_val}"

def test_fwd_return_does_not_cross_trading_day_boundary():
    df = pl.DataFrame(
        {
            "trade_time": [
                datetime(2024, 1, 2, 14, 59),
                datetime(2024, 1, 2, 15, 0),
                datetime(2024, 1, 3, 9, 30),
                datetime(2024, 1, 3, 9, 31),
            ],
            "ts_code": ["000001.SZ"] * 4,
            "close": [100.0, 101.0, 200.0, 202.0],
        }
    )

    out = compute_intraday_fwd_returns(df, periods=[1])

    values = out["fwd_ret_1bar"].to_list()
    assert values[0] == pytest.approx(0.01)
    assert values[1] is None
    assert values[2] == pytest.approx(0.01)
    assert values[3] is None

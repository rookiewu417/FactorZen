"""合并自: test_minute_data.py, test_sessions_returns.py
目标: test_minute_sessions.py

--- 来源 test_minute_data.py ---
test_minute_ingest.py：分钟入库 schema 归一、幂等、月分区合并与 manifest
test_resample_equiv.py：resample_intraday 与内嵌旧实现对齐的等价性锁定

--- 来源 test_sessions_returns.py ---
test_intraday_sessions.py：A 股 session/频率单一真源：canonicalize/resample/bar_index
test_intraday_returns.py：分钟前向收益列、跨日边界与跨股票不泄漏
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from factorzen.config.settings import DATA_RAW_MINUTE
from factorzen.core.storage import load_parquet, save_parquet
from factorzen.dataio.minute_ingest import ingest_minute_files
from factorzen.intraday.evaluation.returns import compute_intraday_fwd_returns
from factorzen.intraday.sessions import (
    ASHARE_BAR_FREQS,
    BAR_LABEL_CONVENTION,
    _bucket_end_minutes,
    _canonical_mask,
    _empty_bars,
    canonicalize_minute,
    normalize_freq,
    resample_intraday,
    session_bar_index,
)


# ==== 来自 test_minute_data.py ====
# ==== 来自 test_minute_ingest.py ====
def _source_frame(code_col: str = "code") -> pl.DataFrame:
    return pl.DataFrame(
        {
            code_col: ["000001.SZ", "000001.SZ", "000002.SZ"],
            "trade_time": [
                "2024-01-02 09:31:00",
                "2024-01-02 09:32:00",
                "2024-02-01 09:31:00",
            ],
            "open": [10, 11, 20],
            "high": [11, 12, 21],
            "low": [9, 10, 19],
            "close": [10.5, 11.5, 20.5],
            "vol": [100.2, 110.8, 200.0],
            "amount": [1000, 1100, 2000],
            "unused": [1, 2, 3],
        }
    )


def test_minute_ingest_suite(tmp_path):
    """test_minute_setting_matches_loader_storage_namespace；test_ingest_normalizes_schema_preserves_bars_and_is_idempotent；test_ingest_merges_into_existing_partial_month_instead_of_skipping；test_ingest_month_filter_limits_written_partitions；test_cli_writes_reproducibility_manifest_and_sentinel"""
    # -- 原 test_minute_setting_matches_loader_storage_namespace --
    def _section_0_test_minute_setting_matches_loader_storage_namespace():
        assert DATA_RAW_MINUTE.name == "minute_1min"

    _section_0_test_minute_setting_matches_loader_storage_namespace()

    # -- 原 test_ingest_normalizes_schema_preserves_bars_and_is_idempotent --
    def _section_1_test_ingest_normalizes_schema_preserves_bars_and_is_idempotent(tmp_path):
        source = tmp_path / "source.parquet"
        raw = tmp_path / "raw"
        _source_frame().write_parquet(source)

        report = ingest_minute_files([source], base_dir=raw)
        first = load_parquet("minute_1min", base_dir=raw, date_col="trade_time").collect()

        assert report.rows_by_month == {"202401": 2, "202402": 1}
        assert first.columns == [
            "ts_code",
            "trade_time",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "amount",
        ]
        assert first.schema["trade_time"] == pl.Datetime("us")
        assert first.schema["vol"] == pl.Int64
        assert first.height == 3
        assert first.filter(pl.col("ts_code") == "000001.SZ").height == 2

        ingest_minute_files([source], base_dir=raw)
        second = load_parquet("minute_1min", base_dir=raw, date_col="trade_time").collect()
        assert second.sort(["trade_time", "ts_code"]).equals(
            first.sort(["trade_time", "ts_code"])
        )

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_ingest_normalizes_schema_preserves_bars_and_is_idempotent(_tp1)

    # -- 原 test_ingest_merges_into_existing_partial_month_instead_of_skipping --
    def _section_2_test_ingest_merges_into_existing_partial_month_instead_of_skipping(tmp_path):
        raw = tmp_path / "raw"
        existing = pl.DataFrame(
            {
                "ts_code": ["000003.SZ"],
                "trade_time": [datetime(2024, 1, 2, 9, 31)],
                "open": [30.0],
                "high": [31.0],
                "low": [29.0],
                "close": [30.5],
                "vol": [300],
                "amount": [3000.0],
            }
        )
        save_parquet(existing, "minute_1min", date_col="trade_time", base_dir=raw)
        source = tmp_path / "gapfill.parquet"
        _source_frame("ts_code").filter(pl.col("trade_time").str.starts_with("2024-01")).write_parquet(
            source
        )

        ingest_minute_files([source], base_dir=raw)
        merged = load_parquet("minute_1min", base_dir=raw, date_col="trade_time").collect()

        assert merged.height == 3
        assert set(merged["ts_code"].to_list()) == {"000001.SZ", "000003.SZ"}

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_ingest_merges_into_existing_partial_month_instead_of_skipping(_tp2)

    # -- 原 test_ingest_month_filter_limits_written_partitions --
    def _section_3_test_ingest_month_filter_limits_written_partitions(tmp_path):
        source = tmp_path / "source.parquet"
        raw = tmp_path / "raw"
        _source_frame().write_parquet(source)

        report = ingest_minute_files([source], base_dir=raw, months=["202402"])

        assert report.rows_by_month == {"202402": 1}
        assert not (raw / "minute_1min" / "year=2024" / "month=01").exists()
        assert (raw / "minute_1min" / "year=2024" / "month=02" / "data.parquet").is_file()

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_ingest_month_filter_limits_written_partitions(_tp3)

    # -- 原 test_cli_writes_reproducibility_manifest_and_sentinel --
    def _section_4_test_cli_writes_reproducibility_manifest_and_sentinel(tmp_path, mp):
        from tools import ingest_minute as cli

        source = tmp_path / "source.parquet"
        raw = tmp_path / "raw"
        workspace = tmp_path / "workspace"
        _source_frame().write_parquet(source)
        mp.setattr(cli, "WORKSPACE_DIR", workspace)

        assert cli.main([str(source), "--data-root", str(raw), "--run-id", "test-run"]) == 0

        run_dir = workspace / "data_ingest" / "test-run"
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "success"
        assert manifest["git_sha"]
        assert manifest["window"] == {"months": ["202401", "202402"]}
        assert manifest["result"]["source_files"] == 1
        assert (run_dir / "ingest.done").is_file()

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_cli_writes_reproducibility_manifest_and_sentinel(_tp4, mp)


# ==== 来自 test_resample_equiv.py ====
def _dt__minute_data(h: int, m: int, day: int = 2, month: int = 1) -> datetime:
    return datetime(2024, month, day, h, m, 0)


def _resample_intraday_ref(
    bars: pl.DataFrame,
    freq: str,
    *,
    time_col: str = "trade_time",
) -> pl.DataFrame:
    """W3-A 优化前的 resample 实现（锁定语义）。"""
    freq = normalize_freq(freq)
    k = ASHARE_BAR_FREQS[freq].minutes

    if bars.is_empty():
        return _empty_bars()

    filtered = bars.filter(_canonical_mask(time_col))
    if filtered.is_empty():
        return _empty_bars()

    idx = session_bar_index(time_col)
    work = (
        filtered.sort([time_col])
        .with_columns(
            idx.alias("_idx"),
            pl.col(time_col).dt.date().alias("_date"),
        )
        .with_columns(
            pl.when(pl.col("_idx") == 0)
            .then(pl.lit(0, dtype=pl.Int32))
            .otherwise(((pl.col("_idx") - 1) // k).cast(pl.Int32))
            .alias("_bucket")
        )
        .filter(pl.col("_idx").is_not_null())
    )

    if work.is_empty():
        return _empty_bars()

    return (
        work.group_by(["ts_code", "_date", "_bucket"], maintain_order=True)
        .agg(
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("vol").sum().cast(pl.Int64).alias("vol"),
            pl.col("amount").sum().cast(pl.Float64).alias("amount"),
        )
        .with_columns(
            ((pl.col("_bucket") + 1) * k).alias("_end_idx"),
        )
        .with_columns(
            (
                pl.col("_date").cast(pl.Datetime("us"))
                + pl.duration(minutes=_bucket_end_minutes(pl.col("_end_idx")))
            ).alias(time_col)
        )
        .select(
            pl.col("ts_code").cast(pl.String),
            pl.col(time_col).cast(pl.Datetime("us")),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("vol").cast(pl.Int64),
            pl.col("amount").cast(pl.Float64),
        )
        .sort(["ts_code", time_col])
    )


def _assert_equiv(src: pl.DataFrame, freq: str = "5min") -> None:
    keys = ["ts_code", "trade_time"]
    exp = _resample_intraday_ref(src, freq).sort(keys)
    got = resample_intraday(src, freq).sort(keys)
    assert_frame_equal(got, exp, check_column_order=True, check_dtypes=True)


class TestResampleBoundaryFixtures:
    def test_resample_boundary_fixtures_suite(self):
        """午休边界：11:30 与 13:01 不跨桶吞并。；开盘竞价 + 收盘 15:00 标签。；不足一个完整 bar 的残段 + 单股票单日。；乱序多股票输入：first/last 仍按时间正确。；test_empty_frame"""
        # -- 原 test_lunch_boundary_bars --
        rows = [
            ("000001.SZ", _dt__minute_data(11, 28), 10.0, 10.1, 9.9, 10.0, 10, 100.0),
            ("000001.SZ", _dt__minute_data(11, 29), 10.0, 10.2, 9.9, 10.1, 20, 200.0),
            ("000001.SZ", _dt__minute_data(11, 30), 10.1, 10.3, 10.0, 10.2, 30, 300.0),
            ("000001.SZ", _dt__minute_data(13, 1), 10.2, 10.4, 10.1, 10.3, 40, 400.0),
            ("000001.SZ", _dt__minute_data(13, 2), 10.3, 10.5, 10.2, 10.4, 50, 500.0),
        ]
        df = pl.DataFrame(
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
        _assert_equiv(df, "5min")
        out = resample_intraday(df, "5min")
        labels = [
            (t.hour, t.minute) for t in out["trade_time"].to_list()  # type: ignore[union-attr]
        ]
        assert (11, 30) in labels
        assert (13, 5) in labels
        assert (13, 0) not in labels

        # -- 原 test_open_close_session_labels --
        rows = [
            ("000001.SZ", _dt__minute_data(9, 30), 10.0, 10.0, 10.0, 10.0, 100, 1000.0),
            ("000001.SZ", _dt__minute_data(9, 31), 10.0, 10.2, 9.9, 10.1, 200, 2000.0),
            ("000001.SZ", _dt__minute_data(9, 35), 10.1, 10.3, 10.0, 10.2, 50, 500.0),
            ("000001.SZ", _dt__minute_data(14, 58), 11.0, 11.1, 10.9, 11.0, 10, 100.0),
            ("000001.SZ", _dt__minute_data(14, 59), 11.0, 11.0, 11.0, 11.0, 0, 0.0),
            ("000001.SZ", _dt__minute_data(15, 0), 11.0, 11.2, 10.9, 11.1, 500, 5500.0),
            ("000001.SZ", _dt__minute_data(15, 5), 11.1, 11.1, 11.1, 11.1, 1, 1.0),  # drop
        ]
        df = pl.DataFrame(
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
        _assert_equiv(df, "5min")

        # -- 原 test_partial_bucket_and_single_stock_day --
        rows = [
            ("000001.SZ", _dt__minute_data(10, 1), 10.0, 10.1, 9.9, 10.05, 10, 100.0),
            ("000001.SZ", _dt__minute_data(10, 2), 10.05, 10.2, 10.0, 10.1, 20, 200.0),
            # 10:01-10:02 → 桶 end 10:05，残段 3 分钟缺失
        ]
        df = pl.DataFrame(
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
        _assert_equiv(df, "5min")
        out = resample_intraday(df, "5min")
        assert out.height == 1
        assert out["vol"][0] == 30
        assert out["open"][0] == pytest.approx(10.0)
        assert out["close"][0] == pytest.approx(10.1)

        # -- 原 test_unsorted_multi_stock_input --
        rows = [
            ("000002.SZ", _dt__minute_data(9, 32), 20.0, 20.1, 19.9, 20.05, 5, 100.0),
            ("000001.SZ", _dt__minute_data(9, 31), 10.0, 10.1, 9.9, 10.05, 10, 100.0),
            ("000001.SZ", _dt__minute_data(9, 33), 10.1, 10.2, 10.0, 10.15, 20, 200.0),
            ("000002.SZ", _dt__minute_data(9, 31), 20.0, 20.2, 19.8, 20.1, 8, 160.0),
            ("000001.SZ", _dt__minute_data(9, 32), 10.05, 10.15, 10.0, 10.1, 15, 150.0),
        ]
        df = pl.DataFrame(
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
        _assert_equiv(df, "5min")

        # -- 原 test_empty_frame --
        _assert_equiv(_empty_bars(), "5min")


class TestResampleRealMonthSample:
    @pytest.fixture(scope="module")
    def minute_sample(self) -> pl.DataFrame | None:
        path = Path("data/raw/minute_1min/year=2024/month=06/data.parquet")
        if not path.exists():
            return None
        # 截断到若干千只股票以控时
        codes = (
            pl.scan_parquet(path)
            .select("ts_code")
            .unique()
            .collect()
            .head(800)["ts_code"]
            .to_list()
        )
        return (
            pl.scan_parquet(path)
            .filter(pl.col("ts_code").is_in(codes))
            .select(["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"])
            .collect()
        )

    def test_real_2024_06_truncated_equiv(self, minute_sample: pl.DataFrame | None) -> None:
        if minute_sample is None:
            pytest.skip("no data/raw/minute_1min/year=2024/month=06")
        _assert_equiv(minute_sample, "5min")

# ==== 来自 test_sessions_returns.py ====
# ==== 来自 test_intraday_sessions.py ====
def _dt__sessions_returns(h: int, m: int, day: int = 2) -> datetime:
    return datetime(2024, 1, day, h, m, 0)

def _mini_frame() -> pl.DataFrame:
    """含竞价、午休边界、盘后、vol=0 的微型帧。"""
    rows = [
        # 09:30 竞价
        ("000001.SZ", _dt__sessions_returns(9, 30), 10.0, 10.0, 10.0, 10.0, 100, 1000.0),
        # 上午连续
        ("000001.SZ", _dt__sessions_returns(9, 31), 10.1, 10.2, 10.0, 10.15, 200, 2000.0),
        ("000001.SZ", _dt__sessions_returns(9, 32), 10.15, 10.3, 10.1, 10.2, 150, 1500.0),
        ("000001.SZ", _dt__sessions_returns(9, 33), 10.2, 10.25, 10.15, 10.2, 100, 1000.0),
        ("000001.SZ", _dt__sessions_returns(9, 34), 10.2, 10.4, 10.2, 10.35, 120, 1200.0),
        ("000001.SZ", _dt__sessions_returns(9, 35), 10.35, 10.4, 10.3, 10.3, 80, 800.0),
        # 11:26..11:30 边界
        ("000001.SZ", _dt__sessions_returns(11, 26), 11.0, 11.1, 10.9, 11.0, 50, 500.0),
        ("000001.SZ", _dt__sessions_returns(11, 27), 11.0, 11.05, 10.95, 11.0, 40, 400.0),
        ("000001.SZ", _dt__sessions_returns(11, 28), 11.0, 11.0, 10.9, 10.95, 30, 300.0),
        ("000001.SZ", _dt__sessions_returns(11, 29), 10.95, 11.0, 10.9, 10.9, 20, 200.0),
        ("000001.SZ", _dt__sessions_returns(11, 30), 10.9, 10.95, 10.85, 10.9, 60, 600.0),
        # 下午开盘
        ("000001.SZ", _dt__sessions_returns(13, 1), 10.9, 11.0, 10.85, 10.95, 70, 700.0),
        ("000001.SZ", _dt__sessions_returns(13, 2), 10.95, 11.0, 10.9, 11.0, 55, 550.0),
        ("000001.SZ", _dt__sessions_returns(13, 3), 11.0, 11.1, 10.95, 11.05, 45, 450.0),
        ("000001.SZ", _dt__sessions_returns(13, 4), 11.05, 11.1, 11.0, 11.0, 35, 350.0),
        ("000001.SZ", _dt__sessions_returns(13, 5), 11.0, 11.05, 10.95, 11.0, 25, 250.0),
        # 收盘 + vol=0 价格延续
        ("000001.SZ", _dt__sessions_returns(14, 59), 11.2, 11.2, 11.2, 11.2, 0, 0.0),
        ("000001.SZ", _dt__sessions_returns(15, 0), 11.2, 11.3, 11.15, 11.25, 500, 5500.0),
        # 盘后应被剔除
        ("000001.SZ", _dt__sessions_returns(15, 5), 11.25, 11.25, 11.25, 11.25, 10, 100.0),
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

def test_resample_freq_table_suite():
    """test_label_convention_is_end；test_freq_table；test_valid；test_unknown_raises；test_drops_after_hours_keeps_boundaries；test_key_indices"""
    # -- 原 test_label_convention_is_end --
    def _section_0_test_label_convention_is_end():
        assert BAR_LABEL_CONVENTION == "end"

    _section_0_test_label_convention_is_end()

    # -- 原 test_freq_table --
    def _section_1_test_freq_table():
        assert ASHARE_BAR_FREQS["1min"].minutes == 1
        assert ASHARE_BAR_FREQS["1min"].bars_per_day == 240
        assert ASHARE_BAR_FREQS["5min"].bars_per_day == 48
        assert ASHARE_BAR_FREQS["60min"].bars_per_day == 4

    _section_1_test_freq_table()

    # -- 原 test_valid --
    def _section_2_test_valid():
        assert normalize_freq("5min") == "5min"

    _section_2_test_valid()

    # -- 原 test_unknown_raises --
    def _section_3_test_unknown_raises():
        with pytest.raises(ValueError, match="未知频率"):
            normalize_freq("7min")

    _section_3_test_unknown_raises()

    # -- 原 test_drops_after_hours_keeps_boundaries --
    def _section_4_test_drops_after_hours_keeps_boundaries():
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

    _section_4_test_drops_after_hours_keeps_boundaries()

    # -- 原 test_key_indices --
    def _section_5_test_key_indices():
        times = [
            _dt__sessions_returns(9, 30),
            _dt__sessions_returns(9, 31),
            _dt__sessions_returns(11, 30),
            _dt__sessions_returns(13, 1),
            _dt__sessions_returns(15, 0),
            _dt__sessions_returns(15, 5),  # non-canonical → null
        ]
        df = pl.DataFrame(
            {"trade_time": pl.Series(times, dtype=pl.Datetime("us"))}
        ).with_columns(session_bar_index().alias("idx"))
        idxs = df["idx"].to_list()
        assert idxs == [0, 1, 120, 121, 240, None]

    _section_5_test_key_indices()

def test_resample_5_30_60_1_suite():
    """09:30+09:31..09:35 → 标签 09:35；open=竞价 open，vol=六根之和。；11:26..11:30 → 标签 11:30。；13:01..13:05 → 标签 13:05；不吞午休。；15:00 归入标签 15:00 桶（与 14:59 等同桶时合并）。；test_after_hours_dropped；30min：上午末桶标签 11:30。；60min：上午 10:30/11:30，下午 14:00/15:00。；freq=1min：竞价并入 09:31；完整日恰 240 根。"""
    # -- 原 test_auction_merges_into_0935 --
    def _section_0_test_auction_merges_into_0935():
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

    _section_0_test_auction_merges_into_0935()

    # -- 原 test_morning_close_bucket_1130 --
    def _section_1_test_morning_close_bucket_1130():
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

    _section_1_test_morning_close_bucket_1130()

    # -- 原 test_afternoon_open_no_lunch_swallow --
    def _section_2_test_afternoon_open_no_lunch_swallow():
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

    _section_2_test_afternoon_open_no_lunch_swallow()

    # -- 原 test_close_1500_bucket --
    def _section_3_test_close_1500_bucket():
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

    _section_3_test_close_1500_bucket()

    # -- 原 test_after_hours_dropped --
    def _section_4_test_after_hours_dropped():
        df = _mini_frame()
        out = resample_intraday(df, "5min")
        times = [
            (t.hour, t.minute) for t in out["trade_time"].to_list()  # type: ignore[union-attr]
        ]
        assert (15, 5) not in times

    _section_4_test_after_hours_dropped()

    # -- 原 test_morning_buckets --
    def _section_5_test_morning_buckets():
        rows_t = [_dt__sessions_returns(9, 30)] + [_dt__sessions_returns(11, m) for m in range(1, 31)]
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

    _section_5_test_morning_buckets()

    # -- 原 test_four_session_labels --
    def _section_6_test_four_session_labels():
        times = [
            _dt__sessions_returns(9, 30),
            _dt__sessions_returns(9, 31),
            _dt__sessions_returns(10, 30),
            _dt__sessions_returns(10, 31),
            _dt__sessions_returns(11, 30),
            _dt__sessions_returns(13, 1),
            _dt__sessions_returns(14, 0),
            _dt__sessions_returns(14, 1),
            _dt__sessions_returns(15, 0),
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

    _section_6_test_four_session_labels()

    # -- 原 test_auction_into_0931_and_240_bars --
    def _section_7_test_auction_into_0931_and_240_bars():
        times = [_dt__sessions_returns(9, 30)]
        # 09:31..11:30
        for mins in range(9 * 60 + 31, 11 * 60 + 30 + 1):
            times.append(_dt__sessions_returns(mins // 60, mins % 60))
        # 13:01..15:00
        for mins in range(13 * 60 + 1, 15 * 60 + 0 + 1):
            times.append(_dt__sessions_returns(mins // 60, mins % 60))
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

    _section_7_test_auction_into_0931_and_240_bars()

def test_edge_and_fwd_ret_suite():
    """test_empty_frame；临停：某 5min 桶只有 2 根 bar，正常聚合不崩。；test_fwd_ret_1bar_value；test_no_cross_stock_leakage；test_fwd_return_does_not_cross_trading_day_boundary"""
    # -- 原 test_empty_frame --
    def _section_0_test_empty_frame():
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

    _section_0_test_empty_frame()

    # -- 原 test_partial_bars_halt --
    def _section_1_test_partial_bars_halt():
        times = [_dt__sessions_returns(9, 31), _dt__sessions_returns(9, 32)]  # 不完整 09:35 桶
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

    _section_1_test_partial_bars_halt()

    # -- 原 test_fwd_ret_1bar_value --
    def _section_2_test_fwd_ret_1bar_value():
        df = compute_intraday_fwd_returns(_make_minute_df(), periods=[1])
        row = df.filter((pl.col("ts_code") == "000001.SZ") & (pl.col("trade_time").dt.minute() == 30))
        expected = (10.1 - 10.0) / 10.0
        assert abs(row["fwd_ret_1bar"][0] - expected) < 1e-9

    _section_2_test_fwd_ret_1bar_value()

    # -- 原 test_no_cross_stock_leakage --
    def _section_3_test_no_cross_stock_leakage():
        df = compute_intraday_fwd_returns(_make_minute_df(), periods=[1])
        for ts in ["000001.SZ", "000002.SZ", "000003.SZ"]:
            last_val = df.filter(pl.col("ts_code") == ts).sort("trade_time").tail(1)["fwd_ret_1bar"][0]
            assert last_val is None, f"{ts} 最后一行应为 null，实为 {last_val}"

    _section_3_test_no_cross_stock_leakage()

    # -- 原 test_fwd_return_does_not_cross_trading_day_boundary --
    def _section_4_test_fwd_return_does_not_cross_trading_day_boundary():
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

    _section_4_test_fwd_return_does_not_cross_trading_day_boundary()

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



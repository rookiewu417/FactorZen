"""test_loader_daily_basic_cols.py：test_loader_daily_basic_cols
test_data_quality.py：Tests for daily data quality reporting.
test_data_audit.py：tests/test_data_audit.py — common/data_audit.py 单元测试。
test_partition_repair.py：test_partition_repair
test_storage.py：common/storage.py 的单元测试。
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from factorzen.core.data_audit import (
    build_raw_data_audit,
)
from factorzen.core.storage import load_parquet, save_parquet
from factorzen.dataio.partition_repair import merge_missing_partition_rows


# ==== 来自 test_loader_daily_basic_cols.py ====
def test_daily_basic_cols_include_new_fields():
    from factorzen.core.loader import DAILY_BASIC_COLS
    for f in ["turnover_rate", "turnover_rate_f", "volume_ratio", "float_share"]:
        assert f in DAILY_BASIC_COLS, f"DAILY_BASIC_COLS missing {f}"
    # 原有字段仍在
    for f in ["trade_date", "ts_code", "pe_ttm", "pb", "total_mv", "circ_mv"]:
        assert f in DAILY_BASIC_COLS

# ==== 来自 test_data_quality.py ====
def _base_daily() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 3)],
            "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ"],
            "open": [10.0, 20.0, 10.5],
            "close": [10.2, 19.8, 10.6],
        }
    )

def _base_factor() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2)],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "factor_value": [1.0, None],
        }
    )

def _base_clean() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2)],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "factor_clean": [0.5, -0.5],
        }
    )

def _base_returns() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2)],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "fwd_ret_1d": [0.01, None],
        }
    )

def test_quality_report_records_coverage_and_warnings():
    from factorzen.core.data_quality import build_daily_quality_report

    report = build_daily_quality_report(
        daily_df=_base_daily(),
        factor_df=_base_factor(),
        clean_df=_base_clean(),
        ret_df=_base_returns(),
        universe_codes=["000001.SZ", "000002.SZ", "000003.SZ"],
    )

    assert report["status"] == "warning"
    assert report["checks"]["factor_value"]["coverage"] == 0.5
    assert report["checks"]["universe"]["coverage"] == pytest.approx(2 / 3)
    assert report["warnings"]

def test_quality_report_rejects_duplicate_daily_keys():
    from factorzen.core.data_quality import QualityCheckError, build_daily_quality_report

    duplicate_daily = pl.concat([_base_daily(), _base_daily().head(1)])

    with pytest.raises(QualityCheckError, match="duplicate daily keys"):
        build_daily_quality_report(
            daily_df=duplicate_daily,
            factor_df=_base_factor(),
            clean_df=_base_clean(),
            ret_df=_base_returns(),
            universe_codes=["000001.SZ", "000002.SZ"],
        )

def test_quality_report_rejects_empty_clean_factor():
    from factorzen.core.data_quality import QualityCheckError, build_daily_quality_report

    empty_clean = _base_clean().with_columns(pl.lit(None).cast(pl.Float64).alias("factor_clean"))

    with pytest.raises(QualityCheckError, match="factor_clean has no valid values"):
        build_daily_quality_report(
            daily_df=_base_daily(),
            factor_df=_base_factor(),
            clean_df=empty_clean,
            ret_df=_base_returns(),
            universe_codes=["000001.SZ", "000002.SZ"],
        )

# ==== 来自 test_data_audit.py ====
# ── 辅助函数 ────────────────────────────────────────────────────────────────

def _daily_df(dates: list[date], codes: list[str]) -> pl.DataFrame:
    rows = [(d, c) for d in dates for c in codes]
    return pl.DataFrame(
        {
            "trade_date": pl.Series([r[0] for r in rows], dtype=pl.Date),
            "ts_code": [r[1] for r in rows],
        }
    )

def _daily_basic_df(dates: list[date], codes: list[str], null_pb: bool = False) -> pl.DataFrame:
    rows = [(d, c) for d in dates for c in codes]
    n = len(rows)
    return pl.DataFrame(
        {
            "trade_date": pl.Series([r[0] for r in rows], dtype=pl.Date),
            "ts_code": [r[1] for r in rows],
            "pe": [20.0] * n,
            "pb": [None] * n if null_pb else [2.0] * n,
            "total_mv": [1e9] * n,
            "circ_mv": [5e8] * n,
        }
    )

def _finance_df(ann_date_str: str, n: int = 5) -> pl.DataFrame:
    codes = [f"{i:06d}.SZ" for i in range(n)]
    return pl.DataFrame(
        {
            "end_date": pl.Series([date(2023, 9, 30)] * n, dtype=pl.Date),
            "ts_code": codes,
            "ann_date": [ann_date_str] * n,
            "revenue": [1e9] * n,
            "n_income": [1e8] * n,
            "total_assets": [5e9] * n,
            "total_equity": [2e9] * n,
            "roe": [0.15] * n,
        }
    )

def _mock_load(df: pl.DataFrame):
    lf = MagicMock()
    lf.collect.return_value = df
    return lf

# ── 1. 参数校验 ─────────────────────────────────────────────────────────────

class TestInvalidDataType:
    def test_unsupported_type_returns_error(self):
        result = build_raw_data_audit(data_type="tick", start="20230101", end="20231231")
        assert result["status"] == "error"
        assert any("unsupported" in e for e in result["errors"])

# ── 2. 数据加载失败 ─────────────────────────────────────────────────────────

class TestLoadFailure:
    def test_scan_exception_returns_error(self):
        with patch("factorzen.core.data_audit.load_parquet") as mock_lp:
            mock_lp.return_value.collect.side_effect = Exception("no parquet files found")
            result = build_raw_data_audit(data_type="daily", start="20230101", end="20231231")
        assert result["status"] == "error"
        assert any("failed to load" in e for e in result["errors"])

    def test_empty_dataframe_returns_error(self):
        with (
            patch("factorzen.core.data_audit.load_parquet") as mock_lp,
            patch("factorzen.core.data_audit.get_trade_dates", return_value=[]),
        ):
            mock_lp.return_value.collect.return_value = pl.DataFrame()
            result = build_raw_data_audit(data_type="daily", start="20230101", end="20231231")
        assert result["status"] == "error"
        assert any("empty" in e for e in result["errors"])

# ── 3. daily — 日期缺口 ─────────────────────────────────────────────────────

class TestDailyDateCoverage:
    _dates = [date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5)]
    _codes = ["000001.SZ", "000002.SZ"]

    def test_no_gaps_ok(self):
        df = _daily_df(self._dates, self._codes)
        with (
            patch("factorzen.core.data_audit.load_parquet", return_value=_mock_load(df)),
            patch("factorzen.core.data_audit.get_trade_dates", return_value=self._dates),
        ):
            result = build_raw_data_audit(data_type="daily", start="20230103", end="20230105")
        assert result["status"] == "ok"
        assert result["checks"]["date_coverage"]["missing_count"] == 0

    def test_missing_date_warning(self):
        # Drop middle date from actual data
        df = _daily_df([date(2023, 1, 3), date(2023, 1, 5)], self._codes)
        with (
            patch("factorzen.core.data_audit.load_parquet", return_value=_mock_load(df)),
            patch("factorzen.core.data_audit.get_trade_dates", return_value=self._dates),
        ):
            result = build_raw_data_audit(data_type="daily", start="20230103", end="20230105")
        assert result["status"] == "warning"
        dc = result["checks"]["date_coverage"]
        assert dc["missing_count"] == 1
        assert "20230104" in dc["missing_dates"]
        assert any("missing" in w for w in result["warnings"])

    def test_missing_dates_capped_at_20_in_output(self):
        # 25 expected dates, 0 actual → 25 missing but only 20 shown
        expected = [date(2023, 1, d) for d in range(3, 28)]  # 25 dates
        df = _daily_df([date(2023, 1, 3)], self._codes)  # only 1 date present
        with (
            patch("factorzen.core.data_audit.load_parquet", return_value=_mock_load(df)),
            patch("factorzen.core.data_audit.get_trade_dates", return_value=expected),
        ):
            result = build_raw_data_audit(data_type="daily", start="20230103", end="20230127")
        assert result["checks"]["date_coverage"]["missing_count"] == 24
        assert len(result["checks"]["date_coverage"]["missing_dates"]) == 20

# ── 4. daily — 股票覆盖率 ───────────────────────────────────────────────────

class TestDailyStockCoverage:
    _dates = [date(2023, 1, 3)]
    _universe = [f"{i:06d}.SZ" for i in range(10)]

    def test_full_coverage_ok(self):
        df = _daily_df(self._dates, self._universe)
        with (
            patch("factorzen.core.data_audit.load_parquet", return_value=_mock_load(df)),
            patch("factorzen.core.data_audit.get_trade_dates", return_value=self._dates),
        ):
            result = build_raw_data_audit(
                data_type="daily",
                start="20230103",
                end="20230103",
                universe_codes=self._universe,
            )
        assert result["status"] == "ok"
        sc = result["checks"]["stock_coverage"]
        assert sc["coverage"] == pytest.approx(1.0)
        assert sc["covered"] == 10

    def test_low_coverage_warning(self):
        # Only 5 of 10 universe stocks present
        df = _daily_df(self._dates, self._universe[:5])
        with (
            patch("factorzen.core.data_audit.load_parquet", return_value=_mock_load(df)),
            patch("factorzen.core.data_audit.get_trade_dates", return_value=self._dates),
        ):
            result = build_raw_data_audit(
                data_type="daily",
                start="20230103",
                end="20230103",
                universe_codes=self._universe,
            )
        assert result["status"] == "warning"
        sc = result["checks"]["stock_coverage"]
        assert sc["coverage"] == pytest.approx(0.5)
        assert any("coverage" in w for w in result["warnings"])

    def test_no_universe_skips_coverage(self):
        df = _daily_df(self._dates, self._universe[:3])
        with (
            patch("factorzen.core.data_audit.load_parquet", return_value=_mock_load(df)),
            patch("factorzen.core.data_audit.get_trade_dates", return_value=self._dates),
        ):
            result = build_raw_data_audit(data_type="daily", start="20230103", end="20230103")
        sc = result["checks"]["stock_coverage"]
        assert "coverage" not in sc
        assert "actual_codes" in sc

# ── 5. daily_basic — 字段空值率 ─────────────────────────────────────────────

class TestDailyBasicFieldNulls:
    _dates = [date(2023, 1, 3)]
    _codes = [f"{i:06d}.SZ" for i in range(10)]

    def test_all_fields_present_ok(self):
        df = _daily_basic_df(self._dates, self._codes, null_pb=False)
        with (
            patch("factorzen.core.data_audit.load_parquet", return_value=_mock_load(df)),
            patch("factorzen.core.data_audit.get_trade_dates", return_value=self._dates),
        ):
            result = build_raw_data_audit(data_type="daily_basic", start="20230103", end="20230103")
        assert result["status"] == "ok"
        fr = result["checks"]["field_null_rates"]
        assert fr["pb"]["coverage"] == pytest.approx(1.0)
        assert fr["pe"]["coverage"] == pytest.approx(1.0)

    def test_all_null_pb_warns(self):
        df = _daily_basic_df(self._dates, self._codes, null_pb=True)
        with (
            patch("factorzen.core.data_audit.load_parquet", return_value=_mock_load(df)),
            patch("factorzen.core.data_audit.get_trade_dates", return_value=self._dates),
        ):
            result = build_raw_data_audit(data_type="daily_basic", start="20230103", end="20230103")
        assert result["status"] == "warning"
        fr = result["checks"]["field_null_rates"]
        assert fr["pb"]["coverage"] == pytest.approx(0.0)
        assert any("pb" in w for w in result["warnings"])

    def test_missing_column_warns(self):
        # Build df without 'circ_mv'
        df = _daily_basic_df(self._dates, self._codes).drop("circ_mv")
        with (
            patch("factorzen.core.data_audit.load_parquet", return_value=_mock_load(df)),
            patch("factorzen.core.data_audit.get_trade_dates", return_value=self._dates),
        ):
            result = build_raw_data_audit(data_type="daily_basic", start="20230103", end="20230103")
        assert result["status"] == "warning"
        assert result["checks"]["field_null_rates"]["circ_mv"].get("missing_column")
        assert any("circ_mv" in w for w in result["warnings"])

# ── 6. finance — PIT 陈旧性 ─────────────────────────────────────────────────

class TestFinancePITStaleness:
    def test_fresh_ann_date_ok(self):
        # ann_date is very recent (today-ish)
        from datetime import datetime

        fresh = datetime.now().strftime("%Y%m%d")
        df = _finance_df(ann_date_str=fresh)
        with patch("factorzen.core.data_audit.load_parquet", return_value=_mock_load(df)):
            result = build_raw_data_audit(data_type="finance", start="20230101", end="20231231")
        assert result["status"] == "ok"
        assert result["checks"]["pit_staleness"]["stale_count"] == 0

    def test_stale_ann_date_warns(self):
        # ann_date is from 2018 — well beyond 548-day threshold
        df = _finance_df(ann_date_str="20180101")
        with patch("factorzen.core.data_audit.load_parquet", return_value=_mock_load(df)):
            result = build_raw_data_audit(data_type="finance", start="20230101", end="20231231")
        assert result["status"] == "warning"
        ps = result["checks"]["pit_staleness"]
        assert ps["stale_count"] == 5
        assert any("stale" in w.lower() for w in result["warnings"])

    def test_finance_unique_end_dates_reported(self):
        df = _finance_df(ann_date_str="20231201", n=3)
        with patch("factorzen.core.data_audit.load_parquet", return_value=_mock_load(df)):
            result = build_raw_data_audit(data_type="finance", start="20230101", end="20231231")
        # finance date_coverage reports unique end_dates, not trade date gaps
        assert "unique_end_dates" in result["checks"]["date_coverage"]

    def test_finance_field_null_rate_warning(self):
        df = _finance_df(ann_date_str="20231201", n=5)
        # Drop 'roe' to trigger missing_column warning
        df = df.drop("roe")
        with patch("factorzen.core.data_audit.load_parquet", return_value=_mock_load(df)):
            result = build_raw_data_audit(data_type="finance", start="20230101", end="20231231")
        assert result["checks"]["field_null_rates"]["roe"].get("missing_column")

# ── 7. 输出格式 ─────────────────────────────────────────────────────────────

class TestOutputSchema:
    def test_output_keys_always_present(self):
        df = _daily_df([date(2023, 1, 3)], ["000001.SZ"])
        with (
            patch("factorzen.core.data_audit.load_parquet", return_value=_mock_load(df)),
            patch("factorzen.core.data_audit.get_trade_dates", return_value=[date(2023, 1, 3)]),
        ):
            result = build_raw_data_audit(data_type="daily", start="20230103", end="20230103")
        assert set(result.keys()) == {"status", "checks", "warnings", "errors"}
        assert result["status"] in ("ok", "warning", "error")
        assert isinstance(result["warnings"], list)
        assert isinstance(result["errors"], list)

# ==== 来自 test_partition_repair.py ====
def test_merge_missing_rows_aligns_legacy_schema_without_overwriting_target(tmp_path):
    source = tmp_path / "backup"
    source.mkdir()
    pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3)],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "pb": [1.0, 2.0],
        }
    ).write_parquet(source / "legacy.parquet")

    raw = tmp_path / "raw"
    current = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 3)],
            "ts_code": ["000001.SZ"],
            "pb": [9.0],
            "turnover_rate": [3.0],
        }
    )
    save_parquet(current, "daily_basic", base_dir=raw)

    report = merge_missing_partition_rows(
        source,
        target_data_type="daily_basic",
        base_dir=raw,
        key_cols=("trade_date", "ts_code"),
    )
    merged = load_parquet("daily_basic", base_dir=raw).collect().sort("trade_date")

    assert report.merged_rows == 1
    assert merged.height == 2
    assert merged["pb"].to_list() == [1.0, 9.0]
    assert merged["turnover_rate"].to_list() == [None, 3.0]

    again = merge_missing_partition_rows(
        source,
        target_data_type="daily_basic",
        base_dir=raw,
        key_cols=("trade_date", "ts_code"),
    )
    assert again.merged_rows == 0

# ==== 来自 test_storage.py ====
@pytest.fixture()
def tmp_dir(tmp_path):
    return tmp_path

def _make_df(n: int = 10) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [date(2024, 1, d + 1) for d in range(n)],
            "ts_code": [f"{i:06d}.SH" for i in range(n)],
            "value": list(range(n)),
        }
    )

def test_save_and_load_roundtrip(tmp_dir):
    df = _make_df(5)
    save_parquet(df, "test_data", base_dir=tmp_dir)
    loaded = load_parquet("test_data", base_dir=tmp_dir).collect()
    assert loaded.shape[0] == 5
    assert set(loaded.columns).issuperset({"trade_date", "ts_code", "value"})

def test_save_append_deduplicates(tmp_dir):
    df1 = _make_df(5)
    save_parquet(df1, "test_data", base_dir=tmp_dir, mode="append")
    # 重复写入同样的数据
    save_parquet(df1, "test_data", base_dir=tmp_dir, mode="append")
    loaded = load_parquet("test_data", base_dir=tmp_dir).collect()
    assert loaded.shape[0] == 5  # 去重后仍为 5 行

def test_save_append_replaces_existing_business_key(tmp_dir):
    original = _make_df(1)
    updated = original.with_columns(pl.lit(99).alias("value"))

    save_parquet(original, "test_data", base_dir=tmp_dir, mode="append")
    save_parquet(updated, "test_data", base_dir=tmp_dir, mode="append")

    loaded = load_parquet("test_data", base_dir=tmp_dir).collect()
    assert loaded.height == 1
    assert loaded["value"][0] == 99

def test_save_overwrite_replaces(tmp_dir):
    df1 = _make_df(5)
    save_parquet(df1, "test_data", base_dir=tmp_dir, mode="overwrite")
    df2 = _make_df(3)
    save_parquet(df2, "test_data", base_dir=tmp_dir, mode="overwrite")
    loaded = load_parquet("test_data", base_dir=tmp_dir).collect()
    # overwrite 只覆盖同月分区；1月数据被覆盖为3行
    assert loaded.shape[0] == 3

def test_hive_partitions_created(tmp_dir):
    df = _make_df(5)
    save_parquet(df, "test_data", base_dir=tmp_dir)
    # 应该创建 year=2024/month=01/data.parquet
    assert (tmp_dir / "test_data" / "year=2024" / "month=01" / "data.parquet").exists()

def test_load_with_date_filter(tmp_dir):
    df = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 5), date(2024, 2, 10), date(2024, 3, 15)],
            "ts_code": ["A", "B", "C"],
            "value": [1, 2, 3],
        }
    )
    save_parquet(df, "test_data", base_dir=tmp_dir)
    loaded = load_parquet("test_data", start="20240201", end="20240228", base_dir=tmp_dir).collect()
    assert loaded.shape[0] == 1
    assert loaded["ts_code"][0] == "B"

def test_load_datetime_end_boundary_includes_full_end_day(tmp_dir):
    """Datetime 列（分钟 bar）end 边界须含截止日全天，而非只到当日 00:00。"""
    from datetime import datetime

    df = pl.DataFrame(
        {
            "trade_time": [
                datetime(2024, 1, 30, 9, 31),
                datetime(2024, 1, 31, 9, 31),   # 截止日盘中
                datetime(2024, 1, 31, 15, 0),   # 截止日收盘
                datetime(2024, 2, 1, 9, 31),    # 越界
            ],
            "ts_code": ["A", "A", "A", "A"],
            "value": [1, 2, 3, 4],
        }
    )
    save_parquet(df, "minute_test", date_col="trade_time", base_dir=tmp_dir)
    loaded = load_parquet(
        "minute_test", start="20240131", end="20240131", date_col="trade_time", base_dir=tmp_dir
    ).collect()
    vals = sorted(loaded["value"].to_list())
    assert vals == [2, 3], (
        f"应含 1/31 全天两根 bar，实得 {vals}（修复前 end=1/31 00:00 把盘中 bar 全排除）"
    )

def test_load_date_end_boundary_still_inclusive(tmp_dir):
    """Date 列的 end 仍为闭区间（含截止日）。"""
    df = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 30), date(2024, 1, 31), date(2024, 2, 1)],
            "ts_code": ["A", "B", "C"],
            "value": [1, 2, 3],
        }
    )
    save_parquet(df, "test_data2", base_dir=tmp_dir)
    loaded = load_parquet("test_data2", start="20240130", end="20240131", base_dir=tmp_dir).collect()
    assert sorted(loaded["value"].to_list()) == [1, 2]

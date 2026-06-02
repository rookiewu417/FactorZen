"""tests/test_data_audit.py — common/data_audit.py 单元测试。

全量 mock：不读取文件系统，不调用 Tushare，不依赖本地 data/raw/ 数据。
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from factorzen.core.data_audit import (
    build_raw_data_audit,
)

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

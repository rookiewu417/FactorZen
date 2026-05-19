"""Tests for daily data quality reporting."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest


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
    from common.data_quality import build_daily_quality_report

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
    from common.data_quality import QualityCheckError, build_daily_quality_report

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
    from common.data_quality import QualityCheckError, build_daily_quality_report

    empty_clean = _base_clean().with_columns(pl.lit(None).cast(pl.Float64).alias("factor_clean"))

    with pytest.raises(QualityCheckError, match="factor_clean has no valid values"):
        build_daily_quality_report(
            daily_df=_base_daily(),
            factor_df=_base_factor(),
            clean_df=empty_clean,
            ret_df=_base_returns(),
            universe_codes=["000001.SZ", "000002.SZ"],
        )

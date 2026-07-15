from __future__ import annotations

from datetime import date

import polars as pl

from factorzen.core.storage import load_parquet, save_parquet
from factorzen.dataio.partition_repair import merge_missing_partition_rows


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

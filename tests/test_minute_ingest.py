from __future__ import annotations

import json
from datetime import datetime

import polars as pl

from factorzen.config.settings import DATA_RAW_MINUTE
from factorzen.core.storage import load_parquet, save_parquet
from factorzen.dataio.minute_ingest import ingest_minute_files


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


def test_minute_setting_matches_loader_storage_namespace():
    assert DATA_RAW_MINUTE.name == "minute_1min"


def test_ingest_normalizes_schema_preserves_bars_and_is_idempotent(tmp_path):
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


def test_ingest_merges_into_existing_partial_month_instead_of_skipping(tmp_path):
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


def test_ingest_month_filter_limits_written_partitions(tmp_path):
    source = tmp_path / "source.parquet"
    raw = tmp_path / "raw"
    _source_frame().write_parquet(source)

    report = ingest_minute_files([source], base_dir=raw, months=["202402"])

    assert report.rows_by_month == {"202402": 1}
    assert not (raw / "minute_1min" / "year=2024" / "month=01").exists()
    assert (raw / "minute_1min" / "year=2024" / "month=02" / "data.parquet").is_file()


def test_cli_writes_reproducibility_manifest_and_sentinel(tmp_path, monkeypatch):
    from tools import ingest_minute as cli

    source = tmp_path / "source.parquet"
    raw = tmp_path / "raw"
    workspace = tmp_path / "workspace"
    _source_frame().write_parquet(source)
    monkeypatch.setattr(cli, "WORKSPACE_DIR", workspace)

    assert cli.main([str(source), "--data-root", str(raw), "--run-id", "test-run"]) == 0

    run_dir = workspace / "data_ingest" / "test-run"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["git_sha"]
    assert manifest["window"] == {"months": ["202401", "202402"]}
    assert manifest["result"]["source_files"] == 1
    assert (run_dir / "ingest.done").is_file()

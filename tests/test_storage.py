"""common/storage.py 的单元测试。"""

from datetime import date

import polars as pl
import pytest

from common.storage import load_parquet, save_parquet


@pytest.fixture()
def tmp_dir(tmp_path):
    return tmp_path


def _make_df(n: int = 10) -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": [date(2024, 1, d + 1) for d in range(n)],
        "ts_code": [f"{i:06d}.SH" for i in range(n)],
        "value": list(range(n)),
    })


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
    df = pl.DataFrame({
        "trade_date": [date(2024, 1, 5), date(2024, 2, 10), date(2024, 3, 15)],
        "ts_code": ["A", "B", "C"],
        "value": [1, 2, 3],
    })
    save_parquet(df, "test_data", base_dir=tmp_dir)
    loaded = load_parquet(
        "test_data", start="20240201", end="20240228", base_dir=tmp_dir
    ).collect()
    assert loaded.shape[0] == 1
    assert loaded["ts_code"][0] == "B"

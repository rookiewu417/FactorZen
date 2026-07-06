"""Polars Parquet 读写封装，支持 Hive 分区。

分区路径格式: ``{base_dir}/{data_type}/year={YYYY}/month={MM}/data.parquet``
"""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from factorzen.config.settings import DATA_RAW


def save_parquet(
    df: pl.DataFrame,
    data_type: str,
    date_col: str = "trade_date",
    base_dir: Path | None = None,
    mode: str = "append",
) -> None:
    """按 data_type/年/月 分区写入 Parquet。

    Args:
        df: 待写入数据，必须包含 date_col 列。
        data_type: 数据类型子目录名，例如 ``"daily"``、``"finance"``。
        date_col: 日期列名，用于提取分区年月。
        base_dir: 基础目录，默认 ``DATA_RAW``。
        mode: ``"append"`` — 合并去重；``"overwrite"`` — 覆盖。

    Raises:
        pl.ColumnNotFoundError: 如果 ``df`` 中不存在 ``date_col`` 列。
    """
    base = DATA_RAW if base_dir is None else base_dir

    df_with_part = df.with_columns(
        pl.col(date_col).dt.year().cast(pl.Utf8).alias("_year"),
        pl.col(date_col).dt.month().cast(pl.Utf8).alias("_month"),
    )

    for (year, month), group in df_with_part.group_by(["_year", "_month"]):
        # 月份零填充：month=05 而非 month=5
        month_str = month.zfill(2)
        part_dir = base / data_type / f"year={year}" / f"month={month_str}"
        part_dir.mkdir(parents=True, exist_ok=True)
        file_path = part_dir / "data.parquet"

        group = group.drop(["_year", "_month"])

        if mode == "append" and file_path.exists():
            existing = pl.read_parquet(file_path)
            combined = pl.concat([existing, group], how="vertical_relaxed")
            key_cols = [date_col, "ts_code"] if "ts_code" in combined.columns else None
            combined = combined.unique(
                subset=key_cols,
                keep="last",
                maintain_order=True,
            )
            combined.write_parquet(file_path)
        else:
            group.write_parquet(file_path)


def load_parquet(
    data_type: str,
    start: str | None = None,
    end: str | None = None,
    date_col: str = "trade_date",
    base_dir: Path | None = None,
) -> pl.LazyFrame:
    """惰性读取指定类型的分区数据，支持日期区间谓词下推。

    Args:
        data_type: 数据类型子目录名。
        start: 起始日期 ``"%Y%m%d"``，含。传入 ``None`` 不过滤左边界。
        end: 截止日期 ``"%Y%m%d"``，含。传入 ``None`` 不过滤右边界。
        date_col: 日期列名。
        base_dir: 基础目录，默认 ``DATA_RAW``。

    Returns:
        LazyFrame，调用 ``.collect()`` 后才实际加载数据。

    Raises:
        ValueError: 如果 ``start`` 或 ``end`` 字符串格式不是 ``"%Y%m%d"``。
    """
    base = DATA_RAW if base_dir is None else base_dir
    lf = pl.scan_parquet(str(base / data_type / "**/*.parquet"))

    if start is not None:
        start_dt = datetime.strptime(start, "%Y%m%d")
        lf = lf.filter(pl.col(date_col) >= start_dt)
    if end is not None:
        # 半开区间 [start, end+1day)：对 Date 列等价于闭区间含 end；对 Datetime 列
        # （分钟 bar）则含 end 当天全部盘中 bar——用 `<= end` 会把 end 解析成当日 00:00，
        # 静默排除截止日所有盘中数据。
        end_next = datetime.strptime(end, "%Y%m%d") + timedelta(days=1)
        lf = lf.filter(pl.col(date_col) < end_next)

    return lf


def scan_parquet(
    data_type: str,
    base_dir: Path | None = None,
) -> pl.LazyFrame:
    """全量惰性扫描指定类型的所有分区，不设日期过滤。

    Args:
        data_type: 数据类型子目录名。
        base_dir: 基础目录，默认 ``DATA_RAW``。

    Returns:
        LazyFrame，调用 ``.collect()`` 后才实际加载数据。
    """
    base = DATA_RAW if base_dir is None else base_dir
    return pl.scan_parquet(str(base / data_type / "**/*.parquet"))


def partition_exists(
    data_type: str,
    year: int,
    month: int,
    base_dir: Path | None = None,
) -> bool:
    """检查指定分区是否存在且非空。

    Args:
        data_type: 数据类型子目录名。
        year: 年份。
        month: 月份。
        base_dir: 基础目录，默认 ``DATA_RAW``。

    Returns:
        分区 Parquet 文件存在且大小大于 0 返回 ``True``。
    """
    base = DATA_RAW if base_dir is None else base_dir
    file_path = base / data_type / f"year={year}" / f"month={month:02d}" / "data.parquet"
    return file_path.exists() and file_path.stat().st_size > 0

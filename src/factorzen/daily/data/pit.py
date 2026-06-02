"""Point-In-Time 财务数据对齐。确保月末调仓不使用未公告的财报。"""

from datetime import date

import polars as pl


def pit_align(
    fina_df: pl.DataFrame,
    snapshot_dates: list[date],
) -> pl.DataFrame:
    """对财务数据做 Point-In-Time 对齐。

    对每个月频快照日期，找出每只股票「最新已公告」的财务报告——
    即 ann_date <= snapshot_date 中 end_date 最大的那条。

    Args:
        fina_df: 财务数据，必须含 ts_code, end_date(Date), ann_date(Date) 及指标列
        snapshot_dates: 月频快照日列表(升序)

    Returns:
        pl.DataFrame, 列: snapshot_date, ts_code, end_date, ann_date, 指标列
    """
    if fina_df.is_empty() or not snapshot_dates:
        return pl.DataFrame()

    # ann_date 在存储中为 String "YYYYMMDD"，比较前统一转成 Date
    if fina_df["ann_date"].dtype == pl.Utf8:
        fina_df = fina_df.with_columns(
            pl.col("ann_date").str.strptime(pl.Date, "%Y%m%d", strict=False)
        )

    # 过滤掉 ann_date 为 null 的记录（未公告的财报）
    fina_df = fina_df.filter(pl.col("ann_date").is_not_null())

    # 按 ts_code + end_date 降序预排序，后续 group_by().first() 即可取到 max end_date
    fina_sorted = fina_df.sort(["ts_code", "end_date"], descending=[False, True])

    results: list[pl.DataFrame] = []
    for sd in snapshot_dates:
        # 只保留快照日之前（含当天）已公告的财报
        valid = fina_sorted.filter(pl.col("ann_date") <= sd)
        if valid.is_empty():
            continue

        # 每个 ts_code 取 end_date 最大的那条（已排好序，first() 即是）
        best = (
            valid.group_by("ts_code")
            .first()
            .with_columns(pl.lit(sd).cast(pl.Date).alias("snapshot_date"))
        )
        results.append(best)

    if not results:
        return pl.DataFrame()

    return pl.concat(results, how="vertical")

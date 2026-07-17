"""pit_align 向量化等价性：golden 以旧实现输出为 ground truth。"""

from __future__ import annotations

from datetime import date

import polars as pl

from factorzen.daily.data.pit import pit_align


def _pit_align_reference(
    fina_df: pl.DataFrame,
    snapshot_dates: list[date],
) -> pl.DataFrame:
    """旧实现逐字拷贝（Wave1 前 master），作 golden 基准。"""
    if fina_df.is_empty() or not snapshot_dates:
        return pl.DataFrame()

    if fina_df["ann_date"].dtype == pl.Utf8:
        fina_df = fina_df.with_columns(
            pl.col("ann_date").str.strptime(pl.Date, "%Y%m%d", strict=False)
        )

    fina_df = fina_df.filter(pl.col("ann_date").is_not_null())

    fina_sorted = fina_df.sort(["ts_code", "end_date"], descending=[False, True])

    results: list[pl.DataFrame] = []
    for sd in snapshot_dates:
        valid = fina_sorted.filter(pl.col("ann_date") <= sd)
        if valid.is_empty():
            continue

        best = (
            valid.group_by("ts_code")
            .first()
            .with_columns(pl.lit(sd).cast(pl.Date).alias("snapshot_date"))
        )
        results.append(best)

    if not results:
        return pl.DataFrame()

    return pl.concat(results, how="vertical")


def _assert_equiv(got: pl.DataFrame, expected: pl.DataFrame) -> None:
    """列集合、dtype、行集合一致；行序无契约，sort 后再比。"""
    if expected.is_empty():
        assert got.is_empty(), f"expected empty, got height={got.height}"
        return
    assert not got.is_empty()
    assert set(got.columns) == set(expected.columns), (
        f"cols got={got.columns} expected={expected.columns}"
    )
    for c in expected.columns:
        assert got[c].dtype == expected[c].dtype, (
            f"dtype {c}: got={got[c].dtype} expected={expected[c].dtype}"
        )
    sort_keys = [c for c in ("snapshot_date", "ts_code") if c in expected.columns]
    g = got.select(expected.columns).sort(sort_keys)
    e = expected.select(expected.columns).sort(sort_keys)
    assert g.equals(e), (
        f"mismatch\ngot:\n{g}\nexpected:\n{e}"
    )


def test_pit_align_multi_stock_multi_quarter():
    """常规多股票多季度。"""
    fina = pl.DataFrame(
        {
            "ts_code": [
                "A", "A", "A",
                "B", "B",
                "C",
            ],
            "end_date": [
                date(2023, 3, 31), date(2023, 6, 30), date(2023, 9, 30),
                date(2023, 3, 31), date(2023, 6, 30),
                date(2023, 6, 30),
            ],
            "ann_date": [
                date(2023, 4, 20), date(2023, 8, 15), date(2023, 10, 25),
                date(2023, 4, 22), date(2023, 8, 20),
                date(2023, 8, 10),
            ],
            "roe": [10.0, 12.0, 14.0, 20.0, 22.0, 30.0],
        }
    )
    snaps = [
        date(2023, 5, 1),
        date(2023, 9, 1),
        date(2023, 11, 1),
    ]
    expected = _pit_align_reference(fina, snaps)
    got = pit_align(fina, snaps)
    _assert_equiv(got, expected)
    # 语义抽检：9/1 时 A 应取 Q2 而非尚未公告的 Q3
    row = got.filter(
        (pl.col("snapshot_date") == date(2023, 9, 1)) & (pl.col("ts_code") == "A")
    )
    assert row[0, "end_date"] == date(2023, 6, 30)
    assert row[0, "roe"] == 12.0


def test_pit_align_correction_later_ann_older_end():
    """更正公告反例：后公告但 end_date 更旧，naive asof-ann 会答错。

    某股票先公告了 Q2，后又公告了一份更早的 Q1 更正；在 Q2 已可见后，
    必须仍取 Q2（max end_date），不能被更晚的 ann_date 覆盖。
    """
    fina = pl.DataFrame(
        {
            "ts_code": ["X", "X"],
            "end_date": [date(2023, 6, 30), date(2023, 3, 31)],
            "ann_date": [date(2023, 8, 15), date(2023, 9, 1)],  # 更正更晚
            "roe": [15.0, 99.0],  # 99 是陷阱：按 ann 最新会错取
        }
    )
    snaps = [date(2023, 8, 20), date(2023, 9, 15)]
    expected = _pit_align_reference(fina, snaps)
    got = pit_align(fina, snaps)
    _assert_equiv(got, expected)

    # 两日都应取 Q2 (end=6/30, roe=15)，绝不能取更正的 Q1
    for sd in snaps:
        row = got.filter(pl.col("snapshot_date") == sd)
        assert row.height == 1
        assert row[0, "end_date"] == date(2023, 6, 30)
        assert row[0, "roe"] == 15.0


def test_pit_align_ann_date_null_and_string_dtype():
    """ann_date 为 null / String YYYYMMDD 两种 dtype。"""
    # String dtype
    fina_str = pl.DataFrame(
        {
            "ts_code": ["A", "A", "B"],
            "end_date": [date(2023, 3, 31), date(2023, 6, 30), date(2023, 6, 30)],
            "ann_date": ["20230420", "20230815", "20230810"],
            "roe": [1.0, 2.0, 3.0],
        }
    )
    snaps = [date(2023, 5, 1), date(2023, 9, 1)]
    expected = _pit_align_reference(fina_str, snaps)
    got = pit_align(fina_str, snaps)
    _assert_equiv(got, expected)

    # null ann_date 被过滤
    fina_null = pl.DataFrame(
        {
            "ts_code": ["A", "A", "A"],
            "end_date": [date(2023, 3, 31), date(2023, 6, 30), date(2023, 9, 30)],
            "ann_date": [date(2023, 4, 20), None, date(2023, 10, 25)],
            "roe": [1.0, 2.0, 3.0],
        }
    )
    snaps2 = [date(2023, 9, 1), date(2023, 11, 1)]
    expected2 = _pit_align_reference(fina_null, snaps2)
    got2 = pit_align(fina_null, snaps2)
    _assert_equiv(got2, expected2)
    # 9/1 时只有 Q1 可见（Q2 ann 为 null 被丢）
    row = got2.filter(pl.col("snapshot_date") == date(2023, 9, 1))
    assert row[0, "end_date"] == date(2023, 3, 31)


def test_pit_align_snapshot_before_any_announcement():
    """快照日早于一切公告 → 该日无输出行。"""
    fina = pl.DataFrame(
        {
            "ts_code": ["A"],
            "end_date": [date(2023, 6, 30)],
            "ann_date": [date(2023, 8, 15)],
            "roe": [10.0],
        }
    )
    snaps = [date(2023, 1, 1), date(2023, 8, 20)]
    expected = _pit_align_reference(fina, snaps)
    got = pit_align(fina, snaps)
    _assert_equiv(got, expected)
    assert got.filter(pl.col("snapshot_date") == date(2023, 1, 1)).is_empty()
    assert got.filter(pl.col("snapshot_date") == date(2023, 8, 20)).height == 1


def test_pit_align_same_end_date_tie_break():
    """同 end_date 双记录：旧实现 sort 后 group_by().first() 的 tie-break。"""
    # 原相对顺序：先 v1 再 v2；同 end_date 应取原相对顺序第一条 (v1)
    fina = pl.DataFrame(
        {
            "ts_code": ["T", "T"],
            "end_date": [date(2023, 12, 31), date(2023, 12, 31)],
            "ann_date": [date(2024, 3, 1), date(2024, 4, 1)],
            "roe": [10.0, 20.0],
            "version": ["v1", "v2"],
        }
    )
    snaps = [date(2024, 3, 15), date(2024, 4, 15)]
    expected = _pit_align_reference(fina, snaps)
    got = pit_align(fina, snaps)
    _assert_equiv(got, expected)

    # 两日均应取 v1（原相对顺序第一条），即便 v2 更晚公告
    for sd in snaps:
        row = got.filter(pl.col("snapshot_date") == sd)
        assert row.height == 1
        assert row[0, "roe"] == 10.0
        assert row[0, "version"] == "v1"


def test_pit_align_tie_break_later_ann_earlier_in_file():
    """同 end_date：原文件中更晚公告的行反而排在前面 → 两日可见后应取该行。"""
    fina = pl.DataFrame(
        {
            "ts_code": ["T", "T"],
            "end_date": [date(2023, 12, 31), date(2023, 12, 31)],
            # 行序：先 late 再 early（与 ann 时间相反）
            "ann_date": [date(2024, 4, 1), date(2024, 3, 1)],
            "roe": [20.0, 10.0],
            "version": ["late_first", "early_second"],
        }
    )
    snaps = [date(2024, 3, 15), date(2024, 4, 15)]
    expected = _pit_align_reference(fina, snaps)
    got = pit_align(fina, snaps)
    _assert_equiv(got, expected)

    # 3/15 只有 early_second 可见
    r1 = got.filter(pl.col("snapshot_date") == date(2024, 3, 15))
    assert r1[0, "version"] == "early_second"
    # 4/15 两者可见：原相对顺序第一条是 late_first
    r2 = got.filter(pl.col("snapshot_date") == date(2024, 4, 15))
    assert r2[0, "version"] == "late_first"


def test_pit_align_empty_inputs():
    assert pit_align(pl.DataFrame(), [date(2024, 1, 1)]).is_empty()
    fina = pl.DataFrame(
        {
            "ts_code": ["A"],
            "end_date": [date(2023, 6, 30)],
            "ann_date": [date(2023, 8, 15)],
            "roe": [1.0],
        }
    )
    assert pit_align(fina, []).is_empty()

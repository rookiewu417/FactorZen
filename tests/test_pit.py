"""测试 Point-In-Time 财务数据对齐。"""

import time
from datetime import date, timedelta

import polars as pl

from daily.data.pit import pit_align


# ── helpers ────────────────────────────────────────────────────────────────

def _make_fina(rows: list[tuple]) -> pl.DataFrame:
    """从 (ts_code, end_date, ann_date, roe) 元组列表构造财务数据。"""
    return pl.DataFrame(
        rows,
        schema={"ts_code": pl.Utf8, "end_date": pl.Date, "ann_date": pl.Date, "roe": pl.Float64},
        orient="row",
    )


# ── correctness ─────────────────────────────────────────────────────────────

def test_pit_align_correctness():
    """验证无前视偏差：快照日只使用「已公告」的财报中 end_date 最新的那条。"""
    # Stock A:
    #   Q2 report: end_date=2024-06-30, ann_date=2024-08-15
    #   Q3 report: end_date=2024-09-30, ann_date=2024-10-30
    fina = _make_fina([
        ("000001.SZ", date(2024, 6, 30), date(2024, 8, 15), 12.0),
        ("000001.SZ", date(2024, 9, 30), date(2024, 10, 30), 15.0),
    ])

    snapshots = [
        date(2024, 8, 31),   # Q2 已公告，Q3 未公告 → 应取 Q2（roe=12.0）
        date(2024, 10, 31),  # Q2/Q3 均已公告 → 应取 Q3（roe=15.0）
    ]

    result = pit_align(fina, snapshots)

    # 两个快照日各返回一条
    assert result.height == 2

    row_aug = result.filter(pl.col("snapshot_date") == date(2024, 8, 31))
    assert row_aug.height == 1
    assert row_aug[0, "end_date"] == date(2024, 6, 30)
    assert row_aug[0, "roe"] == 12.0

    row_oct = result.filter(pl.col("snapshot_date") == date(2024, 10, 31))
    assert row_oct.height == 1
    assert row_oct[0, "end_date"] == date(2024, 9, 30)
    assert row_oct[0, "roe"] == 15.0


def test_pit_align_multiple_stocks():
    """多股票场景：各自独立取最新已公告财报。"""
    d1, d2, d3 = date(2024, 3, 31), date(2024, 6, 30), date(2024, 9, 30)
    fina = _make_fina([
        ("A", d1, date(2024, 4, 25), 10.0),
        ("A", d2, date(2024, 8, 28), 12.0),
        ("A", d3, date(2024, 10, 30), 14.0),
        ("B", d1, date(2024, 4, 25), 20.0),
        ("B", d2, date(2024, 8, 30), 22.0),
        # B 没有 Q3
    ])

    snapshots = [date(2024, 9, 1), date(2024, 11, 1)]

    result = pit_align(fina, snapshots)

    # Sep: A 取 Q2(12.0), B 取 Q2(22.0)
    sep_a = result.filter(pl.col("snapshot_date") == date(2024, 9, 1), pl.col("ts_code") == "A")
    assert sep_a[0, "roe"] == 12.0
    sep_b = result.filter(pl.col("snapshot_date") == date(2024, 9, 1), pl.col("ts_code") == "B")
    assert sep_b[0, "roe"] == 22.0

    # Nov: A 取 Q3(14.0), B 仍取 Q2(22.0)（无 Q3）
    nov_a = result.filter(pl.col("snapshot_date") == date(2024, 11, 1), pl.col("ts_code") == "A")
    assert nov_a[0, "roe"] == 14.0
    nov_b = result.filter(pl.col("snapshot_date") == date(2024, 11, 1), pl.col("ts_code") == "B")
    assert nov_b[0, "roe"] == 22.0

    assert result.height == 4


# ── empty input ─────────────────────────────────────────────────────────────

def test_pit_align_empty_input():
    """空 DataFrame 或空 snapshot 列表 → 返回空 DataFrame。"""
    fina = _make_fina([
        ("A", date(2024, 6, 30), date(2024, 8, 1), 10.0),
    ])

    # 空 fina_df
    assert pit_align(pl.DataFrame(), [date(2024, 9, 1)]).is_empty()

    # 空 snapshot_dates
    assert pit_align(fina, []).is_empty()

    # 两者皆空
    assert pit_align(pl.DataFrame(), []).is_empty()


# ── performance ─────────────────────────────────────────────────────────────

def test_pit_align_performance():
    """1000 只股票 × 40 个月频快照 → 2 秒内完成。"""
    n_stocks = 1000
    n_periods = 40

    base = date(2020, 1, 1)
    # 每只股票 4 份年报（end_date 在 2020-2023）
    rows = []
    for s in range(n_stocks):
        for y in range(4):
            end_d = date(2020 + y, 12, 31)
            ann_d = date(2021 + y, 4, 30)
            rows.append((f"stock_{s:04d}", end_d, ann_d, y * 5.0 + s * 0.01))

    fina = _make_fina(rows)

    snapshots = [base + timedelta(days=30 * i) for i in range(n_periods)]

    start = time.perf_counter()
    result = pit_align(fina, snapshots)
    elapsed = time.perf_counter() - start

    assert not result.is_empty(), "结果不应为空"
    assert elapsed < 2.0, f"耗时 {elapsed:.2f}s ≥ 2s，算法效率不达标"

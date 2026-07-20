"""
test_pit.py：测试 Point-In-Time 财务数据对齐。
test_pit_align_equiv.py：pit_align 向量化等价性：golden 以旧实现输出为 ground truth。
test_fundamentals_pit.py：基本面叶子 PIT 对齐:确保 attach_fundamentals 无未来函数(铁律#1)。
test_fwd_returns.py：compute_fwd_returns 缺列 fail-fast 与 1d/5d 前瞻收益计算。
"""

from __future__ import annotations

import datetime as dt
import time
from datetime import date, timedelta

import polars as pl
import pytest

from factorzen.daily.data.pit import attach_fundamentals, pit_align
from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns

# ==== 来自 test_pit.py ====
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
    fina = _make_fina(
        [
            ("000001.SZ", date(2024, 6, 30), date(2024, 8, 15), 12.0),
            ("000001.SZ", date(2024, 9, 30), date(2024, 10, 30), 15.0),
        ]
    )

    snapshots = [
        date(2024, 8, 31),  # Q2 已公告，Q3 未公告 → 应取 Q2（roe=12.0）
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
    fina = _make_fina(
        [
            ("A", d1, date(2024, 4, 25), 10.0),
            ("A", d2, date(2024, 8, 28), 12.0),
            ("A", d3, date(2024, 10, 30), 14.0),
            ("B", d1, date(2024, 4, 25), 20.0),
            ("B", d2, date(2024, 8, 30), 22.0),
            # B 没有 Q3
        ]
    )

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
    fina = _make_fina(
        [
            ("A", date(2024, 6, 30), date(2024, 8, 1), 10.0),
        ]
    )

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

# ==== 来自 test_pit_align_equiv.py ====
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

# ==== 来自 test_fundamentals_pit.py ====
def _fina() -> pl.DataFrame:
    """两份报告:Q1(end 0331)0420 公告、Q2(end 0630)0815 公告——真实数据 ann/end 为 String。

    含全套质量/成长字段,验证扩充后的叶子一并 PIT 对齐。
    """
    return pl.DataFrame({
        "ts_code": ["000001.SZ", "000001.SZ"],
        "end_date": ["20200331", "20200630"],
        "ann_date": ["20200420", "20200815"],
        "roe": [10.0, 12.0], "roa": [1.0, 1.2],
        "grossprofit_margin": [40.0, 41.0], "netprofit_margin": [20.0, 21.0],
        "debt_to_assets": [50.0, 51.0],
        "or_yoy": [8.0, 9.0], "netprofit_yoy": [15.0, 16.0], "assets_yoy": [5.0, 6.0],
    })


def _daily(dates: list[str]) -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": [dt.datetime.strptime(d, "%Y%m%d").date() for d in dates],
        "ts_code": ["000001.SZ"] * len(dates),
        "close": [10.0] * len(dates),
    })


def test_no_future_leak_before_announcement():
    """t 日在 Q1 公告(0420)之前 → roe 必须是 null,绝不能把 0420 才公告的报告泄漏回 0410。"""
    out = attach_fundamentals(_daily(["20200410"]), fina_df=_fina())
    row = out.filter(pl.col("trade_date") == dt.date(2020, 4, 10))
    assert row["roe"][0] is None, "Q1 报告在公告日前泄漏 → 未来函数!"
    assert row["assets_yoy"][0] is None


def test_uses_latest_announced_report():
    """公告后取最新已公告报告:0420~0814 用 Q1(10.0);0815 起用 Q2(end 更大,12.0)。"""
    out = attach_fundamentals(_daily(["20200410", "20200501", "20200820"]), fina_df=_fina())
    by_date = {r["trade_date"]: r["roe"] for r in out.iter_rows(named=True)}
    assert by_date[dt.date(2020, 4, 10)] is None       # 公告前
    assert by_date[dt.date(2020, 5, 1)] == 10.0         # Q1 已公告
    assert by_date[dt.date(2020, 8, 20)] == 12.0        # Q2 已公告(end_date 更大)


def test_missing_finance_returns_daily_with_null_cols():
    """无 finance 数据(空帧)→ 原样返回但补齐 roe/assets_yoy 为 null(表达式引用不崩)。"""
    out = attach_fundamentals(_daily(["20200501"]), fina_df=pl.DataFrame())
    assert "roe" in out.columns and "assets_yoy" in out.columns
    assert out["roe"][0] is None


def test_expanded_fields_pit_aligned():
    """扩充的质量/成长字段(毛利率/营收增速等)与 roe 同套 PIT 对齐,公告后取最新报告。"""
    out = attach_fundamentals(_daily(["20200410", "20200820"]), fina_df=_fina())
    pre = out.filter(pl.col("trade_date") == dt.date(2020, 4, 10))
    post = out.filter(pl.col("trade_date") == dt.date(2020, 8, 20))
    for col in ("grossprofit_margin", "or_yoy", "netprofit_yoy", "debt_to_assets", "roa"):
        assert pre[col][0] is None, f"{col} 公告前泄漏 → 未来函数!"
    assert post["grossprofit_margin"][0] == 41.0   # Q2
    assert post["or_yoy"][0] == 9.0


def test_all_fundamental_leaves_registered_and_parse():
    """全套质量/成长叶子已注册且可解析(否则 LLM/搜索碰不到、prompt 广告了却用不了)。"""
    from factorzen.discovery.expression import feature_names, parse_expr
    from factorzen.discovery.operators import FUNDAMENTAL_FEATURES, LEAF_FEATURES
    expected = {"roe", "roa", "grossprofit_margin", "netprofit_margin", "debt_to_assets",
                "or_yoy", "netprofit_yoy", "assets_yoy"}
    assert expected <= FUNDAMENTAL_FEATURES
    for leaf in expected:
        assert leaf in LEAF_FEATURES, f"{leaf} 未注册为叶子"
        assert leaf in feature_names(parse_expr(f"rank({leaf})")), f"{leaf} 解析不出"

# ==== 来自 test_fwd_returns.py ====
def test_compute_fwd_returns_raises_on_missing_key_columns():
    # 缺少 ts_code → 应早失败并给出清晰错误,而非晦涩的 polars 异常
    df = pl.DataFrame({"trade_date": [date(2024, 1, 2)], "close": [100.0]})
    with pytest.raises(ValueError) as exc:
        compute_fwd_returns(df, horizons=[1])
    assert "ts_code" in str(exc.value)


def test_compute_fwd_returns_raises_when_no_price_or_ret_column():
    # 既无 close 也无 ret_col → 应早失败
    df = pl.DataFrame({"trade_date": [date(2024, 1, 2)], "ts_code": ["000001.SZ"]})
    with pytest.raises(ValueError) as exc:
        compute_fwd_returns(df, horizons=[1], ret_col="ret")
    msg = str(exc.value)
    assert "close" in msg and "ret" in msg


def test_fwd_ret_1d_uses_next_close_over_current_close():
    df = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "ts_code": ["000001.SZ"] * 3,
            "close": [100.0, 110.0, 121.0],
        }
    ).with_columns((pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("ret"))

    out = compute_fwd_returns(df, horizons=[1], ret_col="ret")

    assert out["fwd_ret_1d"].to_list() == pytest.approx([0.10, 0.10, None])


def test_fwd_ret_5d_is_cumulative_holding_period_return():
    closes = [100.0, 101.0, 103.0, 106.0, 110.0, 115.0, 121.0]
    df = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 5),
                date(2024, 1, 8),
                date(2024, 1, 9),
                date(2024, 1, 10),
            ],
            "ts_code": ["000001.SZ"] * len(closes),
            "close": closes,
        }
    ).with_columns((pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("ret"))

    out = compute_fwd_returns(df, horizons=[5], ret_col="ret")

    assert out["fwd_ret_5d"][0] == pytest.approx(115.0 / 100.0 - 1.0)
    assert out["fwd_ret_5d"][1] == pytest.approx(121.0 / 101.0 - 1.0)
    assert out["fwd_ret_5d"].to_list()[-5:] == [None, None, None, None, None]


def test_fwd_returns_compound_from_ret_when_close_is_absent():
    df = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 5),
            ],
            "ts_code": ["000001.SZ"] * 4,
            "ret": [0.0, 0.10, 0.20, -0.05],
        }
    )

    out = compute_fwd_returns(df, horizons=[2], ret_col="ret")

    assert out["fwd_ret_2d"][0] == pytest.approx((1.10 * 1.20) - 1.0)
    assert out["fwd_ret_2d"][1] == pytest.approx((1.20 * 0.95) - 1.0)
    assert out["fwd_ret_2d"].to_list()[-2:] == [None, None]


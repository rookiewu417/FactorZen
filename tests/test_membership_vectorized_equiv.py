"""universe membership 向量化等价性：与逐日 _load_index_members 完全一致。"""

from __future__ import annotations

import time
from datetime import date

import polars as pl
import pytest


def _membership_index_reference(
    start: str,
    end: str,
    universe_name: str,
) -> pl.DataFrame:
    """旧实现：逐交易日 _load_index_members（Wave1 前语义）。"""
    from factorzen.core.calendar import get_trade_dates
    from factorzen.core.universe import _INDEX_CODE_MAP, _load_index_members

    trade_dates = get_trade_dates(start, end)
    if not trade_dates:
        return pl.DataFrame(schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8})

    index_names = (
        ("csi300", "csi500") if universe_name == "csi800" else (universe_name,)
    )
    rows: list[dict[str, str]] = []
    for d in trade_dates:
        day_str = d.strftime("%Y%m%d")
        members: set[str] = set()
        for uname in index_names:
            code = _INDEX_CODE_MAP[uname]
            members.update(_load_index_members(code, day_str))
        for code in members:
            rows.append({"trade_date": day_str, "ts_code": code})

    if not rows:
        return pl.DataFrame(schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8})
    return pl.DataFrame(rows).select(["trade_date", "ts_code"]).unique()


def _assert_membership_equal(got: pl.DataFrame, expected: pl.DataFrame) -> None:
    g = (
        got.select(["trade_date", "ts_code"])
        .unique()
        .sort(["trade_date", "ts_code"])
    )
    e = (
        expected.select(["trade_date", "ts_code"])
        .unique()
        .sort(["trade_date", "ts_code"])
    )
    assert g.equals(e), (
        f"row mismatch: got={g.height} expected={e.height}\n"
        f"got-only sample: {g.join(e, on=['trade_date','ts_code'], how='anti').head(10)}\n"
        f"exp-only sample: {e.join(g, on=['trade_date','ts_code'], how='anti').head(10)}"
    )


def _has_csi500_cache() -> bool:
    from factorzen.config.settings import DATA_CACHE

    return any(DATA_CACHE.glob("index_member_000905_SH_*.parquet"))


@pytest.mark.skipif(not _has_csi500_cache(), reason="需要本地 csi500 成分缓存")
def test_membership_vectorized_matches_daily_loop_cross_month():
    """跨月窗口：逐日成分集合与旧实现完全一致。"""
    from factorzen.core.universe import (
        _INDEX_MEMBER_MEMORY_CACHE,
        get_universe_membership,
    )

    _INDEX_MEMBER_MEMORY_CACHE.clear()
    start, end = "20230101", "20230630"
    expected = _membership_index_reference(start, end, "csi500")
    _INDEX_MEMBER_MEMORY_CACHE.clear()
    got = get_universe_membership(start, end, "csi500")
    _assert_membership_equal(got, expected)


@pytest.mark.skipif(not _has_csi500_cache(), reason="需要本地 csi500 成分缓存")
def test_membership_vectorized_month_boundary_and_rebalance():
    """含月初/月末边界与成分调整月（6 月调样窗口）。"""
    from factorzen.core.universe import (
        _INDEX_MEMBER_MEMORY_CACHE,
        get_universe_membership,
    )

    _INDEX_MEMBER_MEMORY_CACHE.clear()
    # 5–7 月覆盖半年调样
    start, end = "20230501", "20230731"
    expected = _membership_index_reference(start, end, "csi500")
    _INDEX_MEMBER_MEMORY_CACHE.clear()
    got = get_universe_membership(start, end, "csi500")
    _assert_membership_equal(got, expected)

    # 抽检：月初与月末集合应各自非空（有缓存时）
    days = got["trade_date"].unique().sort().to_list()
    assert days
    for d in (days[0], days[len(days) // 2], days[-1]):
        assert got.filter(pl.col("trade_date") == d).height > 0


@pytest.mark.skipif(not _has_csi500_cache(), reason="需要本地 csi500 成分缓存")
def test_membership_vectorized_two_year_perf():
    """2 年 csi500 membership ≤ 0.5s。"""
    from factorzen.core.universe import (
        _INDEX_MEMBER_MEMORY_CACHE,
        get_universe_membership,
    )

    _INDEX_MEMBER_MEMORY_CACHE.clear()
    # warmup disk into OS cache
    _ = get_universe_membership("20230101", "20230131", "csi500")
    _INDEX_MEMBER_MEMORY_CACHE.clear()

    t0 = time.perf_counter()
    mem = get_universe_membership("20230101", "20241231", "csi500")
    elapsed = time.perf_counter() - t0
    assert mem.height > 100_000
    assert elapsed <= 0.5, f"membership {elapsed:.3f}s > 0.5s"


@pytest.mark.skipif(not _has_csi500_cache(), reason="需要本地 csi500 成分缓存")
def test_membership_csi800_union_equiv():
    from factorzen.core.universe import (
        _INDEX_MEMBER_MEMORY_CACHE,
        get_universe_membership,
    )

    _INDEX_MEMBER_MEMORY_CACHE.clear()
    start, end = "20240101", "20240331"
    expected = _membership_index_reference(start, end, "csi800")
    _INDEX_MEMBER_MEMORY_CACHE.clear()
    got = get_universe_membership(start, end, "csi800")
    _assert_membership_equal(got, expected)

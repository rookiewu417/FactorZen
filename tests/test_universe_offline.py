"""universe.py 离线单测：覆盖被 @needs_tushare 跳过的预设池、过滤器降级分支、
create_universe、_load_index_members 缓存/拉取，以及 get_index_members。

全部用 monkeypatch 注入合成数据，不依赖 TUSHARE_TOKEN 或本地 data/。
"""

from datetime import date
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

from factorzen.core import universe as U
from factorzen.core.universe import (
    create_universe,
    filter_limit,
    filter_liquidity,
    filter_new_listing,
    filter_st,
    filter_suspended,
    get_index_members,
    get_stock_basic,
    get_universe,
)

# ══════════════════════════════════════════════════════════
# fixtures & helpers
# ══════════════════════════════════════════════════════════


def _fake_pro():
    """init_tushare() 的桩：仅需具备被 _retry 引用的 index_weight 属性。"""
    return SimpleNamespace(index_weight=lambda **kw: None)


def _stock_basic() -> pl.DataFrame:
    """合成全 A 股基础信息：含正常股、ST 股、次新股、创业板股。"""
    return pl.DataFrame(
        {
            "ts_code": ["600000.SH", "600001.SH", "600002.SH", "300003.SZ"],
            "symbol": ["600000", "600001", "600002", "300003"],
            "name": ["正常股", "ST退市风险", "次新股", "创业板正常"],
            "area": ["上海"] * 4,
            "industry": ["银行"] * 4,
            "market": ["主板", "主板", "主板", "创业板"],
            "list_date": [
                date(2005, 1, 1),
                date(2005, 1, 1),
                date(2024, 1, 1),  # 距 20240115 仅 14 天 → 次新
                date(2005, 1, 1),
            ],
            "delist_date": [None, None, None, None],
        }
    )


@pytest.fixture
def stock_basic(monkeypatch):
    df = _stock_basic()
    monkeypatch.setattr(U, "get_stock_basic", lambda use_cache=True: df)
    return df


def _fake_daily(df: pl.DataFrame):
    """构造 load_parquet 的惰性返回桩：.collect() 返回给定 daily DataFrame。"""

    class _Lazy:
        def collect(self):
            return df

    def _load(category, start=None, end=None):
        return _Lazy()

    return _load


def _daily_all_tradeable() -> pl.DataFrame:
    """全部在市、低涨跌幅、高成交额的日线（让所有过滤器均放行）。"""
    codes = ["600000.SH", "600001.SH", "600002.SH", "300003.SZ"]
    return pl.DataFrame(
        {
            "ts_code": codes,
            "vol": [1000.0] * len(codes),
            "pct_chg": [1.0] * len(codes),
            "amount": [5_000_000_000.0] * len(codes),
            "open": [10.0] * len(codes),
            "close": [10.0] * len(codes),
        }
    )


# ══════════════════════════════════════════════════════════
# get_stock_basic / get_universe 校验
# ══════════════════════════════════════════════════════════


def test_get_stock_basic_delegates(monkeypatch):
    sentinel = pl.DataFrame({"ts_code": ["000001.SZ"]})
    monkeypatch.setattr(U, "fetch_stock_basic", lambda: sentinel)
    assert get_stock_basic().equals(sentinel)


def test_unknown_universe_raises(stock_basic):
    with pytest.raises(ValueError, match="未知 universe_name"):
        get_universe("20240115", "does_not_exist")


# ══════════════════════════════════════════════════════════
# 预设指数池（mock _load_index_members，离线）
# ══════════════════════════════════════════════════════════


def test_csi300_filters_to_members(stock_basic, monkeypatch):
    """csi300 应只保留指数成分股，与全 A 求交集。"""
    monkeypatch.setattr(U, "_load_index_members", lambda code, ds: ["600000.SH", "300003.SZ"])
    result = get_universe("20240115", "csi300")
    assert set(result["ts_code"].to_list()) == {"600000.SH", "300003.SZ"}


def test_csi800_is_union_of_300_and_500(stock_basic, monkeypatch):
    """csi800 = csi300 ∪ csi500（去重）。"""
    members = {"000300.SH": ["600000.SH"], "000905.SH": ["600001.SH", "300003.SZ"]}
    monkeypatch.setattr(U, "_load_index_members", lambda code, ds: members[code])
    result = get_universe("20240115", "csi800")
    assert set(result["ts_code"].to_list()) == {"600000.SH", "600001.SH", "300003.SZ"}


def test_csi_index_failure_falls_back_to_all_a(stock_basic, monkeypatch):
    """指数加载抛异常时应降级为全 A 股，而非崩溃。"""

    def _boom(code, ds):
        raise RuntimeError("tushare down")

    monkeypatch.setattr(U, "_load_index_members", _boom)
    result = get_universe("20240115", "csi500")
    # 全 A（PIT 过滤后 4 只均在市）
    assert result.height == 4


# ══════════════════════════════════════════════════════════
# daily_default / intraday_default / 别名
# ══════════════════════════════════════════════════════════


def test_daily_default_applies_full_filter_chain(stock_basic, monkeypatch):
    """daily_default：剔除 ST（600001）+ 次新（600002），保留正常股。"""
    monkeypatch.setattr("factorzen.core.storage.load_parquet", _fake_daily(_daily_all_tradeable()))
    result = get_universe("20240115", "daily_default")
    codes = set(result["ts_code"].to_list())
    assert codes == {"600000.SH", "300003.SZ"}


def test_intraday_default_adds_liquidity_filter(stock_basic, monkeypatch):
    """intraday_default：在 daily_default 基础上再剔除低流动性股。"""
    daily = _daily_all_tradeable().with_columns(
        # 600000 成交额低于 1000 万 → 被流动性过滤剔除
        pl.when(pl.col("ts_code") == "600000.SH")
        .then(1_000_000.0)
        .otherwise(pl.col("amount"))
        .alias("amount")
    )
    monkeypatch.setattr("factorzen.core.storage.load_parquet", _fake_daily(daily))
    result = get_universe("20240115", "intraday_default")
    codes = set(result["ts_code"].to_list())
    # daily_default 留下 {600000, 300003}，再剔除低流动性的 600000
    assert codes == {"300003.SZ"}


def test_lft_default_alias_equals_daily_default(stock_basic, monkeypatch):
    monkeypatch.setattr("factorzen.core.storage.load_parquet", _fake_daily(_daily_all_tradeable()))
    lft = set(get_universe("20240115", "lft_default")["ts_code"].to_list())
    daily = set(get_universe("20240115", "daily_default")["ts_code"].to_list())
    assert lft == daily


def test_mft_default_alias_equals_intraday_default(stock_basic, monkeypatch):
    monkeypatch.setattr("factorzen.core.storage.load_parquet", _fake_daily(_daily_all_tradeable()))
    mft = set(get_universe("20240115", "mft_default")["ts_code"].to_list())
    intraday = set(get_universe("20240115", "intraday_default")["ts_code"].to_list())
    assert mft == intraday


# ══════════════════════════════════════════════════════════
# 过滤器（纯逻辑 + 降级分支）
# ══════════════════════════════════════════════════════════


def test_filter_st_removes_st_and_pt():
    stocks = pl.DataFrame({"ts_code": ["a", "b", "c"], "name": ["正常", "*ST东方", "PT水仙"]})
    result = filter_st(stocks, "20240115")
    assert result["ts_code"].to_list() == ["a"]


def test_filter_new_listing_removes_recent():
    stocks = pl.DataFrame(
        {
            "ts_code": ["old", "new"],
            "list_date": [date(2005, 1, 1), date(2024, 1, 10)],
        }
    )
    # 20240115 - 250 天 ≈ 2023-05-10，new(2024-01-10) 在其之后 → 剔除
    result = filter_new_listing(stocks, "20240115", min_days=250)
    assert result["ts_code"].to_list() == ["old"]


def test_filter_new_listing_min_days_zero_keeps_all():
    stocks = pl.DataFrame(
        {"ts_code": ["a", "b"], "list_date": [date(2005, 1, 1), date(2024, 1, 10)]}
    )
    result = filter_new_listing(stocks, "20240115", min_days=0)
    assert set(result["ts_code"].to_list()) == {"a", "b"}


def test_filter_suspended_drops_zero_volume(monkeypatch):
    daily = pl.DataFrame({"ts_code": ["a", "b"], "vol": [1000.0, 0.0]})
    monkeypatch.setattr("factorzen.core.storage.load_parquet", _fake_daily(daily))
    stocks = pl.DataFrame({"ts_code": ["a", "b"], "name": ["x", "y"]})
    result = filter_suspended(stocks, "20240115")
    assert result["ts_code"].to_list() == ["a"]


def test_filter_suspended_empty_daily_no_filter(monkeypatch):
    """无日线数据时优雅降级：原样返回。"""
    empty = pl.DataFrame({"ts_code": [], "vol": []})
    monkeypatch.setattr("factorzen.core.storage.load_parquet", _fake_daily(empty))
    stocks = pl.DataFrame({"ts_code": ["a", "b"], "name": ["x", "y"]})
    result = filter_suspended(stocks, "20240115")
    assert result.height == 2


def test_filter_suspended_exception_no_filter(monkeypatch):
    """读取日线抛异常时优雅降级：原样返回。"""

    def _boom(category, start=None, end=None):
        raise OSError("disk gone")

    monkeypatch.setattr("factorzen.core.storage.load_parquet", _boom)
    stocks = pl.DataFrame({"ts_code": ["a", "b"], "name": ["x", "y"]})
    result = filter_suspended(stocks, "20240115")
    assert result.height == 2


def test_filter_limit_empty_daily_no_filter(monkeypatch):
    """无日线数据时 filter_limit 优雅降级：原样返回。"""
    empty = pl.DataFrame({"ts_code": [], "pct_chg": []})
    monkeypatch.setattr("factorzen.core.storage.load_parquet", _fake_daily(empty))
    stocks = pl.DataFrame({"ts_code": ["600000.SH"], "name": ["x"]})
    assert filter_limit(stocks, "20240115").height == 1


def test_filter_limit_exception_no_filter(monkeypatch):
    """读取日线抛异常时 filter_limit 优雅降级：原样返回。"""

    def _boom(category, start=None, end=None):
        raise OSError("io error")

    monkeypatch.setattr("factorzen.core.storage.load_parquet", _boom)
    stocks = pl.DataFrame({"ts_code": ["600000.SH"], "name": ["x"]})
    assert filter_limit(stocks, "20240115").height == 1


def test_filter_liquidity_drops_low_amount(monkeypatch):
    daily = pl.DataFrame({"ts_code": ["rich", "poor"], "amount": [2_000_000_000.0, 100.0]})
    monkeypatch.setattr("factorzen.core.storage.load_parquet", _fake_daily(daily))
    stocks = pl.DataFrame({"ts_code": ["rich", "poor"], "name": ["x", "y"]})
    result = filter_liquidity(stocks, "20240115", min_amount=10_000_000.0)
    assert result["ts_code"].to_list() == ["rich"]


def test_filter_liquidity_empty_daily_no_filter(monkeypatch):
    empty = pl.DataFrame({"ts_code": [], "amount": []})
    monkeypatch.setattr("factorzen.core.storage.load_parquet", _fake_daily(empty))
    stocks = pl.DataFrame({"ts_code": ["a"], "name": ["x"]})
    assert filter_liquidity(stocks, "20240115").height == 1


def test_filter_liquidity_exception_no_filter(monkeypatch):
    def _boom(category, start=None, end=None):
        raise ValueError("bad parquet")

    monkeypatch.setattr("factorzen.core.storage.load_parquet", _boom)
    stocks = pl.DataFrame({"ts_code": ["a"], "name": ["x"]})
    assert filter_liquidity(stocks, "20240115").height == 1


# ══════════════════════════════════════════════════════════
# create_universe
# ══════════════════════════════════════════════════════════


def test_create_universe_no_filters_returns_base(stock_basic):
    result = create_universe("20240115", base="all_a", filters=None)
    assert result.height == 4


def test_create_universe_applies_named_filters(stock_basic, monkeypatch):
    monkeypatch.setattr("factorzen.core.storage.load_parquet", _fake_daily(_daily_all_tradeable()))
    result = create_universe("20240115", base="all_a", filters=["st", "new_listing"])
    assert set(result["ts_code"].to_list()) == {"600000.SH", "300003.SZ"}


def test_create_universe_unknown_filter_skipped(stock_basic):
    """未知过滤项应跳过而非报错。"""
    result = create_universe("20240115", base="all_a", filters=["nope"])
    assert result.height == 4


def test_create_universe_liquidity_min_amount_passthrough(stock_basic, monkeypatch):
    """min_amount 应透传给 filter_liquidity。"""
    daily = pl.DataFrame(
        {
            "ts_code": ["600000.SH", "600001.SH", "600002.SH", "300003.SZ"],
            "amount": [3e8, 50.0, 50.0, 50.0],
        }
    )
    monkeypatch.setattr("factorzen.core.storage.load_parquet", _fake_daily(daily))
    result = create_universe("20240115", base="all_a", filters=["liquidity"], min_amount=1e8)
    assert result["ts_code"].to_list() == ["600000.SH"]


# ══════════════════════════════════════════════════════════
# _load_index_members 缓存/拉取/空
# ══════════════════════════════════════════════════════════


def test_load_index_members_cache_hit(monkeypatch, tmp_path):
    """缓存文件存在时直接读取，不调用 Tushare。"""
    monkeypatch.setattr(U, "DATA_CACHE", tmp_path)
    cache_file = tmp_path / "index_member_000300_SH_202401.parquet"
    pl.DataFrame({"con_code": ["600000.SH", "600519.SH"]}).write_parquet(cache_file)

    def _should_not_call():
        raise AssertionError("缓存命中时不应调用 init_tushare")

    monkeypatch.setattr("factorzen.core.loader.init_tushare", _should_not_call)
    result = U._load_index_members("000300.SH", "20240115")
    assert result == ["600000.SH", "600519.SH"]


def test_load_index_members_fetch_and_cache(monkeypatch, tmp_path):
    """缓存未命中时从 Tushare 拉取，并写入缓存。"""
    monkeypatch.setattr(U, "DATA_CACHE", tmp_path)
    monkeypatch.setattr("factorzen.core.loader.init_tushare", _fake_pro)
    df_pd = pd.DataFrame({"con_code": ["000001.SZ", "000002.SZ"]})
    monkeypatch.setattr("factorzen.core.loader._retry", lambda fn, **kw: df_pd)

    result = U._load_index_members("000905.SH", "20240115")
    assert result == ["000001.SZ", "000002.SZ"]
    # 写缓存：再次读取应命中（_retry 改为爆炸验证走缓存）
    cache_file = tmp_path / "index_member_000905_SH_202401.parquet"
    assert cache_file.exists()


def test_load_index_members_empty_returns_empty_list(monkeypatch, tmp_path):
    monkeypatch.setattr(U, "DATA_CACHE", tmp_path)
    monkeypatch.setattr("factorzen.core.loader.init_tushare", _fake_pro)
    monkeypatch.setattr("factorzen.core.loader._retry", lambda fn, **kw: pd.DataFrame())
    assert U._load_index_members("000300.SH", "20240115") == []


def test_csi500_uses_latest_cached_members_when_requested_month_empty(
    stock_basic, monkeypatch, tmp_path
):
    monkeypatch.setattr(U, "DATA_CACHE", tmp_path)
    cache_file = tmp_path / "index_member_000905_SH_202405.parquet"
    pl.DataFrame({"con_code": ["600000.SH", "300003.SZ"]}).write_parquet(cache_file)
    monkeypatch.setattr("factorzen.core.loader.init_tushare", _fake_pro)

    def _empty_current_month(_fn, **_kw):
        raise RuntimeError("Tushare 返回空结果")

    monkeypatch.setattr("factorzen.core.loader._retry", _empty_current_month)

    result = get_universe("20240615", "csi500")

    assert set(result["ts_code"].to_list()) == {"600000.SH", "300003.SZ"}


def test_load_index_members_reuses_fallback_members_in_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(U, "DATA_CACHE", tmp_path)
    U._INDEX_MEMBER_MEMORY_CACHE.clear()
    cache_file = tmp_path / "index_member_000905_SH_202405.parquet"
    pl.DataFrame({"con_code": ["600000.SH", "300003.SZ"]}).write_parquet(cache_file)
    monkeypatch.setattr("factorzen.core.loader.init_tushare", _fake_pro)
    calls = 0

    def _fail_current_month(_fn, **_kw):
        nonlocal calls
        calls += 1
        raise RuntimeError("current month unavailable")

    monkeypatch.setattr("factorzen.core.loader._retry", _fail_current_month)

    first = U._load_index_members("000905.SH", "20240615")
    second = U._load_index_members("000905.SH", "20240615")

    assert first == ["600000.SH", "300003.SZ"]
    assert second == ["600000.SH", "300003.SZ"]
    assert calls == 1


# ══════════════════════════════════════════════════════════
# get_index_members
# ══════════════════════════════════════════════════════════


def test_get_index_members_joins_stock_info(stock_basic, monkeypatch):
    monkeypatch.setattr(U, "_load_index_members", lambda code, ds: ["600000.SH"])
    result = get_index_members("000300.SH", "20240115")
    assert result["ts_code"].to_list() == ["600000.SH"]
    assert "industry" in result.columns


def test_get_index_members_empty_falls_back_to_all_market(stock_basic, monkeypatch):
    """无成分股时降级为全市场。"""
    monkeypatch.setattr(U, "_load_index_members", lambda code, ds: [])
    result = get_index_members("000300.SH", "20240115")
    assert result.height == 4


def test_get_index_members_exception_falls_back_to_all_market(stock_basic, monkeypatch):
    def _boom(code, ds):
        raise RuntimeError("api error")

    monkeypatch.setattr(U, "_load_index_members", _boom)
    result = get_index_members("000300.SH", "20240115")
    assert result.height == 4

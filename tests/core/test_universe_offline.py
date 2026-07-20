"""universe.py 离线单测：覆盖被 @needs_tushare 跳过的预设池、过滤器降级分支、
create_universe、_load_index_members 缓存/拉取，以及 get_index_members。

全部用 monkeypatch 注入合成数据，不依赖 TUSHARE_TOKEN 或本地 data/。
"""

import logging
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
    get_universe,
    get_universe_snapshot,
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


@pytest.fixture(autouse=True)
def _no_namechange_by_default(monkeypatch):
    """默认 namechange 不可用，统一走降级（按当前 name 字符串匹配）。

    本文件多数用例未显式 mock namechange；若不在此兜底，filter_st/filter_limit/
    get_universe_snapshot 会尝试调用真实 fetch_namechange()——在本机若 .env
    恰好配置了真实 TUSHARE_TOKEN，会触发真实网络请求，使离线测试变得非离线、
    非确定。需要测试 namechange 可用路径的用例，在测试体内自行
    monkeypatch.setattr(U, "fetch_namechange", ...) 覆盖本 fixture 的默认行为
    即可（universe.py 用 ``from factorzen.core.loader import fetch_namechange``
    在模块级绑定，须 patch ``U.fetch_namechange`` 而非
    ``factorzen.core.loader.fetch_namechange`` 才能生效）。
    """

    def _boom() -> pl.DataFrame:
        raise RuntimeError("namechange unavailable in offline tests")

    monkeypatch.setattr(U, "fetch_namechange", _boom)


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


def test_unknown_universe_raises(stock_basic):
    with pytest.raises(ValueError, match="未知 universe_name"):
        get_universe("20240115", "does_not_exist")


# ══════════════════════════════════════════════════════════
# 预设指数池（mock _load_index_members，离线）
# ══════════════════════════════════════════════════════════


def test_universe_presets_suite(stock_basic):
    """csi300 应只保留指数成分股，与全 A 求交集。；csi800 = csi300 ∪ csi500（去重）。；指数加载抛异常时 fail closed：raise ValueError，拒绝静默降级为全 A。；all_a 路径不经过指数加载；即便 _load_index_members 会炸也不应影响。"""
    # -- 原 test_csi300_filters_to_members --
    def _section_0_test_csi300_filters_to_members(stock_basic, mp):
        mp.setattr(U, "_load_index_members", lambda code, ds: ["600000.SH", "300003.SZ"])
        result = get_universe("20240115", "csi300")
        assert set(result["ts_code"].to_list()) == {"600000.SH", "300003.SZ"}

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_csi300_filters_to_members(stock_basic, mp)

    # -- 原 test_csi800_is_union_of_300_and_500 --
    def _section_1_test_csi800_is_union_of_300_and_500(stock_basic, mp):
        members = {"000300.SH": ["600000.SH"], "000905.SH": ["600001.SH", "300003.SZ"]}
        mp.setattr(U, "_load_index_members", lambda code, ds: members[code])
        result = get_universe("20240115", "csi800")
        assert set(result["ts_code"].to_list()) == {"600000.SH", "600001.SH", "300003.SZ"}

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_csi800_is_union_of_300_and_500(stock_basic, mp)

    # -- 原 test_csi_index_failure_fail_closed --
    def _section_2_test_csi_index_failure_fail_closed(stock_basic, mp):
        def _boom(code, ds):
            raise RuntimeError("tushare down")

        mp.setattr(U, "_load_index_members", _boom)
        with pytest.raises(ValueError, match=r"指数成分加载失败.*拒绝静默降级为全 A") as ei:
            get_universe("20240115", "csi300")
        assert ei.value.__cause__ is not None
        assert "tushare down" in str(ei.value.__cause__)
        assert "all_a" in str(ei.value)

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_csi_index_failure_fail_closed(stock_basic, mp)

    # -- 原 test_all_a_unaffected_by_index_loader --
    def _section_3_test_all_a_unaffected_by_index_loader(stock_basic, mp):
        def _boom(code, ds):
            raise RuntimeError("should not be called for all_a")

        mp.setattr(U, "_load_index_members", _boom)
        result = get_universe("20240115", "all_a")
        assert result.height == 4
        assert set(result["ts_code"].to_list()) == {
            "600000.SH",
            "600001.SH",
            "600002.SH",
            "300003.SZ",
        }

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_all_a_unaffected_by_index_loader(stock_basic, mp)


# ══════════════════════════════════════════════════════════
# daily_default / intraday_default / 别名
# ══════════════════════════════════════════════════════════


def test_default_and_alias_suite(stock_basic):
    """daily_default：剔除 ST（600001）+ 次新（600002），保留正常股。；intraday_default：在 daily_default 基础上再剔除低流动性股。；test_lft_default_alias_equals_daily_default；test_mft_default_alias_equals_intraday_default"""
    # -- 原 test_daily_default_applies_full_filter_chain --
    def _section_0_test_daily_default_applies_full_filter_chain(stock_basic, mp):
        mp.setattr("factorzen.core.storage.load_parquet", _fake_daily(_daily_all_tradeable()))
        result = get_universe("20240115", "daily_default")
        codes = set(result["ts_code"].to_list())
        assert codes == {"600000.SH", "300003.SZ"}

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_daily_default_applies_full_filter_chain(stock_basic, mp)

    # -- 原 test_intraday_default_adds_liquidity_filter --
    def _section_1_test_intraday_default_adds_liquidity_filter(stock_basic, mp):
        daily = _daily_all_tradeable().with_columns(
            # amount 单位=千元；5_000 千元=500万元 < 1000万元 门槛 → 被流动性过滤剔除
            pl.when(pl.col("ts_code") == "600000.SH")
            .then(5_000.0)
            .otherwise(pl.col("amount"))
            .alias("amount")
        )
        mp.setattr("factorzen.core.storage.load_parquet", _fake_daily(daily))
        result = get_universe("20240115", "intraday_default")
        codes = set(result["ts_code"].to_list())
        # daily_default 留下 {600000, 300003}，再剔除低流动性的 600000
        assert codes == {"300003.SZ"}

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_intraday_default_adds_liquidity_filter(stock_basic, mp)

    # -- 原 test_lft_default_alias_equals_daily_default --
    def _section_2_test_lft_default_alias_equals_daily_default(stock_basic, mp):
        mp.setattr("factorzen.core.storage.load_parquet", _fake_daily(_daily_all_tradeable()))
        lft = set(get_universe("20240115", "lft_default")["ts_code"].to_list())
        daily = set(get_universe("20240115", "daily_default")["ts_code"].to_list())
        assert lft == daily

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_lft_default_alias_equals_daily_default(stock_basic, mp)

    # -- 原 test_mft_default_alias_equals_intraday_default --
    def _section_3_test_mft_default_alias_equals_intraday_default(stock_basic, mp):
        mp.setattr("factorzen.core.storage.load_parquet", _fake_daily(_daily_all_tradeable()))
        mft = set(get_universe("20240115", "mft_default")["ts_code"].to_list())
        intraday = set(get_universe("20240115", "intraday_default")["ts_code"].to_list())
        assert mft == intraday

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_mft_default_alias_equals_intraday_default(stock_basic, mp)


# ══════════════════════════════════════════════════════════
# 过滤器（纯逻辑 + 降级分支）
# ══════════════════════════════════════════════════════════


def test_filter_st_namechange_suite(caplog):
    """namechange 不可用（autouse fixture 兜底）时，filter_st 走原 name 匹配降级路径。；namechange 可用时，filter_st 按 PIT 状态过滤，不再依赖当前最新 name。；namechange 获取失败时 filter_st 优雅降级为按 name 匹配，不崩溃，且仅警告一次。"""
    # -- 原 test_filter_st_removes_st_and_pt --
    def _section_0_test_filter_st_removes_st_and_pt():
        stocks = pl.DataFrame({"ts_code": ["a", "b", "c"], "name": ["正常", "*ST东方", "PT水仙"]})
        result = filter_st(stocks, "20240115")
        assert result["ts_code"].to_list() == ["a"]

    _section_0_test_filter_st_removes_st_and_pt()

    # -- 原 test_filter_st_uses_namechange_pit_over_current_name --
    def _section_1_test_filter_st_uses_namechange_pit_over_current_name(mp):
        namechange_df = _namechange_df(
            [
                {
                    "ts_code": "600001.SH",
                    "name": "ST退市股",
                    "start_date": date(2024, 1, 1),
                    "end_date": None,
                    "ann_date": date(2024, 1, 1),
                    "change_reason": "ST",
                },
            ]
        )
        mp.setattr(U, "fetch_namechange", lambda: namechange_df)

        # 600001.SH 当前 name 字段已不含 "ST"（如同已被改名/数据源更新），
        # 但 namechange 显示该代码在 date_str 当天仍处于 ST 状态区间内 → 仍应被剔除。
        stocks = pl.DataFrame(
            {"ts_code": ["600001.SH", "600005.SH"], "name": ["已正常化股票", "无关股票"]}
        )
        result = filter_st(stocks, "20240601")
        assert result["ts_code"].to_list() == ["600005.SH"]

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_filter_st_uses_namechange_pit_over_current_name(mp)

    # -- 原 test_filter_st_namechange_failure_falls_back_and_warns_once --
    def _section_2_test_filter_st_namechange_failure_falls_back_and_warns_once(mp, caplog):
        mp.setattr(U, "_namechange_unavailable_warned", False)

        calls = 0

        def _boom():
            nonlocal calls
            calls += 1
            raise RuntimeError("network down")

        mp.setattr(U, "fetch_namechange", _boom)

        stocks = pl.DataFrame({"ts_code": ["a", "b", "c"], "name": ["正常", "*ST东方", "PT水仙"]})

        with caplog.at_level(logging.WARNING, logger="factorzen.core.universe"):
            result1 = filter_st(stocks, "20240115")
            result2 = filter_st(stocks, "20240116")

        assert result1["ts_code"].to_list() == ["a"]
        assert result2["ts_code"].to_list() == ["a"]
        assert calls == 2, "降级模式下每次调用仍应尝试 namechange（不应有进程级跳过）"

        namechange_warnings = [
            r.message
            for r in caplog.records
            if r.levelno == logging.WARNING and "namechange" in r.message
        ]
        assert len(namechange_warnings) == 1, (
            f"namechange 失败警告应仅出现一次（不刷屏），实际出现 {len(namechange_warnings)} 次: "
            f"{namechange_warnings}"
        )

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_filter_st_namechange_failure_falls_back_and_warns_once(mp, caplog)


# ══════════════════════════════════════════════════════════
# PIT ST 状态判定（namechange）
# ══════════════════════════════════════════════════════════


def _namechange_df(rows: list[dict]) -> pl.DataFrame:
    """构造最小 namechange DataFrame，缺失列补 None。"""
    base = {
        "ts_code": None,
        "name": None,
        "start_date": None,
        "end_date": None,
        "ann_date": None,
        "change_reason": None,
    }
    filled = [{**base, **row} for row in rows]
    return pl.DataFrame(filled)


def test_is_st_asof_suite():
    """同一股票转 ST 前后两个查询日期，_is_st_asof 应返回不同的 ST 状态判断。；change_reason 含「撤销」不计入 ST 状态（表示该记录是 ST 被撤销后的新名称）。；change_reason 为 "*ST" 也应判定为 ST 状态（"*ST" 含 "ST" 子串）。；end_date 为空表示 ST 状态持续至今；查询日期须 >= start_date 才判定为 ST。；change_reason="摘星"（从 *ST 降级为 ST，即摘掉"星号"，不是摘帽/彻底摘星）"""
    # -- 原 test_is_st_asof_distinguishes_query_dates_around_transition --
    def _section_0_test_is_st_asof_distinguishes_query_dates_around_transition():
        namechange_df = _namechange_df(
            [
                {
                    "ts_code": "600001.SH",
                    "name": "正常股份",
                    "start_date": date(2020, 1, 1),
                    "end_date": date(2024, 6, 1),
                    "ann_date": date(2020, 1, 1),
                    "change_reason": None,
                },
                {
                    "ts_code": "600001.SH",
                    "name": "ST正常",
                    "start_date": date(2024, 6, 1),
                    "end_date": None,
                    "ann_date": date(2024, 6, 1),
                    "change_reason": "ST",
                },
            ]
        )
        before = U._is_st_asof(["600001.SH"], "20240501", namechange_df)
        after = U._is_st_asof(["600001.SH"], "20240701", namechange_df)
        assert before == set()
        assert after == {"600001.SH"}

    _section_0_test_is_st_asof_distinguishes_query_dates_around_transition()

    # -- 原 test_is_st_asof_excludes_revoked_st_reason --
    def _section_1_test_is_st_asof_excludes_revoked_st_reason():
        namechange_df = _namechange_df(
            [
                {
                    "ts_code": "600002.SH",
                    "name": "正常股份",
                    "start_date": date(2024, 6, 1),
                    "end_date": None,
                    "ann_date": date(2024, 6, 1),
                    "change_reason": "撤销ST",
                },
            ]
        )
        result = U._is_st_asof(["600002.SH"], "20240701", namechange_df)
        assert result == set()

    _section_1_test_is_st_asof_excludes_revoked_st_reason()

    # -- 原 test_is_st_asof_matches_star_st --
    def _section_2_test_is_st_asof_matches_star_st():
        namechange_df = _namechange_df(
            [
                {
                    "ts_code": "600003.SH",
                    "name": "*ST危困",
                    "start_date": date(2024, 6, 1),
                    "end_date": None,
                    "ann_date": date(2024, 6, 1),
                    "change_reason": "*ST",
                },
            ]
        )
        result = U._is_st_asof(["600003.SH"], "20240701", namechange_df)
        assert result == {"600003.SH"}

    _section_2_test_is_st_asof_matches_star_st()

    # -- 原 test_is_st_asof_end_date_null_means_ongoing --
    def _section_3_test_is_st_asof_end_date_null_means_ongoing():
        namechange_df = _namechange_df(
            [
                {
                    "ts_code": "600004.SH",
                    "name": "ST恒续",
                    "start_date": date(2024, 1, 1),
                    "end_date": None,
                    "ann_date": date(2024, 1, 1),
                    "change_reason": "ST",
                },
            ]
        )
        far_future = U._is_st_asof(["600004.SH"], "20300101", namechange_df)
        before_start = U._is_st_asof(["600004.SH"], "20231231", namechange_df)
        assert far_future == {"600004.SH"}
        assert before_start == set()

    _section_3_test_is_st_asof_end_date_null_means_ongoing()

    # -- 原 test_is_st_asof_treats_downgrade_from_star_st_as_still_st --
    def _section_4_test_is_st_asof_treats_downgrade_from_star_st_as_still_st():
        namechange_df = _namechange_df(
            [
                {
                    "ts_code": "000410.SZ",
                    "name": "ST沈机",
                    "start_date": date(2021, 6, 24),
                    "end_date": None,
                    "ann_date": date(2021, 6, 23),
                    "change_reason": "摘星",
                },
            ]
        )
        result = U._is_st_asof(["000410.SZ"], "20240101", namechange_df)
        assert result == {"000410.SZ"}

    _section_4_test_is_st_asof_treats_downgrade_from_star_st_as_still_st()


def test_build_is_st_by_date_suite():
    """build_is_st_by_date 应对回测窗口内每个交易日独立判断 ST 状态（PIT）。；回测窗口横跨多个交易日时，namechange 全量数据只应拉取一次，不应逐日重复拉取。；namechange 不可用（autouse fixture 兜底）且提供 name_source 时，按当前名称降级判断，且对所有交易日一致。；namechange 不可用且未提供 name_source 时返回空 dict，等价于不区分 ST（与引入本函数前行为一致）。"""
    # -- 原 test_build_is_st_by_date_reflects_per_date_st_transition --
    def _section_0_test_build_is_st_by_date_reflects_per_date_st_transition(mp):
        namechange_df = _namechange_df(
            [
                {
                    "ts_code": "600001.SH",
                    "name": "正常股份",
                    "start_date": date(2020, 1, 1),
                    "end_date": date(2024, 6, 1),
                    "ann_date": date(2020, 1, 1),
                    "change_reason": None,
                },
                {
                    "ts_code": "600001.SH",
                    "name": "ST正常",
                    "start_date": date(2024, 6, 1),
                    "end_date": None,
                    "ann_date": date(2024, 6, 1),
                    "change_reason": "ST",
                },
            ]
        )
        mp.setattr(U, "fetch_namechange", lambda: namechange_df)

        trade_dates = [date(2024, 5, 1), date(2024, 7, 1)]
        result = U.build_is_st_by_date(["600001.SH"], trade_dates)

        assert result[date(2024, 5, 1)] == set()
        assert result[date(2024, 7, 1)] == {"600001.SH"}

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_build_is_st_by_date_reflects_per_date_st_transition(mp)

    # -- 原 test_build_is_st_by_date_fetches_namechange_only_once --
    def _section_1_test_build_is_st_by_date_fetches_namechange_only_once(mp):
        namechange_df = _namechange_df([])
        calls = 0

        def _counting_fetch():
            nonlocal calls
            calls += 1
            return namechange_df

        mp.setattr(U, "fetch_namechange", _counting_fetch)

        trade_dates = [date(2024, 1, i) for i in range(1, 11)]
        U.build_is_st_by_date(["600001.SH"], trade_dates)

        assert calls == 1, f"namechange 应只拉取一次，实际拉取 {calls} 次"

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_build_is_st_by_date_fetches_namechange_only_once(mp)

    # -- 原 test_build_is_st_by_date_falls_back_to_name_source_when_namechange_unavailable --
    def _section_2_test_build_is_st_by_date_falls_back_to_name_source_when_namechange_unavailable():
        name_source = pl.DataFrame(
            {"ts_code": ["600001.SH", "600005.SH"], "name": ["*ST东方", "正常股"]}
        )
        trade_dates = [date(2024, 1, 2), date(2024, 1, 3)]

        result = U.build_is_st_by_date(
            ["600001.SH", "600005.SH"], trade_dates, name_source=name_source
        )

        assert result == {
            date(2024, 1, 2): {"600001.SH"},
            date(2024, 1, 3): {"600001.SH"},
        }

    _section_2_test_build_is_st_by_date_falls_back_to_name_source_when_namechange_unavailable()

    # -- 原 test_build_is_st_by_date_returns_empty_when_no_namechange_and_no_name_source --
    def _section_3_test_build_is_st_by_date_returns_empty_when_no_namechange_and_no_name_source():
        trade_dates = [date(2024, 1, 2), date(2024, 1, 3)]
        result = U.build_is_st_by_date(["600001.SH"], trade_dates)
        assert result == {}

    _section_3_test_build_is_st_by_date_returns_empty_when_no_namechange_and_no_name_source()


def test_snapshot_is_st_suite(stock_basic):
    """namechange 可用时，get_universe_snapshot 的 is_st 列按 PIT 状态判断。；namechange 获取失败时，get_universe_snapshot 的 is_st 列优雅降级为按 name 匹配。"""
    # -- 原 test_universe_snapshot_is_st_uses_namechange_pit --
    def _section_0_test_universe_snapshot_is_st_uses_namechange_pit(stock_basic, mp):
        mp.setattr("factorzen.core.storage.load_parquet", _fake_daily(_daily_all_tradeable()))
        namechange_df = _namechange_df(
            [
                {
                    # _stock_basic() 中 600000.SH name="正常股"（不含 ST），
                    # 但 namechange 显示其 PIT 当天处于 ST 状态 → is_st 应为 True。
                    "ts_code": "600000.SH",
                    "name": "ST正常股",
                    "start_date": date(2024, 1, 1),
                    "end_date": None,
                    "ann_date": date(2024, 1, 1),
                    "change_reason": "ST",
                },
            ]
        )
        mp.setattr(U, "fetch_namechange", lambda: namechange_df)

        result = get_universe_snapshot("20240115", "all_a")
        is_st_map = dict(zip(result["ts_code"].to_list(), result["is_st"].to_list(), strict=False))
        assert is_st_map["600000.SH"] is True
        # 600001.SH 当前 name="ST退市风险"（含 "ST"），但 namechange 中无该代码记录，
        # 说明 PIT 路径不再回退到 name 匹配——namechange 可用时以它为唯一依据。
        assert is_st_map["600001.SH"] is False

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_universe_snapshot_is_st_uses_namechange_pit(stock_basic, mp)

    # -- 原 test_universe_snapshot_is_st_namechange_failure_falls_back --
    def _section_1_test_universe_snapshot_is_st_namechange_failure_falls_back(stock_basic, mp):
        mp.setattr("factorzen.core.storage.load_parquet", _fake_daily(_daily_all_tradeable()))

        def _boom():
            raise RuntimeError("network down")

        mp.setattr(U, "fetch_namechange", _boom)

        result = get_universe_snapshot("20240115", "all_a")
        is_st_map = dict(zip(result["ts_code"].to_list(), result["is_st"].to_list(), strict=False))
        # _stock_basic() fixture: 600001.SH name="ST退市风险" → 降级模式下按 name 判定 is_st=True
        assert is_st_map["600001.SH"] is True
        assert is_st_map["600000.SH"] is False

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_universe_snapshot_is_st_namechange_failure_falls_back(stock_basic, mp)


def test_filter_new_listing_suite():
    """test_filter_new_listing_removes_recent；test_filter_new_listing_min_days_zero_keeps_all"""
    # -- 原 test_filter_new_listing_removes_recent --
    def _section_0_test_filter_new_listing_removes_recent():
        stocks = pl.DataFrame(
            {
                "ts_code": ["old", "new"],
                "list_date": [date(2005, 1, 1), date(2024, 1, 10)],
            }
        )
        # 20240115 - 250 天 ≈ 2023-05-10，new(2024-01-10) 在其之后 → 剔除
        result = filter_new_listing(stocks, "20240115", min_days=250)
        assert result["ts_code"].to_list() == ["old"]

    _section_0_test_filter_new_listing_removes_recent()

    # -- 原 test_filter_new_listing_min_days_zero_keeps_all --
    def _section_1_test_filter_new_listing_min_days_zero_keeps_all():
        stocks = pl.DataFrame(
            {"ts_code": ["a", "b"], "list_date": [date(2005, 1, 1), date(2024, 1, 10)]}
        )
        result = filter_new_listing(stocks, "20240115", min_days=0)
        assert set(result["ts_code"].to_list()) == {"a", "b"}

    _section_1_test_filter_new_listing_min_days_zero_keeps_all()


def test_filter_suspended_suite():
    """test_filter_suspended_drops_zero_volume；无日线数据时优雅降级：原样返回。；读取日线抛异常时优雅降级：原样返回。"""
    # -- 原 test_filter_suspended_drops_zero_volume --
    def _section_0_test_filter_suspended_drops_zero_volume(mp):
        daily = pl.DataFrame({"ts_code": ["a", "b"], "vol": [1000.0, 0.0]})
        mp.setattr("factorzen.core.storage.load_parquet", _fake_daily(daily))
        stocks = pl.DataFrame({"ts_code": ["a", "b"], "name": ["x", "y"]})
        result = filter_suspended(stocks, "20240115")
        assert result["ts_code"].to_list() == ["a"]

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_filter_suspended_drops_zero_volume(mp)

    # -- 原 test_filter_suspended_empty_daily_no_filter --
    def _section_1_test_filter_suspended_empty_daily_no_filter(mp):
        empty = pl.DataFrame({"ts_code": [], "vol": []})
        mp.setattr("factorzen.core.storage.load_parquet", _fake_daily(empty))
        stocks = pl.DataFrame({"ts_code": ["a", "b"], "name": ["x", "y"]})
        result = filter_suspended(stocks, "20240115")
        assert result.height == 2

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_filter_suspended_empty_daily_no_filter(mp)

    # -- 原 test_filter_suspended_exception_no_filter --
    def _section_2_test_filter_suspended_exception_no_filter(mp):
        def _boom(category, start=None, end=None):
            raise OSError("disk gone")

        mp.setattr("factorzen.core.storage.load_parquet", _boom)
        stocks = pl.DataFrame({"ts_code": ["a", "b"], "name": ["x", "y"]})
        result = filter_suspended(stocks, "20240115")
        assert result.height == 2

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_filter_suspended_exception_no_filter(mp)


def test_filter_limit_degrade_suite():
    """无日线数据时 filter_limit 优雅降级：原样返回。；读取日线抛异常时 filter_limit 优雅降级：原样返回。"""
    # -- 原 test_filter_limit_empty_daily_no_filter --
    def _section_0_test_filter_limit_empty_daily_no_filter(mp):
        empty = pl.DataFrame({"ts_code": [], "pct_chg": []})
        mp.setattr("factorzen.core.storage.load_parquet", _fake_daily(empty))
        stocks = pl.DataFrame({"ts_code": ["600000.SH"], "name": ["x"]})
        assert filter_limit(stocks, "20240115").height == 1

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_filter_limit_empty_daily_no_filter(mp)

    # -- 原 test_filter_limit_exception_no_filter --
    def _section_1_test_filter_limit_exception_no_filter(mp):
        def _boom(category, start=None, end=None):
            raise OSError("io error")

        mp.setattr("factorzen.core.storage.load_parquet", _boom)
        stocks = pl.DataFrame({"ts_code": ["600000.SH"], "name": ["x"]})
        assert filter_limit(stocks, "20240115").height == 1

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_filter_limit_exception_no_filter(mp)


def test_filter_limit_floating_point_tolerance_chuang_ye_limit_up(monkeypatch):
    """创业板涨停浮点容差回归。

    open=11.98/pre_close=10.0 算出 pct_chg=(11.98/10.0-1)*100≈19.799999999999997
    （由除法算出，非字面量 19.8），与板块阈值 19.8 比较时若不加 1e-9 容差，
    19.799999999999997 < 19.8 为 True，会被误判为「未到涨停」而漏过滤。
    口径需与 backtest.py 的涨跌停判断一致。
    """
    open_px = 11.98
    pre_close = 10.0
    pct_chg = (open_px / pre_close - 1.0) * 100
    daily = pl.DataFrame(
        {
            "ts_code": ["300003.SZ", "600000.SH"],
            "pct_chg": [pct_chg, 1.0],
        }
    )
    monkeypatch.setattr("factorzen.core.storage.load_parquet", _fake_daily(daily))
    stocks = pl.DataFrame({"ts_code": ["300003.SZ", "600000.SH"], "name": ["创业板", "主板"]})
    result = filter_limit(stocks, "20240115")
    assert result["ts_code"].to_list() == ["600000.SH"], (
        f"创业板涨停股 300003.SZ (pct_chg={pct_chg!r}) 应被过滤，实际={result['ts_code'].to_list()}"
    )


def test_snapshot_limit_float_suite(stock_basic):
    """get_universe_snapshot 的 is_limit_up 同样需要浮点容差。；get_universe_snapshot 的 is_limit_down 同样需要浮点容差。"""
    # -- 原 test_universe_snapshot_floating_point_tolerance_is_limit_up --
    def _section_0_test_universe_snapshot_floating_point_tolerance_is_limit_up(stock_basic, mp):
        open_px = 11.98
        pre_close = 10.0
        pct_chg = (open_px / pre_close - 1.0) * 100
        daily = pl.DataFrame(
            {
                "ts_code": ["600000.SH", "600001.SH", "600002.SH", "300003.SZ"],
                "vol": [1000.0] * 4,
                "pct_chg": [1.0, 1.0, 1.0, pct_chg],
            }
        )
        mp.setattr("factorzen.core.storage.load_parquet", _fake_daily(daily))
        result = get_universe_snapshot("20240115", "all_a")
        row = result.filter(pl.col("ts_code") == "300003.SZ")
        assert row["is_limit_up"].to_list() == [True], (
            f"创业板 pct_chg={pct_chg!r} 应判定为涨停，实际 is_limit_up={row['is_limit_up'].to_list()}"
        )

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_universe_snapshot_floating_point_tolerance_is_limit_up(stock_basic, mp)

    # -- 原 test_universe_snapshot_floating_point_tolerance_is_limit_down --
    def _section_1_test_universe_snapshot_floating_point_tolerance_is_limit_down(stock_basic, mp):
        open_px = 80.2
        pre_close = 100.0
        pct_chg = (open_px / pre_close - 1.0) * 100
        daily = pl.DataFrame(
            {
                "ts_code": ["600000.SH", "600001.SH", "600002.SH", "300003.SZ"],
                "vol": [1000.0] * 4,
                "pct_chg": [1.0, 1.0, 1.0, pct_chg],
            }
        )
        mp.setattr("factorzen.core.storage.load_parquet", _fake_daily(daily))
        result = get_universe_snapshot("20240115", "all_a")
        row = result.filter(pl.col("ts_code") == "300003.SZ")
        assert row["is_limit_down"].to_list() == [True], (
            f"创业板 pct_chg={pct_chg!r} 应判定为跌停，实际 is_limit_down={row['is_limit_down'].to_list()}"
        )

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_universe_snapshot_floating_point_tolerance_is_limit_down(stock_basic, mp)


def test_snapshot_suspended_suite(stock_basic):
    """无日线行的股票（A股停牌当日 Tushare 不发日线行）应标 is_suspended=True。；有日线行但 vol==0（部分数据源发零量行）仍应判停牌——修复不能回归此路径。"""
    # -- 原 test_universe_snapshot_missing_daily_bar_marked_suspended --
    def _section_0_test_universe_snapshot_missing_daily_bar_marked_suspended(stock_basic, mp):
        daily = pl.DataFrame(
            {
                "ts_code": ["600001.SH", "600002.SH", "300003.SZ"],
                "vol": [1000.0] * 3,
                "pct_chg": [1.0] * 3,
            }
        )
        mp.setattr("factorzen.core.storage.load_parquet", _fake_daily(daily))
        result = get_universe_snapshot("20240115", "all_a")
        susp = result.filter(pl.col("ts_code") == "600000.SH")["is_suspended"].to_list()
        assert susp == [True], f"停牌股（无日线行）应 is_suspended=True，实得 {susp}"
        # 有日线行、正常交易的股仍为 False（不误伤）
        ok = result.filter(pl.col("ts_code") == "600001.SH")["is_suspended"].to_list()
        assert ok == [False], f"正常交易股应 is_suspended=False，实得 {ok}"

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_universe_snapshot_missing_daily_bar_marked_suspended(stock_basic, mp)

    # -- 原 test_universe_snapshot_zero_volume_bar_marked_suspended --
    def _section_1_test_universe_snapshot_zero_volume_bar_marked_suspended(stock_basic, mp):
        daily = pl.DataFrame(
            {
                "ts_code": ["600000.SH", "600001.SH", "600002.SH", "300003.SZ"],
                "vol": [0.0, 1000.0, 1000.0, 1000.0],
                "pct_chg": [0.0, 1.0, 1.0, 1.0],
            }
        )
        mp.setattr("factorzen.core.storage.load_parquet", _fake_daily(daily))
        result = get_universe_snapshot("20240115", "all_a")
        susp = result.filter(pl.col("ts_code") == "600000.SH")["is_suspended"].to_list()
        assert susp == [True], f"vol==0 应 is_suspended=True，实得 {susp}"

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_universe_snapshot_zero_volume_bar_marked_suspended(stock_basic, mp)


def test_filter_liquidity_suite():
    """test_filter_liquidity_drops_low_amount；test_filter_liquidity_empty_daily_no_filter；test_filter_liquidity_exception_no_filter"""
    # -- 原 test_filter_liquidity_drops_low_amount --
    def _section_0_test_filter_liquidity_drops_low_amount(mp):
        daily = pl.DataFrame({"ts_code": ["rich", "poor"], "amount": [2_000_000_000.0, 100.0]})
        mp.setattr("factorzen.core.storage.load_parquet", _fake_daily(daily))
        stocks = pl.DataFrame({"ts_code": ["rich", "poor"], "name": ["x", "y"]})
        result = filter_liquidity(stocks, "20240115", min_amount=10_000_000.0)
        assert result["ts_code"].to_list() == ["rich"]

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_filter_liquidity_drops_low_amount(mp)

    # -- 原 test_filter_liquidity_empty_daily_no_filter --
    def _section_1_test_filter_liquidity_empty_daily_no_filter(mp):
        empty = pl.DataFrame({"ts_code": [], "amount": []})
        mp.setattr("factorzen.core.storage.load_parquet", _fake_daily(empty))
        stocks = pl.DataFrame({"ts_code": ["a"], "name": ["x"]})
        assert filter_liquidity(stocks, "20240115").height == 1

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_filter_liquidity_empty_daily_no_filter(mp)

    # -- 原 test_filter_liquidity_exception_no_filter --
    def _section_2_test_filter_liquidity_exception_no_filter(mp):
        def _boom(category, start=None, end=None):
            raise ValueError("bad parquet")

        mp.setattr("factorzen.core.storage.load_parquet", _boom)
        stocks = pl.DataFrame({"ts_code": ["a"], "name": ["x"]})
        assert filter_liquidity(stocks, "20240115").height == 1

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_filter_liquidity_exception_no_filter(mp)


# ══════════════════════════════════════════════════════════
# create_universe
# ══════════════════════════════════════════════════════════


def test_create_universe_suite(stock_basic):
    """test_create_universe_no_filters_returns_base；test_create_universe_applies_named_filters；未知过滤项应跳过而非报错。；min_amount 应透传给 filter_liquidity。"""
    # -- 原 test_create_universe_no_filters_returns_base --
    def _section_0_test_create_universe_no_filters_returns_base(stock_basic):
        result = create_universe("20240115", base="all_a", filters=None)
        assert result.height == 4

    _section_0_test_create_universe_no_filters_returns_base(stock_basic)

    # -- 原 test_create_universe_applies_named_filters --
    def _section_1_test_create_universe_applies_named_filters(stock_basic, mp):
        mp.setattr("factorzen.core.storage.load_parquet", _fake_daily(_daily_all_tradeable()))
        result = create_universe("20240115", base="all_a", filters=["st", "new_listing"])
        assert set(result["ts_code"].to_list()) == {"600000.SH", "300003.SZ"}

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_create_universe_applies_named_filters(stock_basic, mp)

    # -- 原 test_create_universe_unknown_filter_skipped --
    def _section_2_test_create_universe_unknown_filter_skipped(stock_basic):
        result = create_universe("20240115", base="all_a", filters=["nope"])
        assert result.height == 4

    _section_2_test_create_universe_unknown_filter_skipped(stock_basic)

    # -- 原 test_create_universe_liquidity_min_amount_passthrough --
    def _section_3_test_create_universe_liquidity_min_amount_passthrough(stock_basic, mp):
        daily = pl.DataFrame(
            {
                "ts_code": ["600000.SH", "600001.SH", "600002.SH", "300003.SZ"],
                "amount": [3e8, 50.0, 50.0, 50.0],
            }
        )
        mp.setattr("factorzen.core.storage.load_parquet", _fake_daily(daily))
        result = create_universe("20240115", base="all_a", filters=["liquidity"], min_amount=1e8)
        assert result["ts_code"].to_list() == ["600000.SH"]

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_create_universe_liquidity_min_amount_passthrough(stock_basic, mp)


# ══════════════════════════════════════════════════════════
# _load_index_members 缓存/拉取/空
# ══════════════════════════════════════════════════════════


def test_load_index_members_io_suite(tmp_path):
    """缓存文件存在时直接读取，不调用 Tushare。；缓存未命中时从 Tushare 拉取，并写入缓存。；test_load_index_members_empty_returns_empty_list"""
    # -- 原 test_load_index_members_cache_hit --
    def _section_0_test_load_index_members_cache_hit(mp, tmp_path):
        mp.setattr(U, "DATA_CACHE", tmp_path)
        cache_file = tmp_path / "index_member_000300_SH_202401.parquet"
        pl.DataFrame(
            {
                "con_code": ["600000.SH", "600519.SH"],
                "trade_date": ["20240110", "20240110"],
            }
        ).write_parquet(cache_file)

        def _should_not_call():
            raise AssertionError("缓存命中时不应调用 init_tushare")

        mp.setattr("factorzen.core.loader.init_tushare", _should_not_call)
        result = U._load_index_members("000300.SH", "20240115")
        assert result == ["600000.SH", "600519.SH"]

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_load_index_members_cache_hit(mp, _tp0)

    # -- 原 test_load_index_members_fetch_and_cache --
    def _section_1_test_load_index_members_fetch_and_cache(mp, tmp_path):
        mp.setattr(U, "DATA_CACHE", tmp_path)
        mp.setattr("factorzen.core.loader.init_tushare", _fake_pro)
        df_pd = pd.DataFrame(
            {
                "con_code": ["000001.SZ", "000002.SZ"],
                "trade_date": ["20240105", "20240105"],
            }
        )
        mp.setattr("factorzen.core.loader._retry", lambda fn, **kw: df_pd)

        result = U._load_index_members("000905.SH", "20240115")
        assert result == ["000001.SZ", "000002.SZ"]
        # 写缓存：再次读取应命中（_retry 改为爆炸验证走缓存）
        cache_file = tmp_path / "index_member_000905_SH_202401.parquet"
        assert cache_file.exists()

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_load_index_members_fetch_and_cache(mp, _tp1)

    # -- 原 test_load_index_members_empty_returns_empty_list --
    def _section_2_test_load_index_members_empty_returns_empty_list(mp, tmp_path):
        mp.setattr(U, "DATA_CACHE", tmp_path)
        mp.setattr("factorzen.core.loader.init_tushare", _fake_pro)
        mp.setattr("factorzen.core.loader._retry", lambda fn, **kw: pd.DataFrame())
        assert U._load_index_members("000300.SH", "20240115") == []

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_load_index_members_empty_returns_empty_list(mp, _tp2)


def test_load_index_members_fallback_suite(stock_basic, tmp_path):
    """test_csi500_uses_latest_cached_members_when_requested_month_empty；test_load_index_members_reuses_fallback_members_in_memory；当月数据非空但没有任何 trade_date<=查询日 的记录（如当月首个快照本身就"""
    # -- 原 test_csi500_uses_latest_cached_members_when_requested_month_empty --
    def _section_0_test_csi500_uses_latest_cached_members_when_requested_month_empty(stock_basic, mp, tmp_path):
        mp.setattr(U, "DATA_CACHE", tmp_path)
        cache_file = tmp_path / "index_member_000905_SH_202405.parquet"
        pl.DataFrame(
            {
                "con_code": ["600000.SH", "300003.SZ"],
                "trade_date": ["20240510", "20240510"],
            }
        ).write_parquet(cache_file)
        mp.setattr("factorzen.core.loader.init_tushare", _fake_pro)

        def _empty_current_month(_fn, **_kw):
            raise RuntimeError("Tushare 返回空结果")

        mp.setattr("factorzen.core.loader._retry", _empty_current_month)

        result = get_universe("20240615", "csi500")

        assert set(result["ts_code"].to_list()) == {"600000.SH", "300003.SZ"}

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_csi500_uses_latest_cached_members_when_requested_month_empty(stock_basic, mp, _tp0)

    # -- 原 test_load_index_members_reuses_fallback_members_in_memory --
    def _section_1_test_load_index_members_reuses_fallback_members_in_memory(mp, tmp_path):
        mp.setattr(U, "DATA_CACHE", tmp_path)
        U._INDEX_MEMBER_MEMORY_CACHE.clear()
        cache_file = tmp_path / "index_member_000905_SH_202405.parquet"
        pl.DataFrame(
            {
                "con_code": ["600000.SH", "300003.SZ"],
                "trade_date": ["20240510", "20240510"],
            }
        ).write_parquet(cache_file)
        mp.setattr("factorzen.core.loader.init_tushare", _fake_pro)
        calls = 0

        def _fail_current_month(_fn, **_kw):
            nonlocal calls
            calls += 1
            raise RuntimeError("current month unavailable")

        mp.setattr("factorzen.core.loader._retry", _fail_current_month)

        first = U._load_index_members("000905.SH", "20240615")
        second = U._load_index_members("000905.SH", "20240615")

        assert first == ["600000.SH", "300003.SZ"]
        assert second == ["600000.SH", "300003.SZ"]
        assert calls == 1

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_load_index_members_reuses_fallback_members_in_memory(mp, _tp1)

    # -- 原 test_load_index_members_falls_back_when_current_month_has_no_eligible_snapshot --
    def _section_2_test_load_index_members_falls_back_when_current_month_has_no_eligible_snapshot(mp, tmp_path):
        mp.setattr(U, "DATA_CACHE", tmp_path)
        U._INDEX_MEMBER_MEMORY_CACHE.clear()

        # 上一个月(1月)已有缓存，15号生效；查询日2月5日晚于1月15日，可作为回退结果
        prior_cache = tmp_path / "index_member_000300_SH_202401.parquet"
        pl.DataFrame({"con_code": ["600000.SH"], "trade_date": ["20240115"]}).write_parquet(
            prior_cache
        )

        mp.setattr("factorzen.core.loader.init_tushare", _fake_pro)
        # 当月(2月)拉取"成功"，但唯一一条记录的 trade_date(2月20日)晚于查询日(2月5日)，
        # _members_as_of 对这份数据会返回空列表——不该被当成"该指数当月无成分股"处理。
        mp.setattr(
            "factorzen.core.loader._retry",
            lambda fn, **kw: _index_weight_df(["999999.SZ"], ["20240220"]),
        )

        result = U._load_index_members("000300.SH", "20240205")

        assert result == ["600000.SH"], (
            f"当月数据无 trade_date<=查询日 的记录时应回退到上月缓存，实际: {result}"
        )

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_load_index_members_falls_back_when_current_month_has_no_eligible_snapshot(mp, _tp2)


# ══════════════════════════════════════════════════════════
# _load_index_members 按 trade_date 精确截取（回归：曾经按整月并集返回，
# 在调样生效日（6月/12月中旬）前的查询会提前看到尚未生效的新成分，即未来函数）
# ══════════════════════════════════════════════════════════


def _index_weight_df(con_codes: list[str], trade_dates: list[str]) -> pd.DataFrame:
    """构造含 trade_date 的 index_weight 原始返回（同一个月内可含多个调样快照）。"""
    return pd.DataFrame({"con_code": con_codes, "trade_date": trade_dates})


def test_load_index_members_date_filter_suite(tmp_path):
    """同一个月内：前半月成分 {A,B}，6/17 调样后下半月成分 {A,C}。；同一份月度原始数据内，调样生效日及之后的查询应看到新成分集合。；缓存命中路径同样需要按 trade_date 精确截取，不能整月并集返回。"""
    # -- 原 test_load_index_members_filters_by_exact_trade_date_not_whole_month --
    def _section_0_test_load_index_members_filters_by_exact_trade_date_not_whole_month(mp, tmp_path):
        mp.setattr(U, "DATA_CACHE", tmp_path)
        U._INDEX_MEMBER_MEMORY_CACHE.clear()
        mp.setattr("factorzen.core.loader.init_tushare", _fake_pro)
        df_pd = _index_weight_df(
            ["A", "B", "A", "C"],
            ["20240601", "20240601", "20240617", "20240617"],
        )
        mp.setattr("factorzen.core.loader._retry", lambda fn, **kw: df_pd)

        result = U._load_index_members("000300.SH", "20240610")

        assert set(result) == {"A", "B"}, f"调样生效日(6/17)前不应看到新成分 C，实际={result}"

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_load_index_members_filters_by_exact_trade_date_not_whole_month(mp, _tp0)

    # -- 原 test_load_index_members_reflects_resample_after_effective_date --
    def _section_1_test_load_index_members_reflects_resample_after_effective_date(mp, tmp_path):
        mp.setattr(U, "DATA_CACHE", tmp_path)
        U._INDEX_MEMBER_MEMORY_CACHE.clear()
        mp.setattr("factorzen.core.loader.init_tushare", _fake_pro)
        df_pd = _index_weight_df(
            ["A", "B", "A", "C"],
            ["20240601", "20240601", "20240617", "20240617"],
        )
        mp.setattr("factorzen.core.loader._retry", lambda fn, **kw: df_pd)

        before = U._load_index_members("000300.SH", "20240610")
        after = U._load_index_members("000300.SH", "20240620")

        assert set(before) == {"A", "B"}
        assert set(after) == {"A", "C"}

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_load_index_members_reflects_resample_after_effective_date(mp, _tp1)

    # -- 原 test_load_index_members_cache_hit_filters_by_trade_date --
    def _section_2_test_load_index_members_cache_hit_filters_by_trade_date(mp, tmp_path):
        mp.setattr(U, "DATA_CACHE", tmp_path)
        U._INDEX_MEMBER_MEMORY_CACHE.clear()
        cache_file = tmp_path / "index_member_000300_SH_202406.parquet"
        pl.DataFrame(
            {
                "con_code": ["A", "B", "A", "C"],
                "trade_date": ["20240601", "20240601", "20240617", "20240617"],
            }
        ).write_parquet(cache_file)

        def _should_not_call():
            raise AssertionError("缓存命中时不应调用 init_tushare")

        mp.setattr("factorzen.core.loader.init_tushare", _should_not_call)

        result = U._load_index_members("000300.SH", "20240610")

        assert set(result) == {"A", "B"}

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_load_index_members_cache_hit_filters_by_trade_date(mp, _tp2)


# ══════════════════════════════════════════════════════════
# get_index_members
# ══════════════════════════════════════════════════════════


def test_get_index_members_suite(stock_basic):
    """test_get_index_members_joins_stock_info；无成分股时降级为全市场。；test_get_index_members_exception_falls_back_to_all_market"""
    # -- 原 test_get_index_members_joins_stock_info --
    def _section_0_test_get_index_members_joins_stock_info(stock_basic, mp):
        mp.setattr(U, "_load_index_members", lambda code, ds: ["600000.SH"])
        result = get_index_members("000300.SH", "20240115")
        assert result["ts_code"].to_list() == ["600000.SH"]
        assert "industry" in result.columns

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_get_index_members_joins_stock_info(stock_basic, mp)

    # -- 原 test_get_index_members_empty_falls_back_to_all_market --
    def _section_1_test_get_index_members_empty_falls_back_to_all_market(stock_basic, mp):
        mp.setattr(U, "_load_index_members", lambda code, ds: [])
        result = get_index_members("000300.SH", "20240115")
        assert result.height == 4

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_get_index_members_empty_falls_back_to_all_market(stock_basic, mp)

    # -- 原 test_get_index_members_exception_falls_back_to_all_market --
    def _section_2_test_get_index_members_exception_falls_back_to_all_market(stock_basic, mp):
        def _boom(code, ds):
            raise RuntimeError("api error")

        mp.setattr(U, "_load_index_members", _boom)
        result = get_index_members("000300.SH", "20240115")
        assert result.height == 4

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_get_index_members_exception_falls_back_to_all_market(stock_basic, mp)



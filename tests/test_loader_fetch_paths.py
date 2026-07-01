"""loader.py 拉取路径离线单测（补 test_loader.py 未覆盖的 fetch_* 主体）。

全量 mock：通过 patch init_tushare / partition_exists / save_parquet / load_parquet /
fetch_trade_cal，覆盖各 fetch 函数的「拉取→pandas转polars→合并→保存→回读」主路径，
及无数据 / 异常 / 缓存降级分支。不调用真实 Tushare，不读写 data/。
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl

import factorzen.core.loader as loader_module
from factorzen.core.loader import (
    _rate_limit,
    _str_to_date,
    fetch_adj_factor,
    fetch_daily,
    fetch_daily_basic,
    fetch_finance,
    fetch_index_daily,
    fetch_index_member_all,
    fetch_stock_basic,
    fetch_trade_cal,
)

# ── 合成 pandas 输出 ────────────────────────────────────────


def _pd_ohlc(trade_date: str = "20220103", code: str = "000001.SZ") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [trade_date],
            "ts_code": [code],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "pre_close": [10.0],
            "change": [0.5],
            "pct_chg": [5.0],
            "vol": [1000.0],
            "amount": [10000.0],
        }
    )


def _pd_basic(trade_date: str = "20220103", code: str = "000001.SZ") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [trade_date],
            "ts_code": [code],
            "pe": [15.0],
            "pb": [1.5],
            "total_mv": [1e9],
        }
    )


def _pd_adj(trade_date: str = "20220103", code: str = "000001.SZ") -> pd.DataFrame:
    return pd.DataFrame(
        {"trade_date": [trade_date], "ts_code": [code], "adj_factor": [1.23]}
    )


def _cal(dates: list[str]) -> pl.DataFrame:
    """构造交易日历 DataFrame（cal_date 为 Date，is_open=1）。"""
    return pl.DataFrame(
        {
            "cal_date": [datetime.strptime(d, "%Y%m%d").date() for d in dates],
            "is_open": [1] * len(dates),
        }
    )


def _lf(df: pl.DataFrame) -> MagicMock:
    """load_parquet 的返回桩：.collect() 返回给定 DataFrame。"""
    m = MagicMock()
    m.collect.return_value = df
    return m


# ══════════════════════════════════════════════════════════
# 基础设施：_rate_limit / _str_to_date
# ══════════════════════════════════════════════════════════


def test_rate_limit_sleeps_when_called_too_soon(monkeypatch):
    """距上次调用不足 min_interval 时应 sleep 补足。"""
    monkeypatch.setattr(loader_module, "_last_call", time.time())
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    _rate_limit()
    assert slept and slept[0] > 0


def test_str_to_date_parses_yyyymmdd():
    df = pl.DataFrame({"d": ["20240115"]}).with_columns(_str_to_date(pl.col("d")))
    assert df["d"].item() == date(2024, 1, 15)


def test_str_to_date_invalid_becomes_null():
    df = pl.DataFrame({"d": ["not-a-date"]}).with_columns(_str_to_date(pl.col("d")))
    assert df["d"].item() is None


# ══════════════════════════════════════════════════════════
# fetch_trade_cal
# ══════════════════════════════════════════════════════════


def test_fetch_trade_cal_normal():
    mock_pro = MagicMock()
    mock_pro.trade_cal.return_value = pd.DataFrame(
        {"cal_date": ["20240102", "20240103"], "is_open": [1, 1]}
    )
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
    ):
        result = fetch_trade_cal("20240101", "20240131")
    assert result.height == 2
    assert result["cal_date"].dtype == pl.Date


def test_fetch_trade_cal_empty_returns_empty():
    """_retry 返回空时（patch 绕过重试），返回空 DataFrame。"""
    mock_pro = MagicMock()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_retry", return_value=pd.DataFrame()),
    ):
        result = fetch_trade_cal("20240101", "20240131")
    assert result.is_empty()


def test_fetch_trade_cal_exception_reraises():
    mock_pro = MagicMock()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_retry", side_effect=RuntimeError("net down")),
    ):
        try:
            fetch_trade_cal("20240101", "20240131")
        except RuntimeError as e:
            assert "net down" in str(e)
        else:
            raise AssertionError("应抛出 RuntimeError")


# ══════════════════════════════════════════════════════════
# fetch_daily：逐股模式 / 全市场模式 / 无数据
# ══════════════════════════════════════════════════════════


def test_fetch_daily_ts_codes_mode_saves_and_returns():
    mock_pro = MagicMock()
    mock_pro.daily.return_value = _pd_ohlc()
    expected = pl.DataFrame({"ts_code": ["000001.SZ"]})
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "partition_exists", return_value=False),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(expected)),
    ):
        result = fetch_daily("20220101", "20220131", ts_codes=["000001.SZ"])
    save.assert_called_once()
    assert result.equals(expected)


def test_fetch_daily_market_mode_iterates_trade_dates():
    mock_pro = MagicMock()
    mock_pro.daily.return_value = _pd_ohlc()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "partition_exists", return_value=False),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "fetch_trade_cal", return_value=_cal(["20220104", "20220105"])),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_daily("20220101", "20220131")
    # 每个交易日各一次 pro.daily
    assert mock_pro.daily.call_count == 2
    save.assert_called_once()


def test_fetch_daily_ts_codes_per_stock_error_continues():
    """逐股模式：某只股票拉取异常应跳过并继续其余股票。"""
    mock_pro = MagicMock()

    def _daily(ts_code, start_date, end_date):
        if ts_code == "BAD.SZ":
            raise RuntimeError("single stock error")
        return _pd_ohlc(code=ts_code)

    mock_pro.daily.side_effect = _daily
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "partition_exists", return_value=False),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_daily("20220101", "20220131", ts_codes=["BAD.SZ", "000002.SZ"])
    # BAD.SZ 失败被跳过，000002.SZ 成功 → 仍保存
    save.assert_called_once()


def test_fetch_daily_market_mode_no_data_skips_save():
    """全市场模式下当年无数据，不保存。"""
    mock_pro = MagicMock()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "partition_exists", return_value=False),
        patch.object(loader_module, "_retry", side_effect=RuntimeError("空")),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "fetch_trade_cal", return_value=_cal(["20220104"])),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_daily("20220101", "20220131")
    save.assert_not_called()


# ══════════════════════════════════════════════════════════
# fetch_daily_basic：逐股 / 全市场
# ══════════════════════════════════════════════════════════


def test_fetch_daily_basic_ts_codes_mode():
    mock_pro = MagicMock()
    mock_pro.daily_basic.return_value = _pd_basic()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "partition_exists", return_value=False),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_daily_basic("20220101", "20220131", ts_codes=["000001.SZ"])
    save.assert_called_once()


def test_fetch_daily_basic_market_mode():
    mock_pro = MagicMock()
    mock_pro.daily_basic.return_value = _pd_basic()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "partition_exists", return_value=False),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "fetch_trade_cal", return_value=_cal(["20220104"])),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_daily_basic("20220101", "20220131")
    save.assert_called_once()


# ══════════════════════════════════════════════════════════
# fetch_adj_factor
# ══════════════════════════════════════════════════════════


def test_fetch_adj_factor_normal():
    mock_pro = MagicMock()
    mock_pro.adj_factor.return_value = _pd_adj()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "partition_exists", return_value=False),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "fetch_trade_cal", return_value=_cal(["20220104"])),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_adj_factor("20220101", "20220131")
    save.assert_called_once()


def test_fetch_adj_factor_no_data_skips_save():
    mock_pro = MagicMock()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "partition_exists", return_value=False),
        patch.object(loader_module, "_retry", side_effect=RuntimeError("空")),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "fetch_trade_cal", return_value=_cal(["20220104"])),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_adj_factor("20220101", "20220131")
    save.assert_not_called()


# ══════════════════════════════════════════════════════════
# fetch_index_daily：正常 / 无数据 / 异常 / 已缓存
# ══════════════════════════════════════════════════════════


def test_fetch_index_daily_normal():
    mock_pro = MagicMock()
    mock_pro.index_daily.return_value = _pd_ohlc(code="000300.SH")
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "partition_exists", return_value=False),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_index_daily("000300.SH", "20220101", "20220131")
    save.assert_called_once()


def test_fetch_index_daily_exception_skips_save():
    mock_pro = MagicMock()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "partition_exists", return_value=False),
        patch.object(loader_module, "_retry", side_effect=RuntimeError("api error")),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_index_daily("000300.SH", "20220101", "20220131")
    save.assert_not_called()


def test_fetch_index_daily_cached_skips_api():
    mock_pro = MagicMock()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "partition_exists", return_value=True),
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_index_daily("000300.SH", "20220101", "20220131")
    mock_pro.index_daily.assert_not_called()


# ══════════════════════════════════════════════════════════
# fetch_finance：合并/类型对齐路径 + 全市场模式
# ══════════════════════════════════════════════════════════


def test_fetch_finance_merges_and_casts():
    """财报多批返回含 String 数值列，应统一 cast 到 Float64 后合并去重。"""
    df_pd = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["20220430"],
            "end_date": ["20220331"],
            "roe": ["12.5"],  # String 数值 → 应被 cast
        }
    )
    mock_pro = MagicMock()
    mock_pro.fina_indicator.return_value = df_pd
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "partition_exists", return_value=False),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_finance("fina_indicator", "20220101", "20220331", ts_codes=["000001.SZ"])
    saved_df = save.call_args.args[0]
    assert saved_df["roe"].dtype == pl.Float64
    assert saved_df["end_date"].dtype == pl.Date


def test_fetch_finance_market_mode_uses_stock_basic():
    """ts_codes=None 时应调用 fetch_stock_basic 获取全市场代码。"""
    mock_pro = MagicMock()
    mock_pro.income.return_value = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "ann_date": ["20220430"], "end_date": ["20220331"]}
    )
    basic = pl.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"]})
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "partition_exists", return_value=False),
        patch.object(loader_module, "fetch_stock_basic", return_value=basic) as fsb,
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "save_parquet"),
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_finance("income", "20220101", "20220331")
    fsb.assert_called_once()


# ══════════════════════════════════════════════════════════
# fetch_stock_basic：无数据降级分支
# ══════════════════════════════════════════════════════════


def test_fetch_stock_basic_no_data_no_cache_returns_empty(tmp_path: Path):
    """所有 status 拉取失败且无缓存 → 返回空 DataFrame。"""
    mock_pro = MagicMock()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "DATA_CACHE", tmp_path),
        patch.object(loader_module, "_retry", side_effect=RuntimeError("空")),
    ):
        result = fetch_stock_basic(list_status="L")
    assert result.is_empty()


def test_fetch_stock_basic_no_data_falls_back_to_stale_cache(tmp_path: Path):
    """拉取全失败但有（过期）缓存 → 回退读取缓存。"""
    cache_file = tmp_path / "stock_basic_L.parquet"
    pl.DataFrame({"ts_code": ["000001.SZ"], "name": ["旧"]}).write_parquet(cache_file)
    stale = time.time() - 30 * 86400
    os.utime(cache_file, (stale, stale))

    mock_pro = MagicMock()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "DATA_CACHE", tmp_path),
        patch.object(loader_module, "_retry", side_effect=RuntimeError("空")),
    ):
        result = fetch_stock_basic(list_status="L")
    assert result["ts_code"].to_list() == ["000001.SZ"]


def test_fetch_stock_basic_merges_multiple_statuses(tmp_path: Path):
    """L,D 两个 status 分别拉取再按 ts_code 去重合并。"""
    mock_pro = MagicMock()

    def _basic(list_status, fields):
        if list_status == "L":
            return _pd_stock_basic_status("000001.SZ")
        return _pd_stock_basic_status("000002.SZ")

    mock_pro.stock_basic.side_effect = _basic
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "DATA_CACHE", tmp_path),
        patch.object(loader_module, "_rate_limit"),
    ):
        result = fetch_stock_basic(list_status="L,D")
    assert set(result["ts_code"].to_list()) == {"000001.SZ", "000002.SZ"}


def _pd_stock_basic_status(code: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [code],
            "symbol": [code[:6]],
            "name": ["x"],
            "area": ["广东"],
            "industry": ["银行"],
            "market": ["主板"],
            "list_date": ["19910101"],
            "delist_date": [None],
        }
    )


# ══════════════════════════════════════════════════════════
# fetch_index_member_all：PIT 申万一级行业历史成分（循环 l1_code 拉全市场）
#
# 真实字段名已对项目 .env 中的 TUSHARE_TOKEN 实打 index_member_all 接口确认
# （非凭空猜测）：l1_code/l1_name/l2_code/l2_name/l3_code/l3_name/ts_code/name/
# in_date/out_date/is_new。同时确认该接口不带过滤条件时单次调用截断在 3000 行
# （全市场远超 3000 只股票的成分历史），所以必须按 l1_code 循环拉取才能覆盖全市场，
# 不能直接裸调用。
# ══════════════════════════════════════════════════════════


def _pd_l1_classify() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index_code": ["801780.SI", "801150.SI"],
            "industry_name": ["银行", "医药生物"],
            "level": ["L1", "L1"],
            "industry_code": ["480000", "370000"],
            "is_pub": ["1", "1"],
            "parent_code": ["0", "0"],
            "src": ["SW2021", "SW2021"],
        }
    )


def _pd_member_all(l1_code: str, l1_name: str, ts_code: str, name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "l1_code": [l1_code],
            "l1_name": [l1_name],
            "l2_code": ["xxxxxx.SI"],
            "l2_name": ["二级行业"],
            "l3_code": ["yyyyyy.SI"],
            "l3_name": ["三级行业"],
            "ts_code": [ts_code],
            "name": [name],
            "in_date": ["19910403"],
            "out_date": [None],
            "is_new": ["Y"],
        }
    )


def test_fetch_index_member_all_cache_hit_skips_api(tmp_path: Path):
    """缓存新鲜时直接读取，不调用 Tushare（index_classify / index_member_all 均不调用）。"""
    loader_module._INDEX_MEMBER_ALL_MEMORY_CACHE.clear()
    cache_file = tmp_path / "index_member_all.parquet"
    pl.DataFrame({"ts_code": ["000001.SZ"], "l1_name": ["银行"]}).write_parquet(cache_file)

    mock_pro = MagicMock()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "DATA_CACHE", tmp_path),
    ):
        result = fetch_index_member_all()

    mock_pro.index_classify.assert_not_called()
    mock_pro.index_member_all.assert_not_called()
    assert result is not None
    assert result["ts_code"].to_list() == ["000001.SZ"]


def test_fetch_index_member_all_loops_l1_codes_and_caches(tmp_path: Path):
    """无缓存：先枚举一级行业(index_classify)，再逐个拉取成分(index_member_all)，
    合并、转换日期列、写入缓存。"""
    loader_module._INDEX_MEMBER_ALL_MEMORY_CACHE.clear()
    mock_pro = MagicMock()
    mock_pro.index_classify.return_value = _pd_l1_classify()

    def _member(l1_code, fields):
        if l1_code == "801780.SI":
            return _pd_member_all("801780.SI", "银行", "000001.SZ", "平安银行")
        return _pd_member_all("801150.SI", "医药生物", "600196.SH", "复星医药")

    mock_pro.index_member_all.side_effect = _member

    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "DATA_CACHE", tmp_path),
        patch.object(loader_module, "_rate_limit"),
    ):
        result = fetch_index_member_all()

    assert mock_pro.index_member_all.call_count == 2
    assert result is not None
    assert set(result["ts_code"].to_list()) == {"000001.SZ", "600196.SH"}
    assert result["in_date"].dtype == pl.Date
    assert result["out_date"].dtype == pl.Date
    assert (tmp_path / "index_member_all.parquet").exists()


def test_fetch_index_member_all_failure_no_cache_returns_none(tmp_path: Path):
    """无权限/网络失败且无缓存：优雅降级返回 None，不抛异常（不卡住调用方）。"""
    loader_module._INDEX_MEMBER_ALL_MEMORY_CACHE.clear()
    mock_pro = MagicMock()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "DATA_CACHE", tmp_path),
        patch.object(
            loader_module, "_retry", side_effect=RuntimeError("抱歉，您没有访问该接口的权限")
        ),
    ):
        result = fetch_index_member_all()

    assert result is None


def test_fetch_index_member_all_failure_falls_back_to_stale_cache(tmp_path: Path):
    """拉取失败但存在（过期）缓存：回退读取缓存而非返回 None。"""
    loader_module._INDEX_MEMBER_ALL_MEMORY_CACHE.clear()
    cache_file = tmp_path / "index_member_all.parquet"
    pl.DataFrame({"ts_code": ["000001.SZ"], "l1_name": ["银行"]}).write_parquet(cache_file)
    stale = time.time() - 30 * 86400
    os.utime(cache_file, (stale, stale))

    mock_pro = MagicMock()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "DATA_CACHE", tmp_path),
        patch.object(loader_module, "_retry", side_effect=RuntimeError("网络错误")),
    ):
        result = fetch_index_member_all()

    assert result is not None
    assert result["ts_code"].to_list() == ["000001.SZ"]


def test_fetch_index_member_all_reuses_in_process_cache_across_calls(tmp_path: Path):
    """同一进程内重复调用应只真正拉取一次（index_classify/index_member_all
    均只调一次），避免 RiskModel.build() 对长窗口每个交易日都重新从磁盘/网络
    读取同一份全市场行业成分表。"""
    loader_module._INDEX_MEMBER_ALL_MEMORY_CACHE.clear()
    mock_pro = MagicMock()
    mock_pro.index_classify.return_value = _pd_l1_classify()
    mock_pro.index_member_all.side_effect = lambda l1_code, fields: _pd_member_all(
        l1_code, "行业", "000001.SZ", "平安银行"
    )

    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "DATA_CACHE", tmp_path),
        patch.object(loader_module, "_rate_limit"),
    ):
        first = fetch_index_member_all()
        second = fetch_index_member_all()

    assert mock_pro.index_classify.call_count == 1, "第二次调用不应再拉取 index_classify"
    assert first is not None
    assert second is first, "第二次调用应直接返回进程内缓存的同一个对象，而非重新拉取"


def test_fetch_index_member_all_no_token_returns_none_fast(tmp_path: Path):
    """无 TUSHARE_TOKEN（离线 CI 场景）：init_tushare 内 ensure_token 抛错，
    应被优雅捕获并立即返回 None，而不是抛异常或卡住。"""
    loader_module._INDEX_MEMBER_ALL_MEMORY_CACHE.clear()
    with (
        patch.object(loader_module, "DATA_CACHE", tmp_path),
        patch.object(
            loader_module, "init_tushare", side_effect=RuntimeError("请设置 TUSHARE_TOKEN 环境变量")
        ),
    ):
        result = fetch_index_member_all()

    assert result is None

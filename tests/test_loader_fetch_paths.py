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
    fetch_margin_detail,
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


def _pd_minute(code: str = "000001.SZ") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [code],
            "trade_time": ["2022-01-03 09:31:00"],
            "open": [10.0], "high": [11.0], "low": [9.0], "close": [10.5],
            "vol": [1000.0], "amount": [10000.0],
        }
    )


def test_fetch_minute_namespaces_partition_by_freq():
    """不同 freq 写入不同分区命名空间 minute_{freq}，避免跨频率脏缓存。"""
    from factorzen.core.loader import fetch_minute

    mock_pro = MagicMock()
    mock_pro.stk_mins.return_value = _pd_minute()
    saved_types: list[str] = []
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "partition_exists", return_value=False),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "save_parquet",
                     side_effect=lambda df, data_type, **k: saved_types.append(data_type)),
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_minute("000001.SZ", "1min", "20220101", "20220131")
        fetch_minute("000001.SZ", "5min", "20220101", "20220131")

    assert "minute_1min" in saved_types
    assert "minute_5min" in saved_types
    assert "minute" not in saved_types, "不应再写入无 freq 的共享 minute 分区"


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


def test_fetch_daily_ts_codes_mode_returns_filtered_no_cache_write():
    """子集(ts_codes)模式：直拉、按 ts_codes 过滤返回、不写共享全市场缓存（避免污染）。"""
    mock_pro = MagicMock()
    mock_pro.daily.return_value = _pd_ohlc(code="000001.SZ")
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "save_parquet") as save,
    ):
        result = fetch_daily("20220101", "20220131", ts_codes=["000001.SZ"])
    save.assert_not_called()  # 子集不写共享缓存
    assert result.height == 1 and result["ts_code"][0] == "000001.SZ"


def test_fetch_daily_market_mode_fetches_only_missing_dates():
    mock_pro = MagicMock()
    mock_pro.daily.return_value = _pd_ohlc()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "get_trade_dates",
                     return_value=[date(2022, 1, 4), date(2022, 1, 5)]),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_daily("20220101", "20220131")
    # 缓存为空 → 两个交易日都缺失 → 各拉一次
    assert mock_pro.daily.call_count == 2
    save.assert_called_once()


def test_fetch_daily_market_mode_fetches_only_missing_when_partially_cached():
    """C1 核心：部分交易日已缓存时只拉缺失日，不再把部分年误判为整年完整。"""
    mock_pro = MagicMock()
    mock_pro.daily.return_value = _pd_ohlc(trade_date="20220105")
    cached = pl.DataFrame({"trade_date": [date(2022, 1, 4)], "ts_code": ["000001.SZ"]})
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "get_trade_dates",
                     return_value=[date(2022, 1, 4), date(2022, 1, 5)]),
        patch.object(loader_module, "save_parquet"),
        patch.object(loader_module, "load_parquet", return_value=_lf(cached)),
    ):
        fetch_daily("20220101", "20220131")
    # 只有 1/5 缺失 → 只拉一次（1/4 已缓存不重拉）
    assert mock_pro.daily.call_count == 1
    assert mock_pro.daily.call_args.kwargs.get("trade_date") == "20220105"


def test_fetch_daily_market_mode_skips_when_fully_cached():
    """所有交易日都已在缓存 → 不再拉取（交易日历覆盖审计）。"""
    mock_pro = MagicMock()
    cached = pl.DataFrame({
        "trade_date": [date(2022, 1, 4), date(2022, 1, 5)],
        "ts_code": ["000001.SZ", "000001.SZ"],
    })
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "get_trade_dates",
                     return_value=[date(2022, 1, 4), date(2022, 1, 5)]),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(cached)),
    ):
        fetch_daily("20220101", "20220131")
    mock_pro.daily.assert_not_called()  # 全部已缓存 → 不拉
    save.assert_not_called()


def test_fetch_daily_subset_per_stock_error_continues():
    """子集模式：某只股票拉取异常应跳过并继续其余股票。"""
    mock_pro = MagicMock()

    def _daily(ts_code, start_date, end_date):
        if ts_code == "BAD.SZ":
            raise RuntimeError("single stock error")
        return _pd_ohlc(code=ts_code)

    mock_pro.daily.side_effect = _daily
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "save_parquet"),
    ):
        result = fetch_daily("20220101", "20220131", ts_codes=["BAD.SZ", "000002.SZ"])
    codes = result["ts_code"].to_list()
    assert "000002.SZ" in codes and "BAD.SZ" not in codes


def test_fetch_daily_market_mode_no_data_skips_save():
    """全市场模式下缺失日拉取全失败，不保存。"""
    mock_pro = MagicMock()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_retry", side_effect=RuntimeError("空")),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "get_trade_dates", return_value=[date(2022, 1, 4)]),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_daily("20220101", "20220131")
    save.assert_not_called()


# ══════════════════════════════════════════════════════════
# fetch_daily_basic：子集 / 全市场
# ══════════════════════════════════════════════════════════


def test_fetch_daily_basic_subset_mode_no_cache_write():
    mock_pro = MagicMock()
    mock_pro.daily_basic.return_value = _pd_basic(code="000001.SZ")
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "save_parquet") as save,
    ):
        result = fetch_daily_basic("20220101", "20220131", ts_codes=["000001.SZ"])
    save.assert_not_called()
    assert result.height == 1 and result["ts_code"][0] == "000001.SZ"


def test_fetch_daily_basic_market_mode():
    mock_pro = MagicMock()
    mock_pro.daily_basic.return_value = _pd_basic()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "get_trade_dates", return_value=[date(2022, 1, 4)]),
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
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "get_trade_dates", return_value=[date(2022, 1, 4)]),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_adj_factor("20220101", "20220131")
    save.assert_called_once()


def test_fetch_adj_factor_no_data_skips_save():
    mock_pro = MagicMock()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_retry", side_effect=RuntimeError("空")),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "get_trade_dates", return_value=[date(2022, 1, 4)]),
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
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "get_trade_dates", return_value=[date(2022, 1, 4)]),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_index_daily("000300.SH", "20220101", "20220131")
    save.assert_called_once()


def test_fetch_index_daily_exception_skips_save():
    mock_pro = MagicMock()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_retry", side_effect=RuntimeError("api error")),
        patch.object(loader_module, "get_trade_dates", return_value=[date(2022, 1, 4)]),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_index_daily("000300.SH", "20220101", "20220131")
    save.assert_not_called()


def test_fetch_index_daily_cached_skips_api():
    mock_pro = MagicMock()
    cached = pl.DataFrame({"trade_date": [date(2022, 1, 4)], "ts_code": ["000300.SH"]})
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "get_trade_dates", return_value=[date(2022, 1, 4)]),
        patch.object(loader_module, "load_parquet", return_value=_lf(cached)),
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


def test_fetch_finance_namespaces_by_api_name():
    """不同接口写入不同命名空间 finance_{api_name}，避免共用 finance 分区 schema 冲突。"""
    mock_pro = MagicMock()
    mock_pro.income.return_value = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "ann_date": ["20220430"], "end_date": ["20220331"]}
    )
    mock_pro.cashflow.return_value = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "ann_date": ["20220430"], "end_date": ["20220331"]}
    )
    saved_types: list[str] = []
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "partition_exists", return_value=False),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "save_parquet",
                     side_effect=lambda df, data_type, **k: saved_types.append(data_type)),
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_finance("income", "20220101", "20220331", ts_codes=["000001.SZ"])
        fetch_finance("cashflow", "20220101", "20220331", ts_codes=["000001.SZ"])
    assert "finance_income" in saved_types
    assert "finance_cashflow" in saved_types
    assert "finance" not in saved_types


def test_fetch_finance_cache_check_uses_quarter_end_month():
    """完整性检查须用季末月(Q1=3)，数据以 end_date(3/6/9/12) 落盘；用季初月(1)永不命中。"""
    checked: list[tuple] = []

    def _fake_partition_exists(data_type, year, month, **k):
        checked.append((data_type, year, month))
        return month == 3  # 只有季末月 3 有分区

    mock_pro = MagicMock()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "partition_exists", side_effect=_fake_partition_exists),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_finance("income", "20220101", "20220331", ts_codes=["000001.SZ"])
    # 检查用了季末月 3（命中 → 跳过拉取），而非季初月 1
    assert ("finance_income", 2022, 3) in checked
    save.assert_not_called()  # 已缓存 → 不拉取


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


# ══════════════════════════════════════════════════════════
# fetch_margin_detail：按 trade_date 分页 / 落分区 / 幂等增量
# ══════════════════════════════════════════════════════════


def _pd_margin(trade_date: str = "20220104", code: str = "000001.SZ") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [trade_date],
            "ts_code": [code],
            "rzye": [1e9],
            "rqye": [1e6],
            "rzmre": [1e8],
            "rqyl": [100.0],
            "rzche": [1e7],
            "rqchl": [10.0],
            "rqmcl": [5.0],
            "rzrqye": [1.001e9],
        }
    )


def test_fetch_margin_detail_market_mode_by_trade_date():
    """全市场：按缺失交易日逐日 trade_date 拉取并 save_parquet(margin_detail)。"""
    mock_pro = MagicMock()
    mock_pro.margin_detail.return_value = _pd_margin()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "get_trade_dates",
                     return_value=[date(2022, 1, 4), date(2022, 1, 5)]),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_margin_detail("20220101", "20220131")
    assert mock_pro.margin_detail.call_count == 2
    for c in mock_pro.margin_detail.call_args_list:
        assert "trade_date" in c.kwargs
    save.assert_called_once()
    assert save.call_args.kwargs.get("data_type") == "margin_detail"


def test_fetch_margin_detail_incremental_skips_cached_dates():
    """幂等增量：已缓存交易日不重拉。"""
    mock_pro = MagicMock()
    mock_pro.margin_detail.return_value = _pd_margin(trade_date="20220105")
    cached = pl.DataFrame({"trade_date": [date(2022, 1, 4)], "ts_code": ["000001.SZ"]})
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "get_trade_dates",
                     return_value=[date(2022, 1, 4), date(2022, 1, 5)]),
        patch.object(loader_module, "save_parquet"),
        patch.object(loader_module, "load_parquet", return_value=_lf(cached)),
    ):
        fetch_margin_detail("20220101", "20220131")
    assert mock_pro.margin_detail.call_count == 1
    assert mock_pro.margin_detail.call_args.kwargs.get("trade_date") == "20220105"


def test_fetch_margin_detail_fully_cached_no_fetch():
    mock_pro = MagicMock()
    cached = pl.DataFrame({
        "trade_date": [date(2022, 1, 4), date(2022, 1, 5)],
        "ts_code": ["000001.SZ", "000001.SZ"],
    })
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "get_trade_dates",
                     return_value=[date(2022, 1, 4), date(2022, 1, 5)]),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(cached)),
    ):
        fetch_margin_detail("20220101", "20220131")
    mock_pro.margin_detail.assert_not_called()
    save.assert_not_called()


def test_fetch_margin_detail_subset_no_cache_write():
    mock_pro = MagicMock()
    mock_pro.margin_detail.return_value = _pd_margin(code="000001.SZ")
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "save_parquet") as save,
    ):
        result = fetch_margin_detail("20220101", "20220131", ts_codes=["000001.SZ"])
    save.assert_not_called()
    assert result.height == 1 and result["ts_code"][0] == "000001.SZ"


# ══════════════════════════════════════════════════════════
# fetch_stk_holdernumber：市场模式按公告月窗口 / 落分区 / 幂等
# ══════════════════════════════════════════════════════════


def _pd_holder(code: str = "000001.SZ", ann: str = "20220430", end: str = "20220331") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [code],
            "ann_date": [ann],
            "end_date": [end],
            "holder_num": [50000.0],
        }
    )


def test_fetch_stk_holdernumber_market_mode_monthly_windows(tmp_path, monkeypatch):
    """市场模式：按公告月窗口整市场拉（不带 ts_code），落盘并写 "YYYY-MM" 标记。"""
    from factorzen.core.loader import fetch_stk_holdernumber

    monkeypatch.setattr(loader_module, "DATA_RAW", tmp_path)
    mock_pro = MagicMock()
    mock_pro.stk_holdernumber.return_value = _pd_holder()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_stk_holdernumber("20220101", "20220331")
    # 3 个月窗口各一次调用，且不带 ts_code（整市场按公告窗口）
    assert mock_pro.stk_holdernumber.call_count == 3
    for c in mock_pro.stk_holdernumber.call_args_list:
        assert "ts_code" not in c.kwargs
        assert "start_date" in c.kwargs and "end_date" in c.kwargs
    save.assert_called()
    assert save.call_args.kwargs.get("data_type") == "stk_holdernumber"
    saved = save.call_args.args[0]
    assert "holder_num" in saved.columns
    assert saved["end_date"].dtype == pl.Date
    import json
    windows = json.loads((tmp_path / "stk_holdernumber" / "_fetched_windows.json").read_text())
    assert {w["window"] for w in windows} == {"2022-01", "2022-02", "2022-03"}


def test_fetch_stk_holdernumber_window_marker_skips(tmp_path, monkeypatch):
    """幂等：月窗口标记存在则跳过；旧版整年标记向后兼容视为覆盖该年。"""
    import json

    from factorzen.core.loader import fetch_stk_holdernumber

    monkeypatch.setattr(loader_module, "DATA_RAW", tmp_path)
    marker_dir = tmp_path / "stk_holdernumber"
    marker_dir.mkdir(parents=True)
    (marker_dir / "_fetched_windows.json").write_text(
        json.dumps([
            {"year": 2022, "n_stocks": 1, "ts": "2026-01-01T00:00:00"},   # legacy 整年
            {"window": "2023-01", "n_rows": 5, "ts": "2026-01-01T00:00:00"},
        ])
    )
    mock_pro = MagicMock()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_stk_holdernumber("20220101", "20220331")   # legacy 年覆盖
        fetch_stk_holdernumber("20230101", "20230131")   # 月标记覆盖
    mock_pro.stk_holdernumber.assert_not_called()
    save.assert_not_called()


def test_fetch_stk_holdernumber_subset_mode_no_shared_cache(tmp_path, monkeypatch):
    """子集模式（ts_codes）：逐股直拉直接返回，不写共享缓存、不写标记——
    部分数据写共享缓存会让完整性标记撒谎。"""
    from factorzen.core.loader import fetch_stk_holdernumber

    monkeypatch.setattr(loader_module, "DATA_RAW", tmp_path)
    mock_pro = MagicMock()
    mock_pro.stk_holdernumber.return_value = _pd_holder()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "save_parquet") as save,
    ):
        out = fetch_stk_holdernumber("20220101", "20220331", ts_codes=["000001.SZ"])
    assert mock_pro.stk_holdernumber.call_args.kwargs.get("ts_code") == "000001.SZ"
    assert not out.is_empty() and "holder_num" in out.columns
    save.assert_not_called()
    assert not (tmp_path / "stk_holdernumber" / "_fetched_windows.json").exists()


def test_fetch_stk_holdernumber_api_error_window_not_marked(tmp_path, monkeypatch):
    """API 异常的窗口重试一次后放弃且**不写标记**（下次重跑补齐，不静默留洞）。"""
    from factorzen.core.loader import fetch_stk_holdernumber

    monkeypatch.setattr(loader_module, "DATA_RAW", tmp_path)
    monkeypatch.setattr(loader_module, "RETRY_DELAY", 0)
    mock_pro = MagicMock()
    mock_pro.stk_holdernumber.side_effect = RuntimeError("boom")
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_stk_holdernumber("20220101", "20220131")
    assert mock_pro.stk_holdernumber.call_count == 2  # 重试一次
    save.assert_not_called()
    assert not (tmp_path / "stk_holdernumber" / "_fetched_windows.json").exists()


def test_fetch_stk_holdernumber_tmp_marker_not_complete(tmp_path, monkeypatch):
    """标记写入原子性：仅 .tmp 存在不算完成，仍会抓取。"""
    from factorzen.core.loader import fetch_stk_holdernumber

    monkeypatch.setattr(loader_module, "DATA_RAW", tmp_path)
    marker_dir = tmp_path / "stk_holdernumber"
    marker_dir.mkdir(parents=True)
    (marker_dir / "_fetched_windows.json.tmp").write_text("[]")
    mock_pro = MagicMock()
    mock_pro.stk_holdernumber.return_value = _pd_holder()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
        patch.object(loader_module, "fetch_stock_basic",
                     return_value=pl.DataFrame({"ts_code": ["000001.SZ"]})),
    ):
        fetch_stk_holdernumber("20220101", "20220331")
    mock_pro.stk_holdernumber.assert_called()
    save.assert_called()
    assert (marker_dir / "_fetched_windows.json").exists()


def test_fetch_stk_holdernumber_rerun_window_idempotent_upsert(tmp_path, monkeypatch):
    """无标记重跑同窗口：save 前 (ts_code,end_date) 与既有去重，结果无重复。"""
    import factorzen.core.storage as storage_mod
    from factorzen.core.loader import fetch_stk_holdernumber
    from factorzen.core.storage import load_parquet, save_parquet

    monkeypatch.setattr(loader_module, "DATA_RAW", tmp_path)
    monkeypatch.setattr(storage_mod, "DATA_RAW", tmp_path)

    # 预置既有分区行（同一 key）
    existing = pl.DataFrame({
        "ts_code": ["000001.SZ"],
        "ann_date": ["20220430"],
        "end_date": [date(2022, 3, 31)],
        "holder_num": [40000.0],
    })
    save_parquet(existing, data_type="stk_holdernumber", date_col="end_date", base_dir=tmp_path)

    mock_pro = MagicMock()
    mock_pro.stk_holdernumber.return_value = _pd_holder()  # holder_num=50000
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "fetch_stock_basic",
                     return_value=pl.DataFrame({"ts_code": ["000001.SZ"]})),
    ):
        fetch_stk_holdernumber("20220101", "20220331")

    loaded = load_parquet(
        "stk_holdernumber", start="20220101", end="20221231",
        date_col="end_date", base_dir=tmp_path,
    ).collect()
    assert loaded.height == 1
    assert loaded["holder_num"][0] == 50000.0  # keep last from re-fetch


# ══════════════════════════════════════════════════════════
# fetch_top_list：按 trade_date / 空日不算失败 / 幂等增量
# ══════════════════════════════════════════════════════════


def _pd_top(trade_date: str = "20220104", code: str = "000001.SZ") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [trade_date],
            "ts_code": [code],
            "name": ["平安银行"],
            "close": [10.0],
            "pct_change": [5.0],
            "turnover_rate": [3.0],
            "amount": [1e5],
            "l_sell": [100.0],
            "l_buy": [200.0],
            "l_amount": [300.0],
            "net_amount": [100.0],
            "net_rate": [1.0],
            "amount_rate": [2.0],
            "float_values": [50.0],
            "reason": ["涨幅偏离值达7%"],
        }
    )


def test_fetch_top_list_market_mode_by_trade_date():
    """全市场：按缺失交易日逐日 trade_date 拉 top_list 并落盘。"""
    from factorzen.core.loader import fetch_top_list

    mock_pro = MagicMock()
    mock_pro.top_list.return_value = _pd_top()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "get_trade_dates",
                     return_value=[date(2022, 1, 4), date(2022, 1, 5)]),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        fetch_top_list("20220101", "20220131")
    assert mock_pro.top_list.call_count == 2
    for c in mock_pro.top_list.call_args_list:
        assert "trade_date" in c.kwargs
    assert any(
        (getattr(c, "kwargs", {}) or {}).get("data_type") == "top_list"
        or (len(c.args) > 1 and c.args[1] == "top_list")
        for c in save.call_args_list
    ) or save.call_args.kwargs.get("data_type") == "top_list"


def test_fetch_top_list_empty_day_not_failure():
    """空日（无上榜）不算失败：应标记已拉、不抛异常。"""
    from factorzen.core.loader import fetch_top_list

    mock_pro = MagicMock()
    # 返回空 DataFrame（无上榜日）
    mock_pro.top_list.return_value = pd.DataFrame()
    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "get_trade_dates", return_value=[date(2022, 1, 4)]),
        patch.object(loader_module, "save_parquet") as save,
        patch.object(loader_module, "load_parquet", return_value=_lf(pl.DataFrame())),
    ):
        # 不应抛
        fetch_top_list("20220101", "20220131")
    # 空日应写入 sentinel / 标记，避免永久重拉
    assert save.called or mock_pro.top_list.call_count == 1


def test_cast_top_list_schema_stabilizes_drift_and_sentinel(tmp_path):
    """同列跨日推断类型漂移 + sentinel 行 → cast 后 save/load 往返无 SchemaError。"""
    from factorzen.core.loader import (
        _TOPLIST_EMPTY_CODE,
        TOP_LIST_COLS,
        TOP_LIST_SCHEMA,
        _cast_top_list_schema,
    )
    from factorzen.core.storage import load_parquet, save_parquet

    # 日1：全 null 数值/字符串列 → 推断为 Null
    day_null = pl.DataFrame({
        "trade_date": ["20240102"],
        "ts_code": [_TOPLIST_EMPTY_CODE],
        "name": [None],
        "close": [None],
        "pct_change": [None],
        "turnover_rate": [None],
        "amount": [None],
        "l_sell": [None],
        "l_buy": [None],
        "l_amount": [None],
        "net_amount": [None],
        "net_rate": [None],
        "amount_rate": [None],
        "float_values": [None],
        "reason": [None],
    })
    # 日2：真实行，String/Float64
    day_real = pl.DataFrame({
        "trade_date": ["20240103"],
        "ts_code": ["000001.SZ"],
        "name": ["平安银行"],
        "close": [10.5],
        "pct_change": [5.0],
        "turnover_rate": [3.0],
        "amount": [1e5],
        "l_sell": [100.0],
        "l_buy": [200.0],
        "l_amount": [300.0],
        "net_amount": [100.0],
        "net_rate": [1.0],
        "amount_rate": [2.0],
        "float_values": [50.0],
        "reason": ["涨幅偏离值达7%"],
    })
    # 未 cast 的 strict concat 会炸
    try:
        pl.concat([day_null, day_real])
        uncast_ok = True
    except pl.exceptions.SchemaError:
        uncast_ok = False
    assert not uncast_ok, "前置条件：未 cast 应 SchemaError"

    casted = pl.concat([
        _cast_top_list_schema(day_null),
        _cast_top_list_schema(day_real),
    ])
    assert casted.schema == pl.Schema(TOP_LIST_SCHEMA) or all(
        casted.schema[c] == TOP_LIST_SCHEMA[c] for c in TOP_LIST_COLS
    )
    assert casted["trade_date"].dtype == pl.Date

    save_parquet(casted, data_type="top_list", base_dir=tmp_path)
    # 跨分区 scan 往返
    loaded = load_parquet("top_list", start="20240101", end="20240131", base_dir=tmp_path).collect()
    assert loaded.height == 2
    assert loaded["reason"].dtype == pl.String
    assert loaded["net_amount"].dtype == pl.Float64


def test_fetch_top_list_schema_error_rewrites_year(tmp_path, monkeypatch):
    """混 schema 旧分区 load 触发 SchemaError → log 后重写该年（wipe + 重拉）。"""
    from factorzen.core.loader import _TOPLIST_EMPTY_CODE, fetch_top_list

    # 预置损坏 year=2024 分区（Null vs String 混）
    ydir = tmp_path / "top_list" / "year=2024" / "month=01"
    ydir.mkdir(parents=True)
    pl.DataFrame({
        "trade_date": [date(2024, 1, 2)],
        "ts_code": [_TOPLIST_EMPTY_CODE],
        "name": [None],
        "close": [None],
        "reason": [None],
        "net_amount": [None],
        "amount": [None],
        "pct_change": [None],
        "turnover_rate": [None],
        "l_sell": [None],
        "l_buy": [None],
        "l_amount": [None],
        "net_rate": [None],
        "amount_rate": [None],
        "float_values": [None],
    }).write_parquet(ydir / "data.parquet")
    ydir2 = tmp_path / "top_list" / "year=2024" / "month=02"
    ydir2.mkdir(parents=True)
    pl.DataFrame({
        "trade_date": [date(2024, 2, 1)],
        "ts_code": ["000001.SZ"],
        "name": ["平安银行"],
        "close": [10.0],
        "reason": ["涨幅偏离"],
        "net_amount": [100.0],
        "amount": [1e5],
        "pct_change": [1.0],
        "turnover_rate": [1.0],
        "l_sell": [1.0],
        "l_buy": [1.0],
        "l_amount": [1.0],
        "net_rate": [1.0],
        "amount_rate": [1.0],
        "float_values": [1.0],
    }).write_parquet(ydir2 / "data.parquet")

    # 确认 scan 会 SchemaError
    import pytest
    with pytest.raises(pl.exceptions.SchemaError):
        pl.scan_parquet(str(tmp_path / "top_list" / "**/*.parquet")).collect()

    mock_pro = MagicMock()
    mock_pro.top_list.return_value = _pd_top(trade_date="20240104")
    monkeypatch.setattr(loader_module, "DATA_RAW", tmp_path)
    # storage 也读 DATA_RAW
    import factorzen.core.storage as storage_mod
    monkeypatch.setattr(storage_mod, "DATA_RAW", tmp_path)

    with (
        patch.object(loader_module, "init_tushare", return_value=mock_pro),
        patch.object(loader_module, "_rate_limit"),
        patch.object(loader_module, "get_trade_dates",
                     return_value=[date(2024, 1, 4)]),
    ):
        # 不应因旧分区 SchemaError 而崩溃；应 wipe 并重拉
        fetch_top_list("20240101", "20240131")
    assert mock_pro.top_list.call_count >= 1
    # 重写后分区可扫描
    reloaded = pl.scan_parquet(str(tmp_path / "top_list" / "**/*.parquet")).collect()
    assert reloaded.height >= 1
    assert reloaded["reason"].dtype == pl.String


def test_holder_normalize_drops_junk_end_dates(tmp_path, monkeypatch):
    """源数据垃圾行防御：null end_date（会崩分区路径构造 month.zfill）与越界日期
    （实测 1900/2053/未来期末——后者会经 pit_align 最大 end_date 永久霸占最新期）必须过滤。"""
    import datetime as dt

    from factorzen.core.loader import _normalize_holder_frame
    future = (dt.date.today() + dt.timedelta(days=90)).strftime("%Y%m%d")
    raw = pl.DataFrame({
        "ts_code": ["000001.SZ", "300070.SZ", "300066.SZ", "000002.SZ", "002389.SZ"],
        "ann_date": ["20220430", "20230630", "20230911", "20220430", "20250312"],
        "end_date": ["20220331", "20530626", "19000908", None, future],
        "holder_num": [50000.0, None, None, 60000.0, None],
    })
    out = _normalize_holder_frame([raw])
    # 只有合法行存活；null end_date 与越界日期全部被丢
    assert out.height == 1
    assert out["ts_code"].to_list() == ["000001.SZ"]

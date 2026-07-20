"""tests/test_loader.py — common/loader.py 的 mock 单元测试。

全量 mock：不调用 Tushare API，不读写本地 data/raw/ 数据。
覆盖：_retry 重试逻辑、缓存跳过、pandas→polars 转换、finance 批次计数、
fetch_stock_basic 缓存命中/失效。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import polars as pl
import pytest

import factorzen.core.loader as loader_module
from factorzen.config.tushare_config import MAX_RETRIES
from factorzen.core.loader import (
    _retry,
    fetch_daily_basic,
    fetch_finance,
    fetch_namechange,
    fetch_stock_basic,
)

# ── 辅助：合成 pandas 输出 ──────────────────────────────────────────────────


def _pd_daily(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20220103"] * n,
            "ts_code": [f"{i:06d}.SZ" for i in range(n)],
            "open": [10.0] * n,
            "high": [11.0] * n,
            "low": [9.0] * n,
            "close": [10.5] * n,
            "pre_close": [10.0] * n,
            "change": [0.5] * n,
            "pct_chg": [5.0] * n,
            "vol": [1000.0] * n,
            "amount": [10000.0] * n,
        }
    )


def _pd_stock_basic(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [f"{i:06d}.SZ" for i in range(n)],
            "symbol": [f"{i:06d}" for i in range(n)],
            "name": [f"股票{i}" for i in range(n)],
            "area": ["广东"] * n,
            "industry": ["银行"] * n,
            "market": ["主板"] * n,
            "list_date": ["19910101"] * n,
            "delist_date": [None] * n,
        }
    )


def _pd_finance(n: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [f"{i:06d}.SZ" for i in range(n)],
            "ann_date": ["20230430"] * n,
            "end_date": ["20230331"] * n,
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. _retry — 核心重试逻辑
# ══════════════════════════════════════════════════════════════════════════════


class TestRetrySuccess:
    def test_returns_immediately_on_success(self):
        """成功返回非空结果时只调用一次。"""
        mock_func = MagicMock(return_value=_pd_daily())
        with patch.object(loader_module, "_rate_limit"):
            result = _retry(mock_func)
        assert mock_func.call_count == 1
        assert result is not None

    def test_passes_args_and_kwargs(self):
        mock_func = MagicMock(return_value=_pd_daily())
        with patch.object(loader_module, "_rate_limit"):
            _retry(mock_func, "arg1", key="val")
        mock_func.assert_called_once_with("arg1", key="val")


class TestRetryNetworkError:
    def test_retries_on_network_error(self):
        """网络/超时类错误重试 MAX_RETRIES 次，总计 MAX_RETRIES+1 次调用。"""
        mock_func = MagicMock(side_effect=Exception("connection timeout"))
        with (
            patch.object(loader_module, "_rate_limit"),
            patch("time.sleep"),
            pytest.raises(Exception, match="timeout"),
        ):
            _retry(mock_func)
        assert mock_func.call_count == MAX_RETRIES + 1

    def test_sleep_called_max_retries_times(self):
        """每次重试前 sleep 一次，共 MAX_RETRIES 次。"""
        mock_func = MagicMock(side_effect=Exception("network error"))
        with (
            patch.object(loader_module, "_rate_limit"),
            patch("time.sleep") as mock_sleep,
            pytest.raises(Exception, match="network error"),
        ):
            _retry(mock_func)
        assert mock_sleep.call_count == MAX_RETRIES

    def test_succeeds_on_second_attempt(self):
        """前几次失败、后面成功时只抛出一次异常就恢复。"""
        mock_func = MagicMock(
            side_effect=[Exception("timeout"), _pd_daily()]
        )
        with (
            patch.object(loader_module, "_rate_limit"),
            patch("time.sleep"),
        ):
            result = _retry(mock_func)
        assert mock_func.call_count == 2
        assert result is not None


class TestRetryPermissionError:
    @pytest.mark.parametrize("msg", ["token invalid", "权限不足", "参数错误", "积分不够"])
    def test_permission_error_no_retry(self, msg: str):
        """参数/权限/积分类错误立即抛出，不重试。"""
        mock_func = MagicMock(side_effect=Exception(msg))
        with (
            patch.object(loader_module, "_rate_limit"),
            patch("time.sleep") as mock_sleep,
            pytest.raises(Exception, match=msg),
        ):
            _retry(mock_func)
        assert mock_func.call_count == 1
        mock_sleep.assert_not_called()


class TestRetryRateLimit:
    def test_rate_limit_waits_62s(self):
        """频率超限错误等待 62 秒。"""
        mock_func = MagicMock(side_effect=Exception("频率超限，请稍后重试"))
        with (
            patch.object(loader_module, "_rate_limit"),
            patch("time.sleep") as mock_sleep,
            pytest.raises(Exception, match="频率超限"),
        ):
            _retry(mock_func)
        # 每次重试都 sleep(62)
        for c in mock_sleep.call_args_list:
            assert c == call(62.0)

    def test_rate_limit_keyword_triggers_62s(self):
        """消息含'频率'（不含'超限'）同样等待 62 秒。"""
        mock_func = MagicMock(side_effect=Exception("api频率限制"))
        with (
            patch.object(loader_module, "_rate_limit"),
            patch("time.sleep") as mock_sleep,
            pytest.raises(Exception, match="频率限制"),
        ):
            _retry(mock_func)
        for c in mock_sleep.call_args_list:
            assert c == call(62.0)


class TestRetryEmptyResult:
    def test_empty_dataframe_triggers_retry(self):
        """返回空 DataFrame 等同于无数据，触发重试直到耗尽。"""
        mock_func = MagicMock(return_value=pd.DataFrame())
        with (
            patch.object(loader_module, "_rate_limit"),
            patch("time.sleep"),pytest.raises(RuntimeError, match="空结果")
        ):
            _retry(mock_func)
        assert mock_func.call_count == MAX_RETRIES + 1

    def test_none_result_triggers_retry(self):
        """返回 None 等同于无数据，触发重试。"""
        mock_func = MagicMock(return_value=None)
        with (
            patch.object(loader_module, "_rate_limit"),
            patch("time.sleep"),pytest.raises((RuntimeError, Exception))
        ):
            _retry(mock_func)
        assert mock_func.call_count == MAX_RETRIES + 1


# ══════════════════════════════════════════════════════════════════════════════
# 3. fetch_daily_basic — pandas → polars 输出类型
# ══════════════════════════════════════════════════════════════════════════════


class TestFetchDailyBasicOutputType:
    def test_cache_hit_returns_polars(self):
        """缓存命中时 load_parquet 返回 polars DataFrame（不是 pandas）。"""
        mock_pro = MagicMock()
        expected_pl = pl.DataFrame(
            {
                "trade_date": pl.Series([None], dtype=pl.Date),
                "ts_code": ["000001.SZ"],
            }
        )
        lf_mock = MagicMock()
        lf_mock.collect.return_value = expected_pl

        with (
            patch.object(loader_module, "init_tushare", return_value=mock_pro),
            patch.object(loader_module, "get_trade_dates", return_value=[]),
            patch.object(loader_module, "load_parquet", return_value=lf_mock),
        ):
            result = fetch_daily_basic("20220101", "20221231")

        assert isinstance(result, pl.DataFrame)


# ══════════════════════════════════════════════════════════════════════════════
# 4. fetch_stock_basic — 缓存命中 / 失效
# ══════════════════════════════════════════════════════════════════════════════


class TestFetchStockBasicCache:
    def test_fresh_cache_skips_api(self, tmp_path: Path):
        """缓存文件刚写入（mtime 几乎为 now）→ 跳过 API 调用。"""
        cache_file = tmp_path / "stock_basic_L_D_P.parquet"
        fake = pl.DataFrame({"ts_code": ["000001.SZ"], "name": ["平安银行"]})
        fake.write_parquet(cache_file)

        mock_pro = MagicMock()
        with (
            patch.object(loader_module, "init_tushare", return_value=mock_pro),
            patch.object(loader_module, "DATA_CACHE", tmp_path),
        ):
            result = fetch_stock_basic()

        mock_pro.stock_basic.assert_not_called()
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 1

    def test_missing_cache_calls_api_and_writes(self, tmp_path: Path):
        """缓存文件不存在 → 调用 API，并将结果写入缓存文件。"""
        mock_pro = MagicMock()
        mock_pro.stock_basic.return_value = _pd_stock_basic(n=5)

        with (
            patch.object(loader_module, "init_tushare", return_value=mock_pro),
            patch.object(loader_module, "DATA_CACHE", tmp_path),
            patch.object(loader_module, "_rate_limit"),
        ):
            result = fetch_stock_basic(list_status="L")

        mock_pro.stock_basic.assert_called_once()
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 5
        # Cache file should now exist
        assert (tmp_path / "stock_basic_L.parquet").exists()

    def test_stale_cache_calls_api(self, tmp_path: Path, monkeypatch):
        """缓存文件过期（模拟 mtime 是 8 天前）→ 调用 API。"""
        import time

        cache_file = tmp_path / "stock_basic_L_D_P.parquet"
        fake = pl.DataFrame({"ts_code": ["000001.SZ"], "name": ["旧数据"]})
        fake.write_parquet(cache_file)

        # 把 mtime 设为 8 天前（超过 CACHE_EXPIRE_DAYS=7）
        stale_mtime = time.time() - 8 * 86400
        import os

        os.utime(cache_file, (stale_mtime, stale_mtime))

        mock_pro = MagicMock()
        mock_pro.stock_basic.return_value = _pd_stock_basic(n=2)

        with (
            patch.object(loader_module, "init_tushare", return_value=mock_pro),
            patch.object(loader_module, "DATA_CACHE", tmp_path),
            patch.object(loader_module, "_rate_limit"),
        ):
            result = fetch_stock_basic()

        mock_pro.stock_basic.assert_called()
        assert isinstance(result, pl.DataFrame)

    def test_pandas_to_polars_conversion(self, tmp_path: Path):
        """API 返回 pandas DataFrame，fetch_stock_basic 输出必须是 polars。"""
        mock_pro = MagicMock()
        mock_pro.stock_basic.return_value = _pd_stock_basic(n=3)

        with (
            patch.object(loader_module, "init_tushare", return_value=mock_pro),
            patch.object(loader_module, "DATA_CACHE", tmp_path),
            patch.object(loader_module, "_rate_limit"),
        ):
            result = fetch_stock_basic(list_status="L")

        assert isinstance(result, pl.DataFrame), "输出必须是 polars.DataFrame"
        assert "ts_code" in result.columns
        assert "list_date" in result.columns  # _str_to_date 已转换


# ══════════════════════════════════════════════════════════════════════════════
# 5. fetch_finance — 分批次计数
# ══════════════════════════════════════════════════════════════════════════════


class TestFetchFinanceBatchCount:
    def test_120_stocks_2_quarters_6_calls(self):
        """120 只股票 × 2 季度 → ceil(120/50)=3 批/季 × 2 = 6 次 API 调用。"""
        codes = [f"{i:06d}.SZ" for i in range(120)]
        fin_api_mock = MagicMock(return_value=_pd_finance())
        mock_pro = MagicMock()
        mock_pro.fina_indicator = fin_api_mock

        lf_mock = MagicMock()
        lf_mock.collect.return_value = pl.DataFrame()

        with (
            patch.object(loader_module, "init_tushare", return_value=mock_pro),
            patch.object(loader_module, "partition_exists", return_value=False),
            patch.object(loader_module, "save_parquet"),
            patch.object(loader_module, "load_parquet", return_value=lf_mock),
            patch.object(loader_module, "_rate_limit"),
            patch("time.sleep"),
        ):
            fetch_finance("fina_indicator", "20230101", "20230630", ts_codes=codes)

        # Q1 (Jan-Mar): 3 batches; Q2 (Apr-Jun): 3 batches = 6 total
        assert fin_api_mock.call_count == 6

    def test_50_stocks_1_quarter_1_call(self):
        """50 只股票恰好一批，1 季度 → 1 次 API 调用。"""
        codes = [f"{i:06d}.SZ" for i in range(50)]
        fin_api_mock = MagicMock(return_value=_pd_finance())
        mock_pro = MagicMock()
        mock_pro.income = fin_api_mock

        lf_mock = MagicMock()
        lf_mock.collect.return_value = pl.DataFrame()

        with (
            patch.object(loader_module, "init_tushare", return_value=mock_pro),
            patch.object(loader_module, "partition_exists", return_value=False),
            patch.object(loader_module, "save_parquet"),
            patch.object(loader_module, "load_parquet", return_value=lf_mock),
            patch.object(loader_module, "_rate_limit"),
            patch("time.sleep"),
        ):
            fetch_finance("income", "20230101", "20230331", ts_codes=codes)

        assert fin_api_mock.call_count == 1

    def test_cached_quarter_skips_all_batches(self):
        """已缓存分区 → 当季所有批次都跳过，API 不被调用。"""
        codes = [f"{i:06d}.SZ" for i in range(120)]
        fin_api_mock = MagicMock(return_value=_pd_finance())
        mock_pro = MagicMock()
        mock_pro.fina_indicator = fin_api_mock

        lf_mock = MagicMock()
        lf_mock.collect.return_value = pl.DataFrame()

        with (
            patch.object(loader_module, "init_tushare", return_value=mock_pro),
            patch.object(loader_module, "partition_exists", return_value=True),  # all cached
            patch.object(loader_module, "load_parquet", return_value=lf_mock),
            patch.object(loader_module, "_rate_limit"),
        ):
            fetch_finance("fina_indicator", "20230101", "20230630", ts_codes=codes)

        fin_api_mock.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# 6. fetch_namechange — 缓存命中/失效、不传日期参数（PIT 坑回归）、失败降级
# ══════════════════════════════════════════════════════════════════════════════


def _pd_namechange(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [f"{i:06d}.SZ" for i in range(n)],
            "name": [f"ST股票{i}" for i in range(n)],
            "start_date": ["20240101"] * n,
            "end_date": [None] * n,
            "ann_date": ["20240101"] * n,
            "change_reason": ["ST"] * n,
        }
    )


class TestFetchNamechangeCache:
    def test_fresh_cache_skips_api(self, tmp_path: Path):
        """缓存文件刚写入（mtime 几乎为 now）→ 跳过 API 调用。"""
        cache_file = tmp_path / "namechange.parquet"
        fake = pl.DataFrame({"ts_code": ["000001.SZ"], "change_reason": ["ST"]})
        fake.write_parquet(cache_file)

        mock_pro = MagicMock()
        with (
            patch.object(loader_module, "init_tushare", return_value=mock_pro),
            patch.object(loader_module, "DATA_CACHE", tmp_path),
        ):
            result = fetch_namechange()

        mock_pro.namechange.assert_not_called()
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 1

    def test_missing_cache_calls_api_without_date_params_and_writes(self, tmp_path: Path):
        """缓存不存在 → 调用 API。

        已知坑回归：调用时不应传 start_date/end_date —— Tushare namechange 接口
        底层按 ann_date 过滤日期参数，会把早期 ann_date 为空的记录静默丢弃，
        必须全量拉取后本地切片。同时验证结果写入本地缓存。
        """
        mock_pro = MagicMock()
        mock_pro.namechange.return_value = _pd_namechange(n=3)

        with (
            patch.object(loader_module, "init_tushare", return_value=mock_pro),
            patch.object(loader_module, "DATA_CACHE", tmp_path),
            patch.object(loader_module, "_rate_limit"),
        ):
            result = fetch_namechange()

        mock_pro.namechange.assert_called_once()
        _, kwargs = mock_pro.namechange.call_args
        assert "start_date" not in kwargs, (
            "不应传 start_date：namechange 接口按 ann_date 过滤，会静默丢弃早期空值记录"
        )
        assert "end_date" not in kwargs, "不应传 end_date：同上"
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 3
        assert (tmp_path / "namechange.parquet").exists()

    def test_stale_cache_calls_api(self, tmp_path: Path):
        """缓存文件过期（模拟 mtime 是 8 天前）→ 调用 API。"""
        import os
        import time

        cache_file = tmp_path / "namechange.parquet"
        fake = pl.DataFrame({"ts_code": ["000001.SZ"], "change_reason": ["旧数据"]})
        fake.write_parquet(cache_file)

        stale_mtime = time.time() - 8 * 86400
        os.utime(cache_file, (stale_mtime, stale_mtime))

        mock_pro = MagicMock()
        mock_pro.namechange.return_value = _pd_namechange(n=2)

        with (
            patch.object(loader_module, "init_tushare", return_value=mock_pro),
            patch.object(loader_module, "DATA_CACHE", tmp_path),
            patch.object(loader_module, "_rate_limit"),
        ):
            result = fetch_namechange()

        mock_pro.namechange.assert_called_once()
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 2

    def test_pandas_to_polars_date_casting(self, tmp_path: Path):
        """API 返回 pandas，fetch_namechange 输出必须是 polars 且日期列已转换为 pl.Date。"""
        mock_pro = MagicMock()
        mock_pro.namechange.return_value = _pd_namechange(n=2)

        with (
            patch.object(loader_module, "init_tushare", return_value=mock_pro),
            patch.object(loader_module, "DATA_CACHE", tmp_path),
            patch.object(loader_module, "_rate_limit"),
        ):
            result = fetch_namechange()

        assert isinstance(result, pl.DataFrame), "输出必须是 polars.DataFrame"
        for col in ("ts_code", "name", "start_date", "end_date", "ann_date", "change_reason"):
            assert col in result.columns
        assert result.schema["start_date"] == pl.Date
        assert result.schema["ann_date"] == pl.Date

    def test_fetch_failure_no_cache_raises(self, tmp_path: Path):
        """拉取失败且无可用本地缓存 → 向上抛出异常，由调用方决定如何降级。"""
        mock_pro = MagicMock()
        mock_pro.namechange.side_effect = RuntimeError("network down")

        with (
            patch.object(loader_module, "init_tushare", return_value=mock_pro),
            patch.object(loader_module, "DATA_CACHE", tmp_path),
            patch.object(loader_module, "_rate_limit"),
            patch("time.sleep"),
            pytest.raises(RuntimeError),
        ):
            fetch_namechange()

    def test_fetch_failure_with_stale_cache_falls_back(self, tmp_path: Path):
        """拉取失败但本地存在（即使过期的）缓存 → 回退读取缓存，不向上抛异常。"""
        import os
        import time

        cache_file = tmp_path / "namechange.parquet"
        fake = pl.DataFrame({"ts_code": ["000001.SZ"], "change_reason": ["ST"]})
        fake.write_parquet(cache_file)

        stale_mtime = time.time() - 8 * 86400
        os.utime(cache_file, (stale_mtime, stale_mtime))

        mock_pro = MagicMock()
        mock_pro.namechange.side_effect = RuntimeError("network down")

        with (
            patch.object(loader_module, "init_tushare", return_value=mock_pro),
            patch.object(loader_module, "DATA_CACHE", tmp_path),
            patch.object(loader_module, "_rate_limit"),
            patch("time.sleep"),
        ):
            result = fetch_namechange()

        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 1

    def test_empty_result_no_cache_returns_empty_df(self, tmp_path: Path):
        """_retry 返回空结果且无缓存 → 返回空 DataFrame，不抛异常。

        注：真实 _retry 对空结果会重试至 MAX_RETRIES 后抛异常（不会把空结果
        透传给调用方），所以这里直接 patch _retry 本身来隔离测试
        fetch_namechange 自己的「空结果」分支，而非测试 _retry 的重试语义。
        """
        mock_pro = MagicMock()

        with (
            patch.object(loader_module, "init_tushare", return_value=mock_pro),
            patch.object(loader_module, "DATA_CACHE", tmp_path),
            patch.object(loader_module, "_retry", return_value=pd.DataFrame()),
        ):
            result = fetch_namechange()

        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()

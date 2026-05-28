from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pytest

import common.data_ensure as data_ensure


def _daily_frame(trade_date: date | str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [trade_date],
            "ts_code": ["000001.SZ"],
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


def _pd_daily(trade_date: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [trade_date],
            "ts_code": ["000001.SZ"],
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


def test_audit_daily_like_reports_missing_trading_dates(tmp_path, monkeypatch):
    part_dir = tmp_path / "daily" / "year=2024" / "month=01"
    part_dir.mkdir(parents=True)
    _daily_frame(date(2024, 1, 2)).write_parquet(
        part_dir / "data.parquet",
    )
    monkeypatch.setattr(
        data_ensure,
        "get_trade_dates",
        lambda start, end: [date(2024, 1, 2), date(2024, 1, 3)],
    )

    result = data_ensure.audit_daily_like("daily", "20240102", "20240103", base_dir=tmp_path)

    assert result.ok is False
    assert result.present_dates == ["20240102"]
    assert result.missing_dates == ["20240103"]


def test_ensure_daily_fetches_only_missing_dates(tmp_path, monkeypatch):
    part_dir = tmp_path / "daily" / "year=2024" / "month=01"
    part_dir.mkdir(parents=True)
    _daily_frame(date(2024, 1, 2)).write_parquet(
        part_dir / "data.parquet",
    )
    monkeypatch.setattr(
        data_ensure,
        "get_trade_dates",
        lambda start, end: [date(2024, 1, 2), date(2024, 1, 3)],
    )
    pro = MagicMock()
    pro.daily.return_value = _pd_daily("20240103")
    monkeypatch.setattr(data_ensure, "init_tushare", lambda: pro)
    monkeypatch.setattr(data_ensure, "_retry", lambda func, **kwargs: func(**kwargs))

    result = data_ensure.ensure_daily("20240102", "20240103", base_dir=tmp_path)

    assert result.ok is True
    pro.daily.assert_called_once_with(trade_date="20240103")
    loaded = pl.scan_parquet(str(tmp_path / "daily" / "**/*.parquet")).collect()
    assert sorted(loaded["trade_date"].dt.strftime("%Y%m%d").unique().to_list()) == [
        "20240102",
        "20240103",
    ]


def test_ensure_daily_does_not_fetch_when_cache_is_complete(tmp_path, monkeypatch):
    part_dir = tmp_path / "daily" / "year=2024" / "month=01"
    part_dir.mkdir(parents=True)
    _daily_frame(date(2024, 1, 2)).write_parquet(
        part_dir / "data.parquet",
    )
    monkeypatch.setattr(data_ensure, "get_trade_dates", lambda start, end: [date(2024, 1, 2)])
    pro = MagicMock()
    monkeypatch.setattr(data_ensure, "init_tushare", lambda: pro)

    result = data_ensure.ensure_daily("20240102", "20240102", base_dir=tmp_path)

    assert result.ok is True
    pro.daily.assert_not_called()


def test_ensure_daily_raises_when_fetch_does_not_fill_gap(tmp_path, monkeypatch):
    monkeypatch.setattr(data_ensure, "get_trade_dates", lambda start, end: [date(2024, 1, 2)])
    pro = MagicMock()
    pro.daily.return_value = pd.DataFrame()
    monkeypatch.setattr(data_ensure, "init_tushare", lambda: pro)
    monkeypatch.setattr(data_ensure, "_retry", lambda func, **kwargs: pd.DataFrame())

    with pytest.raises(data_ensure.DataEnsureError, match="daily still missing"):
        data_ensure.ensure_daily("20240102", "20240102", base_dir=tmp_path)


def test_ensure_index_daily_fetches_missing_range(tmp_path, monkeypatch):
    monkeypatch.setattr(
        data_ensure,
        "get_trade_dates",
        lambda start, end: [date(2024, 1, 2), date(2024, 1, 3)],
    )
    pro = MagicMock()
    pro.index_daily.return_value = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103"],
            "ts_code": ["000300.SH", "000300.SH"],
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "pre_close": [1.0, 1.0],
            "change": [0.0, 0.0],
            "pct_chg": [0.0, 0.0],
            "vol": [1.0, 1.0],
            "amount": [1.0, 1.0],
        }
    )
    monkeypatch.setattr(data_ensure, "init_tushare", lambda: pro)
    monkeypatch.setattr(data_ensure, "_retry", lambda func, **kwargs: func(**kwargs))

    result = data_ensure.ensure_index_daily("000300.SH", "20240102", "20240103", base_dir=tmp_path)

    assert result.ok is True
    pro.index_daily.assert_called_once_with(
        ts_code="000300.SH",
        start_date="20240102",
        end_date="20240103",
    )

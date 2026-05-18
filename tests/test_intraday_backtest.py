"""tests/test_intraday_backtest.py"""

import datetime
import random

import polars as pl

from daily.evaluation.backtest import BacktestResult
from intraday.evaluation.backtest import aggregate_intraday_factor, run_intraday_backtest


def _make_minute_factor(
    n_stocks: int = 5, n_days: int = 10, bars_per_day: int = 20
) -> pl.DataFrame:
    random.seed(42)
    rows = []
    for day in range(n_days):
        base_date = datetime.date(2026, 1, 2) + datetime.timedelta(days=day)
        base_time = datetime.datetime(2026, 1, 2 + day, 9, 30)
        trade_date = base_date.strftime("%Y%m%d")
        for stock_i in range(1, n_stocks + 1):
            ts = f"00000{stock_i}.SZ"
            for b in range(bars_per_day):
                rows.append(
                    {
                        "trade_date": trade_date,
                        "trade_time": base_time + datetime.timedelta(minutes=b),
                        "ts_code": ts,
                        "factor_value": random.gauss(0, 1),
                    }
                )
    return pl.DataFrame(rows).with_columns(
        pl.col("trade_time").cast(pl.Datetime),
        pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d"),
    )


def _make_daily_price(n_stocks: int = 5, n_days: int = 10) -> pl.DataFrame:
    random.seed(0)
    rows = []
    for day in range(n_days):
        trade_date = (datetime.date(2026, 1, 2) + datetime.timedelta(days=day)).strftime("%Y%m%d")
        for stock_i in range(1, n_stocks + 1):
            open_price = 10.0 + stock_i
            close_price = open_price * (1.0 + random.gauss(0, 0.02))
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": f"00000{stock_i}.SZ",
                    "open": open_price,
                    "close": close_price,
                    "pre_close": open_price,
                    "pct_chg": (close_price / open_price - 1.0) * 100,
                    "vol": 1000.0,
                    "amount": 1_000_000.0,
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d"))


def test_aggregate_returns_one_row_per_stock_per_day():
    minute_factor = _make_minute_factor()
    daily = aggregate_intraday_factor(minute_factor)
    assert "trade_date" in daily.columns
    assert "ts_code" in daily.columns
    assert "factor_value" in daily.columns
    n_dates = minute_factor["trade_date"].n_unique()
    n_stocks = minute_factor["ts_code"].n_unique()
    assert len(daily) == n_dates * n_stocks


def test_aggregate_takes_last_value():
    """聚合后每日每股的因子值应是当日最后一根 bar 的值。"""
    df = _make_minute_factor(n_stocks=1, n_days=1, bars_per_day=5)
    daily = aggregate_intraday_factor(df)
    expected_last = df.sort("trade_time").tail(1)["factor_value"][0]
    assert abs(daily["factor_value"][0] - expected_last) < 1e-9


def test_run_intraday_backtest_returns_backtest_result():
    result = run_intraday_backtest(_make_minute_factor(), _make_daily_price(), n_groups=5)
    assert isinstance(result, BacktestResult)
    assert result.n_groups == 5


def test_run_intraday_backtest_has_long_short():
    result = run_intraday_backtest(_make_minute_factor(), _make_daily_price(), n_groups=5)
    assert "long_short" in result.summary_stats
    assert not result.nav.is_empty()

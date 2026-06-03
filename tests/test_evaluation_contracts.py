"""评估入口的数据契约(列校验 fail-fast)测试。"""

from __future__ import annotations

import polars as pl
import pytest

from factorzen.daily.evaluation.backtest import _prepare_factor_df, _prepare_price_df
from factorzen.daily.evaluation.turnover import compute_turnover


def test_compute_turnover_raises_on_missing_factor_column():
    df = pl.DataFrame({"trade_date": ["20240101"], "ts_code": ["000001.SZ"]})
    with pytest.raises(ValueError) as exc:
        compute_turnover(df, factor_col="factor_clean")
    msg = str(exc.value)
    assert "factor_clean" in msg
    assert "实际列" in msg


def test_compute_turnover_raises_on_missing_key_columns():
    df = pl.DataFrame({"factor_clean": [1.0]})
    with pytest.raises(ValueError) as exc:
        compute_turnover(df, factor_col="factor_clean")
    msg = str(exc.value)
    assert "trade_date" in msg and "ts_code" in msg


def test_prepare_factor_df_error_lists_actual_columns():
    df = pl.DataFrame({"trade_date": ["20240101"], "ts_code": ["000001.SZ"]})
    with pytest.raises(ValueError) as exc:
        _prepare_factor_df(df, "factor_clean")
    msg = str(exc.value)
    assert "factor_clean" in msg
    assert "实际列" in msg


def test_prepare_price_df_error_lists_actual_columns():
    df = pl.DataFrame({"trade_date": ["20240101"], "ts_code": ["000001.SZ"]})
    with pytest.raises(ValueError) as exc:
        _prepare_price_df(df)
    msg = str(exc.value)
    assert "close" in msg
    assert "实际列" in msg

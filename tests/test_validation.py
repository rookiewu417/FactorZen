"""Tests for core.validation column-contract helper."""

from __future__ import annotations

import polars as pl
import pytest

from factorzen.core.validation import require_columns


def test_require_columns_passes_when_all_present():
    df = pl.DataFrame({"trade_date": ["20240101"], "ts_code": ["000001.SZ"], "close": [1.0]})
    # 不应抛错
    require_columns(df, ["trade_date", "ts_code"], context="t")


def test_require_columns_raises_listing_missing_and_actual():
    df = pl.DataFrame({"trade_date": ["20240101"], "close": [1.0]})
    with pytest.raises(ValueError) as exc:
        require_columns(df, ["trade_date", "ts_code", "factor"], context="compute_ic")
    msg = str(exc.value)
    assert "compute_ic" in msg
    assert "ts_code" in msg and "factor" in msg
    # 实际存在的列也应在错误信息中,便于排查
    assert "trade_date" in msg


def test_require_columns_does_not_flag_present_columns():
    df = pl.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError) as exc:
        require_columns(df, ["a", "c"])
    assert "c" in str(exc.value)
    # 'a' 已存在,不应被列为缺失(出现在“缺少必需列 [...]”片段里)
    missing_part = str(exc.value).split("实际列")[0]
    assert "'a'" not in missing_part

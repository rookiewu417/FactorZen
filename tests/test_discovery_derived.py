from __future__ import annotations

import polars as pl


def _mask_df() -> pl.DataFrame:
    # 单股 3 天,已排序;含派生所需全部列
    return pl.DataFrame({
        "trade_date": [1, 2, 3],
        "ts_code": ["A", "A", "A"],
        "open": [10.0, 11.0, 12.0],
        "high": [11.0, 12.0, 13.0],
        "low": [9.0, 10.0, 11.0],
        "close": [10.5, 11.5, 12.5],
        "close_adj": [10.5, 11.5, 12.5],
        "pre_close": [10.0, 10.5, 11.5],
        "vol": [1e5, 1e5, 1e5],
        "amount": [1e6, 1e6, 1e6],
    }).sort(["ts_code", "trade_date"])


def test_add_derived_columns_values():
    from factorzen.discovery.derived import add_derived_columns
    out = add_derived_columns(_mask_df())
    for col in ["vwap", "log_vol", "ret_1d", "amplitude", "intraday_ret", "overnight_ret"]:
        assert col in out.columns
    row0 = out.row(0, named=True)
    assert abs(row0["amplitude"] - (11.0 - 9.0) / 10.0) < 1e-9          # (high-low)/pre_close
    assert abs(row0["intraday_ret"] - (10.5 / 10.0 - 1.0)) < 1e-9        # close/open-1
    assert abs(row0["overnight_ret"] - (10.0 / 10.0 - 1.0)) < 1e-9       # open/pre_close-1


def test_add_derived_columns_safe_when_pre_close_zero():
    from factorzen.discovery.derived import add_derived_columns
    df = _mask_df().with_columns(
        pl.when(pl.col("trade_date") == 1).then(0.0)
        .otherwise(pl.col("pre_close")).alias("pre_close"))
    out = add_derived_columns(df)
    assert out.row(0, named=True)["overnight_ret"] is None  # 分母 0 → null,不崩

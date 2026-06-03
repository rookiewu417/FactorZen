from datetime import date

import polars as pl
import pytest

from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns


def test_compute_fwd_returns_raises_on_missing_key_columns():
    # 缺少 ts_code → 应早失败并给出清晰错误,而非晦涩的 polars 异常
    df = pl.DataFrame({"trade_date": [date(2024, 1, 2)], "close": [100.0]})
    with pytest.raises(ValueError) as exc:
        compute_fwd_returns(df, horizons=[1])
    assert "ts_code" in str(exc.value)


def test_compute_fwd_returns_raises_when_no_price_or_ret_column():
    # 既无 close 也无 ret_col → 应早失败
    df = pl.DataFrame({"trade_date": [date(2024, 1, 2)], "ts_code": ["000001.SZ"]})
    with pytest.raises(ValueError) as exc:
        compute_fwd_returns(df, horizons=[1], ret_col="ret")
    msg = str(exc.value)
    assert "close" in msg and "ret" in msg


def test_fwd_ret_1d_uses_next_close_over_current_close():
    df = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "ts_code": ["000001.SZ"] * 3,
            "close": [100.0, 110.0, 121.0],
        }
    ).with_columns((pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("ret"))

    out = compute_fwd_returns(df, horizons=[1], ret_col="ret")

    assert out["fwd_ret_1d"].to_list() == pytest.approx([0.10, 0.10, None])


def test_fwd_ret_5d_is_cumulative_holding_period_return():
    closes = [100.0, 101.0, 103.0, 106.0, 110.0, 115.0, 121.0]
    df = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 5),
                date(2024, 1, 8),
                date(2024, 1, 9),
                date(2024, 1, 10),
            ],
            "ts_code": ["000001.SZ"] * len(closes),
            "close": closes,
        }
    ).with_columns((pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("ret"))

    out = compute_fwd_returns(df, horizons=[5], ret_col="ret")

    assert out["fwd_ret_5d"][0] == pytest.approx(115.0 / 100.0 - 1.0)
    assert out["fwd_ret_5d"][1] == pytest.approx(121.0 / 101.0 - 1.0)
    assert out["fwd_ret_5d"].to_list()[-5:] == [None, None, None, None, None]


def test_fwd_returns_compound_from_ret_when_close_is_absent():
    df = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 5),
            ],
            "ts_code": ["000001.SZ"] * 4,
            "ret": [0.0, 0.10, 0.20, -0.05],
        }
    )

    out = compute_fwd_returns(df, horizons=[2], ret_col="ret")

    assert out["fwd_ret_2d"][0] == pytest.approx((1.10 * 1.20) - 1.0)
    assert out["fwd_ret_2d"][1] == pytest.approx((1.20 * 0.95) - 1.0)
    assert out["fwd_ret_2d"].to_list()[-2:] == [None, None]

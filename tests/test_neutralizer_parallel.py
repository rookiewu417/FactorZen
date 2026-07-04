"""Tests for joblib-parallel neutralize_ols."""

import numpy as np
import polars as pl


def make_factor_df(n_dates=20, n_stocks=50) -> pl.DataFrame:
    from datetime import date, timedelta

    rng = np.random.default_rng(42)
    start = date(2023, 1, 3)
    rows = []
    industries = [f"ind_{i}" for i in range(5)]
    for d in range(n_dates):
        dt = (start + timedelta(days=d)).isoformat()
        for s in range(n_stocks):
            rows.append(
                {
                    "trade_date": dt,
                    "ts_code": f"00{s:04d}.SZ",
                    "factor": float(rng.standard_normal()),
                    "log_mktcap": float(rng.uniform(10, 20)),
                    "industry": industries[s % 5],
                }
            )
    return pl.DataFrame(rows)


def test_parallel_matches_serial():
    from factorzen.daily.preprocessing.neutralizer import neutralize_ols

    df = make_factor_df()
    serial = neutralize_ols(df, "factor", n_jobs=1)
    parallel = neutralize_ols(df, "factor", n_jobs=2)
    # Results should be numerically identical
    np.testing.assert_allclose(
        serial["factor"].to_numpy(),
        parallel["factor"].to_numpy(),
        rtol=1e-8,
        atol=1e-10,
    )


def test_serial_baseline():
    from factorzen.daily.preprocessing.neutralizer import neutralize_ols

    df = make_factor_df()
    result = neutralize_ols(df, "factor", n_jobs=1)
    assert len(result) == len(df)
    assert "factor" in result.columns


def test_neutralize_ols_handles_mixed_low_and_high_sample_days():
    """neutralize_ols 混合低样本日(<30)和正常样本日时不应因 dtype 不一致崩溃。

    低样本日的 OLS 分支早退分支此前返回未 cast 的 ``pl.lit(None)``
    (dtype=Null)，与其余日期 Float64 残差列 ``pl.concat`` 时报错。姊妹函数
    ``neutralize_by_styles`` 同分支已是 ``pl.lit(None).cast(pl.Float64)``
    的正确写法。必须传入 stock_basic/daily_basic 才会真正走 OLS 回归路径
    （而非"无数据跳过"的提前返回分支）。
    """
    from factorzen.daily.preprocessing.neutralizer import neutralize_ols

    rng = np.random.default_rng(7)
    industries = [f"ind_{i}" for i in range(5)]
    codes = [f"{i:06d}.SZ" for i in range(35)]

    stock_basic = pl.DataFrame(
        {
            "ts_code": codes,
            "industry": [industries[i % len(industries)] for i in range(len(codes))],
        }
    )

    low_date = "2023-01-03"  # 有效样本数 10 < 30
    high_date = "2023-01-04"  # 有效样本数 35 >= 30
    low_codes = codes[:10]
    high_codes = codes

    df_rows = []
    daily_basic_rows = []
    for code in low_codes:
        df_rows.append(
            {"trade_date": low_date, "ts_code": code, "factor": float(rng.standard_normal())}
        )
        daily_basic_rows.append(
            {"trade_date": low_date, "ts_code": code, "total_mv": float(rng.uniform(1e9, 1e11))}
        )
    for code in high_codes:
        df_rows.append(
            {"trade_date": high_date, "ts_code": code, "factor": float(rng.standard_normal())}
        )
        daily_basic_rows.append(
            {"trade_date": high_date, "ts_code": code, "total_mv": float(rng.uniform(1e9, 1e11))}
        )

    df = pl.DataFrame(df_rows)
    daily_basic = pl.DataFrame(daily_basic_rows)

    result = neutralize_ols(df, col="factor", stock_basic=stock_basic, daily_basic=daily_basic)

    assert result.height == df.height
    low_result = result.filter(pl.col("trade_date") == low_date)
    high_result = result.filter(pl.col("trade_date") == high_date)
    assert low_result["factor_neutral"].null_count() == low_result.height
    assert high_result["factor_neutral"].null_count() == 0
    assert np.all(np.isfinite(high_result["factor_neutral"].to_numpy()))

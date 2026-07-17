"""neutralize_ols 批量化等价性：残差与 statsmodels OLS 对齐（atol=1e-8）。"""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm


def _sm_ols_residuals_per_day(
    df: pl.DataFrame,
    col: str,
    stock_basic: pl.DataFrame | None,
    daily_basic: pl.DataFrame | None,
) -> pl.DataFrame:
    """旧路径语义的 statsmodels 参考（与 Wave1 前 neutralize_ols 同逻辑）。"""
    out_col = f"{col}_neutral"
    if stock_basic is None and daily_basic is None:
        return df.with_columns(pl.col(col).alias(out_col))

    industry_map: dict[str, str] = {}
    if stock_basic is not None:
        industry_map = dict(
            zip(
                stock_basic["ts_code"].to_list(),
                [
                    industry if industry is not None and industry != "" else "未知"
                    for industry in stock_basic["industry"].to_list()
                ],
                strict=False,
            )
        )

    dates = df["trade_date"].unique().sort().to_list()
    results: list[pl.DataFrame] = []
    for d in dates:
        cross = df.filter(pl.col("trade_date") == d)
        codes = cross["ts_code"].to_list()
        y = cross[col].to_numpy().astype(float)
        valid = ~np.isnan(y)

        log_mv: np.ndarray | None = None
        if daily_basic is not None:
            mv_cross = daily_basic.filter(pl.col("trade_date") == d).select(
                ["ts_code", "total_mv"]
            )
            mv_map = dict(
                zip(mv_cross["ts_code"].to_list(), mv_cross["total_mv"].to_list(), strict=False)
            )
            mv_arr = np.array([mv_map.get(c, np.nan) for c in codes], dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                log_mv = np.where(mv_arr > 0, np.log(mv_arr), np.nan)
            valid = valid & ~np.isnan(log_mv)

        if valid.sum() < 30:
            results.append(cross.with_columns(pl.lit(None).cast(pl.Float64).alias(out_col)))
            continue

        X_parts: list[np.ndarray] = [np.ones((len(codes), 1), dtype=float)]
        if industry_map:
            industries = [
                industry
                if (industry := industry_map.get(c)) is not None and industry != ""
                else "未知"
                for c in codes
            ]
            unique_ind = sorted(set(industries))
            ind_to_idx = {ind: i for i, ind in enumerate(unique_ind)}
            ind_dummies = np.zeros((len(codes), len(unique_ind) - 1))
            for i, ind in enumerate(industries):
                if ind_to_idx[ind] > 0:
                    ind_dummies[i, ind_to_idx[ind] - 1] = 1
            X_parts.append(ind_dummies)

        if log_mv is not None:
            X_parts.append(np.nan_to_num(log_mv, nan=0.0).reshape(-1, 1))

        X = np.hstack(X_parts)
        try:
            model = sm.OLS(y[valid], X[valid]).fit()
        except Exception:
            results.append(cross.with_columns(pl.lit(None).cast(pl.Float64).alias(out_col)))
            continue

        residuals = np.full(len(y), np.nan)
        residuals[valid] = y[valid] - model.predict(X[valid])
        results.append(cross.with_columns(pl.Series(out_col, residuals)))

    return pl.concat(results)


def _make_panel(
    n_dates: int = 8,
    n_stocks: int = 50,
    n_industries: int = 5,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    industries = [f"ind_{i}" for i in range(n_industries)]
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    stock_basic = pl.DataFrame(
        {
            "ts_code": codes,
            "industry": [industries[i % n_industries] for i in range(n_stocks)],
        }
    )
    df_rows = []
    db_rows = []
    for d in range(n_dates):
        td = f"2023-01-{d + 3:02d}"
        for i, code in enumerate(codes):
            ind_effect = (i % n_industries) * 0.5
            log_mv = float(rng.uniform(10, 20))
            mv = float(np.exp(log_mv))
            factor = float(ind_effect + 0.3 * log_mv + rng.standard_normal())
            df_rows.append({"trade_date": td, "ts_code": code, "factor_value": factor})
            db_rows.append({"trade_date": td, "ts_code": code, "total_mv": mv})
    return pl.DataFrame(df_rows), stock_basic, pl.DataFrame(db_rows)


def test_neutralize_ols_matches_statsmodels():
    """随机多日截面：新残差与 statsmodels 残差 atol=1e-8。"""
    from factorzen.daily.preprocessing.neutralizer import neutralize_ols

    df, stock_basic, daily_basic = _make_panel()
    got = neutralize_ols(
        df, col="factor_value", stock_basic=stock_basic, daily_basic=daily_basic
    )
    expected = _sm_ols_residuals_per_day(df, "factor_value", stock_basic, daily_basic)

    g = got.sort(["trade_date", "ts_code"])
    e = expected.sort(["trade_date", "ts_code"])
    np.testing.assert_allclose(
        g["factor_value_neutral"].to_numpy(),
        e["factor_value_neutral"].to_numpy(),
        atol=1e-8,
        rtol=0,
        equal_nan=True,
    )


def test_neutralize_ols_industry_only_and_size_only():
    from factorzen.daily.preprocessing.neutralizer import neutralize_ols

    df, stock_basic, daily_basic = _make_panel(n_dates=5, n_stocks=40)

    got_ind = neutralize_ols(df, col="factor_value", stock_basic=stock_basic, daily_basic=None)
    exp_ind = _sm_ols_residuals_per_day(df, "factor_value", stock_basic, None)
    np.testing.assert_allclose(
        got_ind.sort(["trade_date", "ts_code"])["factor_value_neutral"].to_numpy(),
        exp_ind.sort(["trade_date", "ts_code"])["factor_value_neutral"].to_numpy(),
        atol=1e-8,
        equal_nan=True,
    )

    got_sz = neutralize_ols(df, col="factor_value", stock_basic=None, daily_basic=daily_basic)
    exp_sz = _sm_ols_residuals_per_day(df, "factor_value", None, daily_basic)
    np.testing.assert_allclose(
        got_sz.sort(["trade_date", "ts_code"])["factor_value_neutral"].to_numpy(),
        exp_sz.sort(["trade_date", "ts_code"])["factor_value_neutral"].to_numpy(),
        atol=1e-8,
        equal_nan=True,
    )


def test_neutralize_ols_skip_low_sample():
    """有效样本 < 30 → 该日全 NaN（旧行为）。"""
    from factorzen.daily.preprocessing.neutralizer import neutralize_ols

    codes = [f"{i:06d}.SZ" for i in range(10)]
    df = pl.DataFrame(
        {
            "trade_date": ["2023-01-03"] * 10,
            "ts_code": codes,
            "factor_value": list(range(10)),
        }
    )
    stock_basic = pl.DataFrame(
        {"ts_code": codes, "industry": ["银行"] * 5 + ["医药"] * 5}
    )
    daily_basic = pl.DataFrame(
        {
            "trade_date": ["2023-01-03"] * 10,
            "ts_code": codes,
            "total_mv": [1e10] * 10,
        }
    )
    out = neutralize_ols(df, col="factor_value", stock_basic=stock_basic, daily_basic=daily_basic)
    assert out["factor_value_neutral"].null_count() == 10


def test_neutralize_ols_single_industry_day():
    """某日全市场同一行业：哑变量退化为仅 const(+log_mv)，应仍可回归。"""
    from factorzen.daily.preprocessing.neutralizer import neutralize_ols

    n = 40
    codes = [f"{i:06d}.SZ" for i in range(n)]
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "trade_date": ["2023-01-03"] * n,
            "ts_code": codes,
            "factor_value": rng.standard_normal(n).tolist(),
        }
    )
    stock_basic = pl.DataFrame({"ts_code": codes, "industry": ["同一行业"] * n})
    daily_basic = pl.DataFrame(
        {
            "trade_date": ["2023-01-03"] * n,
            "ts_code": codes,
            "total_mv": rng.uniform(1e9, 1e11, n).tolist(),
        }
    )
    got = neutralize_ols(df, col="factor_value", stock_basic=stock_basic, daily_basic=daily_basic)
    exp = _sm_ols_residuals_per_day(df, "factor_value", stock_basic, daily_basic)
    np.testing.assert_allclose(
        got["factor_value_neutral"].to_numpy(),
        exp["factor_value_neutral"].to_numpy(),
        atol=1e-8,
        equal_nan=True,
    )
    assert got["factor_value_neutral"].null_count() == 0


def test_neutralize_ols_industry_with_one_stock():
    """某行业当日仅 1 只：哑变量设计仍与 statsmodels 一致。"""
    from factorzen.daily.preprocessing.neutralizer import neutralize_ols

    n = 40
    codes = [f"{i:06d}.SZ" for i in range(n)]
    # 前 39 只行业 A，最后 1 只行业 B
    industries = ["A"] * (n - 1) + ["B"]
    rng = np.random.default_rng(1)
    df = pl.DataFrame(
        {
            "trade_date": ["2023-01-03"] * n,
            "ts_code": codes,
            "factor_value": rng.standard_normal(n).tolist(),
        }
    )
    stock_basic = pl.DataFrame({"ts_code": codes, "industry": industries})
    daily_basic = pl.DataFrame(
        {
            "trade_date": ["2023-01-03"] * n,
            "ts_code": codes,
            "total_mv": rng.uniform(1e9, 1e11, n).tolist(),
        }
    )
    got = neutralize_ols(df, col="factor_value", stock_basic=stock_basic, daily_basic=daily_basic)
    exp = _sm_ols_residuals_per_day(df, "factor_value", stock_basic, daily_basic)
    np.testing.assert_allclose(
        got.sort("ts_code")["factor_value_neutral"].to_numpy(),
        exp.sort("ts_code")["factor_value_neutral"].to_numpy(),
        atol=1e-8,
        equal_nan=True,
    )


def test_neutralize_ols_all_nan_factor_column():
    """全 NaN 因子列 → valid 不足 → 全日 NaN。"""
    from factorzen.daily.preprocessing.neutralizer import neutralize_ols

    n = 40
    codes = [f"{i:06d}.SZ" for i in range(n)]
    df = pl.DataFrame(
        {
            "trade_date": ["2023-01-03"] * n,
            "ts_code": codes,
            "factor_value": [None] * n,
        }
    ).with_columns(pl.col("factor_value").cast(pl.Float64))
    stock_basic = pl.DataFrame(
        {"ts_code": codes, "industry": ["银行" if i % 2 else "医药" for i in range(n)]}
    )
    out = neutralize_ols(df, col="factor_value", stock_basic=stock_basic)
    assert out["factor_value_neutral"].null_count() == n


def test_neutralize_ols_near_constant_factor():
    """近常数因子：仍应与 statsmodels 残差一致（接近 0）。"""
    from factorzen.daily.preprocessing.neutralizer import neutralize_ols

    n = 40
    codes = [f"{i:06d}.SZ" for i in range(n)]
    rng = np.random.default_rng(2)
    y = 5.0 + rng.normal(0, 1e-12, n)
    df = pl.DataFrame(
        {
            "trade_date": ["2023-01-03"] * n,
            "ts_code": codes,
            "factor_value": y.tolist(),
        }
    )
    stock_basic = pl.DataFrame(
        {"ts_code": codes, "industry": ["银行" if i % 2 else "医药" for i in range(n)]}
    )
    daily_basic = pl.DataFrame(
        {
            "trade_date": ["2023-01-03"] * n,
            "ts_code": codes,
            "total_mv": rng.uniform(1e9, 1e11, n).tolist(),
        }
    )
    got = neutralize_ols(df, col="factor_value", stock_basic=stock_basic, daily_basic=daily_basic)
    exp = _sm_ols_residuals_per_day(df, "factor_value", stock_basic, daily_basic)
    np.testing.assert_allclose(
        got.sort("ts_code")["factor_value_neutral"].to_numpy(),
        exp.sort("ts_code")["factor_value_neutral"].to_numpy(),
        atol=1e-8,
        equal_nan=True,
    )


def test_neutralize_ols_no_side_data_returns_original():
    from factorzen.daily.preprocessing.neutralizer import neutralize_ols

    df = pl.DataFrame(
        {
            "trade_date": ["2023-01-03"] * 3,
            "ts_code": ["a", "b", "c"],
            "factor_value": [1.0, 2.0, 3.0],
        }
    )
    out = neutralize_ols(df, col="factor_value", stock_basic=None, daily_basic=None)
    assert out["factor_value_neutral"].to_list() == [1.0, 2.0, 3.0]


def test_neutralize_ols_fwl_large_panel_matches_statsmodels():
    """更大截面（模拟全 A 量级的行业数）：FWL 与 statsmodels 仍 atol=1e-8。"""
    from factorzen.daily.preprocessing.neutralizer import neutralize_ols

    df, stock_basic, daily_basic = _make_panel(
        n_dates=15, n_stocks=120, n_industries=25, seed=99
    )
    got = neutralize_ols(
        df, col="factor_value", stock_basic=stock_basic, daily_basic=daily_basic
    )
    expected = _sm_ols_residuals_per_day(df, "factor_value", stock_basic, daily_basic)
    g = got.sort(["trade_date", "ts_code"])
    e = expected.sort(["trade_date", "ts_code"])
    np.testing.assert_allclose(
        g["factor_value_neutral"].to_numpy(),
        e["factor_value_neutral"].to_numpy(),
        atol=1e-8,
        rtol=0,
        equal_nan=True,
    )


def test_neutralize_ols_preserves_row_count_and_keys():
    from factorzen.daily.preprocessing.neutralizer import neutralize_ols

    df, stock_basic, daily_basic = _make_panel(n_dates=4, n_stocks=40)
    out = neutralize_ols(
        df, col="factor_value", stock_basic=stock_basic, daily_basic=daily_basic
    )
    assert out.height == df.height
    assert out["trade_date"].to_list() == df["trade_date"].to_list()
    assert out["ts_code"].to_list() == df["ts_code"].to_list()


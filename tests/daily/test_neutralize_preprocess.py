"""
test_neutralize_ols_equiv.py：neutralize_ols 批量化等价性：残差与 statsmodels OLS 对齐（atol=1e-8）。
test_neutralizer_parallel.py：Tests for joblib-parallel neutralize_ols.
test_neutralizer_mv_scale.py：市值中性化在真实 A 股市值量级下必须真正剥离 size 暴露（P0）。
test_normalizer.py：测试截面 Z-score 标准化。
test_normalizer_nan.py：截面标准化的 NaN 处理回归测试。
test_preprocessing_winsorize.py：Tests for winsorize_percentile and sigma_clip.
test_preprocessing_rank.py：Tests for cross_sectional_rank and quantile_transform.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import statsmodels.api as sm

from factorzen.daily.preprocessing.normalizer import cross_sectional_zscore


# ==== 来自 test_neutralize_ols_equiv.py ====
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


def test_neutralize_ols_equiv_suite():
    """随机多日截面：新残差与 statsmodels 残差 atol=1e-8。；test_neutralize_ols_industry_only_and_size_only；有效样本 < 30 → 该日全 NaN（旧行为）。；某日全市场同一行业：哑变量退化为仅 const(+log_mv)，应仍可回归。；某行业当日仅 1 只：哑变量设计仍与 statsmodels 一致。；全 NaN 因子列 → valid 不足 → 全日 NaN。；近常数因子：仍应与 statsmodels 残差一致（接近 0）。"""
    # -- 原 test_neutralize_ols_matches_statsmodels --
    def _section_0_test_neutralize_ols_matches_statsmodels():
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

    _section_0_test_neutralize_ols_matches_statsmodels()

    # -- 原 test_neutralize_ols_industry_only_and_size_only --
    def _section_1_test_neutralize_ols_industry_only_and_size_only():
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

    _section_1_test_neutralize_ols_industry_only_and_size_only()

    # -- 原 test_neutralize_ols_skip_low_sample --
    def _section_2_test_neutralize_ols_skip_low_sample():
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

    _section_2_test_neutralize_ols_skip_low_sample()

    # -- 原 test_neutralize_ols_single_industry_day --
    def _section_3_test_neutralize_ols_single_industry_day():
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

    _section_3_test_neutralize_ols_single_industry_day()

    # -- 原 test_neutralize_ols_industry_with_one_stock --
    def _section_4_test_neutralize_ols_industry_with_one_stock():
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

    _section_4_test_neutralize_ols_industry_with_one_stock()

    # -- 原 test_neutralize_ols_all_nan_factor_column --
    def _section_5_test_neutralize_ols_all_nan_factor_column():
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

    _section_5_test_neutralize_ols_all_nan_factor_column()

    # -- 原 test_neutralize_ols_near_constant_factor --
    def _section_6_test_neutralize_ols_near_constant_factor():
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

    _section_6_test_neutralize_ols_near_constant_factor()


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

# ==== 来自 test_neutralizer_parallel.py ====
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


def test_neutralize_ols_regression_failure_returns_nan(monkeypatch):
    """回归失败（>=30 样本但 OLS 抛异常）时中性化列应为 NaN，与 docstring 承诺
    及 <30 样本分支一致。此前静默返回未中性化的原值 y，会让因子带满行业/市值
    暴露漏到下游，下游却以为已中性化（研究可信度隐患）。
    """
    import factorzen.daily.preprocessing.neutralizer as neut
    from factorzen.daily.preprocessing.neutralizer import neutralize_ols

    n = 40  # >= 30 有效样本 → 进入 OLS 回归分支
    codes = [f"{i:06d}.SZ" for i in range(n)]
    df = pl.DataFrame({
        "trade_date": ["2023-01-04"] * n,
        "ts_code": codes,
        "factor": [float(i) for i in range(n)],
    })
    stock_basic = pl.DataFrame({
        "ts_code": codes,
        "industry": ["银行" if i % 2 else "医药" for i in range(n)],
    })

    def _boom_fwl(*args, **kwargs):
        raise RuntimeError("singular design matrix")

    # FWL 路径走 _fwl_attach_residuals；失败时整表标 NaN（与旧 lstsq 失败语义一致）
    monkeypatch.setattr(neut, "_fwl_attach_residuals", _boom_fwl)
    result = neutralize_ols(df, col="factor", stock_basic=stock_basic)

    assert result["factor_neutral"].null_count() == result.height, (
        "回归失败时中性化列应全为 NaN，而非静默返回未中性化的原值"
    )

# ==== 来自 test_neutralizer_mv_scale.py ====
def _cross_section(n=60, seed=0):
    """构造单日截面：total_mv 在 2万~90万（万元）即 2亿~90亿元，A股典型量级。"""
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}.SZ" for i in range(n)]
    total_mv = rng.uniform(2e4, 9e5, n)  # 单位万元
    # 因子几乎完全由 log(市值) 解释 → 正确中性化后残差应与 log(mv) 近乎不相关
    factor = 2.0 * np.log(total_mv) + rng.normal(0, 0.01, n)
    d = date(2024, 3, 1)
    df = pl.DataFrame({"trade_date": [d] * n, "ts_code": codes, "factor_value": factor})
    daily_basic = pl.DataFrame({"trade_date": [d] * n, "ts_code": codes, "total_mv": total_mv})
    return df, daily_basic, total_mv


def test_neutralizer_mv_scale_suite():
    """test_size_neutralization_removes_log_mv_exposure；缺失 total_mv 的股票应被剔除（残差 NaN），而不是用 1e8 巨常数冒充参与回归。"""
    # -- 原 test_size_neutralization_removes_log_mv_exposure --
    def _section_0_test_size_neutralization_removes_log_mv_exposure():
        from factorzen.daily.preprocessing.neutralizer import neutralize_ols

        df, daily_basic, total_mv = _cross_section()
        out = neutralize_ols(df, col="factor_value", daily_basic=daily_basic)
        resid = out["factor_value_neutral"].to_numpy()

        finite = ~np.isnan(resid)
        assert finite.sum() >= 30
        corr = np.corrcoef(resid[finite], np.log(total_mv[finite]))[0, 1]
        assert abs(corr) < 0.15, (
            f"市值中性化后残差与 log(mv) 相关系数 {corr:.3f} 应≈0；修复前因 log 被夹成常数、"
            "size 回归列退化，相关系数≈1（中性化失效）"
        )

    _section_0_test_size_neutralization_removes_log_mv_exposure()

    # -- 原 test_missing_mv_row_excluded_not_filled_with_giant_constant --
    def _section_1_test_missing_mv_row_excluded_not_filled_with_giant_constant():
        from factorzen.daily.preprocessing.neutralizer import neutralize_ols

        df, daily_basic, _ = _cross_section(n=60, seed=1)
        # 抹掉一只股票的 total_mv 行
        missing_code = df["ts_code"][0]
        daily_basic = daily_basic.filter(pl.col("ts_code") != missing_code)
        out = neutralize_ols(df, col="factor_value", daily_basic=daily_basic).sort("ts_code")
        row = out.filter(pl.col("ts_code") == missing_code)
        assert row["factor_value_neutral"].to_numpy()[0] != row["factor_value_neutral"].to_numpy()[0], (
            "缺失市值的股票中性化结果应为 NaN（被剔除），而非用巨常数冒充"
        )

    _section_1_test_missing_mv_row_excluded_not_filled_with_giant_constant()


# ==== 来自 test_normalizer.py ====
def _make_test_data(values: list[float], stocks: list[str] | None = None):
    """构造测试用 DataFrame。"""
    n = len(values)
    if stocks is None:
        stocks = [f"stock_{i}" for i in range(n)]
    return pl.DataFrame(
        {
            "stock_code": stocks,
            "trade_date": ["2026-01-05"] * n,
            "factor_value_clip_fill": values,
        }
    )


def test_normalizer_zscore_suite():
    """所有股票在同一截面上的值相同 → std=0 → zscore 全为 0.0。；多只股票不同值 → 正常计算 Z-score。；test_zscore_single_nan_does_not_poison_whole_day；test_rank_nan_not_ranked_highest"""
    # -- 原 test_zero_std --
    def _section_0_test_zero_std():
        df = _make_test_data([5.0, 5.0, 5.0])
        result = cross_sectional_zscore(df)
        col = "factor_value_clip_fill_z"
        assert result[col].to_list() == [0.0, 0.0, 0.0]

    _section_0_test_zero_std()

    # -- 原 test_normal_case --
    def _section_1_test_normal_case():
        df = _make_test_data([1.0, 2.0, 3.0])
        result = cross_sectional_zscore(df)
        col = "factor_value_clip_fill_z"
        # Polars std 默认 ddof=1：std([1,2,3]) = 1.0, mean = 2.0
        # z = (x - 2.0) / 1.0 → [-1.0, 0.0, 1.0]
        expected = [-1.0, 0.0, 1.0]
        assert result[col].to_list() == expected

    _section_1_test_normal_case()

    # -- 原 test_zscore_single_nan_does_not_poison_whole_day --
    def _section_2_test_zscore_single_nan_does_not_poison_whole_day():
        from factorzen.daily.preprocessing.normalizer import cross_sectional_zscore

        d = date(2024, 3, 1)
        df = pl.DataFrame({
            "trade_date": [d, d, d, d],
            "factor_value_clip_fill": [1.0, 2.0, 3.0, float("nan")],
        })
        out = cross_sectional_zscore(df, col="factor_value_clip_fill")
        z = out["factor_value_clip_fill_z"].to_numpy()

        finite = np.isfinite(z)
        assert finite.sum() == 3, f"3 个有效值应得到有效 z（修复前整日被 NaN 传染），实得 {finite.sum()}"
        # 有效行是标准 z-score：均值≈0
        assert abs(np.nanmean(z[finite])) < 1e-9
        assert not np.isfinite(z[3]), "NaN 输入行应保持非有限，不应被置 0 或污染"

    _section_2_test_zscore_single_nan_does_not_poison_whole_day()

    # -- 原 test_rank_nan_not_ranked_highest --
    def _section_3_test_rank_nan_not_ranked_highest():
        from factorzen.daily.preprocessing.normalizer import cross_sectional_rank

        d = date(2024, 3, 1)
        df = pl.DataFrame({
            "trade_date": [d] * 5,
            "ts_code": ["A", "B", "C", "D", "E"],
            "factor_value": [1.0, 2.0, 3.0, float("nan"), 4.0],
        })
        out = cross_sectional_rank(df, factor_col="factor_value", method="uniform").sort("ts_code")
        vals = dict(zip(out["ts_code"].to_list(), out["factor_value"].to_list(), strict=False))

        nan_score = vals["D"]
        max_real_score = vals["E"]  # 真实最大值 4.0
        assert nan_score is None or not np.isfinite(nan_score), (
            f"NaN 因子应得缺失分位（修复前 polars rank 排最大得最高分位 {nan_score}）"
        )
        assert max_real_score is not None and np.isfinite(max_real_score)
        # 真实最大值应拿到最高的有效分位
        finite_scores = [v for v in vals.values() if v is not None and np.isfinite(v)]
        assert max_real_score == max(finite_scores), "真实最大值应获最高有效分位，而非被 NaN 顶替"

    _section_3_test_rank_nan_not_ranked_highest()


# ==== 来自 test_normalizer_nan.py ====


# ==== 来自 test_preprocessing_winsorize.py ====

def test_winsorize_sigma_suite():
    """test_sigma_clip_removes_outliers；两个日期分布差异极大时，裁剪边界应按日期独立计算。；sigma_clip 应按日期截面计算 mean/std。"""
    # -- 原 test_sigma_clip_removes_outliers --
    def _section_0_test_sigma_clip_removes_outliers():
        from factorzen.daily.preprocessing.outlier import sigma_clip

        rng = np.random.default_rng(42)
        vals = [*list(rng.standard_normal(99)), 100.0]  # one huge outlier
        df = pl.DataFrame(
            {
                "trade_date": ["2024-01-01"] * 100,
                "ts_code": [f"00{i:04d}.SZ" for i in range(100)],
                "factor": vals,
            }
        )
        result = sigma_clip(df, "factor", n_sigma=3.0)
        # The 100.0 outlier is clipped to mean + 3*std.  With the outlier included
        # in the distribution the clipped max is ~31 — still well below 100.
        assert result["factor"].max() < 100.0  # outlier was reduced
        assert result["factor"].max() < 50.0  # meaningfully reduced from 100

    _section_0_test_sigma_clip_removes_outliers()

    # -- 原 test_winsorize_is_per_date --
    def _section_1_test_winsorize_is_per_date():
        from factorzen.daily.preprocessing.outlier import winsorize_percentile

        # Date A: values 1-10; Date B: values 1000-10000 (completely different scale)
        rows = []
        for i in range(1, 11):
            rows.append({"trade_date": "2024-01-01", "ts_code": f"00{i:04d}.SZ", "factor": float(i)})
        for i in range(1000, 10001, 1000):
            rows.append({"trade_date": "2024-01-02", "ts_code": f"00{i:04d}.SZ", "factor": float(i)})
        df = pl.DataFrame(rows)
        result = winsorize_percentile(df, "factor", lower=0.1, upper=0.9)

        date_a = result.filter(pl.col("trade_date") == "2024-01-01")["factor"]
        date_b = result.filter(pl.col("trade_date") == "2024-01-02")["factor"]
        # Date A max should be << 1000 (not contaminated by Date B)
        assert date_a.max() < 100, f"Date A max should be < 100, got {date_a.max()}"
        # Date B min should be >> 10 (not contaminated by Date A)
        assert date_b.min() > 100, f"Date B min should be > 100, got {date_b.min()}"

    _section_1_test_winsorize_is_per_date()

    # -- 原 test_sigma_clip_is_per_date --
    def _section_2_test_sigma_clip_is_per_date():
        from factorzen.daily.preprocessing.outlier import sigma_clip

        rows = []
        # Date A: normal values around 0
        for i in range(99):
            rows.append({"trade_date": "2024-01-01", "ts_code": f"00{i:04d}.SZ", "factor": float(i % 10 - 5)})
        # Date A outlier
        rows.append({"trade_date": "2024-01-01", "ts_code": "00099.SZ", "factor": 1000.0})
        # Date B: all values around 5000 (different mean)
        for i in range(10):
            rows.append({"trade_date": "2024-01-02", "ts_code": f"01{i:04d}.SZ", "factor": 5000.0 + float(i)})
        df = pl.DataFrame(rows)
        result = sigma_clip(df, "factor", n_sigma=3.0)

        date_a = result.filter(pl.col("trade_date") == "2024-01-01")["factor"]
        date_b = result.filter(pl.col("trade_date") == "2024-01-02")["factor"]
        # Date A outlier (1000) should be clipped to mean+3*std — well below 1000.
        # With the outlier itself pulling up std, the clipped ceiling is ~310,
        # which is still much less than 1000 (proving per-date isolation works).
        assert date_a.max() < 1000, f"Date A outlier should be clipped, got {date_a.max()}"
        assert date_a.max() < 500, f"Date A outlier should be meaningfully clipped, got {date_a.max()}"
        # Date B values (~5000) should remain ~5000, not dragged down by Date A's sigma
        assert date_b.min() > 100, f"Date B values should stay ~5000, got {date_b.min()}"

    _section_2_test_sigma_clip_is_per_date()


# ==== 来自 test_preprocessing_rank.py ====
def make_df(n=100) -> pl.DataFrame:
    """100 stocks × 5 dates."""
    dates = [f"2024-01-{d+1:02d}" for d in range(5)]
    rows = []
    rng = np.random.default_rng(42)
    for d in dates:
        vals = rng.standard_normal(n)
        for i, v in enumerate(vals):
            rows.append({"trade_date": d, "ts_code": f"00{i:04d}.SZ", "factor": v})
    return pl.DataFrame(rows)


def test_rank_quantile_suite():
    """test_rank_uniform_in_01；test_rank_normal_approx_standard_normal；test_quantile_transform_constant_column；test_quantile_transform_schema_preserved"""
    # -- 原 test_rank_uniform_in_01 --
    def _section_0_test_rank_uniform_in_01():
        from factorzen.daily.preprocessing.normalizer import cross_sectional_rank

        df = make_df()
        result = cross_sectional_rank(df, "factor", method="uniform")
        vals = result["factor"].drop_nulls()
        assert vals.min() > 0.0
        assert vals.max() < 1.0

    _section_0_test_rank_uniform_in_01()

    # -- 原 test_rank_normal_approx_standard_normal --
    def _section_1_test_rank_normal_approx_standard_normal():
        from scipy.stats import kstest

        from factorzen.daily.preprocessing.normalizer import cross_sectional_rank

        df = make_df(500)
        result = cross_sectional_rank(df, "factor", method="normal")
        vals = result["factor"].drop_nulls().to_numpy()
        _stat, p = kstest(vals, "norm")
        assert p > 0.01  # not rejected at 1%

    _section_1_test_rank_normal_approx_standard_normal()

    # -- 原 test_quantile_transform_constant_column --
    def _section_2_test_quantile_transform_constant_column():
        from factorzen.daily.preprocessing.normalizer import quantile_transform

        df = pl.DataFrame(
            {
                "trade_date": ["2024-01-01"] * 10,
                "ts_code": [f"00{i:04d}.SZ" for i in range(10)],
                "factor": [1.0] * 10,
            }
        )
        result = quantile_transform(df, "factor")
        # Should not raise; all values same (constant)
        assert len(result) == 10

    _section_2_test_quantile_transform_constant_column()

    # -- 原 test_quantile_transform_schema_preserved --
    def _section_3_test_quantile_transform_schema_preserved():
        from factorzen.daily.preprocessing.normalizer import quantile_transform

        df = make_df()
        result = quantile_transform(df, "factor")
        assert result.schema == df.schema
        assert len(result) == len(df)

    _section_3_test_quantile_transform_schema_preserved()



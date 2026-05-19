"""行业+市值中性化。截面 OLS 回归取残差。"""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm

from common.logger import get_logger

logger = get_logger(__name__)


def neutralize_ols(
    df: pl.DataFrame,
    col: str = "factor_value",
    stock_basic: pl.DataFrame | None = None,
    daily_basic: pl.DataFrame | None = None,
    n_jobs: int = 1,
) -> pl.DataFrame:
    """对每个 trade_date 截面做因子值 ~ 行业哑变量 + log(市值) 的 OLS，取残差。

    对每个交易日期，在截面上以因子值为被解释变量，
    行业哑变量和对数市值为解释变量做 OLS 回归，取残差作为中性化后的因子值。
    这样处理后的因子值已剥离行业和市值的影响，更适合做跨行业比较。

    如果 stock_basic 为 None 则跳过行业中性化（但保留市值中性化）；
    如果 daily_basic 为 None 则跳过市值中性化（但保留行业中性化）；
    两者均为 None 时直接返回原值。

    Parameters
    ----------
    df : pl.DataFrame
        因子 DataFrame，必须包含 trade_date, ts_code, {col} 列。
        trade_date 应为 int/str 格式（如 20240101）。
    col : str, default "factor_value_clip_fill_z"
        待中性化的因子列名。
    stock_basic : pl.DataFrame | None, default None
        股票基本信息，必须包含 ts_code 和 industry 列。
    daily_basic : pl.DataFrame | None, default None
        每日估值数据，必须包含 trade_date, ts_code, total_mv 列。
    n_jobs : int, default 1
        并行 worker 数。1 表示串行（默认）；-1 表示使用所有 CPU；
        其他正整数表示指定数量的线程（prefer="threads"）。

    Returns
    -------
    pl.DataFrame
        输入 DataFrame + 新增列 {col}_neutral。
        当中性化因缺失值或回归失败时，对应行的中性化结果为 NaN。
    """
    out_col = f"{col}_neutral"

    if stock_basic is None and daily_basic is None:
        logger.warning("neutralize_ols: 无行业/市值数据，跳过中性化")
        return df.with_columns(pl.col(col).alias(out_col))

    # 构建行业映射
    industry_map: dict[str, str] = {}
    if stock_basic is not None:
        industry_map = dict(
            zip(stock_basic["ts_code"].to_list(), stock_basic["industry"].to_list(), strict=False)
        )

    dates = df["trade_date"].unique().sort().to_list()

    def _process_date(d: object) -> pl.DataFrame:
        cross = df.filter(pl.col("trade_date") == d)
        codes = cross["ts_code"].to_list()
        y = cross[col].to_numpy()
        valid = ~np.isnan(y)

        if valid.sum() < 30:
            logger.warning(f"neutralize_ols: {d} 有效样本数 {valid.sum()} < 30，跳过")
            return cross.with_columns(pl.lit(None).alias(out_col))

        # 构建设计矩阵
        X_parts: list[np.ndarray] = [np.ones((len(codes), 1), dtype=float)]

        if industry_map:
            industries = [industry_map.get(c, "未知") for c in codes]
            unique_ind = sorted(set(industries))
            ind_to_idx = {ind: i for i, ind in enumerate(unique_ind)}
            ind_dummies = np.zeros((len(codes), len(unique_ind) - 1))
            for i, ind in enumerate(industries):
                if ind_to_idx[ind] > 0:
                    ind_dummies[i, ind_to_idx[ind] - 1] = 1
            X_parts.append(ind_dummies)

        if daily_basic is not None:
            mv_cross = daily_basic.filter(pl.col("trade_date") == d).select(["ts_code", "total_mv"])
            mv_map = dict(
                zip(mv_cross["ts_code"].to_list(), mv_cross["total_mv"].to_list(), strict=False)
            )
            log_mv = np.array([np.log(max(mv_map.get(c, 1e8), 1e6)) for c in codes])
            X_parts.append(log_mv.reshape(-1, 1))

        X = np.hstack(X_parts)

        try:
            model = sm.OLS(y[valid], X[valid]).fit()
            residuals = y - model.predict(X)
        except Exception as e:
            logger.warning(f"neutralize_ols: {d} 回归失败 ({e})，使用原值")
            residuals = y

        return cross.with_columns(pl.Series(out_col, residuals))

    # 对每个日期做截面回归（支持并行）
    if n_jobs == 1:
        results = [_process_date(d) for d in dates]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_process_date)(d) for d in dates
        )

    return pl.concat(results)


def neutralize_by_styles(
    factor_df: pl.DataFrame,
    style_dfs: list[pl.DataFrame],
    industry_map: dict[str, str] | None = None,
    col: str = "factor_value",
    n_jobs: int = 1,
) -> pl.DataFrame:
    """Barra-style 多因子中性化：截面 OLS(factor ~ style_1 + ... + industry_dummies)，取残差。

    Args:
        factor_df: 含 trade_date, ts_code, {col} 的因子 DataFrame。
        style_dfs: style 因子 DataFrame 列表，每个含 trade_date, ts_code, factor_value 列。
                   按传入顺序作为解释变量（style_0, style_1, ...）。
        industry_map: {ts_code: industry_name}，若提供则加入行业哑变量。
        col: 待中性化的因子列名。
        n_jobs: 并行 worker 数，1 为串行（默认），-1 为全部 CPU，其他正整数为线程数。

    Returns:
        输入 DataFrame + 新增列 {col}_style_neutral（OLS 残差）。
    """
    out_col = f"{col}_style_neutral"

    if not style_dfs:
        logger.warning("neutralize_by_styles: style_dfs 为空，直接返回原值")
        return factor_df.with_columns(pl.col(col).alias(out_col))

    # 合并所有 style 因子（rename factor_value 避免列名冲突）
    merged = factor_df
    for i, sdf in enumerate(style_dfs):
        style_col = f"_style_{i}"
        sdf_renamed = sdf.rename({"factor_value": style_col}).select(
            ["trade_date", "ts_code", style_col]
        )
        merged = merged.join(sdf_renamed, on=["trade_date", "ts_code"], how="left")

    style_cols = [f"_style_{i}" for i in range(len(style_dfs))]
    dates = merged["trade_date"].unique().sort().to_list()

    def _process_date_styles(d: object) -> pl.DataFrame:
        cross = merged.filter(pl.col("trade_date") == d)
        codes = cross["ts_code"].to_list()
        y = cross[col].to_numpy().astype(float)
        valid_mask = np.isfinite(y)

        # style 列
        style_matrix = []
        for sc in style_cols:
            sv = cross[sc].to_numpy().astype(float)
            style_matrix.append(sv)
            valid_mask &= np.isfinite(sv)

        if valid_mask.sum() < 30:
            cross = cross.with_columns(pl.lit(None).cast(pl.Float64).alias(out_col))
            return cross.select(
                [c for c in cross.columns if not c.startswith("_style_")] + [out_col]
            )

        # 构建设计矩阵
        X_parts: list[np.ndarray] = [np.ones((len(codes), 1), dtype=float)]
        for sv in style_matrix:
            X_parts.append(sv.reshape(-1, 1))

        if industry_map:
            industries = [industry_map.get(c, "未知") for c in codes]
            unique_ind = sorted(set(industries))
            ind_to_idx = {ind: i for i, ind in enumerate(unique_ind)}
            if len(unique_ind) > 1:
                ind_dummies = np.zeros((len(codes), len(unique_ind) - 1))
                for i, ind in enumerate(industries):
                    if ind_to_idx[ind] > 0:
                        ind_dummies[i, ind_to_idx[ind] - 1] = 1
                X_parts.append(ind_dummies)

        X = np.hstack(X_parts)
        residuals = y.copy()

        try:
            model = sm.OLS(y[valid_mask], X[valid_mask]).fit()
            residuals[valid_mask] = y[valid_mask] - model.predict(X[valid_mask])
            residuals[~valid_mask] = np.nan
        except Exception as e:
            logger.warning(f"neutralize_by_styles: {d} 回归失败 ({e})，使用原值")
            residuals[~valid_mask] = np.nan

        cross = cross.with_columns(pl.Series(out_col, residuals))
        return cross.select(
            [c for c in cross.columns if not c.startswith("_style_")] + [out_col]
        )

    if n_jobs == 1:
        results = [_process_date_styles(d) for d in dates]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_process_date_styles)(d) for d in dates
        )

    return pl.concat(results)

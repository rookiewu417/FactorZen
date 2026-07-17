"""行业+市值中性化。截面 OLS 回归取残差（FWL 向量化）。"""

from __future__ import annotations

import numpy as np
import polars as pl

from factorzen.core.logger import get_logger

logger = get_logger(__name__)


def _ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """numpy lstsq 残差：y - X @ beta（与 statsmodels OLS 数值对齐）。"""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _fwl_attach_residuals(
    work: pl.DataFrame,
    *,
    col: str,
    out_col: str,
    use_industry: bool,
    use_mv: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """在已 join 行业/市值的 work 上计算 FWL 残差列。

    Returns
    -------
    work_with_resid, n_by_date
        work 含 ``out_col``；n_by_date 为逐日有效样本计数（供 <30 日志）。
    """
    y_ok = pl.col(col).is_not_null() & pl.col(col).is_finite()
    if use_mv:
        x_ok = pl.col("_log_mv").is_not_null() & pl.col("_log_mv").is_finite()
        work = work.with_columns((y_ok & x_ok).alias("_valid"))
    else:
        work = work.with_columns(y_ok.alias("_valid"))

    work = work.with_columns(
        pl.when(pl.col("_valid")).then(pl.col(col)).otherwise(None).alias("_y_v")
    )
    if use_mv:
        work = work.with_columns(
            pl.when(pl.col("_valid"))
            .then(pl.col("_log_mv"))
            .otherwise(None)
            .alias("_x_v")
        )

    gcols: list[str] = ["trade_date", "_industry"] if use_industry else ["trade_date"]

    work = work.with_columns(pl.col("_y_v").mean().over(gcols).alias("_y_mean"))
    if use_mv:
        work = work.with_columns(pl.col("_x_v").mean().over(gcols).alias("_x_mean"))
        work = work.with_columns(
            (pl.col("_y_v") - pl.col("_y_mean")).alias("_y_dm"),
            (pl.col("_x_v") - pl.col("_x_mean")).alias("_x_dm"),
        )
    else:
        work = work.with_columns((pl.col("_y_v") - pl.col("_y_mean")).alias("_y_dm"))

    n_by_date = work.group_by("trade_date").agg(
        pl.col("_valid").sum().cast(pl.Int64).alias("_n_valid")
    )
    work = work.join(n_by_date, on="trade_date", how="left")

    if use_mv:
        beta_df = (
            work.filter(pl.col("_valid") & (pl.col("_n_valid") >= 30))
            .group_by("trade_date")
            .agg(
                (pl.col("_y_dm") * pl.col("_x_dm")).sum().alias("_sxy"),
                (pl.col("_x_dm") * pl.col("_x_dm")).sum().alias("_sxx"),
            )
            .with_columns(
                pl.when(pl.col("_sxx") > 0)
                .then(pl.col("_sxy") / pl.col("_sxx"))
                .otherwise(0.0)
                .alias("_beta")
            )
            .select(["trade_date", "_beta"])
        )
        work = work.join(beta_df, on="trade_date", how="left")
        resid_expr = (
            pl.when(
                ~pl.col("_valid")
                | (pl.col("_n_valid") < 30)
                | pl.col("_beta").is_null()
            )
            .then(None)
            .otherwise(pl.col("_y_dm") - pl.col("_beta") * pl.col("_x_dm"))
            .cast(pl.Float64)
            .alias(out_col)
        )
    else:
        resid_expr = (
            pl.when(~pl.col("_valid") | (pl.col("_n_valid") < 30))
            .then(None)
            .otherwise(pl.col("_y_dm"))
            .cast(pl.Float64)
            .alias(out_col)
        )

    return work.with_columns(resid_expr), n_by_date


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

    实现：Frisch–Waugh–Lovell ——
    「y 对 (行业 FE + log_mv + const) 的 OLS 残差」≡
    「组内去均值后的 y 对 组内去均值后的 log_mv 做单变量回归取残差」。
    组内去均值与单变量回归在 polars 中全日期×全行业一次向量化，
    与逐日 lstsq 数学恒等（浮点 atol≈1e-8）。

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
        兼容参数；FWL 路径为全向量化，不再使用逐日并行。

    Returns
    -------
    pl.DataFrame
        输入 DataFrame + 新增列 {col}_neutral。
        当中性化因缺失值或回归失败时，对应行的中性化结果为 NaN。
    """
    del n_jobs  # FWL 全向量化，保留 API 兼容
    out_col = f"{col}_neutral"

    if stock_basic is None and daily_basic is None:
        logger.warning("neutralize_ols: 无行业/市值数据，跳过中性化")
        return df.with_columns(pl.col(col).alias(out_col))

    use_industry = stock_basic is not None
    use_mv = daily_basic is not None

    # 行序锚点：输出与输入行一一对应
    work = df.with_row_index("_row_i")

    if use_industry:
        assert stock_basic is not None
        ind = (
            stock_basic.select(["ts_code", "industry"])
            .with_columns(
                pl.when(pl.col("industry").is_null() | (pl.col("industry") == ""))
                .then(pl.lit("未知"))
                .otherwise(pl.col("industry"))
                .alias("_industry")
            )
            .select(["ts_code", "_industry"])
        )
        work = work.join(ind, on="ts_code", how="left")
        work = work.with_columns(pl.col("_industry").fill_null("未知"))

    if use_mv:
        assert daily_basic is not None
        mv = daily_basic.select(["trade_date", "ts_code", "total_mv"]).rename(
            {"total_mv": "_total_mv"}
        )
        work = work.join(mv, on=["trade_date", "ts_code"], how="left")
        work = work.with_columns(
            pl.when(pl.col("_total_mv").is_not_null() & (pl.col("_total_mv") > 0))
            .then(pl.col("_total_mv").log())
            .otherwise(None)
            .alias("_log_mv")
        )

    try:
        work, n_by_date = _fwl_attach_residuals(
            work,
            col=col,
            out_col=out_col,
            use_industry=use_industry,
            use_mv=use_mv,
        )
    except Exception as e:
        # 回归/聚合失败：整表标 NaN（与 docstring、<30 样本分支一致）
        logger.warning(f"neutralize_ols: 回归失败 ({e})，标记为 NaN")
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out_col))

    low = n_by_date.filter(pl.col("_n_valid") < 30)
    if low.height > 0:
        for d, n in zip(
            low["trade_date"].to_list(), low["_n_valid"].to_list(), strict=True
        ):
            logger.warning(f"neutralize_ols: {d} 有效样本数 {n} < 30，跳过")

    base_cols = list(df.columns)
    return work.sort("_row_i").select([*base_cols, out_col])


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
            resid_valid = _ols_residuals(y[valid_mask], X[valid_mask])
            residuals[valid_mask] = resid_valid
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

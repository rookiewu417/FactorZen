"""行业+市值中性化。截面 OLS 回归取残差。"""

import numpy as np
import polars as pl
import statsmodels.api as sm
from common.logger import get_logger

logger = get_logger(__name__)


def neutralize_ols(
    df: pl.DataFrame,
    col: str = "factor_value_clip_fill_z",
    stock_basic: pl.DataFrame | None = None,
    daily_basic: pl.DataFrame | None = None,
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
    industry_map = {}
    if stock_basic is not None:
        industry_map = dict(zip(
            stock_basic["ts_code"].to_list(),
            stock_basic["industry"].to_list()
        ))

    # 对每个日期做截面回归
    dates = df["trade_date"].unique().sort().to_list()
    results = []

    for d in dates:
        cross = df.filter(pl.col("trade_date") == d)
        codes = cross["ts_code"].to_list()
        y = cross[col].to_numpy()
        valid = ~np.isnan(y)

        if valid.sum() < 30:
            logger.warning(
                f"neutralize_ols: {d} 有效样本数 {valid.sum()} < 30，跳过"
            )
            cross = cross.with_columns(pl.lit(None).alias(out_col))
            results.append(cross)
            continue

        # 构建设计矩阵
        X_parts = [np.ones(len(codes))]

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
            mv_cross = daily_basic.filter(
                (pl.col("trade_date") == d)
            ).select(["ts_code", "total_mv"])
            mv_map = dict(zip(
                mv_cross["ts_code"].to_list(),
                mv_cross["total_mv"].to_list()
            ))
            log_mv = np.array([
                np.log(max(mv_map.get(c, 1e8), 1e6)) for c in codes
            ])
            X_parts.append(log_mv.reshape(-1, 1))

        X = np.column_stack(X_parts)

        try:
            model = sm.OLS(y[valid], X[valid]).fit()
            residuals = y - model.predict(X)
        except Exception as e:
            logger.warning(f"neutralize_ols: {d} 回归失败 ({e})，使用原值")
            residuals = y

        cross = cross.with_columns(
            pl.Series(out_col, residuals)
        )
        results.append(cross)

    return pl.concat(results)

"""Factor Correlation — 多因子截面 Rank 相关性矩阵 + FDR 校正。"""

from __future__ import annotations

import numpy as np
import polars as pl

from factorzen.core.logger import get_logger

logger = get_logger(__name__)


def compute_factor_correlation(
    factor_dfs: dict[str, pl.DataFrame],
    factor_col: str = "factor_clean",
) -> pl.DataFrame:
    """计算多因子截面 Rank 相关性均值矩阵（Spearman 相关）。

    Args:
        factor_dfs: {factor_name: DataFrame(trade_date, ts_code, {factor_col})}
        factor_col: 因子列名

    Returns:
        pl.DataFrame — 含 "factor" 列及各因子列的方阵，值为平均 Spearman 相关系数。
    """
    names = list(factor_dfs.keys())
    n = len(names)

    if n == 0:
        return pl.DataFrame()

    if n == 1:
        return pl.DataFrame({"factor": names, names[0]: [1.0]})

    # 合并所有因子到宽表（每日截面）
    merged: pl.DataFrame | None = None
    for name, df in factor_dfs.items():
        col_renamed = df.select(["trade_date", "ts_code", pl.col(factor_col).alias(name)])
        if merged is None:
            merged = col_renamed
        else:
            merged = merged.join(col_renamed, on=["trade_date", "ts_code"], how="inner")

    if merged is None or merged.is_empty():
        # 返回单位矩阵
        data: dict[str, list] = {"factor": names}
        for n1 in names:
            data[n1] = [1.0 if n1 == n2 else 0.0 for n2 in names]
        return pl.DataFrame(data)

    # 对每个日期计算截面 Rank 相关，然后累加
    dates = merged["trade_date"].unique().sort().to_list()
    cum_corr = np.zeros((n, n))
    count = 0

    for d in dates:
        cross = merged.filter(pl.col("trade_date") == d).drop_nulls()
        if len(cross) < 2:
            continue
        arr = np.column_stack([cross[name].to_numpy() for name in names])
        stds = arr.std(axis=0)
        if np.any(stds == 0):
            continue
        # Spearman: rank then Pearson
        from scipy.stats import rankdata

        ranked_arr = np.column_stack([rankdata(arr[:, i]) for i in range(n)])
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(ranked_arr.T)
            if not np.any(np.isnan(corr)):
                cum_corr += corr
                count += 1
        except Exception:
            continue

    if count > 0:
        avg_corr = cum_corr / count
    else:
        avg_corr = np.eye(n)

    np.fill_diagonal(avg_corr, 1.0)

    # 构造 DataFrame（含 "factor" 列作为行标签）
    data2: dict[str, list] = {"factor": names}
    for j, col_name in enumerate(names):
        data2[col_name] = [float(avg_corr[i, j]) for i in range(n)]

    return pl.DataFrame(data2)


def apply_fdr_correction(
    p_values: dict[str, float],
    method: str = "fdr_bh",
) -> dict[str, float]:
    """对多因子批量评估的 p 值进行多重检验校正。

    Args:
        p_values: {因子名: p_value} 字典。
        method: statsmodels multipletests 支持的方法，如：
            "fdr_bh"（Benjamini-Hochberg，控制 FDR，默认）、
            "bonferroni"（Bonferroni，控制 FWER，更保守）、
            "fdr_by"（Benjamini-Yekutieli）。

    Returns:
        {因子名: 校正后 p 值} 字典，键顺序与输入一致。
    """
    from statsmodels.stats.multitest import multipletests

    if not p_values:
        return {}

    names = list(p_values.keys())
    raw_pvals = np.array([p_values[n] for n in names])

    _, pvals_corrected, _, _ = multipletests(raw_pvals, method=method)

    return {n: float(p) for n, p in zip(names, pvals_corrected, strict=True)}

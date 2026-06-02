"""多因子合成方法。

三种合成策略：等权平均、IC 加权、最大化 IR（闭式解）。
所有方法均在 z-score 化后的因子值上操作，输入需包含 trade_date, ts_code, factor_value。
"""

from __future__ import annotations

import numpy as np
import polars as pl


def _zscore_factor(df: pl.DataFrame, col: str = "factor_value") -> pl.DataFrame:
    """截面 z-score 标准化。"""
    return (
        df.with_columns(
            [
                pl.col(col).mean().over("trade_date").alias("_mean"),
                pl.col(col).std(ddof=1).over("trade_date").alias("_std"),
            ]
        )
        .with_columns(
            pl.when(pl.col("_std") > 0)
            .then((pl.col(col) - pl.col("_mean")) / pl.col("_std"))
            .otherwise(0.0)
            .alias(col)
        )
        .drop(["_mean", "_std"])
    )


def _compute_ic_series(
    factor_df: pl.DataFrame,
    ret_df: pl.DataFrame,
    factor_name: str,
) -> np.ndarray:
    """计算因子 vs 前向收益的截面 IC 序列（Pearson(rank(f), rank(r))）。"""
    merged = factor_df.rename({"factor_value": "_fv"}).join(
        ret_df.rename({"ret": "_ret"}), on=["trade_date", "ts_code"], how="inner"
    )
    ic_rows = []
    for _date, group in merged.group_by("trade_date"):
        g = group.drop_nulls(subset=["_fv", "_ret"])
        if len(g) < 10:
            continue
        fv = g["_fv"].to_numpy().astype(float)
        rv = g["_ret"].to_numpy().astype(float)
        if np.std(fv) < 1e-12 or np.std(rv) < 1e-12:
            continue
        fv_rank = fv.argsort().argsort().astype(float)
        rv_rank = rv.argsort().argsort().astype(float)
        ic = float(np.corrcoef(fv_rank, rv_rank)[0, 1])
        if np.isfinite(ic):
            ic_rows.append(ic)
    return np.array(ic_rows)


def equal_weight(
    factor_dfs: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """等权合成：对每个因子截面 z-score 后取均值。

    Args:
        factor_dfs: {factor_name: DataFrame(trade_date, ts_code, factor_value)}

    Returns:
        DataFrame(trade_date, ts_code, factor_value) — 合成后因子
    """
    if not factor_dfs:
        raise ValueError("factor_dfs 不能为空")

    normed = []
    for name, df in factor_dfs.items():
        z = _zscore_factor(df.select(["trade_date", "ts_code", "factor_value"]))
        normed.append(z.rename({"factor_value": f"_f_{name}"}))

    merged = normed[0]
    for z in normed[1:]:
        merged = merged.join(z, on=["trade_date", "ts_code"], how="inner")

    factor_cols = [f"_f_{n}" for n in factor_dfs]
    combined = merged.with_columns(
        pl.concat_list(factor_cols).list.mean().alias("factor_value")
    ).select(["trade_date", "ts_code", "factor_value"])
    return combined


def ic_weighted(
    factor_dfs: dict[str, pl.DataFrame],
    ret_df: pl.DataFrame,
    ic_window: int = 60,
) -> pl.DataFrame:
    """IC 加权合成：以历史 IC 均值（仅正向）为权重，加权平均各因子 z-score。

    使用全历史 IC 作为权重（in-sample 研究口径）。
    权重 = max(0, IC_mean) / sum(max(0, IC_mean_i))，若所有因子 IC ≤ 0 则退化为等权。

    Args:
        factor_dfs: {factor_name: DataFrame(trade_date, ts_code, factor_value)}
        ret_df: DataFrame(trade_date, ts_code, ret) — 对齐到因子的前向收益
        ic_window: 计算 IC 的最近窗口天数（-1 表示全历史）

    Returns:
        DataFrame(trade_date, ts_code, factor_value) — 加权合成因子
    """
    weights: dict[str, float] = {}
    for name, df in factor_dfs.items():
        ic_series = _compute_ic_series(df, ret_df, name)
        if len(ic_series) == 0:
            weights[name] = 0.0
        else:
            tail = ic_series[-ic_window:] if ic_window > 0 else ic_series
            weights[name] = float(max(0.0, np.mean(tail)))

    total_w = sum(weights.values())
    if total_w < 1e-12:
        # 退化为等权
        weights = {n: 1.0 / len(factor_dfs) for n in factor_dfs}
    else:
        weights = {n: w / total_w for n, w in weights.items()}

    normed = []
    for name, df in factor_dfs.items():
        z = _zscore_factor(df.select(["trade_date", "ts_code", "factor_value"]))
        normed.append((name, z.rename({"factor_value": f"_f_{name}"})))

    merged = normed[0][1]
    for _, z in normed[1:]:
        merged = merged.join(z, on=["trade_date", "ts_code"], how="inner")

    # 加权求和（用循环避免 sum() 从 int(0) 起步导致的类型歧义）
    weight_exprs = [pl.col(f"_f_{n}") * weights[n] for n in factor_dfs]
    expr: pl.Expr = weight_exprs[0]
    for e in weight_exprs[1:]:
        expr = expr + e
    combined = merged.with_columns(expr.alias("factor_value")).select(
        ["trade_date", "ts_code", "factor_value"]
    )
    return combined


def max_ir(
    factor_dfs: dict[str, pl.DataFrame],
    ret_df: pl.DataFrame,
    lookback: int = 120,
) -> pl.DataFrame:
    """最大化 IR 合成：闭式解 w = Σ^{-1} · μ（Ledoit-Wolf 收缩正则化）。

    μ = 各因子 IC 均值向量，Σ = 因子 IC 协方差矩阵。
    w 归一化为 L1 单位（仅取正权重因子）。

    Args:
        factor_dfs: {factor_name: DataFrame(trade_date, ts_code, factor_value)}
        ret_df: DataFrame(trade_date, ts_code, ret)
        lookback: IC 历史窗口长度

    Returns:
        DataFrame(trade_date, ts_code, factor_value)
    """
    names = list(factor_dfs.keys())
    k = len(names)

    min_len = None

    ic_series_map: dict[str, np.ndarray] = {}
    for name, df in factor_dfs.items():
        ic = _compute_ic_series(df, ret_df, name)
        ic_series_map[name] = ic
        if min_len is None or len(ic) < min_len:
            min_len = len(ic)

    if min_len is None or min_len < k + 1:
        # 数据不足，退化为等权
        return equal_weight(factor_dfs)

    # 截取最近 lookback 期，并对齐长度
    tail_len = min(lookback, min_len)
    ic_mat = np.column_stack([ic_series_map[n][-tail_len:] for n in names])  # (T, K)

    mu = ic_mat.mean(axis=0)
    try:
        from sklearn.covariance import LedoitWolf  # type: ignore[import]

        lw = LedoitWolf().fit(ic_mat)
        sigma = lw.covariance_
    except ImportError:
        sigma = np.cov(ic_mat, rowvar=False) + np.eye(k) * 1e-6

    try:
        sigma_inv = np.linalg.inv(sigma + np.eye(k) * 1e-6)
    except np.linalg.LinAlgError:
        sigma_inv = np.eye(k)

    w_raw = sigma_inv @ mu
    w_pos = np.maximum(w_raw, 0.0)
    if w_pos.sum() < 1e-12:
        w_pos = np.ones(k)
    weights = dict(zip(names, w_pos / w_pos.sum(), strict=True))

    normed = []
    for name, df in factor_dfs.items():
        z = _zscore_factor(df.select(["trade_date", "ts_code", "factor_value"]))
        normed.append((name, z.rename({"factor_value": f"_f_{name}"})))

    merged = normed[0][1]
    for _, z in normed[1:]:
        merged = merged.join(z, on=["trade_date", "ts_code"], how="inner")

    weight_exprs = [pl.col(f"_f_{n}") * weights[n] for n in names]
    expr: pl.Expr = weight_exprs[0]
    for e in weight_exprs[1:]:
        expr = expr + e
    combined = merged.with_columns(expr.alias("factor_value")).select(
        ["trade_date", "ts_code", "factor_value"]
    )
    return combined

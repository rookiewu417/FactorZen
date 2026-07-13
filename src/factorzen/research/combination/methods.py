"""多因子合成方法。

估权(estimate_*)与应用(apply_weights)拆开:估权只吃「因子 + 前向收益」产出权重向量,
应用只吃「因子 + 权重」产出合成因子。三种公开方法(equal_weight/ic_weighted/max_ir)
是「估权 + 应用」的薄包装;OOS 协议(oos.combine_oos)则逐折用 train 估权、test 应用。
所有方法在截面 z-score 化后的因子值上操作,输入需含 trade_date, ts_code, factor_value。
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
    ic_rows: list[tuple[object, float]] = []
    for date_key, group in merged.group_by("trade_date"):
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
            ic_rows.append((date_key[0], ic))
    # 按交易日排序:ic_window 取「最近」窗口依赖时序,而 group_by 迭代顺序不保证
    ic_rows.sort(key=lambda x: x[0])  # type: ignore[return-value,arg-type]
    return np.array([ic for _, ic in ic_rows])


def _zscore_and_merge(
    factor_dfs: dict[str, pl.DataFrame],
) -> tuple[pl.DataFrame, list[str]]:
    """各因子截面 z-score 后 **outer join** 成宽表(列名 `_f_<name>`),缺失补 0。

    因子库覆盖常异质(不同因子覆盖的股票/日期不同)。inner join 会把并集缩到交集、
    甚至塌空;改外连接取并集,某股票缺某因子时该因子补 0(z-score 后 0=截面均值=中性),
    等价于「缺失因子不表态」,不至于整行被丢或组合崩。
    """
    if not factor_dfs:
        raise ValueError("factor_dfs 不能为空")
    normed = []
    for name, df in factor_dfs.items():
        z = _zscore_factor(df.select(["trade_date", "ts_code", "factor_value"]))
        normed.append(z.rename({"factor_value": f"_f_{name}"}))
    merged = normed[0]
    for z in normed[1:]:
        merged = merged.join(z, on=["trade_date", "ts_code"], how="full", coalesce=True)
    fcols = [f"_f_{n}" for n in factor_dfs]
    merged = merged.with_columns([pl.col(c).fill_null(0.0) for c in fcols])
    return merged, list(factor_dfs.keys())


def apply_weights(
    factor_dfs: dict[str, pl.DataFrame], weights: dict[str, float]
) -> pl.DataFrame:
    """按给定权重加权合成(先各因子截面 z-score)。

    Args:
        factor_dfs: {factor_name: DataFrame(trade_date, ts_code, factor_value)}
        weights: {factor_name: weight}

    Returns:
        DataFrame(trade_date, ts_code, factor_value) — 加权合成因子
    """
    merged, names = _zscore_and_merge(factor_dfs)
    exprs = [pl.col(f"_f_{n}") * weights[n] for n in names]
    expr: pl.Expr = exprs[0]
    for e in exprs[1:]:
        expr = expr + e
    return merged.with_columns(expr.alias("factor_value")).select(
        ["trade_date", "ts_code", "factor_value"]
    )


def estimate_equal_weights(factor_dfs: dict[str, pl.DataFrame]) -> dict[str, float]:
    """等权:每因子 1/k。"""
    if not factor_dfs:
        raise ValueError("factor_dfs 不能为空")
    k = len(factor_dfs)
    return {n: 1.0 / k for n in factor_dfs}


def estimate_ic_weights(
    factor_dfs: dict[str, pl.DataFrame],
    ret_df: pl.DataFrame,
    ic_window: int = 60,
) -> dict[str, float]:
    """IC 加权:权重 = max(0, IC_mean) 归一化;全非正则退化等权。

    Args:
        ic_window: 计算 IC 的最近窗口天数(-1 表示全历史)。
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
        return {n: 1.0 / len(factor_dfs) for n in factor_dfs}
    return {n: w / total_w for n, w in weights.items()}


def estimate_max_ir_weights(
    factor_dfs: dict[str, pl.DataFrame],
    ret_df: pl.DataFrame,
    lookback: int = 120,
) -> dict[str, float] | None:
    """最大化 IR 闭式解 w = Σ⁻¹μ(Ledoit-Wolf 收缩)。数据不足返回 None(调用方退化等权)。"""
    names = list(factor_dfs.keys())
    k = len(names)
    ic_series_map: dict[str, np.ndarray] = {}
    min_len: int | None = None
    for name, df in factor_dfs.items():
        ic = _compute_ic_series(df, ret_df, name)
        ic_series_map[name] = ic
        if min_len is None or len(ic) < min_len:
            min_len = len(ic)
    if min_len is None or min_len < k + 1:
        return None
    tail_len = min(lookback, min_len)
    ic_mat = np.column_stack([ic_series_map[n][-tail_len:] for n in names])  # (T, K)
    mu = ic_mat.mean(axis=0)
    try:
        from sklearn.covariance import LedoitWolf  # type: ignore[import]

        sigma = LedoitWolf().fit(ic_mat).covariance_
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
    return dict(zip(names, (w_pos / w_pos.sum()).tolist(), strict=True))


def equal_weight(factor_dfs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """等权合成(薄包装:估等权 + 应用)。"""
    return apply_weights(factor_dfs, estimate_equal_weights(factor_dfs))


def ic_weighted(
    factor_dfs: dict[str, pl.DataFrame],
    ret_df: pl.DataFrame,
    ic_window: int = 60,
) -> pl.DataFrame:
    """IC 加权合成(薄包装;in-sample 研究口径,OOS 请用 oos.combine_oos)。"""
    return apply_weights(factor_dfs, estimate_ic_weights(factor_dfs, ret_df, ic_window))


def max_ir(
    factor_dfs: dict[str, pl.DataFrame],
    ret_df: pl.DataFrame,
    lookback: int = 120,
) -> pl.DataFrame:
    """最大化 IR 合成(薄包装;数据不足退化等权)。"""
    w = estimate_max_ir_weights(factor_dfs, ret_df, lookback)
    if w is None:
        return equal_weight(factor_dfs)
    return apply_weights(factor_dfs, w)

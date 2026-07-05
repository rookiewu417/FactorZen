"""滚动样本外(OOS)多因子组合器。

逐折用 train 段估权、test 段应用,拼接为完整 OOS 组合因子——消除 methods 里
「全样本估权」的样本内偏差(README 自承的缺陷)。估权只用 train 的因子+收益,
应用只用 test 的因子(不碰收益),配合 PurgedWalkForwardCV 的 purge/embargo 防泄漏。
"""
from __future__ import annotations

from typing import Any

import polars as pl

from factorzen.research.combination.cv import PurgedWalkForwardCV
from factorzen.research.combination.methods import (
    apply_weights,
    estimate_equal_weights,
    estimate_ic_weights,
    estimate_max_ir_weights,
)


def _estimate_fold(
    method: str,
    all_factor_dfs: dict[str, pl.DataFrame],
    train_factor_dfs: dict[str, pl.DataFrame],
    train_ret: pl.DataFrame,
    kwargs: dict[str, Any],
) -> dict[str, float]:
    if method == "equal_weight":
        return estimate_equal_weights(all_factor_dfs)
    if method == "ic_weighted":
        return estimate_ic_weights(train_factor_dfs, train_ret, **kwargs)
    if method == "max_ir":
        w = estimate_max_ir_weights(train_factor_dfs, train_ret, **kwargs)
        return w if w is not None else estimate_equal_weights(all_factor_dfs)
    raise ValueError(f"未知 method: {method}(支持 equal_weight/ic_weighted/max_ir)")


def combine_oos(
    factor_dfs: dict[str, pl.DataFrame],
    ret_df: pl.DataFrame,
    cv: PurgedWalkForwardCV,
    method: str,
    **method_kwargs: Any,
) -> pl.DataFrame:
    """逐折 train 估权 → test 应用,拼接为样本外组合因子(带 fold_id)。

    Args:
        factor_dfs: {name: DataFrame(trade_date, ts_code, factor_value)}
        ret_df: DataFrame(trade_date, ts_code, ret) — 前向收益
        cv: 切分协议
        method: equal_weight | ic_weighted | max_ir
        method_kwargs: 透传给估权(如 ic_window / lookback)

    Returns:
        DataFrame(trade_date, ts_code, factor_value, fold_id)
    """
    if not factor_dfs:
        raise ValueError("factor_dfs 不能为空")
    # 统一 trade_date 为字符串,保证与 cv 切分的日期类型一致
    fdfs = {
        n: df.with_columns(pl.col("trade_date").cast(pl.Utf8))
        for n, df in factor_dfs.items()
    }
    rdf = ret_df.with_columns(pl.col("trade_date").cast(pl.Utf8))
    all_dates = sorted({d for df in fdfs.values() for d in df["trade_date"].to_list()})

    parts: list[pl.DataFrame] = []
    for fid, (train_dates, test_dates) in enumerate(cv.split(all_dates)):
        train_f = {
            n: df.filter(pl.col("trade_date").is_in(train_dates)) for n, df in fdfs.items()
        }
        train_r = rdf.filter(pl.col("trade_date").is_in(train_dates))
        weights = _estimate_fold(method, fdfs, train_f, train_r, method_kwargs)
        test_f = {
            n: df.filter(pl.col("trade_date").is_in(test_dates)) for n, df in fdfs.items()
        }
        combined = apply_weights(test_f, weights).with_columns(
            pl.lit(fid).alias("fold_id")
        )
        parts.append(combined)

    if not parts:
        return pl.DataFrame(
            schema={
                "trade_date": pl.Utf8,
                "ts_code": pl.Utf8,
                "factor_value": pl.Float64,
                "fold_id": pl.Int32,
            }
        )
    return pl.concat(parts)

"""滚动样本外(OOS)多因子组合 + 逐折骨架。

for_each_fold 是 combine_oos(线性权重)与 combine_lgbm(树模型)共用的逐折骨架:
逐折 filter train/test → 对每折调 fold_fn → 拼接加 fold_id。估权/训练只用 train,
应用/预测只用 test 因子(不碰收益),配合 CV 的 purge/embargo 防泄漏。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl

from factorzen.research.combination.cv import PurgedWalkForwardCV
from factorzen.research.combination.methods import (
    apply_weights,
    estimate_equal_weights,
    estimate_ic_weights,
    estimate_max_ir_weights,
)

# fold_fn(all_factor_dfs, train_factor_dfs, train_ret, test_factor_dfs) -> df(trade_date,ts_code,factor_value)
FoldFn = Callable[
    [
        dict[str, pl.DataFrame],
        dict[str, pl.DataFrame],
        pl.DataFrame,
        dict[str, pl.DataFrame],
    ],
    pl.DataFrame,
]

_EMPTY_SCHEMA = {
    "trade_date": pl.Utf8,
    "ts_code": pl.Utf8,
    "factor_value": pl.Float64,
    "fold_id": pl.Int32,
}


def drop_degenerate_factors(
    factor_dfs: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """剔除无法贡献信号的退化因子:0 行(物化为空)或 factor_value 全缺。

    因子库常混入这类因子(如陈旧基本面经 ``ts_skew`` 退化成全 NaN 空帧)。留着它们会
    在 inner join 时拖垮整个面板;组合器应丢弃后用其余因子继续,而非崩掉整个 OOS run。
    """
    kept: dict[str, pl.DataFrame] = {}
    for name, df in factor_dfs.items():
        if df.height == 0 or "factor_value" not in df.columns:
            continue
        if df["factor_value"].null_count() >= df.height:  # 全缺
            continue
        kept[name] = df
    return kept


def for_each_fold(
    factor_dfs: dict[str, pl.DataFrame],
    ret_df: pl.DataFrame,
    cv: PurgedWalkForwardCV,
    fold_fn: FoldFn,
) -> pl.DataFrame:
    """逐折切分 train/test,对每折调 fold_fn,拼接结果并标 fold_id。"""
    if not factor_dfs:
        raise ValueError("factor_dfs 不能为空")
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
        test_f = {
            n: df.filter(pl.col("trade_date").is_in(test_dates)) for n, df in fdfs.items()
        }
        combined = fold_fn(fdfs, train_f, train_r, test_f)
        parts.append(combined.with_columns(pl.lit(fid).alias("fold_id")))

    if not parts:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)
    return pl.concat(parts)


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
    """线性权重滚动 OOS 组合(equal_weight/ic_weighted/max_ir)。"""
    factor_dfs = drop_degenerate_factors(factor_dfs)
    if not factor_dfs:
        raise ValueError("去除全缺因子后无有效因子,无法组合")

    def _fold(all_f, train_f, train_r, test_f):
        weights = _estimate_fold(method, all_f, train_f, train_r, method_kwargs)
        return apply_weights(test_f, weights)

    return for_each_fold(factor_dfs, ret_df, cv, _fold)

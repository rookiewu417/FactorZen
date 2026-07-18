"""滚动样本外(OOS)多因子组合 + 逐折骨架。

for_each_fold 是 combine_oos(线性权重)与 combine_lgbm(树模型)共用的逐折骨架:
逐折 filter train/test → 对每折调 fold_fn → 拼接加 fold_id。估权/训练只用 train,
应用/预测只用 test 因子(不碰收益),配合 CV 的 purge/embargo 防泄漏。

性能:combine_oos 对 ic_weighted/max_ir 全样本预计算 IC 序列,按 train 日期切片估权;
截面 z-score 全样本一次,test 切片直接加权。数值与逐折重算一致(按日独立)。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl

from factorzen.research.combination.cv import PurgedWalkForwardCV
from factorzen.research.combination.methods import (
    IcCache,
    apply_weights,
    build_ic_cache,
    estimate_equal_weights,
    estimate_ic_weights,
    estimate_max_ir_weights,
    pre_zscore_factors,
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

    ``CompactLibraryPool``：丢全 null 列，仍返回 compact（不物化 dict）。
    ``HybridLibraryPool``：基线 drop + extras 按长表规则。
    """
    from factorzen.research.combination.pool import CompactLibraryPool, HybridLibraryPool

    if isinstance(factor_dfs, CompactLibraryPool):
        return factor_dfs.drop_degenerate()  # type: ignore[return-value]
    if isinstance(factor_dfs, HybridLibraryPool):
        base = factor_dfs.base.drop_degenerate()
        extras = drop_degenerate_factors(factor_dfs.extras)
        if not extras:
            return base  # type: ignore[return-value]
        return base.with_extra_factors(extras)  # type: ignore[return-value]

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
    # 并集日期(覆盖异质);用 polars unique 避免 Python 层 to_list 全量扫
    all_dates = (
        pl.concat([df.select("trade_date") for df in fdfs.values()])
        .unique()
        .sort("trade_date")["trade_date"]
        .to_list()
    )

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
    *,
    ic_cache: IcCache | None = None,
    train_dates: set[str] | None = None,
) -> dict[str, float]:
    if method == "equal_weight":
        return estimate_equal_weights(all_factor_dfs)
    if method == "ic_weighted":
        return estimate_ic_weights(
            train_factor_dfs,
            train_ret,
            ic_cache=ic_cache,
            train_dates=train_dates,
            **{k: v for k, v in kwargs.items() if k in ("ic_window",)},
        )
    if method == "max_ir":
        w = estimate_max_ir_weights(
            train_factor_dfs,
            train_ret,
            ic_cache=ic_cache,
            train_dates=train_dates,
            **{k: v for k, v in kwargs.items() if k in ("lookback",)},
        )
        return w if w is not None else estimate_equal_weights(all_factor_dfs)
    raise ValueError(f"未知 method: {method}(支持 equal_weight/ic_weighted/max_ir)")


def combine_oos(
    factor_dfs: dict[str, pl.DataFrame],
    ret_df: pl.DataFrame,
    cv: PurgedWalkForwardCV,
    method: str,
    *,
    ic_cache: IcCache | None = None,
    z_factor_dfs: dict[str, pl.DataFrame] | None = None,
    **method_kwargs: Any,
) -> pl.DataFrame:
    """线性权重滚动 OOS 组合(equal_weight/ic_weighted/max_ir)。

    Args:
        ic_cache: 可选跨方法共享的全样本 IC 缓存;None 时本函数内按需构建。
        z_factor_dfs: 可选预 z-score 因子面板;None 时本函数内构建。
        method_kwargs: 传给估权的额外参数(ic_window / lookback)。
    """
    factor_dfs = drop_degenerate_factors(factor_dfs)
    if not factor_dfs:
        raise ValueError("去除全缺因子后无有效因子,无法组合")

    # 预计算:z-score 与 IC 全样本一次(按日独立 → 与逐折重算数值等价)
    zdfs = z_factor_dfs if z_factor_dfs is not None else pre_zscore_factors(factor_dfs)
    # 只保留仍存活的因子键
    zdfs = {n: zdfs[n] for n in factor_dfs if n in zdfs}
    cache = ic_cache
    if method in ("ic_weighted", "max_ir") and cache is None:
        cache = build_ic_cache(factor_dfs, ret_df)

    def _fold(all_f, train_f, train_r, test_f):
        train_dates = set(train_r["trade_date"].cast(pl.Utf8).to_list())
        weights = _estimate_fold(
            method,
            all_f,
            train_f,
            train_r,
            method_kwargs,
            ic_cache=cache,
            train_dates=train_dates,
        )
        # test 切片走预 z-score 面板,跳过重复截面标准化
        test_dates = (
            next(iter(test_f.values()))["trade_date"].cast(pl.Utf8).unique().to_list()
            if test_f
            else []
        )
        test_z = {
            n: zdfs[n].filter(pl.col("trade_date").is_in(test_dates)) for n in factor_dfs
        }
        return apply_weights(test_z, weights, already_zscored=True)

    return for_each_fold(factor_dfs, ret_df, cv, _fold)

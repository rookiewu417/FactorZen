"""LightGBM 多因子组合模型。

把「各因子截面值 → 预测下期收益排序」交给梯度提升树学习(捕捉非线性/交互),
标签用截面 rank 归一(稳健、对齐 RankIC 目标)。滚动训练走 oos.for_each_fold:
train 折 fit、test 折 predict,与线性方法共用同一防泄漏骨架。
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import polars as pl

from factorzen.research.combination.cv import PurgedWalkForwardCV
from factorzen.research.combination.oos import for_each_fold


def _factor_panel(factor_dfs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """各因子 inner join 成宽表 [trade_date, ts_code, <name>...]。"""
    merged: pl.DataFrame | None = None
    for name, df in factor_dfs.items():
        d = df.select(["trade_date", "ts_code", "factor_value"]).rename(
            {"factor_value": name}
        )
        merged = d if merged is None else merged.join(d, on=["trade_date", "ts_code"], how="inner")
    assert merged is not None  # factor_dfs 非空由调用方保证
    return merged


def build_panel(
    factor_dfs: dict[str, pl.DataFrame], ret_df: pl.DataFrame
) -> pl.DataFrame:
    """因子宽表 join 前向收益,丢弃缺值行;丢弃率 >30% 告警。"""
    panel = _factor_panel(factor_dfs).join(
        ret_df.select(["trade_date", "ts_code", "ret"]),
        on=["trade_date", "ts_code"],
        how="inner",
    )
    before = panel.height
    panel = panel.drop_nulls()
    if before > 0 and (before - panel.height) / before > 0.3:
        warnings.warn(
            f"build_panel 丢弃 {(before - panel.height) / before:.0%} 行(缺值),样本可能不足",
            stacklevel=2,
        )
    return panel


def _rank_label(panel: pl.DataFrame) -> pl.Series:
    """前向收益截面 rank 归一到 [-0.5, 0.5](单元素组记 0)。"""
    n = pl.col("ret").count().over("trade_date")
    return panel.with_columns(
        pl.when(n > 1)
        .then((pl.col("ret").rank().over("trade_date") - 1) / (n - 1) - 0.5)
        .otherwise(0.0)
        .alias("_y")
    )["_y"]


class LGBMCombiner:
    """LightGBM 回归组合器(确定性:同 seed 输出可复现)。"""

    def __init__(
        self,
        *,
        num_leaves: int = 31,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        min_child_samples: int = 100,
        seed: int = 0,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.params: dict[str, Any] = {
            "num_leaves": num_leaves,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 1,
            "seed": seed,
            "verbosity": -1,
            "importance_type": "gain",
        }
        if params:
            self.params.update(params)
        self._model: Any = None
        self._feature_names: list[str] = []

    def fit(self, X: pl.DataFrame, y: pl.Series) -> None:
        import lightgbm as lgb

        self._feature_names = X.columns
        self._model = lgb.LGBMRegressor(**self.params)
        # 传 pandas(带列名)保证 fit/predict feature names 一致,消除 sklearn 校验告警
        self._model.fit(X.to_pandas(), y.to_pandas())

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("须先 fit 再 predict")
        return np.asarray(self._model.predict(X.to_pandas()))

    def importances(self) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("须先 fit 再取 importances")
        imp = self._model.feature_importances_
        return dict(zip(self._feature_names, imp.tolist(), strict=True))


def combine_lgbm(
    factor_dfs: dict[str, pl.DataFrame],
    ret_df: pl.DataFrame,
    cv: PurgedWalkForwardCV,
    **model_kwargs: Any,
) -> pl.DataFrame:
    """LightGBM 滚动 OOS 组合:train 折 fit(rank 标签)、test 折 predict。"""
    if not factor_dfs:
        raise ValueError("factor_dfs 不能为空")
    names = list(factor_dfs.keys())

    def _fold(all_f, train_f, train_r, test_f):
        train_panel = build_panel(train_f, train_r)
        if train_panel.height == 0:
            raise ValueError("训练面板为空(因子全缺值或无对齐样本)")
        model = LGBMCombiner(**model_kwargs)
        model.fit(train_panel.select(names), _rank_label(train_panel))
        test_panel = _factor_panel(test_f).drop_nulls(subset=names)
        preds = model.predict(test_panel.select(names))
        return test_panel.select(["trade_date", "ts_code"]).with_columns(
            pl.Series("factor_value", preds)
        )

    return for_each_fold(factor_dfs, ret_df, cv, _fold)

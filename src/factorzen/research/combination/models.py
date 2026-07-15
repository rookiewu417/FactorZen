"""LightGBM 多因子组合模型。

把「各因子截面值 → 预测下期收益排序」交给梯度提升树学习(捕捉非线性/交互),
标签用截面 rank 归一(稳健、对齐 RankIC 目标)。滚动训练走 oos.for_each_fold:
train 折 fit、test 折 predict,与线性方法共用同一防泄漏骨架。

性能:全样本 factor panel 只 outer-join 一次,逐折按日期切片(join 与日期无关,
切片 ≡ 对子集 rebuild)。LGBM 保持 deterministic + num_threads=1 + 固定 seed。

``base_panel`` 共享:lift 批处理中 43 列基线宽面板只构建一次,候选只 join 新列
(语义与全量 rebuild 逐值一致;LGBM 路径不做截面 zscore,因子间无耦合)。
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import polars as pl

from factorzen.research.combination.cv import PurgedWalkForwardCV
from factorzen.research.combination.oos import drop_degenerate_factors, for_each_fold

# 某折 train/test 面板为空时只告警一次(逐折刷屏无意义)。
_warned_empty_fold = False

_META_COLS = frozenset({"trade_date", "ts_code", "ret"})


def _warn_empty_fold_once() -> None:
    global _warned_empty_fold
    if _warned_empty_fold:
        return
    _warned_empty_fold = True
    warnings.warn(
        "combine_lgbm: 某折 train/test 面板为空,已跳过该折(因子在该窗口覆盖不足)",
        stacklevel=3,
    )


def _feature_names(panel: pl.DataFrame) -> list[str]:
    """宽表面板中的因子列(排除键与标签),保持列序。"""
    return [c for c in panel.columns if c not in _META_COLS]


def _factor_panel(factor_dfs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """各因子 **outer join** 成宽表 [trade_date, ts_code, <name>...]。

    覆盖异质时 inner join 会把并集缩到交集甚至塌空;改外连接取并集,缺失特征留空
    (LGBM 原生把 null 当缺失处理,无需插补)。
    """
    merged: pl.DataFrame | None = None
    for name, df in factor_dfs.items():
        d = (
            df.select(["trade_date", "ts_code", "factor_value"])
            .with_columns(pl.col("trade_date").cast(pl.Utf8))
            .rename({"factor_value": name})
        )
        merged = (
            d
            if merged is None
            else merged.join(d, on=["trade_date", "ts_code"], how="full", coalesce=True)
        )
    assert merged is not None  # factor_dfs 非空由调用方保证
    return merged


def _norm_ret(ret_df: pl.DataFrame) -> pl.DataFrame:
    return ret_df.select(["trade_date", "ts_code", "ret"]).with_columns(
        pl.col("trade_date").cast(pl.Utf8)
    )


def _join_ret(feat: pl.DataFrame, ret_df: pl.DataFrame) -> pl.DataFrame:
    """特征宽表 inner join 前向收益,丢弃 ret 空行。"""
    return (
        feat.join(_norm_ret(ret_df), on=["trade_date", "ts_code"], how="inner")
        .filter(pl.col("ret").is_not_null())
    )


def _warn_incomplete(panel: pl.DataFrame) -> None:
    names = _feature_names(panel)
    if panel.height > 0 and names:
        complete = panel.drop_nulls(subset=names).height
        if complete / panel.height < 0.7:
            warnings.warn(
                f"build_panel: 仅 {complete / panel.height:.0%} 行因子齐全,"
                "其余按缺失喂入(因子覆盖异质)",
                stacklevel=3,
            )


def _extend_feat_from_base(
    base_panel: pl.DataFrame,
    new_dfs: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """从 base_panel 抽出特征列,full join 新因子(保持与 _factor_panel 相同的并集语义)。

    base_panel 可含 ret(会被丢掉);新因子按 dict 插入序依次 full join。
    无跨因子耦合:每列独立,行集=并集。
    """
    base_feat = (
        base_panel.drop("ret") if "ret" in base_panel.columns else base_panel
    )
    if not new_dfs:
        return base_feat
    new_feat = _factor_panel(new_dfs)
    return base_feat.join(
        new_feat, on=["trade_date", "ts_code"], how="full", coalesce=True
    )


def build_panel(
    factor_dfs: dict[str, pl.DataFrame],
    ret_df: pl.DataFrame,
    *,
    base_panel: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """因子宽表 join 前向收益(标签)。保留特征缺失行交给 LGBM 原生 NaN 处理,
    仅要求标签存在;完整行占比过低时告警(覆盖异质提示)。

    ``base_panel``: 已构建的基线宽面板(可含 ret)。传入时只对**不在** base 列中的
    新因子做 full join,再重新 join ret——与全量 ``_factor_panel`` + ret 逐值一致
    (含「候选行超出基线行集」:并集行 + ret 回填)。
    """
    if base_panel is None:
        if not factor_dfs:
            raise ValueError("factor_dfs 不能为空")
        panel = _join_ret(_factor_panel(factor_dfs), ret_df)
        _warn_incomplete(panel)
        return panel

    base_feats = set(_feature_names(base_panel))
    new_dfs = {k: v for k, v in factor_dfs.items() if k not in base_feats}
    feat = _extend_feat_from_base(base_panel, new_dfs)
    panel = _join_ret(feat, ret_df)
    _warn_incomplete(panel)
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
    *,
    base_panel: pl.DataFrame | None = None,
    **model_kwargs: Any,
) -> pl.DataFrame:
    """LightGBM 滚动 OOS 组合:train 折 fit(rank 标签)、test 折 predict。

    先剔除退化因子(空/全缺),再逐折外连接容缺;某折面板为空则跳过(不崩)。
    全样本 factor panel / labeled panel 只构建一次,逐折按日期切片。

    ``base_panel``: 已完成基线因子 join(+ret) 的宽面板。传入时跳过基线列重建,
    只对不在 base 中的新因子 full join。``factor_dfs`` 可只含新因子,也可含全量
    (基线键会被识别并跳过重建)。None → 与旧路径逐字节一致。
    """
    if base_panel is None:
        if not factor_dfs:
            raise ValueError("factor_dfs 不能为空")
        factor_dfs = drop_degenerate_factors(factor_dfs)
        if not factor_dfs:
            raise ValueError("去除全缺因子后无有效因子,无法组合")
        names = list(factor_dfs.keys())
        fold_dfs = factor_dfs
        full_feat = _factor_panel(factor_dfs)
        full_panel = build_panel(factor_dfs, ret_df)
    else:
        factor_dfs = drop_degenerate_factors(factor_dfs) if factor_dfs else {}
        base_names = _feature_names(base_panel)
        if not base_names and not factor_dfs:
            raise ValueError("base_panel 无因子列且 factor_dfs 为空,无法组合")
        base_set = set(base_names)
        new_dfs = {k: v for k, v in factor_dfs.items() if k not in base_set}
        # 调用方传了基线键 → 按 factor_dfs 插入序;只传新因子 → base 列序 + 新键
        if factor_dfs and any(k in base_set for k in factor_dfs):
            names = list(factor_dfs.keys())
            fold_dfs = factor_dfs
        else:
            names = list(base_names) + list(new_dfs.keys())
            if not names:
                raise ValueError("去除全缺因子后无有效因子,无法组合")
            # for_each_fold 日期并集须覆盖基线+新因子(候选行可超出基线)
            if base_names:
                anchor = base_names[0]
                fold_dfs = {
                    anchor: base_panel.select(
                        [
                            "trade_date",
                            "ts_code",
                            pl.col(anchor).alias("factor_value"),
                        ]
                    ),
                    **new_dfs,
                }
            else:
                fold_dfs = dict(new_dfs)
            if not fold_dfs:
                # 仅 base 且 base 有列但 new 空:用 anchor
                fold_dfs = {
                    base_names[0]: base_panel.select(
                        [
                            "trade_date",
                            "ts_code",
                            pl.col(base_names[0]).alias("factor_value"),
                        ]
                    )
                }

        full_feat = _extend_feat_from_base(base_panel, new_dfs)
        full_panel = _join_ret(full_feat, ret_df)
        _warn_incomplete(full_panel)

    _empty = pl.DataFrame(
        schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8, "factor_value": pl.Float64}
    )

    def _fold(all_f, train_f, train_r, test_f):
        train_dates = train_r["trade_date"].cast(pl.Utf8).unique().to_list()
        # 各 test 因子日期并集：与 full_feat 行集对齐，覆盖「候选行超出基线」。
        # 因子日期同质时 ≡ 取首个因子（旧路径）；异质时更完整且与 base_panel 增量路径一致。
        if test_f:
            test_dates = (
                pl.concat(
                    [
                        df.select(pl.col("trade_date").cast(pl.Utf8))
                        for df in test_f.values()
                    ]
                )
                .unique()["trade_date"]
                .to_list()
            )
        else:
            test_dates = []
        train_panel = full_panel.filter(pl.col("trade_date").is_in(train_dates))
        # 特征全缺的 test 行无任何信号,丢弃;其余按 NaN 交给 LGBM
        test_panel = full_feat.filter(pl.col("trade_date").is_in(test_dates)).filter(
            ~pl.all_horizontal([pl.col(n).is_null() for n in names])
        )
        if train_panel.height == 0 or test_panel.height == 0:
            _warn_empty_fold_once()
            return _empty
        model = LGBMCombiner(**model_kwargs)
        model.fit(train_panel.select(names), _rank_label(train_panel))
        preds = model.predict(test_panel.select(names))
        return test_panel.select(["trade_date", "ts_code"]).with_columns(
            pl.Series("factor_value", preds)
        )

    return for_each_fold(fold_dfs, ret_df, cv, _fold)

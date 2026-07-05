"""因子重要性归因:SHAP(可选,更忠实)+ LightGBM gain 兜底。

method='auto':shap 可用则用 TreeExplainer 的 mean(|SHAP|),否则退回 gain。
shap 只在 dev 依赖,缺失时自动降级——报告如实标注实际用了哪种方法(method 列)。
"""
from __future__ import annotations

import numpy as np
import polars as pl

from factorzen.research.combination.models import LGBMCombiner


def explain(
    combiner: LGBMCombiner, x: pl.DataFrame, *, method: str = "auto"
) -> pl.DataFrame:
    """计算因子重要性。

    Args:
        combiner: 已 fit 的 LGBMCombiner。
        x: 特征宽表(shap 计算用样本;列须与 fit 一致)。
        method: auto | shap | gain。

    Returns:
        DataFrame(factor, importance, method) — method 标注实际所用方法。
    """
    if method not in ("auto", "shap", "gain"):
        raise ValueError(f"未知 method: {method}(auto/shap/gain)")

    if method in ("auto", "shap"):
        try:
            import shap

            explainer = shap.TreeExplainer(combiner._model)
            vals = np.asarray(explainer.shap_values(x.to_pandas()))
            imp = np.abs(vals).mean(axis=0)
            return pl.DataFrame(
                {
                    "factor": combiner._feature_names,
                    "importance": imp.tolist(),
                    "method": "shap",
                }
            )
        except ImportError:
            if method == "shap":
                raise

    imp_dict = combiner.importances()
    return pl.DataFrame(
        {
            "factor": list(imp_dict.keys()),
            "importance": list(imp_dict.values()),
            "method": "gain",
        }
    )

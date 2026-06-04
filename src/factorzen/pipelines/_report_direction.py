"""report 流程的回测方向判定：依据 IC 显著性决定是否反向。"""

from __future__ import annotations

import json
from typing import Any

import polars as pl

from factorzen.daily.evaluation.ic_analysis import ICAnalysisResult
from factorzen.pipelines._report_persistence import _meta_path


def _decide_backtest_direction(ic_result: ICAnalysisResult) -> dict[str, Any]:
    """Decide whether the report backtest should invert factor direction."""
    ic_mean = float(getattr(ic_result, "ic_mean", 0.0) or 0.0)
    ic_tstat = float(getattr(ic_result, "ic_tstat", 0.0) or 0.0)
    ic_pvalue = float(getattr(ic_result, "ic_pvalue", 1.0) or 1.0)
    oos_ic = getattr(ic_result, "oos_ic", {}) or {}
    oos_train = oos_ic.get("train")
    oos_test = oos_ic.get("test")
    oos_both_negative = (
        oos_train is not None and oos_test is not None and oos_train < 0 and oos_test < 0
    )

    statistically_negative = ic_mean < 0 and ic_pvalue <= 0.10
    if statistically_negative or (ic_mean < 0 and oos_both_negative):
        reason = (
            "IC 均值为负且 p 值小于等于 0.10"
            if statistically_negative
            else "IC 均值为负，且历史观察期/未来验证期 IC 均为负"
        )
        return {
            "direction": "reversed",
            "should_reverse": True,
            "reason": reason,
            "ic_mean": ic_mean,
            "ic_tstat": ic_tstat,
            "ic_pvalue": ic_pvalue,
            "oos_train_ic": oos_train,
            "oos_test_ic": oos_test,
        }

    reason = (
        "IC 均值为负，但显著性或样本外一致性不足，保持原方向"
        if ic_mean < 0
        else "IC 均值非负，保持原方向"
    )
    return {
        "direction": "normal",
        "should_reverse": False,
        "reason": reason,
        "ic_mean": ic_mean,
        "ic_tstat": ic_tstat,
        "ic_pvalue": ic_pvalue,
        "oos_train_ic": oos_train,
        "oos_test_ic": oos_test,
    }


def _apply_backtest_direction(
    clean_df: pl.DataFrame, decision: dict[str, Any] | None
) -> pl.DataFrame:
    """Flip factor_clean for backtesting when the IC decision requires reverse direction."""
    if not decision or decision.get("direction") != "reversed":
        return clean_df
    return clean_df.with_columns((-pl.col("factor_clean")).alias("factor_clean"))


def _load_backtest_direction(factor_name: str, start: str, end: str) -> dict[str, Any] | None:
    mp = _meta_path(factor_name, start, end)
    if not mp.exists():
        return None
    meta = json.loads(mp.read_text(encoding="utf-8"))
    direction = meta.get("backtest_direction")
    return direction if isinstance(direction, dict) else None

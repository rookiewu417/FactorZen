# src/factorzen/agents/evaluation.py
"""把 LLM 产出的表达式字符串批量评估为 Rank IC/IR。
全部用 discovery 的公开接口，不重构 run_session（零回归）。"""
from __future__ import annotations

import polars as pl

from factorzen.discovery.expression import evaluate as eval_node
from factorzen.discovery.expression import parse_expr, to_expr_string
from factorzen.discovery.scoring import quick_fitness


def _preprocess_daily(daily: pl.DataFrame) -> pl.DataFrame:
    """补全 *_adj 等列名映射，使 evaluate() 能找到正确列。
    测试/生产 daily 若只含 close 而无 close_adj，自动 rename。"""
    renames: dict[str, str] = {}
    for base in ("close", "open", "high", "low"):
        adj = f"{base}_adj"
        if adj not in daily.columns and base in daily.columns:
            renames[base] = adj
    if renames:
        daily = daily.rename(renames)
    extras = []
    if "vwap" not in daily.columns and "amount" in daily.columns and "vol" in daily.columns:
        extras.append((pl.col("amount") / pl.col("vol")).alias("vwap"))
    if "log_vol" not in daily.columns and "vol" in daily.columns:
        extras.append((pl.col("vol") + 1.0).log().alias("log_vol"))
    if extras:
        daily = daily.with_columns(extras)
    return daily


def _node_to_factor_df(node, daily: pl.DataFrame) -> pl.DataFrame:
    """用公开 evaluate(node, df) 算因子值，组装成 [trade_date, ts_code, factor_value]。"""
    df = _preprocess_daily(daily).sort(["ts_code", "trade_date"])
    series = eval_node(node, df)
    return (
        df.select(["trade_date", "ts_code"])
        .with_columns(series.alias("factor_value"))
        .filter(pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
    )


def evaluate_expressions(
    expr_strs: list[str], daily: pl.DataFrame, bundle
) -> list[dict]:
    """批量评估表达式集。非法表达式（parse_expr 抛 ValueError）记 compile_ok=False。"""
    results: list[dict] = []
    for s in expr_strs:
        try:
            node = parse_expr(s)
        except ValueError as exc:
            results.append(
                {
                    "expression": s,
                    "node": None,
                    "compile_ok": False,
                    "ic_train": None,
                    "ir_train": None,
                    "error": str(exc),
                }
            )
            continue
        try:
            fdf = _node_to_factor_df(node, daily)
            fit = quick_fitness(fdf, bundle, segment="train")
            results.append(
                {
                    "expression": to_expr_string(node),
                    "node": node,
                    "compile_ok": True,
                    "ic_train": float(fit["ic_mean"]),
                    "ir_train": float(fit["ir"]),
                    "error": None,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "expression": to_expr_string(node),
                    "node": node,
                    "compile_ok": True,
                    "ic_train": None,
                    "ir_train": None,
                    "error": str(exc),
                }
            )
    return results

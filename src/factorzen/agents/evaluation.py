# src/factorzen/agents/evaluation.py
"""把 LLM 产出的表达式字符串批量评估为 Rank IC/IR。
全部用 discovery 的公开接口，不重构 run_session（零回归）。"""
from __future__ import annotations

import polars as pl

from factorzen.discovery.derived import add_derived_columns
from factorzen.discovery.expression import evaluate as eval_node
from factorzen.discovery.expression import parse_expr, to_expr_string
from factorzen.discovery.scoring import quick_fitness

_PRICE_COLS = ("close", "open", "high", "low", "vol", "amount",
               "close_adj", "open_adj", "high_adj", "low_adj")


def _preprocess_daily(daily: pl.DataFrame) -> pl.DataFrame:
    """把评估帧准备成与 run_session/ExpressionFactor **同一套** prep（复权价 + 停牌掩码 +
    全套派生列），使 LLM 被广告的 22 个叶子（含 ret_1d/amplitude/total_mv 等）在评估帧里
    真实存在，消除 agent 评估路径与挖掘搜索路径的双路径漂移。

    复权价：生产由 CLI 经 FactorDataContext 提供真实 ``*_adj``（不 fake）；仅合成测试帧缺
    ``*_adj``/``pre_close`` 时回退未复权价/前一日 close——生产路径已提供真实列、不触发。
    """
    df = daily
    for base in ("close", "open", "high", "low"):
        adj = f"{base}_adj"
        if adj not in df.columns and base in df.columns:
            df = df.with_columns(pl.col(base).alias(adj))
    df = df.sort(["ts_code", "trade_date"])
    if "pre_close" not in df.columns and "close" in df.columns:
        df = df.with_columns(
            pl.col("close").shift(1).over("ts_code").fill_null(pl.col("close")).alias("pre_close")
        )
    # 停牌掩码：vol==0 行价量列置 null（与挖掘路径一致）
    df = df.with_columns([
        pl.when(pl.col("vol") > 0).then(pl.col(c)).otherwise(None).alias(c)
        for c in _PRICE_COLS if c in df.columns
    ])
    return add_derived_columns(df)


def _node_to_factor_df(node, daily: pl.DataFrame) -> pl.DataFrame:
    """用公开 evaluate(node, df) 算因子值，组装成 [trade_date, ts_code, factor_value]。"""
    df = _preprocess_daily(daily)
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

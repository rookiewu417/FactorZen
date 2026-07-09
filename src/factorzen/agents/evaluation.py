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
    """把评估帧准备成与 run_session/ExpressionFactor 同一套 prep（复权价 + 停牌掩码 + 全套派生列）。"""
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


def make_health_check(daily: pl.DataFrame, *, max_null_ratio: float = 0.5):
    """建一个「表达式 → 诊断信息 | None」的检查器，供自愈循环回灌 LLM。

    对齐 CoSTEER 的评估器：它在沙箱里真正执行代码，把 **Traceback 和 NaN 比例** 交回给模型修正。
    本项目是 DSL，无 exec 沙箱，故在求值层取同样两类信号：求值抛的异常、以及因子值的
    null/NaN 占比。`div(close, sub(close, close))` 这类 parse 通过却全 null 的「静默失明」
    表达式（PR #61 嵌套 .over() bug 同型），旧循环只查 parse，一次修正机会都不给。

    返回 None 表示健康。daily 只在建检查器时预处理一次（`add_derived_columns` 较重）。
    """
    df = _preprocess_daily(daily)

    def check(expr: str) -> str | None:
        try:
            node = parse_expr(expr)
        except ValueError as exc:
            return f"解析失败: {exc}"
        try:
            series = eval_node(node, df)
        except Exception as exc:
            return f"求值失败: {type(exc).__name__}: {exc}"
        n = series.len()
        if n == 0:
            return "求值结果为空序列，无任何因子值"
        # polars: null 与 NaN 是两回事；is_nan() 遇 null 返回 null，须 fill_null(False)
        n_null = int(series.is_null().sum())
        n_nan = int(series.is_nan().fill_null(False).sum()) if series.dtype.is_float() else 0
        ratio = (n_null + n_nan) / n
        if ratio > max_null_ratio:
            return (f"因子值 {ratio:.1%} 为 null/NaN（上限 {max_null_ratio:.0%}），"
                    f"几乎没有有效截面信号；常见成因：分母恒零、窗口长于样本、"
                    f"截面算子套时序算子导致分组键冲突")
        return None

    return check


def _factor_turnover(factor_df: pl.DataFrame, quantile: float = 0.2) -> float | None:
    """纯多头 top-quantile 组合的单边换手率 ∈ [0,1]（交易成本代理，多目标评估用）。

    每日按 factor_value 取 top-⌈n·quantile⌉ 只等权多头，换手率 = 相邻调仓日 0.5·Σ|w_t−w_{t-1}| 均值。
    常数排序→0，每日重排→接近 1。空帧/有效交易日<2/每截面<5 只 → None。
    """
    if factor_df.is_empty():
        return None
    fdf = factor_df.filter(
        pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite()
    )
    dates = fdf.select("trade_date").unique().sort("trade_date")["trade_date"].to_list()
    if len(dates) < 2:
        return None
    prev: dict[str, float] | None = None
    turnovers: list[float] = []
    for d in dates:
        cross = fdf.filter(pl.col("trade_date") == d)
        n = cross.height
        if n < 5:
            continue
        k = max(1, round(n * quantile))
        top = cross.sort("factor_value", descending=True).head(k)["ts_code"].to_list()
        w = {c: 1.0 / k for c in top}
        if prev is not None:
            keys = set(w) | set(prev)
            l1 = sum(abs(w.get(c, 0.0) - prev.get(c, 0.0)) for c in keys)
            turnovers.append(0.5 * l1)
        prev = w
    if not turnovers:
        return None
    return float(sum(turnovers) / len(turnovers))


def evaluate_expressions(
    expr_strs: list[str], daily: pl.DataFrame, bundle
) -> list[dict]:
    """批量评估表达式集。非法表达式（parse_expr 抛 ValueError）记 compile_ok=False。"""
    results: list[dict] = []
    for s in expr_strs:
        try:
            node = parse_expr(s)
        except ValueError as exc:
            results.append({"expression": s, "node": None, "compile_ok": False,
                            "ic_train": None, "ir_train": None, "turnover": None,
                            "error": str(exc)})
            continue
        try:
            fdf = _node_to_factor_df(node, daily)
            fit = quick_fitness(fdf, bundle, segment="train")
            results.append({"expression": to_expr_string(node), "node": node, "compile_ok": True,
                            "ic_train": float(fit["ic_mean"]), "ir_train": float(fit["ir"]),
                            "turnover": _factor_turnover(fdf), "error": None})
        except Exception as exc:
            results.append({"expression": to_expr_string(node), "node": node, "compile_ok": True,
                            "ic_train": None, "ir_train": None, "turnover": None,
                            "error": str(exc)})
    return results

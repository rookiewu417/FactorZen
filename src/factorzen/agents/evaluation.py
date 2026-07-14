# src/factorzen/agents/evaluation.py
"""把 LLM 产出的表达式字符串批量评估为 Rank IC/IR。
全部用 discovery 的公开接口，不重构 run_session（零回归）。"""
from __future__ import annotations

import logging

import polars as pl

from factorzen.discovery.derived import add_derived_columns
from factorzen.discovery.expression import (
    evaluate_materialized,
    parse_expr,
    to_expr_string,
    warmup_shortfall,
)
from factorzen.discovery.scoring import quick_fitness

_LOG = logging.getLogger(__name__)

_PRICE_COLS = ("close", "open", "high", "low", "vol", "amount",
               "close_adj", "open_adj", "high_adj", "low_adj")


def _preprocess_daily(daily: pl.DataFrame, profile=None) -> pl.DataFrame:
    """把评估帧准备成与 run_session/ExpressionFactor 同一套 prep（复权价 + 停牌掩码 + 全套派生列）。

    ``profile``：市场 profile。``None``（默认）→ A 股 prep（复权价别名 + 停牌掩码 + pre_close +
    `add_derived_columns`），逐字节零回归。非 None → 市场特有派生列
    ``profile.factors.derived_columns``（与 M1 `run_session(profile=...)` 同口径：crypto 无复权、
    无停牌掩码、无 pre_close，只按标的排序后追加 vwap/log_vol/ret_1d/taker_buy_ratio）。
    """
    if profile is not None:
        return profile.factors.derived_columns(daily.sort(["ts_code", "trade_date"]))
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


def _factor_df_from_prepped(node, prepped: pl.DataFrame,
                            eval_start=None, eval_end=None, leaf_map=None) -> pl.DataFrame:
    """在**已 `_preprocess_daily` 过**的帧上求值，裁剪到 [eval_start, eval_end]。

    ``leaf_map``：叶子名→列名映射（默认 None → A 股 `LEAF_FEATURES`）。crypto 等传各自映射，
    使 funding_rate/open_interest 等叶子正确编译到对应列。

    调用方若要对同一帧评估多个表达式，先 `_preprocess_daily` 一次再走这里，
    避免每个表达式重复 `add_derived_columns`（较重）。

    先在整帧（含预热段）上求值、再裁剪，而不是只喂 [eval_start, eval_end] 段：
    滚动算子在段首用截断窗口时，`operators._MIN = 3` 让窗口不满照常出**噪声值**
    而非 NaN——只喂本段会把这段噪声当作真实首段信号留在结果里。求值后裁剪保证
    结果只含扩窗预热后的干净值，且不泄漏 eval_start 之前任何未来不可得的信息（PIT）。
    """
    series = evaluate_materialized(node, prepped, leaf_map)
    # 先在整帧（含预热/非成分日）上物化，再裁评估截面——滚动算子需要连续时序。
    # in_universe 列存在时只保留成分内 (date, stock)；列不存在=未启用 membership，零回归。
    cols = ["trade_date", "ts_code"]
    has_univ = "in_universe" in prepped.columns
    if has_univ:
        cols = [*cols, "in_universe"]
    out = (
        prepped.select(cols)
        .with_columns(series.alias("factor_value"))
        .filter(pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
    )
    if eval_start is not None:
        out = out.filter(pl.col("trade_date") >= eval_start)
    if eval_end is not None:
        out = out.filter(pl.col("trade_date") <= eval_end)
    if has_univ:
        out = out.filter(pl.col("in_universe")).drop("in_universe")
    return out


def _node_to_factor_df(node, daily: pl.DataFrame,
                       eval_start=None, eval_end=None, profile=None, leaf_map=None) -> pl.DataFrame:
    """用公开 evaluate(node, df) 算因子值，组装成 [trade_date, ts_code, factor_value]。

    `eval_start` / `eval_end`：**先在整帧上求值、再裁剪到 [eval_start, eval_end]**（扩窗预热）。
    train 段与 holdout 段都必须这样做——只喂本段会让滚动算子在段首用截断窗口，
    发出偏差值（`operators._MIN = 3`，窗口不满**不产生 NaN**，产生噪声值）。
    train 段漏裁会让预热段进 IC 序列，系统性拖低 train IC，制造「holdout 优于 train」的假象。

    PIT 安全：时序算子的滚动窗口只向过去看，段首日用到的是前一段末尾的数据（≤t）；
    截面算子逐日独立。求值后裁剪保证不保留段外任何行。

    ``profile`` / ``leaf_map``：市场 profile 与叶子映射（默认 None → A 股，零回归）。
    """
    return _factor_df_from_prepped(
        node, _preprocess_daily(daily, profile), eval_start, eval_end, leaf_map)


def make_health_check(daily: pl.DataFrame, *, max_null_ratio: float = 0.5,
                      profile=None, leaf_map=None):
    """建一个「表达式 → 诊断信息 | None」的检查器，供自愈循环回灌 LLM。

    ``profile`` / ``leaf_map``：市场 profile 与叶子映射（默认 None → A 股，零回归）。crypto
    必须传，否则 `parse_expr` 把合法 crypto 叶子（funding_rate 等）判为「解析失败」，让
    健康的 crypto 表达式被自愈循环误当病态送修。

    对齐 CoSTEER 的评估器：它在沙箱里真正执行代码，把 **Traceback 和 NaN 比例** 交回给模型修正。
    本项目是 DSL，无 exec 沙箱，故在求值层取同样两类信号：求值抛的异常、以及因子值的
    null/NaN 占比。`div(close, sub(close, close))` 这类 parse 通过却全 null 的「静默失明」
    表达式（PR #61 嵌套 .over() bug 同型），旧循环只查 parse，一次修正机会都不给。

    返回 None 表示健康。daily 只在建检查器时预处理一次（`add_derived_columns` 较重）。
    """
    df = _preprocess_daily(daily, profile)

    def check(expr: str) -> str | None:
        try:
            node = parse_expr(expr, leaf_map)
        except ValueError as exc:
            return f"解析失败: {exc}"
        try:
            series = evaluate_materialized(node, df, leaf_map)
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
    expr_strs: list[str], daily: pl.DataFrame, bundle,
    *, eval_start=None, eval_end=None, profile=None,
) -> list[dict]:
    """批量评估表达式集。非法表达式（parse_expr 抛 ValueError）记 compile_ok=False。

    ``profile``：市场 profile（默认 None → A 股，逐字节零回归）。非 None 时预处理走
    `profile.factors.derived_columns`、叶子集/映射取 `profile.factors.leaf_features()`，
    透传到 `parse_expr`/`warmup_shortfall`/求值——crypto 表达式（funding_rate 等）方能解析与求值。

    `daily` 是**含预热段的完整帧**；`eval_start`/`eval_end` 是 train 段边界。
    求值在整帧上做、再裁剪到该区间——与 holdout 段同一条路径（`nodes.py` 的
    `_holdout_values`），也与 M1 的 `run_session(eval_start=start)` 同口径。
    漏裁 train 段会让预热噪声进 IC 序列、系统性拖低 train IC，把
    「holdout 优于 train」伪造成「无过拟合」的证据。

    `n_train` = 该因子在 train 段的**有效 IC 天数**（不是日历交易日数），供 DSR 的 n_obs 用，
    与 M1 的 `c["n_train"]` 同口径。

    `n_train == 0`（求值后无任何有效截面）时记 ic/ir=None 而非 `quick_fitness` 返回的
    sentinel `0.0`——否则这类死表达式会以「IC 恰好为 0」的身份混进护栏的 `passed` 集：
    既膨胀多重检验的 N，又把 0.0 灌进 DSR 的 IR 池拉低经验方差，使 deflation 基准算在垃圾上。
    **预热不足的表达式同样记 None**：窗口不满时 `operators._MIN = 3` 让它照常出值（噪声而非 NaN），
    静默放行等于把噪声 IC 灌进 DSR 池。
    """
    if eval_end is not None and eval_start is None:
        raise ValueError(
            "eval_end 不能脱离 eval_start 单独传入：下界裁剪与预热门（`warmup_bars`）"
            "都只在 eval_start is not None 时触发，单传 eval_end 会静默重新引入"
            "预热噪声进 train IC 序列的 bug——两者必须同传或都不传。"
        )

    prepped = _preprocess_daily(daily, profile)  # 整帧只预处理一次（add_derived_columns 较重）
    leaf_map = profile.factors.leaf_features() if profile is not None else None
    results: list[dict] = []
    for s in expr_strs:
        try:
            node = parse_expr(s, leaf_map)
        except ValueError as exc:
            results.append({"expression": s, "node": None, "compile_ok": False,
                            "ic_train": None, "ir_train": None, "turnover": None,
                            "n_train": 0, "error": str(exc)})
            continue

        if eval_start is not None:
            sf = warmup_shortfall(node, prepped, eval_start, leaf_map)
            if sf is not None:
                leaf, need, have = sf
                results.append({
                    "expression": to_expr_string(node), "node": node, "compile_ok": True,
                    "ic_train": None, "ir_train": None, "turnover": None, "n_train": 0,
                    "error": f"预热不足: 叶 {leaf} 需要 {need} 根历史，可用 {have} 根"})
                continue

        try:
            fdf = _factor_df_from_prepped(node, prepped, eval_start, eval_end, leaf_map)
            fit = quick_fitness(fdf, bundle, segment="train")
            n_train = int(fit["n"])
            if n_train == 0:
                results.append({
                    "expression": to_expr_string(node), "node": node, "compile_ok": True,
                    "ic_train": None, "ir_train": None, "turnover": None, "n_train": 0,
                    "error": "求值后 train 段无有效截面（因子值全 null/NaN、分母恒零或窗口长于样本）"})
                continue
            results.append({"expression": to_expr_string(node), "node": node, "compile_ok": True,
                            "ic_train": float(fit["ic_mean"]), "ir_train": float(fit["ir"]),
                            "turnover": _factor_turnover(fdf), "n_train": n_train, "error": None})
        except Exception as exc:
            _LOG.warning("表达式 %s 求值失败: %s: %s", s, type(exc).__name__, exc)
            results.append({"expression": to_expr_string(node), "node": node, "compile_ok": True,
                            "ic_train": None, "ir_train": None, "turnover": None,
                            "n_train": 0, "error": str(exc)})
    return results

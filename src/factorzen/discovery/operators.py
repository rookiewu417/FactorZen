# src/factorzen/discovery/operators.py
"""算子库：每个算子是一个把子表达式（pl.Expr）组合成新 pl.Expr 的工厂。

约定（编译前提）：求值表已按 (ts_code, trade_date) 排序。
- 时序算子(ts)用 .over("ts_code")；截面算子(cs)用 .over("trade_date")；算术(arith)逐元素。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import polars as pl

from factorzen.core.feature_schema import (
    BASIC_FEATURES,
    FLOW_FEATURES,
    FUNDAMENTAL_FEATURES,
    HOLDER_FEATURES,
    LEAF_FEATURES,
    MARGIN_FEATURES,
    TOPLIST_FEATURES,
)

__all__ = [
    "BASIC_FEATURES",
    "FLOW_FEATURES",
    "FUNDAMENTAL_FEATURES",
    "HOLDER_FEATURES",
    "LEAF_FEATURES",
    "MARGIN_FEATURES",
    "OPERATORS",
    "TOPLIST_FEATURES",
]

_MIN = 3  # rolling 最小样本


def _safe_div(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    # is_finite() 显式排除 NaN 分母：polars 中 NaN.abs() > 1e-12 判 True，只查 abs 会让
    # NaN 分母穿透守卫、输出 a/NaN=NaN。
    return pl.when(b.is_finite() & (b.abs() > 1e-12)).then(a / b).otherwise(None)


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    category: Literal["ts", "cs", "arith"]
    arity: int
    has_window: bool
    build: Callable[[list[pl.Expr], int | None], pl.Expr]


def _ts(name, fn):  # window 时序算子
    return OperatorSpec(name, "ts", 1, True, lambda c, w: fn(c[0], w))


def _cs(name, fn):  # 截面算子
    return OperatorSpec(name, "cs", 1, False, lambda c, w: fn(c[0]))


def _ar(name, arity, fn):  # 算术算子
    return OperatorSpec(name, "arith", arity, False, lambda c, w: fn(*c))


def _ts2(name, fn):  # 双输入时序算子（arity 2, 有 window）
    return OperatorSpec(name, "ts", 2, True, lambda c, w: fn(c[0], c[1], w))


def _ts_corr(a: pl.Expr, b: pl.Expr, w: int | None) -> pl.Expr:
    ma = a.rolling_mean(w, min_samples=_MIN).over("ts_code")  # type: ignore[arg-type]
    mb = b.rolling_mean(w, min_samples=_MIN).over("ts_code")  # type: ignore[arg-type]
    mab = (a * b).rolling_mean(w, min_samples=_MIN).over("ts_code")  # type: ignore[arg-type]
    va = (a * a).rolling_mean(w, min_samples=_MIN).over("ts_code") - ma * ma  # type: ignore[arg-type]
    vb = (b * b).rolling_mean(w, min_samples=_MIN).over("ts_code") - mb * mb  # type: ignore[arg-type]
    cov = mab - ma * mb
    # clip(0) 吸收近常数窗口 E[x²]−E[x]² 的浮点微负（否则 va*vb<0 → sqrt=NaN，而
    # NaN>1e-12 判 True 会让守卫放行、输出 NaN）；再叠 is_finite 双保险。
    denom = (va.clip(0.0, None) * vb.clip(0.0, None)).sqrt()
    return (
        pl.when(denom.is_finite() & (denom > 1e-12))
        .then((cov / denom).clip(-1.0, 1.0))
        .otherwise(None)
    )


def _ts_cov(a: pl.Expr, b: pl.Expr, w: int | None) -> pl.Expr:
    ma = a.rolling_mean(w, min_samples=_MIN).over("ts_code")  # type: ignore[arg-type]
    mb = b.rolling_mean(w, min_samples=_MIN).over("ts_code")  # type: ignore[arg-type]
    mab = (a * b).rolling_mean(w, min_samples=_MIN).over("ts_code")  # type: ignore[arg-type]
    return mab - ma * mb


OPERATORS: dict[str, OperatorSpec] = {
    # ── 时序（.over("ts_code")）──
    "ts_mean": _ts("ts_mean", lambda x, w: x.rolling_mean(w, min_samples=_MIN).over("ts_code")),
    "ts_std":  _ts("ts_std",  lambda x, w: x.rolling_std(w, min_samples=_MIN).over("ts_code")),
    "ts_sum":  _ts("ts_sum",  lambda x, w: x.rolling_sum(w, min_samples=_MIN).over("ts_code")),
    "ts_min":  _ts("ts_min",  lambda x, w: x.rolling_min(w, min_samples=_MIN).over("ts_code")),
    "ts_max":  _ts("ts_max",  lambda x, w: x.rolling_max(w, min_samples=_MIN).over("ts_code")),
    # polars 1.41.2 原生 rolling_rank，标记 unstable(升级 polars 时需重验语义)。
    # method="average" 并列取均值排名；除以窗口内**实际非空样本数**归一化到 (0,1]——
    # 除以固定 w 会让历史不足 w 天的股票(warm-up/次新)ts_rank 上限只有 count/w，被系统性压低。
    "ts_rank": _ts("ts_rank", lambda x, w: _safe_div(
        x.rolling_rank(w, method="average", min_samples=_MIN).over("ts_code"),
        x.is_not_null().cast(pl.Int64).rolling_sum(w, min_samples=_MIN).over("ts_code"),
    )),
    "delay":   _ts("delay",   lambda x, w: x.shift(w).over("ts_code")),
    "delta":   _ts("delta",   lambda x, w: (x - x.shift(w)).over("ts_code")),
    "pct_change": _ts("pct_change", lambda x, w:
        (pl.when(x.shift(w) > 1e-12).then(x / x.shift(w) - 1.0).otherwise(None)).over("ts_code")),
    "ts_decay_linear": _ts("ts_decay_linear", lambda x, w:
        x.rolling_mean(w, min_samples=_MIN).over("ts_code")),  # MVP：等权近似线性衰减
    "ts_corr": _ts2("ts_corr", _ts_corr),
    "ts_cov": _ts2("ts_cov", _ts_cov),
    "ts_median": _ts("ts_median", lambda x, w:
        x.rolling_median(w, min_samples=_MIN).over("ts_code")),
    "ts_zscore": _ts("ts_zscore", lambda x, w: _safe_div(
        x - x.rolling_mean(w, min_samples=_MIN).over("ts_code"),
        x.rolling_std(w, min_samples=_MIN).over("ts_code"))),
    # polars 1.41.2 原生 rolling_skew，标记 unstable(升级 polars 时需重验语义)。
    # bias=True = 总体偏度(ddof=0)，对应现有 numpy ground-truth 测试口径。
    # fill_nan(None)：零方差窗口原生实现是 0/0 → NaN 而非 null，必须显式转 null，
    # 否则又是一次"NaN 泄漏"（polars 中 NaN > 阈值 判定为 True，下游 pl.when 防护会被绕过）。
    "ts_skew": _ts("ts_skew", lambda x, w:
        x.rolling_skew(w, bias=True, min_samples=_MIN).over("ts_code").fill_nan(None)),
    # ── 截面（.over("trade_date")）──
    # 分母用 x.count()（非空计数）而非 pl.len()（含 null 行）：x.rank() 只给非空值
    # 1..k 排名，若除以含 null 的总行数，归一化尺度会随当日 null 比例漂移（同样的
    # 非空排名，null 多的日被系统性压小），破坏截面可比性。
    "rank":  _cs("rank",  lambda x: (x.rank().over("trade_date") / (x.count().over("trade_date") + 1))),
    "zscore": _cs("zscore", lambda x:
        _safe_div(x - x.mean().over("trade_date"), x.std().over("trade_date"))),
    "scale": _cs("scale", lambda x: _safe_div(x, x.abs().sum().over("trade_date"))),
    # ── 算术 ──
    "add": _ar("add", 2, lambda a, b: a + b),
    "sub": _ar("sub", 2, lambda a, b: a - b),
    "mul": _ar("mul", 2, lambda a, b: a * b),
    "div": _ar("div", 2, lambda a, b: _safe_div(a, b)),
    "abs": _ar("abs", 1, lambda a: a.abs()),
    "log": _ar("log", 1, lambda a: pl.when(a > 0).then(a.log()).otherwise(None)),
    "sign": _ar("sign", 1, lambda a: a.sign()),
    "sqrt": _ar("sqrt", 1, lambda a: pl.when(a >= 0).then(a.sqrt()).otherwise(None)),
    "neg": _ar("neg", 1, lambda a: -a),
    "inv": _ar("inv", 1, lambda a: _safe_div(pl.lit(1.0), a)),
    "square": _ar("square", 1, lambda a: a * a),
    "max": _ar("max", 2, lambda a, b: pl.max_horizontal(a, b)),
    "min": _ar("min", 2, lambda a, b: pl.min_horizontal(a, b)),
}

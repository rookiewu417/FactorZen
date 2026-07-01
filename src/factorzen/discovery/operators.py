# src/factorzen/discovery/operators.py
"""算子库：每个算子是一个把子表达式（pl.Expr）组合成新 pl.Expr 的工厂。

约定（编译前提）：求值表已按 (ts_code, trade_date) 排序。
- 时序算子(ts)用 .over("ts_code")；截面算子(cs)用 .over("trade_date")；算术(arith)逐元素。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

# 叶子名 → 求值表中的列名。vwap/log_vol/ret_1d 为派生列（ExpressionFactor 预计算）。
LEAF_FEATURES: dict[str, str] = {
    "close": "close_adj", "open": "open_adj", "high": "high_adj", "low": "low_adj",
    "vol": "vol", "amount": "amount", "vwap": "vwap", "log_vol": "log_vol", "ret_1d": "ret_1d",
    "total_mv": "total_mv", "circ_mv": "circ_mv", "pb": "pb", "pe_ttm": "pe_ttm",
    "ps_ttm": "ps_ttm", "dv_ttm": "dv_ttm",
}
BASIC_FEATURES: set[str] = {"total_mv", "circ_mv", "pb", "pe_ttm", "ps_ttm", "dv_ttm"}

_MIN = 3  # rolling 最小样本


def _safe_div(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return pl.when(b.abs() > 1e-12).then(a / b).otherwise(None)


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
    denom = (va * vb).sqrt()
    return pl.when(denom > 1e-12).then((cov / denom).clip(-1.0, 1.0)).otherwise(None)


def _ts_cov(a: pl.Expr, b: pl.Expr, w: int | None) -> pl.Expr:
    ma = a.rolling_mean(w, min_samples=_MIN).over("ts_code")  # type: ignore[arg-type]
    mb = b.rolling_mean(w, min_samples=_MIN).over("ts_code")  # type: ignore[arg-type]
    mab = (a * b).rolling_mean(w, min_samples=_MIN).over("ts_code")  # type: ignore[arg-type]
    return mab - ma * mb


def _rolling_argmax(s: pl.Series) -> float | None:
    if s.len() < _MIN or s.null_count() == s.len():
        return None
    idx = s.arg_max()
    return float(idx) / (s.len() - 1) if idx is not None and s.len() > 1 else None


def _rolling_argmin(s: pl.Series) -> float | None:
    if s.len() < _MIN or s.null_count() == s.len():
        return None
    idx = s.arg_min()
    return float(idx) / (s.len() - 1) if idx is not None and s.len() > 1 else None


def _rolling_skew(s: pl.Series) -> float | None:
    a: np.ndarray = s.drop_nulls().to_numpy()
    if len(a) < _MIN:
        return None
    m = float(a.mean())
    sd = float(a.std())
    if sd < 1e-12:
        return None
    return float((((a - m) / sd) ** 3).mean())


OPERATORS: dict[str, OperatorSpec] = {
    # ── 时序（.over("ts_code")）──
    "ts_mean": _ts("ts_mean", lambda x, w: x.rolling_mean(w, min_samples=_MIN).over("ts_code")),
    "ts_std":  _ts("ts_std",  lambda x, w: x.rolling_std(w, min_samples=_MIN).over("ts_code")),
    "ts_sum":  _ts("ts_sum",  lambda x, w: x.rolling_sum(w, min_samples=_MIN).over("ts_code")),
    "ts_min":  _ts("ts_min",  lambda x, w: x.rolling_min(w, min_samples=_MIN).over("ts_code")),
    "ts_max":  _ts("ts_max",  lambda x, w: x.rolling_max(w, min_samples=_MIN).over("ts_code")),
    "ts_rank": _ts("ts_rank", lambda x, w:
        x.rolling_map(lambda s: (float(s.rank()[-1]) / s.len()) if s.len() >= _MIN else None, w).over("ts_code")),
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
    "ts_argmax": _ts("ts_argmax", lambda x, w:
        x.rolling_map(_rolling_argmax, w).over("ts_code")),
    "ts_argmin": _ts("ts_argmin", lambda x, w:
        x.rolling_map(_rolling_argmin, w).over("ts_code")),
    "ts_skew": _ts("ts_skew", lambda x, w:
        x.rolling_map(_rolling_skew, w).over("ts_code")),
    # ── 截面（.over("trade_date")）──
    "rank":  _cs("rank",  lambda x: (x.rank().over("trade_date") / (pl.len().over("trade_date") + 1))),
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

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
    EVENT_FILL0_FEATURES,
    EVENT_MASK_LEAVES,
    EXPRESS_FEATURES,
    FLOW_FEATURES,
    FORECAST_FEATURES,
    FUNDAMENTAL_FEATURES,
    HOLDER_FEATURES,
    LEAF_FEATURES,
    MARGIN_FEATURES,
    TOPLIST_FEATURES,
)

__all__ = [
    "BASIC_FEATURES",
    "EVENT_FILL0_FEATURES",
    "EVENT_MASK_LEAVES",
    "EXPRESS_FEATURES",
    "FLOW_FEATURES",
    "FORECAST_FEATURES",
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


def _decay_linear(x: pl.Expr, w: int) -> pl.Expr:
    """线性衰减加权均值：权重 1..w（最新一期最大），按窗口内**有效**权重和归一化。

    **不能用 `rolling_mean(weights=...)`**——polars 1.41.2 的加权 rolling 在遇到
    null 时是 Rust 层 `panic!`（"weights not yet supported on array with null values"），
    不是 Python 异常，接不住、会崩掉整个求值进程。而真实因子几乎必然含 null
    （warm-up / 停牌 / 财务缺失），所以这条路在生产上是死的。

    实现走 cumsum 恒等式，**O(1) 个 rolling 而非 O(w) 个 shift**：
    令 ``C`` 为累计和，则
    ``Σ_{k=0}^{w-1}(w-k)·x_{t-k} = w·C_t − Σ_{j=1}^{w} C_{t-j} = w·C_t − rolling_sum(C,w)[t-1]``。
    朴素 O(w) 位移版在全 A（12.5M 行）w=63 实测 14.4s（`ts_mean` 0.19s），
    挖掘会被拖垮；本式实测与位移版最大相对误差 ~1e-11（z-score / 千元 / 小收益率
    三种量级同量级结论），远低于 IC 分辨率。位移参考实现保留在
    ``tests/test_ts_decay_linear.py`` 作 parity 锚。

    **有限性守卫**：cumsum 是全序列累加器，一个 inf/NaN 会污染其后**全部**取值
    （位移版只波及 w 个位置）。故非有限值一律按缺失处理——与
    ``_panel_to_compact`` 的 ``np.isfinite`` 过滤、以及全库「NaN 不得穿透」的口径一致。

    分母按有效权重重归一化（而非固定 Σ1..w）：缺失期不会把结果向 0 稀释，
    常数序列含 null 时仍还原该常数（量纲与 ``ts_mean`` 同尺度）。
    窗口内有效样本数 < ``_MIN`` → null，与其它 ts 算子的 min_samples 语义一致。

    调用方在外层套 ``.over("ts_code")``：本函数只组合表达式、不自带窗口分组，
    避免嵌套 over（截面 over 套时序 over 会产出全 null，见 test_expression_nested_over）。
    """
    ok = (x.is_not_null() & x.is_finite()).fill_null(False)
    filled = pl.when(ok).then(x).otherwise(0.0)
    mask = ok.cast(pl.Float64)
    # rolling_sum(C, w).shift(1) = Σ_{j=1..w} C_{t-j}；头部不足 w 期时 shift 产生的
    # null 补 0（等价于 C_{<0}=0，即窗口自然截断到序列起点）。
    csum = filled.cum_sum()
    msum = mask.cum_sum()
    num = w * csum - csum.rolling_sum(w, min_samples=1).shift(1).fill_null(0.0)
    den = w * msum - msum.rolling_sum(w, min_samples=1).shift(1).fill_null(0.0)
    cnt = ok.cast(pl.Int64).rolling_sum(w, min_samples=1)
    return pl.when(cnt >= _MIN).then(_safe_div(num, den)).otherwise(None)


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


def _ts_count_gt(x: pl.Expr, y: pl.Expr, w: int | None) -> pl.Expr:
    """过去 w 日 x>y 的占比 ∈[0,1]。null/NaN 对不计入分子分母;有效样本 < w/2 → null。

    比较前 fill_nan(None):polars 中 NaN>x 恒 True,会泄漏假阳性。
    实现 O(n):rolling_sum 分子/分母,无逐窗 Python 循环。
    """
    ww = int(w)  # type: ignore[arg-type]
    xf = x.fill_nan(None)
    yf = y.fill_nan(None)
    valid = xf.is_not_null() & yf.is_not_null()
    gt_flag = pl.when(valid).then((xf > yf).cast(pl.Float64)).otherwise(0.0)
    valid_flag = valid.cast(pl.Float64)
    num = gt_flag.rolling_sum(ww, min_samples=1).over("ts_code")
    den = valid_flag.rolling_sum(ww, min_samples=1).over("ts_code")
    thr = ww / 2.0
    return pl.when(den >= thr).then(num / den).otherwise(None)


def _ts_streak_gt(x: pl.Expr, y: pl.Expr, w: int | None) -> pl.Expr:
    """截至当日连续 x>y 的天数,截断于 w(∈[0,w])。

    当日 x>y 为假 → 0;x 或 y 为 null/NaN → null。
    O(n):row_nr − last_break(forward_fill),无 O(n·w) 位移叠窗。
    """
    ww = int(w)  # type: ignore[arg-type]
    xf = x.fill_nan(None)
    yf = y.fill_nan(None)
    valid = xf.is_not_null() & yf.is_not_null()
    is_gt = valid & (xf > yf)
    row_nr = pl.int_range(0, pl.len()).over("ts_code")
    # 非 True(含 False 与 invalid)打断游程;invalid 当日输出 null,False 输出 0
    breaker = pl.when(~is_gt).then(row_nr).otherwise(None)
    last_break = breaker.forward_fill().over("ts_code").fill_null(-1)
    raw = (row_nr - last_break).clip(upper_bound=ww)
    return pl.when(~valid).then(None).when(~is_gt).then(pl.lit(0)).otherwise(raw)


def _ts_count_cross_up(x: pl.Expr, y: pl.Expr, w: int | None) -> pl.Expr:
    """过去 w 日内 x 上穿 y 的次数(昨日 x≤y 且今日 x>y)。

    任一侧 null/NaN → 该日不构成上穿。O(n):shift + rolling_sum。
    """
    ww = int(w)  # type: ignore[arg-type]
    xf = x.fill_nan(None)
    yf = y.fill_nan(None)
    px = xf.shift(1).over("ts_code")
    py = yf.shift(1).over("ts_code")
    both_ok = (
        xf.is_not_null()
        & yf.is_not_null()
        & px.is_not_null()
        & py.is_not_null()
    )
    cross = both_ok & (px <= py) & (xf > yf)
    return cross.cast(pl.Float64).rolling_sum(ww, min_samples=1).over("ts_code")


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
        _decay_linear(x, w).over("ts_code")),
    "ts_corr": _ts2("ts_corr", _ts_corr),
    "ts_cov": _ts2("ts_cov", _ts_cov),
    # 阈值/游程(arity=2+window):离散状态类 alpha——占比/连续天数/上穿次数
    "ts_count_gt": _ts2("ts_count_gt", _ts_count_gt),
    "ts_streak_gt": _ts2("ts_streak_gt", _ts_streak_gt),
    "ts_count_cross_up": _ts2("ts_count_cross_up", _ts_count_cross_up),
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

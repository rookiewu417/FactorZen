"""``ts_decay_linear`` 真实现：线性衰减加权均值（非等权）。

历史：该算子曾以 ``rolling_mean`` 占位（注释「MVP：等权近似线性衰减」），
与 ``ts_mean`` 输出**逐位相同**——每个 ts_decay_linear 表达式都是对应 ts_mean
表达式的伪重复，且表达式层去重抓不到（AST 不同、语义相同），搜索空间与
DSR 的 N 被虚增。

本文件的 ground-truth 一律**手算**（numpy 显式循环），
**禁止**用 ``rolling_mean(weights=...)` 自证——那是被测实现本身。
"""
from __future__ import annotations

import numpy as np
import polars as pl

_MIN = 3  # 与 operators._MIN 对齐


def _manual_decay_linear(vals: list[float | None], w: int) -> list[float | None]:
    """手算线性衰减加权均值 ground-truth。

    权重 1,2,...,w（最新一期权重最大），归一化到 Σw=1。
    窗口内非空样本数 < _MIN → None（与其它 _ts 算子的 min_samples 语义一致）。
    """
    out: list[float | None] = []
    for i in range(len(vals)):
        lo = max(0, i - w + 1)
        window = vals[lo:i + 1]
        # 权重与窗口右端对齐：窗口最后一个元素权重最大
        weights = list(range(w - len(window) + 1, w + 1))
        pairs = [(v, wt) for v, wt in zip(window, weights, strict=True) if v is not None]
        if len(pairs) < _MIN:
            out.append(None)
            continue
        num = sum(v * wt for v, wt in pairs)
        den = sum(wt for _, wt in pairs)
        out.append(num / den)
    return out


def _series_df(vals: list[float], code: str = "A") -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": list(range(len(vals))),
        "ts_code": [code] * len(vals),
        "x": vals,
    }).sort(["ts_code", "trade_date"])


def _apply(df: pl.DataFrame, w: int) -> list:
    from factorzen.discovery.operators import OPERATORS
    expr = OPERATORS["ts_decay_linear"].build([pl.col("x")], w)
    return df.with_columns(expr.alias("f"))["f"].to_list()


# ── 1. ground-truth 对拍（手算，非自证）──────────────────────────────────────

def test_matches_manual_linear_decay_ground_truth():
    rng = np.random.default_rng(7)
    vals = [float(v) for v in rng.standard_normal(40) * 3 + 10]
    df = _series_df(vals)
    for w in (5, 10, 20):
        got = _apply(df, w)
        want = _manual_decay_linear(vals, w)
        assert len(got) == len(want)
        for i, (g, e) in enumerate(zip(got, want, strict=True)):
            if e is None:
                assert g is None, f"w={w} i={i}: 期望 None，得 {g}"
            else:
                assert g is not None, f"w={w} i={i}: 期望 {e}，得 None"
                assert abs(g - e) < 1e-9, f"w={w} i={i}: {g} != {e}"


def test_known_closed_form_small_case():
    """写死的小例子：w=3、值 [1,2,3] → (1*1 + 2*2 + 3*3)/(1+2+3) = 14/6。

    手写常数，任何实现漂移都会红。
    """
    df = _series_df([1.0, 2.0, 3.0])
    got = _apply(df, 3)
    assert got[0] is None and got[1] is None  # 非空样本不足 _MIN
    assert abs(got[2] - 14.0 / 6.0) < 1e-12


# ── 2. 反例锚：不得再退化成等权 ──────────────────────────────────────────────

def test_differs_from_ts_mean():
    """回归锚：ts_decay_linear 曾就是 rolling_mean。单调序列上两者必须不同。"""
    from factorzen.discovery.operators import OPERATORS

    vals = [float(i) for i in range(30)]
    df = _series_df(vals)
    decay = _apply(df, 10)
    mean_expr = OPERATORS["ts_mean"].build([pl.col("x")], 10)
    mean = df.with_columns(mean_expr.alias("m"))["m"].to_list()

    diffs = [
        abs(d - m) for d, m in zip(decay, mean, strict=True)
        if d is not None and m is not None
    ]
    assert diffs, "两侧全 null，测试无判别力"
    assert max(diffs) > 1e-6, "ts_decay_linear 与 ts_mean 逐位相同——算子又退化成等权了"


def test_weights_recent_more_than_old():
    """方向锚：递增序列上，加权均值应高于等权均值（近期权重更大）。"""
    from factorzen.discovery.operators import OPERATORS

    vals = [float(i) for i in range(30)]
    df = _series_df(vals)
    decay = _apply(df, 10)
    mean_expr = OPERATORS["ts_mean"].build([pl.col("x")], 10)
    mean = df.with_columns(mean_expr.alias("m"))["m"].to_list()

    pairs = [
        (d, m) for d, m in zip(decay, mean, strict=True) if d is not None and m is not None
    ]
    assert pairs
    assert all(d > m for d, m in pairs), "递增序列上衰减加权均值应严格大于等权均值"


# ── 3. 归一化 / 量纲 ─────────────────────────────────────────────────────────

def test_constant_series_preserves_level():
    """Σw=1 归一化锚：常数序列的加权均值必须等于该常数（不引入水平漂移）。"""
    df = _series_df([5.0] * 20)
    got = _apply(df, 10)
    vals = [v for v in got if v is not None]
    assert vals, "全 null，无判别力"
    assert all(abs(v - 5.0) < 1e-12 for v in vals)


# ── 4. 分组 / 退化截面语义 ───────────────────────────────────────────────────

def test_grouped_by_ts_code_no_leakage():
    """.over("ts_code")：A 的窗口不得吃进 B 的值。"""
    a = [1.0] * 10
    b = [100.0] * 10
    df = pl.concat([_series_df(a, "A"), _series_df(b, "B")]).sort(["ts_code", "trade_date"])
    got = df.with_columns(
        __import__(
            "factorzen.discovery.operators", fromlist=["OPERATORS"]
        ).OPERATORS["ts_decay_linear"].build([pl.col("x")], 5).alias("f")
    )
    a_vals = [v for v in got.filter(pl.col("ts_code") == "A")["f"].to_list() if v is not None]
    b_vals = [v for v in got.filter(pl.col("ts_code") == "B")["f"].to_list() if v is not None]
    assert a_vals and b_vals
    assert all(abs(v - 1.0) < 1e-12 for v in a_vals)
    assert all(abs(v - 100.0) < 1e-12 for v in b_vals)


def test_warmup_null_semantics_match_ts_mean():
    """warm-up 的 null 位置必须与 ts_mean 一致（min_samples 语义同族）。"""
    from factorzen.discovery.operators import OPERATORS

    rng = np.random.default_rng(3)
    vals = [float(v) for v in rng.standard_normal(25)]
    df = _series_df(vals)
    decay = _apply(df, 8)
    mean_expr = OPERATORS["ts_mean"].build([pl.col("x")], 8)
    mean = df.with_columns(mean_expr.alias("m"))["m"].to_list()
    assert [v is None for v in decay] == [v is None for v in mean]


def _shift_reference(x: pl.Expr, w: int) -> pl.Expr:
    """O(w) 位移参考实现——生产实现走 cumsum 恒等式（O(1) rolling）换性能，
    本函数是它的独立 parity 锚：两者由**不同代数路径**得出，非自证。
    """
    ok = (x.is_not_null() & x.is_finite()).fill_null(False)
    filled = pl.when(ok).then(x).otherwise(0.0)
    num = den = cnt = None
    for k in range(w):
        wt = float(w - k)
        v = filled.shift(k).fill_null(0.0) * wt
        m = ok.shift(k).fill_null(value=False)
        d = m.cast(pl.Float64) * wt
        c = m.cast(pl.Int64)
        num = v if num is None else num + v
        den = d if den is None else den + d
        cnt = c if cnt is None else cnt + c
    return (
        pl.when(cnt >= _MIN)
        .then(pl.when(den.abs() > 1e-12).then(num / den).otherwise(None))
        .otherwise(None)
    )


def test_parity_with_shift_reference_across_magnitudes():
    """cumsum 恒等式 vs O(w) 位移：不同量级下都须逐位一致（相对误差 < 1e-9）。

    量级覆盖 z-score(1) / amount 千元(1e6) / 小收益率(1e-4)——cumsum 的
    灾难性抵消风险随量级放大，必须按量级验，不能只测一个尺度。
    """
    rng = np.random.default_rng(11)
    for scale in (1.0, 1e6, 1e-4):
        n = 600
        vals = rng.standard_normal(n) * scale
        vals[rng.random(n) < 0.05] = np.nan
        df = pl.DataFrame({
            "trade_date": list(range(n)),
            "ts_code": ["A"] * n,
            "x": vals,
        }).with_columns(pl.col("x").fill_nan(None)).sort(["ts_code", "trade_date"])
        for w in (5, 20, 63):
            got = np.asarray(_apply(df, w), dtype=float)
            want = np.asarray(
                df.with_columns(_shift_reference(pl.col("x"), w).over("ts_code").alias("r"))
                ["r"].to_list(), dtype=float)
            assert (np.isnan(got) == np.isnan(want)).all(), f"scale={scale} w={w} null 位置不一致"
            both = ~np.isnan(got)
            if both.any():
                rel = np.abs(got[both] - want[both]) / np.maximum(np.abs(want[both]), 1e-300)
                assert rel.max() < 1e-9, f"scale={scale} w={w} 相对误差 {rel.max():.2e}"


def test_non_finite_does_not_poison_downstream():
    """有限性守卫锚：cumsum 是全序列累加器，一个 inf 若不拦会污染其后**全部**取值。

    构造：序列中段插入 inf，断言 inf 之后的取值仍有限，且等于「把 inf 当缺失」的结果。
    """
    n = 40
    vals = [1.0] * n
    vals[10] = float("inf")
    df = _series_df(vals)
    got = _apply(df, 5)
    tail = [v for v in got[15:] if v is not None]
    assert tail, "尾段全 null，测试无判别力"
    assert all(np.isfinite(v) for v in tail), "inf 穿透了 cumsum，污染下游全部取值"
    # inf 按缺失处理 → 其余全是常数 1.0，加权均值仍为 1.0
    assert all(abs(v - 1.0) < 1e-12 for v in tail)


def test_all_null_series_yields_all_null():
    """退化截面守卫：全 null 输入不得抛、不得产出 NaN。"""
    df = pl.DataFrame({
        "trade_date": list(range(10)),
        "ts_code": ["A"] * 10,
        "x": [None] * 10,
    }, schema_overrides={"x": pl.Float64}).sort(["ts_code", "trade_date"])
    got = _apply(df, 5)
    assert all(v is None for v in got)

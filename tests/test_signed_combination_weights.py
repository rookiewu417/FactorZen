"""P1-①：组合层允许负权——让权重自己处理符号。

背景：准入用**残差**口径（`residual_ic_v1`），部署 `combine_from_library` 走的却是
**裸 factor_value**。实锤 top_list 家族 5 条候选残差 lift +0.0066~+0.0094（过阈准入），
裸 IC 却是 −0.0029~−0.0175（负）。准入认可的是「与库正交后那部分」，进组合的是整条裸因子。

三个候选解里用户裁决走「换权重估计」。但实查发现三个方法**全是 long-only**：
`estimate_ic_weights` 的 `max(0.0, ...)`、`estimate_max_ir_weights` 的
`np.maximum(w_raw, 0.0)`——单纯换方法**并不能**表达负贡献。故本轮真正要做的是给估权器
加「允许负权」能力，并把归一化从 `Σw` 换成 L1 `Σ|w|`（负权下 `Σw` 可能≈0 → 爆炸/翻号）。

参数化带现默认值：`allow_negative=False` 时行为逐位不变（A股基线零回归是底线）。
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

_DATES = [f"2024-{m:02d}-{d:02d}" for m in (1, 2, 3) for d in range(1, 21)]
_CODES = [f"{i:06d}.SZ" for i in range(60)]


def _mk(values: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": np.repeat(_DATES, len(_CODES)),
        "ts_code": np.tile(_CODES, len(_DATES)),
        "factor_value": values.reshape(-1).astype(float),
    })


def _ret(values: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": np.repeat(_DATES, len(_CODES)),
        "ts_code": np.tile(_CODES, len(_DATES)),
        "ret": values.reshape(-1).astype(float),
    })


def _scenario(seed: int = 0):
    """构造 IC 符号**已知**的三因子场景（ground truth 由构造给定，非由被测实现给出）：

    - ``pos``：与收益同向 → IC 明显为正
    - ``neg``：与收益反向 → IC 明显为负
    - ``noise``：纯噪声 → IC ≈ 0
    """
    rng = np.random.default_rng(seed)
    shape = (len(_DATES), len(_CODES))
    signal = rng.standard_normal(shape)
    noise = rng.standard_normal(shape)
    ret = signal + 0.5 * rng.standard_normal(shape)
    return (
        {
            "pos": _mk(signal + 0.3 * rng.standard_normal(shape)),
            "neg": _mk(-signal + 0.3 * rng.standard_normal(shape)),
            "noise": _mk(noise),
        },
        _ret(ret),
    )


# ── 1. 现状锚：三个方法都给不出负权（本轮改动的前提）────────────────────────

def test_default_estimators_are_long_only():
    """默认口径下 IC 加权 / max_IR **都不会**给负权——这正是「换方法」解决不了问题的原因。"""
    from factorzen.research.combination.methods import (
        estimate_ic_weights,
        estimate_max_ir_weights,
    )

    dfs, ret = _scenario()
    w_ic = estimate_ic_weights(dfs, ret)
    assert all(v >= 0.0 for v in w_ic.values()), "默认 IC 加权不应给负权（现语义）"
    assert w_ic["neg"] == pytest.approx(0.0, abs=1e-12), "负 IC 因子应被裁到 0"

    w_ir = estimate_max_ir_weights(dfs, ret)
    assert w_ir is not None
    assert all(v >= 0.0 for v in w_ir.values()), "默认 max_IR 不应给负权（现语义）"


# ── 2. 允许负权后：符号必须跟着 IC 符号走 ────────────────────────────────────

def test_ic_weights_signed_follows_known_ic_sign():
    """ground truth 由**构造**给定：pos 正权、neg 负权。

    不用被测实现算出的 IC 反过来断言权重（那是恒真）——因子的 IC 符号是构造时决定的。
    """
    from factorzen.research.combination.methods import estimate_ic_weights

    dfs, ret = _scenario()
    w = estimate_ic_weights(dfs, ret, allow_negative=True)
    assert w["pos"] > 0.0, f"同向因子应得正权，得 {w['pos']}"
    assert w["neg"] < 0.0, f"反向因子应得负权，得 {w['neg']}"
    assert abs(w["noise"]) < abs(w["pos"]), "噪声因子权重量级应远小于信号因子"


def test_ic_weights_signed_l1_normalized():
    """负权下必须用 L1 归一化：Σ|w| = 1。

    若沿用 Σw 归一化，正负相消时分母≈0 → 权重爆炸甚至整体翻号。
    """
    from factorzen.research.combination.methods import estimate_ic_weights

    dfs, ret = _scenario()
    w = estimate_ic_weights(dfs, ret, allow_negative=True)
    assert sum(abs(v) for v in w.values()) == pytest.approx(1.0, abs=1e-9)


def test_l1_normalization_survives_near_cancelling_weights():
    """正负近乎抵消（Σw≈0）时不得爆炸——这正是 Σw 归一化会炸的场景。"""
    from factorzen.research.combination.methods import estimate_ic_weights

    rng = np.random.default_rng(5)
    shape = (len(_DATES), len(_CODES))
    signal = rng.standard_normal(shape)
    ret = signal + 0.5 * rng.standard_normal(shape)
    # 两条对称的反向因子 → mean IC 近似 +c 与 −c，Σw ≈ 0
    dfs = {"a": _mk(signal), "b": _mk(-signal)}
    w = estimate_ic_weights(dfs, _ret(ret), allow_negative=True)
    assert all(np.isfinite(v) for v in w.values())
    assert sum(abs(v) for v in w.values()) == pytest.approx(1.0, abs=1e-9)
    assert max(abs(v) for v in w.values()) <= 1.0 + 1e-9, "权重被放大 = 归一化炸了"


def test_max_ir_signed_closed_form_ground_truth():
    """max_IR 闭式解 w ∝ Σ⁻¹μ 的**手算**对拍：构造让最优权重必为负的 μ/Σ。

    两个强正相关因子、其中一个 μ 明显更小时，Σ⁻¹μ 会给后者负权（对冲共同成分）。
    裁剪到 0 会破坏闭式解的最优性——这是独立于 P1-① 的既有缺陷。
    """
    from factorzen.research.combination.methods import _solve_max_ir_weights

    mu = np.array([0.05, 0.01])
    sigma = np.array([[1.0, 0.9], [0.9, 1.0]])
    w = _solve_max_ir_weights(mu, sigma, allow_negative=True)
    # 手算纯闭式解：Σ⁻¹ = 1/(1-0.81) · [[1,-0.9],[-0.9,1]]，再 L1 归一化
    want = np.linalg.inv(sigma) @ mu
    want = want / np.abs(want).sum()
    assert want[1] < 0, "前提不成立：构造的场景应让第二个权重为负"
    # rtol 放到 1e-5：实现对 Σ 加了文档化的 1e-6 岭正则，与纯闭式解的偏差正是该量级。
    # 这里要测的是「是否还是那个闭式解」，不是「是否逐位复刻实现的正则项」。
    np.testing.assert_allclose(w, want, rtol=1e-5)
    assert w[1] < 0, "允许负权后第二个权重必须真为负（裁剪会让它变 0）"


# ── 3. 判别力核心：允许负权确实能救回负 IC 因子 ──────────────────────────────

def test_signed_weighting_beats_clipped_when_negative_ic_factor_present():
    """机制测试：IC 符号**稳定且估准**时，允许负权的合成因子 IC 更高。

    clipped 口径把 neg 裁到 0（信息浪费）；signed 口径给它负权（信息被利用）。
    比较的是**合成因子对收益的 IC**——外部 ground truth，不是权重自证。

    ⚠️ **本测试证明的是机制可行，不是「signed 在实践中更好」。**
    真实库 OOS（csi300/2020-2026/85 因子，含 21 条负 ic_train）结论**相反**：
    signed 0.0276 < clipped 0.0374。差异来源是本测试**没有**的估计噪声——
    这里 neg 因子的 IC 由构造保证稳定，真实库里 85 个因子的 IC 估计噪声很大，
    放开负权只是放大噪声（Jagannathan & Ma 2003：禁止做空 ≈ 协方差收缩）。
    """
    from factorzen.research.combination.methods import (
        _rank_ic_numpy,
        apply_weights,
        estimate_ic_weights,
    )

    dfs, ret = _scenario(seed=3)

    def _composite_ic(weights):
        comp = apply_weights(dfs, weights)
        j = comp.join(ret, on=["trade_date", "ts_code"], how="inner")
        ics = []
        for _, g in j.group_by("trade_date"):
            v = _rank_ic_numpy(
                g["factor_value"].to_numpy(), g["ret"].to_numpy()
            )
            if v is not None:
                ics.append(v)
        return float(np.mean(ics))

    ic_clipped = _composite_ic(estimate_ic_weights(dfs, ret))
    ic_signed = _composite_ic(estimate_ic_weights(dfs, ret, allow_negative=True))
    assert ic_signed > ic_clipped, (
        f"允许负权未能提升合成 IC：signed={ic_signed:.4f} vs clipped={ic_clipped:.4f}"
    )


# ── 4. 零回归锚：默认行为逐位不变 ────────────────────────────────────────────

def test_default_behavior_bitwise_unchanged():
    """新增参数必须带默认值且默认行为逐位不变（A股基线零回归底线）。"""
    from factorzen.research.combination.methods import (
        estimate_ic_weights,
        estimate_max_ir_weights,
    )

    dfs, ret = _scenario(seed=7)
    w_ic = estimate_ic_weights(dfs, ret)
    w_ic2 = estimate_ic_weights(dfs, ret, allow_negative=False)
    assert w_ic == w_ic2
    assert sum(w_ic.values()) == pytest.approx(1.0, abs=1e-9), "默认仍是 Σw=1"

    w_ir = estimate_max_ir_weights(dfs, ret)
    w_ir2 = estimate_max_ir_weights(dfs, ret, allow_negative=False)
    assert w_ir == w_ir2


def test_all_zero_ic_falls_back_to_equal_weights():
    """退化守卫：全零 IC 时（Σ|w|≈0）退化等权，不得除零。"""
    from factorzen.research.combination.methods import estimate_ic_weights

    shape = (len(_DATES), len(_CODES))
    const = np.ones(shape)
    dfs = {"a": _mk(const), "b": _mk(const)}
    rng = np.random.default_rng(1)
    w = estimate_ic_weights(dfs, _ret(rng.standard_normal(shape)), allow_negative=True)
    assert all(np.isfinite(v) for v in w.values())
    assert w["a"] == pytest.approx(0.5)
    assert w["b"] == pytest.approx(0.5)


# ── 5. OOS 协议分派：新方法名接得上 ──────────────────────────────────────────

def test_oos_dispatch_supports_signed_methods():
    """`ic_weighted_signed` / `max_ir_signed` 必须能进 OOS 协议的方法分派。

    否则能力做完了但 `combine from-library` 的对照表里看不到 = 接线层漂移。
    """
    from factorzen.research.combination.oos import _estimate_fold

    dfs, ret = _scenario()
    for method in ("ic_weighted_signed", "max_ir_signed"):
        w = _estimate_fold(method, dfs, dfs, ret, {})
        assert set(w) == set(dfs), f"{method} 权重键不全"
        assert all(np.isfinite(v) for v in w.values()), f"{method} 产出非有限权重"
    # 有负 IC 因子在场时，signed 方法应真的给出负权（否则等于没接上）
    w = _estimate_fold("ic_weighted_signed", dfs, dfs, ret, {})
    assert w["neg"] < 0.0, "signed 方法没给出负权 = 分派接到了 clipped 实现"


def test_unknown_method_still_raises_valueerror():
    """异常契约不变：未知方法名仍抛 ValueError（解析外部输入只抛一类）。"""
    from factorzen.research.combination.oos import _estimate_fold

    dfs, ret = _scenario()
    with pytest.raises(ValueError, match="未知 method"):
        _estimate_fold("no_such_method", dfs, dfs, ret, {})

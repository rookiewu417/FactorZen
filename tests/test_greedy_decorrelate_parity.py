"""_greedy_decorrelate 决策 parity：紧凑矩阵加速不得改变 kept/dropped 决策。

真源 = `_greedy_decorrelate_reference`（旧版 max_correlation 逐对路径）。
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
import polars as pl


def _panel_from_matrix(
    mat: np.ndarray,
    *,
    start: date = date(2020, 1, 2),
    stock_prefix: str = "",
    drop_dates: set[int] | None = None,
    drop_stocks: set[int] | None = None,
) -> pl.DataFrame:
    """(D×S) 矩阵 → [trade_date, ts_code, factor_value] 面板；可选缺日期/缺股票。"""
    d_n, s_n = mat.shape
    trade_dates: list[date] = []
    ts_codes: list[str] = []
    values: list[float] = []
    for di in range(d_n):
        if drop_dates and di in drop_dates:
            continue
        dt = start + timedelta(days=di)
        for si in range(s_n):
            if drop_stocks and si in drop_stocks:
                continue
            v = mat[di, si]
            trade_dates.append(dt)
            ts_codes.append(f"{stock_prefix}{si:06d}.SH")
            # 用 nan 而非 None，避免前段全缺时 schema 推断成 Null
            values.append(float(v) if np.isfinite(v) else float("nan"))
    return pl.DataFrame({
        "trade_date": trade_dates,
        "ts_code": ts_codes,
        "factor_value": values,
    })

def _synth_suite(seed: int, *, n_factors: int = 10, n_days: int = 60, n_stocks: int = 40):
    """构造一组合成因子：近亲≈0.95、边界≈0.70、全常数退化、含 NaN 块。"""
    rng = np.random.default_rng(seed)
    mats: list[np.ndarray] = []
    exprs: list[str] = []

    # f0: 基准独立信号
    base = rng.standard_normal((n_days, n_stocks))
    mats.append(base)
    exprs.append(f"f0_base_s{seed}")

    # f1: 近亲（corr≈0.95）— 与 f0 同向高相关
    twin = 0.95 * base + 0.05 * rng.standard_normal((n_days, n_stocks))
    mats.append(twin)
    exprs.append(f"f1_near_twin_s{seed}")

    # f2: 边界相关（目标 |corr| ∈ [0.69, 0.71] 附近，与 f0）
    orth = rng.standard_normal((n_days, n_stocks))
    alpha = 0.70
    boundary = alpha * base + np.sqrt(max(1.0 - alpha * alpha, 0.0)) * orth
    mats.append(boundary)
    exprs.append(f"f2_boundary_s{seed}")

    # f3: 全常数退化
    mats.append(np.full((n_days, n_stocks), 3.14))
    exprs.append(f"f3_const_s{seed}")

    # f4: 含 NaN 块（前 1/3 日全缺 + 随机点缺）
    nan_block = rng.standard_normal((n_days, n_stocks))
    nan_block[: n_days // 3, :] = np.nan
    mask = rng.random((n_days, n_stocks)) < 0.1
    nan_block[mask] = np.nan
    mats.append(nan_block)
    exprs.append(f"f4_nan_block_s{seed}")

    # 其余独立噪声，补满 n_factors
    k = 5
    while len(mats) < n_factors:
        mats.append(rng.standard_normal((n_days, n_stocks)))
        exprs.append(f"f{k}_noise_s{seed}")
        k += 1

    materialized = [(e, _panel_from_matrix(m)) for e, m in zip(exprs, mats, strict=False)]
    return materialized


def _assert_decision_parity(new_kept, new_dropped, ref_kept, ref_dropped, *, atol: float = 1e-9):
    assert [e for e, _ in new_kept] == [e for e, _ in ref_kept], (
        f"kept 表达式序列不一致\n new={[e for e,_ in new_kept]}\n ref={[e for e,_ in ref_kept]}"
    )
    # kept 必须是原面板对象（下游写 parquet）
    for (ne, nf), (re, rf) in zip(new_kept, ref_kept, strict=False):
        assert ne == re
        assert nf is rf or nf.equals(rf), f"kept 面板被替换: {ne}"

    assert len(new_dropped) == len(ref_dropped)
    for nd, rd in zip(new_dropped, ref_dropped, strict=False):
        assert nd["expression"] == rd["expression"]
        assert nd["corr_with"] == rd["corr_with"], (
            f"corr_with 不一致 for {nd['expression']}: "
            f"new={nd['corr_with']} ref={rd['corr_with']}"
        )
        assert abs(float(nd["corr"]) - float(rd["corr"])) <= atol, (
            f"corr 超容差 for {nd['expression']}: "
            f"new={nd['corr']} ref={rd['corr']} |diff|={abs(float(nd['corr'])-float(rd['corr']))}"
        )


# ── 1. 决策 parity（核心）────────────────────────────────────────────────────

def test_greedy_decorrelate_decision_parity_three_seeds():
    """随机 3 组合成面板：新旧 kept 序 + dropped(expression/corr_with) 完全一致，corr≤1e-9。"""
    from factorzen.pipelines.factor_combine import (
        _greedy_decorrelate,
        _greedy_decorrelate_reference,
    )

    threshold = 0.7
    for seed in (0, 7, 42):
        mats = _synth_suite(seed)
        new_k, new_d = _greedy_decorrelate(mats, threshold)
        ref_k, ref_d = _greedy_decorrelate_reference(mats, threshold)
        _assert_decision_parity(new_k, new_d, ref_k, ref_d)


# ── 2. 逃生口 threshold=1.0 ──────────────────────────────────────────────────

def test_greedy_decorrelate_threshold_one_keeps_all():
    """threshold=1.0 → >1.0 恒 False → 全 kept、dropped 空。"""
    from factorzen.pipelines.factor_combine import (
        _greedy_decorrelate,
        _greedy_decorrelate_reference,
    )

    mats = _synth_suite(1)
    new_k, new_d = _greedy_decorrelate(mats, 1.0)
    ref_k, ref_d = _greedy_decorrelate_reference(mats, 1.0)
    assert new_d == [] and ref_d == []
    assert [e for e, _ in new_k] == [e for e, _ in mats]
    assert [e for e, _ in ref_k] == [e for e, _ in mats]
    _assert_decision_parity(new_k, new_d, ref_k, ref_d)


# ── 3. 异质覆盖（缺日期 / 缺股票）───────────────────────────────────────────

def test_greedy_decorrelate_heterogeneous_coverage_parity():
    """一因子只有半段日期、另一因子缺部分股票：不崩且与旧实现一致。"""
    from factorzen.pipelines.factor_combine import (
        _greedy_decorrelate,
        _greedy_decorrelate_reference,
    )

    rng = np.random.default_rng(99)
    n_days, n_stocks = 60, 40
    a = rng.standard_normal((n_days, n_stocks))
    b = 0.96 * a + 0.04 * rng.standard_normal((n_days, n_stocks))  # 近亲
    c = rng.standard_normal((n_days, n_stocks))

    mats = [
        ("half_dates", _panel_from_matrix(a, drop_dates=set(range(n_days // 2, n_days)))),
        ("full_near", _panel_from_matrix(b)),
        ("missing_stocks", _panel_from_matrix(c, drop_stocks=set(range(0, 10)))),
        ("noise", _panel_from_matrix(rng.standard_normal((n_days, n_stocks)))),
    ]
    new_k, new_d = _greedy_decorrelate(mats, 0.7)
    ref_k, ref_d = _greedy_decorrelate_reference(mats, 0.7)
    _assert_decision_parity(new_k, new_d, ref_k, ref_d)


# ── 4. 缩尺性能冒烟（打印 + 宽松注释，CI 不依赖时序）────────────────────────

def test_greedy_decorrelate_scaled_perf_smoke():
    """~20 因子 × 250 日 × 100 股：打印 A/B 耗时；新实现预期 ≪ 旧（不硬断言防抖动）。"""
    from factorzen.pipelines.factor_combine import (
        _greedy_decorrelate,
        _greedy_decorrelate_reference,
    )

    rng = np.random.default_rng(123)
    n_factors, n_days, n_stocks = 20, 250, 100
    mats = []
    base = rng.standard_normal((n_days, n_stocks))
    for i in range(n_factors):
        if i == 1:
            m = 0.93 * base + 0.07 * rng.standard_normal((n_days, n_stocks))
        else:
            m = rng.standard_normal((n_days, n_stocks))
        mats.append((f"perf_f{i}", _panel_from_matrix(m)))

    t0 = time.perf_counter()
    ref_k, ref_d = _greedy_decorrelate_reference(mats, 0.7)
    t_ref = time.perf_counter() - t0

    t1 = time.perf_counter()
    new_k, new_d = _greedy_decorrelate(mats, 0.7)
    t_new = time.perf_counter() - t1

    _assert_decision_parity(new_k, new_d, ref_k, ref_d)
    # 打印供人工观察；目标约 <1/5，但不写入硬断言以免 CI 抖动。
    print(
        f"\n[decorr A/B] n={n_factors}×{n_days}×{n_stocks} "
        f"ref={t_ref:.3f}s new={t_new:.3f}s speedup={t_ref / max(t_new, 1e-9):.1f}x"
    )
    # 宽松护栏：仅在旧实现足够慢时检查加速（本地/CI 都极快则跳过）
    if t_ref > 1.0:
        assert t_new < t_ref / 5.0, (
            f"加速不足: ref={t_ref:.3f}s new={t_new:.3f}s (期望 <1/5)"
        )

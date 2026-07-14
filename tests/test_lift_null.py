"""lift 统计层 null 校准工具测试（block sign-flip / AR(1) 日差分 H0）。"""
from __future__ import annotations

import numpy as np
import polars as pl

# ── 1. 管道一致性（防恒真） ──────────────────────────────────────────────────


def test_fixed_diff_matches_paired_lift_stats():
    """固定 diff 序列：手工 lift/SE/half 与 paired_lift_stats 一致。"""
    from factorzen.discovery.lift_test import paired_lift_stats

    # 60 日，block_days=20 → 3 块；奇数中位归前半 → 前 2 块 vs 后 1 块
    n_days = 60
    block_days = 20
    diffs = np.array(
        [0.01] * 20 + [0.03] * 20 + [-0.01] * 20,
        dtype=float,
    )
    dates = [f"d{i:04d}" for i in range(n_days)]
    cand = pl.DataFrame(
        {"trade_date": dates, "ic": diffs.tolist()},
        schema={"trade_date": pl.Utf8, "ic": pl.Float64},
    )
    base = pl.DataFrame(
        {"trade_date": dates, "ic": [0.0] * n_days},
        schema={"trade_date": pl.Utf8, "ic": pl.Float64},
    )
    stats = paired_lift_stats(cand, base, block_days=block_days)

    # 手算
    lift = float(np.mean(diffs))
    block_means = np.array([
        float(np.mean(diffs[0:20])),
        float(np.mean(diffs[20:40])),
        float(np.mean(diffs[40:60])),
    ])
    se = float(block_means.std(ddof=1) / np.sqrt(3))
    first_half = float(np.mean(diffs[0:40]))  # 前 2 块
    second_half = float(np.mean(diffs[40:60]))  # 后 1 块

    assert stats["n_days"] == 60
    assert stats["n_blocks"] == 3
    assert abs(stats["lift"] - lift) < 1e-12
    assert abs(stats["lift_se"] - se) < 1e-12
    assert abs(stats["lift_first_half"] - first_half) < 1e-12
    assert abs(stats["lift_second_half"] - second_half) < 1e-12


def test_null_admission_rates_calls_lift_admission(monkeypatch):
    """null_admission_rates 内部必须走生产 lift_admission（探针计数）。"""
    import factorzen.discovery.lift_null as mod
    from factorzen.discovery.lift_test import lift_admission as real_adm

    calls = {"n": 0}

    def _probe(row, *, threshold=0.001, se_mult=1.0):
        calls["n"] += 1
        return real_adm(row, threshold=threshold, se_mult=se_mult)

    monkeypatch.setattr(mod, "lift_admission", _probe)

    n_sims = 30
    out = mod.null_admission_rates(
        n_days=80,
        block_days=20,
        daily_sigma=0.02,
        ar1=0.0,
        n_sims=n_sims,
        seed=7,
        n_candidates_batch=5,
    )
    assert calls["n"] == n_sims
    assert "p_active" in out
    assert 0.0 <= out["p_active"] <= 1.0


# ── 2. 方向性 ────────────────────────────────────────────────────────────────


def test_higher_se_mult_not_more_active():
    """同 seed 同序列集：se_mult=2.0 的 p_active ≤ se_mult=1.0。"""
    from factorzen.discovery.lift_null import null_admission_rates

    common = dict(
        n_days=400,
        block_days=20,
        daily_sigma=0.03,
        ar1=0.0,
        n_sims=800,
        seed=42,
        n_candidates_batch=10,
    )
    r1 = null_admission_rates(**common, se_mult=1.0)
    r2 = null_admission_rates(**common, se_mult=2.0)
    assert r2["p_active"] <= r1["p_active"]


def test_min_blocks_not_more_active():
    """校准附加规则：min_blocks=10 的 p_active ≤ min_blocks=0。

    n_days=100 → 恒有 n_blocks=5；min_blocks=10 前置 reject 全员，
    故 p_active(10)=0 ≤ p_active(0)。（固定 n_days 下 n_blocks 确定，
    该规则是「窗口够不够」的门槛，不是随机过滤。）
    """
    from factorzen.discovery.lift_null import calibration_table

    rows = calibration_table(
        n_days=100,  # block_days=20 → 5 块
        daily_sigma=0.03,
        ar1=0.0,
        se_mults=(1.0,),
        min_blocks_options=(0, 10),
        n_sims=600,
        seed=11,
    )
    by_mb = {r["min_blocks"]: r for r in rows}
    assert by_mb[10]["p_active"] <= by_mb[0]["p_active"]
    assert by_mb[10]["p_active"] == 0.0


# ── 3. 量级回归（复现审查 §7.2 ~14.8%） ────────────────────────────────────


def test_p_active_magnitude_92_blocks():
    """n_days≈1840（92×20）、ar1=0、se_mult=1.0 → p_active ∈ [0.10, 0.20]。

    宽区间防 seed 敏感；n_sims=2000 固定 seed。scale-free 区 daily_sigma
    取 0.05（SE 通常 > threshold，规则由 se_mult×SE 主导）。
    """
    from factorzen.discovery.lift_null import null_admission_rates

    out = null_admission_rates(
        n_days=1840,
        block_days=20,
        daily_sigma=0.05,
        ar1=0.0,
        se_mult=1.0,
        n_sims=2000,
        seed=0,
        n_candidates_batch=10,
    )
    assert 0.10 <= out["p_active"] <= 0.20, (
        f"p_active={out['p_active']:.4f} 不在 [0.10, 0.20]"
    )


# ── 4. 确定性 ────────────────────────────────────────────────────────────────


def test_null_admission_rates_deterministic():
    """同 seed 两次调用结果完全一致。"""
    from factorzen.discovery.lift_null import null_admission_rates

    kw = dict(
        n_days=120,
        daily_sigma=0.02,
        ar1=0.2,
        n_sims=200,
        seed=99,
        n_candidates_batch=10,
    )
    a = null_admission_rates(**kw)
    b = null_admission_rates(**kw)
    # 率与 FWER 完全一致
    assert a["p_active"] == b["p_active"]
    assert a["p_probation"] == b["p_probation"]
    assert a["p_pass"] == b["p_pass"]
    assert a["fwer_active"]["analytic"] == b["fwer_active"]["analytic"]
    assert a["fwer_active"]["simulated"] == b["fwer_active"]["simulated"]
    assert a["p_active_ci"] == b["p_active_ci"]


# ── 5. AR(1) 生效 ────────────────────────────────────────────────────────────


def test_ar1_changes_lift_se_distribution():
    """正自相关下块均值方差更大 → mean_lift_se(ar1=0.8) > mean_lift_se(ar1=0)。

    原因：AR(1) ρ>0 时块内日差分正相关，块均值的抽样方差升高，
    块 SE = std(block_means)/√n_blocks 随之抬高。
    """
    from factorzen.discovery.lift_null import null_admission_rates

    common = dict(
        n_days=400,
        block_days=20,
        daily_sigma=0.03,
        n_sims=500,
        seed=3,
        n_candidates_batch=10,
        se_mult=1.0,
    )
    r0 = null_admission_rates(**common, ar1=0.0)
    r8 = null_admission_rates(**common, ar1=0.8)
    assert r8["mean_lift_se"] > r0["mean_lift_se"], (
        f"期望 AR(1)=0.8 的 mean_lift_se 更大: "
        f"0.8→{r8['mean_lift_se']}, 0→{r0['mean_lift_se']}"
    )


# ── 6. Wilson CI 边界 ────────────────────────────────────────────────────────


def test_wilson_ci_extremes_no_crash():
    """p=0 / p=1 时 Wilson 区间不炸，且落在 [0,1]。"""
    from factorzen.discovery.lift_null import wilson_ci

    lo0, hi0 = wilson_ci(0, 100)
    assert 0.0 <= lo0 <= hi0 <= 1.0
    assert lo0 == 0.0 or lo0 < 0.05  # 下界贴 0 或极小

    lo1, hi1 = wilson_ci(100, 100)
    assert 0.0 <= lo1 <= hi1 <= 1.0
    assert hi1 == 1.0 or hi1 > 0.95

    lo_empty, hi_empty = wilson_ci(0, 0)
    assert 0.0 <= lo_empty <= hi_empty <= 1.0


# ── 辅助：校准表 / 经验参数 / markdown ───────────────────────────────────────


def test_calibration_table_and_markdown():
    from factorzen.discovery.lift_null import (
        calibration_table,
        format_calibration_markdown,
    )

    rows = calibration_table(
        n_days=100,
        daily_sigma=0.02,
        ar1=0.0,
        se_mults=(1.0, 2.0),
        min_blocks_options=(0, 6),
        n_sims=100,
        seed=1,
    )
    assert len(rows) == 4  # 2 × 2
    md = format_calibration_markdown(rows)
    assert "se_mult" in md
    assert "p_active" in md
    assert "|" in md


def test_estimate_daily_sigma_from_run():
    """粗估：lift_se ≈ σ_block/√n → σ_daily ≈ σ_block × √block_days。"""
    from factorzen.discovery.lift_null import estimate_daily_sigma_from_run

    out = estimate_daily_sigma_from_run(
        {"lift_se": 0.002, "n_blocks": 25, "n_days": 500},
    )
    assert out["n_days"] == 500
    assert out["daily_sigma"] > 0
    # 手算：block_mean_std = 0.002 * √25 = 0.01
    # avg_block_days = 500/25 = 20 → daily_sigma ≈ 0.01 * √20
    expected = 0.01 * np.sqrt(20)
    assert abs(out["daily_sigma"] - expected) < 1e-12

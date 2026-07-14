"""lift 统计层 null 校准（H0 = 无真实 lift）。

本模块只覆盖**统计层**准入规则本身（``lift_admission`` + ``paired_lift_stats``），
不含 Agent / LLM 选择偏差、挖掘筛选、多重尝试等链路。

因此这里估计的 false admission rate 是**下界**：真实线上误准入只会更高，
不会更低。用途是给 ``se_mult`` / ``min_blocks`` / economic threshold 解冻
提供可复跑的下界证据（对齐审查报告 §12.3 第三批）。

H0 生成：均值 0 的 AR(1) 日差分序列 → 构造 cand/base 日 IC DataFrame →
**必须**调用生产 ``paired_lift_stats`` 与 ``lift_admission``（禁止本模块重写
lift/SE/half，防双路径漂移）。
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import polars as pl

from factorzen.discovery.guardrails import DEFAULT_LIFT_THRESHOLD
from factorzen.discovery.lift_test import (
    DEFAULT_BLOCK_DAYS,
    lift_admission,
    paired_lift_stats,
)

__all__ = [
    "calibration_table",
    "estimate_daily_sigma_from_run",
    "format_calibration_markdown",
    "null_admission_rates",
    "wilson_ci",
]


# ── Wilson 95% 置信区间 ──────────────────────────────────────────────────────


def wilson_ci(k: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """二项比例 Wilson score 区间（默认 95%，z=1.96）。

    n=0 → (0, 1)；k=0 / k=n 时不炸，区间仍落在 [0, 1]。
    """
    if n <= 0:
        return (0.0, 1.0)
    k = int(k)
    n = int(n)
    p = k / n
    z2 = float(z) * float(z)
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    # 根号内在 p∈{0,1} 时仍 ≥0
    inner = (p * (1.0 - p) + z2 / (4.0 * n)) / n
    margin = float(z) * np.sqrt(max(0.0, inner)) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (float(lo), float(hi))


# ── H0 路径生成 ──────────────────────────────────────────────────────────────


def _ar1_series(
    n: int,
    *,
    daily_sigma: float,
    ar1: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """均值 0 的 AR(1) 日差分，无条件标准差 ≈ daily_sigma。

    x_t = ρ x_{t-1} + ε_t，ε ~ N(0, σ_ε²)，σ_ε = daily_sigma · √(1-ρ²)
    （|ρ|<1；|ρ|≥1 时退化为白噪声尺度，避免炸）。
    """
    rho = float(ar1)
    sig = float(daily_sigma)
    if abs(rho) >= 1.0:
        # 非平稳边界：按白噪声生成，避免爆炸
        return rng.normal(0.0, sig, size=int(n)).astype(float)
    eps_scale = sig * np.sqrt(max(0.0, 1.0 - rho * rho))
    x = np.empty(int(n), dtype=float)
    x[0] = rng.normal(0.0, sig)  # 从稳态起步
    if n == 1:
        return x
    eps = rng.normal(0.0, eps_scale, size=n - 1)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t - 1]
    return x


# 日期标签缓存：同 n_days 重复模拟时避免反复 f-string
_DATES_CACHE: dict[int, list[str]] = {}


def _daily_dfs_from_diff(diffs: np.ndarray) -> tuple[pl.DataFrame, pl.DataFrame]:
    """base_ic=0、cand_ic=diff，使日差分恰为模拟序列。"""
    n = len(diffs)
    dates = _DATES_CACHE.get(n)
    if dates is None:
        dates = [f"d{i:06d}" for i in range(n)]
        _DATES_CACHE[n] = dates
    ic = np.asarray(diffs, dtype=float)
    cand = pl.DataFrame(
        {"trade_date": dates, "ic": ic},
        schema={"trade_date": pl.Utf8, "ic": pl.Float64},
    )
    base = pl.DataFrame(
        {"trade_date": dates, "ic": np.zeros(n, dtype=float)},
        schema={"trade_date": pl.Utf8, "ic": pl.Float64},
    )
    return cand, base


def _one_decision(
    diffs: np.ndarray,
    *,
    block_days: int,
    threshold: float,
    se_mult: float,
    min_blocks: int,
) -> tuple[str, dict[str, Any]]:
    """单条 diff → paired_lift_stats →（可选 min_blocks 前置）→ lift_admission。"""
    cand, base = _daily_dfs_from_diff(diffs)
    stats = paired_lift_stats(cand, base, block_days=block_days)
    # 校准候选规则：n_blocks < min_blocks → reject（生产 lift_admission 无此参数）
    if int(min_blocks) > 0 and int(stats.get("n_blocks") or 0) < int(min_blocks):
        return "reject", stats
    decision = lift_admission(stats, threshold=threshold, se_mult=se_mult)
    return decision, stats


# ── 核心：单点 null 率 ───────────────────────────────────────────────────────


def null_admission_rates(
    *,
    n_days: int,
    block_days: int = DEFAULT_BLOCK_DAYS,
    daily_sigma: float,
    ar1: float = 0.0,
    threshold: float = DEFAULT_LIFT_THRESHOLD,
    se_mult: float = 1.0,
    n_sims: int = 10_000,
    seed: int = 0,
    n_candidates_batch: int = 10,
    min_blocks: int = 0,
) -> dict[str, Any]:
    """在 H0（无真实 lift）下估计单候选准入率与批 FWER。

    Parameters
    ----------
    n_days :
        配对评分日数（如 holdout ~290）。
    block_days :
        与 ``lift_test.DEFAULT_BLOCK_DAYS`` 一致（默认 20）。
    daily_sigma :
        日差分（cand_ic − base_ic）的标准差量级。
    ar1 :
        日差分 AR(1) 自相关（重叠前向收益导致；0=白噪声）。
    threshold / se_mult :
        透传 ``lift_admission``。
    n_sims :
        独立候选模拟次数。
    seed :
        复现种子。
    n_candidates_batch :
        同批候选数，用于 FWER（至少一个误 active）。
    min_blocks :
        校准**附加**规则：n_blocks < min_blocks → reject（默认 0=不设；
        生产代码无此参数，仅模拟层前置）。

    Returns
    -------
    dict
        ``p_active`` / ``p_probation`` / ``p_pass`` 及 Wilson 95% CI；
        ``fwer_active`` 含 analytic（1-(1-p)^n）与 simulated（按批抽）；
        ``mean_lift_se``（有限 SE 的均值，诊断用）；全部输入参数 + seed。
    """
    n_sims = int(n_sims)
    n_days = int(n_days)
    batch_n = max(1, int(n_candidates_batch))
    rng = np.random.default_rng(int(seed))

    n_active = 0
    n_probation = 0
    n_pass = 0
    se_sum = 0.0
    se_count = 0
    decisions: list[str] = []

    for _ in range(n_sims):
        diffs = _ar1_series(
            n_days, daily_sigma=daily_sigma, ar1=ar1, rng=rng,
        )
        decision, stats = _one_decision(
            diffs,
            block_days=int(block_days),
            threshold=float(threshold),
            se_mult=float(se_mult),
            min_blocks=int(min_blocks),
        )
        decisions.append(decision)
        if decision == "active":
            n_active += 1
            n_pass += 1
        elif decision == "probation":
            n_probation += 1
            n_pass += 1
        se = stats.get("lift_se")
        if se is not None and np.isfinite(float(se)):
            se_sum += float(se)
            se_count += 1

    p_active = n_active / n_sims if n_sims else 0.0
    p_probation = n_probation / n_sims if n_sims else 0.0
    p_pass = n_pass / n_sims if n_sims else 0.0

    # FWER：解析式 1-(1-p)^n；模拟 = 把独立候选按批切，批内任一 active 即 hit
    fwer_analytic = float(1.0 - (1.0 - p_active) ** batch_n)
    n_full_batches = n_sims // batch_n
    if n_full_batches > 0:
        hits = 0
        for b in range(n_full_batches):
            chunk = decisions[b * batch_n : (b + 1) * batch_n]
            if any(d == "active" for d in chunk):
                hits += 1
        fwer_sim = hits / n_full_batches
    else:
        fwer_sim = float("nan")

    return {
        "p_active": p_active,
        "p_probation": p_probation,
        "p_pass": p_pass,
        "p_active_ci": wilson_ci(n_active, n_sims),
        "p_probation_ci": wilson_ci(n_probation, n_sims),
        "p_pass_ci": wilson_ci(n_pass, n_sims),
        "fwer_active": {
            "analytic": fwer_analytic,
            "simulated": fwer_sim,
            "n_candidates_batch": batch_n,
            "n_batches_sim": n_full_batches,
        },
        "mean_lift_se": (se_sum / se_count) if se_count else float("nan"),
        "n_days": n_days,
        "block_days": int(block_days),
        "daily_sigma": float(daily_sigma),
        "ar1": float(ar1),
        "threshold": float(threshold),
        "se_mult": float(se_mult),
        "n_sims": n_sims,
        "seed": int(seed),
        "n_candidates_batch": batch_n,
        "min_blocks": int(min_blocks),
    }


# ── 扫参表 ───────────────────────────────────────────────────────────────────


def _rates_from_decisions(
    decisions: list[str],
    *,
    batch_n: int,
    se_vals: list[float],
) -> dict[str, Any]:
    """由决策列表与对应 lift_se 汇总率 / FWER / mean_lift_se。"""
    n_sims = len(decisions)
    n_active = sum(1 for d in decisions if d == "active")
    n_probation = sum(1 for d in decisions if d == "probation")
    n_pass = n_active + n_probation
    p_active = n_active / n_sims if n_sims else 0.0
    p_probation = n_probation / n_sims if n_sims else 0.0
    p_pass = n_pass / n_sims if n_sims else 0.0
    fwer_analytic = float(1.0 - (1.0 - p_active) ** batch_n)
    n_full = n_sims // batch_n
    if n_full > 0:
        hits = 0
        for b in range(n_full):
            chunk = decisions[b * batch_n : (b + 1) * batch_n]
            if any(d == "active" for d in chunk):
                hits += 1
        fwer_sim: float = hits / n_full
    else:
        fwer_sim = float("nan")
    finite_se = [s for s in se_vals if np.isfinite(s)]
    mean_se = float(np.mean(finite_se)) if finite_se else float("nan")
    return {
        "p_active": p_active,
        "p_probation": p_probation,
        "p_pass": p_pass,
        "p_active_ci": wilson_ci(n_active, n_sims),
        "fwer_active": fwer_analytic,
        "fwer_active_sim": fwer_sim,
        "mean_lift_se": mean_se,
        "n_sims": n_sims,
    }


def _decide(stats: dict[str, Any], *, threshold: float, se_mult: float, min_blocks: int) -> str:
    """模拟层决策：min_blocks 前置 + 生产 lift_admission。"""
    if int(min_blocks) > 0 and int(stats.get("n_blocks") or 0) < int(min_blocks):
        return "reject"
    return lift_admission(stats, threshold=threshold, se_mult=se_mult)


def calibration_table(
    *,
    n_days: int,
    daily_sigma: float,
    ar1: float = 0.0,
    se_mults: Sequence[float] = (1.0, 1.645, 2.0),
    min_blocks_options: Sequence[int] = (0, 6, 10),
    n_sims: int = 5000,
    seed: int = 0,
    block_days: int = DEFAULT_BLOCK_DAYS,
    threshold: float = DEFAULT_LIFT_THRESHOLD,
    n_candidates_batch: int = 10,
) -> list[dict[str, Any]]:
    """扫 (se_mult × min_blocks) 网格，返回校准表行。

    每组合一行：se_mult、min_blocks、p_active、p_pass、fwer_active(batch)。

    **共用底层样本**：先按 ``seed`` 生成 ``n_sims`` 条 H0 序列并只算一次
    ``paired_lift_stats``，再对全部 (se_mult, min_blocks) 复用同一 stats 集做
    决策——保证方向性比较干净，且扫参复杂度 ≈ 单点而非网格积。
    """
    n_sims = int(n_sims)
    n_days = int(n_days)
    batch_n = max(1, int(n_candidates_batch))
    bd = int(block_days)
    thr = float(threshold)
    rng = np.random.default_rng(int(seed))

    # 一次生成全部 H0 stats（昂贵路径只走一遍）
    all_stats: list[dict[str, Any]] = []
    se_vals: list[float] = []
    for _ in range(n_sims):
        diffs = _ar1_series(
            n_days, daily_sigma=daily_sigma, ar1=ar1, rng=rng,
        )
        cand, base = _daily_dfs_from_diff(diffs)
        stats = paired_lift_stats(cand, base, block_days=bd)
        all_stats.append(stats)
        se = stats.get("lift_se")
        se_vals.append(float(se) if se is not None else float("nan"))

    rows: list[dict[str, Any]] = []
    for se_m in se_mults:
        for mb in min_blocks_options:
            decisions = [
                _decide(s, threshold=thr, se_mult=float(se_m), min_blocks=int(mb))
                for s in all_stats
            ]
            rates = _rates_from_decisions(decisions, batch_n=batch_n, se_vals=se_vals)
            rows.append({
                "se_mult": float(se_m),
                "min_blocks": int(mb),
                "p_active": rates["p_active"],
                "p_pass": rates["p_pass"],
                "p_probation": rates["p_probation"],
                "fwer_active": rates["fwer_active"],
                "fwer_active_sim": rates["fwer_active_sim"],
                "p_active_ci": rates["p_active_ci"],
                "mean_lift_se": rates["mean_lift_se"],
                "n_sims": n_sims,
                "seed": int(seed),
            })
    return rows


def format_calibration_markdown(rows: Iterable[dict[str, Any]]) -> str:
    """把 ``calibration_table`` 行渲染为 markdown 表。"""
    header = (
        "| se_mult | min_blocks | p_active | p_pass | fwer_active(10) | "
        "p_active 95% CI |"
    )
    sep = "|--------:|-----------:|---------:|-------:|----------------:|:---------------|"
    lines = [header, sep]
    for r in rows:
        ci = r.get("p_active_ci", (None, None))
        if ci and ci[0] is not None:
            ci_s = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        else:
            ci_s = "—"
        lines.append(
            f"| {r['se_mult']:.3g} | {r['min_blocks']} | "
            f"{r['p_active']:.4f} | {r['p_pass']:.4f} | "
            f"{r['fwer_active']:.4f} | {ci_s} |"
        )
    return "\n".join(lines) + "\n"


# ── 经验参数反推 ─────────────────────────────────────────────────────────────


def estimate_daily_sigma_from_run(
    row: dict[str, Any],
    *,
    block_days: int | None = None,
) -> dict[str, Any]:
    """从一次真实 lift 结果行粗估 null 模拟用的 daily_sigma / n_days。

    假设（诚实标注，粗估即可）
    --------------------------------
    生产：``lift_se ≈ std(block_means) / √n_blocks``
    → ``block_mean_std ≈ lift_se · √n_blocks``。

    若块内日差分近独立，块均值方差 ≈ σ_daily² / L（L = 块长交易日），
    故 ``σ_daily ≈ block_mean_std · √L``，其中
    ``L = n_days / n_blocks``（有 n_days 时）或 ``block_days`` 默认 20。

    块内若存在正自相关，本估计**偏小**（真实 σ_daily 更大）。
    重叠前向收益导致的 AR(1) 需另用 ``ar1`` 参数显式建模，不在此反推。

    Parameters
    ----------
    row :
        形态对齐 manifest lift 行：至少 ``lift_se``、``n_blocks``；
        可选 ``n_days``。
    block_days :
        缺 ``n_days`` 时用此作块长（默认 ``DEFAULT_BLOCK_DAYS``）。

    Returns
    -------
    dict
        ``n_days``、``daily_sigma``、``block_mean_std``、``avg_block_days``、
        ``assumptions`` 说明字符串。
    """
    se = row.get("lift_se")
    n_blocks = row.get("n_blocks")
    if se is None or n_blocks is None:
        raise ValueError("row 需要 lift_se 与 n_blocks")
    se_f = float(se)
    nb = int(n_blocks)
    if nb < 1:
        raise ValueError("n_blocks 必须 ≥ 1")
    if not np.isfinite(se_f) or se_f < 0:
        raise ValueError("lift_se 必须为非负有限数")

    bd_default = int(block_days) if block_days is not None else int(DEFAULT_BLOCK_DAYS)
    if row.get("n_days") is not None:
        n_days = int(row["n_days"])
    else:
        n_days = nb * bd_default

    avg_block_days = n_days / nb if nb else float(bd_default)
    block_mean_std = se_f * np.sqrt(nb)
    daily_sigma = float(block_mean_std * np.sqrt(avg_block_days))

    return {
        "n_days": int(n_days),
        "daily_sigma": daily_sigma,
        "block_mean_std": float(block_mean_std),
        "avg_block_days": float(avg_block_days),
        "n_blocks": nb,
        "lift_se": se_f,
        "assumptions": (
            "块内近独立：σ_daily ≈ lift_se · √n_blocks · √(n_days/n_blocks)；"
            "块内正自相关时本估计偏小；AR(1) 请另设 ar1 参数。"
        ),
    }

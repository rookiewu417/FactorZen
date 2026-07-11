# src/factorzen/discovery/guardrails.py
"""防过拟合护栏的单点判定 + DSR deflation 配方 + 池级 PBO——消除 M1 与 M5/M6 双路径漂移。"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from factorzen.validation.deflated_sharpe import deflated_sharpe
from factorzen.validation.pbo import compute_pbo

# 护栏 DSR 显著性水平的**单一真源**——M1 与 M5/M6 一律引用它，防默认值漂移。
# 2026-07「松一档」：0.05 → 0.10（放宽多重检验后的显著性门槛）。改这一处即全局生效。
DEFAULT_DSR_ALPHA = 0.10


@dataclass(frozen=True)
class DeflationBasis:
    """DSR deflation 的基准：trial 池 IR 的经验方差 + 与之**同源**的 N + 统计量的边数。

    R8：``n_trials`` 与 ``sharpe_variance`` 必须来自同一批 trial，否则 ``expected_max_sharpe``
    的 deflation 基准不自洽。因 ``expected_max_sharpe ∝ sqrt(sharpe_variance)``，而多样化
    trial 池的经验方差恒大于 ``deflated_sharpe`` 的 H0 默认值 ``1/n_obs``，漏传 sharpe_variance
    会让门槛系统性偏小、放行过拟合因子（漂移倍数 ``sqrt(var_emp × n_obs)``，实测 1.60x）。

    ``two_sided`` 描述**选择规则**：按 ``|IR|`` 排序 / 接纳任一符号的路径填 True。它决定
    ``effective_trials``，并让 `deflated_pvalue` 自行对统计量取绝对值——调用方一律传带符号 IR。
    把两者绑在同一个字段上，是为了让「统计量 abs 了没有」与「基准 N 还是 2N」无法各说各话。

    M1(`mining_session`) 与 Agent(`agents/nodes`) 必须**共同调用**本类构造基准、经
    `deflated_pvalue` 求 p 值。有架构守卫测试禁止任一路径直接调 ``deflated_sharpe``。
    """

    n_trials: int
    sharpe_variance: float
    two_sided: bool = False

    @property
    def effective_trials(self) -> int:
        """deflation 实际使用的试验数。``n_trials`` 保持诚实计数（manifest 写它）。

        取绝对值 ⇒ 试验数翻倍。对称零分布下::

            P(max_{i≤N}|Z_i| ≤ t) = [2Φ(t)−1]^N ≈ [Φ(t)²]^N = Φ(t)^{2N} = P(max_{j≤2N} Z_j ≤ t)

        实测（400k 次重复）：拿 N 当 ``max|Z|`` 的基准会少算 0.20σ–0.41σ（N 越小越糟）；
        改 2N 后残差 ≤0.02σ，与公式自身对 ``E[max Z]`` 的逼近误差同量级。
        """
        return 2 * self.n_trials if self.two_sided else self.n_trials

    @classmethod
    def from_ir_pool(cls, ir_pool: Sequence[float | None], *,
                     two_sided: bool = False) -> DeflationBasis:
        """从「评估过且拿到有效 IR」的 trial 池构造。

        None（死表达式）与 nan/inf 一律剔除：它们会同时污染方差与计数——把 0.0 之类的
        sentinel 灌进池子会拉低经验方差，使 deflation 基准算在垃圾上。
        池大小 < 2 时经验方差无意义，退化为 1.0（与 M1 既有行为一致）。
        """
        arr = np.asarray([x for x in ir_pool if x is not None], dtype=float)
        arr = arr[np.isfinite(arr)]
        n = int(arr.size)
        return cls(n_trials=n, sharpe_variance=float(arr.var()) if n > 1 else 1.0,
                   two_sided=two_sided)


def deflated_pvalue(sharpe: float, basis: DeflationBasis, n_obs: int) -> tuple[float, float]:
    """(dsr, pvalue)。两条挖掘路径的 DSR 唯一入口。

    ``sharpe`` 一律传**带符号** IR；是否取绝对值由 ``basis.two_sided`` 决定——绝对值与
    ``effective_trials`` 必须成对出现，故只在此处施加，调用方不得自行 ``abs()``
    （有 ast 架构守卫）。

    ``n_obs`` 须是**该因子自己的有效 IC 天数**，不是 train 段日历交易日数——后者更大，
    会系统性放大显著性（``z ∝ sqrt(n_obs − 1)``）。
    """
    statistic = abs(sharpe) if basis.two_sided else sharpe
    return deflated_sharpe(statistic, basis.effective_trials, n_obs,
                           sharpe_variance=basis.sharpe_variance)


def guardrail_reasons(
    *,
    ic_train: float | None,
    holdout_ic: float | None,
    dsr_pvalue: float | None,
    ci_low: float | None = None,
    ci_high: float | None = None,
    dsr_alpha: float = DEFAULT_DSR_ALPHA,
) -> list[str]:
    """返回**未通过**的护栏门（空列表 = 全过）。`guardrail_passed` 委托本函数（无失败即通过），
    两者共用同一套门，杜绝「判定 / 解释」双路径漂移（陷阱#2）。

    门（2026-07「松一档」口径）：DSR 显著(pval<dsr_alpha，默认 0.10) + holdout 与 train
    **点估计同号**。必需量 ic_train/holdout_ic/dsr_pvalue 任一 None/NaN → 判缺失。

    历史（收紧口径）曾额外要求 holdout 95%CI 单边不跨零。实测该门在短 holdout 上对**真**因子
    误杀率高（97 天 holdout、真 IC=0.03 时 ~12–15%），且**从不独立生效**（大样本诊断里
    22/22 未过因子的 CI 门总与 DSR 同时亮红，0 个是仅被 CI 冤枉）。松一档移除它，holdout
    方向仅由点估计同号把关。``ci_low``/``ci_high`` 仍接收（供报告与向后兼容），不再参与判定。
    """
    required = {"ic_train": ic_train, "holdout_ic": holdout_ic, "dsr_pvalue": dsr_pvalue}
    missing = [k for k, v in required.items() if v is None or v != v]  # v != v 即 NaN
    if missing:
        return [f"缺失/NaN: {', '.join(missing)}"]
    reasons: list[str] = []
    if not (dsr_pvalue < dsr_alpha):  # type: ignore[operator]
        reasons.append(f"DSR 不显著(p={dsr_pvalue:.4f}≥{dsr_alpha})")
    if (holdout_ic > 0) != (ic_train > 0):  # type: ignore[operator]
        reasons.append(f"holdout 反号(train={ic_train:.4f}/holdout={holdout_ic:.4f})")
    return reasons


def guardrail_passed(
    *,
    ic_train: float | None,
    holdout_ic: float | None,
    dsr_pvalue: float | None,
    ci_low: float | None = None,
    ci_high: float | None = None,
    dsr_alpha: float = DEFAULT_DSR_ALPHA,
) -> bool:
    """DSR 显著(pval<dsr_alpha，默认 0.10) + holdout 点估计同号。必需量 None/NaN → False。

    委托 `guardrail_reasons`（无失败原因即通过），保证「过/不过」与「为什么不过」同源。
    2026-07「松一档」：默认 alpha 0.05→0.10，且移除 holdout CI 单边门（详见 guardrail_reasons）。
    """
    return not guardrail_reasons(
        ic_train=ic_train, holdout_ic=holdout_ic, dsr_pvalue=dsr_pvalue,
        ci_low=ci_low, ci_high=ci_high, dsr_alpha=dsr_alpha)


def pool_pbo(
    factor_dfs: list[pl.DataFrame],
    fwd_returns: pl.DataFrame,
    *,
    n_splits: int = 10,
    max_cand: int = 30,
) -> float:
    """对候选池因子帧算池级 PBO（CSCV）。候选<2 或周期不足 → nan。与 mining_session._pool_pbo 共享 compute_pbo。"""
    from factorzen.daily.evaluation.ic_analysis import compute_rank_ic
    from factorzen.daily.preprocessing.normalizer import cross_sectional_zscore

    series: list[np.ndarray] = []
    dates_ref = None
    for fdf in factor_dfs[:max_cand]:
        try:
            clean = cross_sectional_zscore(fdf, col="factor_value").rename(
                {"factor_value_z": "factor_clean"}
            )
            ic_res = compute_rank_ic(
                clean.select(["trade_date", "ts_code", "factor_clean"]),
                fwd_returns, factor_col="factor_clean", frequency="daily",
            )
            ser = ic_res.ic_series.sort("trade_date")
            if dates_ref is None:
                dates_ref = ser["trade_date"]
            ser = ser.join(
                pl.DataFrame({"trade_date": dates_ref}), on="trade_date", how="right"
            ).sort("trade_date")
            series.append(ser["ic"].fill_null(0.0).to_numpy())
        except Exception:
            continue
    if len(series) < 2:
        return float("nan")
    return compute_pbo(np.vstack(series), n_splits=n_splits)

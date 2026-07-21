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

# 因子**库**入池的 |train_IC| 下限——低于此视为纯噪声（非「弱但真」）。改这一处即全局生效。
DEFAULT_IC_FLOOR = 0.015

# 挖掘残差目标（对库正交后的残差 IC）的 |train residual IC| 下限。
# 残差分量天然小于裸 IC（共享方向已被剔除），故低于 DEFAULT_IC_FLOOR。
# **初值 0.010，待真实 team/search run 校准**——若放行过松/过紧，只改这一处。
DEFAULT_RESIDUAL_IC_FLOOR = 0.010

# holdout 有效 IC 天数下限。低于此视为「覆盖不足」而非「反号/无预测力」——
# 空/稀疏 holdout 的 ic_mean 哨兵 0.0 曾被同号门误杀（train>0）或假过关（train<0）。
DEFAULT_HOLDOUT_MIN_DAYS = 60

# lift 队列 |IC| 下界。噪声地板 ≈1SE（0.003–0.0045；SE≈σ/√n，日 IC σ 典型 0.1–0.15）；
# 新值 ≈2SE。依据 = 2026-07-16 战役实证（69 条队列 abs mean 0.0051、43 条落 [0.003,0.005)
# 噪声区、两次实弹 20 测 0 过）。若放行过松/过紧，只改这一处。
# 命名保留 GRAY：旧 manifest / 调用方兼容；语义上是 lift_queue 下界，**无上界**。
DEFAULT_GRAY_IC_FLOOR = 0.008
# raw 模式 lift 队列下界（裸 IC 噪声地板略高于残差；仍 < DEFAULT_IC_FLOOR 0.015；无上界）。
# 若放行过松/过紧，只改这一处。
DEFAULT_RAW_GRAY_IC_FLOOR = 0.010
# 组合增量 RankIC 阈值（初值待校准，lift 噪声地板需实测标定）。
DEFAULT_LIFT_THRESHOLD = 0.001
# 与库内因子「重复」硬拒阈值：corr > 此值 → library_correlated 一票否决。
# （0.7, 0.95] 为软信号「库相关持保留」，不否决、挡快速通道、可入 lift 队列。）
DEFAULT_DUPLICATE_CORR = 0.95

# 入池判据的两种口径：
#   "library"（默认，因子库化）：真(holdout 同号) + 有信号(|IC|≥floor)，不含 DSR 单星显著性。
#   "strict"（单明星）：DSR 显著 + holdout 同号（历史口径，供需要单因子独立显著时选用）。
DEFAULT_GATE = "library"

# reject_category：coverage 失败不得进 known_invalid 负例回灌（非方向性证据）。
REJECT_CATEGORY_HOLDOUT_COVERAGE = "holdout_coverage"
# 与库内 active 因子重复（corr > DEFAULT_DUPLICATE_CORR）：是「重复方向」非「无效」——不得混进 known_invalid。
REJECT_CATEGORY_LIBRARY_CORRELATED = "library_correlated"
# 历史灰区类别：已被 lift_queue 取代，仅旧 manifest 读取 / 负例过滤兼容保留，新写入勿用。
REJECT_CATEGORY_GRAY_ZONE = "gray_zone"
# 单因子主门不过但 |IC|≥下界、非重复 → 等待 session 末组合 lift 裁决（非 known_invalid）。
REJECT_CATEGORY_LIFT_QUEUE = "lift_queue"
# 组合层 lift 裁决 reject（组门不过 / 单候选 below_bar）→ 写回 experiment_index；
# 不进 known_invalid（组合无增量 ≠ 单因子无信号），走 known_lift_rejects 独立通道。
REJECT_CATEGORY_LIFT_REJECTED = "lift_rejected"
# 主门 |IC| 低于 floor（「残差IC太弱」/「train_IC 太弱」）：强度不足 ≠ 方向已验证无效。
# 不进 known_invalid 负例回灌，避免事件式提案被稀释后自强化「方向无效」死循环。
REJECT_CATEGORY_IC_TOO_WEAK = "ic_too_weak"

# ── 稀疏因子 sleeve 旁路（事件子集口径；不改主门阈值）────────────────────────
# |subset_ic_train| 下限：战役 express_yoy 真子集 ~0.02–0.05；0.03 为工程入队门槛。
SLEEVE_SUBSET_IC_FLOOR = 0.03
# train 子集有效 IC 天数下限（低于此视为样本不足，不进 sleeve 旁路）。
SLEEVE_SUBSET_MIN_DAYS = 40


def _holdout_direction_reasons(
    ic_train: float, holdout_ic: float, *, reason_style: str = "raw",
) -> list[str]:
    """覆盖充足后的方向门：严格同号（sign 积 > 0）；holdout 精确 0 →「无信号」非「反号」。

    ``reason_style="residual"`` 时文案加「残差」前缀，与裸 IC 死因区分。
    """
    if reason_style == "residual":
        if holdout_ic == 0.0:
            return [f"残差holdout无信号(train={ic_train:.4f}/holdout={holdout_ic:.4f})"]
        if (holdout_ic > 0) == (ic_train > 0) and ic_train != 0.0:
            return []
        return [f"残差holdout反号(train={ic_train:.4f}/holdout={holdout_ic:.4f})"]
    if holdout_ic == 0.0:
        return [f"holdout无信号(train={ic_train:.4f}/holdout={holdout_ic:.4f})"]
    # sign(h)*sign(t) > 0 ⇔ 双方同为正或同为负（0 已在上支处理）
    if (holdout_ic > 0) == (ic_train > 0) and ic_train != 0.0:
        return []
    return [f"holdout 反号(train={ic_train:.4f}/holdout={holdout_ic:.4f})"]


def _coverage_reason(
    holdout_n_days: int | None, holdout_min_days: int,
) -> str | None:
    """n_days 已知且不足 → 覆盖不足文案；None（旧调用方未传）→ 跳过，零回归。"""
    if holdout_n_days is None:
        return None
    if holdout_n_days < holdout_min_days:
        return f"holdout覆盖不足(days={holdout_n_days}/需{holdout_min_days})"
    return None


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
    holdout_n_days: int | None = None,
    holdout_min_days: int = DEFAULT_HOLDOUT_MIN_DAYS,
) -> list[str]:
    """返回**未通过**的护栏门（空列表 = 全过）。`guardrail_passed` 委托本函数（无失败即通过），
    两者共用同一套门，杜绝「判定 / 解释」双路径漂移（陷阱#2）。

    门（2026-07「松一档」+ 覆盖守卫）：DSR 显著(pval<dsr_alpha，默认 0.10) + holdout 覆盖充足
    + holdout 与 train **严格同号**。必需量 ic_train/holdout_ic/dsr_pvalue 任一 None/NaN → 判缺失。
    覆盖不足优先于同号门（不叫反号）；holdout 精确 0 →「无信号」。

    ``ci_low``/``ci_high`` 仍接收（供报告与向后兼容），不再参与判定。
    ``holdout_n_days=None``（旧调用方）跳过覆盖门，零回归。
    """
    # 覆盖不足时 holdout_ic 常为 nan（空因子帧）——覆盖门优先，不把缺数伪装成「缺失/NaN」。
    cov = _coverage_reason(holdout_n_days, holdout_min_days)
    if cov is not None:
        reasons: list[str] = []
        if dsr_pvalue is None or dsr_pvalue != dsr_pvalue:
            reasons.append("缺失/NaN: dsr_pvalue")
        elif not (dsr_pvalue < dsr_alpha):
            reasons.append(f"DSR 不显著(p={dsr_pvalue:.4f}≥{dsr_alpha})")
        if ic_train is None or ic_train != ic_train:
            reasons.append("缺失/NaN: ic_train")
        reasons.append(cov)
        return reasons
    required = {"ic_train": ic_train, "holdout_ic": holdout_ic, "dsr_pvalue": dsr_pvalue}
    missing = [k for k, v in required.items() if v is None or v != v]  # v != v 即 NaN
    if missing:
        return [f"缺失/NaN: {', '.join(missing)}"]
    reasons = []
    if not (dsr_pvalue < dsr_alpha):  # type: ignore[operator]
        reasons.append(f"DSR 不显著(p={dsr_pvalue:.4f}≥{dsr_alpha})")
    reasons.extend(_holdout_direction_reasons(ic_train, holdout_ic))  # type: ignore[arg-type]
    return reasons


def guardrail_passed(
    *,
    ic_train: float | None,
    holdout_ic: float | None,
    dsr_pvalue: float | None,
    ci_low: float | None = None,
    ci_high: float | None = None,
    dsr_alpha: float = DEFAULT_DSR_ALPHA,
    holdout_n_days: int | None = None,
    holdout_min_days: int = DEFAULT_HOLDOUT_MIN_DAYS,
) -> bool:
    """DSR 显著(pval<dsr_alpha，默认 0.10) + holdout 覆盖 + 点估计同号。必需量 None/NaN → False。

    委托 `guardrail_reasons`（无失败原因即通过），保证「过/不过」与「为什么不过」同源。
    """
    return not guardrail_reasons(
        ic_train=ic_train, holdout_ic=holdout_ic, dsr_pvalue=dsr_pvalue,
        ci_low=ci_low, ci_high=ci_high, dsr_alpha=dsr_alpha,
        holdout_n_days=holdout_n_days, holdout_min_days=holdout_min_days)


def _ic_weak_reason(ic_train: float, ic_floor: float, *, reason_style: str = "raw") -> str:
    if reason_style == "residual":
        return f"残差IC太弱(|{ic_train:.4f}|<{ic_floor})"
    return f"train_IC 太弱(|{ic_train:.4f}|<{ic_floor})"


def library_reasons(
    *,
    ic_train: float | None,
    holdout_ic: float | None,
    ic_floor: float = DEFAULT_IC_FLOOR,
    holdout_n_days: int | None = None,
    holdout_min_days: int = DEFAULT_HOLDOUT_MIN_DAYS,
    reason_style: str = "raw",
) -> list[str]:
    """因子**库**入池判据（2026-07 因子库化 + 覆盖守卫）：
    覆盖充足 + 真（holdout 与 train 严格同号）+ 有信号（``|train_IC| >= ic_floor``）。

    **不含 DSR 单星显著性**——显著性挪到组合层。去相关由 `max_correlation` 另判。

    挡三类：**覆盖不足**（缺数据，非方向证据）、**假**（真反号）、**纯噪声**（|IC| 太弱）。
    覆盖不足优先报告、不与「反号」并用；holdout 精确 0 →「无信号」。
    ``holdout_n_days=None`` 跳过覆盖门（旧调用方零回归）。
    必需量 None/NaN → 判缺失（保守不入池）。

    ``reason_style="residual"``：挖掘残差目标路径——死因文案写「残差IC太弱/残差holdout反号」，
    调用方应传入残差指标 + ``DEFAULT_RESIDUAL_IC_FLOOR``。因子库 upsert/rebuild **不得**
    用 residual 口径（库是参照系，对自身残差化是循环定义）。
    """
    # 覆盖不足优先：空 holdout 常伴随 holdout_ic=nan，不得落到「缺失/NaN」或「反号」。
    cov = _coverage_reason(holdout_n_days, holdout_min_days)
    if cov is not None:
        reasons = []
        if ic_train is None or ic_train != ic_train:
            return ["缺失/NaN: ic_train", cov]
        if abs(ic_train) < ic_floor:
            reasons.append(_ic_weak_reason(ic_train, ic_floor, reason_style=reason_style))
        reasons.append(cov)
        return reasons
    required = {"ic_train": ic_train, "holdout_ic": holdout_ic}
    missing = [k for k, v in required.items() if v is None or v != v]
    if missing:
        return [f"缺失/NaN: {', '.join(missing)}"]
    reasons = []
    if abs(ic_train) < ic_floor:  # type: ignore[arg-type]
        reasons.append(_ic_weak_reason(ic_train, ic_floor, reason_style=reason_style))  # type: ignore[arg-type]
    reasons.extend(_holdout_direction_reasons(
        ic_train, holdout_ic, reason_style=reason_style))  # type: ignore[arg-type]
    return reasons


def classify_reject_category(reasons: list[str]) -> str | None:
    """从护栏 reason 列表提取死因类别（供 experiment_index 过滤）。无匹配 → None。

    优先级：覆盖不足 > 反号/无信号（维持 None，方向证据可进 known_invalid）
    > 太弱（``ic_too_weak``，强度不足不进 known_invalid）。
    """
    for r in reasons:
        if "覆盖不足" in r:
            return REJECT_CATEGORY_HOLDOUT_COVERAGE
    # 反号 / 无信号：方向性证据，明确不打类别（None → known_invalid 可吸收）
    for r in reasons:
        if "反号" in r or "无信号" in r:
            return None
    for r in reasons:
        if "太弱" in r:
            return REJECT_CATEGORY_IC_TOO_WEAK
    return None


def _use_residual_objective(candidate: dict, objective: str | None) -> bool:
    """候选按残差口径还是裸 IC 口径判定。"""
    if objective == "raw":
        return False
    if objective == "residual":
        return True
    # 未显式指定：有残差 train IC 即 residual 口径（挖掘落盘字段）
    return candidate.get("residual_ic_train") is not None


def ic_floor_for_candidate(candidate: dict, *, objective: str | None = None) -> float:
    """候选适用的 lift 队列 |IC| 噪声地板。

    残差口径 → ``DEFAULT_GRAY_IC_FLOOR``；裸 IC 口径 → ``DEFAULT_RAW_GRAY_IC_FLOOR``。
    """
    if _use_residual_objective(candidate, objective):
        return DEFAULT_GRAY_IC_FLOOR
    return DEFAULT_RAW_GRAY_IC_FLOOR


def candidate_train_ic(candidate: dict, *, objective: str | None = None) -> float | None:
    """候选用于与地板比较的 train |IC| 原值；None/NaN → None。"""
    if _use_residual_objective(candidate, objective):
        ic = candidate.get("residual_ic_train")
    else:
        ic = candidate.get("ic_train")
    if ic is None:
        return None
    try:
        val = float(ic)
    except (TypeError, ValueError):
        return None
    if val != val:  # NaN
        return None
    return val


def is_sub_floor_candidate(
    candidate: dict,
    *,
    objective: str | None = None,
    floor: float | None = None,
) -> bool:
    """候选的 |train IC| 是否低于噪声地板（= lift 队列下界）。

    **只看地板这一条**（不含 ``is_lift_queue_candidate`` 的覆盖门 / 库重复门），
    供组门连坐防呆预过滤使用：把「按当前地板本就不该在队列里」的历史积压剔出组门，
    避免噪声在等权残差组合里稀释真信号、导致整组连坐拒
    （2026-07-17 事故：130/150 条 sub-floor → 组 lift=-0.0007 全体拒）。

    ``floor``：显式覆盖；``None`` 时按候选口径取
    ``DEFAULT_GRAY_IC_FLOOR`` / ``DEFAULT_RAW_GRAY_IC_FLOOR``。

    **无可比 IC 指标（None/NaN）→ False**（不算 sub-floor，不过滤）。
    只拦「实测到的噪声」，不拦「没测过的」——``--factor`` 注入的 registry python
    候选不带 ic_train，判 True 会把该路径整个过滤空。
    """
    ic = candidate_train_ic(candidate, objective=objective)
    if ic is None:
        return False
    lo = floor if floor is not None else ic_floor_for_candidate(
        candidate, objective=objective,
    )
    return abs(ic) < float(lo)


def is_lift_queue_candidate(candidate: dict, *, objective: str | None = None) -> bool:
    """lift 队列判定：对组合 lift 实验仍有价值的候选（无 IC 上界）。

    条件（全部满足）：
    1. **覆盖门过**——``n_holdout_days ≥ DEFAULT_HOLDOUT_MIN_DAYS``
       （residual 模式用 ``n_residual_holdout_days``）。
    2. **非库重复**——未被 ``library_correlated`` 拒
       （``reject_category`` / ``library_correlated`` 旗标 /
       ``max_corr_library > DEFAULT_DUPLICATE_CORR``）。
    3. **|IC| ≥ 下界（无上界）**——残差：``|residual_ic_train| ≥ DEFAULT_GRAY_IC_FLOOR``；
       裸 IC：``|ic_train| ≥ DEFAULT_RAW_GRAY_IC_FLOOR``。

    **不检查 passed**：调用方负责「不给已通过主门/快速通道的候选打队列标记」
    （``if not passed and is_lift_queue_candidate(...)``）。本函数本身不读 ``passed``。

    **不要求 holdout 同号**：lift 实验本身就是 OOS 裁决；弱因子反号税不在此门重复征收。

    ``objective``：``"residual"`` / ``"raw"``；``None`` 时若候选带 residual 指标则 residual。
    """
    if candidate.get("reject_category") == REJECT_CATEGORY_LIBRARY_CORRELATED:
        return False
    if candidate.get("library_correlated"):
        return False
    mc = candidate.get("max_corr_library")
    if mc is not None:
        try:
            if abs(float(mc)) > DEFAULT_DUPLICATE_CORR:
                return False
        except (TypeError, ValueError):
            pass

    if _use_residual_objective(candidate, objective):
        n_days = candidate.get("n_residual_holdout_days")
    else:
        n_days = candidate.get("n_holdout_days")
        if n_days is None:
            n_days = candidate.get("holdout_n_days")

    if n_days is None:
        return False
    try:
        if int(n_days) < DEFAULT_HOLDOUT_MIN_DAYS:
            return False
    except (TypeError, ValueError):
        return False
    # 条件 3（地板）与 is_sub_floor_candidate 共用单一真源
    if candidate_train_ic(candidate, objective=objective) is None:
        return False
    return not is_sub_floor_candidate(candidate, objective=objective)


def is_gray_zone(candidate: dict, *, objective: str | None = None) -> bool:
    """**Deprecated**：语义已变为 lift 队列（无 IC 上界），请改用 ``is_lift_queue_candidate``。

    薄别名，行为与 ``is_lift_queue_candidate`` 完全一致（上界已取消；库重复阈 0.95）。
    """
    return is_lift_queue_candidate(candidate, objective=objective)


def is_sleeve_lift_candidate(
    candidate: dict,
    *,
    sleeve_gate: bool = True,
) -> bool:
    """稀疏 / 事件掩码因子 sleeve 旁路：主门不过时可否进 ``lift_queue``（**不**直接 passed）。

    条件（全部满足）：
    1. ``sleeve_gate`` 开启
    2. **触发通道**：``is_sparse``（值稀疏，一期）**或** ``subset_mask_leaves`` 非空
       （事件掩码，二期；LLM 包装后值覆盖可到 1.0）
    3. ``|subset_ic_train| ≥ SLEEVE_SUBSET_IC_FLOOR``（默认 0.03）
    4. ``subset_ic_holdout`` 与 train **严格同号**（任一侧 0 / None → 拒）
    5. ``subset_n_days_train ≥ SLEEVE_SUBSET_MIN_DAYS``（默认 40）

    **独立于** ``is_lift_queue_candidate`` 的全截面 |IC| 地板——事件叶 fill-0
    全截面 IC 被稀释后常低于 gray floor，子集才是真口径。

    调用方契约：``if not passed and is_sleeve_lift_candidate(...)`` 时打
    ``reject_category=lift_queue`` + ``sleeve_candidate=True``，绝不改 ``passed``。
    """
    if not sleeve_gate:
        return False
    mask_leaves = candidate.get("subset_mask_leaves")
    mask_triggered = bool(mask_leaves)
    if not candidate.get("is_sparse") and not mask_triggered:
        return False
    sic_tr = candidate.get("subset_ic_train")
    sic_h = candidate.get("subset_ic_holdout")
    n_days = candidate.get("subset_n_days_train")
    if sic_tr is None or sic_h is None or n_days is None:
        return False
    try:
        tr = float(sic_tr)
        ho = float(sic_h)
        nd = int(n_days)
    except (TypeError, ValueError):
        return False
    if tr != tr or ho != ho:  # NaN
        return False
    if abs(tr) < SLEEVE_SUBSET_IC_FLOOR:
        return False
    if nd < SLEEVE_SUBSET_MIN_DAYS:
        return False
    # 严格同号；0 视为无信号
    if tr == 0.0 or ho == 0.0:
        return False
    return (ho > 0) == (tr > 0)


def acceptance_reasons(
    *,
    gate: str = DEFAULT_GATE,
    ic_train: float | None,
    holdout_ic: float | None,
    dsr_pvalue: float | None = None,
    ci_low: float | None = None,
    ci_high: float | None = None,
    dsr_alpha: float = DEFAULT_DSR_ALPHA,
    ic_floor: float = DEFAULT_IC_FLOOR,
    holdout_n_days: int | None = None,
    holdout_min_days: int = DEFAULT_HOLDOUT_MIN_DAYS,
    reason_style: str = "raw",
) -> list[str]:
    """按 ``gate`` 口径返回未通过的入池判据（空=入池）。两条挖掘路径的**统一入口**，防漂移。

    ``gate="library"``（默认，因子库化）→ `library_reasons`（真+有信号，DSR 挪到组合层）；
    ``gate="strict"``（单明星）→ `guardrail_reasons`（DSR 显著+holdout 同号）。

    ``reason_style="residual"`` 只影响 library 门文案（残差IC太弱/残差holdout反号）；
    调用方负责把残差指标填进 ic_train/holdout_ic 并把 ic_floor 设为
    ``DEFAULT_RESIDUAL_IC_FLOOR``。
    """
    if gate == "strict":
        return guardrail_reasons(
            ic_train=ic_train, holdout_ic=holdout_ic, dsr_pvalue=dsr_pvalue,
            ci_low=ci_low, ci_high=ci_high, dsr_alpha=dsr_alpha,
            holdout_n_days=holdout_n_days, holdout_min_days=holdout_min_days)
    if gate != "library":
        raise ValueError(f"未知 gate={gate!r}，应为 'library' 或 'strict'")
    return library_reasons(
        ic_train=ic_train, holdout_ic=holdout_ic, ic_floor=ic_floor,
        holdout_n_days=holdout_n_days, holdout_min_days=holdout_min_days,
        reason_style=reason_style)


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

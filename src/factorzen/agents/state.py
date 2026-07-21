"""Agent 闭环的显式状态（JSON 可序列化 dataclass）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class AttemptRecord:
    iteration: int
    hypothesis: str
    expression: str
    compile_ok: bool
    ic_train: float | None
    passed_guardrails: bool
    critic_verdict: str | None
    error: str | None
    ir_train: float | None = None
    turnover: float | None = None
    # 该因子在 train 段的有效 IC 天数（DSR 的 n_obs，对齐 M1 的 c["n_train"]）；
    # 不是 train 段日历交易日数——后者更大，会系统性放大显著性。
    n_train: int | None = None
    # 过了护栏但与已有候选高度相关，故未入候选池。这是**决策**，与 passed_guardrails
    # 这个**事实**分开记：把它标成 passed=False 会让它落进 known_invalid 被当作
    # 「已验证无效」喂给 LLM——语义污染。
    decorrelated: bool = False
    # 未过护栏/未入候选池的原因（人类可读，供进度与收尾"近失表"展示）。None=通过或未评估。
    reject_reason: str | None = None
    # 死因类别（如 holdout_coverage）；known_invalid 据此过滤非方向性失败。
    reject_category: str | None = None
    # holdout 段有效 IC 天数（覆盖守卫 / Critic 摘要）；None=未跑 holdout。
    n_holdout_days: int | None = None
    # 残差目标双指标（objective=residual 时由 node_guardrails 填；裸 IC 仍在 ic_train/holdout）
    residual_ic_train: float | None = None
    residual_holdout_ic: float | None = None
    n_residual_holdout_days: int | None = None
    # 稀疏因子子集评估（evaluation 层；消费方缺字段须容忍）
    nonzero_coverage: float | None = None
    is_sparse: bool = False
    subset_ic_train: float | None = None
    subset_n_days_train: int | None = None
    subset_ic_holdout: float | None = None
    subset_n_days_holdout: int | None = None
    # 事件掩码触发叶列表（二期；None/[] = 未走掩码通道）
    subset_mask_leaves: list[str] | None = None
    # sleeve 旁路进 lift 队列标记（不直接 passed；lift 层零改）
    sleeve_candidate: bool = False


@dataclass
class AgentState:
    seed: int
    iteration: int = 0
    attempts: list[AttemptRecord] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    seen_expressions: set[str] = field(default_factory=set)
    # 截面 rank 指纹去重（W4；session 级持久 set，evaluate_expressions 就地更新）
    seen_fingerprints: set[str] = field(default_factory=set)
    negative_examples: list[str] = field(default_factory=list)
    pbo: float | None = None
    # 库级正交：session 开始物化的库池大小 + 因 library_correlated 被拒的累计数（manifest 用）
    library_pool_size: int = 0
    n_library_correlated_rejects: int = 0
    # 灰区候选计数（单因子门不过但落 is_gray_zone；后置 lift 通道；manifest 用）
    n_gray_zone: int = 0
    # 挖掘评估目标：raw | residual（库空时 residual 自动退化 raw，由 resolve_objective 写回）
    objective: str = "residual"

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "iteration": self.iteration,
            "attempts": [asdict(a) for a in self.attempts],
            "candidates": self.candidates,
            "seen_expressions": sorted(self.seen_expressions),
            "negative_examples": self.negative_examples,
            "pbo": self.pbo,
            "library_pool_size": self.library_pool_size,
            "n_library_correlated_rejects": self.n_library_correlated_rejects,
            "n_gray_zone": self.n_gray_zone,
            "objective": self.objective,
        }

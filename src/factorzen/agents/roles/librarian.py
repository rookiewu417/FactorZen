# src/factorzen/agents/roles/librarian.py
"""Librarian 角色：跨 session 长期记忆的读（recall）与写（record）。"""
from __future__ import annotations

from dataclasses import dataclass

from factorzen.discovery.expression import parse_expr, to_expr_string

# 叶子级指导阈值：方向尝试（排除 coverage 失败）≥ 此值且 0 过关 → 挖穿区。
EXHAUSTED_MIN_TRIES = 15
# 本 session 存活叶子中，历史唯一表达式数 ≤ 此值 → 未探索区（优先考虑）。
UNEXPLORED_MAX_TRIES = 2


def _normalize(expr: str) -> str:
    try:
        return to_expr_string(parse_expr(expr))
    except ValueError:
        return expr


@dataclass
class Recall:
    """Librarian 检索结果（亦称 LibrarianBriefing）。

    ``leaf_guidance``：叶子级挖穿/未探索指导；``leaf_names`` 未传入时为 None（零回归）。
    ``library_covered``：库内 active 高 |IC| 表达式（供 Hypothesis 追求正交）；None → 不注入。
    ``lift_rejected``：组合层 lift 拒绝方向（``known_lift_rejects``）；空列表落 None。
    ``exhausted_leaves``：原始挖穿叶名列表（硬过滤用，非格式化文案）；无 dig 时 None。
    """
    seen: set[str]
    known_invalid: list[str]
    known_valid: list[str]
    leaf_guidance: dict[str, list[str]] | None = None
    library_covered: list[str] | None = None
    lift_rejected: list[dict] | None = None
    exhausted_leaves: list[str] | None = None


# 向后兼容别名（任务文档称 LibrarianBriefing）
LibrarianBriefing = Recall


def build_leaf_guidance(
    stats: dict[str, dict],
    leaf_names: list[str],
    *,
    exhausted_min: int | None = None,
    unexplored_max: int | None = None,
) -> dict[str, list[str]]:
    """从 leaf_stats 构建挖穿/未探索列表（只含 ``leaf_names`` 中的存活叶子）。

    - 挖穿区：``n_exprs - n_coverage_fail >= exhausted_min`` 且 ``n_passed == 0``
      （coverage 失败不算方向尝试）
    - 未探索区：``n_exprs <= unexplored_max``

    阈值默认读模块常量（调用时解析，便于测试 monkeypatch）。
    """
    if exhausted_min is None:
        exhausted_min = EXHAUSTED_MIN_TRIES
    if unexplored_max is None:
        unexplored_max = UNEXPLORED_MAX_TRIES
    exhausted: list[str] = []
    unexplored: list[str] = []
    for name in leaf_names:
        st = stats.get(name) or {
            "n_exprs": 0, "n_passed": 0, "best_abs_ic": 0.0, "n_coverage_fail": 0,
        }
        n_exprs = int(st.get("n_exprs") or 0)
        n_passed = int(st.get("n_passed") or 0)
        n_cov = int(st.get("n_coverage_fail") or 0)
        best = float(st.get("best_abs_ic") or 0.0)
        direction_tries = n_exprs - n_cov
        if direction_tries >= exhausted_min and n_passed == 0:
            exhausted.append(
                f"{name}(试 {direction_tries} 次 {n_passed} 过关, best|IC|={best:.3f})"
            )
        if n_exprs <= unexplored_max:
            unexplored.append(name)
    return {"exhausted": exhausted, "unexplored": unexplored}


def raw_exhausted_leaves(
    stats: dict[str, dict],
    leaf_names: list[str],
    *,
    exhausted_min: int | None = None,
) -> list[str]:
    """挖穿叶的**裸名**列表（硬过滤用；口径与 ``build_leaf_guidance`` 一致）。"""
    if exhausted_min is None:
        exhausted_min = EXHAUSTED_MIN_TRIES
    out: list[str] = []
    for name in leaf_names:
        st = stats.get(name) or {
            "n_exprs": 0, "n_passed": 0, "best_abs_ic": 0.0, "n_coverage_fail": 0,
        }
        n_exprs = int(st.get("n_exprs") or 0)
        n_passed = int(st.get("n_passed") or 0)
        n_cov = int(st.get("n_coverage_fail") or 0)
        direction_tries = n_exprs - n_cov
        if direction_tries >= exhausted_min and n_passed == 0:
            out.append(name)
    return out


def recall(
    index,
    *,
    k: int = 5,
    data_window: dict | None = None,
    leaf_names: list[str] | None = None,
    library_covered: list[str] | None = None,
) -> Recall:
    """召回本数据窗口内的历史。

    `data_window=None` → 不限定窗口（向后兼容）。限定时，跨窗口的历史不会被喂给 LLM：
    一个窗口上「已验证有效」的因子，换个窗口未必成立。

    ``leaf_names``：本 session **存活**叶子（leaf_health 摘除后）。传入时重算
    ``leaf_stats`` 并生成 ``leaf_guidance``；死叶不出现在挖穿/未探索任一侧。

    ``library_covered``：库内 active 高 IC 表达式列表（预构建）；None → 不注入（零回归）。
    """
    leaf_guidance = None
    exhausted_leaves: list[str] | None = None
    if leaf_names is not None:
        stats = index.leaf_stats(leaf_names, data_window=data_window)
        leaf_guidance = build_leaf_guidance(stats, list(leaf_names))
        raw = raw_exhausted_leaves(stats, list(leaf_names))
        exhausted_leaves = raw or None
    lift_rej = index.known_lift_rejects(k=k, data_window=data_window)
    return Recall(
        seen=index.seen_expressions(data_window=data_window),
        known_invalid=index.known_invalid(k=k, data_window=data_window),
        known_valid=index.known_valid(k=k, data_window=data_window),
        leaf_guidance=leaf_guidance,
        library_covered=library_covered,
        lift_rejected=lift_rej or None,
        exhausted_leaves=exhausted_leaves,
    )


def record(
    index,
    attempts,
    run_id: str,
    *,
    candidates: list[dict] | None = None,
    data_window: dict | None = None,
    campaign_id: str | None = None,
) -> None:
    """把本 run 所有 AttemptRecord 写入 experiment_index。

    落盘的是**事实**：`passed`（过了定量护栏）、`verdict`（Critic 裁决）、
    `decorrelated`（因与已有候选高度相关而未入候选池）。
    「可否借鉴」这个**决策**不在此计算，由 `ExperimentIndex.known_valid()` 综合三者推出
    ——一处判定，避免同一语义散落在写入侧的多个分支里互相矛盾。

    candidates: 可选。含 holdout_ic 的候选列表，用于归一化匹配后回填 holdout_ic 到记录，
    供 known_valid 按 |holdout_ic| 排序。

    campaign_id: 完整统计问题 key（market/universe/start/end/holdout/objective/horizon/gate
    的哈希）。写在行顶层，**不**塞进 data_window——window_key 语义是数据窗，
    被 seen_expressions/known_invalid 等消费，不能混入 objective/horizon/gate。
    """
    # 构建 holdout_ic 查找字典（归一化匹配，Important 2）
    hic_map: dict[str, float] = {}
    if candidates:
        for c in candidates:
            if "expression" in c and c.get("holdout_ic") is not None:
                hic_map[_normalize(c["expression"])] = c["holdout_ic"]

    records = []
    for a in attempts:
        # 编译失败的表达式也要落盘：seen_expressions() 靠它跨 session 去重，
        # 否则下个 session 会重新生成同一个语法坑，白烧 LLM 调用与自愈轮次。
        # 它们的 ir_train=None，被 DeflationBasis.from_ir_pool 剔除，不污染 deflation 池；
        # known_invalid() 也会排除它们（那里的语义是「能编译但无效」）。
        rec: dict = {
            "expression": a.expression,
            "hypothesis": a.hypothesis,
            "ic_train": a.ic_train,
            # DSR 的 deflation 池要的是 **IR**，不是 IC。没有 ir_train / n_train，
            # 将来永远无法从 index 重建历史 IR 池去做跨 session 的多重检验 N 累积。
            # 记录它们不承诺任何统计立场，只是保住那个可能性（见 F2 的设计讨论）。
            "ir_train": a.ir_train,
            "n_train": a.n_train,
            "passed": a.passed_guardrails,          # 事实：过了定量护栏
            "verdict": a.critic_verdict,            # 决策：Critic 裁决（known_valid 会读它）
            "decorrelated": a.decorrelated,         # 决策：与已有候选高度相关，未入候选池
            "compile_ok": a.compile_ok,             # 事实：表达式是否可解析
            "error": a.error,
            "reject_reason": a.reject_reason,       # 护栏/去相关死因文案
            "reject_category": a.reject_category,   # 如 holdout_coverage → known_invalid 过滤
            "n_holdout_days": a.n_holdout_days,
            "data_window": data_window,             # 数据窗：(start,end,universe,market)
            "campaign_id": campaign_id,             # 完整统计问题族（顶层，非 data_window）
            "run_id": run_id,
        }
        # 回填 holdout_ic（归一化匹配）
        hic = hic_map.get(_normalize(a.expression))
        if hic is not None:
            rec["holdout_ic"] = hic
        records.append(rec)
    index.append(records)

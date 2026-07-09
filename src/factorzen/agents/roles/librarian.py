# src/factorzen/agents/roles/librarian.py
"""Librarian 角色：跨 session 长期记忆的读（recall）与写（record）。"""
from __future__ import annotations

from dataclasses import dataclass

from factorzen.discovery.expression import parse_expr, to_expr_string


def _normalize(expr: str) -> str:
    try:
        return to_expr_string(parse_expr(expr))
    except ValueError:
        return expr


@dataclass
class Recall:
    seen: set[str]
    known_invalid: list[str]
    known_valid: list[str]


def recall(index, *, k: int = 5, data_window: dict | None = None) -> Recall:
    """召回本数据窗口内的历史。

    `data_window=None` → 不限定窗口（向后兼容）。限定时，跨窗口的历史不会被喂给 LLM：
    一个窗口上「已验证有效」的因子，换个窗口未必成立。
    """
    return Recall(
        seen=index.seen_expressions(data_window=data_window),
        known_invalid=index.known_invalid(k=k, data_window=data_window),
        known_valid=index.known_valid(k=k, data_window=data_window),
    )


def record(
    index,
    attempts,
    run_id: str,
    *,
    candidates: list[dict] | None = None,
    data_window: dict | None = None,
) -> None:
    """把本 run 所有 AttemptRecord 写入 experiment_index。

    落盘的是**事实**：`passed`（过了定量护栏）、`verdict`（Critic 裁决）、
    `decorrelated`（因与已有候选高度相关而未入候选池）。
    「可否借鉴」这个**决策**不在此计算，由 `ExperimentIndex.known_valid()` 综合三者推出
    ——一处判定，避免同一语义散落在写入侧的多个分支里互相矛盾。

    candidates: 可选。含 holdout_ic 的候选列表，用于归一化匹配后回填 holdout_ic 到记录，
    供 known_valid 按 |holdout_ic| 排序。
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
            "data_window": data_window,             # 族边界：(start,end,universe,market)
            "run_id": run_id,
        }
        # 回填 holdout_ic（归一化匹配）
        hic = hic_map.get(_normalize(a.expression))
        if hic is not None:
            rec["holdout_ic"] = hic
        records.append(rec)
    index.append(records)

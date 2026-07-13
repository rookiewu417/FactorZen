# tests/test_experiment_index_contract.py
"""长期记忆的契约：`passed_guardrails` 是**事实**，「可否借鉴」是**决策**。

三个症状看似独立，实则纠缠于同一个问题——`passed_guardrails` 究竟是
「过了定量护栏」这个事实，还是「背书为已知有效、可供后续 session 借鉴」这个决策？
它们拉向相反方向：

| 症状 | 修复前 | 事实语义下应为 |
|---|---|---|
| 去相关剔除（`corr>0.7`） | `passed=False` | `True` —— 它确实过了护栏 |
| Critic `revise_hypothesis` | `passed=True` 且进 known_valid | `True`，但不该被借鉴 |
| Critic `drop`（commit 1e0bda4） | 被 **mutate** 成 `False` | 不该 mutate |

1e0bda4 已经在 mutate 一个事实字段来编码复用决策——那是矛盾的根源。分别打三个补丁
只会烙进一个自相矛盾的索引契约。

本文件锁定的契约：
- `passed_guardrails` **一旦为 True 永不改回 False**（不可变事实）
- 「可否借鉴」由 `known_valid()` 计算：
  `passed AND verdict 未否决 AND 未被去相关剔除`
- 顺带修掉 `verdict` 这个「写入 index 却无人读」的死字段
"""
from __future__ import annotations

from factorzen.agents.experiment_index import ExperimentIndex
from factorzen.agents.roles.librarian import record
from factorzen.discovery.expression import is_lookahead_expr


def _idx(tmp_path) -> ExperimentIndex:
    return ExperimentIndex(str(tmp_path / "idx.jsonl"))


def _rec(expr: str, *, passed: bool, verdict: str | None = "keep",
         holdout_ic: float | None = None, decorrelated: bool = False,
         ic_train: float = 0.03) -> dict:
    r = {"expression": expr, "hypothesis": "h", "ic_train": ic_train,
         "passed": passed, "verdict": verdict, "decorrelated": decorrelated,
         "run_id": "t"}
    if holdout_ic is not None:
        r["holdout_ic"] = holdout_ic
    return r


# ── F4：known_valid 必须按 |holdout_ic| 排序 ────────────────────────────────


def test_known_valid_ranks_by_abs_holdout_ic_so_reversal_factors_survive(tmp_path):
    """护栏明确接纳负 IC 反转因子（`guardrail_passed` 的 same_sign + ci_high<0 分支）。

    带符号降序会把**最强的反转因子**（holdout_ic=-0.09，|IC| 最大）排到最后，
    top-k 截断时被弱多头挤出「已验证有效，可借鉴其思路方向」清单——系统性把 LLM
    的借鉴方向偏离反转因子族。
    """
    idx = _idx(tmp_path)
    idx.append([
        _rec("reversal", passed=True, holdout_ic=-0.09),      # |IC| 最大，方向为负
        *[_rec(f"weak_long_{i}", passed=True, holdout_ic=0.01 + 0.01 * i) for i in range(5)],
    ])

    valid = idx.known_valid(k=5)

    assert "reversal" in valid, f"最强的反转因子被挤出 known_valid: {valid}"
    assert valid[0] == "reversal", "按 |holdout_ic| 排序时反转因子应排第一"


def test_known_invalid_still_ranks_by_abs_ic_train(tmp_path):
    """回归：known_invalid 一直用 abs()，本次不动。"""
    idx = _idx(tmp_path)
    idx.append([
        _rec("useless", passed=False, ic_train=0.001),
        _rec("strong_neg", passed=False, ic_train=-0.08),
    ])
    assert idx.known_invalid(k=1) == ["useless"], "最没用的（|IC| 最小）优先"


# ── passed_guardrails 是事实，不是决策 ──────────────────────────────────────


def test_decorrelated_factor_is_passed_but_not_reusable(tmp_path):
    """去相关剔除的因子**确实过了定量护栏**，只是与已有候选高度相关、不入候选池。

    把它标成 `passed=False` 会让它落进 `known_invalid`，被当作「已验证无效」喂给 LLM
    ——语义污染，比 Critic drop 更隐蔽（drop 至少是显式判定）。
    """
    idx = _idx(tmp_path)
    idx.append([_rec("decorr", passed=True, verdict="keep",
                     holdout_ic=0.06, decorrelated=True)])

    assert "decorr" not in idx.known_valid(k=5), "与已有候选重复，不该被借鉴"
    assert "decorr" not in idx.known_invalid(k=5), "它过了护栏，不是无效因子"


def test_revise_hypothesis_candidate_is_passed_but_not_reusable(tmp_path):
    """Critic 说「方向要换」，却同时以 passed=True 进 known_valid 当「已验证有效可借鉴」
    ——与同时喂进 feedback 的「换方向」自相矛盾。"""
    idx = _idx(tmp_path)
    idx.append([_rec("wrong_dir", passed=True, verdict="revise_hypothesis", holdout_ic=0.07)])

    assert "wrong_dir" not in idx.known_valid(k=5)
    assert "wrong_dir" not in idx.known_invalid(k=5), "它过了护栏，不是无效因子"


def test_dropped_candidate_is_passed_but_not_reusable(tmp_path):
    """commit 1e0bda4 靠 mutate passed_guardrails=False 来实现；现在改由 verdict 判定，
    `passed` 保持它的事实值。否决回路的语义不变（drop 的因子不会被借鉴）。"""
    idx = _idx(tmp_path)
    idx.append([_rec("dropped", passed=True, verdict="drop", holdout_ic=0.08)])

    assert "dropped" not in idx.known_valid(k=5), "被 Critic drop 的因子不得进 known_valid"
    assert "dropped" not in idx.known_invalid(k=5), "它过了护栏，不是无效因子"


def test_revise_expr_candidate_stays_reusable(tmp_path):
    """`revise_expr` = 方向对、表达式需改 → 思路仍值得借鉴，保留在 known_valid。"""
    idx = _idx(tmp_path)
    idx.append([_rec("right_dir", passed=True, verdict="revise_expr", holdout_ic=0.07)])
    assert "right_dir" in idx.known_valid(k=5)


def test_lookahead_factor_never_fed_back_to_llm(tmp_path):
    """P0：前视因子（负窗口）即便历史误记 passed，也绝不进 known_valid/known_invalid 喂回 LLM
    ——否则引导 LLM 继续生成前视。parse 层根治新生成，这里堵历史产物回灌口子。干净同伴不受影响。"""
    idx = _idx(tmp_path)
    idx.append([
        _rec("ts_sum(delay(ret_1d, -1), 60)", passed=True, holdout_ic=0.09),   # 前视，原库 #1
        _rec("neg(ret_1d)", passed=True, holdout_ic=0.04),                      # 干净，应保留
        _rec("delta(close, -5)", passed=False, ic_train=0.02),                  # 前视且未过护栏
    ])
    valid = idx.known_valid(k=5)
    invalid = idx.known_invalid(k=5)
    assert not any(is_lookahead_expr(e) for e in valid), f"known_valid 混入前视: {valid}"
    assert not any(is_lookahead_expr(e) for e in invalid), f"known_invalid 混入前视: {invalid}"
    assert "neg(ret_1d)" in valid, "干净因子仍应可借鉴"


def test_legacy_records_without_new_fields_still_readable(tmp_path):
    """老 index 没有 decorrelated 字段；缺失应视为 False，不得让整条记录消失。"""
    idx = _idx(tmp_path)
    idx.append([{"expression": "old", "passed": True, "verdict": "keep",
                 "holdout_ic": 0.05, "ic_train": 0.03, "run_id": "t"}])
    assert "old" in idx.known_valid(k=5)


# ── 契约不变量：事实字段不可被决策 mutate ────────────────────────────────────


def test_node_guardrails_marks_passed_before_decorrelation_cut(tmp_path):
    """`guardrail_passed()` 为真即 `passed_guardrails=True`（事实），随后才做去相关剔除。

    修复前 `corr>0.7 → continue` 发生在 `a.passed_guardrails = True` **之前**，事实丢失。
    """
    import datetime as dt

    import numpy as np
    import polars as pl

    from factorzen.agents.nodes import node_guardrails
    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.multiple_testing import TrialLedger

    rng = np.random.default_rng(7)
    days, d = [], dt.date(2021, 1, 4)
    while len(days) < 300:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{600000 + i:06d}.SH" for i in range(40)]:
        base = rng.uniform(8, 15)
        for i, dd in enumerate(days):
            px = base * (1 + 0.001 * i) + rng.normal(0, 0.1)
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px,
                         "high": px * 1.01, "low": px * 0.99,
                         "vol": 1e6 + rng.normal(0, 1e4), "amount": 1e7})
    daily = pl.DataFrame(rows)
    bundle = DataBundle.build(daily)

    import pytest
    _ = pytest  # 保持 import 一致性

    state = AgentState(seed=1)
    for i, ir in enumerate([0.45, 0.44]):     # 两个强因子，第二个必与第一个高度相关
        state.attempts.append(AttemptRecord(
            iteration=0, hypothesis="h", expression=f"rank(neg(ts_min(low, {5 + i})))",
            compile_ok=True, ic_train=ir / 10.0, passed_guardrails=False,
            critic_verdict=None, error=None, ir_train=ir, turnover=0.3, n_train=305))

    import factorzen.validation.holdout as hmod
    from factorzen.validation.holdout import HoldoutICResult
    orig = hmod.holdout_ic_result
    hmod.holdout_ic_result = lambda fdf, hdf: HoldoutICResult(
        0.05, 0.5, (0.01, 0.09), n_days=100)
    try:
        node_guardrails(state, daily=daily, holdout_df=daily, bundle=bundle,
                        ledger=TrialLedger(), top_k=5)
    finally:
        hmod.holdout_ic_result = orig

    passed_facts = [a.passed_guardrails for a in state.attempts]
    assert all(passed_facts), (
        "两个因子都过了定量护栏 → passed_guardrails 都该是 True（事实）；"
        f"被去相关剔除的那个不该丢失这个事实，实得 {passed_facts}"
    )
    decorr = [a for a in state.attempts if a.decorrelated]
    assert len(decorr) == 1, "第二个因子与第一个高度相关，应被标 decorrelated"
    assert len(state.candidates) == 1, "去相关剔除者不入候选池"


def test_team_drop_does_not_mutate_passed_guardrails_fact(tmp_path):
    """Critic drop 是决策，不得改写「过了定量护栏」这个事实。

    修复前 team_orchestrator 在 drop 分支把 passed_guardrails 重置为 False——
    那是用事实字段编码复用决策，正是本文件要拆开的混淆。
    """
    from factorzen.agents.state import AttemptRecord

    a = AttemptRecord(iteration=0, hypothesis="h", expression="x", compile_ok=True,
                      ic_train=0.05, passed_guardrails=True, critic_verdict="drop",
                      error=None, ir_train=0.3, n_train=300)
    idx = _idx(tmp_path)
    record(idx, [a], run_id="t", candidates=[])

    stored = idx.load()[0]
    assert stored["passed"] is True, "事实不得被 drop 决策改写"
    assert stored["verdict"] == "drop"
    assert "x" not in idx.known_valid(k=5), "但它不该被借鉴"

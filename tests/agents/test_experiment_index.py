"""合并自 agents 相关碎片测试（test_experiment_index.py）。

test_experiment_index_contract.py：长期记忆的契约：passed_guardrails 是事实，「可否借鉴」是决策
test_experiment_index_concurrency.py：experiment_index 并发写：多进程并发 append 不得交错/损坏/丢行
test_index_last_wins.py：长期记忆是事件日志：同一表达式的当前状态 = 它最新的那条记录
test_index_window_scoping.py：长期记忆按数据窗口分族 + 为跨 session N 累积预留字段
test_team_experiment_index.py：experiment_index append/load 与 known_invalid/valid 去重
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from pathlib import Path

from factorzen.agents.experiment_index import ExperimentIndex
from factorzen.agents.roles.librarian import recall, record
from factorzen.agents.state import AttemptRecord
from factorzen.discovery.expression import is_lookahead_expr


# ==== 来自 test_experiment_index_contract.py ====
def _idx(tmp_path) -> ExperimentIndex:
    return ExperimentIndex(str(tmp_path / "idx.jsonl"))


def _rec__index_contract(expr: str, *, passed: bool, verdict: str | None = "keep",
         holdout_ic: float | None = None, decorrelated: bool = False,
         ic_train: float = 0.03) -> dict:
    r = {"expression": expr, "hypothesis": "h", "ic_train": ic_train,
         "passed": passed, "verdict": verdict, "decorrelated": decorrelated,
         "run_id": "t"}
    if holdout_ic is not None:
        r["holdout_ic"] = holdout_ic
    return r


# ── F4：known_valid 必须按 |holdout_ic| 排序 ────────────────────────────────


def test_known_valid_rank_suite(tmp_path):
    """护栏明确接纳负 IC 反转因子（`guardrail_passed` 的 same_sign + ci_high<0 分支）。；回归：known_invalid 一直用 abs()，本次不动。"""
    # -- 原 test_known_valid_ranks_by_abs_holdout_ic_so_reversal_factors_survive --
    def _section_0_test_known_valid_ranks_by_abs_holdout_ic_so_reversal_factors_survive(tmp_path):
        idx = _idx(tmp_path)
        idx.append([
            _rec__index_contract("reversal", passed=True, holdout_ic=-0.09),      # |IC| 最大，方向为负
            *[_rec__index_contract(f"weak_long_{i}", passed=True, holdout_ic=0.01 + 0.01 * i) for i in range(5)],
        ])

        valid = idx.known_valid(k=5)

        assert "reversal" in valid, f"最强的反转因子被挤出 known_valid: {valid}"
        assert valid[0] == "reversal", "按 |holdout_ic| 排序时反转因子应排第一"

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_known_valid_ranks_by_abs_holdout_ic_so_reversal_factors_survive(_tp0)

    # -- 原 test_known_invalid_still_ranks_by_abs_ic_train --
    def _section_1_test_known_invalid_still_ranks_by_abs_ic_train(tmp_path):
        idx = _idx(tmp_path)
        idx.append([
            _rec__index_contract("useless", passed=False, ic_train=0.001),
            _rec__index_contract("strong_neg", passed=False, ic_train=-0.08),
        ])
        assert idx.known_invalid(k=1) == ["useless"], "最没用的（|IC| 最小）优先"

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_known_invalid_still_ranks_by_abs_ic_train(_tp1)


# ── passed_guardrails 是事实，不是决策 ──────────────────────────────────────


def test_passed_fact_vs_reusable_suite(tmp_path):
    """去相关剔除的因子**确实过了定量护栏**，只是与已有候选高度相关、不入候选池。；Critic 说「方向要换」，却同时以 passed=True 进 known_valid 当「已验证有效可借鉴」；commit 1e0bda4 靠 mutate passed_guardrails=False 来实现；现在改由 verdict 判定，；`revise_expr` = 方向对、表达式需改 → 思路仍值得借鉴，保留在 known_valid。；P0：前视因子（负窗口）即便历史误记 passed，也绝不进 known_valid/known_invalid 喂回 LLM；老 index 没有 decorrelated 字段；缺失应视为 False，不得让整条记录消失。"""
    # -- 原 test_decorrelated_factor_is_passed_but_not_reusable --
    def _section_0_test_decorrelated_factor_is_passed_but_not_reusable(tmp_path):
        idx = _idx(tmp_path)
        idx.append([_rec__index_contract("decorr", passed=True, verdict="keep",
                         holdout_ic=0.06, decorrelated=True)])

        assert "decorr" not in idx.known_valid(k=5), "与已有候选重复，不该被借鉴"
        assert "decorr" not in idx.known_invalid(k=5), "它过了护栏，不是无效因子"

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_decorrelated_factor_is_passed_but_not_reusable(_tp0)

    # -- 原 test_revise_hypothesis_candidate_is_passed_but_not_reusable --
    def _section_1_test_revise_hypothesis_candidate_is_passed_but_not_reusable(tmp_path):
        idx = _idx(tmp_path)
        idx.append([_rec__index_contract("wrong_dir", passed=True, verdict="revise_hypothesis", holdout_ic=0.07)])

        assert "wrong_dir" not in idx.known_valid(k=5)
        assert "wrong_dir" not in idx.known_invalid(k=5), "它过了护栏，不是无效因子"

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_revise_hypothesis_candidate_is_passed_but_not_reusable(_tp1)

    # -- 原 test_dropped_candidate_is_passed_but_not_reusable --
    def _section_2_test_dropped_candidate_is_passed_but_not_reusable(tmp_path):
        idx = _idx(tmp_path)
        idx.append([_rec__index_contract("dropped", passed=True, verdict="drop", holdout_ic=0.08)])

        assert "dropped" not in idx.known_valid(k=5), "被 Critic drop 的因子不得进 known_valid"
        assert "dropped" not in idx.known_invalid(k=5), "它过了护栏，不是无效因子"

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_dropped_candidate_is_passed_but_not_reusable(_tp2)

    # -- 原 test_revise_expr_candidate_stays_reusable --
    def _section_3_test_revise_expr_candidate_stays_reusable(tmp_path):
        idx = _idx(tmp_path)
        idx.append([_rec__index_contract("right_dir", passed=True, verdict="revise_expr", holdout_ic=0.07)])
        assert "right_dir" in idx.known_valid(k=5)

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_revise_expr_candidate_stays_reusable(_tp3)

    # -- 原 test_lookahead_factor_never_fed_back_to_llm --
    def _section_4_test_lookahead_factor_never_fed_back_to_llm(tmp_path):
        idx = _idx(tmp_path)
        idx.append([
            _rec__index_contract("ts_sum(delay(ret_1d, -1), 60)", passed=True, holdout_ic=0.09),   # 前视，原库 #1
            _rec__index_contract("neg(ret_1d)", passed=True, holdout_ic=0.04),                      # 干净，应保留
            _rec__index_contract("delta(close, -5)", passed=False, ic_train=0.02),                  # 前视且未过护栏
        ])
        valid = idx.known_valid(k=5)
        invalid = idx.known_invalid(k=5)
        assert not any(is_lookahead_expr(e) for e in valid), f"known_valid 混入前视: {valid}"
        assert not any(is_lookahead_expr(e) for e in invalid), f"known_invalid 混入前视: {invalid}"
        assert "neg(ret_1d)" in valid, "干净因子仍应可借鉴"

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_lookahead_factor_never_fed_back_to_llm(_tp4)

    # -- 原 test_legacy_records_without_new_fields_still_readable --
    def _section_5_test_legacy_records_without_new_fields_still_readable(tmp_path):
        idx = _idx(tmp_path)
        idx.append([{"expression": "old", "passed": True, "verdict": "keep",
                     "holdout_ic": 0.05, "ic_train": 0.03, "run_id": "t"}])
        assert "old" in idx.known_valid(k=5)

    _tp5 = tmp_path / "_s5"
    _tp5.mkdir(exist_ok=True)
    _section_5_test_legacy_records_without_new_fields_still_readable(_tp5)


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
    hmod.holdout_ic_result = lambda *a, **k: HoldoutICResult(
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


# ── 无 IC 的行不配当负例（预热不足 / duplicate_fingerprint 挤占 top-k）────────


def test_known_invalid_excludes_rows_without_ic(tmp_path):
    """ic_train=None 的行（预热不足 / duplicate_fingerprint 等评估未出值）零方向信息，
    排序键 abs(None or 0)=0 会挤占 known_invalid top-k——与编译失败被排除同理。
    seen_expressions 的跨 session 去重价值不受影响。"""
    idx = _idx(tmp_path)
    r_warm = _rec__index_contract("warmup_short", passed=False)
    r_warm["ic_train"] = None
    r_warm["error"] = "预热不足: 叶 roe 需要 504 根历史，可用 400 根"
    r_dup = _rec__index_contract("dup_fp", passed=False)
    r_dup["ic_train"] = None
    r_dup["error"] = "duplicate_fingerprint"
    idx.append([r_warm, r_dup, _rec__index_contract("weak_real", passed=False, ic_train=0.002)])

    inv = idx.known_invalid(k=3)
    assert "warmup_short" not in inv, "无 IC 的预热不足行不该进负例"
    assert "dup_fp" not in inv, "指纹重复行不该进负例"
    assert inv == ["weak_real"], f"仅真实弱 IC 行可当负例: {inv}"
    # 去重集合仍见它们
    seen = idx.seen_expressions()
    assert "warmup_short" in seen and "dup_fp" in seen

# ==== 来自 test_experiment_index_concurrency.py ====
_PER_PROC = 40


def _worker(args: tuple[str, int]) -> None:
    path, k = args
    from factorzen.agents.experiment_index import ExperimentIndex
    idx = ExperimentIndex(path)
    idx.append([{"expression": f"ts_mean(close, {k})_{i}", "passed": bool(i % 2)}
                for i in range(_PER_PROC)])


def test_concurrent_process_appends_no_corruption(tmp_path):
    path = str(tmp_path / "experiment_index.jsonl")
    n_proc = 8
    ctx = mp.get_context("fork")
    with ctx.Pool(n_proc) as pool:
        pool.map(_worker, [(path, k) for k in range(n_proc)])

    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    assert len(lines) == n_proc * _PER_PROC, "并发写丢行/多行"
    for line in lines:            # 每行都是完整合法 JSON（无交错截断）
        rec = json.loads(line)
        assert "expression" in rec

    from factorzen.agents.experiment_index import ExperimentIndex
    assert len(ExperimentIndex(path).load()) == n_proc * _PER_PROC


# ==== 来自 test_index_last_wins.py ====
def _rec__last_wins(expr: str, *, passed: bool, ic: float, holdout: float | None = None,
         verdict: str | None = None, decorrelated: bool = False) -> dict:
    return {"expression": expr, "passed": passed, "ic_train": ic, "compile_ok": True,
            "holdout_ic": holdout, "verdict": verdict, "decorrelated": decorrelated,
            "data_window": {"start": "20220101", "end": "20231229",
                            "universe": "csi800", "market": "ashare"}}


_WINDOW = {"start": "20220101", "end": "20231229", "universe": "csi800", "market": "ashare"}


def _write(path, records):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def test_last_wins_override_suite(tmp_path):
    """收尾降级：先写 passed=True，后写 passed=False → 不得再出现在 known_valid。；反向断言：后写 passed=True 也必须覆盖先写的 False，否则「后写覆盖」是假的。；覆盖判定按**归一化**表达式，而非裸字符串——否则空格差异就能绕过。；判别力：别把「后写覆盖」实现成「只留最后一条记录」。；不同数据窗口的同名表达式互不覆盖——族边界优先于时间顺序。；去重不该让「见过的表达式」漏掉任何一个。"""
    # -- 原 test_later_record_overrides_earlier_for_same_expression --
    def _section_0_test_later_record_overrides_earlier_for_same_expression(tmp_path):
        p = tmp_path / "idx.jsonl"
        _write(p, [
            _rec__last_wins("rank(neg(pb))", passed=True, ic=0.02, holdout=0.05),   # 早轮结论
            _rec__last_wins("rank(neg(pb))", passed=False, ic=0.02, holdout=0.05),  # 收尾复核后的更正
        ])
        index = ExperimentIndex(str(p))

        assert index.known_valid(data_window=_WINDOW) == [], "被降级的因子不该还算「已验证有效」"
        assert "rank(neg(pb))" in index.known_invalid(data_window=_WINDOW)

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_later_record_overrides_earlier_for_same_expression(_tp0)

    # -- 原 test_override_works_in_the_other_direction_too --
    def _section_1_test_override_works_in_the_other_direction_too(tmp_path):
        p = tmp_path / "idx.jsonl"
        _write(p, [
            _rec__last_wins("rank(neg(pb))", passed=False, ic=0.02, holdout=0.05),
            _rec__last_wins("rank(neg(pb))", passed=True, ic=0.02, holdout=0.05),
        ])
        index = ExperimentIndex(str(p))

        assert index.known_valid(data_window=_WINDOW) == ["rank(neg(pb))"]
        assert index.known_invalid(data_window=_WINDOW) == []

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_override_works_in_the_other_direction_too(_tp1)

    # -- 原 test_override_matches_on_normalized_expression --
    def _section_2_test_override_matches_on_normalized_expression(tmp_path):
        p = tmp_path / "idx.jsonl"
        _write(p, [
            _rec__last_wins("rank(neg(pb))", passed=True, ic=0.02, holdout=0.05),
            _rec__last_wins("rank( neg( pb ) )", passed=False, ic=0.02, holdout=0.05),
        ])
        index = ExperimentIndex(str(p))

        assert index.known_valid(data_window=_WINDOW) == []

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_override_matches_on_normalized_expression(_tp2)

    # -- 原 test_distinct_expressions_are_not_collapsed --
    def _section_3_test_distinct_expressions_are_not_collapsed(tmp_path):
        p = tmp_path / "idx.jsonl"
        _write(p, [
            _rec__last_wins("rank(neg(pb))", passed=True, ic=0.02, holdout=0.09),
            _rec__last_wins("rank(neg(pe_ttm))", passed=True, ic=0.03, holdout=0.05),
        ])
        index = ExperimentIndex(str(p))

        assert sorted(index.known_valid(data_window=_WINDOW)) == [
            "rank(neg(pb))", "rank(neg(pe_ttm))"
        ]

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_distinct_expressions_are_not_collapsed(_tp3)

    # -- 原 test_override_is_scoped_per_data_window --
    def _section_4_test_override_is_scoped_per_data_window(tmp_path):
        p = tmp_path / "idx.jsonl"
        other = dict(_WINDOW, end="20241231")
        r_old = _rec__last_wins("rank(neg(pb))", passed=True, ic=0.02, holdout=0.05)
        r_new = _rec__last_wins("rank(neg(pb))", passed=False, ic=0.02, holdout=0.05)
        r_new["data_window"] = other
        _write(p, [r_old, r_new])
        index = ExperimentIndex(str(p))

        assert index.known_valid(data_window=_WINDOW) == ["rank(neg(pb))"], \
            "另一个窗口的 False 不该影响本窗口"
        assert index.known_valid(data_window=other) == []

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_override_is_scoped_per_data_window(_tp4)

    # -- 原 test_seen_expressions_unaffected_by_dedup --
    def _section_5_test_seen_expressions_unaffected_by_dedup(tmp_path):
        p = tmp_path / "idx.jsonl"
        _write(p, [
            _rec__last_wins("rank(neg(pb))", passed=True, ic=0.02),
            _rec__last_wins("rank(neg(pb))", passed=False, ic=0.02),
            _rec__last_wins("rank(neg(pe_ttm))", passed=False, ic=0.01),
        ])
        index = ExperimentIndex(str(p))

        assert index.seen_expressions(data_window=_WINDOW) == {
            "rank(neg(pb))", "rank(neg(pe_ttm))"
        }

    _tp5 = tmp_path / "_s5"
    _tp5.mkdir(exist_ok=True)
    _section_5_test_seen_expressions_unaffected_by_dedup(_tp5)


# ── 接线守卫：收尾降级后，team 必须把更正写回 index ──────────────────────
#
# 变异实证：把 team_orchestrator 里补写更正的分支关掉，上面 6 个测试全绿——
# 它们只测 ExperimentIndex 的读语义，没人验证「降级后真的补写了」。


def test_team_writes_correction_to_index_after_demotion(tmp_path, monkeypatch):
    """收尾把候选降级 → index 里那条 passed=True 必须被后写的 passed=False 覆盖。

    否则被最终 N 否掉的因子仍以「已验证有效」喂给后续 session——长期记忆被污染。
    """
    import datetime as dt
    import json

    import numpy as np
    import polars as pl

    import factorzen.agents.team_orchestrator as team
    from factorzen.agents.state import AttemptRecord
    from factorzen.discovery.guardrails import DeflationBasis

    def _daily(n_stocks=40, n_days=180, seed=1):
        rng = np.random.default_rng(seed)
        days, d = [], dt.date(2022, 1, 3)
        while len(days) < n_days:
            if d.weekday() < 5:
                days.append(d)
            d += dt.timedelta(days=1)
        rows = []
        for c in [f"{i:06d}.SZ" for i in range(n_stocks)]:
            px = 10.0
            for dd in days:
                px *= 1 + rng.standard_normal() * 0.02
                rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                             "high": px * 1.01, "low": px * 0.98,
                             "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                             "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
        return pl.DataFrame(rows)

    def fake_guardrails(state, *, daily, holdout_df, bundle, ledger, top_k=5,
                        dsr_alpha=0.05, warmup_daily=None, eval_start=None, **_kwargs):
        """每轮产一个「当轮 N 下过关」的候选（passed=True 会被 Librarian 落盘）。"""
        ledger.record(1)
        state.attempts.append(AttemptRecord(
            iteration=state.iteration, hypothesis="h", expression="rank(neg(pb))",
            compile_ok=True, ic_train=0.05, passed_guardrails=True, critic_verdict=None,
            error=None, ir_train=0.4, turnover=0.3, n_train=300))
        if not state.candidates:
            state.candidates.append({
                "expression": "rank(neg(pb))", "hypothesis": "h", "ic_train": 0.05,
                "ir_train": 0.4, "turnover": 0.3, "holdout_ic": 0.04, "holdout_ir": 0.3,
                "dsr": 0.9, "dsr_pvalue": 0.01, "n_train": 300,
                "ic_ci_low": 0.01, "ic_ci_high": 0.07})
        return state

    def fake_finalize(state, *, dsr_alpha=0.05, daily=None, bundle=None, **_kwargs):
        """模拟「最终 N 下不再显著」：清空候选并回落事实。"""
        for a in state.attempts:
            a.passed_guardrails = False
        state.candidates = []
        return DeflationBasis(n_trials=3, sharpe_variance=0.01, two_sided=True)

    monkeypatch.setattr(team, "node_guardrails", fake_guardrails)
    monkeypatch.setattr(team, "node_finalize_guardrails", fake_finalize)

    seq = [json.dumps({"hypotheses": ["h"]}),
           json.dumps({"expressions": ["rank(neg(pb))"]}),
           json.dumps({"verdict": "keep", "reason": "ok"})]
    i = {"k": 0}

    def fn(_m):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v

    idx_path = tmp_path / "e.jsonl"
    team.run_team_agent(_daily(), fn, n_rounds=2, seed=1, index_path=str(idx_path),
                        heal_rounds=0)

    index = ExperimentIndex(str(idx_path))
    assert index.known_valid() == [], (
        "收尾已把候选降级，index 却仍把它当「已验证有效」—— 更正记录没补写，"
        "或后写覆盖没生效"
    )
    assert "rank(neg(pb))" in index.known_invalid()

# ==== 来自 test_index_window_scoping.py ====
_W1 = {"start": "20220101", "end": "20231229", "universe": "csi800", "market": "ashare"}
_W2 = {"start": "20150101", "end": "20211231", "universe": "csi300", "market": "ashare"}


def _attempt(expr: str, *, ir: float = 0.3, passed: bool = True,
             verdict: str | None = "keep") -> AttemptRecord:
    return AttemptRecord(iteration=0, hypothesis="h", expression=expr, compile_ok=True,
                         ic_train=0.05, passed_guardrails=passed, critic_verdict=verdict,
                         error=None, ir_train=ir, turnover=0.3, n_train=300)


# ── 前提字段：没有它们，将来永远无法重建历史 IR 池 ────────────────────────────


def test_window_scoping_suite(tmp_path, caplog):
    """DSR 的 deflation 池要的是 **IR**，不是 IC。`record()` 此前只落 `ic_train`。；test_record_persists_data_window；一个窗口上「已验证有效」的因子，换个窗口未必成立——不得跨窗口喂给 LLM。；test_seen_and_known_lists_filter_by_window；不传 data_window → 不过滤（向后兼容既有调用方与老 index）。；老记录不知道来自哪个窗口 → 过滤时保守排除，并告警一次（不静默丢数据）。；不过滤时老记录照常可见——排除只发生在显式按窗口查询时。"""
    # -- 原 test_record_persists_ir_train_and_n_train --
    def _section_0_test_record_persists_ir_train_and_n_train(tmp_path):
        idx = ExperimentIndex(str(tmp_path / "i.jsonl"))
        record(idx, [_attempt("rank(close)", ir=0.42)], run_id="r1", data_window=_W1)

        r = idx.load()[0]
        assert r["ir_train"] == 0.42
        assert r["n_train"] == 300

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_record_persists_ir_train_and_n_train(_tp0)

    # -- 原 test_record_persists_data_window --
    def _section_1_test_record_persists_data_window(tmp_path):
        idx = ExperimentIndex(str(tmp_path / "i.jsonl"))
        record(idx, [_attempt("rank(close)")], run_id="r1", data_window=_W1)

        assert idx.load()[0]["data_window"] == _W1

    caplog.clear()
    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_record_persists_data_window(_tp1)

    # -- 原 test_recall_is_scoped_to_the_data_window --
    def _section_2_test_recall_is_scoped_to_the_data_window(tmp_path):
        idx = ExperimentIndex(str(tmp_path / "i.jsonl"))
        record(idx, [_attempt("in_window", ir=0.4)], run_id="r1", data_window=_W1)
        record(idx, [_attempt("other_window", ir=0.4)], run_id="r2", data_window=_W2)

        rec = recall(idx, k=5, data_window=_W1)

        assert "in_window" in rec.seen
        assert "other_window" not in rec.seen, "跨窗口的历史不该进本窗口的去重集"
        assert "other_window" not in rec.known_valid

    caplog.clear()
    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_recall_is_scoped_to_the_data_window(_tp2)

    # -- 原 test_seen_and_known_lists_filter_by_window --
    def _section_3_test_seen_and_known_lists_filter_by_window(tmp_path):
        idx = ExperimentIndex(str(tmp_path / "i.jsonl"))
        record(idx, [_attempt("w1_valid", ir=0.4)], run_id="r1", data_window=_W1)
        record(idx, [_attempt("w2_invalid", ir=0.01, passed=False)], run_id="r2", data_window=_W2)

        assert idx.known_valid(k=5, data_window=_W1) == ["w1_valid"]
        assert idx.known_valid(k=5, data_window=_W2) == []
        assert idx.known_invalid(k=5, data_window=_W1) == []
        assert idx.known_invalid(k=5, data_window=_W2) == ["w2_invalid"]

    caplog.clear()
    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_seen_and_known_lists_filter_by_window(_tp3)

    # -- 原 test_no_window_means_no_filtering --
    def _section_4_test_no_window_means_no_filtering(tmp_path):
        idx = ExperimentIndex(str(tmp_path / "i.jsonl"))
        record(idx, [_attempt("a", ir=0.4)], run_id="r1", data_window=_W1)
        record(idx, [_attempt("b", ir=0.4)], run_id="r2", data_window=_W2)

        assert idx.seen_expressions() == {"a", "b"}
        assert set(idx.known_valid(k=5)) == {"a", "b"}

    caplog.clear()
    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_no_window_means_no_filtering(_tp4)

    # -- 原 test_legacy_records_without_window_are_excluded_when_filtering --
    def _section_5_test_legacy_records_without_window_are_excluded_when_filtering(tmp_path, caplog):
        idx = ExperimentIndex(str(tmp_path / "i.jsonl"))
        idx.append([{"expression": "legacy", "passed": True, "verdict": "keep",
                     "ic_train": 0.05, "holdout_ic": 0.04, "run_id": "old"}])
        record(idx, [_attempt("fresh", ir=0.4)], run_id="r1", data_window=_W1)

        with caplog.at_level(logging.WARNING, logger="factorzen.agents.experiment_index"):
            valid = idx.known_valid(k=5, data_window=_W1)

        assert valid == ["fresh"], "无窗口标记的老记录不得混进本窗口的召回"
        assert any("data_window" in r.getMessage() for r in caplog.records), \
            "排除老记录必须告警，不能静默"

    caplog.clear()
    _tp5 = tmp_path / "_s5"
    _tp5.mkdir(exist_ok=True)
    _section_5_test_legacy_records_without_window_are_excluded_when_filtering(_tp5, caplog)

    # -- 原 test_legacy_records_visible_when_not_filtering --
    def _section_6_test_legacy_records_visible_when_not_filtering(tmp_path):
        idx = ExperimentIndex(str(tmp_path / "i.jsonl"))
        idx.append([{"expression": "legacy", "passed": True, "verdict": "keep",
                     "ic_train": 0.05, "holdout_ic": 0.04, "run_id": "old"}])

        assert idx.known_valid(k=5) == ["legacy"]

    caplog.clear()
    _tp6 = tmp_path / "_s6"
    _tp6.mkdir(exist_ok=True)
    _section_6_test_legacy_records_visible_when_not_filtering(_tp6)


# ── 按窗口分族 ──────────────────────────────────────────────────────────────


# ── 端到端接线：能力实现了不算，team 路径得真的传下去 ────────────────────────


def test_team_agent_scopes_index_to_its_data_window(tmp_path):
    """`run_team_agent` 必须把 data_window 透传给 Librarian，否则分族形同虚设。"""
    import datetime as dt
    import json

    import numpy as np
    import polars as pl

    from factorzen.agents.team_orchestrator import run_team_agent

    rng = np.random.default_rng(1)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < 180:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{i:06d}.SZ" for i in range(40)]:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    daily = pl.DataFrame(rows)

    def llm(messages):
        system = messages[0]["content"]
        if "verdict" in system:
            return json.dumps({"verdict": "keep", "reason": "ok"})
        if '"expressions"' in system:
            return json.dumps({"expressions": ["ts_mean(close,5)"]})
        return json.dumps({"hypotheses": ["动量"]})

    idx_path = str(tmp_path / "i.jsonl")
    run_team_agent(daily, llm, n_rounds=1, seed=1, index_path=idx_path, data_window=_W1)

    recs = ExperimentIndex(idx_path).load()
    assert recs, "本轮应有 attempt 落盘"
    assert all(r.get("data_window") == _W1 for r in recs), \
        "落盘记录必须带上本次运行的数据窗口"
    assert all("ir_train" in r for r in recs)

# ==== 来自 test_team_experiment_index.py ====
def _recs():
    return [
        {"expression": "ts_mean(close,5)", "hypothesis": "动量", "ic_train": 0.05,
         "holdout_ic": 0.03, "dsr": 0.7, "passed": True, "verdict": "keep"},
        {"expression": "rank(vol)", "hypothesis": "换手", "ic_train": 0.001,
         "holdout_ic": 0.0, "dsr": 0.1, "passed": False, "verdict": "drop"},
    ]


def test_index_smoke_load_suite(tmp_path):
    """test_seen_expressions_normalized；test_load_missing_file_empty；test_append_empty_is_noop"""
    # -- 原 test_seen_expressions_normalized --
    def _section_0_test_seen_expressions_normalized(tmp_path):
        idx = ExperimentIndex(str(tmp_path / "exp.jsonl"))
        idx.append(_recs())
        seen = idx.seen_expressions()
        # 归一化形式（带空格）应能匹配无空格原始查询
        assert "ts_mean(close, 5)" in seen           # 归一化后带空格
        assert "rank(vol)" in seen

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_seen_expressions_normalized(_tp0)

    # -- 原 test_load_missing_file_empty --
    def _section_1_test_load_missing_file_empty(tmp_path):
        idx = ExperimentIndex(str(tmp_path / "nope.jsonl"))
        assert idx.load() == [] and idx.seen_expressions() == set()

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_load_missing_file_empty(_tp1)

    # -- 原 test_append_empty_is_noop --
    def _section_2_test_append_empty_is_noop(tmp_path):
        from factorzen.agents.experiment_index import ExperimentIndex
        path = tmp_path / "idx.jsonl"
        idx = ExperimentIndex(str(path))
        idx.append([])
        assert not path.exists() or path.read_text() == ""

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_append_empty_is_noop(_tp2)



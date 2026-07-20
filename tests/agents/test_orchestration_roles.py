"""
test_orchestration.py：合并自 agents 相关碎片测试（test_orchestration.py）。
test_team_roles.py：合并自 agents 相关碎片测试（test_team_roles.py）。
"""

from __future__ import annotations

import datetime as dt
import json
from unittest.mock import patch

import numpy as np
import polars as pl

from factorzen.agents.experiment_index import ExperimentIndex
from factorzen.agents.orchestrator import run_llm_agent
from factorzen.agents.roles.coder import revise_expressions, write_expressions
from factorzen.agents.roles.critic import CriticVerdict, critique
from factorzen.agents.roles.hypothesis import propose_hypotheses
from factorzen.agents.roles.librarian import recall, record
from factorzen.agents.state import AttemptRecord
from factorzen.agents.team_orchestrator import run_team_agent


# ==== 来自 test_orchestration.py ====
# ==== 来自 test_team_orchestrator.py ====
def _mock_daily__team_orch(n_stocks=40, n_days=180, seed=1):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


def _scripted_team():
    """Hypothesis→Coder→Critic(keep) 一轮脚本，循环复用。"""
    hyp = json.dumps({"hypotheses": ["动量"]})
    code = json.dumps({"expressions": ["ts_mean(close,5)"]})
    crit = json.dumps({"verdict": "keep", "reason": "ok"})
    seq = [hyp, code, crit] * 50
    i = {"k": 0}

    def fn(messages):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v

    return fn


def _inject_cands_for_critic(state, *, ledger, **_kw):
    """把本轮 attempts 注入 candidates，使 Critic 走「有新候选」路径（非 W5c 空轮跳过）。

    真实护栏在 mock 日线上常因 IC 弱/holdout 覆盖拒光 → new_cands=[] → W5c 确定性
    revise_hypothesis、不调 LLM。本测关注的是 revise_expr 跨轮反馈，需强制有候选。
    """
    n = 0
    for a in state.attempts:
        if a.iteration != state.iteration or a.ic_train is None:
            continue
        a.passed_guardrails = True
        state.candidates.append({
            "expression": a.expression, "hypothesis": a.hypothesis,
            "ic_train": a.ic_train or 0.05, "ir_train": a.ir_train or 0.4,
            "turnover": a.turnover or 0.1, "holdout_ic": 0.04, "holdout_ir": 0.3,
            "dsr": 0.7, "dsr_pvalue": 0.05,
            "n_train": a.n_train if a.n_train is not None else 100,
            "n_holdout_days": 80, "ic_ci_low": 0.01, "ic_ci_high": 0.08,
        })
        n += 1
    if n:
        ledger.record(n)
    return state


def test_team_orchestrator_loop_suite(tmp_path):
    """轮1 Critic revise_expr → 轮2 Coder 改写（跨轮 feedback），两表达式都评估、都计入 N。；任务 D：hypotheses_per_round=2 → propose 收到 n=2、两个假设的表达式都被评估，；默认 hypotheses_per_round=1 → propose 收到 n=1（零回归）。；共享 experiment_index：第二次 run 重复表达式被跳过（seen 去重）。；scripted Critic drop → 被 drop 候选从 TeamResult.candidates 移除，且否决回路真实生效。；轮1 Critic revise_expr → 轮2 必须**同时**评估修订产物与新假设产物。"""
    # -- 原 test_run_team_revise_loop_counts_n --
    def _section_0_test_run_team_revise_loop_counts_n(tmp_path):
        from unittest.mock import patch

        hyp = json.dumps({"hypotheses": ["动量"]})
        code1 = json.dumps({"expressions": ["ts_mean(close,5)"]})
        crit_revise = json.dumps({"verdict": "revise_expr", "reason": "窗口太短"})
        code2 = json.dumps({"expressions": ["ts_mean(close,20)"]})  # 下一轮 revise 产物
        crit_keep = json.dumps({"verdict": "keep", "reason": "ok"})
        # 轮1: propose,write,critic(revise) ; 轮2: revise(不再 propose),critic(keep)
        seq = [hyp, code1, crit_revise, code2, crit_keep]
        i = {"k": 0}

        def fn(messages):
            v = seq[i["k"]] if i["k"] < len(seq) else crit_keep
            i["k"] += 1
            return v

        daily = _mock_daily__team_orch()
        with patch("factorzen.agents.team_orchestrator.node_guardrails",
                   side_effect=_inject_cands_for_critic):
            res = run_team_agent(daily, fn, n_rounds=2, seed=1, index_path=str(tmp_path / "e.jsonl"))
        assert res.n_trials >= 2     # 两轮各评估一个表达式(原始 + 改写)，都计入 N
        assert any("ts_mean(close, 20)" in r["expressions"] for r in res.rounds_log)  # 轮2 是改写产物

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_run_team_revise_loop_counts_n(_tp0)

    # -- 原 test_hypotheses_per_round_evaluates_all --
    def _section_1_test_hypotheses_per_round_evaluates_all(tmp_path):
        hyp = json.dumps({"hypotheses": ["动量因子", "反转因子"]})
        keep = json.dumps({"verdict": "keep", "reason": "ok"})
        seen_propose: list[str] = []

        def fn(messages):
            text = "\n".join(m["content"] for m in messages)
            if "风控审计员" in text:            # critic
                return keep
            if "翻译成" in text:               # write_expressions（按假设分流）
                if "动量因子" in text:
                    return json.dumps({"expressions": ["ts_mean(close,5)"]})
                if "反转因子" in text:
                    return json.dumps({"expressions": ["ts_std(close,10)"]})
                return json.dumps({"expressions": ["rank(vol)"]})
            seen_propose.append(text)          # propose_hypotheses
            return hyp

        daily = _mock_daily__team_orch()
        res = run_team_agent(daily, fn, n_rounds=1, seed=1, heal_rounds=0,
                             index_path=str(tmp_path / "e.jsonl"), hypotheses_per_round=2)

        assert any("提出 2 个新方向" in t for t in seen_propose), "propose 应收到 n=2"
        exprs = {a.expression for a in res.state.attempts}
        assert "ts_mean(close, 5)" in exprs and "ts_std(close, 10)" in exprs, \
            f"两个假设的表达式都应被评估: {exprs}"
        by_expr = {a.expression: a.hypothesis for a in res.state.attempts}
        assert by_expr["ts_mean(close, 5)"] == "动量因子"
        assert by_expr["ts_std(close, 10)"] == "反转因子"
        # rounds_log 记两个假设（"；" 连接）
        assert "动量因子" in res.rounds_log[0]["hypothesis"]
        assert "反转因子" in res.rounds_log[0]["hypothesis"]

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_hypotheses_per_round_evaluates_all(_tp1)

    # -- 原 test_hypotheses_per_round_default_is_single --
    def _section_2_test_hypotheses_per_round_default_is_single(tmp_path):
        seen: list[str] = []

        def fn(messages):
            text = "\n".join(m["content"] for m in messages)
            seen.append(text)
            if "风控审计员" in text:
                return json.dumps({"verdict": "keep", "reason": "ok"})
            if "翻译成" in text:
                return json.dumps({"expressions": ["ts_mean(close,5)"]})
            return json.dumps({"hypotheses": ["动量"]})

        run_team_agent(_mock_daily__team_orch(), fn, n_rounds=1, seed=1, heal_rounds=0,
                       index_path=str(tmp_path / "e.jsonl"))
        assert any("提出 1 个新方向" in t for t in seen), "默认应 n=1"
        assert not any("提出 2 个新方向" in t for t in seen)

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_hypotheses_per_round_default_is_single(_tp2)

    # -- 原 test_cross_session_dedup --
    def _section_3_test_cross_session_dedup(tmp_path):
        daily = _mock_daily__team_orch()
        idx_path = str(tmp_path / "shared.jsonl")
        run_team_agent(daily, _scripted_team(), n_rounds=1, seed=1, index_path=idx_path)
        res2 = run_team_agent(daily, _scripted_team(), n_rounds=1, seed=1, index_path=idx_path)
        # 第二次 run 产同样的 ts_mean(close,5)，已在 index → 本轮无新评估（n_trials 可能为 0）
        assert res2.n_trials == 0 or all(
            a.expression != "ts_mean(close, 5)" for a in res2.state.attempts)

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_cross_session_dedup(_tp3)

    # -- 原 test_critic_drop_removes_candidate --
    def _section_4_test_critic_drop_removes_candidate(tmp_path):
        from unittest.mock import patch

        from factorzen.agents.experiment_index import ExperimentIndex

        drop_expr = "ts_mean(close, 5)"

        def fake_guardrails(state, *, daily, holdout_df, bundle, ledger, top_k=5, warmup_daily=None,
                            eval_start=None, profile=None, lib_pool=None, **_kwargs):
            """注入候选并计 N，模拟本轮过了护栏（含真实 node_guardrails 会做的状态写入）。"""
            ledger.record(1)  # N 诚实：记 1 个试验
            # 忠实复刻 node_guardrails 第140行的副作用：候选过护栏时标记对应 AttemptRecord
            for a in state.attempts:
                if a.iteration == state.iteration and a.expression == drop_expr:
                    a.passed_guardrails = True
            state.candidates.append({
                "expression": drop_expr,
                "hypothesis": "动量",
                "ic_train": 0.05,
                "holdout_ic": 0.04,
                "holdout_ir": 0.3,
                "dsr": 0.7,
                "dsr_pvalue": 0.05,
            })
            return state

        hyp = json.dumps({"hypotheses": ["动量"]})
        code = json.dumps({"expressions": ["ts_mean(close,5)"]})
        crit_drop = json.dumps({"verdict": "drop", "reason": "过拟合"})
        seq = [hyp, code, crit_drop]
        i = {"k": 0}

        def fn(messages):
            v = seq[i["k"]] if i["k"] < len(seq) else crit_drop
            i["k"] += 1
            return v

        daily = _mock_daily__team_orch()
        idx_path = str(tmp_path / "e.jsonl")
        with patch("factorzen.agents.team_orchestrator.node_guardrails", fake_guardrails):
            res = run_team_agent(daily, fn, n_rounds=1, seed=42, index_path=idx_path)

        # 系统层回归断言：Critic drop → 本轮候选必须从 candidates 移除
        assert all(c["expression"] != drop_expr for c in res.candidates), \
            f"drop 候选未被移除: {res.candidates}"
        # N 诚实：drop 不影响 ledger（fake_guardrails 已调用 ledger.record(1)）
        assert res.n_trials >= 1

        # 内存状态：drop 是**决策**，不得改写「过了定量护栏」这个**事实**。
        # 早先的实现把 passed_guardrails 重置为 False 来实现否决；那会让该因子以 passed=False
        # 落进 known_invalid 被当作「已验证无效」——同样是污染，只是方向相反。
        # 现在 passed_guardrails 是不可变事实，否决由 known_valid() 读 verdict 完成。
        dropped_attempts = [a for a in res.state.attempts if a.expression == drop_expr]
        assert dropped_attempts, f"未找到 {drop_expr} 的 AttemptRecord"
        assert all(a.passed_guardrails for a in dropped_attempts), \
            "drop 不得改写 passed_guardrails 这个事实（它确实过了定量护栏）"
        assert all(a.critic_verdict == "drop" for a in dropped_attempts), \
            "否决必须记在 critic_verdict 上"

        # 持久化层：落盘的是事实 + 裁决，二者并存不矛盾
        index = ExperimentIndex(idx_path)
        persisted = [r for r in index.load() if r.get("expression") == drop_expr]
        assert persisted, f"experiment_index.jsonl 未找到 {drop_expr} 的记录"
        assert all(r.get("passed") is True for r in persisted)
        assert all(r.get("verdict") == "drop" for r in persisted)

        # 最终不变量（无论用哪种修复方案，这条都必须成立）：
        # 被 Critic 否决的表达式不能被 known_valid() 当作"已验证有效"
        assert drop_expr not in index.known_valid(k=10), \
            f"被否决的表达式 {drop_expr} 不应出现在 known_valid() 中"
        # 对偶不变量：它过了护栏，也不该被当作"已验证无效"喂给 LLM
        assert drop_expr not in index.known_invalid(k=10), \
            f"过了定量护栏的表达式 {drop_expr} 不应出现在 known_invalid() 中"

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_critic_drop_removes_candidate(_tp4)

    # -- 原 test_revise_runs_alongside_new_hypotheses --
    def _section_5_test_revise_runs_alongside_new_hypotheses(tmp_path):
        from unittest.mock import patch

        state = {"crit": 0, "prop": 0}

        def fn(messages):
            text = "\n".join(m["content"] for m in messages)
            if "风控审计员" in text:                      # critic：首轮 revise，其后 keep
                state["crit"] += 1
                v = "revise_expr" if state["crit"] == 1 else "keep"
                return json.dumps({"verdict": v, "reason": "窗口太短"})
            if "风控反馈" in text and "改写出" in text:    # revise_expressions
                return json.dumps({"expressions": ["ts_mean(close,20)"]})
            if "HYPREV" in text:                          # write(新假设2)
                return json.dumps({"expressions": ["ts_std(close,10)"]})
            if "HYPMOM" in text:                          # write(新假设1)
                return json.dumps({"expressions": ["ts_mean(close,5)"]})
            state["prop"] += 1                            # propose：轮1 HYPMOM、轮2 HYPREV
            return json.dumps({"hypotheses": ["HYPMOM" if state["prop"] == 1 else "HYPREV"]})

        daily = _mock_daily__team_orch()
        with patch("factorzen.agents.team_orchestrator.node_guardrails",
                   side_effect=_inject_cands_for_critic):
            res = run_team_agent(daily, fn, n_rounds=2, seed=1, heal_rounds=0,
                                 index_path=str(tmp_path / "e.jsonl"))
        exprs = {a.expression for a in res.state.attempts}
        assert "ts_mean(close, 20)" in exprs, f"轮2 应有修订产物: {exprs}"
        assert "ts_std(close, 10)" in exprs, f"轮2 还应有新假设产物(修订不得挤占): {exprs}"
        assert state["prop"] == 2, f"两轮都应 propose(修订轮不得跳过新假设), 实得 {state['prop']}"

    _tp5 = tmp_path / "_s5"
    _tp5.mkdir(exist_ok=True)
    _section_5_test_revise_runs_alongside_new_hypotheses(_tp5)


# ==== 来自 test_agent_orchestrator.py ====
def _mock_daily__agent_orch(n_stocks=40, n_days=180, seed=1):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


def _scripted_llm():
    """每轮：1 个 proposal + semantic(pass) + critic(keep)。无限循环复用。"""
    prop = json.dumps({"hypothesis": "动量", "expressions": ["ts_mean(close,5)"], "rationale": "r"})
    sem = json.dumps({"consistent": True, "reason": "ok"})
    crit = json.dumps({"verdict": "keep", "reason": "ok"})
    seq = [prop, sem, crit] * 50
    i = {"k": 0}
    def fn(messages):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v
    return fn


def test_run_llm_agent_reproducible():
    daily = _mock_daily__agent_orch()
    r1 = run_llm_agent(daily, _scripted_llm(), n_rounds=2, seed=7, library_orthogonal=False)
    r2 = run_llm_agent(daily, _scripted_llm(), n_rounds=2, seed=7, library_orthogonal=False)
    # 同 seed + 同 scripted LLM → 尝试序列逐字节一致
    assert [a.expression for a in r1.state.attempts] == [a.expression for a in r2.state.attempts]
    assert r1.n_trials == r2.n_trials

# ==== 来自 test_agent_patience.py ====
def _mock_daily__patience(n_stocks=20, n_days=180, seed=1):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


def _fn_invalid():
    """始终产非法表达式 → 永无候选过护栏 → 触发 patience 早停。"""
    seq = [json.dumps({"hypotheses": ["动量"]}),
           json.dumps({"expressions": ["not_a_func("]}),
           json.dumps({"verdict": "keep", "reason": "ok"})]
    i = {"k": 0}

    def fn(_m):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v
    return fn


def test_patience_suite(tmp_path, monkeypatch):
    """test_team_patience_early_stops；patience=None（默认）→ 跑满 n_rounds（零回归）。；每轮都出新候选 → 计数器必须重置 → 不该被 patience=2 早停。；单 Agent 路径的 patience **行为**此前从未被测（只断言了 signature 里有这个参数）。；test_m5_patience_resets_when_a_new_candidate_appears；`no_improve >= patience` 在 patience=0 时于第 2 轮开头恒成立——**即使刚产出新候选**。"""
    # -- 原 test_team_patience_early_stops --
    def _section_0_test_team_patience_early_stops(tmp_path):
        res = run_team_agent(_mock_daily__patience(), _fn_invalid(), n_rounds=8, seed=1,
                             index_path=str(tmp_path / "e.jsonl"), patience=2)
        assert res.state.iteration < 8, f"patience 未早停, iteration={res.state.iteration}"

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_team_patience_early_stops(_tp0)

    # -- 原 test_team_patience_none_runs_all_rounds --
    def _section_1_test_team_patience_none_runs_all_rounds(tmp_path):
        res = run_team_agent(_mock_daily__patience(), _fn_invalid(), n_rounds=3, seed=1,
                             index_path=str(tmp_path / "e.jsonl"))
        assert res.state.iteration == 3

    monkeypatch.undo()
    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_team_patience_none_runs_all_rounds(_tp1)

    # -- 原 test_team_patience_resets_when_a_new_candidate_appears --
    def _section_2_test_team_patience_resets_when_a_new_candidate_appears(tmp_path, monkeypatch):
        import factorzen.agents.team_orchestrator as team

        monkeypatch.setattr(team, "node_guardrails", _stub_guardrails(yields_candidate=True))
        res = run_team_agent(_mock_daily__patience(), _fn_valid(), n_rounds=6, seed=1,
                             index_path=str(tmp_path / "e.jsonl"), patience=2)

        assert res.state.iteration == 6, \
            f"每轮都有新候选，patience 不该触发；实得 iteration={res.state.iteration}"

    monkeypatch.undo()
    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_team_patience_resets_when_a_new_candidate_appears(_tp2, monkeypatch)

    # -- 原 test_m5_patience_early_stops --
    def _section_3_test_m5_patience_early_stops(monkeypatch):
        import factorzen.agents.orchestrator as orch
        from factorzen.agents.orchestrator import run_llm_agent

        monkeypatch.setattr(orch, "node_guardrails", _stub_guardrails(yields_candidate=False))
        res = run_llm_agent(_mock_daily__patience(n_stocks=40), _fn_m5(), n_rounds=8, seed=1, patience=2, library_orthogonal=False)

        assert res.state.iteration == 2, f"连续 2 轮无新候选应早停，实得 {res.state.iteration}"

    monkeypatch.undo()
    _section_3_test_m5_patience_early_stops(monkeypatch)

    # -- 原 test_m5_patience_resets_when_a_new_candidate_appears --
    def _section_4_test_m5_patience_resets_when_a_new_candidate_appears(monkeypatch):
        import factorzen.agents.orchestrator as orch
        from factorzen.agents.orchestrator import run_llm_agent

        monkeypatch.setattr(orch, "node_guardrails", _stub_guardrails(yields_candidate=True))
        res = run_llm_agent(_mock_daily__patience(n_stocks=40), _fn_m5(), n_rounds=5, seed=1, patience=2, library_orthogonal=False)

        assert res.state.iteration == 5, "每轮都有新候选，patience 不该触发"

    monkeypatch.undo()
    _section_4_test_m5_patience_resets_when_a_new_candidate_appears(monkeypatch)

    # -- 原 test_cli_rejects_non_positive_patience --
    def _section_5_test_cli_rejects_non_positive_patience():
        import pytest

        from factorzen.cli.main import main

        for bad in ("0", "-1"):
            with pytest.raises(SystemExit) as ei:
                main(["mine", "agent", "--start", "20220101", "--end", "20231229",
                      "--patience", bad])
            assert ei.value.code == 2, "argparse 参数校验失败应退出码 2"

    monkeypatch.undo()
    _section_5_test_cli_rejects_non_positive_patience()


# ── 计数器重置分支：此前两条路径都没测 ──────────────────────────────────────
#
# 唯一的行为测试用 `_fn_invalid`（候选**永不增长**）。在那个退化场景下，「正确逻辑」与
# 「计数器永不重置」的变异体行为完全一致——测试对重置分支零判别力。
# 「每轮持续产出候选却被误早停」这个真实 bug，此前没有任何测试能抓到。


def _fn_valid():
    """始终产出合法表达式；候选由 stub 的 node_guardrails 注入。"""
    seq = [json.dumps({"hypotheses": ["动量"]}),
           json.dumps({"expressions": ["ts_mean(close,5)"]}),
           json.dumps({"verdict": "keep", "reason": "ok"})]
    i = {"k": 0}

    def fn(_m):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v
    return fn


def _stub_guardrails(*, yields_candidate: bool):
    """替身护栏：`yields_candidate=True` 时每轮新增一个候选。

    候选形状必须与真实 `node_guardrails` 一致——`n_train` / `ic_ci_low` / `ic_ci_high`
    是收尾复核（`node_finalize_guardrails`）重算 DSR 所必需的。少写字段会让替身造出
    生产中不存在的形状，测试就跑在幻觉上了。
    """
    def fake(state, *, daily, holdout_df, bundle, ledger, top_k=5, dsr_alpha=0.05,
             warmup_daily=None, eval_start=None, **_kwargs):
        ledger.record(1)
        if yields_candidate:
            state.attempts.append(AttemptRecord(
                iteration=state.iteration, hypothesis="h", expression=f"e{state.iteration}",
                compile_ok=True, ic_train=0.05, passed_guardrails=True, critic_verdict=None,
                error=None, ir_train=0.4, turnover=0.3, n_train=300))
            state.candidates.append({"expression": f"e{state.iteration}", "hypothesis": "h",
                                     "ic_train": 0.05, "ir_train": 0.4, "turnover": 0.3,
                                     "holdout_ic": 0.04, "holdout_ir": 0.3,
                                     "dsr": 0.9, "dsr_pvalue": 0.01,
                                     "n_train": 300, "ic_ci_low": 0.01, "ic_ci_high": 0.07})
        return state
    return fake


def _fn_m5():
    """单 Agent 的 LLM 脚本：proposal → semantic → critic，每轮表达式不同以避开去重。"""
    st = {"round": -1}

    def fn(messages):
        system = messages[0]["content"]
        if "consistent" in system:
            return json.dumps({"consistent": True, "reason": "ok"})
        if "verdict" in system:
            return json.dumps({"verdict": "keep", "reason": "ok"})
        st["round"] += 1
        return json.dumps({"hypothesis": f"h{st['round']}",
                           "expressions": [f"ts_mean(close,{4 + st['round']})"],
                           "rationale": "r"})
    return fn


# ── patience=0 的边界：CLI 不该放行一个语义反直觉的值 ────────────────────────


# ==== 来自 test_team_roles.py ====
# ==== 来自 test_team_coder.py ====
class FakeLLM__coder:
    def __init__(self, responses):
        self._r = list(responses)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return self._r.pop(0) if self._r else "{}"


def test_coder_role_suite():
    """test_write_expressions_lists_ops；test_revise_uses_critic_reason；test_write_garbage_returns_empty"""
    # -- 原 test_write_expressions_lists_ops --
    def _section_0_test_write_expressions_lists_ops():
        llm = FakeLLM__coder([json.dumps({"expressions": ["ts_mean(close,5)", "rank(vol)"]})])
        out = write_expressions("动量", llm)
        assert out == ["ts_mean(close,5)", "rank(vol)"]
        blob = " ".join(m["content"] for m in llm.calls[0])
        assert "ts_mean" in blob and "close" in blob  # 算子/特征清单进 prompt

    _section_0_test_write_expressions_lists_ops()

    # -- 原 test_revise_uses_critic_reason --
    def _section_1_test_revise_uses_critic_reason():
        llm = FakeLLM__coder([json.dumps({"expressions": ["ts_mean(close,20)"]})])
        out = revise_expressions("动量", ["ts_mean(close,5)"], "窗口太短", llm)
        assert out == ["ts_mean(close,20)"]
        blob = " ".join(m["content"] for m in llm.calls[0])
        assert "窗口太短" in blob and "ts_mean(close,5)" in blob  # 反馈+原表达式进 prompt

    _section_1_test_revise_uses_critic_reason()

    # -- 原 test_write_garbage_returns_empty --
    def _section_2_test_write_garbage_returns_empty():
        llm = FakeLLM__coder(["非 JSON"])
        assert write_expressions("动量", llm) == []

    _section_2_test_write_garbage_returns_empty()


# ==== 来自 test_team_critic.py ====
class FakeLLM__critic:
    def __init__(self, responses):
        self._r = list(responses)

    def __call__(self, messages):
        return self._r.pop(0) if self._r else "{}"


def _cand(**kw):
    base = {"expression": "ts_mean(close,5)", "hypothesis": "动量", "ic_train": 0.05,
            "holdout_ic": 0.03, "dsr": 0.7, "dsr_pvalue": 0.01}
    base.update(kw)
    return base


def test_critic_role_suite():
    """test_critique_keep；test_critique_drop_overfit；test_critique_revise_variants；test_critique_garbage_defaults_keep；test_critique_unknown_verdict_defaults_keep"""
    # -- 原 test_critique_keep --
    def _section_0_test_critique_keep():
        llm = FakeLLM__critic([json.dumps({"verdict": "keep", "reason": "稳健"})])
        v = critique(_cand(), llm)
        assert isinstance(v, CriticVerdict) and v.verdict == "keep"

    _section_0_test_critique_keep()

    # -- 原 test_critique_drop_overfit --
    def _section_1_test_critique_drop_overfit():
        llm = FakeLLM__critic([json.dumps({"verdict": "drop", "reason": "DSR 不显著疑过拟合"})])
        v = critique(_cand(dsr=0.2, dsr_pvalue=0.4), llm)
        assert v.verdict == "drop" and v.reason

    _section_1_test_critique_drop_overfit()

    # -- 原 test_critique_revise_variants --
    def _section_2_test_critique_revise_variants():
        llm = FakeLLM__critic([json.dumps({"verdict": "revise_expr", "reason": "窗口太短"}),
                       json.dumps({"verdict": "revise_hypothesis", "reason": "方向牵强"})])
        assert critique(_cand(), llm).verdict == "revise_expr"
        assert critique(_cand(), llm).verdict == "revise_hypothesis"

    _section_2_test_critique_revise_variants()

    # -- 原 test_critique_garbage_defaults_keep --
    def _section_3_test_critique_garbage_defaults_keep():
        llm = FakeLLM__critic(["不是 JSON"])
        assert critique(_cand(), llm).verdict == "keep"

    _section_3_test_critique_garbage_defaults_keep()

    # -- 原 test_critique_unknown_verdict_defaults_keep --
    def _section_4_test_critique_unknown_verdict_defaults_keep():
        llm = FakeLLM__critic([json.dumps({"verdict": "explode", "reason": "x"})])
        assert critique(_cand(), llm).verdict == "keep"   # 非法 verdict 归一到 keep

    _section_4_test_critique_unknown_verdict_defaults_keep()


# ==== 来自 test_team_critic_grouping.py ====
def _mock_daily(n_stocks=40, n_days=180, seed=1):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                "high": px * 1.01, "low": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
            })
    return pl.DataFrame(rows)


def _inject_guardrails_from_attempts(state, *, ledger, **_kwargs):
    """把本轮 attempts 全量注入 candidates（含 hypothesis），模拟护栏通过。

    字段对齐 nodes.py cand_row；ic/holdout 强制同号正值，保证 library 收尾复核不误杀。
    """
    n = 0
    for a in state.attempts:
        if a.iteration != state.iteration:
            continue
        a.passed_guardrails = True
        state.candidates.append({
            "expression": a.expression,
            "hypothesis": a.hypothesis,
            "ic_train": 0.05,
            "ir_train": 0.4,
            "turnover": 0.1,
            "holdout_ic": 0.04,
            "holdout_ir": 0.3,
            "dsr": 0.7,
            "dsr_pvalue": 0.05,
            "n_train": a.n_train if a.n_train is not None else 100,
            "n_holdout_days": 80,  # ≥ DEFAULT_HOLDOUT_MIN_DAYS，收尾 library 覆盖门
            "ic_ci_low": 0.01,
            "ic_ci_high": 0.08,
        })
        n += 1
    if n:
        ledger.record(n)
    return state


def test_critic_grouping_suite(tmp_path):
    """两假设各 1 候选：H1 drop / H2 keep → 只杀 H1，verdict 不交叉污染。；两假设轮恰好调用 2 次 critique（每假设一次）。"""
    # -- 原 test_critic_groups_by_hypothesis_no_cross_kill --
    def _section_0_test_critic_groups_by_hypothesis_no_cross_kill(tmp_path):
        h1, h2 = "HYPG1", "HYPG2"
        expr1, expr2 = "ts_mean(close,5)", "ts_std(close,10)"
        # 评估规范化后带空格
        norm1, norm2 = "ts_mean(close, 5)", "ts_std(close, 10)"

        def fn(messages):
            text = "\n".join(m["content"] for m in messages)
            if "风控审计员" in text:
                # critique user 内容含「假设: ...」；按代表候选的 hypothesis 分流
                if f"假设: {h1}" in text:
                    return json.dumps({"verdict": "drop", "reason": "H1 过拟合"})
                if f"假设: {h2}" in text:
                    return json.dumps({"verdict": "keep", "reason": "H2 稳健"})
                return json.dumps({"verdict": "keep", "reason": "fallback"})
            if "翻译成" in text:
                if h1 in text:
                    return json.dumps({"expressions": [expr1]})
                if h2 in text:
                    return json.dumps({"expressions": [expr2]})
                return json.dumps({"expressions": ["rank(vol)"]})
            return json.dumps({"hypotheses": [h1, h2]})

        daily = _mock_daily()
        with patch(
            "factorzen.agents.team_orchestrator.node_guardrails",
            _inject_guardrails_from_attempts,
        ):
            res = run_team_agent(
                daily, fn, n_rounds=1, seed=1, heal_rounds=0,
                index_path=str(tmp_path / "e.jsonl"), hypotheses_per_round=2,
            )

        cand_exprs = {c["expression"] for c in res.candidates}
        assert norm2 in cand_exprs, f"H2 keep 候选应保留: {cand_exprs}"
        assert norm1 not in cand_exprs, f"H1 drop 候选应移除: {cand_exprs}"

        by_expr = {a.expression: a for a in res.state.attempts}
        assert by_expr[norm1].critic_verdict == "drop"
        assert by_expr[norm2].critic_verdict == "keep"
        # 事实字段不许被 verdict 改写
        assert by_expr[norm1].passed_guardrails is True
        assert by_expr[norm2].passed_guardrails is True

        last = res.rounds_log[-1]
        assert "verdicts" in last and len(last["verdicts"]) == 2
        by_h = {v["hypothesis"]: v for v in last["verdicts"]}
        assert by_h[h1]["verdict"] == "drop"
        assert by_h[h2]["verdict"] == "keep"
        # 原键 = 最后一组（H2 keep）零回归语义
        assert last["verdict"] == "keep"
        assert last["reason"] == "H2 稳健"

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_critic_groups_by_hypothesis_no_cross_kill(_tp0)

    # -- 原 test_critic_called_once_per_hypothesis --
    def _section_1_test_critic_called_once_per_hypothesis(tmp_path):
        h1, h2 = "HYPC1", "HYPC2"
        n_crit = {"k": 0}

        def fn(messages):
            text = "\n".join(m["content"] for m in messages)
            if "风控审计员" in text:
                n_crit["k"] += 1
                return json.dumps({"verdict": "keep", "reason": "ok"})
            if "翻译成" in text:
                if h1 in text:
                    return json.dumps({"expressions": ["ts_mean(close,5)"]})
                if h2 in text:
                    return json.dumps({"expressions": ["ts_std(close,10)"]})
                return json.dumps({"expressions": ["rank(vol)"]})
            return json.dumps({"hypotheses": [h1, h2]})

        daily = _mock_daily()
        with patch(
            "factorzen.agents.team_orchestrator.node_guardrails",
            _inject_guardrails_from_attempts,
        ):
            run_team_agent(
                daily, fn, n_rounds=1, seed=1, heal_rounds=0,
                index_path=str(tmp_path / "e.jsonl"), hypotheses_per_round=2,
            )

        assert n_crit["k"] == 2, f"两假设应 critique 恰好 2 次，实得 {n_crit['k']}"

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_critic_called_once_per_hypothesis(_tp1)


# ==== 来自 test_team_hypothesis.py ====
class FakeLLM__hypothesis:
    def __init__(self, responses):
        self._r = list(responses)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return self._r.pop(0) if self._r else "{}"


def test_hypothesis_role_suite():
    """test_propose_returns_directions；test_known_invalid_injected_into_prompt；test_propose_garbage_returns_empty"""
    # -- 原 test_propose_returns_directions --
    def _section_0_test_propose_returns_directions():
        llm = FakeLLM__hypothesis([json.dumps({"hypotheses": ["小市值反转", "高换手动量"]})])
        out = propose_hypotheses(llm, known_invalid=[], known_valid=[], n=2)
        assert out == ["小市值反转", "高换手动量"]

    _section_0_test_propose_returns_directions()

    # -- 原 test_known_invalid_injected_into_prompt --
    def _section_1_test_known_invalid_injected_into_prompt():
        llm = FakeLLM__hypothesis([json.dumps({"hypotheses": ["x"]})])
        propose_hypotheses(llm, known_invalid=["rank(vol)"], known_valid=["ts_mean(close, 5)"], n=1)
        blob = " ".join(m["content"] for m in llm.calls[0])
        assert "rank(vol)" in blob  # 已知无效注入(避开)
        assert "ts_mean(close, 5)" in blob  # 已知有效作方向参考

    _section_1_test_known_invalid_injected_into_prompt()

    # -- 原 test_propose_garbage_returns_empty --
    def _section_2_test_propose_garbage_returns_empty():
        llm = FakeLLM__hypothesis(["非 JSON"])
        assert propose_hypotheses(llm, known_invalid=[], known_valid=[]) == []

    _section_2_test_propose_garbage_returns_empty()


# ==== 来自 test_team_librarian.py ====
def test_librarian_role_suite(tmp_path):
    """test_record_then_recall_roundtrip；test_recall_empty_index；record(candidates=...) → holdout_ic 写入 index → known_valid 按 holdout_ic 降序。；AttemptRecord.critic_verdict 非 None 时，record 正确写入 verdict 字段。"""
    # -- 原 test_record_then_recall_roundtrip --
    def _section_0_test_record_then_recall_roundtrip(tmp_path):
        idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
        attempts = [
            AttemptRecord(0, "动量", "ts_mean(close,5)", True, 0.05, True, "keep", None, ir_train=0.4),
            AttemptRecord(0, "换手", "rank(vol)", True, 0.001, False, "drop", None, ir_train=0.01),
        ]
        record(idx, attempts, run_id="r1")
        r = recall(idx, k=5)
        assert "ts_mean(close, 5)" in r.seen and "rank(vol)" in r.seen   # 归一化查重集
        assert "rank(vol)" in r.known_invalid                            # 未过护栏
        assert "ts_mean(close, 5)" in r.known_valid                      # 过护栏

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_record_then_recall_roundtrip(_tp0)

    # -- 原 test_recall_empty_index --
    def _section_1_test_recall_empty_index(tmp_path):
        r = recall(ExperimentIndex(str(tmp_path / "none.jsonl")), k=5)
        assert r.seen == set() and r.known_invalid == [] and r.known_valid == []

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_recall_empty_index(_tp1)

    # -- 原 test_record_backfills_holdout_ic --
    def _section_2_test_record_backfills_holdout_ic(tmp_path):
        idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
        attempts = [
            AttemptRecord(0, "动量", "ts_mean(close,5)", True, 0.05, True, "keep", None, ir_train=0.4),
            AttemptRecord(0, "反转", "rank(vol)", True, 0.03, True, "keep", None, ir_train=0.2),
        ]
        # rank(vol) 的 holdout_ic 更高——若排序正确，known_valid[0] 应为 rank(vol)
        # candidates 用归一化形式（空格）验证 _normalize 匹配路径
        candidates = [
            {"expression": "ts_mean(close, 5)", "holdout_ic": 0.02, "ic_train": 0.05},
            {"expression": "rank(vol)", "holdout_ic": 0.06, "ic_train": 0.03},
        ]
        record(idx, attempts, run_id="r1", candidates=candidates)
        recs = idx.load()
        # idx.load() 返回原始 expression（AttemptRecord.expression，无空格）
        hic_map = {r["expression"]: r.get("holdout_ic") for r in recs}
        assert hic_map.get("ts_mean(close,5)") == 0.02, f"holdout_ic 未写入: {hic_map}"
        assert hic_map.get("rank(vol)") == 0.06
        # known_valid 按 holdout_ic 降序，返回归一化形式：rank(vol) 排第一
        r = recall(idx, k=5)
        assert r.known_valid[0] == "rank(vol)", f"期望 rank(vol) 排第一，实际 {r.known_valid}"

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_record_backfills_holdout_ic(_tp2)

    # -- 原 test_record_backfills_critic_verdict --
    def _section_3_test_record_backfills_critic_verdict(tmp_path):
        idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
        attempts = [
            AttemptRecord(0, "动量", "ts_mean(close,5)", True, 0.05, True, "keep", None, ir_train=0.4),
            AttemptRecord(0, "换手", "rank(vol)", True, 0.001, False, "drop", None, ir_train=0.01),
        ]
        record(idx, attempts, run_id="r1")
        recs = idx.load()
        # idx.load() 返回原始 expression（AttemptRecord.expression，无空格）
        verdict_map = {r["expression"]: r.get("verdict") for r in recs}
        assert verdict_map.get("ts_mean(close,5)") == "keep"
        assert verdict_map.get("rank(vol)") == "drop"

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_record_backfills_critic_verdict(_tp3)



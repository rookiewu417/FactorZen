"""合并自 agents 相关碎片测试（test_orchestration.py）。

test_team_orchestrator.py：team 编排闭环：修订循环、跨 session 去重、critic drop
test_agent_orchestrator.py：run_llm_agent 闭环跑通与同 seed 可复现
test_agent_patience.py：Workstream G：自适应终止（连续 patience 轮无新 passed 候选则早停）
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.agents.orchestrator import run_llm_agent
from factorzen.agents.state import AttemptRecord
from factorzen.agents.team_orchestrator import run_team_agent


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


def test_run_team_closes_loop(tmp_path: Path):
    daily = _mock_daily__team_orch()
    res = run_team_agent(daily, _scripted_team(), n_rounds=2, seed=42,
                         index_path=str(tmp_path / "e.jsonl"))
    assert res.state.iteration == 2
    assert res.n_trials >= 1
    assert len(res.rounds_log) >= 1     # 角色决策可审计


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


def test_run_team_revise_loop_counts_n(tmp_path: Path):
    """轮1 Critic revise_expr → 轮2 Coder 改写（跨轮 feedback），两表达式都评估、都计入 N。"""
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


def test_hypotheses_per_round_evaluates_all(tmp_path: Path):
    """任务 D：hypotheses_per_round=2 → propose 收到 n=2、两个假设的表达式都被评估，
    且每个 attempt 的 hypothesis 归属正确（护栏/Critic 仍每轮一次）。"""
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


def test_hypotheses_per_round_default_is_single(tmp_path: Path):
    """默认 hypotheses_per_round=1 → propose 收到 n=1（零回归）。"""
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


def test_cross_session_dedup(tmp_path: Path):
    """共享 experiment_index：第二次 run 重复表达式被跳过（seen 去重）。"""
    daily = _mock_daily__team_orch()
    idx_path = str(tmp_path / "shared.jsonl")
    run_team_agent(daily, _scripted_team(), n_rounds=1, seed=1, index_path=idx_path)
    res2 = run_team_agent(daily, _scripted_team(), n_rounds=1, seed=1, index_path=idx_path)
    # 第二次 run 产同样的 ts_mean(close,5)，已在 index → 本轮无新评估（n_trials 可能为 0）
    assert res2.n_trials == 0 or all(
        a.expression != "ts_mean(close, 5)" for a in res2.state.attempts)


def test_critic_drop_removes_candidate(tmp_path: Path):
    """scripted Critic drop → 被 drop 候选从 TeamResult.candidates 移除，且否决回路真实生效。

    回归覆盖（原 bug：跨轮否决名存实亡）：`node_guardrails` 把过了定量护栏的 AttemptRecord
    标为 `passed_guardrails=True`，而 Critic 判定 drop 后若无任何机制阻断，该记录会被
    `ExperimentIndex.known_valid()` 当作"已验证有效"喂给后续轮次/session 的假设生成
    ——否决回路被绕过。

    实现几经变化，**不变量始终是最后两条断言**：被否决者不进 known_valid，
    且（因它确实过了护栏）也不进 known_invalid。早先靠重置 `passed_guardrails=False` 实现，
    那是用事实字段编码复用决策，会把它推进 known_invalid（反向污染）；
    现在 `passed_guardrails` 是不可变事实，否决由 `known_valid()` 读 `critic_verdict` 完成。

    fake_guardrails 忠实复刻真实 node_guardrails 的副作用（设 passed_guardrails=True），
    否则该 mock 形同虚设（被否决的 attempt 从未被标记为"已过护栏"）。

    N 诚实验证：drop 移除候选不影响 ledger（attempt 已计入 n_trials）。
    """
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


def test_revise_runs_alongside_new_hypotheses(tmp_path: Path):
    """轮1 Critic revise_expr → 轮2 必须**同时**评估修订产物与新假设产物。

    修复动机：GPT 类引擎的候选常被 Critic 判 revise_expr，纯修订轮会把吞吐
    塌缩（实测 19→3→2）——修订价值保留，但不得挤占新假设配额。
    路由用独特假设名 HYPMOM/HYPREV（样板文案不可能含它们，防误触发假绿）。

    W5c：空轮跳 critic；本测需强制 new_cands 非空才能让 scripted critic 出 revise_expr。
    """
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


def test_run_llm_agent_closes_loop():
    daily = _mock_daily__agent_orch()
    res = run_llm_agent(daily, _scripted_llm(), n_rounds=3, seed=42, library_orthogonal=False)
    assert res.state.iteration == 3
    assert res.n_trials >= 1            # N 累加了
    assert len(res.state.attempts) >= 1


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


def test_team_patience_early_stops(tmp_path):
    res = run_team_agent(_mock_daily__patience(), _fn_invalid(), n_rounds=8, seed=1,
                         index_path=str(tmp_path / "e.jsonl"), patience=2)
    assert res.state.iteration < 8, f"patience 未早停, iteration={res.state.iteration}"


def test_team_patience_none_runs_all_rounds(tmp_path):
    """patience=None（默认）→ 跑满 n_rounds（零回归）。"""
    res = run_team_agent(_mock_daily__patience(), _fn_invalid(), n_rounds=3, seed=1,
                         index_path=str(tmp_path / "e.jsonl"))
    assert res.state.iteration == 3


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


def test_team_patience_resets_when_a_new_candidate_appears(tmp_path, monkeypatch):
    """每轮都出新候选 → 计数器必须重置 → 不该被 patience=2 早停。"""
    import factorzen.agents.team_orchestrator as team

    monkeypatch.setattr(team, "node_guardrails", _stub_guardrails(yields_candidate=True))
    res = run_team_agent(_mock_daily__patience(), _fn_valid(), n_rounds=6, seed=1,
                         index_path=str(tmp_path / "e.jsonl"), patience=2)

    assert res.state.iteration == 6, \
        f"每轮都有新候选，patience 不该触发；实得 iteration={res.state.iteration}"


def test_m5_patience_early_stops(monkeypatch):
    """单 Agent 路径的 patience **行为**此前从未被测（只断言了 signature 里有这个参数）。"""
    import factorzen.agents.orchestrator as orch
    from factorzen.agents.orchestrator import run_llm_agent

    monkeypatch.setattr(orch, "node_guardrails", _stub_guardrails(yields_candidate=False))
    res = run_llm_agent(_mock_daily__patience(n_stocks=40), _fn_m5(), n_rounds=8, seed=1, patience=2, library_orthogonal=False)

    assert res.state.iteration == 2, f"连续 2 轮无新候选应早停，实得 {res.state.iteration}"


def test_m5_patience_resets_when_a_new_candidate_appears(monkeypatch):
    import factorzen.agents.orchestrator as orch
    from factorzen.agents.orchestrator import run_llm_agent

    monkeypatch.setattr(orch, "node_guardrails", _stub_guardrails(yields_candidate=True))
    res = run_llm_agent(_mock_daily__patience(n_stocks=40), _fn_m5(), n_rounds=5, seed=1, patience=2, library_orthogonal=False)

    assert res.state.iteration == 5, "每轮都有新候选，patience 不该触发"


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


def test_cli_rejects_non_positive_patience():
    """`no_improve >= patience` 在 patience=0 时于第 2 轮开头恒成立——**即使刚产出新候选**。

    于是 `--patience 0` 静默变成「只跑 1 轮」，无视 `--iterations`。而 help 文案说的是
    「连续 N 轮无新候选则早停」，用户传 0 期望「不早停/更激进」，得到的却相反。
    """
    import pytest

    from factorzen.cli.main import main

    for bad in ("0", "-1"):
        with pytest.raises(SystemExit) as ei:
            main(["mine", "agent", "--start", "20220101", "--end", "20231229",
                  "--patience", bad])
        assert ei.value.code == 2, "argparse 参数校验失败应退出码 2"

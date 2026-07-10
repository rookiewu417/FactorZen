# tests/test_team_orchestrator.py
import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.agents.team_orchestrator import run_team_agent


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
    daily = _mock_daily()
    res = run_team_agent(daily, _scripted_team(), n_rounds=2, seed=42,
                         index_path=str(tmp_path / "e.jsonl"))
    assert res.state.iteration == 2
    assert res.n_trials >= 1
    assert len(res.rounds_log) >= 1     # 角色决策可审计


def test_run_team_revise_loop_counts_n(tmp_path: Path):
    """轮1 Critic revise_expr → 轮2 Coder 改写（跨轮 feedback），两表达式都评估、都计入 N。"""
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

    daily = _mock_daily()
    res = run_team_agent(daily, fn, n_rounds=2, seed=1, index_path=str(tmp_path / "e.jsonl"))
    assert res.n_trials >= 2     # 两轮各评估一个表达式(原始 + 改写)，都计入 N
    assert any("ts_mean(close, 20)" in r["expressions"] for r in res.rounds_log)  # 轮2 是改写产物


def test_cross_session_dedup(tmp_path: Path):
    """共享 experiment_index：第二次 run 重复表达式被跳过（seen 去重）。"""
    daily = _mock_daily()
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
                        eval_start=None):
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

    daily = _mock_daily()
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

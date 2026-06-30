"""验收侧节点测试：node_guardrails / node_critic / node_reflect。

判别力约束：
- N 记账诚实（ledger.n_trials 累加）
- holdout 隔离（mining 段 max_date < holdout 段 min_date）
- family-aware 去冗余（高度相关的两个候选只入选一个）
- critic keep/drop 改判生效
- reflect 低 IC 进负例库
"""
from __future__ import annotations

import json

from factorzen.agents.nodes import node_critic, node_reflect
from factorzen.agents.state import AgentState, AttemptRecord


class FakeLLM:
    def __init__(self, responses):
        self._r = list(responses)

    def __call__(self, messages):
        return self._r.pop(0) if self._r else '{"verdict":"keep","reason":"ok"}'


def _state_with_attempts():
    s = AgentState(seed=1)
    s.attempts = [
        AttemptRecord(0, "h1", "ts_mean(close,5)", True, 0.05, True, None, None),
        AttemptRecord(0, "h2", "rank(vol)", True, 0.001, False, None, None),  # 低 IC 未过护栏
    ]
    return s


# ---------------------------------------------------------------------------
# node_critic 测试
# ---------------------------------------------------------------------------


def test_node_critic_marks_verdict():
    s = _state_with_attempts()
    llm = FakeLLM(
        [
            json.dumps({"verdict": "keep", "reason": "经济直觉成立"}),
            json.dumps({"verdict": "drop", "reason": "疑似数据窥探"}),
        ]
    )
    s = node_critic(s, llm)
    verdicts = [a.critic_verdict for a in s.attempts]
    assert "keep" in verdicts and "drop" in verdicts


def test_node_critic_skips_already_judged():
    """已有 verdict 的 attempt 不再调用 llm。"""
    s = AgentState(seed=1)
    s.attempts = [
        AttemptRecord(0, "h1", "ts_mean(close,5)", True, 0.05, True, "keep", None),
    ]
    call_count = [0]

    def counting_llm(msgs):
        call_count[0] += 1
        return '{"verdict":"drop","reason":"x"}'

    node_critic(s, counting_llm)
    assert call_count[0] == 0, "已判定 attempt 不应再调用 LLM"


# ---------------------------------------------------------------------------
# node_reflect 测试
# ---------------------------------------------------------------------------


def test_node_reflect_feeds_low_ic_to_negatives():
    s = _state_with_attempts()
    s = node_reflect(s, ic_threshold=0.01)
    # 低 IC 的 rank(vol) 进负例库，高 IC 的不进
    assert "rank(vol)" in s.negative_examples
    assert "ts_mean(close,5)" not in s.negative_examples


def test_node_reflect_increments_iteration():
    s = _state_with_attempts()
    assert s.iteration == 0
    s = node_reflect(s, ic_threshold=0.01)
    assert s.iteration == 1


# ---------------------------------------------------------------------------
# node_guardrails 测试：N 记账 + holdout 隔离
# ---------------------------------------------------------------------------


def test_node_guardrails_n_accounting_and_holdout_isolation():
    """① ledger.n_trials 累加；② 候选带 holdout_ic/dsr；③ holdout 隔离。"""
    import datetime as dt

    import numpy as np
    import polars as pl

    from factorzen.agents.evaluation import evaluate_expressions
    from factorzen.agents.nodes import node_guardrails
    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import split_holdout
    from factorzen.validation.multiple_testing import TrialLedger

    rng = np.random.default_rng(3)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < 180:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(20)]
    rows = []
    for c in codes:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append(
                {
                    "trade_date": dd,
                    "ts_code": c,
                    "close": px,
                    "open": px * 0.99,
                    "high": px * 1.01,
                    "low": px * 0.98,
                    "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                    "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
                }
            )
    daily = pl.DataFrame(rows)
    mining_df, holdout_df, _ = split_holdout(daily, holdout_ratio=0.2)
    bundle = DataBundle.build(mining_df)

    s = AgentState(seed=1)
    for r in evaluate_expressions(["ts_mean(close,5)", "rank(vol)"], mining_df, bundle):
        s.attempts.append(
            AttemptRecord(
                0, "h", r["expression"], r["compile_ok"],
                r["ic_train"], False, None, r["error"],
            )
        )
    ledger = TrialLedger()
    s = node_guardrails(s, daily=mining_df, holdout_df=holdout_df, bundle=bundle, ledger=ledger, top_k=5)

    # ① N 诚实累加本轮评估数
    assert ledger.n_trials >= 1
    # ② 入选候选带 holdout 证据（DSR 门槛可能过滤掉全部，断言兼容 0 候选）
    for c in s.candidates:
        assert "holdout_ic" in c and "dsr" in c
    # ③ holdout 隔离：mining 段时间 < holdout 段时间
    assert mining_df["trade_date"].max() < holdout_df["trade_date"].min()


# ---------------------------------------------------------------------------
# node_guardrails 测试：family-aware 去冗余
# ---------------------------------------------------------------------------


def test_node_guardrails_family_aware_dedup():
    """高度相关的两个候选（ts_mean 5日 vs 6日）只有一个入选。"""
    import datetime as dt

    import numpy as np
    import polars as pl

    from factorzen.agents.evaluation import evaluate_expressions
    from factorzen.agents.nodes import node_guardrails
    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import split_holdout
    from factorzen.validation.multiple_testing import TrialLedger

    rng = np.random.default_rng(7)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < 180:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    # 50 stocks：compute_factor_correlation 需截面 >= 30
    codes = [f"{i:06d}.SZ" for i in range(50)]
    rows = []
    for c in codes:
        px = float(10.0 + rng.uniform(0, 90))  # 各股票不同基础价格
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append(
                {
                    "trade_date": dd,
                    "ts_code": c,
                    "close": px,
                    "open": px * 0.99,
                    "high": px * 1.01,
                    "low": px * 0.98,
                    "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                    "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
                }
            )
    daily = pl.DataFrame(rows)
    mining_df, holdout_df, _ = split_holdout(daily, holdout_ratio=0.2)
    bundle = DataBundle.build(mining_df)

    s = AgentState(seed=3)
    # ts_mean(close,5) 和 ts_mean(close,6) 是同族高度相关变体
    for r in evaluate_expressions(["ts_mean(close,5)", "ts_mean(close,6)"], mining_df, bundle):
        s.attempts.append(
            AttemptRecord(
                0, "h", r["expression"], r["compile_ok"],
                r.get("ic_train"), False, None, r.get("error"),
            )
        )

    ledger = TrialLedger()
    # dsr_threshold=0.0 绕过 DSR 门槛，专注测试 family-aware 去冗余逻辑
    s = node_guardrails(
        s, daily=mining_df, holdout_df=holdout_df, bundle=bundle, ledger=ledger,
        top_k=5, dsr_threshold=0.0,
    )

    # 两个高度相关候选只能入选 <= 1 个（family-aware 过滤同族冗余）
    assert len(s.candidates) <= 1, (
        f"Family-aware 去冗余失效：入选 {len(s.candidates)} 个候选，预期 <= 1"
    )


# ---------------------------------------------------------------------------
# node_guardrails 测试：N 诚实记账灵魂回归（多轮场景）
# ---------------------------------------------------------------------------


def test_node_guardrails_n_honest_accounting():
    """灵魂回归：2轮各评估2个不同表达式 → ledger.n_trials == 4（非三角和6）。

    修复前：passed 取全量 attempts，第2轮记 4 → n_trials=2+4=6（三角和）。
    修复后：passed 仅取本轮 iteration 的 attempts，每轮记2 → n_trials=2+2=4。
    """
    import datetime as dt

    import numpy as np
    import polars as pl

    from factorzen.agents.evaluation import evaluate_expressions
    from factorzen.agents.nodes import node_guardrails, node_reflect
    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import split_holdout
    from factorzen.validation.multiple_testing import TrialLedger

    rng = np.random.default_rng(42)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < 180:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(20)]
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
    daily = pl.DataFrame(rows)
    mining_df, holdout_df, _ = split_holdout(daily, holdout_ratio=0.2)
    bundle = DataBundle.build(mining_df)

    s = AgentState(seed=1)
    ledger = TrialLedger()

    # 轮0：2 个表达式，iteration=0
    exprs_r0 = ["ts_mean(close,5)", "rank(vol)"]
    results_r0 = evaluate_expressions(exprs_r0, mining_df, bundle)
    n_valid_r0 = 0
    for r in results_r0:
        s.attempts.append(AttemptRecord(
            0, "h", r["expression"], r["compile_ok"],
            r["ic_train"], False, None, r["error"], r.get("ir_train"),
        ))
        if r["compile_ok"] and r["ic_train"] is not None:
            n_valid_r0 += 1

    s = node_guardrails(s, daily=mining_df, holdout_df=holdout_df, bundle=bundle,
                        ledger=ledger, top_k=5)
    assert ledger.n_trials == n_valid_r0, (
        f"轮0后 n_trials={ledger.n_trials}，期望 {n_valid_r0}"
    )

    s = node_reflect(s)  # iteration → 1

    # 轮1：2 个不同表达式，iteration=1
    exprs_r1 = ["ts_mean(close,10)", "rank(amount)"]
    results_r1 = evaluate_expressions(exprs_r1, mining_df, bundle)
    n_valid_r1 = 0
    for r in results_r1:
        s.attempts.append(AttemptRecord(
            1, "h", r["expression"], r["compile_ok"],
            r["ic_train"], False, None, r["error"], r.get("ir_train"),
        ))
        if r["compile_ok"] and r["ic_train"] is not None:
            n_valid_r1 += 1

    s = node_guardrails(s, daily=mining_df, holdout_df=holdout_df, bundle=bundle,
                        ledger=ledger, top_k=5)

    expected_total = n_valid_r0 + n_valid_r1
    # 灵魂断言：N 为每轮独立记账之和，非三角和
    assert ledger.n_trials == expected_total, (
        f"N 诚实记账失败：ledger.n_trials={ledger.n_trials}，"
        f"期望 {expected_total}（轮0={n_valid_r0} + 轮1={n_valid_r1}）。"
        f"若为三角和应得 {n_valid_r0 + (n_valid_r0 + n_valid_r1)}，说明 Fix1 未生效。"
    )

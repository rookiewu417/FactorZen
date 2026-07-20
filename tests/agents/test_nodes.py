"""
test_nodes_gen.py：合并自 agents 相关碎片测试（test_nodes_gen.py）。
test_nodes_eval.py：验收侧节点测试：node_guardrails / node_critic / node_reflect。
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from factorzen.agents.nodes import node_critic, node_evaluate, node_generate, node_reflect
from factorzen.agents.roles.coder import decompose_tasks
from factorzen.agents.roles.hypothesis import format_structured, propose_structured
from factorzen.agents.self_heal import heal_expressions
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.discovery.expression import parse_expr
from factorzen.discovery.scoring import DataBundle


# ==== 来自 test_nodes_gen.py ====
# ==== 来自 test_agent_nodes_gen.py ====
class FakeLLM__nodes_gen:
    def __init__(self, responses):
        self._r = list(responses)
    def __call__(self, messages):
        return self._r.pop(0) if self._r else "{}"


def _mock_daily__nodes_gen(n_stocks=40, n_days=120, seed=1):
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


def test_node_generate_eval_suite():
    """test_node_generate_then_evaluate_populates_attempts；eval_start 已设但漏传 warmup_daily → 必须出声，而不是静默裸求值。；test_node_generate_rejects_illegal_and_records_error；验证去重用归一化形式：原始 vs 归一化两种写法不重复进 attempts。"""
    # -- 原 test_node_generate_then_evaluate_populates_attempts --
    def _section_0_test_node_generate_then_evaluate_populates_attempts():
        daily = _mock_daily__nodes_gen()
        bundle = DataBundle.build(daily)
        raw = json.dumps({"hypothesis": "动量", "expressions": ["ts_mean(close,5)", "rank(vol)"],
                          "rationale": "r"})
        # semantic_check 也走 llm：两次 consistent=true
        sem = json.dumps({"consistent": True, "reason": "ok"})
        llm = FakeLLM__nodes_gen([raw, sem, sem])
        state = AgentState(seed=42)
        state = node_generate(state, llm, daily=daily, bundle=bundle)
        state = node_evaluate(state, daily=daily, bundle=bundle)
        assert len(state.attempts) == 2
        assert all(a.compile_ok for a in state.attempts)
        assert all(a.ic_train is not None for a in state.attempts)
        # 验证归一化形式（带空格）在 seen_expressions 中
        assert "ts_mean(close, 5)" in state.seen_expressions

    _section_0_test_node_generate_then_evaluate_populates_attempts()

    # -- 原 test_node_evaluate_raises_when_eval_start_set_without_warmup_daily --
    def _section_1_test_node_evaluate_raises_when_eval_start_set_without_warmup_daily():
        daily = _mock_daily__nodes_gen()
        bundle = DataBundle.build(daily)
        raw = json.dumps({"hypothesis": "动量", "expressions": ["ts_mean(close,5)"], "rationale": "r"})
        sem = json.dumps({"consistent": True, "reason": "ok"})
        llm = FakeLLM__nodes_gen([raw, sem])
        state = AgentState(seed=1)
        state = node_generate(state, llm, daily=daily, bundle=bundle)

        with pytest.raises(ValueError, match="warmup_daily"):
            node_evaluate(state, daily=daily, bundle=bundle,
                          eval_start=dt.date(2022, 2, 1), warmup_daily=None)

    _section_1_test_node_evaluate_raises_when_eval_start_set_without_warmup_daily()

    # -- 原 test_node_generate_rejects_illegal_and_records_error --
    def _section_2_test_node_generate_rejects_illegal_and_records_error():
        daily = _mock_daily__nodes_gen()
        bundle = DataBundle.build(daily)
        raw = json.dumps({"hypothesis": "h", "expressions": ["bogus_op(close)"], "rationale": "r"})
        sem = json.dumps({"consistent": True, "reason": "ok"})
        llm = FakeLLM__nodes_gen([raw, sem])
        state = AgentState(seed=1)
        state = node_generate(state, llm, daily=daily, bundle=bundle)
        state = node_evaluate(state, daily=daily, bundle=bundle)
        assert state.attempts[0].compile_ok is False and state.attempts[0].error

    _section_2_test_node_generate_rejects_illegal_and_records_error()

    # -- 原 test_node_generate_dedup_with_normalized_form --
    def _section_3_test_node_generate_dedup_with_normalized_form():
        daily = _mock_daily__nodes_gen()
        bundle = DataBundle.build(daily)
        # 第一轮：无空格形式
        raw1 = json.dumps({"hypothesis": "h1", "expressions": ["ts_mean(close,5)"], "rationale": "r"})
        sem1 = json.dumps({"consistent": True, "reason": "ok"})
        # 第二轮：有空格形式（归一化后相同）
        raw2 = json.dumps({"hypothesis": "h2", "expressions": ["ts_mean(close, 5)"], "rationale": "r"})
        sem2 = json.dumps({"consistent": True, "reason": "ok"})
        llm = FakeLLM__nodes_gen([raw1, sem1, raw2, sem2])
        state = AgentState(seed=42)
        # 第一轮
        state = node_generate(state, llm, daily=daily, bundle=bundle)
        state = node_evaluate(state, daily=daily, bundle=bundle)
        assert len(state.attempts) == 1
        assert "ts_mean(close, 5)" in state.seen_expressions
        # 第二轮：同一表达式的不同写法应被去重
        state = node_generate(state, llm, daily=daily, bundle=bundle)
        assert len(state._pending) == 0  # type: ignore[attr-defined]
        state = node_evaluate(state, daily=daily, bundle=bundle)
        assert len(state.attempts) == 1  # 仍为 1，没有增加

    _section_3_test_node_generate_dedup_with_normalized_form()


# ==== 来自 test_agent_structured.py ====
def test_structured_propose_suite(tmp_path):
    """test_propose_structured_returns_four_fields；test_propose_structured_skips_malformed；test_format_structured_renders_fields；test_decompose_tasks_returns_tasks；structured=True 走结构化假设路径，闭环完成不崩。"""
    # -- 原 test_propose_structured_returns_four_fields --
    def _section_0_test_propose_structured_returns_four_fields():
        def fake(_m):
            return json.dumps({"hypotheses": [{"direction": "高动量", "mechanism": "趋势延续",
                                               "expected_sign": 1, "falsification": "IC<0则证伪"}]})
        out = propose_structured(fake, known_invalid=[], known_valid=[])
        assert len(out) == 1
        for k in ["direction", "mechanism", "expected_sign", "falsification"]:
            assert k in out[0]
        assert out[0]["expected_sign"] == 1

    _section_0_test_propose_structured_returns_four_fields()

    # -- 原 test_propose_structured_skips_malformed --
    def _section_1_test_propose_structured_skips_malformed():
        def fake(_m):
            return json.dumps({"hypotheses": [{"no_direction": "x"}, {"direction": "ok"}]})
        out = propose_structured(fake, known_invalid=[], known_valid=[])
        assert len(out) == 1 and out[0]["direction"] == "ok"

    _section_1_test_propose_structured_skips_malformed()

    # -- 原 test_format_structured_renders_fields --
    def _section_2_test_format_structured_renders_fields():
        h = {"direction": "高动量", "mechanism": "趋势延续", "expected_sign": 1, "falsification": "IC<0"}
        txt = format_structured(h)
        assert "高动量" in txt and "机制" in txt and "证伪" in txt

    _section_2_test_format_structured_renders_fields()

    # -- 原 test_decompose_tasks_returns_tasks --
    def _section_3_test_decompose_tasks_returns_tasks():
        def fake(_m):
            return json.dumps({"tasks": [{"name": "mom20", "description": "20日动量", "rationale": "趋势"}]})
        tasks = decompose_tasks("高动量", fake)
        assert len(tasks) == 1
        assert tasks[0]["name"] == "mom20" and tasks[0]["rationale"] == "趋势"

    _section_3_test_decompose_tasks_returns_tasks()

    # -- 原 test_team_structured_opt_in_closes_loop --
    def _section_4_test_team_structured_opt_in_closes_loop(tmp_path):
        from factorzen.agents.team_orchestrator import run_team_agent
        seq = [json.dumps({"hypotheses": [{"direction": "动量", "mechanism": "m",
                                           "expected_sign": 1, "falsification": "f"}]}),
               json.dumps({"expressions": ["ts_mean(close,5)"]}),
               json.dumps({"verdict": "keep", "reason": "ok"})] * 20
        i = {"k": 0}

        def fn(_m):
            v = seq[i["k"] % len(seq)]
            i["k"] += 1
            return v
        res = run_team_agent(_mock_daily__structured(), fn, n_rounds=1, seed=1,
                             index_path=str(tmp_path / "e.jsonl"), structured=True)
        assert res.state.iteration == 1

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_team_structured_opt_in_closes_loop(_tp4)


def _mock_daily__structured():
    rng = np.random.default_rng(1)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < 180:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{i:06d}.SZ" for i in range(20)]:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


# ==== 来自 test_agent_self_heal.py ====
def test_heal_expressions_suite():
    """语法错（非未知算子）→ 报错回灌 LLM 修正 → 产出可解析表达式。；可解析表达式不触发 LLM（零额外成本）。；LLM 持续产语法错 → max_rounds 耗尽后丢弃（不死循环）。；有效 + 语法错混合：有效直通，语法错修正，结果去重。；heal_rounds=0 → 关闭自愈：非法表达式原样进入 pending，不触发 revise LLM 调用。"""
    # -- 原 test_heal_fixes_parse_error --
    def _section_0_test_heal_fixes_parse_error():
        def fake(_msgs):
            return json.dumps({"expressions": ["ts_mean(close, 5)"]})
        healed = heal_expressions(["ts_mean()"], "动量", fake, max_rounds=2)
        assert len(healed) >= 1
        for h in healed:
            parse_expr(h)  # 全部可解析

    _section_0_test_heal_fixes_parse_error()

    # -- 原 test_heal_valid_expr_no_llm_call --
    def _section_1_test_heal_valid_expr_no_llm_call():
        def fake(_msgs):
            raise AssertionError("valid expr 不应触发 LLM 修正")
        healed = heal_expressions(["ts_mean(close, 5)"], "h", fake, max_rounds=2)
        assert len(healed) == 1
        parse_expr(healed[0])

    _section_1_test_heal_valid_expr_no_llm_call()

    # -- 原 test_heal_gives_up_after_max_rounds --
    def _section_2_test_heal_gives_up_after_max_rounds():
        def fake(_msgs):
            return json.dumps({"expressions": ["add(close)"]})
        healed = heal_expressions(["add(close)"], "h", fake, max_rounds=2)
        assert healed == []

    _section_2_test_heal_gives_up_after_max_rounds()

    # -- 原 test_heal_dedup_and_mixed --
    def _section_3_test_heal_dedup_and_mixed():
        def fake(_msgs):
            return json.dumps({"expressions": ["rank(vol)"]})
        healed = heal_expressions(["ts_mean(close, 5)", "ts_mean()"], "h", fake, max_rounds=2)
        assert len(healed) == len(set(healed))
        for h in healed:
            parse_expr(h)

    _section_3_test_heal_dedup_and_mixed()

    # -- 原 test_node_generate_heal_rounds_zero_disables_healing --
    def _section_4_test_node_generate_heal_rounds_zero_disables_healing():
        import datetime as dt

        import numpy as np
        import polars as pl

        from factorzen.agents.nodes import node_generate
        from factorzen.agents.state import AgentState
        from factorzen.discovery.scoring import DataBundle

        rng = np.random.default_rng(1)
        days, d = [], dt.date(2022, 1, 3)
        while len(days) < 90:
            if d.weekday() < 5:
                days.append(d)
            d += dt.timedelta(days=1)
        rows = []
        for c in [f"{i:06d}.SZ" for i in range(6)]:
            px = 10.0
            for dd in days:
                px *= 1 + rng.standard_normal() * 0.02
                rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                             "high": px * 1.01, "low": px * 0.98, "vol": 1e6, "amount": 1e7})
        daily = pl.DataFrame(rows)

        calls: list[list] = []
        seq = [json.dumps({"hypothesis": "h", "expressions": ["bad("], "rationale": "r"}),
               json.dumps({"consistent": True, "reason": "ok"})]

        def fn(msgs):
            calls.append(msgs)
            return seq[min(len(calls) - 1, len(seq) - 1)]

        state = node_generate(AgentState(seed=1), fn, daily=daily,
                              bundle=DataBundle.build(daily), heal_rounds=0)

        assert len(calls) == 2, f"应只有 proposal + semantic_check 两次调用，实得 {len(calls)}"
        assert [p.expression for p in state._pending] == ["bad("]

    _section_4_test_node_generate_heal_rounds_zero_disables_healing()


# ==== 来自 test_nodes_eval.py ====
class FakeLLM__nodes_eval:
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


def test_node_critic_reflect_suite():
    """test_node_critic_marks_verdict；已有 verdict 的 attempt 不再调用 llm。；test_node_reflect_feeds_low_ic_to_negatives；test_node_reflect_increments_iteration"""
    # -- 原 test_node_critic_marks_verdict --
    def _section_0_test_node_critic_marks_verdict():
        s = _state_with_attempts()
        llm = FakeLLM__nodes_eval(
            [
                json.dumps({"verdict": "keep", "reason": "经济直觉成立"}),
                json.dumps({"verdict": "drop", "reason": "疑似数据窥探"}),
            ]
        )
        s = node_critic(s, llm)
        verdicts = [a.critic_verdict for a in s.attempts]
        assert "keep" in verdicts and "drop" in verdicts

    _section_0_test_node_critic_marks_verdict()

    # -- 原 test_node_critic_skips_already_judged --
    def _section_1_test_node_critic_skips_already_judged():
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

    _section_1_test_node_critic_skips_already_judged()

    # -- 原 test_node_reflect_feeds_low_ic_to_negatives --
    def _section_2_test_node_reflect_feeds_low_ic_to_negatives():
        s = _state_with_attempts()
        s = node_reflect(s, ic_threshold=0.01)
        # 低 IC 的 rank(vol) 进负例库，高 IC 的不进
        assert "rank(vol)" in s.negative_examples
        assert "ts_mean(close,5)" not in s.negative_examples

    _section_2_test_node_reflect_feeds_low_ic_to_negatives()

    # -- 原 test_node_reflect_increments_iteration --
    def _section_3_test_node_reflect_increments_iteration():
        s = _state_with_attempts()
        assert s.iteration == 0
        s = node_reflect(s, ic_threshold=0.01)
        assert s.iteration == 1

    _section_3_test_node_reflect_increments_iteration()


# ---------------------------------------------------------------------------
# node_reflect 测试
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# node_guardrails 测试：N 记账 + holdout 隔离
# ---------------------------------------------------------------------------


def test_node_guardrails_eval_suite():
    """① ledger.n_trials 累加；② 候选带 holdout_ic/dsr；③ holdout 隔离。；高度相关的两个候选（ts_mean 5日 vs 6日）只有一个入选。；#128：强负 IC(做空)因子——train 段按 abs(ic) 排序纳入 top_k，DSR 须用 abs(ir_train)。；#135：train 正 IC 过 DSR，但 holdout IC 反号 → OOS 护栏须拒(不能只查 NaN)。；灵魂回归：2轮各评估2个不同表达式 → ledger.n_trials == 4（非三角和6）。"""
    # -- 原 test_node_guardrails_n_accounting_and_holdout_isolation --
    def _section_0_test_node_guardrails_n_accounting_and_holdout_isolation():
        import datetime as dt

        import numpy as np
        import polars as pl

        from factorzen.agents.nodes import node_guardrails
        from factorzen.agents.state import AgentState, AttemptRecord
        from factorzen.discovery.evaluation import evaluate_expressions
        from factorzen.discovery.scoring import DataBundle
        from factorzen.validation.holdout import split_holdout
        from factorzen.validation.multiple_testing import TrialLedger

        rng = np.random.default_rng(3)
        days, d = [], dt.date(2022, 1, 3)
        while len(days) < 180:
            if d.weekday() < 5:
                days.append(d)
            d += dt.timedelta(days=1)
        codes = [f"{i:06d}.SZ" for i in range(40)]
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

    _section_0_test_node_guardrails_n_accounting_and_holdout_isolation()

    # -- 原 test_node_guardrails_family_aware_dedup --
    def _section_1_test_node_guardrails_family_aware_dedup():
        import datetime as dt

        import numpy as np
        import polars as pl

        from factorzen.agents.nodes import node_guardrails
        from factorzen.agents.state import AgentState, AttemptRecord
        from factorzen.discovery.evaluation import evaluate_expressions
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
        # dsr_alpha=1.0 绕过 DSR 门槛(pval<1.0 恒真)，专注测试 family-aware 去冗余逻辑
        s = node_guardrails(
            s, daily=mining_df, holdout_df=holdout_df, bundle=bundle, ledger=ledger,
            top_k=5, dsr_alpha=1.0,
        )

        # 两个高度相关候选只能入选 <= 1 个（family-aware 过滤同族冗余）
        assert len(s.candidates) <= 1, (
            f"Family-aware 去冗余失效：入选 {len(s.candidates)} 个候选，预期 <= 1"
        )

    _section_1_test_node_guardrails_family_aware_dedup()

    # -- 原 test_node_guardrails_admits_bidirectional_negative_ic --
    def _section_2_test_node_guardrails_admits_bidirectional_negative_ic():
        from factorzen.agents.nodes import node_guardrails
        from factorzen.agents.state import AgentState, AttemptRecord
        from factorzen.discovery.scoring import DataBundle
        from factorzen.validation.holdout import split_holdout
        from factorzen.validation.multiple_testing import TrialLedger

        daily = _controlled_daily(holdout_sign=-1.0)  # holdout IC(amount) ≡ -1
        mining_df, holdout_df, _ = split_holdout(daily, holdout_ratio=0.2)
        bundle = DataBundle.build(mining_df)

        s = AgentState(seed=1)
        # 注入：强负 train IC + 负 ir_train（做空因子）；holdout 同向为负 → OOS 一致
        # n_train = 该因子 train 段有效 IC 天数（DSR 的 n_obs）；单元素 IR 池 → sharpe_var 退化为 1.0
        s.attempts.append(
            AttemptRecord(0, "h", "amount", True, -0.06, False, None, None, -2.5, n_train=100)
        )
        ledger = TrialLedger()
        node_guardrails(s, daily=mining_df, holdout_df=holdout_df, bundle=bundle,
                        ledger=ledger, top_k=5, dsr_alpha=0.05)

        assert len(s.candidates) == 1, (
            f"强负 IC 做空因子应过关(abs(ir) 喂 DSR + holdout 同向负一致)，实得 "
            f"{len(s.candidates)} —— signed ir_train 喂 DSR 会把负 IC 因子 DSR 压到 ≈0 误杀"
        )

    _section_2_test_node_guardrails_admits_bidirectional_negative_ic()

    # -- 原 test_node_guardrails_rejects_holdout_sign_flip --
    def _section_3_test_node_guardrails_rejects_holdout_sign_flip():
        from factorzen.agents.nodes import node_guardrails
        from factorzen.agents.state import AgentState, AttemptRecord
        from factorzen.discovery.scoring import DataBundle
        from factorzen.validation.holdout import split_holdout
        from factorzen.validation.multiple_testing import TrialLedger

        daily = _controlled_daily(holdout_sign=-1.0)  # holdout IC(amount) ≡ -1（与正 train 反号）
        mining_df, holdout_df, _ = split_holdout(daily, holdout_ratio=0.2)
        bundle = DataBundle.build(mining_df)

        s = AgentState(seed=1)
        # 注入：正 train IC/IR（过 DSR），但 holdout 反号 → 过拟合，OOS 应拒
        s.attempts.append(AttemptRecord(0, "h", "amount", True, 0.06, False, None, None, 2.5))
        ledger = TrialLedger()
        node_guardrails(s, daily=mining_df, holdout_df=holdout_df, bundle=bundle,
                        ledger=ledger, top_k=5, dsr_alpha=0.05)

        assert len(s.candidates) == 0, (
            f"train 正、holdout 反号的过拟合因子应被 OOS 护栏拒，实得 {len(s.candidates)} "
            f"—— holdout 只查 NaN 时反号候选照样入选，OOS 隔离形同虚设"
        )

    _section_3_test_node_guardrails_rejects_holdout_sign_flip()

    # -- 原 test_node_guardrails_n_honest_accounting --
    def _section_4_test_node_guardrails_n_honest_accounting():
        import datetime as dt

        import numpy as np
        import polars as pl

        from factorzen.agents.nodes import node_guardrails, node_reflect
        from factorzen.agents.state import AgentState, AttemptRecord
        from factorzen.discovery.evaluation import evaluate_expressions
        from factorzen.discovery.scoring import DataBundle
        from factorzen.validation.holdout import split_holdout
        from factorzen.validation.multiple_testing import TrialLedger

        rng = np.random.default_rng(42)
        days, d = [], dt.date(2022, 1, 3)
        while len(days) < 180:
            if d.weekday() < 5:
                days.append(d)
            d += dt.timedelta(days=1)
        codes = [f"{i:06d}.SZ" for i in range(40)]
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

    _section_4_test_node_guardrails_n_honest_accounting()


# ---------------------------------------------------------------------------
# node_guardrails 测试：family-aware 去冗余
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# node_guardrails 测试：N 诚实记账灵魂回归（多轮场景）
# ---------------------------------------------------------------------------


def _controlled_daily(holdout_sign: float, n_stocks: int = 30, n_days: int = 400,
                      eps: float = 0.001):
    """构造 holdout 段 IC(amount) 符号确定的数据（供 OOS 护栏/双向 DSR 测试）。

    每只股票 amount 截面严格单调（= 1e6 + i·1000），holdout 段每股日收益 = holdout_sign·eps·i。
    → 因子 `amount` 与 holdout 段前向收益完全（反）单调 → holdout rank IC ≡ holdout_sign，
    bootstrap CI = (holdout_sign, holdout_sign)（离 0）。mining 段收益固定正序（不影响注入的
    train IC/IR——node_guardrails 只读 AttemptRecord 的注入值，不重算 train IC）。

    n_days 默认 400：holdout_ratio=0.2 时 holdout 约 80 个交易日，满足
    DEFAULT_HOLDOUT_MIN_DAYS=60 的覆盖门（旧默认 180 只有 ~36 日，会被覆盖门误杀）。
    """
    import datetime as dt

    import polars as pl

    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    cut = int(len(days) * 0.8)  # 与 split_holdout(holdout_ratio=0.2) 一致
    rows = []
    for i in range(n_stocks):
        amt = float(1_000_000 + i * 1000)  # 截面严格单调 → rank(amount)=i
        px = 100.0
        for di, dd in enumerate(days):
            sign = holdout_sign if di >= cut else 1.0
            px *= 1.0 + sign * eps * i  # 高 i → 高/低收益(按 sign) → 与 amount 同/反序
            rows.append({
                "trade_date": dd, "ts_code": f"{i:06d}.SZ", "close": px,
                "open": px, "high": px, "low": px, "vol": amt, "amount": amt,
            })
    return pl.DataFrame(rows)



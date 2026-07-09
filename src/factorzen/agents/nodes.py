"""Agent 闭环的函数式节点：node(State) -> State。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from factorzen.agents.evaluation import evaluate_expressions, make_health_check
from factorzen.agents.memory import negative_recall
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.discovery.expression import parse_expr, to_expr_string
from factorzen.discovery.operators import LEAF_FEATURES, OPERATORS
from factorzen.llm.generation import (
    LLMFn,
    build_agent_messages,
    generate_factor_proposal,
    semantic_check,
)


@dataclass
class _PendingExpr:
    hypothesis: str
    expression: str


@dataclass
class AgentContext:
    op_names: list[str] = field(default_factory=lambda: list(OPERATORS.keys()))
    leaf_names: list[str] = field(default_factory=lambda: list(LEAF_FEATURES.keys()))


def node_generate(state: AgentState, llm_fn: LLMFn, *, daily, bundle,
                  n_hypotheses: int = 1, feedback: str = "", heal_rounds: int = 0) -> AgentState:
    """生成假设+表达式 → 语义对齐自检 → 暂存待评估（compile/eval 在 node_evaluate）。"""
    ctx = AgentContext()
    msgs = build_agent_messages(ctx.op_names, ctx.leaf_names, feedback, state.negative_examples)
    proposals = generate_factor_proposal(msgs, llm_fn, n_hypotheses=n_hypotheses)
    pending: list[_PendingExpr] = []
    # 求值层诊断器只建一次（预处理较重）；heal_rounds=0 时不建，零开销
    health = make_health_check(daily) if heal_rounds > 0 else None
    for p in proposals:
        # 自愈：把解析报错**与求值诊断**（异常/因子值近乎全 null）回灌 Coder 修正
        # （heal_rounds>0 时启用，CoSTEER 轻量版）
        exprs = p.expressions
        if heal_rounds > 0:
            from factorzen.agents.self_heal import heal_expressions
            exprs = heal_expressions(p.expressions, p.hypothesis, llm_fn,
                                     max_rounds=heal_rounds, health_check=health)
        for expr in exprs:
            try:
                norm = to_expr_string(parse_expr(expr))
            except ValueError:
                norm = expr
            if norm in state.seen_expressions:
                continue
            ok, _reason = semantic_check(p.hypothesis, expr, llm_fn)
            if ok:
                pending.append(_PendingExpr(p.hypothesis, norm))
    state.__dict__.setdefault("_pending", [])
    state._pending = pending  # type: ignore[attr-defined]
    return state


def node_evaluate(state: AgentState, *, daily, bundle) -> AgentState:
    """对暂存表达式批量评估，写 AttemptRecord + 更新 seen。"""
    pending = getattr(state, "_pending", [])
    exprs = [p.expression for p in pending]
    results = evaluate_expressions(exprs, daily, bundle) if exprs else []
    for p, r in zip(pending, results, strict=True):
        state.attempts.append(AttemptRecord(
            iteration=state.iteration, hypothesis=p.hypothesis, expression=r["expression"],
            compile_ok=r["compile_ok"], ic_train=r["ic_train"], passed_guardrails=False,
            critic_verdict=None, error=r["error"], ir_train=r["ir_train"],
            turnover=r.get("turnover"), n_train=r.get("n_train")))
        state.seen_expressions.add(r["expression"])
    state._pending = []  # type: ignore[attr-defined]
    return state


def node_guardrails(
    state: AgentState,
    *,
    daily,
    holdout_df,
    bundle,
    ledger,
    top_k: int = 5,
    dsr_alpha: float = 0.05,
    warmup_daily=None,
) -> AgentState:
    """对过编译的候选记账 N、跑 holdout_ic/DSR，过关者进 candidates。

    ``warmup_daily``：含 mining + holdout 的**完整帧**。holdout 段的因子值在它上面求值、
    再裁剪到 ``>= holdout_start``（扩窗预热）。否则滚动算子在 holdout 边界只有截断窗口，
    发出的偏差值直接进 holdout_ic/CI，扭曲护栏验收。PIT 安全：mining 段整体早于 holdout，
    时序算子只向过去看。缺省 None → 退回旧行为（仅供不便传完整帧的调用方，会有边界偏差）。

    passed 判定委托 discovery.guardrails.guardrail_passed（DSR p 值口径），与 M1 统一，
    消除双路径漂移（旧 dsr>0.5 松约 10 倍，收紧到 pval<dsr_alpha）。池级 PBO 记入 state.pbo。

    DSR 的三个入参都与 M1（mining_session.py:292-307）同口径，否则 deflation 基准不自洽：
    - ``sharpe_variance`` = trial 池 signed IR 的**经验方差**，而非 deflated_sharpe 的 H0
      默认 ``1/n_obs``。因 ``expected_max_sharpe ∝ sqrt(sharpe_variance)`` 而多样化 trial 池
      的经验方差恒大于 ``1/n_obs``，用默认值会让 deflation 基准系统性偏小 → 放行 M1 拒绝的因子
      （实测漂移 ``sqrt(var_emp × n_obs)`` 倍）。
    - ``n_trials`` 与该方差**同源**（同一批 trial）：都取「评估过且拿到有效 IR」的 attempts。
    - ``n_obs`` = 该因子自己的 train 段有效 IC 天数 ``a.n_train``，不是 train 段日历交易日数
      （后者更大，会系统性放大显著性）。
    """
    from factorzen.agents.evaluation import _node_to_factor_df
    from factorzen.discovery.guardrails import (
        DeflationBasis,
        deflated_pvalue,
        guardrail_passed,
        pool_pbo,
    )
    from factorzen.discovery.scoring import max_correlation
    from factorzen.validation.holdout import holdout_ic

    passed = [a for a in state.attempts
              if a.iteration == state.iteration and a.compile_ok and a.ic_train is not None]
    ledger.record(len(passed))
    passed.sort(key=lambda a: abs(a.ic_train or 0.0), reverse=True)

    # DSR 的 N 与 sharpe_variance 同源：跨轮累积的「评估过且有有效 IR」的 signed IR 池。
    # 与 M1 共用 DeflationBasis 这一份配方（架构守卫测试禁止绕过它直接调 deflated_sharpe）。
    # ledger.n_trials 是逐轮 len(passed) 之和，与 basis.n_trials 等长
    # （ic_train 与 ir_train 同时为 None）。
    basis = DeflationBasis.from_ir_pool([a.ir_train for a in state.attempts if a.compile_ok])

    # holdout 段扩窗预热：在完整帧上求值、裁剪到 >= holdout_start。
    # 只喂 holdout_df 会让滚动算子在边界用截断窗口，发出偏差值。
    if warmup_daily is not None:
        _hold_frame, _hold_start = warmup_daily, holdout_df["trade_date"].min()
    else:
        _hold_frame, _hold_start = holdout_df, None

    def _holdout_values(node):
        return _node_to_factor_df(node, _hold_frame, eval_start=_hold_start)

    existing_exprs: set[str] = {c["expression"] for c in state.candidates}

    pool: dict = {}
    for i, c in enumerate(state.candidates):
        try:
            pool[f"prev_{i}"] = _holdout_values(parse_expr(c["expression"]))
        except Exception:
            continue

    for a in passed[:top_k]:
        if a.expression in existing_exprs:
            continue
        try:
            node = parse_expr(a.expression)
            fdf_hold = _holdout_values(node)
            ic_h, ir_h, (ci_lo, ci_hi) = holdout_ic(fdf_hold, holdout_df)
            sharpe = abs(a.ir_train) if a.ir_train is not None else abs(a.ic_train or 0.0)
            dsr, pval = deflated_pvalue(sharpe, basis, a.n_train or 0)
            ic_tr = a.ic_train or 0.0
            if guardrail_passed(
                ic_train=ic_tr, holdout_ic=ic_h, dsr_pvalue=pval,
                ci_low=ci_lo, ci_high=ci_hi, dsr_alpha=dsr_alpha,
            ):
                # 事实先落定：过了定量护栏。去相关剔除是随后的**决策**，不得改写它——
                # 否则该因子会以 passed=False 落进 known_invalid，被当作「已验证无效」
                # 喂给 LLM（它其实过了全部定量护栏，只是与已有候选重复）。
                a.passed_guardrails = True
                corr = max_correlation(fdf_hold, pool)
                if corr > 0.7:
                    a.decorrelated = True
                    continue
                pool[a.expression] = fdf_hold
                existing_exprs.add(a.expression)
                state.candidates.append({
                    "expression": a.expression,
                    "hypothesis": a.hypothesis,
                    "ic_train": a.ic_train,
                    "ir_train": a.ir_train,
                    "turnover": a.turnover,
                    "holdout_ic": ic_h,
                    "holdout_ir": ir_h,
                    "dsr": dsr,
                    "dsr_pvalue": pval,
                })
        except Exception:
            continue

    try:
        cand_fdfs = [
            _node_to_factor_df(parse_expr(c["expression"]), daily) for c in state.candidates
        ]
        state.pbo = pool_pbo(cand_fdfs, bundle.fwd_returns)
    except Exception:
        state.pbo = float("nan")
    return state


def node_critic(state: AgentState, llm_fn: LLMFn) -> AgentState:
    """LLM 以风控审计员身份批判每个候选：keep/drop/mutate。"""
    for a in state.attempts:
        if a.critic_verdict is not None:
            continue
        msgs = [
            {"role": "system", "content": (
                "你是风控审计员，判断因子是否过拟合/经济直觉是否成立。"
                "注意：换手率高意味着交易成本侵蚀，train_IC 高但换手率高的因子未必可实现"
                "超额收益（成本双杀）；结合 ICIR（信息比率，越高越稳定）综合判断。"
                '只输出 JSON: {"verdict":"keep"|"drop"|"mutate","reason":"..."}')},
            {"role": "user", "content": (
                f"假设:{a.hypothesis} 表达式:{a.expression} "
                f"train_IC:{a.ic_train} ICIR:{a.ir_train} "
                f"换手率(单边,成本代理):{a.turnover} 过护栏:{a.passed_guardrails}")},
        ]
        try:
            obj = json.loads(llm_fn(msgs))
            a.critic_verdict = str(obj.get("verdict", "keep"))
        except Exception:
            a.critic_verdict = "keep"
    return state


def node_reflect(state: AgentState, *, ic_threshold: float = 0.01) -> AgentState:
    """更新 Negative RAG 负例库 + 推进迭代计数。"""
    seen = [(a.expression, a.ic_train) for a in state.attempts if a.ic_train is not None]
    state.negative_examples = negative_recall(seen, k=5, ic_threshold=ic_threshold)
    state.iteration += 1
    return state

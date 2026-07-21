"""Agent 闭环的函数式节点：node(State) -> State。"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from factorzen.agents.memory import negative_recall
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.discovery.evaluation import evaluate_expressions, make_health_check
from factorzen.discovery.expression import parse_expr, to_expr_string
from factorzen.discovery.guardrails import (
    DEFAULT_DSR_ALPHA,
    DEFAULT_GATE,
    DEFAULT_RESIDUAL_IC_FLOOR,
)
from factorzen.discovery.operators import LEAF_FEATURES, OPERATORS
from factorzen.llm.generation import (
    LLMFn,
    _extract_json,
    build_agent_messages,
    generate_factor_proposal,
    semantic_check,
)

_LOG = logging.getLogger(__name__)


@dataclass
class _PendingExpr:
    hypothesis: str
    expression: str


def _print_rejections(label: str, state: AgentState) -> None:
    """打印本轮被护栏拒（含去相关剔除）的候选 + 原因，供挖掘过程实时展示「为什么没过护栏」。

    ``reject_reason`` 由 `node_guardrails` 记；只取**本轮**（`iteration == state.iteration`）
    的 attempt，避免把往轮已展示过的重复打印。挖掘由 CLI 触发，故用 print 直出终端。
    """
    for a in state.attempts:
        if a.iteration == state.iteration and a.reject_reason:
            print(f"[{label}]     ✗ {a.expression} → {a.reject_reason}", flush=True)


@dataclass
class AgentContext:
    """生成侧的市场上下文：算子集（市场无关）+ 叶子集/映射 + 市场名。

    默认构造 = A 股（op=全算子、leaf=`LEAF_FEATURES`、market="ashare"、leaf_map=None → 求值走
    A 股默认列映射）。crypto 等经 `from_profile` 构造。
    """
    op_names: list[str] = field(default_factory=lambda: list(OPERATORS.keys()))
    leaf_names: list[str] = field(default_factory=lambda: list(LEAF_FEATURES.keys()))
    market: str = "ashare"
    leaf_map: dict[str, str] | None = None

    @classmethod
    def from_profile(cls, profile) -> AgentContext:
        """从 MarketProfile 构造。``profile=None`` → A 股默认（零回归）。

        op_names 恒取全算子（算子市场无关）；leaf_names/leaf_map 取
        `profile.factors.leaf_features()`；market 取 `profile.name`。
        """
        if profile is None:
            return cls()
        lm = profile.factors.leaf_features()
        return cls(op_names=list(OPERATORS.keys()), leaf_names=list(lm.keys()),
                   market=profile.name, leaf_map=lm)


def node_generate(state: AgentState, llm_fn: LLMFn, *, daily, bundle,
                  n_hypotheses: int = 1, feedback: str = "", heal_rounds: int = 0,
                  leaf_budgets: dict[str, int] | None = None, profile=None,
                  leaf_guidance: dict[str, list[str]] | None = None,
                  library_covered: list[str] | None = None,
                  lift_rejected: list[dict] | None = None,
                  library_crowded: list[tuple[str, int]] | None = None,
                  ctx: AgentContext | None = None) -> AgentState:
    """生成假设+表达式 → 语义对齐自检 → 暂存待评估（compile/eval 在 node_evaluate）。

    ``leaf_budgets``：短历史叶子的可用预热预算，透传给 `build_agent_messages` 提示 LLM
    别对短叶写超预热长窗口（默认 None → prompt 零回归）。

    ``leaf_guidance``：Librarian 叶子级挖穿/未探索，与 team Hypothesis 共用
    ``format_leaf_guidance`` 注入（默认 None → 不注入）。

    ``library_covered``：库内 active 高 IC 表达式，与 team Hypothesis 共用
    ``format_library_covered`` 注入（默认 None → 不注入）。

    ``lift_rejected``：组合层 lift 拒绝方向。参数为 M5/M6 对齐预留，接线待
    M5 引入 experiment_index；默认 None → 不注入（零回归）。

    ``library_crowded``：库内拥挤叶子；None → 不注入（零回归）。

    ``ctx``：调用方已构造的市场上下文（含 leaf_health 摘除后的存活叶）。默认 None 时
    从 ``profile`` 重建（旧调用方零回归）。**注意**：``run_llm_agent`` 开局摘叶后须
    传入同一 ``ctx``，否则 prompt 仍广告死叶。

    ``profile``：市场 profile（默认 None → A 股，零回归）。经 `AgentContext.from_profile`
    得叶子集/映射/市场名，透传给 prompt（market/leaf_names）、health_check、自愈、规范化
    （parse_expr 的 leaf_map）——crypto 表达式方能解析、且 `norm` 与 `evaluate_expressions`
    产出的规范 `seen_expressions` 对齐（否则 dedup 失配致 N over-count）。
    """
    if ctx is None:
        ctx = AgentContext.from_profile(profile)
    msgs = build_agent_messages(ctx.op_names, ctx.leaf_names, feedback,
                                state.negative_examples, leaf_budgets=leaf_budgets,
                                market=ctx.market, leaf_guidance=leaf_guidance,
                                library_covered=library_covered,
                                lift_rejected=lift_rejected,
                                library_crowded=library_crowded)
    proposals = generate_factor_proposal(msgs, llm_fn, n_hypotheses=n_hypotheses)
    pending: list[_PendingExpr] = []
    # 求值层诊断器只建一次（预处理较重）；heal_rounds=0 时不建，零开销
    health = make_health_check(daily, profile=profile, leaf_map=ctx.leaf_map) \
        if heal_rounds > 0 else None
    for p in proposals:
        # 自愈：把解析报错**与求值诊断**（异常/因子值近乎全 null）回灌 Coder 修正
        # （heal_rounds>0 时启用，CoSTEER 轻量版）
        exprs = p.expressions
        if heal_rounds > 0:
            from factorzen.agents.self_heal import heal_expressions
            exprs = heal_expressions(p.expressions, p.hypothesis, llm_fn,
                                     max_rounds=heal_rounds, health_check=health,
                                     leaf_map=ctx.leaf_map, market=ctx.market,
                                     leaf_names=ctx.leaf_names)
        for expr in exprs:
            try:
                norm = to_expr_string(parse_expr(expr, ctx.leaf_map))
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


def node_evaluate(state: AgentState, *, daily, bundle,
                  eval_start=None, eval_end=None, warmup_daily=None, profile=None,
                  leaf_budgets: dict[str, int] | None = None,
                  prepped=None) -> AgentState:
    """对暂存表达式批量评估，写 AttemptRecord + 更新 seen。

    ``eval_start``/``eval_end``：会话级 train 段边界（date，或 None）。**None-gating**：
    为 None（旧调用方默认）时走裸 `evaluate_expressions(exprs, daily, bundle)`，与之前
    逐字节相同、零回归；非 None 时改在 ``warmup_daily``（含预热前缀的完整帧）上求值，
    裁剪到 ``[eval_start, eval_end]``——门槛只挂在 `eval_start` 本身是否为 None，
    不能用 `daily`/`mining_df` 的起点判断（`eval_start=None` 时二者的起点相同，
    误用会让 `evaluate_expressions` 的预热门把可用预热样本数误判成 0）。

    ``eval_start`` 非 None 却漏传 ``warmup_daily`` 时**出声**（ValueError），不静默退回裸
    求值：那样会在已裁到 eval_start 的 ``daily`` 上求值，预热裁剪与预热门双双失效，段首
    截断窗口噪声（`operators._MIN = 3` 不产 NaN）灌回 train IC——与 `evaluate_expressions`
    里「eval_end 不能脱离 eval_start 单传」同一条异常契约（陷阱#7）。

    ``leaf_budgets``：短历史叶子可用预热预算；非空时评估前 ``clamp_window_literals``（W5b）。
    指纹去重走 ``state.seen_fingerprints``（session 级，W4）。

    ``prepped``：session 级已 prep 帧（可选）；透传 ``evaluate_expressions`` 跳过内部 prep。
    """
    pending = getattr(state, "_pending", [])
    exprs = [p.expression for p in pending]
    if exprs and leaf_budgets:
        from factorzen.discovery.expression import clamp_window_literals
        leaf_map = profile.factors.leaf_features() if profile is not None else None
        clamped: list[str] = []
        for e in exprs:
            ce, _did = clamp_window_literals(e, leaf_budgets, leaf_map)
            clamped.append(ce)
        exprs = clamped
        # 同步 pending 表达式（钳后串进评估与 AttemptRecord）
        for p, e in zip(pending, exprs, strict=True):
            p.expression = e  # type: ignore[misc]
    if not exprs:
        results = []
    elif eval_start is not None:
        if warmup_daily is None:
            raise ValueError(
                "eval_start 非 None 时必须提供 warmup_daily（含预热前缀的完整帧）："
                "否则会在已裁到 eval_start 的 daily 上裸求值，预热裁剪与预热门（warmup_bars）"
                "双双失效，静默把段首截断窗口噪声灌回 train IC。")
        results = evaluate_expressions(
            exprs, warmup_daily, bundle,
            eval_start=eval_start, eval_end=eval_end, profile=profile,
            seen_fingerprints=state.seen_fingerprints, prepped=prepped,
        )
    else:
        results = evaluate_expressions(
            exprs, daily, bundle, profile=profile,
            seen_fingerprints=state.seen_fingerprints, prepped=prepped,
        )
    for p, r in zip(pending, results, strict=True):
        state.attempts.append(AttemptRecord(
            iteration=state.iteration, hypothesis=p.hypothesis, expression=r["expression"],
            compile_ok=r["compile_ok"], ic_train=r["ic_train"], passed_guardrails=False,
            critic_verdict=None, error=r["error"], ir_train=r["ir_train"],
            turnover=r.get("turnover"), n_train=r.get("n_train"),
            nonzero_coverage=r.get("nonzero_coverage"),
            is_sparse=bool(r.get("is_sparse") or False),
            subset_ic_train=r.get("subset_ic_train"),
            subset_n_days_train=r.get("subset_n_days_train"),
        ))
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
    dsr_alpha: float = DEFAULT_DSR_ALPHA,
    gate: str = DEFAULT_GATE,
    warmup_daily=None,
    eval_start=None,
    profile=None,
    lib_pool: dict | Any | None = None,
    objective: str = "residual",
    residual_projector=None,
    prepped=None,
    exec_lag: int = 0,
    exec_price_col: str | None = None,
    sleeve_gate: bool = True,
) -> AgentState:
    """对过编译的候选记账 N、跑 holdout_ic/DSR，过关者进 candidates。

    ``profile``：市场 profile（默认 None → A 股，零回归）。透传到 holdout/PBO 段的预处理
    （`profile.factors.derived_columns`）与全部 `parse_expr`/求值的 leaf_map——护栏统计口径
    （DSR/holdout/PBO/去相关）本身市场无关，只有「用哪套派生列/叶子映射求因子值」随市场变。

    ``warmup_daily``：含 mining + holdout 的**完整帧**。holdout 段的因子值在它上面求值、
    再裁剪到 ``>= holdout_start``（扩窗预热）。否则滚动算子在 holdout 边界只有截断窗口，
    发出的偏差值直接进 holdout_ic/CI，扭曲护栏验收。PIT 安全：mining 段整体早于 holdout，
    时序算子只向过去看。缺省 None → 退回旧行为（仅供不便传完整帧的调用方，会有边界偏差）。

    ``eval_start``：会话级 train 段起点（date，或 None）。**只**用于池级 PBO 的 None-gating——
    为 None（旧调用方默认）时 PBO 池走裸 `_node_to_factor_df(node, daily)`（零回归）；
    非 None 时改在 ``warmup_daily`` 上求值、裁剪到 ``[eval_start, daily 的 train 段终点]``，
    与 `evaluate_expressions` 的扩窗预热同一理由。`_node_to_factor_df` 本身没有预热门
    （不会拒绝任何表达式），只是求值窗口的选择——不会引入本任务要修的『预热不足被拒』问题。

    入池判定委托 discovery.guardrails.acceptance_reasons（``gate`` 口径，默认 "library"：真+有
    信号，不含 DSR；"strict" 才用 DSR），与 M1 `_guard_passed` 统一，消除双路径漂移。DSR 仍算出来
    存进候选供组合层/报告用（只是 library 口径下不当门）。池级 PBO 记入 state.pbo。

    ``lib_pool``：库内 active 因子在**与 session 去相关同帧**上的物化面板
    （``build_library_pool`` 产出）。与之算 ``library_orthogonal_check``：
    corr > 0.95 → 硬拒 ``library_correlated``；corr ∈ (0.7, 0.95] 软 reason 挡快速通道、
    可入 lift 队列；corr ≤ 0.7 与旧行为一致。None/空 → 跳过（零回归）。

    ``objective``：``"residual"``（默认）在库非空时用对库残差 IC 做 library 门判定
    （floor=``DEFAULT_RESIDUAL_IC_FLOOR``），裸 IC 仍落盘对照；库空自动退化为 ``"raw"``
    （行为=现状）。``"raw"`` 强制裸 IC 门。

    ``residual_projector``：可选 ``ResidualProjector``（session 级预计算 QR）。residual
    模式下对**全部**本轮候选算 train 残差 IC 并按 ``|residual_ic_train|`` 选 top-K 槽
    （修泄漏 A：槽位键与目标错位）；缺省时本函数现场从 ``lib_panel`` 建一次。

    ``prepped``：session 级已 ``_preprocess_daily`` 的完整帧（可选）。非 None 时 holdout
    扩窗 / 池级 PBO 复用该帧，跳过对 ``warmup_daily`` 的再 prep（P5 峰值省一份全帧）。
    **契约**：须与 evaluate 同源、含预热前缀。train residual 仍对 ``daily``（mining 段）
    单独 prep——mining-only 边界语义（``ret_1d`` 段首 null）与历史数值零回归。

    DSR 的三个入参都与 M1（mining_session.py:292-307）同口径，否则 deflation 基准不自洽：
    - ``sharpe_variance`` = trial 池 signed IR 的**经验方差**，而非 deflated_sharpe 的 H0
      默认 ``1/n_obs``。因 ``expected_max_sharpe ∝ sqrt(sharpe_variance)`` 而多样化 trial 池
      的经验方差恒大于 ``1/n_obs``，用默认值会让 deflation 基准系统性偏小 → 放行 M1 拒绝的因子
      （实测漂移 ``sqrt(var_emp × n_obs)`` 倍）。
    - ``n_trials`` 与该方差**同源**（同一批 trial）：都取「评估过且拿到有效 IR」的 attempts。
    - ``n_obs`` = 该因子自己的 train 段有效 IC 天数 ``a.n_train``，不是 train 段日历交易日数
      （后者更大，会系统性放大显著性）。
    """
    from tqdm import tqdm

    from factorzen.discovery.evaluation import (
        _factor_df_from_prepped,
        _preprocess_daily,
        compute_subset_rank_ic,
    )
    from factorzen.discovery.guardrails import (
        DEFAULT_DUPLICATE_CORR,
        DEFAULT_GRAY_IC_FLOOR,
        REJECT_CATEGORY_LIBRARY_CORRELATED,
        REJECT_CATEGORY_LIFT_QUEUE,
        DeflationBasis,
        acceptance_reasons,
        classify_reject_category,
        deflated_pvalue,
        is_lift_queue_candidate,
        is_sleeve_lift_candidate,
        pool_pbo,
    )
    from factorzen.discovery.residual import (
        ResidualProjector,
        build_library_panel,
        compute_residual_ic,
        resolve_objective,
    )
    from factorzen.discovery.scoring import (
        DEFAULT_DECORR_THRESHOLD,
        build_library_corr_panel,
        library_orthogonal_check,
        max_correlation,
    )
    from factorzen.validation.holdout import holdout_fwd_returns, holdout_ic_result

    leaf_map = profile.factors.leaf_features() if profile is not None else None
    passed = [a for a in state.attempts
              if a.iteration == state.iteration and a.compile_ok and a.ic_train is not None]
    ledger.record(len(passed))

    # DSR 的 N 与 sharpe_variance 同源：跨轮累积的「评估过且有有效 IR」的 signed IR 池。
    # 与 M1 共用 DeflationBasis 这一份配方（架构守卫测试禁止绕过它直接调 deflated_sharpe）。
    # ledger.n_trials 是逐轮 len(passed) 之和，与 basis.n_trials 等长
    # （ic_train 与 ir_train 同时为 None）。
    # two_sided=True：本路径按 |residual_ic_train|（residual）或 |ic_train|（raw）排序且
    # `guardrail_passed` 经 `ci_high < 0` 分支接纳负 IC 反转因子 ⇒ 统计量是 max|IR|，
    # deflation 基准须按 2N 算。
    # （M1 的 fitness 用**带符号** tstat 降序，是单边搜索，反转因子以 neg(x) 形式出现，故用 N。）
    basis = DeflationBasis.from_ir_pool(
        [a.ir_train for a in state.attempts if a.compile_ok], two_sided=True
    )

    # holdout 段扩窗预热：在完整帧上求值、裁剪到 >= holdout_start。
    # 只喂 holdout_df 会让滚动算子在边界用截断窗口，发出偏差值。
    if warmup_daily is not None:
        _hold_frame, _hold_start = warmup_daily, holdout_df["trade_date"].min()
    else:
        _hold_frame, _hold_start = holdout_df, None
    # 整帧预处理（add_derived_columns + 排序，较重）只做一次，循环内每个表达式复用
    # `_factor_df_from_prepped`——否则 all_a×多年帧上逐表达式重跑预处理，护栏这步会慢到像卡死。
    # P5：session 注入 prepped 时跳过对 warmup 的再 prep（与 evaluate 同源）。
    _prepped_hold = prepped if prepped is not None else _preprocess_daily(_hold_frame, profile)

    def _holdout_values(node):
        return _factor_df_from_prepped(node, _prepped_hold, eval_start=_hold_start,
                                       leaf_map=leaf_map)

    # 残差目标：库面板 session 级物化一次（z-score+null→0）；空库 → objective 退化 raw。
    # residual_projector 注入时复用其持有的同一 build_library_panel(lib_pool) 产物
    # （ResidualProjector.__init__ 持 panel 引用）——每轮重建在全 A 是 ~3G 纯重复物化
    # （探针 v23 ⑤ 护栏 OOM 三大头之一）。projector 缺席（M5/测试）时现建，零回归。
    lib_panel = (
        residual_projector.panel if residual_projector is not None
        else build_library_panel(lib_pool)
    )
    eff_objective = resolve_objective(objective, lib_panel is not None)
    state.objective = eff_objective
    # 库相关矩阵面板：session 级构建一次，本轮全部候选复用
    lib_corr_panel = build_library_corr_panel(lib_pool) if lib_pool else None
    # train 段因子值求值帧（残差 train IC）。
    # 注意：即使 hold 用了 session prepped，train 仍对 mining-only daily 单独 prep——
    # 全帧 prepped 再 filter 会让段首 ret_1d 等与 mining-only prep 不一致（数值红线）。
    _prepped_train = (
        _prepped_hold if daily is _hold_frame
        else _preprocess_daily(daily, profile)
    )
    _train_fwd = bundle.fwd_returns
    _hold_fwd = None  # lazy：残差 holdout 时再算

    # ── 全量残差 train IC + 槽位键（修泄漏 A）────────────────────────────────
    # residual 模式：对**全部**本轮候选（非仅 top-K）算 residual_ic_train，槽位按
    # |residual| 排序；缺残差值退回裸 |ic_train|。raw 模式排序键不变。
    projector = residual_projector
    if (
        projector is None
        and eff_objective == "residual"
        and lib_panel is not None
        and lib_panel.k > 0
    ):
        try:
            projector = ResidualProjector.from_panel(lib_panel)
        except Exception as exc:
            _LOG.warning("ResidualProjector 构建失败，回退逐候选 lstsq: %s: %s",
                         type(exc).__name__, exc)
            projector = None

    if eff_objective == "residual" and lib_panel is not None:
        for a in passed:
            try:
                node = parse_expr(a.expression, leaf_map)
                fdf_train = _factor_df_from_prepped(
                    node, _prepped_train, leaf_map=leaf_map,
                )
                r_tr = compute_residual_ic(
                    fdf_train, lib_panel, _train_fwd, projector=projector,
                )
                # n_days=0 → ic_mean=NaN：存 None，槽位键退回裸 IC
                ric = r_tr.ic_mean
                a.residual_ic_train = (  # type: ignore[attr-defined]
                    None if ric != ric else float(ric)
                )
            except Exception as exc:
                _LOG.debug(
                    "全量残差 train IC 失败 %s: %s: %s",
                    a.expression, type(exc).__name__, exc,
                )

        def _slot_key(a) -> float:
            ric = getattr(a, "residual_ic_train", None)
            if ric is not None and ric == ric:
                return abs(float(ric))
            return abs(a.ic_train or 0.0)

        passed.sort(key=_slot_key, reverse=True)
    else:
        passed.sort(key=lambda a: abs(a.ic_train or 0.0), reverse=True)

    existing_exprs: set[str] = {c["expression"] for c in state.candidates}

    pool: dict = {}
    for i, c in enumerate(tqdm(state.candidates, desc="  ⑤ 护栏·去相关池",
                               leave=False, unit="因子")):
        try:
            pool[f"prev_{i}"] = _holdout_values(parse_expr(c["expression"], leaf_map))
        except Exception as exc:
            # 去相关池少一个成员 → 后续候选的 max_corr 偏低 → 可能放进重复因子。不是无害的。
            _LOG.warning("已有候选 %s 的 holdout 求值失败，未计入去相关池: %s",
                         c["expression"], exc)
            continue

    for a in tqdm(passed[:top_k], desc="  ⑤ 护栏·候选验收", leave=False, unit="因子"):
        if a.expression in existing_exprs:
            continue
        try:
            node = parse_expr(a.expression, leaf_map)
            fdf_hold = _holdout_values(node)
            hres = holdout_ic_result(
                fdf_hold, holdout_df,
                exec_lag=exec_lag, exec_price_col=exec_price_col,
            )
            ic_h, ir_h, (ci_lo, ci_hi), n_h = hres.ic_mean, hres.ir, hres.ci, hres.n_days
            a.n_holdout_days = n_h
            # 传**带符号** IR：取绝对值由 basis.two_sided 在 deflated_pvalue 内部完成，
            # 与 effective_trials=2N 成对生效。调用方自己 abs 会让两者脱钩（PR #71 前的 bug）。
            sharpe = a.ir_train if a.ir_train is not None else (a.ic_train or 0.0)
            dsr, pval = deflated_pvalue(sharpe, basis, a.n_train or 0)
            ic_tr = a.ic_train or 0.0

            residual_ic_tr = residual_ic_h = None
            n_residual_h = None
            mc_lib, nearest = 0.0, None

            if eff_objective == "residual" and lib_panel is not None:
                # residual：仅 corr>0.95 硬拒（重复）；(0.7,0.95] 继续残差评估。
                ok_lib, mc_lib, nearest = library_orthogonal_check(
                    fdf_hold, lib_pool, threshold=DEFAULT_DUPLICATE_CORR,
                    panel=lib_corr_panel,
                )
                if not ok_lib:
                    a.passed_guardrails = True  # 方向重复非「无效」；known_invalid 排除
                    a.decorrelated = True
                    a.reject_category = REJECT_CATEGORY_LIBRARY_CORRELATED
                    nearest_s = (nearest or "")[:60]
                    a.reject_reason = (
                        f"与库内因子重复(corr={mc_lib:.2f}, 最相近={nearest_s})"
                    )
                    state.n_library_correlated_rejects = (
                        getattr(state, "n_library_correlated_rejects", 0) + 1
                    )
                    continue
                # train 残差：复用全量预计算；holdout 残差仅 top-K 验收需要
                residual_ic_tr = getattr(a, "residual_ic_train", None)
                if residual_ic_tr is None:
                    fdf_train = _factor_df_from_prepped(
                        node, _prepped_train, leaf_map=leaf_map,
                    )
                    r_tr = compute_residual_ic(
                        fdf_train, lib_panel, _train_fwd, projector=projector,
                    )
                    residual_ic_tr = (
                        None if r_tr.ic_mean != r_tr.ic_mean else float(r_tr.ic_mean)
                    )
                    a.residual_ic_train = residual_ic_tr  # type: ignore[attr-defined]
                if _hold_fwd is None:
                    # 与 train bundle / holdout 主门同源成交口径（禁止恒 close→close）
                    _hold_fwd = holdout_fwd_returns(
                        holdout_df,
                        exec_lag=exec_lag, exec_price_col=exec_price_col,
                    )
                r_h = compute_residual_ic(
                    fdf_hold, lib_panel, _hold_fwd, projector=projector,
                )
                residual_ic_h = r_h.ic_mean
                n_residual_h = r_h.n_days
                a.residual_holdout_ic = residual_ic_h  # type: ignore[attr-defined]
                a.n_residual_holdout_days = n_residual_h  # type: ignore[attr-defined]
                reasons = acceptance_reasons(
                    gate=gate, ic_train=residual_ic_tr, holdout_ic=residual_ic_h,
                    dsr_pvalue=pval, ci_low=ci_lo, ci_high=ci_hi, dsr_alpha=dsr_alpha,
                    ic_floor=DEFAULT_RESIDUAL_IC_FLOOR,
                    holdout_n_days=n_residual_h,
                    reason_style="residual",
                )
                # 库相关软信号：挡快速通道，不硬拒、不进 known_invalid
                if abs(float(mc_lib)) >= DEFAULT_DECORR_THRESHOLD:
                    reasons = [*reasons, f"库相关持保留(corr={mc_lib:.2f})"]
            else:
                # raw（或库空退化）：裸 IC 门，顺序与 P4 零回归一致
                reasons = acceptance_reasons(
                    gate=gate, ic_train=ic_tr, holdout_ic=ic_h, dsr_pvalue=pval,
                    ci_low=ci_lo, ci_high=ci_hi, dsr_alpha=dsr_alpha,
                    holdout_n_days=n_h,
                )

            if not reasons:
                # 事实先落定：过了定量护栏。去相关/库相关是随后的**决策**。
                a.passed_guardrails = True
                # raw 模式：定量过关后再做库相关（P4 旧顺序，零回归）
                if eff_objective != "residual":
                    ok_lib, mc_lib, nearest = library_orthogonal_check(
                        fdf_hold, lib_pool, threshold=DEFAULT_DUPLICATE_CORR,
                        panel=lib_corr_panel,
                    )
                    if not ok_lib:
                        a.decorrelated = True
                        a.reject_category = REJECT_CATEGORY_LIBRARY_CORRELATED
                        nearest_s = (nearest or "")[:60]
                        a.reject_reason = (
                            f"与库内因子重复(corr={mc_lib:.2f}, 最相近={nearest_s})"
                        )
                        state.n_library_correlated_rejects = (
                            getattr(state, "n_library_correlated_rejects", 0) + 1
                        )
                        continue
                    # 软区：不入 active，可打 lift 队列
                    if lib_pool and abs(float(mc_lib)) >= DEFAULT_DECORR_THRESHOLD:
                        a.passed_guardrails = False
                        a.reject_reason = f"库相关持保留(corr={mc_lib:.2f})"
                        lift_probe = {
                            "ic_train": ic_tr,
                            "n_holdout_days": n_h,
                            "residual_ic_train": residual_ic_tr,
                            "n_residual_holdout_days": n_residual_h,
                            "reject_category": a.reject_category,
                            "max_corr_library": mc_lib,
                        }
                        if is_lift_queue_candidate(lift_probe, objective=eff_objective):
                            a.reject_category = REJECT_CATEGORY_LIFT_QUEUE
                            a.reject_reason = (
                                (a.reject_reason or "") + "(lift队列,待组合裁决)"
                            )
                            state.n_gray_zone = getattr(state, "n_gray_zone", 0) + 1
                        continue
                corr = max_correlation(fdf_hold, pool)
                # 恰等阈值 = 拒（与 M1 ``mc < threshold`` / library_orthogonal_check 一致）
                if corr >= DEFAULT_DECORR_THRESHOLD:
                    a.decorrelated = True
                    a.reject_reason = (
                        f"与已有候选高度相关(corr={corr:.2f}≥{DEFAULT_DECORR_THRESHOLD})"
                    )
                    continue
                pool[a.expression] = fdf_hold
                existing_exprs.add(a.expression)
                cand_row = {
                    "expression": a.expression,
                    "hypothesis": a.hypothesis,
                    "ic_train": a.ic_train,
                    "ir_train": a.ir_train,
                    "turnover": a.turnover,
                    "holdout_ic": ic_h,
                    "holdout_ir": ir_h,
                    "dsr": dsr,
                    "dsr_pvalue": pval,
                    # 收尾复核与「拿 manifest 复算 p」都需要这三个：
                    # n_obs 是因子自己的有效 IC 天数，CI 两端喂 guardrail_passed 的方向门槛。
                    "n_train": a.n_train,
                    "n_holdout_days": n_h,
                    "ic_ci_low": ci_lo,
                    "ic_ci_high": ci_hi,
                }
                if lib_pool:
                    cand_row["max_corr_library"] = round(float(mc_lib), 4)
                if residual_ic_tr is not None:
                    cand_row["residual_ic_train"] = residual_ic_tr
                    cand_row["residual_holdout_ic"] = residual_ic_h
                    cand_row["n_residual_holdout_days"] = n_residual_h
                state.candidates.append(cand_row)
            else:
                # 记下未过原因，供进度与收尾"近失表"展示（为什么没进候选池）。
                a.reject_reason = "；".join(reasons)
                a.reject_category = classify_reject_category(reasons)
                # 第二通道：单因子门不过但可入 lift 队列 → 标记待组合裁决。
                lift_probe = {
                    "ic_train": ic_tr,
                    "n_holdout_days": n_h,
                    "residual_ic_train": residual_ic_tr,
                    "n_residual_holdout_days": n_residual_h,
                    "reject_category": a.reject_category,
                    "max_corr_library": mc_lib if lib_pool else None,
                }
                if is_lift_queue_candidate(lift_probe, objective=eff_objective):
                    a.reject_category = REJECT_CATEGORY_LIFT_QUEUE
                    a.reject_reason = (a.reject_reason or "") + "(lift队列,待组合裁决)"
                    state.n_gray_zone = getattr(state, "n_gray_zone", 0) + 1
        except Exception as exc:
            # 静默 continue 会让「这个候选炸了」与「这个候选没过护栏」不可区分。
            a.reject_reason = f"护栏计算异常({type(exc).__name__})"
            _LOG.warning("候选 %s 的护栏计算失败，已跳过: %s: %s",
                         a.expression, type(exc).__name__, exc)
            continue

    # ── 非 top-K 全量残差 → lift 队列（与 top-K 统一 is_lift_queue_candidate 门）──
    # 控成本：先按 train residual ≥ gray floor 过滤，再对幸存者补算 holdout 残差/覆盖
    # （与 top-K 同函数同参数：_holdout_values + compute_residual_ic），统一走
    # is_lift_queue_candidate；reason 后缀一律「(lift队列,待组合裁决)」。
    # session 末 lift_dropped_coverage 机制保留（旁路已前置覆盖门；双保险）。
    if eff_objective == "residual" and lib_panel is not None:
        for a in passed[top_k:]:
            if getattr(a, "passed_guardrails", False):
                continue
            if a.reject_category == REJECT_CATEGORY_LIBRARY_CORRELATED:
                continue
            if a.reject_category == REJECT_CATEGORY_LIFT_QUEUE:
                continue
            ric_nk = getattr(a, "residual_ic_train", None)
            if ric_nk is None or ric_nk != ric_nk:
                continue
            if abs(float(ric_nk)) < DEFAULT_GRAY_IC_FLOOR:
                continue
            # 幸存者：补算 holdout 残差（与 top-K 路径同一计算）
            try:
                node_nk = parse_expr(a.expression, leaf_map)
                fdf_hold_nk = _holdout_values(node_nk)
                if _hold_fwd is None:
                    _hold_fwd = holdout_fwd_returns(
                        holdout_df,
                        exec_lag=exec_lag, exec_price_col=exec_price_col,
                    )
                r_h_nk = compute_residual_ic(
                    fdf_hold_nk, lib_panel, _hold_fwd, projector=projector,
                )
                residual_ic_h_nk = r_h_nk.ic_mean
                n_residual_h_nk = r_h_nk.n_days
                a.residual_holdout_ic = residual_ic_h_nk  # type: ignore[attr-defined]
                a.n_residual_holdout_days = n_residual_h_nk  # type: ignore[attr-defined]
            except Exception as exc:
                _LOG.debug(
                    "非 top-K holdout 残差失败 %s: %s: %s",
                    a.expression, type(exc).__name__, exc,
                )
                continue
            lift_probe = {
                "ic_train": a.ic_train,
                "n_holdout_days": getattr(a, "n_holdout_days", None),
                "residual_ic_train": ric_nk,
                "n_residual_holdout_days": n_residual_h_nk,
                "reject_category": a.reject_category,
                "max_corr_library": getattr(a, "max_corr_library", None),
            }
            if is_lift_queue_candidate(lift_probe, objective="residual"):
                a.reject_category = REJECT_CATEGORY_LIFT_QUEUE
                suffix = "(lift队列,待组合裁决)"
                a.reject_reason = (
                    ((a.reject_reason or "") + suffix) if a.reject_reason else suffix
                )
                state.n_gray_zone = getattr(state, "n_gray_zone", 0) + 1

    # ── 稀疏因子 sleeve 旁路：子集 IC 达标 → lift_queue（不直接 passed）────────
    # 与 is_lift_queue_candidate 全截面地板独立：fill-0 事件叶全截面被稀释后
    # 常过不了 gray floor，事件子集才是真口径。稠密因子 is_sparse=False 零开销跳过。
    if sleeve_gate:
        for a in passed:
            if getattr(a, "passed_guardrails", False):
                continue
            if a.reject_category == REJECT_CATEGORY_LIBRARY_CORRELATED:
                continue
            if not getattr(a, "is_sparse", False):
                continue
            # holdout 子集 IC：尚未算则补算（top-K 与非 top-K 统一）
            if getattr(a, "subset_ic_holdout", None) is None:
                try:
                    node_sl = parse_expr(a.expression, leaf_map)
                    fdf_hold_sl = _holdout_values(node_sl)
                    if _hold_fwd is None:
                        _hold_fwd = holdout_fwd_returns(
                            holdout_df,
                            exec_lag=exec_lag, exec_price_col=exec_price_col,
                        )
                    sic_h, sn_h = compute_subset_rank_ic(fdf_hold_sl, _hold_fwd)
                    a.subset_ic_holdout = sic_h  # type: ignore[attr-defined]
                    a.subset_n_days_holdout = sn_h  # type: ignore[attr-defined]
                except Exception as exc:
                    _LOG.debug(
                        "sleeve holdout 子集 IC 失败 %s: %s: %s",
                        a.expression, type(exc).__name__, exc,
                    )
                    continue
            sleeve_probe = {
                "is_sparse": True,
                "subset_ic_train": getattr(a, "subset_ic_train", None),
                "subset_ic_holdout": getattr(a, "subset_ic_holdout", None),
                "subset_n_days_train": getattr(a, "subset_n_days_train", None),
            }
            if not is_sleeve_lift_candidate(sleeve_probe, sleeve_gate=True):
                continue
            already_queue = a.reject_category == REJECT_CATEGORY_LIFT_QUEUE
            a.reject_category = REJECT_CATEGORY_LIFT_QUEUE
            a.sleeve_candidate = True  # type: ignore[attr-defined]
            sleeve_suffix = "(sleeve候选,lift队列,待组合裁决)"
            if a.reject_reason and sleeve_suffix not in a.reject_reason:
                a.reject_reason = a.reject_reason + sleeve_suffix
            elif not a.reject_reason:
                a.reject_reason = sleeve_suffix
            if not already_queue:
                state.n_gray_zone = getattr(state, "n_gray_zone", 0) + 1

    try:
        # None-gating：eval_start 是 None（旧调用方默认）时裸求值，与之前逐字节相同；
        # 非 None 时在完整帧上求值、裁剪到 [eval_start, daily 的 train 段终点]。
        _pbo_bar = tqdm(state.candidates, desc="  ⑤ 护栏·池级PBO", leave=False, unit="因子")
        if eval_start is not None and warmup_daily is not None:
            # session prepped 或 warmup 与 _hold_frame 同对象 → 复用；否则再 prep。
            if prepped is not None:
                _pbo_prepped = prepped
            elif warmup_daily is _hold_frame:
                _pbo_prepped = _prepped_hold
            else:
                _pbo_prepped = _preprocess_daily(warmup_daily, profile)
            _pbo_end = daily["trade_date"].max()
            cand_fdfs = [
                _factor_df_from_prepped(parse_expr(c["expression"], leaf_map), _pbo_prepped,
                                        eval_start=eval_start, eval_end=_pbo_end, leaf_map=leaf_map)
                for c in _pbo_bar
            ]
        else:
            _pbo_prepped = _prepped_hold if daily is _hold_frame else _preprocess_daily(daily, profile)
            cand_fdfs = [
                _factor_df_from_prepped(parse_expr(c["expression"], leaf_map), _pbo_prepped,
                                        leaf_map=leaf_map)
                for c in _pbo_bar
            ]
        state.pbo = pool_pbo(cand_fdfs, bundle.fwd_returns)
    except Exception as exc:
        _LOG.warning("池级 PBO 计算失败，记为 nan: %s: %s", type(exc).__name__, exc)
        state.pbo = float("nan")
    return state


def node_finalize_guardrails(state: AgentState, *, dsr_alpha: float = DEFAULT_DSR_ALPHA,
                             gate: str = DEFAULT_GATE, daily=None, bundle=None, profile=None,
                             prior=None):
    """收尾：用**最终** basis 统一重算候选的 DSR，据此复核入池。返回该 basis。

    ``gate="library"``（默认）下入池判据 N-**无关**（真+有信号，不含 DSR），故收尾只重算 DSR
    供报告/组合层用、不因 N 剔除候选；``gate="strict"`` 下才按最终 N 重判 DSR、剔除不再显著者。

    `node_guardrails` 每轮调用，其 basis 只覆盖「截至当轮」的 N。于是 round 0 的候选按
    N@轮=3 定 p、round 5 的候选按 N=18 定 p —— **门槛取决于候选碰巧在第几轮被找到**。
    但候选集是从整个 session 的 N 次试验里选出来的，多重检验记账必须覆盖整个搜索。
    实测 `team_51_6r` 的 round-0 候选记录 p=0.0011，按最终 N=18 复算是 p=0.0212（差 19 倍），
    且 manifest 报 n_trials=18 —— 拿 manifest 复算不出产物里的 p。

    **campaign prior（跨 session 族）**：同一评价配置下多个 team session 若各自从零计数，
    则 DSR 的 N 只在 session 内诚实——跨 session 的多重检验漏记。``prior`` 为
    :class:`~factorzen.discovery.campaign.CampaignPrior` 时，basis 用
    **历史唯一表达式 IR ∪ 本 session 新增**（表达式级去重、历史优先）构造，
    消除「N 取决于候选碰巧在第几个 session 被找到」的旧缺陷。``prior=None`` 零回归。

    只重算 DSR：`holdout_ic` / CI / `ic_train` 都与 N 无关。**不逐轮重跑护栏**——那会让
    N 三角和 over-count（见 discovery 的多轮累积计数陷阱）。

    收尾复核与首轮判定同口径：residual 候选按 residual 指标+floor 复核，防 objective 漂移误杀
    （raw 弱、residual 强的候选首轮通过后不得被 raw 门二次杀掉）。库空退化无 residual 字段时
    回退 raw。

    已知取舍：被降级的候选此前可能以 `corr≥0.7` 压制过其它因子，那些因子仍留在
    `decorrelated=True`，此处不复活。复活需重跑去相关，会引入新的 N 记账问题；
    且方向保守（候选只减不增）。

    `daily`/`bundle` 给出时重算池级 PBO——候选集变了，旧 PBO 描述的是另一个池。
    返回的 basis 供调用方落 manifest（``basis.n_trials`` = family 总 N）。
    """
    from factorzen.discovery.evaluation import _factor_df_from_prepped, _preprocess_daily
    from factorzen.discovery.guardrails import (
        DeflationBasis,
        acceptance_reasons,
        deflated_pvalue,
        pool_pbo,
    )

    # 表达式级 trial 池：本 session compile_ok 的 (expr, ir)
    session_pool = [(a.expression, a.ir_train) for a in state.attempts if a.compile_ok]
    if prior is not None:
        # 历史优先：同表达式不双计；union = prior.irs + 本 session 新增 IR
        union_irs = list(prior.irs) + [
            ir for expr, ir in session_pool if expr not in prior.expressions
        ]
        basis = DeflationBasis.from_ir_pool(union_irs, two_sided=True)
    else:
        basis = DeflationBasis.from_ir_pool(
            [ir for _, ir in session_pool], two_sided=True
        )
    if not state.candidates:
        return basis

    by_expr = {a.expression: a for a in state.attempts}
    survivors: list[dict] = []
    for c in state.candidates:
        dsr, pval = deflated_pvalue(c["ir_train"] or 0.0, basis, c["n_train"] or 0)
        c["dsr"], c["dsr_pvalue"] = dsr, pval
        # 与 node_guardrails residual 分支同口径；无 residual 字段则回退 raw（库空退化）
        use_residual = (
            getattr(state, "objective", "raw") == "residual"
            and c.get("residual_ic_train") is not None
        )
        if use_residual:
            reasons = acceptance_reasons(
                gate=gate, ic_train=c["residual_ic_train"],
                holdout_ic=c.get("residual_holdout_ic"), dsr_pvalue=pval,
                ci_low=c["ic_ci_low"], ci_high=c["ic_ci_high"], dsr_alpha=dsr_alpha,
                ic_floor=DEFAULT_RESIDUAL_IC_FLOOR,
                holdout_n_days=c.get("n_residual_holdout_days"),
                reason_style="residual",
            )
        else:
            reasons = acceptance_reasons(
                gate=gate, ic_train=c["ic_train"], holdout_ic=c["holdout_ic"], dsr_pvalue=pval,
                ci_low=c["ic_ci_low"], ci_high=c["ic_ci_high"], dsr_alpha=dsr_alpha,
                holdout_n_days=c.get("n_holdout_days"),
            )
        if not reasons:
            survivors.append(c)
        elif (a := by_expr.get(c["expression"])) is not None:
            # 事实被更完整的 N 修正：它并没有过定量护栏。不同步的话 Librarian
            # 会把它当「已验证有效」写进长期记忆。
            from factorzen.discovery.guardrails import classify_reject_category
            a.passed_guardrails = False
            a.reject_reason = "收尾复核(最终N)：" + "；".join(reasons)
            a.reject_category = classify_reject_category(reasons)

    n_dropped = len(state.candidates) - len(survivors)
    if n_dropped:
        _LOG.info("收尾复核：%d 个候选在最终 N=%d（双边 ⇒ %d）下不再显著，已剔除",
                  n_dropped, basis.n_trials, basis.effective_trials)
    state.candidates = survivors

    if n_dropped and daily is not None and bundle is not None:
        try:
            leaf_map = profile.factors.leaf_features() if profile is not None else None
            _prepped = _preprocess_daily(daily, profile)  # 预处理一次，逐候选复用（同 node_guardrails）
            cand_fdfs = [
                _factor_df_from_prepped(parse_expr(c["expression"], leaf_map), _prepped,
                                        leaf_map=leaf_map)
                for c in state.candidates
            ]
            state.pbo = pool_pbo(cand_fdfs, bundle.fwd_returns)
        except Exception as exc:
            _LOG.warning("收尾 PBO 重算失败，记为 nan: %s: %s", type(exc).__name__, exc)
            state.pbo = float("nan")
    return basis


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
        # 与 roles/critic.py 同用容错解析：request_chat 显式关掉 json_object 模式且不剥
        # markdown 围栏，裸 json.loads 遇 ```json 围栏必抛，会被下面的 except 静默降级为 keep。
        try:
            obj = _extract_json(llm_fn(msgs))
            a.critic_verdict = str(obj.get("verdict", "keep")) if obj else "keep"
        except Exception as exc:
            _LOG.warning("Critic 裁决解析失败，保守判 keep（不误杀）: %s", exc)
            a.critic_verdict = "keep"
    return state


def node_reflect(state: AgentState, *, ic_threshold: float = 0.01) -> AgentState:
    """更新 Negative RAG 负例库 + 推进迭代计数。"""
    seen = [(a.expression, a.ic_train) for a in state.attempts if a.ic_train is not None]
    state.negative_examples = negative_recall(seen, k=5, ic_threshold=ic_threshold)
    state.iteration += 1
    return state

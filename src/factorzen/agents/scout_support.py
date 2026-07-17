"""日内 Feature Scout 共享编排帮手：单 Agent / 团队共用，防双路径漂移。

流程：LLM 提案 → make_expr_spec 校验 → materialize → screen → 注入三帧 →
leaf_health / budgets；session 末仅对「准入因子引用的 ix_*」永久化。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.agents.roles.feature_scout import propose_intraday_features
from factorzen.discovery.intraday_expr import (
    IntradayExprSpec,
    ensure_expr_panel,
    make_expr_spec,
    materialize_expr_features,
    register_expr_features,
    screen_expr_panel,
)
from factorzen.llm.generation import LLMFn

_LOG = logging.getLogger(__name__)


@dataclass
class ScoutState:
    """跨轮 scout 状态：注入叶、已试表达式、审计流水。"""

    injected: list[str] = field(default_factory=list)
    tried_exprs: list[str] = field(default_factory=list)  # 规范化键，avoid 用
    audit: list[dict] = field(default_factory=list)
    # name → IntradayExprSpec，供 session 末 promote 注册
    specs: dict[str, IntradayExprSpec] = field(default_factory=dict)


def _norm_try_key(bar_expr: str, agg: str, freq: str) -> str:
    return f"{agg}|{bar_expr}|{freq}"


def _frame_yyyymmdd_bounds(df: pl.DataFrame) -> tuple[str | None, str | None]:
    """帧内最早/最晚 trade_date → YYYYMMDD。"""
    from factorzen.discovery.intraday_expr import _frame_date_bounds

    return _frame_date_bounds(df)


def _align_trade_date(sel: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    """面板 trade_date dtype 对齐到 frame（与 attach_intraday 同款）。"""
    from factorzen.discovery.intraday_expr import _align_trade_date as _align

    return _align(sel, frame)


def _join_ix_cols(
    frame: pl.DataFrame, panel: pl.DataFrame, names: list[str],
) -> pl.DataFrame:
    """left join ix_* 列；同名先 drop 再 join（对齐 attach_intraday）。"""
    have = [c for c in names if c in panel.columns]
    if not have or frame.is_empty():
        return frame
    sel = panel.select(["trade_date", "ts_code", *have])
    sel = _align_trade_date(sel, frame)
    drop_cols = [c for c in have if c in frame.columns]
    if drop_cols:
        frame = frame.drop(drop_cols)
    return frame.join(sel, on=["trade_date", "ts_code"], how="left")


def _ensure_ctx_leaf_map(ctx: Any) -> dict[str, str]:
    """保证 ctx.leaf_map 为可变 dict（A 股默认 None 时物化 LEAF_FEATURES）。"""
    if ctx.leaf_map is None:
        from factorzen.discovery.operators import LEAF_FEATURES

        ctx.leaf_map = dict(LEAF_FEATURES)
    return ctx.leaf_map


def _append_ctx_leaf(ctx: Any, name: str) -> None:
    """把新 ix_* 叶写入 ctx.leaf_names / leaf_map。"""
    lm = _ensure_ctx_leaf_map(ctx)
    lm[name] = name
    if name not in ctx.leaf_names:
        ctx.leaf_names = [*list(ctx.leaf_names), name]


def _drop_ctx_leaf(ctx: Any, name: str) -> None:
    if name in ctx.leaf_names:
        ctx.leaf_names = [n for n in ctx.leaf_names if n != name]
    if ctx.leaf_map is not None and name in ctx.leaf_map:
        ctx.leaf_map = {k: v for k, v in ctx.leaf_map.items() if k != name}


def _build_known_features(frames: dict[str, pl.DataFrame], state: ScoutState) -> str:
    """现有 i_* 列摘要 + 已注入 / 已试 ix 表达式，作 avoid 上下文。"""
    parts: list[str] = []
    mining = frames.get("mining")
    if mining is not None and not mining.is_empty():
        i_cols = sorted(c for c in mining.columns if c.startswith("i_"))
        if i_cols:
            parts.append("帧内 i_* 叶子: " + ", ".join(i_cols))
    if state.injected:
        inj = []
        for n in state.injected:
            sp = state.specs.get(n)
            if sp is not None:
                inj.append(f"{n}={sp.agg}({sp.bar_expr})")
            else:
                inj.append(n)
        parts.append("本 session 已注入 ix_*: " + "; ".join(inj))
    if state.tried_exprs:
        parts.append("已试表达式键: " + "; ".join(state.tried_exprs[-40:]))
    return "\n".join(parts)


def _reference_from_mining(mining: pl.DataFrame, injected: list[str]) -> pl.DataFrame:
    """screen 参照：mining 帧的 i_* + 已注入 ix_*。"""
    cols = ["trade_date", "ts_code"]
    for c in mining.columns:
        if c not in cols and (
            c.startswith("i_")
            or c in injected
            or (c.startswith("ix_") and c in injected)
        ):
            cols.append(c)
    keep = [c for c in cols if c in mining.columns]
    return mining.select(keep)


def run_scout_round(
    *,
    llm_fn: LLMFn,
    state: ScoutState,
    k: int,
    max_leaves: int,
    start: str,
    end: str,
    freq: str,
    frames: dict[str, pl.DataFrame],
    ctx: Any,
    reference: pl.DataFrame | None = None,
    holdout_start: Any = None,
    eval_start: Any = None,
    leaf_budgets: dict[str, int] | None = None,
    profile: Any = None,
    market_notes: str = "",
    known_features: str | None = None,
) -> dict[str, pl.DataFrame]:
    """一轮 scout：提案 → 校验 → 物化 → 筛 → 注入三帧；单条异常不崩轮次。

    Args:
        frames: ``{"mining", "holdout", "daily"}`` 日频帧（daily=含预热完整帧）。
        reference: screen 相关列参照；None → 从 mining 的 i_*+已注入 ix_* 推导。
        leaf_budgets: 可变 dict，新叶预算并入（None → 跳过 budgets）。
        start/end: 物化窗口 YYYYMMDD（宜含预热段，用 daily 帧最早日期）。

    Returns:
        更新后的 frames（可能与入参为同一 dict 的新 DataFrame 值）。
    """
    if len(state.injected) >= max_leaves:
        return frames

    known = known_features if known_features is not None else _build_known_features(frames, state)
    try:
        proposals = propose_intraday_features(
            llm_fn,
            k=k,
            avoid=list(state.tried_exprs),
            known_features=known,
            market_notes=market_notes,
        )
    except Exception as exc:
        _LOG.warning("feature_scout 提案失败: %s: %s", type(exc).__name__, exc)
        proposals = []

    if not proposals:
        return frames

    # 1) 校验 → 存活 specs
    alive: list[IntradayExprSpec] = []
    for p in proposals:
        bar_expr = str(p.get("bar_expr") or "").strip()
        agg = str(p.get("agg") or "").strip()
        hyp = str(p.get("hypothesis") or "").strip()
        raw_key = _norm_try_key(bar_expr, agg, freq)
        try:
            spec = make_expr_spec(bar_expr, agg, freq=freq, hypothesis=hyp)
        except ValueError as exc:
            if raw_key not in state.tried_exprs:
                state.tried_exprs.append(raw_key)
            state.audit.append({
                "name": None,
                "bar_expr": bar_expr,
                "agg": agg,
                "hypothesis": hyp,
                "verdict": f"invalid:{exc}",
            })
            continue

        try_key = _norm_try_key(spec.bar_expr, spec.agg, spec.freq)
        if try_key in state.tried_exprs or spec.name in state.injected:
            state.audit.append({
                "name": spec.name,
                "bar_expr": spec.bar_expr,
                "agg": spec.agg,
                "hypothesis": hyp or spec.hypothesis,
                "verdict": "duplicate",
            })
            if try_key not in state.tried_exprs:
                state.tried_exprs.append(try_key)
            continue
        if try_key not in state.tried_exprs:
            state.tried_exprs.append(try_key)
        # 容量：本轮注入后不得超过 max_leaves
        if len(state.injected) + len(alive) >= max_leaves:
            state.audit.append({
                "name": spec.name,
                "bar_expr": spec.bar_expr,
                "agg": spec.agg,
                "hypothesis": hyp or spec.hypothesis,
                "verdict": "max_leaves",
            })
            continue
        alive.append(spec)

    if not alive:
        return frames

    # 2) 物化
    try:
        panel = materialize_expr_features(alive, start, end, freq=freq)
    except Exception as exc:
        _LOG.warning("scout materialize 失败: %s: %s", type(exc).__name__, exc)
        for spec in alive:
            state.audit.append({
                "name": spec.name,
                "bar_expr": spec.bar_expr,
                "agg": spec.agg,
                "hypothesis": spec.hypothesis,
                "verdict": f"materialize_error:{type(exc).__name__}",
            })
        return frames

    # 3) screen
    mining = frames["mining"]
    ref = reference if reference is not None else _reference_from_mining(
        mining, state.injected,
    )
    try:
        verdicts = screen_expr_panel(panel, ref)
    except Exception as exc:
        _LOG.warning("scout screen 失败: %s: %s", type(exc).__name__, exc)
        verdicts = {s.name: f"screen_error:{type(exc).__name__}" for s in alive}

    keep_names: list[str] = []
    keep_specs: list[IntradayExprSpec] = []
    for spec in alive:
        v = verdicts.get(spec.name, "low_coverage")
        if v != "keep":
            state.audit.append({
                "name": spec.name,
                "bar_expr": spec.bar_expr,
                "agg": spec.agg,
                "hypothesis": spec.hypothesis,
                "verdict": v,
            })
            continue
        keep_names.append(spec.name)
        keep_specs.append(spec)

    if not keep_names:
        return frames

    # 4) 三帧 left join
    out = dict(frames)
    for key in ("mining", "holdout", "daily"):
        if key not in out or out[key] is None:
            continue
        try:
            out[key] = _join_ix_cols(out[key], panel, keep_names)
        except Exception as exc:
            _LOG.warning("scout join %s 失败: %s: %s", key, type(exc).__name__, exc)
            for spec in keep_specs:
                state.audit.append({
                    "name": spec.name,
                    "bar_expr": spec.bar_expr,
                    "agg": spec.agg,
                    "hypothesis": spec.hypothesis,
                    "verdict": f"join_error:{type(exc).__name__}",
                })
            return frames

    # 5) leaf_health + budgets + ctx
    holdout_df = out["holdout"]
    if holdout_start is None and holdout_df is not None and not holdout_df.is_empty():
        holdout_start = holdout_df["trade_date"].min()

    final_keep: list[str] = []
    for spec in keep_specs:
        name = spec.name
        dead = False
        if holdout_start is not None and holdout_df is not None:
            try:
                from factorzen.discovery.leaf_health import leaf_holdout_coverage

                cov = leaf_holdout_coverage(
                    holdout_df, [name], holdout_start, leaf_map={name: name},
                )
                if float(cov.get(name, 0.0)) <= 0.0:
                    dead = True
            except Exception as exc:
                _LOG.warning("scout leaf_health %s 失败: %s", name, exc)

        if dead:
            # 从三帧摘除列
            for key in ("mining", "holdout", "daily"):
                if key in out and name in out[key].columns:
                    out[key] = out[key].drop(name)
            state.audit.append({
                "name": name,
                "bar_expr": spec.bar_expr,
                "agg": spec.agg,
                "hypothesis": spec.hypothesis,
                "verdict": "dead_on_holdout",
            })
            continue

        _append_ctx_leaf(ctx, name)
        state.injected.append(name)
        state.specs[name] = spec
        state.audit.append({
            "name": name,
            "bar_expr": spec.bar_expr,
            "agg": spec.agg,
            "hypothesis": spec.hypothesis,
            "verdict": "keep",
        })
        final_keep.append(name)

        # budgets：短历史叶并入
        if leaf_budgets is not None and eval_start is not None:
            try:
                from factorzen.config.constants import AGENT_WARMUP_LOOKBACK
                from factorzen.discovery.evaluation import _preprocess_daily
                from factorzen.discovery.expression import leaf_warmup_budgets

                daily_full = out.get("daily")
                if daily_full is not None and not daily_full.is_empty():
                    prepped = _preprocess_daily(daily_full, profile)
                    bud = leaf_warmup_budgets(
                        prepped, eval_start, [name], leaf_map=ctx.leaf_map,
                    )
                    for lk, lv in bud.items():
                        if lv < AGENT_WARMUP_LOOKBACK:
                            leaf_budgets[lk] = lv
            except Exception as exc:
                _LOG.warning("scout leaf_budgets %s 失败: %s", name, exc)

    return out


def promote_admitted_exprs(
    *,
    session_dir: str | Path | None,
    admitted_exprs: list[str],
    state: ScoutState,
    session: str,
    full_start: str,
    full_end: str,
    freq: str,
    base_dir: Path | None = None,
    leaf_map: dict[str, str] | None = None,
) -> list[str]:
    """准入/probation 因子引用的 ix_* ∩ state.injected → 注册 + 全历史 ensure 缓存。

    未引用的不注册（audit 里已有记录）。``base_dir`` 默认全局 INTRADAY_FEATURES_DIR；
    测试传 tmp。``session_dir`` 仅溯源占位，不改变注册路径。
    """
    del session_dir  # 溯源预留；注册路径由 base_dir 决定
    del freq  # name 已绑定 freq；ensure 从 registry 读
    if not state.injected or not admitted_exprs:
        return []

    from factorzen.discovery.preparation import intraday_expr_leaf_names

    # 扩展 leaf_map 以便 parse 已注入 ix_*
    lm = dict(leaf_map) if leaf_map is not None else {}
    for n in state.injected:
        lm[n] = n
    referenced = set(intraday_expr_leaf_names(admitted_exprs, leaf_map=lm or None))
    # 词法回退：parse 失败时仍能从表达式字面抓 ix_*
    for expr in admitted_exprs:
        if not expr:
            continue
        s = str(expr)
        for n in state.injected:
            if n in s:
                referenced.add(n)

    to_promote = [n for n in state.injected if n in referenced]
    if not to_promote:
        return []

    specs = [state.specs[n] for n in to_promote if n in state.specs]
    if not specs:
        return []

    try:
        register_expr_features(specs, session=session, base_dir=base_dir)
    except Exception as exc:
        _LOG.warning("promote register 失败: %s: %s", type(exc).__name__, exc)
        return []

    promoted: list[str] = []
    for sp in specs:
        try:
            ensure_expr_panel(
                sp.name, full_start, full_end, base_dir=base_dir,
            )
            promoted.append(sp.name)
        except Exception as exc:
            _LOG.warning(
                "promote ensure %s 失败: %s: %s", sp.name, type(exc).__name__, exc,
            )
            # 已 register 仍算 promote（缓存可后续补）
            promoted.append(sp.name)
    return promoted


def scout_manifest_block(
    state: ScoutState | None,
    *,
    promoted: list[str] | None = None,
) -> dict | None:
    """session manifest 的 ``intraday_scout`` 块；state is None → None（flag-off）。"""
    if state is None:
        return None
    proposed = sum(1 for a in state.audit if a.get("name") is not None or a.get("bar_expr"))
    # 更稳：提案数 = audit 条数（含 invalid）
    proposed = len(state.audit)
    return {
        "proposed": proposed,
        "injected": list(state.injected),
        "promoted": list(promoted or []),
        "audit": list(state.audit),
    }


def filter_exhausted_expressions(
    exprs: list[str],
    *,
    exhausted: set[str] | list[str] | None,
    leaf_map: dict[str, str] | set[str] | None,
    quota_used: dict[str, int],
    per_leaf_quota: int = 2,
) -> tuple[list[str], int]:
    """硬过滤：纯 exhausted 叶表达式丢弃；混族按轮内配额。

    规则（按序）：
    1. parse 失败 → 保留（语法坑归自愈，不在这里杀）；
    2. 叶集合非空且 **全部 ∈ exhausted** → 丢弃（纯重挖死方向）；
    3. 否则对表达式中每个 exhausted 叶查 ``quota_used``：
       任一叶已达 ``per_leaf_quota`` → 丢弃；未达则全部计数 +1、保留。

    ``exhausted`` 为 None/空 → 直通零回归。``quota_used`` 由调用方持有，跨假设共享。
    返回 ``(保留列表, 丢弃数)``。
    """
    if not exprs:
        return [], 0
    if not exhausted:
        return list(exprs), 0

    from factorzen.discovery.expression import feature_names, parse_expr

    exh = set(exhausted)
    kept: list[str] = []
    n_drop = 0
    for e in exprs:
        try:
            node = parse_expr(e, leaf_map)
            leaves = feature_names(node)
        except (ValueError, TypeError, IndexError):
            kept.append(e)
            continue
        if leaves and leaves <= exh:
            n_drop += 1
            continue
        exh_in = leaves & exh
        if exh_in and any(quota_used.get(leaf, 0) >= per_leaf_quota for leaf in exh_in):
            n_drop += 1
            continue
        for leaf in exh_in:
            quota_used[leaf] = quota_used.get(leaf, 0) + 1
        kept.append(e)
    return kept, n_drop

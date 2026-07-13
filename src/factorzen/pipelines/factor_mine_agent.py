# src/factorzen/pipelines/factor_mine_agent.py
"""LLM Agent 闭环挖掘 pipeline：跑 Agent → 落 manifest + 导出候选。"""
from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

from factorzen.agents.manifest import write_session_manifest
from factorzen.agents.orchestrator import run_llm_agent
from factorzen.llm.config import load_llm_config


def _default_llm_fn():
    """生产 LLMFn：包 request_chat + load_llm_config。"""
    from factorzen.llm.client import request_chat
    config = load_llm_config(enabled=True)
    if not config.is_ready:
        raise RuntimeError("LLM 未配置：设置 .env 的 FACTORZEN_LLM_* 或注入 llm_fn")
    return lambda messages: request_chat(config, messages)


def _llm_meta(llm_fn) -> dict:
    """记录本次挖掘实际使用的 LLM 身份——结果强依赖模型，缺了它 manifest 不可复现。

    注入 llm_fn 时不去读 env（可能根本没配），但要标记出来，免得读者误以为用了 .env 里的模型。
    绝不写入 api_key。
    """
    if llm_fn is not None:
        return {"injected": True}
    c = load_llm_config(enabled=True)
    return {
        "model": c.model,
        "provider": c.provider,
        "temperature": c.temperature,
        "max_tokens": c.max_tokens,
        "thinking": c.thinking or None,
        "max_retries": c.max_retries,
        # 双 profile 审计：事后必须能还原用的是 AIPing 还是 openai 兼容网关
        "flavor": c.flavor,
        "profile": c.profile,
    }


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _fmt(x, nd: int = 4) -> str:
    """把 IC/DSR 等数值格式化成定宽字符串；None/NaN 显示为 —（缺 ≠ 0）。"""
    if x is None:
        return "—"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "—" if f != f else f"{f:.{nd}f}"  # f != f 即 NaN


def _print_startup(daily, params: dict, *, label: str, rid: str) -> None:
    """挖掘启动横幅：窗口 / universe / 数据规模 / 关键参数 / LLM 身份。

    在调 orchestrator 前打印，让用户一眼看清这次挖的是哪段数据、什么配置、用哪个模型。
    """
    n_rows = daily.height
    cols = daily.columns
    n_stocks = daily["ts_code"].n_unique() if "ts_code" in cols else "?"
    n_days = daily["trade_date"].n_unique() if "trade_date" in cols else "?"
    llm = params.get("llm") or {}
    model = llm.get("model") or ("注入 llm_fn" if llm.get("injected") else "?")
    extra = f" ｜ structured={params['structured']}" if "structured" in params else ""
    print(f"\n{'═' * 72}")
    print(f"[{label}] 挖掘启动 ▸ {rid}")
    # universe 缺省的兜底随市场变：A 股=全A；其它市场=Top-N 快照（别把 crypto 标成"全A"）。
    _uni_default = "全A" if params.get("market", "ashare") == "ashare" else "Top-N 快照"
    print(f"[{label}] 窗口 {params.get('start', '?')}~{params.get('end', '?')}"
          f" ｜ universe={params.get('universe') or _uni_default}"
          f" ｜ market={params.get('market', 'ashare')}")
    print(f"[{label}] 载入 {n_rows} 行 / {n_stocks} 只票 / {n_days} 交易日（含预热前缀）")
    print(f"[{label}] 轮数={params.get('n_rounds')} ｜ top_k={params.get('top_k')}"
          f" ｜ seed={params.get('seed')} ｜ holdout={params.get('holdout_ratio')}"
          f" ｜ patience={params.get('patience')} ｜ heal_rounds={params.get('heal_rounds')}{extra}")
    print(f"[{label}] LLM={model}")
    print(f"{'═' * 72}", flush=True)


def _print_round_progress(result, *, label: str) -> None:
    """每轮末进度：本轮评估/有效数、最佳 |train_IC|、累计候选与 N（team 附带 Critic 裁决）。

    由 `_checkpoint`（每个成功轮次回调）调用。``result.state.iteration`` 在轮末已 +1，
    故本轮 = ``iteration - 1``；据此筛出本轮 attempts 统计。
    """
    state = result.state
    rnd = state.iteration - 1
    this = [a for a in state.attempts if a.iteration == rnd]
    n_ok = sum(1 for a in this if a.compile_ok and a.ic_train is not None)
    best = max((abs(a.ic_train) for a in this if a.ic_train is not None), default=None)
    best_s = f"{best:.4f}" if best is not None else "—"
    line = (f"[{label}] 第 {rnd + 1} 轮 ▸ 评估 {len(this)}（有效 {n_ok}）"
            f" ▸ 本轮最佳|train_IC|={best_s}"
            f" ▸ 累计候选 {len(result.candidates)} ▸ N={result.n_trials}")
    rounds_log = getattr(result, "rounds_log", None)
    if rounds_log:
        line += f" ▸ 裁决={rounds_log[-1].get('verdict', '?')}"
    print(line, flush=True)


def _print_final_stats(result, run_dir: str, *, label: str) -> None:
    """收尾因子统计：每候选一行 IC/ICIR/holdout/DSR/换手 + 池级 PBO / N / sharpe_var。

    候选数取**收尾复核后**的最终集（可能少于最后一轮进度显示的累计数——最终 N 下不再
    显著的早轮候选已被 `node_finalize_guardrails` 剔除）。
    """
    cands = result.candidates
    print(f"\n{'═' * 72}")
    print(f"[{label}] 挖掘完成 ▸ 候选 {len(cands)} 个通过护栏"
          f" ▸ 共评估 N={result.n_trials} 个唯一表达式"
          f" ▸ 池级 PBO={_fmt(result.state.pbo, 3)}"
          f" ▸ sharpe_var={_fmt(result.sharpe_variance)}")
    if not cands:
        print(f"[{label}] 无候选通过防过拟合护栏（DSR/holdout/CI）")
    for i, c in enumerate(cands, 1):
        print(f"[{label}] #{i} ▸ train_IC={_fmt(c.get('ic_train'))}"
              f" ICIR={_fmt(c.get('ir_train'))}"
              f" holdout_IC={_fmt(c.get('holdout_ic'))}"
              f" DSR_p={_fmt(c.get('dsr_pvalue'))}"
              f" 换手={_fmt(c.get('turnover'), 3)}")
        print(f"[{label}]      表达式：{c.get('expression')}")
        if c.get("hypothesis"):
            print(f"[{label}]      假设：{c['hypothesis']}")
    _print_near_miss(result, cands, label=label)
    print(f"[{label}] 产物目录：{run_dir}")
    print(f"{'═' * 72}", flush=True)


def _print_near_miss(result, cands, *, label: str, top: int = 8) -> None:
    """近失表：进过 top-k 护栏评估、但未入候选池的表达式 + 原因（为什么没过护栏）。

    尤其当 0 候选时，回答用户「到底卡在哪道门」——按 |train_IC| 排序取最强的若干个。
    `reject_reason` 由 `node_guardrails` / `node_finalize_guardrails` 记；已入选者剔除。
    """
    cand_exprs = {c.get("expression") for c in cands}
    rejected = [a for a in result.state.attempts
                if a.reject_reason and a.ic_train is not None
                and a.expression not in cand_exprs]
    if not rejected:
        return
    rejected.sort(key=lambda a: abs(a.ic_train or 0.0), reverse=True)
    shown = rejected[:top]
    print(f"[{label}] 未入选候选的原因（近失，按 |train_IC| 排序，共 {len(rejected)} 个"
          f"{'，列前 ' + str(top) if len(rejected) > top else ''}）：")
    for a in shown:
        print(f"[{label}]   ✗ train_IC={_fmt(a.ic_train)} ｜ {a.expression}")
        print(f"[{label}]       原因：{a.reject_reason}")


def run_agent_mine(daily, *, n_rounds: int, seed: int, out_dir: str = "workspace/mine_agent",
                   llm_fn=None, top_k: int = 5, holdout_ratio: float = 0.2,
                   human_review: bool = False, run_id: str | None = None,
                   export: bool = True, patience: int | None = None,
                   heal_rounds: int = 2,
                   data_window: dict | None = None, command: str | None = None,
                   eval_start: str | None = None, profile=None,
                   library_orthogonal: bool = True,
                   objective: str = "residual") -> dict:
    """跑单 Agent 挖掘闭环，每轮增量落 manifest，收尾导出候选。

    ``data_window``：``{start, end, universe, market}``。落进 manifest 的 params，
    否则事后无从得知这批因子挖自哪段数据、哪个票池（铁律#3）。
    ``command``：触发本次运行的命令行。

    ``profile``：市场 profile（默认 None → A 股，零回归）。crypto 等传各自 profile，透传到
    `run_llm_agent`；数据装配（含预热前缀的 crypto daily）由调用方（CLI）负责。

    ``eval_start``：``"YYYYMMDD"``，训练段的干净起点。``daily`` 由 `prepare_mining_daily`
    带 ``lookback_days`` 预热前缀，须把该前缀的边界（= 挖掘窗口 ``start``）透传给
    `run_llm_agent`，否则预热段随 `split_holdout` 进 train IC（与 M1 `run_session(eval_start=)`
    同口径）。``None``（默认）退化为旧行为，对现有调用方零回归。
    """
    fn = llm_fn or _default_llm_fn()
    rid = run_id or f"{_timestamp()}_agent_{seed}_{n_rounds}r"
    params = {
        "n_rounds": n_rounds, "seed": seed, "top_k": top_k, "holdout_ratio": holdout_ratio,
        "patience": patience, "heal_rounds": heal_rounds, "eval_start": eval_start,
        **(data_window or {}),
        "command": command,
        "llm": _llm_meta(llm_fn),
    }

    def _checkpoint(partial_result) -> None:
        """每轮末增量落盘 + 打印进度：进程若在下一轮崩溃，已找到的候选不至于全损。"""
        write_session_manifest(partial_result, out_dir=out_dir, run_id=rid,
                               params=params, partial=True)
        _print_round_progress(partial_result, label="mine-agent")

    _print_startup(daily, params, label="mine-agent", rid=rid)
    result = run_llm_agent(daily, fn, n_rounds=n_rounds, seed=seed, top_k=top_k,
                           holdout_ratio=holdout_ratio, human_review=human_review,
                           patience=patience, heal_rounds=heal_rounds, eval_start=eval_start,
                           on_round_end=_checkpoint, profile=profile,
                           library_orthogonal=library_orthogonal,
                           library_root=str(Path(out_dir).parent / "factor_library"),
                           objective=objective)
    write_session_manifest(result, out_dir=out_dir, run_id=rid, params=params, partial=False)
    run_dir = Path(out_dir) / rid
    # candidates.csv —— 兼容 fz mine leaderboard/export-alpha（含 rank + passed 列）
    run_dir.mkdir(parents=True, exist_ok=True)
    from factorzen.discovery.export import agent_candidates_csv_df
    agent_candidates_csv_df(result.candidates).write_csv(run_dir / "candidates.csv")
    if export:
        exp_dir = run_dir / "exported"
        # 清空必须独立于「本次有无候选」：复用 run_id 时若本次候选更少（乃至为 0），
        # 上次 run 的多余因子文件会残留并被下游消费。
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
        if result.candidates:
            from factorzen.discovery.export import export_candidate
            exp_dir.mkdir(parents=True, exist_ok=True)
            for i, c in enumerate(result.candidates):
                export_candidate(c["expression"], f"agent_{rid}_{i}", str(exp_dir))
    _print_final_stats(result, str(run_dir), label="mine-agent")
    return {"run_dir": str(run_dir), "n_candidates": len(result.candidates),
            "n_trials": result.n_trials, "candidates": result.candidates}

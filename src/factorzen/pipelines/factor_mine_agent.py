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
    }


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_agent_mine(daily, *, n_rounds: int, seed: int, out_dir: str = "workspace/mine_agent",
                   llm_fn=None, top_k: int = 5, holdout_ratio: float = 0.2,
                   human_review: bool = False, run_id: str | None = None,
                   export: bool = True, patience: int | None = None,
                   heal_rounds: int = 2,
                   data_window: dict | None = None, command: str | None = None) -> dict:
    """跑单 Agent 挖掘闭环，每轮增量落 manifest，收尾导出候选。

    ``data_window``：``{start, end, universe, market}``。落进 manifest 的 params，
    否则事后无从得知这批因子挖自哪段数据、哪个票池（铁律#3）。
    ``command``：触发本次运行的命令行。
    """
    fn = llm_fn or _default_llm_fn()
    rid = run_id or f"agent_{seed}_{n_rounds}r_{_timestamp()}"
    params = {
        "n_rounds": n_rounds, "seed": seed, "top_k": top_k, "holdout_ratio": holdout_ratio,
        "patience": patience, "heal_rounds": heal_rounds,
        **(data_window or {}),
        "command": command,
        "llm": _llm_meta(llm_fn),
    }

    def _checkpoint(partial_result) -> None:
        """每轮末增量落盘：进程若在下一轮崩溃，已找到的候选不至于全损。"""
        write_session_manifest(partial_result, out_dir=out_dir, run_id=rid,
                               params=params, partial=True)

    result = run_llm_agent(daily, fn, n_rounds=n_rounds, seed=seed, top_k=top_k,
                           holdout_ratio=holdout_ratio, human_review=human_review,
                           patience=patience, heal_rounds=heal_rounds,
                           on_round_end=_checkpoint)
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
    return {"run_dir": str(run_dir), "n_candidates": len(result.candidates),
            "n_trials": result.n_trials, "candidates": result.candidates}

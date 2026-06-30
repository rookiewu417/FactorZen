# src/factorzen/pipelines/factor_mine_agent.py
"""LLM Agent 闭环挖掘 pipeline：跑 Agent → 落 manifest + 导出候选。"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from factorzen.agents.manifest import write_session_manifest
from factorzen.agents.orchestrator import run_llm_agent


def _default_llm_fn():
    """生产 LLMFn：包 request_chat + load_llm_config。"""
    from factorzen.llm.client import request_chat
    from factorzen.llm.config import load_llm_config
    config = load_llm_config(enabled=True)
    if not config.is_ready:
        raise RuntimeError("LLM 未配置：设置 .env 的 FACTORZEN_LLM_* 或注入 llm_fn")
    return lambda messages: request_chat(config, messages)


def run_agent_mine(daily, *, n_rounds: int, seed: int, out_dir: str = "workspace/mine_agent",
                   llm_fn=None, top_k: int = 5, holdout_ratio: float = 0.2,
                   human_review: bool = False, run_id: str | None = None,
                   export: bool = True) -> dict:
    fn = llm_fn or _default_llm_fn()
    result = run_llm_agent(daily, fn, n_rounds=n_rounds, seed=seed, top_k=top_k,
                           holdout_ratio=holdout_ratio, human_review=human_review)
    rid = run_id or f"agent_{seed}_{n_rounds}r"
    params = {"n_rounds": n_rounds, "seed": seed, "top_k": top_k, "holdout_ratio": holdout_ratio}
    write_session_manifest(result, out_dir=out_dir, run_id=rid, params=params)
    run_dir = Path(out_dir) / rid
    # candidates.csv —— 兼容 fz mine leaderboard（M1 读取格式）
    run_dir.mkdir(parents=True, exist_ok=True)
    cand_df = pl.DataFrame(result.candidates) if result.candidates else pl.DataFrame(
        {"expression": [], "holdout_ic": [], "dsr": []})
    cand_df.write_csv(run_dir / "candidates.csv")
    if export and result.candidates:
        from factorzen.discovery.export import export_candidate
        exp_dir = run_dir / "exported"
        exp_dir.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(result.candidates):
            export_candidate(c["expression"], f"agent_{rid}_{i}", str(exp_dir))
    return {"run_dir": str(run_dir), "n_candidates": len(result.candidates),
            "n_trials": result.n_trials, "candidates": result.candidates}

# src/factorzen/pipelines/factor_mine_team.py
"""多 Agent 团队挖掘 pipeline：跑 team → 落 team manifest + candidates.csv + 导出候选。"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from factorzen.agents.team_orchestrator import run_team_agent, write_team_manifest


def _default_llm_fn():
    """生产 LLMFn：包 request_chat + load_llm_config。"""
    from factorzen.llm.client import request_chat
    from factorzen.llm.config import load_llm_config

    config = load_llm_config(enabled=True)
    if not config.is_ready:
        raise RuntimeError("LLM 未配置：设置 .env 的 FACTORZEN_LLM_* 或注入 llm_fn")
    return lambda messages: request_chat(config, messages)


def run_team_mine(
    daily,
    *,
    n_rounds: int,
    seed: int,
    index_path: str,
    out_dir: str = "workspace/mine_team",
    llm_fn=None,
    top_k: int = 5,
    holdout_ratio: float = 0.2,
    run_id: str | None = None,
    export: bool = True,
) -> dict:
    """跑多 Agent 团队挖掘，落 team manifest + candidates.csv + 导出候选。

    Returns
    -------
    dict with keys: run_dir, n_candidates, n_trials, candidates
    """
    fn = llm_fn or _default_llm_fn()
    result = run_team_agent(
        daily, fn,
        n_rounds=n_rounds,
        seed=seed,
        index_path=index_path,
        top_k=top_k,
        holdout_ratio=holdout_ratio,
    )
    rid = run_id or f"team_{seed}_{n_rounds}r"
    params = {
        "n_rounds": n_rounds,
        "seed": seed,
        "top_k": top_k,
        "holdout_ratio": holdout_ratio,
        "index_path": index_path,
    }
    write_team_manifest(result, out_dir=out_dir, run_id=rid, params=params)
    run_dir = Path(out_dir) / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    # candidates.csv —— 兼容 fz mine leaderboard（M1 读取格式）
    cand_df = (
        pl.DataFrame(result.candidates)
        if result.candidates
        else pl.DataFrame({"expression": [], "holdout_ic": [], "dsr": []})
    )
    cand_df.write_csv(run_dir / "candidates.csv")
    if export and result.candidates:
        from factorzen.discovery.export import export_candidate

        exp_dir = run_dir / "exported"
        exp_dir.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(result.candidates):
            export_candidate(c["expression"], f"team_{rid}_{i}", str(exp_dir))
    return {
        "run_dir": str(run_dir),
        "n_candidates": len(result.candidates),
        "n_trials": result.n_trials,
        "candidates": result.candidates,
    }

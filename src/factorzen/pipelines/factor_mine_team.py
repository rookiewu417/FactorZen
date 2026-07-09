# src/factorzen/pipelines/factor_mine_team.py
"""多 Agent 团队挖掘 pipeline：跑 team → 落 team manifest + candidates.csv + 导出候选。"""
from __future__ import annotations

import shutil
from pathlib import Path

from factorzen.agents.team_orchestrator import run_team_agent, write_team_manifest
from factorzen.pipelines.factor_mine_agent import _llm_meta, _timestamp


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
    structured: bool = False,
    patience: int | None = None,
    heal_rounds: int = 2,
    data_window: dict | None = None,
    command: str | None = None,
) -> dict:
    """跑多 Agent 团队挖掘，每轮增量落 manifest，收尾写 candidates.csv + 导出候选。

    ``data_window``：``{start, end, universe, market}``；``command``：触发本次运行的命令行。
    二者落进 manifest 的 params，否则事后无从复现（铁律#3）。

    Returns
    -------
    dict with keys: run_dir, n_candidates, n_trials, candidates
    """
    fn = llm_fn or _default_llm_fn()
    rid = run_id or f"team_{seed}_{n_rounds}r_{_timestamp()}"
    params = {
        "n_rounds": n_rounds,
        "seed": seed,
        "top_k": top_k,
        "holdout_ratio": holdout_ratio,
        "index_path": index_path,
        "structured": structured,
        "patience": patience,
        "heal_rounds": heal_rounds,
        **(data_window or {}),
        "command": command,
        "llm": _llm_meta(llm_fn),
    }

    def _checkpoint(partial_result) -> None:
        """每轮末增量落盘：进程若在下一轮崩溃，已找到的候选不至于全损。"""
        write_team_manifest(partial_result, out_dir=out_dir, run_id=rid,
                            params=params, partial=True)

    result = run_team_agent(
        daily, fn,
        n_rounds=n_rounds,
        seed=seed,
        index_path=index_path,
        top_k=top_k,
        holdout_ratio=holdout_ratio,
        structured=structured,
        patience=patience,
        heal_rounds=heal_rounds,
        on_round_end=_checkpoint,
    )
    write_team_manifest(result, out_dir=out_dir, run_id=rid, params=params, partial=False)
    run_dir = Path(out_dir) / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    # candidates.csv —— 兼容 fz mine leaderboard/export-alpha（含 rank + passed 列）
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
                export_candidate(c["expression"], f"team_{rid}_{i}", str(exp_dir))
    return {
        "run_dir": str(run_dir),
        "n_candidates": len(result.candidates),
        "n_trials": result.n_trials,
        "candidates": result.candidates,
    }

# src/factorzen/pipelines/factor_mine_team.py
"""多 Agent 团队挖掘 pipeline：跑 team → 落 team manifest + candidates.csv + 导出候选。"""
from __future__ import annotations

import shutil
from pathlib import Path

from factorzen.agents.team_orchestrator import run_team_agent, write_team_manifest
from factorzen.pipelines.factor_mine_agent import (
    _llm_meta,
    _print_final_stats,
    _print_round_progress,
    _print_startup,
    _timestamp,
)


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
    eval_start: str | None = None,
    hypotheses_per_round: int = 1,
    profile=None,
    update_library: bool = True,
    library_orthogonal: bool = True,
    objective: str = "residual",
    llm_workers: int = 1,
    auto_lift: bool = True,
    lift_se_mult: float = 1.0,
) -> dict:
    """跑多 Agent 团队挖掘，每轮增量落 manifest，收尾写 candidates.csv + 导出候选。

    ``data_window``：``{start, end, universe, market}``；``command``：触发本次运行的命令行。
    二者落进 manifest 的 params，否则事后无从复现（铁律#3）。

    ``profile``：市场 profile（默认 None → A 股，零回归）。crypto 等传各自 profile，逐层透传到
    `run_team_agent`——数据装配（含预热前缀的 crypto daily）由调用方（CLI）负责。

    ``eval_start``：``"YYYYMMDD"``，训练段的干净起点。``daily`` 由 `prepare_mining_daily`
    带预热前缀，须把该前缀边界（= 挖掘窗口 ``start``）透传给 `run_team_agent`，否则预热段
    随 `split_holdout` 进 train IC。``None``（默认）退化为旧行为，对现有调用方零回归。

    Returns
    -------
    dict with keys: run_dir, n_candidates, n_trials, candidates
    """
    fn = llm_fn or _default_llm_fn()
    rid = run_id or f"{_timestamp()}_team_{seed}_{n_rounds}r"
    params = {
        "n_rounds": n_rounds,
        "seed": seed,
        "top_k": top_k,
        "holdout_ratio": holdout_ratio,
        "index_path": index_path,
        "structured": structured,
        "patience": patience,
        "heal_rounds": heal_rounds,
        "eval_start": eval_start,
        "hypotheses_per_round": hypotheses_per_round,
        "llm_workers": llm_workers,
        **(data_window or {}),
        "command": command,
        "llm": _llm_meta(llm_fn),
    }

    def _checkpoint(partial_result) -> None:
        """每轮末增量落盘 + 打印进度：进程若在下一轮崩溃，已找到的候选不至于全损。"""
        write_team_manifest(partial_result, out_dir=out_dir, run_id=rid,
                            params=params, partial=True)
        _print_round_progress(partial_result, label="mine-team")

    _print_startup(daily, params, label="mine-team", rid=rid)
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
        data_window=data_window,
        eval_start=eval_start,
        hypotheses_per_round=hypotheses_per_round,
        profile=profile,
        update_library=update_library,
        # 库根固定到 workspace/factor_library（out_dir=workspace/mine_team 的同级），
        # 不用 run_team_agent 从 index_path 推导的默认（那会落到 mine_team/factor_library）。
        library_root=str(Path(out_dir).parent / "factor_library"),
        library_orthogonal=library_orthogonal,
        objective=objective,
        llm_workers=llm_workers,
        auto_lift=auto_lift,
        lift_se_mult=lift_se_mult,
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
    _print_final_stats(result, str(run_dir), label="mine-team")
    return {
        "run_dir": str(run_dir),
        "n_candidates": len(result.candidates),
        "n_trials": result.n_trials,
        "candidates": result.candidates,
    }

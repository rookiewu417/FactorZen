"""Agent session manifest：把假设/表达式/分数/候选/参数全程落盘（可审计、可复现）。"""
from __future__ import annotations

import json
from pathlib import Path

from factorzen.core.experiment import get_git_sha


def write_session_manifest(
    result, *, out_dir: str, run_id: str, params: dict, partial: bool = False
) -> Path:
    """落 session manifest。

    ``partial=True`` 表示这是轮末的增量快照——挖掘尚未跑完，进程若在此后崩溃，
    留在磁盘上的就是它。消费方据此区分「跑完的结果」与「崩溃现场」。
    """
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state = result.state
    manifest = {
        "run_id": run_id, "seed": state.seed, "n_trials": result.n_trials,
        "iterations": state.iteration, "params": params,
        "partial": partial,
        "pbo": state.pbo,
        "attempts": [a.__dict__ for a in state.attempts],
        "candidates": result.candidates,
        "git_sha": get_git_sha(),
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return path

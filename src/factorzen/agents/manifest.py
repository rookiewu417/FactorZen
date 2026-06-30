"""Agent session manifest：把假设/表达式/分数/候选/参数全程落盘（可审计、可复现）。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_session_manifest(result, *, out_dir: str, run_id: str, params: dict) -> Path:
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state = result.state
    manifest = {
        "run_id": run_id, "seed": state.seed, "n_trials": result.n_trials,
        "iterations": state.iteration, "params": params,
        "attempts": [a.__dict__ for a in state.attempts],
        "candidates": result.candidates,
        "git_sha": _git_sha(),
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return path

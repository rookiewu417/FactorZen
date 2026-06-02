"""实验记录：写 manifest.json 到 workspace/factor_evaluations/{run_id}/"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from factorzen.config.settings import FACTOR_EVALUATIONS_DIR, ROOT

EXPERIMENTS_DIR = FACTOR_EVALUATIONS_DIR
_EXPERIMENT_INDEX = EXPERIMENTS_DIR / "experiment_index.jsonl"
_RUN_ID_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def _get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def _get_git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False


def _get_pixi_lock_hash() -> str:
    lock_path = ROOT / "pixi.lock"
    if not lock_path.exists():
        return "missing"
    return hashlib.sha256(lock_path.read_bytes()).hexdigest()


def _update_experiment_index(manifest: dict[str, Any], exp_dir: Path) -> None:
    """Append a one-line summary to experiment_index.jsonl for cross-run lookup."""
    config = manifest.get("config", {})
    entry: dict[str, Any] = {
        "run_id": manifest.get("run_id"),
        "timestamp": manifest.get("start_ts"),
        "factor": config.get("factor"),
        "universe": config.get("universe"),
        "start": config.get("start"),
        "end": config.get("end"),
        "status": manifest.get("status"),
        "manifest_path": str(exp_dir / "manifest.json"),
    }
    try:
        index_path = exp_dir.parent / "experiment_index.jsonl"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # index write failure must not affect the experiment outcome


def record_experiment_output(exp_dir: Path, key: str, value: str) -> None:
    """Record an output path in an existing experiment manifest."""
    manifest_path = exp_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("outputs", {})[key] = value
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _config_to_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump()
    if hasattr(config, "__dict__"):
        return vars(config)

    from factorzen.core.logger import get_logger

    _logger = get_logger(__name__)
    _logger.warning(
        "run_experiment: config type %s has no model_dump/__dict__, recording repr",
        type(config).__name__,
    )
    return {"repr": repr(config)}


def _safe_run_id_part(value: object) -> str:
    cleaned = _RUN_ID_SAFE_CHARS.sub("_", str(value)).strip("._-")
    return cleaned or "unknown"


@contextmanager
def run_experiment(
    config: Any,
    run_id: str | None = None,
    command: list[str] | None = None,
):
    """记录实验 manifest。config 可以是 RunConfig 或任意可序列化对象。

    Args:
        config: 运行配置，可以是 RunConfig 或任意含 model_dump()/__ dict__ 的对象。
        run_id: 实验 ID，None 时自动生成时间戳字符串。

    Yields:
        exp_dir: 实验输出目录 Path 对象。
    """
    config_dict = _config_to_dict(config)
    if run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        factor_name = config_dict.get("factor")
        run_id = (
            f"{_safe_run_id_part(factor_name)}_{timestamp}"
            if factor_name
            else timestamp
        )

    exp_dir = EXPERIMENTS_DIR / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    start_ts = datetime.now().isoformat()

    manifest: dict[str, Any] = {
        "schema_version": "1",
        "run_id": run_id,
        "git_sha": _get_git_sha(),
        "git_dirty": _get_git_dirty(),
        "pixi_lock_sha256": _get_pixi_lock_hash(),
        "command": command,
        "config": config_dict,
        "outputs": {},
        "start_ts": start_ts,
        "end_ts": None,
        "status": "running",
        "error": None,
    }

    manifest_path = exp_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    try:
        yield exp_dir
        manifest["end_ts"] = datetime.now().isoformat()
        manifest["status"] = "success"
    except Exception as exc:
        manifest["end_ts"] = datetime.now().isoformat()
        manifest["status"] = "failure"
        manifest["error"] = str(exc)
        raise
    finally:
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["outputs"] = existing.get("outputs", manifest.get("outputs", {}))
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        _update_experiment_index(manifest, exp_dir)


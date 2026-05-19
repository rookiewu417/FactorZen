"""实验记录：写 manifest.json 到 output/experiments/{run_id}/"""
from __future__ import annotations

import json
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

EXPERIMENTS_DIR = Path("output/experiments")


def _get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


@contextmanager
def run_experiment(config: Any, run_id: str | None = None):
    """记录实验 manifest。config 可以是 RunConfig 或任意可序列化对象。

    Args:
        config: 运行配置，可以是 RunConfig 或任意含 model_dump()/__ dict__ 的对象。
        run_id: 实验 ID，None 时自动生成时间戳字符串。

    Yields:
        exp_dir: 实验输出目录 Path 对象。
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_dir = EXPERIMENTS_DIR / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    start_ts = datetime.now().isoformat()
    if hasattr(config, "model_dump"):
        config_dict = config.model_dump()
    elif hasattr(config, "__dict__"):
        config_dict = dict(config.__dict__)
    else:
        config_dict = {}

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "git_sha": _get_git_sha(),
        "config": config_dict,
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
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

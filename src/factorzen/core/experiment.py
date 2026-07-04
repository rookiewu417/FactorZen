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
from factorzen.core.logger import get_logger

logger = get_logger(__name__)

EXPERIMENTS_DIR = FACTOR_EVALUATIONS_DIR
_EXPERIMENT_INDEX = EXPERIMENTS_DIR / "experiment_index.jsonl"
_RUN_ID_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def get_git_sha() -> str:
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


# run_experiment 自身管理的标准字段;其余键视为运行期写入的元数据,finally 时予以保留。
_MANAGED_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "git_sha",
        "git_dirty",
        "pixi_lock_sha256",
        "command",
        "config",
        "outputs",
        "start_ts",
        "end_ts",
        "duration_seconds",
        "status",
        "error",
    }
)


def record_experiment_metadata(exp_dir: Path, key: str, value: Any) -> None:
    """Record an arbitrary top-level metadata key in an existing manifest.

    与 ``record_experiment_output``(写入 ``outputs`` 子字典)不同,此处写入顶层键,
    适合运行期产生的观测数据(如 ``stage_timings``)。run_experiment 的 finally
    会保留这些键。
    """
    manifest_path = exp_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[key] = value
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _config_to_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "model_dump"):
        return config.model_dump()
    if hasattr(config, "__dict__"):
        return vars(config)

    logger.warning(
        "run_experiment: config type %s has no model_dump/__dict__, recording repr",
        type(config).__name__,
    )
    return {"repr": repr(config)}


def _safe_run_id_part(value: object) -> str:
    cleaned = _RUN_ID_SAFE_CHARS.sub("_", str(value)).strip("._-")
    return cleaned or "unknown"


def build_manifest_base(
    command: list[str] | None,
    config: Any,
    *,
    start_dt: datetime | None = None,
) -> dict[str, Any]:
    """构造 manifest 中与可复现性相关的基础字段。

    与 ``run_experiment`` 解耦：供不便走完整 run_experiment() 流程（manifest 目录结构/
    run_id 生成约定与自身耦合较深）的 pipeline（如 risk_build/portfolio_build）独立复用，
    避免各自重复手写 ``_git_sha()`` 之类精简版逻辑，导致 manifest 缺 command/git_dirty/
    pixi_lock_sha256/schema_version 等可复现性字段。

    Args:
        command: 触发本次运行的命令行（如 ``sys.argv``），不可得时传 None。
        config: 运行配置；可以是 RunConfig 等带 model_dump()/__dict__ 的对象，也可以是
            调用方已自行拼好的 plain dict（pipeline 自身参数集）。
        start_dt: 记录的起始时间；为 None 时取 ``datetime.now()``。

    Returns:
        含 schema_version/git_sha/git_dirty/pixi_lock_sha256/command/config/start_ts 的 dict。
    """
    if start_dt is None:
        start_dt = datetime.now()
    return {
        "schema_version": "1",
        "git_sha": get_git_sha(),
        "git_dirty": _get_git_dirty(),
        "pixi_lock_sha256": _get_pixi_lock_hash(),
        "command": command,
        "config": _config_to_dict(config),
        "start_ts": start_dt.isoformat(),
    }


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

    start_dt = datetime.now()
    base = build_manifest_base(command, config_dict, start_dt=start_dt)

    manifest: dict[str, Any] = {
        "schema_version": base["schema_version"],
        "run_id": run_id,
        "git_sha": base["git_sha"],
        "git_dirty": base["git_dirty"],
        "pixi_lock_sha256": base["pixi_lock_sha256"],
        "command": base["command"],
        "config": base["config"],
        "outputs": {},
        "start_ts": base["start_ts"],
        "end_ts": None,
        "duration_seconds": None,
        "status": "running",
        "error": None,
    }

    if manifest["git_dirty"]:
        logger.warning(
            "git_dirty=true：工作树存在未提交改动，本次运行无法仅凭 git SHA 复现；已记录到 manifest。"
        )

    manifest_path = exp_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    try:
        yield exp_dir
        manifest["status"] = "success"
    except Exception as exc:
        manifest["status"] = "failure"
        manifest["error"] = str(exc)
        raise
    finally:
        end_dt = datetime.now()
        manifest["end_ts"] = end_dt.isoformat()
        manifest["duration_seconds"] = round((end_dt - start_dt).total_seconds(), 3)
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["outputs"] = existing.get("outputs", manifest.get("outputs", {}))
            # 保留运行期通过 record_experiment_metadata 写入的顶层元数据
            for key, value in existing.items():
                if key not in _MANAGED_MANIFEST_KEYS:
                    manifest[key] = value
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        _update_experiment_index(manifest, exp_dir)


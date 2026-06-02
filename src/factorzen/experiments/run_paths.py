"""Workspace run artifact layout."""

from __future__ import annotations

import shutil
from pathlib import Path

from factorzen.config.settings import FACTOR_EVALUATIONS_DIR, ROOT

WORKSPACE_DIR = ROOT / "workspace"

STANDARD_ARTIFACT_NAMES = {
    "factor": "factor.parquet",
    "ic": "ic.parquet",
    "quality_report": "quality.json",
    "walk_forward_summary": "walk_forward.json",
    "universe_snapshot": "universe.parquet",
    "report": "report.html",
    "llm_explanation": "llm_explanation.json",
    "meta": "meta.json",
}


def run_dir(run_id: str) -> Path:
    """Return the self-contained workspace run directory for a run id."""
    return FACTOR_EVALUATIONS_DIR / run_id


def standard_artifact_name(key: str, source: Path) -> str:
    """Return the stable filename used inside a run directory."""
    return STANDARD_ARTIFACT_NAMES.get(key, source.name)


def copy_outputs_to_run_dir(outputs: dict[str, str], destination: Path) -> dict[str, str]:
    """Copy produced artifacts into a run directory with stable names."""
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for key, raw_path in outputs.items():
        source = Path(raw_path)
        if not source.exists() or not source.is_file():
            continue
        target = destination / standard_artifact_name(key, source)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied[f"run_{key}"] = str(target)
    return copied

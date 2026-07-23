"""Workspace run artifact layout.

评估产物落点：
- 有因子名 → ``workspace/factors/<market>/<name>/evaluations/{run_id}/``
- 无因子名 → ``workspace/factors/_runs/{run_id}/``
旧 ``runs/artifacts/daily/`` 中间层与 ``copy_outputs_to_run_dir`` 双写已废除。
"""

from __future__ import annotations

import re
from pathlib import Path

from factorzen.config.settings import FACTOR_STORE_DIR, ROOT

WORKSPACE_DIR = ROOT / "workspace"

STANDARD_ARTIFACT_NAMES = {
    "factor": "factor.parquet",
    "ic": "ic.parquet",
    "quality_report": "quality.json",
    "walk_forward_summary": "walk_forward.json",
    "signal": "signal.json",
    "signal_group_nav": "signal_group_nav.parquet",
    "universe_snapshot": "universe.parquet",
    "report": "report.html",
    "llm_explanation": "llm_explanation.json",
    "meta": "meta.json",
}

# 因子数值面板 schema（唯一落点：factors；evaluations 不落 parquet）。
FACTOR_PANEL_COLUMNS = (
    "trade_date",
    "ts_code",
    "factor_value",
    "factor_clean",
)
# 兼容旧名
EVAL_FACTOR_PANEL_COLUMNS = FACTOR_PANEL_COLUMNS

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_TS_SUFFIX_RE = re.compile(r"_(\d{8}_\d{6})$")
_SAFE_PART = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_part(value: object) -> str:
    cleaned = _SAFE_PART.sub("_", str(value)).strip("._-")
    return cleaned or "unknown"


def run_dir(
    run_id: str,
    *,
    factor: str | None = None,
    market: str = "ashare",
) -> Path:
    """Return the workspace run directory for a run id（新布局构造，不查磁盘）。

    - ``factor`` 非空 → ``factors/<market>/<factor>/evaluations/<run_id>``
    - ``factor`` 为 None → 用正则 ``_\\d{8}_\\d{6}$`` 从 run_id 剥因子名试新布局；
      剥不出 → ``factors/_runs/<run_id>``
    """
    if factor:
        return (
            FACTOR_STORE_DIR
            / market
            / _safe_part(factor)
            / "evaluations"
            / run_id
        )
    m = _TS_SUFFIX_RE.search(run_id)
    if m:
        factor_part = run_id[: m.start()]
        if factor_part:
            return (
                FACTOR_STORE_DIR
                / market
                / _safe_part(factor_part)
                / "evaluations"
                / run_id
            )
    return FACTOR_STORE_DIR / "_runs" / run_id


def find_run_dir(run_id: str) -> Path | None:
    """按嵌套布局查找已存在的 run 目录。

    扫描 ``factors/*/*/evaluations/<run_id>`` 与 ``factors/_runs/<run_id>``。
    ``run_id`` 先过 ``[A-Za-z0-9_.-]+`` 校验防路径遍历；找不到返回 None。
    """
    if not _RUN_ID_RE.fullmatch(run_id):
        return None
    root = FACTOR_STORE_DIR
    # factors/_runs/<run_id>
    candidate = root / "_runs" / run_id
    if candidate.is_dir():
        return candidate
    # factors/<market>/<name>/evaluations/<run_id>
    if not root.is_dir():
        return None
    try:
        markets = [p for p in root.iterdir() if p.is_dir() and p.name not in ("reports", "_runs")]
    except OSError:
        return None
    for market_dir in markets:
        try:
            names = [p for p in market_dir.iterdir() if p.is_dir()]
        except OSError:
            continue
        for name_dir in names:
            d = name_dir / "evaluations" / run_id
            if d.is_dir():
                return d
    return None


def standard_artifact_name(key: str, source: Path | None = None) -> str:
    """Return the stable filename used inside a run directory."""
    if key in STANDARD_ARTIFACT_NAMES:
        return STANDARD_ARTIFACT_NAMES[key]
    if source is not None:
        return source.name
    return key


def artifact_path(destination: Path, key: str) -> Path:
    """``factors/.../evaluations/{run_id}/{stable_name}``。"""
    return destination / standard_artifact_name(key)

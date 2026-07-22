"""workspace 产物的只读索引。

扫描各域 `<workspace>/<domain>/<run_id>/manifest.json` 建索引,读 metrics/nav 供
API 与 Dashboard 消费。损坏/缺字段的 manifest 跳过并记 warning,绝不因单个坏产物炸接口。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.core.logger import get_logger

logger = get_logger("factorzen.server.artifacts")

DOMAINS = [
    "factor_evaluations",
    "mining_sessions",
    "portfolios",
    "sim",
    "execution",
    "combinations",
    "combine_backtests",  # 天然带 nav.parquet 的回测域
    "mine_team",
    # risk_models：目前仅有少量 manifest、无 nav；有产物再收
]


class ArtifactIndex:
    """只读产物索引(零侵入:不触发计算)。"""

    def __init__(self, workspace_dir: str | Path) -> None:
        self.root = Path(workspace_dir)

    def list_runs(self, domain: str) -> list[dict[str, Any]]:
        base = self.root / domain
        out: list[dict[str, Any]] = []
        if not base.exists():
            return out
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            mani = d / "manifest.json"
            if not mani.exists():
                continue
            try:
                m = json.loads(mani.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"[artifacts] 跳过损坏 manifest {mani}: {exc}")
                continue
            if not isinstance(m, dict):
                continue
            out.append(
                {
                    "run_id": d.name,
                    "domain": domain,
                    "git_sha": m.get("git_sha"),
                    "status": m.get("status"),
                    "manifest": m,
                }
            )
        return out

    def _safe_run_dir(self, domain: str, run_id: str) -> Path:
        """校验 domain 白名单 + run_id 无路径遍历，返回安全的 run 目录。

        非白名单 domain、或 run_id 含 ../ 等导致逃出 <root>/<domain> 时 raise
        FileNotFoundError（防路径遍历读到 workspace 外的任意文件）。
        """
        if domain not in DOMAINS:
            raise FileNotFoundError(f"未知 domain: {domain}")
        base = (self.root / domain).resolve()
        target = (base / run_id).resolve()
        if target.parent != base or not target.is_relative_to(base):
            raise FileNotFoundError(f"非法 run_id: {run_id}")
        return target

    def run_detail(self, domain: str, run_id: str) -> dict[str, Any]:
        d = self._safe_run_dir(domain, run_id)
        mani = d / "manifest.json"
        if not mani.exists():
            raise FileNotFoundError(f"产物不存在: {domain}/{run_id}")
        detail: dict[str, Any] = {
            "run_id": run_id,
            "domain": domain,
            "manifest": json.loads(mani.read_text(encoding="utf-8")),
        }
        metrics_f = d / "metrics.json"
        if metrics_f.exists():
            try:
                detail["metrics"] = json.loads(metrics_f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning(f"[artifacts] metrics.json 损坏: {metrics_f}")
        return detail

    def nav_series(self, domain: str, run_id: str) -> list[tuple[str, float]]:
        try:
            d = self._safe_run_dir(domain, run_id)
        except FileNotFoundError:
            return []
        nav_f = d / "nav.parquet"
        if not nav_f.exists():
            return []
        try:
            df = pl.read_parquet(nav_f)
        except Exception as exc:
            logger.warning(f"[artifacts] nav.parquet 读取失败 {nav_f}: {exc}")
            return []
        cols = df.columns
        date_col = next(
            (c for c in ("as_of_date", "trade_date", "date") if c in cols), cols[0]
        )
        nav_col = next(
            (c for c in ("nav_after", "nav", "value") if c in cols), cols[-1]
        )
        return [
            (str(r[date_col]), float(r[nav_col])) for r in df.iter_rows(named=True)
        ]

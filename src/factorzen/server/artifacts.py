"""workspace 产物的只读索引。

扫描各域 `<workspace>/<domain>/<run_id>/manifest.json` 建索引,读 metrics/nav 供
API 与 Dashboard 消费。损坏/缺字段的 manifest 跳过并记 warning,绝不因单个坏产物炸接口。

**特例**：API domain 名 ``factor_evaluations`` 保持不变（前端零改动），但磁盘落点为
嵌套布局 ``factors/*/*/evaluations/<run_id>/`` 与 ``factors/_runs/<run_id>/``。
"""
from __future__ import annotations

import json
import re
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
    "strategies",  # 规则型策略预置权重回测
    "mine_agent",  # LLM 单 Agent 挖掘
    "risk_models",  # Barra 风格/行业暴露与协方差
]

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class ArtifactIndex:
    """只读产物索引(零侵入:不触发计算)。"""

    def __init__(self, workspace_dir: str | Path) -> None:
        self.root = Path(workspace_dir)

    def _factors_root(self) -> Path:
        return self.root / "factors"

    def _list_factor_evaluation_runs(self) -> list[dict[str, Any]]:
        """扫 factors/*/*/evaluations/*/manifest.json + factors/_runs/*/manifest.json。

        跳过 factors/reports；市场层/因子层只认目录。
        """
        out: list[dict[str, Any]] = []
        factors = self._factors_root()
        if not factors.exists():
            return out

        # factors/_runs/<run_id>
        runs_root = factors / "_runs"
        if runs_root.is_dir():
            try:
                run_dirs = sorted(p for p in runs_root.iterdir() if p.is_dir())
            except OSError as exc:
                logger.warning(f"[artifacts] 列举 _runs 失败 {runs_root}: {exc}")
                run_dirs = []
            for d in run_dirs:
                self._append_run_if_valid(out, d, domain="factor_evaluations")

        # factors/<market>/<name>/evaluations/<run_id>
        try:
            market_dirs = sorted(
                p
                for p in factors.iterdir()
                if p.is_dir() and p.name not in ("reports", "_runs")
            )
        except OSError as exc:
            logger.warning(f"[artifacts] 列举 factors 失败 {factors}: {exc}")
            return out

        for market_dir in market_dirs:
            try:
                name_dirs = sorted(p for p in market_dir.iterdir() if p.is_dir())
            except OSError:
                continue
            for name_dir in name_dirs:
                evals = name_dir / "evaluations"
                if not evals.is_dir():
                    continue
                try:
                    run_dirs = sorted(p for p in evals.iterdir() if p.is_dir())
                except OSError:
                    continue
                for d in run_dirs:
                    self._append_run_if_valid(out, d, domain="factor_evaluations")
        return out

    def _append_run_if_valid(
        self, out: list[dict[str, Any]], d: Path, *, domain: str
    ) -> None:
        mani = d / "manifest.json"
        if not mani.exists():
            return
        try:
            m = json.loads(mani.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"[artifacts] 跳过损坏 manifest {mani}: {exc}")
            return
        if not isinstance(m, dict):
            return
        out.append(
            {
                "run_id": d.name,
                "domain": domain,
                "git_sha": m.get("git_sha"),
                "status": m.get("status"),
                "manifest": m,
            }
        )

    def list_runs(self, domain: str) -> list[dict[str, Any]]:
        if domain == "factor_evaluations":
            return self._list_factor_evaluation_runs()

        base = self.root / domain
        out: list[dict[str, Any]] = []
        if not base.exists():
            return out
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            self._append_run_if_valid(out, d, domain=domain)
        return out

    def overview(self) -> list[dict[str, Any]]:
        """汇总各域产物数量与最新 run。

        遍历 DOMAINS；latest 取 list_runs 目录名排序后的最后一项
        （坏 manifest 已在 list_runs 中跳过）。无产物时 latest 为 None。
        """
        result: list[dict[str, Any]] = []
        for domain in DOMAINS:
            runs = self.list_runs(domain)
            latest: dict[str, Any] | None = None
            if runs:
                last = runs[-1]
                latest = {
                    "run_id": last["run_id"],
                    "status": last["status"],
                    "git_sha": last["git_sha"],
                }
            result.append(
                {
                    "domain": domain,
                    "count": len(runs),
                    "latest": latest,
                }
            )
        return result

    def _find_factor_eval_run_dir(self, run_id: str) -> Path | None:
        """定位 factor_evaluations 域的 run 目录（与 find_run_dir 同逻辑）。"""
        if not _RUN_ID_RE.fullmatch(run_id):
            return None
        factors = self._factors_root()
        candidate = factors / "_runs" / run_id
        if candidate.is_dir():
            return candidate
        if not factors.is_dir():
            return None
        try:
            market_dirs = [
                p
                for p in factors.iterdir()
                if p.is_dir() and p.name not in ("reports", "_runs")
            ]
        except OSError:
            return None
        for market_dir in market_dirs:
            try:
                name_dirs = [p for p in market_dir.iterdir() if p.is_dir()]
            except OSError:
                continue
            for name_dir in name_dirs:
                d = name_dir / "evaluations" / run_id
                if d.is_dir():
                    return d
        return None

    def _safe_run_dir(self, domain: str, run_id: str) -> Path:
        """校验 domain 白名单 + run_id 无路径遍历，返回安全的 run 目录。

        非白名单 domain、或 run_id 含 ../ 等导致逃出域根时 raise
        FileNotFoundError（防路径遍历读到 workspace 外的任意文件）。
        """
        if domain not in DOMAINS:
            raise FileNotFoundError(f"未知 domain: {domain}")

        if domain == "factor_evaluations":
            if not _RUN_ID_RE.fullmatch(run_id):
                raise FileNotFoundError(f"非法 run_id: {run_id}")
            found = self._find_factor_eval_run_dir(run_id)
            if found is None:
                raise FileNotFoundError(f"产物不存在: {domain}/{run_id}")
            # 必须落在 factors 根下
            factors = self._factors_root().resolve()
            target = found.resolve()
            if not target.is_relative_to(factors):
                raise FileNotFoundError(f"非法 run_id: {run_id}")
            return target

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
            # workspace 相对真实路径：平铺域 = <domain>/<run_id>；
            # factor_evaluations = factors/<market>/<name>/evaluations/<run_id>。
            # 前端产物列表/直开链接用它，不再手拼 <domain>/<run_id>。
            # d 已 resolve，root 需同样 resolve（root 可能以相对路径传入）。
            "path": str(d.relative_to(self.root.resolve())),
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

"""运营任务与报告的只读索引。

扫描 workspace/_ops 与 workspace/factors/reports，供 REST API 消费。
损坏/缺文件跳过并记 warning，绝不因单个坏产物炸接口。
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from factorzen.core.logger import get_logger

logger = get_logger("factorzen.server.opsview")

# 报告允许的扩展名
REPORT_EXTENSIONS = frozenset({".json", ".md", ".html", ".txt"})

# 报告文件读取大小上限（字节）；测试可 monkeypatch
REPORT_MAX_BYTES = 1_000_000

# 报告递归深度上限
REPORT_MAX_DEPTH = 3


def _mtime_iso(path: Path) -> str:
    """目录/文件 mtime 转 ISO 字符串（UTC）。"""
    try:
        ts = path.stat().st_mtime
    except OSError:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class OpsViewIndex:
    """运营 + 报告只读索引（零侵入）。"""

    def __init__(self, workspace_dir: str | Path) -> None:
        self.root = Path(workspace_dir)
        self.ops_root = self.root / "_ops"
        self.campaigns_root = self.ops_root / "campaigns"
        self.reports_root = self.root / "factors" / "reports"

    def list_campaigns(self) -> dict[str, Any]:
        """列出 _ops/campaigns 下各 campaign 摘要。"""
        campaigns: list[dict[str, Any]] = []
        base = self.campaigns_root
        if not base.exists():
            return {"campaigns": []}

        try:
            dirs = sorted(p for p in base.iterdir() if p.is_dir())
        except OSError as exc:
            logger.warning(f"[opsview] 列举 campaigns 失败 {base}: {exc}")
            return {"campaigns": []}

        for d in dirs:
            done = (d / "done").exists()
            exitcode: str | None = None
            ec_path = d / "exitcode"
            if ec_path.exists():
                try:
                    exitcode = ec_path.read_text(encoding="utf-8").strip() or None
                except OSError as exc:
                    logger.warning(f"[opsview] 读取 exitcode 失败 {ec_path}: {exc}")

            command: str | None = None
            cmd_path = d / "command.txt"
            if cmd_path.exists():
                try:
                    raw = cmd_path.read_text(encoding="utf-8", errors="replace")
                    command = raw[:200] if raw else None
                except OSError as exc:
                    logger.warning(f"[opsview] 读取 command.txt 失败 {cmd_path}: {exc}")

            campaigns.append(
                {
                    "name": d.name,
                    "done": done,
                    "exitcode": exitcode,
                    "mtime": _mtime_iso(d),
                    "command": command,
                }
            )

        # 按 mtime 降序（新的在前）
        campaigns.sort(key=lambda c: c.get("mtime") or "", reverse=True)
        return {"campaigns": campaigns}

    def _safe_campaign_dir(self, name: str) -> Path:
        """校验 campaign name 无路径遍历。"""
        base = self.campaigns_root.resolve()
        target = (base / name).resolve()
        if not target.is_relative_to(base) or target == base:
            raise FileNotFoundError(f"非法 campaign name: {name}")
        if target.parent != base:
            raise FileNotFoundError(f"非法 campaign name: {name}")
        return target

    def campaign_log(self, name: str, tail: int = 200) -> dict[str, Any]:
        """取 campaign 目录下最新 *.log 的尾部 N 行。"""
        if tail < 1:
            tail = 1
        if tail > 2000:
            tail = 2000

        d = self._safe_campaign_dir(name)
        if not d.is_dir():
            raise FileNotFoundError(f"campaign 不存在: {name}")

        log_file: str | None = None
        lines: list[str] = []

        try:
            logs = sorted(
                (p for p in d.iterdir() if p.is_file() and p.suffix == ".log"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError as exc:
            logger.warning(f"[opsview] 列举 log 失败 {d}: {exc}")
            return {"name": name, "log_file": None, "lines": []}

        if not logs:
            return {"name": name, "log_file": None, "lines": []}

        latest = logs[0]
        log_file = latest.name
        try:
            text = latest.read_text(encoding="utf-8", errors="replace")
            all_lines = text.splitlines()
            lines = all_lines[-tail:] if tail < len(all_lines) else all_lines
        except OSError as exc:
            logger.warning(f"[opsview] 读取 log 失败 {latest}: {exc}")

        return {"name": name, "log_file": log_file, "lines": lines}

    def list_reports(self) -> dict[str, Any]:
        """递归列出 reports 下允许扩展名的文件（深度 ≤ REPORT_MAX_DEPTH）。"""
        files: list[dict[str, Any]] = []
        base = self.reports_root
        if not base.exists():
            return {"files": []}

        base_resolved = base.resolve()

        def _walk(current: Path, depth: int) -> None:
            if depth > REPORT_MAX_DEPTH:
                return
            try:
                children = sorted(current.iterdir(), key=lambda p: p.name)
            except OSError as exc:
                logger.warning(f"[opsview] 列举 reports 失败 {current}: {exc}")
                return
            for p in children:
                try:
                    if p.is_dir():
                        _walk(p, depth + 1)
                    elif p.is_file() and p.suffix.lower() in REPORT_EXTENSIONS:
                        rel = p.resolve().relative_to(base_resolved)
                        # 相对路径用 POSIX 风格
                        rel_str = rel.as_posix()
                        st = p.stat()
                        files.append(
                            {
                                "path": rel_str,
                                "size": st.st_size,
                                "mtime": _mtime_iso(p),
                            }
                        )
                except (OSError, ValueError) as exc:
                    logger.warning(f"[opsview] 跳过报告文件 {p}: {exc}")

        _walk(base, 0)
        files.sort(key=lambda f: f.get("mtime") or "", reverse=True)
        return {"files": files}

    def _safe_report_path(self, rel_path: str) -> Path:
        """校验相对路径无遍历，且扩展名合法。"""
        if not rel_path or rel_path.startswith("/") or "\\" in rel_path:
            raise FileNotFoundError(f"非法 path: {rel_path}")
        # 规范化：去掉 leading ./
        cleaned = rel_path.lstrip("./")
        if not cleaned or ".." in Path(cleaned).parts:
            raise FileNotFoundError(f"非法 path: {rel_path}")

        base = self.reports_root.resolve()
        target = (base / cleaned).resolve()
        if not target.is_relative_to(base) or target == base:
            raise FileNotFoundError(f"非法 path: {rel_path}")

        suffix = target.suffix.lower()
        if suffix not in REPORT_EXTENSIONS:
            raise FileNotFoundError(f"不允许的扩展名: {suffix}")

        return target

    def read_report(self, rel_path: str) -> dict[str, Any]:
        """读取报告文本内容；超限 raise PermissionError 供 API 转 413。"""
        target = self._safe_report_path(rel_path)
        if not target.is_file():
            raise FileNotFoundError(f"报告不存在: {rel_path}")

        try:
            size = target.stat().st_size
        except OSError as exc:
            raise FileNotFoundError(f"报告不可读: {rel_path}") from exc

        if size > REPORT_MAX_BYTES:
            raise PermissionError(
                f"报告过大 ({size} > {REPORT_MAX_BYTES} bytes): {rel_path}"
            )

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise FileNotFoundError(f"报告读取失败: {rel_path}") from exc

        return {
            "path": rel_path,
            "size": size,
            "content": content,
        }

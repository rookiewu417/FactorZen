"""后台 CLI / 脚本任务管理（Web UI 任务中心底座）。

生产 jobs 目录：``WORKSPACE_DIR / "_ops" / "webui_jobs"``；测试可注入 tmp。
命令白名单：kind=cli 仅走 ``python -m factorzen.cli.main``；kind=script 仅
``workspace/configs/*.py``。不经 shell。
"""
from __future__ import annotations

import contextlib
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from factorzen.config.settings import ROOT
from factorzen.core.logger import get_logger
from factorzen.server.files import _normalize_rel

logger = get_logger("factorzen.server.jobs")


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _new_job_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{secrets.token_hex(3)}"


class JobManager:
    """提交 / 列表 / 详情 / 日志 / 终止后台任务。"""

    def __init__(
        self,
        jobs_dir: str | Path,
        *,
        workspace_dir: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        self.jobs_dir = Path(jobs_dir)
        self.workspace_dir = Path(workspace_dir) if workspace_dir is not None else None
        self.project_root = Path(project_root) if project_root is not None else ROOT
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    # ---- 命令白名单（可测缝：测试可 monkeypatch ``_build_command``）----

    def _build_command(self, kind: str, argv: list[str]) -> list[str]:
        """按 kind 构造真实可执行命令列表。

        测试可 monkeypatch 本方法，注入 ``[sys.executable, "-c", "..."]`` 级轻命令。
        """
        if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
            raise ValueError("argv 必须是 str 列表")
        for a in argv:
            if "\x00" in a:
                raise ValueError("argv 项不得含空字节")

        if kind == "cli":
            return [sys.executable, "-m", "factorzen.cli.main", *argv]

        if kind == "script":
            if not argv:
                raise ValueError("script 需要 argv[0] 为 configs/ 下 .py 路径")
            if self.workspace_dir is None:
                raise ValueError("JobManager 未配置 workspace_dir，无法校验 script 路径")
            rel = argv[0]
            cleaned = _normalize_rel(rel)
            if cleaned is None or cleaned == "":
                raise ValueError(f"非法 script 路径: {rel!r}")
            parts = Path(cleaned).parts
            if not parts or parts[0] != "configs":
                raise ValueError(f"script 必须位于 configs/ 下: {rel!r}")
            if Path(cleaned).suffix.lower() != ".py":
                raise ValueError(f"script 必须是 .py 文件: {rel!r}")

            ws = self.workspace_dir.resolve()
            configs_root = (ws / "configs").resolve()
            target = (ws / cleaned).resolve()
            if not target.is_relative_to(configs_root):
                raise ValueError(f"script 路径逃逸 configs/: {rel!r}")
            if not target.is_file():
                raise ValueError(f"script 文件不存在: {rel!r}")
            return [sys.executable, str(target), *argv[1:]]

        raise ValueError(f"未知 kind: {kind!r}（仅支持 cli / script）")

    def _safe_job_dir(self, job_id: str) -> Path:
        """校验 job_id 无路径遍历，返回 job 目录。"""
        if not job_id or not isinstance(job_id, str):
            raise FileNotFoundError(f"非法 job_id: {job_id!r}")
        if "/" in job_id or "\\" in job_id or ".." in job_id:
            raise FileNotFoundError(f"非法 job_id: {job_id!r}")
        # 宽松：允许非标准 id（测试伪造 orphaned），但禁止逃逸
        base = self.jobs_dir.resolve()
        target = (base / job_id).resolve()
        if target.parent != base or not target.is_relative_to(base):
            raise FileNotFoundError(f"非法 job_id: {job_id!r}")
        return target

    def submit(self, argv: list[str], kind: str, title: str) -> dict[str, Any]:
        """提交后台任务，立即返回 meta。"""
        if kind not in ("cli", "script"):
            raise ValueError(f"未知 kind: {kind!r}")
        cmd = self._build_command(kind, argv)

        job_id = _new_job_id()
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)

        log_path = job_dir / "job.log"
        log_f = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 — 由 waiter 关闭
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            log_f.close()
            raise

        started_at = _now_iso()
        meta: dict[str, Any] = {
            "job_id": job_id,
            "kind": kind,
            "title": title,
            "argv": list(argv),
            "pid": proc.pid,
            "started_at": started_at,
        }
        (job_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        def _wait_and_finish() -> None:
            exit_code: int | None = None
            try:
                exit_code = proc.wait()
            except Exception as exc:
                logger.warning(f"[jobs] wait 失败 {job_id}: {exc}")
                exit_code = -1
            finally:
                with contextlib.suppress(Exception):
                    log_f.close()
            status = {
                "exit_code": exit_code,
                "ended_at": _now_iso(),
            }
            try:
                (job_dir / "status.json").write_text(
                    json.dumps(status, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError as exc:
                logger.warning(f"[jobs] 写 status.json 失败 {job_id}: {exc}")

        t = threading.Thread(target=_wait_and_finish, name=f"job-{job_id}", daemon=True)
        t.start()
        return meta

    def _pid_alive(self, pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # 进程存在但无权信号 → 视为存活
            return True
        except OSError:
            return False

    def _job_status(self, job_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
        """合并 meta 与运行状态。"""
        out = dict(meta)
        status_path = job_dir / "status.json"
        if status_path.is_file():
            try:
                st = json.loads(status_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"[jobs] 读 status.json 失败 {status_path}: {exc}")
                st = {}
            out["status"] = "finished"
            out["exit_code"] = st.get("exit_code")
            out["ended_at"] = st.get("ended_at")
            return out

        pid = meta.get("pid")
        if isinstance(pid, int) and self._pid_alive(pid):
            out["status"] = "running"
            out["exit_code"] = None
            out["ended_at"] = None
            return out

        out["status"] = "orphaned"
        out["exit_code"] = None
        out["ended_at"] = None
        return out

    def list_jobs(self) -> list[dict[str, Any]]:
        """扫描 jobs_dir，新在前。"""
        if not self.jobs_dir.exists():
            return []
        items: list[dict[str, Any]] = []
        try:
            dirs = [p for p in self.jobs_dir.iterdir() if p.is_dir()]
        except OSError as exc:
            logger.warning(f"[jobs] 列举失败 {self.jobs_dir}: {exc}")
            return []

        # 目录名含时间戳，降序即新在前
        for d in sorted(dirs, key=lambda p: p.name, reverse=True):
            meta_path = d / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"[jobs] 跳过损坏 meta {meta_path}: {exc}")
                continue
            if not isinstance(meta, dict):
                continue
            items.append(self._job_status(d, meta))
        return items

    def job_detail(self, job_id: str) -> dict[str, Any]:
        """单任务详情（meta + 状态）。"""
        job_dir = self._safe_job_dir(job_id)
        meta_path = job_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"job 不存在: {job_id}")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise FileNotFoundError(f"job meta 损坏: {job_id}") from exc
        if not isinstance(meta, dict):
            raise FileNotFoundError(f"job meta 无效: {job_id}")
        return self._job_status(job_dir, meta)

    def job_log(self, job_id: str, tail: int = 200) -> dict[str, Any]:
        """取 job.log 尾部 N 行（上限 2000）。"""
        if tail < 1:
            tail = 1
        if tail > 2000:
            tail = 2000

        job_dir = self._safe_job_dir(job_id)
        if not (job_dir / "meta.json").is_file():
            raise FileNotFoundError(f"job 不存在: {job_id}")

        log_path = job_dir / "job.log"
        lines: list[str] = []
        if log_path.is_file():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                all_lines = text.splitlines()
                lines = all_lines[-tail:] if tail < len(all_lines) else all_lines
            except OSError as exc:
                logger.warning(f"[jobs] 读 log 失败 {log_path}: {exc}")

        return {"job_id": job_id, "lines": lines}

    def kill(self, job_id: str) -> dict[str, Any]:
        """终止 running 任务（向进程组发 SIGTERM）。"""
        detail = self.job_detail(job_id)
        if detail.get("status") != "running":
            raise RuntimeError(f"job 非 running，无法终止: {job_id} status={detail.get('status')}")

        pid = detail.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            raise RuntimeError(f"job pid 无效: {job_id}")

        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            logger.warning(f"[jobs] kill 时进程已消失 {job_id} pid={pid}")
        except OSError as exc:
            logger.warning(f"[jobs] kill 失败 {job_id} pid={pid}: {exc}")

        return {"job_id": job_id, "killed": True}

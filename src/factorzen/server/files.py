"""workspace 内受控文件管理（列目录 / 读 / 写文本 / 删）。

所有 path 为相对 workspace 根的相对路径；拒绝绝对路径、反斜杠与 ``..`` 段，
resolve 后必须仍在 workspace 内（仿 opsview._safe_report_path）。
"""
from __future__ import annotations

import math
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.core.logger import get_logger

logger = get_logger("factorzen.server.files")

# 文本读写大小上限（字节）；测试可 monkeypatch
FILE_MAX_BYTES = 1_000_000

# 允许读写的文本类扩展名（小写）；无扩展名也按文本试读
TEXT_EXTENSIONS = frozenset(
    {
        ".json",
        ".jsonl",
        ".md",
        ".txt",
        ".yaml",
        ".yml",
        ".py",
        ".sh",
        ".csv",
        ".html",
        ".log",
        ".cfg",
        ".toml",
    }
)


def _mtime_iso(path: Path) -> str:
    """文件/目录 mtime 转 ISO 字符串（UTC）。"""
    try:
        ts = path.stat().st_mtime
    except OSError:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _is_text_path(path: Path) -> bool:
    """是否文本类：有扩展名则查白名单，无扩展名按文本处理。"""
    suffix = path.suffix.lower()
    if not suffix:
        return True
    return suffix in TEXT_EXTENSIONS


def _jsonable_cell(v: Any) -> Any:
    """将单元格转为 JSON 合法值。

    int/float/bool/None 保留（float 的 NaN/inf → None）；
    datetime/date → iso 字符串；其余 str()。
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    # polars 可能给出 Decimal 等
    try:
        # 处理 numpy 标量等
        if hasattr(v, "item"):
            return _jsonable_cell(v.item())
    except Exception:
        pass
    return str(v)


def _normalize_rel(rel_path: str) -> str | None:
    """规范化相对路径；非法则返回 None。

    注意：不能用 ``str.lstrip("./")``——那会按字符集剥离，把 ``../x`` 误变成 ``x``。
    """
    if rel_path is None:
        return None
    if "\\" in rel_path or rel_path.startswith("/"):
        return None

    cleaned = rel_path
    # 仅剥前缀 "./" 段（完整两字符），不碰 ".."
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]

    if cleaned in ("", "."):
        return ""  # 表示根

    # 任何 ".." 段一律拒绝（含单独 ".."）
    parts = Path(cleaned).parts
    if any(p == ".." for p in parts):
        return None
    if any(p == "" for p in parts):
        return None
    return cleaned


class FileManager:
    """workspace 文件管理器（列/读/写/删）。"""

    def __init__(self, workspace_dir: str | Path) -> None:
        self.root = Path(workspace_dir).resolve()

    def _safe_path(self, rel_path: str, *, allow_root: bool = False) -> Path:
        """校验相对路径无遍历，返回 resolve 后的绝对路径。

        allow_root=True 时允许空串指向 workspace 根（仅 list 用）。
        违规 raise FileNotFoundError（API 映射 404）。
        """
        cleaned = _normalize_rel(rel_path if rel_path is not None else "")
        if cleaned is None:
            raise FileNotFoundError(f"非法 path: {rel_path!r}")

        if cleaned == "":
            if allow_root:
                return self.root
            raise FileNotFoundError(f"非法 path: {rel_path!r}")

        target = (self.root / cleaned).resolve()
        if not target.is_relative_to(self.root):
            raise FileNotFoundError(f"非法 path: {rel_path}")
        if target == self.root and not allow_root:
            raise FileNotFoundError(f"非法 path: {rel_path}")
        return target

    def list_dir(self, rel_path: str = "") -> dict[str, Any]:
        """列出目录内容。path 空串=根；目录不存在 404。"""
        target = self._safe_path(rel_path, allow_root=True)
        if not target.exists() or not target.is_dir():
            raise FileNotFoundError(f"目录不存在: {rel_path or '/'}")

        dirs: list[dict[str, Any]] = []
        files: list[dict[str, Any]] = []
        try:
            children = list(target.iterdir())
        except OSError as exc:
            logger.warning(f"[files] 列举失败 {target}: {exc}")
            raise FileNotFoundError(f"目录不可读: {rel_path or '/'}") from exc

        for p in children:
            try:
                if p.is_dir():
                    dirs.append({"name": p.name, "mtime": _mtime_iso(p)})
                elif p.is_file():
                    st = p.stat()
                    files.append(
                        {
                            "name": p.name,
                            "size": st.st_size,
                            "mtime": _mtime_iso(p),
                        }
                    )
            except OSError as exc:
                logger.warning(f"[files] 跳过 {p}: {exc}")

        dirs.sort(key=lambda x: x["name"])
        files.sort(key=lambda x: x["name"])
        # 返回的 path 用调用方传入的相对形式
        display = rel_path if rel_path not in (None,) else ""
        return {"path": display, "dirs": dirs, "files": files}

    def read_content(self, rel_path: str) -> dict[str, Any]:
        """读文件：text / parquet / binary 三种 kind。"""
        target = self._safe_path(rel_path, allow_root=False)
        if not target.is_file():
            raise FileNotFoundError(f"文件不存在: {rel_path}")

        try:
            size = target.stat().st_size
        except OSError as exc:
            raise FileNotFoundError(f"文件不可读: {rel_path}") from exc

        suffix = target.suffix.lower()

        # parquet 预览
        if suffix == ".parquet":
            return self._read_parquet(rel_path, target, size)

        # 文本类
        if _is_text_path(target):
            if size > FILE_MAX_BYTES:
                raise PermissionError(
                    f"文件过大 ({size} > {FILE_MAX_BYTES} bytes): {rel_path}"
                )
            try:
                content = target.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise FileNotFoundError(f"文件读取失败: {rel_path}") from exc
            return {
                "kind": "text",
                "path": rel_path,
                "size": size,
                "content": content,
            }

        # 其他二进制
        return {"kind": "binary", "path": rel_path, "size": size}

    def _read_parquet(
        self, rel_path: str, target: Path, size: int
    ) -> dict[str, Any]:
        """polars 读 parquet，返回 schema + head 50；读失败 raise ValueError → 422。"""
        try:
            lf = pl.scan_parquet(target)
            # lazy:只物化前 50 行与行数,避免大 parquet 整文件进内存
            head_df = lf.head(50).collect()
            n_rows = int(lf.select(pl.len()).collect().item())
        except Exception as exc:
            logger.warning(f"[files] parquet 读取失败 {target}: {exc}")
            raise ValueError(f"parquet 读取失败: {rel_path}: {exc}") from exc

        schema = [
            {"name": name, "dtype": str(dtype)}
            for name, dtype in head_df.schema.items()
        ]
        head: list[dict[str, Any]] = []
        for row in head_df.iter_rows(named=True):
            head.append({c: _jsonable_cell(v) for c, v in row.items()})

        return {
            "kind": "parquet",
            "path": rel_path,
            "n_rows": n_rows,
            "schema": schema,
            "head": head,
            "size": size,
        }

    def write_content(self, rel_path: str, content: str) -> dict[str, Any]:
        """覆盖写文本；仅允许文本类扩展名；父目录必须已存在。"""
        cleaned = _normalize_rel(rel_path if rel_path is not None else "")
        if cleaned is None or cleaned == "":
            raise FileNotFoundError(f"非法 path: {rel_path}")

        p = Path(cleaned)
        if not _is_text_path(p):
            raise PermissionError(f"不允许写入的扩展名: {p.suffix or '(binary)'}")

        target = self._safe_path(rel_path, allow_root=False)
        parent = target.parent
        if not parent.exists() or not parent.is_dir():
            raise FileNotFoundError(f"父目录不存在: {rel_path}")

        try:
            target.write_text(content, encoding="utf-8")
            size = target.stat().st_size
        except OSError as exc:
            raise FileNotFoundError(f"写入失败: {rel_path}") from exc

        return {"path": rel_path, "size": size}

    def delete(self, rel_path: str, *, recursive: bool = False) -> dict[str, Any]:
        """删除文件或目录。

        - path 空/指向根 → ValueError（API 400）
        - 文件直接删
        - 空目录 rmdir
        - 非空目录需 recursive=True，否则 RuntimeError（API 409）
        """
        cleaned = _normalize_rel(rel_path if rel_path is not None else "")
        if cleaned is None:
            raise FileNotFoundError(f"非法 path: {rel_path}")
        if cleaned == "":
            raise ValueError("禁止删除 workspace 根")

        target = self._safe_path(rel_path, allow_root=False)
        if target == self.root:
            raise ValueError("禁止删除 workspace 根")

        if not target.exists():
            raise FileNotFoundError(f"路径不存在: {rel_path}")

        if target.is_file():
            try:
                target.unlink()
            except OSError as exc:
                raise FileNotFoundError(f"删除失败: {rel_path}") from exc
            return {"deleted": rel_path}

        if target.is_dir():
            try:
                is_empty = not any(target.iterdir())
            except OSError as exc:
                raise FileNotFoundError(f"目录不可读: {rel_path}") from exc

            if is_empty:
                try:
                    target.rmdir()
                except OSError as exc:
                    raise FileNotFoundError(f"删除失败: {rel_path}") from exc
                return {"deleted": rel_path}

            if not recursive:
                raise RuntimeError(f"目录非空，需 recursive=true: {rel_path}")

            try:
                shutil.rmtree(target)
            except OSError as exc:
                raise FileNotFoundError(f"递归删除失败: {rel_path}") from exc
            return {"deleted": rel_path}

        raise FileNotFoundError(f"不支持的路径类型: {rel_path}")

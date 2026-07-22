"""因子库与因子资产的只读索引。

扫描 workspace/factor_library 与 workspace/factor_store，供 REST API 消费。
损坏行/文件跳过并记 warning，绝不因单个坏产物炸接口。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from factorzen.core.logger import get_logger

logger = get_logger("factorzen.server.library")

# 市场白名单
MARKETS = frozenset({"ashare", "crypto", "us", "futures"})


class FactorLibraryIndex:
    """因子库 + 因子资产只读索引（零侵入：不触发计算）。"""

    def __init__(self, workspace_dir: str | Path) -> None:
        self.root = Path(workspace_dir)
        self.lib_root = self.root / "factor_library"
        self.store_root = self.root / "factor_store"
        self.track_root = self.lib_root / "forward_track"

    def _check_market(self, market: str) -> None:
        if market not in MARKETS:
            raise FileNotFoundError(f"未知 market: {market}")

    def list_factors(self, market: str) -> dict[str, Any]:
        """读取 factor_library/<market>.jsonl，返回 count/by_status/factors。"""
        self._check_market(market)
        path = self.lib_root / f"{market}.jsonl"
        factors: list[dict[str, Any]] = []
        if not path.exists():
            return {"market": market, "count": 0, "by_status": {}, "factors": []}

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(f"[library] 读取失败 {path}: {exc}")
            return {"market": market, "count": 0, "by_status": {}, "factors": []}

        for i, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(f"[library] 跳过坏行 {path}:{i}: {exc}")
                continue
            if not isinstance(obj, dict):
                logger.warning(f"[library] 跳过非对象行 {path}:{i}")
                continue
            factors.append(obj)

        status_counter: Counter[str] = Counter()
        for f in factors:
            st = f.get("status")
            key = str(st) if st is not None else "unknown"
            status_counter[key] += 1

        return {
            "market": market,
            "count": len(factors),
            "by_status": dict(status_counter),
            "factors": factors,
        }

    def forward_track(self, market: str, expression: str) -> dict[str, Any]:
        """从 forward_track/<market>.jsonl 过滤 expression，按 date 升序。"""
        self._check_market(market)
        path = self.track_root / f"{market}.jsonl"
        points: list[dict[str, Any]] = []
        if not path.exists():
            return {"expression": expression, "points": []}

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(f"[library] 读取 track 失败 {path}: {exc}")
            return {"expression": expression, "points": []}

        for i, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(f"[library] 跳过 track 坏行 {path}:{i}: {exc}")
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("expression") != expression:
                continue
            points.append(
                {
                    "date": obj.get("date"),
                    "ic": obj.get("ic"),
                    "n_stocks": obj.get("n_stocks"),
                }
            )

        points.sort(key=lambda p: str(p.get("date") or ""))
        return {"expression": expression, "points": points}

    def list_store(self, market: str) -> dict[str, Any]:
        """遍历 factor_store/<market>/<name>/meta.json。"""
        self._check_market(market)
        base = self.store_root / market
        entries: list[dict[str, Any]] = []
        if not base.exists():
            return {"market": market, "entries": []}

        try:
            subdirs = sorted(p for p in base.iterdir() if p.is_dir())
        except OSError as exc:
            logger.warning(f"[library] 列举 store 失败 {base}: {exc}")
            return {"market": market, "entries": []}

        for d in subdirs:
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"[library] 跳过损坏 meta {meta_path}: {exc}")
                continue
            if not isinstance(meta, dict):
                continue
            entry = dict(meta)
            entry["name"] = meta.get("name") or d.name
            entries.append(entry)

        return {"market": market, "entries": entries}

    def _safe_store_dir(self, market: str, name: str) -> Path:
        """校验 market 白名单 + name 无路径遍历，返回安全的资产目录。"""
        self._check_market(market)
        base = (self.store_root / market).resolve()
        target = (base / name).resolve()
        if not target.is_relative_to(base) or target == base:
            raise FileNotFoundError(f"非法 name: {name}")
        # 必须是 base 的直接子目录（不允许更深的相对逃逸后落在 base 内但非直属）
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise FileNotFoundError(f"非法 name: {name}") from exc
        if target.parent != base:
            raise FileNotFoundError(f"非法 name: {name}")
        return target

    def store_detail(self, market: str, name: str) -> dict[str, Any]:
        """读取 meta.json + factor.py 源码。"""
        d = self._safe_store_dir(market, name)
        meta_path = d / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"资产不存在: {market}/{name}")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise FileNotFoundError(f"资产 meta 损坏: {market}/{name}") from exc
        if not isinstance(meta, dict):
            raise FileNotFoundError(f"资产 meta 非法: {market}/{name}")

        source: str | None = None
        src_path = d / "factor.py"
        if src_path.exists():
            try:
                source = src_path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning(f"[library] 读取 factor.py 失败 {src_path}: {exc}")
                source = None

        return {
            "market": market,
            "name": meta.get("name") or name,
            "meta": meta,
            "source": source,
        }

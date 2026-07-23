"""因子库与因子资产的索引。

扫描 workspace/factor_library 与 workspace/factor_store，供 REST API 消费。
损坏行/文件跳过并记 warning，绝不因单个坏产物炸接口。

改 factor_library 的 status 必须按 jsonl 行直接读改写（不经 FactorRecord），
以免 from_dict 丢弃未知字段。
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

# 合法因子状态（含手写 manual）
VALID_STATUSES = frozenset(
    {"active", "correlated", "probation", "no_lift", "manual"}
)

# store 手写因子并入列表时从 ledger_snapshot 透出的指标键
_SNAPSHOT_METRIC_KEYS = (
    "ic_train",
    "holdout_ic",
    "dsr",
    "turnover",
)


class FactorLibraryIndex:
    """因子库 + 因子资产索引（list 合并手写因子；status 可写）。"""

    def __init__(self, workspace_dir: str | Path) -> None:
        self.root = Path(workspace_dir)
        self.lib_root = self.root / "factor_library"
        self.store_root = self.root / "factor_store"
        self.track_root = self.lib_root / "forward_track"

    def _check_market(self, market: str) -> None:
        if market not in MARKETS:
            raise FileNotFoundError(f"未知 market: {market}")

    def _read_library_factors(self, market: str) -> list[dict[str, Any]]:
        """读取 factor_library/<market>.jsonl，每条补 source=library。坏行跳过。"""
        path = self.lib_root / f"{market}.jsonl"
        factors: list[dict[str, Any]] = []
        if not path.exists():
            return factors

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(f"[library] 读取失败 {path}: {exc}")
            return factors

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
            row = dict(obj)
            row["source"] = "library"
            factors.append(row)
        return factors

    def _store_handwritten_extras(
        self, market: str, lib_exprs: set[str]
    ) -> list[dict[str, Any]]:
        """扫描 factor_store 中 kind=python 且 expression 不在 lib 的手写因子。"""
        base = self.store_root / market
        extras: list[dict[str, Any]] = []
        if not base.exists():
            return extras

        try:
            subdirs = sorted(p for p in base.iterdir() if p.is_dir())
        except OSError as exc:
            logger.warning(f"[library] 列举 store 失败 {base}: {exc}")
            return extras

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
                logger.warning(f"[library] 跳过非法 meta {meta_path}")
                continue
            if meta.get("kind") != "python":
                continue

            expr = meta.get("expression")
            if not isinstance(expr, str) or not expr:
                logger.warning(f"[library] 跳过无 expression 的 python 因子 {meta_path}")
                continue
            if expr in lib_exprs:
                # 已在 library 登记，用 library 那条，不重复
                continue

            snap = meta.get("ledger_snapshot")
            if not isinstance(snap, dict):
                snap = {}
            status = snap.get("status")
            if status is None or status == "":
                status = "manual"

            name = meta.get("name") or d.name
            row: dict[str, Any] = {
                "expression": expr,
                "market": market,
                "name": name,
                "kind": "python",
                "created_at": meta.get("created_at"),
                "status": status,
                "source": "store",
                "admission_track": "manual",
            }
            for k in _SNAPSHOT_METRIC_KEYS:
                row[k] = snap.get(k)
            extras.append(row)
        return extras

    def list_factors(self, market: str) -> dict[str, Any]:
        """合并 library jsonl + 未入库的 store 手写因子，返回 count/by_status/factors。"""
        self._check_market(market)
        lib_factors = self._read_library_factors(market)
        lib_exprs: set[str] = set()
        for f in lib_factors:
            expr = f.get("expression")
            if isinstance(expr, str) and expr:
                lib_exprs.add(expr)

        store_extras = self._store_handwritten_extras(market, lib_exprs)
        factors = lib_factors + store_extras

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

    def update_status(
        self,
        market: str,
        expression: str,
        new_status: str,
        source: str,
    ) -> dict[str, Any]:
        """更新因子 status。library 按行改写 jsonl；store 改 meta.ledger_snapshot。"""
        self._check_market(market)
        if new_status not in VALID_STATUSES:
            raise ValueError(
                f"非法 status: {new_status}；合法值: {sorted(VALID_STATUSES)}"
            )
        if source == "library":
            self._update_library_status(market, expression, new_status)
        elif source == "store":
            self._update_store_status(market, expression, new_status)
        else:
            raise ValueError(f"非法 source: {source}；合法值: library | store")
        return {
            "market": market,
            "expression": expression,
            "status": new_status,
            "source": source,
        }

    def _update_library_status(
        self, market: str, expression: str, new_status: str
    ) -> None:
        """按行读改写 factor_library/<market>.jsonl，不经 FactorRecord。"""
        path = self.lib_root / f"{market}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"因子库不存在: {market}")

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise FileNotFoundError(f"读取因子库失败: {market}") from exc

        # 保留末尾换行风格：原文件若以 \n 结尾则回写也带
        ends_with_nl = text.endswith("\n") if text else True
        raw_lines = text.splitlines()
        out_lines: list[str] = []
        found = False

        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                # 空行原样保留（保持行结构）
                out_lines.append(line if line == "" else stripped)
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                # 坏行原样保留
                out_lines.append(stripped)
                continue
            if not isinstance(obj, dict):
                out_lines.append(stripped)
                continue

            if (
                not found
                and obj.get("expression") == expression
            ):
                obj["status"] = new_status
                found = True
                out_lines.append(
                    json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                )
            else:
                # 未改行：再 dumps 保持字段，或直接保留原文更稳妥
                # 为「未知字段原样」与格式稳定，未匹配行保留原文
                out_lines.append(stripped)

        if not found:
            raise FileNotFoundError(
                f"未找到 expression: {expression} (market={market})"
            )

        body = "\n".join(out_lines)
        if ends_with_nl:
            body += "\n"
        path.write_text(body, encoding="utf-8")

    def _resolve_store_name_for_expression(
        self, market: str, expression: str
    ) -> str:
        """从 expression 或 store 目录解析资产 name；路径遍历由 _safe_store_dir 兜底。"""
        # 优先 py::<name> 解析
        if expression.startswith("py::"):
            name = expression[4:]
            if name and "/" not in name and "\\" not in name and ".." not in name:
                try:
                    d = self._safe_store_dir(market, name)
                    meta_path = d / "meta.json"
                    if meta_path.exists():
                        return name
                except FileNotFoundError:
                    pass

        # 回退：扫描 store 找 meta.expression 匹配
        base = self.store_root / market
        if not base.exists():
            raise FileNotFoundError(
                f"未找到 store 因子 expression={expression} (market={market})"
            )
        try:
            subdirs = sorted(p for p in base.iterdir() if p.is_dir())
        except OSError as exc:
            raise FileNotFoundError(
                f"列举 store 失败: {market}"
            ) from exc

        for d in subdirs:
            # 只认 base 直接子目录（防异常项）
            try:
                self._safe_store_dir(market, d.name)
            except FileNotFoundError:
                continue
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(meta, dict) and meta.get("expression") == expression:
                return d.name

        raise FileNotFoundError(
            f"未找到 store 因子 expression={expression} (market={market})"
        )

    def _update_store_status(
        self, market: str, expression: str, new_status: str
    ) -> None:
        """改 factor_store/<market>/<name>/meta.json 的 ledger_snapshot.status。"""
        name = self._resolve_store_name_for_expression(market, expression)
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

        snap = meta.get("ledger_snapshot")
        if not isinstance(snap, dict):
            snap = {}
        snap["status"] = new_status
        meta["ledger_snapshot"] = snap

        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

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

"""ops 阶段级幂等状态。

每个交易日一个 ``<state_dir>/<YYYY-MM-DD>.json``,记录各阶段
``{status: done|failed, ts, detail}``。runner 据此在重跑时跳过已完成阶段、
从失败处续跑。写入走「临时文件 + os.replace」保证原子(崩溃不留半截 JSON)。
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any


class OpsState:
    """单个交易日的阶段状态,落盘可跨进程恢复。"""

    def __init__(self, state_dir: str | Path, as_of: date) -> None:
        self.path = Path(state_dir) / f"{as_of.isoformat()}.json"
        self._data: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        if self.path.exists():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        return {}

    def is_done(self, stage: str) -> bool:
        return self._data.get(stage, {}).get("status") == "done"

    def mark_done(self, stage: str, detail: str = "") -> None:
        self._set(stage, "done", detail)

    def mark_failed(self, stage: str, detail: str) -> None:
        self._set(stage, "failed", detail)

    def summary(self) -> dict[str, dict[str, Any]]:
        return dict(self._data)

    def _set(self, stage: str, status: str, detail: str) -> None:
        self._data[stage] = {
            "status": status,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "detail": detail,
        }
        self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.parent / (self.path.name + ".tmp")
        tmp.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, self.path)

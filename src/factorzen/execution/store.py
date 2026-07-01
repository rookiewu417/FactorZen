"""向前执行会话落盘/续跑：workspace/execution/<session_id>/。"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl

from factorzen.core.experiment import build_manifest_base, get_git_sha


class SessionStore:
    def __init__(self, session_dir: str | Path) -> None:
        self.dir = Path(session_dir)
        self._ledger = self.dir / "ledger.parquet"
        self._state = self.dir / "state.json"
        self._nav = self.dir / "nav.parquet"

    def init(self, config: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest_base(list(config.get("command", [])), config)
        manifest.update({"git_sha": get_git_sha(), "config": config})
        (self.dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2)
        )

    def has_date(self, as_of: date) -> bool:
        if not self._ledger.exists():
            return False
        got = pl.read_parquet(self._ledger).select("as_of_date")["as_of_date"].to_list()
        return as_of.isoformat() in got

    def append(self, record: dict) -> None:
        row = {k: record[k] for k in ("as_of_date", "nav_before", "nav_after")}
        row["payload"] = json.dumps(
            {"orders": record["orders"], "fills": record["fills"]}, ensure_ascii=False
        )
        df = pl.DataFrame([row])
        if self._ledger.exists():
            df = pl.concat([pl.read_parquet(self._ledger), df], how="vertical")
        df.write_parquet(self._ledger)
        df.select(["as_of_date", "nav_after"]).write_parquet(self._nav)
        self._state.write_text(json.dumps(record["broker_state"], ensure_ascii=False))

    def load_state(self) -> dict | None:
        if not self._state.exists():
            return None
        return json.loads(self._state.read_text())

    def nav_frame(self) -> pl.DataFrame:
        return pl.read_parquet(self._nav) if self._nav.exists() else pl.DataFrame()

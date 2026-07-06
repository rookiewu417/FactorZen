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
            {
                "orders": record["orders"],
                "acks": record.get("acks", []),
                "fills": record["fills"],
            },
            ensure_ascii=False,
        )
        df = pl.DataFrame([row])
        if self._ledger.exists():
            df = pl.concat([pl.read_parquet(self._ledger), df], how="vertical")
        df.write_parquet(self._ledger)
        df.select(["as_of_date", "nav_after"]).write_parquet(self._nav)
        # 在续跑态里嵌入 _last_as_of，供 run_daily_step 做日期单调性守卫(E2)。
        # broker.load_state 只读 cash/pos/order_seq/last_price，忽略 _last_as_of。
        state_out = {**record["broker_state"], "_last_as_of": record["as_of_date"]}
        self._state.write_text(json.dumps(state_out, ensure_ascii=False))

    def ledger_records(self) -> list[dict]:
        """逐行还原 {as_of_date,nav_before,nav_after,orders,acks,fills}；旧 payload 无 acks 视为 []。"""
        if not self._ledger.exists():
            return []
        out: list[dict] = []
        for row in pl.read_parquet(self._ledger).iter_rows(named=True):
            payload = json.loads(row["payload"])
            out.append(
                {
                    "as_of_date": row["as_of_date"],
                    "nav_before": row["nav_before"],
                    "nav_after": row["nav_after"],
                    "orders": payload.get("orders", []),
                    "acks": payload.get("acks", []),
                    "fills": payload.get("fills", []),
                }
            )
        return out

    def load_state(self) -> dict | None:
        if not self._state.exists():
            return None
        return json.loads(self._state.read_text())

    def nav_frame(self) -> pl.DataFrame:
        return pl.read_parquet(self._nav) if self._nav.exists() else pl.DataFrame()

"""向前执行会话落盘/续跑：workspace/execution/<session_id>/。"""
from __future__ import annotations

import json
import os
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
        manifest_path = self.dir / "manifest.json"
        # 已有会话再 init（如 fz live replay 复用 fz live init 建的 session）不覆盖——
        # 否则 init 设的 slippage_bps/initial_cash 会被 replay 的默认 config 静默清掉。
        if manifest_path.exists():
            return
        manifest = build_manifest_base(list(config.get("command", [])), config)
        manifest.update({"git_sha": get_git_sha(), "config": config})
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

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
        # 三个文件各自 tmp + os.replace 原子替换，避免写到一半崩溃留下损坏 parquet/json。
        self._atomic_parquet(df, self._ledger)
        self._atomic_parquet(df.select(["as_of_date", "nav_after"]), self._nav)
        # 在续跑态里嵌入 _last_as_of，供 run_daily_step 做日期单调性守卫(E2) +
        # 崩溃恢复一致性校验（state._last_as_of 须与 ledger 末行日期一致）。
        # broker.load_state 只读 cash/pos/order_seq/last_price，忽略 _last_as_of。
        state_out = {**record["broker_state"], "_last_as_of": record["as_of_date"]}
        self._atomic_text(json.dumps(state_out, ensure_ascii=False), self._state)

    @staticmethod
    def _atomic_parquet(df: pl.DataFrame, path: Path) -> None:
        tmp = path.with_name(path.name + ".tmp")
        df.write_parquet(tmp)
        os.replace(tmp, path)

    @staticmethod
    def _atomic_text(text: str, path: Path) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def last_ledger_date(self) -> str | None:
        """ledger 末行(最大)交易日 ISO 字符串；空/无 ledger 返回 None。"""
        if not self._ledger.exists():
            return None
        dates = pl.read_parquet(self._ledger)["as_of_date"].to_list()
        return max(dates) if dates else None

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

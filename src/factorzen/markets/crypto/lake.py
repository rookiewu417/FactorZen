"""crypto 本地数据湖:parquet 分区读写(纯本地 IO,不联网)。

布局(时间戳一律 naive-UTC Datetime("us")):
    <root>/klines_1m/symbol=BTCUSDT/2026-05.parquet
    <root>/funding/symbol=BTCUSDT/2026-05.parquet
    <root>/metrics/symbol=BTCUSDT/2026-06-27.parquet
    <root>/meta.parquet            # ts_code, name, list_date
    <root>/manifest.json           # 回填区间/gaps/git_sha,可复现
写入时自动追加 ``ts_code`` 列,读取即拼接过滤,无需目录名解析。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl


def month_range(start: str, end: str) -> list[str]:
    """[start, end]("YYYYMMDD")覆盖到的月份列表("YYYY-MM")。"""
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end, "%Y%m%d")
    out, y, m = [], s.year, s.month
    while (y, m) <= (e.year, e.month):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def day_range(month: str, start: str, end: str) -> list[str]:
    """month("YYYY-MM")内与 [start, end] 相交的日期列表("YYYYMMDD")。"""
    y, m = int(month[:4]), int(month[5:7])
    d = datetime(y, m, 1)
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end, "%Y%m%d")
    out = []
    while d.month == m:
        if s <= d <= e:
            out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


def _read_partitions(dir_: Path, symbols: list[str] | None, start: str, end: str,
                     time_col: str, schema: dict[str, Any]) -> pl.DataFrame:
    lo = datetime.strptime(start, "%Y%m%d")
    hi = datetime.strptime(end, "%Y%m%d") + timedelta(days=1)
    files: list[Path] = []
    syms = symbols if symbols is not None else [p.name.split("=", 1)[1]
                                                for p in sorted(dir_.glob("symbol=*"))]
    for sym in syms:
        sdir = dir_ / f"symbol={sym}"
        if sdir.is_dir():
            files += sorted(sdir.glob("*.parquet"))
    if not files:
        return pl.DataFrame(schema={"ts_code": pl.String, **schema})
    lf = pl.scan_parquet([str(f) for f in files])
    return (
        lf.filter((pl.col(time_col) >= lo) & (pl.col(time_col) < hi))
        .collect()
        .sort(["ts_code", time_col])
    )


_KLINE_SCHEMA = {"trade_date": pl.Datetime("us"), "open": pl.Float64, "high": pl.Float64,
                 "low": pl.Float64, "close": pl.Float64, "vol": pl.Float64,
                 "amount": pl.Float64, "taker_buy_volume": pl.Float64}
_FUNDING_SCHEMA = {"event_time": pl.Datetime("us"), "funding_rate": pl.Float64}
_METRICS_SCHEMA = {"event_time": pl.Datetime("us"), "open_interest": pl.Float64}


class CryptoLake:
    def __init__(self, root: str | Path = "workspace/crypto_lake") -> None:
        self.root = Path(root)

    # ── 路径 ──────────────────────────────────────────────
    def kline_path(self, symbol: str, month: str) -> Path:
        return self.root / "klines_1m" / f"symbol={symbol}" / f"{month}.parquet"

    def funding_path(self, symbol: str, month: str) -> Path:
        return self.root / "funding" / f"symbol={symbol}" / f"{month}.parquet"

    def metrics_path(self, symbol: str, day: str) -> Path:
        return self.root / "metrics" / f"symbol={symbol}" / f"{day}.parquet"

    # ── 写入(追加 ts_code 列)───────────────────────────
    def _write(self, path: Path, symbol: str, df: pl.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.with_columns(pl.lit(symbol).alias("ts_code")).write_parquet(path)

    def write_klines(self, symbol: str, month: str, df: pl.DataFrame) -> None:
        self._write(self.kline_path(symbol, month), symbol, df)

    def write_funding(self, symbol: str, month: str, df: pl.DataFrame) -> None:
        self._write(self.funding_path(symbol, month), symbol, df)

    def write_metrics(self, symbol: str, day: str, df: pl.DataFrame) -> None:
        self._write(self.metrics_path(symbol, day), symbol, df)

    # ── 读取 ──────────────────────────────────────────────
    def read_klines(self, symbols: list[str] | None, start: str, end: str) -> pl.DataFrame:
        return _read_partitions(self.root / "klines_1m", symbols, start, end,
                                "trade_date", _KLINE_SCHEMA)

    def read_funding(self, symbols: list[str] | None, start: str, end: str) -> pl.DataFrame:
        return _read_partitions(self.root / "funding", symbols, start, end,
                                "event_time", _FUNDING_SCHEMA)

    def read_metrics(self, symbols: list[str] | None, start: str, end: str) -> pl.DataFrame:
        return _read_partitions(self.root / "metrics", symbols, start, end,
                                "event_time", _METRICS_SCHEMA)

    # ── meta / manifest ───────────────────────────────────
    def write_meta(self, df: pl.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        df.write_parquet(self.root / "meta.parquet")

    def read_meta(self) -> pl.DataFrame:
        p = self.root / "meta.parquet"
        if not p.exists():
            return pl.DataFrame(
                schema={"ts_code": pl.String, "name": pl.String, "list_date": pl.Date})
        return pl.read_parquet(p)

    def write_manifest(self, d: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "manifest.json").write_text(
            json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    def read_manifest(self) -> dict[str, Any]:
        p = self.root / "manifest.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def symbols(self) -> list[str]:
        d = self.root / "klines_1m"
        return sorted(p.name.split("=", 1)[1] for p in d.glob("symbol=*")) if d.is_dir() else []

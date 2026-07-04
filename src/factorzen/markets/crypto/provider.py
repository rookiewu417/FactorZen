"""crypto 行情接入（CCXT）。

唯一与交易所直接交互的 crypto 模块。默认 Binance USDT-M 永续（``binanceusdm``），
换交易所只改 ``exchange_id``。测试注入 fake ``client`` 走离线路径，CI 无网络可跑。

标的键约定：``ts_code`` 用交易所 native 形式（如 ``BTCUSDT``），内部映射到
ccxt unified 形式（如 ``BTC/USDT:USDT``）调接口。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import polars as pl

from factorzen.markets.base import DataProvider
from factorzen.markets.crypto.frequency import BAR_FREQS, normalize_freq

_BAR_SCHEMA = ["ts_code", "open", "high", "low", "close", "vol", "_ms"]


def _date_to_ms(d: str) -> int:
    """``YYYYMMDD`` → UTC 00:00 的毫秒时间戳。"""
    dt = datetime.strptime(d, "%Y%m%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


class CryptoDataProvider(DataProvider):
    """CCXT 行情/资金费/持仓量/元数据接入。"""

    def __init__(
        self,
        exchange_id: str = "binanceusdm",
        client: Any = None,
        quote: str = "USDT",
    ) -> None:
        self.exchange_id = exchange_id
        self._client = client
        self.quote = quote

    @property
    def client(self) -> Any:
        """惰性创建 ccxt 客户端（注入 client 时直接用，测试离线路径）。"""
        if self._client is None:
            import ccxt

            self._client = getattr(ccxt, self.exchange_id)({"enableRateLimit": True})
        return self._client

    # ── symbol 映射 ────────────────────────────────────────
    def _to_unified(self, ts_code: str) -> str:
        """``BTCUSDT`` → ``BTC/USDT:USDT``（USDT 本位永续）。"""
        base = ts_code[: -len(self.quote)]
        return f"{base}/{self.quote}:{self.quote}"

    def _to_ts_code(self, unified: str) -> str:
        """``BTC/USDT:USDT`` → ``BTCUSDT``。"""
        left = unified.split(":")[0]
        base, quote = left.split("/")
        return f"{base}{quote}"

    # ── 行情 ───────────────────────────────────────────────
    def fetch_bars(
        self, symbols: list[str] | None, start: str, end: str, freq: str = "daily"
    ) -> pl.DataFrame:
        tf = BAR_FREQS[normalize_freq(freq)].timeframe
        start_ms = _date_to_ms(start)
        end_ms = _date_to_ms(end)
        rows: list[tuple] = []
        for sym in symbols or []:
            unified = self._to_unified(sym)
            since = start_ms
            while since <= end_ms:
                batch = self.client.fetch_ohlcv(unified, timeframe=tf, since=since, limit=1000)
                if not batch:
                    break
                stop = False
                for ts, o, h, lo, c, v in batch:
                    if ts > end_ms:
                        stop = True
                        break
                    rows.append((sym, float(o), float(h), float(lo), float(c), float(v), int(ts)))
                last_ts = batch[-1][0]
                if stop or last_ts < since:
                    break
                since = last_ts + 1
        if not rows:
            return pl.DataFrame(
                schema={
                    "ts_code": pl.String, "open": pl.Float64, "high": pl.Float64,
                    "low": pl.Float64, "close": pl.Float64, "vol": pl.Float64,
                    "trade_date": pl.Date, "amount": pl.Float64,
                }
            )
        df = pl.DataFrame(rows, schema=_BAR_SCHEMA, orient="row")
        return df.with_columns(
            pl.from_epoch(pl.col("_ms"), time_unit="ms").cast(pl.Date).alias("trade_date"),
            (pl.col("close") * pl.col("vol")).alias("amount"),
        ).drop("_ms").sort(["ts_code", "trade_date"])

    # ── 资金费（perps 特有）────────────────────────────────
    def fetch_funding(self, symbols: list[str] | None, start: str, end: str) -> pl.DataFrame:
        start_ms = _date_to_ms(start)
        end_ms = _date_to_ms(end) + 86_400_000 - 1  # 含 end 当日所有档
        rows: list[tuple] = []
        for sym in symbols or []:
            unified = self._to_unified(sym)
            hist = self.client.fetch_funding_rate_history(unified, since=start_ms, limit=1000)
            for rec in hist:
                ts = int(rec["timestamp"])
                if ts > end_ms:
                    continue
                rows.append((sym, int(ts), float(rec["fundingRate"])))
        df = pl.DataFrame(rows, schema=["ts_code", "_ms", "funding_rate"], orient="row")
        if df.is_empty():
            return pl.DataFrame(
                schema={"ts_code": pl.String, "trade_date": pl.Date, "funding_rate": pl.Float64}
            )
        # 日频 = 当日多档 funding 之和（Binance 每 8h 一档）
        return (
            df.with_columns(
                pl.from_epoch(pl.col("_ms"), time_unit="ms").cast(pl.Date).alias("trade_date")
            )
            .group_by(["ts_code", "trade_date"])
            .agg(pl.col("funding_rate").sum())
            .sort(["ts_code", "trade_date"])
        )

    # ── 持仓量（best-effort）───────────────────────────────
    def fetch_open_interest(
        self, symbols: list[str] | None, start: str, end: str
    ) -> pl.DataFrame:
        empty = pl.DataFrame(
            schema={"ts_code": pl.String, "trade_date": pl.Date, "open_interest": pl.Float64}
        )
        if not hasattr(self.client, "fetch_open_interest_history"):
            return empty
        start_ms = _date_to_ms(start)
        end_ms = _date_to_ms(end) + 86_400_000 - 1
        rows: list[tuple] = []
        for sym in symbols or []:
            unified = self._to_unified(sym)
            hist = self.client.fetch_open_interest_history(unified, since=start_ms, limit=1000)
            for rec in hist:
                ts = int(rec["timestamp"])
                if ts > end_ms:
                    continue
                amount = rec.get("openInterestAmount") or rec.get("openInterestValue") or 0.0
                rows.append((sym, int(ts), float(amount)))
        if not rows:
            return empty
        return (
            pl.DataFrame(rows, schema=["ts_code", "_ms", "open_interest"], orient="row")
            .with_columns(
                pl.from_epoch(pl.col("_ms"), time_unit="ms").cast(pl.Date).alias("trade_date")
            )
            .drop("_ms")
            .sort(["ts_code", "trade_date"])
        )

    # ── 元数据 ─────────────────────────────────────────────
    def fetch_symbol_meta(self) -> pl.DataFrame:
        markets = self.client.load_markets()
        rows: list[tuple] = []
        for _unified, m in markets.items():
            if not m.get("swap"):
                continue  # 只要永续
            if m.get("quote") != self.quote:
                continue
            ts_code = f"{m['base']}{m['quote']}"
            list_date = None
            onboard = (m.get("info") or {}).get("onboardDate")
            if onboard:
                dt = datetime.fromtimestamp(int(onboard) / 1000, tz=timezone.utc)
                list_date = dt.date()
            rows.append((ts_code, m.get("base"), list_date))
        return pl.DataFrame(
            rows,
            schema={"ts_code": pl.String, "name": pl.String, "list_date": pl.Date},
            orient="row",
        )

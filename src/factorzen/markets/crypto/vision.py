"""Binance Vision(data.binance.vision)历史包下载与解析。

为什么用 Vision 而非 CCXT REST:本环境 fapi.binance.com 返回 451(地域封锁),
而 Vision 静态站可达;且 zip 批量回填远快于 REST 分页。所有联网经注入的
``fetch(url)->bytes``,测试离线。月包为主、当月日包补齐;缺口记 manifest["gaps"],
不静默。funding 仅有月包 → 当月尚未发布的 funding 记 gap(缺失=fill 0 中性)。

网络前提:本机直连不出网,``_http_get`` 走 ``urllib`` 默认 opener,自动继承
env 的 ``http_proxy/https_proxy`` 代理(见 spec §1);离线测试全部注入 fetch。
"""
from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from collections.abc import Callable

import polars as pl

from factorzen.markets.crypto.lake import CryptoLake, day_range, month_range

_BASE = "https://data.binance.vision"
_UM = "data/futures/um"

Fetch = Callable[[str], bytes]


def _http_get(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "factorzen-lake"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # 固定 https 静态站
        return bytes(r.read())


# ── URL 生成 ───────────────────────────────────────────────
def kline_month_url(symbol: str, month: str, interval: str = "1m") -> str:
    return f"{_BASE}/{_UM}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"


def kline_day_url(symbol: str, day: str, interval: str = "1m") -> str:
    d = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    return f"{_BASE}/{_UM}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{d}.zip"


def funding_month_url(symbol: str, month: str) -> str:
    return f"{_BASE}/{_UM}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"


def metrics_day_url(symbol: str, day: str) -> str:
    d = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    return f"{_BASE}/{_UM}/daily/metrics/{symbol}/{symbol}-metrics-{d}.zip"


# ── 解析(CSV 可能无表头,探测首行)─────────────────────
_KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def _read_csv(raw: bytes, header_prefix: bytes, cols: list[str]) -> pl.DataFrame:
    has_header = raw.split(b"\n", 1)[0].startswith(header_prefix)
    return pl.read_csv(io.BytesIO(raw), has_header=has_header,
                       new_columns=None if has_header else cols,
                       infer_schema_length=10000)


def parse_kline_csv(raw: bytes) -> pl.DataFrame:
    df = _read_csv(raw, b"open_time", _KLINE_COLS)
    return df.select(
        pl.from_epoch(pl.col("open_time").cast(pl.Int64), time_unit="ms")
        .cast(pl.Datetime("us")).alias("trade_date"),
        pl.col("open").cast(pl.Float64), pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64), pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64).alias("vol"),
        pl.col("quote_volume").cast(pl.Float64).alias("amount"),
        pl.col("taker_buy_volume").cast(pl.Float64),
    ).sort("trade_date")


def parse_funding_csv(raw: bytes) -> pl.DataFrame:
    df = _read_csv(raw, b"calc_time", ["calc_time", "funding_interval_hours", "last_funding_rate"])
    return df.select(
        pl.from_epoch(pl.col("calc_time").cast(pl.Int64), time_unit="ms")
        .cast(pl.Datetime("us")).alias("event_time"),
        pl.col("last_funding_rate").cast(pl.Float64).alias("funding_rate"),
    ).sort("event_time")


def parse_metrics_csv(raw: bytes) -> pl.DataFrame:
    cols = ["create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
            "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
            "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
    df = _read_csv(raw, b"create_time", cols)
    return df.select(
        pl.col("create_time").str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="us")
        .alias("event_time"),
        pl.col("sum_open_interest").cast(pl.Float64).alias("open_interest"),
    ).sort("event_time")


# ── S3 listing ────────────────────────────────────────────
def _list_prefixes(prefix: str, fetch: Fetch) -> list[str]:
    out: list[str] = []
    marker = ""
    while True:
        url = f"{_BASE}/?prefix={prefix}&delimiter=/" + (f"&marker={marker}" if marker else "")
        xml = fetch(url).decode()
        page = re.findall(r"<Prefix>([^<]+)</Prefix>", xml)
        out += [p for p in page if p != prefix]
        if "<IsTruncated>true</IsTruncated>" not in xml:
            return out
        nm = re.search(r"<NextMarker>([^<]+)</NextMarker>", xml)
        marker = nm.group(1) if nm else out[-1]


def list_um_symbols(quote: str = "USDT", fetch: Fetch = _http_get) -> list[str]:
    prefix = f"{_UM}/monthly/klines/"
    syms = [p[len(prefix):].strip("/") for p in _list_prefixes(prefix, fetch)]
    return sorted(s for s in syms if s.endswith(quote))


def list_symbol_months(symbol: str, fetch: Fetch = _http_get) -> list[str]:
    prefix = f"{_UM}/monthly/klines/{symbol}/1m/"
    xml = fetch(f"{_BASE}/?prefix={prefix}").decode()
    pat = rf"{symbol}-1m-(\d{{4}}-\d{{2}})\.zip</Key>"
    return sorted(set(re.findall(pat, xml)))


# ── 下载 ──────────────────────────────────────────────────
def fetch_zip_csv(url: str, fetch: Fetch = _http_get, retries: int = 2) -> bytes | None:
    for attempt in range(retries + 1):
        try:
            raw = fetch(url)
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                return zf.read(zf.namelist()[0])
        except Exception:  # 网络/解压任何异常都重试,达上限记 gap
            if attempt == retries:
                return None
    return None


def rank_symbols_by_amount(symbols: list[str], month: str, top_n: int,
                           fetch: Fetch = _http_get) -> list[str]:
    """用 1d 月包(每标的≈30 行,极小)按月度 quote 成交额排序选 Top-N。"""
    scored: list[tuple[float, str]] = []
    for sym in symbols:
        csv = fetch_zip_csv(kline_month_url(sym, month, interval="1d"), fetch=fetch)
        if csv is None:
            continue
        amt = float(parse_kline_csv(csv)["amount"].sum())
        scored.append((amt, sym))
    return [s for _, s in sorted(scored, reverse=True)[:top_n]]


# ── 回填编排 ──────────────────────────────────────────────
def backfill(lake: CryptoLake, symbols: list[str], start: str, end: str, *,
             fetch: Fetch = _http_get, log: Callable[..., object] = print) -> dict[str, object]:
    """月包为主、当月日包补齐;已有分区跳过(增量);缺口记 gaps 不静默。"""
    gaps: list[str] = []
    months = month_range(start, end)
    last_full = _prev_month(months[-1])  # 末月可能未出月包
    for sym in symbols:
        for month in months:
            # klines:先月包,404 再逐日日包
            if not lake.kline_path(sym, month).exists():
                csv = fetch_zip_csv(kline_month_url(sym, month), fetch=fetch)
                if csv is not None:
                    lake.write_klines(sym, month, parse_kline_csv(csv))
                else:
                    frames = []
                    for day in day_range(month, start, end):
                        dcsv = fetch_zip_csv(kline_day_url(sym, day), fetch=fetch)
                        if dcsv is None:
                            gaps.append(f"klines/{sym}/{day}")
                        else:
                            frames.append(parse_kline_csv(dcsv))
                    if frames:
                        lake.write_klines(sym, month, pl.concat(frames))
                    elif month <= last_full:
                        gaps.append(f"klines/{sym}/{month}")
            # funding:仅月包
            if not lake.funding_path(sym, month).exists():
                csv = fetch_zip_csv(funding_month_url(sym, month), fetch=fetch)
                if csv is not None:
                    lake.write_funding(sym, month, parse_funding_csv(csv))
                else:
                    gaps.append(f"funding/{sym}/{month}")
            # metrics(OI):仅日包
            for day in day_range(month, start, end):
                if lake.metrics_path(sym, day).exists():
                    continue
                mcsv = fetch_zip_csv(metrics_day_url(sym, day), fetch=fetch)
                if mcsv is None:
                    gaps.append(f"metrics/{sym}/{day}")
                else:
                    lake.write_metrics(sym, day, parse_metrics_csv(mcsv))
        log(f"[backfill] {sym} 完成")
    manifest: dict[str, object] = {"start": start, "end": end, "symbols": symbols, "gaps": gaps}
    lake.write_manifest(manifest)
    if gaps:
        log(f"[backfill] ⚠ {len(gaps)} 个缺口(详见 manifest.json gaps)")
    return manifest


def _prev_month(month: str) -> str:
    y, m = int(month[:4]), int(month[5:7])
    return f"{y - 1:04d}-12" if m == 1 else f"{y:04d}-{m - 1:02d}"

"""美股行情接入（Yahoo Finance chart API 自建 provider，不引 yfinance 库）。

唯一与 Yahoo 直接交互的模块。端点
``https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1=&period2=&interval=1d``
返回 timestamp + 未复权 OHLCV + **已复权** adjclose。urllib 直连（继承 env 代理）+ 频率控制
（分批 + 间隔避 429 + 退避重试，参考 crypto Vision `_http_get`）+ 按 symbol parquet 缓存
（覆盖审计而非「文件存在」启发式：``_coverage.json`` 记每标的已回补 [start,end]）。

**复权硬契约（ground-truth 逐值测试见 tests/test_us_provider.py）：**
Yahoo OHLC 未复权、adjclose 为复权收盘。``adj_factor = adjclose / close_raw`` 比率复权 OHLC
（拆股/分红调整），使 ``ret_1d`` 无拆股跳变、PIT 安全。**量列不复权**：``vol`` = 原始股数、
``amount`` = ``close_raw × vol_raw`` = 美元成交额（拆股不变量，见 factors.py 单位注释）。

**已知限制（诚实标注）：** Yahoo adjclose 是**后视锚定**（latest-anchored）——发生新公司行动后
历史复权值会变；本 provider 扩窗时重拉全并集窗口覆写，使单次 run 内自洽（不做前视锚定的
增量分红/拆股事件回放，留二期）。

测试注入 ``fetch(url)->bytes``（canned JSON），CI 离线可跑。
"""
from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from factorzen.config.settings import DATA_RAW
from factorzen.markets.base import DataProvider
from factorzen.markets.us.sp500_snapshot import sp500_symbols

Fetch = Callable[[str], bytes]

_CHART_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
_CACHE_SUBDIR = "us_daily"

_OUT_COLS = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "adj_factor"]


def _http_get(url: str, timeout: float = 30.0) -> bytes:
    # 固定 https 静态 API；继承 env http(s)_proxy（同 crypto Vision 直连口径）。
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (factorzen-us)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return bytes(r.read())


def _to_date(d: date | str) -> date:
    if isinstance(d, str):
        return datetime.strptime(d, "%Y%m%d").date()
    return d


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "ts_code": pl.String, "trade_date": pl.Date,
        "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
        "vol": pl.Float64, "amount": pl.Float64, "adj_factor": pl.Float64,
    })


def parse_chart_json(raw: bytes, ts_code: str) -> pl.DataFrame:
    """解析 Yahoo chart JSON → 后复权日线帧（纯函数，离线可测）。

    返回列 ``ts_code, trade_date(Date), open/high/low/close(后复权), vol(原始股数),
    amount(=close_raw×vol_raw 美元成交额), adj_factor(=adjclose/close_raw)``，按 trade_date 排序。
    close/adjclose 为 null 的行丢弃（Yahoo 在区间里塞的休市/停牌行，避免 NaN 穿透）。
    """
    try:
        data = json.loads(raw)
    except Exception:
        return _empty()
    results = ((data.get("chart") or {}).get("result")) or []
    if not results:
        return _empty()
    res = results[0]
    ts = res.get("timestamp") or []
    ind = res.get("indicators") or {}
    quote = (ind.get("quote") or [{}])[0] or {}
    adj_list = ind.get("adjclose") or []
    adjclose = (adj_list[0].get("adjclose") if adj_list and adj_list[0] else None)
    if not ts or adjclose is None:
        return _empty()
    n = len(ts)

    def _col(name: str) -> list:
        v = quote.get(name)
        return v if isinstance(v, list) and len(v) == n else [None] * n

    df = pl.DataFrame({
        "_ts": ts,
        "open_raw": _col("open"),
        "high_raw": _col("high"),
        "low_raw": _col("low"),
        "close_raw": _col("close"),
        "vol_raw": _col("volume"),
        "adjclose": adjclose if isinstance(adjclose, list) and len(adjclose) == n else [None] * n,
    })
    df = df.filter(pl.col("close_raw").is_not_null() & pl.col("adjclose").is_not_null()
                   & (pl.col("close_raw").abs() > 1e-12))
    if df.is_empty():
        return _empty()
    df = df.with_columns(
        pl.from_epoch(pl.col("_ts").cast(pl.Int64), time_unit="s").dt.date().alias("trade_date"),
        (pl.col("adjclose").cast(pl.Float64) / pl.col("close_raw").cast(pl.Float64)).alias("adj_factor"),
        pl.col("vol_raw").cast(pl.Float64).fill_null(0.0),
    )
    return df.select(
        pl.lit(ts_code).alias("ts_code"),
        "trade_date",
        (pl.col("open_raw").cast(pl.Float64) * pl.col("adj_factor")).alias("open"),
        (pl.col("high_raw").cast(pl.Float64) * pl.col("adj_factor")).alias("high"),
        (pl.col("low_raw").cast(pl.Float64) * pl.col("adj_factor")).alias("low"),
        (pl.col("close_raw").cast(pl.Float64) * pl.col("adj_factor")).alias("close"),
        pl.col("vol_raw").alias("vol"),
        (pl.col("close_raw").cast(pl.Float64) * pl.col("vol_raw")).alias("amount"),
        pl.col("adj_factor"),
    ).unique(subset=["trade_date"], keep="last").sort("trade_date")


class USDataProvider(DataProvider):
    def __init__(
        self,
        cache_root: str | Path | None = None,
        fetch: Fetch = _http_get,
        request_interval: float = 0.3,
        retries: int = 3,
        universe_symbols: list[str] | None = None,
    ) -> None:
        self._fetch = fetch
        self.request_interval = request_interval
        self.retries = retries
        self.cache_root = Path(cache_root) if cache_root is not None else DATA_RAW
        self._universe = universe_symbols  # None → 静态 S&P500 快照
        self._throttle = fetch is _http_get  # 注入 fetch(测试)不 sleep

    def _universe_symbols(self) -> list[str]:
        return self._universe if self._universe is not None else sp500_symbols()

    # ── 缓存路径/覆盖审计 ──────────────────────────────────
    @property
    def _cache_dir(self) -> Path:
        return self.cache_root / _CACHE_SUBDIR

    def _symbol_path(self, sym: str) -> Path:
        return self._cache_dir / f"{sym}.parquet"

    @property
    def _coverage_path(self) -> Path:
        return self._cache_dir / "_coverage.json"

    def _read_coverage(self) -> dict[str, list[str]]:
        p = self._coverage_path
        if p.exists():
            try:
                return dict(json.loads(p.read_text()))
            except Exception:
                return {}
        return {}

    def _write_coverage(self, cov: dict[str, list[str]]) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._coverage_path.write_text(json.dumps(cov))

    # ── 网络拉取（限流 + 退避重试）────────────────────────
    def _chart_url(self, sym: str, start: str, end: str) -> str:
        p1 = int(datetime.strptime(start, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp())
        # period2 半开：+1 天含 end 当日
        p2 = int((datetime.strptime(end, "%Y%m%d") + timedelta(days=1))
                 .replace(tzinfo=timezone.utc).timestamp())
        return f"{_CHART_BASE}/{sym}?period1={p1}&period2={p2}&interval=1d"

    def _get_with_retry(self, url: str) -> bytes | None:
        for attempt in range(self.retries + 1):
            if self._throttle and self.request_interval > 0:
                time.sleep(self.request_interval)  # 限流：每次网络前歇一下避 429
            try:
                return self._fetch(url)
            except Exception:
                if attempt == self.retries:
                    return None
                if self._throttle:
                    time.sleep(2.0 * (attempt + 1))  # 429/网络退避
        return None

    def _fetch_symbol(self, sym: str, start: str, end: str) -> pl.DataFrame:
        raw = self._get_with_retry(self._chart_url(sym, start, end))
        if raw is None:
            return _empty()
        return parse_chart_json(raw, sym)

    def _load_symbol(self, sym: str) -> pl.DataFrame:
        p = self._symbol_path(sym)
        if not p.exists():
            return _empty()
        try:
            return pl.read_parquet(p)
        except Exception:
            return _empty()

    def _ensure_symbol(self, sym: str, start: str, end: str, cov: dict[str, list[str]]) -> None:
        """覆盖审计：若该标的已回补窗口覆盖 [start,end] 则跳过；否则拉并集窗口覆写。

        记「请求覆盖窗口」而非数据 min/max——否则次新股请求早于其上市日会被判「未覆盖」而反复重拉。
        """
        rec = cov.get(sym)
        if rec and rec[0] <= start and rec[1] >= end:
            return
        fs = min(start, rec[0]) if rec else start
        fe = max(end, rec[1]) if rec else end
        df = self._fetch_symbol(sym, fs, fe)
        if not df.is_empty():
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            df.write_parquet(self._symbol_path(sym))
        cov[sym] = [fs, fe]  # 即使空也记，避免反复重拉已知无数据标的

    # ── DataProvider Port ──────────────────────────────────
    def fetch_bars(
        self, symbols: list[str] | None, start: str, end: str, freq: str = "daily"
    ) -> pl.DataFrame:
        if freq != "daily":
            raise ValueError(f"USDataProvider 仅支持 freq='daily'（Yahoo 日线），收到 {freq!r}")
        syms = list(symbols) if symbols else self._universe_symbols()
        cov = self._read_coverage()
        frames: list[pl.DataFrame] = []
        for sym in syms:
            self._ensure_symbol(sym, start, end, cov)
            df = self._load_symbol(sym)
            if not df.is_empty():
                frames.append(df)
        self._write_coverage(cov)
        if not frames:
            return _empty()
        s, e = _to_date(start), _to_date(end)
        out = pl.concat(frames, how="vertical_relaxed").filter(
            (pl.col("trade_date") >= s) & (pl.col("trade_date") <= e)
        )
        return out.select(_OUT_COLS).sort(["ts_code", "trade_date"])

    def fetch_symbol_meta(self) -> pl.DataFrame:
        """静态 S&P500 快照元数据 [ts_code, name, list_date(null)]。

        list_date=null（幸存者偏差 MVP 不做 PIT 上市日）；universe 快照据此不做次新过滤。
        """
        syms = self._universe_symbols()
        return pl.DataFrame({
            "ts_code": syms,
            "name": syms,
            "list_date": [None] * len(syms),
        }, schema={"ts_code": pl.String, "name": pl.String, "list_date": pl.Date})

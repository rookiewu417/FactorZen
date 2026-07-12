"""国内商品期货交易日历（Tushare trade_cal，默认 SHFE）。

商品期货与 A 股同守国家法定节假日，交易日基本一致；此处仍独立按期货交易所拉取，
避免与 A 股日历耦合。首次拉取缓存到 ``data/cache/trade_cal_futures_{exchange}.parquet``，
7 天过期刷新。测试可注入 ``cal_df`` 走离线路径（不联网）。

年化周期数：商品期货约 243 交易日/年（``periods_per_year("daily")=243``）。
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl

from factorzen.config.settings import DATA_CACHE
from factorzen.config.tushare_config import CACHE_EXPIRE_DAYS
from factorzen.markets.base import Calendar

_PERIODS_PER_YEAR: dict[str, float] = {"daily": 243.0}


def _to_date(d: date | str) -> date:
    if isinstance(d, str):
        return datetime.strptime(d, "%Y%m%d").date()
    return d


class FuturesCalendar(Calendar):
    def __init__(self, exchange: str = "SHFE", cal_df: pl.DataFrame | None = None) -> None:
        self.exchange = exchange
        self._cal = cal_df  # 注入 → 离线（测试）

    # ── 缓存管理 ───────────────────────────────────────────
    @property
    def _cache_file(self) -> Path:
        return DATA_CACHE / f"trade_cal_futures_{self.exchange}.parquet"

    def _cache_valid(self) -> bool:
        f = self._cache_file
        if not f.exists():
            return False
        return (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days < CACHE_EXPIRE_DAYS

    def _fetch(self) -> pl.DataFrame:
        from factorzen.core.loader import init_tushare

        pro = init_tushare()
        df = pro.trade_cal(exchange=self.exchange, start_date="19950101", end_date="20301231")
        if df is None or df.empty:
            raise RuntimeError(f"Tushare trade_cal(exchange={self.exchange}) 返回空，检查网络/权限。")
        out = pl.from_pandas(df).with_columns(
            pl.col("cal_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d"),
            pl.col("is_open").cast(pl.Int8),
        )
        DATA_CACHE.mkdir(parents=True, exist_ok=True)
        out.write_parquet(self._cache_file)
        return out

    def _load(self) -> pl.DataFrame:
        if self._cal is not None:
            return self._cal
        if self._cache_valid():
            return pl.read_parquet(self._cache_file)
        return self._fetch()

    # ── Calendar Port ──────────────────────────────────────
    def sessions(self, start: str, end: str) -> list[date]:
        cal = self._load()
        s, e = _to_date(start), _to_date(end)
        return (
            cal.filter(
                (pl.col("is_open") == 1) & (pl.col("cal_date") >= s) & (pl.col("cal_date") <= e)
            )
            .sort("cal_date")["cal_date"]
            .to_list()
        )

    def is_session(self, d: date | str) -> bool:
        cal = self._load()
        row = cal.filter(pl.col("cal_date") == _to_date(d))
        return (not row.is_empty()) and row["is_open"].item() == 1

    def next_session(self, d: date | str, n: int = 1) -> date:
        cal = self._load()
        rows = (
            cal.filter((pl.col("is_open") == 1) & (pl.col("cal_date") > _to_date(d)))
            .sort("cal_date")
            .limit(n)
        )
        if len(rows) < n:
            raise ValueError(f"不足 {n} 个后续交易日（从 {d} 往后）")
        return rows["cal_date"].to_list()[n - 1]

    def prev_session(self, d: date | str, n: int = 1) -> date:
        cal = self._load()
        rows = (
            cal.filter((pl.col("is_open") == 1) & (pl.col("cal_date") < _to_date(d)))
            .sort("cal_date", descending=True)
            .limit(n)
        )
        if len(rows) < n:
            raise ValueError(f"不足 {n} 个前序交易日（从 {d} 往前）")
        return rows["cal_date"].to_list()[n - 1]

    def periods_per_year(self, freq: str = "daily") -> float:
        if freq not in _PERIODS_PER_YEAR:
            raise ValueError(f"futures 未知频率: {freq!r}，支持 {sorted(_PERIODS_PER_YEAR)}")
        return _PERIODS_PER_YEAR[freq]

"""国内商品期货行情接入（Tushare fut_daily / fut_mapping / fut_basic）。

唯一与 Tushare 期货接口直接交互的模块。原始 fut_daily / fut_mapping 按 trade_date 落
parquet 缓存，用**期货交易日历覆盖审计**（非「文件存在」启发式）判缺失日增量拉取；fut_basic
（品种元数据）缓存 7 天。``fetch_bars`` 返回**主力连续后复权**帧（经 continuous.build_continuous）。

测试注入 ``pro``（fake，具 fut_daily/fut_mapping/fut_basic 方法）+ ``calendar``（离线日历）+
``cache_root``（临时目录），CI 无 token/网络可跑。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.config.settings import DATA_RAW
from factorzen.core.storage import load_parquet, save_parquet
from factorzen.markets.base import DataProvider
from factorzen.markets.futures.calendar import FuturesCalendar
from factorzen.markets.futures.continuous import build_continuous

# 国内商品期货交易所（不含 CFFEX 金融期货，本 Phase 只做商品）
COMMODITY_EXCHANGES: tuple[str, ...] = ("SHFE", "DCE", "CZCE", "INE", "GFEX")

_FUT_DAILY_COLS = [
    "ts_code", "trade_date", "pre_close", "open", "high", "low", "close",
    "settle", "vol", "amount", "oi",
]
_FUT_MAPPING_COLS = ["ts_code", "trade_date", "mapping_ts_code"]


def _str_to_date(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)


class FuturesDataProvider(DataProvider):
    def __init__(
        self,
        exchanges: tuple[str, ...] = COMMODITY_EXCHANGES,
        pro: Any = None,
        calendar: FuturesCalendar | None = None,
        cache_root: str | Path | None = None,
    ) -> None:
        self.exchanges = exchanges
        self._pro = pro
        # 真实 client(惰性建)走限流+重试；注入 pro(测试)不限流(避免每调 sleep 0.2s 拖慢单测)
        self._throttle = pro is None
        self.calendar = calendar or FuturesCalendar()
        self.cache_root = Path(cache_root) if cache_root is not None else DATA_RAW
        self._fut_codes_cache: set[str] | None = None
        self._meta_cache: pl.DataFrame | None = None

    @property
    def pro(self) -> Any:
        if self._pro is None:
            from factorzen.core.loader import init_tushare

            self._pro = init_tushare()
        return self._pro

    # ── 缓存审计 + 增量拉取 ────────────────────────────────
    def _expected_sessions(self, start: str, end: str) -> list[str]:
        return [d.strftime("%Y%m%d") for d in self.calendar.sessions(start, end)]

    def _cached_dates(self, data_type: str, start: str, end: str) -> set[str]:
        try:
            cached = load_parquet(
                data_type, start=start, end=end, base_dir=self.cache_root
            ).collect()
        except Exception:
            return set()
        if cached.is_empty() or "trade_date" not in cached.columns:
            return set()
        return set(cached.select(pl.col("trade_date").dt.strftime("%Y%m%d")).to_series().to_list())

    def _fetch_by_date(
        self, api: Any, data_type: str, missing: list[str], std_cols: list[str]
    ) -> None:
        """逐缺失交易日拉全市场并写缓存（按年 flush 控内存），覆盖审计已保证 PIT/完整性。

        真实 client 经 ``core.loader._retry`` 走限流(_rate_limit)+重试(含频率超限 62s 退避)——直接裸调
        api(trade_date=) 会被 Tushare 频率超限拒绝且被 except 静默吞掉致数据缺口（陷阱：静默降级）。
        单日彻底失败仅记 warning 并跳过；覆盖审计使下次重跑自愈补齐该日。
        """
        if not missing:
            return
        from factorzen.core.loader import logger

        buf: list[pl.DataFrame] = []

        def _flush() -> None:
            if not buf:
                return
            merged = (
                pl.concat(buf, how="vertical_relaxed")
                .with_columns(_str_to_date("trade_date"))
                .sort(["trade_date", "ts_code"])
            )
            merged = merged.select([c for c in std_cols if c in merged.columns])
            save_parquet(merged, data_type=data_type, base_dir=self.cache_root)
            buf.clear()

        last_ym: str | None = None
        for d in missing:
            if last_ym is not None and d[:6] != last_ym:  # 按年月 flush：进度可观测 + kill 可续
                _flush()
            last_ym = d[:6]
            try:
                df_pd = self._call(api, d)
            except Exception as e:
                logger.warning(f"[{data_type}] {d} 拉取失败(跳过,重跑自愈): {e}")
                continue
            if df_pd is not None and not df_pd.empty:
                buf.append(pl.from_pandas(df_pd))
        _flush()

    def _call(self, api: Any, d: str) -> Any:
        """单日 API 调用：真实 client 走 loader 限流+重试；注入 pro(测试)裸调。"""
        if self._throttle:
            from factorzen.core.loader import _retry

            return _retry(api, trade_date=d)
        return api(trade_date=d)

    def _load_raw(self, data_type: str, api: Any, start: str, end: str, std_cols: list[str]) -> pl.DataFrame:
        expected = self._expected_sessions(start, end)
        if expected:
            present = self._cached_dates(data_type, start, end)
            missing = [d for d in expected if d not in present]
            self._fetch_by_date(api, data_type, missing, std_cols)
        try:
            return load_parquet(data_type, start=start, end=end, base_dir=self.cache_root).collect()
        except Exception:
            return pl.DataFrame()

    # ── fut_codes（品种字母集，主力连续过滤用）────────────
    def _fut_codes(self) -> set[str]:
        if self._fut_codes_cache is not None:
            return self._fut_codes_cache
        meta = self.fetch_symbol_meta()
        codes = set(meta["fut_code"].to_list()) if not meta.is_empty() else set()
        self._fut_codes_cache = codes
        return codes

    # ── DataProvider Port ──────────────────────────────────
    def fetch_bars(
        self, symbols: list[str] | None, start: str, end: str, freq: str = "daily"
    ) -> pl.DataFrame:
        if freq != "daily":
            raise ValueError(
                f"FuturesDataProvider 仅支持 freq='daily'（Tushare fut_daily），收到 {freq!r}"
            )
        daily = self._load_raw("fut_daily", self.pro.fut_daily, start, end, _FUT_DAILY_COLS)
        mapping = self._load_raw("fut_mapping", self.pro.fut_mapping, start, end, _FUT_MAPPING_COLS)
        if daily.is_empty() or mapping.is_empty():
            from factorzen.markets.futures.continuous import _empty

            return _empty()
        cont = build_continuous(mapping, daily, self._fut_codes())
        if symbols:
            cont = cont.filter(pl.col("ts_code").is_in(list(symbols)))
        return cont.sort(["ts_code", "trade_date"])

    def fetch_symbol_meta(self) -> pl.DataFrame:
        """品种级元数据 [ts_code(连续码), name, fut_code, exchange, list_date]，缓存 7 天。

        由 fut_basic（合约级）聚合出品种：每 (fut_code, exchange) 取一行，连续码 =
        ``{fut_code}.{exch_suffix}``（SHF/DCE/ZCE/INE/GFE），list_date = 该品种最早合约上市日。
        """
        if self._meta_cache is not None:
            return self._meta_cache
        from factorzen.config.tushare_config import CACHE_EXPIRE_DAYS

        # 缓存挂在 cache_root 下并以 exchanges 命名：避免不同 cache_root/交易所集串味
        # （测试临时目录隔离；生产 data/raw 下按交易所集分文件，切换交易所不吃陈旧 meta）。
        cache_dir = self.cache_root / "fut_meta"
        cache_file = cache_dir / f"fut_basic_meta_{'_'.join(self.exchanges)}.parquet"
        if cache_file.exists():
            age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
            if age < CACHE_EXPIRE_DAYS:
                self._meta_cache = pl.read_parquet(cache_file)
                return self._meta_cache

        parts: list[pl.DataFrame] = []
        for exch in self.exchanges:
            try:
                df_pd = self.pro.fut_basic(exchange=exch, fut_type="1")
            except Exception:
                continue
            if df_pd is None or df_pd.empty:
                continue
            df = pl.from_pandas(df_pd)
            cols = [c for c in ("ts_code", "symbol", "name", "fut_code", "exchange", "list_date") if c in df.columns]
            parts.append(df.select(cols))
        if not parts:
            self._meta_cache = _empty_meta()
            return self._meta_cache

        raw = pl.concat(parts, how="vertical_relaxed")
        # fut_code 为空的（如连续/指数合约行）剔除；list_date 转 Date
        raw = raw.filter(pl.col("fut_code").is_not_null() & (pl.col("fut_code").str.len_chars() > 0))
        if "list_date" in raw.columns:
            raw = raw.with_columns(_str_to_date("list_date"))
        else:
            raw = raw.with_columns(pl.lit(None, dtype=pl.Date).alias("list_date"))
        meta = (
            raw.group_by(["fut_code", "exchange"])
            .agg(
                pl.col("name").first().alias("name"),
                pl.col("list_date").min().alias("list_date"),
            )
            .with_columns(
                (pl.col("fut_code") + "." + pl.col("exchange").replace(_EXCH_SUFFIX)).alias("ts_code")
            )
            .select("ts_code", "name", "fut_code", "exchange", "list_date")
            .sort("ts_code")
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta.write_parquet(cache_file)
        self._meta_cache = meta
        return meta


# 交易所 → 连续码后缀（Tushare 约定）
_EXCH_SUFFIX: dict[str, str] = {
    "SHFE": "SHF", "DCE": "DCE", "CZCE": "ZCE", "INE": "INE", "GFEX": "GFE",
}


def _empty_meta() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "ts_code": pl.String, "name": pl.String, "fut_code": pl.String,
        "exchange": pl.String, "list_date": pl.Date,
    })

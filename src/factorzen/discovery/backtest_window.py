# src/factorzen/discovery/backtest_window.py
"""默认回测窗口——因子库 rebuild 与挖掘共用的枢纽。

「最近约 6 年滚动到数据最新端」：``end = min(该市场缓存最新可用交易日, today?)``，
``start = end − years``。crypto 因数据成熟度另设起点下限 20210101。返回 ``YYYYMMDD``。
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

# 各市场缓存根（探测最新可用交易日用）。改这里即改探测源。
_ASHARE_DAILY_ROOT = "data/raw/daily"
_FUTURES_DAILY_ROOT = "data/raw/fut_daily"
_CRYPTO_LAKE_ROOT = "workspace/crypto_lake"
# 美股缓存：provider 按 symbol 落 ``data/raw/us_daily/{sym}.parquet``（非 Hive 年月分区）。
_US_DAILY_ROOT = "data/raw/us_daily"

# crypto 数据成熟度起点下限（早于此的 K 线覆盖不足，不作训练窗口）。
_CRYPTO_START_FLOOR = date(2021, 1, 1)


def _coerce_date(v) -> date:
    """把 parquet/JSON 里的 trade_date 值（date/datetime/str/int）统一成 ``date``。"""
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v).strip().replace("-", "").replace("/", "")
    if len(s) >= 8 and s[:8].isdigit():
        return datetime.strptime(s[:8], "%Y%m%d").date()
    raise ValueError(f"无法解析交易日: {v!r}")


def _max_trade_date_partitioned(root: str) -> date | None:
    """扫 Hive 分区 ``root/year=YYYY/month=MM/*.parquet``，读最新分区 parquet 的 trade_date 最大值。

    只读最深（最新 year→month）的那个分区文件，避免全量扫描。目录/文件缺失或无 trade_date 列
    → None（调用方回退）。
    """
    base = Path(root)
    if not base.is_dir():
        return None
    parquets = sorted(base.glob("year=*/month=*/*.parquet"))
    if not parquets:
        return None
    import polars as pl

    # 从最新分区往前找，直到取到有效 trade_date（防末尾空/损坏分区）。
    for pq in reversed(parquets):
        try:
            mx = (pl.scan_parquet(pq).select(pl.col("trade_date").max())
                  .collect().item())
        except Exception:
            continue
        if mx is not None:
            try:
                return _coerce_date(mx)
            except ValueError:
                continue
    return None


def _crypto_lake_end() -> date | None:
    """crypto 数据湖最新覆盖日：优先读 ``manifest.json`` 的 ``end``，缺失回退扫 klines 月文件名。"""
    manifest = Path(_CRYPTO_LAKE_ROOT) / "manifest.json"
    if manifest.is_file():
        try:
            meta = json.loads(manifest.read_text())
            end = meta.get("end")
            if end:
                return _coerce_date(end)
        except Exception:
            pass
    # 回退：klines_1m/symbol=*/YYYY-MM.parquet 的最大月份 → 该月月末（保守）。
    kl = Path(_CRYPTO_LAKE_ROOT) / "klines_1m"
    if kl.is_dir():
        months = sorted(p.stem for p in kl.glob("symbol=*/*.parquet") if p.stem[:7].replace("-", "").isdigit())
        if months:
            y, m = months[-1][:4], months[-1][5:7]
            try:
                nm = date(int(y) + (m == "12"), (int(m) % 12) + 1, 1)
                from datetime import timedelta
                return nm - timedelta(days=1)
            except Exception:
                return None
    return None


def _us_latest_date(root: str) -> date | None:
    """美股缓存最新交易日：扫 ``root/*.parquet``（每 symbol 一文件）取 trade_date 最大值。"""
    base = Path(root)
    if not base.is_dir():
        return None
    if not any(base.glob("*.parquet")):
        return None
    import polars as pl

    try:
        mx = (pl.scan_parquet(str(base / "*.parquet"))
              .select(pl.col("trade_date").max()).collect().item())
    except Exception:
        return None
    if mx is None:
        return None
    try:
        return _coerce_date(mx)
    except ValueError:
        return None


def latest_data_date(market: str) -> date | None:
    """探测该市场缓存的最大可用 ``trade_date``。缓存缺失 → None（调用方须回退/报错）。

    A股扫 ``data/raw/daily``；期货扫 ``data/raw/fut_daily``；crypto 读 ``workspace/crypto_lake``；
    美股扫 ``data/raw/us_daily``（每 symbol 一 parquet）。
    """
    if market == "ashare":
        return _max_trade_date_partitioned(_ASHARE_DAILY_ROOT)
    if market == "futures":
        return _max_trade_date_partitioned(_FUTURES_DAILY_ROOT)
    if market == "crypto":
        return _crypto_lake_end()
    if market == "us":
        return _us_latest_date(_US_DAILY_ROOT)
    raise ValueError(f"未知 market={market!r}，应为 ashare/crypto/futures/us")


def _minus_years(d: date, years: int) -> date:
    """d 回退 years 年；2/29 落到无闰年时退到 2/28。"""
    try:
        return d.replace(year=d.year - years)
    except ValueError:
        return d.replace(year=d.year - years, day=28)


def default_window(market: str, *, years: int = 6, today: date | None = None) -> tuple[str, str]:
    """(start, end) as ``YYYYMMDD``。``end = min(latest_data_date(market), today?)``，
    ``start = end − years 年``。crypto 起点封到 ≥ 20210101。

    缓存探测不到最新交易日时抛 ``ValueError``（诚实报错，不静默取 today——那会把无数据的
    未来窗口伪造成有数据）。
    """
    latest = latest_data_date(market)
    if latest is None:
        raise ValueError(
            f"探测不到 {market} 缓存的最新交易日（缓存可能未回补）；"
            f"请先补数据或显式传 --start/--end 覆盖默认窗口。"
        )
    end = latest if today is None else min(latest, today)
    start = _minus_years(end, years)
    if market == "crypto" and start < _CRYPTO_START_FLOOR:
        start = _CRYPTO_START_FLOOR
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

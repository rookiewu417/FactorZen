"""5min（及任意 A 股 bar 频率）预物化中间层：canonicalize + resample 读穿式缓存。

落盘：``{cache_dir}/bars_{freq}/year=YYYY/month=MM/data.parquet`` +
``{cache_dir}/bars_{freq}/manifest.json``。

缓存有效性（不做「文件存在」启发式）：
1. manifest ``resample_hash`` 与当前 ``resample_semantics_hash(freq)`` 一致；
2. manifest ``coverage.months`` 覆盖请求月；
3. 分区 parquet 存在且非空。

语义哈希纳入 ``resample_intraday`` 关键参数/版本；语义变更时旧缓存整库失效。
"""

from __future__ import annotations

import hashlib
import json
from calendar import monthrange
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.config.settings import DATA_DERIVED, DATA_RAW
from factorzen.core.storage import load_parquet, partition_exists, save_parquet
from factorzen.intraday.sessions import (
    AFTER_HOURS_POLICY,
    ASHARE_BAR_FREQS,
    BAR_LABEL_CONVENTION,
    FIRST_BAR_INCLUDES_AUCTION,
    canonicalize_minute,
    normalize_freq,
    resample_intraday,
)

# 算法语义版本：resample_intraday 行为变更时必须 bump，触发缓存失效。
_RESAMPLE_ALGO_VERSION: str = "bucket_argminmax_v1"

_BARS_SCHEMA = {
    "ts_code": pl.String,
    "trade_time": pl.Datetime("us"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "vol": pl.Int64,
    "amount": pl.Float64,
}


def resample_semantics_hash(freq: str) -> str:
    """把 resample_intraday 关键语义参数/版本号纳入哈希。"""
    freq_n = normalize_freq(freq)
    meta = ASHARE_BAR_FREQS[freq_n]
    payload = {
        "algo": _RESAMPLE_ALGO_VERSION,
        "freq": freq_n,
        "minutes": meta.minutes,
        "bars_per_day": meta.bars_per_day,
        "bar_label": BAR_LABEL_CONVENTION,
        "after_hours": AFTER_HOURS_POLICY,
        "first_bar_auction": FIRST_BAR_INCLUDES_AUCTION,
        "already_canonical_path": True,
        "open_close": "arg_min_max_trade_time",
        "agg": {"high": "max", "low": "min", "vol": "sum", "amount": "sum"},
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def bars_data_type(freq: str) -> str:
    """分区 data_type 名：``bars_5min`` / ``bars_15min`` …"""
    return f"bars_{normalize_freq(freq)}"


def bars_cache_root(freq: str, *, cache_dir: Path | None = None) -> Path:
    """``{cache_dir}/bars_{freq}``。"""
    base = DATA_DERIVED if cache_dir is None else Path(cache_dir)
    return base / bars_data_type(freq)


def _manifest_path(freq: str, *, cache_dir: Path | None = None) -> Path:
    return bars_cache_root(freq, cache_dir=cache_dir) / "manifest.json"


def read_bars_manifest(
    freq: str = "5min",
    *,
    cache_dir: Path | None = None,
) -> dict[str, Any] | None:
    """读取 bars 缓存 manifest；不存在返回 ``None``。"""
    path = _manifest_path(freq, cache_dir=cache_dir)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _parse_month_label(month: str) -> tuple[int, int, str, str]:
    """``YYYY-MM`` → (year, month, month_start_YYYYMMDD, month_end_YYYYMMDD)。"""
    if len(month) != 7 or month[4] != "-":
        raise ValueError(f"month 须为 YYYY-MM，得到 {month!r}")
    y, m = int(month[:4]), int(month[5:7])
    if not (1 <= m <= 12):
        raise ValueError(f"非法月份: {month!r}")
    last = monthrange(y, m)[1]
    return y, m, f"{y:04d}{m:02d}01", f"{y:04d}{m:02d}{last:02d}"


def _cache_valid(
    month: str,
    freq: str,
    *,
    cache_dir: Path | None,
    rhash: str,
) -> bool:
    """覆盖 + 哈希 + 分区非空，三者同时成立才命中。"""
    existing = read_bars_manifest(freq, cache_dir=cache_dir)
    if existing is None:
        return False
    if existing.get("resample_hash") != rhash:
        return False
    if existing.get("freq") != normalize_freq(freq):
        return False
    months = list((existing.get("coverage") or {}).get("months") or [])
    if month not in months:
        return False
    y, m, _, _ = _parse_month_label(month)
    root = DATA_DERIVED if cache_dir is None else Path(cache_dir)
    return partition_exists(bars_data_type(freq), y, m, base_dir=root)


def build_bars_from_minute(minute: pl.DataFrame, freq: str) -> pl.DataFrame:
    """纯函数：1min → canonicalize → resample（不读写缓存）。"""
    freq_n = normalize_freq(freq)
    if minute.is_empty():
        return pl.DataFrame(schema=_BARS_SCHEMA)
    canon = canonicalize_minute(minute.lazy()).collect()
    if canon.is_empty():
        return pl.DataFrame(schema=_BARS_SCHEMA)
    return resample_intraday(canon, freq_n, already_canonical=True)


def _merge_coverage(
    existing: dict[str, Any] | None,
    month: str,
    m_start: str,
    m_end: str,
) -> dict[str, Any]:
    if existing is None:
        return {"start": m_start, "end": m_end, "months": [month]}
    cov = existing.get("coverage") or {}
    old_months = list(cov.get("months") or [])
    merged = sorted(set(old_months) | {month})
    old_start = str(cov.get("start") or m_start)
    old_end = str(cov.get("end") or m_end)
    return {
        "start": min(old_start, m_start),
        "end": max(old_end, m_end),
        "months": merged,
    }


def _write_month_bars(
    bars: pl.DataFrame,
    month: str,
    freq: str,
    *,
    cache_dir: Path | None,
    rhash: str,
) -> None:
    """写分区 + 更新 manifest（hash 不匹配时重置 coverage）。"""
    freq_n = normalize_freq(freq)
    y, m, m_start, m_end = _parse_month_label(month)
    root = DATA_DERIVED if cache_dir is None else Path(cache_dir)
    dtype = bars_data_type(freq_n)

    if bars.is_empty():
        # 仍写空 schema 分区，避免反复 miss；coverage 也记上
        empty = pl.DataFrame(schema=_BARS_SCHEMA)
        part = root / dtype / f"year={y}" / f"month={m:02d}"
        part.mkdir(parents=True, exist_ok=True)
        empty.write_parquet(part / "data.parquet")
    else:
        save_parquet(
            bars,
            data_type=dtype,
            date_col="trade_time",
            base_dir=root,
            mode="overwrite",
        )

    existing = read_bars_manifest(freq_n, cache_dir=cache_dir)
    if existing is not None and existing.get("resample_hash") != rhash:
        # 语义变更：覆盖重建 coverage
        existing = None

    coverage = _merge_coverage(existing, month, m_start, m_end)
    payload: dict[str, Any] = {
        "freq": freq_n,
        "resample_hash": rhash,
        "resample_algo": _RESAMPLE_ALGO_VERSION,
        "coverage": coverage,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _write_manifest(_manifest_path(freq_n, cache_dir=cache_dir), payload)


def _read_month_bars(
    month: str,
    freq: str,
    *,
    cache_dir: Path | None,
) -> pl.DataFrame:
    y, m, m_start, m_end = _parse_month_label(month)
    root = DATA_DERIVED if cache_dir is None else Path(cache_dir)
    try:
        lf = load_parquet(
            bars_data_type(freq),
            start=m_start,
            end=m_end,
            date_col="trade_time",
            base_dir=root,
        )
        return lf.collect()
    except Exception:
        return pl.DataFrame(schema=_BARS_SCHEMA)


def load_or_build_bars(
    month: str,
    freq: str = "5min",
    *,
    source_dir: Path | None = None,
    cache_dir: Path | None = None,
    force: bool = False,
    minute: pl.DataFrame | None = None,
    write_cache: bool = True,
) -> pl.DataFrame:
    """读穿式：命中且哈希/覆盖有效则直接读，否则 canonicalize+resample 并写缓存。

    Parameters
    ----------
    month:
        ``YYYY-MM`` 自然月标签。
    freq:
        A 股 bar 频率（进缓存键 ``bars_{freq}``）。
    source_dir:
        1min 源湖根；``minute`` 未传入且需重建时从此加载。
    cache_dir:
        缓存根（其下建 ``bars_{freq}/``）；默认 ``DATA_DERIVED``。
    force:
        忽略命中，强制重算（仍写缓存，除非 ``write_cache=False``）。
    minute:
        可选预加载的 1min 帧（已含该月）；避免重复读湖。
    write_cache:
        是否落盘；测试对比「强制重算不污染」时可关。

    Returns
    -------
    pl.DataFrame
        schema 与 ``resample_intraday`` 一致（行序无契约）。
    """
    freq_n = normalize_freq(freq)
    rhash = resample_semantics_hash(freq_n)
    _parse_month_label(month)  # validate

    if not force and _cache_valid(month, freq_n, cache_dir=cache_dir, rhash=rhash):
        return _read_month_bars(month, freq_n, cache_dir=cache_dir)

    # 重建
    if minute is None:
        src = DATA_RAW if source_dir is None else Path(source_dir)
        _, _, m_start, m_end = _parse_month_label(month)
        try:
            lf = load_parquet(
                "minute_1min",
                start=m_start,
                end=m_end,
                date_col="trade_time",
                base_dir=src,
            )
            minute = lf.collect()
        except Exception:
            minute = pl.DataFrame(
                schema={
                    "ts_code": pl.String,
                    "trade_time": pl.Datetime("us"),
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "vol": pl.Int64,
                    "amount": pl.Float64,
                }
            )

    bars = build_bars_from_minute(minute, freq_n)
    if write_cache:
        _write_month_bars(bars, month, freq_n, cache_dir=cache_dir, rhash=rhash)
    return bars


__all__ = [
    "bars_cache_root",
    "bars_data_type",
    "build_bars_from_minute",
    "load_or_build_bars",
    "read_bars_manifest",
    "resample_semantics_hash",
]

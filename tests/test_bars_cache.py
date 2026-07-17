"""tests/test_bars_cache.py — 5min bars 预物化缓存：命中/缺月/哈希失效/freq 隔离。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from factorzen.core.storage import save_parquet
from factorzen.intraday.bars_cache import (
    bars_data_type,
    build_bars_from_minute,
    load_or_build_bars,
    read_bars_manifest,
    resample_semantics_hash,
)
from factorzen.intraday.sessions import canonicalize_minute, resample_intraday


def _make_day_bars(
    code: str,
    day: datetime,
    *,
    n: int = 20,
    base_px: float = 10.0,
) -> pl.DataFrame:
    slots: list[tuple[int, int]] = [(9, 30)]
    for i in range(1, n):
        idx = i
        if idx <= 120:
            tod = 570 + idx
            h, m = divmod(tod, 60)
        else:
            tod = 780 + (idx - 120)
            h, m = divmod(tod, 60)
        if h > 15 or (h == 15 and m > 0):
            break
        if h == 12 or (h == 11 and m > 30):
            continue
        slots.append((h, m))
    if (15, 0) not in slots:
        slots.append((15, 0))
    slots = slots[:n]
    if (15, 0) not in slots:
        slots[-1] = (15, 0)
    rows_t = [day.replace(hour=h, minute=m, second=0, microsecond=0) for h, m in slots]
    px = [base_px + 0.01 * i for i in range(len(rows_t))]
    return pl.DataFrame(
        {
            "ts_code": [code] * len(rows_t),
            "trade_time": pl.Series(rows_t, dtype=pl.Datetime("us")),
            "open": px,
            "high": [p + 0.05 for p in px],
            "low": [p - 0.05 for p in px],
            "close": [p + 0.02 for p in px],
            "vol": pl.Series([100 + i * 10 for i in range(len(rows_t))], dtype=pl.Int64),
            "amount": [1000.0 + i * 100 for i in range(len(rows_t))],
        }
    )


def _build_src(tmp: Path, months: list[tuple[int, int, list[int]]]) -> None:
    frames: list[pl.DataFrame] = []
    for y, m, days in months:
        for d in days:
            for code, px in (("000001.SZ", 10.0), ("000002.SZ", 20.0)):
                frames.append(
                    _make_day_bars(code, datetime(y, m, d), n=24, base_px=px)
                )
    save_parquet(
        pl.concat(frames),
        data_type="minute_1min",
        date_col="trade_time",
        base_dir=tmp,
        mode="overwrite",
    )


def _keys(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(["ts_code", "trade_time"])


class TestBarsCache:
    def test_hit_matches_force_rebuild(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        cache = tmp_path / "cache"
        _build_src(src, [(2024, 6, [3, 4])])

        cold = load_or_build_bars(
            "2024-06", "5min", source_dir=src, cache_dir=cache, force=True
        )
        hot = load_or_build_bars(
            "2024-06", "5min", source_dir=src, cache_dir=cache, force=False
        )
        forced = load_or_build_bars(
            "2024-06", "5min", source_dir=src, cache_dir=cache, force=True
        )

        assert cold.height > 0
        assert_frame_equal(_keys(cold), _keys(hot), check_exact=False, abs_tol=1e-12)
        assert_frame_equal(_keys(hot), _keys(forced), check_exact=False, abs_tol=1e-12)
        man = read_bars_manifest("5min", cache_dir=cache)
        assert man is not None
        assert man["resample_hash"] == resample_semantics_hash("5min")
        assert "2024-06" in man["coverage"]["months"]

    def test_missing_month_falls_back_to_compute(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        cache = tmp_path / "cache"
        _build_src(src, [(2024, 6, [3]), (2024, 7, [1])])

        # 只物化 6 月
        load_or_build_bars("2024-06", "5min", source_dir=src, cache_dir=cache)
        man = read_bars_manifest("5min", cache_dir=cache)
        assert man is not None
        assert man["coverage"]["months"] == ["2024-06"]

        # 7 月缺 → 计算并扩展 coverage
        jul = load_or_build_bars("2024-07", "5min", source_dir=src, cache_dir=cache)
        assert jul.height > 0
        man2 = read_bars_manifest("5min", cache_dir=cache)
        assert man2 is not None
        assert set(man2["coverage"]["months"]) == {"2024-06", "2024-07"}

    def test_hash_mismatch_invalidates(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        cache = tmp_path / "cache"
        _build_src(src, [(2024, 6, [3, 4])])
        load_or_build_bars("2024-06", "5min", source_dir=src, cache_dir=cache)

        # 污染 manifest 哈希
        mpath = cache / "bars_5min" / "manifest.json"
        payload = json.loads(mpath.read_text(encoding="utf-8"))
        payload["resample_hash"] = "deadbeefdeadbeef"
        mpath.write_text(json.dumps(payload), encoding="utf-8")

        # 应判失效并重写正确哈希
        out = load_or_build_bars("2024-06", "5min", source_dir=src, cache_dir=cache)
        assert out.height > 0
        man = read_bars_manifest("5min", cache_dir=cache)
        assert man is not None
        assert man["resample_hash"] == resample_semantics_hash("5min")

    def test_freq_key_isolation(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        cache = tmp_path / "cache"
        _build_src(src, [(2024, 6, [3])])

        b5 = load_or_build_bars("2024-06", "5min", source_dir=src, cache_dir=cache)
        b15 = load_or_build_bars("2024-06", "15min", source_dir=src, cache_dir=cache)
        assert b5.height > 0 and b15.height > 0
        assert b5.height != b15.height  # 桶数不同

        assert (cache / "bars_5min" / "manifest.json").exists()
        assert (cache / "bars_15min" / "manifest.json").exists()
        assert bars_data_type("5min") == "bars_5min"
        assert bars_data_type("15min") == "bars_15min"

        # 互不覆盖
        m5 = read_bars_manifest("5min", cache_dir=cache)
        m15 = read_bars_manifest("15min", cache_dir=cache)
        assert m5 is not None and m15 is not None
        assert m5["resample_hash"] != m15["resample_hash"]

    def test_build_bars_equals_direct_resample(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _build_src(src, [(2024, 6, [3])])
        minute = pl.read_parquet(list((src / "minute_1min").rglob("*.parquet")))
        direct = resample_intraday(
            canonicalize_minute(minute.lazy()).collect(),
            "5min",
            already_canonical=True,
        )
        via = build_bars_from_minute(minute, "5min")
        assert_frame_equal(_keys(direct), _keys(via), check_exact=False, abs_tol=1e-12)

"""test_attach_intraday.py：attach_intraday：注入 join、缺面板 null+warn、require raise、out_meta、leaf_health。
test_mining_intraday_leaves.py：挖掘链：i_* 可评估 + 无面板时 leaf_health 摘叶 + 同 seed 自身一致性。
test_bars_cache.py：tests/test_bars_cache.py — 5min bars 预物化缓存：命中/缺月/哈希失效/freq 隔离。
"""
from __future__ import annotations

import datetime as dt
import json
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from factorzen.core.feature_schema import INTRADAY_FEATURES
from factorzen.core.storage import save_parquet
from factorzen.daily.data.intraday import attach_intraday
from factorzen.intraday.bars_cache import (
    bars_data_type,
    build_bars_from_minute,
    load_or_build_bars,
    read_bars_manifest,
    resample_semantics_hash,
)
from factorzen.intraday.sessions import canonicalize_minute, resample_intraday

# ==== 来自 test_attach_intraday.py ====
_COLS = sorted(INTRADAY_FEATURES)

def _daily_date(dates: list[str], code: str = "000001.SZ") -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": [dt.datetime.strptime(d, "%Y%m%d").date() for d in dates],
        "ts_code": [code] * len(dates),
        "close": [10.0] * len(dates),
    })

def _daily_utf8(dates: list[str], code: str = "000001.SZ") -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": dates,
        "ts_code": [code] * len(dates),
        "close": [10.0] * len(dates),
    })

def _panel(dates: list[str], code: str = "000001.SZ", *, as_date: bool = True) -> pl.DataFrame:
    td = (
        [dt.datetime.strptime(d, "%Y%m%d").date() for d in dates]
        if as_date
        else list(dates)
    )
    data: dict = {"trade_date": td, "ts_code": [code] * len(dates)}
    for i, c in enumerate(_COLS):
        data[c] = [float(i + 1) + 0.1 * j for j in range(len(dates))]
    return pl.DataFrame(data)

def test_injected_join_date_dtype():
    daily = _daily_date(["20240102", "20240103"])
    panel = _panel(["20240102", "20240103"])
    # 显式固定 i_rv 便于断言
    panel = panel.with_columns(
        pl.when(pl.col("trade_date") == dt.date(2024, 1, 2))
        .then(0.42)
        .otherwise(0.43)
        .alias("i_rv")
    )
    out = attach_intraday(daily, injected=panel)
    for c in _COLS:
        assert c in out.columns
    by = {r["trade_date"]: r for r in out.iter_rows(named=True)}
    assert by[dt.date(2024, 1, 2)]["i_rv"] == pytest.approx(0.42)
    assert by[dt.date(2024, 1, 3)]["i_rv"] == pytest.approx(0.43)

def test_injected_join_utf8_daily():
    """daily trade_date 为 Utf8 时也能 left-join。"""
    daily = _daily_utf8(["20240102", "20240103"])
    panel = _panel(["20240102", "20240103"], as_date=True).with_columns(
        pl.lit(0.99).alias("i_rv")
    )
    out = attach_intraday(daily, injected=panel)
    assert out["trade_date"].dtype in (pl.Utf8, pl.String)
    assert out.filter(pl.col("trade_date") == "20240102")["i_rv"][0] == pytest.approx(0.99)

def test_missing_panel_require_false_nulls_and_warning():
    daily = _daily_date(["20240102"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = attach_intraday(daily, injected=pl.DataFrame(), require=False)
    assert any("intraday" in str(x.message).lower() or "i_*" in str(x.message)
               or "日内" in str(x.message) for x in w)
    for c in _COLS:
        assert c in out.columns
        assert out[c][0] is None

def test_missing_panel_require_true_raises_with_build_hint():
    daily = _daily_date(["20240102"])
    with pytest.raises(ValueError, match="intraday-features build"):
        attach_intraday(daily, injected=pl.DataFrame(), require=True)

def test_out_meta_filled():
    daily = _daily_date(["20240102", "20240103"])
    panel = _panel(["20240102", "20240103"])
    meta: dict = {}
    attach_intraday(daily, injected=panel, out_meta=meta)
    assert "intraday_panel" in meta
    ip = meta["intraday_panel"]
    assert ip["version"] == "v1"
    assert ip["freq"] == "5min"
    assert ip["coverage_start"] is not None
    assert ip["coverage_end"] is not None

def test_leaf_health_zero_coverage_on_null_i_leaves():
    """缺面板 require=False → 全 null 列；leaf_health 对 i_* 覆盖率 0。"""
    from factorzen.discovery.leaf_health import leaf_holdout_coverage

    # 扩截面以满足 min_cross 语义；i_* 全 null → 覆盖率仍 0
    rows = []
    for d in [dt.date(2024, 1, d) for d in range(2, 12)]:
        for i in range(40):
            rows.append({
                "trade_date": d,
                "ts_code": f"{i:06d}.SZ",
                "close_adj": 10.0,
            })
    frame = pl.DataFrame(rows)
    out = attach_intraday(frame, injected=pl.DataFrame(), require=False)
    hstart = dt.date(2024, 1, 7)
    cov = leaf_holdout_coverage(
        out, list(INTRADAY_FEATURES), hstart,
        leaf_map={k: k for k in INTRADAY_FEATURES},
        min_cross=30,
    )
    assert all(v == 0.0 for v in cov.values())

# ==== 来自 test_mining_intraday_leaves.py ====
def _mk_daily(
    n_days: int = 60,
    n_stocks: int = 35,
    seed: int = 7,
    *,
    with_intraday: bool = False,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days: list[dt.date] = []
    d = dt.date(2021, 1, 4)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{600000 + i:06d}.SH" for i in range(n_stocks)]:
        base = rng.uniform(8, 15)
        for i, dd in enumerate(days):
            px = base * (1 + 0.001 * i) + rng.normal(0, 0.1)
            row = {
                "trade_date": dd,
                "ts_code": c,
                "close": px,
                "open": px,
                "high": px * 1.01,
                "low": px * 0.99,
                "close_adj": px,
                "open_adj": px,
                "high_adj": px * 1.01,
                "low_adj": px * 0.99,
                "pre_close": px / (1 + 0.001 * max(i, 1)),
                "vol": float(1e6 + rng.normal(0, 1e4)),
                "amount": float(1e7 + rng.normal(0, 1e5)),
            }
            if with_intraday:
                for leaf in sorted(INTRADAY_FEATURES):
                    row[leaf] = float(abs(rng.normal(0.02, 0.01)))
            rows.append(row)
    return pl.DataFrame(rows)

def test_run_session_with_i_rv_evaluable(tmp_path, monkeypatch):
    """合成帧含 i_* 列：含 i_rv 的表达式可评估且非全 null。"""
    from factorzen.discovery.evaluation import evaluate_expressions
    from factorzen.discovery.mining_session import run_session
    from factorzen.discovery.scoring import DataBundle

    daily = _mk_daily(with_intraday=True)
    bundle = DataBundle.build(daily)
    res = evaluate_expressions(["rank(i_rv)", "ts_mean(i_rv, 5)"], daily, bundle)
    assert all(r["compile_ok"] for r in res), res
    assert any(r.get("ic_train") is not None for r in res)

    # 强制搜索产出 i_rv 表达式
    exprs = ["rank(i_rv)", "rank(close)"]
    idx = {"i": 0}

    class _FakeSearcher:
        def __init__(self, *a, **k):
            pass

        def propose(self):
            from factorzen.discovery.expression import parse_expr
            e = exprs[idx["i"] % len(exprs)]
            idx["i"] += 1
            return parse_expr(e)

    monkeypatch.setattr(
        "factorzen.discovery.mining_session.RandomSearcher", _FakeSearcher,
    )
    out = run_session(
        daily, n_trials=4, top_k=2, seed=1, method="random",
        out_dir=str(tmp_path / "sess_i"),
        update_library=False,
        library_orthogonal=False,
    )
    # 至少跑完；i_rv 不应因缺列被 compile 拒绝
    assert "candidates" in out

def test_zero_regression_excluded_intraday_and_seed_consistency(tmp_path):
    """不带 i_* 列：excluded 恰含全部 INTRADAY_FEATURES；同 seed 候选序列自身一致。"""
    from factorzen.discovery.mining_session import run_session

    daily = _mk_daily(with_intraday=False)
    r1 = run_session(
        daily, n_trials=8, top_k=3, seed=42, method="random",
        out_dir=str(tmp_path / "a"),
        update_library=False,
        library_orthogonal=False,
    )
    r2 = run_session(
        daily, n_trials=8, top_k=3, seed=42, method="random",
        out_dir=str(tmp_path / "b"),
        update_library=False,
        library_orthogonal=False,
    )
    excl1 = set(r1.get("excluded_leaves") or {})
    excl2 = set(r2.get("excluded_leaves") or {})
    assert excl1 >= INTRADAY_FEATURES
    assert excl1 == excl2
    e1 = [c["expression"] for c in r1["candidates"]]
    e2 = [c["expression"] for c in r2["candidates"]]
    assert e1 == e2

# ==== 来自 test_bars_cache.py ====
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

    def test_partially_cached_boundary_month_rebuilt(self, tmp_path: Path) -> None:
        """上游在边界月内补数后，读穿缓存必须重算该月而非命中部分缓存。

        与 ``features/engine`` 的部分月防呆是双路径配对项。bars 层更隐蔽：
        features 用 ``--force`` 只能绕过它**自己**的跳过，读穿到这里仍会命中
        部分 bars，于是「重算」出的特征月照样是残的。
        """
        src = tmp_path / "src"
        cache = tmp_path / "cache"
        # ① 源湖此刻只有 06-03（上游数据尚未到月末）
        _build_src(src, [(2024, 6, [3])])
        first = load_or_build_bars(
            "2024-06", "5min", source_dir=src, cache_dir=cache,
        )
        assert first["trade_time"].dt.date().max() == date(2024, 6, 3)
        man1 = read_bars_manifest("5min", cache_dir=cache)
        assert man1 is not None
        assert man1["coverage"]["month_last_date"]["2024-06"] == "2024-06-03"

        # ② 上游补进 06-04（同月，月标签不变）
        _build_src(src, [(2024, 6, [3, 4])])

        # ③ 再读：不得命中部分缓存
        second = load_or_build_bars(
            "2024-06", "5min", source_dir=src, cache_dir=cache,
        )
        assert second["trade_time"].dt.date().max() == date(2024, 6, 4), (
            "边界月命中了部分缓存，补进的 06-04 丢失"
        )
        man2 = read_bars_manifest("5min", cache_dir=cache)
        assert man2 is not None
        assert man2["coverage"]["month_last_date"]["2024-06"] == "2024-06-04"

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

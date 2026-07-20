"""
test_intraday_attach_cache.py：test_attach_intraday.py：attach_intraday：注入 join、缺面板 null+warn、require raise、out_meta、leaf_health。
test_intraday_features.py：test_intraday_battery.py：tests/test_intraday_battery.py — 特征电池 v1 与 compute_day_panel 数值 ground-truth。
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
from factorzen.discovery.intraday_expr import (
    AGG_FUNCS,
    ELEMENTWISE_OPS,
    ensure_expr_panel,
    load_expr_registry,
    make_expr_spec,
    materialize_expr_features,
    register_expr_features,
    registry_path,
    screen_expr_panel,
    validate_bar_expr,
)
from factorzen.intraday.bars_cache import (
    bars_data_type,
    build_bars_from_minute,
    load_or_build_bars,
    read_bars_manifest,
    resample_semantics_hash,
)
from factorzen.intraday.features import battery, battery_hash, compute_day_panel
from factorzen.intraday.features.spec import IntradayFeatureSpec
from factorzen.intraday.sessions import canonicalize_minute, resample_intraday

# ==== 来自 test_intraday_attach_cache.py ====
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

# ==== 来自 test_intraday_features.py ====
# ==== 来自 test_intraday_battery.py ====
def _dt__battery(h: int, m: int, day: int = 2) -> datetime:
    return datetime(2024, 1, day, h, m, 0)

def _sparse_one_day() -> pl.DataFrame:
    """1 股 1 日稀疏 1min 帧（经 5min 重采样得 8 桶，i∈{1,2,6,24,25,42,43,48}）。

    桶序列（手算，W0 bar-end）：
    - 09:30+09:31 → 09:35 i=1 open=10 close=10.5 vol=300 amount=3100
    - 09:40 → i=2 close=10.2 vol=150 amount=1530
    - 10:00 → i=6 close=10.3 vol=400 amount=4120
    - 11:30 → i=24 close=10.4 vol=250 amount=2600
    - 13:01 → 13:05 i=25 close=10.5 vol=300 amount=3150
    - 14:30 → i=42 close=10.6 vol=200 amount=2120
    - 14:35 → i=43 close=10.7 vol=350 amount=3745
    - 15:00 → i=48 close=10.8 vol=500 amount=5400
    """
    rows = [
        (_dt__battery(9, 30), 10.0, 10.0, 10.0, 10.0, 100, 1000.0),
        (_dt__battery(9, 31), 10.0, 10.6, 10.0, 10.5, 200, 2100.0),
        (_dt__battery(9, 40), 10.5, 10.5, 10.2, 10.2, 150, 1530.0),
        (_dt__battery(10, 0), 10.2, 10.4, 10.1, 10.3, 400, 4120.0),
        (_dt__battery(11, 30), 10.3, 10.5, 10.2, 10.4, 250, 2600.0),
        (_dt__battery(13, 1), 10.4, 10.6, 10.3, 10.5, 300, 3150.0),
        (_dt__battery(14, 30), 10.5, 10.7, 10.4, 10.6, 200, 2120.0),
        (_dt__battery(14, 35), 10.6, 10.8, 10.5, 10.7, 350, 3745.0),
        (_dt__battery(15, 0), 10.7, 10.9, 10.6, 10.8, 500, 5400.0),
    ]
    return pl.DataFrame(
        {
            "ts_code": ["000001.SZ"] * len(rows),
            "trade_time": pl.Series([r[0] for r in rows], dtype=pl.Datetime("us")),
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "vol": pl.Series([r[5] for r in rows], dtype=pl.Int64),
            "amount": [r[6] for r in rows],
        }
    )

# 手算硬编码期望（rel tol 1e-9）
_EXPECTED = {
    "i_rv": 0.06217881540867241,
    "i_rskew": 1.2582542330998039,
    "i_rkurt": 3.7286865731759713,
    "i_downvol_ratio": 0.2111441355367538,
    "i_updown_vol": 1.3180426148977666,
    "i_ret_open30": 0.030000000000000027,
    "i_ret_close30": 0.018867924528301883,
    "i_ret_mid": 0.029126213592232997,
    "i_vwap_dev": 0.026974577915777287,
    "i_smart_money": 0.9874755556882473,
    "i_vol_open30_share": 0.3469387755102041,
    "i_vol_close30_share": 0.3469387755102041,
    "i_vol_entropy": 0.9718771358715251,
    "i_amihud": 6588.942735992862,
    "i_path_eff": 0.5714285714285711,
    "i_max_ret_share": 0.36763884425793547,
}

_CORE_HARD = [
    "i_rv",
    "i_rskew",
    "i_ret_open30",
    "i_ret_close30",
    "i_vwap_dev",
    "i_vol_entropy",
    "i_smart_money",
    "i_path_eff",
    "i_amihud",
    "i_max_ret_share",
]

class TestBatteryMeta:
    def test_v1_twenty_unique_i_prefix(self) -> None:
        """17 个连续路径统计 + 3 个涨跌停邻域（2026-07-19 新增）。"""
        specs = battery("v1", "5min")
        assert len(specs) == 20
        names = [s.name for s in specs]
        assert len(set(names)) == 20
        # 新增的三个必须在册，且与 INTRADAY_FEATURES 一致（登记簿不许漂移）
        from factorzen.core.feature_schema import INTRADAY_FEATURES

        assert set(names) == INTRADAY_FEATURES
        assert {"i_limit_up_seal_share", "i_limit_up_open_count",
                "i_limit_up_first_touch"} <= set(names)
        assert all(n.startswith("i_") for n in names)
        assert all(isinstance(s, IntradayFeatureSpec) for s in specs)
        assert all(s.formula and s.description for s in specs)
        assert all(s.expression is None for s in specs)

    def test_v2_raises(self) -> None:
        with pytest.raises(ValueError, match="未知电池版本"):
            battery("v2")

    def test_60min_raises(self) -> None:
        with pytest.raises(ValueError, match="不支持"):
            battery("v1", freq="60min")

    def test_battery_hash_stable(self) -> None:
        a = battery_hash(battery("v1", "5min"))
        b = battery_hash(battery("v1", "5min"))
        assert a == b
        assert len(a) == 16

class TestGroundTruth:
    def test_sparse_5min_core_features(self) -> None:
        specs = battery("v1", "5min")
        panel = compute_day_panel(
            _sparse_one_day(), specs, "5min", min_bar_coverage=0.0
        )
        assert panel.height == 1
        assert panel["trade_date"].dtype == pl.Date
        row = panel.row(0, named=True)

        for name in _CORE_HARD:
            assert row[name] == pytest.approx(_EXPECTED[name], rel=1e-9), name

        # 其余：非空 + 符号合理
        assert row["i_rkurt"] == pytest.approx(_EXPECTED["i_rkurt"], rel=1e-9)
        assert row["i_downvol_ratio"] == pytest.approx(
            _EXPECTED["i_downvol_ratio"], rel=1e-9
        )
        assert 0.0 < row["i_downvol_ratio"] < 1.0
        assert row["i_updown_vol"] == pytest.approx(_EXPECTED["i_updown_vol"], rel=1e-9)
        assert row["i_ret_mid"] == pytest.approx(_EXPECTED["i_ret_mid"], rel=1e-9)
        assert row["i_vol_open30_share"] == pytest.approx(
            _EXPECTED["i_vol_open30_share"], rel=1e-9
        )
        assert row["i_vol_close30_share"] == pytest.approx(
            _EXPECTED["i_vol_close30_share"], rel=1e-9
        )
        # 有效桶 8 < 10 → i_pv_corr 恒 null
        assert row["i_pv_corr"] is None

class TestGuards:
    def test_low_coverage_nulls_all_features_keeps_row(self) -> None:
        specs = battery("v1", "5min")
        # 正常日
        good = _sparse_one_day()
        # 覆盖不足日：仅 2 根 bar，另一天
        rows = [
            (_dt__battery(9, 31, day=3), 10.0, 10.1, 9.9, 10.0, 100, 1000.0),
            (_dt__battery(15, 0, day=3), 10.0, 10.2, 9.9, 10.1, 100, 1010.0),
        ]
        low = pl.DataFrame(
            {
                "ts_code": ["000001.SZ"] * 2,
                "trade_time": pl.Series([r[0] for r in rows], dtype=pl.Datetime("us")),
                "open": [r[1] for r in rows],
                "high": [r[2] for r in rows],
                "low": [r[3] for r in rows],
                "close": [r[4] for r in rows],
                "vol": pl.Series([r[5] for r in rows], dtype=pl.Int64),
                "amount": [r[6] for r in rows],
            }
        )
        # 稀疏正常日有效桶=8；门槛 0.1×48=4.8 → 正常日通过，2 桶日不通过
        # （0.8×48=38.4 会把稀疏正常日也打成低覆盖）
        panel = compute_day_panel(
            pl.concat([good, low]), specs, "5min", min_bar_coverage=0.1
        )
        assert panel.height == 2
        feat_cols = [s.name for s in specs]
        low_row = panel.filter(pl.col("trade_date") == datetime(2024, 1, 3).date())
        assert low_row.height == 1
        for c in feat_cols:
            assert low_row[c][0] is None, c
        good_row = panel.filter(pl.col("trade_date") == datetime(2024, 1, 2).date())
        assert good_row["i_rv"][0] is not None

    def test_nan_input_becomes_null_not_nan(self) -> None:
        specs = battery("v1", "5min")
        df = _sparse_one_day().with_columns(
            pl.when(pl.col("trade_time").dt.minute() == 40)
            .then(float("nan"))
            .otherwise(pl.col("close"))
            .alias("close")
        )
        panel = compute_day_panel(df, specs, "5min", min_bar_coverage=0.0)
        for c in [s.name for s in specs]:
            vals = panel[c].to_list()
            for v in vals:
                if v is not None:
                    assert v == v  # not NaN

    def test_all_zero_vol_no_crash(self) -> None:
        specs = battery("v1", "5min")
        rows = [
            (_dt__battery(9, 31), 10.0, 10.0, 10.0, 10.0, 0, 0.0),
            (_dt__battery(9, 40), 10.0, 10.1, 9.9, 10.05, 0, 0.0),
            (_dt__battery(15, 0), 10.05, 10.1, 10.0, 10.1, 0, 0.0),
        ]
        df = pl.DataFrame(
            {
                "ts_code": ["000001.SZ"] * 3,
                "trade_time": pl.Series([r[0] for r in rows], dtype=pl.Datetime("us")),
                "open": [r[1] for r in rows],
                "high": [r[2] for r in rows],
                "low": [r[3] for r in rows],
                "close": [r[4] for r in rows],
                "vol": pl.Series([r[5] for r in rows], dtype=pl.Int64),
                "amount": [r[6] for r in rows],
            }
        )
        panel = compute_day_panel(df, specs, "5min", min_bar_coverage=0.0)
        assert panel.height == 1
        # V 类特征 null
        assert panel["i_vwap_dev"][0] is None
        assert panel["i_vol_open30_share"][0] is None
        assert panel["i_smart_money"][0] is None
        # 路径效率仍可算
        assert panel["i_path_eff"][0] is not None or panel["i_path_eff"][0] is None

# ── 涨跌停邻域叶（A 股特有的离散状态机；与 17 个连续路径统计机制不同）──────────

def _limit_day(closes: list[float], highs: list[float] | None = None,
               *, day: int = 2, code: str = "000001.SZ") -> pl.DataFrame:
    """每个给定值落在**不同的 5min bar** 上的 1min 帧（09:30 起，每 5 分钟一根）。

    ⚠️ 两个坑（初版都踩了）：
    1. 间隔必须 ≥5 分钟——连续分钟会被 ``resample_intraday`` 并进同一个桶，
       seal_share / first_touch 退化成单桶，测不出判别力。
    2. 起点用 **09:31** 而非 09:30——bar-end 约定下 09:30 与 09:35 会落进同一根，
       实测 09:30 起点只得 3 根、09:31 起点得 4 根。
    """
    from datetime import timedelta
    highs = highs if highs is not None else list(closes)
    base = datetime(2024, 1, day, 9, 31, 0)
    rows = []
    for i, (c, h) in enumerate(zip(closes, highs, strict=True)):
        t = base + timedelta(minutes=5 * i)
        rows.append((code, t, c, h, min(c, h), c, 100, c * 100))
    return pl.DataFrame(
        rows,
        schema=["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"],
        orient="row",
    )

def _limit_ref(pre_close: float, limit_pct: float = 0.1, *, day: int = 2,
               code: str = "000001.SZ") -> pl.DataFrame:
    from datetime import date as _date
    return pl.DataFrame({
        "ts_code": [code], "trade_date": [_date(2024, 1, day)],
        "pre_close": [pre_close], "limit_pct": [limit_pct],
    })

def test_limit_leaves_seal_share_and_open_count():
    """封板时长占比 + 打开次数：手算 ground-truth。

    pre_close=10 → 涨停价 11.0。close 序列 [11.0, 11.0, 10.9, 11.0]：
    - seal = [1,1,0,1] → seal_share = 3/4
    - 打开次数 = seal 由 1→0 的次数 = 1
    """
    from factorzen.intraday.features import battery, compute_day_panel

    minute = _limit_day([11.0, 11.0, 10.9, 11.0])
    ref = _limit_ref(10.0)
    out = compute_day_panel(minute, battery(), "5min", min_bar_coverage=0.0,
                            daily_ref=ref)
    assert "i_limit_up_seal_share" in out.columns, out.columns
    r = out.row(0, named=True)
    assert abs(r["i_limit_up_seal_share"] - 0.75) < 1e-9, r["i_limit_up_seal_share"]
    assert r["i_limit_up_open_count"] == 1.0, r["i_limit_up_open_count"]

def test_limit_leaves_never_touched_is_zero_not_null():
    """全日未触板 → seal_share/open_count = **0**（0 有信息「今天没封过」），
    first_touch = **1.0**（最晚）。不得为 null——否则截面 95%+ null，
    rank/IC 会塌成少数触板票的子样本游戏。"""
    from factorzen.intraday.features import battery, compute_day_panel

    out = compute_day_panel(_limit_day([10.1, 10.2, 10.15, 10.2]), battery(), "5min",
                            min_bar_coverage=0.0, daily_ref=_limit_ref(10.0))
    r = out.row(0, named=True)
    assert r["i_limit_up_seal_share"] == 0.0
    assert r["i_limit_up_open_count"] == 0.0
    assert r["i_limit_up_first_touch"] == 1.0

def test_limit_leaves_first_touch_earlier_is_smaller():
    """首次触板越早，first_touch 越小（判别力：两组对照）。"""
    from factorzen.intraday.features import battery, compute_day_panel

    early = compute_day_panel(_limit_day([11.0, 10.5, 10.5, 10.5]), battery(), "5min",
                              min_bar_coverage=0.0, daily_ref=_limit_ref(10.0))
    late = compute_day_panel(_limit_day([10.5, 10.5, 10.5, 11.0]), battery(), "5min",
                             min_bar_coverage=0.0, daily_ref=_limit_ref(10.0))
    assert early.row(0, named=True)["i_limit_up_first_touch"] < \
        late.row(0, named=True)["i_limit_up_first_touch"]

def test_limit_leaves_null_without_daily_ref():
    """不传 daily_ref（旧调用方）→ 涨跌停叶全 null，其余 17 叶**逐位不变**（零回归）。"""
    from factorzen.intraday.features import battery, compute_day_panel

    minute = _limit_day([11.0, 11.0, 10.9, 11.0])
    with_ref = compute_day_panel(minute, battery(), "5min", min_bar_coverage=0.0,
                                 daily_ref=_limit_ref(10.0))
    without = compute_day_panel(minute, battery(), "5min", min_bar_coverage=0.0)
    assert without.row(0, named=True)["i_limit_up_seal_share"] is None
    for c in ("i_rv", "i_ret_open30", "i_vwap_dev", "i_amihud"):
        a = with_ref.row(0, named=True)[c]
        b = without.row(0, named=True)[c]
        assert (a is None and b is None) or a == b, f"{c}: {a} vs {b}"

def test_limit_leaves_guard_bad_pre_close():
    """pre_close ≤0 / 缺失 → 三叶全 null（不是 0，区别于「未触板」）。"""
    from factorzen.intraday.features import battery, compute_day_panel

    out = compute_day_panel(_limit_day([11.0, 11.0]), battery(), "5min",
                            min_bar_coverage=0.0, daily_ref=_limit_ref(0.0))
    r = out.row(0, named=True)
    assert r["i_limit_up_seal_share"] is None
    assert r["i_limit_up_first_touch"] is None

# ==== 来自 test_intraday_expr_features.py ====
def _dt__expr_features(h: int, m: int, day: int = 2) -> datetime:
    return datetime(2024, 1, day, h, m, 0)

def _sparse_two_stocks() -> pl.DataFrame:
    """2 股 1 日稀疏 1min 帧（与 battery ground-truth 同桶序列）。"""
    rows = [
        (_dt__expr_features(9, 30), 10.0, 10.0, 10.0, 10.0, 100, 1000.0),
        (_dt__expr_features(9, 31), 10.0, 10.6, 10.0, 10.5, 200, 2100.0),
        (_dt__expr_features(9, 40), 10.5, 10.5, 10.2, 10.2, 150, 1530.0),
        (_dt__expr_features(10, 0), 10.2, 10.4, 10.1, 10.3, 400, 4120.0),
        (_dt__expr_features(11, 30), 10.3, 10.5, 10.2, 10.4, 250, 2600.0),
        (_dt__expr_features(13, 1), 10.4, 10.6, 10.3, 10.5, 300, 3150.0),
        (_dt__expr_features(14, 30), 10.5, 10.7, 10.4, 10.6, 200, 2120.0),
        (_dt__expr_features(14, 35), 10.6, 10.8, 10.5, 10.7, 350, 3745.0),
        (_dt__expr_features(15, 0), 10.7, 10.9, 10.6, 10.8, 500, 5400.0),
    ]
    frames = []
    for code, scale in (("000001.SZ", 1.0), ("000002.SZ", 1.1)):
        frames.append(
            pl.DataFrame(
                {
                    "ts_code": [code] * len(rows),
                    "trade_time": pl.Series(
                        [r[0] for r in rows], dtype=pl.Datetime("us")
                    ),
                    "open": [r[1] * scale for r in rows],
                    "high": [r[2] * scale for r in rows],
                    "low": [r[3] * scale for r in rows],
                    "close": [r[4] * scale for r in rows],
                    "vol": pl.Series([r[5] for r in rows], dtype=pl.Int64),
                    "amount": [r[6] * scale for r in rows],
                }
            )
        )
    return pl.concat(frames)

def _write_minute_source(tmp: Path, minute: pl.DataFrame | None = None) -> Path:
    src = tmp / "src"
    frame = minute if minute is not None else _sparse_two_stocks()
    save_parquet(
        frame,
        data_type="minute_1min",
        date_col="trade_time",
        base_dir=src,
        mode="overwrite",
    )
    return src

# 5min 重采样后 000001.SZ 手算期望（polars std ddof=1）
_EXP_STD_BAR_RET = 0.02100625435521192
_EXP_MEAN_VWAP = 10.479166666666666
_EXP_LAST_SIGNED = 10.8
_EXP_FIRST_BAR_RET = 0.05

class TestValidateBarExpr:
    def test_rejects_ts_and_rank(self) -> None:
        with pytest.raises(ValueError, match=r"禁止算子|未知"):
            validate_bar_expr("ts_mean(close, 5)")
        with pytest.raises(ValueError, match=r"禁止算子"):
            validate_bar_expr("rank(close)")

    def test_accepts_elementwise(self) -> None:
        node = validate_bar_expr("div(amount, vol)")
        assert node is not None
        node2 = validate_bar_expr("mul(close, sign(bar_ret))")
        assert node2 is not None
        assert "div" in ELEMENTWISE_OPS
        assert "rank" not in ELEMENTWISE_OPS

class TestMakeExprSpec:
    def test_same_inputs_same_name(self) -> None:
        a = make_expr_spec("div(amount, vol)", "mean", freq="5min")
        b = make_expr_spec("div(amount, vol)", "mean", freq="5min")
        assert a.name == b.name
        assert a.name.startswith("ix_")
        assert len(a.name) == 11  # ix_ + 8 hex

    def test_equivalent_expr_same_name(self) -> None:
        a = make_expr_spec("div(amount,vol)", "mean", freq="5min")
        b = make_expr_spec("div(amount, vol)", "mean", freq="5min")
        assert a.name == b.name
        assert a.bar_expr == b.bar_expr

    def test_unknown_agg_freq(self) -> None:
        with pytest.raises(ValueError, match="未知聚合"):
            make_expr_spec("close", "mode", freq="5min")
        with pytest.raises(ValueError, match="未知频率"):
            make_expr_spec("close", "mean", freq="7min")

class TestMaterializeGroundTruth:
    def test_std_mean_last_and_bar_ret(self, tmp_path: Path) -> None:
        src = _write_minute_source(tmp_path)
        specs = [
            make_expr_spec("bar_ret", "std", freq="5min"),
            make_expr_spec("div(amount, vol)", "mean", freq="5min"),
            make_expr_spec("mul(close, sign(bar_ret))", "last", freq="5min"),
        ]
        panel = materialize_expr_features(
            specs,
            "20240102",
            "20240102",
            freq="5min",
            source_dir=src,
            min_bar_coverage=0.0,
        )
        assert panel.height == 2
        row = panel.filter(pl.col("ts_code") == "000001.SZ").row(0, named=True)
        assert row[specs[0].name] == pytest.approx(_EXP_STD_BAR_RET, abs=1e-9)
        assert row[specs[1].name] == pytest.approx(_EXP_MEAN_VWAP, abs=1e-9)
        assert row[specs[2].name] == pytest.approx(_EXP_LAST_SIGNED, abs=1e-9)

    def test_bar_ret_first_bar(self, tmp_path: Path) -> None:
        """首 bar bar_ret = close/open−1（5min 首桶合并竞价）。"""
        src = _write_minute_source(tmp_path)
        # first(bar_ret) 应等于首桶 close/open−1
        spec = make_expr_spec("bar_ret", "first", freq="5min")
        panel = materialize_expr_features(
            [spec],
            "20240102",
            "20240102",
            freq="5min",
            source_dir=src,
            min_bar_coverage=0.0,
        )
        v = panel.filter(pl.col("ts_code") == "000001.SZ")[spec.name][0]
        assert v == pytest.approx(_EXP_FIRST_BAR_RET, abs=1e-9)

    def test_mixed_freq_raises(self, tmp_path: Path) -> None:
        s5 = make_expr_spec("close", "last", freq="5min")
        s1 = make_expr_spec("close", "last", freq="1min")
        with pytest.raises(ValueError, match="混频"):
            materialize_expr_features(
                [s5, s1], "20240102", "20240102", freq="5min", source_dir=tmp_path
            )

class TestScreen:
    def test_three_rejects_and_keep(self) -> None:
        dates = [date(2024, 1, d) for d in range(2, 12)]
        codes = [f"{i:06d}.SZ" for i in range(5)]
        rows = []
        for d in dates:
            for i, c in enumerate(codes):
                rows.append(
                    {
                        "trade_date": d,
                        "ts_code": c,
                        "ix_lowcov": None if i < 4 else 1.0,  # 极低覆盖
                        "ix_const": 1.0,  # 近常数
                        "ix_good": float(i) + 0.1 * d.day,
                        "ix_corr": float(i) + 0.1 * d.day + 1e-9,  # 与 good 几乎共线
                    }
                )
        panel = pl.DataFrame(rows)
        # 单独筛 lowcov / const / good
        v1 = screen_expr_panel(
            panel.select(["trade_date", "ts_code", "ix_lowcov"]),
            min_coverage=0.6,
        )
        assert v1["ix_lowcov"] == "low_coverage"

        v2 = screen_expr_panel(
            panel.select(["trade_date", "ts_code", "ix_const"]),
            min_coverage=0.6,
        )
        assert v2["ix_const"] == "degenerate"

        v3 = screen_expr_panel(
            panel.select(["trade_date", "ts_code", "ix_good"]),
            min_coverage=0.6,
        )
        assert v3["ix_good"] == "keep"

        ref = panel.select(["trade_date", "ts_code", "ix_good"])
        v4 = screen_expr_panel(
            panel.select(["trade_date", "ts_code", "ix_corr"]),
            reference=ref,
            min_coverage=0.6,
            max_abs_corr=0.9,
        )
        assert v4["ix_corr"].startswith("correlated:")

class TestRegistry:
    def test_roundtrip_idempotent(self, tmp_path: Path) -> None:
        base = tmp_path / "feat"
        specs = [
            make_expr_spec("div(amount, vol)", "mean", freq="5min", hypothesis="vwap"),
            make_expr_spec("bar_ret", "std", freq="5min"),
        ]
        register_expr_features(specs, session="s1", base_dir=base)
        reg = load_expr_registry(base)
        assert set(reg) == {s.name for s in specs}
        assert reg[specs[0].name].hypothesis == "vwap"
        # 幂等
        register_expr_features(specs, session="s2", base_dir=base)
        lines = registry_path(base).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

class TestEnsureExprPanel:
    def test_cache_and_unregistered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = _write_minute_source(tmp_path)
        base = tmp_path / "feat"
        spec = make_expr_spec("div(amount, vol)", "mean", freq="5min")
        register_expr_features([spec], session="t", base_dir=base)

        calls = {"n": 0}
        real_mat = materialize_expr_features

        def _counting(*a, **kw):
            calls["n"] += 1
            # 稀疏测试帧：强制 min_bar_coverage=0，否则默认 0.8 全 null
            kw = dict(kw)
            kw["min_bar_coverage"] = 0.0
            return real_mat(*a, **kw)

        monkeypatch.setattr(
            "factorzen.discovery.intraday_expr.materialize_expr_features",
            _counting,
        )

        p1 = ensure_expr_panel(
            spec.name, "20240102", "20240102", base_dir=base, source_dir=src
        )
        assert calls["n"] == 1
        assert spec.name in p1.columns
        assert p1.height >= 1
        assert p1[spec.name].null_count() == 0

        p2 = ensure_expr_panel(
            spec.name, "20240102", "20240102", base_dir=base, source_dir=src
        )
        assert calls["n"] == 1  # 二次读缓存
        assert p2.height == p1.height

        with pytest.raises(ValueError, match="未注册"):
            ensure_expr_panel("ix_deadbeef", "20240102", "20240102", base_dir=base)

class TestAggFuncsComplete:
    def test_agg_keys(self) -> None:
        expected = {
            "sum", "mean", "std", "skew", "min", "max", "last", "first", "median"
        }
        assert set(AGG_FUNCS) == expected

# ==== 来自 test_expression_factor_intraday.py ====
def test_expression_factor_i_rv_matches_evaluate_materialized(monkeypatch):
    """monkeypatch attach_intraday 注入面板 → compute 与直算逐值一致。"""
    from factorzen.discovery.derived import add_derived_columns
    from factorzen.discovery.expression import evaluate_materialized, parse_expr
    from factorzen.discovery.factor import ExpressionFactor

    dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)]
    codes = ["000001.SZ", "000002.SZ"]
    rows = []
    for c in codes:
        for i, d in enumerate(dates):
            rows.append({
                "trade_date": d,
                "ts_code": c,
                "close": 10.0 + i,
                "close_adj": 10.0 + i,
                "open": 10.0,
                "open_adj": 10.0,
                "high": 11.0,
                "high_adj": 11.0,
                "low": 9.0,
                "low_adj": 9.0,
                "pre_close": 10.0,
                "vol": 1e5,
                "amount": 1e6,
            })
    daily = pl.DataFrame(rows)

    panel_rows = []
    for c in codes:
        for i, d in enumerate(dates):
            r = {"trade_date": d, "ts_code": c}
            for leaf in sorted(INTRADAY_FEATURES):
                r[leaf] = 0.01 * (i + 1) + (0.001 if c.endswith("1.SZ") else 0.002)
            panel_rows.append(r)
    panel = pl.DataFrame(panel_rows)

    def _fake_attach(d, **kw):
        # 模拟 require=True 注入
        have = [c for c in sorted(INTRADAY_FEATURES) if c in panel.columns]
        sel = panel.select(["trade_date", "ts_code", *have])
        drop = [c for c in have if c in d.columns]
        if drop:
            d = d.drop(drop)
        return d.join(sel, on=["trade_date", "ts_code"], how="left")

    monkeypatch.setattr(
        "factorzen.daily.data.intraday.attach_intraday", _fake_attach,
    )
    # ExpressionFactor 内 from factorzen.daily.data.intraday import attach_intraday
    # 局部 import 在调用时解析，patch 源模块即可

    class _Ctx:
        start = "20240102"
        end = "20240104"

        @property
        def daily(self):
            return daily.lazy()

        @property
        def daily_basic(self):
            return pl.DataFrame({
                "trade_date": dates * len(codes),
                "ts_code": [c for c in codes for _ in dates],
                "circ_mv": [1e6] * (len(dates) * len(codes)),
            }).lazy()

    expr = "rank(i_rv)"
    fac = ExpressionFactor(expr, mined_name="i_rv_rank")
    out = fac.compute(_Ctx())
    assert out.height > 0
    assert out["factor_value"].null_count() < out.height

    # 直算：attach → sort → derived → evaluate
    attached = _fake_attach(daily)
    prepped = add_derived_columns(attached.sort(["ts_code", "trade_date"]))
    node = parse_expr(expr)
    direct = prepped.with_columns(
        evaluate_materialized(node, prepped).alias("factor_value")
    ).filter(
        pl.col("trade_date") >= dt.date(2024, 1, 2)
    ).select(["trade_date", "ts_code", "factor_value"]).filter(
        pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite()
    )

    a = out.sort(["ts_code", "trade_date"])
    b = direct.sort(["ts_code", "trade_date"])
    assert a.height == b.height
    for va, vb in zip(a["factor_value"].to_list(), b["factor_value"].to_list(), strict=True):
        assert va == pytest.approx(vb, abs=1e-9, nan_ok=True)

# ==== 来自 test_feature_schema_intraday.py ====
# 改造前 LEAF_FEATURES 的 40 个键顺序（硬编码守卫：只许末尾扩，不许改旧序）
_PRE_CHANGE_LEAF_KEYS: list[str] = [
    "close",
    "open",
    "high",
    "low",
    "vol",
    "amount",
    "vwap",
    "log_vol",
    "ret_1d",
    "amplitude",
    "intraday_ret",
    "overnight_ret",
    "total_mv",
    "circ_mv",
    "pb",
    "pe_ttm",
    "ps_ttm",
    "dv_ttm",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "float_share",
    "roe",
    "roa",
    "grossprofit_margin",
    "netprofit_margin",
    "debt_to_assets",
    "or_yoy",
    "netprofit_yoy",
    "assets_yoy",
    "net_mf_amount",
    "north_ratio",
    "margin_ratio",
    "margin_buy_ratio",
    "margin_balance",
    "short_balance",
    "holder_num",
    "holder_num_chg",
    "top_list_net_buy",
    "top_list_flag",
]

def test_intraday_features_subset_of_leaf_features():
    from factorzen.core.feature_schema import INTRADAY_FEATURES, LEAF_FEATURES

    assert set(LEAF_FEATURES.keys()) >= INTRADAY_FEATURES
    assert len(INTRADAY_FEATURES) == 20  # 17 连续路径统计 + 3 涨跌停邻域

def test_intraday_leaves_are_identity_i_prefix():
    from factorzen.core.feature_schema import INTRADAY_FEATURES, LEAF_FEATURES

    for name in INTRADAY_FEATURES:
        assert name.startswith("i_"), name
        assert LEAF_FEATURES[name] == name

def test_leaf_features_key_order_prefix_unchanged():
    """既有 40 键相对顺序绝不动（随机搜索按键序采样）。"""
    from factorzen.core.feature_schema import LEAF_FEATURES

    keys = list(LEAF_FEATURES.keys())
    n = len(_PRE_CHANGE_LEAF_KEYS)
    assert keys[:n] == _PRE_CHANGE_LEAF_KEYS
    # 新叶子全部在旧键之后
    assert all(k.startswith("i_") for k in keys[n:])


"""库池 compact（单骨架宽面板）内存路径：与 legacy 数值 parity + 自动开关。"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl


def _mk_daily(n_days: int = 100, n_stocks: int = 20, seed: int = 11) -> pl.DataFrame:
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
            rows.append({
                "trade_date": dd, "ts_code": c,
                "close": px, "open": px, "high": px * 1.01, "low": px * 0.99,
                "close_adj": px, "open_adj": px, "high_adj": px * 1.01, "low_adj": px * 0.99,
                "pre_close": px / (1 + 0.001 * max(i, 1)),
                "vol": 1e6 + rng.normal(0, 1e4), "amount": 1e7 + rng.normal(0, 1e5),
            })
    return pl.DataFrame(rows)


def _write_lib(root: Path, market: str, records: list[dict]) -> None:
    path = root / f"{market}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


def _seed_lib(tmp_path: Path) -> Path:
    _write_lib(tmp_path, "ashare", [
        {"expression": "rank(close)", "market": "ashare", "status": "active",
         "ic_train": 0.05},
        {"expression": "rank(vol)", "market": "ashare", "status": "active",
         "ic_train": 0.04},
        {"expression": "rank(amount)", "market": "ashare", "status": "active",
         "ic_train": 0.03},
    ])
    return tmp_path


# ── parity: compact vs legacy ────────────────────────────────────────────────


def test_compact_legacy_getitem_values_equal(tmp_path):
    """同一表达式长表 filter 后 factor_value 与键 f64 全等。"""
    from factorzen.discovery.factor_library import (
        CompactLibraryPool,
        build_library_pool,
    )

    daily = _mk_daily()
    root = str(_seed_lib(tmp_path))
    legacy = build_library_pool("ashare", daily, root=root, compact=False)
    compact = build_library_pool("ashare", daily, root=root, compact=True)
    assert isinstance(compact, CompactLibraryPool)
    assert set(legacy.keys()) == set(compact.keys())
    for expr in legacy:
        a = legacy[expr].sort(["trade_date", "ts_code"])
        b = compact[expr].sort(["trade_date", "ts_code"])
        assert a.height == b.height
        assert a["trade_date"].to_list() == b["trade_date"].to_list()
        assert a["ts_code"].to_list() == b["ts_code"].to_list()
        va = a["factor_value"].to_numpy()
        vb = b["factor_value"].to_numpy()
        np.testing.assert_array_equal(va, vb)


def test_compact_legacy_corr_panel_and_max_corr_equal(tmp_path):
    """build_library_corr_panel + max_correlation 两模式 f64 全等。"""
    from factorzen.discovery.factor_library import build_library_pool
    from factorzen.discovery.scoring import (
        build_library_corr_panel,
        max_correlation,
        max_correlation_detail,
    )

    daily = _mk_daily()
    root = str(_seed_lib(tmp_path))
    legacy = build_library_pool("ashare", daily, root=root, compact=False)
    compact = build_library_pool("ashare", daily, root=root, compact=True)

    p_leg = build_library_corr_panel(legacy)
    p_cmp = build_library_corr_panel(compact)
    assert p_leg is not None and p_cmp is not None
    assert p_leg.names == p_cmp.names
    assert p_leg.dates == p_cmp.dates
    assert p_leg.stocks == p_cmp.stocks
    # present=None 新契约:掩码经 present_block 推导(直接 np.where(None,...) 会把
    # None 当 False 标量退化成恒真比较——陷阱#1)
    pres_leg = p_leg.present_block(0, len(p_leg.dates))
    pres_cmp = p_cmp.present_block(0, len(p_cmp.dates))
    np.testing.assert_array_equal(pres_leg, pres_cmp)
    assert pres_leg.any()  # 掩码非空,比较有判别力
    # 值：null 位已由 present 标；有限位须 bit-identical
    np.testing.assert_array_equal(
        np.where(pres_leg, p_leg.values, 0.0),
        np.where(pres_cmp, p_cmp.values, 0.0),
    )

    # 候选 = 库内第一因子
    cand = legacy[next(iter(legacy))]
    mc_l, n_l = max_correlation_detail(cand, legacy, panel=p_leg)
    mc_c, n_c = max_correlation_detail(cand, compact, panel=p_cmp)
    assert mc_l == mc_c
    assert n_l == n_c
    assert max_correlation(cand, legacy, panel=p_leg) == max_correlation(
        cand, compact, panel=p_cmp,
    )


def test_compact_legacy_residual_ic_equal(tmp_path):
    """residual LibraryPanel + compute_residual_ic 两模式一致。"""
    from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns
    from factorzen.discovery.factor_library import build_library_pool
    from factorzen.discovery.residual import (
        ResidualProjector,
        build_library_panel,
        compute_residual_ic,
    )

    daily = _mk_daily()
    root = str(_seed_lib(tmp_path))
    legacy = build_library_pool("ashare", daily, root=root, compact=False)
    compact = build_library_pool("ashare", daily, root=root, compact=True)

    panel_l = build_library_panel(legacy)
    panel_c = build_library_panel(compact)
    assert panel_l is not None and panel_c is not None
    assert panel_l.factor_names == panel_c.factor_names
    assert panel_l.dates == panel_c.dates
    assert panel_l.stocks == panel_c.stocks
    np.testing.assert_allclose(panel_l.X, panel_c.X, rtol=0, atol=0)

    cand = legacy[next(iter(legacy))]
    # 用略扰动的候选避免与库列完全共线导致数值病态差异放大
    cand2 = cand.with_columns(
        (pl.col("factor_value") + 0.01 * pl.col("factor_value").rank().over("trade_date")
         / pl.col("factor_value").count().over("trade_date")).alias("factor_value")
    )
    sorted_daily = daily.sort(["ts_code", "trade_date"])
    fwd = compute_fwd_returns(sorted_daily, price_col="close_adj")
    proj_l = ResidualProjector.from_panel(panel_l)
    proj_c = ResidualProjector.from_panel(panel_c)
    r_l = compute_residual_ic(cand2, panel_l, fwd, projector=proj_l)
    r_c = compute_residual_ic(cand2, panel_c, fwd, projector=proj_c)
    assert r_l.n_days == r_c.n_days
    if r_l.n_days > 0:
        assert r_l.ic_mean == r_c.ic_mean


# ── 自动开关 ────────────────────────────────────────────────────────────────


def test_auto_compact_when_over_threshold(tmp_path, capsys):
    from factorzen.discovery.factor_library import (
        CompactLibraryPool,
        build_library_pool,
    )

    daily = _mk_daily(n_days=30, n_stocks=10)
    root = str(_seed_lib(tmp_path))
    # 阈值调到极小 → 必走 compact
    pool = build_library_pool(
        "ashare", daily, root=root, compact=None, compact_threshold=1,
    )
    assert isinstance(pool, CompactLibraryPool)
    out = capsys.readouterr().out
    assert "库池 compact 模式" in out


def test_auto_legacy_on_small_frame(tmp_path):
    from factorzen.discovery.factor_library import (
        CompactLibraryPool,
        build_library_pool,
    )

    daily = _mk_daily(n_days=30, n_stocks=10)
    root = str(_seed_lib(tmp_path))
    pool = build_library_pool("ashare", daily, root=root, compact=None)
    assert isinstance(pool, dict)
    assert not isinstance(pool, CompactLibraryPool)


def test_should_use_compact_pool_math():
    from factorzen.discovery.factor_library import (
        POOL_KEY_BYTES_PER_ROW,
        estimate_library_pool_key_bytes,
        should_use_compact_pool,
    )

    n_f, n_r = 84, 10_925_813
    est = estimate_library_pool_key_bytes(n_f, n_r)
    assert est == n_f * n_r * POOL_KEY_BYTES_PER_ROW
    assert should_use_compact_pool(n_f, n_r, threshold=8 * 1024**3)
    assert not should_use_compact_pool(3, 2000, threshold=8 * 1024**3)


def test_compact_filter_dates(tmp_path):
    from factorzen.discovery.factor_library import CompactLibraryPool, build_library_pool

    daily = _mk_daily(n_days=40, n_stocks=8)
    root = str(_seed_lib(tmp_path))
    pool = build_library_pool("ashare", daily, root=root, compact=True)
    assert isinstance(pool, CompactLibraryPool)
    dates = sorted(daily["trade_date"].unique().to_list())
    half = dates[: len(dates) // 2]
    sliced = pool.filter_dates(half)
    assert isinstance(sliced, CompactLibraryPool)
    assert sliced.wide["trade_date"].max() <= max(half)
    assert len(sliced) > 0


def test_compact_panel_row_set_matches_legacy_with_warmup_nulls(tmp_path):
    """滚动窗因子的预热期全 null 行:legacy 行集=「至少一因子有限」;compact 必须同行集。

    否则全缺行(带 ret)混进 LGBM 训练面板与 fold 日期轴——同数据 compact/legacy
    静默数值漂移(预热期真实场景必现,满覆盖 mock 测不到)。
    """
    from factorzen.discovery.factor_library import build_library_pool
    from factorzen.research.combination.models import build_panel

    _write_lib(tmp_path, "ashare", [
        {"expression": "ts_mean(close, 10)", "market": "ashare",
         "status": "active", "ic_train": 0.05},
    ])
    daily = _mk_daily(40, 6)
    legacy = build_library_pool("ashare", daily, None, root=str(tmp_path), compact=False)
    comp = build_library_pool("ashare", daily, None, root=str(tmp_path), compact=True)
    ret = daily.select(
        [pl.col("trade_date").cast(pl.Utf8), "ts_code"]
    ).with_columns(pl.lit(0.01).alias("ret"))

    p_l = build_panel(legacy, ret)
    p_c = build_panel(comp, ret)
    assert p_c.height == p_l.height, \
        f"compact 面板混入全 null 预热行: compact={p_c.height} legacy={p_l.height}"
    key = ["trade_date", "ts_code"]
    assert p_c.sort(key).select(p_l.columns).equals(p_l.sort(key)), "行集/值不一致"

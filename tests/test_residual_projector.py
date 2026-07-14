"""ResidualProjector：per-date QR 预计算与 residualize_cross_section / compute_residual_ic 语义 parity。

覆盖：
1. golden：合成面板（≥3 库因子、≥80 日、含 NaN、含薄截面日）两路径逐值对齐
2. 秩亏：库内完全相同列 → 残差仍与 lstsq 路径一致
3. 对齐：候选含库外 ts_code / 缺日 → 掉行/掉日行为一致
4. compute_residual_ic(projector=...) 与慢路径 IC 一致
"""
from __future__ import annotations

import datetime as dt
import inspect

import numpy as np
import polars as pl

# ── 合成工具 ────────────────────────────────────────────────────────────────


def _dates(n: int = 80) -> list[dt.date]:
    days, d = [], dt.date(2021, 1, 4)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days


def _codes(n: int = 50) -> list[str]:
    return [f"{600000 + i:06d}.SH" for i in range(n)]


def _panel_long(
    M: np.ndarray, dates: list, codes: list, *, col: str = "factor_value",
) -> pl.DataFrame:
    """M: (n_dates, n_stocks) → long panel；非有限 → null。"""
    rows = []
    for i, d in enumerate(dates):
        for j, c in enumerate(codes):
            v = float(M[i, j])
            rows.append({
                "trade_date": d,
                "ts_code": c,
                col: None if not np.isfinite(v) else v,
            })
    return pl.DataFrame(rows)


def _slow_residualize_panel(factor_df: pl.DataFrame, panel) -> pl.DataFrame:
    """独立慢路径：逐日 residualize_cross_section，镜像 ResidualProjector 语义。

    用于 golden 互证——**不**调用 ResidualProjector，只复用公开单日函数。
    """
    from factorzen.discovery.residual import (
        _day_min_samples,
        residualize_cross_section,
    )

    empty_schema = {
        "trade_date": pl.Date,
        "ts_code": pl.Utf8,
        "factor_value": pl.Float64,
    }
    if factor_df is None or factor_df.is_empty():
        return pl.DataFrame(schema=empty_schema)

    cand = factor_df.with_columns(pl.col("factor_value").fill_nan(None)).filter(
        pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite()
    )
    if cand.is_empty():
        return pl.DataFrame(schema=empty_schema)

    min_n = _day_min_samples(panel.k)
    out_rows: list[dict] = []

    for date, day_df in cand.group_by("trade_date", maintain_order=True):
        d = date[0] if isinstance(date, tuple) else date
        di = panel.date_idx.get(d)
        if di is None:
            continue
        codes = day_df["ts_code"].to_list()
        y = day_df["factor_value"].to_numpy().astype(np.float64, copy=False)
        si = np.fromiter(
            (panel.stock_idx.get(c, -1) for c in codes),
            dtype=np.int64,
            count=len(codes),
        )
        valid = si >= 0
        if int(valid.sum()) < min_n:
            continue
        si_v = si[valid]
        y_v = y[valid]
        codes_v = [c for c, ok in zip(codes, valid, strict=True) if ok]
        if y_v.shape[0] < min_n:
            continue
        X_day = panel.X[di, si_v, :]
        resid = residualize_cross_section(y_v, X_day)
        for c, r in zip(codes_v, resid, strict=True):
            out_rows.append({"trade_date": d, "ts_code": c, "factor_value": float(r)})

    if not out_rows:
        return pl.DataFrame(schema=empty_schema)
    return pl.DataFrame(out_rows)


def _sort_panel(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(["trade_date", "ts_code"])


def _build_synth_fixture(*, rank_def: bool = False, seed: int = 7):
    """≥3 库因子、≥80 日、含 NaN、含薄截面日。"""
    from factorzen.discovery.residual import build_library_panel

    rng = np.random.default_rng(seed)
    dates = _dates(85)
    codes = _codes(55)
    n_d, n_s = len(dates), len(codes)

    f1 = rng.normal(0, 1, size=(n_d, n_s))
    f2 = rng.normal(0, 1, size=(n_d, n_s))
    f3 = f1.copy() if rank_def else rng.normal(0, 1, size=(n_d, n_s))

    lib_pool = {
        "lib_a": _panel_long(f1, dates, codes),
        "lib_b": _panel_long(f2, dates, codes),
        "lib_c": _panel_long(f3, dates, codes),
    }
    panel = build_library_panel(lib_pool)
    assert panel is not None and panel.k == 3

    cand_m = 0.4 * f1 + 0.3 * f2 + rng.normal(0, 0.8, size=(n_d, n_s))
    cand_m = cand_m.copy()
    cand_m[rng.random(size=(n_d, n_s)) < 0.05] = np.nan
    # 最后 5 日只保留 12 只股票（min_n = max(30, k+10)=30 → 必掉）
    cand_m[-5:, 12:] = np.nan
    # 中间 3 日只保留 25 只（仍 < 30 → 掉）
    cand_m[40:43, 25:] = np.nan

    cand = _panel_long(cand_m, dates, codes).filter(
        pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite()
    )

    noise = rng.normal(0, 0.4, size=(n_d, n_s))
    fwd_m = np.where(np.isfinite(cand_m), cand_m + noise, 0.0)
    fwd = _panel_long(fwd_m, dates, codes, col="fwd_ret_1d")

    return dates, codes, panel, cand, fwd


# ── 1. golden parity ────────────────────────────────────────────────────────


def test_residualize_parity_with_cross_section_slow_path():
    """ResidualProjector.residualize ≡ 逐日 residualize_cross_section（atol 1e-9）。"""
    from factorzen.discovery.residual import ResidualProjector

    _, _, panel, cand, _ = _build_synth_fixture()
    proj = ResidualProjector(panel)
    fast = _sort_panel(proj.residualize(cand))
    slow = _sort_panel(_slow_residualize_panel(cand, panel))

    assert set(fast.columns) == {"trade_date", "ts_code", "factor_value"}
    assert fast.shape == slow.shape, f"行数不一致 fast={fast.shape} slow={slow.shape}"
    assert fast.height > 0, "golden 应保留有效日"

    joined = fast.join(slow, on=["trade_date", "ts_code"], how="inner", suffix="_slow")
    assert joined.height == fast.height == slow.height
    a = joined["factor_value"].to_numpy()
    b = joined["factor_value_slow"].to_numpy()
    assert np.allclose(a, b, atol=1e-9, equal_nan=True), (
        f"残差不对齐 max|Δ|={np.nanmax(np.abs(a - b))}"
    )


def test_compute_residual_ic_projector_matches_slow():
    """compute_residual_ic 快/慢路径 IC 与 n_days 一致。"""
    from factorzen.discovery.residual import ResidualProjector, compute_residual_ic

    _, _, panel, cand, fwd = _build_synth_fixture()
    proj = ResidualProjector(panel)
    slow = compute_residual_ic(cand, panel, fwd)
    fast = compute_residual_ic(cand, panel, fwd, projector=proj)

    assert slow.n_days == fast.n_days
    assert slow.n_days > 0
    assert abs(slow.ic_mean - fast.ic_mean) < 1e-12


# ── 2. 秩亏 parity ──────────────────────────────────────────────────────────


def test_rank_deficient_library_parity():
    """库内两列完全相同 → QR 路径与 lstsq 残差仍逐值一致。"""
    from factorzen.discovery.residual import ResidualProjector, compute_residual_ic

    _, _, panel, cand, fwd = _build_synth_fixture(rank_def=True, seed=99)
    X0 = panel.X[0]
    corr = float(np.corrcoef(X0[:, 0], X0[:, 2])[0, 1])
    assert abs(corr) > 0.999, f"秩亏夹具失效 corr={corr}"

    proj = ResidualProjector(panel)
    fast = _sort_panel(proj.residualize(cand))
    slow = _sort_panel(_slow_residualize_panel(cand, panel))
    assert fast.height == slow.height and fast.height > 0
    joined = fast.join(slow, on=["trade_date", "ts_code"], suffix="_slow")
    a = joined["factor_value"].to_numpy()
    b = joined["factor_value_slow"].to_numpy()
    assert np.allclose(a, b, atol=1e-9, equal_nan=True), (
        f"秩亏残差不对齐 max|Δ|={np.nanmax(np.abs(a - b))}"
    )

    r_slow = compute_residual_ic(cand, panel, fwd)
    r_fast = compute_residual_ic(cand, panel, fwd, projector=proj)
    assert r_slow.n_days == r_fast.n_days
    if r_slow.n_days > 0:
        assert abs(r_slow.ic_mean - r_fast.ic_mean) < 1e-12


# ── 3. 对齐行为 ─────────────────────────────────────────────────────────────


def test_unknown_ts_code_and_missing_dates_alignment():
    """库外 ts_code 丢弃；库无缺日丢弃——快慢路径一致。"""
    from factorzen.discovery.residual import ResidualProjector, build_library_panel

    rng = np.random.default_rng(3)
    dates = _dates(40)
    codes = _codes(45)
    n_d, n_s = len(dates), len(codes)
    lib_pool = {
        f"f{j}": _panel_long(rng.normal(0, 1, size=(n_d, n_s)), dates, codes)
        for j in range(3)
    }
    panel = build_library_panel(lib_pool)
    assert panel is not None

    base = rng.normal(0, 1, size=(n_d, n_s))
    cand = _panel_long(base, dates, codes)
    ghost_date = dt.date(2099, 1, 6)  # 工作日但不在库
    # 库外股票（同日）+ 库内股票但库无日；避免与 cand 已有 (date, code) 重复
    extra = pl.DataFrame({
        "trade_date": [
            dates[0], dates[0],
            ghost_date, ghost_date, ghost_date, ghost_date, ghost_date,
        ],
        "ts_code": [
            "999001.SH", "999002.SH",
            codes[0], codes[1], codes[2], codes[3], codes[4],
        ],
        "factor_value": [1.0, 2.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    })
    cand = pl.concat([cand, extra], how="vertical_relaxed")

    proj = ResidualProjector(panel)
    fast = _sort_panel(proj.residualize(cand))
    slow = _sort_panel(_slow_residualize_panel(cand, panel))

    assert fast.height == slow.height
    assert fast.filter(pl.col("ts_code").str.starts_with("999")).height == 0
    assert fast.filter(pl.col("trade_date") == ghost_date).height == 0
    joined = fast.join(slow, on=["trade_date", "ts_code"], suffix="_slow")
    assert joined.height == fast.height
    assert np.allclose(
        joined["factor_value"].to_numpy(),
        joined["factor_value_slow"].to_numpy(),
        atol=1e-9,
    )


def test_projector_none_keeps_compute_residual_ic_signature():
    """不传 projector 时行为与旧签名兼容。"""
    from factorzen.discovery.residual import compute_residual_ic

    sig = inspect.signature(compute_residual_ic)
    assert "projector" in sig.parameters
    assert sig.parameters["projector"].default is None

    _, _, panel, cand, fwd = _build_synth_fixture(seed=1)
    res = compute_residual_ic(cand, panel, fwd)
    assert res.n_days >= 0


def test_day_guard_drops_thin_days_both_paths():
    """薄截面日两路径均掉日。"""
    from factorzen.discovery.residual import ResidualProjector, _day_min_samples

    _, _, panel, cand, _ = _build_synth_fixture()
    min_n = _day_min_samples(panel.k)
    proj = ResidualProjector(panel)
    out = proj.residualize(cand)
    if out.height > 0:
        counts = out.group_by("trade_date").len()
        assert int(counts["len"].min()) >= min_n
    slow = _slow_residualize_panel(cand, panel)
    assert set(out["trade_date"].unique().to_list()) == set(
        slow["trade_date"].unique().to_list()
    )

"""护栏 corr panel 内存瘦身：present=None / f32 存储 / 分块归约 / 刀 1 projector 复用。

数值红线：f64 下与逐对 ``max_correlation_detail(..., panel=None)`` 逐位等价；
f32 存储下 allclose(atol=1e-6)。不改 ``tests/test_library_corr_panel_equiv.py`` 断言。
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from tests.test_library_corr_panel_equiv import _assert_same, _dates, _panel_df

# ── helpers ──────────────────────────────────────────────────────────────────


def _mk_pool_and_cand(
    *,
    n_days: int = 40,
    n_stocks: int = 40,
    seed: int = 0,
    null_frac: float = 0.05,
    single_day: bool = False,
    degenerate_lib: bool = False,
):
    """小池 dict + 候选；可选缺行 / 单日 / 退化截面。"""
    rng = np.random.default_rng(seed)
    if single_day:
        n_days = 1
    days = _dates(n_days)
    stocks = [f"{i:06d}.SH" for i in range(n_stocks)]
    base = rng.standard_normal((len(days), len(stocks)))
    null_c = rng.random(base.shape) < null_frac
    null_a = rng.random(base.shape) < null_frac
    null_b = rng.random(base.shape) < (null_frac + 0.02)
    cand = _panel_df(days, stocks, base, null_mask=null_c)
    lib_a_v = base + 0.05 * rng.standard_normal(base.shape)
    if degenerate_lib:
        lib_b_v = np.ones_like(base)
    else:
        lib_b_v = rng.standard_normal(base.shape)
    pool = {
        "lib_a": _panel_df(days, stocks, lib_a_v, null_mask=null_a),
        "lib_b": _panel_df(days, stocks, lib_b_v, null_mask=null_b),
    }
    return cand, pool, days, stocks


def _to_compact(pool: dict[str, pl.DataFrame]):
    """dict 长表 → CompactLibraryPool（键并集 outer + 值列）。"""
    from factorzen.research.combination.pool import CompactLibraryPool

    names = tuple(pool.keys())
    wide = None
    for name, df in pool.items():
        col = "factor_value" if "factor_value" in df.columns else "factor_clean"
        piece = df.select(
            ["trade_date", "ts_code", pl.col(col).alias(name)]
        )
        if wide is None:
            wide = piece
        else:
            wide = wide.join(piece, on=["trade_date", "ts_code"], how="full", coalesce=True)
    assert wide is not None
    return CompactLibraryPool(wide, names)


# ── 1. 等价 ground truth ─────────────────────────────────────────────────────


def test_dict_pool_panel_matches_pairwise():
    """legacy dict 分支：present=None 面板 vs 逐对 f64 精确一致。"""
    from factorzen.discovery.scoring import build_library_corr_panel, max_correlation_detail

    cand, pool, _, _ = _mk_pool_and_cand(seed=1, null_frac=0.08)
    pairwise = max_correlation_detail(cand, pool)
    panel = build_library_corr_panel(pool)
    assert panel is not None
    assert panel.present is None
    matrixed = max_correlation_detail(cand, pool, panel=panel)
    _assert_same(pairwise, matrixed)


def test_compact_pool_panel_matches_pairwise():
    """CompactLibraryPool 分支：present=None 面板 vs 逐对 f64 精确一致。"""
    from factorzen.discovery.scoring import build_library_corr_panel, max_correlation_detail

    cand, pool, _, _ = _mk_pool_and_cand(seed=2, null_frac=0.06)
    compact = _to_compact(pool)
    pairwise = max_correlation_detail(cand, pool)
    panel = build_library_corr_panel(compact)
    assert panel is not None
    assert panel.present is None
    matrixed = max_correlation_detail(cand, compact, panel=panel)
    _assert_same(pairwise, matrixed)


def test_single_day_and_degenerate_still_equiv():
    """单日截面 + 库因子截面常数(std=0)：panel 与逐对一致。"""
    from factorzen.discovery.scoring import build_library_corr_panel, max_correlation_detail

    cand, pool, _, _ = _mk_pool_and_cand(
        n_days=1, n_stocks=40, seed=3, single_day=True, degenerate_lib=True,
    )
    # 单日但 n_stocks>=30 才有有效相关；退化因子得 0，高相关 lib_a 应胜出
    pairwise = max_correlation_detail(cand, pool)
    panel = build_library_corr_panel(pool)
    assert panel is not None and panel.present is None
    _assert_same(pairwise, max_correlation_detail(cand, pool, panel=panel))

    compact = _to_compact(pool)
    panel_c = build_library_corr_panel(compact)
    assert panel_c is not None and panel_c.present is None
    _assert_same(pairwise, max_correlation_detail(cand, compact, panel=panel_c))


def test_missing_lib_rows_independent_pairs():
    """库帧缺行（覆盖不齐）不因一次对齐互相污染。"""
    from factorzen.discovery.scoring import build_library_corr_panel, max_correlation_detail

    n_days, n_stocks = 35, 50
    days = _dates(n_days)
    stocks = [f"{i:06d}.SH" for i in range(n_stocks)]
    rng = np.random.default_rng(11)
    base = rng.standard_normal((n_days, n_stocks))
    null_a = np.zeros((n_days, n_stocks), dtype=bool)
    null_a[:, 30:] = True
    null_b = np.zeros((n_days, n_stocks), dtype=bool)
    null_b[:, :15] = True
    cand = _panel_df(days, stocks, base)
    pool = {
        "a": _panel_df(days, stocks, base + 0.02 * rng.standard_normal(base.shape), null_mask=null_a),
        "b": _panel_df(days, stocks, rng.standard_normal(base.shape), null_mask=null_b),
    }
    pairwise = max_correlation_detail(cand, pool)
    for p in (pool, _to_compact(pool)):
        panel = build_library_corr_panel(p)
        assert panel is not None and panel.present is None
        _assert_same(pairwise, max_correlation_detail(cand, p, panel=panel))


# ── 2. f32 模式 ──────────────────────────────────────────────────────────────


def test_f32_mode_dtype_and_allclose(monkeypatch, capsys):
    """阈值压到 1 → values float32、present is None、与逐对 allclose + nearest 一致。"""
    from factorzen.discovery import scoring as scoring_mod
    from factorzen.discovery.scoring import build_library_corr_panel, max_correlation_detail

    monkeypatch.setattr(scoring_mod, "CORR_PANEL_F32_BYTES_THRESHOLD", 1)
    cand, pool, _, _ = _mk_pool_and_cand(seed=4, n_days=45, n_stocks=40)
    panel = build_library_corr_panel(pool)
    assert panel is not None
    assert panel.values.dtype == np.float32
    assert panel.present is None
    out = capsys.readouterr().out
    assert "f32" in out and "corr-panel" in out

    pairwise = max_correlation_detail(cand, pool)
    matrixed = max_correlation_detail(cand, pool, panel=panel)
    assert matrixed[1] == pairwise[1]
    assert matrixed[0] == pytest.approx(pairwise[0], abs=1e-6)

    # compact 同样触发
    panel_c = build_library_corr_panel(_to_compact(pool))
    assert panel_c is not None
    assert panel_c.values.dtype == np.float32
    assert panel_c.present is None
    mc_c, n_c = max_correlation_detail(cand, pool, panel=panel_c)
    assert n_c == pairwise[1]
    assert mc_c == pytest.approx(pairwise[0], abs=1e-6)


# ── 3. 分块边界 ──────────────────────────────────────────────────────────────


def test_chunk_blk1_and_single_block_bit_identical(monkeypatch):
    """blk=1（逐日）与 blk>n_d（单块）与默认分块 f64 逐位一致。"""
    from factorzen.discovery import scoring as scoring_mod
    from factorzen.discovery.scoring import build_library_corr_panel, max_correlation_detail

    cand, pool, _, _ = _mk_pool_and_cand(seed=5, n_days=50, n_stocks=40)
    panel = build_library_corr_panel(pool)
    assert panel is not None
    assert panel.values.dtype == np.float64
    default = max_correlation_detail(cand, pool, panel=panel)

    # 极小预算 → blk=1
    monkeypatch.setattr(scoring_mod, "CORR_PANEL_CHUNK_BYTES", 1)
    blk1 = max_correlation_detail(cand, pool, panel=panel)
    assert blk1 == default
    assert blk1[0] == default[0]  # 逐位
    assert blk1[1] == default[1]

    # 极大预算 → 单块覆盖全部日期
    monkeypatch.setattr(scoring_mod, "CORR_PANEL_CHUNK_BYTES", 10**18)
    one = max_correlation_detail(cand, pool, panel=panel)
    assert one == default
    assert one[0] == default[0]
    assert one[1] == default[1]


# ── 4. NaN 候选毒化 ──────────────────────────────────────────────────────────


def test_candidate_nan_poisons_day_panel_and_pairwise():
    """候选某日含 float NaN（非 null）→ panel 与逐对均排除该日、终值一致。"""
    from factorzen.discovery.scoring import build_library_corr_panel, max_correlation_detail

    n_stocks = 40
    days = _dates(3)
    stocks = [f"{i:06d}.SH" for i in range(n_stocks)]
    rng = np.random.default_rng(99)
    v = rng.standard_normal((3, n_stocks))
    # day1 仅 10 只 → 不够 30；有效日只剩 day0/day2
    null_d1 = np.zeros((3, n_stocks), dtype=bool)
    null_d1[1, 10:] = True

    cand_vals = v.copy()
    cand_vals[0, 0] = np.nan  # day0 毒化 → 只剩 day2
    cand = _panel_df(days, stocks, cand_vals, null_mask=null_d1)
    lib = _panel_df(days, stocks, v + 0.01, null_mask=null_d1)
    pool = {"lib": lib}

    # 对照：无毒化候选应有高相关
    cand_clean = _panel_df(days, stocks, v, null_mask=null_d1)
    assert max_correlation_detail(cand_clean, pool)[0] > 0.9

    pairwise = max_correlation_detail(cand, pool)
    panel = build_library_corr_panel(pool)
    assert panel is not None and panel.present is None
    matrixed = max_correlation_detail(cand, pool, panel=panel)
    _assert_same(pairwise, matrixed)
    # 毒化 day0 后若 day2 仍存活应有相关；与 clean 路径不同即可证明毒化生效
    # （day2 有 40 只，高相关；day0 被排除）
    assert matrixed[0] > 0.5
    # 若全日毒化：再毒 day2
    cand_vals2 = cand_vals.copy()
    cand_vals2[2, 5] = np.nan
    cand2 = _panel_df(days, stocks, cand_vals2, null_mask=null_d1)
    pw2 = max_correlation_detail(cand2, pool)
    mx2 = max_correlation_detail(cand2, pool, panel=panel)
    _assert_same(pw2, mx2)
    assert pw2[0] == 0.0


# ── 5. present_block ─────────────────────────────────────────────────────────


def test_present_block_none_matches_explicit_bool():
    """present=None 时 present_block 与显式 bool 面板推导一致。"""
    from factorzen.discovery.scoring import LibraryCorrPanel, build_library_corr_panel

    _, pool, _, _ = _mk_pool_and_cand(seed=6, n_days=20, n_stocks=30, null_frac=0.1)
    panel = build_library_corr_panel(pool)
    assert panel is not None
    assert panel.present is None

    derived = panel.present_block(0, panel.values.shape[0])
    expected = ~np.isnan(panel.values)
    np.testing.assert_array_equal(derived, expected)

    # 切片
    d0, d1 = 3, 12
    np.testing.assert_array_equal(
        panel.present_block(d0, d1),
        ~np.isnan(panel.values[d0:d1]),
    )

    # 显式 bool 面板：present_block 走切片分支
    explicit = LibraryCorrPanel(
        names=panel.names,
        dates=panel.dates,
        stocks=panel.stocks,
        date_idx=panel.date_idx,
        stock_idx=panel.stock_idx,
        values=panel.values,
        present=expected,
    )
    np.testing.assert_array_equal(
        explicit.present_block(d0, d1),
        expected[d0:d1],
    )
    # 与 None 推导逐格一致
    np.testing.assert_array_equal(
        panel.present_block(d0, d1),
        explicit.present_block(d0, d1),
    )


# ── 6. 刀 1：residual_projector 复用 lib_panel，不重调 build_library_panel ───


def test_node_guardrails_reuses_projector_panel(monkeypatch):
    """residual_projector 非 None → build_library_panel 不被调用；残差结果与现建一致。"""
    import datetime as dt

    from factorzen.agents.nodes import node_guardrails
    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.discovery.residual import (
        ResidualProjector,
        build_library_panel,
    )
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import HoldoutICResult
    from factorzen.validation.multiple_testing import TrialLedger

    rng = np.random.default_rng(7)
    days: list[dt.date] = []
    d = dt.date(2022, 1, 3)
    while len(days) < 60:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(35)]
    rows = []
    for c in codes:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.015
            rows.append({
                "trade_date": dd, "ts_code": c,
                "close": px, "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                "close_adj": px, "open_adj": px * 0.99,
                "high_adj": px * 1.01, "low_adj": px * 0.98,
                "pre_close": px, "vol": 1e6, "amount": 1e7,
            })
    daily = pl.DataFrame(rows)
    bundle = DataBundle.build(daily)

    # 库池：两只与 close 相关的合成因子（足够让 residual 路径走通）
    lib_rows_a, lib_rows_b = [], []
    for c in codes:
        for dd in days:
            lib_rows_a.append({
                "trade_date": dd, "ts_code": c,
                "factor_value": float(rng.standard_normal()),
            })
            lib_rows_b.append({
                "trade_date": dd, "ts_code": c,
                "factor_value": float(rng.standard_normal()),
            })
    lib_pool = {
        "lib_a": pl.DataFrame(lib_rows_a),
        "lib_b": pl.DataFrame(lib_rows_b),
    }
    lib_panel = build_library_panel(lib_pool)
    assert lib_panel is not None and lib_panel.k > 0
    projector = ResidualProjector.from_panel(lib_panel)

    # holdout / 去相关 / 相关 放行，聚焦 residual projector 路径
    monkeypatch.setattr(
        "factorzen.validation.holdout.holdout_ic_result",
        lambda *a, **k: HoldoutICResult(0.05, 0.5, (0.01, 0.09), n_days=80),
    )
    monkeypatch.setattr(
        "factorzen.discovery.scoring.library_orthogonal_check",
        lambda *a, **k: (True, 0.1, None),
    )
    monkeypatch.setattr(
        "factorzen.discovery.scoring.max_correlation",
        lambda *a, **k: 0.0,
    )
    monkeypatch.setattr(
        "factorzen.discovery.guardrails.pool_pbo",
        lambda *a, **k: 0.1,
    )

    calls = {"n": 0}
    real_build = build_library_panel

    def counting_build(pool):
        calls["n"] += 1
        return real_build(pool)

    # nodes 函数内 from-import residual.build_library_panel；patch 源模块即可
    monkeypatch.setattr(
        "factorzen.discovery.residual.build_library_panel",
        counting_build,
    )

    def _run(residual_projector=None):
        state = AgentState(seed=1)
        state.attempts.append(AttemptRecord(
            iteration=0, hypothesis="h", expression="rank(close)",
            compile_ok=True, ic_train=0.06, passed_guardrails=False,
            critic_verdict=None, error=None, ir_train=0.5, turnover=0.3, n_train=50,
        ))
        node_guardrails(
            state, daily=daily, holdout_df=daily, bundle=bundle,
            ledger=TrialLedger(), top_k=3, lib_pool=lib_pool,
            objective="residual", residual_projector=residual_projector,
        )
        return state

    # 有 projector：不应调用 build_library_panel
    calls["n"] = 0
    state_proj = _run(residual_projector=projector)
    assert calls["n"] == 0, (
        f"注入 residual_projector 后仍调用了 build_library_panel {calls['n']} 次"
    )

    # 无 projector：应调用至少一次
    calls["n"] = 0
    state_fresh = _run(residual_projector=None)
    assert calls["n"] >= 1, "未注入 projector 时应现场 build_library_panel"

    # 残差 train IC 一致（同一 panel / 同一候选）
    a_proj = next(a for a in state_proj.attempts if a.expression == "rank(close)")
    a_fresh = next(a for a in state_fresh.attempts if a.expression == "rank(close)")
    assert a_proj.residual_ic_train is not None
    assert a_fresh.residual_ic_train is not None
    assert a_proj.residual_ic_train == pytest.approx(a_fresh.residual_ic_train, abs=1e-12)

    # 候选落盘 residual 字段（若入选）也应一致
    if state_proj.candidates and state_fresh.candidates:
        c0 = state_proj.candidates[0]
        c1 = state_fresh.candidates[0]
        if "residual_ic_train" in c0 and "residual_ic_train" in c1:
            assert c0["residual_ic_train"] == pytest.approx(
                c1["residual_ic_train"], abs=1e-12,
            )

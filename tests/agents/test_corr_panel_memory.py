"""护栏 corr panel 内存瘦身：present=None / f32 存储 / 分块归约 / 刀 1 projector 复用。

数值红线：f64 下与逐对 ``max_correlation_detail(..., panel=None)`` 逐位等价；
f32 存储下 allclose(atol=1e-6)。不改 ``tests/test_library_corr_panel_equiv.py`` 断言。
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from tests.discovery.test_library_residual import (
    _assert_same,
    _panel_df,
)
from tests.discovery.test_library_residual import (
    _dates__library_corr_panel_equiv as _dates,
)

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


def test_pool_panel_ground_truth_suite():
    """legacy dict 分支：present=None 面板 vs 逐对 f64 精确一致。；CompactLibraryPool 分支：present=None 面板 vs 逐对 f64 精确一致。；单日截面 + 库因子截面常数(std=0)：panel 与逐对一致。；库帧缺行（覆盖不齐）不因一次对齐互相污染。"""
    # -- 原 test_dict_pool_panel_matches_pairwise --
    def _section_0_test_dict_pool_panel_matches_pairwise():
        from factorzen.discovery.scoring import build_library_corr_panel, max_correlation_detail

        cand, pool, _, _ = _mk_pool_and_cand(seed=1, null_frac=0.08)
        pairwise = max_correlation_detail(cand, pool)
        panel = build_library_corr_panel(pool)
        assert panel is not None
        assert panel.present is None
        matrixed = max_correlation_detail(cand, pool, panel=panel)
        _assert_same(pairwise, matrixed)

    _section_0_test_dict_pool_panel_matches_pairwise()

    # -- 原 test_compact_pool_panel_matches_pairwise --
    def _section_1_test_compact_pool_panel_matches_pairwise():
        from factorzen.discovery.scoring import build_library_corr_panel, max_correlation_detail

        cand, pool, _, _ = _mk_pool_and_cand(seed=2, null_frac=0.06)
        compact = _to_compact(pool)
        pairwise = max_correlation_detail(cand, pool)
        panel = build_library_corr_panel(compact)
        assert panel is not None
        assert panel.present is None
        matrixed = max_correlation_detail(cand, compact, panel=panel)
        _assert_same(pairwise, matrixed)

    _section_1_test_compact_pool_panel_matches_pairwise()

    # -- 原 test_single_day_and_degenerate_still_equiv --
    def _section_2_test_single_day_and_degenerate_still_equiv():
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

    _section_2_test_single_day_and_degenerate_still_equiv()

    # -- 原 test_missing_lib_rows_independent_pairs --
    def _section_3_test_missing_lib_rows_independent_pairs():
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

    _section_3_test_missing_lib_rows_independent_pairs()


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


# ── 7. lazy-wide 模式（超阈值免物化） ─────────────────────────────────────────


def _assert_block_equal(a: tuple, b: tuple) -> None:
    """block() 返回的 (vals, pres) 逐位等价（含 NaN 位置）。"""
    va, pa = a
    vb, pb = b
    np.testing.assert_array_equal(pa, pb)
    # NaN 位置一致 + 有限位逐位相等
    nan_a, nan_b = np.isnan(va), np.isnan(vb)
    np.testing.assert_array_equal(nan_a, nan_b)
    np.testing.assert_array_equal(va[~nan_a], vb[~nan_b])


def test_lazy_wide_bit_identical_suite(monkeypatch, capsys):
    """默认阈值下小池仍返回 LibraryCorrPanel(f64 物化)。；lazy vs 物化 vs 逐对：f64 逐位一致；覆盖 null/单日/退化/NaN 毒化。；lazy.block 与物化.block 在多切窗（含 0 起/尾/单日）vals/pres 逐位等价。；lazy 下 CORR_PANEL_CHUNK_BYTES=1(blk=1) 与极大(单块)与默认逐位一致。；f32 wide + lazy 与 f32 物化 LibraryCorrPanel 结果逐位一致（同源升 f64）。；结构守卫：_wide is pool.wide；block 后无缓存大数组属性；打印 lazy 提示。"""
    # -- 原 test_lazy_wide_default_threshold_still_materializes --
    def _section_0_test_lazy_wide_default_threshold_still_materializes():
        from factorzen.discovery.scoring import LibraryCorrPanel, build_library_corr_panel

        _, pool, _, _ = _mk_pool_and_cand(seed=33, n_days=30, n_stocks=30)
        compact = _to_compact(pool)
        panel = build_library_corr_panel(compact)
        assert isinstance(panel, LibraryCorrPanel)
        assert panel.values.dtype == np.float64
        assert panel.present is None

    _section_0_test_lazy_wide_default_threshold_still_materializes()

    # -- 原 test_lazy_wide_tripartite_bit_identical --
    def _section_1_test_lazy_wide_tripartite_bit_identical(mp):
        from factorzen.discovery import scoring as scoring_mod
        from factorzen.discovery.scoring import (
            LazyWideCorrGrid,
            LibraryCorrPanel,
            build_library_corr_panel,
            max_correlation_detail,
        )

        cases = [
            dict(seed=20, null_frac=0.08),
            dict(seed=21, n_days=1, n_stocks=40, single_day=True, degenerate_lib=True),
            dict(seed=22, null_frac=0.12),
        ]
        for kw in cases:
            cand, pool, _, _ = _mk_pool_and_cand(**kw)
            compact = _to_compact(pool)

            # 物化（默认阈值）
            mat = build_library_corr_panel(compact)
            assert isinstance(mat, LibraryCorrPanel)
            assert mat.values.dtype == np.float64

            # lazy（阈值压到 1）
            mp.setattr(scoring_mod, "CORR_PANEL_LAZY_BYTES_THRESHOLD", 1)
            lazy = build_library_corr_panel(compact)
            assert isinstance(lazy, LazyWideCorrGrid)
            assert lazy.present is None
            assert not hasattr(lazy, "values") or getattr(lazy, "values", None) is None

            pairwise = max_correlation_detail(cand, pool)
            out_mat = max_correlation_detail(cand, compact, panel=mat)
            out_lazy = max_correlation_detail(cand, compact, panel=lazy)
            # lazy ↔ 物化：同一散射语义，f64 逐位一致（==）
            assert out_lazy == out_mat
            assert out_lazy[0] == out_mat[0]
            assert out_lazy[1] == out_mat[1]
            # 与逐对 ground truth：沿用既有 panel 红线（_assert_same / atol=1e-12）
            # （矩阵路径 vs compute_factor_correlation 求和顺序可差 ULP，既有测试同）
            _assert_same(pairwise, out_lazy)
            _assert_same(pairwise, out_mat)
            mp.setattr(
                scoring_mod, "CORR_PANEL_LAZY_BYTES_THRESHOLD",
                scoring_mod.CORR_PANEL_F32_BYTES_THRESHOLD,
            )

        # NaN 候选毒化：与既有 fixture 语义一致
        n_stocks = 40
        days = _dates(3)
        stocks = [f"{i:06d}.SH" for i in range(n_stocks)]
        rng = np.random.default_rng(99)
        v = rng.standard_normal((3, n_stocks))
        null_d1 = np.zeros((3, n_stocks), dtype=bool)
        null_d1[1, 10:] = True
        cand_vals = v.copy()
        cand_vals[0, 0] = np.nan
        cand = _panel_df(days, stocks, cand_vals, null_mask=null_d1)
        lib = _panel_df(days, stocks, v + 0.01, null_mask=null_d1)
        pool = {"lib": lib}
        compact = _to_compact(pool)
        mat = build_library_corr_panel(compact)
        mp.setattr(scoring_mod, "CORR_PANEL_LAZY_BYTES_THRESHOLD", 1)
        lazy = build_library_corr_panel(compact)
        assert isinstance(lazy, LazyWideCorrGrid)
        pairwise = max_correlation_detail(cand, pool)
        out_lazy = max_correlation_detail(cand, compact, panel=lazy)
        out_mat = max_correlation_detail(cand, compact, panel=mat)
        assert out_lazy == out_mat
        _assert_same(pairwise, out_lazy)
        _assert_same(pairwise, out_mat)

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_lazy_wide_tripartite_bit_identical(mp)

    # -- 原 test_lazy_wide_block_equiv_windows --
    def _section_2_test_lazy_wide_block_equiv_windows(mp):
        from factorzen.discovery import scoring as scoring_mod
        from factorzen.discovery.scoring import (
            LazyWideCorrGrid,
            LibraryCorrPanel,
            build_library_corr_panel,
        )

        _, pool, _, _ = _mk_pool_and_cand(seed=30, n_days=25, n_stocks=35, null_frac=0.1)
        compact = _to_compact(pool)
        mat = build_library_corr_panel(compact)
        assert isinstance(mat, LibraryCorrPanel)
        mp.setattr(scoring_mod, "CORR_PANEL_LAZY_BYTES_THRESHOLD", 1)
        lazy = build_library_corr_panel(compact)
        assert isinstance(lazy, LazyWideCorrGrid)

        n_d = len(mat.dates)
        windows = [(0, n_d), (0, 1), (n_d - 1, n_d), (3, 12), (5, 6), (0, 5), (10, n_d)]
        for d0, d1 in windows:
            _assert_block_equal(lazy.block(d0, d1), mat.block(d0, d1))
            # present_block 兼容路径
            np.testing.assert_array_equal(
                lazy.present_block(d0, d1), mat.present_block(d0, d1),
            )

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_lazy_wide_block_equiv_windows(mp)

    # -- 原 test_lazy_wide_chunk_boundaries_bit_identical --
    def _section_3_test_lazy_wide_chunk_boundaries_bit_identical(mp):
        from factorzen.discovery import scoring as scoring_mod
        from factorzen.discovery.scoring import (
            LazyWideCorrGrid,
            build_library_corr_panel,
            max_correlation_detail,
        )

        cand, pool, _, _ = _mk_pool_and_cand(seed=31, n_days=50, n_stocks=40)
        compact = _to_compact(pool)
        mp.setattr(scoring_mod, "CORR_PANEL_LAZY_BYTES_THRESHOLD", 1)
        lazy = build_library_corr_panel(compact)
        assert isinstance(lazy, LazyWideCorrGrid)

        default = max_correlation_detail(cand, compact, panel=lazy)

        mp.setattr(scoring_mod, "CORR_PANEL_CHUNK_BYTES", 1)
        blk1 = max_correlation_detail(cand, compact, panel=lazy)
        assert blk1 == default
        assert blk1[0] == default[0]
        assert blk1[1] == default[1]

        mp.setattr(scoring_mod, "CORR_PANEL_CHUNK_BYTES", 10**18)
        one = max_correlation_detail(cand, compact, panel=lazy)
        assert one == default
        assert one[0] == default[0]
        assert one[1] == default[1]

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_lazy_wide_chunk_boundaries_bit_identical(mp)

    # -- 原 test_lazy_wide_f32_source_bit_identical_vs_f32_panel --
    def _section_4_test_lazy_wide_f32_source_bit_identical_vs_f32_panel(mp):
        from factorzen.discovery import scoring as scoring_mod
        from factorzen.discovery.scoring import (
            LazyWideCorrGrid,
            LibraryCorrPanel,
            build_library_corr_panel,
            max_correlation_detail,
        )
        from factorzen.research.combination.pool import CompactLibraryPool

        cand, pool, _, _ = _mk_pool_and_cand(seed=32, n_days=40, n_stocks=40, null_frac=0.07)
        compact = _to_compact(pool)
        # 值列压 f32（模拟 POOL_VALUE_F32 大池存储）
        f32_exprs = [
            pl.col(n).cast(pl.Float32) for n in compact.factor_names
        ]
        wide_f32 = compact.wide.with_columns(f32_exprs)
        pool_f32 = CompactLibraryPool(wide_f32, compact.factor_names)
        assert all(pool_f32.wide[n].dtype == pl.Float32 for n in pool_f32.factor_names)

        # f32 物化面板：lazy 阈值抬高，f32 阈值压到 1
        mp.setattr(scoring_mod, "CORR_PANEL_LAZY_BYTES_THRESHOLD", 10**18)
        mp.setattr(scoring_mod, "CORR_PANEL_F32_BYTES_THRESHOLD", 1)
        mat_f32 = build_library_corr_panel(pool_f32)
        assert isinstance(mat_f32, LibraryCorrPanel)
        assert mat_f32.values.dtype == np.float32
        assert mat_f32.present is None

        # lazy：阈值压到 1
        mp.setattr(scoring_mod, "CORR_PANEL_LAZY_BYTES_THRESHOLD", 1)
        lazy = build_library_corr_panel(pool_f32)
        assert isinstance(lazy, LazyWideCorrGrid)

        out_mat = max_correlation_detail(cand, pool_f32, panel=mat_f32)
        out_lazy = max_correlation_detail(cand, pool_f32, panel=lazy)
        assert out_lazy == out_mat
        assert out_lazy[0] == out_mat[0]
        assert out_lazy[1] == out_mat[1]

        # block 也逐位一致
        n_d = len(mat_f32.dates)
        _assert_block_equal(lazy.block(0, n_d), mat_f32.block(0, n_d))

    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_lazy_wide_f32_source_bit_identical_vs_f32_panel(mp)

    # -- 原 test_lazy_wide_holds_wide_ref_no_big_cache --
    def _section_5_test_lazy_wide_holds_wide_ref_no_big_cache(mp, capsys):
        from factorzen.discovery import scoring as scoring_mod
        from factorzen.discovery.scoring import LazyWideCorrGrid, build_library_corr_panel

        _, pool, _, _ = _mk_pool_and_cand(seed=34, n_days=20, n_stocks=30)
        compact = _to_compact(pool)
        mp.setattr(scoring_mod, "CORR_PANEL_LAZY_BYTES_THRESHOLD", 1)
        lazy = build_library_corr_panel(compact)
        assert isinstance(lazy, LazyWideCorrGrid)
        assert lazy._wide is compact.wide  # 引用非复制

        out = capsys.readouterr().out
        assert "lazy-wide" in out and "corr-panel" in out and "免物化" in out

        n_d = len(lazy.dates)
        n_s = len(lazy.stocks)
        n_f = len(lazy.names)
        # 块预算：临时块上限量级（允许 2× 余量）
        blk = max(1, min(n_d, 3))
        budget = n_s * n_f * blk * 8 * 2

        vals, pres = lazy.block(0, blk)
        # 返回临时块在预算附近（非整网格）
        full_grid = n_d * n_s * n_f * 8
        assert vals.nbytes <= budget * 2
        assert vals.nbytes < full_grid or n_d <= blk  # 非整网格常驻
        # block 后不新增大数组属性（slots 固定；无 values 缓存）
        assert set(LazyWideCorrGrid.__slots__) == {
            "_day_starts", "_di_by_day", "_row_by_day", "_si_by_day", "_wide",
            "date_idx", "dates", "names", "present", "stock_idx", "stocks",
        }
        assert not hasattr(lazy, "values")
        # 行级索引 O(n_rows)，不是 (n_d,n_s,n_f) 值网格
        assert lazy._row_by_day.ndim == 1 and lazy._si_by_day.ndim == 1
        del vals, pres

    with pytest.MonkeyPatch.context() as mp:
        _section_5_test_lazy_wide_holds_wide_ref_no_big_cache(mp, capsys)



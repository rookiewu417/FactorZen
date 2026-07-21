"""sleeve 三期：overlay 叠加口径 lift 裁决（TDD）。

契约：
- sleeve_candidate=True → overlay_v1（main_z + w·sleeve_z）；非事件日保持基线
- 合成：稀疏强事件信号 → overlay lift 显著为正；旧 residual 全截面口径不给正增量
- 非 sleeve 候选 residual 路径逐位不变
- 掩码天数 < OVERLAY_MIN_MASK_DAYS（40）→ 不评 overlay
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl


def _dates(n: int, start: date = date(2023, 1, 2)) -> list[str]:
    out: list[str] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def _panel_from(
    dates: list[str], stocks: list[str], mat: np.ndarray, *, col: str = "factor_value",
) -> pl.DataFrame:
    rows = []
    for di, d in enumerate(dates):
        for si, s in enumerate(stocks):
            rows.append({"trade_date": d, "ts_code": s, col: float(mat[di, si])})
    return pl.DataFrame(rows)


def _panel(dates: list[str], n_stocks: int, value_fn) -> pl.DataFrame:
    rows = []
    for d in dates:
        for s in range(n_stocks):
            rows.append({
                "trade_date": d,
                "ts_code": f"{s:04d}.SZ",
                "factor_value": float(value_fn(d, s)),
            })
    return pl.DataFrame(rows)


def _ret_by_stock(dates: list[str], n_stocks: int) -> pl.DataFrame:
    return _panel(dates, n_stocks, lambda d, s: float(s)).rename(
        {"factor_value": "ret"}
    )


# ── 常量 / 导入契约 ──────────────────────────────────────────────────────────


def test_overlay_constants_exported():
    from factorzen.discovery.guardrails import SLEEVE_SUBSET_MIN_DAYS
    from factorzen.discovery.lift_test import (
        DEFAULT_OVERLAY_W,
        OVERLAY_MIN_MASK_DAYS,
    )

    assert DEFAULT_OVERLAY_W == 0.25
    assert OVERLAY_MIN_MASK_DAYS == SLEEVE_SUBSET_MIN_DAYS == 40


# ── 合成：overlay 正、旧 residual 口径不给正增量 ────────────────────────────


def _synth_sparse_event_vs_dense_lib(
    *,
    n_days: int = 100,
    n_stocks: int = 100,
    n_event: int = 8,
    seed: int = 99,
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame, pl.DataFrame]:
    """库因子吃掉截面主序；稀疏事件只在中间包事件股带子集 alpha。

    形态对齐 team_1002：全截面 residual_ic ≈ 0 / 负，子集 IC 强；
    overlay 叠权应给出正增量，residual 口径不得给出显著正增量。
    """
    rng = np.random.default_rng(seed)
    dates = _dates(n_days)
    stocks = [f"{s:04d}.SZ" for s in range(n_stocks)]
    common = np.linspace(-2.0, 2.0, n_stocks)
    # 中段事件股（非端点极值，避免全截面 RankIC 被事件股主导）
    ev_idx = np.arange(10, 10 + n_event)

    active: dict[str, pl.DataFrame] = {}
    for j in range(5):
        mat = np.zeros((n_days, n_stocks))
        for di in range(n_days):
            mat[di] = common + rng.standard_normal(n_stocks) * 0.15
        active[f"lib_{j}"] = _panel_from(dates, stocks, mat)

    cand = np.zeros((n_days, n_stocks))
    ret = np.zeros((n_days, n_stocks))
    for di in range(n_days):
        a = np.linspace(-1.0, 1.0, n_event)
        cand[di, ev_idx] = a
        ret[di] = common + rng.standard_normal(n_stocks) * 0.1
        ret[di, ev_idx] += a * 0.8  # 子集 alpha：事件内强相关

    return (
        active,
        _panel_from(dates, stocks, cand),
        _panel_from(dates, stocks, ret, col="ret"),
    )


def test_overlay_lift_positive_while_residual_not_positive():
    """两口径对照：overlay lift >0；同候选 residual 口径 ≯0（修复判别力）。"""
    from factorzen.discovery.lift_test import (
        library_equal_weight_score_panel,
        run_lift_tests,
    )
    from factorzen.discovery.residual import build_library_panel

    active, cand, ret = _synth_sparse_event_vs_dense_lib()
    mats = {"sparse_event": cand}

    # ── 旧 residual 全截面口径 ─────────────────────────────────────────
    resid_rows = run_lift_tests(
        [{"expression": "sparse_event", "residual_ic_train": 0.01}],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: mats[e],
        lift_workers=1,
        block_days=10,
        threshold=-1.0,
    )
    rr = resid_rows[0]
    assert rr["error"] is None, rr
    assert rr.get("lift_metric") == "residual_ic_v1"
    assert rr.get("overlay") is not True
    # fill-0 稀释后 residual 不得给出显著正增量（team_1002 组门 −0.0008 形态）
    assert float(rr["lift"]) <= 0.005, (
        f"residual 应≈0/负, got {rr['lift']}"
    )

    # ── overlay 口径 ───────────────────────────────────────────────────
    ov_rows = run_lift_tests(
        [{
            "expression": "sparse_event",
            "sleeve_candidate": True,
            "residual_ic_train": 0.01,
            "subset_ic_train": 0.5,
        }],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: mats[e],
        lift_workers=1,
        block_days=10,
        threshold=-1.0,
    )
    orow = ov_rows[0]
    assert orow["error"] is None, orow
    assert orow.get("overlay") is True
    assert orow.get("lift_metric") == "overlay_v1"
    assert orow.get("overlay_w") == 0.25
    assert orow.get("n_mask_days") is not None and int(orow["n_mask_days"]) >= 40
    assert orow["lift"] is not None
    assert float(orow["lift"]) > 0.001, f"overlay lift 应显著为正, got {orow['lift']}"
    # 判别力：overlay 正且 residual 不给正 → 两口径结论相反
    assert float(orow["lift"]) > float(rr["lift"]) + 0.002, (
        f"overlay={orow['lift']} residual={rr['lift']}"
    )
    assert orow["lift_se"] is not None
    assert orow["baseline"] is not None
    # 子集方向权威：admission_ic 应与子集同号（此处正）
    assert float(orow["admission_ic"]) > 0.1, orow["admission_ic"]

    panel = build_library_panel(active)
    assert panel is not None
    assert library_equal_weight_score_panel(panel).height > 0


def test_non_sleeve_path_bit_identical():
    """非 sleeve 候选：有/无 sleeve_candidate=False 时 residual 行字段逐位一致。"""
    from factorzen.discovery.lift_test import run_lift_tests

    active, _sparse, ret = _synth_sparse_event_vs_dense_lib(seed=3)
    dates = ret["trade_date"].unique().sort().to_list()
    n_stocks = ret["ts_code"].n_unique()
    dense = _panel(dates, n_stocks, lambda d, s: float(s) + 0.1)

    common = dict(
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: dense,
        lift_workers=1,
        block_days=10,
        threshold=-1.0,
    )
    a = run_lift_tests(
        [{"expression": "dense_a", "residual_ic_train": 0.02}],
        **common,
    )[0]
    b = run_lift_tests(
        [{"expression": "dense_a", "residual_ic_train": 0.02, "sleeve_candidate": False}],
        **common,
    )[0]
    for k in (
        "lift", "lift_se", "n_blocks", "lift_first_half", "lift_second_half",
        "candidate_rank_ic", "admission_ic", "lift_metric", "baseline", "error",
    ):
        va, vb = a.get(k), b.get(k)
        if isinstance(va, float) and isinstance(vb, float):
            assert abs(va - vb) < 1e-12, (k, va, vb)
        else:
            assert va == vb, (k, va, vb)
    assert a.get("lift_metric") == "residual_ic_v1"
    assert a.get("overlay") is not True


def test_insufficient_mask_days_skips_overlay():
    """掩码天数 < 40 → error=insufficient_mask_days，lift=None。"""
    from factorzen.discovery.lift_test import OVERLAY_MIN_MASK_DAYS, run_lift_tests

    n_days, n_stocks = 60, 40
    dates = _dates(n_days)
    rng = np.random.default_rng(1)
    active = {
        "lib_a": _panel(dates, n_stocks, lambda d, s: float(rng.standard_normal())),
    }
    event_dates = set(dates[:20])  # 仅 20 天 < 40

    def sparse(d, s):
        if d in event_dates and s < 4:
            return float(s + 1)
        return 0.0

    cand = _panel(dates, n_stocks, sparse)
    ret = _ret_by_stock(dates, n_stocks)

    rows = run_lift_tests(
        [{"expression": "few_mask", "sleeve_candidate": True}],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        lift_workers=1,
        threshold=-1.0,
    )
    r = rows[0]
    assert r["lift"] is None
    assert r.get("error") == "insufficient_mask_days"
    assert r.get("overlay") is True
    assert int(r.get("n_mask_days") or 0) < OVERLAY_MIN_MASK_DAYS


def test_split_sleeve_dense_mixed_batch():
    """同批混合：sleeve 走 overlay，稠密走 residual；互不污染。"""
    from factorzen.discovery.lift_test import run_lift_tests

    active, sparse_cand, ret = _synth_sparse_event_vs_dense_lib(seed=5)
    dates = sparse_cand["trade_date"].unique().sort().to_list()
    n_stocks = sparse_cand["ts_code"].n_unique()
    dense = _panel(dates, n_stocks, lambda d, s: float(s))

    mats = {"sparse_ev": sparse_cand, "dense_ev": dense}
    rows = run_lift_tests(
        [
            {"expression": "sparse_ev", "sleeve_candidate": True, "residual_ic_train": 0.05},
            {"expression": "dense_ev", "residual_ic_train": 0.08},
        ],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: mats[e],
        lift_workers=1,
        block_days=10,
        threshold=-1.0,
    )
    by = {r["expression"]: r for r in rows}
    assert by["sparse_ev"].get("lift_metric") == "overlay_v1"
    assert by["sparse_ev"].get("overlay") is True
    assert by["dense_ev"].get("lift_metric") == "residual_ic_v1"
    assert by["dense_ev"].get("overlay") is not True


def test_partition_helpers():
    """稠密/sleeve 分流 helper 供 CLI 与 session 同源。"""
    from factorzen.discovery.lift_test import partition_lift_queue_by_sleeve

    dense, sleeve = partition_lift_queue_by_sleeve([
        {"expression": "a", "sleeve_candidate": True},
        {"expression": "b"},
        {"expression": "c", "sleeve_candidate": False},
        {"expression": "d", "sleeve_candidate": True},
    ])
    assert [c["expression"] for c in dense] == ["b", "c"]
    assert [c["expression"] for c in sleeve] == ["a", "d"]

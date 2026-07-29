"""lift 裁决链内存收尾：CLI lift-test 与 team session 末钩子的三处峰值。

实测死点（2026-07-29，23G 机器）：

1. ``fz factor-library lift-test``（79 条 active 基线 + 17 候选，全 A 2016-2026）
   exit=137 / anon-rss 23.6G，死在**基线池物化全部完成后**的装配/面板构建阶段；
2. mine team campaign 的 session 末 lift 钩子 exit=137 @ 22.85G——钩子在**主进程**
   （session 常驻 ~17G）重新物化基线池。

本文件的判别力来源（不写恒真断言）：

- 池：``run_group_lift`` 紧接 ``run_lift_tests``（CLI lift-test 的真实调用序）时数
  ``build_library_pool`` 的**构造次数**——修前 2 次（两个不同池对象 ⇒ panel 缓存
  必然 miss ⇒ 面板也建 2 次），修后 1 次；
- 面板 golden：consume 与非 consume 两条路径的 ``LibraryPanel.X`` **逐值**比对
  （含 NaN 位置），并用「删一列的池」做反例证明断言非恒真；
- 白名单：拿一个**故意不覆盖全部叶**的库 + 候选集，断言算出的叶集恰等于
  「候选叶 ∪ 库内全部记录叶」，并断言库外的叶不在里面。
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

MARKET = "ashare"


def _prepped(n_days: int = 120, n_stocks: int = 40, seed: int = 11) -> pl.DataFrame:
    """已 prep 形态的小帧（含 A 股 leaf_map 会用到的列，按 (ts_code, trade_date) 排序）。"""
    rng = np.random.default_rng(seed)
    days: list[dt.date] = []
    d = dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        px = 10.0 + rng.uniform(0, 5)
        for day in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({
                "trade_date": day,
                "ts_code": c,
                "close": px,
                "close_adj": px,
                "open_adj": px * (1 + rng.standard_normal() * 0.003),
                "high_adj": px * 1.01,
                "low_adj": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
            })
    return pl.DataFrame(rows).sort(["ts_code", "trade_date"])


_ACTIVE_EXPRS = ("rank(close)", "rank(vol)", "rank(amount)")


def _seed_library(root) -> None:
    from factorzen.discovery.factor_library import FactorRecord, _save_library

    recs = [
        FactorRecord(
            expression=e, market=MARKET, status="active",
            admission_track="single", ic_train=0.05 - 0.01 * i, holdout_ic=0.04,
            n_train=100, added_at="2026-07-01", updated_at="2026-07-01",
        )
        for i, e in enumerate(_ACTIVE_EXPRS)
    ]
    _save_library(MARKET, recs, root=str(root))


def _pool(prepped, root):
    from factorzen.discovery.factor_library import build_library_pool

    return build_library_pool(
        MARKET, prepped, None, root=str(root), statuses=("active",), compact=True,
    )


# ── 1. 面板 golden：consume 逐列释放宽表，X 必须逐值一致 ────────────────────────


def test_panel_consume_golden_x_identical(tmp_path):
    """``consume=True`` 只改「什么时候释放列」，X 必须与 consume=False 逐值一致。"""
    from factorzen.discovery.residual import build_library_panel

    _seed_library(tmp_path)
    prepped = _prepped()
    ref = build_library_panel(_pool(prepped, tmp_path))
    got = build_library_panel(_pool(prepped, tmp_path), consume=True)
    assert ref is not None and got is not None
    assert ref.factor_names == got.factor_names
    assert ref.dates == got.dates and ref.stocks == got.stocks
    assert ref.X.dtype == got.X.dtype
    # 逐值（含 NaN 位置）；X 由 _cs_zscore_null0 产出本无 NaN，equal_nan 兜住未来变更
    assert np.array_equal(ref.X, got.X, equal_nan=True)


def test_panel_consume_releases_wide_columns(tmp_path):
    """consume=True 后宽帧值列必须真的被丢掉（否则「宽帧+X 共存」没被压掉）。"""
    from factorzen.discovery.residual import build_library_panel

    _seed_library(tmp_path)
    prepped = _prepped()
    pool = _pool(prepped, tmp_path)
    n_before = len(pool.wide.columns)
    assert n_before == 2 + len(_ACTIVE_EXPRS)
    panel = build_library_panel(pool, consume=True)
    assert panel is not None and panel.k == len(_ACTIVE_EXPRS)
    # 键列保留（baseline_hash / 行轴），值列全释放
    assert set(pool.wide.columns) == {"trade_date", "ts_code"}
    # factor_names 不动：_baseline_hash / len(pool) 语义不变
    assert len(pool) == len(_ACTIVE_EXPRS)
    # 释放后再取值列 = 大声失败，不许静默返回空
    import pytest

    with pytest.raises(KeyError):
        _ = pool[_ACTIVE_EXPRS[0]]


def test_panel_consume_default_off_keeps_wide(tmp_path):
    """默认 consume=False：注入池（team 钩子复用 session lib_pool）零回归。"""
    from factorzen.discovery.residual import build_library_panel

    _seed_library(tmp_path)
    pool = _pool(_prepped(), tmp_path)
    build_library_panel(pool)
    assert set(pool.wide.columns) == {"trade_date", "ts_code", *_ACTIVE_EXPRS}
    assert pool[_ACTIVE_EXPRS[0]].height > 0


def test_panel_golden_reference_is_discriminating(tmp_path):
    """反例：少一条 active 的池 → X 不同。证明上面的逐值断言不是恒真。"""
    from factorzen.discovery.factor_library import FactorRecord, _save_library
    from factorzen.discovery.residual import build_library_panel

    _seed_library(tmp_path)
    prepped = _prepped()
    full = build_library_panel(_pool(prepped, tmp_path))
    # build_library_pool 无 only 形参：直接砍库文件重建一个「少两条」的池
    _save_library(MARKET, [
        FactorRecord(
            expression=_ACTIVE_EXPRS[0], market=MARKET, status="active",
            admission_track="single", ic_train=0.05, holdout_ic=0.04,
            n_train=100, added_at="2026-07-01", updated_at="2026-07-01",
        ),
    ], root=str(tmp_path))
    short_pool = _pool(prepped, tmp_path)
    short = build_library_panel(short_pool)
    assert full is not None and short is not None
    assert full.X.shape != short.X.shape


# ── 2. 池复用：组门 + 逐条 = 一个池、一个面板 ──────────────────────────────────


def _ctx(prepped, root, **kw):
    from factorzen.discovery.lift_test import make_lift_context

    return make_lift_context(
        MARKET, prepped, prepped=prepped, horizon=5,
        library_root=str(root), **kw,
    )


def _queue() -> list[dict]:
    return [
        {"expression": "rank(low)", "ic_train": 0.03, "residual_ic_train": 0.02},
        {"expression": "rank(high)", "ic_train": 0.02, "residual_ic_train": 0.015},
    ]


def test_group_then_individual_builds_pool_once(tmp_path, monkeypatch):
    """CLI lift-test 的真实调用序（组门 → 逐条）只应构造 1 次库池 / 1 次面板。

    修前：两处各自 ``build_library_pool``，两个不同对象 ⇒ ``panel_for`` 的对象
    同一性缓存必然 miss ⇒ 面板/QR 也各建一次（全 A 各 ~4.7G，正是 CLI 探针死点）。
    """
    from factorzen.discovery import factor_library as fl
    from factorzen.discovery import residual as res
    from factorzen.discovery.lift_test import run_group_lift, run_lift_tests

    _seed_library(tmp_path)
    prepped = _prepped()
    ctx = _ctx(prepped, tmp_path)

    n_pool = {"n": 0}
    n_panel = {"n": 0}
    real_pool = fl.build_library_pool
    real_panel = res.build_library_panel

    def c_pool(*a, **kw):
        n_pool["n"] += 1
        return real_pool(*a, **kw)

    def c_panel(*a, **kw):
        n_panel["n"] += 1
        return real_panel(*a, **kw)

    monkeypatch.setattr(fl, "build_library_pool", c_pool)
    monkeypatch.setattr(res, "build_library_panel", c_panel)

    q = _queue()
    run_group_lift(
        q, market=MARKET, daily=prepped, leaf_map=None,
        library_root=str(tmp_path), ctx=ctx,
    )
    rows = run_lift_tests(
        q, market=MARKET, daily=prepped, leaf_map=None,
        library_root=str(tmp_path), ctx=ctx, lift_workers=1,
    )
    assert len(rows) == len(q)
    # 真跑出了 lift 值（不是 materialize_failed 的空壳行）——否则计数断言没判别力
    assert all(r.get("lift") is not None for r in rows), rows
    assert n_pool["n"] == 1, f"库池构造 {n_pool['n']} 次（应 1 次）"
    assert n_panel["n"] == 1, f"库面板构造 {n_panel['n']} 次（应 1 次）"


def test_pool_reuse_does_not_change_lift_values(tmp_path):
    """池复用是纯内存改造：逐条 lift 数值必须与「无 ctx 缓存」路径逐值一致。"""
    import dataclasses

    from factorzen.discovery.lift_test import (
        LiftBaselineCache,
        run_group_lift,
        run_lift_tests,
    )

    _seed_library(tmp_path)
    prepped = _prepped()
    q = _queue()

    ctx_a = _ctx(prepped, tmp_path)
    run_group_lift(
        q, market=MARKET, daily=prepped, leaf_map=None,
        library_root=str(tmp_path), ctx=ctx_a,
    )
    rows_a = run_lift_tests(
        q, market=MARKET, daily=prepped, leaf_map=None,
        library_root=str(tmp_path), ctx=ctx_a, lift_workers=1,
    )
    # 每次都换新缓存 = 旧的「各建各的」路径
    ctx_b = dataclasses.replace(_ctx(prepped, tmp_path), baseline_cache=LiftBaselineCache())
    rows_b = run_lift_tests(
        q, market=MARKET, daily=prepped, leaf_map=None,
        library_root=str(tmp_path), ctx=ctx_b, lift_workers=1,
    )
    assert rows_a and all(r.get("lift") is not None for r in rows_a), rows_a
    for a, b in zip(rows_a, rows_b, strict=True):
        assert a["expression"] == b["expression"]
        for key in ("lift", "lift_se", "lift_second_half", "candidate_rank_ic", "passed"):
            av, bv = a.get(key), b.get(key)
            if isinstance(av, float) and isinstance(bv, float):
                assert av == bv, f"{key}: {av} != {bv}"
            else:
                assert av == bv, f"{key}: {av!r} != {bv!r}"


def test_injected_pool_is_not_consumed(tmp_path):
    """注入池（team 钩子传 session lib_pool）不得被 consume——session 还要用它。"""
    from factorzen.discovery.lift_test import run_lift_tests

    _seed_library(tmp_path)
    prepped = _prepped()
    pool = _pool(prepped, tmp_path)
    ctx = _ctx(prepped, tmp_path)
    run_lift_tests(
        _queue(), market=MARKET, daily=prepped, leaf_map=None,
        library_root=str(tmp_path), ctx=ctx, lift_workers=1,
        active_factor_dfs=pool,
    )
    assert set(pool.wide.columns) == {"trade_date", "ts_code", *_ACTIVE_EXPRS}


# ── 3. lift-test 的装配帧列白名单 ────────────────────────────────────────────


def test_lift_test_keep_leaves_is_candidates_union_library(tmp_path):
    """白名单 = 候选叶 ∪ 库内**全部**记录叶；库外叶不得混入。"""
    from factorzen.cli.main import _rebuild_keep_leaves
    from factorzen.discovery.factor_library import FactorRecord, _save_library

    _save_library(MARKET, [
        FactorRecord(
            expression="rank(close)", market=MARKET, status="active",
            admission_track="single", ic_train=0.05, holdout_ic=0.04,
            n_train=100, added_at="2026-07-01", updated_at="2026-07-01",
        ),
        # 非 active 也要进白名单：去相关/复审会碰到
        FactorRecord(
            expression="rank(pe_ttm)", market=MARKET, status="correlated",
            admission_track="lift", ic_train=0.01, holdout_ic=0.0,
            n_train=100, added_at="2026-07-01", updated_at="2026-07-01",
        ),
    ], root=str(tmp_path))
    leaves = _rebuild_keep_leaves(
        [], ["div(turnover_rate, amount)"], MARKET, str(tmp_path),
    )
    assert leaves is not None
    assert set(leaves) == {"turnover_rate", "amount", "close", "pe_ttm"}
    assert "roe" not in set(leaves)


def test_lift_test_cli_sets_keep_leaves(tmp_path, monkeypatch):
    """``fz factor-library lift-test`` 必须在装配前置位 ``args.keep_leaves``。

    修前它一列不裁 —— 全 A × 10 年的 prepped 全宽帧是 CLI 探针峰值的最大常驻项。
    """
    import argparse
    import json

    from factorzen.cli import main as cli
    from factorzen.discovery.factor_library import FactorRecord, _save_library

    _save_library(MARKET, [
        FactorRecord(
            expression="rank(close)", market=MARKET, status="active",
            admission_track="single", ic_train=0.05, holdout_ic=0.04,
            n_train=100, added_at="2026-07-01", updated_at="2026-07-01",
        ),
    ], root=str(tmp_path))

    sess = tmp_path / "sess"
    sess.mkdir()
    (sess / "manifest.json").write_text(json.dumps({
        "attempts": [
            {"expression": "rank(turnover_rate)", "ic_train": 0.03,
             "residual_ic_train": 0.02, "reject_category": "lift_queue"},
        ],
        "params": {"data_window": {"start": "20220101", "end": "20221231"}},
    }), encoding="utf-8")

    seen: dict = {}

    def fake_prep(args):
        seen["keep_leaves"] = getattr(args, "keep_leaves", None)
        return None, None, {}

    monkeypatch.setattr(cli, "_prepare_agent_mining_data", fake_prep)
    args = argparse.Namespace(
        session=[str(sess)], factor=None, market=MARKET, library_root=str(tmp_path),
        start="20220101", end="20221231", universe=None, set=None,
        admission_start=None, admission_end=None, horizon=None, apply=False,
    )
    rc = cli._cmd_factor_library_lift_test(args)
    assert rc == 1  # 装配帧为空 → 退出 1（本用例只关心 keep_leaves 已置位）
    assert seen.get("keep_leaves") is not None, "lift-test 未置位 keep_leaves"
    assert set(seen["keep_leaves"]) == {"turnover_rate", "close"}

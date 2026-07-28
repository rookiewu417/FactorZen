"""rebuild lift 复审的内存路径：active 池走 compact 入口 + 基线设施跨调用复用。

背景（全 A 2016-2026 rebuild 实测 exit=137 / 峰值 23.2G，两个死因）：

1. ``factor_library.rebuild`` **自建** active 池（逐条 ``materialize(expr)`` 塞 dict），
   绕过 ``build_library_pool`` 的 compact / f32 / spill 三件套；
2. ``run_lift_tests`` 是**逐条记录**调的，每次都对同一个 active 池重建
   ``build_library_panel`` + ``ResidualProjector`` + ``library_equal_weight_score_panel``
   + ret 面板 —— 峰值随记录数反复出现。

本文件的判别力来源：**不是**断言「函数被调用了」，而是
- 池：断言复审 runner 拿到的对象**就是** ``build_library_pool`` 的返回值（强制 compact
  时它是 ``CompactLibraryPool``，旧自建路径只可能是普通 dict）；
- 复用：真跑 ``run_lift_tests``，数**构造次数**（修前 = 记录数，修后 = 1）；
- 等价：新旧两条池构建路径端到端跑同一批记录，lift/裁决字段**逐值**比对，
  并用「故意删一条 active 的池」做反例证明这些断言非恒真。
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl

MARKET = "ashare"


# ── 合成 prepped 帧：可直接喂 evaluate_materialized / _build_ret_panel ──────────


def _prepped(n_days: int = 140, n_stocks: int = 45, seed: int = 7) -> pl.DataFrame:
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
# lift 轨复审目标（多条 → 逐条调 runner，正是峰值反复出现的形态）
_LIFT_EXPRS = ("rank(high)", "rank(low)", "rank(open)")


def _seed_library(root, *, with_lift: bool = True) -> None:
    from factorzen.discovery.factor_library import FactorRecord, _save_library

    recs = []
    # ic_train 递减 —— 让文件序与 build_library_pool 的 (-|ic_train|, expr) 序一致，
    # 使「旧自建池」与「新 compact 入口池」列序相同，等价性断言可以严格逐值比。
    for i, e in enumerate(_ACTIVE_EXPRS):
        recs.append(FactorRecord(
            expression=e, market=MARKET, status="active",
            admission_track="single", ic_train=0.05 - 0.01 * i, holdout_ic=0.04,
            n_train=100, added_at="2026-07-01", updated_at="2026-07-01",
        ))
    if with_lift:
        for i, e in enumerate(_LIFT_EXPRS):
            recs.append(FactorRecord(
                expression=e, market=MARKET, status="active",
                admission_track="lift", ic_train=0.012 - 0.001 * i, holdout_ic=0.0,
                lift=0.01, lift_se=0.001, lift_second_half=0.005, horizon=5,
                added_at="2026-07-02", updated_at="2026-07-02",
            ))
    _save_library(MARKET, recs, root=str(root))


def _evaluate_stub(exprs):
    """single 轨评估桩：把 _ACTIVE_EXPRS 原样收回库（upsert 后它们才是 active 池成员）。"""
    out = []
    for i, e in enumerate(_ACTIVE_EXPRS):
        if e in exprs:
            out.append({
                "expression": e, "ic_train": 0.05 - 0.01 * i, "holdout_ic": 0.04,
                "dsr_pvalue": 0.2, "n_train": 100, "n_holdout_days": 100,
            })
    return out


def _run_rebuild(root, prepped, **kw):
    from factorzen.discovery.factor_library import rebuild

    return rebuild(
        MARKET,
        sources=list(_ACTIVE_EXPRS),
        eval_window=("20220101", "20221231"),
        universe=None, horizon=5,
        evaluate=_evaluate_stub, git_sha="t", now="2026-07-28",
        root=str(root), fresh=True,
        prepped=prepped, daily=None,
        admission_start=None, admission_end=None,
        **kw,
    )


# ── 1. 池：rebuild 不再自建 dict，改走 build_library_pool（compact 入口）────────


def test_rebuild_active_pool_uses_compact_entry(tmp_path, monkeypatch):
    """复审 runner 拿到的 active 池 **就是** ``build_library_pool`` 的返回对象。

    判别力：强制 compact 后返回 ``CompactLibraryPool``——旧自建循环
    （``pool_dfs[expr] = materialize(expr)``）在任何配置下都只可能产出普通 dict；
    且此处 ``materialize=None``，旧路径的池必然是空 dict。
    """
    import factorzen.discovery.factor_library as fl
    from factorzen.discovery.factor_library import CompactLibraryPool

    _seed_library(tmp_path)
    prepped = _prepped()

    monkeypatch.setattr(fl, "should_use_compact_pool", lambda *a, **k: True)
    seen_pool_calls: list[dict] = []
    real_build = fl.build_library_pool

    def spy_build(market, daily, leaf_map=None, **kw):
        seen_pool_calls.append(dict(kw, _n_cols=len(daily.columns)))
        return real_build(market, daily, leaf_map, **kw)

    monkeypatch.setattr(fl, "build_library_pool", spy_build)

    got: list = []

    def lift_runner(cands, *, active_factor_dfs=None, **kw):
        got.append(active_factor_dfs)
        return [{
            "expression": cands[0]["expression"], "lift": 0.006, "lift_se": 0.001,
            "lift_second_half": 0.003, "baseline": 0.05, "passed": True,
        }]

    _run_rebuild(tmp_path, prepped, lift_runner=lift_runner, materialize=None)

    assert len(seen_pool_calls) == 1, f"active 池应恰好建一次，got {len(seen_pool_calls)}"
    assert seen_pool_calls[0]["statuses"] == ("active",)
    assert seen_pool_calls[0]["root"] == str(tmp_path)
    # 列裁剪：只喂键 + active 表达式引用的叶列（close_adj/vol/amount + 2 键 = 5）
    assert seen_pool_calls[0]["_n_cols"] < len(prepped.columns)

    assert len(got) == len(_LIFT_EXPRS), "每条 lift 记录调一次 runner"
    pool = got[0]
    assert isinstance(pool, CompactLibraryPool), (
        f"应走 compact 入口，got {type(pool).__name__}（旧自建 dict 路径没被替掉）"
    )
    assert set(pool.factor_names) == set(_ACTIVE_EXPRS)
    for other in got[1:]:
        assert other is pool, "逐条复审必须复用同一个池对象，不得每条重建"


def test_rebuild_active_pool_falls_back_without_frame(tmp_path):
    """无 prepped/daily（只注入 materialize 的桩）→ 回退旧自建循环，零回归。"""
    _seed_library(tmp_path)
    panel = pl.DataFrame({
        "trade_date": [dt.date(2024, 1, 2)],
        "ts_code": ["000001.SZ"],
        "factor_value": [1.0],
    })
    got: list = []

    def lift_runner(cands, *, active_factor_dfs=None, **kw):
        got.append(active_factor_dfs)
        return [{"expression": cands[0]["expression"], "lift": 0.006,
                 "lift_se": 0.001, "lift_second_half": 0.003, "passed": True}]

    _run_rebuild(
        tmp_path, None, lift_runner=lift_runner, materialize=lambda e: panel,
    )
    assert got and isinstance(got[0], dict)
    assert set(got[0]) == set(_ACTIVE_EXPRS)


def test_rebuild_active_pool_honors_in_universe(tmp_path):
    """帧含 ``in_universe`` 时池只留成分内行 —— 与旧 ``_factor_df_from_prepped`` 同口径。

    判别力：非成分行的值被构造成**唯一可识别**的极端值；若过滤丢失，池里就会出现
    这些 (date, code) 键。
    """
    _seed_library(tmp_path)
    prepped = _prepped(n_days=40, n_stocks=35)
    out_codes = {"000000.SZ", "000001.SZ"}
    prepped = prepped.with_columns(
        (~pl.col("ts_code").is_in(list(out_codes))).alias("in_universe"),
    )

    got: list = []

    def lift_runner(cands, *, active_factor_dfs=None, **kw):
        got.append(active_factor_dfs)
        return [{"expression": cands[0]["expression"], "lift": 0.006,
                 "lift_se": 0.001, "lift_second_half": 0.003, "passed": True}]

    _run_rebuild(tmp_path, prepped, lift_runner=lift_runner, materialize=None)

    pool = got[0]
    assert pool, "池不应为空"
    codes = set(pool["rank(close)"]["ts_code"].to_list())
    assert codes and not (codes & out_codes), (
        f"in_universe=False 的票不得进基线池，泄漏 {codes & out_codes}"
    )


# ── 2. 复用：逐条复审只建一次 panel / projector / ret ───────────────────────────


def _count_baseline_builds(monkeypatch) -> dict[str, int]:
    """给四个基线设施构造点装计数器（真实实现照常执行）。"""
    import factorzen.discovery.lift_test as lt
    import factorzen.discovery.residual as res

    n = {"panel": 0, "projector": 0, "ret": 0, "scores": 0}

    real_panel = res.build_library_panel
    real_proj = res.ResidualProjector
    real_ret = lt._build_ret_panel
    real_scores = lt.library_equal_weight_score_panel

    def c_panel(pool):
        n["panel"] += 1
        return real_panel(pool)

    class CProj(real_proj):  # type: ignore[misc, valid-type]
        def __init__(self, panel):
            n["projector"] += 1
            super().__init__(panel)

    def c_ret(daily, **kw):
        n["ret"] += 1
        return real_ret(daily, **kw)

    def c_scores(panel):
        n["scores"] += 1
        return real_scores(panel)

    # run_lift_tests / run_group_lift 是函数内 import residual 的，patch 源模块
    monkeypatch.setattr(res, "build_library_panel", c_panel)
    monkeypatch.setattr(res, "ResidualProjector", CProj)
    monkeypatch.setattr(lt, "_build_ret_panel", c_ret)
    monkeypatch.setattr(lt, "library_equal_weight_score_panel", c_scores)
    return n


def test_rebuild_lift_review_builds_baseline_once(tmp_path, monkeypatch):
    """3 条 lift 记录 → 3 次 ``run_lift_tests``，但 panel/QR/ret 各只建 1 次。

    修复前：各建 3 次（峰值随记录数反复出现，全 A 第 2 条就被 OOM-kill）。
    ``base_scores``（等权主分数长表）稠密路径根本不读 → 应为 0 次。
    """
    _seed_library(tmp_path)
    prepped = _prepped()
    n = _count_baseline_builds(monkeypatch)

    res = _run_rebuild(tmp_path, prepped, materialize=None)

    assert res.lift_review_error is None, res.lift_review_error
    assert n["panel"] == 1, f"库面板应只建 1 次，got {n['panel']}"
    assert n["projector"] == 1, f"ResidualProjector（逐日 QR）应只建 1 次，got {n['projector']}"
    assert n["ret"] == 1, f"ret 面板应只建 1 次，got {n['ret']}"
    assert n["scores"] == 0, (
        f"稠密 residual 路径不读等权主分数，不该构造，got {n['scores']}"
    )


def test_group_lift_and_per_candidate_share_ctx_cache(monkeypatch):
    """双路径检查：``run_group_lift`` → ``run_lift_tests`` 共用同一 ctx 的缓存。

    这是 CLI ``factor-library lift-test`` 与 team session 末钩子的调用形态
    （组门跑完紧接着逐条测），且 CLI 逐组 ``dataclasses.replace`` 出来的 ctx
    必须共享同一缓存实例。
    """
    import dataclasses

    from factorzen.discovery.lift_test import (
        LiftEvalContext,
        run_group_lift,
        run_lift_tests,
    )

    prepped = _prepped(n_days=90, n_stocks=45)
    n = _count_baseline_builds(monkeypatch)

    pool = {
        "lib_a": prepped.select(
            ["trade_date", "ts_code", pl.col("vol").alias("factor_value")],
        ),
    }
    cand = prepped.select(
        ["trade_date", "ts_code", pl.col("amount").alias("factor_value")],
    )
    ctx = LiftEvalContext(
        market=MARKET, prepped=prepped, leaf_map=None, horizon=5,
        admission_start=None, admission_end=None, library_root="/tmp/none",
    )
    grp_ctx = dataclasses.replace(ctx, admission_start="2022-03-01")

    run_group_lift(
        [{"expression": "c"}], market=MARKET, daily=prepped,
        active_factor_dfs=pool, materialize_candidate=lambda e: cand, ctx=ctx,
    )
    run_lift_tests(
        [{"expression": "c"}], market=MARKET, daily=prepped,
        active_factor_dfs=pool, materialize_candidate=lambda e: cand,
        ctx=grp_ctx, lift_workers=1,
    )

    assert n["panel"] == 1, f"组门与逐条应共用库面板，got {n['panel']}"
    assert n["projector"] == 1, f"组门与逐条应共用逐日 QR，got {n['projector']}"
    assert n["ret"] == 1, f"组门与逐条应共用 ret 面板，got {n['ret']}"
    assert ctx.baseline_cache is grp_ctx.baseline_cache, (
        "dataclasses.replace 必须沿用同一缓存实例，否则 CLI 逐组仍会重建"
    )


def test_baseline_cache_rebuilds_on_different_pool(monkeypatch):
    """换了池对象必须重建 —— 缓存按对象同一性命中，绝不跨基线复用。"""
    from factorzen.discovery.lift_test import LiftEvalContext, run_lift_tests

    prepped = _prepped(n_days=90, n_stocks=45)
    n = _count_baseline_builds(monkeypatch)
    base = prepped.select(
        ["trade_date", "ts_code", pl.col("vol").alias("factor_value")],
    )
    cand = prepped.select(
        ["trade_date", "ts_code", pl.col("amount").alias("factor_value")],
    )
    ctx = LiftEvalContext(
        market=MARKET, prepped=prepped, leaf_map=None, horizon=5,
        admission_start=None, admission_end=None, library_root="/tmp/none",
    )
    for pool in ({"lib_a": base}, {"lib_a": base}):   # 值相同、对象不同
        run_lift_tests(
            [{"expression": "c"}], market=MARKET, daily=prepped,
            active_factor_dfs=pool, materialize_candidate=lambda e: cand,
            ctx=ctx, lift_workers=1,
        )
    assert n["panel"] == 2, f"不同池对象必须各建各的，got {n['panel']}"


# ── 3. golden 等价：新旧两条池构建路径的 lift 裁决逐值一致 ────────────────────

_GOLDEN_FIELDS = (
    "status", "admission_track", "admission_decision",
    "lift", "lift_se", "lift_baseline", "lift_first_half", "lift_second_half",
    "admission_ic", "lift_metric", "n_lib_factors", "block_days",
    "lift_scored_start", "lift_scored_end", "lift_admission_start",
    "lift_admission_end", "lift_horizon", "lift_baseline_hash",
)


def _legacy_pool(root, prepped) -> dict[str, pl.DataFrame]:
    """逐字节复刻旧 ``rebuild`` 自建池：``materialize(expr)`` 逐条塞 dict。

    列序刻意按 ``build_library_pool`` 的 ``(-|ic_train|, expression)`` 排 —— 两条路径
    的 QR 基列序相同，等价性断言才落在「池怎么算出来的」上，而不是浮点列序噪声上。
    """
    from factorzen.discovery.factor_library import load_library
    from factorzen.discovery.lift_test import _materializer_from_prepped

    mat = _materializer_from_prepped(prepped, None, python_market=MARKET)
    recs = [r for r in load_library(MARKET, root=str(root)) if r.status == "active"]
    recs.sort(key=lambda r: (-abs(r.ic_train or 0.0), r.expression))
    pool: dict[str, pl.DataFrame] = {}
    for r in recs:
        panel = mat(r.expression)
        if panel is not None:
            pool[r.expression] = panel
    return pool


def _lift_view(root) -> dict[str, dict]:
    from factorzen.discovery.factor_library import load_library

    return {
        r.expression: {f: getattr(r, f, None) for f in _GOLDEN_FIELDS}
        for r in load_library(MARKET, root=str(root))
        if (r.admission_track or "single") == "lift"
    }


def test_rebuild_lift_review_golden_pool_equivalence(tmp_path):
    """新池（build_library_pool）与旧自建池端到端跑真 ``run_lift_tests``，裁决逐值一致。

    反例（判别力证明）：故意从池里删掉一条 active 后，同样的比对必须**失败**——
    否则说明这些断言对池内容不敏感，等于没测。
    """
    prepped = _prepped()

    new_root = tmp_path / "new"
    new_root.mkdir()
    _seed_library(new_root)
    res_new = _run_rebuild(new_root, prepped, materialize=None)
    assert res_new.lift_review_error is None, res_new.lift_review_error
    view_new = _lift_view(new_root)
    assert len(view_new) == len(_LIFT_EXPRS)
    assert all(v["lift"] is not None for v in view_new.values()), (
        f"真跑 lift 必须算出数（否则等价性恒真）：{view_new}"
    )

    # 旧路径：把「旧自建池」原样注入（rebuild 对注入池短路，不走新构建）
    old_root = tmp_path / "old"
    old_root.mkdir()
    _seed_library(old_root)
    ref_pool = _legacy_pool(new_root, prepped)
    assert set(ref_pool) == set(_ACTIVE_EXPRS)
    res_old = _run_rebuild(
        old_root, prepped, materialize=None, active_factor_dfs=ref_pool,
    )
    assert res_old.lift_review_error is None, res_old.lift_review_error
    view_old = _lift_view(old_root)

    assert view_new == view_old, (
        "新旧池路径的 lift 裁决字段必须逐值一致\n"
        f"new={view_new}\nold={view_old}"
    )

    # ── 反例：池少一条 active → 残差基线变 → 指标必须变 ────────────────────
    ctrl_root = tmp_path / "ctrl"
    ctrl_root.mkdir()
    _seed_library(ctrl_root)
    short_pool = {k: v for k, v in ref_pool.items() if k != _ACTIVE_EXPRS[-1]}
    _run_rebuild(
        ctrl_root, prepped, materialize=None, active_factor_dfs=short_pool,
    )
    view_ctrl = _lift_view(ctrl_root)
    assert view_ctrl != view_new, (
        "删掉一条 active 后裁决字段竟然没变——上面的等价性断言是恒真的"
    )


def test_rebuild_lift_review_golden_cache_equivalence(tmp_path, monkeypatch):
    """缓存开/关的裁决必须逐值一致：缓存只决定「算不算」，不参与任何数值。"""
    import factorzen.discovery.lift_test as lt

    prepped = _prepped()

    on_root = tmp_path / "cache_on"
    on_root.mkdir()
    _seed_library(on_root)
    _run_rebuild(on_root, prepped, materialize=None)
    view_on = _lift_view(on_root)

    # 把缓存打成「永不命中」（每次都真建）→ 等价于修复前的逐条重建
    class _NoCache(lt.LiftBaselineCache):
        def panel_for(self, pool, build):
            return build(pool)

        def projector_for(self, panel, build):
            return build(panel)

        def scores_for(self, panel, build):
            return build(panel)

        def ret_for(self, src, key, build):
            return build(src)

    monkeypatch.setattr(lt, "LiftBaselineCache", _NoCache)
    off_root = tmp_path / "cache_off"
    off_root.mkdir()
    _seed_library(off_root)
    _run_rebuild(off_root, prepped, materialize=None)
    view_off = _lift_view(off_root)

    assert view_on and view_on == view_off, (
        f"缓存不得改变任何数值\non={view_on}\noff={view_off}"
    )


# ── 4. 两处「白付」的内存：池源帧重复 sort、逐日 Q 全量预计算 ────────────────


def test_build_library_pool_skips_redundant_sort(tmp_path, monkeypatch):
    """源帧已按 (ts_code, trade_date) 排序 → 不再 sort（那是一次全量物化拷贝）。

    判别力：分两次断言——已排序帧 sort 调用数 = 0，乱序帧 = 1，且两条路径产出的
    池**逐值一致**（省掉的是拷贝，不是排序语义）。
    """
    import factorzen.discovery.factor_library as fl

    _seed_library(tmp_path, with_lift=False)
    prepped = _prepped(n_days=30, n_stocks=12)
    shuffled = prepped.sample(fraction=1.0, shuffle=True, seed=3)

    n_sort = {"n": 0}
    real_sort = pl.DataFrame.sort

    def spy_sort(self, *a, **k):
        n_sort["n"] += 1
        return real_sort(self, *a, **k)

    monkeypatch.setattr(pl.DataFrame, "sort", spy_sort)

    pool_sorted = fl.build_library_pool(
        MARKET, prepped, None, statuses=("active",), root=str(tmp_path),
    )
    assert n_sort["n"] == 0, f"已排序帧不该再 sort，调了 {n_sort['n']} 次"

    n_sort["n"] = 0
    pool_shuffled = fl.build_library_pool(
        MARKET, shuffled, None, statuses=("active",), root=str(tmp_path),
    )
    assert n_sort["n"] >= 1, "乱序帧必须照旧 sort"

    monkeypatch.undo()
    assert set(pool_sorted) == set(pool_shuffled) == set(_ACTIVE_EXPRS)
    for e in _ACTIVE_EXPRS:
        a = pool_sorted[e].sort(["ts_code", "trade_date"])
        b = pool_shuffled[e].sort(["ts_code", "trade_date"])
        assert a.height == b.height > 0, e
        assert a["factor_value"].to_list() == b["factor_value"].to_list(), e


def test_sorted_by_pool_keys_detects_disorder():
    """排序判据本身要有判别力：只错一行也必须判 False。"""
    from factorzen.discovery.factor_library import _sorted_by_pool_keys

    df = _prepped(n_days=6, n_stocks=4).select(["ts_code", "trade_date"])
    assert _sorted_by_pool_keys(df)
    # 同一 ts_code 内把两天对调 → 日期倒序
    swapped = df[[1, 0, *range(2, df.height)]]
    assert not _sorted_by_pool_keys(swapped)
    # ts_code 倒序
    assert not _sorted_by_pool_keys(df.sort(["ts_code"], descending=True))
    # 空帧 / 缺列 → 保守判 False（照旧 sort）
    assert not _sorted_by_pool_keys(df.head(0))
    assert not _sorted_by_pool_keys(df.select(["ts_code"]))


def test_residual_projector_qr_is_lazy_and_value_identical():
    """逐日 Q 改惰性：候选铺不满全日轴时一次都不算，且残差逐值不变。

    判别力：拿「构造后立刻全量预计算」的旧行为当 golden——先跑惰性版拿残差，
    再手工把每日 Q 全算出来重跑，两次残差逐值必须相同；同时断言子集候选跑完
    之后 Q 缓存仍是**全空**（旧实现构造完就满了）。
    """
    from factorzen.discovery.residual import (
        _Q_UNSET,
        ResidualProjector,
        build_library_panel,
    )

    prepped = _prepped(n_days=40, n_stocks=45)
    pool = {
        "lib_a": prepped.select(
            ["trade_date", "ts_code", pl.col("vol").alias("factor_value")],
        ),
        "lib_b": prepped.select(
            ["trade_date", "ts_code", pl.col("amount").alias("factor_value")],
        ),
    }
    panel = build_library_panel(pool)
    assert panel is not None and panel.k == 2

    # 候选只覆盖一部分票 → 永远走子集分支
    subset = set(prepped["ts_code"].unique().to_list()[:40])
    cand = prepped.filter(pl.col("ts_code").is_in(list(subset))).select(
        ["trade_date", "ts_code", pl.col("close").alias("factor_value")],
    )

    lazy = ResidualProjector(panel)
    assert all(q is _Q_UNSET for q in lazy._Q), "构造期不该预计算任何 Q"
    out_lazy = lazy.residualize(cand).sort(["ts_code", "trade_date"])
    assert out_lazy.height > 0
    assert all(q is _Q_UNSET for q in lazy._Q), (
        "候选铺不满全日轴，全日 Q 一次都不该被算"
    )

    # golden：复刻旧构造期全量预计算，再跑同一候选
    eager = ResidualProjector(panel)
    from factorzen.discovery.residual import _qr_basis

    eager._Q = [_qr_basis(panel.X[di]) for di in range(panel.n_dates)]
    out_eager = eager.residualize(cand).sort(["ts_code", "trade_date"])
    assert out_lazy.height == out_eager.height
    assert (
        out_lazy["factor_value"].to_list() == out_eager["factor_value"].to_list()
    ), "惰性化改变了残差数值"

    # 全覆盖候选：快路径命中 → Q 必须真的被算出来并缓存
    full_cand = prepped.select(
        ["trade_date", "ts_code", pl.col("close").alias("factor_value")],
    )
    lazy2 = ResidualProjector(panel)
    lazy2.residualize(full_cand)
    assert any(q is not _Q_UNSET for q in lazy2._Q), "快路径必须算并缓存全日 Q"


def test_sleeve_path_still_gets_base_scores(monkeypatch):
    """``base_scores`` 改惰性后，sleeve/overlay 分支必须照常拿到它。"""
    from factorzen.discovery.lift_test import LiftEvalContext, run_lift_tests

    prepped = _prepped(n_days=120, n_stocks=45)
    n = _count_baseline_builds(monkeypatch)
    pool = {
        "lib_a": prepped.select(
            ["trade_date", "ts_code", pl.col("vol").alias("factor_value")],
        ),
    }
    cand = prepped.select(
        ["trade_date", "ts_code", pl.col("amount").alias("factor_value")],
    )
    ctx = LiftEvalContext(
        market=MARKET, prepped=prepped, leaf_map=None, horizon=5,
        admission_start=None, admission_end=None, library_root="/tmp/none",
    )
    rows = run_lift_tests(
        [{"expression": "c", "sleeve_candidate": True}],
        market=MARKET, daily=prepped, active_factor_dfs=pool,
        materialize_candidate=lambda e: cand, ctx=ctx, lift_workers=1,
    )
    assert n["scores"] == 1, f"sleeve 路径必须构造等权主分数，got {n['scores']}"
    assert rows and rows[0].get("lift_metric") == "overlay_v1"
    assert rows[0].get("error") is None, rows[0].get("error")
    # baseline = 等权主分数的日 IC 均值：拿不到 base_scores 时这里必是 None
    assert isinstance(rows[0].get("baseline"), float)
    assert math.isfinite(float(rows[0]["baseline"]))
    assert isinstance(rows[0].get("lift"), float) and math.isfinite(rows[0]["lift"])

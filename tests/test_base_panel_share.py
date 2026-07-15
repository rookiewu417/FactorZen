"""任务 G：base_panel 共享 + 自适应 lift_workers。

parity golden 硬约束：全量 build vs base_panel 增量路径逐值一致。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl


def _dates(n_days: int):
    days, d = [], date(2024, 1, 2)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return days


def _long_factor(
    dates: list[str],
    codes: list[str],
    rng: np.random.Generator,
    *,
    scale: float = 1.0,
    extra_rows: list[tuple[str, str, float]] | None = None,
) -> pl.DataFrame:
    rows = []
    for d in dates:
        vals = rng.standard_normal(len(codes)) * scale
        for s, code in enumerate(codes):
            rows.append({
                "trade_date": d,
                "ts_code": code,
                "factor_value": float(vals[s]),
            })
    if extra_rows:
        for d, code, v in extra_rows:
            rows.append({
                "trade_date": d,
                "ts_code": code,
                "factor_value": float(v),
            })
    return pl.DataFrame(rows)


def _synth_five_plus_one(
    *,
    n_days: int = 80,
    n_stocks: int = 24,
    seed: int = 7,
    candidate_extra: bool = True,
):
    """5 基线因子 + 1 候选；可选候选独有行（基线 5 列全缺）。"""
    rng = np.random.default_rng(seed)
    dates = _dates(n_days)
    codes = [f"{i:04d}.SZ" for i in range(n_stocks)]
    baseline = {
        f"b{i}": _long_factor(dates, codes, rng, scale=1.0 + 0.1 * i)
        for i in range(5)
    }
    extra = None
    if candidate_extra:
        # 候选覆盖超出基线：新日期 + 新股票（基线全缺）
        extra_date = _dates(n_days + 5)[-1]
        extra = [
            (extra_date, "9999.SZ", 1.5),
            (extra_date, codes[0], -0.8),
            (dates[10], "8888.SZ", 2.0),
        ]
        # ret 覆盖这些边角行
        ret_extra = list(extra)
    else:
        ret_extra = []

    cand = _long_factor(dates, codes, rng, scale=1.3, extra_rows=extra)
    ret_rows = []
    for d in dates:
        rets = 0.05 * rng.standard_normal(len(codes))
        for s, code in enumerate(codes):
            ret_rows.append({"trade_date": d, "ts_code": code, "ret": float(rets[s])})
    for d, code, _v in ret_extra:
        ret_rows.append({
            "trade_date": d,
            "ts_code": code,
            "ret": float(0.01 * rng.standard_normal()),
        })
    ret = pl.DataFrame(ret_rows)
    full = {**baseline, "cand": cand}
    return baseline, cand, ret, full


def _assert_panel_equal(a: pl.DataFrame, b: pl.DataFrame, *, atol: float = 1e-12):
    """宽表面板逐值一致（按键排序后比列）。"""
    keys = ["trade_date", "ts_code"]
    a_s = a.sort(keys)
    b_s = b.sort(keys)
    assert a_s.columns == b_s.columns, f"列序/列名不一致: {a_s.columns} vs {b_s.columns}"
    assert a_s.height == b_s.height
    assert a_s.select(keys).equals(b_s.select(keys))
    for col in a_s.columns:
        if col in keys:
            continue
        av = a_s[col].to_numpy().astype(float)
        bv = b_s[col].to_numpy().astype(float)
        # null 对齐
        a_null = a_s[col].is_null().to_numpy()
        b_null = b_s[col].is_null().to_numpy()
        assert np.array_equal(a_null, b_null), f"null 掩码不一致: {col}"
        both = ~a_null
        if both.any():
            np.testing.assert_allclose(av[both], bv[both], atol=atol, rtol=0)


def test_build_panel_base_share_parity_with_extra_rows():
    """build_panel 全量 vs base+候选：含候选超出基线行集边角。"""
    from factorzen.research.combination.models import build_panel

    baseline, cand, ret, full = _synth_five_plus_one(candidate_extra=True)
    full_panel = build_panel(full, ret)
    base = build_panel(baseline, ret)
    incr = build_panel({"cand": cand}, ret, base_panel=base)

    # 列：基线序 + cand 在最后（与 full dict 插入序一致）
    assert _feature_tail(full_panel) == ["b0", "b1", "b2", "b3", "b4", "cand"]
    assert _feature_tail(incr) == ["b0", "b1", "b2", "b3", "b4", "cand"]
    _assert_panel_equal(full_panel, incr)

    # 边角：存在基线全缺、候选有值、且有 ret 的行
    feat_cols = ["b0", "b1", "b2", "b3", "b4"]
    edge = incr.filter(
        pl.all_horizontal([pl.col(c).is_null() for c in feat_cols])
        & pl.col("cand").is_not_null()
    )
    assert edge.height >= 1, "应含候选超出基线行集的边角行"


def _feature_tail(panel: pl.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in ("trade_date", "ts_code", "ret")]


def test_combine_lgbm_base_panel_parity_golden():
    """核心 golden：combine_lgbm(全量) vs combine_lgbm(候选, base_panel=)。

    两条独立路径互证；禁止恒真。atol=1e-12。
    """
    from factorzen.research.combination.cv import PurgedWalkForwardCV
    from factorzen.research.combination.models import build_panel, combine_lgbm

    baseline, cand, ret, full = _synth_five_plus_one(
        n_days=90, n_stocks=20, seed=11, candidate_extra=True,
    )
    cv = PurgedWalkForwardCV(
        train_days=40, test_days=15, purge_days=5, embargo_days=0, expanding=False,
    )
    kw = dict(seed=3, n_estimators=30, min_child_samples=10, num_leaves=15)

    out_full = combine_lgbm(full, ret, cv, **kw)
    base = build_panel(baseline, ret)
    out_incr = combine_lgbm({"cand": cand}, ret, cv, base_panel=base, **kw)

    assert out_full.height > 0 and out_incr.height > 0
    # 两条路径都必须有真实预测，且非全零（防恒真）
    assert float(out_full["factor_value"].std()) > 1e-6
    assert float(out_incr["factor_value"].std()) > 1e-6

    a = out_full.sort(["trade_date", "ts_code"])
    b = out_incr.sort(["trade_date", "ts_code"])
    assert a.height == b.height
    assert a.select(["trade_date", "ts_code"]).equals(b.select(["trade_date", "ts_code"]))
    np.testing.assert_allclose(
        a["factor_value"].to_numpy().astype(float),
        b["factor_value"].to_numpy().astype(float),
        atol=1e-12,
        rtol=0,
    )

    # 全量 dict + base_panel 同样一致
    out_full_bp = combine_lgbm(full, ret, cv, base_panel=base, **kw)
    c = out_full_bp.sort(["trade_date", "ts_code"])
    np.testing.assert_allclose(
        a["factor_value"].to_numpy().astype(float),
        c["factor_value"].to_numpy().astype(float),
        atol=1e-12,
        rtol=0,
    )


def test_combine_lgbm_base_none_matches_no_kw():
    """base_panel=None 与不传 kw 一致（组合层零回归接口）。"""
    from factorzen.research.combination.cv import PurgedWalkForwardCV
    from factorzen.research.combination.models import combine_lgbm

    _baseline, _cand, ret, full = _synth_five_plus_one(
        n_days=70, n_stocks=16, seed=2, candidate_extra=False,
    )
    cv = PurgedWalkForwardCV(train_days=30, test_days=10, purge_days=3)
    kw = dict(seed=1, n_estimators=20, min_child_samples=8)
    a = combine_lgbm(full, ret, cv, **kw).sort(["trade_date", "ts_code"])
    b = combine_lgbm(full, ret, cv, base_panel=None, **kw).sort(
        ["trade_date", "ts_code"]
    )
    np.testing.assert_allclose(
        a["factor_value"].to_numpy().astype(float),
        b["factor_value"].to_numpy().astype(float),
        atol=1e-12,
        rtol=0,
    )


def test_run_lift_tests_base_panel_path_matches_mock():
    """run_lift_tests：mock combine 下 base_panel 路径与关闭共享路径结果一致。

    通过注入 combine_fn 关闭共享 vs 真 lgbm 路径另测。
    """
    from factorzen.discovery.lift_test import run_lift_tests

    active = {"lib_a": _long_factor(_dates(40), [f"{i:04d}.SZ" for i in range(8)],
                                    np.random.default_rng(0))}
    dates = _dates(40)
    codes = [f"{i:04d}.SZ" for i in range(8)]
    cand = _long_factor(dates, codes, np.random.default_rng(1))
    ret_rows = [
        {"trade_date": d, "ts_code": c, "ret": 0.01 * (i + 1)}
        for d in dates
        for i, c in enumerate(codes)
    ]
    ret = pl.DataFrame(ret_rows)

    def det_combine(fds, rdf, cv, **kw):
        n = len(fds)
        if n <= len(active):
            return ret.select(["trade_date", "ts_code"]).with_columns(
                pl.lit(0.0).alias("factor_value")
            )
        return rdf.select(
            ["trade_date", "ts_code", pl.col("ret").alias("factor_value")]
        )

    grays = [
        {"expression": "c0", "residual_ic_train": 0.01},
        {"expression": "c1", "residual_ic_train": 0.009},
    ]
    common = dict(
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        combine_fn=det_combine,
        top_m=None,
        threshold=0.001,
        block_days=10,
        seed=0,
        lift_workers=1,
    )
    # mock 关闭 base_panel 共享；两次调用应一致
    a = run_lift_tests(grays, **common)
    b = run_lift_tests(grays, **common)
    assert len(a) == len(b) == 2
    for x, y in zip(a, b, strict=True):
        assert x["lift"] == y["lift"]
        assert x["passed"] == y["passed"]
        assert x["expression"] == y["expression"]


def test_run_lift_tests_real_lgbm_base_panel_parity():
    """真 lgbm：生产 base_panel 共享路径与强制关闭共享（monkeypatch）结果一致。"""
    import factorzen.discovery.lift_test as lt
    from factorzen.research.combination.models import combine_lgbm as real_combine

    baseline, cand, ret, _full = _synth_five_plus_one(
        n_days=80, n_stocks=18, seed=5, candidate_extra=False,
    )
    # 用 2 个基线因子加速
    active = {"lib_a": baseline["b0"], "lib_b": baseline["b1"]}

    def thin_combine(fds, rdf, cv, **kw):
        return real_combine(
            fds, rdf, cv,
            seed=0, n_estimators=25, min_child_samples=10, num_leaves=12,
            **{k: v for k, v in kw.items() if k == "base_panel"},
        )

    grays = [{"expression": "cand_x", "residual_ic_train": 0.02}]
    daily = pl.DataFrame({"trade_date": [], "ts_code": [], "close": []})

    # 生产路径：combine_fn=None 内部走 combine_lgbm + base_panel
    # 用 thin wrapper 替真实 combine_lgbm
    import factorzen.research.combination.models as models_mod

    original = models_mod.combine_lgbm
    models_mod.combine_lgbm = thin_combine  # type: ignore[assignment]
    try:
        rows_share = lt.run_lift_tests(
            grays,
            market="ashare",
            daily=daily,
            active_factor_dfs=active,
            ret_df=ret,
            materialize_candidate=lambda e: cand,
            combine_fn=None,
            lift_workers=1,
            seed=0,
            top_m=None,
            threshold=-1.0,
            cv_params={
                "train_days": 35,
                "test_days": 12,
                "purge_days": 3,
                "expanding": False,
            },
        )
    finally:
        models_mod.combine_lgbm = original  # type: ignore[assignment]

    # 关闭共享：注入 combine_fn 不传 base_panel
    def no_share_combine(fds, rdf, cv, **kw):
        return real_combine(
            fds, rdf, cv,
            seed=0, n_estimators=25, min_child_samples=10, num_leaves=12,
        )

    rows_full = lt.run_lift_tests(
        grays,
        market="ashare",
        daily=daily,
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        combine_fn=no_share_combine,
        lift_workers=1,
        seed=0,
        top_m=None,
        threshold=-1.0,
        cv_params={
            "train_days": 35,
            "test_days": 12,
            "purge_days": 3,
            "expanding": False,
        },
    )

    assert len(rows_share) == len(rows_full) == 1
    assert rows_share[0]["error"] is None
    assert rows_full[0]["error"] is None
    assert rows_share[0]["lift"] is not None
    assert abs(float(rows_share[0]["lift"]) - float(rows_full[0]["lift"])) < 1e-10
    assert abs(
        float(rows_share[0]["candidate_rank_ic"]) - float(rows_full[0]["candidate_rank_ic"])
    ) < 1e-10
    assert abs(float(rows_share[0]["baseline"]) - float(rows_full[0]["baseline"])) < 1e-10


# ── G1 自适应 workers ──────────────────────────────────────────────────────


def test_adaptive_lift_workers_from_sysconf(monkeypatch):
    """可用内存 → workers = max(2, min(4, gb//5))。"""
    from factorzen.discovery import lift_test as lt

    # 23GB → 4；12GB → 2；4GB → 2（下限）；0 → 2；cap 100GB → 4
    cases = [
        (23 * 1024**3, 4),
        (12 * 1024**3, 2),
        (4 * 1024**3, 2),
        (0, 2),
        (100 * 1024**3, 4),  # cap
    ]
    page = 4096

    for avail_bytes, expected in cases:
        pages = avail_bytes // page

        def _sysconf(name, _pages=pages, _page=page):
            if name == "SC_AVPHYS_PAGES":
                return _pages
            if name == "SC_PAGE_SIZE":
                return _page
            raise ValueError(name)

        monkeypatch.setattr(lt.os, "sysconf", _sysconf)
        assert lt.adaptive_lift_workers() == expected


def test_adaptive_lift_workers_sysconf_error_fallback(monkeypatch):
    from factorzen.discovery import lift_test as lt

    def boom(_name):
        raise OSError("no sysconf")

    monkeypatch.setattr(lt.os, "sysconf", boom)
    assert lt.adaptive_lift_workers() == 2
    assert lt.resolve_lift_workers(None) == 2


def test_resolve_lift_workers_explicit_not_overridden(monkeypatch):
    from factorzen.discovery import lift_test as lt

    def _sysconf(name):
        # 假装只有 4GB → 自适应下限 2；显式 1 仍串行
        if name == "SC_AVPHYS_PAGES":
            return (4 * 1024**3) // 4096
        if name == "SC_PAGE_SIZE":
            return 4096
        raise ValueError(name)

    monkeypatch.setattr(lt.os, "sysconf", _sysconf)
    assert lt.adaptive_lift_workers() == 2
    assert lt.resolve_lift_workers(6) == 6
    assert lt.resolve_lift_workers(1) == 1
    assert lt.resolve_lift_workers(0) == 0


def test_run_lift_tests_default_workers_adaptive(monkeypatch):
    """lift_workers=None（默认）走自适应；低内存仍 ≥2 建池；显式 1 不建池。"""
    from factorzen.discovery import lift_test as lt

    def _sysconf(name):
        if name == "SC_AVPHYS_PAGES":
            return (8 * 1024**3) // 4096  # 8GB → max(2, 1)=2
        if name == "SC_PAGE_SIZE":
            return 4096
        raise ValueError(name)

    monkeypatch.setattr(lt.os, "sysconf", _sysconf)

    created = {"n": 0, "max_workers": None}
    real = lt.ThreadPoolExecutor

    class SpyPool:
        def __init__(self, *a, **k):
            created["n"] += 1
            created["max_workers"] = k.get("max_workers", a[0] if a else None)
            self._inner = real(*a, **k)

        def __enter__(self):
            return self._inner.__enter__()

        def __exit__(self, *a):
            return self._inner.__exit__(*a)

    monkeypatch.setattr(lt, "ThreadPoolExecutor", SpyPool)

    dates = _dates(30)
    codes = [f"{i:04d}.SZ" for i in range(6)]
    active = {"lib": _long_factor(dates, codes, np.random.default_rng(0))}
    cand = _long_factor(dates, codes, np.random.default_rng(1))
    ret = pl.DataFrame([
        {"trade_date": d, "ts_code": c, "ret": 0.01}
        for d in dates for c in codes
    ])

    def combine(fds, rdf, cv, **kw):
        return rdf.select(
            ["trade_date", "ts_code", pl.col("ret").alias("factor_value")]
        )

    common = dict(
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        combine_fn=combine,
        top_m=None,
    )
    rows = lt.run_lift_tests(
        [{"expression": "c0", "residual_ic_train": 0.01}],
        lift_workers=None,  # 显式 None = 自适应
        **common,
    )
    assert len(rows) == 1
    # 8GB → workers=2 → 建池
    assert created["n"] == 1
    assert created["max_workers"] == 2

    created["n"] = 0
    created["max_workers"] = None
    rows1 = lt.run_lift_tests(
        [{"expression": "c0", "residual_ic_train": 0.01}],
        lift_workers=1,  # 显式串行
        **common,
    )
    assert len(rows1) == 1
    assert created["n"] == 0


# ── H2 全零行：退化候选 / 安全名一致性 ──────────────────────────────────────


def test_degenerate_candidate_shared_path_is_error_not_zero_lift():
    """先红契约：物化后全 null 候选走共享路径 → error，禁止 lift=0 假结论。

    正常候选同批不受影响。
    """
    import factorzen.discovery.lift_test as lt
    from factorzen.research.combination.models import combine_lgbm as real_combine

    baseline, cand_ok, ret, _full = _synth_five_plus_one(
        n_days=80, n_stocks=16, seed=9, candidate_extra=False,
    )
    active = {"lib_a": baseline["b0"], "lib_b": baseline["b1"]}
    dates = _dates(80)
    codes = [f"{i:04d}.SZ" for i in range(16)]
    # 全 null 候选（与 drop_degenerate 同口径）
    cand_null = pl.DataFrame({
        "trade_date": [d for d in dates for _ in codes],
        "ts_code": codes * len(dates),
        "factor_value": [None] * (len(dates) * len(codes)),
    }).with_columns(pl.col("factor_value").cast(pl.Float64))

    def thin_combine(fds, rdf, cv, **kw):
        return real_combine(
            fds, rdf, cv,
            seed=0, n_estimators=20, min_child_samples=8, num_leaves=12,
            **{k: v for k, v in kw.items() if k == "base_panel"},
        )

    mats = {"null_cand": cand_null, "ok_cand": cand_ok}
    grays = [
        {"expression": "null_cand", "residual_ic_train": 0.03},
        {"expression": "ok_cand", "residual_ic_train": 0.02},
    ]
    daily = pl.DataFrame({"trade_date": [], "ts_code": [], "close": []})
    cv_params = {
        "train_days": 35, "test_days": 12, "purge_days": 3, "expanding": False,
    }

    import factorzen.research.combination.models as models_mod

    original = models_mod.combine_lgbm
    models_mod.combine_lgbm = thin_combine  # type: ignore[assignment]
    try:
        rows = lt.run_lift_tests(
            grays,
            market="ashare",
            daily=daily,
            active_factor_dfs=active,
            ret_df=ret,
            materialize_candidate=lambda e: mats[e],
            combine_fn=None,  # 生产共享路径
            lift_workers=1,
            seed=0,
            top_m=None,
            threshold=-1.0,
            cv_params=cv_params,
        )
    finally:
        models_mod.combine_lgbm = original  # type: ignore[assignment]

    assert len(rows) == 2
    by = {r["expression"]: r for r in rows}
    bad = by["null_cand"]
    good = by["ok_cand"]
    # 退化：显式 error，不是 lift=0 / se=0 的假结论
    assert bad["error"] == "degenerate_candidate"
    assert bad["lift"] is None
    assert bad["passed"] is False
    # 正常候选不受影响
    assert good["error"] is None
    assert good["lift"] is not None


def test_safe_pool_append_never_collides_with_base_keys():
    """集合一致性：追加安全名与 base_panel 列集不交；撞名表达式仍得新列。"""
    from factorzen.discovery.lift_test import (
        _safe_pool_with_new_factors,
        _with_safe_feature_names,
    )
    from factorzen.research.combination.models import _feature_names, build_panel

    baseline, cand, ret, _ = _synth_five_plus_one(
        n_days=40, n_stocks=8, seed=3, candidate_extra=False,
    )
    # 候选表达式已在 active 中（撞名）
    active = {"lib_a": baseline["b0"], "cand_x": baseline["b1"]}
    safe_active = _with_safe_feature_names(active)
    base = build_panel(safe_active, ret)
    base_feats = set(_feature_names(base))

    # 旧整表重映射：键集 ⊆ base → 静默不 join
    pool_collide = dict(active)
    pool_collide["cand_x"] = cand
    remapped = _with_safe_feature_names(pool_collide)
    assert set(remapped) <= base_feats

    # 修复：追加新键
    appended = _safe_pool_with_new_factors(safe_active, {"cand_x": cand})
    new_keys = set(appended) - base_feats
    assert new_keys == {f"f{len(safe_active):03d}"}
    assert set(safe_active).issubset(set(appended))
    assert set(safe_active) == base_feats


def test_combine_lgbm_rejects_all_degenerate_new_factors():
    """combine 层防御：意图新增的因子全被 drop → 显式 ValueError。"""
    import pytest

    from factorzen.research.combination.cv import PurgedWalkForwardCV
    from factorzen.research.combination.models import build_panel, combine_lgbm

    baseline, _cand, ret, _ = _synth_five_plus_one(
        n_days=50, n_stocks=10, seed=1, candidate_extra=False,
    )
    active = {"b0": baseline["b0"], "b1": baseline["b1"]}
    base = build_panel(active, ret)
    dates = _dates(50)
    codes = [f"{i:04d}.SZ" for i in range(10)]
    null_cand = pl.DataFrame({
        "trade_date": [d for d in dates for _ in codes],
        "ts_code": codes * len(dates),
        "factor_value": [None] * (len(dates) * len(codes)),
    }).with_columns(pl.col("factor_value").cast(pl.Float64))
    cv = PurgedWalkForwardCV(
        train_days=25, test_days=10, purge_days=2, expanding=False,
    )
    with pytest.raises(ValueError, match="degenerate_new_factors"):
        combine_lgbm(
            {"new_null": null_cand},
            ret,
            cv,
            base_panel=base,
            seed=0,
            n_estimators=10,
            min_child_samples=5,
        )


def test_paired_lift_stats_all_zero_diff_se_is_none():
    """diff 全零：lift=0 但 lift_se=None（不许当 SE=0 强结论）。"""
    from factorzen.discovery.lift_test import paired_lift_stats

    dates = [f"202401{d:02d}" for d in range(1, 41)]
    ics = [0.01 + 0.001 * (i % 5) for i in range(40)]
    daily = pl.DataFrame(
        {"trade_date": dates, "ic": ics},
        schema={"trade_date": pl.Utf8, "ic": pl.Float64},
    )
    stats = paired_lift_stats(daily, daily, block_days=10)
    assert stats["lift"] == 0.0
    assert stats["n_days"] == 40
    assert stats["lift_se"] is None
    assert stats["n_blocks"] == 4


def test_fold_test_dates_invariant_to_factor_dict_order():
    """判别测试：fold test 日期不得依赖因子 dict 插入序（旧实现取首因子,是潜伏 bug）。

    构造覆盖异质的两因子（f_wide 覆盖全部日期,f_narrow 只覆盖前半）,两种插入序的
    combine_lgbm 输出必须一致——旧 next(iter(...)) 实现下 narrow 在前会丢后半 test 行。
    """
    import datetime as dt

    import polars as pl

    from factorzen.research.combination.cv import PurgedWalkForwardCV
    from factorzen.research.combination.models import combine_lgbm

    days = []
    d = dt.date(2024, 1, 2)
    while len(days) < 90:
        if d.weekday() < 5:
            days.append(d.strftime("%Y%m%d"))
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(12)]

    def _panel(dates):
        return pl.DataFrame({
            "trade_date": [dd for dd in dates for _ in codes],
            "ts_code": codes * len(dates),
            "factor_value": [
                (hash((dd, c)) % 1000) / 1000.0 for dd in dates for c in codes
            ],
        })

    f_wide = _panel(days)
    f_narrow = _panel(days[: len(days) // 2])
    ret = pl.DataFrame({
        "trade_date": [dd for dd in days for _ in codes],
        "ts_code": codes * len(days),
        "ret": [(hash((c, dd)) % 200 - 100) / 5000.0 for dd in days for c in codes],
    })
    cv = PurgedWalkForwardCV(train_days=30, test_days=10, purge_days=2,
                             embargo_days=0, expanding=False)
    out_a = combine_lgbm({"w": f_wide, "n": f_narrow}, ret, cv, seed=7)
    out_b = combine_lgbm({"n": f_narrow, "w": f_wide}, ret, cv, seed=7)
    a = out_a.sort(["trade_date", "ts_code"])
    b = out_b.sort(["trade_date", "ts_code"])
    # 判别点=行集(fold test 日期):旧「取首因子」实现 narrow 在前会丢后半 test 行。
    # 不断言预测值:lgbm 对特征列序有平手裁决差异,值级不变性不成立(已知特性)。
    assert a.height == b.height and a.height > 0
    assert (a["trade_date"] == b["trade_date"]).all()
    assert (a["ts_code"] == b["ts_code"]).all()
    wide_only_dates = set(d for d in a["trade_date"].unique().to_list())
    # 后半日期(narrow 无覆盖)必须出现在 test 行里——旧实现会整段缺失
    assert any(d >= "20240401" for d in wide_only_dates)

"""LiftEvalContext 统一评估上下文 + admission 评分窗口。TDD、mock 离线。

residual_ic_v1：无 combine_fn；注入 active_factor_dfs + ret_df + materialize。
"""
from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl


def _dates(n_days: int, start: date | None = None):
    days, d = [], start or date(2024, 1, 2)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return days


def _iso(compact: str) -> str:
    """``YYYYMMDD`` → ``YYYY-MM-DD``：生产 scored_* 的形态。"""
    return f"{compact[0:4]}-{compact[4:6]}-{compact[6:8]}"


def _panel_from_values(dates, n_stocks, value_fn, *, col="factor_value"):
    """value_fn(date, stock_idx) → float。"""
    rows = []
    for d in dates:
        for s in range(n_stocks):
            rows.append({
                "trade_date": d,
                "ts_code": f"{s:04d}.SZ",
                col: float(value_fn(d, s)),
            })
    return pl.DataFrame(rows)


def _ret_by_stock_rank(dates, n_stocks):
    """ret = stock 序号 → 与 factor=s 完美正相关、与 factor=-s 完美负相关。"""
    return _panel_from_values(dates, n_stocks, lambda d, s: float(s), col="ret")


def _active_noise(dates, n_stocks, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "lib_a": _panel_from_values(
            dates, n_stocks, lambda d, s: float(rng.standard_normal()),
        ),
    }


# ── 1. 窗口裁剪改变结论 ──────────────────────────────────────────────────────


def test_admission_window_flips_lift_sign():
    """全窗 residual lift>0，admission 只看后半段 → lift<0；scored_* 落在窗内。"""
    from factorzen.discovery.lift_test import LiftEvalContext, run_lift_tests

    # n_stocks≥40：residual 日守卫 max(30, k+10)
    n_days, n_stocks = 50, 40
    dates = _dates(n_days)
    # 前 30 日强、后 20 日弱 → 全窗 lift>0；admission 从 mid_late 起 → lift<0
    mid_late = dates[30]
    active = _active_noise(dates, n_stocks)
    ret = _ret_by_stock_rank(dates, n_stocks)

    def cand_flip():
        rows = []
        for d in dates:
            for s in range(n_stocks):
                fv = float(s) if d < mid_late else -float(s)
                rows.append({
                    "trade_date": d, "ts_code": f"{s:04d}.SZ", "factor_value": fv,
                })
        return pl.DataFrame(rows)

    cand = cand_flip()

    common = dict(
        gray_candidates=[{"expression": "flip_cand", "residual_ic_train": 0.006}],
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        threshold=0.001,
        block_days=10,
        top_m=10,
        lift_workers=1,
    )

    full = run_lift_tests(**common)
    assert full[0]["error"] is None, full[0]
    assert full[0]["lift"] is not None and full[0]["lift"] > 0, full[0]

    ctx = LiftEvalContext(
        market="ashare",
        prepped=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),
        leaf_map=None,
        horizon=5,
        admission_start=mid_late,
        admission_end=None,
    )
    windowed = run_lift_tests(**common, ctx=ctx)
    assert windowed[0]["error"] is None, windowed[0]
    assert windowed[0]["lift"] is not None and windowed[0]["lift"] < 0, windowed[0]
    assert full[0]["passed"] is True or full[0]["lift"] > 0
    assert windowed[0]["lift"] < 0

    assert windowed[0]["admission_start"] == mid_late
    assert windowed[0]["admission_end"] is None
    assert windowed[0]["scored_start"] is not None
    assert windowed[0]["scored_end"] is not None
    assert windowed[0]["scored_start"] >= _iso(mid_late)
    assert windowed[0]["scored_end"] >= windowed[0]["scored_start"]
    assert windowed[0]["horizon"] == 5
    assert windowed[0]["baseline"] is None
    assert windowed[0].get("lift_metric") == "residual_ic_v1"


# ── 2. 对称性：pool 与 materializer 共用同一 prepped ─────────────────────────


def test_make_lift_context_shared_prepped_for_pool_and_materializer(monkeypatch):
    """make_lift_context prep 一次；pool 与 materializer 收到同一 prepped 对象。"""
    from factorzen.discovery import lift_test as lt
    from factorzen.discovery.lift_test import make_lift_context, run_lift_tests

    class _Factors:
        def derived_columns(self, df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(pl.lit(42.0).alias("probe_derived"))

    profile = SimpleNamespace(name="mock_mkt", factors=_Factors())

    daily = pl.DataFrame({
        "trade_date": ["20240102", "20240103"],
        "ts_code": ["000001.SZ", "000001.SZ"],
        "close": [10.0, 10.5],
        "open": [9.5, 10.0],
        "high": [10.2, 10.6],
        "low": [9.4, 9.9],
        "vol": [1e6, 1.1e6],
        "amount": [1e7, 1.1e7],
    })

    ctx = make_lift_context(
        "mock_mkt", daily, profile=profile, leaf_map={"close": "close"},
        horizon=3, admission_start="20240103",
    )
    assert "probe_derived" in ctx.prepped.columns
    assert ctx.profile_name == "mock_mkt"
    assert ctx.horizon == 3
    assert ctx.admission_start == "20240103"
    prepped_id = id(ctx.prepped)

    captured: dict = {}

    def fake_pool(market, daily_df, leaf_map, **kw):
        captured["pool_id"] = id(daily_df)
        captured["pool_has_probe"] = "probe_derived" in daily_df.columns
        # 非空 active 让 run 继续；返回极简面板
        return {
            "lib_a": pl.DataFrame({
                "trade_date": ["20240102", "20240103"],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "factor_value": [0.1, 0.2],
            }),
        }

    def spy_mat_from_prepped(prepped, leaf_map, **_kw):
        captured["mat_id"] = id(prepped)
        captured["mat_has_probe"] = "probe_derived" in prepped.columns

        def _mat(expr: str):
            return pl.DataFrame({
                "trade_date": ["20240102", "20240103"],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "factor_value": [0.3, 0.4],
            })

        return _mat

    monkeypatch.setattr(
        "factorzen.discovery.factor_library.build_library_pool", fake_pool,
    )
    monkeypatch.setattr(lt, "_materializer_from_prepped", spy_mat_from_prepped)

    monkeypatch.setattr(
        lt, "_build_ret_panel",
        # **_kw 接住 exec_lag/exec_price_col 等后加 kwargs——桩签名写死
        # 会在真实签名扩展时假报错（本会话已栽过两次）
        lambda daily_df, *, horizon=5, **_kw: pl.DataFrame({
            "trade_date": ["20240102", "20240103"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "ret": [0.01, -0.01],
        }),
    )

    # 短面板会 no_residual_days；本测只验证 prepped 共享，不关心 lift 值
    run_lift_tests(
        [{"expression": "rank(close)", "residual_ic_train": 0.006}],
        market="mock_mkt",
        daily=daily,
        ctx=ctx,
        top_m=1,
        lift_workers=1,
    )

    assert captured.get("pool_id") == prepped_id
    assert captured.get("mat_id") == prepped_id
    assert captured.get("pool_has_probe") is True
    assert captured.get("mat_has_probe") is True


# ── 3. 显式注入优先于 ctx ───────────────────────────────────────────────────


def test_explicit_injection_overrides_ctx(monkeypatch):
    """ctx 与显式 active/ret/materialize 同时给 → 用注入的。"""
    from factorzen.discovery import lift_test as lt
    from factorzen.discovery.lift_test import LiftEvalContext, run_lift_tests

    dates = _dates(40)
    n_stocks = 40
    active = _active_noise(dates, n_stocks, seed=1)
    ret = _ret_by_stock_rank(dates, n_stocks)
    cand = _panel_from_values(dates, n_stocks, lambda d, s: float(s))

    ctx = LiftEvalContext(
        market="should_not_use",
        prepped=pl.DataFrame({"trade_date": ["x"], "ts_code": ["y"], "close": [1.0]}),
        leaf_map=None,
        horizon=99,
        admission_start=None,
        admission_end=None,
        library_root="/should/not/touch",
    )

    pool_called = {"n": 0}
    mat_from_called = {"n": 0}
    ret_build_called = {"n": 0}

    monkeypatch.setattr(
        "factorzen.discovery.factor_library.build_library_pool",
        lambda *a, **k: pool_called.__setitem__("n", pool_called["n"] + 1) or {},
    )
    monkeypatch.setattr(
        lt, "_materializer_from_prepped",
        lambda *a, **k: (
            mat_from_called.__setitem__("n", mat_from_called["n"] + 1)
            or (lambda e: cand)
        ),
    )
    monkeypatch.setattr(
        lt, "_build_ret_panel",
        lambda *a, **k: (
            ret_build_called.__setitem__("n", ret_build_called["n"] + 1)
            or ret
        ),
    )

    injected_mat = {"n": 0}

    def mat(expr):
        injected_mat["n"] += 1
        return cand

    rows = run_lift_tests(
        [{"expression": "c0", "residual_ic_train": 0.01}],
        market="ashare",
        daily=pl.DataFrame(),
        ctx=ctx,
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=mat,
        horizon=5,  # 显式覆盖 ctx.horizon=99
        lift_workers=1,
    )

    assert pool_called["n"] == 0
    assert mat_from_called["n"] == 0
    assert ret_build_called["n"] == 0
    assert injected_mat["n"] == 1
    assert rows[0]["horizon"] == 5  # 显式优先
    assert rows[0]["error"] is None or rows[0]["lift"] is not None


# ── 4. 零回归：ctx=None ─────────────────────────────────────────────────────


def test_ctx_none_zero_regression_same_inputs():
    """ctx=None / 不传 ctx 同一 mock 输入结果一致。"""
    from factorzen.discovery.lift_test import run_lift_tests

    dates = _dates(50)
    n_stocks = 40
    active = _active_noise(dates, n_stocks, seed=2)
    ret = _ret_by_stock_rank(dates, n_stocks)
    cand = _panel_from_values(dates, n_stocks, lambda d, s: float(s) + 0.01 * hash(d) % 7)

    kwargs = dict(
        gray_candidates=[
            {"expression": "c0", "residual_ic_train": 0.008},
            {"expression": "c1", "residual_ic_train": 0.007},
        ],
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        top_m=10,
        threshold=0.001,
        block_days=10,
        seed=0,
        lift_workers=1,
    )

    a = run_lift_tests(**kwargs)
    b = run_lift_tests(**kwargs, ctx=None)
    assert len(a) == len(b) == 2
    for ra, rb in zip(a, b, strict=True):
        # elapsed_s 是墙钟遥测,两次调用必然不同;零回归只比结果字段
        assert {k: v for k, v in ra.items() if k != "elapsed_s"} == {
            k: v for k, v in rb.items() if k != "elapsed_s"
        }
    for r in a:
        assert r["admission_start"] is None
        assert r["admission_end"] is None
        assert r["horizon"] == 5
        assert "scored_start" in r and "scored_end" in r
        assert r.get("lift_metric") == "residual_ic_v1"


# ── 5. horizon 透传 ──────────────────────────────────────────────────────────


def test_ctx_horizon_passed_to_build_ret_panel(monkeypatch):
    """ctx.horizon=1 时 _build_ret_panel 收到 horizon=1，结果行 horizon==1。"""
    from factorzen.discovery import lift_test as lt
    from factorzen.discovery.lift_test import LiftEvalContext, run_lift_tests

    dates = _dates(40)
    n_stocks = 40
    active = _active_noise(dates, n_stocks, seed=3)
    ret = _ret_by_stock_rank(dates, n_stocks)
    cand = _panel_from_values(dates, n_stocks, lambda d, s: float(s))

    seen = {}

    def spy_ret(daily_df, *, horizon=5, exec_lag=0, exec_price_col=None):
        # 默认口径必须原样传下来（exec_lag=0 = 历史行为）
        seen["exec_lag"] = exec_lag
        seen["exec_price_col"] = exec_price_col
        seen["horizon"] = horizon
        return ret

    monkeypatch.setattr(lt, "_build_ret_panel", spy_ret)

    ctx = LiftEvalContext(
        market="ashare",
        prepped=pl.DataFrame({"trade_date": ["20240102"], "ts_code": ["x"], "close": [1.0]}),
        leaf_map=None,
        horizon=1,
        admission_start=None,
        admission_end=None,
    )

    rows = run_lift_tests(
        [{"expression": "c0", "residual_ic_train": 0.01}],
        market="ashare",
        daily=pl.DataFrame(),
        ctx=ctx,
        active_factor_dfs=active,
        # 不注入 ret_df → 走 _build_ret_panel
        materialize_candidate=lambda e: cand,
        lift_workers=1,
        # 不传 horizon → 从 ctx 派生
    )

    assert seen.get("horizon") == 1
    assert rows[0]["horizon"] == 1
    # ctx 未指定成交口径 ⇒ 必须原样传下默认值（exec_lag=0 = 历史 close→close，
    # 见 compute_fwd_returns docstring）。若这里变成 1，等于默认行为被悄悄改了。
    assert seen.get("exec_lag") == 0
    assert seen.get("exec_price_col") is None


def test_group_lift_admission_window_and_provenance():
    """run_group_lift 透传 admission 窗并写 provenance 字段。"""
    from factorzen.discovery.lift_test import LiftEvalContext, run_group_lift

    n_days, n_stocks = 50, 40
    dates = _dates(n_days)
    mid_late = dates[30]
    active = _active_noise(dates, n_stocks, seed=4)
    ret = _ret_by_stock_rank(dates, n_stocks)

    def cand_flip():
        rows = []
        for d in dates:
            for s in range(n_stocks):
                fv = float(s) if d < mid_late else -float(s)
                rows.append({
                    "trade_date": d, "ts_code": f"{s:04d}.SZ", "factor_value": fv,
                })
        return pl.DataFrame(rows)

    cand = cand_flip()

    ctx = LiftEvalContext(
        market="ashare",
        prepped=pl.DataFrame(),
        leaf_map=None,
        horizon=5,
        admission_start=mid_late,
        admission_end=None,
    )

    out = run_group_lift(
        [{"expression": "g1", "residual_ic_train": 0.006}],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        ctx=ctx,
        threshold=0.001,
    )
    assert out["error"] is None, out
    assert out["lift"] is not None and out["lift"] < 0
    assert out["admission_start"] == mid_late
    assert out["scored_start"] is not None and out["scored_start"] >= _iso(mid_late)
    assert out["horizon"] == 5
    assert out["baseline"] is None
    assert "base_daily" not in out
    assert out.get("lift_metric") == "residual_ic_v1"

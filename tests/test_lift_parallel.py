"""任务 F：lift 批量裁决提速——候选并行 / DEFAULT_LIFT_CV / base_daily / CLI 透传。

全部 mock 离线；不碰真实数据或 lgbm 训练。
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from tests._cli_lift_mocks import patch_cli_lift_pre_gates


def _dates(n_days: int):
    days, d = [], date(2024, 1, 2)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return days


def _tiny_panels(n_days: int = 40, n_stocks: int = 8):
    """合成 active / 候选 / ret 面板（combine_fn mock 不读数值）。"""
    dates = _dates(n_days)
    codes = [f"{i:04d}.SZ" for i in range(n_stocks)]
    lib_rows, cand_rows, ret_rows = [], [], []
    for d in dates:
        for s, code in enumerate(codes):
            lib_rows.append({"trade_date": d, "ts_code": code, "factor_value": float(s)})
            cand_rows.append({"trade_date": d, "ts_code": code, "factor_value": float(s + 1)})
            ret_rows.append({"trade_date": d, "ts_code": code, "ret": 0.01 * (s + 1)})
    active = {"lib_a": pl.DataFrame(lib_rows)}
    cand = pl.DataFrame(cand_rows)
    ret = pl.DataFrame(ret_rows)
    return active, cand, ret


def _det_combine_factory(active: dict, ret: pl.DataFrame):
    """确定性 combine：基线弱噪声、加候选后用 ret 作预测 → 稳定正 lift。"""

    def combine_fn(fds, rdf, cv, **kw):
        n = len(fds)
        if n <= len(active):
            # 基线：常数预测（截面无区分 → IC 弱/空，paired 仍可算）
            return ret.select(["trade_date", "ts_code"]).with_columns(
                pl.lit(0.0).alias("factor_value")
            )
        # 候选池：用 ret 本身 → 近完美 IC
        return rdf.select(
            ["trade_date", "ts_code", pl.col("ret").alias("factor_value")]
        )

    return combine_fn


def test_parallel_vs_serial_row_identical():
    """workers=4 与 workers=1 结果逐行一致（确定性 combine_fn mock）。"""
    from factorzen.discovery.lift_test import run_lift_tests

    active, cand, ret = _tiny_panels()
    combine_fn = _det_combine_factory(active, ret)
    grays = [
        {"expression": f"c{i}", "residual_ic_train": 0.009 - i * 0.0001}
        for i in range(6)
    ]
    mats = {f"c{i}": cand for i in range(6)}
    common = dict(
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: mats[e],
        combine_fn=combine_fn,
        top_m=None,
        threshold=0.001,
        block_days=10,
        seed=0,
    )
    serial = run_lift_tests(grays, lift_workers=1, **common)
    parallel = run_lift_tests(grays, lift_workers=4, **common)
    assert len(serial) == len(parallel) == 6
    for a, b in zip(serial, parallel, strict=True):
        assert a["expression"] == b["expression"]
        assert a["lift"] == b["lift"]
        assert a["lift_se"] == b["lift_se"]
        assert a["baseline"] == b["baseline"]
        assert a["candidate_rank_ic"] == b["candidate_rank_ic"]
        assert a["passed"] == b["passed"]
        assert a["error"] == b["error"]
        assert a["n_blocks"] == b["n_blocks"]
        assert a["lift_first_half"] == b["lift_first_half"]
        assert a["lift_second_half"] == b["lift_second_half"]


def test_workers_one_skips_thread_pool(monkeypatch):
    """workers=1 时不实例化 ThreadPoolExecutor（同 _llm_map 零回归约定）。"""
    import factorzen.discovery.lift_test as lt

    created = {"n": 0}
    real = lt.ThreadPoolExecutor

    class SpyPool:
        def __init__(self, *a, **k):
            created["n"] += 1
            raise AssertionError("workers=1 不应创建 ThreadPoolExecutor")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(lt, "ThreadPoolExecutor", SpyPool)

    active, cand, ret = _tiny_panels()
    grays = [{"expression": "c0", "residual_ic_train": 0.01}]
    rows = lt.run_lift_tests(
        grays,
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        combine_fn=_det_combine_factory(active, ret),
        lift_workers=1,
        top_m=None,
    )
    assert created["n"] == 0
    assert len(rows) == 1
    assert rows[0]["expression"] == "c0"

    # 对照：workers>1 应建池（恢复真类后）
    monkeypatch.setattr(lt, "ThreadPoolExecutor", real)
    # 仅断言并行路径可跑通且结果有行
    rows4 = lt.run_lift_tests(
        grays,
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        combine_fn=_det_combine_factory(active, ret),
        lift_workers=4,
        top_m=None,
    )
    assert len(rows4) == 1


def test_default_lift_cv_shared_by_run_lift_and_group(monkeypatch):
    """DEFAULT_LIFT_CV 被 run_lift_tests / run_group_lift 共用（改常量两处同变）。"""
    import factorzen.discovery.lift_test as lt
    from factorzen.research.combination.cv import PurgedWalkForwardCV

    captured: list[dict] = []

    class CapturingCV(PurgedWalkForwardCV):
        def __init__(self, **kw):
            captured.append(dict(kw))
            super().__init__(**kw)

    monkeypatch.setattr(
        "factorzen.research.combination.cv.PurgedWalkForwardCV", CapturingCV,
    )
    # 改常量后两入口应同变
    custom = {
        "train_days": 99,
        "test_days": 11,
        "purge_days": 2,
        "embargo_days": 1,
        "expanding": False,
    }
    monkeypatch.setattr(lt, "DEFAULT_LIFT_CV", custom)

    active, cand, ret = _tiny_panels()
    combine = _det_combine_factory(active, ret)
    daily = pl.DataFrame({"trade_date": [], "ts_code": [], "close": []})
    grays = [{"expression": "c0", "residual_ic_train": 0.01}]

    lt.run_lift_tests(
        grays,
        market="ashare",
        daily=daily,
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        combine_fn=combine,
        lift_workers=1,
    )
    lt.run_group_lift(
        grays,
        market="ashare",
        daily=daily,
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        combine_fn=combine,
    )
    assert len(captured) >= 2
    for kw in captured:
        assert kw["train_days"] == 99
        assert kw["test_days"] == 11
        assert kw["purge_days"] == 2
        assert kw["embargo_days"] == 1
        assert kw["expanding"] is False


def test_base_daily_injection_skips_baseline_combine():
    """base_daily 注入时基线 combine 不被调用（mock 计数）。"""
    from factorzen.discovery.lift_test import run_group_lift, run_lift_tests

    active, cand, ret = _tiny_panels()
    call_n = {"n": 0}

    def counting_combine(fds, rdf, cv, **kw):
        call_n["n"] += 1
        n = len(fds)
        if n <= len(active):
            return ret.select(["trade_date", "ts_code"]).with_columns(
                pl.lit(0.0).alias("factor_value")
            )
        return rdf.select(
            ["trade_date", "ts_code", pl.col("ret").alias("factor_value")]
        )

    daily = pl.DataFrame({"trade_date": [], "ts_code": [], "close": []})
    grays = [
        {"expression": "c0", "residual_ic_train": 0.01},
        {"expression": "c1", "residual_ic_train": 0.009},
    ]
    mats = {"c0": cand, "c1": cand}

    # 无注入：基线 1 + 2 候选 = 3
    call_n["n"] = 0
    run_lift_tests(
        grays,
        market="ashare",
        daily=daily,
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: mats[e],
        combine_fn=counting_combine,
        lift_workers=1,
    )
    assert call_n["n"] == 3

    # 预造 base_daily
    dates = ret["trade_date"].unique().sort().to_list()
    base_daily = pl.DataFrame({
        "trade_date": dates,
        "ic": [0.01] * len(dates),
    })
    call_n["n"] = 0
    rows = run_lift_tests(
        grays,
        market="ashare",
        daily=daily,
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: mats[e],
        combine_fn=counting_combine,
        lift_workers=1,
        base_daily=base_daily,
    )
    # 仅 2 候选 combine，无基线
    assert call_n["n"] == 2, f"期望 2 次 combine，实际 {call_n['n']}"
    for r in rows:
        assert r["baseline"] is not None
        assert abs(float(r["baseline"]) - 0.01) < 1e-12

    # run_group_lift 注入同样跳过基线
    call_n["n"] = 0
    out = run_group_lift(
        grays,
        market="ashare",
        daily=daily,
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: mats[e],
        combine_fn=counting_combine,
        base_daily=base_daily,
    )
    assert call_n["n"] == 1  # 仅组 combine
    assert out["error"] is None
    assert out["baseline"] is not None
    assert abs(float(out["baseline"]) - 0.01) < 1e-12
    assert out["base_daily"] is not None


def test_run_group_lift_returns_base_daily():
    """成功路径返回 base_daily 供 session 钩子复用。"""
    from factorzen.discovery.lift_test import run_group_lift

    active, cand, ret = _tiny_panels()
    out = run_group_lift(
        [{"expression": "c0", "residual_ic_train": 0.01}],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        combine_fn=_det_combine_factory(active, ret),
    )
    assert out["error"] is None
    assert out["base_daily"] is not None
    assert "trade_date" in out["base_daily"].columns
    assert "ic" in out["base_daily"].columns


def test_cli_lift_workers_from_outer_parser(tmp_path, monkeypatch):
    """CLI --lift-workers 从 parser 最外层透传到 run_lift_tests。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.lift_test as lt_mod
    from factorzen.cli.main import build_parser

    run_dir = tmp_path / "sess"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "attempts": [{
                "expression": "rank(close)",
                "reject_category": "lift_queue",
                "residual_ic_train": 0.01,
            }],
            "candidates": [],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda args: (
            pl.DataFrame({
                "trade_date": [date(2024, 1, 2)],
                "ts_code": ["000001.SZ"],
                "close": [10.0],
            }),
            None,
            {},
        ),
    )
    seen: dict = {}

    def fake_lift(gray, **kw):
        seen["lift_workers"] = kw.get("lift_workers")
        return [{
            "expression": "rank(close)",
            "lift": None,
            "baseline": None,
            "passed": False,
        }]

    patch_cli_lift_pre_gates(monkeypatch)
    monkeypatch.setattr(lt_mod, "run_lift_tests", fake_lift)

    args = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--start", "20240102",
        "--end", "20240301",
        "--library-root", str(tmp_path / "lib"),
        "--dry-run",
        "--lift-workers", "8",
    ])
    assert args.lift_workers == 8
    assert args.func.__name__ == "_cmd_factor_library_lift_test"
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert seen["lift_workers"] == 8

    # 默认 4
    args_def = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--start", "20240102",
        "--end", "20240301",
        "--library-root", str(tmp_path / "lib"),
        "--dry-run",
    ])
    assert args_def.lift_workers is None  # None→run_lift_tests 自适应(按可用内存)


def test_session_hook_reuses_group_base_daily(monkeypatch):
    """组门返回 base_daily 时，run_lift_tests 收到同一序列（省基线 combine）。"""
    import datetime as dt

    import numpy as np

    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.agents.team_orchestrator import _session_end_auto_lift
    from factorzen.discovery.guardrails import REJECT_CATEGORY_LIFT_QUEUE

    def _mock_daily(n_stocks=40, n_days=180, seed=1):
        rng = np.random.default_rng(seed)
        days, d = [], dt.date(2022, 1, 3)
        while len(days) < n_days:
            if d.weekday() < 5:
                days.append(d)
            d += dt.timedelta(days=1)
        codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
        rows = []
        for c in codes:
            px = 10.0
            for dd in days:
                px *= 1 + rng.standard_normal() * 0.02
                rows.append({
                    "trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                    "high": px * 1.01, "low": px * 0.98,
                    "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                    "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
                })
        return pl.DataFrame(rows)

    def _panel(n_dates: int, n_stocks: int = 10, start=dt.date(2022, 1, 3)):
        days, d = [], start
        while len(days) < n_dates:
            if d.weekday() < 5:
                days.append(d)
            d += dt.timedelta(days=1)
        codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
        rows = []
        for c in codes:
            for dd in days:
                rows.append({
                    "trade_date": dd, "ts_code": c,
                    "factor_value": float(hash((c, dd)) % 100) / 100.0,
                })
        return pl.DataFrame(rows)

    dates = _dates(30)
    base_daily = pl.DataFrame({
        "trade_date": dates,
        "ic": [0.02] * len(dates),
    })
    seen: dict = {}

    def fake_group(*a, **k):
        return {
            "lift": 0.05,
            "lift_se": 0.001,
            "error": None,
            "n_candidates": 1,
            "expressions": ["ts_mean(close, 5)"],
            "base_daily": base_daily,
        }

    def fake_per(*a, **k):
        seen["base_daily"] = k.get("base_daily")
        seen["lift_workers"] = k.get("lift_workers")
        return [{
            "expression": "ts_mean(close, 5)",
            "lift": 0.01,
            "lift_se": 0.001,
            "lift_second_half": 0.01,
            "passed": True,
            "baseline": 0.02,
        }]

    monkeypatch.setattr("factorzen.discovery.lift_test.run_group_lift", fake_group)
    monkeypatch.setattr("factorzen.discovery.lift_test.run_lift_tests", fake_per)
    monkeypatch.setattr(
        "factorzen.discovery.factor_library.upsert_lift_admissions",
        lambda *a, **k: {
            "added_active": 0, "added_probation": 1, "rejected": 0, "errors": [],
        },
        raising=False,
    )

    state = AgentState(seed=1)
    state.attempts.append(AttemptRecord(
        iteration=0, hypothesis="h", expression="ts_mean(close, 5)",
        compile_ok=True, ic_train=0.02, passed_guardrails=False,
        critic_verdict=None, error=None, ir_train=1.0, n_train=100,
        residual_ic_train=0.01,
        reject_category=REJECT_CATEGORY_LIFT_QUEUE,
        reject_reason="x(lift队列)",
    ))
    state.n_gray_zone = 1

    daily = _mock_daily(n_days=120)
    cut = daily["trade_date"].unique().sort()[int(daily["trade_date"].n_unique() * 0.8)]
    holdout = daily.filter(pl.col("trade_date") >= cut)

    def mat(expr):
        return _panel(80, start=cut)

    class _FakeCtx:
        leaf_map = None

    meta = _session_end_auto_lift(
        state,
        daily=daily,
        holdout_df=holdout,
        profile=None,
        ctx=_FakeCtx(),
        market="ashare",
        library_root=str(Path("/tmp/lib_lift_par")),
        seed=1,
        auto_lift=True,
        lift_se_mult=1.0,
        lift_workers=3,
        horizon=1,
        materialize_candidate=mat,
        active_factor_dfs={"base": _panel(100)},
        ret_df=_panel(100).rename({"factor_value": "ret"}),
    )
    assert seen.get("base_daily") is base_daily
    assert seen.get("lift_workers") == 3
    # manifest 视图不得含 base_daily（JSON 不可序列化）
    assert "base_daily" not in (meta.get("lift_group") or {})
    assert meta.get("lift_results")

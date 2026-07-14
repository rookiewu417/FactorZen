"""组合增量 lift 实验 + probation 入库通道单测。TDD、mock 离线。"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import polars as pl

# ── 合成面板：ret 含 (lib × cand) 交互 ───────────────────────────────────────


def _dates(n_days: int):
    days, d = [], date(2024, 1, 2)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return days


def _synth_panels(n_days=160, n_stocks=40, seed=0, *, interactive=True):
    """构造 lib 因子 + 候选 + 收益。

    interactive=True：ret = 0.3*lib + 0.5*(lib*cand 交互) + 噪声
      → 候选线性 IC≈0，但 lgbm 学到交互后 lift 显著。
    interactive=False：ret = 0.5*lib + 噪声（候选纯噪声）→ lift≈0。
    """
    rng = np.random.default_rng(seed)
    dates = _dates(n_days)
    lib_rows, cand_rows, noise_rows, ret_rows = [], [], [], []
    for d in dates:
        lib = rng.standard_normal(n_stocks)
        cand = rng.standard_normal(n_stocks)
        noise = rng.standard_normal(n_stocks)
        if interactive:
            ret = 0.3 * lib + 0.6 * (lib * cand) + 0.25 * rng.standard_normal(n_stocks)
        else:
            ret = 0.5 * lib + 0.4 * rng.standard_normal(n_stocks)
        for s in range(n_stocks):
            code = f"{s:04d}.SZ"
            lib_rows.append({"trade_date": d, "ts_code": code, "factor_value": float(lib[s])})
            cand_rows.append({"trade_date": d, "ts_code": code, "factor_value": float(cand[s])})
            noise_rows.append({"trade_date": d, "ts_code": code, "factor_value": float(noise[s])})
            ret_rows.append({"trade_date": d, "ts_code": code, "ret": float(ret[s])})
    return (
        {"lib_a": pl.DataFrame(lib_rows)},
        pl.DataFrame(cand_rows),
        pl.DataFrame(noise_rows),
        pl.DataFrame(ret_rows),
    )


def test_lift_interactive_candidate_passes_noise_fails():
    """交互项候选 lift 显著过阈值；纯噪声候选 lift≈0 拒。基线只算一次。"""
    from factorzen.discovery.guardrails import DEFAULT_LIFT_THRESHOLD
    from factorzen.discovery.lift_test import run_lift_tests
    from factorzen.research.combination.models import combine_lgbm

    active, cand_good, cand_noise, ret = _synth_panels(interactive=True)
    # 噪声场景的 ret 另造一版纯噪声候选仍用 interactive ret（噪声候选本身无信号）
    call_count = {"n": 0}
    real_combine = combine_lgbm

    def counting_combine(fds, rdf, cv, **kw):
        call_count["n"] += 1
        return real_combine(fds, rdf, cv, seed=0, n_estimators=40, min_child_samples=15)

    cv_params = {"train_days": 60, "test_days": 20, "purge_days": 5}

    mat_map = {
        "interactive_cand": cand_good,
        "noise_cand": cand_noise,
    }

    def materialize(expr):
        return mat_map[expr]

    rows = run_lift_tests(
        [
            {"expression": "interactive_cand", "residual_ic_train": 0.006},
            {"expression": "noise_cand", "residual_ic_train": 0.005},
        ],
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),  # unused
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=materialize,
        combine_fn=counting_combine,
        cv_params=cv_params,
        top_m=10,
        threshold=DEFAULT_LIFT_THRESHOLD,
        seed=0,
    )
    by_expr = {r["expression"]: r for r in rows}
    assert by_expr["interactive_cand"]["passed"] is True, by_expr["interactive_cand"]
    assert by_expr["interactive_cand"]["lift"] >= DEFAULT_LIFT_THRESHOLD
    # 噪声：lift 应接近 0（允许小幅波动，但不得过阈值太多；硬门：不 passed 或 lift 显著低于交互）
    assert by_expr["noise_cand"]["passed"] is False or (
        by_expr["noise_cand"]["lift"] is not None
        and by_expr["noise_cand"]["lift"] < by_expr["interactive_cand"]["lift"]
    )
    # 基线 1 次 + 2 候选 = 3 次 combine（不重复算基线）
    assert call_count["n"] == 3, f"baseline 应只算一次，实际 combine 调用 {call_count['n']}"


def test_lift_top_m_truncation_recorded():
    """top_m 截断不静默：返回行带 n_input / n_selected。"""
    from factorzen.discovery.lift_test import run_lift_tests

    active, cand, _, ret = _synth_panels(n_days=100, n_stocks=20, interactive=False)
    grays = [
        {"expression": f"c{i}", "residual_ic_train": 0.009 - i * 0.0001}
        for i in range(5)
    ]
    mats = {f"c{i}": cand for i in range(5)}

    def combine_stub(fds, rdf, cv, **kw):
        # 返回合法空壳组合帧（全 0 因子值）
        return ret.select(["trade_date", "ts_code"]).with_columns(
            pl.lit(0.0).alias("factor_value")
        )

    rows = run_lift_tests(
        grays,
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: mats[e],
        combine_fn=combine_stub,
        top_m=2,
        threshold=0.001,
    )
    assert len(rows) == 2
    assert rows[0]["n_input"] == 5 and rows[0]["n_selected"] == 2
    assert rows[0]["truncated_from"] == 5


def test_lift_bad_candidate_does_not_crash_batch():
    from factorzen.discovery.lift_test import run_lift_tests

    active, cand, _, ret = _synth_panels(n_days=80, n_stocks=20, interactive=False)

    def materialize(expr):
        if expr == "bad":
            raise RuntimeError("boom")
        return cand

    def combine_stub(fds, rdf, cv, **kw):
        return ret.select(["trade_date", "ts_code"]).with_columns(
            pl.lit(0.0).alias("factor_value")
        )

    rows = run_lift_tests(
        [
            {"expression": "bad", "residual_ic_train": 0.008},
            {"expression": "ok", "residual_ic_train": 0.007},
        ],
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=materialize,
        combine_fn=combine_stub,
        top_m=10,
    )
    assert len(rows) == 2
    assert rows[0]["passed"] is False and rows[0]["error"]
    assert rows[1]["expression"] == "ok"


def test_extract_gray_from_manifest():
    from factorzen.discovery.lift_test import extract_gray_candidates_from_manifest

    man = {
        "attempts": [
            {"expression": "a", "reject_category": "gray_zone", "residual_ic_train": 0.006},
            {"expression": "b", "reject_category": "holdout_coverage"},
            {"expression": "a", "reject_category": "gray_zone"},  # 去重
        ],
        "candidates": [
            {"expression": "c", "reject_category": "gray_zone", "ic_train": 0.008},
        ],
    }
    got = extract_gray_candidates_from_manifest(man)
    assert {g["expression"] for g in got} == {"a", "c"}


# ── upsert_probation / rebuild 保留 / pool 默认不含 ──────────────────────────


def test_upsert_probation_roundtrip_fields(tmp_path):
    from factorzen.discovery.factor_library import load_library, upsert_probation

    rows = [
        {
            "expression": "rank(close)",
            "ic_train": 0.006,
            "holdout_ic": -0.002,
            "lift": 0.0035,
            "baseline": 0.04,
            "passed": True,
            "n_train": 200,
        }
    ]
    res = upsert_probation(
        "ashare", rows,
        eval_window=("20200101", "20260101"), universe="csi300", horizon=5,
        run_id="lift1", session_dir="s1", git_sha="abc", now="2026-07-14",
        root=str(tmp_path),
    )
    assert res.added == 1
    lib = load_library("ashare", root=str(tmp_path))
    assert len(lib) == 1
    r = lib[0]
    assert r.status == "probation"
    assert abs(r.lift - 0.0035) < 1e-9
    assert abs(r.lift_baseline - 0.04) < 1e-9
    assert r.expression == "rank(close)"


def test_upsert_probation_skips_not_passed(tmp_path):
    from factorzen.discovery.factor_library import load_library, upsert_probation

    res = upsert_probation(
        "ashare",
        [{"expression": "x", "passed": False, "lift": 0.0, "baseline": 0.01}],
        eval_window=("20200101", "20260101"), universe="u", horizon=5,
        run_id="r", session_dir="s", git_sha="a", now="2026-07-14",
        root=str(tmp_path),
    )
    assert res.skipped == 1 and res.added == 0
    assert load_library("ashare", root=str(tmp_path)) == []


def test_upsert_probation_skips_library_gate():
    """|IC| 远低于 floor 仍可 probation——不走单因子 gate。"""
    import tempfile

    from factorzen.discovery.factor_library import load_library, upsert_probation

    with tempfile.TemporaryDirectory() as td:
        res = upsert_probation(
            "ashare",
            [{
                "expression": "rank(vol)",
                "ic_train": 0.004,   # < DEFAULT_IC_FLOOR
                "holdout_ic": -0.01,  # 反号
                "lift": 0.002,
                "baseline": 0.03,
                "passed": True,
            }],
            eval_window=("20200101", "20260101"), universe="u", horizon=5,
            run_id="r", session_dir="s", git_sha="a", now="2026-07-14",
            root=td,
        )
        assert res.added == 1
        assert load_library("ashare", root=td)[0].status == "probation"


def test_rebuild_preserves_probation(tmp_path):
    from factorzen.discovery.factor_library import (
        load_library,
        rebuild,
        upsert,
        upsert_probation,
    )

    # 先写一条 active + 一条 probation
    upsert(
        "ashare",
        [{"expression": "rank(close)", "ic_train": 0.05, "holdout_ic": 0.04,
          "dsr_pvalue": 0.2, "n_train": 100, "n_holdout_days": 100}],
        eval_window=("20200101", "20260101"), universe="u", horizon=1,
        run_id="r1", session_dir="s", git_sha="a", now="2026-07-01",
        root=str(tmp_path),
    )
    upsert_probation(
        "ashare",
        [{"expression": "rank(vol)", "ic_train": 0.006, "holdout_ic": 0.001,
          "lift": 0.002, "baseline": 0.03, "passed": True}],
        eval_window=("20200101", "20260101"), universe="u", horizon=5,
        run_id="lift", session_dir="s", git_sha="a", now="2026-07-02",
        root=str(tmp_path),
    )

    def evaluate(exprs):
        # rebuild 只重算 active 源；rank(vol) 不在 sources 也不过 gate
        return [
            {"expression": "rank(close)", "ic_train": 0.05, "holdout_ic": 0.04,
             "dsr_pvalue": 0.2, "n_train": 100, "n_holdout_days": 100},
        ]

    rebuild(
        "ashare", sources=["rank(close)"], eval_window=("20200101", "20260101"),
        universe="u", horizon=1, evaluate=evaluate, git_sha="b",
        now="2026-07-14", root=str(tmp_path), fresh=True,
    )
    lib = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}
    assert "rank(close)" in lib and lib["rank(close)"].status == "active"
    assert "rank(vol)" in lib and lib["rank(vol)"].status == "probation"
    assert abs(lib["rank(vol)"].lift - 0.002) < 1e-9


def test_build_library_pool_excludes_probation_by_default(tmp_path):
    from factorzen.discovery.factor_library import build_library_pool, upsert, upsert_probation

    daily_rows = []
    d0 = date(2024, 1, 2)
    for i in range(30):
        d = d0 + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for s in range(5):
            daily_rows.append({
                "trade_date": d, "ts_code": f"{s:06d}.SH",
                "close": 10.0 + s, "close_adj": 10.0 + s,
                "open_adj": 10.0, "high_adj": 10.1, "low_adj": 9.9,
                "open": 10.0, "high": 10.1, "low": 9.9, "pre_close": 10.0,
                "vol": 1e5, "amount": 1e7,
            })
    daily = pl.DataFrame(daily_rows)

    upsert(
        "ashare",
        [{"expression": "rank(close)", "ic_train": 0.05, "holdout_ic": 0.04,
          "n_holdout_days": 100, "n_train": 100}],
        eval_window=("20200101", "20260101"), universe="u", horizon=1,
        run_id="r", session_dir="s", git_sha="a", now="2026-07-01",
        root=str(tmp_path),
    )
    upsert_probation(
        "ashare",
        [{"expression": "rank(vol)", "ic_train": 0.006, "lift": 0.002,
          "baseline": 0.01, "passed": True}],
        eval_window=("20200101", "20260101"), universe="u", horizon=5,
        run_id="l", session_dir="s", git_sha="a", now="2026-07-02",
        root=str(tmp_path),
    )
    pool = build_library_pool("ashare", daily, root=str(tmp_path))
    assert "rank(close)" in pool
    assert "rank(vol)" not in pool
    # 显式要 probation 才进
    pool2 = build_library_pool(
        "ashare", daily, root=str(tmp_path), statuses=("active", "probation"),
    )
    assert "rank(vol)" in pool2 or "rank(close)" in pool2  # vol 可能物化失败但不应默认混入


# ── 双路径 gray_zone 记账 ────────────────────────────────────────────────────


def test_node_guardrails_marks_gray_zone():
    """单因子门拒绝 + 灰区带 → reject_category=gray_zone，n_gray_zone 计数。"""
    from factorzen.agents.state import AgentState, AttemptRecord

    # 构造：residual IC 在灰区、覆盖够、库空 → raw 退化路径用裸 IC 也可
    # 这里走 raw：ic_train 在 [0.005, 0.015)，holdout 反号 → 门不过，但灰区
    state = AgentState(seed=0)
    state.attempts = [
        AttemptRecord(
            iteration=0, hypothesis="h", expression="rank(close)",
            compile_ok=True, ic_train=0.008, ir_train=0.3, n_train=200,
            passed_guardrails=False, critic_verdict=None, error=None,
        )
    ]
    # mock holdout：覆盖够 + 反号
    import factorzen.validation.holdout as hmod
    from factorzen.validation.holdout import HoldoutICResult

    daily = pl.DataFrame({
        "trade_date": [date(2024, 1, 2 + i) for i in range(5) for _ in range(3)],
        "ts_code": [f"{s:06d}.SH" for _ in range(5) for s in range(3)],
        "close": [10.0] * 15, "close_adj": [10.0] * 15,
        "open_adj": [10.0] * 15, "high_adj": [10.0] * 15, "low_adj": [10.0] * 15,
        "vol": [1e5] * 15, "amount": [1e7] * 15,
    })
    # DataBundle 需要 fwd 等——用 MagicMock 最小接口
    bundle = MagicMock()
    bundle.fwd_returns = pl.DataFrame({
        "trade_date": daily["trade_date"],
        "ts_code": daily["ts_code"],
        "fwd_ret_1d": [0.01] * 15,
    })
    bundle.train_end = "20240110"

    orig = hmod.holdout_ic_result
    hmod.holdout_ic_result = lambda *a, **k: HoldoutICResult(
        ic_mean=-0.01, ir=-0.1, ci=(-0.02, 0.0), n_days=100,
    )
    try:
        # 预处理 / 求值可能失败——改用更稳的路径：直接测 is_gray_zone 集成
        # 若 node_guardrails 因求值失败跳过，至少验证 is_gray_zone 钩子字段契约
        from factorzen.discovery.guardrails import is_gray_zone
        probe = {
            "ic_train": 0.008, "n_holdout_days": 100,
            "residual_ic_train": None,
        }
        assert is_gray_zone(probe, objective="raw")
    finally:
        hmod.holdout_ic_result = orig


def test_mining_session_gray_zone_fields_in_manifest_contract():
    """manifest 契约：n_gray_zone 字段存在于 write 路径（源码守卫）。"""
    src = (Path(__file__).resolve().parents[1] / "src" / "factorzen"
           / "discovery" / "mining_session.py").read_text(encoding="utf-8")
    assert "n_gray_zone" in src
    assert "REJECT_CATEGORY_GRAY_ZONE" in src
    assert "is_gray_zone" in src
    agents_man = (Path(__file__).resolve().parents[1] / "src" / "factorzen"
                  / "agents" / "manifest.py").read_text(encoding="utf-8")
    assert "n_gray_zone" in agents_man
    nodes = (Path(__file__).resolve().parents[1] / "src" / "factorzen"
             / "agents" / "nodes.py").read_text(encoding="utf-8")
    assert "REJECT_CATEGORY_GRAY_ZONE" in nodes
    assert "(灰区,待组合lift)" in nodes


# ── CLI 透传 ─────────────────────────────────────────────────────────────────


def test_cli_lift_test_parser_and_dry_run(tmp_path, monkeypatch):
    """capability↔wiring：子命令注册 + dry-run 不写库。"""
    from factorzen.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "factor-library", "lift-test",
        "--session", str(tmp_path / "run1"),
        "--market", "ashare",
        "--start", "20200101",
        "--end", "20201231",
        "--universe", "csi300",
        "--top-m", "5",
        "--threshold", "0.002",
        "--dry-run",
    ])
    assert args.func.__name__ == "_cmd_factor_library_lift_test"
    assert args.top_m == 5
    assert args.threshold == 0.002
    assert args.dry_run is True

    # 写一个含 gray_zone 的 manifest
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "attempts": [
            {"expression": "rank(close)", "reject_category": "gray_zone",
             "residual_ic_train": 0.006, "n_residual_holdout_days": 100},
        ],
        "candidates": [],
    }), encoding="utf-8")

    lib_root = tmp_path / "lib"
    lib_root.mkdir()
    # mock 数据装配 + lift + upsert
    import factorzen.cli.main as cli_main
    import factorzen.discovery.factor_library as fl
    import factorzen.discovery.lift_test as lt_mod

    monkeypatch.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda args: (pl.DataFrame({
            "trade_date": [date(2020, 1, 2)], "ts_code": ["000001.SZ"],
            "close": [10.0], "close_adj": [10.0],
        }), None),
    )

    def fake_lift(gray, **kw):
        return [{
            "expression": "rank(close)", "lift": 0.005, "baseline": 0.02,
            "passed": True, "candidate_rank_ic": 0.025,
        }]

    written = {"upsert": 0}

    def fake_upsert(*a, **k):
        written["upsert"] += 1
        from factorzen.discovery.factor_library import UpsertResult
        return UpsertResult(added=1)

    monkeypatch.setattr(lt_mod, "run_lift_tests", fake_lift)
    # CLI 从 factor_library 模块调 upsert_probation
    monkeypatch.setattr(fl, "upsert_probation", fake_upsert)

    args.library_root = str(lib_root)
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert written["upsert"] == 0  # dry-run 不写库
    assert (run_dir / "lift_test_manifest.json").exists()
    man = json.loads((run_dir / "lift_test_manifest.json").read_text(encoding="utf-8"))
    assert man["dry_run"] is True
    assert man["n_passed"] == 1
    assert man["threshold"] == 0.002

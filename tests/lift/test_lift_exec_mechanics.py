"""test_lift_group_gate_guard.py：组门连坐防呆 + lift manifest 审计保全。
test_lift_parallel.py：lift 并行 / residual 引擎确定性 / CLI workers 透传。
test_lift_intraday_autoattach.py：lift / forward_track / combine 自动检测 i_* → intraday 装帧。
"""


from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tests._cli_lift_mocks import patch_cli_lift_pre_gates

# ==== 来自 test_lift_group_gate_guard.py ====

def test_sub_floor_predicate_suite():
    """带 residual_ic_train 的候选按 DEFAULT_GRAY_IC_FLOOR(0.008) 判。；无 residual、只有裸 ic_train → 按 DEFAULT_RAW_GRAY_IC_FLOOR(0.010) 判。；无任何 IC 指标（--factor 注入的 python 候选）→ 不算 sub-floor。；显式 floor 覆盖默认口径分派。"""
    # -- 原 test_sub_floor_uses_residual_floor_for_residual_candidates --
    def _section_0_test_sub_floor_uses_residual_floor_for_residual_candidates():
        from factorzen.discovery.guardrails import (
            DEFAULT_GRAY_IC_FLOOR,
            is_sub_floor_candidate,
        )

        assert DEFAULT_GRAY_IC_FLOOR == 0.008
        assert is_sub_floor_candidate({"residual_ic_train": 0.005}) is True
        assert is_sub_floor_candidate({"residual_ic_train": -0.005}) is True  # 取绝对值
        assert is_sub_floor_candidate({"residual_ic_train": 0.008}) is False  # 边界含等号
        assert is_sub_floor_candidate({"residual_ic_train": 0.02}) is False

    _section_0_test_sub_floor_uses_residual_floor_for_residual_candidates()

    # -- 原 test_sub_floor_uses_raw_floor_for_raw_candidates --
    def _section_1_test_sub_floor_uses_raw_floor_for_raw_candidates():
        from factorzen.discovery.guardrails import (
            DEFAULT_RAW_GRAY_IC_FLOOR,
            is_sub_floor_candidate,
        )

        assert DEFAULT_RAW_GRAY_IC_FLOOR == 0.010
        assert is_sub_floor_candidate({"ic_train": 0.009}) is True
        assert is_sub_floor_candidate({"ic_train": 0.011}) is False

    _section_1_test_sub_floor_uses_raw_floor_for_raw_candidates()

    # -- 原 test_sub_floor_missing_ic_is_not_filtered --
    def _section_2_test_sub_floor_missing_ic_is_not_filtered():
        from factorzen.discovery.guardrails import is_sub_floor_candidate

        assert is_sub_floor_candidate({"expression": "py::mom", "kind": "python"}) is False
        assert is_sub_floor_candidate({"ic_train": None}) is False
        assert is_sub_floor_candidate({"residual_ic_train": float("nan")}) is False

    _section_2_test_sub_floor_missing_ic_is_not_filtered()

    # -- 原 test_sub_floor_explicit_floor_override --
    def _section_3_test_sub_floor_explicit_floor_override():
        from factorzen.discovery.guardrails import is_sub_floor_candidate

        assert is_sub_floor_candidate({"residual_ic_train": 0.02}, floor=0.05) is True
        assert is_sub_floor_candidate({"residual_ic_train": 0.005}, floor=0.001) is False

    _section_3_test_sub_floor_explicit_floor_override()


# ---------------------------------------------------------------- CLI 层


def _write_session(
    tmp_path: Path, *, n_good: int = 20, n_sub_floor: int = 130,
) -> Path:
    """事故形态 session：n_good 条真信号 + n_sub_floor 条 sub-floor 噪声。"""
    run_dir = tmp_path / "run1"
    run_dir.mkdir(exist_ok=True)
    attempts = [
        {
            "expression": f"good_{i}",
            "reject_category": "lift_queue",
            "residual_ic_train": 0.02 + 0.0001 * i,
            "n_residual_holdout_days": 100,
        }
        for i in range(n_good)
    ] + [
        {
            "expression": f"noise_{i}",
            "reject_category": "lift_queue",
            "residual_ic_train": 0.001 + 0.000001 * i,  # 远低于 0.008
            "n_residual_holdout_days": 100,
        }
        for i in range(n_sub_floor)
    ]
    (run_dir / "manifest.json").write_text(
        json.dumps({"attempts": attempts, "candidates": []}), encoding="utf-8",
    )
    return run_dir


def _base_args(run_dir: Path, lib_root: Path, extra: list[str] | None = None):
    from factorzen.cli.main import build_parser

    argv = [
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--market", "ashare",
        "--start", "20200101",
        "--end", "20201231",
        "--universe", "csi300",
        "--dry-run",
    ]
    if extra:
        argv.extend(extra)
    parser = build_parser()
    args = parser.parse_args(argv)
    args.library_root = str(lib_root)
    return args


def _patch_cli(monkeypatch, *, group_lift: float = -0.0007, seen: dict | None = None):
    """mock 掉数据准备 / 覆盖门 / 组门 / 逐候选 lift；捕获组门收到的队列。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.lift_test as lt_mod

    monkeypatch.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda a: (pl.DataFrame({
            "trade_date": [date(2020, 1, 2)], "ts_code": ["000001.SZ"],
            "close": [10.0], "close_adj": [10.0],
        }), None, {}),
    )
    monkeypatch.setattr(
        lt_mod, "filter_candidates_by_coverage",
        lambda cands, **k: (list(cands), []),
    )

    def fake_group(queue, **k):
        if seen is not None:
            seen["queue"] = [c.get("expression") for c in queue]
        return {
            "lift": group_lift, "lift_se": 0.0001, "error": None,
            "lift_metric": "residual_ic_v1",
        }

    monkeypatch.setattr(lt_mod, "run_group_lift", fake_group)
    monkeypatch.setattr(
        lt_mod, "run_lift_tests",
        lambda gray, **kw: [
            {
                "expression": c.get("expression"), "lift": 0.01,
                "baseline": 0.02, "passed": True, "elapsed_s": 0.01,
            }
            for c in gray
        ],
    )
    monkeypatch.setattr(lt_mod, "resolve_lift_workers", lambda w: 2)


def test_sub_floor_cli_group_gate_suite(tmp_path, capsys):
    """事故复现：130 噪声 + 20 真信号，--top-m 0 全测。；--include-sub-floor → sub-floor 照旧进组门（可复现旧行为/连坐）。；--queue-ic-floor 显式抬高地板 → 更多候选被判 sub-floor。；全员 sub-floor → 组门根本不跑，零 results 行（不产生连坐拒绝）。"""
    # -- 原 test_sub_floor_candidates_excluded_from_group_gate_by_default --
    def _section_0_test_sub_floor_candidates_excluded_from_group_gate_by_default(tmp_path, mp, capsys):
        import factorzen.cli.main as cli_main

        run_dir = _write_session(tmp_path, n_good=20, n_sub_floor=130)
        lib_root = tmp_path / "lib"
        lib_root.mkdir()
        args = _base_args(run_dir, lib_root, ["--set", "top_m=0"])
        seen: dict = {}
        # 组门给负 lift：若噪声进了组门就会全体连坐
        _patch_cli(mp, group_lift=-0.0007, seen=seen)

        rc = cli_main._cmd_factor_library_lift_test(args)
        assert rc == 0

        # 组门只应看到 20 条 good
        assert len(seen["queue"]) == 20
        assert all(e.startswith("good_") for e in seen["queue"])

        man = json.loads((run_dir / "lift_test_manifest.json").read_text(encoding="utf-8"))
        assert man["n_sub_floor"] == 130
        assert man["sub_floor_filtered"] is True
        dropped = {d["expression"] for d in man["lift_dropped_sub_floor"]}
        assert len(dropped) == 130
        assert all(e.startswith("noise_") for e in dropped)

        # 被过滤者不产生 results 行 → 不会被写回 lift_rejected
        exprs = {r.get("expression") for r in man["results"]}
        assert not any(str(e).startswith("noise_") for e in exprs)

        err = capsys.readouterr().err
        assert "sub-floor" in err

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_sub_floor_candidates_excluded_from_group_gate_by_default(_tp0, mp, capsys)

    # -- 原 test_include_sub_floor_escape_hatch_restores_old_behavior --
    def _section_1_test_include_sub_floor_escape_hatch_restores_old_behavior(tmp_path, mp):
        import factorzen.cli.main as cli_main

        run_dir = _write_session(tmp_path, n_good=20, n_sub_floor=130)
        lib_root = tmp_path / "lib"
        lib_root.mkdir()
        args = _base_args(
            run_dir, lib_root, ["--set", "top_m=0", "--set", "include_sub_floor=true"],
        )
        seen: dict = {}
        _patch_cli(mp, group_lift=-0.0007, seen=seen)

        rc = cli_main._cmd_factor_library_lift_test(args)
        assert rc == 0
        assert len(seen["queue"]) == 150  # 全量进组门

        man = json.loads((run_dir / "lift_test_manifest.json").read_text(encoding="utf-8"))
        assert man["n_sub_floor"] == 130      # 仍如实记账
        assert man["sub_floor_filtered"] is False
        # 旧行为：组门不过 → 全体连坐
        assert all(
            str(r.get("error") or "").startswith("group_gate") for r in man["results"]
        )
        assert len(man["results"]) == 150

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_include_sub_floor_escape_hatch_restores_old_behavior(_tp1, mp)

    # -- 原 test_queue_ic_floor_flag_overrides_default --
    def _section_2_test_queue_ic_floor_flag_overrides_default(tmp_path, mp):
        import factorzen.cli.main as cli_main

        run_dir = _write_session(tmp_path, n_good=20, n_sub_floor=10)
        lib_root = tmp_path / "lib"
        lib_root.mkdir()
        # good 的 residual_ic_train 在 [0.02, 0.0219]；地板抬到 0.021 → 只剩少数存活
        args = _base_args(
            run_dir, lib_root, ["--set", "top_m=0", "--set", "queue_ic_floor=0.021"],
        )
        seen: dict = {}
        _patch_cli(mp, group_lift=0.01, seen=seen)

        rc = cli_main._cmd_factor_library_lift_test(args)
        assert rc == 0
        man = json.loads((run_dir / "lift_test_manifest.json").read_text(encoding="utf-8"))
        assert man["queue_ic_floor"] == 0.021
        # good 的 residual_ic_train = 0.02 + 0.0001*i (i=0..19) → 仅 i≥10 的 10 条 ≥0.021
        # 加上 10 条 noise 全剔 → n_sub_floor = 10 + 10 = 20，组门只剩 10 条
        assert man["n_sub_floor"] == 20
        assert len(seen["queue"]) == 10
        assert all(e.startswith("good_") for e in seen["queue"])

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_queue_ic_floor_flag_overrides_default(_tp2, mp)

    # -- 原 test_all_sub_floor_skips_group_gate_without_collateral_rejects --
    def _section_3_test_all_sub_floor_skips_group_gate_without_collateral_rejects(tmp_path, mp):
        import factorzen.cli.main as cli_main

        run_dir = _write_session(tmp_path, n_good=0, n_sub_floor=30)
        lib_root = tmp_path / "lib"
        lib_root.mkdir()
        args = _base_args(run_dir, lib_root, ["--set", "top_m=0"])
        seen: dict = {}
        _patch_cli(mp, group_lift=-0.0007, seen=seen)

        rc = cli_main._cmd_factor_library_lift_test(args)
        assert rc == 0
        assert "queue" not in seen  # 组门未被调用
        man = json.loads((run_dir / "lift_test_manifest.json").read_text(encoding="utf-8"))
        assert man["n_sub_floor"] == 30
        assert man["results"] == []

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_all_sub_floor_skips_group_gate_without_collateral_rejects(_tp3, mp)


# ---------------------------------------------------------------- 审计保全


def test_lift_manifest_audit_suite(tmp_path):
    """落盘 = 时间戳归档 + 稳定 latest 指针，两者内容一致。；事故核心：成功 run 的证据不被后续失败 run 覆写。"""
    # -- 原 test_manifest_archived_with_timestamp_and_latest_pointer --
    def _section_0_test_manifest_archived_with_timestamp_and_latest_pointer(tmp_path, mp):
        import factorzen.cli.main as cli_main

        run_dir = _write_session(tmp_path, n_good=5, n_sub_floor=0)
        lib_root = tmp_path / "lib"
        lib_root.mkdir()
        args = _base_args(run_dir, lib_root, ["--set", "top_m=0"])
        _patch_cli(mp, group_lift=0.01)

        rc = cli_main._cmd_factor_library_lift_test(args)
        assert rc == 0

        latest = run_dir / "lift_test_manifest.json"
        assert latest.is_file()  # 稳定入口零回归
        archives = sorted(run_dir.glob("lift_test_manifest_*.json"))
        assert len(archives) == 1
        assert json.loads(archives[0].read_text(encoding="utf-8")) == json.loads(
            latest.read_text(encoding="utf-8")
        )

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_manifest_archived_with_timestamp_and_latest_pointer(_tp0, mp)

    # -- 原 test_failed_rerun_cannot_destroy_earlier_success_evidence --
    def _section_1_test_failed_rerun_cannot_destroy_earlier_success_evidence(tmp_path, mp):
        import factorzen.cli.main as cli_main

        run_dir = _write_session(tmp_path, n_good=5, n_sub_floor=0)
        lib_root = tmp_path / "lib"
        lib_root.mkdir()

        # run 1：组门过 + 逐候选全 passed
        args1 = _base_args(run_dir, lib_root, ["--set", "top_m=0"])
        _patch_cli(mp, group_lift=0.01)
        assert cli_main._cmd_factor_library_lift_test(args1) == 0
        archives1 = sorted(run_dir.glob("lift_test_manifest_*.json"))
        assert len(archives1) == 1
        first = json.loads(archives1[0].read_text(encoding="utf-8"))
        assert first["n_passed"] == 5

        # run 2：组门全拒（同秒重跑也必须另起归档，不许覆写）
        args2 = _base_args(run_dir, lib_root, ["--set", "top_m=0", "--set", "include_sub_floor=true"])
        _patch_cli(mp, group_lift=-0.0007)
        assert cli_main._cmd_factor_library_lift_test(args2) == 0

        archives2 = sorted(run_dir.glob("lift_test_manifest_*.json"))
        assert len(archives2) == 2, "同秒重跑覆写了归档 = 证据丢失"
        # 第一次的归档内容原样留存
        assert json.loads(archives1[0].read_text(encoding="utf-8")) == first
        assert first["n_passed"] == 5
        # latest 指针指向最新（失败）那次
        latest = json.loads(
            (run_dir / "lift_test_manifest.json").read_text(encoding="utf-8")
        )
        assert latest["n_passed"] == 0

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_failed_rerun_cannot_destroy_earlier_success_evidence(_tp1, mp)


# ==== 来自 test_lift_parallel.py ====

def _dates(n_days: int):
    days, d = [], date(2024, 1, 2)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return days


def _panels(n_days: int = 50, n_stocks: int = 40, seed: int = 0):
    """合成 active / 正交信号候选 / ret（满足 residual 日守卫）。"""
    rng = np.random.default_rng(seed)
    dates = _dates(n_days)
    codes = [f"{i:04d}.SZ" for i in range(n_stocks)]
    lib_rows, cand_rows, ret_rows = [], [], []
    for d in dates:
        lib = rng.standard_normal(n_stocks)
        ortho = rng.standard_normal(n_stocks)
        ortho = ortho - (np.dot(ortho, lib) / (np.dot(lib, lib) + 1e-12)) * lib
        ret = 0.3 * lib + 0.7 * ortho + 0.1 * rng.standard_normal(n_stocks)
        for s, code in enumerate(codes):
            lib_rows.append({"trade_date": d, "ts_code": code, "factor_value": float(lib[s])})
            cand_rows.append({"trade_date": d, "ts_code": code, "factor_value": float(ortho[s])})
            ret_rows.append({"trade_date": d, "ts_code": code, "ret": float(ret[s])})
    active = {"lib_a": pl.DataFrame(lib_rows)}
    cand = pl.DataFrame(cand_rows)
    ret = pl.DataFrame(ret_rows)
    return active, cand, ret


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

    active, cand, ret = _panels()
    grays = [{"expression": "c0", "residual_ic_train": 0.01}]
    rows = lt.run_lift_tests(
        grays,
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        lift_workers=1,
        top_m=None,
    )
    assert created["n"] == 0
    assert len(rows) == 1
    assert rows[0]["expression"] == "c0"

    # 对照：workers>1 应建池（恢复真类后）
    monkeypatch.setattr(lt, "ThreadPoolExecutor", real)
    rows4 = lt.run_lift_tests(
        grays,
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        lift_workers=4,
        top_m=None,
    )
    assert len(rows4) == 1


def test_residual_meta_shared_by_run_lift_and_group():
    """run_lift_tests / run_group_lift 共用 residual_ic_v1 meta 契约。"""
    from factorzen.discovery.lift_test import run_group_lift, run_lift_tests

    active, cand, ret = _panels()
    daily = pl.DataFrame({"trade_date": [], "ts_code": [], "close": []})
    grays = [{"expression": "c0", "residual_ic_train": 0.01}]

    rows = run_lift_tests(
        grays,
        market="ashare",
        daily=daily,
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        lift_workers=1,
    )
    out = run_group_lift(
        grays,
        market="ashare",
        daily=daily,
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
    )
    assert rows[0].get("lift_metric") == "residual_ic_v1"
    assert rows[0].get("cv_train_days") is None
    assert rows[0].get("cv_test_days") is None
    assert out.get("lift_metric") == "residual_ic_v1"
    assert out.get("n_lib_factors") == 1
    assert "base_daily" not in out
    assert out["baseline"] is None


def test_cli_lift_workers_from_outer_parser(tmp_path, monkeypatch):
    """CLI --lift-workers 从 parser 最外层透传到 run_lift_tests。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.lift_test as lt_mod
    from factorzen.cli.main import build_parser

    run_dir = tmp_path / "sess"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "attempts": [
                {
                    "expression": "rank(close)",
                    "reject_category": "lift_queue",
                    "residual_ic_train": 0.01,
                    "n_residual_holdout_days": 100,
                },
            ],
            "candidates": [],
        }),
        encoding="utf-8",
    )
    lib_root = tmp_path / "lib"
    lib_root.mkdir()

    captured: dict = {}

    def fake_lift(gray, **kw):
        captured["lift_workers"] = kw.get("lift_workers")
        return [{
            "expression": "rank(close)",
            "lift": 0.01,
            "baseline": None,
            "passed": True,
            "candidate_rank_ic": 0.01,
            "elapsed_s": 0.01,
        }]

    monkeypatch.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda args: (
            pl.DataFrame({
                "trade_date": [date(2020, 1, 2)],
                "ts_code": ["000001.SZ"],
                "close": [10.0],
                "close_adj": [10.0],
            }),
            None,
            {},
        ),
    )
    patch_cli_lift_pre_gates(monkeypatch)
    monkeypatch.setattr(lt_mod, "run_lift_tests", fake_lift)
    import factorzen.discovery.factor_library as fl
    from factorzen.discovery.factor_library import UpsertResult

    monkeypatch.setattr(fl, "upsert_probation", lambda *a, **k: UpsertResult(added=0))

    parser = build_parser()
    args = parser.parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--market", "ashare",
        "--start", "20200101",
        "--end", "20201231",
        "--universe", "csi300",
        "--set", "lift_workers=3",
        "--dry-run",
        "--set", "library_root=" + str(lib_root),
    ])
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert captured.get("lift_workers") == 3

    # 默认 None → 自适应
    args_def = parser.parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--market", "ashare",
        "--start", "20200101",
        "--end", "20201231",
        "--universe", "csi300",
    ])
    assert args_def.lift_workers is None  # None→run_lift_tests 自适应


# ==== 来自 test_lift_intraday_autoattach.py ====

def test_intraday_autoattach_suite():
    """test_expressions_need_intraday_basic；_assemble_daily 对含 i_* 的表达式集合带 intraday=True。；含 i_* leaf_names 时 system 含 NOTES；不含时与同参数对照逐字节相等。；`ix_*` 表达式叶必须从 `run_mine` 一路透传到 `prepare_mining_daily`。"""
    # -- 原 test_expressions_need_intraday_basic --
    def _section_0_test_expressions_need_intraday_basic():
        from factorzen.discovery.preparation import expressions_need_intraday

        assert expressions_need_intraday(["rank(close)"]) is False
        assert expressions_need_intraday(["rank(i_rv)"]) is True
        assert expressions_need_intraday(["rank(close)", "ts_mean(i_amihud, 5)"]) is True
        # parse 失败跳过
        assert expressions_need_intraday(["not_a_real_op(foo)"]) is False

    _section_0_test_expressions_need_intraday_basic()

    # -- 原 test_forward_track_assemble_passes_intraday_when_i_star --
    def _section_1_test_forward_track_assemble_passes_intraday_when_i_star(mp):
        import factorzen.discovery.forward_track as ft

        calls: list[dict] = []

        def _fake_prepare(*a, **kw):
            calls.append(kw)
            return pl.DataFrame({
                "trade_date": [],
                "ts_code": [],
            })

        mp.setattr(ft, "prepare_mining_daily", _fake_prepare)
        # expressions_need_intraday 用真实现
        ft._assemble_daily(
            "ashare", "20240115", 60, universe="csi300",
            expressions=["rank(i_rv)"],
        )
        assert calls, "应调用 prepare_mining_daily"
        assert calls[0].get("intraday") is True

        calls.clear()
        ft._assemble_daily(
            "ashare", "20240115", 60, universe="csi300",
            expressions=["rank(close)"],
        )
        assert calls[0].get("intraday") is False

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_forward_track_assemble_passes_intraday_when_i_star(mp)

    # -- 原 test_prompt_injects_notes_only_when_i_star_leaves --
    def _section_2_test_prompt_injects_notes_only_when_i_star_leaves():
        from factorzen.llm.generation import build_agent_messages
        from factorzen.llm.prompt_fragments import ASHARE_INTRADAY_LEAF_NOTES

        # 不含 i_*：同参数两次逐字节相等（零回归锚；golden 见 test_mining_multimarket）
        base = build_agent_messages(["ts_mean", "rank"], ["close", "vol"], "FB", ["neg1"])
        base2 = build_agent_messages(["ts_mean", "rank"], ["close", "vol"], "FB", ["neg1"])
        assert base[0]["content"] == base2[0]["content"]
        assert ASHARE_INTRADAY_LEAF_NOTES not in base[0]["content"]

        with_i = build_agent_messages(
            ["ts_mean", "rank"], ["close", "vol", "i_rv"], "FB", ["neg1"],
        )
        assert ASHARE_INTRADAY_LEAF_NOTES in with_i[0]["content"]
        # 去掉 NOTES 后，与「同 leaf 列表但不注入」的预期差仅在 NOTES 段
        stripped = with_i[0]["content"].replace("\n" + ASHARE_INTRADAY_LEAF_NOTES, "", 1)
        # leaf 列表含 i_rv，故与 base 不同；但 stripped 不应再含 NOTES
        assert ASHARE_INTRADAY_LEAF_NOTES not in stripped
        assert "i_rv" in stripped

    _section_2_test_prompt_injects_notes_only_when_i_star_leaves()

    # -- 原 test_run_mine_passes_intraday_expr_leaves_through --
    def _section_3_test_run_mine_passes_intraday_expr_leaves_through(mp):
        import factorzen.pipelines.factor_mine  # noqa: F401  （见上方首次导入陷阱注释）

        calls: list[dict] = []

        def _fake_prepare(*a, **kw):
            calls.append(kw)
            return pl.DataFrame({
                "trade_date": [__import__("datetime").date(2024, 1, 2)] * 2,
                "ts_code": ["A", "B"], "close": [1.0, 2.0], "close_adj": [1.0, 2.0],
            })

        mp.setattr(
            "factorzen.discovery.preparation.prepare_mining_daily", _fake_prepare,
        )
        mp.setattr(
            "factorzen.pipelines.factor_mine.prepare_mining_daily", _fake_prepare,
        )
        mp.setattr(
            "factorzen.pipelines.factor_mine.run_session",
            lambda *a, **kw: {"session_dir": None, "candidates": []},
        )
        mp.setattr(
            "factorzen.pipelines.factor_mine._inject_membership_into_session_manifest",
            lambda *a, **kw: None,
        )

        from factorzen.pipelines.factor_mine import run_mine

        run_mine(start="20240101", end="20240601", universe=None,
                 intraday=True, intraday_expr_leaves=["ix_abc12345"])

        assert calls, "prepare_mining_daily 未被调用"
        assert calls[0].get("intraday_expr_leaves") == ["ix_abc12345"], calls[0]
        # 不传时保持 None（零回归）
        calls.clear()
        run_mine(start="20240101", end="20240601", universe=None, intraday=True)
        assert calls[0].get("intraday_expr_leaves") is None, calls[0]

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_run_mine_passes_intraday_expr_leaves_through(mp)



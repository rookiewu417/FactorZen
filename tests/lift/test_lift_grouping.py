"""多 session lift-test：按各自 admission/holdout 窗分组评分，禁止并集窗。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from tests._cli_lift_mocks import patch_cli_lift_pre_gates

# ── 纯函数单测 ───────────────────────────────────────────────────────────────


def test_group_lift_candidates_pure_suite():
    """跨 session 去重 + 按窗分组；重复 expr 归首次窗，绝无并集窗。；同 expression 跨 session 首次出现胜出；被去重者连同其窗丢弃。；相同 admission 窗的多个 session 合并为一组（去重后）。；test_group_lift_candidates_skips_empty_expression；分组顺序 = 首次出现的窗顺序；组内候选 = 首次出现顺序。"""
    # -- 原 test_group_lift_candidates_by_admission_two_windows_no_union --
    def _section_0_test_group_lift_candidates_by_admission_two_windows_no_union():
        from factorzen.cli.main import _group_lift_candidates_by_admission

        items = [
            {
                "session": "A",
                "candidates": [
                    {"expression": "exprA1", "reject_category": "gray_zone"},
                    {"expression": "exprA2", "reject_category": "gray_zone"},
                ],
                "adm_start": "2024-01-01",
                "adm_end": "2024-03-31",
            },
            {
                "session": "B",
                "candidates": [
                    {"expression": "exprB1", "reject_category": "gray_zone"},
                    {"expression": "exprA1", "reject_category": "gray_zone"},  # 与 A 重复
                ],
                "adm_start": "2024-10-01",
                "adm_end": "2024-12-31",
            },
        ]
        groups = _group_lift_candidates_by_admission(items)

        assert len(groups) == 2
        (s1, e1, c1), (s2, e2, c2) = groups

        assert (s1, e1) == ("2024-01-01", "2024-03-31")
        assert [c["expression"] for c in c1] == ["exprA1", "exprA2"]

        assert (s2, e2) == ("2024-10-01", "2024-12-31")
        assert [c["expression"] for c in c2] == ["exprB1"]

        # 没有任何候选落入并集窗
        union = ("2024-01-01", "2024-12-31")
        for s, e, _ in groups:
            assert (s, e) != union

    _section_0_test_group_lift_candidates_by_admission_two_windows_no_union()

    # -- 原 test_group_lift_candidates_dedup_keeps_first_window --
    def _section_1_test_group_lift_candidates_dedup_keeps_first_window():
        from factorzen.cli.main import _group_lift_candidates_by_admission

        items = [
            {
                "session": "early",
                "candidates": [{"expression": "shared"}],
                "adm_start": "2023-01-01",
                "adm_end": "2023-06-30",
            },
            {
                "session": "late",
                "candidates": [{"expression": "shared"}, {"expression": "only_late"}],
                "adm_start": "2024-01-01",
                "adm_end": "2024-06-30",
            },
        ]
        groups = _group_lift_candidates_by_admission(items)
        by_win = {(s, e): [c["expression"] for c in cs] for s, e, cs in groups}
        assert by_win[("2023-01-01", "2023-06-30")] == ["shared"]
        assert by_win[("2024-01-01", "2024-06-30")] == ["only_late"]

    _section_1_test_group_lift_candidates_dedup_keeps_first_window()

    # -- 原 test_group_lift_candidates_same_window_merges --
    def _section_2_test_group_lift_candidates_same_window_merges():
        from factorzen.cli.main import _group_lift_candidates_by_admission

        items = [
            {
                "session": "s1",
                "candidates": [{"expression": "a"}, {"expression": "b"}],
                "adm_start": "2024-01-01",
                "adm_end": "2024-03-31",
            },
            {
                "session": "s2",
                "candidates": [{"expression": "b"}, {"expression": "c"}],
                "adm_start": "2024-01-01",
                "adm_end": "2024-03-31",
            },
        ]
        groups = _group_lift_candidates_by_admission(items)
        assert len(groups) == 1
        s, e, cands = groups[0]
        assert (s, e) == ("2024-01-01", "2024-03-31")
        assert [c["expression"] for c in cands] == ["a", "b", "c"]

    _section_2_test_group_lift_candidates_same_window_merges()

    # -- 原 test_group_lift_candidates_skips_empty_expression --
    def _section_3_test_group_lift_candidates_skips_empty_expression():
        from factorzen.cli.main import _group_lift_candidates_by_admission

        items = [
            {
                "session": "s",
                "candidates": [
                    {"expression": ""},
                    {"expression": None},
                    {"expression": "ok"},
                ],
                "adm_start": None,
                "adm_end": None,
            },
        ]
        groups = _group_lift_candidates_by_admission(items)
        assert len(groups) == 1
        assert [c["expression"] for c in groups[0][2]] == ["ok"]

    _section_3_test_group_lift_candidates_skips_empty_expression()

    # -- 原 test_group_lift_candidates_stable_order_by_first_appearance --
    def _section_4_test_group_lift_candidates_stable_order_by_first_appearance():
        from factorzen.cli.main import _group_lift_candidates_by_admission

        items = [
            {
                "session": "B_first",
                "candidates": [{"expression": "b1"}],
                "adm_start": "2024-10-01",
                "adm_end": "2024-12-31",
            },
            {
                "session": "A_second",
                "candidates": [{"expression": "a1"}],
                "adm_start": "2024-01-01",
                "adm_end": "2024-03-31",
            },
        ]
        groups = _group_lift_candidates_by_admission(items)
        assert [g[0] for g in groups] == ["2024-10-01", "2024-01-01"]

    _section_4_test_group_lift_candidates_stable_order_by_first_appearance()


# ── CLI seam：多 session 分窗调用 run_lift_tests ─────────────────────────────


def _write_session_with_holdout(
    root: Path,
    name: str,
    *,
    expressions: list[str],
    holdout_start: str,
    holdout_end: str,
    horizon: int | None = None,
    horizon_in_params: bool = False,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    attempts = [
        {
            "expression": expr,
            "reject_category": "gray_zone",
            "residual_ic_train": 0.02,  # ≥ DEFAULT_GRAY_IC_FLOOR（避开 sub-floor 防呆）
            "n_residual_holdout_days": 100,
        }
        for expr in expressions
    ]
    man: dict = {
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
        "end": holdout_end,
        "attempts": attempts,
        "candidates": [],
    }
    if horizon is not None:
        if horizon_in_params:
            man["params"] = {"horizon": horizon}
        else:
            man["horizon"] = horizon
    (run_dir / "manifest.json").write_text(
        json.dumps(man),
        encoding="utf-8",
    )
    return run_dir


def _fake_daily() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [date(2020, 1, 2)],
            "ts_code": ["000001.SZ"],
            "close": [10.0],
            "close_adj": [10.0],
        }
    )


def test_cli_lift_admission_window_suite(tmp_path, capsys):
    """两个不相交 holdout session → run_lift_tests 调 2 次，绝不用并集窗。；--admission-start/end 任一非空 → 所有候选同一旗标窗，只调一次。；单 session：一次 run_lift_tests，窗 = 该 session holdout（零回归）。"""
    # -- 原 test_cli_lift_test_multi_session_calls_per_admission_window --
    def _section_0_test_cli_lift_test_multi_session_calls_per_admission_window(tmp_path, mp, capsys):
        import factorzen.cli.main as cli_main
        import factorzen.discovery.lift_test as lt_mod
        from factorzen.cli.main import build_parser

        sa = _write_session_with_holdout(
            tmp_path,
            "sess_a",
            expressions=["exprA1", "exprA2"],
            holdout_start="2024-01-01",
            holdout_end="2024-03-31",
        )
        sb = _write_session_with_holdout(
            tmp_path,
            "sess_b",
            expressions=["exprB1", "exprA1"],
            holdout_start="2024-10-01",
            holdout_end="2024-12-31",
        )

        mp.setattr(
            cli_main,
            "_prepare_agent_mining_data",
            lambda args: (_fake_daily(), None, {}),
        )

        calls: list[tuple[int, str | None, str | None]] = []

        def fake_lift(gray, **kw):
            ctx = kw.get("ctx")
            calls.append(
                (
                    len(gray),
                    getattr(ctx, "admission_start", None),
                    getattr(ctx, "admission_end", None),
                )
            )
            return [
                {
                    "expression": g.get("expression"),
                    "lift": 0.001,
                    "lift_se": 0.001,
                    "lift_second_half": 0.001,
                    "baseline": 0.02,
                    "passed": False,
                }
                for g in gray
            ]

        patch_cli_lift_pre_gates(mp)
        mp.setattr(lt_mod, "run_lift_tests", fake_lift)

        args = build_parser().parse_args(
            [
                "factor-library",
                "lift-test",
                # nargs="+"：多 session 写在同一 --session 后，不可重复 --session
                "--session",
                str(sa),
                str(sb),
                "--market",
                "ashare",
                "--start",
                "20200101",
                "--end",
                "20241231",
                "--set",
                "library_root=" + str(tmp_path / "lib"),
            ]
        )
        rc = cli_main._cmd_factor_library_lift_test(args)
        assert rc == 0

        assert len(calls) == 2
        windows = {(start, end) for _, start, end in calls}
        assert ("2024-01-01", "2024-03-31") in windows
        assert ("2024-10-01", "2024-12-31") in windows
        # 并集窗从未出现
        assert ("2024-01-01", "2024-12-31") not in windows

        by_win = {(s, e): n for n, s, e in calls}
        assert by_win[("2024-01-01", "2024-03-31")] == 2  # exprA1, exprA2
        assert by_win[("2024-10-01", "2024-12-31")] == 1  # exprB1 only

        out = capsys.readouterr().out
        # no silent caps：分组与候选数可见
        assert "2024-01-01" in out
        assert "2024-10-01" in out

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_cli_lift_test_multi_session_calls_per_admission_window(_tp0, mp, capsys)

    # -- 原 test_cli_lift_test_flag_admission_forces_single_group --
    def _section_1_test_cli_lift_test_flag_admission_forces_single_group(tmp_path, mp):
        import factorzen.cli.main as cli_main
        import factorzen.discovery.lift_test as lt_mod
        from factorzen.cli.main import build_parser

        sa = _write_session_with_holdout(
            tmp_path,
            "sess_a",
            expressions=["exprA1"],
            holdout_start="2024-01-01",
            holdout_end="2024-03-31",
        )
        sb = _write_session_with_holdout(
            tmp_path,
            "sess_b",
            expressions=["exprB1"],
            holdout_start="2024-10-01",
            holdout_end="2024-12-31",
        )

        mp.setattr(
            cli_main,
            "_prepare_agent_mining_data",
            lambda args: (_fake_daily(), None, {}),
        )

        calls: list[tuple[int, str | None, str | None]] = []

        def fake_lift(gray, **kw):
            ctx = kw.get("ctx")
            calls.append(
                (
                    len(gray),
                    getattr(ctx, "admission_start", None),
                    getattr(ctx, "admission_end", None),
                )
            )
            return [
                {
                    "expression": g.get("expression"),
                    "lift": 0.0,
                    "lift_se": 0.0,
                    "lift_second_half": 0.0,
                    "baseline": 0.0,
                    "passed": False,
                }
                for g in gray
            ]

        patch_cli_lift_pre_gates(mp)
        mp.setattr(lt_mod, "run_lift_tests", fake_lift)

        args = build_parser().parse_args(
            [
                "factor-library",
                "lift-test",
                "--session",
                str(sa),
                str(sb),
                "--start",
                "20200101",
                "--end",
                "20241231",
                "--set",
                "library_root=" + str(tmp_path / "lib"),
                "--admission-start",
                "2025-01-01",
                "--admission-end",
                "2025-06-30",
            ]
        )
        rc = cli_main._cmd_factor_library_lift_test(args)
        assert rc == 0
        assert len(calls) == 1
        n, s, e = calls[0]
        assert n == 2  # exprA1 + exprB1
        assert (s, e) == ("2025-01-01", "2025-06-30")

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_cli_lift_test_flag_admission_forces_single_group(_tp1, mp)

    # -- 原 test_cli_lift_test_single_session_one_group --
    def _section_2_test_cli_lift_test_single_session_one_group(tmp_path, mp):
        import factorzen.cli.main as cli_main
        import factorzen.discovery.lift_test as lt_mod
        from factorzen.cli.main import build_parser

        sa = _write_session_with_holdout(
            tmp_path,
            "only",
            expressions=["rank(close)"],
            holdout_start="2024-06-01",
            holdout_end="2024-08-31",
        )
        mp.setattr(
            cli_main,
            "_prepare_agent_mining_data",
            lambda args: (_fake_daily(), None, {}),
        )
        calls: list = []

        def fake_lift(gray, **kw):
            ctx = kw.get("ctx")
            calls.append(
                {
                    "n": len(gray),
                    "start": getattr(ctx, "admission_start", None),
                    "end": getattr(ctx, "admission_end", None),
                }
            )
            return [
                {
                    "expression": "rank(close)",
                    "lift": 0.005,
                    "lift_se": 0.001,
                    "lift_second_half": 0.004,
                    "baseline": 0.02,
                    "passed": True,
                }
            ]

        patch_cli_lift_pre_gates(mp)
        mp.setattr(lt_mod, "run_lift_tests", fake_lift)

        args = build_parser().parse_args(
            [
                "factor-library",
                "lift-test",
                "--session",
                str(sa),
                "--start",
                "20200101",
                "--end",
                "20241231",
                "--set",
                "library_root=" + str(tmp_path / "lib"),
            ]
        )
        rc = cli_main._cmd_factor_library_lift_test(args)
        assert rc == 0
        assert len(calls) == 1
        assert calls[0]["n"] == 1
        assert calls[0]["start"] == "2024-06-01"
        assert calls[0]["end"] == "2024-08-31"

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_cli_lift_test_single_session_one_group(_tp2, mp)


# ── P5：horizon 从 flag / manifest 解析，禁硬编码 DEFAULT_HORIZON ─────────────


def test_cli_lift_horizon_suite(tmp_path, capsys):
    """纯函数：顶层 horizon / params.horizon / 皆无。；session manifest horizon=3、不传 --horizon → run_lift_tests ctx.horizon==3。；--horizon 7 覆盖 session manifest horizon=3。；team-style manifest：params.horizon=4 → ctx.horizon==4。；多 session horizon 不一致 → stderr 警告，仍用第一个 session 的值。"""
    # -- 原 test_horizon_from_manifest_top_level_and_params --
    def _section_0_test_horizon_from_manifest_top_level_and_params():
        from factorzen.cli.main import _horizon_from_manifest

        assert _horizon_from_manifest({"horizon": 3}) == 3
        assert _horizon_from_manifest({"params": {"horizon": 7}}) == 7
        assert _horizon_from_manifest({"params": {}}) is None
        assert _horizon_from_manifest({}) is None
        # 顶层优先于 params
        assert _horizon_from_manifest({"horizon": 2, "params": {"horizon": 9}}) == 2

    _section_0_test_horizon_from_manifest_top_level_and_params()

    # -- 原 test_cli_lift_test_horizon_from_manifest --
    def _section_1_test_cli_lift_test_horizon_from_manifest(tmp_path, mp):
        import factorzen.cli.main as cli_main
        import factorzen.discovery.lift_test as lt_mod
        from factorzen.cli.main import build_parser

        sa = _write_session_with_holdout(
            tmp_path,
            "h3",
            expressions=["rank(close)"],
            holdout_start="2024-06-01",
            holdout_end="2024-08-31",
            horizon=3,
        )
        mp.setattr(
            cli_main,
            "_prepare_agent_mining_data",
            lambda args: (_fake_daily(), None, {}),
        )
        captured: list = []

        def fake_lift(gray, **kw):
            captured.append(kw)
            return [
                {
                    "expression": g.get("expression"),
                    "lift": 0.0,
                    "lift_se": 0.0,
                    "lift_second_half": 0.0,
                    "baseline": 0.0,
                    "passed": False,
                }
                for g in gray
            ]

        patch_cli_lift_pre_gates(mp)
        mp.setattr(lt_mod, "run_lift_tests", fake_lift)

        args = build_parser().parse_args(
            [
                "factor-library",
                "lift-test",
                "--session",
                str(sa),
                "--start",
                "20200101",
                "--end",
                "20241231",
                "--set",
                "library_root=" + str(tmp_path / "lib"),
            ]
        )
        rc = cli_main._cmd_factor_library_lift_test(args)
        assert rc == 0
        assert captured
        ctx = captured[0].get("ctx")
        assert ctx is not None
        assert ctx.horizon == 3, f"期望 manifest horizon=3，got {ctx.horizon}"

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_cli_lift_test_horizon_from_manifest(_tp1, mp)

    # -- 原 test_cli_lift_test_horizon_flag_overrides_manifest --
    def _section_2_test_cli_lift_test_horizon_flag_overrides_manifest(tmp_path, mp):
        import factorzen.cli.main as cli_main
        import factorzen.discovery.lift_test as lt_mod
        from factorzen.cli.main import build_parser

        sa = _write_session_with_holdout(
            tmp_path,
            "h3",
            expressions=["rank(close)"],
            holdout_start="2024-06-01",
            holdout_end="2024-08-31",
            horizon=3,
        )
        mp.setattr(
            cli_main,
            "_prepare_agent_mining_data",
            lambda args: (_fake_daily(), None, {}),
        )
        captured: list = []

        def fake_lift(gray, **kw):
            captured.append(kw)
            return [
                {
                    "expression": g.get("expression"),
                    "lift": 0.0,
                    "lift_se": 0.0,
                    "lift_second_half": 0.0,
                    "baseline": 0.0,
                    "passed": False,
                }
                for g in gray
            ]

        patch_cli_lift_pre_gates(mp)
        mp.setattr(lt_mod, "run_lift_tests", fake_lift)

        args = build_parser().parse_args(
            [
                "factor-library",
                "lift-test",
                "--session",
                str(sa),
                "--start",
                "20200101",
                "--end",
                "20241231",
                "--set",
                "library_root=" + str(tmp_path / "lib"),
                "--set", "horizon=7",
            ]
        )
        rc = cli_main._cmd_factor_library_lift_test(args)
        assert rc == 0
        assert captured[0]["ctx"].horizon == 7

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_cli_lift_test_horizon_flag_overrides_manifest(_tp2, mp)

    # -- 原 test_cli_lift_test_horizon_params_key --
    def _section_3_test_cli_lift_test_horizon_params_key(tmp_path, mp):
        import factorzen.cli.main as cli_main
        import factorzen.discovery.lift_test as lt_mod
        from factorzen.cli.main import build_parser

        sa = _write_session_with_holdout(
            tmp_path,
            "hp",
            expressions=["rank(vol)"],
            holdout_start="2024-06-01",
            holdout_end="2024-08-31",
            horizon=4,
            horizon_in_params=True,
        )
        mp.setattr(
            cli_main,
            "_prepare_agent_mining_data",
            lambda args: (_fake_daily(), None, {}),
        )
        captured: list = []

        def fake_lift(gray, **kw):
            captured.append(kw)
            return [
                {
                    "expression": g.get("expression"),
                    "lift": 0.0,
                    "lift_se": 0.0,
                    "lift_second_half": 0.0,
                    "baseline": 0.0,
                    "passed": False,
                }
                for g in gray
            ]

        patch_cli_lift_pre_gates(mp)
        mp.setattr(lt_mod, "run_lift_tests", fake_lift)

        args = build_parser().parse_args(
            [
                "factor-library",
                "lift-test",
                "--session",
                str(sa),
                "--start",
                "20200101",
                "--end",
                "20241231",
                "--set",
                "library_root=" + str(tmp_path / "lib"),
            ]
        )
        rc = cli_main._cmd_factor_library_lift_test(args)
        assert rc == 0
        assert captured[0]["ctx"].horizon == 4

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_cli_lift_test_horizon_params_key(_tp3, mp)

    # -- 原 test_cli_lift_test_multi_session_horizon_mismatch_warns --
    def _section_4_test_cli_lift_test_multi_session_horizon_mismatch_warns(tmp_path, mp, capsys):
        import factorzen.cli.main as cli_main
        import factorzen.discovery.lift_test as lt_mod
        from factorzen.cli.main import build_parser

        sa = _write_session_with_holdout(
            tmp_path, "a", expressions=["e1"],
            holdout_start="2024-01-01", holdout_end="2024-03-31", horizon=3,
        )
        sb = _write_session_with_holdout(
            tmp_path, "b", expressions=["e2"],
            holdout_start="2024-10-01", holdout_end="2024-12-31", horizon=10,
        )
        mp.setattr(
            cli_main,
            "_prepare_agent_mining_data",
            lambda args: (_fake_daily(), None, {}),
        )
        captured: list = []

        def fake_lift(gray, **kw):
            captured.append(kw)
            return [
                {
                    "expression": g.get("expression"),
                    "lift": 0.0, "lift_se": 0.0, "lift_second_half": 0.0,
                    "baseline": 0.0, "passed": False,
                }
                for g in gray
            ]

        patch_cli_lift_pre_gates(mp)
        mp.setattr(lt_mod, "run_lift_tests", fake_lift)

        args = build_parser().parse_args(
            [
                "factor-library", "lift-test",
                "--session", str(sa), str(sb),
                "--start", "20200101", "--end", "20241231",
                "--set", "library_root=" + str(tmp_path / "lib"),
            ]
        )
        rc = cli_main._cmd_factor_library_lift_test(args)
        assert rc == 0
        # base_ctx 统一 resolved_horizon=3（首 session）
        assert all(c["ctx"].horizon == 3 for c in captured)
        err = capsys.readouterr().err
        assert "horizon" in err.lower() or "不一致" in err

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_cli_lift_test_multi_session_horizon_mismatch_warns(_tp4, mp, capsys)



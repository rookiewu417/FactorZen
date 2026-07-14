"""多 session lift-test：按各自 admission/holdout 窗分组评分，禁止并集窗。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl

# ── 纯函数单测 ───────────────────────────────────────────────────────────────


def test_group_lift_candidates_by_admission_two_windows_no_union():
    """跨 session 去重 + 按窗分组；重复 expr 归首次窗，绝无并集窗。"""
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


def test_group_lift_candidates_dedup_keeps_first_window():
    """同 expression 跨 session 首次出现胜出；被去重者连同其窗丢弃。"""
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


def test_group_lift_candidates_same_window_merges():
    """相同 admission 窗的多个 session 合并为一组（去重后）。"""
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


def test_group_lift_candidates_skips_empty_expression():
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


def test_group_lift_candidates_stable_order_by_first_appearance():
    """分组顺序 = 首次出现的窗顺序；组内候选 = 首次出现顺序。"""
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


# ── CLI seam：多 session 分窗调用 run_lift_tests ─────────────────────────────


def _write_session_with_holdout(
    root: Path,
    name: str,
    *,
    expressions: list[str],
    holdout_start: str,
    holdout_end: str,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    attempts = [
        {
            "expression": expr,
            "reject_category": "gray_zone",
            "residual_ic_train": 0.006,
            "n_residual_holdout_days": 100,
        }
        for expr in expressions
    ]
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "holdout_start": holdout_start,
                "holdout_end": holdout_end,
                "end": holdout_end,
                "attempts": attempts,
                "candidates": [],
            }
        ),
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


def test_cli_lift_test_multi_session_calls_per_admission_window(
    tmp_path, monkeypatch, capsys
):
    """两个不相交 holdout session → run_lift_tests 调 2 次，绝不用并集窗。"""
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

    monkeypatch.setattr(
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

    monkeypatch.setattr(lt_mod, "run_lift_tests", fake_lift)

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
            "--library-root",
            str(tmp_path / "lib"),
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


def test_cli_lift_test_flag_admission_forces_single_group(tmp_path, monkeypatch):
    """--admission-start/end 任一非空 → 所有候选同一旗标窗，只调一次。"""
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

    monkeypatch.setattr(
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

    monkeypatch.setattr(lt_mod, "run_lift_tests", fake_lift)

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
            "--library-root",
            str(tmp_path / "lib"),
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


def test_cli_lift_test_single_session_one_group(tmp_path, monkeypatch):
    """单 session：一次 run_lift_tests，窗 = 该 session holdout（零回归）。"""
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
    monkeypatch.setattr(
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

    monkeypatch.setattr(lt_mod, "run_lift_tests", fake_lift)

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
            "--library-root",
            str(tmp_path / "lib"),
        ]
    )
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert len(calls) == 1
    assert calls[0]["n"] == 1
    assert calls[0]["start"] == "2024-06-01"
    assert calls[0]["end"] == "2024-08-31"

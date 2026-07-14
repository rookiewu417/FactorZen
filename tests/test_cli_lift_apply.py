"""CLI：lift-test 默认 dry-run / --apply / --se-mult；rebuild lift 复审 fail-loudly。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest


def _write_gray_session(tmp_path: Path) -> Path:
    """写一个含 gray_zone 候选的假 session 目录（含 manifest.json）。"""
    run_dir = tmp_path / "session1"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "attempts": [
                    {
                        "expression": "rank(close)",
                        "reject_category": "gray_zone",
                        "residual_ic_train": 0.006,
                        "n_residual_holdout_days": 100,
                    },
                ],
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


def _patch_lift_deps(monkeypatch, *, upsert_calls: list | None = None):
    """mock 数据装配 / run_lift_tests / upsert_lift_admissions，不碰真实数据。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.factor_library as fl
    import factorzen.discovery.lift_test as lt_mod

    monkeypatch.setattr(
        cli_main,
        "_prepare_agent_mining_data",
        lambda args: (_fake_daily(), None),
    )

    def fake_lift(gray, **kw):
        return [
            {
                "expression": "rank(close)",
                "lift": 0.005,
                "lift_se": 0.001,
                "lift_second_half": 0.004,
                "baseline": 0.02,
                "passed": True,
                "candidate_rank_ic": 0.025,
            }
        ]

    monkeypatch.setattr(lt_mod, "run_lift_tests", fake_lift)

    calls = upsert_calls if upsert_calls is not None else []

    def fake_upsert(results, **kw):
        calls.append({"results": results, **kw})
        return {"added_active": 0, "added_probation": 1, "rejected": 0}

    monkeypatch.setattr(fl, "upsert_lift_admissions", fake_upsert)
    return calls


# ── D2 / D3：lift-test 默认 dry-run / --apply / --se-mult ─────────────────────


def test_lift_test_default_is_dry_run(tmp_path, monkeypatch):
    """不带旗标时默认 dry-run，upsert_lift_admissions 不被调用。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser

    run_dir = _write_gray_session(tmp_path)
    upsert_calls: list = []
    _patch_lift_deps(monkeypatch, upsert_calls=upsert_calls)

    args = build_parser().parse_args(
        [
            "factor-library",
            "lift-test",
            "--session",
            str(run_dir),
            "--market",
            "ashare",
            "--start",
            "20200101",
            "--end",
            "20201231",
            "--library-root",
            str(tmp_path / "lib"),
        ]
    )
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert upsert_calls == []
    man = json.loads((run_dir / "lift_test_manifest.json").read_text(encoding="utf-8"))
    assert man["dry_run"] is True


def test_lift_test_apply_writes_library(tmp_path, monkeypatch, capsys):
    """--apply 时调用 upsert_lift_admissions 一次。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser

    run_dir = _write_gray_session(tmp_path)
    upsert_calls: list = []
    _patch_lift_deps(monkeypatch, upsert_calls=upsert_calls)

    args = build_parser().parse_args(
        [
            "factor-library",
            "lift-test",
            "--session",
            str(run_dir),
            "--market",
            "ashare",
            "--start",
            "20200101",
            "--end",
            "20201231",
            "--library-root",
            str(tmp_path / "lib"),
            "--apply",
        ]
    )
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert len(upsert_calls) == 1
    man = json.loads((run_dir / "lift_test_manifest.json").read_text(encoding="utf-8"))
    assert man["dry_run"] is False
    out = capsys.readouterr().out
    assert "入库" in out


def test_lift_test_apply_and_dry_run_mutually_exclusive():
    """--apply 与 --dry-run 互斥，argparse 报错 exit 2。"""
    from factorzen.cli.main import build_parser

    p = build_parser()
    with pytest.raises(SystemExit) as ei:
        p.parse_args(
            [
                "factor-library",
                "lift-test",
                "--session",
                "workspace/x",
                "--start",
                "20200101",
                "--end",
                "20201231",
                "--apply",
                "--dry-run",
            ]
        )
    assert ei.value.code == 2


def test_lift_test_se_mult_forwarded(tmp_path, monkeypatch):
    """--apply --se-mult 2.0 时 upsert 收到 se_mult==2.0。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser

    run_dir = _write_gray_session(tmp_path)
    upsert_calls: list = []
    _patch_lift_deps(monkeypatch, upsert_calls=upsert_calls)

    args = build_parser().parse_args(
        [
            "factor-library",
            "lift-test",
            "--session",
            str(run_dir),
            "--market",
            "ashare",
            "--start",
            "20200101",
            "--end",
            "20201231",
            "--library-root",
            str(tmp_path / "lib"),
            "--apply",
            "--se-mult",
            "2.0",
        ]
    )
    assert args.se_mult == 2.0
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert len(upsert_calls) == 1
    assert upsert_calls[0]["se_mult"] == 2.0


def test_lift_test_dry_run_message_mentions_apply(tmp_path, monkeypatch, capsys):
    """dry-run 输出应引导用户加 --apply 写库。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser

    run_dir = _write_gray_session(tmp_path)
    _patch_lift_deps(monkeypatch)

    args = build_parser().parse_args(
        [
            "factor-library",
            "lift-test",
            "--session",
            str(run_dir),
            "--start",
            "20200101",
            "--end",
            "20201231",
            "--library-root",
            str(tmp_path / "lib"),
        ]
    )
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run：通过" in out
    assert "--apply" in out


# ── D1：rebuild fail-loudly ──────────────────────────────────────────────────


def test_rebuild_fail_loudly_on_lift_review_error(monkeypatch, capsys):
    """lift_review_error 非 None → stderr 报错 + return 1。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser
    from factorzen.discovery.factor_library import UpsertResult

    monkeypatch.setattr(
        cli_main,
        "_prepare_agent_mining_data",
        lambda args: (_fake_daily(), None),
    )

    import factorzen.discovery.factor_library as fl

    monkeypatch.setattr(fl, "collect_source_expressions", lambda market: [])
    monkeypatch.setattr(
        fl, "build_library_evaluator", lambda *a, **k: (lambda *x, **y: {}, None)
    )
    monkeypatch.setattr(
        fl,
        "rebuild",
        lambda *a, **k: UpsertResult(
            added=0, updated=0, correlated=0, skipped=0,
            lift_review_error="RuntimeError: x",
        ),
    )

    args = build_parser().parse_args(
        [
            "factor-library",
            "rebuild",
            "--market",
            "ashare",
            "--start",
            "20200101",
            "--end",
            "20201231",
        ]
    )
    rc = cli_main._cmd_factor_library_rebuild(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "lift 轨复审失败" in err
    assert "RuntimeError: x" in err
    assert "不完整" in err


def test_rebuild_ok_when_no_lift_review_error(monkeypatch):
    """lift_review_error=None → return 0。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser
    from factorzen.discovery.factor_library import UpsertResult

    monkeypatch.setattr(
        cli_main,
        "_prepare_agent_mining_data",
        lambda args: (_fake_daily(), None),
    )

    import factorzen.discovery.factor_library as fl

    monkeypatch.setattr(fl, "collect_source_expressions", lambda market: [])
    monkeypatch.setattr(
        fl, "build_library_evaluator", lambda *a, **k: (lambda *x, **y: {}, None)
    )
    monkeypatch.setattr(
        fl,
        "rebuild",
        lambda *a, **k: UpsertResult(added=1, updated=0, correlated=0, skipped=0),
    )

    args = build_parser().parse_args(
        [
            "factor-library",
            "rebuild",
            "--market",
            "ashare",
            "--start",
            "20200101",
            "--end",
            "20201231",
        ]
    )
    rc = cli_main._cmd_factor_library_rebuild(args)
    assert rc == 0

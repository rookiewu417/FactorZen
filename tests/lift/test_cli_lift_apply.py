"""合并自: test_cli_lift_apply.py, test_lift_alignment.py
目标: test_cli_lift_apply.py

--- 来源 test_cli_lift_apply.py ---
test_cli_lift_apply.py：CLI：lift-test 默认 dry-run / --apply / --se-mult；rebuild lift 复审 fail-loudly。
test_cli_lift_w1w2.py：W1c / W2b / W0-fix-2：CLI lift-test 默认 top_m、组门、覆盖过滤。

--- 来源 test_lift_alignment.py ---
test_cli_lift_alignment.py：CLI ``factor-library lift-test`` 批处理路径：候选面板与挖掘装配同源 + top_m 全测。
test_lift_date_alignment.py：lift 链路 trade_date 形态对齐回归。
"""

from __future__ import annotations

import datetime as dt
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# ==== 来自 test_cli_lift_apply.py ====
# ==== 来自 test_cli_lift_apply.py ====

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
                        "residual_ic_train": 0.02,  # ≥ DEFAULT_GRAY_IC_FLOOR（避开 sub-floor 防呆）
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
    from tests._cli_lift_mocks import patch_cli_lift_pre_gates

    monkeypatch.setattr(
        cli_main,
        "_prepare_agent_mining_data",
        lambda args: (_fake_daily(), None, {}),
    )
    patch_cli_lift_pre_gates(monkeypatch)

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
        lambda args: (_fake_daily(), None, {}),
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
        lambda args: (_fake_daily(), None, {}),
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


# ==== 来自 test_cli_lift_w1w2.py ====

def _write_session(tmp_path: Path, n: int = 25, *, residual_base: float = 0.02) -> Path:
    run_dir = tmp_path / "run1"
    run_dir.mkdir(exist_ok=True)
    attempts = [
        {
            "expression": f"rank(ts_mean(close, {i + 1}))",
            "reject_category": "lift_queue",
            "residual_ic_train": residual_base + 0.0001 * (n - i),
            "n_residual_holdout_days": 100,
        }
        for i in range(n)
    ]
    (run_dir / "manifest.json").write_text(
        json.dumps({"attempts": attempts, "candidates": []}),
        encoding="utf-8",
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


def test_cli_top_m_0_tests_all(tmp_path, monkeypatch):
    """--top-m 0 → 全测。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.lift_test as lt_mod

    run_dir = _write_session(tmp_path, n=25)
    lib_root = tmp_path / "lib"
    lib_root.mkdir()
    args = _base_args(run_dir, lib_root, extra=["--top-m", "0"])

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
    monkeypatch.setattr(
        lt_mod, "run_group_lift",
        lambda queue, **k: {
            "lift": 0.01, "lift_se": 0.001, "error": None,
            "lift_metric": "residual_ic_v1",
        },
    )
    called = {"n_cands": 0}

    def fake_lift(gray, **kw):
        called["n_cands"] = len(gray)
        return [
            {"expression": c.get("expression"), "lift": 0.0, "passed": False}
            for c in gray
        ]

    monkeypatch.setattr(lt_mod, "run_lift_tests", fake_lift)
    monkeypatch.setattr(lt_mod, "resolve_lift_workers", lambda w: 2)

    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert called["n_cands"] == 25
    man = json.loads((run_dir / "lift_test_manifest.json").read_text(encoding="utf-8"))
    assert "truncated_from" not in man or man.get("truncated_from") is None
    assert man["top_m"] == 0


def test_cli_group_gate_fail_skips_run_lift_tests(tmp_path, monkeypatch):
    """组 lift 不过 → run_lift_tests 不被调用，manifest 有 lift_group。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.lift_test as lt_mod

    run_dir = _write_session(tmp_path, n=3)
    lib_root = tmp_path / "lib"
    lib_root.mkdir()
    args = _base_args(run_dir, lib_root, extra=["--top-m", "0"])

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
    monkeypatch.setattr(
        lt_mod, "run_group_lift",
        lambda queue, **k: {
            "lift": 0.0001, "lift_se": 0.01, "error": None,
            "lift_metric": "residual_ic_v1",
        },
    )
    # se_mult=1 → bar=max(0.001, 0.01)=0.01 > lift 0.0001 → 不过
    lift_calls = []
    monkeypatch.setattr(
        lt_mod, "run_lift_tests",
        lambda *a, **k: lift_calls.append(1) or [],
    )
    monkeypatch.setattr(lt_mod, "resolve_lift_workers", lambda w: 2)

    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert lift_calls == [], "组门不过不应调 run_lift_tests"
    man = json.loads((run_dir / "lift_test_manifest.json").read_text(encoding="utf-8"))
    assert man["lift_group"] is not None
    assert man["lift_group"].get("lift") == 0.0001
    assert all(
        str(r.get("error") or "").startswith("group_gate")
        for r in man["results"]
    )


def test_cli_coverage_filter_before_group_gate(tmp_path, monkeypatch):
    """覆盖 30 天剔除、200 天保留；dropped 进 manifest；低覆盖不进组门。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.lift_test as lt_mod

    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "attempts": [
            {
                "expression": "low_cov",
                "reject_category": "lift_queue",
                "residual_ic_train": 0.02,
                "n_residual_holdout_days": 100,
            },
            {
                "expression": "high_cov",
                "reject_category": "lift_queue",
                "residual_ic_train": 0.015,
                "n_residual_holdout_days": 100,
            },
        ],
        "candidates": [],
    }), encoding="utf-8")
    lib_root = tmp_path / "lib"
    lib_root.mkdir()
    args = _base_args(run_dir, lib_root, extra=["--top-m", "0"])

    monkeypatch.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda a: (pl.DataFrame({
            "trade_date": [date(2020, 1, 2)], "ts_code": ["000001.SZ"],
            "close": [10.0], "close_adj": [10.0],
        }), None, {}),
    )

    def mat(expr):
        n = 30 if expr == "low_cov" else 200
        return pl.DataFrame({
            "trade_date": [date(2020, 1, 1 + (i % 28)) for i in range(n)],
            "ts_code": ["000001.SZ"] * n,
            "factor_value": [float(i) for i in range(n)],
        })

    # 不 mock filter——走真函数，但注入 materializer 经 memo 困难；
    # 直接 mock filter 结果更稳，并断言组门只收 high_cov
    def fake_filter(cands, **k):
        kept, dropped = [], []
        for c in cands:
            if c.get("expression") == "low_cov":
                dropped.append({
                    "expression": "low_cov", "n_oos_days": 30, "error": "holdout_coverage",
                })
            else:
                kept.append(c)
        return kept, dropped

    monkeypatch.setattr(lt_mod, "filter_candidates_by_coverage", fake_filter)
    group_queues = []

    def fake_group(queue, **k):
        group_queues.append([c.get("expression") for c in queue])
        return {
            "lift": 0.01, "lift_se": 0.001, "error": None,
            "lift_metric": "residual_ic_v1",
        }

    monkeypatch.setattr(lt_mod, "run_group_lift", fake_group)
    monkeypatch.setattr(
        lt_mod, "run_lift_tests",
        lambda gray, **k: [
            {"expression": c.get("expression"), "lift": 0.002, "passed": False}
            for c in gray
        ],
    )
    monkeypatch.setattr(lt_mod, "resolve_lift_workers", lambda w: 2)

    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert group_queues == [["high_cov"]]
    man = json.loads((run_dir / "lift_test_manifest.json").read_text(encoding="utf-8"))
    assert any(d["expression"] == "low_cov" for d in man["lift_dropped_coverage"])
    assert all(r.get("expression") != "low_cov" for r in man["results"])


def test_group_gate_ok_unit():
    from factorzen.discovery.lift_test import group_gate_ok

    ok, bar = group_gate_ok(
        {"lift": 0.01, "lift_se": 0.002, "error": None},
        threshold=0.001, lift_se_mult=1.0,
    )
    assert ok is True
    assert abs(bar - 0.002) < 1e-12

    ok2, _ = group_gate_ok(
        {"lift": 0.01, "lift_se": None, "error": None},
        threshold=0.001, lift_se_mult=1.0,
    )
    assert ok2 is False  # SE 缺失不过

    ok3, _ = group_gate_ok(
        {"lift": 0.0005, "lift_se": 0.001, "error": None},
        threshold=0.001, lift_se_mult=1.0,
    )
    assert ok3 is False


def test_filter_candidates_by_coverage_unit():
    from factorzen.discovery.lift_test import filter_candidates_by_coverage

    def mat(expr):
        n = 30 if expr == "low" else 200
        return pl.DataFrame({
            "trade_date": [f"2020{1 + i // 28:02d}{1 + i % 28:02d}" for i in range(n)],
            "ts_code": ["000001.SZ"] * n,
            "factor_value": [1.0] * n,
        })

    cands = [
        {"expression": "low", "residual_ic_train": 0.02},
        {"expression": "high", "residual_ic_train": 0.015},
    ]
    kept, dropped = filter_candidates_by_coverage(
        cands, materialize_candidate=mat, holdout_start=None,
    )
    assert [c["expression"] for c in kept] == ["high"]
    assert dropped[0]["expression"] == "low"
    assert dropped[0]["error"] == "holdout_coverage"
    assert dropped[0]["n_oos_days"] == 30


def test_run_lift_tests_elapsed_s(monkeypatch):
    """W2c：每候选结果含 elapsed_s（float 秒）。residual_ic_v1：无 combine_fn。"""
    from factorzen.discovery.lift_test import run_lift_tests

    ret = pl.DataFrame({
        "trade_date": ["20200102", "20200103"],
        "ts_code": ["000001.SZ", "000001.SZ"],
        "ret": [0.01, -0.01],
    })
    active = {
        "base": pl.DataFrame({
            "trade_date": ["20200102", "20200103"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "factor_value": [1.0, 2.0],
        }),
    }
    cand = pl.DataFrame({
        "trade_date": ["20200102", "20200103"],
        "ts_code": ["000001.SZ", "000001.SZ"],
        "factor_value": [0.5, 1.5],
    })

    rows = run_lift_tests(
        [{"expression": "cand_a", "residual_ic_train": 0.02}],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        lift_workers=1,
        top_m=None,
    )
    assert len(rows) == 1
    assert "elapsed_s" in rows[0]
    assert isinstance(rows[0]["elapsed_s"], float)
    assert rows[0]["elapsed_s"] >= 0.0

# ==== 来自 test_lift_alignment.py ====
# ==== 来自 test_cli_lift_alignment.py ====

def _trading_dates(n: int, start: date = date(2024, 1, 2)) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _sparse_event_daily(n_days: int = 40, n_stocks: int = 8) -> pl.DataFrame:
    """合成日线 + 稀疏龙虎榜叶子（条件 fill-0 语义的已 attach 结果）。

    - 大部分行 top_list_flag=0（已知日未上榜）
    - 少数行 =1（事件）
    - 首日 null（lag 后无前值）——与 attach_flows 真实语义一致
    """
    dates = _trading_dates(n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for di, d in enumerate(dates):
        for si, code in enumerate(codes):
            # 稀疏事件：约 1/8 股票 × 每隔 7 日有 flag=1
            is_event = (di > 0) and (di % 7 == 0) and (si % 4 == 0)
            if di == 0:
                flag, net = None, None
            else:
                flag = 1.0 if is_event else 0.0
                net = 0.15 if is_event else 0.0
            rows.append({
                "trade_date": d,
                "ts_code": code,
                "close": 10.0 + si * 0.1,
                "close_adj": 10.0 + si * 0.1,
                "open": 10.0, "high": 10.5, "low": 9.5,
                "open_adj": 10.0, "high_adj": 10.5, "low_adj": 9.5,
                "vol": 1e5, "amount": 1e5,
                "circ_mv": 1e6, "total_mv": 2e6,
                "turnover_rate": 1.0,
                "top_list_flag": flag,
                "top_list_net_buy": net,
                "margin_buy_ratio": 0.01 if di > 0 else None,
                "holder_num_chg": 0.0 if di > 0 else None,
            })
    return pl.DataFrame(rows)


def _write_lift_queue_session(tmp_path: Path, expressions: list[str]) -> Path:
    run_dir = tmp_path / "run_session"
    run_dir.mkdir()
    attempts = [
        {
            "expression": e,
            "reject_category": "lift_queue",
            # 基数 0.02：25 条全在 DEFAULT_GRAY_IC_FLOOR 之上，本测试测截断非地板
            "residual_ic_train": 0.02 - i * 0.0001,
            "ic_train": 0.001,
            "n_residual_holdout_days": 100,
        }
        for i, e in enumerate(expressions)
    ]
    (run_dir / "manifest.json").write_text(
        json.dumps({"attempts": attempts, "candidates": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_dir


def _mining_path_nonzero_rows(daily: pl.DataFrame, expr: str) -> int:
    """挖掘路径口径：preprocess → factor_df_from_prepped 非空行数。"""
    from factorzen.discovery.evaluation import _factor_df_from_prepped, _preprocess_daily
    from factorzen.discovery.expression import parse_expr

    prepped = _preprocess_daily(daily)
    node = parse_expr(expr)
    panel = _factor_df_from_prepped(node, prepped)
    return int(panel.height)


# ── a. 稀疏事件叶子：CLI 物化覆盖 = 挖掘装配物化覆盖 ─────────────────────────


def test_cli_lift_sparse_event_panel_matches_mining_path(tmp_path, monkeypatch, capsys):
    """从 parser 最外层进 lift-test；候选面板非空行数与挖掘路径一致。

    同时断言 CLI 调用的是 ``_prepare_agent_mining_data``（与 mine team/agent 同源，
    不许另起 loader）。
    """
    import factorzen.cli.main as cli_main
    import factorzen.discovery.lift_test as lt_mod
    from factorzen.cli.main import build_parser
    from factorzen.discovery.lift_test import _materializer_from_prepped

    expr = "mul(delay(top_list_flag, 1), max(top_list_net_buy, 0.0))"
    daily = _sparse_event_daily()
    n_mining = _mining_path_nonzero_rows(daily, expr)
    assert n_mining > 0, "合成帧应产生非空候选面板（否则测试本身无效）"

    run_dir = _write_lift_queue_session(tmp_path, [expr])
    prep_calls: list = []

    def tracking_prepare(args):
        prep_calls.append({
            "start": getattr(args, "start", None),
            "end": getattr(args, "end", None),
            "universe": getattr(args, "universe", None),
        })
        return daily, None, {"membership_mode": None}

    monkeypatch.setattr(cli_main, "_prepare_agent_mining_data", tracking_prepare)

    captured: dict = {}

    def capturing_run_lift_tests(gray, **kw):
        captured["n_gray"] = len(gray)
        captured["top_m"] = kw.get("top_m")
        ctx = kw.get("ctx")
        assert ctx is not None, "CLI 必须构造 LiftEvalContext"
        # 与 lift 内部默认 materializer 同路径：prepped 上物化
        mat = _materializer_from_prepped(ctx.prepped, ctx.leaf_map)
        panel = mat(expr)
        assert panel is not None and not panel.is_empty()
        captured["n_cli"] = int(panel.height)
        # 叶子列必须仍在 prepped 上（装配未丢事件叶子）
        for col in ("top_list_flag", "top_list_net_buy"):
            assert col in ctx.prepped.columns, f"prepped 缺叶子 {col}"
        return [{
            "expression": expr,
            "lift": 0.0,
            "baseline": 0.01,
            "passed": False,
            "candidate_rank_ic": 0.01,
            "n_input": len(gray),
            "n_selected": len(gray),
        }]

    from tests._cli_lift_mocks import patch_cli_lift_pre_gates
    patch_cli_lift_pre_gates(monkeypatch)
    monkeypatch.setattr(lt_mod, "run_lift_tests", capturing_run_lift_tests)

    # parser 最外层（禁止 inspect.signature）
    args = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--market", "ashare",
        "--start", "20240102",
        "--end", "20240301",
        "--universe", "csi300",
        "--dry-run",
        "--library-root", str(tmp_path / "lib"),
        "--top-m", "0",  # 全测逃生口（默认 20）
    ])
    assert args.func.__name__ == "_cmd_factor_library_lift_test"
    assert args.top_m == 0

    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert len(prep_calls) == 1, "必须经 _prepare_agent_mining_data 装配"
    assert captured.get("n_cli") == n_mining, (
        f"CLI 物化非空行 {captured.get('n_cli')} ≠ 挖掘路径 {n_mining}"
    )
    assert captured.get("top_m") is None


def test_cli_and_mine_team_share_prepare_fn(monkeypatch, tmp_path):
    """mine team 与 lift-test 共用同一 ``_prepare_agent_mining_data`` 绑定。

    从 parser 最外层分别进两条命令，断言命中同一 tracking 替身（同源、非复制第二份）。
    """
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser

    daily = _sparse_event_daily(n_days=5, n_stocks=2)
    hits: list[str] = []

    def tracking_prepare(args):
        hits.append(getattr(args, "factor_library_command", None)
                    or getattr(args, "mine_command", None)
                    or "unknown")
        return daily, None, {}

    monkeypatch.setattr(cli_main, "_prepare_agent_mining_data", tracking_prepare)

    # lift-test
    run_dir = _write_lift_queue_session(tmp_path, ["rank(close)"])
    import factorzen.discovery.lift_test as lt_mod
    from tests._cli_lift_mocks import patch_cli_lift_pre_gates

    patch_cli_lift_pre_gates(monkeypatch)
    monkeypatch.setattr(
        lt_mod, "run_lift_tests",
        lambda gray, **kw: [{
            "expression": "rank(close)", "lift": None, "baseline": None,
            "passed": False,
        }],
    )
    args_lt = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--start", "20240102", "--end", "20240301",
        "--library-root", str(tmp_path / "lib"),
        "--dry-run",
    ])
    assert cli_main._cmd_factor_library_lift_test(args_lt) == 0

    # mine team（短路 run_team_mine，只验证装配入口）
    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine_team.run_team_mine",
        lambda *a, **k: {"n_candidates": 0, "n_trials": 0, "run_dir": str(tmp_path)},
    )
    args_mt = build_parser().parse_args([
        "mine", "team", "--start", "20240102", "--end", "20240301",
    ])
    assert cli_main._cmd_mine_team(args_mt) == 0

    assert len(hits) == 2, f"两条路径都应调用同源 prepare，got {hits}"


# ── b. top_m 默认 20 截断 + --top-m 0 全测 + 显式截断告警 ────────────────────


def test_cli_lift_top_m_default_truncates_to_20(tmp_path, monkeypatch, capsys):
    """默认 top_m=20：38 候选截到 20 进 run_lift_tests，stderr 截断告警。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.lift_test as lt_mod
    from factorzen.cli.main import build_parser
    from tests._cli_lift_mocks import patch_cli_lift_pre_gates

    # 唯一 expression 串（去重后仍 38 个）；mock lift 不物化
    exprs = [f"add(rank(close), {i}.0)" for i in range(38)]
    run_dir = _write_lift_queue_session(tmp_path, exprs)

    monkeypatch.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda args: (_sparse_event_daily(n_days=5, n_stocks=2), None, {}),
    )
    patch_cli_lift_pre_gates(monkeypatch)
    seen: dict = {}

    def fake_lift(gray, **kw):
        seen["n"] = len(gray)
        seen["top_m"] = kw.get("top_m")
        return [
            {
                "expression": g["expression"], "lift": -0.0001, "baseline": 0.01,
                "passed": False, "n_input": len(gray),
                "n_selected": len(gray),
            }
            for g in gray
        ]

    monkeypatch.setattr(lt_mod, "run_lift_tests", fake_lift)

    args = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--start", "20240102", "--end", "20240301",
        "--library-root", str(tmp_path / "lib"),
        "--dry-run",
    ])
    assert args.top_m == 20
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert seen["top_m"] is None  # CLI 已截断
    assert seen["n"] == 20

    err = capsys.readouterr().err
    assert "--top-m=20" in err
    assert "truncated_from=38" in err


def test_cli_lift_top_m_explicit_truncates_with_warning(tmp_path, monkeypatch, capsys):
    """显式 ``--top-m 10``：截断并打印告警行（no silent caps）。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.lift_test as lt_mod
    from factorzen.cli.main import build_parser
    from tests._cli_lift_mocks import patch_cli_lift_pre_gates

    exprs = [f"add(rank(close), {i}.0)" for i in range(38)]
    run_dir = _write_lift_queue_session(tmp_path, exprs)

    monkeypatch.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda args: (_sparse_event_daily(n_days=5, n_stocks=2), None, {}),
    )
    patch_cli_lift_pre_gates(monkeypatch)
    seen: dict = {}

    def fake_lift(gray, **kw):
        seen["top_m"] = kw.get("top_m")
        seen["n_input"] = len(gray)
        return [
            {
                "expression": g["expression"], "lift": -0.0001, "baseline": 0.01,
                "passed": False, "n_input": len(gray), "n_selected": len(gray),
            }
            for g in gray
        ]

    monkeypatch.setattr(lt_mod, "run_lift_tests", fake_lift)

    args = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--start", "20240102", "--end", "20240301",
        "--library-root", str(tmp_path / "lib"),
        "--top-m", "10",
        "--dry-run",
    ])
    assert args.top_m == 10
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert seen["top_m"] is None  # CLI 已截断，run_lift_tests 收 kept
    assert seen["n_input"] == 10

    err = capsys.readouterr().err
    assert "--top-m=10" in err
    assert "截断" in err
    assert "truncated_from=38" in err


def test_cli_lift_parser_top_m_default_is_20():
    """parser 契约：不传 --top-m → args.top_m == 20；--top-m 0 为全测逃生口。"""
    from factorzen.cli.main import build_parser

    args = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", "workspace/x",
        "--start", "20200101", "--end", "20201231",
    ])
    assert args.top_m == 20
    args0 = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", "workspace/x",
        "--start", "20200101", "--end", "20201231",
        "--top-m", "0",
    ])
    assert args0.top_m == 0


# ==== 来自 test_lift_date_alignment.py ====

def _codes(n: int = 12) -> list[str]:
    return [f"{600000 + i:06d}.SH" for i in range(n)]


def _panels(dates: list, *, col: str = "factor_value", value_col: str = "ret"):
    """构造 (因子面板, 收益面板)；因子值与收益单调同向 → 每日 IC 恒为 +1。"""
    codes = _codes()
    fac = pl.DataFrame([
        {"trade_date": d, "ts_code": c, col: float(j)}
        for d in dates for j, c in enumerate(codes)
    ])
    ret = pl.DataFrame([
        {"trade_date": d, "ts_code": c, value_col: float(j) * 0.5}
        for d in dates for j, c in enumerate(codes)
    ])
    return fac, ret


# ── 1. join 形态对齐：Date 候选面板不得被静默丢空 ────────────────────────────


def test_daily_oos_rank_ic_date_candidate_joins_utf8_returns():
    """候选 pl.Date × 收益 Utf8(ISO)——生产真实组合，必须匹配上。"""
    from factorzen.discovery.lift_test import _daily_oos_rank_ic

    days = [dt.date(2026, 4, 5), dt.date(2026, 4, 7)]
    fac, ret = _panels(days)
    # 复刻 _build_ret_panel：收益侧显式 cast Utf8
    ret = ret.with_columns(pl.col("trade_date").cast(pl.Utf8))
    assert fac.schema["trade_date"] == pl.Date

    out = _daily_oos_rank_ic(fac, ret)
    assert out.height == 2, f"Date 候选面板被静默丢空: {out}"
    # 因子与收益严格同序 → 每日 spearman = 1.0
    assert all(abs(v - 1.0) < 1e-12 for v in out["ic"].to_list())


def test_daily_oos_rank_ic_both_date_panels():
    """两侧都 pl.Date 也必须匹配（cast 后形态须一致）。"""
    from factorzen.discovery.lift_test import _daily_oos_rank_ic

    days = [dt.date(2026, 4, 5), dt.date(2026, 4, 7)]
    fac, ret = _panels(days)
    out = _daily_oos_rank_ic(fac, ret)
    assert out.height == 2, f"两侧 Date 被静默丢空: {out}"


# ── 2. admission 窗形态：生产窗串必须真的裁到正确日集 ────────────────────────


def test_admission_window_accepts_production_iso_bounds():
    """窗界用 cli._lift_admission_str 的真实产出（YYYY-MM-DD）。"""
    from factorzen.cli.main import _lift_admission_str
    from factorzen.discovery.lift_test import _daily_oos_rank_ic

    days = [dt.date(2026, 4, 5), dt.date(2026, 4, 7), dt.date(2026, 4, 15)]
    fac, ret = _panels(days)
    ret = ret.with_columns(pl.col("trade_date").cast(pl.Utf8))

    end = _lift_admission_str(dt.date(2026, 4, 10))
    assert end == "2026-04-10"  # 契约锚定：变了说明上游改了形态

    out = _daily_oos_rank_ic(fac, ret, end=end)
    # 4/5 与 4/7 在窗内，4/15 在窗外
    assert out.height == 2, f"窗内日被误裁: {out}"

    start = _lift_admission_str(dt.date(2026, 4, 6))
    out2 = _daily_oos_rank_ic(fac, ret, start=start, end=end)
    assert out2.height == 1, f"闭区间窗错行: {out2}"


def test_admission_window_accepts_compact_bounds_equivalently():
    """紧凑 YYYYMMDD 与带横杠 ISO 必须裁出同一日集（形态无关）。"""
    from factorzen.discovery.lift_test import _daily_oos_rank_ic

    days = [dt.date(2026, 4, 5), dt.date(2026, 4, 7), dt.date(2026, 4, 15)]
    fac, ret = _panels(days)
    ret = ret.with_columns(pl.col("trade_date").cast(pl.Utf8))

    iso = _daily_oos_rank_ic(fac, ret, end="2026-04-10")["ic"].to_list()
    compact = _daily_oos_rank_ic(fac, ret, end="20260410")["ic"].to_list()
    assert iso == compact and len(iso) == 2


# ── 3. 残差侧同契约 ──────────────────────────────────────────────────────────


def test_daily_residual_rank_ic_window_format_agnostic():
    """残差日序列的窗过滤对两种日期形态等价，且输出 ISO。"""
    from factorzen.discovery.residual import (
        build_library_panel,
        daily_residual_rank_ic,
    )

    rng = np.random.default_rng(7)
    days = [dt.date(2024, 2, 5), dt.date(2024, 2, 6), dt.date(2024, 2, 7)]
    codes = _codes(45)
    lib_m = rng.normal(0, 1, size=(3, 45))
    cand_m = lib_m + rng.normal(0, 0.8, size=(3, 45))
    fwd_m = cand_m + rng.normal(0, 0.3, size=(3, 45))

    def _long(M, col):
        return pl.DataFrame([
            {"trade_date": d, "ts_code": c, col: float(M[i, j])}
            for i, d in enumerate(days) for j, c in enumerate(codes)
        ])

    panel = build_library_panel({"lib": _long(lib_m, "factor_value")})
    assert panel is not None
    cand = _long(cand_m, "factor_value")
    fwd = _long(fwd_m, "fwd_ret_1d")

    iso = daily_residual_rank_ic(
        cand, panel, fwd, start="2024-02-06", end="2024-02-06",
    )
    compact = daily_residual_rank_ic(
        cand, panel, fwd, start="20240206", end="20240206",
    )
    assert iso.height == 1, f"ISO 窗裁错: {iso}"
    assert compact.height == 1, f"紧凑窗裁错: {compact}"
    assert iso["ic"].to_list() == compact["ic"].to_list()
    # 输出形态锚定 ISO（与 _lift_admission_str / 库内 scored_* 既有形态一致）
    assert iso["trade_date"].to_list() == ["2024-02-06"]


def test_daily_residual_rank_ic_joins_date_candidate_with_utf8_returns():
    """候选 pl.Date × 收益 Utf8——残差引擎的生产真实组合，不得抛 SchemaError。

    候选面板由 ``_materializer_from_prepped`` 产出（prepped 帧原生 pl.Date），
    收益面板由 ``_build_ret_panel`` 显式 ``cast(pl.Utf8)``。旧实现直接 join
    两个不同 dtype 的键 → ``SchemaError``（2026-07-15 apply 全灭事故同款）。
    """
    from factorzen.discovery.residual import (
        build_library_panel,
        daily_residual_rank_ic,
    )

    rng = np.random.default_rng(3)
    days = [dt.date(2024, 2, 5), dt.date(2024, 2, 6), dt.date(2024, 2, 7)]
    codes = _codes(45)

    def _long(M, col):
        return pl.DataFrame([
            {"trade_date": d, "ts_code": c, col: float(M[i, j])}
            for i, d in enumerate(days) for j, c in enumerate(codes)
        ])

    lib_m = rng.normal(0, 1, size=(3, 45))
    cand_m = lib_m + rng.normal(0, 0.8, size=(3, 45))
    fwd_m = cand_m + rng.normal(0, 0.3, size=(3, 45))

    panel = build_library_panel({"lib": _long(lib_m, "factor_value")})
    assert panel is not None
    cand = _long(cand_m, "factor_value")
    fwd_date = _long(fwd_m, "ret")
    # 复刻 _build_ret_panel：收益侧 cast Utf8
    fwd_utf8 = fwd_date.with_columns(pl.col("trade_date").cast(pl.Utf8))
    assert cand.schema["trade_date"] == pl.Date

    same = daily_residual_rank_ic(cand, panel, fwd_date, ret_col="ret")
    mixed = daily_residual_rank_ic(cand, panel, fwd_utf8, ret_col="ret")
    assert same.height == 3
    assert mixed.height == 3, f"Date×Utf8 未对齐: {mixed}"
    # 形态对齐不得改变数值
    assert same["ic"].to_list() == mixed["ic"].to_list()


# ── 4. 真实后果：admission_ic 不得因形态错配退化成 0.0 ───────────────────────


def test_admission_ic_not_silently_zero_for_date_panels():
    """端到端：pl.Date 候选面板下 admission_ic 必须是真实 IC，不是空帧 0.0。

    这是库内 2 条 lift 轨记录 ``admission_ic == 0.0`` 的直接回归锚。
    """
    from factorzen.discovery.lift_test import _daily_oos_rank_ic, _mean_ic

    days = [dt.date(2026, 4, 5), dt.date(2026, 4, 7)]
    fac, ret = _panels(days)
    ret = ret.with_columns(pl.col("trade_date").cast(pl.Utf8))

    admission_ic = _mean_ic(_daily_oos_rank_ic(fac, ret))
    assert admission_ic != 0.0, "admission_ic 退化为空帧哨兵 0.0（方向权威失效）"
    assert abs(admission_ic - 1.0) < 1e-12


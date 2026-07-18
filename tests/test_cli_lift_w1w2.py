"""W1c / W2b / W0-fix-2：CLI lift-test 默认 top_m、组门、覆盖过滤。"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl


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


def test_parser_default_top_m_is_20():
    from factorzen.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "factor-library", "lift-test",
        "--session", "/tmp/x",
        "--market", "ashare",
        "--start", "20200101",
        "--end", "20201231",
    ])
    assert args.top_m == 20


def test_cli_top_m_default_truncates_to_20(tmp_path, monkeypatch, capsys):
    """>20 候选 → 只测 20，manifest 含 truncated_from。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.lift_test as lt_mod

    run_dir = _write_session(tmp_path, n=25)
    lib_root = tmp_path / "lib"
    lib_root.mkdir()
    args = _base_args(run_dir, lib_root)  # 默认 top_m=20

    monkeypatch.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda a: (pl.DataFrame({
            "trade_date": [date(2020, 1, 2)], "ts_code": ["000001.SZ"],
            "close": [10.0], "close_adj": [10.0],
        }), None, {}),
    )

    # 覆盖门：全部放行
    monkeypatch.setattr(
        lt_mod, "filter_candidates_by_coverage",
        lambda cands, **k: (list(cands), []),
    )
    # 组门过（residual_ic_v1：无 base_daily）
    monkeypatch.setattr(
        lt_mod, "run_group_lift",
        lambda queue, **k: {
            "lift": 0.01, "lift_se": 0.001, "error": None,
            "lift_metric": "residual_ic_v1",
        },
    )
    called = {"n": 0, "top_m": "unset", "n_cands": 0}

    def fake_lift(gray, **kw):
        called["n"] += 1
        called["top_m"] = kw.get("top_m")
        called["n_cands"] = len(gray)
        return [
            {
                "expression": c.get("expression"),
                "lift": 0.002, "baseline": 0.02, "passed": False,
                "elapsed_s": 0.01,
            }
            for c in gray
        ]

    monkeypatch.setattr(lt_mod, "run_lift_tests", fake_lift)
    monkeypatch.setattr(lt_mod, "resolve_lift_workers", lambda w: 2)

    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert called["n"] == 1
    assert called["n_cands"] == 20
    assert called["top_m"] is None  # CLI 已截断，run_lift_tests 全测 kept
    man = json.loads((run_dir / "lift_test_manifest.json").read_text(encoding="utf-8"))
    assert man["truncated_from"] == 25
    assert man["top_m"] == 20
    assert man["n_tested"] == 20


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

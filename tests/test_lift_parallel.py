"""lift 并行 / residual 引擎确定性 / CLI workers 透传。

全部离线小面板；不碰真实数据。
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import polars as pl

from tests._cli_lift_mocks import patch_cli_lift_pre_gates


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


def test_parallel_vs_serial_row_identical():
    """workers=4 与 workers=1 结果逐行一致（残差路径确定性）。"""
    from factorzen.discovery.lift_test import run_lift_tests

    active, cand, ret = _panels()
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
        assert a.get("lift_metric") == "residual_ic_v1"


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


def test_no_base_daily_in_group_result():
    """成功路径不再返回 base_daily（残差口径无基线序列）。"""
    from factorzen.discovery.lift_test import run_group_lift

    active, cand, ret = _panels()
    out = run_group_lift(
        [{"expression": "c0", "residual_ic_train": 0.01}],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
    )
    assert out["error"] is None
    assert "base_daily" not in out
    assert out["lift"] is not None
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
        "--lift-workers", "3",
        "--dry-run",
        "--library-root", str(lib_root),
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

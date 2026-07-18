"""lift_metric 落库 provenance：新旧口径可区分（阶段 C）。"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl


def _meta(**kw):
    base = {
        "session_dir": "sess/metric",
        "run_id": "run_metric",
        "universe": "csi300",
        "horizon": 5,
        "eval_start": "20200101",
        "eval_end": "20260101",
        "git_sha": "deadbeef",
        "now": "2026-07-18",
    }
    base.update(kw)
    return base


def test_upsert_lift_admissions_persists_lift_metric(tmp_path):
    """run_lift_tests → upsert_lift_admissions → FactorRecord.lift_metric == residual_ic_v1。"""
    from factorzen.discovery.factor_library import load_library, upsert_lift_admissions
    from factorzen.discovery.lift_test import LiftEvalContext, run_lift_tests

    dates: list[str] = []
    d = date(2024, 1, 2)
    while len(dates) < 50:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    n_stocks = 40  # residual 日守卫 max(30, k+10)
    active = {
        "lib_a": pl.DataFrame({
            "trade_date": [dd for dd in dates for _ in range(n_stocks)],
            "ts_code": [f"{s:04d}.SZ" for _ in dates for s in range(n_stocks)],
            "factor_value": [float(s) for _ in dates for s in range(n_stocks)],
        }),
    }
    ret = pl.DataFrame({
        "trade_date": [dd for dd in dates for _ in range(n_stocks)],
        "ts_code": [f"{s:04d}.SZ" for _ in dates for s in range(n_stocks)],
        "ret": [0.01 * s for _ in dates for s in range(n_stocks)],
    })
    cand = pl.DataFrame({
        "trade_date": [dd for dd in dates for _ in range(n_stocks)],
        "ts_code": [f"{s:04d}.SZ" for _ in dates for s in range(n_stocks)],
        "factor_value": [float(s) + 0.5 for _ in dates for s in range(n_stocks)],
    })

    ctx = LiftEvalContext(
        market="ashare",
        prepped=pl.DataFrame({
            "trade_date": ["x"], "ts_code": ["y"], "close": [1.0],
        }),
        leaf_map=None,
        horizon=5,
        admission_start="20240120",
        admission_end="20240315",
        profile_name="ashare_v1",
    )
    rows = run_lift_tests(
        [{"expression": "rank(close)", "residual_ic_train": 0.02, "ic_train": 0.03}],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        block_days=12,
        threshold=0.001,
        ctx=ctx,
        lift_workers=1,
    )
    assert rows[0].get("lift_metric") == "residual_ic_v1"
    # 强制 passed 以便 upsert 写入（本测关心 provenance 落盘）
    rows[0]["lift"] = 0.05
    rows[0]["lift_se"] = 0.001
    rows[0]["lift_first_half"] = 0.04
    rows[0]["lift_second_half"] = 0.06
    rows[0]["passed"] = True

    upsert_lift_admissions(
        [rows[0]],
        market="ashare",
        root=str(tmp_path),
        meta=_meta(),
        threshold=0.001,
        se_mult=1.0,
        allow_active=True,
    )
    rec = load_library("ashare", root=str(tmp_path))[0]
    assert rec.lift_metric == "residual_ic_v1"
    assert rec.lift_metric is not None
    assert rec.n_lib_factors == rows[0].get("n_lib_factors") == 1


def test_old_jsonl_missing_lift_metric_reads_as_none():
    """旧口径记录（无 lift_metric 键）读回为 None，不崩——新旧可区分。"""
    from factorzen.discovery.factor_library import FactorRecord

    old = {
        "expression": "rank(close)",
        "market": "ashare",
        "ic_train": 0.05,
        "status": "active",
        "admission_track": "lift",
        "lift": 0.01,
        "horizon": 5,
        "eval_start": "20200101",
        "eval_end": "20240101",
        # 故意无 lift_metric / n_lib_factors
    }
    rec = FactorRecord.from_dict(old)
    assert rec.lift_metric is None
    assert rec.n_lib_factors is None
    assert rec.lift == 0.01
    # 再 round-trip 不丢其它字段、不填假值
    again = FactorRecord.from_dict(rec.to_dict())
    assert again.lift_metric is None
    assert again.n_lib_factors is None

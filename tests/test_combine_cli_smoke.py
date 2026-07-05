"""fz combine run CLI 冒烟。"""
from __future__ import annotations

import numpy as np
import polars as pl

from factorzen.cli.main import main


def _write_inputs(tmp_path, n_days=120, n_stocks=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = [f"2025{1 + i // 28:02d}{1 + i % 28:02d}" for i in range(n_days)]
    ra, rb, rr = [], [], []
    for d in dates:
        fa = rng.standard_normal(n_stocks)
        fb = rng.standard_normal(n_stocks)
        ret = 0.8 * fa - 0.4 * fb + rng.standard_normal(n_stocks) * 0.3
        for s in range(n_stocks):
            c = f"{s:04d}.SZ"
            ra.append({"trade_date": d, "ts_code": c, "factor_value": float(fa[s])})
            rb.append({"trade_date": d, "ts_code": c, "factor_value": float(fb[s])})
            rr.append({"trade_date": d, "ts_code": c, "ret": float(ret[s])})
    fa_p = tmp_path / "fa.parquet"
    fb_p = tmp_path / "fb.parquet"
    ret_p = tmp_path / "ret.parquet"
    pl.DataFrame(ra).write_parquet(fa_p)
    pl.DataFrame(rb).write_parquet(fb_p)
    pl.DataFrame(rr).write_parquet(ret_p)
    return fa_p, fb_p, ret_p


def test_fz_combine_run_smoke(tmp_path):
    fa_p, fb_p, ret_p = _write_inputs(tmp_path)
    out = tmp_path / "out"
    rc = main(
        [
            "combine", "run",
            "--factor", str(fa_p),
            "--factor", str(fb_p),
            "--ret", str(ret_p),
            "--train-days", "60",
            "--test-days", "20",
            "--purge-days", "5",
            "--methods", "equal_weight,lgbm",
            "--seed", "0",
            "--run-id", "cli1",
            "--out-dir", str(out),
        ]
    )
    assert rc == 0
    run_dir = out / "cli1"
    assert (run_dir / "comparison.csv").exists()
    assert (run_dir / "report.md").exists()
    comp = pl.read_csv(run_dir / "comparison.csv")
    assert set(comp["method"].to_list()) == {"equal_weight", "lgbm"}

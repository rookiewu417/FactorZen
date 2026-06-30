import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl


def _mock(n_stocks=12, n_days=290, seed=3):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2023, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    daily = pl.DataFrame([{"trade_date": dd, "ts_code": c, "pct_chg": float(rng.standard_normal() * 2)}
                          for c in codes for dd in days])
    db = pl.DataFrame([{"trade_date": dd, "ts_code": c,
                        "total_mv": float(abs(rng.standard_normal()) * 1e9 + 5e9),
                        "pb": float(abs(rng.standard_normal()) + 1.5),
                        "pe_ttm": float(abs(rng.standard_normal()) * 10 + 15)}
                       for c in codes for dd in days])
    stocks = pl.DataFrame({"ts_code": codes,
                           "industry": [["银行", "医药", "电子"][i % 3] for i in range(n_stocks)]})
    return daily, db, stocks, days[260].strftime("%Y%m%d"), days[-1].strftime("%Y%m%d")


def test_run_risk_build_writes_artifacts(tmp_path: Path):
    from factorzen.pipelines.risk_build import run_risk_build
    daily, db, stocks, start, end = _mock()
    res = run_risk_build(daily, db, stocks, start, end, out_dir=str(tmp_path), run_id="t1")
    run_dir = Path(res["run_dir"])
    for f in ["exposures.parquet", "factor_covariance.parquet", "specific_risk.parquet",
              "factor_returns.parquet", "risk_summary.csv", "manifest.json"]:
        assert (run_dir / f).exists(), f
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert 0.0 <= manifest["r_squared"] <= 1.0
    assert "factor_names" in manifest

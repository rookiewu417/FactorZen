import json
from datetime import date
from pathlib import Path

import polars as pl

from factorzen.execution.drivers import run_replay


def _write_portfolio_run(dir_: Path, sig: date, code: str, w: float) -> str:
    dir_.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"ts_code": [code], "target_weight": [w]}).write_parquet(dir_ / "weights.parquet")
    (dir_ / "manifest.json").write_text(json.dumps(
        {"signal_date": sig.isoformat(), "status": "optimal"}))
    return str(dir_)


def _daily(codes, dates):
    rows = []
    for d in dates:
        for c in codes:
            rows.append({"trade_date": d, "ts_code": c, "open": 10.0,
                         "pre_close": 10.0, "close": 10.0, "vol": 1e6})
    return pl.DataFrame(rows)


def test_replay_produces_session_artifacts(tmp_path: Path):
    dates = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    daily = _daily(["A.SZ"], dates)
    rd = _write_portfolio_run(tmp_path / "pf", date(2026, 1, 5), "A.SZ", 0.5)
    out = run_replay(
        session_dir=tmp_path / "sess", portfolio_run_dirs=[rd],
        daily=daily, initial_cash=1_000_000.0,
        from_date=dates[0], to_date=dates[-1], seed=0,
    )
    assert out["n_steps"] >= 1
    assert (tmp_path / "sess" / "nav.parquet").exists()
    assert (tmp_path / "sess" / "ledger.parquet").exists()
    assert (tmp_path / "sess" / "manifest.json").exists()
    nav = pl.read_parquet(tmp_path / "sess" / "nav.parquet")
    assert nav.height == out["n_steps"]

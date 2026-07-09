# tests/test_agent_evaluation.py
import datetime as dt

import numpy as np
import polars as pl

from factorzen.agents.evaluation import evaluate_expressions
from factorzen.discovery.scoring import DataBundle


def _mock_daily(n_stocks=40, n_days=120, seed=1):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px,
                         "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


def test_evaluate_valid_expressions():
    daily = _mock_daily()
    bundle = DataBundle.build(daily)
    out = evaluate_expressions(["ts_mean(close,5)", "rank(vol)"], daily, bundle)
    assert len(out) == 2
    for r in out:
        assert r["compile_ok"] is True
        assert r["ic_train"] is not None        # 真算出了 IC（非 None）
        assert isinstance(r["ic_train"], float)


def test_evaluate_rejects_illegal_expression():
    daily = _mock_daily()
    bundle = DataBundle.build(daily)
    out = evaluate_expressions(["this_is_not_an_operator(close)", "ts_mean(close,5)"], daily, bundle)
    assert out[0]["compile_ok"] is False and out[0]["error"]   # 非法被拒，记错误
    assert out[0]["ic_train"] is None
    assert out[1]["compile_ok"] is True                         # 合法的照常评估

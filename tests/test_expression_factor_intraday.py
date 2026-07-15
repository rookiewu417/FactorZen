"""ExpressionFactor.compute 与 evaluate_materialized 对 i_* 逐值一致。"""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from factorzen.core.feature_schema import INTRADAY_FEATURES


def test_expression_factor_i_rv_matches_evaluate_materialized(monkeypatch):
    """monkeypatch attach_intraday 注入面板 → compute 与直算逐值一致。"""
    from factorzen.discovery.derived import add_derived_columns
    from factorzen.discovery.expression import evaluate_materialized, parse_expr
    from factorzen.discovery.factor import ExpressionFactor

    dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)]
    codes = ["000001.SZ", "000002.SZ"]
    rows = []
    for c in codes:
        for i, d in enumerate(dates):
            rows.append({
                "trade_date": d,
                "ts_code": c,
                "close": 10.0 + i,
                "close_adj": 10.0 + i,
                "open": 10.0,
                "open_adj": 10.0,
                "high": 11.0,
                "high_adj": 11.0,
                "low": 9.0,
                "low_adj": 9.0,
                "pre_close": 10.0,
                "vol": 1e5,
                "amount": 1e6,
            })
    daily = pl.DataFrame(rows)

    panel_rows = []
    for c in codes:
        for i, d in enumerate(dates):
            r = {"trade_date": d, "ts_code": c}
            for leaf in sorted(INTRADAY_FEATURES):
                r[leaf] = 0.01 * (i + 1) + (0.001 if c.endswith("1.SZ") else 0.002)
            panel_rows.append(r)
    panel = pl.DataFrame(panel_rows)

    def _fake_attach(d, **kw):
        # 模拟 require=True 注入
        have = [c for c in sorted(INTRADAY_FEATURES) if c in panel.columns]
        sel = panel.select(["trade_date", "ts_code", *have])
        drop = [c for c in have if c in d.columns]
        if drop:
            d = d.drop(drop)
        return d.join(sel, on=["trade_date", "ts_code"], how="left")

    monkeypatch.setattr(
        "factorzen.daily.data.intraday.attach_intraday", _fake_attach,
    )
    # ExpressionFactor 内 from factorzen.daily.data.intraday import attach_intraday
    # 局部 import 在调用时解析，patch 源模块即可

    class _Ctx:
        start = "20240102"
        end = "20240104"

        @property
        def daily(self):
            return daily.lazy()

        @property
        def daily_basic(self):
            return pl.DataFrame({
                "trade_date": dates * len(codes),
                "ts_code": [c for c in codes for _ in dates],
                "circ_mv": [1e6] * (len(dates) * len(codes)),
            }).lazy()

    expr = "rank(i_rv)"
    fac = ExpressionFactor(expr, mined_name="i_rv_rank")
    out = fac.compute(_Ctx())
    assert out.height > 0
    assert out["factor_value"].null_count() < out.height

    # 直算：attach → sort → derived → evaluate
    attached = _fake_attach(daily)
    prepped = add_derived_columns(attached.sort(["ts_code", "trade_date"]))
    node = parse_expr(expr)
    direct = prepped.with_columns(
        evaluate_materialized(node, prepped).alias("factor_value")
    ).filter(
        pl.col("trade_date") >= dt.date(2024, 1, 2)
    ).select(["trade_date", "ts_code", "factor_value"]).filter(
        pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite()
    )

    a = out.sort(["ts_code", "trade_date"])
    b = direct.sort(["ts_code", "trade_date"])
    assert a.height == b.height
    for va, vb in zip(a["factor_value"].to_list(), b["factor_value"].to_list(), strict=True):
        assert va == pytest.approx(vb, abs=1e-9, nan_ok=True)

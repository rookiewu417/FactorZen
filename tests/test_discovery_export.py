from __future__ import annotations

from pathlib import Path


def test_render_factor_file_contains_expression():
    from factorzen.discovery.export import render_factor_file
    text = render_factor_file("rank(ts_mean(close, 5))", "mined_demo")
    assert "rank(ts_mean(close, 5))" in text
    assert "class" in text and "ExpressionFactor" in text
    assert 'name = "mined_demo"' in text


def test_exported_file_is_importable_and_consistent(tmp_path: Path):
    """导出的 .py 能 import，且其因子 compute 与直接用 ExpressionFactor 一致。"""
    import importlib.util
    from dataclasses import dataclass, field
    from datetime import date, timedelta

    import numpy as np
    import polars as pl

    from factorzen.discovery.export import export_candidate
    from factorzen.discovery.factor import ExpressionFactor

    path = export_candidate("ts_mean(close, 5)", "mined_demo", str(tmp_path))
    assert path.exists()
    spec = importlib.util.spec_from_file_location("mined_demo", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "MinedDemo")

    # 造数据 + mock ctx（复用 Task 4 风格）
    rng = np.random.default_rng(1)
    start = date(2024, 1, 2)
    days, d = [], start
    while len(days) < 70:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for s in [f"{i:06d}.SH" for i in range(6)]:
        p = 10.0
        for day in days:
            p = float(max(p * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": day, "ts_code": s, "close": p, "open": p, "high": p,
                         "low": p, "close_adj": p, "open_adj": p, "high_adj": p, "low_adj": p,
                         "amount": 1e7, "vol": 1e5})
    lf = pl.DataFrame(rows).lazy()

    @dataclass
    class MockCtx:
        start: str = "20240301"
        end: str = "20240331"
        required_data: list = field(default_factory=lambda: ["daily", "daily_basic"])
        lookback_days: int = 30
        universe = None
        snapshot_mode = "daily"
        @property
        def daily(self): return lf
        @property
        def daily_basic(self): return pl.DataFrame({"trade_date": [], "ts_code": []}).lazy()

    direct = ExpressionFactor(expression="ts_mean(close, 5)", mined_name="mined_demo",
                              lookback_days=60).compute(MockCtx())
    assert direct.height > 0
    direct_sorted = direct.sort(["trade_date", "ts_code"])
    exported = mod.MinedDemo().compute(MockCtx()).sort(["trade_date", "ts_code"])
    assert exported.height == direct_sorted.height > 0
    j = direct_sorted.join(exported, on=["trade_date", "ts_code"], suffix="_exp")
    assert j.height == direct_sorted.height
    diff = (j["factor_value"] - j["factor_value_exp"]).abs().max()
    assert diff is not None and diff < 1e-9

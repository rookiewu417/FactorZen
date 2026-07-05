# tests/test_mine_export_alpha.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import polars as pl


def _make_daily_lf(n_stocks=8, n_days=60, seed=42) -> pl.LazyFrame:
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for s in [f"{i:06d}.SH" for i in range(n_stocks)]:
        price = 10.0
        for day in days:
            price = float(max(price * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": day, "ts_code": s, "close": price,
                         "open": price, "high": price, "low": price, "pre_close": price,
                         "close_adj": price, "open_adj": price, "high_adj": price,
                         "low_adj": price,
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
                         "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows).lazy()


@dataclass
class MockCtx:
    start: str = "20240301"
    end: str = "20240301"
    required_data: list = field(default_factory=lambda: ["daily", "daily_basic"])
    lookback_days: int = 40
    universe: list | None = None
    snapshot_mode: str = "daily"
    _daily: pl.LazyFrame | None = None
    _basic: pl.LazyFrame | None = None

    @property
    def daily(self) -> pl.LazyFrame:
        return self._daily

    @property
    def daily_basic(self) -> pl.LazyFrame:
        return self._basic if self._basic is not None else pl.DataFrame(
            {"trade_date": [], "ts_code": []}).lazy()


def test_export_alpha_writes_two_column_parquet(tmp_path):
    """挖掘候选 + 指定日期 → 落 [ts_code, alpha] 两列截面 parquet：非空且值有限。"""
    from factorzen.discovery.export import export_alpha_cross_section

    ctx = MockCtx(_daily=_make_daily_lf())
    out = tmp_path / "alpha.parquet"
    p = export_alpha_cross_section("pct_change(close, 20)", ctx, "20240301", str(out))

    assert p.exists()
    df = pl.read_parquet(p)
    # 恰有 ts_code + alpha 两列
    assert df.columns == ["ts_code", "alpha"]
    assert df.height > 0
    # 值全部有限且非空
    assert df["alpha"].is_finite().all()
    assert df["alpha"].null_count() == 0
    # 单日截面：每只股票至多一行
    assert df["ts_code"].n_unique() == df.height


def test_read_candidate_expression_by_rank(tmp_path):
    """candidates.csv 按 rank 取表达式。"""
    from factorzen.discovery.export import read_candidate_expression

    # 用 polars 写，忠实复现 mining_session 产出的 candidates.csv（含逗号的表达式会被加引号）
    pl.DataFrame({
        "rank": [1, 2],
        "n_trials": [100, 100],
        "expression": ["rank(close)", "ts_mean(close, 5)"],
    }).write_csv(tmp_path / "candidates.csv")
    assert read_candidate_expression(str(tmp_path), 1) == "rank(close)"
    assert read_candidate_expression(str(tmp_path), 2) == "ts_mean(close, 5)"

"""算子库的 property-based 测试(hypothesis)。

对随机维度/随机种子的合成面板,断言算子应满足的不变式:长度守恒、值域、
截面性质、NaN 传播。补齐测试体系短板——example-based 测试易漏的边界由此覆盖。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from factorzen.discovery.expression import evaluate, parse_expr
from factorzen.discovery.operators import LEAF_FEATURES


def _synth(seed: int, n_stocks: int, n_days: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 1)
    rows = []
    for s in range(n_stocks):
        p = 10.0 + s
        for d in range(n_days):
            p = max(1.0, p * (1 + rng.normal(0, 0.02)))
            v = float(rng.uniform(1e5, 1e6))
            rows.append(
                {
                    "ts_code": f"{s:04d}.SZ",
                    "trade_date": start + timedelta(days=d),
                    "close": p, "close_adj": p, "open": p, "open_adj": p,
                    "high": p, "high_adj": p, "low": p, "low_adj": p,
                    "vol": v, "amount": p * v,
                }
            )
    return pl.DataFrame(rows).sort(["ts_code", "trade_date"])


_DIMS = dict(
    seed=st.integers(0, 10_000),
    n_stocks=st.integers(15, 35),
    n_days=st.integers(12, 30),
)


@given(**_DIMS)
@settings(max_examples=25, deadline=None)
def test_rank_length_preserved_and_in_unit_range(seed, n_stocks, n_days):
    df = _synth(seed, n_stocks, n_days)
    r = evaluate(parse_expr("rank(close)", LEAF_FEATURES), df)
    assert len(r) == df.height  # 长度守恒
    v = r.drop_nulls()
    assert v.is_empty() or bool(((v >= 0) & (v <= 1)).all())  # rank ∈ [0,1]


@given(**_DIMS)
@settings(max_examples=25, deadline=None)
def test_delta_length_preserved(seed, n_stocks, n_days):
    df = _synth(seed, n_stocks, n_days)
    r = evaluate(parse_expr("delta(close, 1)", LEAF_FEATURES), df)
    assert len(r) == df.height


@given(**_DIMS)
@settings(max_examples=25, deadline=None)
def test_zscore_cross_sectional_mean_near_zero(seed, n_stocks, n_days):
    df = _synth(seed, n_stocks, n_days)
    out = df.with_columns(
        evaluate(parse_expr("zscore(close)", LEAF_FEATURES), df).alias("z")
    )
    daily = out.drop_nulls("z").group_by("trade_date").agg(pl.col("z").mean().alias("m"))
    if daily.height:
        assert bool((daily["m"].abs() < 1e-6).all())  # 截面 z-score 均值≈0


@given(**_DIMS)
@settings(max_examples=20, deadline=None)
def test_ts_mean_nan_propagation(seed, n_stocks, n_days):
    """rolling min_samples=3:每股前 2 行必为 null(不产生假值)。"""
    df = _synth(seed, n_stocks, n_days)
    out = df.with_columns(
        evaluate(parse_expr("ts_mean(close, 5)", LEAF_FEATURES), df).alias("m")
    )
    first2 = out.group_by("ts_code", maintain_order=True).head(2)
    assert first2["m"].null_count() == first2.height

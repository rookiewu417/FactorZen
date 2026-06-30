from __future__ import annotations

import numpy as np
import polars as pl
import pytest


def test_round_trip_simple():
    from factorzen.discovery.expression import parse_expr, to_expr_string
    s = "rank(ts_mean(close, 5))"
    assert to_expr_string(parse_expr(s)) == s


def test_round_trip_nested():
    from factorzen.discovery.expression import parse_expr, to_expr_string
    s = "div(ts_mean(close, 5), ts_mean(close, 60))"
    assert to_expr_string(parse_expr(s)) == s


def test_constant_and_feature():
    from factorzen.discovery.expression import parse_expr, to_expr_string
    s = "mul(zscore(pb), 2.0)"
    assert to_expr_string(parse_expr(s)) == s


def test_complexity_counts_nodes():
    from factorzen.discovery.expression import complexity, parse_expr
    # rank(1) + ts_mean(1) + close(1) = 3
    assert complexity(parse_expr("rank(ts_mean(close, 5))")) == 3


def test_feature_names():
    from factorzen.discovery.expression import feature_names, parse_expr
    assert feature_names(parse_expr("div(close, pb)")) == {"close", "pb"}


def test_parse_rejects_unknown_op():
    from factorzen.discovery.expression import parse_expr
    with pytest.raises(ValueError):
        parse_expr("frobnicate(close, 5)")


def test_parse_rejects_unknown_leaf():
    from factorzen.discovery.expression import parse_expr
    with pytest.raises(ValueError):
        parse_expr("frobnicate")  # 无括号 → 叶子路径


def test_round_trip_scientific_constant():
    from factorzen.discovery.expression import Constant, parse_expr, to_expr_string
    s = to_expr_string(Constant(1e-5))   # "1e-05"
    assert parse_expr(s) == Constant(1e-5)


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for code in ["A", "B", "C"]:
        p = 10.0
        for d in range(40):
            p = float(max(p * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": d, "ts_code": code, "close_adj": p,
                         "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows).sort(["ts_code", "trade_date"])


def test_compile_ts_mean_ratio():
    from factorzen.discovery.expression import evaluate, parse_expr
    df = _toy()
    series = evaluate(parse_expr("div(ts_mean(close, 5), ts_mean(close, 20))"), df)
    assert series.len() == df.height
    assert series.drop_nulls().is_finite().all()


def test_compile_cross_sectional_rank_per_date():
    from factorzen.discovery.expression import evaluate, parse_expr
    df = _toy()
    out = df.with_columns(evaluate(parse_expr("rank(close)"), df).alias("r"))
    # 每个 trade_date 截面内 rank 落在 (0,1)
    vals = out.filter(pl.col("trade_date") == 30)["r"].drop_nulls().to_list()
    assert all(0.0 < v < 1.0 for v in vals)

from __future__ import annotations
import numpy as np
import polars as pl


def _toy_df(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for code in ["A", "B"]:
        price = 10.0
        for d in range(30):
            price = float(max(price * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": d, "ts_code": code, "close_adj": price,
                         "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows).sort(["ts_code", "trade_date"])


def test_ts_mean_matches_manual():
    from factorzen.discovery.operators import OPERATORS
    df = _toy_df()
    expr = OPERATORS["ts_mean"].build([pl.col("close_adj")], 5)
    got = df.with_columns(expr.alias("f"))
    manual = df.with_columns(
        pl.col("close_adj").rolling_mean(5, min_samples=3).over("ts_code").alias("m"))
    assert got["f"].to_list() == manual["m"].to_list()


def test_cs_rank_is_within_unit_interval():
    from factorzen.discovery.operators import OPERATORS
    df = _toy_df()
    expr = OPERATORS["rank"].build([pl.col("close_adj")], None)
    got = df.with_columns(expr.alias("r"))["r"].drop_nulls().to_list()
    assert all(0.0 < v < 1.0 for v in got)


def test_arith_add():
    from factorzen.discovery.operators import OPERATORS
    df = _toy_df()
    expr = OPERATORS["add"].build([pl.col("close_adj"), pl.col("vol")], None)
    got = df.with_columns(expr.alias("s"))
    assert got["s"].to_list() == (df["close_adj"] + df["vol"]).to_list()


def test_operator_categories_present():
    from factorzen.discovery.operators import OPERATORS
    cats = {spec.category for spec in OPERATORS.values()}
    assert cats == {"ts", "cs", "arith"}

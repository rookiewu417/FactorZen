from __future__ import annotations
import numpy as np
import polars as pl


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for code in ["A", "B", "C", "D"]:
        p = 10.0
        for d in range(30):
            p = float(max(p * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": d, "ts_code": code, "close_adj": p, "open_adj": p,
                         "high_adj": p, "low_adj": p, "vol": 1e5, "amount": 1e6,
                         "vwap": p, "log_vol": 11.0, "ret_1d": 0.0,
                         "total_mv": 5e9, "circ_mv": 4e9, "pb": 2.0,
                         "pe_ttm": 20.0, "ps_ttm": 3.0, "dv_ttm": 1.0})
    return pl.DataFrame(rows).sort(["ts_code", "trade_date"])


def test_random_expression_is_compilable():
    from factorzen.discovery.search.random_search import random_expression
    from factorzen.discovery.expression import compile_expr, to_expr_string, parse_expr
    df = _toy()
    rng = np.random.default_rng(7)
    for _ in range(50):
        node = random_expression(rng, max_depth=3)
        # 可编译
        out = df.with_columns(compile_expr(node).alias("f"))
        assert "f" in out.columns
        # 可 round-trip
        assert to_expr_string(parse_expr(to_expr_string(node))) == to_expr_string(node)


def test_random_searcher_proposes_distinct():
    from factorzen.discovery.search.random_search import RandomSearcher
    from factorzen.discovery.expression import to_expr_string
    s = RandomSearcher(np.random.default_rng(0), max_depth=3)
    exprs = {to_expr_string(s.propose()) for _ in range(30)}
    assert len(exprs) > 5  # 有多样性

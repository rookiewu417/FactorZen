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


def test_crossover_and_mutate_stay_compilable():
    from factorzen.discovery.search.random_search import random_expression
    from factorzen.discovery.search.genetic import crossover, mutate
    from factorzen.discovery.expression import compile_expr
    df = _toy()
    rng = np.random.default_rng(11)
    for _ in range(40):
        a = random_expression(rng, 3)
        b = random_expression(rng, 3)
        child = crossover(a, b, rng)
        mutant = mutate(child, rng, 3)
        for node in (child, mutant):
            df.with_columns(compile_expr(node).alias("f"))  # 不抛异常即合法


def test_genetic_improves_toy_objective():
    """目标：偏好复杂度小的表达式 → GP 平均复杂度应下降或持平。"""
    from factorzen.discovery.search.genetic import GeneticSearcher
    from factorzen.discovery.expression import complexity
    rng = np.random.default_rng(5)
    gs = GeneticSearcher(rng, max_depth=3)
    best = gs.evolve(lambda node: -complexity(node), pop_size=20, generations=5)
    assert complexity(best[0]) <= 4


def test_genetic_terminates_under_complexity_pressure():
    """即使目标偏好高复杂度（防膨胀过滤压力最大），evolve 也必须在有限时间内终止。"""
    from factorzen.discovery.search.genetic import GeneticSearcher
    from factorzen.discovery.expression import complexity
    rng = np.random.default_rng(13)
    gs = GeneticSearcher(rng, max_depth=3)
    best = gs.evolve(lambda node: float(complexity(node)), pop_size=15, generations=6)
    assert len(best) == 15  # 种群规模维持，未因死循环卡住

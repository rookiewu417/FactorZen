"""MC1 T1/T2: discovery 引擎 leaf 集/leaf→列映射/搜索 可注入(默认 A 股不变)。"""
from __future__ import annotations

import numpy as np
import polars as pl

from factorzen.discovery.expression import (
    Feature,
    compile_expr,
    feature_names,
    parse_expr,
)
from factorzen.discovery.operators import LEAF_FEATURES
from factorzen.discovery.search.genetic import GeneticSearcher
from factorzen.discovery.search.random_search import RandomSearcher, random_expression

_CRYPTO_LEAF_MAP = {
    "close": "close", "vol": "vol", "funding_rate": "funding_rate",
    "open_interest": "open_interest",
}


# ── T1: expression 注入 ───────────────────────────────────────────────────────
def test_compile_default_ashare_unchanged():
    """默认走 A 股 LEAF_FEATURES：close→close_adj。"""
    expr = compile_expr(Feature("close"))
    df = pl.DataFrame({"close_adj": [1.0, 2.0], "close": [9.0, 9.0]})
    assert df.with_columns(expr.alias("x"))["x"].to_list() == [1.0, 2.0]


def test_compile_with_crypto_leaf_map():
    """注入 crypto leaf_map：close→close(无复权)，funding_rate 可编译。"""
    df = pl.DataFrame({"close": [1.0, 2.0], "funding_rate": [0.01, 0.02]})
    close_expr = compile_expr(Feature("close"), leaf_map=_CRYPTO_LEAF_MAP)
    assert df.with_columns(close_expr.alias("x"))["x"].to_list() == [1.0, 2.0]
    fr_expr = compile_expr(Feature("funding_rate"), leaf_map=_CRYPTO_LEAF_MAP)
    assert df.with_columns(fr_expr.alias("x"))["x"].to_list() == [0.01, 0.02]


def test_parse_with_crypto_leaves():
    """注入 crypto leaves：funding_rate 合法解析；默认 A 股拒绝。"""
    node = parse_expr("ts_mean(funding_rate, 3)", leaves=_CRYPTO_LEAF_MAP)
    assert "funding_rate" in feature_names(node)
    # 默认 A 股叶子集不含 funding_rate
    import pytest
    with pytest.raises(ValueError, match="未知叶子"):
        parse_expr("funding_rate")


# ── T2: 搜索注入 ──────────────────────────────────────────────────────────────
def test_random_expression_uses_injected_leaves():
    rng = np.random.default_rng(0)
    crypto_leaves = list(_CRYPTO_LEAF_MAP.keys())
    for _ in range(50):
        node = random_expression(rng, max_depth=3, leaves=crypto_leaves)
        assert feature_names(node) <= set(crypto_leaves)


def test_random_expression_default_ashare():
    rng = np.random.default_rng(0)
    for _ in range(30):
        node = random_expression(rng, max_depth=3)
        assert feature_names(node) <= set(LEAF_FEATURES.keys())


def test_random_searcher_leaves():
    rng = np.random.default_rng(1)
    s = RandomSearcher(rng, max_depth=3, leaves=list(_CRYPTO_LEAF_MAP.keys()))
    for _ in range(30):
        assert feature_names(s.propose()) <= set(_CRYPTO_LEAF_MAP.keys())


def test_genetic_searcher_leaves():
    rng = np.random.default_rng(2)
    gs = GeneticSearcher(rng, max_depth=3, leaves=list(_CRYPTO_LEAF_MAP.keys()))
    pop = gs.evolve(lambda n: -float(len(feature_names(n))), pop_size=12, generations=2)
    for node in pop:
        assert feature_names(node) <= set(_CRYPTO_LEAF_MAP.keys())

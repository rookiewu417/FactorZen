from __future__ import annotations
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
    from factorzen.discovery.expression import parse_expr, complexity
    # rank(1) + ts_mean(1) + close(1) = 3
    assert complexity(parse_expr("rank(ts_mean(close, 5))")) == 3


def test_feature_names():
    from factorzen.discovery.expression import parse_expr, feature_names
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

"""表达式 lookback_days 须按 AST 推导，避免大窗口/嵌套表达式硬编码 60 欠预热。

原 render_factor_file 路径已废除；契约迁至 discovery.factor.lookback_for_expression。
"""
from __future__ import annotations


def test_required_lookback_sums_windows_along_deepest_path():
    from factorzen.discovery.expression import parse_expr, required_lookback

    assert required_lookback(parse_expr("close")) == 0
    assert required_lookback(parse_expr("rank(close)")) == 0            # 截面算子不加窗口
    assert required_lookback(parse_expr("ts_mean(close, 20)")) == 20
    assert required_lookback(parse_expr("ts_mean(delta(close, 5), 20)")) == 25  # 嵌套累加
    # 双子树取最深路径
    assert required_lookback(parse_expr("add(ts_mean(close, 20), ts_mean(close, 60))")) == 60


def test_lookback_for_expression_uses_derived_lookback():
    from factorzen.discovery.factor import lookback_for_expression

    # 小窗口/无窗口 → 下限 60
    assert lookback_for_expression("rank(close)") == 60
    # 大窗口 → 按需放大
    assert lookback_for_expression("ts_mean(close, 120)") == 120
    # 嵌套累加
    assert lookback_for_expression("ts_mean(delta(close, 40), 60)") == 100


def test_lookback_for_expression_malformed_falls_back():
    from factorzen.discovery.factor import lookback_for_expression

    # 畸形表达式不应崩，回退到下限 60
    assert lookback_for_expression("ts_mean(close, )") == 60

"""导出因子的 lookback_days 须按表达式 AST 推导，避免大窗口/嵌套表达式硬编码 60 欠预热。"""
from __future__ import annotations


def test_required_lookback_sums_windows_along_deepest_path():
    from factorzen.discovery.expression import parse_expr, required_lookback

    assert required_lookback(parse_expr("close")) == 0
    assert required_lookback(parse_expr("rank(close)")) == 0            # 截面算子不加窗口
    assert required_lookback(parse_expr("ts_mean(close, 20)")) == 20
    assert required_lookback(parse_expr("ts_mean(delta(close, 5), 20)")) == 25  # 嵌套累加
    # 双子树取最深路径
    assert required_lookback(parse_expr("add(ts_mean(close, 20), ts_mean(close, 60))")) == 60


def test_render_factor_file_uses_derived_lookback():
    from factorzen.discovery.export import render_factor_file

    # 小窗口/无窗口 → 下限 60
    assert "lookback_days = 60" in render_factor_file("rank(close)", "f_small")
    # 大窗口 → 按需放大
    src = render_factor_file("ts_mean(close, 120)", "f_big")
    assert "lookback_days = 120" in src
    # 嵌套累加
    src2 = render_factor_file("ts_mean(delta(close, 40), 60)", "f_nest")
    assert "lookback_days = 100" in src2


def test_render_factor_file_malformed_expression_falls_back():
    from factorzen.discovery.export import render_factor_file
    # 畸形表达式不应崩，回退到下限 60
    assert "lookback_days = 60" in render_factor_file("ts_mean(close, )", "f_bad")

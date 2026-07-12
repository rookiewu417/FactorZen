"""parse_expr 对畸形表达式须抛 ValueError（统一异常契约），而非 IndexError（F3）。

根因：window 算子空参数时 int(raw_args[-1]) 越界抛 IndexError，而所有 LLM 输出解析点
（agents/nodes、team_orchestrator、evaluation…）只 except ValueError → 一条畸形 LLM
输出（截断/幻觉如 'ts_mean()'）崩掉整个挖掘 session。
"""
from __future__ import annotations

import pytest

from factorzen.discovery.expression import (
    LookaheadWindowError,
    is_lookahead_expr,
    parse_expr,
)


@pytest.mark.parametrize("expr", ["ts_mean()", "ts_std()", "delay()"])
def test_window_op_empty_args_raises_valueerror(expr):
    with pytest.raises(ValueError):
        parse_expr(expr)


def test_valid_expression_still_parses():
    node = parse_expr("ts_mean(close, 5)")
    assert node is not None


# ── P0：时序算子窗口 < 1 = 前视/未来函数（违反 PIT 铁律），parse 层根治 ────────────────

@pytest.mark.parametrize("expr", [
    "delay(ret_1d, -1)",                 # 头号污染因子的核心：shift(-1)=明日值
    "ts_sum(delay(ret_1d, -1), 60)",     # A股库原 #1（嵌套前视）
    "delta(close, -5)",                  # 前视差分
    "pct_change(close, -1)",             # 前视变化率
    "ts_mean(close, 0)",                 # 零窗口无意义
    "delay(ret_1d, 0)",                  # 零位移=恒等，无意义
])
def test_negative_or_zero_window_raises_lookahead_error(expr):
    """窗口 <1 → LookaheadWindowError（ValueError 子类，异常契约统一）。"""
    with pytest.raises(LookaheadWindowError):
        parse_expr(expr)
    with pytest.raises(ValueError):        # 子类仍被 except ValueError 接住
        parse_expr(expr)


@pytest.mark.parametrize("expr", [
    "delay(ret_1d, 1)", "ts_sum(delay(ret_1d, 1), 60)", "delta(close, 5)",
    "ts_mean(close, 20)", "ts_corr(close, vol, 10)",
])
def test_positive_window_still_parses(expr):
    assert parse_expr(expr) is not None


def test_is_lookahead_expr_detects_negative_window():
    assert is_lookahead_expr("ts_sum(delay(ret_1d, -1), 60)") is True
    assert is_lookahead_expr("delay(ret_1d, -1)") is True
    assert is_lookahead_expr("delta(close, 0)") is True
    # 干净表达式 → False
    assert is_lookahead_expr("ts_mean(close, 20)") is False
    assert is_lookahead_expr("neg(ret_1d)") is False
    # 解析失败但**非前视**（未知叶子，如别的市场表达式）→ False（不误判成前视）
    assert is_lookahead_expr("delay(funding_rate, 1)") is False
    assert is_lookahead_expr("garbage((") is False

"""parse_expr 对畸形表达式须抛 ValueError（统一异常契约），而非 IndexError（F3）。

根因：window 算子空参数时 int(raw_args[-1]) 越界抛 IndexError，而所有 LLM 输出解析点
（agents/nodes、team_orchestrator、evaluation…）只 except ValueError → 一条畸形 LLM
输出（截断/幻觉如 'ts_mean()'）崩掉整个挖掘 session。
"""
from __future__ import annotations

import pytest

from factorzen.discovery.expression import parse_expr


@pytest.mark.parametrize("expr", ["ts_mean()", "ts_std()", "delay()"])
def test_window_op_empty_args_raises_valueerror(expr):
    with pytest.raises(ValueError):
        parse_expr(expr)


def test_valid_expression_still_parses():
    node = parse_expr("ts_mean(close, 5)")
    assert node is not None

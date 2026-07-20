"""Merged discovery tests: test_expression.py

test_discovery_expression.py：表达式 parse/round-trip/complexity/compile 与截面 rank 求值
test_expression_nested_over.py：嵌套冲突 .over() 求值 bug 的回归测试
test_parse_expr_exception_contract.py：parse_expr 畸形表达式统一抛 ValueError 而非 IndexError（F3）
test_export_lookback.py：表达式 lookback_days 按 AST 推导，避免硬编码预热不足
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from factorzen.discovery.expression import (
    LookaheadWindowError,
    compile_expr,
    evaluate_materialized,
    is_lookahead_expr,
    parse_expr,
)
from factorzen.discovery.operators import OPERATORS


# ==== 来自 test_discovery_expression.py ====
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

# ==== 来自 test_expression_nested_over.py ====
def _frame() -> pl.DataFrame:
    """3 只股票 × 6 天，含 rank/ts_std 需要的叶子列，按 (ts_code, trade_date) 排序。"""
    rows = []
    rets = {
        "A": [0.01, -0.02, 0.03, -0.01, 0.02, 0.00],
        "B": [0.00, 0.01, -0.03, 0.02, 0.01, -0.02],
        "C": [-0.01, 0.02, 0.01, -0.02, 0.03, 0.01],
    }
    for code in ("A", "B", "C"):
        for i, r in enumerate(rets[code]):
            rows.append({
                "ts_code": code, "trade_date": 20230100 + i + 1,
                "close_adj": 10.0 + i + hash(code) % 3, "vol": 1000.0 + 10 * i,
                "ret_1d": r, "pb": 1.0 + 0.1 * i + (hash(code) % 5) * 0.2,
            })
    return pl.DataFrame(rows).sort(["ts_code", "trade_date"])

def test_rank_of_rolling_is_not_all_null():
    """核心回归：rank(ts_std(...)) 曾被算成全 null，修复后必须有非空值。"""
    df = _frame()
    node = parse_expr("rank(ts_std(ret_1d, 3))")

    series = evaluate_materialized(node, df)

    non_null = series.drop_nulls().filter(series.drop_nulls().is_finite())
    assert non_null.len() > 0, "rank(ts_std(...)) 不应全 null（嵌套 over bug）"

def test_nested_over_bug_reproduced_by_old_compile_expr():
    """反例守卫：旧的单嵌套 compile_expr 对同一表达式确实产出全 null（证明测试有判别力）。"""
    df = _frame()
    node = parse_expr("rank(ts_std(ret_1d, 3))")

    old = df.with_columns(compile_expr(node).alias("__f"))["__f"]

    finite = old.is_finite().sum() if old.dtype in (pl.Float64, pl.Float32) else 0
    assert finite == 0, "本测试前提是旧路径全 null；若旧路径已非空则场景变了，需重审"

def test_deep_cross_over_time_via_arith_not_null():
    """截面算子套在算术（含时序子式）外层也不能塌：rank(add(ts_std(ret_1d,3), ts_mean(ret_1d,3)))。"""
    df = _frame()
    node = parse_expr("rank(add(ts_std(ret_1d, 3), ts_mean(ret_1d, 3)))")

    series = evaluate_materialized(node, df)

    assert series.is_finite().sum() > 0

PARITY_EXPRS = [
    "mul(close, vol)",            # 纯算术
    "add(pb, ret_1d)",            # 纯算术
    "rank(pb)",                   # 截面套叶子（无冲突，曾正常）
    "ts_std(ret_1d, 3)",          # 时序套叶子（曾正常）
    "ts_mean(ts_std(ret_1d, 3), 3)",  # 同键嵌套 ts∘ts（曾正常）；窗口须 ≥ _MIN=3
    "neg(rank(pb))",              # 算术套截面
]

def test_parity_on_previously_working_shapes():
    """零漂移：修复前就正常的形状，物化求值必须与旧嵌套求值逐值相等（含 null 位置）。"""
    df = _frame()
    for expr in PARITY_EXPRS:
        node = parse_expr(expr)
        old = df.with_columns(compile_expr(node).alias("__f"))["__f"]
        new = evaluate_materialized(node, df)
        # null 位置一致
        assert old.is_null().to_list() == new.is_null().to_list(), f"{expr}: null 位置漂移"
        # 非空值近似相等
        mask = old.is_not_null() & old.is_finite()
        if mask.sum() > 0:
            o = old.filter(mask).to_list()
            n = new.filter(mask).to_list()
            assert all(abs(a - b) < 1e-9 for a, b in zip(o, n, strict=True)), f"{expr}: 值漂移"

def test_arith_operators_carry_no_over_invariant():
    """不变量：'只物化 ts/cs' 的优化依赖 arith 算子不含 .over()。
    若未来有人给 arith 算子误加 .over()，本测试红，提醒改成物化每个节点。"""
    leaves = [pl.col("ret_1d"), pl.col("pb")]
    for name, spec in OPERATORS.items():
        if spec.category != "arith":
            continue
        expr = spec.build(leaves[: spec.arity], None)
        assert "over" not in str(expr).lower(), f"arith 算子 {name} 不应含 .over()"

# ==== 来自 test_parse_expr_exception_contract.py ====
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

# ==== 来自 test_export_lookback.py ====
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


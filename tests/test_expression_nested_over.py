"""嵌套冲突 .over() 求值 bug 的回归测试。

根因：compile_expr 把整棵树拼成单个嵌套 pl.Expr，当截面算子（rank/zscore, .over("trade_date")）
套在时序滚动算子（ts_std 等, .over("ts_code")）外面时，polars 对分组键冲突的嵌套 .over() 求值为
全 null → 因子 ic 退化成 0、永不入选。修法：evaluate_materialized 在 ts/cs 算子处物化成列，
arith 保持内联，任何 .over() 的输入都是已物化列，永不嵌套。
"""
from __future__ import annotations

import polars as pl

from factorzen.discovery.expression import (
    compile_expr,
    evaluate_materialized,
    parse_expr,
)
from factorzen.discovery.operators import OPERATORS


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

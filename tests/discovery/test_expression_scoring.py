"""
test_expression.py：Merged discovery tests: test_expression.py
test_scoring_bundle.py：Merged discovery tests: test_scoring_bundle.py
"""

from __future__ import annotations

from datetime import (
    date,
    timedelta,
)

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


# ==== 来自 test_expression.py ====
# ==== 来自 test_discovery_expression.py ====
def test_parse_ast_roundtrip_suite():
    """test_round_trip_simple；test_round_trip_nested；test_round_trip_scientific_constant；test_constant_and_feature；test_complexity_counts_nodes；test_feature_names"""
    # -- 原 test_round_trip_simple --
    def _section_0_test_round_trip_simple():
        from factorzen.discovery.expression import parse_expr, to_expr_string
        s = "rank(ts_mean(close, 5))"
        assert to_expr_string(parse_expr(s)) == s

    _section_0_test_round_trip_simple()

    # -- 原 test_round_trip_nested --
    def _section_1_test_round_trip_nested():
        from factorzen.discovery.expression import parse_expr, to_expr_string
        s = "div(ts_mean(close, 5), ts_mean(close, 60))"
        assert to_expr_string(parse_expr(s)) == s

    _section_1_test_round_trip_nested()

    # -- 原 test_round_trip_scientific_constant --
    def _section_2_test_round_trip_scientific_constant():
        from factorzen.discovery.expression import Constant, parse_expr, to_expr_string
        s = to_expr_string(Constant(1e-5))   # "1e-05"
        assert parse_expr(s) == Constant(1e-5)

    _section_2_test_round_trip_scientific_constant()

    # -- 原 test_constant_and_feature --
    def _section_3_test_constant_and_feature():
        from factorzen.discovery.expression import parse_expr, to_expr_string
        s = "mul(zscore(pb), 2.0)"
        assert to_expr_string(parse_expr(s)) == s

    _section_3_test_constant_and_feature()

    # -- 原 test_complexity_counts_nodes --
    def _section_4_test_complexity_counts_nodes():
        from factorzen.discovery.expression import complexity, parse_expr
        # rank(1) + ts_mean(1) + close(1) = 3
        assert complexity(parse_expr("rank(ts_mean(close, 5))")) == 3

    _section_4_test_complexity_counts_nodes()

    # -- 原 test_feature_names --
    def _section_5_test_feature_names():
        from factorzen.discovery.expression import feature_names, parse_expr
        assert feature_names(parse_expr("div(close, pb)")) == {"close", "pb"}

    _section_5_test_feature_names()


def test_parse_rejects_unknown_suite():
    """test_parse_rejects_unknown_op；test_parse_rejects_unknown_leaf"""
    # -- 原 test_parse_rejects_unknown_op --
    def _section_0_test_parse_rejects_unknown_op():
        from factorzen.discovery.expression import parse_expr
        with pytest.raises(ValueError):
            parse_expr("frobnicate(close, 5)")

    _section_0_test_parse_rejects_unknown_op()

    # -- 原 test_parse_rejects_unknown_leaf --
    def _section_1_test_parse_rejects_unknown_leaf():
        from factorzen.discovery.expression import parse_expr
        with pytest.raises(ValueError):
            parse_expr("frobnicate")  # 无括号 → 叶子路径

    _section_1_test_parse_rejects_unknown_leaf()


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

def test_compile_basic_eval_suite():
    """test_compile_ts_mean_ratio；test_compile_cross_sectional_rank_per_date"""
    # -- 原 test_compile_ts_mean_ratio --
    def _section_0_test_compile_ts_mean_ratio():
        from factorzen.discovery.expression import evaluate, parse_expr
        df = _toy()
        series = evaluate(parse_expr("div(ts_mean(close, 5), ts_mean(close, 20))"), df)
        assert series.len() == df.height
        assert series.drop_nulls().is_finite().all()

    _section_0_test_compile_ts_mean_ratio()

    # -- 原 test_compile_cross_sectional_rank_per_date --
    def _section_1_test_compile_cross_sectional_rank_per_date():
        from factorzen.discovery.expression import evaluate, parse_expr
        df = _toy()
        out = df.with_columns(evaluate(parse_expr("rank(close)"), df).alias("r"))
        # 每个 trade_date 截面内 rank 落在 (0,1)
        vals = out.filter(pl.col("trade_date") == 30)["r"].drop_nulls().to_list()
        assert all(0.0 < v < 1.0 for v in vals)

    _section_1_test_compile_cross_sectional_rank_per_date()


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

def test_nested_over_regression_suite():
    """核心回归：rank(ts_std(...)) 曾被算成全 null，修复后必须有非空值。；反例守卫：旧的单嵌套 compile_expr 对同一表达式确实产出全 null（证明测试有判别力）。；截面算子套在算术（含时序子式）外层也不能塌：rank(add(ts_std(ret_1d,3), ts_mean(ret_1d,3)))。；零漂移：修复前就正常的形状，物化求值必须与旧嵌套求值逐值相等（含 null 位置）。；不变量：'只物化 ts/cs' 的优化依赖 arith 算子不含 .over()。"""
    # -- 原 test_rank_of_rolling_is_not_all_null --
    def _section_0_test_rank_of_rolling_is_not_all_null():
        df = _frame()
        node = parse_expr("rank(ts_std(ret_1d, 3))")

        series = evaluate_materialized(node, df)

        non_null = series.drop_nulls().filter(series.drop_nulls().is_finite())
        assert non_null.len() > 0, "rank(ts_std(...)) 不应全 null（嵌套 over bug）"

    _section_0_test_rank_of_rolling_is_not_all_null()

    # -- 原 test_nested_over_bug_reproduced_by_old_compile_expr --
    def _section_1_test_nested_over_bug_reproduced_by_old_compile_expr():
        df = _frame()
        node = parse_expr("rank(ts_std(ret_1d, 3))")

        old = df.with_columns(compile_expr(node).alias("__f"))["__f"]

        finite = old.is_finite().sum() if old.dtype in (pl.Float64, pl.Float32) else 0
        assert finite == 0, "本测试前提是旧路径全 null；若旧路径已非空则场景变了，需重审"

    _section_1_test_nested_over_bug_reproduced_by_old_compile_expr()

    # -- 原 test_deep_cross_over_time_via_arith_not_null --
    def _section_2_test_deep_cross_over_time_via_arith_not_null():
        df = _frame()
        node = parse_expr("rank(add(ts_std(ret_1d, 3), ts_mean(ret_1d, 3)))")

        series = evaluate_materialized(node, df)

        assert series.is_finite().sum() > 0

    _section_2_test_deep_cross_over_time_via_arith_not_null()

    # -- 原 test_parity_on_previously_working_shapes --
    def _section_3_test_parity_on_previously_working_shapes():
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

    _section_3_test_parity_on_previously_working_shapes()

    # -- 原 test_arith_operators_carry_no_over_invariant --
    def _section_4_test_arith_operators_carry_no_over_invariant():
        leaves = [pl.col("ret_1d"), pl.col("pb")]
        for name, spec in OPERATORS.items():
            if spec.category != "arith":
                continue
            expr = spec.build(leaves[: spec.arity], None)
            assert "over" not in str(expr).lower(), f"arith 算子 {name} 不应含 .over()"

    _section_4_test_arith_operators_carry_no_over_invariant()


PARITY_EXPRS = [
    "mul(close, vol)",            # 纯算术
    "add(pb, ret_1d)",            # 纯算术
    "rank(pb)",                   # 截面套叶子（无冲突，曾正常）
    "ts_std(ret_1d, 3)",          # 时序套叶子（曾正常）
    "ts_mean(ts_std(ret_1d, 3), 3)",  # 同键嵌套 ts∘ts（曾正常）；窗口须 ≥ _MIN=3
    "neg(rank(pb))",              # 算术套截面
]


# ==== 来自 test_parse_expr_exception_contract.py ====
@pytest.mark.parametrize("expr", [
    "ts_mean()",
])
def test_window_op_empty_args_raises_valueerror(expr):
    with pytest.raises(ValueError):
        parse_expr(expr)


# ── P0：时序算子窗口 < 1 = 前视/未来函数（违反 PIT 铁律），parse 层根治 ────────────────

@pytest.mark.parametrize("expr", [
    "delay(ret_1d, -1)",                 # 头号污染因子的核心：shift(-1)=明日值
    "ts_sum(delay(ret_1d, -1), 60)",     # A股库原 #1（嵌套前视）
    "ts_mean(close, 0)",                 # 零窗口无意义
])
def test_negative_or_zero_window_raises_lookahead_error(expr):
    """窗口 <1 → LookaheadWindowError（ValueError 子类，异常契约统一）。"""
    with pytest.raises(LookaheadWindowError):
        parse_expr(expr)
    with pytest.raises(ValueError):        # 子类仍被 except ValueError 接住
        parse_expr(expr)

@pytest.mark.parametrize("expr", [
    "delay(ret_1d, 1)", "ts_mean(close, 20)",
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
def test_lookback_suite():
    """test_required_lookback_sums_windows_along_deepest_path；test_lookback_for_expression_uses_derived_lookback；test_lookback_for_expression_malformed_falls_back"""
    # -- 原 test_required_lookback_sums_windows_along_deepest_path --
    def _section_0_test_required_lookback_sums_windows_along_deepest_path():
        from factorzen.discovery.expression import parse_expr, required_lookback

        assert required_lookback(parse_expr("close")) == 0
        assert required_lookback(parse_expr("rank(close)")) == 0            # 截面算子不加窗口
        assert required_lookback(parse_expr("ts_mean(close, 20)")) == 20
        assert required_lookback(parse_expr("ts_mean(delta(close, 5), 20)")) == 25  # 嵌套累加
        # 双子树取最深路径
        assert required_lookback(parse_expr("add(ts_mean(close, 20), ts_mean(close, 60))")) == 60

    _section_0_test_required_lookback_sums_windows_along_deepest_path()

    # -- 原 test_lookback_for_expression_uses_derived_lookback --
    def _section_1_test_lookback_for_expression_uses_derived_lookback():
        from factorzen.discovery.factor import lookback_for_expression

        # 小窗口/无窗口 → 下限 60
        assert lookback_for_expression("rank(close)") == 60
        # 大窗口 → 按需放大
        assert lookback_for_expression("ts_mean(close, 120)") == 120
        # 嵌套累加
        assert lookback_for_expression("ts_mean(delta(close, 40), 60)") == 100

    _section_1_test_lookback_for_expression_uses_derived_lookback()

    # -- 原 test_lookback_for_expression_malformed_falls_back --
    def _section_2_test_lookback_for_expression_malformed_falls_back():
        from factorzen.discovery.factor import lookback_for_expression

        # 畸形表达式不应崩，回退到下限 60
        assert lookback_for_expression("ts_mean(close, )") == 60

    _section_2_test_lookback_for_expression_malformed_falls_back()


# ==== 来自 test_scoring_bundle.py ====
# ==== 来自 test_discovery_scoring.py ====
# tests/test_discovery_scoring.py

def _daily(seed=1, n_stocks=40, n_days=120):
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for s in [f"{i:06d}.SH" for i in range(n_stocks)]:
        p = 10.0
        for day in days:
            p = float(max(p * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": day, "ts_code": s, "close": p, "close_adj": p,
                         "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows)

def _signal_factor_df(daily: pl.DataFrame) -> pl.DataFrame:
    """构造与次日收益正相关的因子（用于验证 IC 为正）。"""
    df = daily.sort(["ts_code", "trade_date"]).with_columns(
        (pl.col("close_adj").shift(-1).over("ts_code") / pl.col("close_adj") - 1.0).alias("fwd"))
    return df.select(["trade_date", "ts_code", pl.col("fwd").alias("factor_value")]).drop_nulls()

def _noisy_signal_factor_df(daily: pl.DataFrame, noise: float = 0.6, seed: int = 3) -> pl.DataFrame:
    """与次日收益正相关但含噪的因子：日频 IC 为正但 <1、逐日波动（IR/t-stat 有限且不相等）。"""
    rng = np.random.default_rng(seed)
    df = (daily.sort(["ts_code", "trade_date"])
          .with_columns((pl.col("close_adj").shift(-1).over("ts_code") / pl.col("close_adj") - 1.0).alias("fwd"))
          .drop_nulls())
    vals = df["fwd"].to_numpy() + rng.standard_normal(df.height) * noise
    return df.select(["trade_date", "ts_code"]).with_columns(pl.Series("factor_value", vals))

def test_databundle_split():
    from factorzen.discovery.scoring import DataBundle
    b = DataBundle.build(_daily(), train_ratio=0.7)
    assert b.train_end is not None
    assert "fwd_ret_1d" in b.fwd_returns.columns

def test_quick_fitness_positive_for_signal():
    from factorzen.discovery.scoring import DataBundle, quick_fitness
    daily = _daily()
    b = DataBundle.build(daily, train_ratio=0.7)
    fac = _signal_factor_df(daily)
    res = quick_fitness(fac, b, segment="train")
    assert res["ic_mean"] > 0.05
    assert res["n"] > 0

def test_max_correlation_suite():
    """test_max_correlation_self_is_one；R3 复现：池里混入一个退化(截面常数)因子，不应把候选与真实高相关因子的相关性抹成 0。"""
    # -- 原 test_max_correlation_self_is_one --
    def _section_0_test_max_correlation_self_is_one():
        from factorzen.discovery.scoring import max_correlation
        daily = _daily()
        fac = _signal_factor_df(daily).rename({"factor_value": "factor_clean"})
        corr = max_correlation(fac.rename({"factor_clean": "factor_value"}),
                               {"self": fac})
        assert corr > 0.99

    _section_0_test_max_correlation_self_is_one()

    # -- 原 test_max_correlation_pairwise_ignores_degenerate_pool_factor --
    def _section_1_test_max_correlation_pairwise_ignores_degenerate_pool_factor():
        from factorzen.discovery.scoring import max_correlation
        daily = _daily()
        good = _signal_factor_df(daily).rename({"factor_value": "factor_clean"})  # 好池因子
        # 退化：同一 (trade_date, ts_code) 键上的常数因子，截面 std==0
        degenerate = good.with_columns(pl.lit(1.0).alias("factor_clean"))
        cand = _signal_factor_df(daily)  # 候选 == good（完全相关）
        corr = max_correlation(cand, {"good": good, "degenerate": degenerate})
        assert corr > 0.99  # 修前因退化因子污染整表返回 0.0

    _section_1_test_max_correlation_pairwise_ignores_degenerate_pool_factor()


def test_fitness_tstat_suite():
    """R2：排序键由裸 IR 换成 t-stat。fitness 现在跟 t-stat 走，且 t-stat≠IR（换的是键而非恒等）。；R2 核心：n<=4 时 HAC t-stat=0 → 低样本候选 fitness 不再吃 raw IR 的虚高。；test_score_penalizes_complexity"""
    # -- 原 test_fitness_sort_key_is_tstat_not_raw_ir --
    def _section_0_test_fitness_sort_key_is_tstat_not_raw_ir():
        from factorzen.discovery.expression import parse_expr
        from factorzen.discovery.scoring import DataBundle, score_candidate
        daily = _daily(n_stocks=40, n_days=120)
        b = DataBundle.build(daily, train_ratio=0.7)
        fac = _noisy_signal_factor_df(daily)
        sc = score_candidate(fac, parse_expr("close"), b, pool={}, gamma=0.002)
        assert sc["tstat_train"] != 0.0
        # fitness == t-stat − 复杂度惩罚（pool 空 → mc=0）；若仍用 ir 则会与此不符（因 t-stat≠ir）
        assert sc["fitness"] == pytest.approx(sc["tstat_train"] - 0.002 * sc["complexity"], abs=1e-9)
        assert abs(sc["tstat_train"] - sc["ir_train"]) > 1e-6

    _section_0_test_fitness_sort_key_is_tstat_not_raw_ir()

    # -- 原 test_fitness_low_sample_tstat_gate_kills_ir_illusion --
    def _section_1_test_fitness_low_sample_tstat_gate_kills_ir_illusion():
        from factorzen.discovery.expression import parse_expr
        from factorzen.discovery.scoring import DataBundle, quick_fitness, score_candidate
        daily = _daily(n_stocks=40, n_days=6)          # train 段仅 4 个有效 IC 日
        b = DataBundle.build(daily, train_ratio=0.5)
        fac = _noisy_signal_factor_df(daily)
        train = quick_fitness(fac, b, segment="train")
        sc = score_candidate(fac, parse_expr("close"), b, pool={}, gamma=0.002)
        assert train["n"] <= 4                          # 低样本
        assert sc["tstat_train"] == 0.0                 # t-stat 的 n>4 门槛未过
        assert sc["fitness"] <= 1e-9                    # 只剩复杂度惩罚，raw IR 被无视

    _section_1_test_fitness_low_sample_tstat_gate_kills_ir_illusion()

    # -- 原 test_score_penalizes_complexity --
    def _section_2_test_score_penalizes_complexity():
        from factorzen.discovery.expression import parse_expr
        from factorzen.discovery.scoring import DataBundle, score_candidate
        daily = _daily()
        b = DataBundle.build(daily)
        fac = _signal_factor_df(daily)
        simple = score_candidate(fac, parse_expr("close"), b, pool={}, gamma=0.01)
        # 复杂表达式（节点更多）在相同 IC 下 fitness 更低
        assert simple["complexity"] == 1
        # 相同因子值(IC 相同) + 更复杂的 node → complexity 更大 → fitness 更低（纯复杂度惩罚）
        complex_score = score_candidate(fac, parse_expr("ts_mean(close, 5)"), b, pool={}, gamma=0.01)
        assert complex_score["complexity"] > simple["complexity"]
        assert complex_score["fitness"] < simple["fitness"]

    _section_2_test_score_penalizes_complexity()


def test_quick_fitness_uses_horizon_1_only(monkeypatch):
    """挖掘 quick_fitness 只算 1d IC；5/10/20d 无人消费（审计 Wave2 项 3）。"""
    from factorzen.discovery import scoring as scoring_mod
    from factorzen.discovery.scoring import DataBundle, quick_fitness

    daily = _daily()
    b = DataBundle.build(daily)
    fac = _signal_factor_df(daily)

    seen: list = []
    _orig = scoring_mod.compute_rank_ic

    def _wrap(*args, **kwargs):
        seen.append(kwargs.get("horizons"))
        return _orig(*args, **kwargs)

    monkeypatch.setattr(scoring_mod, "compute_rank_ic", _wrap)
    res = quick_fitness(fac, b, segment="train")
    assert seen == [[1]]
    assert res["n"] > 0
    # 与显式 1d 主 IC 一致：再跑无 mock 对照
    monkeypatch.setattr(scoring_mod, "compute_rank_ic", _orig)
    res2 = quick_fitness(fac, b, segment="train")
    assert res["ic_mean"] == pytest.approx(res2["ic_mean"], abs=1e-12)
    assert res["ir"] == pytest.approx(res2["ir"], abs=1e-12)
    assert res["tstat"] == pytest.approx(res2["tstat"], abs=1e-12)
    assert res["n"] == res2["n"]

# ==== 来自 test_discovery_derived.py ====
def _mask_df() -> pl.DataFrame:
    # 单股 3 天,已排序;含派生所需全部列
    return pl.DataFrame({
        "trade_date": [1, 2, 3],
        "ts_code": ["A", "A", "A"],
        "open": [10.0, 11.0, 12.0],
        "high": [11.0, 12.0, 13.0],
        "low": [9.0, 10.0, 11.0],
        "close": [10.5, 11.5, 12.5],
        "close_adj": [10.5, 11.5, 12.5],
        "pre_close": [10.0, 10.5, 11.5],
        "vol": [1e5, 1e5, 1e5],
        "amount": [1e6, 1e6, 1e6],
    }).sort(["ts_code", "trade_date"])

def test_add_derived_suite():
    """test_add_derived_columns_values；test_add_derived_columns_safe_when_pre_close_zero"""
    # -- 原 test_add_derived_columns_values --
    def _section_0_test_add_derived_columns_values():
        from factorzen.discovery.derived import add_derived_columns
        out = add_derived_columns(_mask_df())
        for col in ["vwap", "log_vol", "ret_1d", "amplitude", "intraday_ret", "overnight_ret"]:
            assert col in out.columns
        row0 = out.row(0, named=True)
        assert abs(row0["amplitude"] - (11.0 - 9.0) / 10.0) < 1e-9          # (high-low)/pre_close
        assert abs(row0["intraday_ret"] - (10.5 / 10.0 - 1.0)) < 1e-9        # close/open-1
        assert abs(row0["overnight_ret"] - (10.0 / 10.0 - 1.0)) < 1e-9       # open/pre_close-1

    _section_0_test_add_derived_columns_values()

    # -- 原 test_add_derived_columns_safe_when_pre_close_zero --
    def _section_1_test_add_derived_columns_safe_when_pre_close_zero():
        from factorzen.discovery.derived import add_derived_columns
        df = _mask_df().with_columns(
            pl.when(pl.col("trade_date") == 1).then(0.0)
            .otherwise(pl.col("pre_close")).alias("pre_close"))
        out = add_derived_columns(df)
        assert out.row(0, named=True)["overnight_ret"] is None  # 分母 0 → null,不崩

    _section_1_test_add_derived_columns_safe_when_pre_close_zero()


# ==== 来自 test_run_mine_joins_daily_basic.py ====
def test_run_mine_joins_daily_basic_into_frame(monkeypatch):
    import factorzen.daily.data.context as ctx_mod
    import factorzen.pipelines.factor_mine as fm

    d = [date(2024, 1, 1), date(2024, 1, 2)]
    daily = pl.DataFrame({
        "trade_date": d * 2,
        "ts_code": ["A.SZ", "A.SZ", "B.SZ", "B.SZ"],
        "close": [10.0, 11.0, 20.0, 21.0], "close_adj": [10.0, 11.0, 20.0, 21.0],
        "open": [10.0, 11.0, 20.0, 21.0], "high": [10.0, 11.0, 20.0, 21.0],
        "low": [10.0, 11.0, 20.0, 21.0], "vol": [1e5, 1e5, 1e5, 1e5],
        "amount": [1e6, 1e6, 1e6, 1e6],
    })
    basic = pl.DataFrame({
        "trade_date": d * 2,
        "ts_code": ["A.SZ", "A.SZ", "B.SZ", "B.SZ"],
        "total_mv": [5e5, 5e5, 8e5, 8e5], "pb": [1.5, 1.5, 2.0, 2.0],
    })

    class _FakeCtx:
        def __init__(self, **kw):
            pass

        @property
        def daily(self):
            return daily.lazy()

        @property
        def daily_basic(self):
            return basic.lazy()

    monkeypatch.setattr(ctx_mod, "FactorDataContext", _FakeCtx)

    captured: dict = {}

    def _fake_run_session(frame, **kw):
        captured["frame"] = frame
        return {"candidates": [], "session_dir": "x", "n_trials": 0, "n_scored": 0}

    monkeypatch.setattr(fm, "run_session", _fake_run_session)

    fm.run_mine(start="20240101", end="20240102", n_trials=1)

    cols = set(captured["frame"].columns)
    assert "total_mv" in cols and "pb" in cols, (
        f"run_mine 传给 run_session 的帧应含 daily_basic 基本面列（否则 BASIC_FEATURES 死叶子），实得 {cols}"
    )

def test_prepare_mining_daily_default_warmup_covers_search_space(monkeypatch):
    """默认预热前缀 = search_space_max_lookback()（覆盖搜索空间最大回看），不再是会误拒
    长窗口/深嵌套因子的旧默认 60。FactorDataContext 收到的 lookback_days 即证据。"""
    import factorzen.daily.data.context as ctx_mod
    import factorzen.pipelines.factor_mine as fm
    from factorzen.discovery.search.random_search import search_space_max_lookback

    captured: dict = {}
    empty = pl.DataFrame({"trade_date": [], "ts_code": []})

    class _FakeCtx:
        def __init__(self, **kw):
            captured["lookback_days"] = kw.get("lookback_days")

        @property
        def daily(self):
            return empty.lazy()

        @property
        def daily_basic(self):
            return empty.lazy()

    monkeypatch.setattr(ctx_mod, "FactorDataContext", _FakeCtx)

    fm.prepare_mining_daily("20240101", "20240201")

    assert captured["lookback_days"] == search_space_max_lookback()


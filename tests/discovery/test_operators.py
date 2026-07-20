"""Merged discovery tests: test_operators.py

test_discovery_operators.py：算子库 ts/cs/算术/叶子目录与手工或 numpy 真值对齐
test_operators_properties.py：算子库 hypothesis 属性测试
test_operators_nan_guards.py：算子除零守卫防 NaN 穿透 + ts_rank 部分窗口归一化
test_ts_decay_linear.py：ts_decay_linear 真线性衰减加权均值（非等权）
"""

from __future__ import annotations

import math
from datetime import (
    date,
    timedelta,
)

import numpy as np
import polars as pl
import pytest
from hypothesis import (
    given,
    settings,
)
from hypothesis import (
    strategies as st,
)

from factorzen.discovery.expression import (
    evaluate,
    parse_expr,
)
from factorzen.discovery.operators import (
    LEAF_FEATURES,
    OPERATORS,
    _safe_div,
)


# ==== 来自 test_discovery_operators.py ====
def _toy_df(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for code in ["A", "B"]:
        price = 10.0
        for d in range(30):
            price = float(max(price * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": d, "ts_code": code, "close_adj": price,
                         "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows).sort(["ts_code", "trade_date"])


def test_cs_rank_suite():
    """test_cs_rank_is_within_unit_interval；截面 rank 归一化尺度不应随当日 null 比例漂移：相同非空值、不同 null 数的两日，"""
    # -- 原 test_cs_rank_is_within_unit_interval --
    def _section_0_test_cs_rank_is_within_unit_interval():
        from factorzen.discovery.operators import OPERATORS
        df = _toy_df()
        expr = OPERATORS["rank"].build([pl.col("close_adj")], None)
        got = df.with_columns(expr.alias("r"))["r"].drop_nulls().to_list()
        assert all(0.0 < v < 1.0 for v in got)

    _section_0_test_cs_rank_is_within_unit_interval()

    # -- 原 test_cs_rank_normalization_scale_invariant_to_null_ratio --
    def _section_1_test_cs_rank_normalization_scale_invariant_to_null_ratio():
        from factorzen.discovery.operators import OPERATORS

        df = pl.DataFrame({
            "trade_date": [1, 1, 1, 2, 2, 2, 2, 2],
            "ts_code": [f"c{i}" for i in range(8)],
            "close_adj": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, None, None],
        })
        expr = OPERATORS["rank"].build([pl.col("close_adj")], None)
        out = df.with_columns(expr.alias("r"))
        d1_max = out.filter(pl.col("trade_date") == 1)["r"].max()
        d2_max = out.filter(pl.col("trade_date") == 2)["r"].max()
        assert abs(d1_max - d2_max) < 1e-12, (
            f"归一化排名尺度随 null 比例漂移：无 null 日 max={d1_max}，含 2 null 日 max={d2_max}"
        )

    _section_1_test_cs_rank_normalization_scale_invariant_to_null_ratio()


def test_arith_ops_suite():
    """test_arith_add；test_arith_neg_square_inv；test_arith_max_min_horizontal；test_arith_inv_null_on_zero"""
    # -- 原 test_arith_add --
    def _section_0_test_arith_add():
        from factorzen.discovery.operators import OPERATORS
        df = _toy_df()
        expr = OPERATORS["add"].build([pl.col("close_adj"), pl.col("vol")], None)
        got = df.with_columns(expr.alias("s"))
        assert got["s"].to_list() == (df["close_adj"] + df["vol"]).to_list()

    _section_0_test_arith_add()

    # -- 原 test_arith_neg_square_inv --
    def _section_1_test_arith_neg_square_inv():
        from factorzen.discovery.operators import OPERATORS
        df = _toy_df()
        neg = df.with_columns(OPERATORS["neg"].build([pl.col("close_adj")], None).alias("n"))
        assert neg["n"].to_list() == (-df["close_adj"]).to_list()
        sq = df.with_columns(OPERATORS["square"].build([pl.col("close_adj")], None).alias("s"))
        assert sq["s"].to_list() == (df["close_adj"] * df["close_adj"]).to_list()
        inv = df.with_columns(OPERATORS["inv"].build([pl.col("close_adj")], None).alias("i"))
        # close_adj 恒 > 0.1 → inv 有限且 = 1/x
        got = inv["i"].to_list()
        exp = (1.0 / df["close_adj"]).to_list()
        assert all(abs(g - e) < 1e-12 for g, e in zip(got, exp, strict=True))

    _section_1_test_arith_neg_square_inv()

    # -- 原 test_arith_max_min_horizontal --
    def _section_2_test_arith_max_min_horizontal():
        from factorzen.discovery.operators import OPERATORS
        df = _toy_df()
        mx = df.with_columns(
            OPERATORS["max"].build([pl.col("close_adj"), pl.col("vol")], None).alias("m"))
        mn = df.with_columns(
            OPERATORS["min"].build([pl.col("close_adj"), pl.col("vol")], None).alias("m"))
        exp_max = [max(a, b) for a, b in zip(df["close_adj"], df["vol"], strict=True)]
        exp_min = [min(a, b) for a, b in zip(df["close_adj"], df["vol"], strict=True)]
        assert mx["m"].to_list() == exp_max
        assert mn["m"].to_list() == exp_min

    _section_2_test_arith_max_min_horizontal()

    # -- 原 test_arith_inv_null_on_zero --
    def _section_3_test_arith_inv_null_on_zero():
        from factorzen.discovery.operators import OPERATORS
        df = pl.DataFrame({"trade_date": [0, 1], "ts_code": ["A", "A"], "x": [0.0, 4.0]})
        got = df.with_columns(OPERATORS["inv"].build([pl.col("x")], None).alias("i"))["i"].to_list()
        assert got[0] is None                      # 1/0 → null(安全除法)
        assert got[1] is not None and abs(got[1] - 0.25) < 1e-12

    _section_3_test_arith_inv_null_on_zero()


def test_ts_corr_cov_suite():
    """test_ts_corr_perfect_positive_correlation；test_ts_corr_null_when_constant_series；近常数序列的微负方差不应经 sqrt 穿透成 NaN。；test_ts_cov_matches_numpy_ground_truth"""
    # -- 原 test_ts_corr_perfect_positive_correlation --
    def _section_0_test_ts_corr_perfect_positive_correlation():
        from factorzen.discovery.operators import OPERATORS
        # b = 2a + 1 完全正相关 → 滚动 corr 恒为 +1
        rows = [{"trade_date": d, "ts_code": "A", "a": float(d), "b": 2.0 * d + 1.0}
                for d in range(10)]
        df = pl.DataFrame(rows).sort(["ts_code", "trade_date"])
        expr = OPERATORS["ts_corr"].build([pl.col("a"), pl.col("b")], 5)
        got = df.with_columns(expr.alias("c"))["c"].drop_nulls().to_list()
        assert got and all(abs(v - 1.0) < 1e-9 for v in got)

    _section_0_test_ts_corr_perfect_positive_correlation()

    # -- 原 test_ts_corr_null_when_constant_series --
    def _section_1_test_ts_corr_null_when_constant_series():
        from factorzen.discovery.operators import OPERATORS
        rows = [{"trade_date": d, "ts_code": "A", "a": float(d), "b": 5.0} for d in range(10)]
        df = pl.DataFrame(rows).sort(["ts_code", "trade_date"])
        expr = OPERATORS["ts_corr"].build([pl.col("a"), pl.col("b")], 5)
        got = df.with_columns(expr.alias("c"))
        # b 无方差 → 分母 0 → 全 null
        assert got["c"].drop_nulls().len() == 0

    _section_1_test_ts_corr_null_when_constant_series()

    # -- 原 test_ts_corr_never_outputs_nan_on_near_constant --
    def _section_2_test_ts_corr_never_outputs_nan_on_near_constant():
        d = pl.DataFrame({"ts_code": ["A"] * 8, "trade_date": list(range(8)),
                          "a": [1.0 + (i % 2) * 1e-9 for i in range(8)], "b": [2.0] * 8})
        e = OPERATORS["ts_corr"].build([pl.col("a"), pl.col("b")], 5)
        out = d.with_columns(e.alias("r"))["r"].to_list()
        assert not any(v is not None and math.isnan(v) for v in out), f"不应含 NaN：{out}"

    _section_2_test_ts_corr_never_outputs_nan_on_near_constant()

    # -- 原 test_ts_cov_matches_numpy_ground_truth --
    def _section_3_test_ts_cov_matches_numpy_ground_truth():
        from factorzen.discovery.operators import OPERATORS
        a = [float(d) for d in range(8)]
        b = [float(d) * 3.0 for d in range(8)]
        rows = [{"trade_date": d, "ts_code": "A", "a": a[d], "b": b[d]} for d in range(8)]
        df = pl.DataFrame(rows).sort(["ts_code", "trade_date"])
        expr = OPERATORS["ts_cov"].build([pl.col("a"), pl.col("b")], 4)
        got = df.with_columns(expr.alias("c"))["c"].to_list()
        # 独立 ground truth：numpy 对每个 trailing 窗口(w=4, min_samples=3)直接算总体协方差
        na, nb = np.array(a), np.array(b)
        exp: list[float | None] = []
        for i in range(8):
            lo = max(0, i - 3)
            wa, wb = na[lo:i + 1], nb[lo:i + 1]
            if len(wa) < 3:
                exp.append(None)
            else:
                exp.append(float((wa * wb).mean() - wa.mean() * wb.mean()))
        assert len(got) == len(exp)
        for g, e in zip(got, exp, strict=True):
            if e is None:
                assert g is None
            else:
                assert g is not None and abs(g - e) < 1e-9

    _section_3_test_ts_cov_matches_numpy_ground_truth()


def test_ts_moments_suite():
    """test_ts_zscore_null_on_constant_and_matches_numpy；test_ts_skew_symmetric_is_zero；test_ts_skew_matches_numpy_ground_truth"""
    # -- 原 test_ts_zscore_null_on_constant_and_matches_numpy --
    def _section_0_test_ts_zscore_null_on_constant_and_matches_numpy():
        import numpy as np

        from factorzen.discovery.operators import OPERATORS
        # 前3个常数(std=0→null),后面有方差;用 numpy 独立算每个 trailing 窗口的 z 分数
        vals = [5.0, 5.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        rows = [{"trade_date": d, "ts_code": "A", "close_adj": vals[d], "vol": 1.0} for d in range(8)]
        df = pl.DataFrame(rows).sort(["ts_code", "trade_date"])
        expr = OPERATORS["ts_zscore"].build([pl.col("close_adj")], 3)
        got = df.with_columns(expr.alias("z"))["z"].to_list()
        na = np.array(vals)
        for i in range(8):
            lo = max(0, i - 2)  # 窗口 w=3
            w = na[lo:i + 1]
            if len(w) < 3:
                assert got[i] is None
            else:
                sd = float(w.std(ddof=1))  # polars rolling_std 默认 ddof=1
                if sd < 1e-12:
                    assert got[i] is None  # 常数窗口 std=0 → null
                else:
                    exp = (float(w[-1]) - float(w.mean())) / sd
                    assert got[i] is not None and abs(got[i] - exp) < 1e-9

    _section_0_test_ts_zscore_null_on_constant_and_matches_numpy()

    # -- 原 test_ts_skew_symmetric_is_zero --
    def _section_1_test_ts_skew_symmetric_is_zero():
        from factorzen.discovery.operators import OPERATORS
        # 关于平均值对称的序列，偏度 = 0（[2,4,3,4,2] 平均值=3，偏差=[-1,1,0,1,-1]，关于0对称）
        vals = [2.0, 4.0, 3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0]
        rows = [{"trade_date": d, "ts_code": "A", "close_adj": v, "vol": 1.0}
                for d, v in enumerate(vals)]
        df = pl.DataFrame(rows).sort(["ts_code", "trade_date"])
        expr = OPERATORS["ts_skew"].build([pl.col("close_adj")], 5)
        got = df.with_columns(expr.alias("s"))["s"].drop_nulls().to_list()
        # window=5、min_samples=3 → 第一个非空点是偏窗 [2,4,3]（均值=3，偏差=[-1,1,0]），
        # 关于 0 对称 → 偏度=0；不是满窗 [2,4,3,4,2](那是第二个非空点，同样对称)。
        assert got and abs(got[0]) < 1e-9

    _section_1_test_ts_skew_symmetric_is_zero()

    # -- 原 test_ts_skew_matches_numpy_ground_truth --
    def _section_2_test_ts_skew_matches_numpy_ground_truth():
        import numpy as np

        from factorzen.discovery.operators import OPERATORS
        vals = [1.0, 1.0, 1.0, 1.0, 10.0, 2.0, 3.0, 4.0]
        rows = [{"trade_date": d, "ts_code": "A", "close_adj": vals[d], "vol": 1.0} for d in range(8)]
        df = pl.DataFrame(rows).sort(["ts_code", "trade_date"])
        expr = OPERATORS["ts_skew"].build([pl.col("close_adj")], 4)
        got = df.with_columns(expr.alias("s"))["s"].to_list()
        na = np.array(vals)
        for i in range(8):
            lo = max(0, i - 3)
            w = na[lo:i + 1]
            if len(w) < 3:
                assert got[i] is None
            else:
                m = float(w.mean())
                sd = float(w.std())  # numpy ddof=0 总体
                if sd < 1e-12:
                    assert got[i] is None
                else:
                    exp = float((((w - m) / sd) ** 3).mean())
                    assert got[i] is not None and abs(got[i] - exp) < 1e-6

    _section_2_test_ts_skew_matches_numpy_ground_truth()


def test_ts_rank_suite():
    """test_ts_rank_matches_manual；warm-up 期(历史不足 w)ts_rank 须除以窗口内实际样本数，而非固定 w。"""
    # -- 原 test_ts_rank_matches_manual --
    def _section_0_test_ts_rank_matches_manual():
        from factorzen.discovery.operators import OPERATORS

        def _avg_rank_of_last(window: np.ndarray) -> float:
            # 平均排名(并列取均值): rank = (#严格小于) + (#等于(含自身) + 1) / 2。
            # 与 polars rolling_rank(method="average") 官方文档示例([1,4,4,1,9], w=3
            # → [null,null,2.5,1.0,3.0])逐点手算核对一致，等价于 scipy
            # rankdata(method="average") 对窗口末尾元素的排名。
            last = window[-1]
            less = float((window < last).sum())
            equal = float((window == last).sum())
            return less + (equal + 1.0) / 2.0

        vals = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 5.0]  # 含并列：1.0×2, 5.0×3
        rows = [{"trade_date": d, "ts_code": "A", "close_adj": vals[d], "vol": 1.0}
                for d in range(len(vals))]
        df = pl.DataFrame(rows).sort(["ts_code", "trade_date"])
        w = 4
        expr = OPERATORS["ts_rank"].build([pl.col("close_adj")], w)
        got = df.with_columns(expr.alias("r"))["r"].to_list()

        na = np.array(vals)
        for i in range(w - 1, len(vals)):  # 只比对满窗区 index>=w-1(偏窗区旧/新实现语义不同，见 fix report)
            window = na[i - w + 1: i + 1]
            exp = _avg_rank_of_last(window) / w
            assert got[i] is not None
            assert abs(got[i] - exp) < 1e-9

    _section_0_test_ts_rank_matches_manual()

    # -- 原 test_ts_rank_normalizes_by_actual_window_count --
    def _section_1_test_ts_rank_normalizes_by_actual_window_count():
        d = pl.DataFrame({"ts_code": ["A"] * 4, "trade_date": list(range(4)),
                          "x": [1.0, 2.0, 3.0, 4.0]})  # 单调上升 → 每行都是窗口内最大
        e = OPERATORS["ts_rank"].build([pl.col("x")], 10)
        out = d.with_columns(e.alias("r"))["r"].to_list()
        # 第3、4行是各自窗口内 top(rank=count) → 归一化应为 1.0，而非 3/10、4/10
        assert out[2] == pytest.approx(1.0), f"warm-up top 应为 1.0，实得 {out[2]}"
        assert out[3] == pytest.approx(1.0), f"warm-up top 应为 1.0，实得 {out[3]}"

    _section_1_test_ts_rank_normalizes_by_actual_window_count()


def test_registry_leaves_suite():
    """test_leaf_features_contains_price_volume_and_fundamental；test_basic_features_include_turnover_and_shares；test_operator_categories_present；test_operator_category_assignments"""
    # -- 原 test_leaf_features_contains_price_volume_and_fundamental --
    def _section_0_test_leaf_features_contains_price_volume_and_fundamental():
        from factorzen.discovery.operators import LEAF_FEATURES
        price_vol_leaves = {"close", "open", "high", "low", "vol", "amount", "vwap", "log_vol", "ret_1d"}
        fundamental_leaves = {"total_mv", "circ_mv", "pb", "pe_ttm", "ps_ttm", "dv_ttm"}
        keys = set(LEAF_FEATURES.keys())
        assert price_vol_leaves <= keys, f"missing price/vol leaves: {price_vol_leaves - keys}"
        assert fundamental_leaves <= keys, f"missing fundamental leaves: {fundamental_leaves - keys}"

    _section_0_test_leaf_features_contains_price_volume_and_fundamental()

    # -- 原 test_basic_features_include_turnover_and_shares --
    def _section_1_test_basic_features_include_turnover_and_shares():
        from factorzen.discovery.operators import BASIC_FEATURES, LEAF_FEATURES
        for f in ["turnover_rate", "turnover_rate_f", "volume_ratio", "float_share"]:
            assert f in BASIC_FEATURES, f"BASIC_FEATURES missing {f}"
            assert f in LEAF_FEATURES, f"LEAF_FEATURES missing {f}"
        # 原有 6 个基本面叶子仍在
        assert {"total_mv", "circ_mv", "pb", "pe_ttm", "ps_ttm", "dv_ttm"} <= BASIC_FEATURES

    _section_1_test_basic_features_include_turnover_and_shares()

    # -- 原 test_operator_categories_present --
    def _section_2_test_operator_categories_present():
        from factorzen.discovery.operators import OPERATORS
        cats = {spec.category for spec in OPERATORS.values()}
        assert cats == {"ts", "cs", "arith"}

    _section_2_test_operator_categories_present()

    # -- 原 test_operator_category_assignments --
    def _section_3_test_operator_category_assignments():
        from factorzen.discovery.operators import OPERATORS
        assert OPERATORS["ts_mean"].category == "ts"
        assert OPERATORS["pct_change"].category == "ts"
        assert OPERATORS["rank"].category == "cs"
        assert OPERATORS["add"].category == "arith"

    _section_3_test_operator_category_assignments()


# ==== 来自 test_operators_properties.py ====
def _synth(seed: int, n_stocks: int, n_days: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 1)
    rows = []
    for s in range(n_stocks):
        p = 10.0 + s
        for d in range(n_days):
            p = max(1.0, p * (1 + rng.normal(0, 0.02)))
            v = float(rng.uniform(1e5, 1e6))
            rows.append(
                {
                    "ts_code": f"{s:04d}.SZ",
                    "trade_date": start + timedelta(days=d),
                    "close": p, "close_adj": p, "open": p, "open_adj": p,
                    "high": p, "high_adj": p, "low": p, "low_adj": p,
                    "vol": v, "amount": p * v,
                }
            )
    return pl.DataFrame(rows).sort(["ts_code", "trade_date"])

_DIMS = dict(
    seed=st.integers(0, 10_000),
    n_stocks=st.integers(15, 35),
    n_days=st.integers(12, 30),
)

@given(**_DIMS)
@settings(max_examples=10, deadline=None)
def test_rank_length_preserved_and_in_unit_range(seed, n_stocks, n_days):
    df = _synth(seed, n_stocks, n_days)
    r = evaluate(parse_expr("rank(close)", LEAF_FEATURES), df)
    assert len(r) == df.height  # 长度守恒
    v = r.drop_nulls()
    assert v.is_empty() or bool(((v >= 0) & (v <= 1)).all())  # rank ∈ [0,1]

@given(**_DIMS)
@settings(max_examples=10, deadline=None)
def test_delta_length_preserved(seed, n_stocks, n_days):
    df = _synth(seed, n_stocks, n_days)
    r = evaluate(parse_expr("delta(close, 1)", LEAF_FEATURES), df)
    assert len(r) == df.height

@given(**_DIMS)
@settings(max_examples=10, deadline=None)
def test_zscore_cross_sectional_mean_near_zero(seed, n_stocks, n_days):
    df = _synth(seed, n_stocks, n_days)
    out = df.with_columns(
        evaluate(parse_expr("zscore(close)", LEAF_FEATURES), df).alias("z")
    )
    daily = out.drop_nulls("z").group_by("trade_date").agg(pl.col("z").mean().alias("m"))
    if daily.height:
        assert bool((daily["m"].abs() < 1e-6).all())  # 截面 z-score 均值≈0

@given(**_DIMS)
@settings(max_examples=10, deadline=None)
def test_ts_mean_nan_propagation(seed, n_stocks, n_days):
    """rolling min_samples=3:每股前 2 行必为 null(不产生假值)。"""
    df = _synth(seed, n_stocks, n_days)
    out = df.with_columns(
        evaluate(parse_expr("ts_mean(close, 5)", LEAF_FEATURES), df).alias("m")
    )
    first2 = out.group_by("ts_code", maintain_order=True).head(2)
    assert first2["m"].null_count() == first2.height

# ==== 来自 test_operators_nan_guards.py ====
def test_safe_div_nan_denominator_returns_none_not_nan():
    d = pl.DataFrame({"b": [1.0, float("nan"), 0.0]})
    out = d.with_columns(_safe_div(pl.lit(1.0), pl.col("b")).alias("r"))["r"].to_list()
    assert out[0] == 1.0
    assert out[1] is None, "NaN 分母应得 None，不应穿透成 NaN"
    assert out[2] is None, "0 分母应得 None"


# ==== 来自 test_ts_decay_linear.py ====
_MIN = 3  # 与 operators._MIN 对齐

def _manual_decay_linear(vals: list[float | None], w: int) -> list[float | None]:
    """手算线性衰减加权均值 ground-truth。

    权重 1,2,...,w（最新一期权重最大），归一化到 Σw=1。
    窗口内非空样本数 < _MIN → None（与其它 _ts 算子的 min_samples 语义一致）。
    """
    out: list[float | None] = []
    for i in range(len(vals)):
        lo = max(0, i - w + 1)
        window = vals[lo:i + 1]
        # 权重与窗口右端对齐：窗口最后一个元素权重最大
        weights = list(range(w - len(window) + 1, w + 1))
        pairs = [(v, wt) for v, wt in zip(window, weights, strict=True) if v is not None]
        if len(pairs) < _MIN:
            out.append(None)
            continue
        num = sum(v * wt for v, wt in pairs)
        den = sum(wt for _, wt in pairs)
        out.append(num / den)
    return out

def _series_df(vals: list[float], code: str = "A") -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": list(range(len(vals))),
        "ts_code": [code] * len(vals),
        "x": vals,
    }).sort(["ts_code", "trade_date"])

def _apply(df: pl.DataFrame, w: int) -> list:
    from factorzen.discovery.operators import OPERATORS
    expr = OPERATORS["ts_decay_linear"].build([pl.col("x")], w)
    return df.with_columns(expr.alias("f"))["f"].to_list()

# ── 1. ground-truth 对拍（手算，非自证）──────────────────────────────────────

def test_ts_decay_linear_suite():
    """test_matches_manual_linear_decay_ground_truth；写死的小例子：w=3、值 [1,2,3] → (1*1 + 2*2 + 3*3)/(1+2+3) = 14/6。；回归锚：ts_decay_linear 曾就是 rolling_mean。单调序列上两者必须不同。；方向锚：递增序列上，加权均值应高于等权均值（近期权重更大）。；Σw=1 归一化锚：常数序列的加权均值必须等于该常数（不引入水平漂移）。；.over("ts_code")：A 的窗口不得吃进 B 的值。；warm-up 的 null 位置必须与 ts_mean 一致（min_samples 语义同族）。；cumsum 恒等式 vs O(w) 位移：不同量级下都须逐位一致（相对误差 < 1e-9）。；有限性守卫锚：cumsum 是全序列累加器，一个 inf 若不拦会污染其后**全部**取值。；退化截面守卫：全 null 输入不得抛、不得产出 NaN。"""
    # -- 原 test_matches_manual_linear_decay_ground_truth --
    def _section_0_test_matches_manual_linear_decay_ground_truth():
        rng = np.random.default_rng(7)
        vals = [float(v) for v in rng.standard_normal(40) * 3 + 10]
        df = _series_df(vals)
        for w in (5, 20):
            got = _apply(df, w)
            want = _manual_decay_linear(vals, w)
            assert len(got) == len(want)
            for i, (g, e) in enumerate(zip(got, want, strict=True)):
                if e is None:
                    assert g is None, f"w={w} i={i}: 期望 None，得 {g}"
                else:
                    assert g is not None, f"w={w} i={i}: 期望 {e}，得 None"
                    assert abs(g - e) < 1e-9, f"w={w} i={i}: {g} != {e}"

    _section_0_test_matches_manual_linear_decay_ground_truth()

    # -- 原 test_known_closed_form_small_case --
    def _section_1_test_known_closed_form_small_case():
        df = _series_df([1.0, 2.0, 3.0])
        got = _apply(df, 3)
        assert got[0] is None and got[1] is None  # 非空样本不足 _MIN
        assert abs(got[2] - 14.0 / 6.0) < 1e-12

    _section_1_test_known_closed_form_small_case()

    # -- 原 test_differs_from_ts_mean --
    def _section_2_test_differs_from_ts_mean():
        from factorzen.discovery.operators import OPERATORS

        vals = [float(i) for i in range(30)]
        df = _series_df(vals)
        decay = _apply(df, 10)
        mean_expr = OPERATORS["ts_mean"].build([pl.col("x")], 10)
        mean = df.with_columns(mean_expr.alias("m"))["m"].to_list()

        diffs = [
            abs(d - m) for d, m in zip(decay, mean, strict=True)
            if d is not None and m is not None
        ]
        assert diffs, "两侧全 null，测试无判别力"
        assert max(diffs) > 1e-6, "ts_decay_linear 与 ts_mean 逐位相同——算子又退化成等权了"

    _section_2_test_differs_from_ts_mean()

    # -- 原 test_weights_recent_more_than_old --
    def _section_3_test_weights_recent_more_than_old():
        from factorzen.discovery.operators import OPERATORS

        vals = [float(i) for i in range(30)]
        df = _series_df(vals)
        decay = _apply(df, 10)
        mean_expr = OPERATORS["ts_mean"].build([pl.col("x")], 10)
        mean = df.with_columns(mean_expr.alias("m"))["m"].to_list()

        pairs = [
            (d, m) for d, m in zip(decay, mean, strict=True) if d is not None and m is not None
        ]
        assert pairs
        assert all(d > m for d, m in pairs), "递增序列上衰减加权均值应严格大于等权均值"

    _section_3_test_weights_recent_more_than_old()

    # -- 原 test_constant_series_preserves_level --
    def _section_4_test_constant_series_preserves_level():
        df = _series_df([5.0] * 20)
        got = _apply(df, 10)
        vals = [v for v in got if v is not None]
        assert vals, "全 null，无判别力"
        assert all(abs(v - 5.0) < 1e-12 for v in vals)

    _section_4_test_constant_series_preserves_level()

    # -- 原 test_grouped_by_ts_code_no_leakage --
    def _section_5_test_grouped_by_ts_code_no_leakage():
        a = [1.0] * 10
        b = [100.0] * 10
        df = pl.concat([_series_df(a, "A"), _series_df(b, "B")]).sort(["ts_code", "trade_date"])
        got = df.with_columns(
            __import__(
                "factorzen.discovery.operators", fromlist=["OPERATORS"]
            ).OPERATORS["ts_decay_linear"].build([pl.col("x")], 5).alias("f")
        )
        a_vals = [v for v in got.filter(pl.col("ts_code") == "A")["f"].to_list() if v is not None]
        b_vals = [v for v in got.filter(pl.col("ts_code") == "B")["f"].to_list() if v is not None]
        assert a_vals and b_vals
        assert all(abs(v - 1.0) < 1e-12 for v in a_vals)
        assert all(abs(v - 100.0) < 1e-12 for v in b_vals)

    _section_5_test_grouped_by_ts_code_no_leakage()

    # -- 原 test_warmup_null_semantics_match_ts_mean --
    def _section_6_test_warmup_null_semantics_match_ts_mean():
        from factorzen.discovery.operators import OPERATORS

        rng = np.random.default_rng(3)
        vals = [float(v) for v in rng.standard_normal(25)]
        df = _series_df(vals)
        decay = _apply(df, 8)
        mean_expr = OPERATORS["ts_mean"].build([pl.col("x")], 8)
        mean = df.with_columns(mean_expr.alias("m"))["m"].to_list()
        assert [v is None for v in decay] == [v is None for v in mean]

    _section_6_test_warmup_null_semantics_match_ts_mean()

    # -- 原 test_parity_with_shift_reference_across_magnitudes --
    def _section_7_test_parity_with_shift_reference_across_magnitudes():
        rng = np.random.default_rng(11)
        for scale in (1.0, 1e6):
            n = 600
            vals = rng.standard_normal(n) * scale
            vals[rng.random(n) < 0.05] = np.nan
            df = pl.DataFrame({
                "trade_date": list(range(n)),
                "ts_code": ["A"] * n,
                "x": vals,
            }).with_columns(pl.col("x").fill_nan(None)).sort(["ts_code", "trade_date"])
            for w in (5, 63):
                got = np.asarray(_apply(df, w), dtype=float)
                want = np.asarray(
                    df.with_columns(_shift_reference(pl.col("x"), w).over("ts_code").alias("r"))
                    ["r"].to_list(), dtype=float)
                assert (np.isnan(got) == np.isnan(want)).all(), f"scale={scale} w={w} null 位置不一致"
                both = ~np.isnan(got)
                if both.any():
                    rel = np.abs(got[both] - want[both]) / np.maximum(np.abs(want[both]), 1e-300)
                    assert rel.max() < 1e-9, f"scale={scale} w={w} 相对误差 {rel.max():.2e}"

    _section_7_test_parity_with_shift_reference_across_magnitudes()

    # -- 原 test_non_finite_does_not_poison_downstream --
    def _section_8_test_non_finite_does_not_poison_downstream():
        n = 40
        vals = [1.0] * n
        vals[10] = float("inf")
        df = _series_df(vals)
        got = _apply(df, 5)
        tail = [v for v in got[15:] if v is not None]
        assert tail, "尾段全 null，测试无判别力"
        assert all(np.isfinite(v) for v in tail), "inf 穿透了 cumsum，污染下游全部取值"
        # inf 按缺失处理 → 其余全是常数 1.0，加权均值仍为 1.0
        assert all(abs(v - 1.0) < 1e-12 for v in tail)

    _section_8_test_non_finite_does_not_poison_downstream()

    # -- 原 test_all_null_series_yields_all_null --
    def _section_9_test_all_null_series_yields_all_null():
        df = pl.DataFrame({
            "trade_date": list(range(10)),
            "ts_code": ["A"] * 10,
            "x": [None] * 10,
        }, schema_overrides={"x": pl.Float64}).sort(["ts_code", "trade_date"])
        got = _apply(df, 5)
        assert all(v is None for v in got)

    _section_9_test_all_null_series_yields_all_null()


# ── 2. 反例锚：不得再退化成等权 ──────────────────────────────────────────────


# ── 3. 归一化 / 量纲 ─────────────────────────────────────────────────────────


# ── 4. 分组 / 退化截面语义 ───────────────────────────────────────────────────


def _shift_reference(x: pl.Expr, w: int) -> pl.Expr:
    """O(w) 位移参考实现——生产实现走 cumsum 恒等式（O(1) rolling）换性能，
    本函数是它的独立 parity 锚：两者由**不同代数路径**得出，非自证。
    """
    ok = (x.is_not_null() & x.is_finite()).fill_null(False)
    filled = pl.when(ok).then(x).otherwise(0.0)
    num = den = cnt = None
    for k in range(w):
        wt = float(w - k)
        v = filled.shift(k).fill_null(0.0) * wt
        m = ok.shift(k).fill_null(value=False)
        d = m.cast(pl.Float64) * wt
        c = m.cast(pl.Int64)
        num = v if num is None else num + v
        den = d if den is None else den + d
        cnt = c if cnt is None else cnt + c
    return (
        pl.when(cnt >= _MIN)
        .then(pl.when(den.abs() > 1e-12).then(num / den).otherwise(None))
        .otherwise(None)
    )



from __future__ import annotations

import numpy as np
import polars as pl


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


def test_ts_mean_matches_manual():
    from factorzen.discovery.operators import OPERATORS
    df = _toy_df()
    expr = OPERATORS["ts_mean"].build([pl.col("close_adj")], 5)
    got = df.with_columns(expr.alias("f"))
    manual = df.with_columns(
        pl.col("close_adj").rolling_mean(5, min_samples=3).over("ts_code").alias("m"))
    assert got["f"].to_list() == manual["m"].to_list()


def test_cs_rank_is_within_unit_interval():
    from factorzen.discovery.operators import OPERATORS
    df = _toy_df()
    expr = OPERATORS["rank"].build([pl.col("close_adj")], None)
    got = df.with_columns(expr.alias("r"))["r"].drop_nulls().to_list()
    assert all(0.0 < v < 1.0 for v in got)


def test_arith_add():
    from factorzen.discovery.operators import OPERATORS
    df = _toy_df()
    expr = OPERATORS["add"].build([pl.col("close_adj"), pl.col("vol")], None)
    got = df.with_columns(expr.alias("s"))
    assert got["s"].to_list() == (df["close_adj"] + df["vol"]).to_list()


def test_operator_categories_present():
    from factorzen.discovery.operators import OPERATORS
    cats = {spec.category for spec in OPERATORS.values()}
    assert cats == {"ts", "cs", "arith"}


def test_ts_corr_perfect_positive_correlation():
    from factorzen.discovery.operators import OPERATORS
    # b = 2a + 1 完全正相关 → 滚动 corr 恒为 +1
    rows = [{"trade_date": d, "ts_code": "A", "a": float(d), "b": 2.0 * d + 1.0}
            for d in range(10)]
    df = pl.DataFrame(rows).sort(["ts_code", "trade_date"])
    expr = OPERATORS["ts_corr"].build([pl.col("a"), pl.col("b")], 5)
    got = df.with_columns(expr.alias("c"))["c"].drop_nulls().to_list()
    assert got and all(abs(v - 1.0) < 1e-9 for v in got)


def test_ts_corr_null_when_constant_series():
    from factorzen.discovery.operators import OPERATORS
    rows = [{"trade_date": d, "ts_code": "A", "a": float(d), "b": 5.0} for d in range(10)]
    df = pl.DataFrame(rows).sort(["ts_code", "trade_date"])
    expr = OPERATORS["ts_corr"].build([pl.col("a"), pl.col("b")], 5)
    got = df.with_columns(expr.alias("c"))
    # b 无方差 → 分母 0 → 全 null
    assert got["c"].drop_nulls().len() == 0


def test_ts_cov_matches_numpy_ground_truth():
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

def test_leaf_features_contains_price_volume_and_fundamental():
    from factorzen.discovery.operators import LEAF_FEATURES
    price_vol_leaves = {"close", "open", "high", "low", "vol", "amount", "vwap", "log_vol", "ret_1d"}
    fundamental_leaves = {"total_mv", "circ_mv", "pb", "pe_ttm", "ps_ttm", "dv_ttm"}
    keys = set(LEAF_FEATURES.keys())
    assert price_vol_leaves <= keys, f"missing price/vol leaves: {price_vol_leaves - keys}"
    assert fundamental_leaves <= keys, f"missing fundamental leaves: {fundamental_leaves - keys}"


def test_basic_features_subset_and_no_turnover():
    from factorzen.discovery.operators import BASIC_FEATURES
    allowed = {"total_mv", "circ_mv", "pb", "pe_ttm", "ps_ttm", "dv_ttm", "pe", "ps", "dv_ratio"}
    assert "turnover_rate" not in BASIC_FEATURES, "BASIC_FEATURES must not contain 'turnover_rate'"
    assert "volume_ratio" not in BASIC_FEATURES, "BASIC_FEATURES must not contain 'volume_ratio'"
    assert allowed >= BASIC_FEATURES, f"unexpected entries: {BASIC_FEATURES - allowed}"


def test_operator_category_assignments():
    from factorzen.discovery.operators import OPERATORS
    assert OPERATORS["ts_mean"].category == "ts"
    assert OPERATORS["pct_change"].category == "ts"
    assert OPERATORS["rank"].category == "cs"
    assert OPERATORS["add"].category == "arith"

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


def test_cs_rank_normalization_scale_invariant_to_null_ratio():
    """截面 rank 归一化尺度不应随当日 null 比例漂移：相同非空值、不同 null 数的两日，
    归一化排名应一致。此前分母用 pl.len()（含 null 行）→ 含 null 日尺度被压小。
    """
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

def test_ts_median_matches_manual():
    from factorzen.discovery.operators import OPERATORS
    df = _toy_df()
    expr = OPERATORS["ts_median"].build([pl.col("close_adj")], 5)
    got = df.with_columns(expr.alias("f"))
    manual = df.with_columns(
        pl.col("close_adj").rolling_median(5, min_samples=3).over("ts_code").alias("m"))
    assert got["f"].to_list() == manual["m"].to_list()


def test_ts_zscore_null_on_constant_and_matches_numpy():
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


def test_ts_skew_symmetric_is_zero():
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


def test_ts_skew_matches_numpy_ground_truth():
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


def test_ts_rank_matches_manual():
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


def test_leaf_features_contains_price_volume_and_fundamental():
    from factorzen.discovery.operators import LEAF_FEATURES
    price_vol_leaves = {"close", "open", "high", "low", "vol", "amount", "vwap", "log_vol", "ret_1d"}
    fundamental_leaves = {"total_mv", "circ_mv", "pb", "pe_ttm", "ps_ttm", "dv_ttm"}
    keys = set(LEAF_FEATURES.keys())
    assert price_vol_leaves <= keys, f"missing price/vol leaves: {price_vol_leaves - keys}"
    assert fundamental_leaves <= keys, f"missing fundamental leaves: {fundamental_leaves - keys}"


def test_basic_features_include_turnover_and_shares():
    from factorzen.discovery.operators import BASIC_FEATURES, LEAF_FEATURES
    for f in ["turnover_rate", "turnover_rate_f", "volume_ratio", "float_share"]:
        assert f in BASIC_FEATURES, f"BASIC_FEATURES missing {f}"
        assert f in LEAF_FEATURES, f"LEAF_FEATURES missing {f}"
    # 原有 6 个基本面叶子仍在
    assert {"total_mv", "circ_mv", "pb", "pe_ttm", "ps_ttm", "dv_ttm"} <= BASIC_FEATURES


def test_operator_category_assignments():
    from factorzen.discovery.operators import OPERATORS
    assert OPERATORS["ts_mean"].category == "ts"
    assert OPERATORS["pct_change"].category == "ts"
    assert OPERATORS["rank"].category == "cs"
    assert OPERATORS["add"].category == "arith"


def test_arith_neg_square_inv():
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


def test_arith_max_min_horizontal():
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


def test_arith_inv_null_on_zero():
    from factorzen.discovery.operators import OPERATORS
    df = pl.DataFrame({"trade_date": [0, 1], "ts_code": ["A", "A"], "x": [0.0, 4.0]})
    got = df.with_columns(OPERATORS["inv"].build([pl.col("x")], None).alias("i"))["i"].to_list()
    assert got[0] is None                      # 1/0 → null(安全除法)
    assert got[1] is not None and abs(got[1] - 0.25) < 1e-12

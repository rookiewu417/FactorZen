# tests/test_stats_rank_parity.py
"""RankIC ties 口径：average-rank Spearman 单一实现 + 四消费方 parity。

契约：
- ties 取组内平均秩（1-based），mergesort 保确定性；行序打乱结果不变
- 与 polars rank(method="average") + Pearson 主口径差 < 1e-12
- lift_test / experiment 日 IC 序列一致
- 退化：n<2、常数列 → None / 跳过
"""
from __future__ import annotations

import numpy as np
import polars as pl

# ── 旧 ordinal 双 argsort（TDD 反例：ties 下行序敏感）──────────────────────


def _ordinal_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """历史双 argsort 实现：ties 依行序，非确定性。"""
    fr = a.argsort().argsort().astype(float)
    rr = b.argsort().argsort().astype(float)
    return float(np.corrcoef(fr, rr)[0, 1])


def _polars_spearman(a: np.ndarray, b: np.ndarray) -> float:
    df = pl.DataFrame({"a": a, "b": b})
    ranked = df.with_columns(
        pl.col("a").rank(method="average").alias("ra"),
        pl.col("b").rank(method="average").alias("rb"),
    )
    return float(ranked.select(pl.corr("ra", "rb")).item())


def _panel_with_ties(n_days: int = 5, n_stocks: int = 12, seed: int = 0):
    """构造含大量 ties 的小面板（factor 重复值多，ret 略相关）。"""
    rng = np.random.default_rng(seed)
    dates = [f"2024010{i + 1}" for i in range(n_days)]
    codes = [f"{600000 + s:06d}.SH" for s in range(n_stocks)]
    rows_f, rows_r = [], []
    for d in dates:
        # 大量 ties：仅 4 个离散档
        f = rng.choice([1.0, 1.0, 1.0, 2.0, 2.0, 3.0], size=n_stocks)
        r = 0.3 * f + rng.normal(0, 1.0, size=n_stocks)
        for s, code in enumerate(codes):
            rows_f.append({
                "trade_date": d, "ts_code": code, "factor_value": float(f[s]),
            })
            rows_r.append({
                "trade_date": d, "ts_code": code, "ret": float(r[s]),
            })
    return pl.DataFrame(rows_f), pl.DataFrame(rows_r)


# ── core.stats 单元 ────────────────────────────────────────────────────────


def test_avg_rank_ties_average_and_one_based():
    from factorzen.core.stats import avg_rank

    x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    # ranks: 1,1,1 → avg 2.0; 2,2 → avg 4.5; 3 → 6.0
    got = avg_rank(x)
    np.testing.assert_allclose(got, [2.0, 2.0, 2.0, 4.5, 4.5, 6.0])


def test_spearman_avg_rank_row_order_invariant_on_ties():
    """含 ties 时打乱行序结果完全相等（修复后）；旧 ordinal 会变。"""
    from factorzen.core.stats import spearman_avg_rank

    f = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    r = np.array([0.1, 0.2, 0.15, 0.5, 0.4, 0.9])
    base = spearman_avg_rank(f, r)
    assert base is not None

    rng = np.random.default_rng(0)
    for _ in range(20):
        perm = rng.permutation(f.size)
        got = spearman_avg_rank(f[perm], r[perm])
        assert got == base, f"avg-rank 行序敏感: {got} vs {base}"

    # TDD 反例：旧 ordinal 对同一配对打乱行序后结果会变
    ord_same_pair = set()
    for _ in range(30):
        perm = rng.permutation(f.size)
        ord_same_pair.add(_ordinal_spearman(f[perm], r[perm]))
    assert len(ord_same_pair) > 1, "旧 ordinal 应在 ties 下随行序变化（TDD 反例）"


def test_spearman_avg_rank_matches_polars_average():
    from factorzen.core.stats import spearman_avg_rank

    rng = np.random.default_rng(7)
    # 无 ties
    a = rng.standard_normal(80)
    b = 0.5 * a + rng.standard_normal(80)
    assert abs(spearman_avg_rank(a, b) - _polars_spearman(a, b)) < 1e-12

    # 含 ties
    a_t = rng.choice([0.0, 1.0, 1.0, 2.0, 3.0, 3.0], size=60).astype(float)
    b_t = 0.4 * a_t + rng.normal(0, 1.0, size=60)
    assert abs(spearman_avg_rank(a_t, b_t) - _polars_spearman(a_t, b_t)) < 1e-12


def test_spearman_avg_rank_degenerate_guards():
    from factorzen.core.stats import spearman_avg_rank

    assert spearman_avg_rank(np.array([1.0]), np.array([2.0])) is None
    assert spearman_avg_rank(np.array([]), np.array([])) is None
    # 常数列
    assert spearman_avg_rank(np.ones(10), np.arange(10, dtype=float)) is None
    assert spearman_avg_rank(np.arange(10, dtype=float), np.ones(10)) is None
    # 双侧常数
    assert spearman_avg_rank(np.ones(5), np.ones(5)) is None


# ── 四消费方：行序不变 + lift/experiment parity ───────────────────────────


def test_daily_oos_rank_ic_row_order_invariant_with_ties():
    """修复后 lift_test 日 IC 不随截面行序变化。"""
    from factorzen.discovery.lift_test import _daily_oos_rank_ic

    combined, ret_df = _panel_with_ties()
    base = _daily_oos_rank_ic(combined, ret_df)
    assert base.height > 0

    shuffled = (
        combined.with_row_index("_i")
        .with_columns((pl.col("_i") * 17 % 97).alias("_k"))
        .sort(["trade_date", "_k"])
        .drop(["_i", "_k"])
    )
    got = _daily_oos_rank_ic(shuffled, ret_df)
    assert got["trade_date"].to_list() == base["trade_date"].to_list()
    np.testing.assert_allclose(
        got["ic"].to_numpy(), base["ic"].to_numpy(), rtol=0, atol=0,
    )


def test_evaluate_oos_row_order_invariant_with_ties():
    from factorzen.research.combination.experiment import _evaluate_oos

    combined, ret_df = _panel_with_ties()
    base = _evaluate_oos(combined, ret_df)["rank_ic_mean"]
    shuffled = (
        combined.with_row_index("_i")
        .with_columns((pl.col("_i") * 13 % 89).alias("_k"))
        .sort(["trade_date", "_k"])
        .drop(["_i", "_k"])
    )
    got = _evaluate_oos(shuffled, ret_df)["rank_ic_mean"]
    assert got == base


def test_lift_and_evaluate_oos_daily_ic_parity_with_ties():
    """同一小面板：lift_test 日 IC 序列与 experiment rank_ic_mean 分量一致。"""
    from factorzen.discovery.lift_test import _daily_oos_rank_ic
    from factorzen.research.combination.experiment import _evaluate_oos

    combined, ret_df = _panel_with_ties(n_days=8, n_stocks=20, seed=11)
    daily = _daily_oos_rank_ic(combined, ret_df)
    mean_daily = float(daily["ic"].mean())
    ref = float(_evaluate_oos(combined, ret_df)["rank_ic_mean"])
    assert abs(mean_daily - ref) < 1e-12


def test_methods_rank_ic_numpy_uses_avg_rank_on_ties():
    from factorzen.core.stats import spearman_avg_rank
    from factorzen.research.combination.methods import _rank_ic_numpy

    f = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0])
    r = np.array([0.1, 0.2, 0.15, 0.5, 0.4, 0.9, 0.85, 1.1, 1.0, 1.3, 1.2, 1.5])
    got = _rank_ic_numpy(f, r)
    exp = spearman_avg_rank(f, r)
    assert got is not None and exp is not None
    assert got == exp

    # 行序不变
    perm = np.array([3, 0, 5, 1, 8, 2, 10, 4, 7, 11, 6, 9])
    assert _rank_ic_numpy(f[perm], r[perm]) == got


def test_residual_spearman_is_avg_rank_reexport():
    """residual._spearman 为 core.stats 薄封装/再导出，语义一致。"""
    from factorzen.core.stats import spearman_avg_rank
    from factorzen.discovery.residual import _spearman

    f = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    r = np.array([0.2, 0.1, 0.5, 0.4, 0.9])
    assert _spearman(f, r) == spearman_avg_rank(f, r)
    assert _spearman(np.ones(4), np.arange(4.0)) is None


def test_daily_oos_skips_constant_cross_section():
    from factorzen.discovery.lift_test import _daily_oos_rank_ic

    dates = ["20240101", "20240102"]
    codes = [f"{i:04d}.SZ" for i in range(12)]
    rows_f, rows_r = [], []
    for d in dates:
        for s, code in enumerate(codes):
            rows_f.append({
                "trade_date": d, "ts_code": code, "factor_value": 1.0,  # 常数
            })
            rows_r.append({
                "trade_date": d, "ts_code": code, "ret": float(s),
            })
    daily = _daily_oos_rank_ic(pl.DataFrame(rows_f), pl.DataFrame(rows_r))
    assert daily.is_empty()

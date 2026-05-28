"""性能基准测试：评估层 polars 化前后耗时对比。

运行方式：
    pixi run python -m pytest benchmarks/bench_evaluation.py -v -s

说明：
- "polars vectorized" 版本使用 _rank_ic_by_date（group_by + pl.corr）
- "legacy loop" 版本模拟旧版逐日 filter + scipy.stats.spearmanr
- 5 年日频 × 3000 只股票的合成数据
"""

import time
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from scipy.stats import spearmanr


def _make_large_dataset(n_years: int = 5, n_stocks: int = 3000, seed: int = 42):
    """生成约 5 年 × 3000 股 的合成因子+收益 DataFrame。"""
    rng = np.random.default_rng(seed)
    n_days = n_years * 252

    start = date(2019, 1, 2)
    # 生成 n_days 个工作日
    trade_dates = []
    d = start
    while len(trade_dates) < n_days:
        if d.weekday() < 5:
            trade_dates.append(d)
        d += timedelta(days=1)

    total_rows = n_days * n_stocks
    stocks = [f"{i:06d}.SH" for i in range(n_stocks)]
    factor_vals = rng.standard_normal(total_rows).astype(np.float32)
    ret_vals = (rng.standard_normal(total_rows) * 0.01).astype(np.float32)

    dates_repeated = trade_dates * n_stocks
    dates_repeated.sort()  # 按日期排序更接近实际

    from datetime import date as date_t

    # 使用 from_epoch 绕过 object dtype 问题
    epoch_days = [(d - date_t(1970, 1, 1)).days for d in np.repeat(trade_dates, n_stocks)]

    return pl.DataFrame(
        {
            "trade_date": pl.Series(epoch_days).cast(pl.Int32).cast(pl.Date),
            "ts_code": np.tile(stocks, n_days),
            "factor_clean": factor_vals,
            "fwd_ret_1d": ret_vals,
        }
    )


@pytest.fixture(scope="module")
def large_df():
    print("\n生成测试数据（5 年 × 3000 股）...")
    t0 = time.perf_counter()
    df = _make_large_dataset()
    print(f"  生成耗时: {time.perf_counter() - t0:.2f}s，形状: {df.shape}")
    return df


def test_polars_vectorized_ic(large_df, benchmark):
    """polars group_by + pl.corr 实现（当前版本）。"""
    from daily.evaluation.ic_analysis import _rank_ic_by_date

    def run():
        return _rank_ic_by_date(large_df, "factor_clean", "fwd_ret_1d")

    result = benchmark(run)
    ic_series = result["ic"].drop_nulls().to_numpy()
    print(f"\n  IC Mean: {np.mean(ic_series):.4f}, periods: {len(ic_series)}")


def test_legacy_loop_ic(large_df, benchmark):
    """旧版 Python for-loop + scipy.stats.spearmanr（仅 200 天，否则太慢）。"""
    subset = large_df.head(200 * 3000)  # 只取 200 个交易日

    def run():
        trade_dates = subset["trade_date"].unique().sort().to_list()
        ics = []
        for d in trade_dates:
            cross = subset.filter(pl.col("trade_date") == d)
            x = cross["factor_clean"].to_numpy()
            y = cross["fwd_ret_1d"].to_numpy()
            valid = ~np.isnan(x) & ~np.isnan(y)
            if valid.sum() < 30:
                continue
            ic, _ = spearmanr(x[valid], y[valid])
            if not np.isnan(ic):
                ics.append(ic)
        return ics

    result = benchmark(run)
    print(f"\n  IC Mean (200 days): {np.mean(result):.4f}")


def test_speedup_report(large_df):
    """打印 polars vs legacy 在 200 天上的实际加速比。"""
    from daily.evaluation.ic_analysis import _rank_ic_by_date

    subset = large_df.head(200 * 3000)

    t0 = time.perf_counter()
    _rank_ic_by_date(subset, "factor_clean", "fwd_ret_1d")
    polars_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    trade_dates = subset["trade_date"].unique().sort().to_list()
    for d in trade_dates:
        cross = subset.filter(pl.col("trade_date") == d)
        x = cross["factor_clean"].to_numpy()
        y = cross["fwd_ret_1d"].to_numpy()
        valid = ~np.isnan(x) & ~np.isnan(y)
        if valid.sum() >= 30:
            spearmanr(x[valid], y[valid])
    legacy_time = time.perf_counter() - t0

    speedup = legacy_time / polars_time if polars_time > 0 else float("inf")
    print(
        f"\n  polars: {polars_time:.3f}s  |  legacy loop: {legacy_time:.3f}s  |  加速比: {speedup:.1f}x"
    )
    assert speedup > 5, f"期望加速比 > 5x，实际 {speedup:.1f}x"

"""市值中性化在真实 A 股市值量级下必须真正剥离 size 暴露（P0）。

根因：neutralize_ols 对 log 市值做 max(mv, 1e6) 下限截断，而 Tushare total_mv 单位是
万元——总市值 <100亿元（total_mv<1e6 万元，A股大多数）全被夹成常数 log(1e6)，size 回归
列退化为常数，市值中性化实际失效；缺失值又填 1e8（万亿元）巨常数污染回归。
"""
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl


def _cross_section(n=60, seed=0):
    """构造单日截面：total_mv 在 2万~90万（万元）即 2亿~90亿元，A股典型量级。"""
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}.SZ" for i in range(n)]
    total_mv = rng.uniform(2e4, 9e5, n)  # 单位万元
    # 因子几乎完全由 log(市值) 解释 → 正确中性化后残差应与 log(mv) 近乎不相关
    factor = 2.0 * np.log(total_mv) + rng.normal(0, 0.01, n)
    d = date(2024, 3, 1)
    df = pl.DataFrame({"trade_date": [d] * n, "ts_code": codes, "factor_value": factor})
    daily_basic = pl.DataFrame({"trade_date": [d] * n, "ts_code": codes, "total_mv": total_mv})
    return df, daily_basic, total_mv


def test_size_neutralization_removes_log_mv_exposure():
    from factorzen.daily.preprocessing.neutralizer import neutralize_ols

    df, daily_basic, total_mv = _cross_section()
    out = neutralize_ols(df, col="factor_value", daily_basic=daily_basic)
    resid = out["factor_value_neutral"].to_numpy()

    finite = ~np.isnan(resid)
    assert finite.sum() >= 30
    corr = np.corrcoef(resid[finite], np.log(total_mv[finite]))[0, 1]
    assert abs(corr) < 0.15, (
        f"市值中性化后残差与 log(mv) 相关系数 {corr:.3f} 应≈0；修复前因 log 被夹成常数、"
        "size 回归列退化，相关系数≈1（中性化失效）"
    )


def test_missing_mv_row_excluded_not_filled_with_giant_constant():
    """缺失 total_mv 的股票应被剔除（残差 NaN），而不是用 1e8 巨常数冒充参与回归。"""
    from factorzen.daily.preprocessing.neutralizer import neutralize_ols

    df, daily_basic, _ = _cross_section(n=60, seed=1)
    # 抹掉一只股票的 total_mv 行
    missing_code = df["ts_code"][0]
    daily_basic = daily_basic.filter(pl.col("ts_code") != missing_code)
    out = neutralize_ols(df, col="factor_value", daily_basic=daily_basic).sort("ts_code")
    row = out.filter(pl.col("ts_code") == missing_code)
    assert row["factor_value_neutral"].to_numpy()[0] != row["factor_value_neutral"].to_numpy()[0], (
        "缺失市值的股票中性化结果应为 NaN（被剔除），而非用巨常数冒充"
    )

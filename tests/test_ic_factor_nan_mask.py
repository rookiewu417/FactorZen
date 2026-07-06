"""Rank IC 有效掩码必须同时过滤因子列的 NaN/inf（D1）。

根因：_rank_ic_by_date（及 advanced/_common、sector_ic、size_ic 的同款掩码）对因子列
只有 is_not_null()，缺 is_finite()——polars 中 NaN 不是 null，rank 把 NaN 排为最大值，
NaN 因子行以最高秩参与 Rank IC，污染结果。剔除 NaN 行后 IC 应不变。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl


def _panel(n_days=6, n_stocks=40, seed=0):
    rng = np.random.default_rng(seed)
    rows_f = []
    rows_r = []
    d0 = date(2024, 1, 1)
    for di in range(n_days):
        d = d0 + timedelta(days=di)
        x = rng.normal(0, 1, n_stocks)
        fwd = x * 0.8 + rng.normal(0, 0.3, n_stocks)  # 强相关截面
        for si in range(n_stocks):
            code = f"{si:06d}.SZ"
            rows_f.append({"trade_date": d, "ts_code": code, "factor_clean": float(x[si])})
            rows_r.append({"trade_date": d, "ts_code": code, "fwd_ret_1d": float(fwd[si])})
    return pl.DataFrame(rows_f), pl.DataFrame(rows_r)


def test_factor_nan_row_equivalent_to_dropped_row():
    """不变量：因子为 NaN 的行应与该行被物理删除等价——NaN 不得以最高秩污染 IC。"""
    from factorzen.daily.evaluation.ic_analysis import compute_rank_ic

    factor_df, ret_df = _panel()
    codes8 = [f"{i:06d}.SZ" for i in range(8)]

    # A) 每日前 8 只股票的因子值置 NaN（收益不变）
    nan_df = factor_df.with_columns(
        pl.when(pl.col("ts_code").is_in(codes8))
        .then(float("nan"))
        .otherwise(pl.col("factor_clean"))
        .alias("factor_clean")
    )
    # B) 同样这 8 只股票的行直接删除（ground truth）
    drop_df = factor_df.filter(~pl.col("ts_code").is_in(codes8))

    nan_ic = compute_rank_ic(nan_df, ret_df, factor_col="factor_clean").ic_mean
    drop_ic = compute_rank_ic(drop_df, ret_df, factor_col="factor_clean").ic_mean

    assert abs(nan_ic - drop_ic) < 1e-9, (
        f"NaN 因子行应等价于删除该行，nan_ic={nan_ic:.6f} vs drop_ic={drop_ic:.6f}"
        "（修复前 NaN 被 rank 排最大参与 IC，两者不等）"
    )


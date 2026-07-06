"""截面标准化的 NaN 处理回归测试。

D2：cross_sectional_zscore 遇截面内任一 NaN，mean/std 被 NaN 传染 → 整个交易日全部
    z 值变 NaN，整日截面被静默摧毁。
D3：cross_sectional_rank 不剔除 NaN，polars rank 把 NaN 排为最大 → NaN 股票获最高分位，
    TopN/分层策略会把因子为 NaN 的股票当最强信号买入。
修复：聚合/排名前把 NaN 视作缺失（fill_nan(None)），NaN 输入 → 缺失输出，不污染同日其他股票。
"""
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl


def test_zscore_single_nan_does_not_poison_whole_day():
    from factorzen.daily.preprocessing.normalizer import cross_sectional_zscore

    d = date(2024, 3, 1)
    df = pl.DataFrame({
        "trade_date": [d, d, d, d],
        "factor_value_clip_fill": [1.0, 2.0, 3.0, float("nan")],
    })
    out = cross_sectional_zscore(df, col="factor_value_clip_fill")
    z = out["factor_value_clip_fill_z"].to_numpy()

    finite = np.isfinite(z)
    assert finite.sum() == 3, f"3 个有效值应得到有效 z（修复前整日被 NaN 传染），实得 {finite.sum()}"
    # 有效行是标准 z-score：均值≈0
    assert abs(np.nanmean(z[finite])) < 1e-9
    assert not np.isfinite(z[3]), "NaN 输入行应保持非有限，不应被置 0 或污染"


def test_rank_nan_not_ranked_highest():
    from factorzen.daily.preprocessing.normalizer import cross_sectional_rank

    d = date(2024, 3, 1)
    df = pl.DataFrame({
        "trade_date": [d] * 5,
        "ts_code": ["A", "B", "C", "D", "E"],
        "factor_value": [1.0, 2.0, 3.0, float("nan"), 4.0],
    })
    out = cross_sectional_rank(df, factor_col="factor_value", method="uniform").sort("ts_code")
    vals = dict(zip(out["ts_code"].to_list(), out["factor_value"].to_list(), strict=False))

    nan_score = vals["D"]
    max_real_score = vals["E"]  # 真实最大值 4.0
    assert nan_score is None or not np.isfinite(nan_score), (
        f"NaN 因子应得缺失分位（修复前 polars rank 排最大得最高分位 {nan_score}）"
    )
    assert max_real_score is not None and np.isfinite(max_real_score)
    # 真实最大值应拿到最高的有效分位
    finite_scores = [v for v in vals.values() if v is not None and np.isfinite(v)]
    assert max_real_score == max(finite_scores), "真实最大值应获最高有效分位，而非被 NaN 顶替"

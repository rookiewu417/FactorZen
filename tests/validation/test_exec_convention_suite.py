"""P0 成交口径贯通：holdout 主门 / 残差 holdout 数值差异与零回归。

修前必须 FAIL（主门不透传 exec → 两口径 IC 相同）。
禁止 inspect.signature 断言；用合成面板上的数值差异做判别。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest


def _synth_exec_panel(n_stocks: int = 40, n_days: int = 50, seed: int = 7) -> pl.DataFrame:
    """≥30 只股票（IC 截面门槛）、≥40 天；open_adj = close_adj * 1.03 使 lag 切换后前向收益不同。"""
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)
    days: list[date] = []
    d = start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for i in range(n_stocks):
        code = f"{i:06d}.SH"
        p = 10.0 + i
        for day in days:
            p = float(max(p * (1.0 + rng.standard_normal() * 0.02), 0.5))
            rows.append({
                "trade_date": day,
                "ts_code": code,
                "close": p,
                "close_adj": p,
                # 同日比例缩放：lag0 close 收益 ≠ lag1 open 收益（时间平移）
                "open_adj": p * 1.03,
                "open": p * 1.03,
                "vol": 1e5,
            })
    return pl.DataFrame(rows)


def _factor_aligned_to_close_fwd(holdout: pl.DataFrame) -> pl.DataFrame:
    """因子 = close→close 次日收益 → 默认口径 holdout IC 应明显为正。"""
    return (
        holdout.sort(["ts_code", "trade_date"])
        .with_columns(
            (pl.col("close_adj").shift(-1).over("ts_code") / pl.col("close_adj") - 1.0)
            .alias("factor_value")
        )
        .select(["trade_date", "ts_code", "factor_value"])
        .drop_nulls()
    )


def test_holdout_ic_exec_convention_numeric_diff_and_zero_regression():
    """A: exec 口径 IC ≠ 默认；B: 默认签名 ≡ compute_fwd_returns 默认口径 ground-truth。"""
    from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns, compute_rank_ic
    from factorzen.daily.preprocessing.normalizer import cross_sectional_zscore
    from factorzen.validation.holdout import holdout_ic_result

    panel = _synth_exec_panel()
    # 整板当 holdout，保证有效截面天数
    holdout = panel
    fac = _factor_aligned_to_close_fwd(holdout)

    # ── A: 两口径数值必须不同（修前恒 close→close → 同值 → FAIL）──────────────
    r_default = holdout_ic_result(fac, holdout)
    r_exec = holdout_ic_result(
        fac, holdout, exec_lag=1, exec_price_col="open_adj",
    )
    assert r_default.n_days > 5 and r_exec.n_days > 5
    assert abs(r_default.ic_mean - r_exec.ic_mean) > 1e-6, (
        f"exec 口径应改变 holdout IC；"
        f"default={r_default.ic_mean}, exec={r_exec.ic_mean}"
    )

    # ── B: 默认参数 ≡ 独立 ground-truth（compute_fwd_returns 默认，非自证）────
    price_col = "close_adj"
    gt_fwd = compute_fwd_returns(
        holdout.sort(["ts_code", "trade_date"]), price_col=price_col,
    )
    clean = cross_sectional_zscore(fac, col="factor_value").rename(
        {"factor_value_z": "factor_clean"}
    )
    gt = compute_rank_ic(
        clean.select(["trade_date", "ts_code", "factor_clean"]),
        gt_fwd,
        factor_col="factor_clean",
        frequency="daily",
    )
    assert r_default.ic_mean == pytest.approx(float(gt.ic_mean), abs=1e-12)
    # 新签名默认参数与无参调用一致
    r_default_kw = holdout_ic_result(fac, holdout, exec_lag=0, exec_price_col=None)
    assert r_default.ic_mean == pytest.approx(r_default_kw.ic_mean, abs=1e-12)
    assert r_default.n_days == r_default_kw.n_days


def test_holdout_fwd_helper_exec_numeric_diff():
    """C: 残差 holdout 路径 helper——exec 前向收益与默认口径数值不同。

    直接测 holdout_fwd_returns（nodes 残差 _hold_fwd 同源）；禁止 signature 断言。
    """
    from factorzen.validation.holdout import holdout_fwd_returns

    holdout = _synth_exec_panel(n_stocks=2)
    fwd0 = holdout_fwd_returns(holdout)
    fwd1 = holdout_fwd_returns(holdout, exec_lag=1, exec_price_col="open_adj")
    col = "fwd_ret_1d"
    joined = (
        fwd0.select(["ts_code", "trade_date", pl.col(col).alias("r0")])
        .join(
            fwd1.select(["ts_code", "trade_date", pl.col(col).alias("r1")]),
            on=["ts_code", "trade_date"],
            how="inner",
        )
        .drop_nulls()
    )
    assert joined.height > 10
    diffs = (joined["r0"] - joined["r1"]).abs().to_numpy()
    assert float(np.nanmax(diffs)) > 1e-6, "exec_lag/open_adj 须系统改变 holdout 前向收益"

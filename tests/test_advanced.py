"""高级评估保留能力：``ic_analysis`` 内置 multi_period / decay，以及单调性 re-export。

独立 IC 衰减模块已删除；衰减口径改由 ``compute_rank_ic`` 的
``decay`` / ``multi_period`` 字段提供。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from factorzen.daily.evaluation.advanced import MonotonicityResult, compute_monotonicity
from factorzen.daily.evaluation.ic_analysis import compute_rank_ic

_N_STOCKS = 40  # 必须 ≥ _MIN_CROSS_SAMPLES(=30)
_N_DAYS = 30


def _make_factor_and_returns(seed: int = 3) -> tuple[pl.DataFrame, pl.DataFrame]:
    """带递减信噪比的合成数据：|IC|(1d) > |IC|(5d) > |IC|(10d)。"""
    rng = np.random.default_rng(seed)
    dates = pl.date_range(
        pl.date(2026, 1, 5),
        pl.date(2026, 1, 5) + pl.duration(days=_N_DAYS - 1),
        interval="1d",
        eager=True,
    )
    codes = [f"{600000 + i:06d}.SH" for i in range(_N_STOCKS)]
    f = rng.standard_normal(_N_STOCKS)

    rows_f, rows_r = [], []
    for d in dates:
        for i, code in enumerate(codes):
            rows_f.append({"trade_date": d, "ts_code": code, "factor_clean": float(f[i])})
            rows_r.append(
                {
                    "trade_date": d,
                    "ts_code": code,
                    "fwd_ret_1d": float(f[i] + 0.5 * rng.standard_normal()),
                    "fwd_ret_5d": float(f[i] + 2.0 * rng.standard_normal()),
                    "fwd_ret_10d": float(f[i] + 5.0 * rng.standard_normal()),
                    "fwd_ret_20d": float(f[i] + 8.0 * rng.standard_normal()),
                }
            )
    return pl.DataFrame(rows_f), pl.DataFrame(rows_r)


def test_rank_ic_multi_period_covers_horizons():
    factor, ret = _make_factor_and_returns()
    result = compute_rank_ic(factor, ret, factor_col="factor_clean", horizons=[1, 5, 10, 20])

    assert sorted(result.multi_period.keys()) == [1, 5, 10, 20]
    assert sorted(result.decay.keys()) == [1, 5, 10, 20]
    for h, stats in result.multi_period.items():
        assert "ic_mean" in stats and "ir" in stats
        assert stats["ic_mean"] == pytest.approx(result.decay[h])


def test_rank_ic_decay_is_monotonically_decreasing_in_abs():
    factor, ret = _make_factor_and_returns()
    result = compute_rank_ic(factor, ret, factor_col="factor_clean", horizons=[1, 5, 10])
    ic = result.decay

    assert all(v == v for v in ic.values()), f"IC 不该是 nan：{ic}"
    assert ic[1] > 0.5, f"1 日 IC 应显著为正，实得 {ic[1]:.4f}"
    assert abs(ic[1]) > abs(ic[5]) > abs(ic[10]), f"IC 未随持有期衰减：{ic}"


def test_rank_ic_series_covers_trading_days():
    factor, ret = _make_factor_and_returns()
    result = compute_rank_ic(factor, ret, factor_col="factor_clean", horizons=[1, 5, 10])

    assert result.n_periods == _N_DAYS
    assert result.ic_series.height == _N_DAYS
    assert result.ic_std > 0


def test_monotonicity_reexport_runs():
    """advanced 包仍导出 compute_monotonicity（管线在用）。"""
    factor, ret = _make_factor_and_returns()
    mono_df = factor.join(
        ret.select(["trade_date", "ts_code", "fwd_ret_1d"]),
        on=["trade_date", "ts_code"],
        how="inner",
    )
    mono = compute_monotonicity(
        mono_df, factor_col="factor_clean", ret_col="fwd_ret_1d", n_groups=5
    )
    assert isinstance(mono, MonotonicityResult)
    assert len(mono.group_means) == 5

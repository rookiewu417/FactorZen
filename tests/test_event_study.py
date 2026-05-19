"""事件研究测试。"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl


def _make_factor_ret_dfs(n_stocks: int = 20, n_dates: int = 60, seed: int = 42):
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 3)
    dates_list = [(start + timedelta(days=i)).isoformat() for i in range(n_dates)]

    factor_rows, ret_rows = [], []
    for dt in dates_list:
        for s in range(n_stocks):
            ts = f"00{s:04d}.SZ"
            factor_rows.append(
                {"trade_date": dt, "ts_code": ts, "factor_clean": float(rng.standard_normal())}
            )
            ret_rows.append(
                {
                    "trade_date": dt,
                    "ts_code": ts,
                    "ret_1d": float(rng.standard_normal() * 0.01),
                }
            )

    return pl.DataFrame(factor_rows), pl.DataFrame(ret_rows)


def test_event_study_returns_correct_windows():
    from daily.evaluation.advanced import compute_event_study

    factor_df, ret_df = _make_factor_ret_dfs()
    result = compute_event_study(factor_df, ret_df, pre_window=3, post_window=10)
    assert result.n_events > 0
    expected_len = 3 + 1 + 10  # pre + event_day + post
    assert len(result.windows) == expected_len
    assert len(result.avg_cumret) == expected_len
    assert len(result.ci_95) == expected_len


def test_event_study_ci_nonnegative():
    from daily.evaluation.advanced import compute_event_study

    factor_df, ret_df = _make_factor_ret_dfs(seed=0)
    result = compute_event_study(factor_df, ret_df, pre_window=2, post_window=5)
    assert (result.ci_95 >= 0).all()


def test_event_study_windows_range():
    """windows 列表应从 -pre_window 到 +post_window。"""
    from daily.evaluation.advanced import compute_event_study

    factor_df, ret_df = _make_factor_ret_dfs()
    pre, post = 4, 8
    result = compute_event_study(factor_df, ret_df, pre_window=pre, post_window=post)
    assert result.windows[0] == -pre
    assert result.windows[-1] == post


def test_event_study_empty_factor():
    """空因子 DataFrame 应返回零结果不崩溃。"""
    from daily.evaluation.advanced import compute_event_study

    empty_factor = pl.DataFrame(
        {"trade_date": [], "ts_code": [], "factor_clean": []},
        schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8, "factor_clean": pl.Float64},
    )
    ret_df = pl.DataFrame(
        {"trade_date": [], "ts_code": [], "ret_1d": []},
        schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8, "ret_1d": pl.Float64},
    )
    result = compute_event_study(empty_factor, ret_df, pre_window=2, post_window=5)
    assert result.n_events == 0
    assert len(result.windows) == 2 + 1 + 5

"""事件研究窗口累计收益的基准与 off-by-one 回归（P0）。

根因：事件日收益被同时计入 w=0 及所有事件前窗口点，w=0 点算成 -r0 而非 0，且事件后
窗口 daily_arr[base_idx:i+1] 含事件日收益 → 曲线在 w=0→w=+1 处出现约 2*r0 跳变。
正确约定：以事件日收盘为基准 cumret[w]=price[w]/price[0]-1，w=0 恒为 0，事件后不含
事件日收益。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl


def _constant_return_panel(daily_ret=0.01, n_days=40, n_stocks=10, seed=0):
    """所有股票每天恒定 +1% 收益；某只固定股票在中段某日为 top 分位事件。"""
    rng = np.random.default_rng(seed)
    d0 = date(2024, 1, 1)
    dates = [d0 + timedelta(days=i) for i in range(n_days)]
    rows_f, rows_r = [], []
    for di, d in enumerate(dates):
        for si in range(n_stocks):
            code = f"{si:06d}.SZ"
            # 让 000000 在第 20 天成为 top 事件，其它日/其它股票因子小
            fval = 100.0 if (si == 0 and di == 20) else rng.uniform(-1, 1)
            rows_f.append({"trade_date": d, "ts_code": code, "factor_clean": fval})
            rows_r.append({"trade_date": d, "ts_code": code, "ret_1d": daily_ret})
    return pl.DataFrame(rows_f), pl.DataFrame(rows_r)


def test_event_day_baseline_is_zero_and_no_double_count():
    from factorzen.daily.evaluation.advanced.event_study import compute_event_study

    factor_df, ret_df = _constant_return_panel(daily_ret=0.01)
    res = compute_event_study(
        factor_df, ret_df, event_threshold=0.95, pre_window=3, post_window=3,
    )
    assert res.n_events >= 1
    w = res.windows
    cum = res.avg_cumret
    base = w.index(0)

    # w=0 必须是基准 0
    assert abs(cum[base]) < 1e-12, f"w=0 应为基准 0，实得 {cum[base]:.4f}（修复前为 -r0）"

    # 事件后 w=+1 只含 r1=1%（不含事件日收益），w=+k = 1.01^k - 1
    assert abs(cum[base + 1] - 0.01) < 1e-9, f"w=+1 应为 r1=0.01，实得 {cum[base+1]:.4f}"
    assert abs(cum[base + 2] - (1.01**2 - 1)) < 1e-9
    assert abs(cum[base + 3] - (1.01**3 - 1)) < 1e-9

    # w=0→w=+1 的跳变应≈r0=1%，而非修复前的 ≈2%
    assert abs((cum[base + 1] - cum[base]) - 0.01) < 1e-9, "w=0→+1 不应出现 2*r0 跳变"

    # 事件前 w=-k = price[-k]/price[0]-1 = 1/1.01^k - 1（负、单调）
    assert abs(cum[base - 1] - (1.0 / 1.01 - 1.0)) < 1e-9
    assert abs(cum[base - 2] - (1.0 / 1.01**2 - 1.0)) < 1e-9
    assert cum[base - 3] < cum[base - 2] < cum[base - 1] < 0.0

"""fz report build 前向收益/IC 标签须用复权收盘价，与 fz factor test 口径一致，
避免除权除息日 close 跳空污染 IC/单调性/分层 IC。"""
from __future__ import annotations

from datetime import date

import polars as pl


def test_attach_close_adj_derives_adjusted_close():
    from factorzen.pipelines.generate_report import _attach_close_adj

    daily = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 1), date(2024, 1, 2)],
            "ts_code": ["A.SZ", "A.SZ"],
            "close": [10.0, 11.0],
        }
    )
    adj = pl.DataFrame(
        {
            "ts_code": ["A.SZ", "A.SZ"],
            "trade_date": [date(2024, 1, 1), date(2024, 1, 2)],
            "adj_factor": [2.0, 2.0],
        }
    )
    out = _attach_close_adj(daily, adj)
    assert out.sort("trade_date")["close_adj"].to_list() == [20.0, 22.0]


def test_attach_close_adj_empty_adj_returns_unchanged():
    from factorzen.pipelines.generate_report import _attach_close_adj

    daily = pl.DataFrame(
        {"trade_date": [date(2024, 1, 1)], "ts_code": ["A.SZ"], "close": [10.0]}
    )
    out = _attach_close_adj(daily, pl.DataFrame())
    assert "close_adj" not in out.columns  # adj 缺失 → 回退，_build_forward_return_frame 用 close


def test_report_forward_returns_use_adjusted_close_no_ex_div_jump():
    """除权日 close 跳空（送转 10→5）不应污染前向收益——复权后 close_adj 连续。"""
    from factorzen.pipelines.daily_single import _build_forward_return_frame
    from factorzen.pipelines.generate_report import _attach_close_adj

    daily = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "ts_code": ["A.SZ", "A.SZ", "A.SZ"],
            "close": [10.0, 5.0, 5.0],  # d2 送转，未复权价腰斩
        }
    )
    adj = pl.DataFrame(
        {
            "ts_code": ["A.SZ", "A.SZ", "A.SZ"],
            "trade_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "adj_factor": [1.0, 2.0, 2.0],  # 除权后翻倍 → close_adj = [10,10,10] 连续
        }
    )
    daily_adj = _attach_close_adj(daily, adj)
    ret_df = _build_forward_return_frame(daily_adj)
    fwd = ret_df.sort("trade_date")["fwd_ret_1d"].to_list()
    # 复权后 close_adj 连续 → d1→d2 的 fwd_ret_1d = 0，而非未复权的虚假 -50%
    assert abs(fwd[0] - 0.0) < 1e-9, f"复权后不应有除权跳空，实际 fwd_ret_1d={fwd[0]}"

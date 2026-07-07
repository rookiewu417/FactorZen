"""discovery 算子除零守卫防 NaN 穿透 + ts_rank 部分窗口归一化（Wave6 correctness-P2）。"""
from __future__ import annotations

import math

import polars as pl
import pytest

from factorzen.discovery.operators import OPERATORS, _safe_div


def test_safe_div_nan_denominator_returns_none_not_nan():
    d = pl.DataFrame({"b": [1.0, float("nan"), 0.0]})
    out = d.with_columns(_safe_div(pl.lit(1.0), pl.col("b")).alias("r"))["r"].to_list()
    assert out[0] == 1.0
    assert out[1] is None, "NaN 分母应得 None，不应穿透成 NaN"
    assert out[2] is None, "0 分母应得 None"


def test_ts_rank_normalizes_by_actual_window_count():
    """warm-up 期(历史不足 w)ts_rank 须除以窗口内实际样本数，而非固定 w。"""
    d = pl.DataFrame({"ts_code": ["A"] * 4, "trade_date": list(range(4)),
                      "x": [1.0, 2.0, 3.0, 4.0]})  # 单调上升 → 每行都是窗口内最大
    e = OPERATORS["ts_rank"].build([pl.col("x")], 10)
    out = d.with_columns(e.alias("r"))["r"].to_list()
    # 第3、4行是各自窗口内 top(rank=count) → 归一化应为 1.0，而非 3/10、4/10
    assert out[2] == pytest.approx(1.0), f"warm-up top 应为 1.0，实得 {out[2]}"
    assert out[3] == pytest.approx(1.0), f"warm-up top 应为 1.0，实得 {out[3]}"


def test_ts_corr_never_outputs_nan_on_near_constant():
    """近常数序列的微负方差不应经 sqrt 穿透成 NaN。"""
    d = pl.DataFrame({"ts_code": ["A"] * 8, "trade_date": list(range(8)),
                      "a": [1.0 + (i % 2) * 1e-9 for i in range(8)], "b": [2.0] * 8})
    e = OPERATORS["ts_corr"].build([pl.col("a"), pl.col("b")], 5)
    out = d.with_columns(e.alias("r"))["r"].to_list()
    assert not any(v is not None and math.isnan(v) for v in out), f"不应含 NaN：{out}"

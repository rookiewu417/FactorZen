"""截面被 `_MIN_CROSS_SAMPLES` 整天丢光时，必须出声。

`_rank_ic_by_date` 把截面股票数 < 30 的交易日**静默**丢弃。若每一天都不足 30 只，
`ic_series` 为空 → `compute_rank_ic` 返回 sentinel `ic_mean=0.0, ir=0.0, n_periods=0`
（不是 nan），于是「所有因子的 IC 恰好为 0」与「因子确实没有预测力」不可区分。

真实脚枪：`discovery/scoring.py` 对 A 股与 crypto 用同一个 `compute_rank_ic`，
`min_samples=30` 写死。crypto 默认 `top_n=50` 安全，但 `fz mine search --market crypto
--universe-size 20` 会让**每个截面被丢弃、所有因子 IC 恒 0、挖掘一无所获且无一句提示**。

本文件只要求「出声」，不改变任何过滤行为——按市场参数化阈值是另一个议题。
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import polars as pl
import pytest

from factorzen.daily.evaluation import ic_analysis as ia

_LOGGER = "factorzen.daily.evaluation.ic_analysis"


@pytest.fixture(autouse=True)
def _reset_warn_flag(monkeypatch):
    """告警只发一次（挖掘会调用它上千次）。每个测试从干净状态开始。"""
    monkeypatch.setattr(ia, "_warned_thin_cross_section", False, raising=False)


def _frame(n_stocks: int, n_days: int = 5) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for d in range(n_days):
        for i in range(n_stocks):
            rows.append({
                "trade_date": dt.date(2023, 1, 2) + dt.timedelta(days=d),
                "ts_code": f"{i:06d}.SZ",
                "factor": float(rng.normal()),
                "fwd_ret_1d": float(rng.normal()),
            })
    return pl.DataFrame(rows)


def test_warns_when_every_day_is_dropped(caplog):
    """20 只 < 30 → 每天都被丢 → 必须 WARNING，且带上最大截面数供用户排查。"""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = ia._rank_ic_by_date(_frame(20), "factor", "fwd_ret_1d")

    assert out.height == 0
    assert "30" in caplog.text, "告警应说明门槛值"
    assert "20" in caplog.text, "告警应说明实际最大截面数，否则用户无从下手"


def test_no_warning_when_cross_sections_are_thick(caplog):
    """40 只 → 全部保留 → 不该有任何告警。没有这条，「无脑每次都警告」也能过上一个测试。"""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = ia._rank_ic_by_date(_frame(40), "factor", "fwd_ret_1d")

    assert out.height == 5
    assert caplog.text == ""


def test_warning_is_emitted_only_once(caplog):
    """挖掘每评估一个表达式就调一次；每次都警告会把日志淹没。"""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        for _ in range(5):
            ia._rank_ic_by_date(_frame(20), "factor", "fwd_ret_1d")

    assert caplog.text.count("截面") == 1, f"应恰好告警一次，实得：\n{caplog.text}"


def test_no_warning_when_data_is_simply_empty(caplog):
    """全 null 因子是另一种病（不是截面太薄），不该报「截面不足」误导排查方向。"""
    df = _frame(40).with_columns(pl.lit(None, dtype=pl.Float64).alias("factor"))
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = ia._rank_ic_by_date(df, "factor", "fwd_ret_1d")

    assert out.height == 0
    assert caplog.text == ""


def test_partial_drop_does_not_warn(caplog):
    """只有**整天丢光**才告警。部分丢弃（早期上市股少）是正常的，警告会变噪音。"""
    thick = _frame(40, n_days=4)
    thin = _frame(20, n_days=1).with_columns(
        (pl.col("trade_date") + pl.duration(days=10)).alias("trade_date")
    )
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = ia._rank_ic_by_date(pl.concat([thick, thin]), "factor", "fwd_ret_1d")

    assert out.height == 4, "薄的那天被丢，厚的四天保留"
    assert caplog.text == ""


def test_pearson_path_warns_too(caplog):
    """双路径：`_pearson_ic_by_date` 有它自己的过滤，同样会静默丢光。"""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = ia._pearson_ic_by_date(_frame(20), "factor", "fwd_ret_1d")

    assert out.height == 0
    assert "30" in caplog.text

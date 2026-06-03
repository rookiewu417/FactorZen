"""Tests for core.timing.StageTimer (per-stage timing observability)."""

from __future__ import annotations

import logging

import pytest

from factorzen.core.timing import StageTimer


def test_stage_timer_accumulates_named_durations():
    timer = StageTimer()
    with timer.stage("ic"):
        pass
    with timer.stage("backtest"):
        pass
    assert set(timer.timings) == {"ic", "backtest"}
    assert all(isinstance(v, float) and v >= 0 for v in timer.timings.values())


def test_stage_timer_logs_stage_name_and_duration(caplog):
    timer = StageTimer()
    with caplog.at_level(logging.INFO), timer.stage("报告生成"):
        pass
    assert any("报告生成" in r.getMessage() and "耗时" in r.getMessage() for r in caplog.records)


def test_stage_timer_records_duration_even_on_exception():
    timer = StageTimer()
    with pytest.raises(ValueError), timer.stage("boom"):
        raise ValueError("x")
    assert "boom" in timer.timings

"""单元测试:reports/_summaries 的纯助手函数边界行为(覆盖防御分支)。"""

from __future__ import annotations

import types

import polars as pl

from factorzen.reports._summaries import (
    _build_benchmark_summary,
    _build_factor_corr_summary,
    _build_holding_period_summary,
    _build_monthly_return_summary,
    _build_neutralized_summary,
    _build_predictive_summary,
    _build_quality_summary,
    _build_regime_summary,
    _display_regime_label,
    _display_status,
)


def _ns(**kw):
    return types.SimpleNamespace(**kw)


# ── 预测能力摘要 ──────────────────────────────────────────────


def test_predictive_summary_missing_core_ic_says_insufficient():
    out = _build_predictive_summary({"ic_mean": None, "ir": None, "ic_positive_ratio": None})
    assert "样本不足" in out["headline"]
    assert out["next_steps"]


def test_predictive_summary_positive_strong_signal():
    out = _build_predictive_summary(
        {
            "ic_mean": 0.05,
            "ir": 0.8,
            "ic_positive_ratio": 0.65,
            "ic_tstat": 3.0,
            "ic_pvalue": 0.001,
            "pearson_ic_mean": 0.04,
        }
    )
    assert "方向为正" in out["headline"]
    assert out["details"]


def test_predictive_summary_negative_direction():
    out = _build_predictive_summary(
        {
            "ic_mean": -0.05,
            "ir": 0.6,
            "ic_positive_ratio": 0.4,
            "ic_tstat": -3.0,
            "ic_pvalue": 0.001,
        }
    )
    assert "方向为负" in out["headline"]


# ── 中性化摘要 ──────────────────────────────────────────────


def test_neutralized_summary_none_when_absent():
    assert _build_neutralized_summary({}) is None


def test_neutralized_summary_high_retention():
    out = _build_neutralized_summary({"ic_mean": 0.05, "neutralized_ic_mean": 0.045})
    assert out is not None and out["retention"] is not None and out["retention"] >= 0.7


def test_neutralized_summary_direction_reversal():
    out = _build_neutralized_summary({"ic_mean": 0.05, "neutralized_ic_mean": -0.03})
    assert out is not None and "反转" in out["headline"]


# ── 基准摘要 ──────────────────────────────────────────────


def test_benchmark_summary_none_when_no_result():
    assert _build_benchmark_summary(None) is None


def test_benchmark_summary_outperform():
    out = _build_benchmark_summary(
        _ns(ann_excess_ret=0.08, information_ratio=0.7, tracking_error=0.10, excess_max_dd=-0.05)
    )
    assert out is not None
    assert out["direction"] == "跑赢基准"
    assert set(out) >= {"direction", "efficiency", "risk", "drawdown"}


# ── 市场状态 / 持有期 / 因子相关性 ──────────────────────────


def test_regime_summary_none_when_empty():
    assert _build_regime_summary({"regime_table": []}) is None


def test_regime_summary_up_down_split():
    out = _build_regime_summary(
        {
            "regime_table": [
                {"regime": "up", "label": "上涨", "ic": 0.03},
                {"regime": "down", "label": "下跌", "ic": -0.03},
            ]
        }
    )
    assert out is not None and "上涨" in out["headline"]


def test_holding_period_summary_none_when_empty():
    assert _build_holding_period_summary({"multi_period_table": []}) is None


def test_factor_corr_summary_none_for_single_factor():
    df = pl.DataFrame({"factor": ["a"], "a": [1.0]})
    assert _build_factor_corr_summary(df) is None


def test_factor_corr_summary_reports_high_overlap():
    df = pl.DataFrame({"factor": ["a", "b"], "a": [1.0, 0.85], "b": [0.85, 1.0]})
    out = _build_factor_corr_summary(df)
    assert out is not None and "a / b" in out["headline"]


# ── 质量 / 显示助手 ──────────────────────────────────────────


def test_quality_summary_not_provided_when_none():
    out = _build_quality_summary(None)
    assert out["status"] == "未传入"
    assert out["checks"] == []


def test_quality_summary_maps_status_and_checks():
    out = _build_quality_summary(
        {
            "status": "warning",
            "checks": {"factor_clean": {"rows": 100, "coverage": 0.95, "valid_count": 95}},
            "warnings": ["factor_clean coverage is low"],
            "errors": [],
        }
    )
    assert out["status"] == "需关注"
    assert out["checks"] and out["checks"][0]["name"] == "清洗后因子值"


def test_display_status_and_regime_label():
    assert _display_status("ok") == "正常"
    assert _display_status("failed") == "失败"
    assert _display_status("disabled") == "已关闭"
    assert _display_regime_label("up") == "上涨"
    assert _display_regime_label("unknown_key") == "unknown_key"


def test_monthly_return_summary_none_without_returns():
    assert _build_monthly_return_summary(_ns(returns=None)) is None

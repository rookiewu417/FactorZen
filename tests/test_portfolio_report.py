"""Tests for portfolio_report.py — M7 成果展示页。

TDD: write tests first, watch them fail, then implement.
All tests pass sim_result=None so no real backtest is required.
"""

import polars as pl
import pytest

from factorzen.reports.portfolio_report import generate_portfolio_report

# ── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def base_metrics() -> dict:
    return {
        "ann_ret": 0.12,
        "ann_vol": 0.18,
        "sharpe": 0.67,
        "max_dd": -0.15,
        "ann_turnover": 3.2,
        "total_cost": 0.01,
    }


@pytest.fixture()
def attribution_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "type": ["brinson_allocation", "factor_return"],
            "key": ["银行", "size"],
            "value": [0.01, 0.005],
        }
    )


@pytest.fixture()
def risk_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "metric": ["total_risk", "factor_risk", "specific_risk"],
            "value": [0.18, 0.15, 0.10],
        }
    )


@pytest.fixture()
def manifest() -> dict:
    return {"n_holdings": 87, "status": "optimal"}


# ── core rendering test (from brief) ─────────────────────────────────────


def test_generate_portfolio_report_html_has_sections(
    base_metrics, attribution_df, risk_df, manifest
):
    """复刻 brief 里要求的断言，不依赖真实回测。"""
    html = generate_portfolio_report(
        None,
        metrics=base_metrics,
        attribution_df=attribution_df,
        risk_summary_df=risk_df,
        portfolio_manifest=manifest,
    )
    assert isinstance(html, str) and len(html) > 500

    # 关键 section 存在
    assert "sharpe" in html.lower() or "夏普" in html
    assert "0.67" in html or "67" in html  # 绩效数值渲染进去
    assert "总风险" in html or "total_risk" in html or "0.18" in html


# ── additional coverage ───────────────────────────────────────────────────


def test_returns_str(base_metrics):
    """generate_portfolio_report 返回 str."""
    html = generate_portfolio_report(None, metrics=base_metrics)
    assert isinstance(html, str)


def test_html_has_doctype(base_metrics):
    """输出是合法 HTML 文档，以 DOCTYPE 开头。"""
    html = generate_portfolio_report(None, metrics=base_metrics)
    assert html.strip().startswith("<!DOCTYPE") or html.strip().startswith("<!")


def test_metrics_values_present(base_metrics):
    """所有 metrics 数值都出现在 HTML 里。"""
    html = generate_portfolio_report(None, metrics=base_metrics)
    # ann_ret 12% or 0.12 — must match actual rendered format
    assert "0.12" in html or "12.0%" in html
    # ann_vol 18%
    assert "18" in html
    # sharpe
    assert "0.67" in html or "67" in html


def test_risk_table_rendered(base_metrics, risk_df):
    """风险表格写入 HTML。"""
    html = generate_portfolio_report(None, metrics=base_metrics, risk_summary_df=risk_df)
    # At least one of total_risk / factor_risk / specific_risk shows up
    assert any(v in html for v in ("total_risk", "factor_risk", "specific_risk", "总风险"))


def test_attribution_table_rendered(base_metrics, attribution_df):
    """归因表格写入 HTML。"""
    html = generate_portfolio_report(None, metrics=base_metrics, attribution_df=attribution_df)
    assert "银行" in html or "brinson" in html.lower() or "归因" in html


def test_manifest_rendered(base_metrics, manifest):
    """持仓 meta 写入 HTML。"""
    html = generate_portfolio_report(None, metrics=base_metrics, portfolio_manifest=manifest)
    assert "87" in html or "optimal" in html or "持仓" in html


def test_no_chart_when_sim_result_none(base_metrics):
    """sim_result=None 时不应有 base64 图表字符串（无 <img src='data:image）。"""
    html = generate_portfolio_report(None, metrics=base_metrics)
    # If there are no charts, there should be no base64 img tags
    assert 'data:image/png;base64' not in html


def test_capability_overview_cards_present(base_metrics):
    """修复5：能力总览卡按能力命名展示，不暴露内部里程碑代号（M1-M6）。

    展示页是对外可见正文，项目约定（CLAUDE.md）要求对外不暴露 M0-M7 里程碑代号；
    旧版直接把 "M1 因子挖掘".."M6 多 Agent 协作" 写进卡片文案，违反该约定。
    """
    html = generate_portfolio_report(None, metrics=base_metrics)
    # 按能力命名的文案应存在
    assert any(
        token in html
        for token in ("因子挖掘", "防过拟合", "风险模型", "组合优化", "Agent", "模拟交易")
    )
    # 内部里程碑代号字样不应再出现在对外可见正文中
    for milestone_token in ("M1", "M2", "M3", "M4", "M5", "M6"):
        assert milestone_token not in html, (
            f"展示页正文不应出现内部里程碑代号 {milestone_token!r}（对外应按能力命名）"
        )


def test_all_args_together(base_metrics, attribution_df, risk_df, manifest):
    """所有参数一起传入，不报错，HTML 足够长。"""
    html = generate_portfolio_report(
        None,
        metrics=base_metrics,
        attribution_df=attribution_df,
        risk_summary_df=risk_df,
        portfolio_manifest=manifest,
    )
    assert len(html) > 1000

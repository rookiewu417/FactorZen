"""Tear Sheet report engine tests."""

from dataclasses import replace
from pathlib import Path

import jinja2
import numpy as np
import polars as pl
import pytest

from factorzen.daily.evaluation.advanced import (
    EventStudyResult,
    ICDecayResult,
    MarketRegimeICResult,
    MonotonicityResult,
    RankAutocorrResult,
    SectorICResult,
    SizeICResult,
)
from factorzen.daily.evaluation.backtest import BacktestResult
from factorzen.daily.evaluation.ic_analysis import ICAnalysisResult
from factorzen.daily.evaluation.turnover import TurnoverResult
from factorzen.daily.evaluation.walk_forward import WalkForwardFoldResult, WalkForwardResult
from factorzen.reports._charts import _prepare_brinson_plot_frame
from factorzen.reports.tear_sheet import (
    _compute_factor_rating,
    _format_metric_number,
    _format_metric_percent,
    generate_tear_sheet,
)

# Fixtures


def _make_dates(n: int = 60) -> list:
    return [f"2025-{(i // 20 + 1):02d}-{(i % 20 + 1):02d}" for i in range(n)]


@pytest.fixture
def ic_result() -> ICAnalysisResult:
    dates = _make_dates()
    n = len(dates)
    rng = np.random.default_rng(42)
    ics = rng.normal(0.03, 0.08, n).tolist()
    return ICAnalysisResult(
        factor_name="test_factor",
        ic_mean=float(np.mean(ics)),
        ic_std=float(np.std(ics, ddof=1)),
        ir=float(np.mean(ics)) / float(np.std(ics, ddof=1)),
        ic_positive_ratio=float(np.mean(np.array(ics) > 0)),
        n_periods=n,
        ic_series=pl.DataFrame({"trade_date": dates, "ic": ics}),
        decay={1: 0.032, 5: 0.025, 10: 0.018, 20: 0.010},
        frequency="daily",
    )


@pytest.fixture
def bt_result() -> BacktestResult:
    dates = _make_dates()
    n_groups = 10
    rng = np.random.default_rng(43)

    records, nav_records, ls_ret = [], [], []
    for d in dates:
        day_rets = {}
        for g in range(n_groups):
            ret = rng.normal(0.0002 * (g - 4.5), 0.015)
            records.append({"trade_date": d, "group": g, "ret": ret})
            day_rets[g] = ret

    # NAV by group
    for g in range(n_groups):
        g_rets = [r["ret"] for r in records if r["group"] == g]
        cum = np.cumprod(1 + np.array(g_rets))
        for i, d in enumerate(dates):
            nav_records.append({"trade_date": d, "group": g, "nav": float(cum[i])})

    # Long-short
    for _i, d in enumerate(dates):
        day_rets_i = {r["group"]: r["ret"] for r in records if r["trade_date"] == d}
        long_ret = day_rets_i.get(n_groups - 1, 0)
        short_ret = day_rets_i.get(0, 0)
        ls_r = long_ret - short_ret
        ls_ret.append(ls_r)
    ls_cum = np.cumprod(1 + np.array(ls_ret))
    long_short_nav = pl.DataFrame(
        {
            "trade_date": dates,
            "ret": ls_ret,
            "nav": ls_cum,
        }
    )

    # Summary stats
    summary_stats = {}
    for g in range(n_groups):
        grets = np.array([r["ret"] for r in records if r["group"] == g])
        cum = np.cumprod(1 + grets)
        summary_stats[g] = {
            "ann_ret": float(np.mean(grets) * 252),
            "ann_vol": float(np.std(grets) * np.sqrt(252)),
            "sharpe": float(np.mean(grets) * 252 / (np.std(grets) * np.sqrt(252) + 1e-9)),
            "max_dd": float(np.min(cum / np.maximum.accumulate(cum) - 1)),
        }
    ls_arr = np.array(ls_ret)
    ls_cum_arr = np.cumprod(1 + ls_arr)
    summary_stats["long_short"] = {
        "ann_ret": float(np.mean(ls_arr) * 252),
        "ann_vol": float(np.std(ls_arr) * np.sqrt(252)),
        "sharpe": float(np.mean(ls_arr) * 252 / (np.std(ls_arr) * np.sqrt(252) + 1e-9)),
        "max_dd": float(np.min(ls_cum_arr / np.maximum.accumulate(ls_cum_arr) - 1)),
    }

    returns = long_short_nav.rename({"ret": "net_return"}).with_columns(
        [
            pl.col("net_return").alias("gross_return"),
            pl.lit(0.0).alias("cost"),
            pl.lit(0.0).alias("borrow_cost"),
            pl.lit(0.0).alias("cash_weight"),
            pl.lit(0.0).alias("turnover"),
        ]
    )
    positions = pl.DataFrame(
        {
            "trade_date": dates,
            "ts_code": ["000001.SZ"] * len(dates),
            "weight": [1.0] * len(dates),
            "market_value": [1.0] * len(dates),
        }
    )
    trades = pl.DataFrame(
        {
            "trade_date": dates[:1],
            "ts_code": ["000001.SZ"],
            "prev_weight": [0.0],
            "target_weight": [1.0],
            "filled_delta_weight": [1.0],
            "turnover": [1.0],
            "cost": [0.0],
            "block_reason": [""],
        }
    )

    return BacktestResult(
        factor_name="test_factor",
        strategy_name="quantile_long_short",
        n_groups=n_groups,
        returns=returns,
        nav=pl.DataFrame(nav_records),
        positions=positions,
        trades=trades,
        summary_stats=summary_stats,
        config={},
        frequency="daily",
    )


@pytest.fixture
def to_result() -> TurnoverResult:
    dates = _make_dates()
    n = len(dates)
    rng = np.random.default_rng(44)
    turnover_vals = rng.uniform(0.15, 0.35, n).tolist()
    return TurnoverResult(
        factor_name="test_factor",
        avg_turnover=float(np.mean(turnover_vals)),
        migration_matrix=pl.DataFrame(),
        daily_turnover=pl.DataFrame({"trade_date": dates, "turnover": turnover_vals}),
        frequency="daily",
    )


@pytest.fixture
def advanced_results() -> dict:
    return {
        "decay_results": [
            ICDecayResult(horizon=1, ic_mean=0.032, ic_std=0.08),
            ICDecayResult(horizon=5, ic_mean=0.025, ic_std=0.07),
            ICDecayResult(horizon=20, ic_mean=0.010, ic_std=0.06),
        ],
        "mono": MonotonicityResult(
            factor_name="test_factor",
            monotonicity_score=0.85,
            group_means=[-0.002, -0.001, 0.001, 0.003],
            direction="positive",
            ols_slope=0.0012,
        ),
        "autocorr": RankAutocorrResult(
            factor_name="test_factor",
            autocorr_values=[0.65],
            mean_autocorr=0.65,
            half_life_est=1.6,
            _lag_to_autocorr={1: 0.65},
        ),
        "sector": SectorICResult(
            factor_name="test_factor",
            sector_ic_df=pl.DataFrame(
                {
                    "sector": ["fin", "tech", "cons"],
                    "ic": [0.028, 0.035, 0.022],
                }
            ),
        ),
        "size": SizeICResult(
            factor_name="test_factor",
            buckets={"Large": 0.030, "Mid": 0.033, "Small": 0.025},
        ),
    }


# Tests: generate_tear_sheet


class TestGenerateTearSheet:
    def test_metric_formatters_suppress_negative_zero(self):
        assert _format_metric_percent(-0.0000001, 2) == "0.00%"
        assert _format_metric_number(-0.0000001, 2) == "0.00"

    def test_report_tables_suppress_negative_zero_percent(self, ic_result, bt_result, to_result):
        tiny_drawdown = replace(
            bt_result,
            summary_stats={
                "portfolio": {
                    "ann_ret": 0.03,
                    "ann_vol": 0.01,
                    "sharpe": 3.0,
                    "max_dd": -0.0000001,
                    "avg_turnover": 0.10,
                    "total_cost": 0.001,
                }
            },
            config={"strategy_type": "optimizer_strategy", "cost_model": "linear"},
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            tiny_drawdown,
            to_result,
            strategy_results={"optimizer_mv_long_only": tiny_drawdown},
            primary_strategy="optimizer_mv_long_only",
        )

        assert "-0.00%" not in html
        assert "0.00%" in html

    def test_basic_generation(self, ic_result, bt_result, to_result):
        """Smoke test: generate HTML without errors."""
        html = generate_tear_sheet(
            "momentum_20d",
            ic_result,
            bt_result,
            to_result,
            frequency="daily",
            date_range="2025-01-01 ~ 2025-05-13",
        )
        assert isinstance(html, str)
        assert len(html) > 1000

    def test_html_contains_key_elements(self, ic_result, bt_result, to_result):
        """HTML contains expected structural elements."""
        html = generate_tear_sheet("momentum_20d", ic_result, bt_result, to_result)
        assert "<!DOCTYPE html>" in html
        assert "momentum_20d" in html
        assert "综合结论" in html
        assert "收益表现" in html
        assert "预测能力" in html
        assert "结构检验" in html
        assert "交易可行性" in html
        assert "风险归因" in html
        assert "</html>" in html

    def test_html_contains_llm_explanation_when_provided(self, ic_result, bt_result, to_result):
        """LLM explanation is rendered when provided."""
        html = generate_tear_sheet(
            "momentum_20d",
            ic_result,
            bt_result,
            to_result,
            llm_explanation={
                "rating": "moderate",
                "confidence": "medium",
                "factor_intuition": "动量因子刻画近期强势的延续性。",
                "evidence_assessment": "IC 为正但样本外强度一般。",
                "risk_flags": ["换手率偏高，需要关注交易成本。"],
                "usage_suggestion": "适合继续研究，不应单独作为交易信号。",
                "next_steps": ["检查行业中性后表现"],
            },
        )

        assert "大模型研究解读" in html
        assert "moderate" in html
        assert "中等（moderate）" in html
        assert "中（medium）" in html
        assert "动量因子刻画近期强势的延续性。" in html

    def test_summary_uses_llm_explanation_when_provided(self, ic_result, bt_result, to_result):
        html = generate_tear_sheet(
            "momentum_20d",
            ic_result,
            bt_result,
            to_result,
            llm_explanation={
                "rating": "weak",
                "confidence": "low",
                "factor_intuition": "signal intuition",
                "evidence_assessment": "LLM evidence conclusion",
                "risk_flags": ["risk one"],
                "usage_suggestion": "LLM usage suggestion",
                "next_steps": ["next one"],
            },
        )

        summary = html.split("<h2>综合评估</h2>", 1)[1]
        assert "LLM evidence conclusion" in summary
        assert "LLM usage suggestion" in summary

    def test_summary_and_llm_explanation_appear_before_analysis(
        self, ic_result, bt_result, to_result
    ):
        html = generate_tear_sheet(
            "momentum_20d",
            ic_result,
            bt_result,
            to_result,
            llm_explanation={
                "rating": "weak",
                "confidence": "low",
                "factor_intuition": "signal intuition",
                "evidence_assessment": "LLM evidence conclusion",
                "risk_flags": [],
                "usage_suggestion": "LLM usage suggestion",
                "next_steps": [],
            },
        )

        summary_pos = html.index("<h2>综合评估</h2>")
        llm_pos = html.index("<h2>大模型研究解读</h2>")
        returns_pos = html.index("<h2>收益表现</h2>")

        assert summary_pos < returns_pos
        assert llm_pos < returns_pos

    def test_brinson_plot_frame_limits_sector_count_and_aggregates_other(self):
        sector_df = pl.DataFrame(
            {
                "sector": [f"Sector{i:02d}" for i in range(20)],
                "allocation": [float(i) / 100 for i in range(20)],
                "selection": [0.0] * 20,
                "interaction": [0.0] * 20,
                "total_contribution": [float(i) / 100 for i in range(20)],
            }
        )

        prepared = _prepare_brinson_plot_frame(sector_df, max_sectors=8)

        assert prepared.height == 9
        assert "其他" in prepared["sector"].to_list()

    def test_generate_html_contains_factor_name(self, ic_result, bt_result, to_result):
        """Generated HTML contains factor name."""
        html = generate_tear_sheet("momentum_20d", ic_result, bt_result, to_result)
        assert "<html" in html
        assert "momentum_20d" in html

    def test_html_size_under_5mb(self, ic_result, bt_result, to_result):
        """Generated HTML is under 5 MB."""
        html = generate_tear_sheet("momentum_20d", ic_result, bt_result, to_result)
        size_bytes = len(html.encode("utf-8"))
        assert size_bytes < 5 * 1024 * 1024, f"HTML size {size_bytes} exceeds 5MB"

    def test_html_contains_chart_base64(self, ic_result, bt_result, to_result):
        """Charts are embedded as base64 in HTML."""
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            date_range="2025-01-01 ~ 2025-05-13",
        )
        assert "data:image/png;base64," in html

    def test_html_contains_table_of_contents(self, ic_result, bt_result, to_result):
        html = generate_tear_sheet("test_factor", ic_result, bt_result, to_result)

        assert 'class="toc"' in html
        assert 'href="#overview"' in html
        assert 'href="#returns"' in html
        assert 'href="#predictive-power"' in html
        assert 'href="#structure-checks"' in html
        assert 'href="#tradability"' in html
        assert 'href="#robustness"' in html
        assert 'href="#risk-attribution"' in html
        assert 'href="#appendix"' in html

    def test_html_contains_multi_strategy_tabs_and_summary(
        self, ic_result, bt_result, to_result
    ):
        topn_result = replace(
            bt_result,
            strategy_name="topn_50",
            config={"strategy_type": "topn_long_only"},
        )
        alt_result = replace(
            bt_result,
            strategy_name="quantile_ls_5",
            config={"strategy_type": "quantile_long_short", "strategy_params": {"quantiles": 5}},
            summary_stats={
                "portfolio": {
                    "ann_ret": 0.12,
                    "ann_vol": 0.18,
                    "sharpe": 0.67,
                    "max_dd": -0.08,
                }
            },
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            topn_result,
            to_result,
            strategy_results={
                "topn_50": topn_result,
                "quantile_ls_5": alt_result,
            },
            primary_strategy="topn_50",
        )

        assert "跨策略对比" in html
        assert 'class="strategy-tabs"' in html
        assert 'data-target="strategy-page-topn-50"' in html
        assert 'data-target="strategy-page-quantile-ls-5"' in html
        assert 'id="strategy-page-topn-50"' in html
        assert 'id="strategy-page-quantile-ls-5"' in html
        assert "quantile_ls_5" in html
        assert "TopN 多头 50（主策略）" in html
        assert "五分位多空" in html
        assert "代码：topn_50" in html
        assert "本报告共运行 2 个策略" in html
        assert "其中 1 个为多空策略" in html
        assert "分页按钮会标明策略方向" in html
        assert 'class="tab-kind"' in html
        assert "代码：quantile_ls_5 | 方向：分位数组合多空" in html
        assert "关键参数" in html
        assert "分位组数=5" in html

    def test_long_only_strategy_does_not_show_long_short_label(self, ic_result, bt_result, to_result):
        long_only = replace(
            bt_result,
            strategy_name="topn_long_only",
            summary_stats={
                "portfolio": {
                    "ann_ret": 0.08,
                    "ann_vol": 0.16,
                    "sharpe": 0.50,
                    "max_dd": -0.08,
                    "avg_turnover": 0.25,
                    "total_cost": 0.01,
                    "ann_turnover": 63.0,
                }
            },
            config={
                "strategy_type": "topn_long_only",
                "cost_model": "linear",
                "max_abs_weight": 0.1,
            },
        )
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            long_only,
            to_result,
            strategy_results={"topn_50": long_only},
            primary_strategy="topn_50",
        )

        assert "组合收益" in html
        assert "多空组合" not in html
        assert "<td>L/S</td>" not in html
        assert "多头 TopN" in html
        assert "主策略组合年化收益 8.0%" in html
        assert "多空年化收益 8.0%" not in html

    def test_long_short_strategy_shows_long_short_label(self, ic_result, bt_result, to_result):
        long_short = replace(
            bt_result,
            strategy_name="quantile_long_short",
            config={
                "strategy_type": "quantile_long_short",
                "cost_model": "linear",
                "max_abs_weight": 0.1,
            },
        )
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            long_short,
            to_result,
            strategy_results={"quantile_ls_5": long_short},
            primary_strategy="quantile_ls_5",
        )

        assert "多空组合" in html
        assert "分位数组合多空" in html

    def test_cross_strategy_table_explains_direction_cost_and_turnover(self, ic_result, bt_result, to_result):
        topn = replace(
            bt_result,
            strategy_name="topn_long_only",
            summary_stats={
                "portfolio": {
                    "ann_ret": 0.08,
                    "ann_vol": 0.16,
                    "sharpe": 0.50,
                    "max_dd": -0.08,
                    "avg_turnover": 0.25,
                    "total_cost": 0.01,
                    "ann_turnover": 63.0,
                }
            },
            config={"strategy_type": "topn_long_only", "cost_model": "linear"},
        )
        quantile = replace(
            bt_result,
            strategy_name="quantile_long_short",
            config={"strategy_type": "quantile_long_short", "cost_model": "linear"},
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            topn,
            to_result,
            strategy_results={"topn_50": topn, "quantile_ls_5": quantile},
            primary_strategy="topn_50",
        )

        for heading in ("策略方向", "收益口径", "平均换手", "交易成本", "成本模型"):
            assert heading in html
        assert "收益质量摘要" in html
        assert "收益质量摘要：</strong>收益质量结论：" not in html
        assert "收益质量摘要：</strong>收益为正" in html
        assert "Sharp e" not in html
        assert "Sharpe 较弱" in html
        assert "回撤可控" in html
        assert "换手适中" in html
        assert "成本可控" in html
        assert "多头 TopN" in html
        assert "分位数组合多空" in html
        assert "线性成本" in html

    def test_strategy_page_explains_missing_nav_curve_when_metrics_exist(
        self, ic_result, bt_result, to_result, monkeypatch
    ):
        import factorzen.reports.tear_sheet as tear_sheet

        result = replace(
            bt_result,
            summary_stats={
                "portfolio": {
                    "ann_ret": 0.08,
                    "ann_vol": 0.16,
                    "sharpe": 0.50,
                    "max_dd": -0.08,
                }
            },
        )
        monkeypatch.setattr(tear_sheet, "_make_returns_chart", lambda *_: None)

        html = generate_tear_sheet("test_factor", ic_result, result, to_result)
        returns_html = html.split('<div class="panel" id="returns">', 1)[1].split(
            '<div class="panel" id="predictive-power">', 1
        )[0]

        assert "收益质量摘要" in returns_html
        assert "净值曲线未生成" in returns_html
        assert "已生成回测指标，但缺少可绘图的日度收益或净值序列" in returns_html
        assert "无回测数据：该策略未产生净值曲线" not in returns_html

    def test_strategy_page_summarizes_monthly_return_concentration(
        self, ic_result, bt_result, to_result
    ):
        returns = pl.DataFrame(
            {
                "trade_date": [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-02-01",
                    "2025-02-02",
                    "2025-03-01",
                    "2025-03-02",
                ],
                "net_return": [0.01, 0.02, -0.01, -0.02, 0.03, 0.04],
            }
        )
        result = replace(bt_result, returns=returns)

        html = generate_tear_sheet("test_factor", ic_result, result, to_result)

        assert "月度收益摘要" in html
        assert "正收益月份 2/3" in html
        assert "最佳月份 2025-03（7.12%）" in html
        assert "最弱月份 2025-02（-2.98%）" in html
        assert "收益有一定月份集中度" in html

    def test_monthly_return_summary_does_not_infer_concentration_from_one_month(
        self, ic_result, bt_result, to_result
    ):
        returns = pl.DataFrame(
            {
                "trade_date": ["2025-01-01", "2025-01-02"],
                "net_return": [0.01, 0.02],
            }
        )
        result = replace(bt_result, returns=returns)

        html = generate_tear_sheet("test_factor", ic_result, result, to_result)

        assert "月度收益摘要" in html
        assert "正收益月份 1/1" in html
        assert "观察月份 2025-01（3.02%）" in html
        assert "最佳月份 2025-01" not in html
        assert "最弱月份 2025-01" not in html
        assert "月度样本不足，仅作区间摘要" in html
        assert "收益高度集中在少数月份" not in html
        assert "收益有一定月份集中度" not in html

    def test_monthly_return_summary_explains_missing_heatmap(
        self, ic_result, bt_result, to_result, monkeypatch
    ):
        import factorzen.reports.tear_sheet as tear_sheet

        returns = pl.DataFrame(
            {
                "trade_date": [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-02-01",
                    "2025-02-02",
                    "2025-03-01",
                    "2025-03-02",
                ],
                "net_return": [0.01, 0.02, -0.01, -0.02, 0.03, 0.04],
            }
        )
        result = replace(bt_result, returns=returns)
        monkeypatch.setattr(tear_sheet, "_make_monthly_return_heatmap", lambda *_: None)

        html = generate_tear_sheet("test_factor", ic_result, result, to_result)
        returns_html = html.split('<div class="panel" id="returns">', 1)[1].split(
            '<div class="panel" id="predictive-power">', 1
        )[0]

        assert "月度收益分析" in returns_html
        assert "月度收益摘要" in returns_html
        assert "月度收益热力图未生成" in returns_html
        assert 'alt="月度收益热力图"' not in returns_html

    def test_cross_strategy_section_contains_selection_summary(
        self, ic_result, bt_result, to_result
    ):
        topn = replace(
            bt_result,
            strategy_name="topn_long_only",
            summary_stats={
                "portfolio": {
                    "ann_ret": 0.08,
                    "ann_vol": 0.16,
                    "sharpe": 0.50,
                    "max_dd": -0.08,
                    "avg_turnover": 0.25,
                    "total_cost": 0.01,
                    "ann_turnover": 63.0,
                }
            },
            config={"strategy_type": "topn_long_only", "cost_model": "linear"},
        )
        quantile = replace(
            bt_result,
            strategy_name="quantile_long_short",
            summary_stats={
                "portfolio": {
                    "ann_ret": 0.12,
                    "ann_vol": 0.20,
                    "sharpe": 0.70,
                    "max_dd": -0.12,
                    "avg_turnover": 0.90,
                    "total_cost": 0.03,
                    "ann_turnover": 226.8,
                }
            },
            config={"strategy_type": "quantile_long_short", "cost_model": "linear"},
        )
        defensive = replace(
            bt_result,
            strategy_name="optimizer_mv_long_only",
            summary_stats={
                "portfolio": {
                    "ann_ret": 0.06,
                    "ann_vol": 0.10,
                    "sharpe": 0.60,
                    "max_dd": -0.03,
                    "avg_turnover": 0.20,
                    "total_cost": 0.005,
                    "ann_turnover": 50.4,
                }
            },
            config={"strategy_type": "optimizer_strategy", "cost_model": "linear"},
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            topn,
            to_result,
            strategy_results={
                "topn_50": topn,
                "quantile_ls_5": quantile,
                "optimizer_mv_long_only": defensive,
            },
            primary_strategy="topn_50",
        )

        assert "跨策略选择摘要" in html
        assert "年化收益最高：五分位多空（12.00%）" in html
        assert "Sharpe 最高：五分位多空（0.70）" in html
        assert "回撤控制最好：均值-方差优化多头（最大回撤 -3.00%）" in html
        assert "年化收益最高的是" not in html
        assert "其中 1 个策略平均换手较高，1 个策略交易成本较高" in html

    def test_cross_strategy_summary_ignores_missing_metrics(
        self, ic_result, bt_result, to_result
    ):
        topn = replace(
            bt_result,
            strategy_name="topn_long_only",
            summary_stats={
                "portfolio": {
                    "ann_ret": -0.02,
                    "ann_vol": 0.16,
                    "sharpe": -0.20,
                    "max_dd": -0.08,
                    "avg_turnover": 0.25,
                    "total_cost": 0.01,
                }
            },
            config={"strategy_type": "topn_long_only", "cost_model": "linear"},
        )
        missing = replace(
            bt_result,
            strategy_name="factor_weighted_ls",
            summary_stats={"portfolio": {}},
            config={"strategy_type": "factor_weighted_long_short", "cost_model": "linear"},
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            topn,
            to_result,
            strategy_results={"topn_50": topn, "factor_weighted_ls": missing},
            primary_strategy="topn_50",
        )

        assert "年化收益最高：TopN 多头 50（-2.00%）" in html
        assert "Sharpe 最高：TopN 多头 50（-0.20）" in html
        assert "回撤控制最好：TopN 多头 50（最大回撤 -8.00%）" in html
        missing_row = html.split("<td>因子加权多空", 1)[1].split("</tr>", 1)[0]
        assert "样本不足" in missing_row
        assert "0.00%" not in missing_row
        assert "收益指标样本不足，暂不参与策略优选" in html
        missing_page = html.split('id="strategy-page-factor-weighted-ls"', 1)[1].split(
            "</section>", 1
        )[0]
        assert "0.000" not in missing_page
        assert "0.00%" not in missing_page

    def test_strategy_parameters_are_displayed_with_reader_labels(
        self, ic_result, bt_result, to_result
    ):
        optimizer = replace(
            bt_result,
            strategy_name="optimizer_mv_long_only",
            summary_stats={
                "portfolio": {
                    "ann_ret": 0.08,
                    "ann_vol": 0.16,
                    "sharpe": 0.50,
                    "max_dd": -0.08,
                    "avg_turnover": 0.25,
                    "total_cost": 0.01,
                    "ann_turnover": 63.0,
                },
                "long_short": {
                    "ann_ret": 0.08,
                    "ann_vol": 0.16,
                    "sharpe": 0.50,
                    "max_dd": -0.08,
                    "avg_turnover": 0.25,
                    "total_cost": 0.01,
                    "ann_turnover": 63.0,
                },
            },
            config={
                "strategy_type": "optimizer_strategy",
                "strategy_params": {
                    "optimizer": "mean_variance",
                    "risk_aversion": 1.0,
                    "lookback_days": 60,
                    "cov_estimator": "ledoit_wolf",
                    "top_n": 100,
                    "long_only": True,
                    "gross_exposure": 1.0,
                    "net_exposure": 1.0,
                    "max_weight": 0.08,
                },
                "cost_model": "linear",
            },
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            optimizer,
            to_result,
            strategy_results={"optimizer_mv_long_only": optimizer},
            primary_strategy="optimizer_mv_long_only",
        )

        for marker in (
            "均值-方差优化多头",
            "优化器组合",
            "优化器=均值-方差",
            "风险厌恶=1.0",
            "协方差回看天数=60",
            "协方差估计=Ledoit-Wolf 收缩",
            "TopN 数量=100",
            "仅多头=是",
            "总敞口=1.0",
            "净敞口=1.0",
            "单票权重上限=0.08",
            "线性成本",
        ):
            assert marker in html

        for raw_key in (
            "risk_aversion=",
            "lookback_days=",
            "cov_estimator=",
            "top_n=",
            "long_only=True",
            "gross_exposure=",
            "net_exposure=",
            "max_weight=",
            "optimizer_strategy",
            "mean_variance",
            "ledoit_wolf",
        ):
            assert raw_key not in html

    def test_overview_dashboard_contains_quality_and_primary_strategy_context(self, ic_result, bt_result, to_result):
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            date_range="20250101 ~ 20250331",
            strategy_results={"quantile_ls_5": bt_result},
            primary_strategy="quantile_ls_5",
        )

        assert "研究仪表盘" in html
        assert "主策略" in html
        assert "分位数组合多空" in html
        assert "代码：quantile_ls_5" in html
        assert "方向：分位数组合多空" in html
        assert "策略覆盖" in html
        assert "有效区间" in html
        assert "样本期数" in html
        assert "数据质量" in html
        assert "阅读口径" in html
        assert "研究边界" in html
        assert "未生成模块会在附录列出状态和原因" in html

    def test_report_contains_professional_layout_css_hooks(self, ic_result, bt_result, to_result):
        html = generate_tear_sheet("test_factor", ic_result, bt_result, to_result)

        for css_hook in (
            ".dashboard-grid",
            ".dashboard-card",
            ".badge",
            ".status-callout",
            ".strategy-header",
            ".table-wrap",
            ".table-wrap table",
            ".definition-grid",
            "border: 1px dashed #cbd5e1",
            ":focus-visible",
            "overflow-wrap: anywhere",
            "max-height: calc(100vh - 32px)",
            "@media (max-width: 900px)",
        ):
            assert css_hook in html

    def test_appendix_contains_reproducibility_strategy_and_module_status(self, ic_result, bt_result, to_result):
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            date_range="20250101 ~ 20250331",
            strategy_results={"quantile_ls_5": bt_result},
            primary_strategy="quantile_ls_5",
            walk_forward_summary={"status": "insufficient_data", "n_folds": 0},
        )

        assert "复现摘要" in html
        assert "策略配置清单" in html
        assert "策略代码" in html
        assert "中文策略名用于阅读和横向比较" in html
        assert "模块状态" in html
        assert "模块完整性总览" in html
        assert "未生成模块不一定影响当前结论" in html
        assert "<th>读法 / 下一步</th>" in html
        assert "延长评估区间或降低验证窗口要求后复跑。" in html
        assert "若该维度影响结论，应补齐配置或输入数据后复跑。" in html
        assert "组合归因（Brinson/Barra）" in html
        assert "真实评估区间" in html
        assert "滚动样本外" in html
        assert "样本不足，未生成滚动验证折数" in html
        assert "未来验证期折叠" not in html
        assert "<td>滚动样本外</td>" in html
        assert "<td>样本不足</td>" in html
        assert "未传入事件研究结果" in html
        assert "<td>因子相关性</td>" not in html
        assert "insufficient_data" not in html
        assert "event_study_result" not in html
        assert "factor_corr" not in html
        assert "benchmark_result" not in html
        assert "walk_forward_result" not in html
        assert "walk_forward_summary" not in html

    def test_appendix_module_status_summary_handles_complete_report(
        self, ic_result, bt_result, to_result
    ):
        from factorzen.daily.evaluation.benchmark import BenchmarkResult

        dates = [f"2025-{(i // 20 + 1):02d}-{(i % 20 + 1):02d}" for i in range(60)]
        zeros = np.zeros(60)
        benchmark = BenchmarkResult(
            benchmark_code="000300.SH",
            benchmark_name="HS300",
            daily=pl.DataFrame(
                {
                    "trade_date": pl.Series(dates).str.strptime(pl.Date, "%Y-%m-%d"),
                    "strategy_ret": zeros,
                    "benchmark_ret": zeros,
                    "excess_ret": zeros,
                    "strategy_nav": np.ones(60),
                    "benchmark_nav": np.ones(60),
                    "excess_nav": np.ones(60),
                }
            ),
            ann_excess_ret=0.0,
            tracking_error=0.0,
            information_ratio=0.0,
            excess_max_dd=0.0,
        )
        event_study = EventStudyResult(
            windows=[0, 1],
            avg_cumret=np.array([0.0, 0.01]),
            ci_95=np.array([0.0, 0.005]),
            n_events=20,
        )
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            benchmark_result=benchmark,
            event_study_result=event_study,
            factor_corr=pl.DataFrame(
                {"factor": ["a", "b"], "a": [1.0, 0.82], "b": [0.82, 1.0]}
            ),
        )

        assert "模块完整性总览" in html
        assert "可作为当前报告证据。" in html
        assert "相关性结论" in html
        assert "最高重叠因子对为 a / b" in html
        assert "高重叠因子" in html

    def test_factor_corr_with_no_valid_off_diagonal_is_not_marked_generated(
        self, ic_result, bt_result, to_result
    ):
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            factor_corr=pl.DataFrame(
                {
                    "factor": ["a", "b"],
                    "a": [1.0, np.nan],
                    "b": [np.nan, 1.0],
                }
            ),
        )
        structure_html = html.split('<div class="panel" id="structure-checks">', 1)[1].split(
            '<div class="panel" id="tradability">', 1
        )[0]
        appendix_html = html.split('<div class="panel" id="appendix">', 1)[1]

        assert "相关性矩阵缺少有效的非对角元素" in structure_html
        assert "暂不绘制因子相关性热力图" in structure_html
        assert '<img class="chart-img"' not in structure_html
        assert "<td>因子相关性</td>" in appendix_html
        assert "<td>样本不足</td>" in appendix_html
        assert "相关性矩阵缺少有效的非对角元素" in appendix_html
        assert ">nan<" not in html
        assert ">NaN<" not in html

    def test_single_factor_corr_input_is_hidden_for_single_factor_report(
        self, ic_result, bt_result, to_result
    ):
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            factor_corr=pl.DataFrame({"factor": ["test_factor"], "test_factor": [1.0]}),
        )
        structure_html = html.split('<div class="panel" id="structure-checks">', 1)[1].split(
            '<div class="panel" id="tradability">', 1
        )[0]
        appendix_html = html.split('<div class="panel" id="appendix">', 1)[1]

        assert "因子相关性" not in structure_html
        assert "相关性结论" not in structure_html
        assert "暂不绘制因子相关性热力图" not in structure_html
        assert "<td>因子相关性</td>" not in appendix_html

    def test_appendix_explains_metric_definitions(self, ic_result, bt_result, to_result):
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            strategy_results={"quantile_ls_5": bt_result},
            primary_strategy="quantile_ls_5",
        )

        for marker in (
            "指标口径说明",
            "Rank IC",
            "信息比率 (IR)",
            "夏普比率",
            "短样本年化",
            "最大回撤",
            "平均换手",
            "交易成本",
            "市值分层 IC",
            "行业分层 IC",
            "多空组合",
        ):
            assert marker in html

    def test_short_sample_annualized_metrics_are_flagged(
        self, ic_result, bt_result, to_result
    ):
        short_ic = replace(ic_result, n_periods=15)
        html = generate_tear_sheet("test_factor", short_ic, bt_result, to_result)

        assert "短样本年化提示" in html
        assert "年化收益、年化波动、Sharpe、最大回撤和基准超额主要用于横向比较策略口径" in html
        assert "短样本年化指标仅适合同区间横向比较" in html

    def test_appendix_contains_quality_report_details(self, ic_result, bt_result, to_result):
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            quality_report={
                "status": "warning",
                "checks": {
                    "factor_clean": {
                        "rows": 100,
                        "valid_count": 95,
                        "null_count": 5,
                        "inf_count": 0,
                        "coverage": 0.95,
                    }
                },
                "warnings": ["factor_clean coverage is low: 95.0%"],
                "errors": [],
            },
        )

        assert "数据质量摘要" in html
        assert "状态：需关注" in html
        assert "质量影响判断" in html
        assert "覆盖率最低的检查项是清洗后因子值（95.0%）" in html
        assert "覆盖率存在缺口，结论仍可阅读" in html
        assert "缺失值合计 5 个，无限值合计 0 个" in html
        assert "<th>读法</th>" in html
        assert "覆盖率需关注" in html
        assert "<td>数据质量</td>" in html
        assert "<td>需关注</td>" in html
        assert "清洗后因子值" in html
        assert "95.0%" in html
        assert "清洗后因子值覆盖率偏低" in html
        assert "factor_clean" not in html
        assert "coverage is low" not in html

    def test_appendix_quality_summary_downgrades_low_coverage(
        self, ic_result, bt_result, to_result
    ):
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            quality_report={
                "status": "error",
                "checks": {
                    "forward_return": {
                        "rows": 100,
                        "valid_count": 82,
                        "null_count": 16,
                        "inf_count": 2,
                        "coverage": 0.82,
                    }
                },
                "warnings": [],
                "errors": ["forward_return contains inf values"],
            },
        )

        assert "覆盖率最低的检查项是前向收益（82.0%）" in html
        assert "覆盖率偏低，因子 IC、分组收益和回测结果的可信度需要降级。" in html
        assert "缺失值合计 16 个，无限值合计 2 个" in html
        assert "存在无限值，需优先复核清洗流程。" in html
        assert "forward_return" not in html

    def test_appendix_quality_summary_treats_nan_coverage_as_missing(
        self, ic_result, bt_result, to_result
    ):
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            quality_report={
                "status": "warning",
                "checks": {
                    "factor_clean": {
                        "rows": 100,
                        "valid_count": 95,
                        "null_count": 5,
                        "inf_count": 0,
                        "coverage": np.nan,
                    }
                },
                "warnings": [],
                "errors": [],
            },
        )
        appendix_html = html.split('<div class="panel" id="appendix">', 1)[1]

        assert "当前质量检查未提供覆盖率" in appendix_html
        assert "未提供覆盖率。" in appendix_html
        assert "覆盖率偏低" not in appendix_html
        assert "结论需降级" not in appendix_html
        assert ">nan<" not in html
        assert ">NaN<" not in html

    def test_report_replaces_nan_metrics_with_sample_notice(self, ic_result, bt_result, to_result):
        ic_result.multi_period = {
            20: {
                "ic_mean": np.nan,
                "ic_std": np.nan,
                "ir": np.nan,
                "ic_positive_ratio": np.nan,
                "tstat": np.nan,
                "pvalue": np.nan,
            }
        }
        advanced_results = {
            "decay_results": [ICDecayResult(horizon=20, ic_mean=np.nan, ic_std=np.nan)]
        }

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )

        assert "样本不足" in html
        assert ">nan<" not in html
        assert ">NaN<" not in html

    def test_daily_report_marks_missing_core_ic_as_sample_insufficient(
        self, ic_result, bt_result, to_result
    ):
        ic_result.ic_mean = None
        ic_result.ic_std = None
        ic_result.ir = None
        ic_result.ic_positive_ratio = None
        ic_result.ic_tstat = None
        ic_result.ic_pvalue = None
        ic_result.ic_series = pl.DataFrame()

        html = generate_tear_sheet("test_factor", ic_result, bt_result, to_result)

        assert '<div class="label">IC 均值</div><div class="value">样本不足</div>' in html
        assert '<div class="label">IC 标准差</div><div class="value">样本不足</div>' in html
        assert '<div class="label">信息比率 (IR)</div><div class="value">样本不足</div>' in html
        assert '<div class="label">IC 胜率</div><div class="value">样本不足</div>' in html
        assert '<div class="label">t 统计量</div><div class="value">样本不足</div>' in html
        assert "核心 IC 指标样本不足，暂不判断预测方向" in html
        assert "IC 均值极低" not in html
        assert "IC 接近 0" not in html

    def test_zero_event_study_is_reported_as_empty_not_generated(
        self, ic_result, bt_result, to_result
    ):
        empty_event_study = EventStudyResult(
            windows=[],
            avg_cumret=np.array([]),
            ci_95=np.array([]),
            n_events=0,
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            event_study_result=empty_event_study,
        )

        assert "事件研究无有效事件" in html
        assert "<td>事件研究</td>" in html
        assert "<td>无有效事件</td>" in html
        assert "没有满足阈值和事件窗口要求的有效事件" in html
        assert "下一步：放宽 Top 分位阈值、缩短事件窗口或扩大股票池后复跑" in html
        assert "事件数量: 0" not in html
        assert 'alt="事件研究图"' not in html

    def test_event_study_with_events_but_empty_windows_is_sample_insufficient(
        self, ic_result, bt_result, to_result
    ):
        incomplete_event_study = EventStudyResult(
            windows=[],
            avg_cumret=np.array([]),
            ci_95=np.array([]),
            n_events=12,
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            event_study_result=incomplete_event_study,
        )
        robustness_html = html.split('<div class="panel" id="robustness">', 1)[1].split(
            '<div class="panel" id="risk-attribution">', 1
        )[0]
        appendix_html = html.split('<div class="panel" id="appendix">', 1)[1]

        assert "事件研究样本不足" in robustness_html
        assert "已找到 12 个事件，但事件窗口收益序列为空或不完整" in robustness_html
        assert "下一步：检查事件窗口收益拼接、停牌过滤和事件窗口长度" in robustness_html
        assert "<td>事件研究</td>" in appendix_html
        assert "<td>样本不足</td>" in appendix_html
        assert "事件窗口收益序列为空或不完整" in appendix_html
        assert "事件数量: 12" not in html
        assert 'alt="事件研究图"' not in html

    def test_event_study_section_contains_result_summary(
        self, ic_result, bt_result, to_result
    ):
        event_study = EventStudyResult(
            windows=[-1, 0, 1, 2],
            avg_cumret=np.array([0.0, 0.01, 0.03, 0.05]),
            ci_95=np.array([0.00, 0.01, 0.015, 0.02]),
            n_events=12,
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            event_study_result=event_study,
        )

        assert "事件研究摘要" in html
        assert "因子 Top 分位事件数 12 个" in html
        assert "事件后 2 期平均累计收益为 5.00%" in html
        assert "事件后收益为正，说明高分位信号在事件窗口内有正向延续。" in html
        assert "95% 置信区间半宽约 2.00%" in html
        assert "事件数量偏少，应将该结果视为辅助证据。" in html

    def test_event_study_section_warns_on_negative_event_return(
        self, ic_result, bt_result, to_result
    ):
        event_study = EventStudyResult(
            windows=[0, 1, 5],
            avg_cumret=np.array([0.0, -0.01, -0.04]),
            ci_95=np.array([0.00, 0.01, 0.03]),
            n_events=45,
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            event_study_result=event_study,
        )

        assert "事件后 5 期平均累计收益为 -4.00%" in html
        assert "事件后收益为负，说明高分位信号在事件窗口内未延续" in html
        assert "事件数量偏少" not in html

    def test_event_study_section_marks_nan_final_return_as_insufficient(
        self, ic_result, bt_result, to_result
    ):
        event_study = EventStudyResult(
            windows=[0, 1, 5],
            avg_cumret=np.array([0.0, 0.01, np.nan]),
            ci_95=np.array([0.00, 0.01, 0.02]),
            n_events=45,
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            event_study_result=event_study,
        )

        assert "事件后 5 期平均累计收益样本不足" in html
        assert "事件后累计收益接近 0" not in html

    def test_event_study_section_marks_nan_ci_as_insufficient(
        self, ic_result, bt_result, to_result
    ):
        event_study = EventStudyResult(
            windows=[0, 1, 5],
            avg_cumret=np.array([0.0, 0.01, 0.05]),
            ci_95=np.array([0.00, 0.01, np.nan]),
            n_events=45,
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            event_study_result=event_study,
        )

        assert "事件后 5 期平均累计收益为 5.00%" in html
        assert "95% 置信区间样本不足" in html
        assert "95% 置信区间半宽约 样本不足" not in html

    def test_event_study_with_none_ci_does_not_crash_report(
        self, ic_result, bt_result, to_result
    ):
        # 报告侧 _make_event_study_chart 容忍 ci_95=None（zeros_like），
        # 因此图表照常生成；模板曾在该分支直接对 None 下标取值导致整份报告崩溃。
        event_study = EventStudyResult(
            windows=[-1, 0, 1, 2],
            avg_cumret=np.array([0.0, 0.01, 0.03, 0.05]),
            ci_95=None,
            n_events=40,
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            event_study_result=event_study,
        )

        assert "事件研究摘要" in html
        assert "事件后 2 期平均累计收益为 5.00%" in html
        assert "95% 置信区间样本不足" in html
        assert 'alt="事件研究图"' in html

    def test_factor_weighted_long_only_is_consistently_long_only(
        self, ic_result, bt_result, to_result
    ):
        # factor_weighted + long_only=True：概览总结判为多头，
        # 策略分页过去却无条件判为多空，导致同一份报告对同一策略
        # 既写"组合收益"又写"多空组合"。
        fw_long_only = replace(
            bt_result,
            strategy_name="factor_weighted",
            summary_stats={
                "portfolio": {
                    "ann_ret": 0.08,
                    "ann_vol": 0.16,
                    "sharpe": 0.50,
                    "max_dd": -0.08,
                    "avg_turnover": 0.25,
                    "total_cost": 0.01,
                    "ann_turnover": 63.0,
                }
            },
            config={
                "strategy_type": "factor_weighted",
                "strategy_params": {"long_only": True},
                "cost_model": "linear",
            },
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            fw_long_only,
            to_result,
            strategy_results={"factor_weighted": fw_long_only},
            primary_strategy="factor_weighted",
        )

        # 策略分页收益口径与概览总结一致，均按多头呈现
        assert "收益口径：组合收益" in html
        assert "收益口径：多空组合" not in html
        assert "主策略组合年化收益 8.0%" in html

    def test_strategy_page_contains_trade_execution_summary(self, ic_result, bt_result, to_result):
        constrained = replace(
            bt_result,
            strategy_name="topn_long_only",
            summary_stats={
                "portfolio": {
                    "ann_ret": 0.08,
                    "ann_vol": 0.16,
                    "sharpe": 0.50,
                    "max_dd": -0.08,
                    "avg_turnover": 0.25,
                    "total_cost": 0.01,
                    "ann_turnover": 63.0,
                }
            },
            trades=pl.DataFrame(
                {
                    "trade_date": ["2025-01-01", "2025-01-02"],
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "prev_weight": [0.0, 0.0],
                    "target_weight": [0.2, 0.2],
                    "filled_delta_weight": [0.1, 0.0],
                    "turnover": [0.1, 0.0],
                    "cost": [0.001, 0.0],
                    "block_reason": ["capacity", "limit_up"],
                }
            ),
            config={"strategy_type": "topn_long_only", "cost_model": "linear"},
        )
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            constrained,
            to_result,
            strategy_results={"topn_50": constrained},
            primary_strategy="topn_50",
        )

        assert "交易执行摘要" in html
        assert "约束原因" in html
        assert "成交容量限制" in html
        assert "涨停限制" in html
        assert "capacity" not in html
        assert "limit_up" not in html

    def test_tradability_section_contains_cross_strategy_execution_summary(
        self, ic_result, bt_result, to_result
    ):
        constrained = replace(
            bt_result,
            strategy_name="topn_long_only",
            summary_stats={
                "portfolio": {
                    "ann_ret": 0.08,
                    "ann_vol": 0.16,
                    "sharpe": 0.50,
                    "max_dd": -0.08,
                    "avg_turnover": 0.25,
                    "total_cost": 0.01,
                    "ann_turnover": 63.0,
                }
            },
            trades=pl.DataFrame(
                {
                    "trade_date": ["2025-01-01", "2025-01-02"],
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "prev_weight": [0.0, 0.0],
                    "target_weight": [0.2, 0.2],
                    "filled_delta_weight": [0.1, 0.0],
                    "turnover": [0.1, 0.0],
                    "cost": [0.001, 0.0],
                    "block_reason": ["capacity", "limit_up"],
                }
            ),
            config={"strategy_type": "topn_long_only", "cost_model": "linear"},
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            constrained,
            to_result,
            strategy_results={"topn_50": constrained},
            primary_strategy="topn_50",
        )

        assert "交易可行性摘要" in html
        assert "执行结论" in html
        assert "成交约束占比最高：TopN 多头 50（100.0%）" in html
        assert "TopN 多头 50 的成交约束" not in html
        assert "TopN 多头 50的成交约束" not in html
        assert "当前执行风险较高" in html
        assert "）。 当前" not in html
        assert "优先复核成交容量限制" in html
        assert "受约束交易越多" in html
        assert "主要约束" in html
        assert "成交容量限制 1 次" in html
        assert "涨停限制 1 次" in html
        assert "<td>2 / 2</td>" in html
        assert "capacity" not in html
        assert "limit_up" not in html

    def test_tradability_summary_marks_missing_execution_metrics_as_insufficient(
        self, ic_result, bt_result, to_result
    ):
        missing_execution = replace(
            bt_result,
            strategy_name="topn_long_only",
            summary_stats={
                "portfolio": {
                    "ann_ret": 0.08,
                    "ann_vol": 0.16,
                    "sharpe": 0.50,
                    "max_dd": -0.08,
                }
            },
            trades=pl.DataFrame(),
            config={"strategy_type": "topn_long_only", "cost_model": "linear"},
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            missing_execution,
            to_result,
            strategy_results={"topn_50": missing_execution},
            primary_strategy="topn_50",
        )
        tradability_html = html.split('<div class="panel" id="tradability">', 1)[1].split(
            '<div class="panel" id="robustness">', 1
        )[0]

        assert "执行结论" in tradability_html
        assert "交易执行指标样本不足" in tradability_html
        assert "暂不判断执行瓶颈" in tradability_html
        assert "未显示明显执行瓶颈" not in tradability_html
        assert "成交约束占比最高" not in tradability_html

    def test_html_table_of_contents_matches_report_order(self, ic_result, bt_result, to_result):
        html = generate_tear_sheet("test_factor", ic_result, bt_result, to_result)

        expected = [
            'href="#overview"',
            'href="#returns"',
            'href="#predictive-power"',
            'href="#structure-checks"',
            'href="#tradability"',
            'href="#robustness"',
            'href="#risk-attribution"',
            'href="#appendix"',
        ]
        toc_positions = [html.index(item) for item in expected]
        panel_positions = [
            html.index('id="overview"'),
            html.index('id="returns"'),
            html.index('id="predictive-power"'),
            html.index('id="structure-checks"'),
            html.index('id="tradability"'),
            html.index('id="robustness"'),
            html.index('id="risk-attribution"'),
            html.index('id="appendix"'),
        ]

        assert toc_positions == sorted(toc_positions)
        assert panel_positions == sorted(panel_positions)

    def test_html_contains_enhanced_visual_gallery(self, ic_result, bt_result, to_result):
        html = generate_tear_sheet("test_factor", ic_result, bt_result, to_result)

        assert 'alt="IC 分布图"' in html
        assert 'alt="月度收益热力图"' in html
        assert html.count("data:image/png;base64,") >= 5

    def test_quantile_spread_explains_positive_and_reverse_direction(
        self, ic_result, bt_result, to_result
    ):
        html = generate_tear_sheet("test_factor", ic_result, bt_result, to_result)

        assert "价差为正表示最高分位跑赢最低分位" in html
        assert "价差为负表示反向分层更强" in html
        assert "价差持续正数说明因子有较强的分层区分能力" not in html

    def test_quantile_spread_explains_missing_chart_when_group_nav_exists(
        self, ic_result, bt_result, to_result, monkeypatch
    ):
        import factorzen.reports.tear_sheet as tear_sheet

        monkeypatch.setattr(tear_sheet, "_make_quantile_spread_chart", lambda *_: None)

        html = generate_tear_sheet("test_factor", ic_result, bt_result, to_result)
        returns_html = html.split('<div class="panel" id="returns">', 1)[1].split(
            '<div class="panel" id="predictive-power">', 1
        )[0]

        assert "分位价差（Quantile Spread）" in returns_html
        assert "分位价差图未生成" in returns_html
        assert "已识别至少两组分位组合净值" in returns_html
        assert 'alt="分位价差图"' not in returns_html

    def test_html_shows_effective_evaluation_window(self, ic_result, bt_result, to_result):
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            date_range="2025-01-01 ~ 2025-12-31",
        )

        assert "真实评估区间" in html
        assert "因子预热" in html
        assert '<p class="note">' in html

    def test_risk_attribution_is_last_analysis_section(
        self, ic_result, bt_result, to_result
    ):
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            walk_forward_summary={"status": "insufficient_data", "n_folds": 0},
        )

        assert html.index('<div class="panel" id="risk-attribution">') > html.index("滚动样本外稳健性")
        assert html.index('<div class="panel" id="risk-attribution">') < html.index('<div class="panel" id="appendix">')

    def test_risk_attribution_orders_size_monotonicity_benchmark_before_sector(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        from factorzen.daily.evaluation.benchmark import BenchmarkResult

        dates = [f"2025-{(i // 20 + 1):02d}-{(i % 20 + 1):02d}" for i in range(60)]
        zeros = np.zeros(60)
        benchmark = BenchmarkResult(
            benchmark_code="000300.SH",
            benchmark_name="HS300",
            daily=pl.DataFrame(
                {
                    "trade_date": pl.Series(dates).str.strptime(pl.Date, "%Y-%m-%d"),
                    "strategy_ret": zeros,
                    "benchmark_ret": zeros,
                    "excess_ret": zeros,
                    "strategy_nav": np.ones(60),
                    "benchmark_nav": np.ones(60),
                    "excess_nav": np.ones(60),
                }
            ),
            ann_excess_ret=0.0,
            tracking_error=0.0,
            information_ratio=0.0,
            excess_max_dd=0.0,
        )
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
            benchmark_result=benchmark,
        )

        sector_pos = html.index("行业分层 IC")
        assert html.index("市值分层 IC") < sector_pos
        assert html.index("单调性得分") < sector_pos
        assert html.index("基准对比") < sector_pos

    def test_sector_ic_table_is_collapsible(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        advanced_results = dict(advanced_results)
        advanced_results["sector"].sector_ic_df = pl.DataFrame(
            {
                "sector": [f"sector_{i}" for i in range(8)],
                "ic": [0.01 + i * 0.001 for i in range(8)],
            }
        )
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )

        assert 'id="sector-ic-toggle"' in html
        assert 'class="sector-ic-table is-collapsed"' in html
        assert 'data-expanded-label="收起全部"' in html
        assert 'data-collapsed-label="展开全部"' in html
        assert "只展示前 6 行" in html

    def test_sector_ic_section_contains_consistency_summary(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )

        risk_html = html.split('<div class="panel" id="risk-attribution">', 1)[1].split(
            '<div class="panel" id="appendix">', 1
        )[0]
        assert "行业一致性摘要" in risk_html
        assert "覆盖行业大多为正向 IC，行业内预测方向较一致。" in risk_html
        assert "IC 数值最高的行业是 tech（IC=0.0350）" in risk_html
        assert "IC 数值最低的是 cons（IC=0.0220）" in risk_html
        assert "绝对 IC 最大的行业是 tech（IC=0.0350）" in risk_html
        assert "行业间差异有限，当前样本未显示强烈行业依赖。" in risk_html
        assert "<th>方向读法</th>" in risk_html
        assert "<td>正向</td>" in risk_html

    def test_sector_ic_section_warns_when_industries_are_split(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        advanced_results = dict(advanced_results)
        advanced_results["sector"].sector_ic_df = pl.DataFrame(
            {
                "sector": ["bank", "tech", "energy"],
                "ic": [-0.050, 0.045, 0.002],
            }
        )
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )

        assert "行业间 IC 正负分化，因子可能依赖行业环境或行业暴露。" in html
        assert "IC 数值最高的行业是 tech（IC=0.0450）" in html
        assert "IC 数值最低的是 bank（IC=-0.0500）" in html
        assert "绝对 IC 最大的行业是 bank（IC=-0.0500）" in html
        assert "行业差异很大，应重点检查行业中性化、行业容量和行业轮动影响。" in html
        assert "<td>反向</td>" in html
        assert "<td>不明显</td>" in html

    def test_sector_ic_summary_handles_all_negative_industries_as_reverse_signal(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        advanced_results = dict(advanced_results)
        advanced_results["sector"].sector_ic_df = pl.DataFrame(
            {
                "sector": ["bank", "tech", "energy"],
                "ic": [-0.020, -0.040, -0.080],
            }
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )

        assert "覆盖行业大多为负向 IC，行业内方向一致但更适合按反向信号理解。" in html
        assert "IC 数值最高的行业是 bank（IC=-0.0200）" in html
        assert "IC 数值最低的是 energy（IC=-0.0800）" in html
        assert "按反向因子理解，绝对 IC 最大的行业是 energy（|IC|=0.0800）" in html
        assert "最强行业是 bank" not in html

    def test_sector_ic_section_handles_missing_ic_values(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        advanced_results = dict(advanced_results)
        advanced_results["sector"].sector_ic_df = pl.DataFrame(
            {
                "sector": ["bank", "tech"],
                "ic": [None, np.nan],
            },
            schema={"sector": pl.Utf8, "ic": pl.Float64},
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )
        risk_html = html.split('<div class="panel" id="risk-attribution">', 1)[1].split(
            '<div class="panel" id="appendix">', 1
        )[0]

        assert "行业分层 IC" in risk_html
        assert "行业一致性摘要" not in risk_html
        assert "样本不足" in risk_html
        assert ">nan<" not in html
        assert ">NaN<" not in html
        assert ">None<" not in html

    def test_size_ic_section_contains_exposure_summary(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )

        risk_html = html.split('<div class="panel" id="risk-attribution">', 1)[1].split(
            '<div class="panel" id="appendix">', 1
        )[0]
        assert "市值敞口摘要" in risk_html
        assert "IC 数值最高的市值段是中盘（IC=0.0330）" in risk_html
        assert "IC 数值最低的是小盘（IC=0.0250）" in risk_html
        assert "绝对 IC 最大的市值段是中盘（IC=0.0330）" in risk_html
        assert "各市值段 IC 接近，当前样本未显示强烈规模敞口。" in risk_html
        assert "<td>大盘</td>" in risk_html
        assert "<td>中盘</td>" in risk_html
        assert "<td>小盘</td>" in risk_html

    def test_size_ic_section_warns_when_size_spread_is_large(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        advanced_results = dict(advanced_results)
        advanced_results["size"] = replace(
            advanced_results["size"],
            buckets={"Large": 0.070, "Mid": 0.020, "Small": -0.010},
        )
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )

        assert "IC 数值最高的市值段是大盘（IC=0.0700）" in html
        assert "IC 数值最低的是小盘（IC=-0.0100）" in html
        assert "绝对 IC 最大的市值段是大盘（IC=0.0700）" in html
        assert "大小盘差异明显，使用时应重点检查市值中性化和组合市值暴露。" in html

    def test_size_ic_summary_handles_all_negative_buckets_as_reverse_signal(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        advanced_results = dict(advanced_results)
        advanced_results["size"] = replace(
            advanced_results["size"],
            buckets={"Large": -0.030, "Mid": -0.060, "Small": -0.090},
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )

        assert "IC 数值最高的市值段是大盘（IC=-0.0300）" in html
        assert "IC 数值最低的是小盘（IC=-0.0900）" in html
        assert "按反向因子理解，绝对 IC 最大的市值段是小盘（|IC|=0.0900）" in html
        assert "预测能力最强的市值段是大盘" not in html

    def test_size_ic_section_handles_missing_bucket_values(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        advanced_results = dict(advanced_results)
        advanced_results["size"] = replace(
            advanced_results["size"],
            buckets={"Large": None, "Mid": np.nan, "Small": 0.018},
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )
        risk_html = html.split('<div class="panel" id="risk-attribution">', 1)[1].split(
            '<div class="panel" id="appendix">', 1
        )[0]

        assert "市值分层 IC" in risk_html
        assert "IC 数值最高的市值段是小盘（IC=0.0180）" in risk_html
        assert "样本不足" in risk_html
        assert ">nan<" not in html
        assert ">NaN<" not in html
        assert ">None<" not in html

    def test_size_ic_section_explains_when_all_bucket_ic_values_are_missing(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        advanced_results = dict(advanced_results)
        advanced_results["size"] = replace(
            advanced_results["size"],
            buckets={"Large": None, "Mid": np.nan, "Small": None},
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )
        risk_html = html.split('<div class="panel" id="risk-attribution">', 1)[1].split(
            '<div class="panel" id="appendix">', 1
        )[0]

        assert "市值分层 IC" in risk_html
        assert "市值分层 IC 已传入，但有效 IC 样本不足" in risk_html
        assert "暂不判断大小盘强弱和规模敞口" in risk_html
        assert "IC 数值最高的市值段是" not in risk_html
        assert ">nan<" not in html
        assert ">NaN<" not in html
        assert ">None<" not in html

    def test_sector_ic_empty_state_includes_actionable_next_step(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        advanced_results = dict(advanced_results)
        advanced_results.pop("sector")

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )

        assert "无行业分解数据" in html
        assert "下一步：补齐股票行业分类" in html
        assert "确认行业 IC 模块已启用后复跑" in html

    def test_none_backtest(self, ic_result, to_result):
        """None backtest is handled gracefully."""
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            None,
            to_result,
            date_range="2025-01-01 ~ 2025-05-13",
        )
        assert isinstance(html, str)
        assert "无回测数据" in html
        assert "当前流程未提供回测结果" in html
        assert "backtest_result" not in html

    def test_html_shows_current_backtest_strategy(self, ic_result, bt_result, to_result):
        """Report shows the current backtest strategy."""
        html = generate_tear_sheet("test_factor", ic_result, bt_result, to_result)

        assert "主回测策略" in html
        assert "分位数组合多空" in html
        assert "quantile_long_short" in html

    def test_html_explains_missing_attribution(self, ic_result, bt_result, to_result):
        """Missing attribution explains why the section is empty."""
        html = generate_tear_sheet("test_factor", ic_result, bt_result, to_result)

        assert "组合归因未生成" in html
        assert "当前流程未提供归因结果" in html
        assert "attribution_result" not in html

    def test_html_marks_reversed_backtest_direction(self, ic_result, bt_result, to_result):
        """Report marks automatic reversed backtest direction."""
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            backtest_direction={
                "direction": "reversed",
                "reason": "IC 均值为负且 p 值小于等于 0.10",
            },
        )

        assert "已自动反向回测" in html
        assert "IC 均值为负" in html
        assert '<p class="note">已自动反向回测' in html

    def test_reversed_backtest_direction_is_visible_in_strategy_context(
        self, ic_result, bt_result, to_result
    ):
        topn_result = replace(
            bt_result,
            strategy_name="topn_50",
            config={"strategy_type": "topn_long_only"},
        )
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            topn_result,
            to_result,
            strategy_results={"topn_50": topn_result},
            primary_strategy="topn_50",
            backtest_direction={
                "direction": "reversed",
                "reason": "IC 均值为负且 p 值小于等于 0.10",
            },
        )

        assert "方向：多头 TopN | 信号口径：反向因子" in html
        assert "策略代码：topn_50 | 方向：多头 TopN | 信号口径：反向因子" in html
        assert "反向因子：已按 IC 方向自动反向回测" in html
        assert "多头 TopN<span class=\"strategy-code\">信号口径：反向因子</span>" in html

    def test_none_turnover(self, ic_result, bt_result):
        """None turnover is handled gracefully."""
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            None,
            date_range="2025-01-01 ~ 2025-05-13",
        )
        assert isinstance(html, str)
        assert "无换手率数据" in html
        assert "未传入调仓序列" in html

    def test_turnover_metrics_explain_missing_chart(
        self, ic_result, bt_result, to_result, monkeypatch
    ):
        import factorzen.reports.tear_sheet as tear_sheet

        monkeypatch.setattr(tear_sheet, "_make_turnover_chart", lambda *_: None)

        html = generate_tear_sheet("test_factor", ic_result, bt_result, to_result)
        tradability_html = html.split('<div class="panel" id="tradability">', 1)[1].split(
            '<div class="panel" id="robustness">', 1
        )[0]

        assert "平均换手率" in html
        assert "换手率曲线未生成" in tradability_html
        assert "已生成平均换手率" in tradability_html
        assert "无换手率数据：未传入调仓序列" not in tradability_html
        assert 'alt="换手率图"' not in tradability_html

    def test_all_none_results(self):
        """All None results still generate valid HTML."""
        html = generate_tear_sheet(
            "empty_factor",
            None,
            None,
            None,
            date_range="2025-01-01 ~ 2025-01-02",
        )
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "empty_factor" in html

    def test_empty_report_no_data_blocks_include_next_actions(self):
        html = generate_tear_sheet(
            "empty_factor",
            None,
            None,
            None,
            date_range="2025-01-01 ~ 2025-01-02",
        )

        assert "下一步：补齐策略回测结果后复跑" in html
        assert "下一步：先生成 Rank IC 时序和 IC 分布" in html
        assert "下一步：启用中性化 IC、多持有期一致性、单调性和信号持续性检查" in html
        assert "下一步：补齐调仓权重、成交记录或换手率序列" in html
        assert "下一步：至少补齐样本外分割或滚动样本外验证" in html
        assert "滚动未来验证期" not in html

    def test_summary_has_stars(self, ic_result, bt_result, to_result):
        """Summary contains star rating."""
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            date_range="2025-01-01 ~ 2025-05-13",
        )
        assert chr(9733) in html  # star
    def test_summary_contains_factor_rating_scorecard(self, ic_result, bt_result, to_result, advanced_results):
        """Summary contains research-grade factor rating scorecard."""
        ic_result.oos_ic = {"train": 0.03, "test": 0.028}
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
            neutralized_ic_result=ic_result,
        )

        assert "评分卡总分" in html
        assert "Alpha 强度" in html
        assert "可交易性" in html
        assert "评级标签" in html
        assert "候选（candidate）" in html
        assert "<th>满分</th>" not in html
        assert "<tr><td>Alpha 强度</td>" in html
        assert "/30</td>" in html
        assert "/25</td>" in html
        assert "/20</td>" in html
        assert "/15</td>" in html
        assert "/10</td>" in html

    def test_summary_rating_scorecard_explains_component_readouts(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        """Rating dimensions explain what the score means and what to inspect next."""
        ic_result.oos_ic = {"train": 0.03, "test": 0.028}
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
            neutralized_ic_result=ic_result,
        )

        assert "<th>读法 / 下一步</th>" in html
        assert "Alpha 证据" in html
        assert "样本外、IC 衰减和跨持有期方向" in html
        assert "执行成本和换手" in html
        assert "中性化、Pearson/Rank 一致性" in html
        assert "分组单调性和信号持续性" in html

    def test_overview_contains_actionable_research_decision(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        ic_result.oos_ic = {"train": 0.03, "test": 0.028}
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
            neutralized_ic_result=ic_result,
            walk_forward_summary={"status": "insufficient_data", "n_folds": 0},
        )

        assert "最终研究决策" in html
        assert "当前结论" in html
        assert "建议动作" in html
        assert "证据强度" in html
        assert "关键缺口" in html
        assert "下一步" in html
        assert "进入候选池观察" in html
        assert "缺少滚动样本外验证" in html

    def test_research_decision_describes_partial_benchmark_and_attribution(
        self, ic_result, bt_result, to_result
    ):
        from factorzen.daily.evaluation.attribution import BrinsonResult
        from factorzen.daily.evaluation.benchmark import BenchmarkResult

        benchmark = BenchmarkResult(
            benchmark_code="000300.SH",
            benchmark_name="HS300",
            daily=pl.DataFrame(),
            ann_excess_ret=0.05,
            tracking_error=0.08,
            information_ratio=0.625,
            excess_max_dd=-0.03,
        )
        brinson = BrinsonResult(
            sector_df=pl.DataFrame(),
            period_df=pl.DataFrame(),
            ann_allocation=0.01,
            ann_selection=0.04,
            ann_interaction=-0.005,
            ann_active_return=0.045,
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            benchmark_result=benchmark,
            attribution_result={"brinson": brinson},
        )
        overview_html = html.split('<div class="panel" id="overview">', 1)[1].split(
            '<div class="panel" id="returns">', 1
        )[0]
        appendix_html = html.split('<div class="panel" id="appendix">', 1)[1]

        assert "基准超额图表或日度明细不完整" in overview_html
        assert "归因图表或行业/风格明细不完整" in overview_html
        assert "缺少基准超额对比" not in overview_html
        assert "缺少组合归因证据" not in overview_html
        assert "补齐基准日度净值明细后复核超额曲线。" in appendix_html
        assert "补齐归因行业或风格明细后复核归因图。" in appendix_html
        assert "先处理质量警告，再提高结论置信度。" not in appendix_html

    def test_rating_caps_are_rendered_as_separated_notice_items(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        ic_result.n_periods = 15
        advanced_results["mono"].monotonicity_score = 0.40
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )

        assert 'class="rating-caps"' in html
        assert 'class="rating-cap-item"' in html
        assert "样本期数少于 60，最高 2 星" in html
        assert "分组单调性不足，最高 3 星" in html
        assert "<p><strong>评级上限：</strong></p><ul>" not in html

    def test_scorecard_explains_when_rating_label_is_capped(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        ic_result.n_periods = 15
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
            neutralized_ic_result=ic_result,
        )

        scorecard_line = html.split("<strong>评分卡总分：</strong>", 1)[1].split("</p>", 1)[0]

        assert "评级标签：</strong>偏弱（weak）" in scorecard_line
        assert "<strong>评级说明：</strong>" in scorecard_line
        assert "原始得分对应 4 星" in scorecard_line
        assert "评级上限后为 2 星" in scorecard_line
        assert "）（原始得分对应" not in scorecard_line

    def test_decision_score_explains_rating_caps(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        ic_result.n_periods = 15
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )

        assert "最终研究决策" in html
        assert "评级上限：样本期数少于 60，最高 2 星" in html

    def test_different_frequency(self, ic_result, bt_result, to_result):
        """Different frequency labels render correctly."""
        for freq in ["daily", "weekly", "monthly"]:
            html = generate_tear_sheet(
                f"test_{freq}",
                ic_result,
                bt_result,
                to_result,
                frequency=freq,
                date_range="2025-01-01 ~ 2025-05-13",
            )
            assert freq in html

    def test_different_factor_names(self, ic_result, bt_result, to_result):
        """Different factor names render correctly."""
        names = ["momentum_20d", "value_ep", "My_Custom_Factor"]
        for name in names:
            html = generate_tear_sheet(
                name,
                ic_result,
                bt_result,
                to_result,
                date_range="2025-01-01 ~ 2025-05-13",
            )
            assert name in html

    def test_with_advanced_results(self, ic_result, bt_result, to_result, advanced_results):
        """Advanced evaluation results are included."""
        html = generate_tear_sheet(
            "momentum_20d",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
            date_range="2025-01-01 ~ 2025-05-13",
        )
        assert "单调性" in html or "monotonicity_score" in html.lower()

    def test_structure_checks_explain_monotonicity_score(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        html = generate_tear_sheet(
            "momentum_20d",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )

        assert "分组单调性" in html
        assert "单调性结论" in html
        assert "强。分组收益排序清晰" in html
        assert "可重点结合 IC 稳定性、换手成本和样本外结果判断是否投产。" in html
        assert "monotonicity_score" not in html

    def test_structure_checks_warn_when_monotonicity_is_weak(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        advanced_results = dict(advanced_results)
        advanced_results["mono"] = replace(
            advanced_results["mono"], monotonicity_score=0.40
        )
        html = generate_tear_sheet(
            "momentum_20d",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )

        assert "弱。分组排序不清晰" in html
        assert "不宜只依据均值 IC 使用" in html

    def test_predictive_power_explains_ic_decay_direction(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        html = generate_tear_sheet(
            "momentum_20d",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )

        assert "IC 衰减摘要" in html
        assert "各持有期 IC 均为正，预测方向一致" in html
        assert "短端 1 天 IC=0.0320，长端 20 天 IC=0.0100" in html
        assert "<th>方向读法</th>" in html
        assert "<td>正向</td>" in html

    def test_structure_section_summarizes_best_holding_period(
        self, ic_result, bt_result, to_result
    ):
        ic_result.multi_period = {
            1: {
                "ic_mean": 0.012,
                "ic_std": 0.090,
                "ir": 0.35,
                "ic_positive_ratio": 0.55,
                "tstat": 1.75,
                "pvalue": 0.080,
            },
            5: {
                "ic_mean": 0.029,
                "ic_std": 0.070,
                "ir": 0.72,
                "ic_positive_ratio": 0.65,
                "tstat": 2.65,
                "pvalue": 0.010,
            },
            20: {
                "ic_mean": 0.018,
                "ic_std": 0.080,
                "ir": 0.40,
                "ic_positive_ratio": 0.60,
                "tstat": 2.05,
                "pvalue": 0.040,
            },
        }

        html = generate_tear_sheet("momentum_20d", ic_result, bt_result, to_result)

        assert "持有期结论" in html
        assert "5d 的综合表现最好" in html
        assert "各持有期方向一致，信号具备跨周期稳定性" in html
        assert "建议以 5d 作为优先验证的调仓周期" in html

    def test_holding_period_summary_labels_all_negative_periods_as_reverse_strength(
        self, ic_result, bt_result, to_result
    ):
        ic_result.multi_period = {
            1: {
                "ic_mean": -0.020,
                "ic_std": 0.080,
                "ir": -0.25,
                "ic_positive_ratio": 0.40,
                "tstat": -1.20,
                "pvalue": 0.200,
            },
            5: {
                "ic_mean": -0.050,
                "ic_std": 0.070,
                "ir": -0.71,
                "ic_positive_ratio": 0.20,
                "tstat": -2.70,
                "pvalue": 0.010,
            },
            20: {
                "ic_mean": -0.030,
                "ic_std": 0.080,
                "ir": -0.38,
                "ic_positive_ratio": 0.35,
                "tstat": -1.80,
                "pvalue": 0.080,
            },
        }

        html = generate_tear_sheet("momentum_20d", ic_result, bt_result, to_result)

        assert "5d 的反向信号强度最集中" in html
        assert "各持有期方向一致，但 IC 均为负，应按反向信号验证。" in html
        assert "建议以 5d 作为优先验证的反向调仓周期" in html
        assert "5d 的综合表现最好" not in html

    def test_holding_period_summary_avoids_default_zero_for_missing_stats(
        self, ic_result, bt_result, to_result
    ):
        ic_result.multi_period = {
            1: {
                "ic_mean": 0.020,
                "ic_std": None,
                "ir": None,
                "ic_positive_ratio": None,
                "tstat": None,
                "pvalue": None,
            },
            5: {
                "ic_mean": 0.030,
                "ic_std": None,
                "ir": np.nan,
                "ic_positive_ratio": np.nan,
                "tstat": None,
                "pvalue": None,
            },
        }

        html = generate_tear_sheet("momentum_20d", ic_result, bt_result, to_result)
        structure_html = html.split('<div class="panel" id="structure-checks">', 1)[1].split(
            '<div class="panel" id="tradability">', 1
        )[0]

        assert "持有期结论" in structure_html
        assert "IR 样本不足" in structure_html
        assert "胜率 样本不足" in structure_html
        assert "IR 0.00" not in structure_html
        assert "胜率 0.0%" not in structure_html
        assert ">nan<" not in html
        assert ">NaN<" not in html

    def test_predictive_power_contains_actionable_summary(
        self, ic_result, bt_result, to_result
    ):
        html = generate_tear_sheet(
            "momentum_20d",
            ic_result,
            bt_result,
            to_result,
            pearson_ic_result=ic_result,
        )

        assert "预测能力总览" in html
        assert "Rank IC 方向为正" in html
        assert "统计显著性不足" in html
        assert "Rank/Pearson 方向一致" in html
        assert "下一步应优先检查" in html

    def test_predictive_power_marks_reversed_ic_decay_direction(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        advanced_results = dict(advanced_results)
        advanced_results["decay_results"] = [
            ICDecayResult(horizon=1, ic_mean=-0.030, ic_std=0.08),
            ICDecayResult(horizon=5, ic_mean=-0.020, ic_std=0.07),
        ]
        html = generate_tear_sheet(
            "momentum_20d",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )

        assert "各持有期 IC 均为负，方向一致但更适合按反向因子理解。" in html
        assert "<td>反向</td>" in html

    def test_predictive_power_handles_missing_ic_decay_values(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        advanced_results = dict(advanced_results)
        advanced_results["decay_results"] = [
            ICDecayResult(horizon=1, ic_mean=None, ic_std=None),
            ICDecayResult(horizon=5, ic_mean=np.nan, ic_std=np.nan),
        ]
        html = generate_tear_sheet(
            "momentum_20d",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )

        predictive_html = html.split('<div class="panel" id="predictive-power">', 1)[1].split(
            '<div class="panel" id="structure-checks">', 1
        )[0]
        assert "IC 衰减摘要" in predictive_html
        assert "IC 衰减关键指标样本不足" in predictive_html
        assert "样本不足" in predictive_html
        assert "各持有期 IC 接近 0" not in predictive_html
        assert "<td>不明显</td>" not in predictive_html
        assert ">nan<" not in html
        assert ">NaN<" not in html
        assert ">None<" not in html

    def test_predictive_power_metrics_explain_missing_ic_charts(
        self, ic_result, bt_result, to_result, monkeypatch
    ):
        import factorzen.reports.tear_sheet as tear_sheet

        monkeypatch.setattr(tear_sheet, "_make_ic_chart", lambda *_: None)
        monkeypatch.setattr(tear_sheet, "_make_ic_distribution_chart", lambda *_: None)

        html = generate_tear_sheet("momentum_20d", ic_result, bt_result, to_result)
        predictive_html = html.split('<div class="panel" id="predictive-power">', 1)[1].split(
            '<div class="panel" id="structure-checks">', 1
        )[0]

        assert "预测能力总览" in predictive_html
        assert "Rank IC 方向为正" in predictive_html
        assert "IC 时序图未生成" in predictive_html
        assert "IC 分布图未生成" in predictive_html
        assert "无 IC 数据：未传入 IC 时序" not in predictive_html
        assert 'alt="IC 时序图"' not in predictive_html
        assert 'alt="IC 分布图"' not in predictive_html

    def test_structure_checks_explain_signal_persistence(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        html = generate_tear_sheet(
            "momentum_20d",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )

        assert "信号持续性" in html
        assert "持续性结论" in html
        assert "中等。因子排序有一定延续性" in html
        assert "应结合交易成本和调仓阈值决定可用频率。" in html
        assert "<th>排名自相关</th><th>半衰期</th><th>交易含义</th>" in html

    def test_structure_checks_warn_when_signal_persistence_is_low(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        advanced_results = dict(advanced_results)
        advanced_results["autocorr"] = replace(
            advanced_results["autocorr"],
            autocorr_values=[0.20],
            mean_autocorr=0.20,
            half_life_est=0.5,
            _lag_to_autocorr={1: 0.20},
        )
        html = generate_tear_sheet(
            "momentum_20d",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )

        assert "低。因子排序变化较快，信号生命周期偏短。" in html
        assert "避免纸面收益被交易成本吞噬" in html

    def test_overview_marks_missing_half_life_as_insufficient(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        advanced_results = dict(advanced_results)
        advanced_results["autocorr"] = replace(
            advanced_results["autocorr"],
            mean_autocorr=0.45,
            half_life_est=None,
        )

        html = generate_tear_sheet(
            "momentum_20d",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )

        overview_html = html.split('<div class="panel" id="overview">', 1)[1].split(
            '<div class="panel" id="returns">', 1
        )[0]
        assert "排名自相关：0.450 | 半衰期：样本不足" in overview_html
        assert "半衰期：0.0 期" not in overview_html

    def test_overview_marks_nan_half_life_without_unit_suffix(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        advanced_results = dict(advanced_results)
        advanced_results["autocorr"] = replace(
            advanced_results["autocorr"],
            mean_autocorr=0.45,
            half_life_est=np.nan,
        )

        html = generate_tear_sheet(
            "momentum_20d",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )
        overview_html = html.split('<div class="panel" id="overview">', 1)[1].split(
            '<div class="panel" id="returns">', 1
        )[0]

        assert "排名自相关：0.450 | 半衰期：样本不足" in overview_html
        assert "样本不足 期" not in overview_html
        assert ">nan<" not in html
        assert ">NaN<" not in html

    def test_overview_explains_nan_rank_autocorr_as_insufficient(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        advanced_results = dict(advanced_results)
        advanced_results["autocorr"] = replace(
            advanced_results["autocorr"],
            autocorr_values=[np.nan],
            mean_autocorr=np.nan,
            half_life_est=np.nan,
            _lag_to_autocorr={1: np.nan},
        )

        html = generate_tear_sheet(
            "momentum_20d",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )
        overview_html = html.split('<div class="panel" id="overview">', 1)[1].split(
            '<div class="panel" id="returns">', 1
        )[0]

        assert "排名自相关样本不足，暂不判断信号持续性" in overview_html
        assert "排名自相关：样本不足 | 半衰期：样本不足" not in overview_html
        assert ">nan<" not in html
        assert ">NaN<" not in html

    def test_structure_checks_handle_nan_monotonicity_and_autocorr(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        advanced_results = dict(advanced_results)
        advanced_results["mono"] = replace(
            advanced_results["mono"], monotonicity_score=np.nan
        )
        advanced_results["autocorr"] = replace(
            advanced_results["autocorr"],
            autocorr_values=[np.nan],
            mean_autocorr=np.nan,
            half_life_est=np.nan,
            _lag_to_autocorr={1: np.nan},
        )

        html = generate_tear_sheet(
            "momentum_20d",
            ic_result,
            bt_result,
            to_result,
            advanced_results=advanced_results,
        )
        structure_html = html.split('<div class="panel" id="structure-checks">', 1)[1].split(
            '<div class="panel" id="tradability">', 1
        )[0]

        assert "单调性指标样本不足" in structure_html
        assert "持续性指标样本不足" in structure_html
        assert "样本不足" in structure_html
        assert "弱。分组排序不清晰" not in structure_html
        assert "低。因子排序变化较快" not in structure_html
        assert ">nan<" not in html
        assert ">NaN<" not in html

    def test_structure_checks_do_not_show_empty_state_when_only_autocorr_exists(
        self, ic_result, bt_result, to_result, advanced_results
    ):
        html = generate_tear_sheet(
            "momentum_20d",
            ic_result,
            bt_result,
            to_result,
            advanced_results={"autocorr": advanced_results["autocorr"]},
        )

        structure_html = html.split('<div class="panel" id="structure-checks">', 1)[1].split(
            '<div class="panel" id="tradability">', 1
        )[0]
        assert "信号持续性" in structure_html
        assert "无结构检验数据" not in structure_html

    def test_html_non_empty(self, ic_result, bt_result, to_result):
        """Report HTML is non-empty."""
        html = generate_tear_sheet("test_factor", ic_result, bt_result, to_result)
        assert html is not None
        assert len(html) > 0

    def test_html_contains_benchmark_section(self, ic_result, bt_result, to_result):
        """Benchmark section is included when benchmark_result is passed."""
        from factorzen.daily.evaluation.benchmark import BenchmarkResult

        dates = [f"2025-{(i // 20 + 1):02d}-{(i % 20 + 1):02d}" for i in range(60)]
        rng = np.random.default_rng(55)
        rets = rng.normal(0.001, 0.01, 60)
        bench_rets = rng.normal(0.0005, 0.01, 60)
        excess_rets = rets - bench_rets

        daily_df = pl.DataFrame(
            {
                "trade_date": pl.Series(dates).str.strptime(pl.Date, "%Y-%m-%d"),
                "strategy_ret": rets,
                "benchmark_ret": bench_rets,
                "excess_ret": excess_rets,
                "strategy_nav": np.cumprod(1 + rets),
                "benchmark_nav": np.cumprod(1 + bench_rets),
                "excess_nav": np.cumprod(1 + excess_rets),
            }
        )

        bm = BenchmarkResult(
            benchmark_code="000300.SH",
            benchmark_name="HS300",
            daily=daily_df,
            ann_excess_ret=0.05,
            tracking_error=0.08,
            information_ratio=0.625,
            excess_max_dd=-0.03,
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            benchmark_result=bm,
        )
        assert "基准对比" in html
        assert "HS300" in html
        assert "基准相对结论" in html
        assert "跑赢基准" in html
        assert "单位主动风险带来的超额收益较好" in html

    def test_benchmark_section_marks_missing_information_ratio_as_insufficient(
        self, ic_result, bt_result, to_result
    ):
        from factorzen.daily.evaluation.benchmark import BenchmarkResult

        daily_df = pl.DataFrame(
            {
                "trade_date": pl.Series(_make_dates()).str.strptime(pl.Date, "%Y-%m-%d"),
                "strategy_ret": [0.0] * 60,
                "benchmark_ret": [0.0] * 60,
                "excess_ret": [0.0] * 60,
                "strategy_nav": [1.0] * 60,
                "benchmark_nav": [1.0] * 60,
                "excess_nav": [1.0] * 60,
            }
        )
        benchmark = BenchmarkResult(
            benchmark_code="000300.SH",
            benchmark_name="HS300",
            daily=daily_df,
            ann_excess_ret=0.02,
            tracking_error=0.0,
            information_ratio=None,
            excess_max_dd=None,
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            benchmark_result=benchmark,
        )

        assert "基准相对结论" in html
        assert "信息比率样本不足，暂不判断超额效率。" in html
        benchmark_html = html.split("基准对比 - HS300", 1)[1].split(
            '<div class="panel" id="predictive-power">', 1
        )[0]
        assert '<div class="label">信息比率 (IR)</div>' in benchmark_html
        assert '<div class="value">样本不足</div>' in benchmark_html
        assert ">None<" not in html

    def test_benchmark_summary_renders_without_nav_series(
        self, ic_result, bt_result, to_result
    ):
        from factorzen.daily.evaluation.benchmark import BenchmarkResult

        benchmark = BenchmarkResult(
            benchmark_code="000300.SH",
            benchmark_name="HS300",
            daily=pl.DataFrame(),
            ann_excess_ret=0.05,
            tracking_error=0.08,
            information_ratio=0.625,
            excess_max_dd=-0.03,
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            benchmark_result=benchmark,
        )
        returns_html = html.split('<div class="panel" id="returns">', 1)[1].split(
            '<div class="panel" id="predictive-power">', 1
        )[0]
        appendix_html = html.split('<div class="panel" id="appendix">', 1)[1]

        assert "基准对比 - HS300" in returns_html
        assert "基准相对结论" in returns_html
        assert "跑赢基准" in returns_html
        assert "基准图未生成" in returns_html
        assert 'alt="基准对比图"' not in returns_html
        assert "<td>基准超额</td>" in appendix_html
        assert "<td>需关注</td>" in appendix_html
        assert "已生成基准摘要，但缺少可绘图的日度净值明细" in appendix_html

    def test_html_contains_attribution_section(self, ic_result, bt_result, to_result):
        """Attribution section is included when attribution_result is passed."""
        from factorzen.daily.evaluation.attribution import BrinsonResult

        sector_df = pl.DataFrame(
            {
                "sector": ["Tech", "Finance"],
                "allocation": [0.01, -0.005],
                "selection": [0.02, 0.01],
                "interaction": [0.001, -0.002],
                "total_contribution": [0.031, 0.003],
            }
        )
        period_df = pl.DataFrame(
            {
                "trade_date": ["2025-01-01", "2025-01-02"],
                "allocation": [0.005, -0.002],
                "selection": [0.01, 0.005],
                "interaction": [0.0005, -0.001],
                "active_ret": [0.0155, 0.002],
            }
        )
        brinson = BrinsonResult(
            sector_df=sector_df,
            period_df=period_df,
            ann_allocation=0.02,
            ann_selection=0.05,
            ann_interaction=-0.005,
            ann_active_return=0.065,
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            attribution_result={"brinson": brinson},
        )
        assert "风格/行业归因" in html or "attribution" in html.lower()
        assert "归因结论" in html
        assert "主要正贡献来自选股贡献" in html
        assert "最大正贡献来自 Tech" in html
        assert "最大拖累来自 Finance" in html

    def test_attribution_summary_renders_without_plot_data(
        self, ic_result, bt_result, to_result
    ):
        from factorzen.daily.evaluation.attribution import BrinsonResult

        brinson = BrinsonResult(
            sector_df=pl.DataFrame(),
            period_df=pl.DataFrame(),
            ann_allocation=0.01,
            ann_selection=0.04,
            ann_interaction=-0.005,
            ann_active_return=0.045,
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            attribution_result={"brinson": brinson},
        )
        risk_html = html.split('<div class="panel" id="risk-attribution">', 1)[1].split(
            '<div class="panel" id="appendix">', 1
        )[0]
        appendix_html = html.split('<div class="panel" id="appendix">', 1)[1]

        assert "风格/行业归因" in risk_html
        assert "归因结论" in risk_html
        assert "Brinson 口径下整体贡献为正" in risk_html
        assert "主要正贡献来自选股贡献" in risk_html
        assert "归因图未生成" in risk_html
        assert 'alt="归因分析图"' not in risk_html
        assert "<td>组合归因（Brinson/Barra）</td>" in appendix_html
        assert "<td>需关注</td>" in appendix_html
        assert "已生成归因摘要，但缺少可绘图的行业或风格明细" in appendix_html

    def test_html_contains_walk_forward_insufficient_state(
        self, ic_result, bt_result, to_result
    ):
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            walk_forward_summary={"status": "insufficient_data", "n_folds": 0},
        )
        assert "滚动样本外稳健性" in html
        assert "样本不足" in html
        assert "下一步：扩大回测区间" in html
        assert "缩短历史观察期或样本外验证期参数" in html
        assert "隔离期（embargo）" in html
        assert "防止历史观察期与样本外验证期信息泄漏" in html

    def test_html_uses_observation_window_terms(self, ic_result, bt_result, to_result):
        ic_result.oos_ic = {"train": 0.03, "test": 0.02}
        html = generate_tear_sheet("test_factor", ic_result, bt_result, to_result)

        assert "历史观察期" in html
        assert "样本外验证期" in html
        assert "未来验证期" not in html
        assert "训练集" not in html

    def test_oos_robustness_section_explains_direction_retention(
        self, ic_result, bt_result, to_result
    ):
        ic_result.oos_ic = {"train": 0.030, "test": 0.024}
        html = generate_tear_sheet("test_factor", ic_result, bt_result, to_result)

        assert "样本外摘要" in html
        assert "历史观察期与样本外验证期 IC 方向一致" in html
        assert "样本外验证期强度约为历史观察期的 80.0%" in html
        assert "样本外强度保留较好" in html

    def test_oos_robustness_section_warns_on_direction_reversal(
        self, ic_result, bt_result, to_result
    ):
        ic_result.oos_ic = {"train": 0.030, "test": -0.012}
        html = generate_tear_sheet("test_factor", ic_result, bt_result, to_result)

        assert "历史观察期与样本外验证期 IC 方向反转" in html
        assert "样本外稳定性需要重点复核" in html
        assert "样本外强度衰减明显" in html

    def test_oos_robustness_ignores_nan_split_metrics(
        self, ic_result, bt_result, to_result
    ):
        ic_result.oos_ic = {"train": np.nan, "test": np.nan}

        html = generate_tear_sheet("test_factor", ic_result, bt_result, to_result)
        robustness_html = html.split('<div class="panel" id="robustness">', 1)[1].split(
            '<div class="panel" id="risk-attribution">', 1
        )[0]

        assert ">样本外稳健性</h3>" not in robustness_html
        assert "前 70% 历史观察期与后 30% 样本外验证期的 IC 分割" not in robustness_html
        assert ">nan<" not in html
        assert ">NaN<" not in html

    def test_oos_robustness_explains_partial_split_metrics(
        self, ic_result, bt_result, to_result
    ):
        ic_result.oos_ic = {"train": 0.030, "test": np.nan}

        html = generate_tear_sheet("test_factor", ic_result, bt_result, to_result)
        robustness_html = html.split('<div class="panel" id="robustness">', 1)[1].split(
            '<div class="panel" id="risk-attribution">', 1
        )[0]
        appendix_html = html.split('<div class="panel" id="appendix">', 1)[1]

        assert "样本外 IC 分割不完整" in robustness_html
        assert "已计算历史观察期 IC，缺少样本外验证期 IC" in robustness_html
        assert "历史观察期与样本外验证期 IC 方向一致" not in robustness_html
        assert "<td>样本外 IC 分割</td>" in appendix_html
        assert "<td>样本不足</td>" in appendix_html
        assert "样本外 IC 分割不完整：已计算历史观察期 IC，缺少样本外验证期 IC。" in appendix_html
        assert ">nan<" not in html
        assert ">NaN<" not in html

    def test_oos_robustness_section_labels_stronger_future_window(
        self, ic_result, bt_result, to_result
    ):
        ic_result.oos_ic = {"train": -0.020, "test": -0.050}
        html = generate_tear_sheet("test_factor", ic_result, bt_result, to_result)

        assert "样本外验证期强度约为历史观察期的 250.0%" in html
        assert "样本外验证期强度高于历史观察期，但短样本下需警惕偶然放大。" in html

    def test_robustness_section_explains_market_regime_ic(
        self, ic_result, bt_result, to_result
    ):
        regime = MarketRegimeICResult(
            factor_name="test_factor",
            regime_type="direction",
            regime_ic=pl.DataFrame(
                {
                    "regime": ["up", "down"],
                    "ic": [0.042, -0.018],
                }
            ),
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results={"regime": regime},
        )

        assert "市场状态适应性" in html
        assert "市场状态结论" in html
        assert "上涨市场 IC 为正，下跌市场 IC 为负" in html
        assert "上涨市场 IC 为正，下跌市场 IC 为负。该信号可能依赖市场方向" in html
        assert "。 该信号" not in html
        assert "该信号可能依赖市场方向" in html
        assert "上涨" in html
        assert "下跌" in html
        assert "市场状态 IC" in html
        assert "已生成市场状态 IC，用于检查因子在不同市场环境下的方向稳定性。" in html

    def test_regime_ic_section_handles_missing_ic_values(
        self, ic_result, bt_result, to_result
    ):
        regime = MarketRegimeICResult(
            factor_name="test_factor",
            regime_type="direction",
            regime_ic=pl.DataFrame(
                {
                    "regime": ["up", "down"],
                    "ic": [None, np.nan],
                },
                schema={"regime": pl.Utf8, "ic": pl.Float64},
            ),
        )

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            advanced_results={"regime": regime},
        )
        robustness_html = html.split('<div class="panel" id="robustness">', 1)[1].split(
            '<div class="panel" id="risk-attribution">', 1
        )[0]

        assert "市场状态适应性" in robustness_html
        assert "市场状态结论" not in robustness_html
        assert "样本不足" in robustness_html
        assert ">nan<" not in html
        assert ">NaN<" not in html
        assert ">None<" not in html

    def test_walk_forward_section_explains_stability_ratio(
        self, ic_result, bt_result, to_result
    ):
        wf = WalkForwardResult(
            folds=[
                WalkForwardFoldResult(
                    fold_id=1,
                    train_start="2025-01-01",
                    train_end="2025-03-31",
                    test_start="2025-04-01",
                    test_end="2025-04-30",
                    is_sharpe=1.20,
                    oos_sharpe=0.72,
                    oos_ann_ret=0.08,
                    oos_max_dd=-0.04,
                )
            ],
            oos_returns=pl.DataFrame(),
            is_sharpe_mean=1.20,
            oos_sharpe_mean=0.72,
            oos_sharpe_std=0.0,
            oos_max_dd=-0.04,
            stability_ratio=0.60,
        )
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            walk_forward_result=wf,
        )

        assert "滚动验证摘要" in html
        assert "样本外验证期 Sharpe 为正且稳定性比率较高" in html
        assert "稳定性比率 = 样本外验证期 Sharpe 均值 / 历史观察期 Sharpe 均值" in html

    def test_walk_forward_section_handles_missing_metrics(
        self, ic_result, bt_result, to_result
    ):
        wf = WalkForwardResult(
            folds=[
                WalkForwardFoldResult(
                    fold_id=1,
                    train_start="2025-01-01",
                    train_end="2025-03-31",
                    test_start="2025-04-01",
                    test_end="2025-04-30",
                    is_sharpe=np.nan,
                    oos_sharpe=None,
                    oos_ann_ret=np.nan,
                    oos_max_dd=None,
                )
            ],
            oos_returns=pl.DataFrame(),
            is_sharpe_mean=np.nan,
            oos_sharpe_mean=np.nan,
            oos_sharpe_std=np.nan,
            oos_max_dd=np.nan,
            stability_ratio=np.nan,
        )
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            walk_forward_result=wf,
        )

        assert "滚动验证摘要" in html
        assert "滚动样本外关键指标样本不足" in html
        assert "样本不足" in html
        assert ">nan<" not in html
        assert ">NaN<" not in html
        assert ">None<" not in html


    def test_html_contains_neutralized_ic_section(self, ic_result, bt_result, to_result):
        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            neutralized_ic_result=ic_result,
        )

        assert "中性化 IC" in html
        assert "中性化" in html
        assert "中性化结论" in html
        assert "中性化后保留率" in html
        assert "方向保持一致" in html
        assert "行业/市值暴露不能单独解释该信号" in html

    def test_neutralized_summary_requires_valid_raw_ic(
        self, ic_result, bt_result, to_result
    ):
        ic_result.ic_mean = np.nan
        neutralized = replace(ic_result, ic_mean=0.020)

        html = generate_tear_sheet(
            "test_factor",
            ic_result,
            bt_result,
            to_result,
            neutralized_ic_result=neutralized,
        )
        structure_html = html.split('<div class="panel" id="structure-checks">', 1)[1].split(
            '<div class="panel" id="tradability">', 1
        )[0]

        assert "原始 Rank IC 样本不足" in structure_html
        assert "中性化后保留率 0.0%" not in structure_html
        assert ">nan<" not in html
        assert ">NaN<" not in html


class TestTearSheetImports:
    """Module import and template file tests."""

    def test_tear_sheet_import(self):
        from factorzen.reports.tear_sheet import generate_tear_sheet

        assert callable(generate_tear_sheet)

    def test_template_dir_exists(self):
        template_dir = (
            Path(__file__).resolve().parent.parent / "src" / "factorzen" / "reports" / "templates"
        )
        assert template_dir.is_dir()


class TestFactorRating:
    def test_strong_factor_gets_candidate_or_better_rating(self):
        metrics = {
            "ic_mean": 0.045,
            "ir": 0.65,
            "ic_positive_ratio": 0.63,
            "ic_tstat": 3.1,
            "ic_pvalue": 0.01,
            "n_periods": 180,
            "oos_train_ic": 0.04,
            "oos_test_ic": 0.035,
            "ls_ann_ret": 0.12,
            "ls_sharpe": 1.4,
            "ls_max_dd": -0.08,
            "avg_turnover": 0.25,
            "neutralized_ic_mean": 0.034,
            "pearson_ic_mean": 0.04,
            "monotonicity_score": 0.85,
            "rank_autocorr": 0.65,
            "decay_table": [
                {"horizon": 1, "ic_mean": 0.045, "ic_std": 0.08},
                {"horizon": 5, "ic_mean": 0.035, "ic_std": 0.07},
                {"horizon": 20, "ic_mean": 0.02, "ic_std": 0.06},
            ],
            "walk_forward_oos_sharpe_mean": 0.9,
            "walk_forward_stability_ratio": 0.8,
        }

        rating = _compute_factor_rating(metrics)

        assert rating.stars >= 4
        assert rating.label in {"candidate", "production_watch"}
        assert rating.components["Alpha 强度"] > 20

    def test_missing_oos_caps_rating_at_three_stars(self):
        metrics = {
            "ic_mean": 0.06,
            "ir": 0.8,
            "ic_positive_ratio": 0.7,
            "ic_tstat": 4.0,
            "ic_pvalue": 0.001,
            "n_periods": 180,
            "ls_ann_ret": 0.18,
            "ls_sharpe": 2.0,
            "ls_max_dd": -0.05,
            "avg_turnover": 0.2,
            "neutralized_ic_mean": 0.055,
            "monotonicity_score": 0.9,
            "rank_autocorr": 0.8,
        }

        rating = _compute_factor_rating(metrics)

        assert rating.stars == 3
        assert "缺少样本外验证期 IC，最高 3 星" in rating.caps

    def test_nan_oos_ic_caps_rating_like_missing_oos(self):
        metrics = {
            "ic_mean": 0.06,
            "ir": 0.8,
            "ic_positive_ratio": 0.7,
            "ic_tstat": 4.0,
            "ic_pvalue": 0.001,
            "n_periods": 180,
            "oos_train_ic": 0.04,
            "oos_test_ic": np.nan,
            "ls_ann_ret": 0.18,
            "ls_sharpe": 2.0,
            "ls_max_dd": -0.05,
            "avg_turnover": 0.2,
            "neutralized_ic_mean": 0.055,
            "monotonicity_score": 0.9,
            "rank_autocorr": 0.8,
        }

        rating = _compute_factor_rating(metrics)

        assert rating.stars == 3
        assert "缺少样本外验证期 IC，最高 3 星" in rating.caps

    def test_neutralized_ic_failure_caps_rating_at_three_stars(self):
        metrics = {
            "ic_mean": 0.05,
            "ir": 0.7,
            "ic_positive_ratio": 0.65,
            "ic_tstat": 3.0,
            "ic_pvalue": 0.01,
            "n_periods": 180,
            "oos_train_ic": 0.04,
            "oos_test_ic": 0.035,
            "ls_ann_ret": 0.12,
            "ls_sharpe": 1.5,
            "ls_max_dd": -0.08,
            "avg_turnover": 0.25,
            "neutralized_ic_mean": 0.01,
            "monotonicity_score": 0.8,
            "rank_autocorr": 0.6,
        }

        rating = _compute_factor_rating(metrics)

        assert rating.stars <= 3
        assert "中性化后 IC 保留不足 50%，最高 3 星" in rating.caps

    def test_nan_structure_metrics_are_scored_like_missing_metrics(self):
        base_metrics = {
            "ic_mean": 0.05,
            "ir": 0.7,
            "ic_positive_ratio": 0.65,
            "ic_tstat": 3.0,
            "ic_pvalue": 0.01,
            "n_periods": 180,
            "oos_train_ic": 0.04,
            "oos_test_ic": 0.035,
            "ls_ann_ret": 0.12,
            "ls_sharpe": 1.5,
            "ls_max_dd": -0.08,
            "avg_turnover": 0.25,
            "neutralized_ic_mean": 0.04,
        }
        missing_rating = _compute_factor_rating(base_metrics)
        nan_rating = _compute_factor_rating(
            {
                **base_metrics,
                "monotonicity_score": np.nan,
                "rank_autocorr": np.nan,
            }
        )

        assert nan_rating.components["结构质量"] == missing_rating.components["结构质量"]

    def test_nan_walk_forward_metrics_are_scored_like_missing_metrics(self):
        base_metrics = {
            "ic_mean": 0.05,
            "ir": 0.7,
            "ic_positive_ratio": 0.65,
            "ic_tstat": 3.0,
            "ic_pvalue": 0.01,
            "n_periods": 180,
            "oos_train_ic": 0.04,
            "oos_test_ic": 0.035,
            "ls_ann_ret": 0.12,
            "ls_sharpe": 1.5,
            "ls_max_dd": -0.08,
            "avg_turnover": 0.25,
            "neutralized_ic_mean": 0.04,
            "monotonicity_score": 0.8,
            "rank_autocorr": 0.6,
        }
        missing_rating = _compute_factor_rating(base_metrics)
        nan_rating = _compute_factor_rating(
            {
                **base_metrics,
                "walk_forward_oos_sharpe_mean": np.nan,
                "walk_forward_stability_ratio": np.nan,
            }
        )

        assert nan_rating.components["稳定性"] == missing_rating.components["稳定性"]

    def test_nan_long_short_metrics_cap_rating_like_missing_backtest(self):
        metrics = {
            "ic_mean": 0.06,
            "ir": 0.8,
            "ic_positive_ratio": 0.7,
            "ic_tstat": 4.0,
            "ic_pvalue": 0.001,
            "n_periods": 180,
            "oos_train_ic": 0.04,
            "oos_test_ic": 0.035,
            "ls_ann_ret": np.nan,
            "ls_sharpe": np.nan,
            "ls_max_dd": np.nan,
            "avg_turnover": 0.2,
            "neutralized_ic_mean": 0.055,
            "monotonicity_score": 0.9,
            "rank_autocorr": 0.8,
        }

        rating = _compute_factor_rating(metrics)

        assert rating.stars == 3
        assert "缺少多空回测绩效，最高 3 星" in rating.caps

    def test_nan_robustness_metrics_are_scored_like_missing_metrics(self):
        base_metrics = {
            "ic_mean": 0.05,
            "ir": 0.7,
            "ic_positive_ratio": 0.65,
            "ic_tstat": 3.0,
            "ic_pvalue": 0.01,
            "n_periods": 180,
            "oos_train_ic": 0.04,
            "oos_test_ic": 0.035,
            "ls_ann_ret": 0.12,
            "ls_sharpe": 1.5,
            "ls_max_dd": -0.08,
            "avg_turnover": 0.25,
            "monotonicity_score": 0.8,
            "rank_autocorr": 0.6,
        }
        missing_rating = _compute_factor_rating(base_metrics)
        nan_rating = _compute_factor_rating(
            {
                **base_metrics,
                "neutralized_ic_mean": np.nan,
                "pearson_ic_mean": np.nan,
            }
        )

        assert nan_rating.components["鲁棒性"] == missing_rating.components["鲁棒性"]
        assert "中性化后 IC 保留不足 50%，最高 3 星" not in nan_rating.caps

    def test_nan_decay_and_multi_period_are_scored_like_missing_metrics(self):
        base_metrics = {
            "ic_mean": 0.05,
            "ir": 0.7,
            "ic_positive_ratio": 0.65,
            "ic_tstat": 3.0,
            "ic_pvalue": 0.01,
            "n_periods": 180,
            "oos_train_ic": 0.04,
            "oos_test_ic": 0.035,
            "ls_ann_ret": 0.12,
            "ls_sharpe": 1.5,
            "ls_max_dd": -0.08,
            "avg_turnover": 0.25,
            "monotonicity_score": 0.8,
            "rank_autocorr": 0.6,
        }
        missing_rating = _compute_factor_rating(base_metrics)
        nan_rating = _compute_factor_rating(
            {
                **base_metrics,
                "decay_table": [
                    {"horizon": 1, "ic_mean": np.nan, "ic_std": np.nan},
                    {"horizon": 5, "ic_mean": np.nan, "ic_std": np.nan},
                ],
                "multi_period_table": [
                    {"horizon": 1, "ic_mean": np.nan},
                    {"horizon": 5, "ic_mean": np.nan},
                ],
            }
        )

        assert nan_rating.components["稳定性"] == missing_rating.components["稳定性"]

    def test_template_file_exists(self):
        template_file = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "factorzen"
            / "reports"
            / "templates"
            / "tear_sheet.html"
        )
        assert template_file.is_file()

    def test_intraday_template_uses_research_report_semantics(self):
        template_file = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "factorzen"
            / "reports"
            / "templates"
            / "intraday_ic.html"
        )
        text = template_file.read_text(encoding="utf-8")

        for marker in (
            "报告目录",
            "综合结论",
            "日度 IC 趋势",
            "分时段 IC",
            "附录",
            "dashboard-grid",
            "status-callout",
            "definition-grid",
            "未生成日度 IC 图",
            "分钟报告用于验证日频因子的盘中一致性",
            "阅读口径",
            "研究边界",
            "指标口径说明",
            "分时段 IC",
            "信息比率 (IR)",
            "分钟粒度",
            "信号方向",
            "短样本提示",
            "阅读含义",
            "border: 1px dashed #cbd5e1",
        ):
            assert marker in text

        for mojibake in ("鈥", "鍒", "鎶", "绋", "涓"):
            assert mojibake not in text

    def test_intraday_template_renders_reader_facing_segment_labels(self):
        template_dir = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "factorzen"
            / "reports"
            / "templates"
        )
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir), autoescape=True)
        template = env.get_template("intraday_ic.html")

        html = template.render(
            factor_name="vwap_deviation",
            bar_size="1min",
            universe="mft_default",
            generated_at="2026-06-01 09:30",
            date_range="20260101 ~ 20260131",
            charts={"daily_ic_chart": None},
            metrics={
                "n_periods": 20,
                "ic_positive_ratio": 0.55,
                "ic_mean": -0.02,
                "ic_std": 0.10,
                "ir": -0.20,
                "daily_ic_table": [],
                "segment_table": [
                    {"segment": "open", "ic": -0.03},
                    {"segment": "midday", "ic": -0.01},
                    {"segment": "close", "ic": -0.02},
                ],
            },
        )

        for marker in (
            "分钟粒度",
            "信号方向",
            "盘中研究决策",
            "反向信号观察",
            "关键缺口",
            "样本 Bar 少于 60",
            "下一步",
            "负向",
            "短样本提示",
            "开盘段",
            "中盘段",
            "收盘段",
            "盘中一致性摘要",
            "各分时段 IC 均为负",
            "绝对 IC 最大的时段是开盘段",
            "<th>信号方向</th>",
            "检验信号是否依赖开盘定价和隔夜信息释放",
        ):
            assert marker in html

        for raw_visible in (
            "<td>open</td>",
            "<td>midday</td>",
            "<td>close</td>",
            "daily_ic_chart",
            "segment_table",
            "Bar Size",
        ):
            assert raw_visible not in html

    def test_intraday_template_no_data_blocks_include_next_actions(self):
        template_dir = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "factorzen"
            / "reports"
            / "templates"
        )
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir), autoescape=True)
        template = env.get_template("intraday_ic.html")

        html = template.render(
            factor_name="vwap_deviation",
            bar_size="1min",
            universe="mft_default",
            generated_at="2026-06-01 09:30",
            date_range="20260101 ~ 20260131",
            charts={"daily_ic_chart": None},
            metrics={
                "n_periods": 0,
                "ic_positive_ratio": None,
                "ic_mean": None,
                "ic_std": None,
                "ir": None,
                "daily_ic_table": [],
                "segment_table": [],
            },
        )

        assert "指标未计算" in html
        assert "盘中 IC 尚未计算" in html
        assert "IC=样本不足 | IR=样本不足" in html
        assert "盘中 IC 接近零" not in html
        assert "0.0000" not in html
        assert "0.00" not in html
        assert "下一步：检查分钟数据覆盖率和日度聚合逻辑" in html
        assert "下一步：确认交易时段切分规则和分钟 Bar 时间戳" in html

    @pytest.mark.xfail(
        reason="intraday 报告 UI 已冻结（见 docs/evolution-plan-2026）；"
        "附录'研究边界'模块状态行属未实现的 intraday 重设计，日频聚焦期不构建",
        strict=False,
    )
    def test_intraday_template_explains_missing_daily_ic_chart_with_table(self):
        template_dir = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "factorzen"
            / "reports"
            / "templates"
        )
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir), autoescape=True)
        template = env.get_template("intraday_ic.html")

        html = template.render(
            factor_name="vwap_deviation",
            bar_size="1min",
            universe="mft_default",
            generated_at="2026-06-01 09:30",
            date_range="20260101 ~ 20260131",
            charts={"daily_ic_chart": None},
            metrics={
                "n_periods": 80,
                "ic_positive_ratio": 0.62,
                "ic_mean": 0.03,
                "ic_std": 0.10,
                "ir": 0.30,
                "daily_ic_table": [
                    {"trade_date": "2026-01-02", "ic": 0.04},
                    {"trade_date": "2026-01-03", "ic": 0.02},
                ],
                "segment_table": [
                    {"segment": "open", "ic": 0.03},
                    {"segment": "midday", "ic": 0.02},
                    {"segment": "close", "ic": 0.01},
                ],
            },
        )
        daily_html = html.split('<div class="panel" id="daily-ic">', 1)[1].split(
            '<div class="panel" id="segment-ic">', 1
        )[0]
        overview_html = html.split('<div class="panel" id="overview">', 1)[1].split(
            '<div class="panel" id="daily-ic">', 1
        )[0]
        appendix_html = html.split('<div class="panel" id="appendix">', 1)[1]

        assert "日度 IC 表已生成，但趋势图缺失" in overview_html
        assert "未生成日度 IC 趋势，尚不能判断跨日期稳定性。" not in overview_html
        assert "日度 IC 趋势图未生成" in daily_html
        assert "已生成日度 IC 表" in daily_html
        assert "2026-01-02" in daily_html
        assert "未生成日度 IC 图：可能是样本为空" not in daily_html
        assert 'alt="日度 IC 趋势图"' not in daily_html
        assert "模块状态" in appendix_html
        assert "<td>日度 IC 趋势</td>" in appendix_html
        assert "<td>需关注</td>" in appendix_html
        assert "已生成日度 IC 表，但缺少趋势图" in appendix_html
        assert "<td>分时段 IC</td>" in appendix_html
        assert "已生成开盘、中盘、收盘分时段 IC" in appendix_html
        assert "<td>交易可行性边界</td>" in appendix_html
        assert "<td>研究边界</td>" in appendix_html
        assert "<td>交易可行性边界</td>\n        <td>未生成</td>" not in appendix_html

    def test_intraday_template_does_not_overstate_partial_segment_coverage(self):
        template_dir = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "factorzen"
            / "reports"
            / "templates"
        )
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir), autoescape=True)
        template = env.get_template("intraday_ic.html")

        html = template.render(
            factor_name="vwap_deviation",
            bar_size="1min",
            universe="mft_default",
            generated_at="2026-06-01 09:30",
            date_range="20260101 ~ 20260131",
            charts={"daily_ic_chart": None},
            metrics={
                "n_periods": 20,
                "ic_positive_ratio": 0.55,
                "ic_mean": 0.02,
                "ic_std": 0.10,
                "ir": 0.20,
                "daily_ic_table": [],
                "segment_table": [{"segment": "open", "ic": 0.03}],
            },
        )

        assert "仅生成 1 个分时段 IC，缺少中盘段、收盘段" in html
        assert "已生成开盘、中盘、收盘分时段 IC" not in html

    def test_intraday_template_replaces_nan_metrics_with_sample_notice(self):
        template_dir = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "factorzen"
            / "reports"
            / "templates"
        )
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir), autoescape=True)
        template = env.get_template("intraday_ic.html")

        html = template.render(
            factor_name="vwap_deviation",
            bar_size="1min",
            universe="mft_default",
            generated_at="2026-06-01 09:30",
            date_range="20260101 ~ 20260131",
            charts={"daily_ic_chart": None},
            metrics={
                "n_periods": 20,
                "ic_positive_ratio": np.nan,
                "ic_mean": np.nan,
                "ic_std": np.nan,
                "ir": np.nan,
                "daily_ic_table": [{"trade_date": "2026-01-02", "ic": np.nan}],
                "segment_table": [{"segment": "open", "ic": np.nan}],
            },
        )

        assert "样本不足" in html
        assert ">nan<" not in html
        assert ">NaN<" not in html
        assert "nan%" not in html
        assert "盘中 IC 尚未计算" in html
        assert "有效分时段 IC 样本不足，暂不比较最强时段" in html
        assert "下一步：检查分时段样本数量、交易时段切分规则和分钟 Bar 时间戳" in html
        assert "绝对 IC 最大的时段是" not in html

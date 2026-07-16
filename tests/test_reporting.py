"""单因子 Tear Sheet 极简单页报告测试。"""

from __future__ import annotations

import types
from datetime import date, timedelta
from typing import Any

import polars as pl

from factorzen.daily.evaluation.ic_analysis import ICAnalysisResult
from factorzen.reports.tear_sheet import generate_tear_sheet

# ── fixtures / builders ──────────────────────────────────────────────────────


def _ic_series(n: int = 80, start: date = date(2023, 1, 3), ic_val: float = 0.03) -> pl.DataFrame:
    dates = [start + timedelta(days=i) for i in range(n)]
    return pl.DataFrame({"trade_date": dates, "ic": [ic_val + 0.001 * (i % 5) for i in range(n)]})


def make_ic_result(
    *,
    ic_mean: float = 0.0350,
    ic_std: float = 0.0800,
    ir: float = 0.44,
    ic_positive_ratio: float = 0.62,
    n_periods: int = 80,
    ic_tstat: float = 3.50,
    ic_pvalue: float = 0.0005,
    ic_series: pl.DataFrame | None = None,
    multi_period: dict[int, dict[str, float]] | None = None,
    decay: dict[int, float] | None = None,
    empty_series: bool = False,
) -> ICAnalysisResult:
    if empty_series:
        series = pl.DataFrame({"trade_date": pl.Series([], dtype=pl.Date), "ic": pl.Series([], dtype=pl.Float64)})
    elif ic_series is not None:
        series = ic_series
    else:
        series = _ic_series(n=max(n_periods, 5))

    if multi_period is None and not empty_series:
        multi_period = {
            1: {"ic_mean": 0.0350, "ic_std": 0.08, "ir": 0.44, "ic_positive_ratio": 0.62, "tstat": 3.5, "pvalue": 0.0005},
            5: {"ic_mean": 0.0280, "ic_std": 0.09, "ir": 0.31, "ic_positive_ratio": 0.58, "tstat": 2.4, "pvalue": 0.016},
            10: {"ic_mean": 0.0200, "ic_std": 0.10, "ir": 0.20, "ic_positive_ratio": 0.55, "tstat": 1.6, "pvalue": 0.11},
            20: {"ic_mean": 0.0120, "ic_std": 0.11, "ir": 0.11, "ic_positive_ratio": 0.52, "tstat": 0.9, "pvalue": 0.37},
        }
    if decay is None and multi_period:
        decay = {h: float(v["ic_mean"]) for h, v in multi_period.items()}

    return ICAnalysisResult(
        factor_name="test_factor",
        ic_mean=ic_mean,
        ic_std=ic_std,
        ir=ir,
        ic_positive_ratio=ic_positive_ratio,
        n_periods=n_periods,
        ic_series=series,
        decay=decay or {},
        frequency="daily",
        ic_tstat=ic_tstat,
        ic_pvalue=ic_pvalue,
        multi_period=multi_period or {},
    )


def make_bt_result(
    *,
    ann_ret: float = 0.1250,
    sharpe: float = 1.35,
    max_dd: float = -0.1820,
    with_nav: bool = True,
    n_days: int = 60,
) -> Any:
    portfolio = {
        "ann_ret": ann_ret,
        "ann_vol": 0.18,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "avg_turnover": 0.25,
        "total_cost": 0.01,
        "ann_turnover": 0.25 * 252,
    }
    if with_nav:
        dates = [date(2023, 1, 3) + timedelta(days=i) for i in range(n_days)]
        rows = []
        for g in (0, 1, 4):
            nav = 1.0
            for i, d in enumerate(dates):
                nav *= 1.0 + 0.001 * (g + 1) * (1 if i % 3 else -0.5)
                rows.append({"trade_date": d, "group": g, "nav": nav})
        nav_df = pl.DataFrame(rows)
    else:
        nav_df = pl.DataFrame()

    return types.SimpleNamespace(
        strategy_name="top_n",
        nav=nav_df,
        summary_stats={"portfolio": portfolio, "long_short": portfolio},
    )


def make_to_result(avg_turnover: float = 0.35) -> Any:
    return types.SimpleNamespace(
        factor_name="test_factor",
        avg_turnover=avg_turnover,
        migration_matrix=pl.DataFrame(),
        daily_turnover=pl.DataFrame(),
        frequency="daily",
    )


def make_mono_result(
    group_means: list[float] | None = None,
    *,
    monotonicity_score: float = 1.0,
    direction: str = "positive",
) -> Any:
    if group_means is None:
        group_means = [0.001, 0.002, 0.003, 0.004, 0.006]
    return types.SimpleNamespace(
        factor_name="test_factor",
        monotonicity_score=monotonicity_score,
        group_means=group_means,
        direction=direction,
        ols_slope=0.001,
    )


def make_benchmark_result(ann_excess_ret: float = 0.0450) -> Any:
    return types.SimpleNamespace(
        benchmark_code="000300.SH",
        benchmark_name="沪深300",
        daily=pl.DataFrame(),
        ann_excess_ret=ann_excess_ret,
        tracking_error=0.08,
        information_ratio=0.56,
        excess_max_dd=-0.10,
    )


# ── full render ──────────────────────────────────────────────────────────────


class TestFullRender:
    def test_core_metrics_and_charts(self):
        ic = make_ic_result(ic_mean=0.0350, ir=0.44, ic_tstat=3.50, ic_pvalue=0.0005, n_periods=80)
        bt = make_bt_result(ann_ret=0.1250, sharpe=1.35, max_dd=-0.1820)
        to = make_to_result(0.35)
        mono = make_mono_result([0.001, 0.002, 0.004, 0.005, 0.008], monotonicity_score=1.0, direction="positive")
        bench = make_benchmark_result(0.0450)
        wf = {
            "status": "ok",
            "n_folds": 4,
            "oos_sharpe_mean": 0.92,
            "stability_ratio": 0.75,
            "oos_max_dd": -0.12,
        }
        quality = {"warnings": ["缺失值比例偏高：open 列 12% 为空"]}

        html = generate_tear_sheet(
            "momentum_20d",
            ic,
            bt,
            to,
            frequency="daily",
            date_range="20230101-20231231",
            universe="hs300",
            mono_result=mono,
            benchmark_result=bench,
            backtest_direction={"direction": "normal", "reason": "IC 均值非负，保持原方向"},
            walk_forward_summary=wf,
            quality_report=quality,
        )

        assert isinstance(html, str) and len(html) > 500
        assert "momentum_20d" in html
        assert "20230101-20231231" in html
        assert "hs300" in html
        assert "daily" in html

        # 核心指标数值（独立构造的期望格式）
        assert "0.0350" in html  # IC 均值 4 位
        assert "0.44" in html  # ICIR 2 位
        assert "3.50" in html  # t
        assert "0.0005" in html  # p
        assert "62.00%" in html  # IC>0 占比
        assert ">80<" in html or ">80</" in html or "80" in html  # N
        assert "12.50%" in html  # 年化收益
        assert "1.35" in html  # Sharpe
        assert "-18.20%" in html  # 最大回撤
        assert "4.50%" in html  # 超额年化
        assert "35.00%" in html  # 换手

        # 正向信号
        assert "正向信号" in html
        assert "反向信号" not in html

        # IC 衰减（multi_period）
        assert "1d" in html and "5d" in html and "10d" in html and "20d" in html
        assert "0.0280" in html  # 5d IC
        assert "0.31" in html  # 5d IR

        # 单调性
        assert "G1" in html and "G5" in html
        assert "Spearman" in html
        assert "正向单调" in html

        # WF ok
        assert "ok" in html
        assert "0.92" in html
        assert "0.75" in html

        # 质量警告
        assert "缺失值比例偏高：open 列 12% 为空" in html
        # n_periods=80 ≥ 60 → 不应有短样本警告
        assert "样本量较少" not in html

        # 两张 base64 图
        assert html.count("data:image/png;base64,") == 2

        # 无 JS / 无外部资源
        assert "<script" not in html.lower()
        assert "http://" not in html
        assert "https://" not in html

    def test_reversed_direction_badge(self):
        html = generate_tear_sheet(
            "rev_factor",
            make_ic_result(ic_mean=-0.04, n_periods=100),
            make_bt_result(),
            make_to_result(0.2),
            backtest_direction={
                "direction": "reversed",
                "reason": "IC 均值为负且 p 值小于等于 0.10",
            },
        )
        assert "反向信号（做多低因子值）" in html
        assert "IC 均值为负且 p 值小于等于 0.10" in html
        assert "正向信号" not in html


# ── degenerate inputs ────────────────────────────────────────────────────────


class TestDegenerateInputs:
    def test_bt_none_and_missing_optionals(self):
        ic = make_ic_result(empty_series=True, n_periods=10, multi_period={}, decay={})
        # 强制空 multi/decay
        ic.multi_period = {}
        ic.decay = {}
        html = generate_tear_sheet(
            "sparse_factor",
            ic,
            None,
            make_to_result(0.1),
            mono_result=None,
            benchmark_result=None,
            backtest_direction=None,
            walk_forward_summary={"status": "disabled", "n_folds": 0},
            quality_report=None,
        )
        assert isinstance(html, str)
        assert "sparse_factor" in html
        assert "未计算" in html
        assert "disabled" in html
        assert "超额年化" not in html
        assert "data:image/png;base64," not in html  # 无 nav / 空 ic_series
        # n_periods=10 → 样本量较少 + 短样本
        assert "样本量较少（10 期）" in html
        assert "短样本年化" in html

    def test_quality_report_none_no_crash(self):
        html = generate_tear_sheet(
            "q_none",
            make_ic_result(n_periods=100, ic_mean=0.05),
            make_bt_result(),
            make_to_result(0.1),
            quality_report=None,
        )
        assert "q_none" in html

    def test_empty_ic_series_still_renders_metrics(self):
        ic = make_ic_result(empty_series=True, n_periods=50, ic_mean=0.02, ir=0.25)
        html = generate_tear_sheet("empty_ic", ic, make_bt_result(), make_to_result(0.2))
        assert "0.0200" in html
        assert "0.25" in html
        # 有 nav 图但无 IC 图
        assert html.count("data:image/png;base64,") == 1


# ── benchmark row ────────────────────────────────────────────────────────────


class TestBenchmarkRow:
    def test_excess_row_present_when_benchmark_given(self):
        html = generate_tear_sheet(
            "with_bench",
            make_ic_result(n_periods=100, ic_mean=0.05),
            make_bt_result(),
            make_to_result(0.2),
            benchmark_result=make_benchmark_result(0.0333),
        )
        assert "超额年化" in html
        assert "3.33%" in html

    def test_excess_row_absent_when_benchmark_none(self):
        html = generate_tear_sheet(
            "no_bench",
            make_ic_result(n_periods=100, ic_mean=0.05),
            make_bt_result(),
            make_to_result(0.2),
            benchmark_result=None,
        )
        assert "超额年化" not in html


# ── warnings thresholds ──────────────────────────────────────────────────────


class TestWarnings:
    def test_low_ic_and_high_turnover(self):
        html = generate_tear_sheet(
            "warn_factor",
            make_ic_result(ic_mean=0.005, n_periods=100),
            make_bt_result(),
            make_to_result(0.85),
            quality_report={"warnings": ["自定义质量警告 A"]},
        )
        assert "IC 均值极低" in html
        assert "换手率较高" in html
        assert "自定义质量警告 A" in html

    def test_no_warnings_when_healthy(self):
        html = generate_tear_sheet(
            "healthy",
            make_ic_result(ic_mean=0.04, n_periods=120),
            make_bt_result(),
            make_to_result(0.3),
            quality_report={"warnings": []},
        )
        assert "警告" not in html or '<div class="warnings"' not in html


# ── decay fallback ───────────────────────────────────────────────────────────


class TestDecayFallback:
    def test_decay_only_when_no_multi_period(self):
        ic = make_ic_result(n_periods=90, multi_period={}, decay={1: 0.0412, 5: 0.0301})
        ic.multi_period = {}
        html = generate_tear_sheet("decay_only", ic, make_bt_result(), make_to_result(0.2))
        assert "0.0412" in html
        assert "0.0301" in html
        # decay 无 IR → 未计算
        assert "IC 衰减" in html

    def test_multi_period_preferred(self):
        ic = make_ic_result(
            n_periods=90,
            multi_period={1: {"ic_mean": 0.0555, "ir": 0.66, "ic_std": 0.1, "ic_positive_ratio": 0.6, "tstat": 2.0, "pvalue": 0.05}},
            decay={1: 0.0999},
        )
        html = generate_tear_sheet("mp_pref", ic, make_bt_result(), make_to_result(0.2))
        assert "0.0555" in html
        assert "0.66" in html
        assert "0.0999" not in html


# ── mono edge ────────────────────────────────────────────────────────────────


class TestMono:
    def test_non_monotonic_conclusion(self):
        mono = make_mono_result(
            [0.01, -0.01, 0.02, -0.02, 0.005],
            monotonicity_score=0.25,
            direction="positive",
        )
        html = generate_tear_sheet(
            "non_mono",
            make_ic_result(n_periods=100, ic_mean=0.03),
            make_bt_result(),
            make_to_result(0.2),
            mono_result=mono,
        )
        assert "非单调" in html
        assert "Spearman" in html

    def test_mono_none_shows_weijisuan(self):
        html = generate_tear_sheet(
            "no_mono",
            make_ic_result(n_periods=100, ic_mean=0.03),
            make_bt_result(),
            make_to_result(0.2),
            mono_result=None,
        )
        assert "单调性" in html
        assert "未计算" in html


# ── wf status ────────────────────────────────────────────────────────────────


class TestWalkForward:
    def test_wf_ok_block(self):
        html = generate_tear_sheet(
            "wf_ok",
            make_ic_result(n_periods=100, ic_mean=0.03),
            make_bt_result(),
            make_to_result(0.2),
            walk_forward_summary={
                "status": "ok",
                "n_folds": 5,
                "oos_sharpe_mean": 1.12,
                "stability_ratio": 0.88,
            },
        )
        assert "OOS Sharpe 均值" in html
        assert "1.12" in html
        assert "稳定率" in html
        assert "0.88" in html

    def test_wf_disabled_no_oos_rows(self):
        html = generate_tear_sheet(
            "wf_off",
            make_ic_result(n_periods=100, ic_mean=0.03),
            make_bt_result(),
            make_to_result(0.2),
            walk_forward_summary={"status": "disabled", "n_folds": 0},
        )
        assert "disabled" in html
        assert "OOS Sharpe 均值" not in html


# ── export / env ─────────────────────────────────────────────────────────────


def test_env_exported_for_portfolio_report():
    from factorzen.reports.tear_sheet import _ENV

    assert _ENV is not None
    assert "tear_sheet.html" in _ENV.list_templates()


def test_package_export():
    from factorzen.reports import generate_tear_sheet as exported

    assert exported is generate_tear_sheet

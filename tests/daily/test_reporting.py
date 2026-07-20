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
    grouped_nav: bool = False,
) -> Any:
    """构造回测结果。

    默认为**生产形态**：单一组合净值（trade_date, net_return, nav），**无 group 列**——
    全仓 5 个策略类都只产组合净值，没有任何策略产出分层 nav。
    ``grouped_nav=True`` 才产带 group 的多曲线形态，用于覆盖 _make_returns_chart 的分支。
    """
    portfolio = {
        "ann_ret": ann_ret,
        "ann_vol": 0.18,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "avg_turnover": 0.25,
        "total_cost": 0.01,
        "ann_turnover": 0.25 * 252,
    }
    if not with_nav:
        nav_df = pl.DataFrame()
    elif grouped_nav:
        dates = [date(2023, 1, 3) + timedelta(days=i) for i in range(n_days)]
        rows = []
        for g in (0, 1, 4):
            nav = 1.0
            for i, d in enumerate(dates):
                nav *= 1.0 + 0.001 * (g + 1) * (1 if i % 3 else -0.5)
                rows.append({"trade_date": d, "group": g, "nav": nav})
        nav_df = pl.DataFrame(rows)
    else:
        dates = [date(2023, 1, 3) + timedelta(days=i) for i in range(n_days)]
        rows = []
        nav = 1.0
        for i, d in enumerate(dates):
            ret = 0.0035 if i % 4 else -0.008  # 有涨有跌，产生真实回撤形态
            nav *= 1.0 + ret
            rows.append({"trade_date": d, "net_return": ret, "nav": nav})
        nav_df = pl.DataFrame(rows)

    return types.SimpleNamespace(
        strategy_name="top_n",
        nav=nav_df,
        returns=nav_df,
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
    with_group_daily: bool = True,
    n_days: int = 40,
) -> Any:
    """构造单调性结果。

    ``group_daily_returns`` 对齐 compute_monotonicity 的真实产出
    （trade_date, group[Int32, 0-indexed], mean_ret），报告层据此画分组净值与逐组绩效。
    """
    if group_means is None:
        group_means = [0.001, 0.002, 0.003, 0.004, 0.006]
    if with_group_daily:
        rows = []
        for g, base in enumerate(group_means):
            for i in range(n_days):
                rows.append(
                    {
                        "trade_date": date(2023, 1, 3) + timedelta(days=i),
                        "group": g,
                        "mean_ret": base * (1 if i % 3 else -0.6),
                    }
                )
        gdr = pl.DataFrame(rows).with_columns(pl.col("group").cast(pl.Int32))
    else:
        gdr = pl.DataFrame(
            schema={"trade_date": pl.Date, "group": pl.Int32, "mean_ret": pl.Float64}
        )
    return types.SimpleNamespace(
        factor_name="test_factor",
        monotonicity_score=monotonicity_score,
        group_means=group_means,
        direction=direction,
        ols_slope=0.001,
        group_daily_returns=gdr,
    )


def make_benchmark_result(
    ann_excess_ret: float = 0.0450,
    *,
    with_daily: bool = True,
    n_days: int = 40,
) -> Any:
    """构造基准对比结果。

    ``daily`` 对齐 BenchmarkResult 的真实列
    （trade_date, strategy_ret, benchmark_ret, excess_ret, *_nav）。
    """
    if with_daily:
        rows = []
        s_nav = b_nav = e_nav = 1.0
        for i in range(n_days):
            s_ret = 0.0035 if i % 4 else -0.006
            b_ret = 0.0020 if i % 5 else -0.004
            s_nav *= 1.0 + s_ret
            b_nav *= 1.0 + b_ret
            e_nav *= 1.0 + (s_ret - b_ret)
            rows.append(
                {
                    "trade_date": date(2023, 1, 3) + timedelta(days=i),
                    "strategy_ret": s_ret,
                    "benchmark_ret": b_ret,
                    "excess_ret": s_ret - b_ret,
                    "strategy_nav": s_nav,
                    "benchmark_nav": b_nav,
                    "excess_nav": e_nav,
                }
            )
        daily = pl.DataFrame(rows)
    else:
        daily = pl.DataFrame()
    return types.SimpleNamespace(
        benchmark_code="000300.SH",
        benchmark_name="沪深300",
        daily=daily,
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

        # 七张决策图各就各位（按 alt 断言，避免写死总数——加图就得改数字且看不出缺哪张）
        for alt in (
            "组合净值",
            "回撤曲线",
            "策略与基准对比",
            "IC 时序",
            "IC 累计曲线",
            "分组累计净值",
            "分组平均收益",
        ):
            assert f'alt="{alt}"' in html, f"缺少图表：{alt}"
        assert html.count("data:image/png;base64,") == 7

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
        # 净值/回撤依赖 bt_result 仍应渲染；两张 IC 图依赖空序列应整体缺席
        assert 'alt="组合净值"' in html
        assert 'alt="回撤曲线"' in html
        assert 'alt="IC 时序"' not in html
        assert 'alt="IC 累计曲线"' not in html


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


# ── 分层有效性区块：逐组绩效表 + 口径标注 ──────────────────────────────────


class TestGroupPerfTable:
    def test_group_perf_metrics_match_hand_computed_values(self):
        """逐组绩效对齐手算值（不依赖被测代码反推）。

        G1 收益序列 [0.01, -0.02, 0.03]：
          年化   = mean * 252 = (0.02/3) * 252 = 168.00%
          年化波动 = std(ddof=0) * sqrt(252) = 0.0205480 * 15.87451 = 0.326190
          Sharpe = 1.68 / 0.326190 = 5.15
          净值   = [1.01, 0.9898, 1.019694]，峰值 1.01 → 回撤 = 0.9898/1.01-1 = -2.00%
          胜率   = 2/3 = 66.67%
        """
        from factorzen.reports.tear_sheet import _build_group_perf_table

        rows = []
        for g, rets in enumerate([[0.01, -0.02, 0.03], [0.02, 0.01, 0.01]]):
            for i, r in enumerate(rets):
                rows.append(
                    {
                        "trade_date": date(2023, 1, 3) + timedelta(days=i),
                        "group": g,
                        "mean_ret": r,
                    }
                )
        mono = types.SimpleNamespace(
            group_daily_returns=pl.DataFrame(rows).with_columns(
                pl.col("group").cast(pl.Int32)
            )
        )

        table = _build_group_perf_table(mono)
        assert table is not None
        g1 = table["rows"][0]
        assert g1["group"] == "G1"
        assert g1["ann_ret"] == "168.00%"
        assert g1["sharpe"] == "5.15"
        assert g1["max_dd"] == "-2.00%"
        assert g1["win_rate"] == "66.67%"
        assert g1["n_periods"] == "3"

    def test_group_perf_table_rendered_with_caveat(self):
        """分组区块必须带口径标注——等权免成本的数字与组合回测不可直接比较。"""
        html = generate_tear_sheet(
            "grouped",
            make_ic_result(n_periods=100, ic_mean=0.03),
            make_bt_result(),
            make_to_result(0.2),
            mono_result=make_mono_result(),
        )
        assert "分层有效性" in html
        assert "胜率" in html
        assert "不含交易成本与交易约束" in html, "缺口径标注会诱导与组合回测数字混比"

    def test_no_group_block_when_group_daily_missing(self):
        """旧结果对象无 group_daily_returns 时应静默降级，不崩、不留空表头。

        只有依赖逐日明细的两项（分组净值图、逐组绩效表）消失；
        分组柱状图与口径标注仍在——它们只依赖 group_means，而 group_means
        同样是等权免成本口径，标注依旧适用。
        """
        html = generate_tear_sheet(
            "legacy_mono",
            make_ic_result(n_periods=100, ic_mean=0.03),
            make_bt_result(),
            make_to_result(0.2),
            mono_result=make_mono_result(with_group_daily=False),
        )
        assert 'alt="分组累计净值"' not in html
        assert "胜率" not in html, "逐组绩效表应缺席"
        # 柱状图与口径标注仍应在
        assert 'alt="分组平均收益"' in html
        assert "不含交易成本与交易约束" in html
        # 单调性表本身仍在（它只依赖 group_means）
        assert "Spearman" in html

    def test_single_group_rejected(self):
        """单组无从比较分层，绩效表应返回 None 而非渲染一行。"""
        from factorzen.reports.tear_sheet import _build_group_perf_table

        rows = [
            {"trade_date": date(2023, 1, 3) + timedelta(days=i), "group": 0, "mean_ret": 0.01}
            for i in range(5)
        ]
        mono = types.SimpleNamespace(
            group_daily_returns=pl.DataFrame(rows).with_columns(
                pl.col("group").cast(pl.Int32)
            )
        )
        assert _build_group_perf_table(mono) is None


# ── export / env ─────────────────────────────────────────────────────────────


def test_env_exported_for_portfolio_report():
    from factorzen.reports.tear_sheet import _ENV

    assert _ENV is not None
    assert "tear_sheet.html" in _ENV.list_templates()


def test_package_export():
    from factorzen.reports import generate_tear_sheet as exported

    assert exported is generate_tear_sheet

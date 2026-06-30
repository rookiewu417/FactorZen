"""单元测试:reports/_charts 的助手函数在 None/空输入下的防御行为。

主要覆盖各 `_make_*_chart` 在数据缺失时的早返回(返回 None,不渲染),以及若干
判定/抽取助手的边界分支。
"""

from __future__ import annotations

import types
from datetime import date

import pandas as pd
import polars as pl

from factorzen.reports._charts import (
    _event_study_has_valid_window_series,
    _extract_quantile_grouped_returns,
    _factor_corr_has_valid_off_diagonal,
    _factor_corr_is_multi_factor_input,
    _make_attribution_chart,
    _make_benchmark_chart,
    _make_event_study_chart,
    _make_factor_corr_heatmap,
    _make_ic_chart,
    _make_ic_distribution_chart,
    _make_monthly_return_heatmap,
    _make_quantile_spread_chart,
    _make_returns_chart,
    _make_turnover_chart,
    _make_walk_forward_chart,
    _prepare_brinson_plot_frame,
    _with_plot_dates,
)


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def test_make_charts_return_none_on_none_result():
    assert _make_returns_chart(None, "f") is None
    assert _make_ic_chart(None) is None
    assert _make_ic_distribution_chart(None) is None
    assert _make_monthly_return_heatmap(None) is None
    assert _make_benchmark_chart(None) is None
    assert _make_attribution_chart(None, None) is None
    assert _make_walk_forward_chart(None) is None
    assert _make_event_study_chart(None) is None
    assert _make_factor_corr_heatmap(None) is None
    assert _make_turnover_chart(None) is None


def test_make_returns_chart_none_when_nav_empty():
    assert _make_returns_chart(_ns(nav=pl.DataFrame()), "f") is None


def test_make_ic_chart_none_when_series_empty():
    assert _make_ic_chart(_ns(ic_series=pl.DataFrame())) is None


def test_make_turnover_chart_none_when_daily_turnover_empty():
    assert _make_turnover_chart(_ns(daily_turnover=pl.DataFrame())) is None


def test_make_quantile_spread_chart_none_for_insufficient_groups():
    assert _make_quantile_spread_chart({}) is None
    assert _make_quantile_spread_chart({0: [0.01, 0.02]}) is None  # 仅一组


def test_make_quantile_spread_chart_renders_with_two_groups():
    b64 = _make_quantile_spread_chart({0: [0.01, -0.01, 0.02], 4: [0.03, 0.01, 0.04]})
    assert isinstance(b64, str) and len(b64) > 100


def test_make_monthly_return_heatmap_renders_with_production_shaped_namespace():
    """修复4：生产路径（cli/main.py::_cmd_report_portfolio）用
    ``SimpleNamespace(nav=nav_df, returns=nav_df)`` 重建 sim_result。
    本函数只读 ``.returns``（不读 ``.nav``），.returns 必须真实可用，
    否则该函数在唯一的生产调用路径下恒返回 None（死代码，图表永不渲染）。
    """
    nav_df = pl.DataFrame(
        {
            "trade_date": [date(2023, 1, 1), date(2023, 1, 15), date(2023, 2, 1)],
            "net_return": [0.01, -0.005, 0.02],
            "nav": [1.01, 1.005, 1.025],
        }
    )
    sim_result = _ns(nav=nav_df, returns=nav_df)
    b64 = _make_monthly_return_heatmap(sim_result)
    assert isinstance(b64, str) and len(b64) > 100, (
        "sim_result.returns 已设置时，月度收益热力图应成功渲染而非返回 None"
    )


def test_make_monthly_return_heatmap_none_when_returns_attr_missing():
    """对照组：只设置 .nav、不设置 .returns 时（修复前的生产路径形状），
    函数应仍然返回 None —— 用于证明上一条用例确实在验证 .returns 生效，
    而不是函数本身的行为已变化到总是返回非 None。
    """
    nav_df = pl.DataFrame(
        {
            "trade_date": [date(2023, 1, 1), date(2023, 1, 15), date(2023, 2, 1)],
            "net_return": [0.01, -0.005, 0.02],
            "nav": [1.01, 1.005, 1.025],
        }
    )
    sim_result = _ns(nav=nav_df)  # 故意不设置 .returns
    assert _make_monthly_return_heatmap(sim_result) is None


def test_make_event_study_chart_none_when_no_events():
    es = _ns(windows=[0, 1], avg_cumret=[0.0, 0.01], ci_95=[0.0, 0.0], n_events=0)
    assert _make_event_study_chart(es) is None


def test_event_study_validity_predicate():
    assert _event_study_has_valid_window_series(None) is False
    ok = _ns(windows=[0, 1, 2], avg_cumret=[0.0, 0.1, 0.2], ci_95=[0.0, 0.0, 0.0], n_events=5)
    assert _event_study_has_valid_window_series(ok) is True
    # avg_cumret 与 windows 长度不一致 → False
    bad = _ns(windows=[0, 1, 2], avg_cumret=[0.0, 0.1], ci_95=[0.0, 0.0], n_events=5)
    assert _event_study_has_valid_window_series(bad) is False


def test_factor_corr_predicates():
    assert _factor_corr_is_multi_factor_input(None) is False
    assert _factor_corr_has_valid_off_diagonal(None) is False
    single = pl.DataFrame({"factor": ["a"], "a": [1.0]})
    assert _factor_corr_is_multi_factor_input(single) is False
    multi = pl.DataFrame({"factor": ["a", "b"], "a": [1.0, 0.5], "b": [0.5, 1.0]})
    assert _factor_corr_is_multi_factor_input(multi) is True
    assert _factor_corr_has_valid_off_diagonal(multi) is True


def test_extract_quantile_grouped_returns_empty_on_none():
    assert _extract_quantile_grouped_returns(None) == {}
    assert _extract_quantile_grouped_returns(_ns(nav=pl.DataFrame(), summary_stats={})) == {}


def test_prepare_brinson_plot_frame_returns_empty_input():
    empty = pl.DataFrame()
    assert _prepare_brinson_plot_frame(empty).is_empty()


def test_with_plot_dates_falls_back_to_index_without_date_column():
    frame = pd.DataFrame({"value": [1, 2, 3]})
    out, x_col, is_date = _with_plot_dates(frame, date_col="trade_date")
    assert x_col == "_plot_index"
    assert is_date is False
    assert list(out["_plot_index"]) == [0, 1, 2]

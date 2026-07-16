"""单元测试: reports/_charts 保留函数的防御行为与月度热力图。"""

from __future__ import annotations

import types
from datetime import date

import pandas as pd
import polars as pl

from factorzen.reports._charts import (
    _make_ic_chart,
    _make_monthly_return_heatmap,
    _make_returns_chart,
    _with_plot_dates,
)


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def test_make_charts_return_none_on_none_result():
    assert _make_returns_chart(None, "f") is None
    assert _make_ic_chart(None) is None
    assert _make_monthly_return_heatmap(None) is None


def test_make_returns_chart_none_when_nav_empty():
    assert _make_returns_chart(_ns(nav=pl.DataFrame()), "f") is None


def test_make_ic_chart_none_when_series_empty():
    assert _make_ic_chart(_ns(ic_series=pl.DataFrame())) is None


def test_make_monthly_return_heatmap_renders_with_production_shaped_namespace():
    """生产路径用 SimpleNamespace(nav=..., returns=...) 重建 sim_result。"""
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
    """只设置 .nav、不设置 .returns 时应返回 None。"""
    nav_df = pl.DataFrame(
        {
            "trade_date": [date(2023, 1, 1), date(2023, 1, 15), date(2023, 2, 1)],
            "net_return": [0.01, -0.005, 0.02],
            "nav": [1.01, 1.005, 1.025],
        }
    )
    sim_result = _ns(nav=nav_df)  # 故意不设置 .returns
    assert _make_monthly_return_heatmap(sim_result) is None


def test_with_plot_dates_falls_back_to_index_without_date_column():
    frame = pd.DataFrame({"value": [1, 2, 3]})
    out, x_col, is_date = _with_plot_dates(frame, date_col="trade_date")
    assert x_col == "_plot_index"
    assert is_date is False
    assert list(out["_plot_index"]) == [0, 1, 2]

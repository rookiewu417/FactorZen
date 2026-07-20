"""
test_charts_helpers.py：reports/_charts 保留函数的防御行为与月度热力图
test_tear_sheet_escape.py：tear_sheet 外部不可信文本须 HTML 转义
test_markets_crypto_report.py：MC6: crypto 成果展示页(市场语境 USDT/365/资金费/sector)
"""

from __future__ import annotations

import types
from datetime import date, timedelta

import pandas as pd
import polars as pl

from factorzen.reports._charts import (
    _make_benchmark_chart,
    _make_drawdown_chart,
    _make_group_bar_chart,
    _make_group_nav_chart,
    _make_ic_chart,
    _make_ic_cumulative_chart,
    _make_monthly_return_heatmap,
    _make_returns_chart,
    _with_plot_dates,
)
from factorzen.reports.portfolio_report import generate_portfolio_report
from factorzen.reports.tear_sheet import generate_tear_sheet


# ==== 来自 test_charts_helpers.py ====
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

# ── 新增决策图：IC 累计 / 回撤 / 分组净值 / 分组柱 / 基准对比 ──────────────────

def _prod_nav(n: int = 40) -> pl.DataFrame:
    """生产形态的组合 nav：单一净值曲线，**无 group 列**。

    全仓 5 个策略类（QuantileLongShort/TopNLongOnly/PrecomputedWeights/
    FactorWeighted/Optimizer）都产单一组合净值，不产分层 nav。
    """
    dates = [date(2023, 1, 3) + timedelta(days=i) for i in range(n)]
    nav, rows = 1.0, []
    for i, d in enumerate(dates):
        nav *= 1.0 + (0.004 if i % 4 else -0.009)  # 有涨有跌，制造真实回撤
        rows.append({"trade_date": d, "net_return": 0.004 if i % 4 else -0.009, "nav": nav})
    return pl.DataFrame(rows)

def _group_daily(n_days: int = 30, n_groups: int = 5) -> pl.DataFrame:
    rows = []
    for g in range(n_groups):
        for i in range(n_days):
            rows.append(
                {
                    "trade_date": date(2023, 1, 3) + timedelta(days=i),
                    "group": g,
                    "mean_ret": 0.0005 * (g + 1) * (1 if i % 3 else -0.6),
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("group").cast(pl.Int32))

def test_new_charts_return_none_on_none_result():
    assert _make_ic_cumulative_chart(None) is None
    assert _make_drawdown_chart(None) is None
    assert _make_group_bar_chart(None) is None
    assert _make_group_nav_chart(None) is None
    assert _make_benchmark_chart(None) is None

def test_ic_cumulative_renders_and_survives_nan_gaps():
    """含 NaN 的 IC 序列不应中断曲线或抛异常。"""
    ic = pl.DataFrame(
        {
            "trade_date": [date(2023, 1, 3) + timedelta(days=i) for i in range(30)],
            "ic": [float("nan") if i % 7 == 0 else 0.02 + 0.001 * (i % 5) for i in range(30)],
        }
    )
    b64 = _make_ic_cumulative_chart(_ns(ic_series=ic))
    assert isinstance(b64, str) and len(b64) > 100

def test_ic_cumulative_none_when_all_values_nan():
    ic = pl.DataFrame(
        {
            "trade_date": [date(2023, 1, 3) + timedelta(days=i) for i in range(5)],
            "ic": [float("nan")] * 5,
        }
    )
    assert _make_ic_cumulative_chart(_ns(ic_series=ic)) is None

def test_drawdown_renders_on_production_shaped_nav():
    b64 = _make_drawdown_chart(_ns(nav=_prod_nav()))
    assert isinstance(b64, str) and len(b64) > 100

def test_drawdown_none_when_nav_has_group_column():
    """带 group 列的 nav 是多条曲线，单一回撤曲线无意义，应跳过而非画错。"""
    grouped = _prod_nav(10).with_columns(pl.lit(0, dtype=pl.Int32).alias("group"))
    assert _make_drawdown_chart(_ns(nav=grouped)) is None

def test_drawdown_none_when_too_few_points():
    tiny = pl.DataFrame({"trade_date": [date(2023, 1, 3)], "nav": [1.0]})
    assert _make_drawdown_chart(_ns(nav=tiny)) is None

def test_group_nav_renders_from_group_daily_returns():
    b64 = _make_group_nav_chart(_ns(group_daily_returns=_group_daily()))
    assert isinstance(b64, str) and len(b64) > 100

def test_group_nav_none_when_single_group():
    """单组无从比较分层，应返回 None 而非画一条孤线。"""
    single = _group_daily(n_days=20, n_groups=1)
    assert _make_group_nav_chart(_ns(group_daily_returns=single)) is None

def test_group_nav_none_when_frame_empty():
    empty = pl.DataFrame(
        schema={"trade_date": pl.Date, "group": pl.Int32, "mean_ret": pl.Float64}
    )
    assert _make_group_nav_chart(_ns(group_daily_returns=empty)) is None

def test_group_bar_renders_and_rejects_degenerate_input():
    assert isinstance(_make_group_bar_chart(_ns(group_means=[0.001, 0.003, 0.006])), str)
    assert _make_group_bar_chart(_ns(group_means=[])) is None
    assert _make_group_bar_chart(_ns(group_means=[0.001])) is None
    assert _make_group_bar_chart(_ns(group_means=[0.001, float("nan")])) is None

def test_benchmark_chart_renders_with_three_series():
    daily = pl.DataFrame(
        {
            "trade_date": [date(2023, 1, 3) + timedelta(days=i) for i in range(30)],
            "strategy_nav": [1.0 + 0.002 * i for i in range(30)],
            "benchmark_nav": [1.0 + 0.001 * i for i in range(30)],
            "excess_nav": [1.0 + 0.001 * i for i in range(30)],
        }
    )
    b64 = _make_benchmark_chart(
        _ns(daily=daily, benchmark_name="沪深300", information_ratio=0.56)
    )
    assert isinstance(b64, str) and len(b64) > 100

def test_benchmark_chart_none_when_daily_empty():
    """生产 fixture 曾用空 daily——此时必须返回 None 而非抛异常。"""
    assert _make_benchmark_chart(_ns(daily=pl.DataFrame(), benchmark_name="沪深300")) is None

# ==== 来自 test_tear_sheet_escape.py ====
def test_tear_sheet_escapes_untrusted_text():
    html = generate_tear_sheet(
        "<script>alert('xss')</script>",
        None,
        None,
        None,
        date_range="2024-01-01 ~ 2024-06-30",
        universe="csi300",
        backtest_direction={
            "direction": "reversed",
            "reason": "<img src=x onerror=alert(1)>",
        },
        quality_report={"warnings": ["<script>alert('q')</script>"]},
    )
    assert "<script>" not in html, "因子名/警告中的 <script> 应被转义"
    assert "&lt;script&gt;" in html, "转义后应出现 &lt;script&gt;"
    assert "<img src=x" not in html, "方向 reason 中的 <img onerror> 应被转义"

# ==== 来自 test_markets_crypto_report.py ====
_METRICS = {
    "ann_ret": 0.25, "ann_vol": 0.35, "sharpe": 0.71, "max_dd": -0.12,
    "avg_turnover": 0.83, "total_cost": 0.008, "total_funding": 0.004,
}
_ATTRIB = pl.DataFrame({
    "type": ["factor_return", "factor_return", "brinson_allocation"],
    "key": ["funding_carry", "btc_beta", "L1"],
    "value": [0.012, -0.003, 0.002],
})
_RISK = pl.DataFrame({"metric": ["total_risk", "funding_carry"], "value": [0.35, 0.05]})

def test_crypto_report_renders_market_context():
    html = generate_portfolio_report(
        None, metrics=_METRICS, attribution_df=_ATTRIB, risk_summary_df=_RISK, market="crypto"
    )
    # 市场语境标注
    assert "USDT" in html
    assert "365" in html
    # 资金费成本卡
    assert "资金费" in html
    # crypto 风格因子名直接渲染（归因表）
    assert "funding_carry" in html and "btc_beta" in html
    # sector 措辞（M3 卡）
    assert "sector" in html
    # 平均换手率卡（avg_turnover，修 key mismatch）
    assert "平均换手率" in html

def test_ashare_report_unchanged():
    html = generate_portfolio_report(
        None, metrics={"ann_ret": 0.1, "ann_vol": 0.2, "sharpe": 0.5, "max_dd": -0.08,
                       "ann_turnover": 3.2, "total_cost": 0.01},
        market="ashare",
    )
    # A 股默认：M3 卡用「行业」，无 crypto 语境
    assert "行业" in html
    assert "USDT" not in html
    assert "资金费" not in html
    # 年化换手率卡（ann_turnover 原路径）
    assert "年化换手率" in html

def test_report_default_market_is_ashare():
    """不传 market 默认 ashare（向后兼容）。"""
    html = generate_portfolio_report(
        None, metrics={"ann_ret": 0.1, "ann_vol": 0.2, "sharpe": 0.5, "max_dd": -0.05},
    )
    assert "USDT" not in html


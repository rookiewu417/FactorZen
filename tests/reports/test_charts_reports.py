"""
test_charts_helpers：reports/_charts 基建函数
test_portfolio_report：组合 dashboard 市场语境
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from factorzen.reports._charts import _with_plot_dates
from factorzen.reports.portfolio_report import generate_portfolio_report
from factorzen.reports.signal_report import generate_signal_report
from factorzen.reports.trading_report import generate_trading_report


# ==== charts infrastructure ====
def test_charts_infra_suite():
    """_with_plot_dates 有日期列解析 / 无日期列回退索引。"""

    # -- 有 trade_date 时解析为日期轴 --
    def _section_0():
        frame = pd.DataFrame(
            {"trade_date": ["20240102", "20240103"], "v": [1.0, 2.0]}
        )
        out, x_col, is_date = _with_plot_dates(frame)
        assert x_col == "_plot_date"
        assert is_date is True
        assert "_plot_date" in out.columns

    _section_0()

    # -- 无日期列回退 index --
    def _section_1():
        frame = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
        out, x_col, is_date = _with_plot_dates(frame)
        assert x_col == "_plot_index"
        assert is_date is False
        assert len(out) == 3

    _section_1()


# ==== portfolio + escape via new reports ====
def test_portfolio_and_escape_suite():
    """crypto/ashare portfolio 语境；新报告 XSS 转义。"""

    # -- crypto 市场语境 --
    def _section_crypto():
        html = generate_portfolio_report(
            None, metrics=_METRICS, attribution_df=_ATTRIB, risk_summary_df=_RISK, market="crypto"
        )
        assert "USDT" in html
        assert "365" in html
        assert "资金费" in html
        assert "funding_carry" in html and "btc_beta" in html
        assert "sector" in html
        assert "平均换手率" in html

    _section_crypto()

    # -- ashare 不变 --
    def _section_ashare():
        html = generate_portfolio_report(
            None,
            metrics={
                "ann_ret": 0.1,
                "ann_vol": 0.2,
                "sharpe": 0.5,
                "max_dd": -0.08,
                "ann_turnover": 3.2,
                "total_cost": 0.01,
            },
            market="ashare",
        )
        assert "行业" in html
        assert "USDT" not in html
        assert "资金费" not in html
        assert "年化换手率" in html

    _section_ashare()

    # -- 默认 market=ashare --
    def _section_default():
        html = generate_portfolio_report(
            None, metrics={"ann_ret": 0.1, "ann_vol": 0.2, "sharpe": 0.5, "max_dd": -0.05}
        )
        assert "USDT" not in html

    _section_default()

    # -- 信号/交易报告转义 --
    def _section_escape():
        sig_html = generate_signal_report(
            None,
            factor_name="<script>alert('xss')</script>",
            universe="csi300",
        )
        assert "<script>" not in sig_html
        assert "&lt;script&gt;" in sig_html

        tr_html = generate_trading_report(
            "<script>alert('xss')</script>",
            None,
            backtest_direction={
                "direction": "reversed",
                "reason": "<img src=x onerror=alert(1)>",
            },
        )
        assert "<script>" not in tr_html
        assert "&lt;script&gt;" in tr_html
        assert "<img src=x" not in tr_html

    _section_escape()


_METRICS = {
    "ann_ret": 0.25,
    "ann_vol": 0.35,
    "sharpe": 0.71,
    "max_dd": -0.12,
    "avg_turnover": 0.83,
    "total_cost": 0.008,
    "total_funding": 0.004,
}
_ATTRIB = pl.DataFrame(
    {
        "type": ["factor_return", "factor_return", "brinson_allocation"],
        "key": ["funding_carry", "btc_beta", "L1"],
        "value": [0.012, -0.003, 0.002],
    }
)
_RISK = pl.DataFrame({"metric": ["total_risk", "funding_carry"], "value": [0.35, 0.05]})

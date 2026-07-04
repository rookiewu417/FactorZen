"""MC6: crypto 成果展示页（市场语境：USDT/365/资金费/sector）。"""
from __future__ import annotations

import polars as pl

from factorzen.reports.portfolio_report import generate_portfolio_report

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

"""Portfolio Dashboard Report — M7 成果展示页。

生成组合策略绩效 HTML dashboard，复用现有 reports 引擎
（Jinja2 FileSystemLoader + metric_number / metric_percent 过滤器 + matplotlib 图表）。
"""

from datetime import datetime
from typing import Any

import polars as pl

from factorzen.core.logger import get_logger
from factorzen.reports._charts import _make_monthly_return_heatmap, _make_returns_chart
from factorzen.reports.tear_sheet import _ENV

logger = get_logger(__name__)


def generate_portfolio_report(
    sim_result: Any,
    *,
    metrics: dict[str, Any],
    attribution_df: pl.DataFrame | None = None,
    risk_summary_df: pl.DataFrame | None = None,
    portfolio_manifest: dict[str, Any] | None = None,
) -> str:
    """生成组合成果展示 HTML dashboard。

    Parameters
    ----------
    sim_result : StrategyBacktestResult or None
        策略回测结果；为 None 时跳过净值图，只渲染 metrics / 归因 / 风险。
    metrics : dict
        绩效指标字典（ann_ret, ann_vol, sharpe, max_dd, ann_turnover, total_cost 等）。
    attribution_df : pl.DataFrame, optional
        归因明细表（type / key / value 列）。
    risk_summary_df : pl.DataFrame, optional
        风险汇总表（metric / value 列）。
    portfolio_manifest : dict, optional
        持仓 meta（n_holdings, status 等）以及 return_attribution_available 标志。

    Returns
    -------
    str
        完整的 HTML 报告字符串。
    """
    manifest = portfolio_manifest or {}
    portfolio_name: str = str(manifest.get("portfolio_name", "组合策略"))

    charts: dict[str, str] = {}
    if sim_result is not None:
        try:
            nav_b64 = _make_returns_chart(sim_result, portfolio_name)
            if nav_b64:
                charts["returns_chart"] = nav_b64
        except Exception:
            logger.warning("生成组合净值图失败", exc_info=True)
        try:
            monthly_b64 = _make_monthly_return_heatmap(sim_result)
            if monthly_b64:
                charts["monthly_return_chart"] = monthly_b64
        except Exception:
            logger.warning("生成月度收益热力图失败", exc_info=True)

    attribution_rows: list[dict[str, Any]] = (
        attribution_df.to_dicts()
        if attribution_df is not None and not attribution_df.is_empty()
        else []
    )
    risk_rows: list[dict[str, Any]] = (
        risk_summary_df.to_dicts()
        if risk_summary_df is not None and not risk_summary_df.is_empty()
        else []
    )

    # 归因可用标志：manifest 明确标注 False 时提示占位
    attribution_available: bool = bool(manifest.get("return_attribution_available", True))

    template = _ENV.get_template("portfolio_dashboard.html")
    return template.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        portfolio_name=portfolio_name,
        metrics=metrics,
        charts=charts,
        attribution_rows=attribution_rows,
        risk_rows=risk_rows,
        portfolio_manifest=manifest,
        has_sim=sim_result is not None,
        attribution_available=attribution_available,
    )

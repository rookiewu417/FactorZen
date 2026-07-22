"""Portfolio Dashboard Report — M7 成果展示页。

生成组合策略绩效 HTML dashboard，复用 reports 基建
（Jinja2 FileSystemLoader + metric_number / metric_percent 过滤器 + matplotlib 图表）。
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl

from factorzen.core.logger import get_logger
from factorzen.reports._charts import (
    _fig_to_base64,
    _format_sparse_x_axis,
    _with_plot_dates,
)
from factorzen.reports._formatting import (
    _format_metric_number,
    _format_metric_percent,
    _is_finite_metric,
    _safe_attr,
)

logger = get_logger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=jinja2.select_autoescape(["html"]),
)
_ENV.filters["metric_number"] = _format_metric_number
_ENV.filters["metric_percent"] = _format_metric_percent
_ENV.tests["finite_metric"] = _is_finite_metric


# 市场语境标签（年化天数/单位/板块措辞/资金费）。crypto 与 A 股差异全走这里。
_MARKET_LABELS: dict[str, dict[str, Any]] = {
    "ashare": {"ann_days": 252, "unit": "", "sector_word": "行业", "has_funding": False},
    "crypto": {"ann_days": 365, "unit": "USDT", "sector_word": "sector", "has_funding": True},
}


def _portfolio_nav_chart(sim_result: Any, portfolio_name: str) -> str | None:
    """组合净值曲线（portfolio dashboard 专用）。"""
    if sim_result is None:
        return None
    nav = _safe_attr(sim_result, "nav")
    if nav is None or nav.is_empty():
        return None
    nav_pd = nav.to_pandas()
    if "nav" not in nav_pd.columns:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.5))
    nav_pd, x_col, is_date_axis = _with_plot_dates(nav_pd)
    if "trade_date" in nav_pd.columns:
        nav_pd = nav_pd.sort_values("trade_date")
    ax.plot(nav_pd[x_col], nav_pd["nav"], linewidth=1.4, label="组合", color="#2c7fb8")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_title(f"组合净值 - {portfolio_name}", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    _format_sparse_x_axis(ax, is_date_axis=is_date_axis)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _portfolio_monthly_heatmap(sim_result: Any) -> str | None:
    """月度收益热力图（portfolio dashboard 专用）。"""
    if sim_result is None:
        return None
    returns = _safe_attr(sim_result, "returns")
    if returns is None or returns.is_empty():
        return None
    ret_pd = returns.to_pandas()
    if "trade_date" not in ret_pd.columns:
        return None
    ret_col = next(
        (col for col in ["net_return", "ret", "return"] if col in ret_pd.columns),
        None,
    )
    if ret_col is None:
        return None
    dates = ret_pd["trade_date"].astype(str)
    parsed = None
    for fmt in ["%Y%m%d", "%Y-%m-%d"]:
        try:
            candidate = __import__("pandas").to_datetime(dates, format=fmt, errors="coerce")
        except Exception:
            candidate = None
        if candidate is not None and candidate.notna().any():
            parsed = candidate
            break
    if parsed is None:
        return None
    frame = ret_pd.assign(_date=parsed).dropna(subset=["_date", ret_col])
    if frame.empty:
        return None
    frame["_year"] = frame["_date"].dt.year
    frame["_month"] = frame["_date"].dt.month
    monthly = (
        frame.groupby(["_year", "_month"])[ret_col]
        .apply(lambda s: float(np.prod(1 + s.to_numpy(dtype=float)) - 1))
        .unstack("_month")
        .reindex(columns=range(1, 13))
    )
    if monthly.empty:
        return None
    data = monthly.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    fig, ax = plt.subplots(figsize=(10, max(2.8, 0.7 * len(monthly.index) + 1.5)))
    vmax = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 0.01
    vmax = max(vmax, 0.01)
    image = ax.imshow(masked, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("月度收益热力图", fontsize=12)
    ax.set_xticks(range(12))
    ax.set_xticklabels([str(i) for i in range(1, 13)])
    ax.set_yticks(range(len(monthly.index)))
    ax.set_yticklabels([str(y) for y in monthly.index])
    ax.set_xlabel("月份")
    ax.set_ylabel("年份")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j] * 100:.1f}%", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, format=mticker.PercentFormatter(1.0))
    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_portfolio_report(
    sim_result: Any,
    *,
    metrics: dict[str, Any],
    attribution_df: pl.DataFrame | None = None,
    risk_summary_df: pl.DataFrame | None = None,
    portfolio_manifest: dict[str, Any] | None = None,
    market: str = "ashare",
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
            nav_b64 = _portfolio_nav_chart(sim_result, portfolio_name)
            if nav_b64:
                charts["returns_chart"] = nav_b64
        except Exception:
            logger.warning("生成组合净值图失败", exc_info=True)
        try:
            monthly_b64 = _portfolio_monthly_heatmap(sim_result)
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

    labels = _MARKET_LABELS.get(market, _MARKET_LABELS["ashare"])

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
        market=market,
        labels=labels,
    )

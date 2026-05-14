"""Factor Tear Sheet 报告引擎。

生成包含 6 个面板的 HTML 因子研究报告：
1. Overview — 核心指标总览
2. Returns Analysis — 分层回测，NAV 曲线
3. IC Analysis — IC 时序 + 衰减表
4. Turnover Analysis — 换手率时序
5. Risk Attribution — 风险 / 市场状态
6. Summary — 星级评估与解读
"""

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import jinja2
import matplotlib

matplotlib.use("Agg")  # 非交互后端，不弹窗
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl

from common.logger import get_logger

logger = get_logger(__name__)

# ── 模板加载 ──────────────────────────────────────────────────────────
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=jinja2.select_autoescape(["html"]),
)


def _fig_to_base64(fig: plt.Figure) -> str:
    """将 matplotlib Figure 转为 base64 PNG 字符串。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _safe_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """安全获取对象属性。"""
    if obj is not None and hasattr(obj, attr):
        return getattr(obj, attr)
    return default


# ── 图表生成 ──────────────────────────────────────────────────────────

def _make_returns_chart(bt_result: Any, factor_name: str) -> Optional[str]:
    """分层回测 NAV 曲线图。"""
    if bt_result is None:
        return None
    nav = _safe_attr(bt_result, "nav")
    if nav is None or nav.is_empty():
        return None

    fig, ax = plt.subplots(figsize=(10, 4.5))
    nav_pd = nav.to_pandas()
    if "trade_date" in nav_pd.columns and "group" in nav_pd.columns and "nav" in nav_pd.columns:
        for g, grp_data in nav_pd.groupby("group"):
            grp_data = grp_data.sort_values("trade_date")
            ax.plot(
                grp_data["trade_date"], grp_data["nav"],
                linewidth=1.2,
                label=f"Q{g+1}" if isinstance(g, (int, float)) else str(g),
            )
    else:
        for col in nav_pd.columns:
            if col == "trade_date":
                continue
            ax.plot(nav_pd["trade_date"], nav_pd[col], linewidth=1.2, label=str(col))

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_title(f"Stratified Backtest NAV &mdash; {factor_name}", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _make_ic_chart(ic_result: Any) -> Optional[str]:
    """IC 时序棒图 + 滚动均值。"""
    if ic_result is None:
        return None
    ic_series = _safe_attr(ic_result, "ic_series")
    if ic_series is None or ic_series.is_empty():
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    ic_pd = ic_series.to_pandas()
    ic_col = "ic" if "ic" in ic_pd.columns else [c for c in ic_pd.columns if c != "trade_date"][0]
    date_col = "trade_date" if "trade_date" in ic_pd.columns else ic_pd.columns[0]

    ax.bar(ic_pd[date_col], ic_pd[ic_col], width=1.0, color="#bdc3c7", alpha=0.6, label="IC")
    if len(ic_pd) >= 5:
        window = min(20, max(3, len(ic_pd) // 3))
        rolling = ic_pd[ic_col].rolling(window=window, min_periods=1).mean()
        ax.plot(ic_pd[date_col], rolling, color="#e74c3c", linewidth=1.2,
                label=f"Rolling Mean({window})")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title("Rank IC Time Series", fontsize=12)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _make_turnover_chart(to_result: Any) -> Optional[str]:
    """换手率填充图。"""
    if to_result is None:
        return None
    dt = _safe_attr(to_result, "daily_turnover")
    if dt is None or dt.is_empty():
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    dt_pd = dt.to_pandas()
    date_col = "trade_date" if "trade_date" in dt_pd.columns else dt_pd.columns[0]
    val_col = [c for c in dt_pd.columns if c != date_col][0]
    ax.fill_between(dt_pd[date_col], dt_pd[val_col], alpha=0.3, color="#9b59b6")
    ax.plot(dt_pd[date_col], dt_pd[val_col], linewidth=1.2, color="#8e44ad")
    ax.set_title("Periodic Turnover", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1%}"))
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


# ── 辅助函数 ──────────────────────────────────────────────────────────

def _build_bt_summary_table(stats: dict) -> list:
    """构建回测分组统计表格行。"""
    rows = []
    group_keys = sorted([k for k in stats if isinstance(k, int)])
    for key in group_keys:
        gs = stats[key]
        if not isinstance(gs, dict):
            continue
        rows.append({
            "group": f"Q{key+1}",
            "ann_ret": f"{gs.get('ann_ret', 0)*100:.2f}%",
            "ann_vol": f"{gs.get('ann_vol', 0)*100:.2f}%",
            "sharpe": f"{gs.get('sharpe', 0):.3f}",
            "max_dd": f"{gs.get('max_dd', 0)*100:.2f}%",
        })
    if "long_short" in stats:
        ls = stats["long_short"]
        rows.append({
            "group": "L/S",
            "ann_ret": f"{ls.get('ann_ret', 0)*100:.2f}%",
            "ann_vol": f"{ls.get('ann_vol', 0)*100:.2f}%",
            "sharpe": f"{ls.get('sharpe', 0):.3f}",
            "max_dd": f"{ls.get('max_dd', 0)*100:.2f}%",
        })
    return rows


def _extract_metrics(ic_result, bt_result, to_result, advanced_results) -> Dict[str, Any]:
    """提取所有关键指标为扁平字典。"""
    m: Dict[str, Any] = {}

    m["ic_mean"] = _safe_attr(ic_result, "ic_mean", 0) or 0
    m["ic_std"] = _safe_attr(ic_result, "ic_std", 0) or 0
    m["ir"] = _safe_attr(ic_result, "ir", 0) or 0
    m["ic_positive_ratio"] = _safe_attr(ic_result, "ic_positive_ratio", 0) or 0
    m["n_periods"] = _safe_attr(ic_result, "n_periods", 0) or 0
    m["decay"] = _safe_attr(ic_result, "decay", {})

    m["bt_stats"] = []
    if bt_result is not None:
        stats = _safe_attr(bt_result, "summary_stats", {})
        if stats:
            m["bt_stats"] = _build_bt_summary_table(stats)
            if "long_short" in stats:
                m["ls_ann_ret"] = stats["long_short"].get("ann_ret", 0)
                m["ls_sharpe"] = stats["long_short"].get("sharpe", 0)
                m["ls_max_dd"] = stats["long_short"].get("max_dd", 0)

    m["avg_turnover"] = _safe_attr(to_result, "avg_turnover", 0) or 0

    if advanced_results:
        mono = advanced_results.get("mono")
        if mono:
            m["monotonicity_score"] = _safe_attr(mono, "monotonicity_score")

        acorr = advanced_results.get("autocorr")
        if acorr:
            m["rank_autocorr"] = _safe_attr(acorr, "mean_autocorr")
            m["half_life"] = _safe_attr(acorr, "half_life_est")

        sector = advanced_results.get("sector")
        if sector:
            m["sector_ic"] = _safe_attr(sector, "sector_ic_df")

        sz = advanced_results.get("size")
        if sz:
            m["size_buckets"] = _safe_attr(sz, "buckets", {})

        decay_list = advanced_results.get("decay_results", [])
        if decay_list:
            m["decay_table"] = [
                {"horizon": d.horizon, "ic_mean": d.ic_mean, "ic_std": d.ic_std}
                for d in decay_list
            ]

    return m


def _compute_star_rating(metrics: Dict[str, Any]) -> int:
    """根据指标计算 1-5 星级评分。"""
    stars = 3
    ic_mean = abs(metrics.get("ic_mean", 0))
    ir = metrics.get("ir", 0)
    ls_sharpe = metrics.get("ls_sharpe", 0)

    if ic_mean > 0.04:
        stars += 1
    if ir > 0.5:
        stars += 1
    if ls_sharpe > 1.0:
        stars += 1
    if ic_mean < 0.015 and ir < 0.2:
        stars = max(1, stars - 1)

    return min(5, max(1, stars))


def _generate_summary_text(factor_name: str, metrics: Dict[str, Any]) -> str:
    """生成总结解读文本，包含星级评级。"""
    star_char = chr(9733)  # ★
    stars = _compute_star_rating(metrics)
    rating = star_char * stars + chr(9734) * (5 - stars)

    lines = [f"<p><strong>评级: {rating} ({stars}/5)</strong></p>"]

    ic_mean = metrics.get("ic_mean") or 0
    ir = metrics.get("ir") or 0

    if abs(ic_mean) < 0.01:
        lines.append("<p>IC 均值极低（|IC| &lt; 0.01），因子对收益的预测能力非常有限。</p>")
    elif ic_mean > 0.03:
        lines.append(f"<p>IC 均值 {ic_mean:.4f}（Spearman &rho;），因子展现出较强的正向预测能力。</p>")
    elif ic_mean < -0.03:
        lines.append(f"<p>IC 均值 {ic_mean:.4f}，因子呈现显著的负向预测能力（可用作反向因子）。</p>")
    else:
        lines.append(f"<p>IC 均值 {ic_mean:.4f}，因子具备一定的预测能力。</p>")

    if ir > 0.5:
        lines.append(f"<p>信息比率 IR = {ir:.2f}，因子稳定性良好，IC 方向一致性高。</p>")
    elif ir > 0.2:
        lines.append(f"<p>信息比率 IR = {ir:.2f}，因子稳定性一般。</p>")

    ls_ret = metrics.get("ls_ann_ret") or 0
    if ls_ret > 0.05:
        lines.append(f"<p>多空年化收益 {ls_ret*100:.1f}%，分层效果显著，Top-Bottom 区分度高。</p>")
    elif ls_ret > 0:
        lines.append(f"<p>多空年化收益 {ls_ret*100:.1f}%，分层效果较弱。</p>")
    elif ls_ret < 0:
        lines.append("<p>多空收益为负，分层回测不理想，因子分组单调性需关注。</p>")

    return "\n".join(lines)


# ── 主函数 ────────────────────────────────────────────────────────────

def generate_tear_sheet(
    factor_name: str,
    ic_result: Any,
    bt_result: Any,
    to_result: Any,
    *,
    frequency: str = "daily",
    date_range: str = "",
    advanced_results: Optional[Dict[str, Any]] = None,
    universe: str = "lft_default",
) -> str:
    """生成因子 Tear Sheet HTML 报告。

    Parameters
    ----------
    factor_name : str
        因子名称。
    ic_result : ICAnalysisResult or None
        IC 分析结果。
    bt_result : BacktestResult or None
        分层回测结果。
    to_result : TurnoverResult or None
        换手率分析结果。
    frequency : str
        数据频率（daily / weekly / monthly）。
    date_range : str
        日期范围字符串，如 "2025-01-01 ~ 2025-05-13"。
    advanced_results : dict, optional
        高级评价结果，键名：
        - decay_results (list[ICDecayResult])
        - mono (MonotonicityResult)
        - autocorr (RankAutocorrResult)
        - sector (SectorICResult)
        - size (SizeICResult)
    universe : str
        股票池。

    Returns
    -------
    str
        完整的 HTML 报告字符串。
    """
    charts: Dict[str, str] = {}

    if bt_result is not None:
        try:
            nav_b64 = _make_returns_chart(bt_result, factor_name)
            if nav_b64:
                charts["returns_chart"] = nav_b64
        except Exception:
            logger.warning("生成分层回测图表失败", exc_info=True)

    if ic_result is not None:
        try:
            ic_b64 = _make_ic_chart(ic_result)
            if ic_b64:
                charts["ic_chart"] = ic_b64
        except Exception:
            logger.warning("生成 IC 图表失败", exc_info=True)

    if to_result is not None:
        try:
            to_b64 = _make_turnover_chart(to_result)
            if to_b64:
                charts["turnover_chart"] = to_b64
        except Exception:
            logger.warning("生成换手率图表失败", exc_info=True)

    metrics = _extract_metrics(ic_result, bt_result, to_result, advanced_results)

    warnings: List[str] = []
    if metrics.get("n_periods", 0) < 30:
        warnings.append(f"样本量较少（{metrics['n_periods']} 期），IC 估计可能不稳定。")
    if metrics.get("ic_mean") is not None and abs(metrics["ic_mean"]) < 0.01:
        warnings.append("IC 均值极低（|IC| < 0.01），因子预测能力有限。")
    if metrics.get("avg_turnover", 0) is not None and metrics.get("avg_turnover", 0) > 0.8:
        warnings.append("换手率较高（>80%），信号稳定性需关注。")

    summary_html = _generate_summary_text(factor_name, metrics)

    template = _ENV.get_template("tear_sheet.html")
    return template.render(
        factor_name=factor_name,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        date_range=date_range,
        frequency=frequency,
        universe=universe,
        metrics=metrics,
        charts=charts,
        warnings=warnings,
        summary_html=summary_html,
    )

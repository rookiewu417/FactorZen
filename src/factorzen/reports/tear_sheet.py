"""Factor Tear Sheet 报告引擎。

生成包含目录、综合结论、收益表现、预测能力、结构检验、交易可行性、
稳健性验证、风险归因和附录的 HTML 因子研究报告。
"""

import base64
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2
import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")  # 非交互后端，不弹窗
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Windows 中文字体支持（优先 Microsoft YaHei，回退 SimHei）
for _font in ["Microsoft YaHei", "SimHei", "sans-serif"]:
    matplotlib.rcParams["font.family"] = _font
    matplotlib.rcParams["axes.unicode_minus"] = False
    break

from factorzen.core.logger import get_logger  # noqa: E402
from factorzen.reports._formatting import (  # noqa: E402  (re-export 供模板与测试使用)
    _finite_float,
    _format_metric_number,
    _format_metric_percent,
    _is_finite_metric,
    _num,
    _safe_attr,
    _same_direction,
)
from factorzen.reports._scoring import (  # noqa: E402  (re-export 供模板与测试使用)
    FactorRating,
    _compute_factor_rating,
    _stars_from_score,
)

logger = get_logger(__name__)

RATING_COMPONENT_MAX_SCORES = {
    "Alpha 强度": 30,
    "稳定性": 25,
    "可交易性": 20,
    "鲁棒性": 15,
    "结构质量": 10,
}


# ── 模板加载 ──────────────────────────────────────────────────────────
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=jinja2.select_autoescape(["html"]),
)
_ENV.filters["metric_number"] = _format_metric_number
_ENV.filters["metric_percent"] = _format_metric_percent
_ENV.tests["finite_metric"] = _is_finite_metric


def _fig_to_base64(fig: plt.Figure) -> str:
    """将 matplotlib Figure 转为 base64 PNG 字符串。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _with_plot_dates(frame: Any, date_col: str = "trade_date") -> tuple[Any, str, bool]:
    """Return a frame with a datetime x-axis column when dates can be parsed."""
    if date_col not in frame.columns:
        out = frame.copy()
        out["_plot_index"] = np.arange(len(out))
        return out, "_plot_index", False
    try:
        pandas = __import__("pandas")
        parsed = pandas.to_datetime(frame[date_col], errors="coerce")
    except Exception:
        return frame, date_col, False
    if not parsed.notna().any():
        return frame, date_col, False
    out = frame.copy()
    out["_plot_date"] = parsed
    return out, "_plot_date", True


def _format_sparse_x_axis(ax: plt.Axes, *, is_date_axis: bool, max_ticks: int = 7) -> None:
    """Keep report chart x-axis labels readable on dense daily samples."""
    if is_date_axis:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=max(3, int(max_ticks)))
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    else:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=max(3, int(max_ticks))))
    ax.tick_params(axis="x", labelsize=8)
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment("center")


# ── 图表生成 ──────────────────────────────────────────────────────────


def _make_returns_chart(bt_result: Any, factor_name: str) -> str | None:
    """分层回测净值曲线图。"""
    if bt_result is None:
        return None
    nav = _safe_attr(bt_result, "nav")
    if nav is None or nav.is_empty():
        return None

    fig, ax = plt.subplots(figsize=(10, 4.5))
    nav_pd = nav.to_pandas()
    nav_pd, x_col, is_date_axis = _with_plot_dates(nav_pd)
    if "trade_date" in nav_pd.columns and "group" in nav_pd.columns and "nav" in nav_pd.columns:
        for g, grp_data in nav_pd.groupby("group"):
            grp_data = grp_data.sort_values("trade_date")
            ax.plot(
                grp_data[x_col],
                grp_data["nav"],
                linewidth=1.2,
                label=f"Q{g + 1}" if isinstance(g, (int, float)) else str(g),
            )
    elif "trade_date" in nav_pd.columns and "nav" in nav_pd.columns:
        nav_pd = nav_pd.sort_values("trade_date")
        ax.plot(nav_pd[x_col], nav_pd["nav"], linewidth=1.4, label="组合")
    else:
        for col in nav_pd.columns:
            if col in {"trade_date", "_plot_date"}:
                continue
            ax.plot(nav_pd[x_col], nav_pd[col], linewidth=1.2, label=str(col))

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_title(f"分层回测净值 - {factor_name}", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    _format_sparse_x_axis(ax, is_date_axis=is_date_axis)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _make_ic_chart(ic_result: Any) -> str | None:
    """IC 时序棒图 + 滚动均值。"""
    if ic_result is None:
        return None
    ic_series = _safe_attr(ic_result, "ic_series")
    if ic_series is None or ic_series.is_empty():
        return None

    ic_pd = ic_series.to_pandas()
    ic_col = (
        "ic"
        if "ic" in ic_pd.columns
        else next((c for c in ic_pd.columns if c != "trade_date"), None)
    )
    if ic_col is None:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    date_col = "trade_date" if "trade_date" in ic_pd.columns else ic_pd.columns[0]
    ic_pd, x_col, is_date_axis = _with_plot_dates(ic_pd, date_col)

    ax.bar(ic_pd[x_col], ic_pd[ic_col], width=1.0, color="#bdc3c7", alpha=0.6, label="IC")
    if len(ic_pd) >= 5:
        window = min(20, max(3, len(ic_pd) // 3))
        rolling = ic_pd[ic_col].rolling(window=window, min_periods=1).mean()
        ax.plot(
            ic_pd[x_col], rolling, color="#e74c3c", linewidth=1.2, label=f"滚动均值({window}期)"
        )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title("Rank IC 时序", fontsize=12)
    ax.legend(fontsize=8)
    _format_sparse_x_axis(ax, is_date_axis=is_date_axis)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _make_ic_distribution_chart(ic_result: Any) -> str | None:
    """IC 分布直方图，用于检查 IC 是否由少数极端日期贡献。"""
    if ic_result is None:
        return None
    ic_series = _safe_attr(ic_result, "ic_series")
    if ic_series is None or ic_series.is_empty():
        return None

    ic_pd = ic_series.to_pandas()
    ic_col = (
        "ic"
        if "ic" in ic_pd.columns
        else next((c for c in ic_pd.columns if c != "trade_date"), None)
    )
    if ic_col is None:
        return None
    values = np.asarray(ic_pd[ic_col].dropna(), dtype=float)
    if values.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = min(30, max(8, int(np.sqrt(values.size))))
    ax.hist(values, bins=bins, color="#4c78a8", alpha=0.78, edgecolor="white")
    ax.axvline(0, color="#6b7280", linestyle="--", linewidth=1.0, label="0")
    ax.axvline(values.mean(), color="#f58518", linewidth=1.5, label=f"均值 {values.mean():.4f}")
    ax.set_title("IC 分布", fontsize=12)
    ax.set_xlabel("IC")
    ax.set_ylabel("频数")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _make_monthly_return_heatmap(bt_result: Any) -> str | None:
    """多空组合月度收益热力图，从回测收益序列派生。"""
    if bt_result is None:
        return None
    returns = _safe_attr(bt_result, "returns")
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
        candidate = None
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


def _monthly_returns(bt_result: Any) -> list[tuple[str, float]]:
    returns = _safe_attr(bt_result, "returns")
    if returns is None or returns.is_empty():
        return []

    ret_pd = returns.to_pandas()
    if "trade_date" not in ret_pd.columns:
        return []
    ret_col = next((col for col in ["net_return", "ret", "return"] if col in ret_pd.columns), None)
    if ret_col is None:
        return []

    dates = ret_pd["trade_date"].astype(str)
    parsed = None
    pandas = __import__("pandas")
    for fmt in ["%Y%m%d", "%Y-%m-%d"]:
        candidate = pandas.to_datetime(dates, format=fmt, errors="coerce")
        if candidate.notna().any():
            parsed = candidate
            break
    if parsed is None:
        return []

    frame = ret_pd.assign(_date=parsed).dropna(subset=["_date", ret_col])
    if frame.empty:
        return []
    frame["_month"] = frame["_date"].dt.to_period("M").astype(str)
    monthly = frame.groupby("_month")[ret_col].apply(
        lambda s: float(np.prod(1 + s.to_numpy(dtype=float)) - 1)
    )
    return [(str(month), float(ret)) for month, ret in monthly.sort_index().items()]


def _build_monthly_return_summary(bt_result: Any) -> dict[str, str] | None:
    monthly = _monthly_returns(bt_result)
    if not monthly:
        return None

    positive_count = sum(1 for _, ret in monthly if ret > 0)
    best_month, best_ret = max(monthly, key=lambda item: item[1])
    worst_month, worst_ret = min(monthly, key=lambda item: item[1])
    total_abs = sum(abs(ret) for _, ret in monthly)
    concentration = (
        max((abs(ret) for _, ret in monthly), default=0.0) / total_abs if total_abs else 0.0
    )
    if len(monthly) < 3:
        concentration_text = "月度样本不足，仅作区间摘要；至少覆盖 3 个自然月后再判断收益集中度。"
    elif concentration >= 0.70:
        concentration_text = "收益高度集中在少数月份，需复核是否由事件或极端行情驱动。"
    elif concentration >= 0.50:
        concentration_text = "收益有一定月份集中度，需确认是否依赖少数月份。"
    else:
        concentration_text = "月度收益分布相对分散，未显示明显单月依赖。"

    return {
        "positive": f"正收益月份 {positive_count}/{len(monthly)}",
        "best": (
            f"观察月份 {best_month}（{best_ret:.2%}）"
            if len(monthly) == 1
            else f"最佳月份 {best_month}（{best_ret:.2%}）"
        ),
        "worst": (
            "无跨月对比" if len(monthly) == 1 else f"最弱月份 {worst_month}（{worst_ret:.2%}）"
        ),
        "concentration": concentration_text,
    }


def _make_benchmark_chart(benchmark_result: Any) -> str | None:
    """基准对比图：策略净值 vs 基准净值（上）+ 超额净值（下）。"""
    if benchmark_result is None:
        return None
    daily = _safe_attr(benchmark_result, "daily")
    if daily is None or daily.is_empty():
        return None

    df = daily.to_pandas()
    if "trade_date" not in df.columns:
        return None
    if not {"strategy_nav", "benchmark_nav", "excess_nav"}.issubset(df.columns):
        return None
    df, x_col, is_date_axis = _with_plot_dates(df)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    benchmark_name = _safe_attr(benchmark_result, "benchmark_name", "基准")

    if "strategy_nav" in df.columns:
        df_sorted = df.sort_values("trade_date")
        ax1.plot(df_sorted[x_col], df_sorted["strategy_nav"], linewidth=1.4, label="策略")
    if "benchmark_nav" in df.columns:
        df_sorted = df.sort_values("trade_date")
        ax1.plot(
            df_sorted[x_col],
            df_sorted["benchmark_nav"],
            linewidth=1.2,
            linestyle="--",
            label=benchmark_name,
        )
    ax1.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax1.set_title(f"基准对比 — {benchmark_name}", fontsize=12)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))

    if "excess_nav" in df.columns:
        df_sorted = df.sort_values("trade_date")
        ax2.plot(
            df_sorted[x_col],
            df_sorted["excess_nav"],
            linewidth=1.2,
            color="#e74c3c",
            label="超额净值",
        )
        ax2.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
        ax2.set_title("超额净值", fontsize=11)
        ax2.legend(fontsize=8, loc="upper left")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))

    _format_sparse_x_axis(ax2, is_date_axis=is_date_axis)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _fig_to_base64(fig)


def _prepare_brinson_plot_frame(sector_df: pl.DataFrame, max_sectors: int = 12) -> pl.DataFrame:
    if sector_df.is_empty():
        return sector_df
    required = {"sector", "allocation", "selection", "interaction", "total_contribution"}
    if not required.issubset(set(sector_df.columns)):
        return sector_df

    clean = (
        sector_df.select(
            [
                "sector",
                pl.col("allocation").cast(pl.Float64).fill_nan(0.0).fill_null(0.0),
                pl.col("selection").cast(pl.Float64).fill_nan(0.0).fill_null(0.0),
                pl.col("interaction").cast(pl.Float64).fill_nan(0.0).fill_null(0.0),
                pl.col("total_contribution").cast(pl.Float64).fill_nan(0.0).fill_null(0.0),
            ]
        )
        .with_columns(pl.col("total_contribution").abs().alias("_abs_total"))
        .sort("_abs_total", descending=True)
    )
    if clean.height <= max_sectors:
        return clean.drop("_abs_total").sort("total_contribution")

    head = clean.head(max_sectors)
    other = (
        clean.slice(max_sectors)
        .select(["allocation", "selection", "interaction", "total_contribution"])
        .sum()
        .with_columns(pl.lit("其他").alias("sector"))
        .select(["sector", "allocation", "selection", "interaction", "total_contribution"])
    )
    return pl.concat([head.drop("_abs_total"), other]).sort("total_contribution")


def _make_attribution_chart(brinson_result: Any, barra_result: Any) -> str | None:
    """归因分析图：Brinson 行业堆积条形（上）+ Barra 风格暴露（下）。"""
    if brinson_result is None and barra_result is None:
        return None

    sector_df = _safe_attr(brinson_result, "sector_df") if brinson_result is not None else None
    exposures = _safe_attr(barra_result, "exposures", {}) if barra_result is not None else {}
    has_brinson_plot = sector_df is not None and not sector_df.is_empty()
    has_barra_plot = bool(exposures)
    if not has_brinson_plot and not has_barra_plot:
        return None

    n_panels = (1 if has_brinson_plot else 0) + (1 if has_barra_plot else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4.5 * n_panels))
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0

    if has_brinson_plot:
        assert sector_df is not None  # has_brinson_plot 已保证非空
        sector_df = _prepare_brinson_plot_frame(sector_df)
        sdf = sector_df.to_pandas()
        ax = axes[ax_idx]
        ax_idx += 1
        sectors = sdf["sector"].tolist() if "sector" in sdf.columns else list(range(len(sdf)))
        alloc = sdf["allocation"].tolist() if "allocation" in sdf.columns else [0] * len(sdf)
        selection = sdf["selection"].tolist() if "selection" in sdf.columns else [0] * len(sdf)
        interaction = (
            sdf["interaction"].tolist() if "interaction" in sdf.columns else [0] * len(sdf)
        )
        y_pos = range(len(sectors))
        ax.barh(
            list(y_pos),
            alloc,
            label="配置效应",
            color="#3498db",
            alpha=0.8,
        )
        ax.barh(
            list(y_pos),
            selection,
            left=alloc,
            label="选股效应",
            color="#2ecc71",
            alpha=0.8,
        )
        left_interaction = [a + s for a, s in zip(alloc, selection, strict=False)]
        ax.barh(
            list(y_pos),
            interaction,
            left=left_interaction,
            label="交互效应",
            color="#e74c3c",
            alpha=0.8,
        )
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(sectors, fontsize=9)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.6)
        ax.set_title("Brinson 行业归因", fontsize=12)
        ax.legend(fontsize=8, loc="best")

    if has_barra_plot:
        ax = axes[ax_idx]
        styles = list(exposures.keys())
        betas = [exposures[s] for s in styles]
        colors = ["#27ae60" if b >= 0 else "#e74c3c" for b in betas]
        y_pos = range(len(styles))
        ax.barh(list(y_pos), betas, color=colors, alpha=0.8)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(styles, fontsize=9)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.6)
        ax.set_title("Barra 风格因子暴露", fontsize=12)

    fig.tight_layout()
    return _fig_to_base64(fig)


def _make_walk_forward_chart(wf_result: Any) -> str | None:
    """滚动样本外图：样本外验证期拼接净值，以及历史观察期/样本外验证期 Sharpe。"""
    if wf_result is None:
        return None
    oos_returns = _safe_attr(wf_result, "oos_returns")
    folds = _safe_attr(wf_result, "folds", [])
    if oos_returns is None or oos_returns.is_empty() or not folds:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # 上：样本外验证期拼接净值
    oos_pd = oos_returns.to_pandas()
    if "trade_date" in oos_pd.columns and "nav" in oos_pd.columns:
        oos_pd, x_col, is_date_axis = _with_plot_dates(oos_pd)
        oos_pd = oos_pd.sort_values("trade_date")
        ax1.plot(
            oos_pd[x_col], oos_pd["nav"], linewidth=1.4, color="#2980b9", label="样本外验证期净值"
        )
        ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        _format_sparse_x_axis(ax1, is_date_axis=is_date_axis)
    ax1.set_title("滚动样本外验证期累计净值", fontsize=12)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax1.legend(fontsize=8)
    fig.autofmt_xdate()

    # 下：历史观察期 vs 样本外验证期 Sharpe 分折柱状图
    fold_ids = [f.fold_id for f in folds]
    is_sharpes = [f.is_sharpe for f in folds]
    oos_sharpes = [f.oos_sharpe for f in folds]
    x = range(len(fold_ids))
    width = 0.35
    ax2.bar(
        [xi - width / 2 for xi in x],
        is_sharpes,
        width,
        label="历史观察期 Sharpe",
        color="#3498db",
        alpha=0.7,
    )
    ax2.bar(
        [xi + width / 2 for xi in x],
        oos_sharpes,
        width,
        label="样本外验证期 Sharpe",
        color="#e74c3c",
        alpha=0.7,
    )
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([f"第 {fid} 折" for fid in fold_ids], fontsize=8)
    ax2.set_title("各折历史观察期 / 样本外验证期 Sharpe 对比", fontsize=12)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    return _fig_to_base64(fig)


def _make_quantile_spread_chart(grouped_returns: dict) -> str | None:
    """Q_N - Q_1 每日价差时序图 + 累计价差图。

    Args:
        grouped_returns: {quantile_int: list[float]} 每组日收益序列

    Returns:
        base64 PNG 字符串，或 None（数据不足时）
    """
    if not grouped_returns:
        return None
    keys = sorted(k for k in grouped_returns if isinstance(k, int))
    if len(keys) < 2:
        return None
    top_key = keys[-1]
    bot_key = keys[0]
    top_rets = np.array(grouped_returns[top_key], dtype=float)
    bot_rets = np.array(grouped_returns[bot_key], dtype=float)
    min_len = min(len(top_rets), len(bot_rets))
    if min_len == 0:
        return None
    spread = top_rets[:min_len] - bot_rets[:min_len]
    cum_spread = np.cumprod(1.0 + spread) - 1.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    x = np.arange(min_len)
    pos_mask = spread >= 0
    ax1.bar(x[pos_mask], spread[pos_mask], color="#27ae60", alpha=0.7, label="正价差")
    ax1.bar(x[~pos_mask], spread[~pos_mask], color="#e74c3c", alpha=0.7, label="负价差")
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax1.set_title(f"Q{top_key + 1} - Q{bot_key + 1} 每日价差", fontsize=12)
    ax1.legend(fontsize=8)

    ax2.plot(x, cum_spread, linewidth=1.4, color="#2980b9", label="累计价差")
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax2.set_title("累计价差", fontsize=12)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2%}"))
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return _fig_to_base64(fig)


def _extract_quantile_grouped_returns(bt_result: Any) -> dict[int, list[float]]:
    """Extract per-quantile return series from grouped NAV data."""
    nav = _safe_attr(bt_result, "nav")
    summary_stats = _safe_attr(bt_result, "summary_stats", {})
    if nav is None or nav.is_empty() or not summary_stats:
        return {}

    nav_pd = nav.to_pandas()
    if "group" not in nav_pd.columns or "nav" not in nav_pd.columns:
        return {}

    grouped_rets: dict[int, list[float]] = {}
    for group, frame in nav_pd.groupby("group"):
        if not isinstance(group, (int, float)):
            continue
        group_id = int(group)
        sort_col = "trade_date" if "trade_date" in frame.columns else None
        ordered = frame.sort_values(sort_col) if sort_col else frame
        navs = ordered["nav"].to_numpy(dtype=float)
        if len(navs) > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                rets = np.diff(navs) / navs[:-1]
            rets = [float(r) for r in rets if np.isfinite(r)]
        else:
            rets = []
        if rets:
            grouped_rets[group_id] = rets
    return grouped_rets


def _event_study_has_valid_window_series(event_study_result: Any) -> bool:
    if event_study_result is None:
        return False
    n_events = int(_safe_attr(event_study_result, "n_events", 0) or 0)
    windows = _safe_attr(event_study_result, "windows", [])
    avg_cumret = _safe_attr(event_study_result, "avg_cumret")
    if n_events <= 0 or avg_cumret is None or len(windows) == 0:
        return False
    try:
        return len(avg_cumret) == len(windows)
    except TypeError:
        return False


def _make_event_study_chart(event_study_result: Any) -> str | None:
    """事件研究图：平均累计收益 + 95% CI 阴影。

    Args:
        event_study_result: EventStudyResult（含 windows, avg_cumret, ci_95, n_events）

    Returns:
        base64 PNG 字符串，或 None
    """
    if event_study_result is None:
        return None
    windows = _safe_attr(event_study_result, "windows", [])
    avg_cumret = _safe_attr(event_study_result, "avg_cumret")
    ci_95 = _safe_attr(event_study_result, "ci_95")
    n_events = _safe_attr(event_study_result, "n_events", 0)
    if n_events <= 0:
        return None
    if not _event_study_has_valid_window_series(event_study_result):
        return None

    avg_cumret = np.asarray(avg_cumret)
    ci_95 = np.asarray(ci_95) if ci_95 is not None else np.zeros_like(avg_cumret)
    windows_arr = np.asarray(windows)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(windows_arr, avg_cumret, linewidth=1.5, color="#2980b9", label="平均累计收益")
    ax.fill_between(
        windows_arr,
        avg_cumret - ci_95,
        avg_cumret + ci_95,
        alpha=0.2,
        color="#2980b9",
        label="95% CI",
    )
    ax.axvline(x=0, color="#e74c3c", linestyle="--", linewidth=0.8, label="事件日")
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)
    ax.set_xlabel("相对事件日（交易日）")
    ax.set_ylabel("平均累计收益")
    ax.set_title(f"事件研究 — 事件前后平均累计收益（n={n_events}）", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2%}"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _factor_corr_has_valid_off_diagonal(corr_df: Any) -> bool:
    if (
        not isinstance(corr_df, pl.DataFrame)
        or corr_df.is_empty()
        or "factor" not in corr_df.columns
    ):
        return False
    factor_names = [str(name) for name in corr_df["factor"].to_list()]
    if len(factor_names) < 2:
        return False
    for row_idx, _row_name in enumerate(factor_names):
        for col_idx, col_name in enumerate(factor_names):
            if col_idx <= row_idx or col_name not in corr_df.columns:
                continue
            if _finite_float(corr_df[col_name][row_idx]) is not None:
                return True
    return False


def _factor_corr_is_multi_factor_input(corr_df: Any) -> bool:
    if (
        not isinstance(corr_df, pl.DataFrame)
        or corr_df.is_empty()
        or "factor" not in corr_df.columns
    ):
        return False
    return len(corr_df["factor"].to_list()) >= 2


def _make_factor_corr_heatmap(corr_df: Any) -> str | None:
    """因子相关性热力图（Spearman 相关矩阵）。

    Args:
        corr_df: pl.DataFrame，含 "factor" 列（行标签）及各因子列

    Returns:
        base64 PNG 字符串，或 None
    """
    if corr_df is None:
        return None
    try:
        factor_names = corr_df["factor"].to_list()
    except Exception:
        return None
    if len(factor_names) == 0:
        return None
    if not _factor_corr_has_valid_off_diagonal(corr_df):
        return None

    n = len(factor_names)
    mat = np.zeros((n, n))
    for i, fname in enumerate(factor_names):
        if fname in corr_df.columns:
            mat[:, i] = corr_df[fname].to_numpy()

    fig, ax = plt.subplots(figsize=(max(5, n * 0.9), max(4, n * 0.8)))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(factor_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(factor_names, fontsize=9)
    ax.set_title("因子截面 Rank 相关性矩阵（Spearman）", fontsize=12)

    # 注释数值
    for i in range(n):
        for j in range(n):
            text_color = "white" if abs(mat[i, j]) > 0.6 else "black"
            ax.text(
                j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color=text_color
            )

    fig.tight_layout()
    return _fig_to_base64(fig)


def _make_turnover_chart(to_result: Any) -> str | None:
    """换手率填充图。"""
    if to_result is None:
        return None
    dt = _safe_attr(to_result, "daily_turnover")
    if dt is None or dt.is_empty():
        return None

    dt_pd = dt.to_pandas()
    date_col = "trade_date" if "trade_date" in dt_pd.columns else dt_pd.columns[0]
    val_col = next((c for c in dt_pd.columns if c != date_col), None)
    if val_col is None:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    dt_pd, x_col, is_date_axis = _with_plot_dates(dt_pd, date_col)
    ax.fill_between(dt_pd[x_col], dt_pd[val_col], alpha=0.3, color="#9b59b6")
    ax.plot(dt_pd[x_col], dt_pd[val_col], linewidth=1.2, color="#8e44ad")
    ax.set_title("周期换手率", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1%}"))
    _format_sparse_x_axis(ax, is_date_axis=is_date_axis)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


# ── 辅助函数 ──────────────────────────────────────────────────────────


def _build_bt_summary_table(stats: dict, *, include_long_short: bool = True) -> list:
    """构建回测分组统计表格行。"""
    rows = []
    portfolio = stats.get("portfolio")
    if isinstance(portfolio, dict):
        rows.append(
            {
                "group": "组合收益",
                "ann_ret": _format_metric_percent(portfolio.get("ann_ret"), 2),
                "ann_vol": _format_metric_percent(portfolio.get("ann_vol"), 2),
                "sharpe": _format_metric_number(portfolio.get("sharpe"), 3),
                "max_dd": _format_metric_percent(portfolio.get("max_dd"), 2),
            }
        )
    group_keys = sorted([k for k in stats if isinstance(k, int)])
    for key in group_keys:
        gs = stats[key]
        if not isinstance(gs, dict):
            continue
        rows.append(
            {
                "group": f"Q{key + 1}",
                "ann_ret": _format_metric_percent(gs.get("ann_ret"), 2),
                "ann_vol": _format_metric_percent(gs.get("ann_vol"), 2),
                "sharpe": _format_metric_number(gs.get("sharpe"), 3),
                "max_dd": _format_metric_percent(gs.get("max_dd"), 2),
            }
        )
    if include_long_short and "long_short" in stats:
        ls = stats["long_short"]
        rows.append(
            {
                "group": "多空组合",
                "ann_ret": _format_metric_percent(ls.get("ann_ret"), 2),
                "ann_vol": _format_metric_percent(ls.get("ann_vol"), 2),
                "sharpe": _format_metric_number(ls.get("sharpe"), 3),
                "max_dd": _format_metric_percent(ls.get("max_dd"), 2),
            }
        )
    return rows


def _resolve_is_long_short(bt_result: Any, stats: dict[str, Any]) -> bool:
    """是否将该策略视为多空——概览与策略分页共用的唯一判定。

    概览总结和各策略分页必须走同一逻辑，否则同一策略可能在一份报告里被
    既描述为多头、又描述为多空（例如 factor_weighted + long_only=True）。
    """
    config = _safe_attr(bt_result, "config", {}) or {}
    config = config if isinstance(config, dict) else {}
    strategy_name = str(_safe_attr(bt_result, "strategy_name", "") or "")
    strategy_type = str(config.get("strategy_type") or strategy_name)
    lowered = strategy_type.lower()
    params = config.get("strategy_params")
    long_only = isinstance(params, dict) and params.get("long_only") is True

    if "topn" in lowered or "long_only" in lowered or long_only:
        return False
    if "long_short" in lowered or lowered.endswith("_ls") or "quantile_ls" in lowered:
        return True
    if "factor_weighted" in lowered:
        return True
    return "long_short" in stats


def _extract_metrics(
    ic_result,
    bt_result,
    to_result,
    advanced_results,
    pearson_ic_result=None,
    neutralized_ic_result=None,
) -> dict[str, Any]:
    """提取所有关键指标为扁平字典。"""
    m: dict[str, Any] = {}

    m["ic_mean"] = _finite_float(_safe_attr(ic_result, "ic_mean"))
    m["ic_std"] = _finite_float(_safe_attr(ic_result, "ic_std"))
    m["ir"] = _finite_float(_safe_attr(ic_result, "ir"))
    m["ic_positive_ratio"] = _finite_float(_safe_attr(ic_result, "ic_positive_ratio"))
    m["n_periods"] = _safe_attr(ic_result, "n_periods", 0) or 0
    m["decay"] = _safe_attr(ic_result, "decay", {})
    m["ic_tstat"] = _finite_float(_safe_attr(ic_result, "ic_tstat"))
    m["ic_pvalue"] = _finite_float(_safe_attr(ic_result, "ic_pvalue"))
    # Multi-period consistency: {horizon: {ic_mean, ic_std, ir, ic_positive_ratio}}
    multi_period = _safe_attr(ic_result, "multi_period", {})
    if multi_period:
        m["multi_period_table"] = [
            {
                "horizon": f"{h}d",
                "ic_mean": _finite_float(v.get("ic_mean")),
                "ic_std": _finite_float(v.get("ic_std")),
                "ir": _finite_float(v.get("ir")),
                "ic_pos": _finite_float(v.get("ic_positive_ratio")),
                "tstat": _finite_float(v.get("tstat")),
                "pvalue": _finite_float(v.get("pvalue")),
            }
            for h, v in sorted(multi_period.items())
        ]
    # Out-of-sample split
    oos_ic = _safe_attr(ic_result, "oos_ic", {})
    if oos_ic:
        train_ic = _finite_float(oos_ic.get("train"))
        test_ic = _finite_float(oos_ic.get("test"))
        if train_ic is not None and test_ic is not None:
            m["oos_train_ic"] = train_ic
            m["oos_test_ic"] = test_ic
        elif train_ic is not None or test_ic is not None:
            if train_ic is not None:
                m["oos_split_incomplete_detail"] = "已计算历史观察期 IC，缺少样本外验证期 IC"
            else:
                m["oos_split_incomplete_detail"] = "已计算样本外验证期 IC，缺少历史观察期 IC"

    if pearson_ic_result is not None:
        m["pearson_ic_mean"] = _finite_float(pearson_ic_result.ic_mean)
        m["pearson_ic_std"] = _finite_float(pearson_ic_result.ic_std)
        m["pearson_ir"] = _finite_float(pearson_ic_result.ir)
        m["pearson_ic_positive_ratio"] = _finite_float(pearson_ic_result.ic_positive_ratio)
        m["pearson_ic_tstat"] = _finite_float(pearson_ic_result.ic_tstat)

    if neutralized_ic_result is not None:
        m["neutralized_ic_mean"] = _finite_float(neutralized_ic_result.ic_mean)
        m["neutralized_ic_std"] = _finite_float(neutralized_ic_result.ic_std)
        m["neutralized_ir"] = _finite_float(neutralized_ic_result.ir)
        m["neutralized_ic_positive_ratio"] = _finite_float(neutralized_ic_result.ic_positive_ratio)
        m["neutralized_ic_tstat"] = _finite_float(neutralized_ic_result.ic_tstat)

    m["bt_stats"] = []
    m["bt_strategy_name"] = _safe_attr(bt_result, "strategy_name", "未运行") or "未运行"
    if bt_result is not None:
        stats = _safe_attr(bt_result, "summary_stats", {})
        if stats:
            m["bt_stats"] = _build_bt_summary_table(stats)
            primary_is_long_short = _resolve_is_long_short(bt_result, stats)
            primary_stats = (
                stats.get("long_short")
                if primary_is_long_short and isinstance(stats.get("long_short"), dict)
                else stats.get("portfolio")
            ) or {}
            m["primary_is_long_short"] = primary_is_long_short
            m["primary_return_label"] = "多空组合" if primary_is_long_short else "组合收益"
            m["primary_ann_ret"] = primary_stats.get("ann_ret")
            m["primary_sharpe"] = primary_stats.get("sharpe")
            m["primary_max_dd"] = primary_stats.get("max_dd")
            if "long_short" in stats:
                m["ls_ann_ret"] = stats["long_short"].get("ann_ret", 0)
                m["ls_sharpe"] = stats["long_short"].get("sharpe", 0)
                m["ls_max_dd"] = stats["long_short"].get("max_dd", 0)
        nav = _safe_attr(bt_result, "nav")
        if nav is not None and not nav.is_empty() and "trade_date" in nav.columns:
            dates = nav.select("trade_date").sort("trade_date")["trade_date"]
            m["effective_start"] = str(dates[0])
            m["effective_end"] = str(dates[-1])

    m["avg_turnover"] = _finite_float(_safe_attr(to_result, "avg_turnover"))

    if advanced_results:
        mono = advanced_results.get("mono")
        if mono:
            m["monotonicity_score"] = _safe_attr(mono, "monotonicity_score")

        acorr = advanced_results.get("autocorr")
        if acorr:
            m["rank_autocorr_available"] = True
            m["rank_autocorr"] = _finite_float(_safe_attr(acorr, "mean_autocorr"))
            m["half_life"] = _finite_float(_safe_attr(acorr, "half_life_est"))

        sector = advanced_results.get("sector")
        if sector:
            m["sector_ic"] = _safe_attr(sector, "sector_ic_df")

        sz = advanced_results.get("size")
        if sz:
            m["size_buckets"] = _safe_attr(sz, "buckets", {})

        regime = advanced_results.get("regime")
        regime_ic = _safe_attr(regime, "regime_ic")
        if isinstance(regime_ic, pl.DataFrame) and not regime_ic.is_empty():
            m["regime_type"] = _safe_attr(regime, "regime_type", "")
            m["regime_table"] = [
                {
                    "regime": str(row.get("regime", "")),
                    "label": _display_regime_label(row.get("regime")),
                    "ic": row.get("ic"),
                }
                for row in regime_ic.iter_rows(named=True)
            ]

        decay_list = advanced_results.get("decay_results", [])
        if decay_list:
            m["decay_table"] = [
                {"horizon": d.horizon, "ic_mean": d.ic_mean, "ic_std": d.ic_std} for d in decay_list
            ]

    return m


def _build_attribution_notice(
    attribution_result: Any,
    has_attribution_chart: bool,
    attribution_summary: dict[str, Any] | None,
) -> str | None:
    """Return a user-facing explanation when attribution output is unavailable."""
    if attribution_result is None:
        return (
            "组合归因未生成：当前流程未提供归因结果；Brinson 归因需要组合权重、"
            "基准权重及行业收益数据，Barra 风格归因需要组合收益和风格因子暴露。"
        )
    if not has_attribution_chart:
        if attribution_summary is not None:
            return "归因图未生成：已生成归因摘要，但缺少可绘图的行业或风格明细。"
        return "组合归因未生成：归因结果为空，或缺少可绘图的 Brinson/Barra 字段。"
    return None


def _build_predictive_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Summarize predictive power into a first-read IC conclusion."""
    ic_mean_raw = _finite_float(metrics.get("ic_mean"))
    ir_raw = _finite_float(metrics.get("ir"))
    positive_ratio_raw = _finite_float(metrics.get("ic_positive_ratio"))
    if ic_mean_raw is None or ir_raw is None or positive_ratio_raw is None:
        return {
            "headline": "核心 IC 指标样本不足，暂不判断预测方向。",
            "details": [
                "需先生成有效 Rank IC 均值、标准差、IR 和方向胜率后，再判断预测能力。",
                "当前不应把缺失值解释为零值信号。",
            ],
            "next_steps": ["检查前向收益、截面样本数和 IC 时序生成结果。"],
        }

    ic_mean = ic_mean_raw
    ir = ir_raw
    positive_ratio = positive_ratio_raw
    tstat = _num(metrics.get("ic_tstat"))
    pvalue = _num(metrics.get("ic_pvalue"), 1.0)

    if ic_mean > 0.01:
        direction = "Rank IC 方向为正"
    elif ic_mean < -0.01:
        direction = "Rank IC 方向为负"
    else:
        direction = "Rank IC 方向不明显"

    if abs(ir) >= 0.5:
        stability = "IC 稳定性较好"
    elif abs(ir) >= 0.2:
        stability = "IC 稳定性中等"
    else:
        stability = "IC 稳定性偏弱"

    if pvalue <= 0.05 or abs(tstat) >= 2.0:
        significance = "统计显著性较强"
    elif pvalue <= 0.10 or abs(tstat) >= 1.65:
        significance = "统计显著性边际成立"
    else:
        significance = "统计显著性不足"

    pearson = metrics.get("pearson_ic_mean")
    if pearson is None:
        pearson_text = "未生成 Pearson IC，对尾部敏感性暂不判断"
    else:
        pearson_val = _num(pearson)
        if abs(pearson_val) <= 0.005 or abs(ic_mean) <= 0.005:
            pearson_text = "Rank/Pearson 至少一项接近零，需检查信号是否主要由排序或极端值驱动"
        elif _same_direction(ic_mean, pearson_val):
            pearson_text = "Rank/Pearson 方向一致"
        else:
            pearson_text = "Rank/Pearson 方向不一致，需重点检查极端值和分布尾部"

    next_steps: list[str] = []
    if significance == "统计显著性不足":
        next_steps.append("下一步应优先检查样本长度、IC 分布和极端日期贡献。")
    if pearson is None:
        next_steps.append("补充 Pearson IC，判断预测能力是否依赖极端收益。")
    elif "不一致" in pearson_text:
        next_steps.append("对因子值和收益做分布诊断，必要时截尾或分段复核。")
    if abs(ic_mean) <= 0.01:
        next_steps.append("重新检查因子方向、收益窗口和预处理口径。")
    if not next_steps:
        next_steps.append("下一步应优先检查样本外、分组单调性和交易成本后的保留程度。")

    return {
        "headline": f"{direction}，{stability}，{significance}。",
        "details": [
            f"IC 均值 {ic_mean:.4f}，IR {ir:.2f}，IC 胜率 {positive_ratio * 100:.1f}%。",
            pearson_text + "。",
        ],
        "next_steps": next_steps[:3],
    }


def _build_neutralized_summary(metrics: dict[str, Any]) -> dict[str, Any] | None:
    """Summarize whether neutralized IC preserves the original signal."""
    if metrics.get("neutralized_ic_mean") is None:
        return None

    raw_ic = _finite_float(metrics.get("ic_mean"))
    neutral_ic = _finite_float(metrics.get("neutralized_ic_mean"))
    if raw_ic is None:
        return {
            "headline": "原始 Rank IC 样本不足，中性化保留率暂不具备解释意义。",
            "detail": "需先补齐原始 Rank IC，再判断行业/市值暴露是否解释了该信号。",
            "retention": None,
        }
    if neutral_ic is None:
        return {
            "headline": "中性化 Rank IC 样本不足，中性化保留率暂不具备解释意义。",
            "detail": "需先确认中性化回归或残差因子计算结果，再比较中性化前后信号。",
            "retention": None,
        }
    if abs(raw_ic) <= 1e-9:
        return {
            "headline": "原始 Rank IC 接近 0，中性化保留率暂不具备解释意义。",
            "detail": "需先确认原始信号方向，再判断行业/市值暴露影响。",
            "retention": None,
        }

    retention = abs(neutral_ic) / abs(raw_ic)
    if abs(neutral_ic) > 0.005 and not _same_direction(raw_ic, neutral_ic):
        headline = "中性化后方向反转，行业/市值暴露可能主导原始信号。"
        detail = "下一步应优先检查行业中性、市值中性和组合暴露。"
    elif retention >= 0.70:
        headline = "中性化后保留率较高，方向保持一致。"
        detail = "行业/市值暴露不能单独解释该信号，仍需结合样本外和交易成本复核。"
    elif retention >= 0.50:
        headline = "中性化后保留率中等，方向保持一致。"
        detail = "部分预测能力可能来自风格或行业暴露，建议比较中性化前后组合收益。"
    else:
        headline = "中性化后保留率偏低，原始信号可能受风格或行业暴露影响。"
        detail = "不宜只依据原始 IC 投产，应先复核暴露来源。"

    return {
        "headline": headline,
        "detail": detail,
        "retention": retention,
    }


def _build_holding_period_summary(metrics: dict[str, Any]) -> dict[str, str] | None:
    """Summarize multi-period IC into a recommended holding horizon."""
    rows = [
        row
        for row in metrics.get("multi_period_table", [])
        if _finite_float(row.get("ic_mean")) is not None
    ]
    if not rows:
        return None

    def _score(row: dict[str, Any]) -> float:
        pvalue = _num(row.get("pvalue"), 1.0)
        significance_bonus = 0.25 if pvalue <= 0.05 else (0.10 if pvalue <= 0.10 else 0.0)
        return (
            abs(_num(row.get("ir")))
            + abs(_num(row.get("ic_mean"))) * 10.0
            + max(0.0, abs(_num(row.get("ic_pos"), 0.5) - 0.5)) * 2.0
            + significance_bonus
        )

    best = max(rows, key=_score)
    best_horizon = str(best.get("horizon"))
    best_ic = _format_metric_number(best.get("ic_mean"), 4)
    best_ir = _format_metric_number(best.get("ir"), 2)
    best_win_rate = _format_metric_percent(best.get("ic_pos"), 1)
    directional_rows = [row for row in rows if abs(_num(row.get("ic_mean"))) >= 0.005]
    positive_count = sum(1 for row in directional_rows if _num(row.get("ic_mean")) > 0)
    negative_count = sum(1 for row in directional_rows if _num(row.get("ic_mean")) < 0)
    if directional_rows and positive_count == len(directional_rows):
        headline = (
            f"{best_horizon} 的综合表现最好，IC {best_ic}，IR {best_ir}，胜率 {best_win_rate}。"
        )
        direction = "各持有期方向一致，信号具备跨周期稳定性。"
        recommendation = f"建议以 {best_horizon} 作为优先验证的调仓周期，同时用长端结果监控衰减。"
    elif directional_rows and negative_count == len(directional_rows):
        headline = (
            f"{best_horizon} 的反向信号强度最集中，IC {best_ic}，"
            f"IR {best_ir}，正向胜率 {best_win_rate}。"
        )
        direction = "各持有期方向一致，但 IC 均为负，应按反向信号验证。"
        recommendation = (
            f"建议以 {best_horizon} 作为优先验证的反向调仓周期，同时用长端结果监控衰减。"
        )
    elif positive_count > 0 and negative_count > 0:
        headline = (
            f"{best_horizon} 的绝对 IC 强度最高，IC {best_ic}，IR {best_ir}，胜率 {best_win_rate}。"
        )
        direction = "持有期之间方向分化，信号可能依赖特定调仓周期。"
        recommendation = f"建议以 {best_horizon} 作为优先验证周期，但需分别测试正向和反向口径。"
    else:
        headline = (
            f"{best_horizon} 的综合证据相对最多，IC {best_ic}，IR {best_ir}，胜率 {best_win_rate}。"
        )
        direction = "部分持有期 IC 接近 0，跨周期稳定性仍需更多样本确认。"
        recommendation = f"建议先延长样本，再决定是否以 {best_horizon} 作为调仓周期。"

    return {
        "headline": headline,
        "direction": direction,
        "recommendation": recommendation,
    }


def _display_regime_label(regime: Any) -> str:
    labels = {
        "up": "上涨",
        "down": "下跌",
        "low_vol": "低波动",
        "mid_vol": "中波动",
        "high_vol": "高波动",
    }
    key = str(regime or "").strip()
    return labels.get(key, key or "未标记")


def _build_regime_summary(metrics: dict[str, Any]) -> dict[str, str] | None:
    rows = [
        row for row in metrics.get("regime_table", []) if _finite_float(row.get("ic")) is not None
    ]
    if not rows:
        return None

    by_regime = {str(row.get("regime")): _num(row.get("ic")) for row in rows}
    if "up" in by_regime and "down" in by_regime:
        up_ic = by_regime["up"]
        down_ic = by_regime["down"]
        if up_ic > 0.005 and down_ic < -0.005:
            headline = "上涨市场 IC 为正，下跌市场 IC 为负。"
            detail = "该信号可能依赖市场方向，应分别评估上涨/下跌环境下的组合暴露。"
        elif up_ic < -0.005 and down_ic > 0.005:
            headline = "上涨市场 IC 为负，下跌市场 IC 为正。"
            detail = "该信号可能偏防御或反向择时，应复核市场方向切分口径。"
        elif _same_direction(up_ic, down_ic):
            headline = "上涨和下跌市场 IC 方向一致。"
            detail = "信号对市场方向的依赖较弱，可继续结合样本外和交易成本验证。"
        else:
            headline = "上涨和下跌市场 IC 均接近 0。"
            detail = "当前市场状态切分未提供明确证据，需要更长样本确认。"
    else:
        strongest = max(rows, key=lambda row: abs(_num(row.get("ic"))))
        headline = (
            f"{strongest.get('label')}状态下 IC 绝对值最高（{_num(strongest.get('ic')):.4f}）。"
        )
        detail = "需比较不同状态下方向和强度，避免仅在单一市场环境中有效。"

    return {"headline": headline, "detail": detail}


def _build_benchmark_summary(benchmark_result: Any) -> dict[str, Any] | None:
    """Summarize benchmark-relative performance in reader-facing terms."""
    if benchmark_result is None:
        return None

    ann_excess = _finite_float(_safe_attr(benchmark_result, "ann_excess_ret"))
    ir = _finite_float(_safe_attr(benchmark_result, "information_ratio"))
    tracking_error = _finite_float(_safe_attr(benchmark_result, "tracking_error"))
    excess_dd = _finite_float(_safe_attr(benchmark_result, "excess_max_dd"))

    if ann_excess is None and ir is None and tracking_error is None and excess_dd is None:
        return None

    if ann_excess is None:
        direction = "超额收益样本不足"
    elif ann_excess > 0.02:
        direction = "跑赢基准"
    elif ann_excess < -0.02:
        direction = "跑输基准"
    else:
        direction = "接近基准"

    if ir is None:
        efficiency = "信息比率样本不足，暂不判断超额效率。"
    elif ir >= 0.5:
        efficiency = "单位主动风险带来的超额收益较好。"
    elif ir > 0:
        efficiency = "超额为正但效率一般，需结合回撤和成本确认。"
    else:
        efficiency = "主动风险未转化为正超额，基准相对收益需要谨慎。"

    if tracking_error is None:
        risk = "跟踪误差样本不足。"
    elif tracking_error >= 0.15:
        risk = "主动偏离较高，适合主动风险预算充足的组合。"
    elif tracking_error <= 0.05:
        risk = "主动偏离较低，更接近稳健增强口径。"
    else:
        risk = "主动偏离中等，需观察超额是否能覆盖波动。"

    if excess_dd is None:
        drawdown = "超额回撤样本不足。"
    elif excess_dd <= -0.10:
        drawdown = "超额回撤较深，需检查失效区间和止损口径。"
    else:
        drawdown = "超额回撤相对可控。"

    return {
        "direction": direction,
        "efficiency": efficiency,
        "risk": risk,
        "drawdown": drawdown,
    }


def _build_benchmark_notice(
    benchmark_result: Any,
    has_benchmark_chart: bool,
    benchmark_summary: dict[str, Any] | None,
) -> str | None:
    """Return a user-facing explanation when benchmark chart output is unavailable."""
    if benchmark_result is None or has_benchmark_chart:
        return None
    if benchmark_summary is not None:
        return "基准图未生成：已生成基准摘要，但缺少可绘图的日度净值明细。"
    return "基准图未生成：基准结果为空，或缺少可绘图的日度策略净值、基准净值和超额净值。"


def _build_attribution_summary(attribution_result: Any) -> dict[str, Any] | None:
    """Summarize Brinson attribution with dominant contribution and sector leaders."""
    if not isinstance(attribution_result, dict):
        return None

    brinson = attribution_result.get("brinson")
    if brinson is None:
        return None

    components = {
        "配置贡献": _finite_float(_safe_attr(brinson, "ann_allocation")),
        "选股贡献": _finite_float(_safe_attr(brinson, "ann_selection")),
        "交互贡献": _finite_float(_safe_attr(brinson, "ann_interaction")),
    }
    finite_components = {k: v for k, v in components.items() if v is not None}
    active = _finite_float(_safe_attr(brinson, "ann_active_return"))
    if not finite_components and active is None:
        return None

    if finite_components:
        driver_name, driver_value = max(finite_components.items(), key=lambda item: abs(item[1]))
        if abs(driver_value) < 0.005:
            driver_text = "超额贡献较分散，暂无单一主导来源。"
        elif driver_value > 0:
            driver_text = f"主要正贡献来自{driver_name}。"
        else:
            driver_text = f"主要拖累来自{driver_name}。"
    else:
        driver_text = "归因贡献项样本不足，暂不判断主导来源。"

    if active is None:
        active_text = "年化超额样本不足。"
    elif active > 0:
        active_text = "Brinson 口径下整体贡献为正。"
    elif active < 0:
        active_text = "Brinson 口径下整体贡献为负。"
    else:
        active_text = "Brinson 口径下整体贡献接近零。"

    sector_text = ""
    sector_df = _safe_attr(brinson, "sector_df")
    if (
        isinstance(sector_df, pl.DataFrame)
        and not sector_df.is_empty()
        and "total_contribution" in sector_df.columns
    ):
        sorted_df = sector_df.with_columns(
            pl.col("total_contribution").cast(pl.Float64).fill_nan(0.0).fill_null(0.0)
        ).sort("total_contribution")
        low = sorted_df.row(0, named=True)
        high = sorted_df.row(sorted_df.height - 1, named=True)
        sector_text = (
            f"行业层面最大正贡献来自 {high.get('sector', '未知行业')}，"
            f"最大拖累来自 {low.get('sector', '未知行业')}。"
        )

    return {
        "active": active_text,
        "driver": driver_text,
        "sector": sector_text,
    }


def _build_factor_corr_summary(factor_corr: Any) -> dict[str, Any] | None:
    """Summarize cross-factor information overlap from a Spearman matrix."""
    if not isinstance(factor_corr, pl.DataFrame) or factor_corr.is_empty():
        return None
    if "factor" not in factor_corr.columns:
        return None

    factor_names = [str(name) for name in factor_corr["factor"].to_list()]
    if len(factor_names) < 2:
        return None

    strongest_pair = ("", "")
    strongest_corr = 0.0
    abs_corrs: list[float] = []
    high_count = 0

    for row_idx, row_name in enumerate(factor_names):
        for col_idx, col_name in enumerate(factor_names):
            if col_idx <= row_idx or col_name not in factor_corr.columns:
                continue
            corr = _finite_float(factor_corr[col_name][row_idx])
            if corr is None:
                continue
            abs_corr = abs(corr)
            abs_corrs.append(abs_corr)
            if abs_corr >= 0.7:
                high_count += 1
            if abs_corr > abs(strongest_corr):
                strongest_corr = corr
                strongest_pair = (row_name, col_name)

    if not abs_corrs:
        return {
            "headline": "相关性矩阵缺少有效的非对角元素。",
            "detail": "需要检查输入矩阵列名是否与 factor 行标签一致。",
        }

    mean_abs = float(np.mean(abs_corrs))
    if high_count:
        detail = f"存在 {high_count} 组 |相关性| >= 0.70 的高重叠因子，组合前应考虑去冗余或降权。"
    elif mean_abs >= 0.4:
        detail = "平均相关性中等，因子组合仍需关注信息重复。"
    else:
        detail = "整体相关性较低，多因子信息互补性较好。"

    pair_text = f"{strongest_pair[0]} / {strongest_pair[1]}"
    return {
        "headline": f"最高重叠因子对为 {pair_text}，相关性 {strongest_corr:.2f}。",
        "detail": detail,
        "mean_abs": mean_abs,
    }


def _display_factor_rating_label(label: str) -> str:
    return _format_label_with_code(
        label,
        {
            "production_watch": "生产观察",
            "candidate": "候选",
            "research": "研究观察",
            "weak": "偏弱",
            "invalid": "不可用",
        },
    )


def _rating_component_readout(name: str, value: float, max_score: int) -> str:
    """Explain a rating component in research-action terms."""
    ratio = value / max_score if max_score else 0.0
    if ratio >= 0.75:
        strength = "较强"
    elif ratio >= 0.45:
        strength = "中等"
    else:
        strength = "偏弱"

    readouts = {
        "Alpha 强度": f"Alpha 证据{strength}；重点看 IC、IR、显著性和多空收益是否同向支持。",
        "稳定性": f"稳定性{strength}；继续核对样本外、IC 衰减和跨持有期方向。",
        "可交易性": f"交易落地{strength}；重点看执行成本和换手、成交约束、回撤压力。",
        "鲁棒性": f"鲁棒性{strength}；重点看中性化、Pearson/Rank 一致性和行业/市值分层。",
        "结构质量": f"结构质量{strength}；重点看分组单调性和信号持续性，决定组合构建方式。",
    }
    return readouts.get(name, f"该维度{strength}；需要结合对应明细模块复核。")


def _generate_summary_text(
    factor_name: str,
    metrics: dict[str, Any],
    llm_explanation: dict[str, Any] | None = None,
) -> str:
    """生成总结解读文本，包含星级评级。"""
    star_char = chr(9733)  # ★
    rating_result = metrics.get("factor_rating")
    if not isinstance(rating_result, FactorRating):
        rating_result = _compute_factor_rating(metrics)
    stars = rating_result.stars
    base_stars = _stars_from_score(rating_result.score)
    rating = star_char * stars + chr(9734) * (5 - stars)
    rating_cap_note = ""
    if rating_result.caps and base_stars > stars:
        rating_cap_note = (
            f" | <strong>评级说明：</strong>原始得分对应 {base_stars} 星，评级上限后为 {stars} 星"
        )

    lines = [
        f"<p><strong>评级: {rating} ({stars}/5)</strong></p>",
        (
            f"<p><strong>评分卡总分：</strong>{rating_result.score:.1f}/100 "
            f"| <strong>评级标签：</strong>{_display_factor_rating_label(rating_result.label)}"
            f"{rating_cap_note}</p>"
        ),
        "<table><tr><th>维度</th><th>得分</th><th>读法 / 下一步</th></tr>",
    ]
    for name, value in rating_result.components.items():
        max_score = RATING_COMPONENT_MAX_SCORES.get(name, 0)
        readout = _rating_component_readout(name, value, max_score)
        lines.append(f"<tr><td>{name}</td><td>{value:.1f}/{max_score}</td><td>{readout}</td></tr>")
    lines.append("</table>")

    if rating_result.positives:
        lines.append(f"<p><strong>主要优势：</strong>{'、'.join(rating_result.positives)}</p>")
    if rating_result.caps:
        lines.append('<div class="rating-caps"><div class="rating-caps-title">评级上限</div>')
        for item in rating_result.caps:
            lines.append(f'<div class="rating-cap-item">{item}</div>')
        lines.append("</div>")
    elif rating_result.warnings:
        lines.append(f"<p><strong>主要风险：</strong>{'、'.join(rating_result.warnings)}</p>")

    ic_mean = _finite_float(metrics.get("ic_mean"))
    ir = _finite_float(metrics.get("ir"))

    if ic_mean is None:
        lines.append(
            '<p class="summary-note">核心 IC 指标样本不足，暂不判断预测方向；需先补齐有效 IC 时序。</p>'
        )
    elif abs(ic_mean) < 0.01:
        lines.append(
            '<p class="summary-note">IC 均值极低（|IC| &lt; 0.01），因子对收益的预测能力非常有限。</p>'
        )
    elif ic_mean > 0.03:
        lines.append(
            f'<p class="summary-note">IC 均值 {ic_mean:.4f}（Spearman &rho;），因子展现出较强的正向预测能力。</p>'
        )
    elif ic_mean < -0.03:
        lines.append(
            f'<p class="summary-note">IC 均值 {ic_mean:.4f}，因子呈现显著的负向预测能力（可用作反向因子）。</p>'
        )
    else:
        lines.append(f'<p class="summary-note">IC 均值 {ic_mean:.4f}，因子具备一定的预测能力。</p>')

    if ir is not None and ir > 0.5:
        lines.append(
            f'<p class="summary-note">信息比率 IR = {ir:.2f}，因子稳定性良好，IC 方向一致性高。</p>'
        )
    elif ir is not None and ir > 0.2:
        lines.append(f'<p class="summary-note">信息比率 IR = {ir:.2f}，因子稳定性一般。</p>')

    primary_ret = metrics.get("primary_ann_ret")
    if primary_ret is None:
        primary_ret = metrics.get("ls_ann_ret") or 0
    if metrics.get("primary_is_long_short"):
        if primary_ret > 0.05:
            lines.append(
                f'<p class="summary-note">多空年化收益 {primary_ret * 100:.1f}%，分层效果显著，Top-Bottom 区分度高。</p>'
            )
        elif primary_ret > 0:
            lines.append(
                f'<p class="summary-note">多空年化收益 {primary_ret * 100:.1f}%，分层效果较弱。</p>'
            )
        elif primary_ret < 0:
            lines.append(
                '<p class="summary-note">多空收益为负，分层回测不理想，因子分组单调性需关注。</p>'
            )
    else:
        if primary_ret > 0.05:
            lines.append(
                f'<p class="summary-note">主策略组合年化收益 {primary_ret * 100:.1f}%，收益表现较强；需结合多空策略页确认因子纯选股价差。</p>'
            )
        elif primary_ret > 0:
            lines.append(
                f'<p class="summary-note">主策略组合年化收益 {primary_ret * 100:.1f}%，收益表现偏弱，需结合回撤、换手和多空策略验证。</p>'
            )
        elif primary_ret < 0:
            lines.append(
                '<p class="summary-note">主策略组合收益为负，组合构建或因子方向需重点复核。</p>'
            )

    if llm_explanation is not None:
        evidence = str(llm_explanation.get("evidence_assessment", "")).strip()
        usage = str(llm_explanation.get("usage_suggestion", "")).strip()
        rating_label = str(llm_explanation.get("rating", "")).strip()
        confidence = str(llm_explanation.get("confidence", "")).strip()
        if evidence or usage:
            label = f"{rating_label}/{confidence}".strip("/")
            label_html = f"（{label}）" if label else ""
            lines.append(f"<p><strong>LLM 综合结论{label_html}：</strong>{evidence}</p>")
            if usage:
                lines.append(f"<p><strong>LLM 使用建议：</strong>{usage}</p>")

    return "\n".join(lines)


def _slugify_strategy_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return slug or "strategy"


def _strategy_portfolio_stats(bt_result: Any) -> dict[str, float]:
    stats = _safe_attr(bt_result, "summary_stats", {}) or {}
    return stats.get("portfolio") or stats.get("long_short") or {}


def _infer_strategy_type(name: str, bt_result: Any) -> str:
    config = _safe_attr(bt_result, "config", {}) or {}
    configured = config.get("strategy_type") if isinstance(config, dict) else None
    raw = configured or _safe_attr(bt_result, "strategy_name", name) or name
    return str(raw)


def _strategy_exposure_label(strategy_type: str, is_long_short: bool) -> str:
    lowered = strategy_type.lower()
    if "topn" in lowered or "long_only" in lowered:
        return "多头 TopN"
    if ("quantile" in lowered or "quantile_ls" in lowered) and is_long_short:
        return "分位数组合多空"
    if "factor_weighted" in lowered and is_long_short:
        return "因子加权多空"
    if "optimizer" in lowered:
        return "优化组合"
    return "多空" if is_long_short else "组合"


def _display_strategy_type(strategy_type: str) -> str:
    labels = {
        "topn_long_only": "TopN 多头",
        "quantile_long_short": "分位数组合多空",
        "factor_weighted": "因子加权",
        "optimizer_strategy": "优化器组合",
    }
    return labels.get(strategy_type, strategy_type or "未指定")


def _display_strategy_name(name: str, strategy_type: str, bt_result: Any) -> str:
    config = _safe_attr(bt_result, "config", {}) or {}
    params = config.get("strategy_params") if isinstance(config, dict) else {}
    params = params if isinstance(params, dict) else {}
    explicit = {
        "topn_50": "TopN 多头 50",
        "quantile_ls_5": "五分位多空",
        "factor_weighted_ls": "因子加权多空",
        "optimizer_mv_long_only": "均值-方差优化多头",
    }
    if name in explicit:
        return explicit[name]

    lowered = f"{name} {strategy_type}".lower()
    if "optimizer" in lowered:
        optimizer = _display_strategy_param_value("optimizer", params.get("optimizer", ""))
        return f"{optimizer}优化组合" if optimizer else "优化组合"
    if "topn" in lowered:
        top_n = params.get("top_n") or re.search(r"topn[_-]?(\d+)", lowered)
        top_n_value = top_n.group(1) if isinstance(top_n, re.Match) else top_n
        return f"TopN 多头 {top_n_value}" if top_n_value else "TopN 多头"
    if "quantile" in lowered or "quantile_ls" in lowered:
        quantiles = params.get("quantiles") or re.search(r"(?:quantile_ls|ls)[_-]?(\d+)", lowered)
        quantile_value = quantiles.group(1) if isinstance(quantiles, re.Match) else quantiles
        return f"{quantile_value} 分位多空" if quantile_value else "分位数组合多空"
    if "factor_weighted" in lowered:
        return "因子加权多空" if "ls" in lowered or "long_short" in lowered else "因子加权组合"
    return _display_strategy_type(strategy_type)


def _strategy_constraints(bt_result: Any) -> dict[str, Any]:
    config = _safe_attr(bt_result, "config", {}) or {}
    if not isinstance(config, dict):
        return {}
    keys = ("cost_model", "max_abs_weight", "rebalance_threshold", "alpha", "fallback_adv")
    return {key: config.get(key) for key in keys if config.get(key) is not None}


def _format_strategy_value(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value)


def _display_strategy_param_value(key: str, value: Any) -> str:
    value_labels = {
        "optimizer": {
            "mean_variance": "均值-方差",
            "max_sharpe": "最大夏普",
            "risk_parity": "风险平价",
        },
        "cov_estimator": {
            "ledoit_wolf": "Ledoit-Wolf 收缩",
            "sample": "样本协方差",
        },
    }
    raw = _format_strategy_value(value)
    return value_labels.get(key, {}).get(raw, raw)


def _display_cost_model(cost_model: Any) -> str:
    labels = {
        "linear": "线性成本",
        "square_root_impact": "平方根冲击成本",
    }
    key = str(cost_model or "").strip()
    return labels.get(key, key or "未指定")


def _strategy_params_summary(bt_result: Any) -> str:
    config = _safe_attr(bt_result, "config", {}) or {}
    if not isinstance(config, dict):
        return "未记录"
    params = config.get("strategy_params") or {}
    if not isinstance(params, dict) or not params:
        return "默认参数"

    priority = (
        "optimizer",
        "risk_aversion",
        "lookback_days",
        "cov_estimator",
        "top_n",
        "quantiles",
        "long_only",
        "gross_exposure",
        "net_exposure",
        "max_weight",
    )
    labels = {
        "optimizer": "优化器",
        "risk_aversion": "风险厌恶",
        "lookback_days": "协方差回看天数",
        "cov_estimator": "协方差估计",
        "top_n": "TopN 数量",
        "quantiles": "分位组数",
        "long_only": "仅多头",
        "gross_exposure": "总敞口",
        "net_exposure": "净敞口",
        "max_weight": "单票权重上限",
        "factor_col": "因子列",
        "long_exposure": "多头敞口",
    }
    ordered_keys = [key for key in priority if key in params]
    ordered_keys.extend(sorted(key for key in params if key not in ordered_keys))
    parts = [
        f"{labels.get(key, key)}={_display_strategy_param_value(key, params[key])}"
        for key in ordered_keys
        if params.get(key) is not None
    ]
    return "；".join(parts) if parts else "默认参数"


def _strategy_stat(stats: dict[str, Any], key: str, default: float = 0.0) -> float:
    return _num(stats.get(key), default)


def _display_trade_constraint_reason(reason: Any) -> str:
    labels = {
        "capacity": "成交容量限制",
        "limit_up": "涨停限制",
        "limit_down": "跌停限制",
        "suspended": "停牌或无成交",
        "missing_price": "缺少价格数据",
        "invalid_portfolio_value": "组合市值无效",
    }
    key = str(reason or "").strip()
    return labels.get(key, key or "未指定")


def _build_trade_summary(bt_result: Any) -> dict[str, Any]:
    trades = _safe_attr(bt_result, "trades")
    summary = {
        "trade_count": 0,
        "constrained_count": 0,
        "constraint_rate": 0.0,
        "constraint_reasons": [],
    }
    if trades is None or trades.is_empty() or "block_reason" not in trades.columns:
        return summary

    summary["trade_count"] = trades.height
    blocked = trades.filter(pl.col("block_reason").is_not_null() & (pl.col("block_reason") != ""))
    summary["constrained_count"] = blocked.height
    summary["constraint_rate"] = blocked.height / trades.height if trades.height else 0.0
    if not blocked.is_empty():
        reason_rows = (
            blocked.group_by("block_reason")
            .len()
            .sort(["len", "block_reason"], descending=[True, False])
            .iter_rows(named=True)
        )
        summary["constraint_reasons"] = [
            {
                "reason": _display_trade_constraint_reason(row["block_reason"]),
                "count": row["len"],
            }
            for row in reason_rows
        ]
    return summary


def _build_strategy_quality_summary(
    portfolio_stats: dict[str, Any],
    trade_summary: dict[str, Any],
) -> dict[str, Any]:
    """Translate per-strategy return and execution metrics into reader-facing labels."""
    ann_ret = _finite_float(portfolio_stats.get("ann_ret"))
    sharpe = _finite_float(portfolio_stats.get("sharpe"))
    max_dd = _finite_float(portfolio_stats.get("max_dd"))
    avg_turnover = _finite_float(portfolio_stats.get("avg_turnover"))
    total_cost = _finite_float(portfolio_stats.get("total_cost"))
    constraint_rate = _num(trade_summary.get("constraint_rate"))

    if sharpe is None:
        sharpe_text = "Sharpe 样本不足"
    elif sharpe >= 1.0:
        sharpe_text = "Sharpe 较好"
    elif sharpe >= 0.5:
        sharpe_text = "Sharpe 较弱"
    else:
        sharpe_text = "Sharpe 偏低"

    if max_dd is None:
        drawdown_text = "回撤样本不足"
    elif max_dd >= -0.10:
        drawdown_text = "回撤可控"
    elif max_dd >= -0.20:
        drawdown_text = "回撤中等"
    else:
        drawdown_text = "回撤偏深"

    if avg_turnover is None:
        turnover_text = "换手样本不足"
    elif avg_turnover <= 0:
        turnover_text = "换手未记录"
    elif avg_turnover <= 0.20:
        turnover_text = "换手较低"
    elif avg_turnover <= 0.60:
        turnover_text = "换手适中"
    else:
        turnover_text = "换手偏高"

    if total_cost is None:
        cost_text = "成本样本不足"
    elif total_cost <= 0.01:
        cost_text = "成本可控"
    elif total_cost <= 0.03:
        cost_text = "成本需关注"
    else:
        cost_text = "成本压力较高"

    if constraint_rate <= 0:
        constraint_text = "暂无成交约束记录"
    elif constraint_rate <= 0.05:
        constraint_text = "成交约束较少"
    else:
        constraint_text = "成交约束需重点复核"

    if ann_ret is None or sharpe is None or max_dd is None:
        conclusion = "收益指标样本不足，暂不参与策略优选；需先补齐回测统计。"
    elif ann_ret > 0 and sharpe >= 0.5 and max_dd >= -0.10:
        conclusion = "收益为正且回撤可控，可继续结合样本外和成本压力复核。"
    elif ann_ret > 0:
        conclusion = "收益为正，但风险调整后质量仍需复核。"
    else:
        conclusion = "当前收益不足，暂不宜单独作为策略选择依据。"

    return {
        "conclusion": conclusion,
        "labels": [sharpe_text, drawdown_text, turnover_text, cost_text, constraint_text],
    }


def _build_execution_summary(strategy_pages: list[dict[str, Any]]) -> dict[str, str] | None:
    if not strategy_pages:
        return None

    constraint_pages = [
        page
        for page in strategy_pages
        if _num(page.get("trade_summary", {}).get("trade_count")) > 0
    ]
    turnover_pages = [
        page for page in strategy_pages if _finite_float(page.get("avg_turnover")) is not None
    ]
    cost_pages = [
        page for page in strategy_pages if _finite_float(page.get("total_cost")) is not None
    ]
    if not constraint_pages and not turnover_pages and not cost_pages:
        return {
            "headline": "交易执行指标样本不足。",
            "risk": "暂不判断执行瓶颈；当前报告缺少成交记录、换手率和交易成本统计。",
            "action": "下一步应补齐调仓成交记录、组合换手和成本字段，再比较策略可执行性。",
        }

    highest_constraint = (
        max(
            constraint_pages,
            key=lambda page: _num(page.get("trade_summary", {}).get("constraint_rate")),
        )
        if constraint_pages
        else None
    )
    highest_turnover = (
        max(turnover_pages, key=lambda page: _finite_float(page.get("avg_turnover")) or 0.0)
        if turnover_pages
        else None
    )
    highest_cost = (
        max(cost_pages, key=lambda page: _finite_float(page.get("total_cost")) or 0.0)
        if cost_pages
        else None
    )

    constraint_rate = (
        _num(highest_constraint.get("trade_summary", {}).get("constraint_rate"))
        if highest_constraint is not None
        else None
    )
    constraint_name = (
        str(highest_constraint.get("display_name", "")).strip()
        if highest_constraint is not None
        else ""
    )
    turnover_name = (
        str(highest_turnover.get("display_name", "")).strip()
        if highest_turnover is not None
        else ""
    )
    cost_name = (
        str(highest_cost.get("display_name", "")).strip() if highest_cost is not None else ""
    )
    if constraint_rate is not None:
        headline = f"成交约束占比最高：{constraint_name}（{constraint_rate:.1%}）。"
    else:
        headline = "成交约束记录样本不足。"

    if constraint_rate is not None and constraint_rate >= 0.20:
        risk = "当前执行风险较高，需先复核约束来源和可成交容量。"
    elif constraint_rate is not None and constraint_rate > 0:
        risk = "当前存在少量执行约束，需在扩大资金规模前复核成交可得性。"
    elif (
        highest_turnover is not None
        and (_finite_float(highest_turnover.get("avg_turnover")) or 0.0) >= 0.80
    ):
        risk = f"{turnover_name} 换手偏高，交易成本和冲击成本需要压力测试。"
    elif (
        highest_cost is not None and (_finite_float(highest_cost.get("total_cost")) or 0.0) >= 0.03
    ):
        risk = f"{cost_name} 成本压力较高，应优先比较扣费后净收益。"
    elif constraint_rate is None:
        risk = "成交约束记录样本不足，但换手和成本未显示明显压力。"
    else:
        risk = "当前策略组未显示明显执行瓶颈，仍需在更长样本和更大资金规模下复核。"

    reasons = (
        highest_constraint.get("trade_summary", {}).get("constraint_reasons")
        if highest_constraint is not None
        else []
    ) or []
    if reasons:
        action = f"优先复核{reasons[0]['reason']}，再检查调仓阈值、单票权重上限和成交量约束。"
    else:
        action = "优先比较低换手、低成本且约束占比较低的策略作为执行候选。"

    return {
        "headline": headline,
        "risk": risk,
        "action": action,
    }


def _display_quality_check_name(name: Any) -> str:
    labels = {
        "factor_value": "原始因子值",
        "factor_clean": "清洗后因子值",
        "forward_return": "前向收益",
        "price": "价格数据",
        "universe": "股票池",
    }
    key = str(name or "").strip()
    return labels.get(key, key or "未指定")


def _display_quality_message(message: Any) -> str:
    text = str(message or "").strip()
    replacements = {
        "factor_value": "原始因子值",
        "factor_clean": "清洗后因子值",
        "forward_return": "前向收益",
        "coverage is low": "覆盖率偏低",
        "has null values": "存在缺失值",
        "has infinite values": "存在无限值",
    }
    for raw, label in replacements.items():
        text = text.replace(raw, label)
    text = re.sub(r"\s*:\s*", "：", text)
    return re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)


def _build_quality_summary(quality_report: dict[str, Any] | None) -> dict[str, Any]:
    def _finite_count(value: Any) -> int | None:
        numeric = _finite_float(value)
        return None if numeric is None else int(numeric)

    if not quality_report:
        return {
            "status": "未传入",
            "warnings": [],
            "errors": [],
            "checks": [],
            "detail": "当前流程未提供结构化质量摘要；完整质量文件以 run 目录中的 quality.json 为准。",
        }
    checks: list[dict[str, Any]] = []
    raw_checks = quality_report.get("checks", {})
    if isinstance(raw_checks, dict):
        for name, payload in raw_checks.items():
            if not isinstance(payload, dict):
                continue
            checks.append(
                {
                    "name": _display_quality_check_name(name),
                    "rows": _finite_count(payload.get("rows")),
                    "coverage": _finite_float(payload.get("coverage")),
                    "valid_count": _finite_count(payload.get("valid_count")),
                    "null_count": _finite_count(payload.get("null_count")),
                    "inf_count": _finite_count(payload.get("inf_count")),
                }
            )
    return {
        "status": _display_status(quality_report.get("status", "unknown")),
        "warnings": [_display_quality_message(item) for item in quality_report.get("warnings", [])],
        "errors": [_display_quality_message(item) for item in quality_report.get("errors", [])],
        "checks": checks,
        "detail": "数据质量检查已生成，覆盖率、缺失值和错误会影响报告结论可信度。",
    }


def _display_status(raw_status: Any) -> str:
    status = str(raw_status or "").strip()
    labels = {
        "ok": "正常",
        "pass": "正常",
        "passed": "正常",
        "warning": "需关注",
        "warn": "需关注",
        "error": "失败",
        "failed": "失败",
        "unknown": "未知",
        "insufficient_data": "样本不足",
        "not_run": "未运行",
        "skipped": "已跳过",
    }
    return labels.get(status.lower(), status or "未知")


def _format_label_with_code(value: Any, labels: dict[str, str]) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "未提供"
    label = labels.get(raw.lower())
    return f"{label}（{raw}）" if label else raw


def _prepare_llm_explanation_view(llm_explanation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(llm_explanation, dict):
        return None

    view = dict(llm_explanation)
    view["rating_display"] = _format_label_with_code(
        view.get("rating"),
        {
            "strong": "强",
            "moderate": "中等",
            "weak": "偏弱",
            "invalid": "不可用",
        },
    )
    view["confidence_display"] = _format_label_with_code(
        view.get("confidence"),
        {
            "high": "高",
            "medium": "中",
            "low": "低",
        },
    )
    return view


def _build_report_statuses(
    *,
    metrics: dict[str, Any],
    quality_summary: dict[str, Any],
    benchmark_result: Any,
    benchmark_summary: dict[str, Any] | None,
    benchmark_notice: str | None,
    attribution_notice: str | None,
    attribution_summary: dict[str, Any] | None,
    walk_forward_result: Any,
    walk_forward_summary: dict[str, Any] | None,
    event_study_result: Any,
    has_regime_ic: bool,
    factor_corr: Any,
    factor_corr_summary: dict[str, Any] | None,
) -> list[dict[str, str]]:
    statuses: list[dict[str, str]] = []
    statuses.append(
        {
            "module": "数据质量",
            "status": _display_status(quality_summary.get("status", "未传入")),
            "detail": str(quality_summary.get("detail", "")),
        }
    )
    if benchmark_result is None:
        benchmark_status = "未启用或未生成"
        benchmark_detail = "当前流程未提供基准对比结果，报告不展示基准超额曲线。"
    elif benchmark_notice is None:
        benchmark_status = "已生成"
        benchmark_detail = "展示策略相对基准的超额收益、IR、跟踪误差与超额回撤。"
    elif benchmark_summary is not None:
        benchmark_status = "需关注"
        benchmark_detail = "已生成基准摘要，但缺少可绘图的日度净值明细。"
    else:
        benchmark_status = "样本不足"
        benchmark_detail = benchmark_notice
    statuses.append(
        {
            "module": "基准超额",
            "status": benchmark_status,
            "detail": benchmark_detail,
        }
    )
    if attribution_notice is None:
        attribution_status = "已生成"
        attribution_detail = "已生成 Brinson/Barra 归因图或归因摘要。"
    elif attribution_summary is not None:
        attribution_status = "需关注"
        attribution_detail = "已生成归因摘要，但缺少可绘图的行业或风格明细。"
    else:
        attribution_status = "未生成"
        attribution_detail = attribution_notice
    statuses.append(
        {
            "module": "组合归因（Brinson/Barra）",
            "status": attribution_status,
            "detail": attribution_detail,
        }
    )
    has_oos_split = (
        _finite_float(metrics.get("oos_train_ic")) is not None
        and _finite_float(metrics.get("oos_test_ic")) is not None
    )
    oos_incomplete_detail = metrics.get("oos_split_incomplete_detail")
    if has_oos_split:
        oos_status = "已生成"
        oos_detail = "已生成历史观察期与样本外验证期 IC 分割。"
    elif oos_incomplete_detail is not None:
        oos_status = "样本不足"
        oos_detail = f"样本外 IC 分割不完整：{oos_incomplete_detail}。"
    else:
        oos_status = "未启用或未生成"
        oos_detail = "当前流程未提供完整的历史观察期与样本外验证期 IC 分割。"
    statuses.append({"module": "样本外 IC 分割", "status": oos_status, "detail": oos_detail})
    if walk_forward_result is not None:
        wf_status = "已生成"
        wf_detail = "已生成滚动验证折数与样本外验证期绩效。"
    elif walk_forward_summary is not None:
        raw_wf_status = str(walk_forward_summary.get("status", "未生成"))
        wf_status = _display_status(raw_wf_status)
        wf_detail = (
            "样本不足，未生成滚动验证折数。"
            if raw_wf_status == "insufficient_data"
            else str(walk_forward_summary.get("error") or "已生成滚动样本外摘要。")
        )
    else:
        wf_status = "未启用或未生成"
        wf_detail = "当前流程未提供滚动样本外结果或摘要。"
    statuses.append({"module": "滚动样本外", "status": wf_status, "detail": wf_detail})
    if event_study_result is None:
        event_status = "未启用或未生成"
        event_detail = "未传入事件研究结果。"
    elif int(_safe_attr(event_study_result, "n_events", 0) or 0) <= 0:
        event_status = "无有效事件"
        event_detail = "事件研究已运行，但没有满足阈值和事件窗口要求的有效事件。"
    elif not _event_study_has_valid_window_series(event_study_result):
        event_status = "样本不足"
        event_detail = (
            f"已找到 {_safe_attr(event_study_result, 'n_events', 0)} 个事件，"
            "但事件窗口收益序列为空或不完整。"
        )
    else:
        event_status = "已生成"
        event_detail = f"事件数量: {_safe_attr(event_study_result, 'n_events', 0)}。"
    statuses.append({"module": "事件研究", "status": event_status, "detail": event_detail})
    if _factor_corr_is_multi_factor_input(factor_corr):
        if _factor_corr_has_valid_off_diagonal(factor_corr):
            factor_corr_status = "已生成"
            factor_corr_detail = "已生成因子相关性热力图。"
        else:
            factor_corr_status = "样本不足"
            factor_corr_detail = (
                f"{factor_corr_summary['headline']} {factor_corr_summary['detail']}"
                if factor_corr_summary
                else "因子相关性矩阵缺少有效的非对角元素。"
            )
        statuses.append(
            {"module": "因子相关性", "status": factor_corr_status, "detail": factor_corr_detail}
        )
    statuses.append(
        {
            "module": "市场状态 IC",
            "status": "已生成" if has_regime_ic else "未启用或未生成",
            "detail": "已生成市场状态 IC，用于检查因子在不同市场环境下的方向稳定性。"
            if has_regime_ic
            else "未传入市场状态 IC 结果，无法判断信号是否依赖上涨、下跌或波动环境。",
        }
    )
    return statuses


def _build_research_decision(
    *,
    metrics: dict[str, Any],
    report_statuses: list[dict[str, str]],
    strategy_pages: list[dict[str, Any]],
    quality_summary: dict[str, Any],
    benchmark_summary: dict[str, Any] | None,
    attribution_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a concise first-screen research decision for readers."""
    rating = metrics.get("factor_rating")
    if not isinstance(rating, FactorRating):
        rating = _compute_factor_rating(metrics)

    if rating.stars >= 4:
        verdict = "进入候选池观察"
        action = "允许进入候选池，但需在样本外、交易成本和风险约束复核后再考虑实盘。"
    elif rating.stars == 3:
        verdict = "继续研究"
        action = "保留研究价值，优先补齐关键缺口后再决定是否进入候选池。"
    elif rating.stars == 2:
        verdict = "暂缓使用"
        action = "当前证据偏弱，仅适合继续诊断，不建议进入策略组合。"
    else:
        verdict = "不纳入候选池"
        action = "当前统计证据不足，应重新审视因子定义、样本和预处理口径。"

    evidence: list[str] = []
    if rating.positives:
        evidence.append(f"优势维度：{'、'.join(rating.positives[:3])}。")
    n_periods = int(_num(metrics.get("n_periods")))
    if n_periods > 0:
        evidence.append(f"有效样本 {n_periods} 期。")
    if strategy_pages:
        long_short_count = sum(1 for page in strategy_pages if page.get("is_long_short"))
        evidence.append(
            f"已覆盖 {len(strategy_pages)} 个策略，其中 {long_short_count} 个多空策略。"
        )
    if benchmark_summary:
        evidence.append(f"基准相对表现：{benchmark_summary['direction']}。")
    if attribution_summary:
        evidence.append(f"归因口径：{attribution_summary['active']}")
    if not evidence:
        evidence.append("当前证据不足，需要先生成 IC、回测和质量检查。")

    gaps = list(rating.caps)
    quality_status = str(quality_summary.get("status", ""))
    if quality_status in {"需关注", "失败", "未知", "未传入"}:
        gaps.append(f"数据质量状态为{quality_status}。")
    for row in report_statuses:
        status = row.get("status", "")
        module = row.get("module", "")
        if module == "滚动样本外" and status != "已生成":
            gaps.append("缺少滚动样本外验证。")
        elif module == "组合归因（Brinson/Barra）" and status != "已生成":
            if status == "需关注":
                gaps.append("归因图表或行业/风格明细不完整。")
            else:
                gaps.append("缺少组合归因证据。")
        elif module == "基准超额" and status != "已生成":
            if status == "需关注":
                gaps.append("基准超额图表或日度明细不完整。")
            else:
                gaps.append("缺少基准超额对比。")
    if not gaps:
        gaps.append("暂无硬性评级上限，但仍需持续监控样本外衰减。")

    next_steps: list[str] = []
    if any("滚动样本外" in gap for gap in gaps):
        next_steps.append("延长样本区间或降低滚动验证窗口要求，补齐滚动样本外验证。")
    if any("归因图表" in gap for gap in gaps):
        next_steps.append("补齐归因行业或风格明细，确认归因摘要是否由少数维度驱动。")
    elif any("组合归因" in gap for gap in gaps):
        next_steps.append("补齐组合权重、基准权重、行业收益或风格暴露后复跑归因。")
    if any("基准超额图表" in gap for gap in gaps):
        next_steps.append("补齐基准日度净值明细，确认超额曲线、跟踪误差和回撤路径。")
    elif any("基准超额" in gap for gap in gaps):
        next_steps.append("指定基准指数并复跑，确认超额收益和主动风险。")
    if quality_status in {"需关注", "失败", "未知", "未传入"}:
        next_steps.append("先处理数据质量缺口，再提高结论置信度。")
    if not next_steps:
        next_steps.append("进入更长区间和更大股票池复核，并监控换手、成本和容量约束。")

    generated = sum(1 for row in report_statuses if row.get("status") in {"已生成", "正常"})
    total = len(report_statuses)
    if generated == total:
        evidence_strength = "强"
    elif generated >= max(1, total // 2):
        evidence_strength = "中等"
    else:
        evidence_strength = "偏弱"

    return {
        "verdict": verdict,
        "action": action,
        "evidence_strength": evidence_strength,
        "score": rating.score,
        "stars": rating.stars,
        "label": rating.label,
        "rating_caps": rating.caps[:2],
        "evidence": evidence[:4],
        "gaps": gaps[:5],
        "next_steps": next_steps[:4],
    }


def _build_strategy_pages(
    factor_name: str,
    strategy_results: dict[str, Any],
    primary_strategy: str | None,
    signal_label: str = "",
) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    primary = (
        primary_strategy
        if primary_strategy in strategy_results
        else next(iter(strategy_results), None)
    )
    for name, result in strategy_results.items():
        page_charts: dict[str, str] = {}
        stats = _safe_attr(result, "summary_stats", {}) or {}
        strategy_type = _infer_strategy_type(name, result)
        is_long_short = _resolve_is_long_short(result, stats)
        portfolio_stats = (
            stats.get("portfolio") or (stats.get("long_short") if is_long_short else {}) or {}
        )
        portfolio_metrics = {
            key: _finite_float(portfolio_stats.get(key))
            for key in (
                "ann_ret",
                "ann_vol",
                "sharpe",
                "max_dd",
                "avg_turnover",
                "total_cost",
                "ann_turnover",
            )
        }
        long_short_stats = stats.get("long_short") if is_long_short else None
        constraints = _strategy_constraints(result)
        trade_summary = _build_trade_summary(result)
        display_name = _display_strategy_name(name, strategy_type, result)
        try:
            returns_chart = _make_returns_chart(result, factor_name)
            if returns_chart:
                page_charts["returns_chart"] = returns_chart
        except Exception:
            logger.warning("生成策略收益图失败: %s", name, exc_info=True)
        try:
            monthly_chart = _make_monthly_return_heatmap(result)
            if monthly_chart:
                page_charts["monthly_return_chart"] = monthly_chart
        except Exception:
            logger.warning("生成策略月度收益图失败: %s", name, exc_info=True)

        pages.append(
            {
                "name": name,
                "display_name": display_name,
                "display_name_with_code": f"{display_name}（{name}）",
                "slug": _slugify_strategy_name(name),
                "is_primary": name == primary,
                "strategy_type": strategy_type,
                "strategy_type_label": _display_strategy_type(strategy_type),
                "exposure_label": _strategy_exposure_label(strategy_type, is_long_short),
                "signal_label": signal_label,
                "is_long_short": is_long_short,
                "return_label": "多空组合" if is_long_short else "组合收益",
                "bt_stats": _build_bt_summary_table(stats, include_long_short=is_long_short),
                "portfolio_stats": portfolio_stats,
                "portfolio_metrics": portfolio_metrics,
                "long_short_stats": long_short_stats,
                "constraints": constraints,
                "params_summary": _strategy_params_summary(result),
                "avg_turnover": portfolio_metrics["avg_turnover"],
                "total_cost": portfolio_metrics["total_cost"],
                "cost_model": _display_cost_model(constraints.get("cost_model")),
                "trade_summary": trade_summary,
                "quality_summary": _build_strategy_quality_summary(portfolio_stats, trade_summary),
                "monthly_return_summary": _build_monthly_return_summary(result),
                "charts": page_charts,
            }
        )
    return pages


# ── 主函数 ────────────────────────────────────────────────────────────


def generate_tear_sheet(
    factor_name: str,
    ic_result: Any,
    bt_result: Any,
    to_result: Any,
    *,
    frequency: str = "daily",
    date_range: str = "",
    advanced_results: dict[str, Any] | None = None,
    universe: str = "lft_default",
    benchmark_result: Any = None,
    attribution_result: Any = None,
    backtest_direction: dict[str, Any] | None = None,
    walk_forward_result: Any = None,
    walk_forward_summary: dict[str, Any] | None = None,
    event_study_result: Any = None,
    factor_corr: Any = None,
    pearson_ic_result: Any = None,
    neutralized_ic_result: Any = None,
    llm_explanation: dict[str, Any] | None = None,
    strategy_results: dict[str, Any] | None = None,
    primary_strategy: str | None = None,
    quality_report: dict[str, Any] | None = None,
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
        交易可行性结果。
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
    benchmark_result : BenchmarkResult or None
        基准对比结果。
    attribution_result : dict or None
        归因分析结果，键名 "brinson"（BrinsonResult）和/或 "barra"（BarraStyleResult）。
    event_study_result : EventStudyResult or None
        事件研究结果（来自 compute_event_study）。
    factor_corr : pl.DataFrame or None
        因子相关性矩阵（来自 compute_factor_correlation）。

    Returns
    -------
    str
        完整的 HTML 报告字符串。
    """
    charts: dict[str, str] = {}
    if strategy_results is None and bt_result is not None:
        default_name = _safe_attr(bt_result, "strategy_name", "primary") or "primary"
        strategy_results = {default_name: bt_result}
    signal_label = (
        "反向因子"
        if backtest_direction and backtest_direction.get("direction") == "reversed"
        else ""
    )
    strategy_pages = (
        _build_strategy_pages(factor_name, strategy_results, primary_strategy, signal_label)
        if strategy_results
        else []
    )
    execution_summary = _build_execution_summary(strategy_pages)

    # 顶层净值图仅在没有策略分页时（即无策略结果）作为兜底展示；
    # 有 bt_result 时 strategy_pages 必非空，各分页会渲染自己的净值图，
    # 此处再生成只会被模板丢弃，属无谓开销。
    if bt_result is not None and not strategy_pages:
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
        try:
            ic_dist_b64 = _make_ic_distribution_chart(ic_result)
            if ic_dist_b64:
                charts["ic_distribution_chart"] = ic_dist_b64
        except Exception:
            logger.warning("生成 IC 分布图失败", exc_info=True)

    if to_result is not None:
        try:
            to_b64 = _make_turnover_chart(to_result)
            if to_b64:
                charts["turnover_chart"] = to_b64
        except Exception:
            logger.warning("生成换手率图表失败", exc_info=True)

    # 同上：顶层月度热力图仅在无策略分页时兜底展示。
    if bt_result is not None and not strategy_pages:
        try:
            monthly_b64 = _make_monthly_return_heatmap(bt_result)
            if monthly_b64:
                charts["monthly_return_chart"] = monthly_b64
        except Exception:
            logger.warning("生成月度收益图失败", exc_info=True)

    if benchmark_result is not None:
        try:
            b64 = _make_benchmark_chart(benchmark_result)
            if b64:
                charts["benchmark_chart"] = b64
        except Exception:
            logger.warning("生成基准对比图表失败", exc_info=True)

    brinson_r = None
    barra_r = None
    if attribution_result is not None:
        brinson_r = attribution_result.get("brinson")
        barra_r = attribution_result.get("barra")
        try:
            b64 = _make_attribution_chart(brinson_r, barra_r)
            if b64:
                charts["attribution_chart"] = b64
        except Exception:
            logger.warning("生成归因图表失败", exc_info=True)

    if walk_forward_result is not None:
        try:
            wf_b64 = _make_walk_forward_chart(walk_forward_result)
            if wf_b64:
                charts["walk_forward_chart"] = wf_b64
        except Exception:
            logger.warning("生成 Walk-Forward 图表失败", exc_info=True)

    if event_study_result is not None:
        try:
            es_b64 = _make_event_study_chart(event_study_result)
            if es_b64:
                charts["event_study_chart"] = es_b64
        except Exception:
            logger.warning("生成事件研究图表失败", exc_info=True)

    if factor_corr is not None:
        try:
            fc_b64 = _make_factor_corr_heatmap(factor_corr)
            if fc_b64:
                charts["factor_corr_chart"] = fc_b64
        except Exception:
            logger.warning("生成因子相关性热力图失败", exc_info=True)

    # Quantile spread chart from backtest result
    quantile_spread_notice = None
    if bt_result is not None:
        try:
            grouped_rets = _extract_quantile_grouped_returns(bt_result)
            if len(grouped_rets) >= 2:
                qs_b64 = _make_quantile_spread_chart(grouped_rets)
                if qs_b64:
                    charts["quantile_spread_chart"] = qs_b64
                else:
                    quantile_spread_notice = (
                        "分位价差图未生成：已识别至少两组分位组合净值，"
                        "但缺少可绘图的价差序列。下一步：检查分组净值长度、"
                        "分组编号和图表生成日志。"
                    )
        except Exception:
            logger.warning("生成分位价差图表失败", exc_info=True)
            quantile_spread_notice = (
                "分位价差图未生成：已传入分位组合净值，但图表生成失败。"
                "下一步：检查分组净值长度、分组编号和图表生成日志。"
            )

    metrics = _extract_metrics(
        ic_result, bt_result, to_result, advanced_results, pearson_ic_result, neutralized_ic_result
    )
    if walk_forward_result is not None:
        metrics["walk_forward_oos_sharpe_mean"] = _finite_float(
            _safe_attr(walk_forward_result, "oos_sharpe_mean")
        )
        metrics["walk_forward_stability_ratio"] = _finite_float(
            _safe_attr(walk_forward_result, "stability_ratio")
        )
        metrics["walk_forward_oos_max_dd"] = _finite_float(
            _safe_attr(walk_forward_result, "oos_max_dd")
        )
    elif walk_forward_summary and walk_forward_summary.get("status") == "ok":
        metrics["walk_forward_oos_sharpe_mean"] = _finite_float(
            walk_forward_summary.get("oos_sharpe_mean")
        )
        metrics["walk_forward_stability_ratio"] = _finite_float(
            walk_forward_summary.get("stability_ratio")
        )
        metrics["walk_forward_oos_max_dd"] = _finite_float(walk_forward_summary.get("oos_max_dd"))
    metrics["factor_rating"] = _compute_factor_rating(metrics)

    warnings: list[str] = []
    if metrics.get("n_periods", 0) < 30:
        warnings.append(f"样本量较少（{metrics['n_periods']} 期），IC 估计可能不稳定。")
    if metrics.get("n_periods", 0) < 60:
        warnings.append("短样本年化指标仅适合同区间横向比较，不代表长期可外推业绩。")
    if metrics.get("ic_mean") is not None and abs(metrics["ic_mean"]) < 0.01:
        warnings.append("IC 均值极低（|IC| < 0.01），因子预测能力有限。")
    if metrics.get("avg_turnover", 0) is not None and metrics.get("avg_turnover", 0) > 0.8:
        warnings.append("换手率较高（>80%），信号稳定性需关注。")

    summary_html = _generate_summary_text(factor_name, metrics, llm_explanation)
    llm_explanation_view = _prepare_llm_explanation_view(llm_explanation)
    predictive_summary = _build_predictive_summary(metrics)
    neutralized_summary = _build_neutralized_summary(metrics)
    holding_period_summary = _build_holding_period_summary(metrics)
    regime_summary = _build_regime_summary(metrics)
    benchmark_summary = _build_benchmark_summary(benchmark_result)
    benchmark_notice = _build_benchmark_notice(
        benchmark_result, "benchmark_chart" in charts, benchmark_summary
    )
    attribution_summary = _build_attribution_summary(attribution_result)
    attribution_notice = _build_attribution_notice(
        attribution_result, "attribution_chart" in charts, attribution_summary
    )
    factor_corr_summary = _build_factor_corr_summary(factor_corr)
    quality_summary = _build_quality_summary(quality_report)
    report_statuses = _build_report_statuses(
        metrics=metrics,
        quality_summary=quality_summary,
        benchmark_result=benchmark_result,
        benchmark_summary=benchmark_summary,
        benchmark_notice=benchmark_notice,
        attribution_notice=attribution_notice,
        attribution_summary=attribution_summary,
        walk_forward_result=walk_forward_result,
        walk_forward_summary=walk_forward_summary,
        event_study_result=event_study_result,
        has_regime_ic=bool(metrics.get("regime_table")),
        factor_corr=factor_corr,
        factor_corr_summary=factor_corr_summary,
    )
    research_decision = _build_research_decision(
        metrics=metrics,
        report_statuses=report_statuses,
        strategy_pages=strategy_pages,
        quality_summary=quality_summary,
        benchmark_summary=benchmark_summary,
        attribution_summary=attribution_summary,
    )
    primary_page = next(
        (page for page in strategy_pages if page["is_primary"]),
        strategy_pages[0] if strategy_pages else None,
    )
    dashboard = {
        "primary_strategy": (
            primary_page["display_name"]
            if primary_page
            else metrics.get("bt_strategy_name", "未运行")
        ),
        "primary_strategy_code": (
            primary_page["name"] if primary_page else metrics.get("bt_strategy_name", "未运行")
        ),
        "primary_exposure": primary_page["exposure_label"] if primary_page else "未运行",
        "primary_signal_label": primary_page["signal_label"] if primary_page else signal_label,
        "effective_range": (
            f"{metrics.get('effective_start')} ~ {metrics.get('effective_end')}"
            if metrics.get("effective_start") and metrics.get("effective_end")
            else "未计算"
        ),
        "sample_periods": metrics.get("n_periods", 0),
        "data_quality": quality_summary["status"]
        if quality_report
        else ("需关注" if warnings else "正常"),
    }

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
        dashboard=dashboard,
        quality_summary=quality_summary,
        summary_html=summary_html,
        predictive_summary=predictive_summary,
        neutralized_summary=neutralized_summary,
        holding_period_summary=holding_period_summary,
        regime_summary=regime_summary,
        attribution_notice=attribution_notice,
        benchmark_summary=benchmark_summary,
        benchmark_notice=benchmark_notice,
        quantile_spread_notice=quantile_spread_notice,
        attribution_summary=attribution_summary,
        factor_corr_summary=factor_corr_summary,
        research_decision=research_decision,
        report_statuses=report_statuses,
        benchmark_result=benchmark_result,
        attribution_result=attribution_result,
        backtest_direction=backtest_direction,
        walk_forward_result=walk_forward_result,
        walk_forward_summary=walk_forward_summary,
        event_study_result=event_study_result,
        factor_corr=factor_corr,
        llm_explanation=llm_explanation_view,
        strategy_pages=strategy_pages,
        execution_summary=execution_summary,
    )

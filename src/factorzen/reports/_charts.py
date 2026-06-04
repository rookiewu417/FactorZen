"""报告图表生成:将评估结果渲染为 base64 PNG(matplotlib)。"""

import base64
import io
from typing import Any

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")  # 非交互后端，不弹窗
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _register_cjk_fonts() -> None:
    """发现并注册常见 CJK 字体，让未预装中文字体的环境也能渲染报告图表。

    重点是 matplotlib 默认不扫描的 WSL 挂载 Windows 字体目录（``/mnt/*/Windows/Fonts``，
    含 Microsoft YaHei / SimHei），以及用户字体目录。任意一项失败都静默，绝不影响 import。
    """
    import contextlib
    import glob
    from pathlib import Path

    from matplotlib import font_manager

    candidates: list[str] = []
    # WSL：Windows 自带中文字体（任意盘符）
    for pat in ("msyh*.tt?", "simhei.ttf", "simsun.tt?"):
        candidates += glob.glob(f"/mnt/*/Windows/Fonts/{pat}")
    # 用户字体目录（含通过 setup 装入的 Noto Sans CJK）
    for d in (Path.home() / ".local/share/fonts", Path.home() / ".fonts"):
        if d.exists():
            for ext in ("*.ttf", "*.ttc", "*.otf"):
                candidates += [str(p) for p in d.glob(ext)]
    for fp in candidates:
        with contextlib.suppress(Exception):
            font_manager.fontManager.addfont(fp)


_register_cjk_fonts()

# 中文字体支持：按优先级回退（Windows / Linux 常见 CJK 字体），末位回落 DejaVu Sans。
# 必须设置 font.sans-serif 列表（而非 font.family 单值），否则 matplotlib 无法
# 在缺失首选字体时自动尝试后续字体，会把中文渲染成豆腐块并刷屏 glyph 缺失告警。
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "Noto Sans CJK JP",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

from factorzen.reports._formatting import _finite_float, _safe_attr  # noqa: E402


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

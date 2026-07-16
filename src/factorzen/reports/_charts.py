"""报告图表生成:将评估结果渲染为 base64 PNG(matplotlib)。"""

import base64
import io
from typing import Any

import matplotlib
import numpy as np

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

from factorzen.reports._formatting import _safe_attr  # noqa: E402


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

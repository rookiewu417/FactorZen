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


def _make_ic_cumulative_chart(ic_result: Any) -> str | None:
    """IC 累计曲线：斜率即平均 IC，斜率转平/转负标示 alpha 失效时点。

    棒图只能看单期强弱，累计曲线才能看出趋势——这是判断因子是否已失效的第一工具。
    虚线为「恒定斜率」参考（首末点连线）；实线持续落在参考线下方＝后段贡献衰减。
    """
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
    date_col = "trade_date" if "trade_date" in ic_pd.columns else ic_pd.columns[0]
    ic_pd = ic_pd.sort_values(date_col)

    values = ic_pd[ic_col].to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return None
    # 缺失期不推进累计（填 0），避免断线；不参与统计
    cum = np.cumsum(np.where(finite, values, 0.0))

    ic_pd, x_col, is_date_axis = _with_plot_dates(ic_pd, date_col)
    x_vals = ic_pd[x_col].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_vals, cum, color="#2c7fb8", linewidth=1.6, label="累计 IC")
    ax.fill_between(x_vals, 0, cum, color="#2c7fb8", alpha=0.12)
    if len(cum) >= 2:
        ax.plot(
            [x_vals[0], x_vals[-1]],
            [0.0, cum[-1]],
            color="#999",
            linestyle="--",
            linewidth=0.9,
            label="恒定斜率参考",
        )
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    mean_ic = float(np.mean(values[finite]))
    ax.set_title(f"IC 累计曲线（累计 {cum[-1]:.2f}，均值 {mean_ic:.4f}）", fontsize=12)
    ax.set_ylabel("累计 IC")
    ax.legend(fontsize=8, loc="best")
    _format_sparse_x_axis(ax, is_date_axis=is_date_axis)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _make_drawdown_chart(bt_result: Any) -> str | None:
    """回撤曲线（underwater）：最大回撤的形态比数值更能说明风险。

    单次崩塌 vs 长期阴跌、水下时长与修复耗时，核心指标表里的一个 max_dd 数字看不出来。
    """
    if bt_result is None:
        return None
    nav = _safe_attr(bt_result, "nav")
    if nav is None or nav.is_empty():
        return None

    nav_pd = nav.to_pandas()
    if "nav" not in nav_pd.columns or "trade_date" not in nav_pd.columns:
        return None
    # 分层 nav（含 group 列）语义是多条曲线，回撤图只描述单一组合净值
    if "group" in nav_pd.columns:
        return None

    nav_pd = nav_pd.sort_values("trade_date")
    navs = nav_pd["nav"].to_numpy(dtype=float)
    finite = np.isfinite(navs) & (navs > 0)
    if finite.sum() < 2:
        return None
    nav_pd = nav_pd.loc[finite]
    navs = navs[finite]

    running_max = np.maximum.accumulate(navs)
    drawdown = navs / running_max - 1.0

    nav_pd, x_col, is_date_axis = _with_plot_dates(nav_pd)
    x_vals = nav_pd[x_col].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.fill_between(x_vals, drawdown, 0, color="#c0392b", alpha=0.28)
    ax.plot(x_vals, drawdown, color="#c0392b", linewidth=1.0)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    trough = int(np.argmin(drawdown))
    max_dd = float(drawdown[trough])
    # 水下占比：净值未创新高的时间比例
    underwater_ratio = float(np.mean(drawdown < -1e-12))
    if max_dd < 0:
        ax.annotate(
            f"最大回撤 {max_dd * 100:.2f}%",
            xy=(x_vals[trough], max_dd),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#7b241c",
        )
    ax.set_title(f"回撤曲线（水下时间占比 {underwater_ratio * 100:.1f}%）", fontsize=12)
    ax.set_ylabel("回撤")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    _format_sparse_x_axis(ax, is_date_axis=is_date_axis)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _make_group_bar_chart(mono_result: Any) -> str | None:
    """分组平均收益柱状图：一眼看出是否单调、收益是否只来自某一端。

    表格逐行读不出形状；柱状图能立刻区分「全程单调」与「只有极端组有效」。
    """
    if mono_result is None:
        return None
    group_means = _safe_attr(mono_result, "group_means", None)
    if not group_means or len(group_means) < 2:
        return None
    try:
        means = np.asarray([float(m) for m in group_means], dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(means).all():
        return None

    fig, ax = plt.subplots(figsize=(10, 3.8))
    labels = [f"G{i + 1}" for i in range(len(means))]
    colors = ["#c0392b" if m < 0 else "#27ae60" for m in means]
    ax.bar(labels, means, color=colors, alpha=0.85, width=0.62)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.6)

    for i, m in enumerate(means):
        ax.annotate(
            f"{m * 100:.3f}%",
            xy=(i, m),
            xytext=(0, 3 if m >= 0 else -11),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
        )

    spread = float(means[-1] - means[0])
    ax.set_title(
        f"分组平均收益（G{len(means)} − G1 = {spread * 100:.3f}%）",
        fontsize=12,
    )
    ax.set_ylabel("期均收益")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.margins(y=0.18)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _make_group_nav_chart(mono_result: Any) -> str | None:
    """分组累计净值曲线：每一组一条线，低→高按色阶排列。

    因子若真单调有效，曲线终点顺序应与色阶一致（G1 最低 → GN 最高）；
    交叉缠绕说明分组区分度不稳定。

    **口径**：等权分组、不含交易成本与交易约束，与组合回测净值不可直接比较。
    """
    if mono_result is None:
        return None
    frame = _safe_attr(mono_result, "group_daily_returns", None)
    if frame is None or getattr(frame, "is_empty", lambda: True)():
        return None

    gdr = frame.to_pandas()
    required = {"trade_date", "group", "mean_ret"}
    if not required.issubset(set(gdr.columns)):
        return None
    gdr = gdr.dropna(subset=["mean_ret"]).sort_values(["group", "trade_date"])
    if gdr.empty:
        return None

    groups = sorted(gdr["group"].unique())
    if len(groups) < 2:
        return None

    # viridis 而非 RdYlGn：后者色阶中点是浅黄，中间组在白底上几乎不可见
    # （实测 5 组时 G3 看不见）。viridis 全程亮度足够，且保留低→高的顺序感。
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    finals: list[tuple[Any, float]] = []
    for idx, g in enumerate(groups):
        part = gdr[gdr["group"] == g]
        rets = part["mean_ret"].to_numpy(dtype=float)
        rets = np.where(np.isfinite(rets), rets, 0.0)
        if rets.size < 2:
            continue
        nav = np.cumprod(1.0 + rets)
        part, x_col, is_date_axis = _with_plot_dates(part)
        # 压到 [0, 0.86]：viridis 末端的亮黄在白底上偏淡，截掉后最高组仍是黄绿而非纯黄
        color = cmap(0.86 * idx / max(1, len(groups) - 1))
        ax.plot(
            part[x_col].to_numpy(),
            nav,
            linewidth=1.3,
            color=color,
            label=f"G{int(g) + 1}",
        )
        finals.append((g, float(nav[-1])))

    if not finals:
        plt.close(fig)
        return None

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_title("分组累计净值（等权，不含成本与交易约束）", fontsize=12)
    ax.set_ylabel("累计净值")
    ax.legend(fontsize=8, loc="upper left", ncol=min(len(finals), 5))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    _format_sparse_x_axis(ax, is_date_axis=is_date_axis)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _make_benchmark_chart(benchmark_result: Any) -> str | None:
    """策略 / 基准 / 超额三线图：区分「真 alpha」与「跟着大盘涨」。

    核心指标表只露一个「超额年化」标量，看不出超额是稳定累积还是集中在某一段。
    """
    if benchmark_result is None:
        return None
    daily = _safe_attr(benchmark_result, "daily")
    if daily is None or daily.is_empty():
        return None

    frame = daily.to_pandas()
    if "trade_date" not in frame.columns:
        return None
    series_specs = [
        ("strategy_nav", "策略", "#2c7fb8", 1.6),
        ("benchmark_nav", "基准", "#7f8c8d", 1.2),
        ("excess_nav", "超额", "#e67e22", 1.4),
    ]
    available = [s for s in series_specs if s[0] in frame.columns]
    if not available:
        return None

    frame = frame.sort_values("trade_date")
    frame, x_col, is_date_axis = _with_plot_dates(frame)
    x_vals = frame[x_col].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    plotted = False
    for col, label, color, width in available:
        vals = frame[col].to_numpy(dtype=float)
        if not np.isfinite(vals).any():
            continue
        ax.plot(x_vals, vals, linewidth=width, color=color, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    bench_name = str(_safe_attr(benchmark_result, "benchmark_name", "") or "基准")
    ir = _safe_attr(benchmark_result, "information_ratio", None)
    title = f"策略 vs {bench_name} vs 超额"
    try:
        ir_val = float(ir)  # type: ignore[arg-type]
        if np.isfinite(ir_val):
            title = f"{title}（IR={ir_val:.2f}）"
    except (TypeError, ValueError):
        pass
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("净值")
    ax.legend(fontsize=8, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    _format_sparse_x_axis(ax, is_date_axis=is_date_axis)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)

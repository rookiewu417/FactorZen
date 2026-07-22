"""信号轨报告：回答「因子有没有预测力？排序对不对？信号衰减多快？」

数据源 ``SignalBacktestResult``（毛收益 / 研究口径），不含停牌/涨跌停/T+1 等交易约束。
"""

from __future__ import annotations

import html as html_lib
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from factorzen.core.logger import get_logger
from factorzen.reports._charts import (
    _fig_to_base64,
    _format_sparse_x_axis,
    _with_plot_dates,
)
from factorzen.reports._formatting import (
    _finite_float,
    _format_metric_number,
    _format_metric_percent,
    _safe_attr,
)

logger = get_logger(__name__)

# 醒目口径横幅（测试与页面均依赖此精确文案）
SIGNAL_BANNER = (
    "信号层毛收益 · 不含停牌/涨跌停/T+1/容量约束与撮合成本"
    " · 不可直接当可实现收益汇报 · 可交易净值请看 `fz factor backtest`"
)

_EMPTY = "未计算"


def _esc(value: Any) -> str:
    return html_lib.escape(str(value if value is not None else ""), quote=True)


def _fmt_ic(value: Any) -> str:
    return _format_metric_number(value, digits=4, empty=_EMPTY)


def _fmt_ratio2(value: Any) -> str:
    return _format_metric_number(value, digits=2, empty=_EMPTY)


def _fmt_pct2(value: Any) -> str:
    return _format_metric_percent(value, digits=2, empty=_EMPTY)


def _fmt_pvalue(value: Any) -> str:
    return _format_metric_number(value, digits=4, empty=_EMPTY)


def _sig_stars(pvalue: float | None) -> str:
    if pvalue is None:
        return ""
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.1:
        return "*"
    return ""


def _is_empty_frame(frame: Any) -> bool:
    if frame is None:
        return True
    try:
        return bool(frame.is_empty())
    except Exception:
        return True


def _safe_chart(label: str, maker: Any) -> str | None:
    try:
        b64 = maker()
    except Exception:
        logger.warning("信号报告生成%s失败", label, exc_info=True)
        return None
    return b64 if b64 else None


# ── charts ───────────────────────────────────────────────────────────────────


def _chart_group_nav(signal_result: Any) -> str | None:
    """分层累计净值 + 多空毛净值虚线。"""
    group_nav = _safe_attr(signal_result, "group_nav")
    if _is_empty_frame(group_nav):
        return None
    gnav = group_nav.to_pandas()
    if not {"trade_date", "group", "nav"}.issubset(gnav.columns):
        return None
    gnav = gnav.dropna(subset=["nav"]).sort_values(["group", "trade_date"])
    if gnav.empty:
        return None
    groups = sorted(gnav["group"].unique())
    if not groups:
        return None

    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    is_date_axis = False
    x_col = "trade_date"
    for idx, g in enumerate(groups):
        part = gnav[gnav["group"] == g]
        if part.empty:
            continue
        part, x_col, is_date_axis = _with_plot_dates(part)
        color = cmap(0.86 * idx / max(1, len(groups) - 1))
        label = f"G{int(g) + 1}" if isinstance(g, (int, float, np.integer)) else str(g)
        ax.plot(part[x_col], part["nav"], linewidth=1.25, color=color, label=label)

    ls_nav = _safe_attr(signal_result, "ls_nav")
    if not _is_empty_frame(ls_nav):
        lsn = ls_nav.to_pandas()
        if "nav_gross" in lsn.columns and "trade_date" in lsn.columns:
            lsn = lsn.dropna(subset=["nav_gross"]).sort_values("trade_date")
            if not lsn.empty:
                lsn, x_col_ls, is_date_axis = _with_plot_dates(lsn)
                ax.plot(
                    lsn[x_col_ls],
                    lsn["nav_gross"],
                    color="#1a1a1a",
                    linewidth=2.0,
                    linestyle="--",
                    label="多空(毛)",
                    zorder=5,
                )

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_title("分层累计净值（毛口径）", fontsize=11)
    ax.set_ylabel("净值")
    ax.legend(fontsize=7, loc="upper left", ncol=min(len(groups) + 1, 6))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    _format_sparse_x_axis(ax, is_date_axis=is_date_axis)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _chart_group_bar(signal_result: Any) -> str | None:
    """各组年化平均收益柱 + 线性趋势。"""
    group_returns = _safe_attr(signal_result, "group_returns")
    if _is_empty_frame(group_returns):
        return None
    gr = group_returns.to_pandas()
    if not {"group", "ret"}.issubset(gr.columns):
        return None
    means = gr.groupby("group", sort=True)["ret"].mean()
    if means.empty or len(means) < 1:
        return None
    vals = means.to_numpy(dtype=float)
    if not np.isfinite(vals).any():
        return None
    # 日均 → 年化（展示派生，非重算绩效引擎）
    ann = vals * 252.0
    labels = [
        f"G{int(g) + 1}" if isinstance(g, (int, float, np.integer)) else str(g)
        for g in means.index
    ]
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in ann]

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    x = np.arange(len(ann))
    ax.bar(x, ann, color=colors, alpha=0.85, width=0.62)
    ax.axhline(y=0, color="gray", linewidth=0.6)
    if len(ann) >= 2 and np.isfinite(ann).sum() >= 2:
        finite_mask = np.isfinite(ann)
        coef = np.polyfit(x[finite_mask], ann[finite_mask], 1)
        ax.plot(x, np.polyval(coef, x), color="#2980b9", linewidth=1.3, linestyle="--", label="线性趋势")
        ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("分层平均收益（年化）", fontsize=11)
    ax.set_ylabel("年化收益")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.margins(y=0.15)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _chart_ic_series(signal_result: Any) -> str | None:
    ic = _safe_attr(signal_result, "ic")
    ic_series = _safe_attr(ic, "ic_series")
    if _is_empty_frame(ic_series):
        return None
    ic_pd = ic_series.to_pandas()
    ic_col = "ic" if "ic" in ic_pd.columns else next(
        (c for c in ic_pd.columns if c != "trade_date"), None
    )
    if ic_col is None:
        return None
    date_col = "trade_date" if "trade_date" in ic_pd.columns else ic_pd.columns[0]
    ic_pd = ic_pd.sort_values(date_col)
    vals = ic_pd[ic_col].to_numpy(dtype=float)
    if not np.isfinite(vals).any():
        return None
    ic_pd, x_col, is_date_axis = _with_plot_dates(ic_pd, date_col)

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.bar(ic_pd[x_col], vals, width=1.0, color="#95a5a6", alpha=0.55, label="IC")
    if len(ic_pd) >= 5:
        window = min(20, max(3, len(ic_pd) // 3))
        rolling = (
            __import__("pandas")
            .Series(vals)
            .rolling(window=window, min_periods=1)
            .mean()
            .to_numpy()
        )
        ax.plot(
            ic_pd[x_col],
            rolling,
            color="#e74c3c",
            linewidth=1.3,
            label=f"滚动{window}日均值",
        )
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.6)
    ax.set_title("IC 时序", fontsize=11)
    ax.legend(fontsize=8)
    _format_sparse_x_axis(ax, is_date_axis=is_date_axis)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _chart_ic_cumulative(signal_result: Any) -> str | None:
    ic = _safe_attr(signal_result, "ic")
    ic_series = _safe_attr(ic, "ic_series")
    if _is_empty_frame(ic_series):
        return None
    ic_pd = ic_series.to_pandas()
    ic_col = "ic" if "ic" in ic_pd.columns else next(
        (c for c in ic_pd.columns if c != "trade_date"), None
    )
    if ic_col is None:
        return None
    date_col = "trade_date" if "trade_date" in ic_pd.columns else ic_pd.columns[0]
    ic_pd = ic_pd.sort_values(date_col)
    values = ic_pd[ic_col].to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return None
    cum = np.cumsum(np.where(finite, values, 0.0))
    ic_pd, x_col, is_date_axis = _with_plot_dates(ic_pd, date_col)
    x_vals = ic_pd[x_col].to_numpy()

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.plot(x_vals, cum, color="#2c7fb8", linewidth=1.5, label="累计 IC")
    ax.fill_between(x_vals, 0, cum, color="#2c7fb8", alpha=0.12)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_title(f"IC 累计曲线（终点 {cum[-1]:.2f}）", fontsize=11)
    ax.set_ylabel("累计 IC")
    ax.legend(fontsize=8)
    _format_sparse_x_axis(ax, is_date_axis=is_date_axis)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _chart_ic_decay(signal_result: Any) -> str | None:
    ic = _safe_attr(signal_result, "ic")
    decay = _safe_attr(ic, "decay", None) or {}
    multi = _safe_attr(ic, "multi_period", None) or {}
    points: list[tuple[int, float]] = []
    if isinstance(decay, dict) and decay:
        for h, v in decay.items():
            try:
                hv = int(h)
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                points.append((hv, fv))
    elif isinstance(multi, dict) and multi:
        for h, v in multi.items():
            if not isinstance(v, dict):
                continue
            try:
                hv = int(h)
                raw_ic = v.get("ic_mean")
                if raw_ic is None:
                    continue
                fv = float(raw_ic)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                points.append((hv, fv))
    if len(points) < 1:
        return None
    points.sort(key=lambda x: x[0])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.plot(xs, ys, marker="o", color="#8e44ad", linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.6)
    ax.set_title("IC 衰减曲线", fontsize=11)
    ax.set_xlabel("持有期（天）")
    ax.set_ylabel("IC 均值")
    fig.tight_layout()
    return _fig_to_base64(fig)


def _chart_ic_hist(signal_result: Any) -> str | None:
    ic = _safe_attr(signal_result, "ic")
    ic_series = _safe_attr(ic, "ic_series")
    if _is_empty_frame(ic_series):
        return None
    ic_pd = ic_series.to_pandas()
    ic_col = "ic" if "ic" in ic_pd.columns else next(
        (c for c in ic_pd.columns if c != "trade_date"), None
    )
    if ic_col is None:
        return None
    vals = ic_pd[ic_col].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return None
    mean_ic = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.hist(vals, bins=min(30, max(8, vals.size // 4)), color="#5dade2", alpha=0.8, edgecolor="white")
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(x=mean_ic, color="#e74c3c", linewidth=1.4, label=f"均值 {mean_ic:.4f}")
    ax.set_title("IC 分布", fontsize=11)
    ax.set_xlabel("IC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _chart_group_year_heatmap(signal_result: Any) -> str | None:
    """行=年，列=分位组，值=该年该组年化收益。"""
    group_returns = _safe_attr(signal_result, "group_returns")
    if _is_empty_frame(group_returns):
        return None
    gr = group_returns.to_pandas()
    if not {"trade_date", "group", "ret"}.issubset(gr.columns):
        return None
    dates = gr["trade_date"].astype(str)
    parsed = None
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            candidate = __import__("pandas").to_datetime(dates, format=fmt, errors="coerce")
        except Exception:
            candidate = None
        if candidate is not None and candidate.notna().any():
            parsed = candidate
            break
    if parsed is None:
        try:
            parsed = __import__("pandas").to_datetime(dates, errors="coerce")
        except Exception:
            return None
    if parsed is None or not parsed.notna().any():
        return None
    frame = gr.assign(_year=parsed.dt.year).dropna(subset=["_year", "ret", "group"])
    if frame.empty:
        return None
    # 年化：日均 * 252
    pivot = (
        frame.groupby(["_year", "group"])["ret"]
        .mean()
        .mul(252.0)
        .unstack("group")
        .sort_index()
    )
    if pivot.empty:
        return None
    data = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    fig, ax = plt.subplots(figsize=(9.5, max(2.6, 0.55 * len(pivot.index) + 1.4)))
    vmax = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 0.01
    vmax = max(vmax, 0.01)
    image = ax.imshow(masked, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    cols = list(pivot.columns)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(
        [f"G{int(c) + 1}" if isinstance(c, (int, float, np.integer)) else str(c) for c in cols]
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(y)) for y in pivot.index])
    ax.set_title("分层收益年度热力图（年化）", fontsize=11)
    ax.set_xlabel("分位组")
    ax.set_ylabel("年份")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j] * 100:.1f}%", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, format=mticker.PercentFormatter(1.0))
    fig.tight_layout()
    return _fig_to_base64(fig)


def _chart_turnover(signal_result: Any) -> str | None:
    ls_returns = _safe_attr(signal_result, "ls_returns")
    if _is_empty_frame(ls_returns):
        return None
    lr = ls_returns.to_pandas()
    if "ls_turnover" not in lr.columns or "trade_date" not in lr.columns:
        return None
    lr = lr.dropna(subset=["ls_turnover"]).sort_values("trade_date")
    if lr.empty:
        return None
    vals = lr["ls_turnover"].to_numpy(dtype=float)
    if not np.isfinite(vals).any():
        return None
    mean_to = float(np.nanmean(vals))
    lr, x_col, is_date_axis = _with_plot_dates(lr)

    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.plot(lr[x_col], vals, color="#16a085", linewidth=1.0, alpha=0.85, label="日换手")
    ax.axhline(y=mean_to, color="#c0392b", linestyle="--", linewidth=1.1, label=f"均值 {mean_to:.2%}")
    ax.set_title("多空换手率时序", fontsize=11)
    ax.set_ylabel("换手率")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=8)
    _format_sparse_x_axis(ax, is_date_axis=is_date_axis)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


# ── metrics / HTML ───────────────────────────────────────────────────────────


def _build_metrics(signal_result: Any) -> dict[str, str]:
    ic = _safe_attr(signal_result, "ic")
    stats = _safe_attr(signal_result, "summary_stats", {}) or {}
    ls = stats.get("long_short") if isinstance(stats, dict) else None
    if not isinstance(ls, dict):
        ls = {}
    mono = _safe_attr(signal_result, "monotonicity")
    turnover = _safe_attr(signal_result, "turnover")

    # n_periods=0 → IC 一天都没算出来（截面样本不足 ic_analysis._MIN_CROSS_SAMPLES，
    # 或 join 零命中），此时 ic_mean/ir/tstat 全是 0.0 哨兵。直接展示会让读者把
    # 「算不出来」误读成「因子恰好无预测力」——两者在决策上天差地别，必须显式区分。
    ic_n = _finite_float(_safe_attr(ic, "n_periods"))
    ic_unavailable = ic_n is None or ic_n <= 0
    if ic_unavailable:
        ic_mean = ir = tstat = pvalue = ic_pos = None
    else:
        ic_mean = _finite_float(_safe_attr(ic, "ic_mean"))
        ir = _finite_float(_safe_attr(ic, "ir"))
        tstat = _finite_float(_safe_attr(ic, "ic_tstat"))
        pvalue = _finite_float(_safe_attr(ic, "ic_pvalue"))
        ic_pos = _finite_float(_safe_attr(ic, "ic_positive_ratio"))

    ann_g = _finite_float(ls.get("ann_ret_gross"))
    sharpe_g = _finite_float(ls.get("sharpe_gross"))
    max_dd_g = _finite_float(ls.get("max_dd_gross"))
    mono_score = _finite_float(_safe_attr(mono, "monotonicity_score"))
    avg_to = _finite_float(ls.get("avg_turnover"))
    if avg_to is None:
        avg_to = _finite_float(_safe_attr(turnover, "avg_turnover"))

    tstat_disp = _fmt_ratio2(tstat)
    stars = _sig_stars(pvalue)
    if tstat_disp != _EMPTY and stars:
        tstat_disp = f"{tstat_disp}{stars}"

    return {
        "ic_mean": _fmt_ic(ic_mean),
        "icir": _fmt_ratio2(ir),
        "tstat": tstat_disp,
        "pvalue": _fmt_pvalue(pvalue),
        "ic_pos": _fmt_pct2(ic_pos),
        "ann_ret_gross": _fmt_pct2(ann_g),
        "sharpe_gross": _fmt_ratio2(sharpe_g),
        "max_dd_gross": _fmt_pct2(max_dd_g),
        "mono_score": _fmt_ratio2(mono_score),
        "avg_turnover": _fmt_pct2(avg_to),
        # 供渲染层决定是否插入「IC 不可用」告警条
        "_ic_unavailable": "1" if ic_unavailable else "",
    }


def _section_html(
    title: str,
    caption: str,
    body: str,
    *,
    missing: bool = False,
    reason: str = "",
) -> str:
    if missing:
        reason_html = f"<p class='muted'>{_esc(reason or '数据缺失或不足以绘图')}</p>"
        return (
            f"<section class='block'>"
            f"<h2>{_esc(title)}</h2>"
            f"<p class='caption'>{_esc(caption)}</p>"
            f"<p class='missing'>未计算</p>{reason_html}"
            f"</section>"
        )
    return (
        f"<section class='block'>"
        f"<h2>{_esc(title)}</h2>"
        f"<p class='caption'>{_esc(caption)}</p>"
        f"{body}"
        f"</section>"
    )


def _chart_block(title: str, caption: str, b64: str | None, *, reason: str = "") -> str:
    if not b64:
        return _section_html(title, caption, "", missing=True, reason=reason)
    img = (
        f"<div class='chart'><img src='data:image/png;base64,{b64}' "
        f"alt='{_esc(title)}'/></div>"
    )
    return _section_html(title, caption, img)


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
  color: #1a1a1a; line-height: 1.55; background: #fff;
  padding: 24px 16px 48px; overflow-x: hidden;
}
.wrap { max-width: 920px; margin: 0 auto; }
h1 { font-size: 1.55rem; font-weight: 700; margin-bottom: 10px; }
h2 {
  font-size: 1.05rem; font-weight: 600; margin: 0 0 6px;
  border-bottom: 1px solid #ddd; padding-bottom: 4px;
}
.banner {
  background: #fff3cd; border: 2px solid #e6a800; color: #5c4500;
  padding: 12px 14px; border-radius: 4px; font-weight: 600;
  font-size: 0.92rem; margin-bottom: 14px; line-height: 1.45;
}
.meta {
  color: #555; font-size: 0.86rem; margin-bottom: 16px;
  display: flex; flex-wrap: wrap; gap: 6px 14px;
}
.meta span { white-space: nowrap; }
.cards {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 10px; margin: 12px 0 8px;
}
.card {
  border: 1px solid #e0e0e0; border-radius: 6px; padding: 10px 12px; background: #fafafa;
}
.card .label { font-size: 0.75rem; color: #666; margin-bottom: 4px; }
.card .value { font-size: 1.05rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.block { margin-top: 22px; }
.caption { color: #888; font-size: 0.82rem; margin-bottom: 8px; }
.chart { margin: 8px 0; text-align: center; overflow-x: auto; }
.chart img { max-width: 100%; height: auto; border: 1px solid #e8e8e8; }
.missing { color: #a94442; font-weight: 600; margin: 6px 0; }
.muted { color: #888; font-size: 0.85rem; }
.table-wrap { overflow-x: auto; width: 100%; }
@media (prefers-color-scheme: dark) {
  body { background: #121212; color: #e8e8e8; }
  h2 { border-bottom-color: #444; }
  .banner { background: #3d3200; border-color: #c9a227; color: #ffe9a0; }
  .meta { color: #aaa; }
  .card { background: #1e1e1e; border-color: #333; }
  .card .label { color: #999; }
  .chart img { border-color: #333; }
  .caption, .muted { color: #999; }
}
"""


def generate_signal_report(
    signal_result: Any,
    *,
    factor_name: str = "",
    date_range: str = "",
    universe: str = "",
    frequency: str = "daily",
    quality_report: dict[str, Any] | None = None,
) -> str:
    """生成信号轨单页 HTML 报告。任何缺失数据对应区块显示「未计算」，不抛异常。"""
    try:
        return _generate_signal_report_inner(
            signal_result,
            factor_name=factor_name,
            date_range=date_range,
            universe=universe,
            frequency=frequency,
            quality_report=quality_report,
        )
    except Exception:
        logger.exception("信号报告生成失败，返回降级页")
        name = _esc(factor_name or "factor")
        return (
            f"<!DOCTYPE html><html lang='zh-CN'><head><meta charset='UTF-8'>"
            f"<title>{name} — 信号轨报告</title></head><body>"
            f"<h1>{name}</h1><div class='banner'>{_esc(SIGNAL_BANNER)}</div>"
            f"<p>未计算：报告生成异常</p></body></html>"
        )


def _generate_signal_report_inner(
    signal_result: Any,
    *,
    factor_name: str,
    date_range: str,
    universe: str,
    frequency: str,
    quality_report: dict[str, Any] | None,
) -> str:
    name = factor_name or str(_safe_attr(signal_result, "factor_name", "") or "factor")
    n_groups = _safe_attr(signal_result, "n_groups", None)
    meta = _safe_attr(signal_result, "meta", {}) or {}
    if not isinstance(meta, dict):
        meta = {}
    exec_lag = meta.get("exec_lag", "—")
    exec_price = meta.get("exec_price_col", "—")
    dropped = meta.get("dropped_days", "—")

    metrics = _build_metrics(signal_result) if signal_result is not None else {
        k: _EMPTY
        for k in (
            "ic_mean", "icir", "tstat", "pvalue", "ic_pos",
            "ann_ret_gross", "sharpe_gross", "max_dd_gross", "mono_score", "avg_turnover",
        )
    }

    chart_specs: list[tuple[str, str, Any, str]] = [
        (
            "分层累计净值",
            "这张图回答：各组排序是否单调、多空毛净值有多强？",
            lambda: _chart_group_nav(signal_result),
            "缺少 group_nav",
        ),
        (
            "分层平均收益",
            "这张图回答：分位组收益是否随因子值单调递增（或递减）？",
            lambda: _chart_group_bar(signal_result),
            "缺少 group_returns",
        ),
        (
            "IC 时序",
            "这张图回答：逐日预测力是否稳定，还是偶发脉冲？",
            lambda: _chart_ic_series(signal_result),
            "缺少 ic.ic_series",
        ),
        (
            "IC 累计曲线",
            "这张图回答：alpha 是持续积累还是集中在某几段？",
            lambda: _chart_ic_cumulative(signal_result),
            "缺少 ic.ic_series",
        ),
        (
            "IC 衰减曲线",
            "这张图回答：信号半衰期有多长？",
            lambda: _chart_ic_decay(signal_result),
            "缺少 ic.decay / multi_period",
        ),
        (
            "IC 分布",
            "这张图回答：IC 是否偏态或厚尾？",
            lambda: _chart_ic_hist(signal_result),
            "缺少 ic.ic_series 或样本过少",
        ),
        (
            "分层收益年度热力图",
            "这张图回答：因子是否只在某几年有效？",
            lambda: _chart_group_year_heatmap(signal_result),
            "缺少 group_returns 或无法按年聚合",
        ),
        (
            "换手率时序",
            "这张图回答：信号组合换手是否过高、是否阶段性飙升？",
            lambda: _chart_turnover(signal_result),
            "缺少 ls_returns.ls_turnover",
        ),
    ]

    sections: list[str] = []
    for title, caption, maker, reason in chart_specs:
        b64 = _safe_chart(title, maker) if signal_result is not None else None
        sections.append(_chart_block(title, caption, b64, reason=reason))

    cards_html = "".join(
        f"<div class='card'><div class='label'>{_esc(lab)}</div>"
        f"<div class='value'>{_esc(metrics[key])}</div></div>"
        for lab, key in (
            ("IC mean", "ic_mean"),
            ("ICIR", "icir"),
            ("t-stat", "tstat"),
            ("p-value", "pvalue"),
            ("IC>0 占比", "ic_pos"),
            ("多空年化(毛)", "ann_ret_gross"),
            ("多空 Sharpe(毛)", "sharpe_gross"),
            ("多空最大回撤(毛)", "max_dd_gross"),
            ("单调性 score", "mono_score"),
            ("平均换手", "avg_turnover"),
        )
    )

    if metrics.get("_ic_unavailable"):
        cards_html += (
            "<div class='card' style='grid-column:1/-1;border-color:#c0392b'>"
            "<div class='label'>⚠️ IC 不可用</div>"
            "<div class='value' style='font-size:0.95rem;line-height:1.5'>"
            "有效 IC 天数为 0——逐日截面有效样本不足最小门槛(30)或因子与收益 join 零命中，"
            "IC 一天都没算出来。上方 IC 类指标显示「未计算」而非 0，"
            "<b>请勿把它读成「因子无预测力」</b>。常见原因：票池过小、日期格式不一致、因子全为空。"
            "</div></div>"
        )

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta_bits = [
        f"因子 {_esc(name)}",
        f"区间 {_esc(date_range or '—')}",
        f"票池 {_esc(universe or '—')}",
        f"分组数 {_esc(n_groups if n_groups is not None else '—')}",
        f"exec_lag {_esc(exec_lag)}",
        f"成交价 {_esc(exec_price)}",
        f"dropped_days {_esc(dropped)}",
        f"频率 {_esc(frequency)}",
        f"生成 {_esc(generated_at)}",
    ]
    meta_html = "".join(f"<span>{b}</span>" for b in meta_bits)

    warn_html = ""
    if isinstance(quality_report, dict):
        warns = [str(w).strip() for w in (quality_report.get("warnings") or []) if str(w).strip()]
        if warns:
            items = "".join(f"<li>{_esc(w)}</li>" for w in warns)
            warn_html = f"<section class='block'><h2>质量警告</h2><ul>{items}</ul></section>"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_esc(name)} — 信号轨报告</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>{_esc(name)} · 信号轨</h1>
  <div class="banner">{_esc(SIGNAL_BANNER)}</div>
  <div class="meta">{meta_html}</div>

  <section class="block">
    <h2>核心指标</h2>
    <p class="caption">这张表回答：因子截面预测力与多空毛收益的一览摘要。</p>
    <div class="cards">{cards_html}</div>
  </section>

  {"".join(sections)}
  {warn_html}
</div>
</body>
</html>
"""

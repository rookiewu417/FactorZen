"""交易轨报告：回答「这套信号拿去交易，能剩下多少？卡在哪里？」

数据源 ``StrategyBacktestResult``（净口径 / 含约束与成本）。
``nav`` 的 gross_return / cost / borrow_cost / cash_weight 与 ``trades.block_reason``
是本报告的灵魂数据。
"""

from __future__ import annotations

import html as html_lib
from datetime import datetime
from typing import Any

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
    _finite_float,
    _format_metric_number,
    _format_metric_percent,
    _safe_attr,
)

logger = get_logger(__name__)

TRADING_BANNER = (
    "模拟交易净口径 · 含停牌/涨跌停/T+1/容量约束与佣金+印花税+滑点+融券成本"
    " · 成交时点 t 日信号 → t+1 开盘"
)

_EMPTY = "未计算"

# block_reason 展示归类
_BLOCK_LABELS: dict[str, str] = {
    "suspended": "停牌",
    "limit_up": "涨停",
    "limit_down": "跌停",
    "capacity": "容量",
    "missing_price": "缺价",
    "invalid_portfolio_value": "组合无效",
}


def _esc(value: Any) -> str:
    return html_lib.escape(str(value if value is not None else ""), quote=True)


def _fmt_ratio2(value: Any) -> str:
    return _format_metric_number(value, digits=2, empty=_EMPTY)


def _fmt_pct2(value: Any) -> str:
    return _format_metric_percent(value, digits=2, empty=_EMPTY)


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
        logger.warning("交易报告生成%s失败", label, exc_info=True)
        return None
    return b64 if b64 else None


def _nav_frame(bt_result: Any) -> Any | None:
    nav = _safe_attr(bt_result, "nav")
    if _is_empty_frame(nav):
        # 部分路径 returns 与 nav 同结构
        ret = _safe_attr(bt_result, "returns")
        if not _is_empty_frame(ret) and "nav" in getattr(ret, "columns", []):
            return ret
        return None
    return nav


# ── pure display metrics (testable) ──────────────────────────────────────────


def compute_cost_erosion(nav: Any) -> dict[str, float] | None:
    """从 nav 帧累计毛收益/成本/净收益，计算成本侵蚀比例。

    侵蚀比例 = (sum(cost) + sum(borrow_cost)) / |sum(gross_return)|
    当 |sum(gross_return)| 接近 0 时返回 None。
    """
    if _is_empty_frame(nav):
        return None
    try:
        df = nav
        cols = set(df.columns)
    except Exception:
        return None
    if "gross_return" not in cols:
        return None
    gross = np.asarray(df["gross_return"].to_numpy(), dtype=float)
    cost = (
        np.asarray(df["cost"].to_numpy(), dtype=float)
        if "cost" in cols
        else np.zeros_like(gross)
    )
    borrow = (
        np.asarray(df["borrow_cost"].to_numpy(), dtype=float)
        if "borrow_cost" in cols
        else np.zeros_like(gross)
    )
    net = (
        np.asarray(df["net_return"].to_numpy(), dtype=float)
        if "net_return" in cols
        else gross - cost - borrow
    )
    gross = np.where(np.isfinite(gross), gross, 0.0)
    cost = np.where(np.isfinite(cost), cost, 0.0)
    borrow = np.where(np.isfinite(borrow), borrow, 0.0)
    net = np.where(np.isfinite(net), net, 0.0)
    sum_gross = float(np.sum(gross))
    sum_cost = float(np.sum(cost))
    sum_borrow = float(np.sum(borrow))
    sum_net = float(np.sum(net))
    denom = abs(sum_gross)
    if denom < 1e-15:
        return {
            "sum_gross": sum_gross,
            "sum_cost": sum_cost,
            "sum_borrow": sum_borrow,
            "sum_net": sum_net,
            "erosion_ratio": float("nan"),
        }
    return {
        "sum_gross": sum_gross,
        "sum_cost": sum_cost,
        "sum_borrow": sum_borrow,
        "sum_net": sum_net,
        "erosion_ratio": (sum_cost + sum_borrow) / denom,
    }


def compute_avg_net_exposure(nav: Any) -> float | None:
    """平均**净**敞口 = mean(1 − cash_weight) = mean(Σw)。

    注意这是净敞口不是资金占用：多空策略两腿对冲后 Σw≈0，该值天然接近零，
    **不能**用来判断「建没建上仓」——那要看总敞口（``compute_avg_gross_exposure``）。
    """
    if _is_empty_frame(nav):
        return None
    try:
        if "cash_weight" not in nav.columns:
            return None
        cw = np.asarray(nav["cash_weight"].to_numpy(), dtype=float)
    except Exception:
        return None
    net = 1.0 - cw
    net = net[np.isfinite(net)]
    if net.size == 0:
        return None
    return float(np.mean(net))


def gross_exposure_series(positions: Any) -> Any | None:
    """逐日总敞口 Σ|weight|（资金实际用了多少的唯一正确度量）。

    long-only 与多空通用：long-only 下 gross≈net，gross 远低于 1 说明大量资金
    未建仓（participation/ADV 容量约束吃掉了下单量）；多空下 gross≈2、net≈0。
    """
    if _is_empty_frame(positions):
        return None
    try:
        if "weight" not in positions.columns or "trade_date" not in positions.columns:
            return None
        return (
            positions.group_by("trade_date")
            .agg(pl.col("weight").abs().sum().alias("gross_exposure"))
            .sort("trade_date")
        )
    except Exception:
        return None


def compute_avg_gross_exposure(positions: Any) -> float | None:
    """平均总敞口 = mean(Σ|w|)。"""
    frame = gross_exposure_series(positions)
    if frame is None or _is_empty_frame(frame):
        return None
    try:
        arr = np.asarray(frame["gross_exposure"].to_numpy(), dtype=float)
    except Exception:
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.mean(arr))


# ── charts ───────────────────────────────────────────────────────────────────


def _chart_nav(bt_result: Any, benchmark_result: Any) -> str | None:
    nav = _nav_frame(bt_result)
    if nav is None:
        return None
    npd = nav.to_pandas()
    if "nav" not in npd.columns or "trade_date" not in npd.columns:
        return None
    if "group" in npd.columns:
        return None
    npd = npd.dropna(subset=["nav"]).sort_values("trade_date")
    if len(npd) < 1:
        return None
    npd, x_col, is_date_axis = _with_plot_dates(npd)

    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.plot(npd[x_col], npd["nav"], color="#2c7fb8", linewidth=1.6, label="策略净值")

    if benchmark_result is not None:
        daily = _safe_attr(benchmark_result, "daily")
        if not _is_empty_frame(daily):
            bpd = daily.to_pandas()
            if "benchmark_nav" in bpd.columns and "trade_date" in bpd.columns:
                bpd = bpd.dropna(subset=["benchmark_nav"]).sort_values("trade_date")
                if not bpd.empty:
                    bpd, bx, is_date_axis = _with_plot_dates(bpd)
                    bname = str(_safe_attr(benchmark_result, "benchmark_name", "") or "基准")
                    ax.plot(
                        bpd[bx],
                        bpd["benchmark_nav"],
                        color="#7f8c8d",
                        linewidth=1.2,
                        label=bname,
                    )

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_title("净值曲线", fontsize=11)
    ax.set_ylabel("净值")
    ax.legend(fontsize=8, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    _format_sparse_x_axis(ax, is_date_axis=is_date_axis)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _chart_drawdown(bt_result: Any) -> str | None:
    nav = _nav_frame(bt_result)
    if nav is None:
        return None
    npd = nav.to_pandas()
    if "nav" not in npd.columns or "trade_date" not in npd.columns:
        return None
    if "group" in npd.columns:
        return None
    npd = npd.sort_values("trade_date")
    navs = npd["nav"].to_numpy(dtype=float)
    finite = np.isfinite(navs) & (navs > 0)
    if finite.sum() < 2:
        return None
    npd = npd.loc[finite]
    navs = navs[finite]
    running_max = np.maximum.accumulate(navs)
    drawdown = navs / running_max - 1.0
    npd, x_col, is_date_axis = _with_plot_dates(npd)
    x_vals = npd[x_col].to_numpy()

    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.fill_between(x_vals, drawdown, 0, color="#c0392b", alpha=0.28)
    ax.plot(x_vals, drawdown, color="#c0392b", linewidth=1.0)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    trough = int(np.argmin(drawdown))
    max_dd = float(drawdown[trough])
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
    ax.set_title("回撤水下图", fontsize=11)
    ax.set_ylabel("回撤")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    _format_sparse_x_axis(ax, is_date_axis=is_date_axis)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _chart_cost_waterfall(bt_result: Any) -> str | None:
    nav = _nav_frame(bt_result)
    stats = compute_cost_erosion(nav) if nav is not None else None
    if stats is None:
        return None
    g = stats["sum_gross"]
    c = stats["sum_cost"]
    b = stats["sum_borrow"]
    n = stats["sum_net"]
    erosion = stats["erosion_ratio"]

    # 瀑布：毛收益 → −交易成本 → −融券成本 → 净收益
    labels = ["毛收益", "交易成本", "融券成本", "净收益"]
    values = [g, -c, -b, n]
    # 瀑布连接
    running = 0.0
    bottoms: list[float] = []
    heights: list[float] = []
    colors: list[str] = []
    for i, v in enumerate(values):
        if i == 0:
            bottoms.append(0.0)
            heights.append(v)
            colors.append("#27ae60" if v >= 0 else "#c0392b")
            running = v
        elif i == len(values) - 1:
            bottoms.append(0.0)
            heights.append(v)
            colors.append("#2980b9")
        else:
            # 成本为负向
            if v <= 0:
                bottoms.append(running + v)
                heights.append(-v)
                colors.append("#e67e22")
                running = running + v
            else:
                bottoms.append(running)
                heights.append(v)
                colors.append("#27ae60")
                running = running + v

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    x = np.arange(len(labels))
    ax.bar(x, heights, bottom=bottoms, color=colors, width=0.55, alpha=0.9)
    # 连接线
    conn_y = g
    for i in range(1, 3):
        ax.plot([i - 0.3, i + 0.3], [conn_y, conn_y], color="#888", linewidth=0.8)
        conn_y = conn_y + values[i]
    for i, v in enumerate(values):
        ax.annotate(
            f"{v * 100:.2f}%",
            xy=(i, bottoms[i] + heights[i] if i not in (0, 3) else max(v, 0)),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    er_txt = (
        f"成本侵蚀比例 {(erosion * 100):.1f}%"
        if np.isfinite(erosion)
        else "成本侵蚀比例 不可计（毛收益≈0）"
    )
    ax.set_title(f"成本侵蚀瀑布图 · {er_txt}", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(y=0, color="gray", linewidth=0.6)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylabel("累计收益贡献")
    fig.tight_layout()
    return _fig_to_base64(fig)


def _chart_utilization(bt_result: Any) -> str | None:
    """总敞口 Σ|w| 与净敞口 Σw 双线。

    只画净敞口会对多空策略失去判别力（两腿对冲后 Σw≈0，看不出建没建上仓）；
    总敞口才是资金占用的度量。两条线同图，long-only 与多空都能读。
    """
    gross = gross_exposure_series(_safe_attr(bt_result, "positions"))
    nav = _nav_frame(bt_result)

    gpd = None
    if gross is not None and not _is_empty_frame(gross):
        gpd = gross.to_pandas().sort_values("trade_date")

    npd = None
    if nav is not None:
        tmp = nav.to_pandas()
        if "cash_weight" in tmp.columns and "trade_date" in tmp.columns:
            tmp = tmp.dropna(subset=["cash_weight"]).sort_values("trade_date")
            if not tmp.empty:
                npd = tmp

    if gpd is None and npd is None:
        return None

    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    note = ""
    if gpd is not None:
        g_arr = gpd["gross_exposure"].to_numpy(dtype=float)
        mean_g = float(np.nanmean(g_arr)) if np.isfinite(g_arr).any() else float("nan")
        gplot, gx, g_is_date = _with_plot_dates(gpd)
        ax.plot(gplot[gx], g_arr, color="#8e44ad", linewidth=1.2, label="总敞口 Σ|w|")
        if np.isfinite(mean_g):
            ax.axhline(
                y=mean_g, color="#c0392b", linestyle="--", linewidth=1.0,
                label=f"总敞口均值 {mean_g:.1%}",
            )
            note = f"总敞口均值 {mean_g:.1%}"
        _format_sparse_x_axis(ax, is_date_axis=g_is_date)
    if npd is not None:
        n_arr = 1.0 - npd["cash_weight"].to_numpy(dtype=float)
        nplot, nx, n_is_date = _with_plot_dates(npd)
        ax.plot(
            nplot[nx], n_arr, color="#2980b9", linewidth=1.0,
            linestyle=":", label="净敞口 Σw",
        )
        if gpd is None:
            _format_sparse_x_axis(ax, is_date_axis=n_is_date)

    ax.axhline(y=0.0, color="#7f8c8d", linewidth=0.8)
    ax.set_title(
        "敞口时序（总敞口=资金实际用了多少；long-only 下远低于 1 = 大量资金未建仓，"
        "多空下 ≈2 属正常，此时净敞口≈0 亦属正常）",
        fontsize=9,
    )
    ax.set_ylabel("权重和")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=8)
    if note:
        ax.margins(y=0.15)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _fig_to_base64(fig)


def _chart_block_reasons(bt_result: Any) -> str | None:
    trades = _safe_attr(bt_result, "trades")
    if _is_empty_frame(trades):
        return None
    try:
        tpd = trades.to_pandas()
    except Exception:
        return None
    if "block_reason" not in tpd.columns:
        return None
    reasons = tpd["block_reason"].fillna("").astype(str)
    # 空串 = 正常成交：单独统计但不主导图；用户要求按实际分布决定
    blocked = reasons[reasons.str.len() > 0]
    if blocked.empty:
        # 全是正常成交：画一个「正常成交」柱
        counts = {"正常成交": len(reasons)}
    else:
        counts = blocked.value_counts().to_dict()
        ok_n = int((reasons.str.len() == 0).sum())
        if ok_n > 0:
            counts["正常成交"] = ok_n

    labels = []
    values = []
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        labels.append(_BLOCK_LABELS.get(k, k))
        values.append(int(v))
    if not values:
        return None

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    colors = ["#95a5a6" if lab == "正常成交" else "#e74c3c" for lab in labels]
    ax.bar(range(len(labels)), values, color=colors, alpha=0.85, width=0.62)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    for i, v in enumerate(values):
        ax.annotate(str(v), xy=(i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_title("拒单原因分布", fontsize=11)
    ax.set_ylabel("笔数")
    ax.margins(y=0.15)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _chart_monthly_heatmap(bt_result: Any) -> str | None:
    returns = _safe_attr(bt_result, "returns")
    if _is_empty_frame(returns):
        nav = _nav_frame(bt_result)
        returns = nav
    if _is_empty_frame(returns):
        return None
    ret_pd = returns.to_pandas()
    if "trade_date" not in ret_pd.columns:
        return None
    ret_col = next(
        (c for c in ("net_return", "ret", "return") if c in ret_pd.columns),
        None,
    )
    if ret_col is None:
        return None
    dates = ret_pd["trade_date"].astype(str)
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
    fig, ax = plt.subplots(figsize=(9.5, max(2.6, 0.55 * len(monthly.index) + 1.4)))
    vmax = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 0.01
    vmax = max(vmax, 0.01)
    image = ax.imshow(masked, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("月度收益热力图（净收益）", fontsize=11)
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
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, format=mticker.PercentFormatter(1.0))
    fig.tight_layout()
    return _fig_to_base64(fig)


def _chart_turnover_cost(bt_result: Any) -> str | None:
    nav = _nav_frame(bt_result)
    returns = _safe_attr(bt_result, "returns")
    src: Any = returns if not _is_empty_frame(returns) else nav
    if src is None or _is_empty_frame(src):
        return None
    pd_ = src.to_pandas().sort_values("trade_date") if "trade_date" in src.columns else src.to_pandas()
    if "trade_date" not in pd_.columns:
        return None
    has_to = "turnover" in pd_.columns
    has_cost = "cost" in pd_.columns
    if not has_to and not has_cost:
        return None
    pd_, x_col, is_date_axis = _with_plot_dates(pd_)

    fig, ax1 = plt.subplots(figsize=(9.5, 3.6))
    if has_to:
        to = pd_["turnover"].to_numpy(dtype=float)
        ax1.plot(pd_[x_col], to, color="#16a085", linewidth=1.0, alpha=0.85, label="换手率")
        ax1.set_ylabel("换手率", color="#16a085")
        ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    if has_cost:
        cost = pd_["cost"].to_numpy(dtype=float)
        cost = np.where(np.isfinite(cost), cost, 0.0)
        cum_cost = np.cumsum(cost)
        ax2 = ax1.twinx()
        ax2.plot(pd_[x_col], cum_cost, color="#e67e22", linewidth=1.3, label="累计成本")
        ax2.set_ylabel("累计成本 / 初始净值", color="#e67e22")
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.set_title("换手率与累计成本", fontsize=11)
    _format_sparse_x_axis(ax1, is_date_axis=is_date_axis)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _chart_rolling_sharpe(bt_result: Any, window: int = 60) -> str | None:
    returns = _safe_attr(bt_result, "returns")
    if _is_empty_frame(returns):
        nav = _nav_frame(bt_result)
        returns = nav
    if _is_empty_frame(returns):
        return None
    rpd = returns.to_pandas()
    if "trade_date" not in rpd.columns:
        return None
    ret_col = next((c for c in ("net_return", "ret") if c in rpd.columns), None)
    if ret_col is None:
        return None
    rpd = rpd.dropna(subset=[ret_col]).sort_values("trade_date")
    if len(rpd) < max(5, window // 4):
        return None
    rets = rpd[ret_col].to_numpy(dtype=float)
    # 滚动年化 Sharpe
    import pandas as pd

    s = pd.Series(rets)
    roll_mean = s.rolling(window, min_periods=max(10, window // 3)).mean()
    roll_std = s.rolling(window, min_periods=max(10, window // 3)).std()
    sharpe = (roll_mean / roll_std) * np.sqrt(252.0)
    rpd, x_col, is_date_axis = _with_plot_dates(rpd)
    vals = sharpe.to_numpy()
    if not np.isfinite(vals).any():
        return None

    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.plot(rpd[x_col], vals, color="#2c3e50", linewidth=1.2)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.6)
    ax.set_title(f"滚动 Sharpe（{window} 日）", fontsize=11)
    ax.set_ylabel("Sharpe")
    _format_sparse_x_axis(ax, is_date_axis=is_date_axis)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


# ── metrics / HTML ───────────────────────────────────────────────────────────


def _portfolio_stats(bt_result: Any) -> dict[str, Any]:
    if bt_result is None:
        return {}
    stats = _safe_attr(bt_result, "summary_stats", {}) or {}
    if not isinstance(stats, dict):
        return {}
    portfolio = stats.get("portfolio") or stats.get("long_short")
    return portfolio if isinstance(portfolio, dict) else {}


def _build_metrics(bt_result: Any) -> dict[str, str]:
    portfolio = _portfolio_stats(bt_result)
    nav = _nav_frame(bt_result)

    ann_ret = _finite_float(portfolio.get("ann_ret"))
    sharpe = _finite_float(portfolio.get("sharpe"))
    max_dd = _finite_float(portfolio.get("max_dd"))
    ann_vol = _finite_float(portfolio.get("ann_vol"))
    avg_to = _finite_float(portfolio.get("avg_turnover"))

    calmar = None
    if ann_ret is not None and max_dd is not None and abs(max_dd) > 1e-12:
        calmar = ann_ret / abs(max_dd)

    erosion = compute_cost_erosion(nav) if nav is not None else None
    cost_ratio = None
    if erosion is not None and np.isfinite(erosion.get("erosion_ratio", float("nan"))):
        cost_ratio = erosion["erosion_ratio"]

    # 总敞口=资金占用(多空/long-only 通用);净敞口对多空恒≈0,单独看无判别力
    gross_exp = compute_avg_gross_exposure(_safe_attr(bt_result, "positions"))
    net_exp = compute_avg_net_exposure(nav) if nav is not None else None

    # 日度胜率
    win_rate: float | None = None
    returns = _safe_attr(bt_result, "returns") if bt_result is not None else None
    src: Any = returns if not _is_empty_frame(returns) else nav
    if src is not None and not _is_empty_frame(src) and "net_return" in getattr(src, "columns", []):
        rets_arr = np.asarray(src["net_return"].to_numpy(), dtype=np.float64)
        rets_arr = rets_arr[np.isfinite(rets_arr)]
        if rets_arr.size > 0:
            win_rate = float(np.mean(rets_arr > 0))

    out: dict[str, str] = {
        "ann_ret": _fmt_pct2(ann_ret),
        "sharpe": _fmt_ratio2(sharpe),
        "max_dd": _fmt_pct2(max_dd),
        "calmar": _fmt_ratio2(calmar),
        "ann_vol": _fmt_pct2(ann_vol),
        "avg_turnover": _fmt_pct2(avg_to),
        "cost_ratio": _fmt_pct2(cost_ratio),
        "avg_gross_exp": _fmt_pct2(gross_exp),
        "avg_net_exp": _fmt_pct2(net_exp),
        "win_rate": _fmt_pct2(win_rate),
    }
    return out


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
  background: #eef2f7; border: 1px solid #8aa0b8; color: #243447;
  padding: 12px 14px; border-radius: 4px; font-weight: 500;
  font-size: 0.9rem; margin-bottom: 14px; line-height: 1.45;
}
.meta {
  color: #555; font-size: 0.86rem; margin-bottom: 16px;
  display: flex; flex-wrap: wrap; gap: 6px 14px;
}
.meta span { white-space: nowrap; }
.dir-badge {
  display: inline-block; font-size: 0.85rem; padding: 4px 10px;
  border: 1px solid #c44; background: #fff0f0; color: #a22;
  border-radius: 3px; margin-bottom: 8px;
}
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
@media (prefers-color-scheme: dark) {
  body { background: #121212; color: #e8e8e8; }
  h2 { border-bottom-color: #444; }
  .banner { background: #1a2533; border-color: #4a6a8a; color: #c5d4e8; }
  .meta { color: #aaa; }
  .card { background: #1e1e1e; border-color: #333; }
  .card .label { color: #999; }
  .chart img { border-color: #333; }
  .caption, .muted { color: #999; }
  .dir-badge { background: #3a1a1a; border-color: #a44; color: #f0a0a0; }
}
"""


def generate_trading_report(
    factor_name: str,
    bt_result: Any,
    *,
    date_range: str = "",
    universe: str = "",
    strategy_name: str = "",
    cost_model: str = "",
    backtest_direction: dict[str, Any] | None = None,
    benchmark_result: Any = None,
    walk_forward_summary: dict[str, Any] | None = None,
    quality_report: dict[str, Any] | None = None,
) -> str:
    """生成交易轨单页 HTML 报告。任何缺失数据对应区块显示「未计算」，不抛异常。"""
    try:
        return _generate_trading_report_inner(
            factor_name,
            bt_result,
            date_range=date_range,
            universe=universe,
            strategy_name=strategy_name,
            cost_model=cost_model,
            backtest_direction=backtest_direction,
            benchmark_result=benchmark_result,
            walk_forward_summary=walk_forward_summary,
            quality_report=quality_report,
        )
    except Exception:
        logger.exception("交易报告生成失败，返回降级页")
        name = _esc(factor_name or "factor")
        return (
            f"<!DOCTYPE html><html lang='zh-CN'><head><meta charset='UTF-8'>"
            f"<title>{name} — 交易轨报告</title></head><body>"
            f"<h1>{name}</h1><div class='banner'>{_esc(TRADING_BANNER)}</div>"
            f"<p>未计算：报告生成异常</p></body></html>"
        )


def _generate_trading_report_inner(
    factor_name: str,
    bt_result: Any,
    *,
    date_range: str,
    universe: str,
    strategy_name: str,
    cost_model: str,
    backtest_direction: dict[str, Any] | None,
    benchmark_result: Any,
    walk_forward_summary: dict[str, Any] | None,
    quality_report: dict[str, Any] | None,
) -> str:
    name = factor_name or str(_safe_attr(bt_result, "factor_name", "") or "factor")
    strat = strategy_name or str(_safe_attr(bt_result, "strategy_name", "") or "—")
    if not cost_model and bt_result is not None:
        cfg = _safe_attr(bt_result, "config", {}) or {}
        if isinstance(cfg, dict):
            cm = cfg.get("cost_model")
            cost_model = str(cm) if cm is not None else ""
    cost_disp = cost_model or "—"

    metrics = _build_metrics(bt_result) if bt_result is not None else {
        k: _EMPTY
        for k in (
            "ann_ret", "sharpe", "max_dd", "calmar", "ann_vol",
            "avg_turnover", "cost_ratio", "avg_gross_exp", "avg_net_exp", "win_rate",
        )
    }

    direction_html = ""
    if backtest_direction:
        direction = str(backtest_direction.get("direction", "normal") or "normal")
        reason = str(backtest_direction.get("reason", "") or "")
        if direction == "reversed":
            label = "反向信号（做多低因子值）"
            direction_html = f"<div class='dir-badge'>{_esc(label)}</div>"
        else:
            label = "正向信号"
            direction_html = f"<p class='muted'>{_esc(label)}</p>"
        if reason:
            direction_html += f"<p class='muted'>判定原因：{_esc(reason)}</p>"

    chart_specs: list[tuple[str, str, Any, str]] = [
        (
            "净值曲线",
            "这张图回答：策略净净值走成什么样？相对基准如何？",
            lambda: _chart_nav(bt_result, benchmark_result),
            "缺少 nav 净值序列",
        ),
        (
            "回撤水下图",
            "这张图回答：回撤是单次崩塌还是长期阴跌？",
            lambda: _chart_drawdown(bt_result),
            "缺少 nav 或有效点不足",
        ),
        (
            "成本侵蚀瀑布图",
            "这张图回答：成本吃掉了毛 alpha 的百分之多少？",
            lambda: _chart_cost_waterfall(bt_result),
            "缺少 gross_return / cost 列",
        ),
        (
            "敞口时序",
            "这张图回答：有多少资金真正建仓（总敞口）？是否存在 participation 假象？"
            "多空策略看总敞口，净敞口≈0 是正常的。",
            lambda: _chart_utilization(bt_result),
            "缺少 cash_weight",
        ),
        (
            "拒单原因分布",
            "这张图回答：可交易性瓶颈卡在停牌、涨跌停还是容量？",
            lambda: _chart_block_reasons(bt_result),
            "缺少 trades 或 block_reason",
        ),
        (
            "月度收益热力图",
            "这张图回答：净收益在哪些年月集中？",
            lambda: _chart_monthly_heatmap(bt_result),
            "缺少 returns.net_return",
        ),
        (
            "换手率与累计成本",
            "这张图回答：换手与成本是否同步抬升？",
            lambda: _chart_turnover_cost(bt_result),
            "缺少 turnover / cost",
        ),
        (
            "滚动 Sharpe",
            "这张图回答：绩效是否稳定，还是靠某段暴涨？",
            lambda: _chart_rolling_sharpe(bt_result),
            "缺少 net_return 或样本过短",
        ),
    ]

    sections: list[str] = []
    for title, caption, maker, reason in chart_specs:
        b64 = _safe_chart(title, maker) if bt_result is not None else None
        sections.append(_chart_block(title, caption, b64, reason=reason))

    cards_html = "".join(
        f"<div class='card'><div class='label'>{_esc(lab)}</div>"
        f"<div class='value'>{_esc(metrics[key])}</div></div>"
        for lab, key in (
            ("年化收益(净)", "ann_ret"),
            ("Sharpe", "sharpe"),
            ("最大回撤", "max_dd"),
            ("Calmar", "calmar"),
            ("年化波动", "ann_vol"),
            ("换手率", "avg_turnover"),
            ("成本占毛收益比", "cost_ratio"),
            ("平均总敞口 Σ|w|", "avg_gross_exp"),
            ("平均净敞口 Σw", "avg_net_exp"),
            ("胜率(日度)", "win_rate"),
        )
    )

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta_bits = [
        f"因子 {_esc(name)}",
        f"区间 {_esc(date_range or '—')}",
        f"票池 {_esc(universe or '—')}",
        f"策略 {_esc(strat)}",
        f"成本模型 {_esc(cost_disp)}",
        f"生成 {_esc(generated_at)}",
    ]
    meta_html = "".join(f"<span>{b}</span>" for b in meta_bits)

    wf_html = ""
    if isinstance(walk_forward_summary, dict) and walk_forward_summary:
        status = walk_forward_summary.get("status", "—")
        wf_html = (
            f"<section class='block'><h2>Walk-forward</h2>"
            f"<p class='caption'>这张表回答：样本外是否仍稳。</p>"
            f"<p class='muted'>status={_esc(status)}"
        )
        if status == "ok":
            oos = walk_forward_summary.get("oos_sharpe_mean")
            stab = walk_forward_summary.get("stability_ratio")
            wf_html += f" · OOS Sharpe={_esc(_fmt_ratio2(oos))} · 稳定率={_esc(_fmt_ratio2(stab))}"
        wf_html += "</p></section>"

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
<title>{_esc(name)} — 交易轨报告</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>{_esc(name)} · 交易轨</h1>
  <div class="banner">{_esc(TRADING_BANNER)}</div>
  <div class="meta">{meta_html}</div>
  {direction_html}

  <section class="block">
    <h2>核心指标</h2>
    <p class="caption">这张表回答：可交易净绩效与成本/仓位约束下的一览摘要。</p>
    <div class="cards">{cards_html}</div>
  </section>

  {"".join(sections)}
  {wf_html}
  {warn_html}
</div>
</body>
</html>
"""

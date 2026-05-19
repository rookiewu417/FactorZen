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
from typing import Any

import jinja2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # 非交互后端，不弹窗
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Windows 中文字体支持（优先 Microsoft YaHei，回退 SimHei）
for _font in ["Microsoft YaHei", "SimHei", "sans-serif"]:
    matplotlib.rcParams["font.family"] = _font
    matplotlib.rcParams["axes.unicode_minus"] = False
    break

from common.logger import get_logger  # noqa: E402
from config.constants import MIN_BACKTEST_IR, STAR_RATING_THRESHOLDS  # noqa: E402

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


def _make_returns_chart(bt_result: Any, factor_name: str) -> str | None:
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
                grp_data["trade_date"],
                grp_data["nav"],
                linewidth=1.2,
                label=f"Q{g + 1}" if isinstance(g, (int, float)) else str(g),
            )
    elif "trade_date" in nav_pd.columns and "nav" in nav_pd.columns:
        nav_pd = nav_pd.sort_values("trade_date")
        ax.plot(nav_pd["trade_date"], nav_pd["nav"], linewidth=1.4, label="Portfolio")
    else:
        for col in nav_pd.columns:
            if col == "trade_date":
                continue
            ax.plot(nav_pd["trade_date"], nav_pd[col], linewidth=1.2, label=str(col))

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_title(f"分层回测 NAV — {factor_name}", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _make_ic_chart(ic_result: Any) -> str | None:
    """IC 时序棒图 + 滚动均值。"""
    if ic_result is None:
        return None
    ic_series = _safe_attr(ic_result, "ic_series")
    if ic_series is None or ic_series.is_empty():
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    ic_pd = ic_series.to_pandas()
    ic_col = "ic" if "ic" in ic_pd.columns else next(c for c in ic_pd.columns if c != "trade_date")
    date_col = "trade_date" if "trade_date" in ic_pd.columns else ic_pd.columns[0]

    ax.bar(ic_pd[date_col], ic_pd[ic_col], width=1.0, color="#bdc3c7", alpha=0.6, label="IC")
    if len(ic_pd) >= 5:
        window = min(20, max(3, len(ic_pd) // 3))
        rolling = ic_pd[ic_col].rolling(window=window, min_periods=1).mean()
        ax.plot(
            ic_pd[date_col], rolling, color="#e74c3c", linewidth=1.2, label=f"滚动均值({window}期)"
        )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title("Rank IC 时序", fontsize=12)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _make_benchmark_chart(benchmark_result: Any) -> str | None:
    """基准对比图：策略 NAV vs 基准 NAV（上）+ 超额 NAV（下）。"""
    if benchmark_result is None:
        return None
    daily = _safe_attr(benchmark_result, "daily")
    if daily is None or daily.is_empty():
        return None

    df = daily.to_pandas()
    if "trade_date" not in df.columns:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    benchmark_name = _safe_attr(benchmark_result, "benchmark_name", "Benchmark")

    if "strategy_nav" in df.columns:
        df_sorted = df.sort_values("trade_date")
        ax1.plot(df_sorted["trade_date"], df_sorted["strategy_nav"], linewidth=1.4, label="策略")
    if "benchmark_nav" in df.columns:
        df_sorted = df.sort_values("trade_date")
        ax1.plot(
            df_sorted["trade_date"],
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
            df_sorted["trade_date"],
            df_sorted["excess_nav"],
            linewidth=1.2,
            color="#e74c3c",
            label="超额 NAV",
        )
        ax2.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
        ax2.set_title("超额净值", fontsize=11)
        ax2.legend(fontsize=8, loc="upper left")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))

    fig.autofmt_xdate()
    fig.tight_layout()
    return _fig_to_base64(fig)


def _make_attribution_chart(brinson_result: Any, barra_result: Any) -> str | None:
    """归因分析图：Brinson 行业堆积条形（上）+ Barra 风格暴露（下）。"""
    if brinson_result is None and barra_result is None:
        return None

    n_panels = (1 if brinson_result is not None else 0) + (1 if barra_result is not None else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4.5 * n_panels))
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0

    if brinson_result is not None:
        sector_df = _safe_attr(brinson_result, "sector_df")
        if sector_df is not None and not sector_df.is_empty():
            sdf = sector_df.to_pandas()
            ax = axes[ax_idx]
            ax_idx += 1
            sectors = sdf["sector"].tolist() if "sector" in sdf.columns else list(range(len(sdf)))
            alloc = sdf["allocation"].tolist() if "allocation" in sdf.columns else [0] * len(sdf)
            selection = (
                sdf["selection"].tolist() if "selection" in sdf.columns else [0] * len(sdf)
            )
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
            ax.legend(fontsize=8, loc="lower right")
        else:
            ax_idx += 1

    if barra_result is not None:
        exposures = _safe_attr(barra_result, "exposures", {})
        if exposures:
            ax = axes[ax_idx]
            ax_idx += 1
            styles = list(exposures.keys())
            betas = [exposures[s] for s in styles]
            colors = ["#27ae60" if b >= 0 else "#e74c3c" for b in betas]
            y_pos = range(len(styles))
            ax.barh(list(y_pos), betas, color=colors, alpha=0.8)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(styles, fontsize=9)
            ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.6)
            ax.set_title("Barra 风格因子暴露", fontsize=12)
        else:
            ax_idx += 1

    fig.tight_layout()
    return _fig_to_base64(fig)


def _make_walk_forward_chart(wf_result: Any) -> str | None:
    """Walk-forward 图：(上) OOS 拼接净值; (下) IS vs OOS Sharpe 分折柱状图。"""
    if wf_result is None:
        return None
    oos_returns = _safe_attr(wf_result, "oos_returns")
    folds = _safe_attr(wf_result, "folds", [])
    if oos_returns is None or oos_returns.is_empty() or not folds:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # 上：OOS 拼接净值
    oos_pd = oos_returns.to_pandas()
    if "trade_date" in oos_pd.columns and "nav" in oos_pd.columns:
        oos_pd = oos_pd.sort_values("trade_date")
        ax1.plot(oos_pd["trade_date"], oos_pd["nav"], linewidth=1.4, color="#2980b9", label="OOS NAV")
        ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax1.set_title("Walk-Forward OOS 累计净值", fontsize=12)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax1.legend(fontsize=8)
    fig.autofmt_xdate()

    # 下：IS vs OOS Sharpe 分折柱状图
    fold_ids = [f.fold_id for f in folds]
    is_sharpes = [f.is_sharpe for f in folds]
    oos_sharpes = [f.oos_sharpe for f in folds]
    x = range(len(fold_ids))
    width = 0.35
    ax2.bar([xi - width / 2 for xi in x], is_sharpes, width, label="IS Sharpe", color="#3498db", alpha=0.7)
    ax2.bar([xi + width / 2 for xi in x], oos_sharpes, width, label="OOS Sharpe", color="#e74c3c", alpha=0.7)
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([f"Fold {fid}" for fid in fold_ids], fontsize=8)
    ax2.set_title("各折 IS / OOS Sharpe 对比", fontsize=12)
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
    if avg_cumret is None or len(windows) == 0:
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
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    fig.tight_layout()
    return _fig_to_base64(fig)


def _make_turnover_chart(to_result: Any) -> str | None:
    """换手率填充图。"""
    if to_result is None:
        return None
    dt = _safe_attr(to_result, "daily_turnover")
    if dt is None or dt.is_empty():
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    dt_pd = dt.to_pandas()
    date_col = "trade_date" if "trade_date" in dt_pd.columns else dt_pd.columns[0]
    val_col = next(c for c in dt_pd.columns if c != date_col)
    ax.fill_between(dt_pd[date_col], dt_pd[val_col], alpha=0.3, color="#9b59b6")
    ax.plot(dt_pd[date_col], dt_pd[val_col], linewidth=1.2, color="#8e44ad")
    ax.set_title("周期换手率", fontsize=12)
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
        rows.append(
            {
                "group": f"Q{key + 1}",
                "ann_ret": f"{gs.get('ann_ret', 0) * 100:.2f}%",
                "ann_vol": f"{gs.get('ann_vol', 0) * 100:.2f}%",
                "sharpe": f"{gs.get('sharpe', 0):.3f}",
                "max_dd": f"{gs.get('max_dd', 0) * 100:.2f}%",
            }
        )
    if "long_short" in stats:
        ls = stats["long_short"]
        rows.append(
            {
                "group": "L/S",
                "ann_ret": f"{ls.get('ann_ret', 0) * 100:.2f}%",
                "ann_vol": f"{ls.get('ann_vol', 0) * 100:.2f}%",
                "sharpe": f"{ls.get('sharpe', 0):.3f}",
                "max_dd": f"{ls.get('max_dd', 0) * 100:.2f}%",
            }
        )
    return rows


def _extract_metrics(ic_result, bt_result, to_result, advanced_results) -> dict[str, Any]:
    """提取所有关键指标为扁平字典。"""
    m: dict[str, Any] = {}

    m["ic_mean"] = _safe_attr(ic_result, "ic_mean", 0) or 0
    m["ic_std"] = _safe_attr(ic_result, "ic_std", 0) or 0
    m["ir"] = _safe_attr(ic_result, "ir", 0) or 0
    m["ic_positive_ratio"] = _safe_attr(ic_result, "ic_positive_ratio", 0) or 0
    m["n_periods"] = _safe_attr(ic_result, "n_periods", 0) or 0
    m["decay"] = _safe_attr(ic_result, "decay", {})
    m["ic_tstat"] = _safe_attr(ic_result, "ic_tstat", 0.0) or 0.0
    m["ic_pvalue"] = _safe_attr(ic_result, "ic_pvalue", 1.0)
    # Multi-period consistency: {horizon: {ic_mean, ic_std, ir, ic_positive_ratio}}
    multi_period = _safe_attr(ic_result, "multi_period", {})
    if multi_period:
        m["multi_period_table"] = [
            {
                "horizon": f"{h}d",
                "ic_mean": v.get("ic_mean", 0),
                "ic_std": v.get("ic_std", 0),
                "ir": v.get("ir", 0),
                "ic_pos": v.get("ic_positive_ratio", 0),
                "tstat": v.get("tstat", 0),
                "pvalue": v.get("pvalue", 1.0),
            }
            for h, v in sorted(multi_period.items())
        ]
    # Out-of-sample split
    oos_ic = _safe_attr(ic_result, "oos_ic", {})
    if oos_ic:
        m["oos_train_ic"] = oos_ic.get("train", 0)
        m["oos_test_ic"] = oos_ic.get("test", 0)

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
                {"horizon": d.horizon, "ic_mean": d.ic_mean, "ic_std": d.ic_std} for d in decay_list
            ]

    return m


def _compute_star_rating(metrics: dict[str, Any]) -> int:
    """根据指标计算 1-5 星级评分。"""
    stars = 3
    ic_mean = abs(metrics.get("ic_mean", 0))
    ir = metrics.get("ir", 0)
    ls_sharpe = metrics.get("ls_sharpe", 0)

    if ic_mean > STAR_RATING_THRESHOLDS[4]:  # > 0.04 → +1
        stars += 1
    if ir > MIN_BACKTEST_IR:  # > 0.5 → +1
        stars += 1
    if ls_sharpe > 1.0:
        stars += 1
    if ic_mean < STAR_RATING_THRESHOLDS[2] and ir < 0.2:  # < 0.02 弱因子降星
        stars = max(1, stars - 1)

    return min(5, max(1, stars))


def _generate_summary_text(factor_name: str, metrics: dict[str, Any]) -> str:
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
        lines.append(
            f"<p>IC 均值 {ic_mean:.4f}（Spearman &rho;），因子展现出较强的正向预测能力。</p>"
        )
    elif ic_mean < -0.03:
        lines.append(
            f"<p>IC 均值 {ic_mean:.4f}，因子呈现显著的负向预测能力（可用作反向因子）。</p>"
        )
    else:
        lines.append(f"<p>IC 均值 {ic_mean:.4f}，因子具备一定的预测能力。</p>")

    if ir > 0.5:
        lines.append(f"<p>信息比率 IR = {ir:.2f}，因子稳定性良好，IC 方向一致性高。</p>")
    elif ir > 0.2:
        lines.append(f"<p>信息比率 IR = {ir:.2f}，因子稳定性一般。</p>")

    ls_ret = metrics.get("ls_ann_ret") or 0
    if ls_ret > 0.05:
        lines.append(
            f"<p>多空年化收益 {ls_ret * 100:.1f}%，分层效果显著，Top-Bottom 区分度高。</p>"
        )
    elif ls_ret > 0:
        lines.append(f"<p>多空年化收益 {ls_ret * 100:.1f}%，分层效果较弱。</p>")
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
    advanced_results: dict[str, Any] | None = None,
    universe: str = "lft_default",
    benchmark_result: Any = None,
    attribution_result: Any = None,
    walk_forward_result: Any = None,
    event_study_result: Any = None,
    factor_corr: Any = None,
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
    if bt_result is not None:
        try:
            nav = _safe_attr(bt_result, "nav")
            summary_stats = _safe_attr(bt_result, "summary_stats", {})
            if nav is not None and not nav.is_empty() and summary_stats:
                # Build grouped_returns dict from returns data
                # Use summary_stats keys to identify groups and reconstruct from nav
                nav_pd = nav.to_pandas()
                if "group" in nav_pd.columns and "nav" in nav_pd.columns:
                    grouped_rets: dict = {}
                    for g, grp in nav_pd.groupby("group"):
                        if isinstance(g, (int, float)):
                            g_int = int(g)
                            navs = grp.sort_values("trade_date")["nav"].values
                            rets = list(np.diff(navs) / navs[:-1]) if len(navs) > 1 else []
                            if rets:
                                grouped_rets[g_int] = rets
                    if len(grouped_rets) >= 2:
                        qs_b64 = _make_quantile_spread_chart(grouped_rets)
                        if qs_b64:
                            charts["quantile_spread_chart"] = qs_b64
        except Exception:
            logger.warning("生成分位价差图表失败", exc_info=True)

    metrics = _extract_metrics(ic_result, bt_result, to_result, advanced_results)

    warnings: list[str] = []
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
        benchmark_result=benchmark_result,
        attribution_result=attribution_result,
        walk_forward_result=walk_forward_result,
        event_study_result=event_study_result,
        factor_corr=factor_corr,
    )

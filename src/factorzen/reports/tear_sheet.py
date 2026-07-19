"""Factor Tear Sheet 报告引擎（极简单页版）。

生成单页 HTML：标题元信息、核心指标表、分层净值图、IC 累计图、
IC 衰减表、单调性表与警告区。不做打分、星级或研究决策文案。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2
import numpy as np

from factorzen.config.constants import TRADING_DAYS_PER_YEAR
from factorzen.core.logger import get_logger
from factorzen.reports._charts import (
    _make_benchmark_chart,
    _make_drawdown_chart,
    _make_group_bar_chart,
    _make_group_nav_chart,
    _make_ic_chart,
    _make_ic_cumulative_chart,
    _make_returns_chart,
)
from factorzen.reports._formatting import (
    _finite_float,
    _format_metric_number,
    _format_metric_percent,
    _is_finite_metric,
    _safe_attr,
)

logger = get_logger(__name__)

# ── 模板加载（portfolio_report 反向依赖 _ENV，原名保留）────────────────
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=jinja2.select_autoescape(["html"]),
)
_ENV.filters["metric_number"] = _format_metric_number
_ENV.filters["metric_percent"] = _format_metric_percent
_ENV.tests["finite_metric"] = _is_finite_metric


def _fmt_ic(value: Any) -> str:
    return _format_metric_number(value, digits=4, empty="未计算")


def _fmt_ratio2(value: Any) -> str:
    """Sharpe / ICIR / 稳定率等，2 位小数。"""
    return _format_metric_number(value, digits=2, empty="未计算")


def _fmt_pct2(value: Any) -> str:
    """收益/回撤/换手/占比等，百分比 2 位小数。"""
    return _format_metric_percent(value, digits=2, empty="未计算")


def _fmt_tstat(value: Any) -> str:
    return _format_metric_number(value, digits=2, empty="未计算")


def _fmt_pvalue(value: Any) -> str:
    return _format_metric_number(value, digits=4, empty="未计算")


def _portfolio_stats(bt_result: Any) -> dict[str, Any]:
    """从 bt_result.summary_stats['portfolio'] 取主策略绩效。"""
    if bt_result is None:
        return {}
    stats = _safe_attr(bt_result, "summary_stats", {}) or {}
    portfolio = stats.get("portfolio") if isinstance(stats, dict) else None
    return portfolio if isinstance(portfolio, dict) else {}


def _build_core_metrics(
    ic_result: Any,
    bt_result: Any,
    to_result: Any,
    *,
    benchmark_result: Any = None,
    walk_forward_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """组装核心指标表所需的显示值。"""
    ic_mean = _finite_float(_safe_attr(ic_result, "ic_mean"))
    ir = _finite_float(_safe_attr(ic_result, "ir"))
    ic_tstat = _finite_float(_safe_attr(ic_result, "ic_tstat"))
    ic_pvalue = _finite_float(_safe_attr(ic_result, "ic_pvalue"))
    ic_pos = _finite_float(_safe_attr(ic_result, "ic_positive_ratio"))
    n_periods = _safe_attr(ic_result, "n_periods", None)
    if n_periods is None:
        n_periods_disp = "未计算"
        n_periods_num = 0
    else:
        try:
            n_periods_num = int(n_periods)
            n_periods_disp = str(n_periods_num)
        except (TypeError, ValueError):
            n_periods_num = 0
            n_periods_disp = "未计算"

    portfolio = _portfolio_stats(bt_result)
    ann_ret = _finite_float(portfolio.get("ann_ret")) if portfolio else None
    sharpe = _finite_float(portfolio.get("sharpe")) if portfolio else None
    max_dd = _finite_float(portfolio.get("max_dd")) if portfolio else None

    excess_ann = None
    if benchmark_result is not None:
        excess_ann = _finite_float(_safe_attr(benchmark_result, "ann_excess_ret"))

    avg_turnover = _finite_float(_safe_attr(to_result, "avg_turnover"))

    wf = walk_forward_summary if isinstance(walk_forward_summary, dict) else None
    if wf is None:
        wf_status = "未计算"
        wf_oos_sharpe = None
        wf_stability = None
        wf_ok = False
    else:
        status = wf.get("status")
        wf_status = str(status) if status is not None else "未计算"
        wf_ok = status == "ok"
        wf_oos_sharpe = _finite_float(wf.get("oos_sharpe_mean")) if wf_ok else None
        wf_stability = _finite_float(wf.get("stability_ratio")) if wf_ok else None

    return {
        "ic_mean": _fmt_ic(ic_mean),
        "ir": _fmt_ratio2(ir),
        "ic_tstat": _fmt_tstat(ic_tstat),
        "ic_pvalue": _fmt_pvalue(ic_pvalue),
        "ic_positive_ratio": _fmt_pct2(ic_pos),
        "n_periods": n_periods_disp,
        "n_periods_num": n_periods_num,
        "ann_ret": _fmt_pct2(ann_ret),
        "sharpe": _fmt_ratio2(sharpe),
        "max_dd": _fmt_pct2(max_dd),
        "has_excess": benchmark_result is not None,
        "ann_excess_ret": _fmt_pct2(excess_ann),
        "avg_turnover": _fmt_pct2(avg_turnover),
        "avg_turnover_raw": avg_turnover,
        "ic_mean_raw": ic_mean,
        "wf_status": wf_status,
        "wf_ok": wf_ok,
        "wf_oos_sharpe": _fmt_ratio2(wf_oos_sharpe) if wf_ok else None,
        "wf_stability": _fmt_ratio2(wf_stability) if wf_ok else None,
    }


def _build_decay_table(ic_result: Any) -> list[dict[str, str]]:
    """IC 衰减：优先 multi_period（含 IC/IR），否则用 decay（仅 IC）。"""
    multi_period = _safe_attr(ic_result, "multi_period", None) or {}
    decay = _safe_attr(ic_result, "decay", None) or {}

    rows: list[dict[str, str]] = []
    if isinstance(multi_period, dict) and multi_period:
        for h, v in sorted(multi_period.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0):
            if not isinstance(v, dict):
                continue
            rows.append(
                {
                    "horizon": f"{h}d",
                    "ic_mean": _fmt_ic(v.get("ic_mean")),
                    "ir": _fmt_ratio2(v.get("ir")),
                }
            )
        return rows

    if isinstance(decay, dict) and decay:
        for h, ic_val in sorted(decay.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0):
            rows.append(
                {
                    "horizon": f"{h}d",
                    "ic_mean": _fmt_ic(ic_val),
                    "ir": "未计算",
                }
            )
    return rows


def _spearman_from_group_means(group_means: list[float]) -> float | None:
    """分组序号 vs 组均收益的 Spearman 相关。"""
    if not group_means or len(group_means) < 2:
        return None
    y = np.asarray(group_means, dtype=float)
    if not np.isfinite(y).all():
        return None
    if np.nanstd(y) == 0:
        return None
    x = np.arange(len(y), dtype=float)
    try:
        from scipy.stats import spearmanr

        corr, _ = spearmanr(x, y)
        return _finite_float(corr)
    except Exception:
        # 无 scipy 时用 Pearson of ranks 兜底
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        if np.std(rx) == 0 or np.std(ry) == 0:
            return None
        return _finite_float(float(np.corrcoef(rx, ry)[0, 1]))


def _build_mono_table(mono_result: Any) -> dict[str, Any] | None:
    """单调性表：分组均收益 + spearman + 单调结论。"""
    if mono_result is None:
        return None
    group_means = _safe_attr(mono_result, "group_means", None)
    if not group_means:
        return None
    try:
        means = [float(m) for m in group_means]
    except (TypeError, ValueError):
        return None

    rows = [
        {"group": f"G{i + 1}", "mean_ret": _fmt_pct2(m), "mean_ret_raw": m}
        for i, m in enumerate(means)
    ]
    spearman = _spearman_from_group_means(means)
    score = _finite_float(_safe_attr(mono_result, "monotonicity_score"))
    direction = str(_safe_attr(mono_result, "direction", "") or "")

    if score is not None and score >= 0.8:
        if direction == "positive":
            conclusion = "正向单调"
        elif direction == "negative":
            conclusion = "负向单调"
        else:
            conclusion = "单调"
    elif score is not None:
        conclusion = "非单调"
    else:
        conclusion = "未计算"

    return {
        "rows": rows,
        "spearman": _fmt_ratio2(spearman),
        "spearman_raw": spearman,
        "score": _fmt_ratio2(score),
        "direction": direction,
        "conclusion": conclusion,
    }


def _build_group_perf_table(mono_result: Any) -> dict[str, Any] | None:
    """逐组绩效表：每一分组的年化 / Sharpe / 最大回撤 / 胜率。

    数据取自 ``MonotonicityResult.group_daily_returns``（逐日 × 分组等权收益）。
    **口径**：等权、不含交易成本与交易约束，与组合回测的 Sharpe / 回撤不可直接比较——
    它回答的是「因子分组本身有没有区分度」，不是「这组能不能交易出来」。
    """
    if mono_result is None:
        return None
    frame = _safe_attr(mono_result, "group_daily_returns", None)
    if frame is None or getattr(frame, "is_empty", lambda: True)():
        return None
    try:
        gdr = frame.to_pandas()
    except Exception:
        return None
    if not {"trade_date", "group", "mean_ret"}.issubset(set(gdr.columns)):
        return None
    gdr = gdr.dropna(subset=["mean_ret"])
    if gdr.empty:
        return None

    groups = sorted(gdr["group"].unique())
    if len(groups) < 2:
        return None

    rows: list[dict[str, Any]] = []
    for g in groups:
        rets = gdr.loc[gdr["group"] == g, "mean_ret"].to_numpy(dtype=float)
        rets = rets[np.isfinite(rets)]
        if rets.size < 2:
            continue
        ann_ret = float(np.mean(rets) * TRADING_DAYS_PER_YEAR)
        ann_vol = float(np.std(rets) * np.sqrt(TRADING_DAYS_PER_YEAR))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else None
        nav = np.cumprod(1.0 + rets)
        max_dd = float(np.min(nav / np.maximum.accumulate(nav) - 1.0))
        win_rate = float(np.mean(rets > 0))
        rows.append(
            {
                "group": f"G{int(g) + 1}",
                "ann_ret": _fmt_pct2(ann_ret),
                "sharpe": _fmt_ratio2(sharpe),
                "max_dd": _fmt_pct2(max_dd),
                "win_rate": _fmt_pct2(win_rate),
                "n_periods": str(int(rets.size)),
            }
        )

    if len(rows) < 2:
        return None
    return {"rows": rows, "n_groups": len(rows)}


def _build_direction_view(backtest_direction: dict[str, Any] | None) -> dict[str, Any]:
    if not backtest_direction:
        return {"is_reversed": False, "label": "正向信号", "reason": ""}
    direction = str(backtest_direction.get("direction", "normal") or "normal")
    reason = str(backtest_direction.get("reason", "") or "")
    if direction == "reversed":
        return {
            "is_reversed": True,
            "label": "反向信号（做多低因子值）",
            "reason": reason,
        }
    return {"is_reversed": False, "label": "正向信号", "reason": reason}


def _build_warnings(
    *,
    n_periods: int,
    ic_mean: float | None,
    avg_turnover: float | None,
    quality_report: dict[str, Any] | None,
) -> list[str]:
    """沿用旧 tear_sheet 阈值：n_periods / |ic_mean| / avg_turnover + quality warnings。"""
    warnings: list[str] = []
    if n_periods < 30:
        warnings.append(f"样本量较少（{n_periods} 期），IC 估计可能不稳定。")
    if n_periods < 60:
        warnings.append("短样本年化指标仅适合同区间横向比较，不代表长期可外推业绩。")
    if ic_mean is not None and abs(ic_mean) < 0.01:
        warnings.append("IC 均值极低（|IC| < 0.01），因子预测能力有限。")
    if avg_turnover is not None and avg_turnover > 0.8:
        warnings.append("换手率较高（>80%），信号稳定性需关注。")
    if isinstance(quality_report, dict):
        for w in quality_report.get("warnings") or []:
            text = str(w).strip()
            if text:
                warnings.append(text)
    return warnings


def generate_tear_sheet(
    factor_name: str,
    ic_result: Any,
    bt_result: Any,
    to_result: Any,
    *,
    frequency: str = "daily",
    date_range: str = "",
    universe: str = "",
    mono_result: Any = None,
    benchmark_result: Any = None,
    backtest_direction: dict[str, Any] | None = None,
    walk_forward_summary: dict[str, Any] | None = None,
    quality_report: dict[str, Any] | None = None,
) -> str:
    """生成极简单页因子 Tear Sheet HTML。

    所有可选输入缺失时对应区块显示「未计算」或省略，不抛异常。
    """
    metrics = _build_core_metrics(
        ic_result,
        bt_result,
        to_result,
        benchmark_result=benchmark_result,
        walk_forward_summary=walk_forward_summary,
    )
    direction = _build_direction_view(backtest_direction)
    decay_table = _build_decay_table(ic_result)
    mono_table = _build_mono_table(mono_result)
    warnings = _build_warnings(
        n_periods=int(metrics.get("n_periods_num") or 0),
        ic_mean=metrics.get("ic_mean_raw"),
        avg_turnover=metrics.get("avg_turnover_raw"),
        quality_report=quality_report,
    )

    group_perf_table = _build_group_perf_table(mono_result)

    # 单图失败不拖垮整页：逐个 try，失败只落 warning 并留空区块
    chart_specs: list[tuple[str, str, Any]] = [
        ("returns_chart", "组合净值图", lambda: _make_returns_chart(bt_result, factor_name)),
        ("drawdown_chart", "回撤曲线", lambda: _make_drawdown_chart(bt_result)),
        ("benchmark_chart", "基准对比图", lambda: _make_benchmark_chart(benchmark_result)),
        ("ic_chart", "IC 时序图", lambda: _make_ic_chart(ic_result)),
        ("ic_cum_chart", "IC 累计图", lambda: _make_ic_cumulative_chart(ic_result)),
        ("group_nav_chart", "分组净值图", lambda: _make_group_nav_chart(mono_result)),
        ("group_bar_chart", "分组收益柱状图", lambda: _make_group_bar_chart(mono_result)),
    ]
    charts: dict[str, str] = {}
    for key, label, maker in chart_specs:
        try:
            b64 = maker()
        except Exception:
            logger.warning("生成%s失败", label, exc_info=True)
            continue
        if b64:
            charts[key] = b64

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta_parts = [p for p in [date_range, universe, frequency, f"生成时间 {generated_at}"] if p]
    meta_line = " | ".join(meta_parts)

    template = _ENV.get_template("tear_sheet.html")
    return template.render(
        factor_name=factor_name,
        meta_line=meta_line,
        date_range=date_range,
        universe=universe,
        frequency=frequency,
        generated_at=generated_at,
        direction=direction,
        metrics=metrics,
        charts=charts,
        decay_table=decay_table,
        mono_table=mono_table,
        group_perf_table=group_perf_table,
        warnings=warnings,
    )

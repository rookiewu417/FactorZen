"""Factor Tear Sheet 报告引擎。

生成包含目录、综合结论、收益表现、预测能力、结构检验、交易可行性、
稳健性验证、风险归因和附录的 HTML 因子研究报告。
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2
import polars as pl

from factorzen.core.logger import get_logger
from factorzen.reports._charts import (
    _event_study_has_valid_window_series,
    _extract_quantile_grouped_returns,
    _factor_corr_has_valid_off_diagonal,
    _factor_corr_is_multi_factor_input,
    _make_attribution_chart,
    _make_benchmark_chart,
    _make_event_study_chart,
    _make_factor_corr_heatmap,
    _make_ic_chart,
    _make_ic_distribution_chart,
    _make_monthly_return_heatmap,
    _make_quantile_spread_chart,
    _make_returns_chart,
    _make_turnover_chart,
    _make_walk_forward_chart,
)
from factorzen.reports._formatting import (
    _finite_float,
    _format_metric_number,
    _format_metric_percent,
    _is_finite_metric,
    _num,
    _safe_attr,
)
from factorzen.reports._scoring import (
    FactorRating,
    _compute_factor_rating,
    _stars_from_score,
)
from factorzen.reports._strategy import (
    _build_bt_summary_table,
    _build_execution_summary,
    _build_strategy_quality_summary,
    _build_trade_summary,
    _display_cost_model,
    _display_strategy_name,
    _display_strategy_type,
    _infer_strategy_type,
    _resolve_is_long_short,
    _slugify_strategy_name,
    _strategy_constraints,
    _strategy_exposure_label,
    _strategy_params_summary,
)
from factorzen.reports._summaries import (
    _build_attribution_notice,
    _build_attribution_summary,
    _build_benchmark_notice,
    _build_benchmark_summary,
    _build_factor_corr_summary,
    _build_holding_period_summary,
    _build_monthly_return_summary,
    _build_neutralized_summary,
    _build_predictive_summary,
    _build_quality_summary,
    _build_regime_summary,
    _display_regime_label,
    _display_status,
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
        "data_quality": quality_summary["status"],
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

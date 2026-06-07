"""报告文本摘要:把评估指标转为面向阅读者的结论/提示与状态显示。"""

import re
from typing import Any

import numpy as np
import polars as pl

from factorzen.reports._formatting import (
    _finite_float,
    _format_metric_number,
    _format_metric_percent,
    _num,
    _safe_attr,
    _same_direction,
)


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
        "disabled": "已关闭",
    }
    return labels.get(status.lower(), status or "未知")

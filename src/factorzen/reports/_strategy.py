"""策略口径辅助:命名/类型/暴露标签/约束/参数与交易成交摘要。"""

import re
from typing import Any

import polars as pl

from factorzen.reports._formatting import (
    _finite_float,
    _format_metric_number,
    _format_metric_percent,
    _num,
    _safe_attr,
)


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

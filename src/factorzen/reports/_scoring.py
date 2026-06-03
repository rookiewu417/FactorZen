"""因子评级评分卡:100 分维度打分 + 1-5 星(含上限规则)。"""

from dataclasses import dataclass
from typing import Any

from factorzen.config.constants import MIN_BACKTEST_IR
from factorzen.reports._formatting import _clamp, _finite_float, _num, _same_direction


@dataclass(frozen=True)
class FactorRating:
    """Research scorecard result for the summary panel."""

    stars: int
    score: float
    label: str
    components: dict[str, float]
    caps: list[str]
    positives: list[str]
    warnings: list[str]


def _score_from_drawdown(max_dd: float | None) -> float:
    if max_dd is None:
        return 2.0
    if max_dd >= -0.10:
        return 4.0
    if max_dd >= -0.20:
        return 2.5
    if max_dd >= -0.30:
        return 1.0
    return 0.0


def _score_from_turnover(avg_turnover: float | None) -> float:
    if avg_turnover is None:
        return 1.5
    if avg_turnover <= 0:
        return 1.5
    if avg_turnover <= 0.20:
        return 3.0
    if avg_turnover <= 0.50:
        return 2.0
    if avg_turnover <= 0.80:
        return 1.0
    return 0.0


def _decay_values(metrics: dict[str, Any]) -> list[float]:
    decay_table = metrics.get("decay_table") or []
    if decay_table:
        rows = sorted(decay_table, key=lambda row: _num(row.get("horizon"), 0))
        return [
            numeric for row in rows if (numeric := _finite_float(row.get("ic_mean"))) is not None
        ]

    decay = metrics.get("decay") or {}
    if decay:
        return [numeric for h in sorted(decay) if (numeric := _finite_float(decay[h])) is not None]

    return []


def _score_alpha_strength(metrics: dict[str, Any]) -> float:
    ic_mean = _num(metrics.get("ic_mean"))
    abs_ic = abs(ic_mean)
    ir = abs(_num(metrics.get("ir")))
    tstat = abs(_num(metrics.get("ic_tstat")))
    pvalue = _num(metrics.get("ic_pvalue"), 1.0)
    pos_ratio = _num(metrics.get("ic_positive_ratio"), 0.5)
    direction_ratio = max(pos_ratio, 1.0 - pos_ratio)

    ic_score = _clamp(abs_ic / 0.04) * 12.0
    ir_score = _clamp(ir / MIN_BACKTEST_IR) * 8.0
    if pvalue <= 0.01 or tstat >= 2.58:
        sig_score = 6.0
    elif pvalue <= 0.05 or tstat >= 1.96:
        sig_score = 4.0
    elif pvalue <= 0.10 or tstat >= 1.65:
        sig_score = 2.0
    else:
        sig_score = 0.0
    direction_score = _clamp((direction_ratio - 0.50) / 0.12) * 4.0

    return ic_score + ir_score + sig_score + direction_score


def _score_stability(metrics: dict[str, Any]) -> float:
    ic_mean = _num(metrics.get("ic_mean"))
    abs_ic = abs(ic_mean)
    train_ic = _finite_float(metrics.get("oos_train_ic"))
    test_ic = _finite_float(metrics.get("oos_test_ic"))
    if train_ic is not None and test_ic is not None:
        denominator = max(abs(train_ic), abs_ic, 1e-9)
        retention = _clamp(abs(test_ic) / denominator)
        oos_score = 10.0 * retention if _same_direction(ic_mean, test_ic) else 0.0
    else:
        oos_score = 3.0

    wf_sharpe = _finite_float(metrics.get("walk_forward_oos_sharpe_mean"))
    wf_stability = _finite_float(metrics.get("walk_forward_stability_ratio"))
    if wf_sharpe is not None or wf_stability is not None:
        wf_score = _clamp((wf_sharpe or 0.0) / 1.0) * 3.0
        wf_score += _clamp(wf_stability or 0.0) * 3.0
    else:
        wf_score = 2.0

    decay_vals = _decay_values(metrics)
    if len(decay_vals) >= 2 and abs(decay_vals[0]) > 1e-9:
        same_sign_count = sum(1 for val in decay_vals if _same_direction(decay_vals[0], val))
        sign_score = same_sign_count / len(decay_vals)
        retention = _clamp(abs(decay_vals[-1]) / abs(decay_vals[0]))
        decay_score = 5.0 * (0.6 * sign_score + 0.4 * retention)
    else:
        decay_score = 2.5

    multi_period = [
        numeric
        for row in metrics.get("multi_period_table") or []
        if (numeric := _finite_float(row.get("ic_mean"))) is not None
    ]
    if multi_period:
        same_sign_count = sum(1 for ic_value in multi_period if _same_direction(ic_mean, ic_value))
        multi_score = 4.0 * same_sign_count / len(multi_period)
    else:
        multi_score = 2.0

    return oos_score + wf_score + decay_score + multi_score


def _score_tradeability(metrics: dict[str, Any]) -> float:
    sharpe = _finite_float(metrics.get("ls_sharpe"))
    ann_ret = _finite_float(metrics.get("ls_ann_ret"))
    max_dd = _finite_float(metrics.get("ls_max_dd"))
    avg_turnover = _finite_float(metrics.get("avg_turnover"))

    sharpe_score = _clamp((sharpe or 0.0) / 1.5) * 8.0
    ret_score = _clamp((ann_ret or 0.0) / 0.10) * 5.0
    dd_score = _score_from_drawdown(max_dd)
    turnover_score = _score_from_turnover(avg_turnover)

    return sharpe_score + ret_score + dd_score + turnover_score


def _score_robustness(metrics: dict[str, Any]) -> float:
    ic_mean = _num(metrics.get("ic_mean"))
    abs_ic = abs(ic_mean)
    neutral_ic = _finite_float(metrics.get("neutralized_ic_mean"))
    if neutral_ic is not None and abs_ic > 1e-9:
        retention = _clamp(abs(neutral_ic) / abs_ic)
        neutral_score = 8.0 * retention if _same_direction(ic_mean, neutral_ic) else 0.0
    else:
        neutral_score = 4.0

    pearson_ic = _finite_float(metrics.get("pearson_ic_mean"))
    if pearson_ic is not None and abs_ic > 1e-9:
        pearson_score = (
            3.0 * _clamp(abs(pearson_ic) / abs_ic) if _same_direction(ic_mean, pearson_ic) else 0.0
        )
    else:
        pearson_score = 1.5

    subgroup_score = 0.0
    sector_ic = metrics.get("sector_ic")
    if sector_ic is not None and hasattr(sector_ic, "is_empty") and not sector_ic.is_empty():
        subgroup_score += 2.0
    else:
        subgroup_score += 0.5

    size_buckets = metrics.get("size_buckets") or {}
    if size_buckets:
        values = [_num(v) for v in size_buckets.values()]
        same_sign_count = sum(1 for val in values if _same_direction(ic_mean, val))
        subgroup_score += 2.0 * same_sign_count / max(len(values), 1)
    else:
        subgroup_score += 0.5

    return neutral_score + pearson_score + subgroup_score


def _score_structure(metrics: dict[str, Any]) -> float:
    mono = _finite_float(metrics.get("monotonicity_score"))
    if mono is not None:
        mono_score = _clamp(mono) * 6.0
    else:
        mono_score = 3.0

    autocorr = _finite_float(metrics.get("rank_autocorr"))
    if autocorr is not None:
        autocorr_score = _clamp(autocorr) * 4.0
    else:
        autocorr_score = 2.0

    return mono_score + autocorr_score


def _stars_from_score(score: float) -> int:
    if score >= 80:
        return 5
    if score >= 65:
        return 4
    if score >= 50:
        return 3
    if score >= 35:
        return 2
    return 1


def _label_from_stars(stars: int) -> str:
    if stars >= 5:
        return "production_watch"
    if stars == 4:
        return "candidate"
    if stars == 3:
        return "research"
    if stars == 2:
        return "weak"
    return "invalid"


def _compute_factor_rating(metrics: dict[str, Any]) -> FactorRating:
    """Compute a 100-point research scorecard and capped 1-5 star rating."""
    components = {
        "Alpha 强度": _score_alpha_strength(metrics),
        "稳定性": _score_stability(metrics),
        "可交易性": _score_tradeability(metrics),
        "鲁棒性": _score_robustness(metrics),
        "结构质量": _score_structure(metrics),
    }
    score = round(sum(components.values()), 1)
    base_stars = _stars_from_score(score)

    caps: list[str] = []
    cap_stars = 5
    n_periods = int(_num(metrics.get("n_periods")))
    ic_mean = _num(metrics.get("ic_mean"))
    abs_ic = abs(ic_mean)
    avg_turnover = _finite_float(metrics.get("avg_turnover"))

    if n_periods < 60:
        cap_stars = min(cap_stars, 2)
        caps.append("样本期数少于 60，最高 2 星")

    oos_test_ic = _finite_float(metrics.get("oos_test_ic"))
    if oos_test_ic is None:
        cap_stars = min(cap_stars, 3)
        caps.append("缺少样本外验证期 IC，最高 3 星")
    else:
        if abs(oos_test_ic) > 0.005 and not _same_direction(ic_mean, oos_test_ic):
            cap_stars = min(cap_stars, 2)
            caps.append("样本外验证期 IC 与全样本方向相反，最高 2 星")

    neutral_ic = _finite_float(metrics.get("neutralized_ic_mean"))
    if neutral_ic is not None and abs_ic > 1e-9:
        neutral_retention = abs(neutral_ic) / abs_ic
        if neutral_retention < 0.50:
            cap_stars = min(cap_stars, 3)
            caps.append("中性化后 IC 保留不足 50%，最高 3 星")

    mono = _finite_float(metrics.get("monotonicity_score"))
    if mono is not None and mono < 0.60:
        cap_stars = min(cap_stars, 3)
        caps.append("分组单调性不足，最高 3 星")

    if avg_turnover is not None and avg_turnover > 0.80:
        cap_stars = min(cap_stars, 3)
        caps.append("平均换手率高于 80%，最高 3 星")

    if (
        _finite_float(metrics.get("ls_sharpe")) is None
        and _finite_float(metrics.get("ls_ann_ret")) is None
    ):
        cap_stars = min(cap_stars, 3)
        caps.append("缺少多空回测绩效，最高 3 星")

    stars = min(base_stars, cap_stars)
    positives = [
        name
        for name, value in components.items()
        if value
        >= {"Alpha 强度": 21.0, "稳定性": 17.5, "可交易性": 14.0, "鲁棒性": 10.5, "结构质量": 7.0}[
            name
        ]
    ]
    warnings = caps.copy()
    for name, value in components.items():
        threshold = {
            "Alpha 强度": 12.0,
            "稳定性": 10.0,
            "可交易性": 8.0,
            "鲁棒性": 6.0,
            "结构质量": 4.0,
        }[name]
        if value < threshold:
            warnings.append(f"{name}得分偏低")

    return FactorRating(
        stars=stars,
        score=score,
        label=_label_from_stars(stars),
        components={name: round(value, 1) for name, value in components.items()},
        caps=caps,
        positives=positives,
        warnings=warnings,
    )


def _compute_star_rating(metrics: dict[str, Any]) -> int:
    """根据评分卡计算 1-5 星级评分。"""
    return _compute_factor_rating(metrics).stars

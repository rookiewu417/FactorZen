"""Tear Sheet 的多空绩效指标(ls_*)必须仅对真正的多空策略填充（T1）。

根因：backtest._summary_stats 恒返回 {"portfolio": stats, "long_short": stats}（同一份
拷贝），tear_sheet 用 `if "long_short" in stats`（恒真）把长仅组合的 Sharpe/收益/回撤写进
ls_*，导致：① 决策面板"多空 Sharpe"显示含 β 的长仅 Sharpe；② _score_tradeability 按假
多空绩效虚高评分；③ "缺少多空回测绩效，最高 3 星"评级上限永不触发。
_resolve_is_long_short 的兜底 `return "long_short" in stats` 同样恒真。
"""
from __future__ import annotations

from types import SimpleNamespace

_STATS = {
    "ann_ret": 0.25, "ann_vol": 0.18, "sharpe": 1.4, "max_dd": -0.2,
    "avg_turnover": 0.3, "total_cost": 0.01, "ann_turnover": 75.0,
}


def _bt(strategy_name, strategy_type=None, long_only=None):
    # _summary_stats 恒把 portfolio 复制成 long_short
    stats = {"portfolio": dict(_STATS), "long_short": dict(_STATS)}
    config = {"strategy_type": strategy_type or strategy_name}
    if long_only is not None:
        config["strategy_params"] = {"long_only": long_only}
    return SimpleNamespace(
        summary_stats=stats, strategy_name=strategy_name, config=config, nav=None
    )


def test_long_only_primary_does_not_fill_ls_metrics():
    from factorzen.reports.tear_sheet import _extract_metrics

    bt = _bt("topn_50", strategy_type="topn_50", long_only=True)
    m = _extract_metrics(None, bt, None, None)

    assert m.get("primary_is_long_short") is False
    assert m.get("ls_sharpe") is None, "长仅策略不应把长仅 Sharpe 冒充多空 Sharpe 填入 ls_*"
    assert m.get("ls_ann_ret") is None
    assert m.get("ls_max_dd") is None


def test_genuine_long_short_fills_ls_metrics():
    from factorzen.reports.tear_sheet import _extract_metrics

    bt = _bt("quantile_ls_5", strategy_type="quantile_ls_5")
    m = _extract_metrics(None, bt, None, None)

    assert m.get("primary_is_long_short") is True
    assert m.get("ls_sharpe") == _STATS["sharpe"]
    assert m.get("ls_ann_ret") == _STATS["ann_ret"]


def test_resolve_is_long_short_custom_name_defaults_long_only():
    from factorzen.reports._strategy import _resolve_is_long_short

    stats = {"portfolio": dict(_STATS), "long_short": dict(_STATS)}
    # 自定义命名、不匹配任何多空模式、未声明 long_only → 应判长仅(False)，
    # 而非因 stats 恒含 long_short 键被误判多空
    bt = SimpleNamespace(strategy_name="my_alpha", config={"strategy_type": "my_alpha"})
    assert _resolve_is_long_short(bt, stats) is False


def test_long_only_capped_at_3_stars_for_missing_ls():
    """长仅策略 ls_* 为 None → 评级上限 3 星（'缺少多空回测绩效'）应生效。"""
    from factorzen.reports._scoring import _compute_factor_rating

    metrics = {
        "ic_mean": 0.08, "ir": 0.9, "ic_positive_ratio": 0.7, "ic_tstat": 5.0,
        "n_periods": 200, "avg_turnover": 0.2,
        # ls_* 缺失（长仅）
    }
    rating = _compute_factor_rating(metrics)
    assert rating.stars <= 3, "长仅（无真实多空绩效）评级应被上限封到 ≤3 星"

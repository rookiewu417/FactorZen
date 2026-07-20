"""
test_backtest_engine.py：test_backtest.py：daily/evaluation/backtest.py 的单元测试。
test_backtest_costs.py：S2 防回归：验证 CostModel 和成本扣除逻辑。
"""

from __future__ import annotations

import datetime
import json
import random
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from factorzen.config.constants import (
    BORROW_RATE_ANNUAL,
    COMMISSION_RATE,
    SLIPPAGE_RATE,
    STAMP_TAX_RATE,
    TRADING_DAYS_PER_YEAR,
)
from factorzen.daily.evaluation.backtest import (
    BacktestConfig,
    CostModel,
    PrecomputedWeightsStrategy,
    run_strategy_backtest,
    run_stratified_backtest,
)
from factorzen.daily.evaluation.cost_models import SquareRootImpactCostModel
from factorzen.daily.evaluation.ic_analysis import ICAnalysisResult
from factorzen.intraday.evaluation.backtest import aggregate_intraday_factor, run_intraday_backtest
from factorzen.pipelines._report_direction import (
    _apply_backtest_direction,
    _decide_backtest_direction,
)


# ==== 来自 test_backtest_engine.py ====
# ==== 来自 test_backtest.py ====
def _make_factor_price(n_dates: int = 60, n_stocks: int = 50, seed: int = 42):
    """生成合成因子+价格 DataFrame。"""
    rng = np.random.default_rng(seed)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    stocks = [f"{i:06d}.SH" for i in range(n_stocks)]

    rows_factor = []
    rows_price = []
    last_close = {s: 10.0 + i for i, s in enumerate(stocks)}
    for idx, d in enumerate(dates):
        for s in stocks:
            if idx < n_dates - 1:
                rows_factor.append(
                    {
                        "trade_date": d,
                        "ts_code": s,
                        "factor_clean": float(rng.standard_normal()),
                    }
                )
            open_price = last_close[s] * (1.0 + float(rng.normal(0, 0.002)))
            close_price = open_price * (1.0 + float(rng.normal(0, 0.01)))
            rows_price.append(
                {
                    "trade_date": d,
                    "ts_code": s,
                    "open": open_price,
                    "close": close_price,
                    "pre_close": last_close[s],
                    "pct_chg": (close_price / last_close[s] - 1.0) * 100,
                    "vol": 1000.0,
                    "amount": 1_000_000.0,
                }
            )
            last_close[s] = close_price

    return pl.DataFrame(rows_factor), pl.DataFrame(rows_price)


def test_backtest_core_engine_suite():
    """test_factor_name_passed_through；test_summary_stats_has_portfolio_and_long_short；test_nav_starts_near_one；test_summary_string_is_non_empty"""
    # -- 原 test_factor_name_passed_through --
    def _section_0_test_factor_name_passed_through():
        factor_df, price_df = _make_factor_price()
        result = run_stratified_backtest(factor_df, price_df, factor_name="momentum")
        assert result.factor_name == "momentum"

    _section_0_test_factor_name_passed_through()

    # -- 原 test_summary_stats_has_portfolio_and_long_short --
    def _section_1_test_summary_stats_has_portfolio_and_long_short():
        factor_df, price_df = _make_factor_price()
        n_groups = 5
        result = run_stratified_backtest(factor_df, price_df, n_groups=n_groups)
        assert "portfolio" in result.summary_stats
        assert "long_short" in result.summary_stats
        assert result.n_groups == n_groups

    _section_1_test_summary_stats_has_portfolio_and_long_short()

    # -- 原 test_nav_starts_near_one --
    def _section_2_test_nav_starts_near_one():
        factor_df, price_df = _make_factor_price()
        result = run_stratified_backtest(factor_df, price_df, n_groups=5)
        first_nav = result.nav.sort("trade_date")["nav"][0]
        assert first_nav == 1.0

    _section_2_test_nav_starts_near_one()

    # -- 原 test_summary_string_is_non_empty --
    def _section_3_test_summary_string_is_non_empty():
        factor_df, price_df = _make_factor_price()
        result = run_stratified_backtest(factor_df, price_df, n_groups=5)
        text = result.summary()
        assert "Portfolio" in text
        assert len(text) > 10

    _section_3_test_summary_string_is_non_empty()


# ==== 来自 test_backtest_direction.py ====


def _ic(**kwargs) -> ICAnalysisResult:
    base = dict(
        factor_name="f",
        ic_mean=0.0,
        ic_std=0.1,
        ir=0.0,
        ic_positive_ratio=0.5,
        n_periods=100,
        ic_series=pl.DataFrame(),
        ic_tstat=0.0,
        ic_pvalue=1.0,
        oos_ic={},
    )
    base.update(kwargs)
    return ICAnalysisResult(**base)


def test_backtest_direction_suite():
    """test_significant_negative_ic_reverses；``ic_pvalue=0.0`` 必须保留，不能被 ``x or 1.0`` 吃成不显著。；test_weak_negative_ic_keeps_normal；IS/OOS 两段 IC 均为负时，即使全样本 p 略高也对齐交易方向。；test_positive_ic_keeps_normal"""
    # -- 原 test_significant_negative_ic_reverses --
    def _section_0_test_significant_negative_ic_reverses():
        d = _decide_backtest_direction(
            _ic(ic_mean=-0.03, ic_tstat=-3.0, ic_pvalue=0.003, ir=-0.3)
        )
        assert d["direction"] == "reversed"
        assert d["should_reverse"] is True
        assert "p 值" in d["reason"] or "负" in d["reason"]

    _section_0_test_significant_negative_ic_reverses()

    # -- 原 test_pvalue_zero_is_not_treated_as_missing --
    def _section_1_test_pvalue_zero_is_not_treated_as_missing():
        d = _decide_backtest_direction(
            _ic(ic_mean=-0.03, ic_tstat=-13.0, ic_pvalue=0.0, ir=-0.3)
        )
        assert d["direction"] == "reversed"
        assert d["ic_pvalue"] == 0.0
        assert "p 值" in d["reason"]

    _section_1_test_pvalue_zero_is_not_treated_as_missing()

    # -- 原 test_weak_negative_ic_keeps_normal --
    def _section_2_test_weak_negative_ic_keeps_normal():
        d = _decide_backtest_direction(
            _ic(
                ic_mean=-0.005,
                ic_tstat=-0.4,
                ic_pvalue=0.7,
                oos_ic={"train": -0.004, "test": 0.002},
            )
        )
        assert d["direction"] == "normal"
        assert d["should_reverse"] is False

    _section_2_test_weak_negative_ic_keeps_normal()

    # -- 原 test_oos_both_negative_reverses_even_if_p_weak --
    def _section_3_test_oos_both_negative_reverses_even_if_p_weak():
        d = _decide_backtest_direction(
            _ic(
                ic_mean=-0.02,
                ic_tstat=-1.5,
                ic_pvalue=0.14,  # > 0.10
                oos_ic={"train": -0.025, "test": -0.015},
            )
        )
        assert d["direction"] == "reversed"
        assert d["should_reverse"] is True

    _section_3_test_oos_both_negative_reverses_even_if_p_weak()

    # -- 原 test_positive_ic_keeps_normal --
    def _section_4_test_positive_ic_keeps_normal():
        d = _decide_backtest_direction(
            _ic(ic_mean=0.04, ic_tstat=4.0, ic_pvalue=0.0001, ir=0.5)
        )
        assert d["direction"] == "normal"
        assert d["should_reverse"] is False

    _section_4_test_positive_ic_keeps_normal()


def test_apply_reversed_flips_factor_clean_only():
    df = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3)],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "factor_clean": [1.5, -0.5],
            "factor_value": [9.0, 8.0],
        }
    )
    out = _apply_backtest_direction(df, {"direction": "reversed", "should_reverse": True})
    assert out["factor_clean"].to_list() == [-1.5, 0.5]
    # 原始语义列不变
    assert out["factor_value"].to_list() == [9.0, 8.0]


def test_apply_normal_is_noop():
    df = pl.DataFrame({"factor_clean": [1.0, 2.0]})
    out = _apply_backtest_direction(df, {"direction": "normal", "should_reverse": False})
    assert out["factor_clean"].to_list() == [1.0, 2.0]
    assert _apply_backtest_direction(df, None)["factor_clean"].to_list() == [1.0, 2.0]


def test_daily_single_wires_direction_helpers():
    """factor run 主路径必须 import 并调用与 report 相同的方向工具。"""
    import inspect

    from factorzen.pipelines import daily_single as mod

    src = inspect.getsource(mod._run)
    assert "_decide_backtest_direction" in src
    assert "_apply_backtest_direction" in src
    assert "backtest_direction=backtest_direction" in src
    # IC 用 clean_df；回测/换手/walk-forward 用 backtest_df
    assert "compute_rank_ic(clean_df" in src or "compute_rank_ic(\n        clean_df" in src
    assert "_run_backtest_strategies(\n            effective_config,\n            backtest_df," in src or (
        "backtest_df" in src and "_run_backtest_strategies" in src
    )
    assert "compute_turnover(backtest_df" in src


def test_meta_path_records_backtest_direction(tmp_path, monkeypatch):
    """daily_single 写入的 meta 形状应可被 report --reuse 读取。"""
    from factorzen.pipelines import _report_direction as direction_mod
    from factorzen.pipelines import _report_persistence as persist_mod

    monkeypatch.setattr(
        persist_mod, "daily_result_output_dir", lambda _name: tmp_path / "results"
    )
    # _meta_path 在 persist 模块解析 daily_result_output_dir
    path = persist_mod._meta_path("hf_resiliency", "20200101", "20201231")
    decision = direction_mod._decide_backtest_direction(
        _ic(ic_mean=-0.03, ic_tstat=-5.0, ic_pvalue=0.001)
    )
    path.write_text(
        json.dumps({"backtest_direction": decision}, ensure_ascii=False),
        encoding="utf-8",
    )
    loaded = direction_mod._load_backtest_direction("hf_resiliency", "20200101", "20201231")
    # _load_backtest_direction 也走 _meta_path → 需同样 patch
    monkeypatch.setattr(
        direction_mod, "_meta_path", lambda *a, **k: path
    )
    loaded = direction_mod._load_backtest_direction("hf_resiliency", "20200101", "20201231")
    assert loaded is not None
    assert loaded["direction"] == "reversed"
    assert loaded["should_reverse"] is True

# ==== 来自 test_backtest_missing_weights_hold.py ====
_DATES = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
_CODES = ["000001.SZ", "000002.SZ", "000003.SZ"]


def _factors():
    # 因子覆盖前 3 日（→ 慢路径 signal_date 01-01/01-02/01-03 都在 factor_by_date）
    rows = []
    for d in _DATES[:3]:
        for i, c in enumerate(_CODES):
            rows.append({"trade_date": d, "ts_code": c, "factor_clean": float(3 - i)})
    return pl.DataFrame(rows)


def _prices():
    rows = []
    for d in _DATES:
        for idx, c in enumerate(_CODES):
            # 让持仓股 000001 每天有正的日内收益 → 持有 vs 清仓 NAV 明显不同
            rows.append({
                "trade_date": d, "ts_code": c,
                "open": 10.0 + idx, "close": 10.0 + idx + 0.5,
                "pre_close": 10.0 + idx, "pct_chg": 5.0, "vol": 1000.0, "amount": 1_000_000.0,
            })
    return pl.DataFrame(rows)


def _weights_first_signal_only():
    # 只对 01-01 这个 signal 日给权重（→ 01-02 执行建仓 000001）；之后 signal 日无权重
    return {_DATES[0]: pl.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [1.0]})}


def test_missing_weights_hold_suite():
    """test_fast_and_slow_path_hold_on_missing_weights；#5：weights_by_date 存在但空表 = 显式空仓 → 平到空；不得 carry 前仓。；对照：sig_date 完全不在 weights_by_date → 仍 carry（#5 不得误伤缺失语义）。"""
    # -- 原 test_fast_and_slow_path_hold_on_missing_weights --
    def _section_0_test_fast_and_slow_path_hold_on_missing_weights():
        cfg = BacktestConfig(initial_capital=1_000_000, max_participation_rate=1.0)
        cost = CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0)
        factors, prices = _factors(), _prices()

        fast = run_strategy_backtest(
            PrecomputedWeightsStrategy(_weights_first_signal_only()),
            factors, prices, config=cfg, cost_model=cost,
            collect_positions=False, collect_trades=False, include_context_positions=False,
        )
        slow = run_strategy_backtest(
            PrecomputedWeightsStrategy(_weights_first_signal_only()),
            factors, prices, config=cfg, cost_model=cost,
            collect_trades=True,  # 强制走慢路径
        )

        assert fast.nav.equals(slow.nav), (
            "缺权重的 signal 日两路径 NAV 应一致（都持有）；修复前慢路径清仓、快路径持有 → 分叉"
        )
        assert fast.returns.equals(slow.returns)

        # 慢路径在缺权重日不应产生卖出换手（持有，非清仓）
        trades = slow.trades
        if trades.height > 0:
            # 01-03 执行日（signal=01-02 无权重）不应有任何成交
            exec_0103 = trades.filter(pl.col("trade_date") == date(2024, 1, 3))
            assert exec_0103.height == 0, "缺权重的 signal 日不应清仓换手"

    _section_0_test_fast_and_slow_path_hold_on_missing_weights()

    # -- 原 test_explicit_empty_weights_flat_not_carry_fast_and_slow --
    def _section_1_test_explicit_empty_weights_flat_not_carry_fast_and_slow():
        cfg = BacktestConfig(
            initial_capital=1_000_000,
            max_participation_rate=1.0,
            fallback_adv=1e15,
        )
        cost = CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0)
        factors, prices = _factors_single(), _prices_trending_up()
        # d0 建仓；d1 显式空权重（flat）；d2 不在表中
        weights_by_date = {
            _DATES[0]: pl.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [1.0]}),
            _DATES[1]: _empty_weight_frame(),
        }
        strategy = PrecomputedWeightsStrategy(weights_by_date)

        fast = run_strategy_backtest(
            strategy,
            factors,
            prices,
            config=cfg,
            cost_model=cost,
            collect_positions=False,
            collect_trades=False,
            include_context_positions=False,
        )
        slow = run_strategy_backtest(
            strategy,
            factors,
            prices,
            config=cfg,
            cost_model=cost,
            collect_trades=True,
        )

        # 01-02 建仓后 NAV=1.05；01-03 执行显式空仓后应 flat，后续不再随标的涨
        for result, path in ((fast, "快路径"), (slow, "慢路径")):
            nav_by_date = {
                row["trade_date"]: row["nav"]
                for row in result.nav.select(["trade_date", "nav"]).iter_rows(named=True)
            }
            assert nav_by_date[date(2024, 1, 2)] == pytest.approx(1.05), path
            # 显式空仓后 NAV 应冻结在建仓后水平（≈1.05），而非 carry 到 1.2 / 1.3
            assert nav_by_date[date(2024, 1, 3)] == pytest.approx(1.05), (
                f"{path} 显式空仓后应 flat，旧快路径会 carry 到 ~1.2"
            )
            assert nav_by_date[date(2024, 1, 4)] == pytest.approx(1.05), (
                f"{path} 显式空仓后 NAV 不应再随标的波动"
            )
            day3 = result.nav.filter(pl.col("trade_date") == date(2024, 1, 3)).row(0, named=True)
            assert day3["cash_weight"] == pytest.approx(1.0), f"{path} 应全现金"

        # 慢路径 positions：01-03 起无持仓
        pos_after = slow.positions.filter(pl.col("trade_date") >= date(2024, 1, 3))
        assert pos_after.height == 0, "显式空仓后不应残留持仓"

        # 双路径 NAV 一致
        assert fast.nav["nav"].to_list() == pytest.approx(slow.nav["nav"].to_list())

    _section_1_test_explicit_empty_weights_flat_not_carry_fast_and_slow()

    # -- 原 test_missing_signal_date_still_carries_not_flat --
    def _section_2_test_missing_signal_date_still_carries_not_flat():
        cfg = BacktestConfig(
            initial_capital=1_000_000,
            max_participation_rate=1.0,
            fallback_adv=1e15,
        )
        cost = CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0)
        factors, prices = _factors_single(), _prices_trending_up()
        # 仅 d0 有权重；d1 缺失（非显式空）
        weights_by_date = {
            _DATES[0]: pl.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [1.0]}),
        }
        strategy = PrecomputedWeightsStrategy(weights_by_date)

        fast = run_strategy_backtest(
            strategy,
            factors,
            prices,
            config=cfg,
            cost_model=cost,
            collect_positions=False,
            collect_trades=False,
            include_context_positions=False,
        )
        slow = run_strategy_backtest(
            strategy,
            factors,
            prices,
            config=cfg,
            cost_model=cost,
            collect_trades=True,
        )

        for result, path in ((fast, "快路径"), (slow, "慢路径")):
            nav_by_date = {
                row["trade_date"]: row["nav"]
                for row in result.nav.select(["trade_date", "nav"]).iter_rows(named=True)
            }
            # 缺权重日 carry：01-03 NAV~1.2，01-04~1.3
            assert nav_by_date[date(2024, 1, 3)] == pytest.approx(1.2), path
            assert nav_by_date[date(2024, 1, 4)] == pytest.approx(1.3), path

        assert fast.nav["nav"].to_list() == pytest.approx(slow.nav["nav"].to_list())
        # 01-03 执行（signal=01-02 缺失）无卖出
        exec_0103 = slow.trades.filter(pl.col("trade_date") == date(2024, 1, 3))
        assert exec_0103.height == 0

    _section_2_test_missing_signal_date_still_carries_not_flat()


def _prices_trending_up() -> pl.DataFrame:
    """单标的上涨行情：持有 vs 显式空仓在 NAV 上可分。"""
    # amount 足够大，避免 capacity 卡死满仓平仓
    rows = [
        {
            "trade_date": date(2024, 1, 1),
            "ts_code": "000001.SZ",
            "open": 10.0,
            "close": 10.0,
            "pre_close": 10.0,
            "pct_chg": 0.0,
            "vol": 1e9,
            "amount": 1e15,
        },
        {
            "trade_date": date(2024, 1, 2),
            "ts_code": "000001.SZ",
            "open": 10.0,
            "close": 10.5,
            "pre_close": 10.0,
            "pct_chg": 5.0,
            "vol": 1e9,
            "amount": 1e15,
        },
        {
            "trade_date": date(2024, 1, 3),
            "ts_code": "000001.SZ",
            "open": 10.5,
            "close": 12.0,
            "pre_close": 10.5,
            "pct_chg": 14.0,
            "vol": 1e9,
            "amount": 1e15,
        },
        {
            "trade_date": date(2024, 1, 4),
            "ts_code": "000001.SZ",
            "open": 12.0,
            "close": 13.0,
            "pre_close": 12.0,
            "pct_chg": 8.0,
            "vol": 1e9,
            "amount": 1e15,
        },
    ]
    return pl.DataFrame(rows)


def _factors_single() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"trade_date": d, "ts_code": "000001.SZ", "factor_clean": 1.0}
            for d in _DATES[:3]
        ]
    )


def _empty_weight_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={"ts_code": pl.Utf8, "target_weight": pl.Float64})


# ==== 来自 test_intraday_backtest.py ====
def _make_minute_factor(
    n_stocks: int = 5, n_days: int = 10, bars_per_day: int = 20
) -> pl.DataFrame:
    random.seed(42)
    rows = []
    for day in range(n_days):
        base_date = datetime.date(2026, 1, 2) + datetime.timedelta(days=day)
        base_time = datetime.datetime(2026, 1, 2 + day, 9, 30)
        trade_date = base_date.strftime("%Y%m%d")
        for stock_i in range(1, n_stocks + 1):
            ts = f"00000{stock_i}.SZ"
            for b in range(bars_per_day):
                rows.append(
                    {
                        "trade_date": trade_date,
                        "trade_time": base_time + datetime.timedelta(minutes=b),
                        "ts_code": ts,
                        "factor_value": random.gauss(0, 1),
                    }
                )
    return pl.DataFrame(rows).with_columns(
        pl.col("trade_time").cast(pl.Datetime),
        pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d"),
    )


def _make_daily_price(n_stocks: int = 5, n_days: int = 10) -> pl.DataFrame:
    random.seed(0)
    rows = []
    for day in range(n_days):
        trade_date = (datetime.date(2026, 1, 2) + datetime.timedelta(days=day)).strftime("%Y%m%d")
        for stock_i in range(1, n_stocks + 1):
            open_price = 10.0 + stock_i
            close_price = open_price * (1.0 + random.gauss(0, 0.02))
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": f"00000{stock_i}.SZ",
                    "open": open_price,
                    "close": close_price,
                    "pre_close": open_price,
                    "pct_chg": (close_price / open_price - 1.0) * 100,
                    "vol": 1000.0,
                    "amount": 1_000_000.0,
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d"))


def test_intraday_backtest_suite():
    """test_aggregate_returns_one_row_per_stock_per_day；聚合后每日每股的因子值应是当日最后一根 bar 的值。；test_run_intraday_backtest_has_long_short"""
    # -- 原 test_aggregate_returns_one_row_per_stock_per_day --
    def _section_0_test_aggregate_returns_one_row_per_stock_per_day():
        minute_factor = _make_minute_factor()
        daily = aggregate_intraday_factor(minute_factor)
        assert "trade_date" in daily.columns
        assert "ts_code" in daily.columns
        assert "factor_value" in daily.columns
        n_dates = minute_factor["trade_date"].n_unique()
        n_stocks = minute_factor["ts_code"].n_unique()
        assert len(daily) == n_dates * n_stocks

    _section_0_test_aggregate_returns_one_row_per_stock_per_day()

    # -- 原 test_aggregate_takes_last_value --
    def _section_1_test_aggregate_takes_last_value():
        df = _make_minute_factor(n_stocks=1, n_days=1, bars_per_day=5)
        daily = aggregate_intraday_factor(df)
        expected_last = df.sort("trade_time").tail(1)["factor_value"][0]
        assert abs(daily["factor_value"][0] - expected_last) < 1e-9

    _section_1_test_aggregate_takes_last_value()

    # -- 原 test_run_intraday_backtest_has_long_short --
    def _section_2_test_run_intraday_backtest_has_long_short():
        result = run_intraday_backtest(_make_minute_factor(), _make_daily_price(), n_groups=5)
        assert "long_short" in result.summary_stats
        assert not result.nav.is_empty()

    _section_2_test_run_intraday_backtest_has_long_short()


# ==== 来自 test_lookahead_safety.py ====
def _make_synthetic_data(n_dates: int = 200, n_stocks: int = 100, seed: int = 42):
    """构造随机游走价格数据。"""
    rng = np.random.default_rng(seed)

    dates = [f"2024-{(i // 28 + 1):02d}-{(i % 28 + 1):02d}" for i in range(n_dates)]
    stocks = [f"{i:06d}.SZ" for i in range(n_stocks)]

    price_rows = []
    same_day_factor_rows = []
    leaked_factor_rows = []
    for s in stocks:
        rets = rng.normal(0.0002, 0.02, n_dates)
        last_close = 1.0
        for j, d in enumerate(dates):
            open_price = last_close
            close_price = open_price * (1.0 + rets[j])
            price_rows.append(
                {
                    "trade_date": d,
                    "ts_code": s,
                    "open": float(open_price),
                    "close": float(close_price),
                    "pre_close": float(last_close),
                    "pct_chg": float(rets[j] * 100),
                    "vol": 1000.0,
                    "amount": 1_000_000_000.0,
                }
            )
            if j < n_dates - 1:
                same_day_factor_rows.append(
                    {"trade_date": d, "ts_code": s, "factor_clean": float(rets[j])}
                )
                leaked_factor_rows.append(
                    {"trade_date": d, "ts_code": s, "factor_clean": float(rets[j + 1])}
                )
            last_close = close_price

    return (
        pl.DataFrame(same_day_factor_rows),
        pl.DataFrame(leaked_factor_rows),
        pl.DataFrame(price_rows),
    )


class TestLookaheadSafety:
    def test_lookahead_safety_suite(self):
        """因子 = t 日收益时，t+1 执行后 Sharpe 应接近 0。；因子 = t+1 收益是人为未来泄漏，Sharpe 应极高。；BacktestResult.ret_definition 应记录新执行收益口径。"""
        # -- 原 test_same_day_return_factor_is_not_traded_same_day --
        same_day_factor, _, price_df = _make_synthetic_data()
        result = run_stratified_backtest(same_day_factor, price_df, n_groups=5)
        sharpe_same_day = result.summary_stats["long_short"]["sharpe"]
        assert abs(sharpe_same_day) < 2.0

        # -- 原 test_future_return_factor_is_inflated --
        _, leaked_factor, price_df = _make_synthetic_data()
        result = run_stratified_backtest(leaked_factor, price_df, n_groups=5)
        sharpe_leaked = result.summary_stats["long_short"]["sharpe"]
        assert sharpe_leaked > 5.0

        # -- 原 test_ret_definition_field --
        same_day_factor, _, price_df = _make_synthetic_data()
        result = run_stratified_backtest(same_day_factor, price_df, n_groups=5)
        assert result.ret_definition == "open_to_close_with_overnight_carry"


# ==== 来自 test_backtest_costs.py ====
def _make_data(n_dates: int = 200, n_stocks: int = 100, seed: int = 0):
    """合成因子+价格数据。"""
    rng = np.random.default_rng(seed)
    dates = [f"2024-{(i // 28 + 1):02d}-{(i % 28 + 1):02d}" for i in range(n_dates)]
    stocks = [f"{i:06d}.SZ" for i in range(n_stocks)]

    factor_rows, price_rows = [], []
    last_close = {s: 10.0 + i for i, s in enumerate(stocks)}
    for day_idx, d in enumerate(dates):
        factor_vals = rng.standard_normal(n_stocks)
        intraday_rets = 0.05 * factor_vals / n_stocks + rng.normal(0, 0.02, n_stocks)
        for i, s in enumerate(stocks):
            if day_idx < n_dates - 1:
                factor_rows.append(
                    {"trade_date": d, "ts_code": s, "factor_clean": float(factor_vals[i])}
                )
            open_price = last_close[s]
            close_price = open_price * (1.0 + float(intraday_rets[i]))
            price_rows.append(
                {
                    "trade_date": d,
                    "ts_code": s,
                    "open": open_price,
                    "close": close_price,
                    "pre_close": last_close[s],
                    "pct_chg": (close_price / last_close[s] - 1.0) * 100,
                    "vol": 1000.0,
                    "amount": 1_000_000_000.0,
                }
            )
            last_close[s] = close_price

    return pl.DataFrame(factor_rows), pl.DataFrame(price_rows)


def test_backtest_costs_suite():
    """CostModel 默认值应与 constants.py 一致。；往返成本 = 2×commission + 2×slippage + stamp_tax。；日频融券费率 = 年化 / 252。；周频融券费率 = 年化 × 5 / 252。；支持自定义成本参数。；启用 CostModel 后，多空年化收益应低于无成本版本。"""
    # -- 原 test_default_values --
    def _section_0_test_default_values():
        cm = CostModel()
        assert cm.commission == COMMISSION_RATE
        assert cm.stamp_tax == STAMP_TAX_RATE
        assert cm.slippage == SLIPPAGE_RATE
        assert cm.borrow_annual == BORROW_RATE_ANNUAL

    _section_0_test_default_values()

    # -- 原 test_round_trip_cost --
    def _section_1_test_round_trip_cost():
        cm = CostModel()
        expected = 2 * cm.commission + 2 * cm.slippage + cm.stamp_tax
        assert abs(cm.round_trip_cost() - expected) < 1e-10

    _section_1_test_round_trip_cost()

    # -- 原 test_borrow_rate_per_period_daily --
    def _section_2_test_borrow_rate_per_period_daily():
        cm = CostModel()
        expected = cm.borrow_annual / TRADING_DAYS_PER_YEAR
        assert abs(cm.borrow_rate_per_period("daily") - expected) < 1e-12

    _section_2_test_borrow_rate_per_period_daily()

    # -- 原 test_borrow_rate_per_period_weekly --
    def _section_3_test_borrow_rate_per_period_weekly():
        cm = CostModel()
        assert (
            abs(cm.borrow_rate_per_period("weekly") - cm.borrow_annual * 5 / TRADING_DAYS_PER_YEAR)
            < 1e-12
        )

    _section_3_test_borrow_rate_per_period_weekly()

    # -- 原 test_custom_values --
    def _section_4_test_custom_values():
        cm = CostModel(commission=0.0001, stamp_tax=0.0, slippage=0.0, borrow_annual=0.05)
        assert abs(cm.round_trip_cost() - 0.0002) < 1e-12  # 2×0.0001

    _section_4_test_custom_values()

    # -- 原 test_costs_reduce_returns --
    def _section_5_test_costs_reduce_returns():
        factor_df, price_df = _make_data()
        r_free = run_stratified_backtest(factor_df, price_df, n_groups=5)
        r_cost = run_stratified_backtest(factor_df, price_df, n_groups=5, cost_model=CostModel())
        ann_ret_free = r_free.summary_stats["long_short"]["ann_ret"]
        ann_ret_cost = r_cost.summary_stats["long_short"]["ann_ret"]
        assert ann_ret_free > ann_ret_cost, (
            f"加入成本后年化收益应下降: 无成本={ann_ret_free:.4f}, 含成本={ann_ret_cost:.4f}"
        )

    _section_5_test_costs_reduce_returns()

class TestBacktestCostIntegration:

    def test_zero_cost_model_matches_no_cost(self):
        """CostModel 全部费率设为 0 时，结果应与 cost_model=None 完全相同。"""
        factor_df, price_df = _make_data()
        r_none = run_stratified_backtest(factor_df, price_df, n_groups=5)
        r_zero = run_stratified_backtest(
            factor_df,
            price_df,
            n_groups=5,
            cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
        )
        # 零成本时结果应近似相同（因 borrow_rate=0 且 round_trip=0）
        s_none = r_none.summary_stats["long_short"]["ann_ret"]
        s_zero = r_zero.summary_stats["long_short"]["ann_ret"]
        # 允许极小浮点误差
        assert abs(s_none - s_zero) < 1e-6, f"零成本模型应与无成本相同: {s_none} vs {s_zero}"


class TestSquareRootImpactCostModel:
    def test_adv_missing_uses_fallback(self):
        """adv=None 时应使用 fallback_adv，成本有限且为正。"""
        m = SquareRootImpactCostModel(alpha=0.1, fallback_adv=1e7)
        cost = m.trade_cost(delta_weight=0.01, adv=None)
        cost_explicit = m.trade_cost(delta_weight=0.01, adv=1e7)
        assert cost == cost_explicit, "adv=None 应等价于 adv=fallback_adv"
        assert cost > 0

    def test_adv_zero_uses_fallback(self):
        """adv=0 时应回退到 fallback_adv，不触发除零。"""
        m = SquareRootImpactCostModel(alpha=0.1, fallback_adv=1e7)
        cost = m.trade_cost(delta_weight=0.01, adv=0.0)
        assert cost > 0
        assert np.isfinite(cost)

    def test_extreme_turnover_finite(self):
        """极端换手（delta_weight=1.0）成本应有限且不溢出。"""
        m = SquareRootImpactCostModel(alpha=0.1, fallback_adv=1e7)
        cost = m.trade_cost(delta_weight=1.0, adv=1e6)
        assert np.isfinite(cost)
        assert cost > 0

    def test_alpha_parameterization(self):
        """alpha 越大，冲击成本越大。"""
        m_low = SquareRootImpactCostModel(alpha=0.01)
        m_high = SquareRootImpactCostModel(alpha=1.0)
        assert m_high.trade_cost(delta_weight=0.05) > m_low.trade_cost(delta_weight=0.05)

    def test_fallback_adv_scales_impact_for_given_adv(self):
        """fallback_adv 是 ADV 归一化基准：基准越大，同等 adv 被视为流动性越差，成本越高。"""
        adv = 1e6  # 固定的实际成交额
        # fallback_adv=1e9：adv_normalized = 1e6/1e9 = 0.001，冲击高
        m_high_ref = SquareRootImpactCostModel(alpha=0.1, fallback_adv=1e9)
        # fallback_adv=1e5：adv_normalized = 1e6/1e5 = 10，冲击低
        m_low_ref = SquareRootImpactCostModel(alpha=0.1, fallback_adv=1e5)
        assert m_high_ref.trade_cost(delta_weight=0.01, adv=adv) > m_low_ref.trade_cost(
            delta_weight=0.01, adv=adv
        )

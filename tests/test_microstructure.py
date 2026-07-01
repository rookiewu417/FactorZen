"""A-share microstructure edge-case tests.

Tests cover:
- Suspended stock (vol=0) blocking buy AND sell
- Limit-up stock blocks buy but allows sell
- Limit-down stock blocks sell but allows buy
- ST stocks filtered by universe
- New listing stocks (<250 days) filtered
- T+1 execution: signal on day t, execution on day t+1
- Fast path suspended stock blocking
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from factorzen.core.universe import _get_board_limit
from factorzen.daily.evaluation.backtest import (
    BacktestConfig,
    BacktestContext,
    CostModel,
    PrecomputedWeightsStrategy,
    Strategy,
    _apply_trade_constraints,
    run_strategy_backtest,
)

# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════


def _make_prices(
    *,
    codes: list[str] | None = None,
    n_days: int = 3,
    base_date: date = date(2024, 1, 1),
    overrides: dict[tuple[int, str], dict] | None = None,
) -> pl.DataFrame:
    """生成合成价格 DataFrame。

    Parameters
    ----------
    codes : list[str]
        股票代码列表，默认 ["000001.SZ"]。
    n_days : int
        天数，默认 3。
    base_date : date
        起始日期。
    overrides : dict
        按 ``(day_index, ts_code)`` 覆盖字段值。
    """
    if codes is None:
        codes = ["000001.SZ"]
    if overrides is None:
        overrides = {}

    rows: list[dict] = []
    for day_idx in range(n_days):
        d = base_date + timedelta(days=day_idx)
        for code in codes:
            row = {
                "trade_date": d,
                "ts_code": code,
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            }
            row.update(overrides.get((day_idx, code), {}))
            rows.append(row)

    return pl.DataFrame(rows)


def _make_factors(
    entries: list[tuple[date, str, float]],
) -> pl.DataFrame:
    """生成合成因子 DataFrame。"""
    return pl.DataFrame(
        [{"trade_date": d, "ts_code": code, "factor_clean": val} for d, code, val in entries]
    )


class BuyOneStrategy(Strategy):
    """做多单只股票。"""

    name = "buy_one"

    def __init__(self, code: str = "000001.SZ") -> None:
        self.code = code

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        return pl.DataFrame({"ts_code": [self.code], "target_weight": [1.0]})


class SellAllStrategy(Strategy):
    """第一天买入，第二天清仓。"""

    name = "sell_all"

    def __init__(self, code: str = "000001.SZ") -> None:
        self.code = code
        self._call_count = 0

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        self._call_count += 1
        target = 1.0 if self._call_count == 1 else 0.0
        return pl.DataFrame({"ts_code": [self.code], "target_weight": [target]})


def _zero_cost() -> CostModel:
    return CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0)


def _default_config(**kwargs) -> BacktestConfig:
    defaults = {
        "initial_capital": 1_000_000,
        "max_participation_rate": 1.0,
        "fallback_adv": 1_000_000.0,
    }
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


@pytest.fixture(autouse=True)
def _no_namechange_by_default(monkeypatch):
    """默认 namechange 不可用，filter_st 统一走降级（按 name 字符串匹配）路径。

    universe.py 用 ``from factorzen.core.loader import fetch_namechange`` 在
    模块级绑定，须 patch ``factorzen.core.universe.fetch_namechange`` 才能
    生效。避免本机 .env 配了真实 token 时，TestSTFiltering 等用例意外触发
    真实网络请求。本文件其余 ST 相关测试（_apply_trade_constraints /
    run_strategy_backtest 的 is_st / is_st_by_date）均由调用方显式传参，
    不经过 namechange，不受此 fixture 影响。
    """

    def _boom() -> pl.DataFrame:
        raise RuntimeError("namechange unavailable in offline tests")

    monkeypatch.setattr("factorzen.core.universe.fetch_namechange", _boom)


# ═══════════════════════════════════════════════════════════
# Test: Suspended stock (vol=0) blocks both buy AND sell
# ═══════════════════════════════════════════════════════════


class TestSuspendedStockBlocking:
    """停牌股票（vol=0）不能买也不能卖。"""

    def test_suspended_blocks_buy_via_apply_trade_constraints(self):
        """Unit test: _apply_trade_constraints 直接检验停牌买入被阻。"""
        price_map = {
            "000001.SZ": {
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 0.0,
                "amount": 0.0,
            }
        }
        filled, reason = _apply_trade_constraints(
            code="000001.SZ",
            delta=1.0,
            price_map=price_map,
            portfolio_value=1_000_000.0,
            config=_default_config(),
        )
        assert filled == 0.0
        assert reason == "suspended"

    def test_suspended_blocks_sell_via_apply_trade_constraints(self):
        """Unit test: _apply_trade_constraints 直接检验停牌卖出被阻。"""
        price_map = {
            "000001.SZ": {
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 0.0,
                "amount": 0.0,
            }
        }
        filled, reason = _apply_trade_constraints(
            code="000001.SZ",
            delta=-0.5,
            price_map=price_map,
            portfolio_value=1_000_000.0,
            config=_default_config(),
        )
        assert filled == 0.0
        assert reason == "suspended"

    def test_suspended_blocks_buy_in_backtest(self):
        """Integration test: 停牌日买入在完整回测中被阻。"""
        prices = _make_prices(
            n_days=3,
            overrides={
                # Day 1 (execution day): suspended
                (1, "000001.SZ"): {"vol": 0.0, "amount": 0.0},
            },
        )
        factors = _make_factors([(date(2024, 1, 1), "000001.SZ", 1.0)])

        result = run_strategy_backtest(
            BuyOneStrategy(),
            factors,
            prices,
            config=_default_config(),
            cost_model=_zero_cost(),
        )

        trade = result.trades.filter(pl.col("trade_date") == date(2024, 1, 2)).row(0, named=True)
        assert trade["filled_delta_weight"] == pytest.approx(0.0)
        assert trade["block_reason"] == "suspended"

    def test_suspended_blocks_sell_in_backtest(self):
        """Integration test: 停牌日卖出在完整回测中被阻。"""
        prices = _make_prices(
            n_days=4,
            overrides={
                # Day 2 (sell execution day): suspended
                (2, "000001.SZ"): {"vol": 0.0, "amount": 0.0},
            },
        )
        factors = _make_factors([
            (date(2024, 1, 1), "000001.SZ", 1.0),
            (date(2024, 1, 2), "000001.SZ", 1.0),
            (date(2024, 1, 3), "000001.SZ", 1.0),
        ])

        result = run_strategy_backtest(
            SellAllStrategy(),
            factors,
            prices,
            config=_default_config(),
            cost_model=_zero_cost(),
        )

        sell_trades = result.trades.filter(
            (pl.col("trade_date") == date(2024, 1, 3))
            & (pl.col("target_weight") < pl.col("prev_weight"))
        )
        if not sell_trades.is_empty():
            trade = sell_trades.row(0, named=True)
            assert trade["filled_delta_weight"] == pytest.approx(0.0)
            assert trade["block_reason"] == "suspended"

    def test_suspended_blocks_in_fast_path(self):
        """Fast path (PrecomputedWeightsStrategy): 停牌阻止交易。"""
        weights_by_date = {
            date(2024, 1, 1): pl.DataFrame(
                {"ts_code": ["000001.SZ"], "target_weight": [1.0]}
            ),
        }
        prices = _make_prices(
            n_days=3,
            overrides={
                (1, "000001.SZ"): {"vol": 0.0, "amount": 0.0},
            },
        )
        # Fast path requires collect_positions=False, collect_trades=False
        result = run_strategy_backtest(
            PrecomputedWeightsStrategy(weights_by_date),
            _make_factors([(date(2024, 1, 1), "000001.SZ", 1.0)]),
            prices,
            config=_default_config(),
            cost_model=_zero_cost(),
            collect_positions=False,
            collect_trades=False,
            include_context_positions=False,
        )

        # If suspended blocking works, the day-2 return should be 0 (no position entered)
        returns_day2 = result.returns.filter(pl.col("trade_date") == date(2024, 1, 2))
        if not returns_day2.is_empty():
            ret = returns_day2["net_return"][0]
            # NAV should stay at 1.0 — no position was taken
            assert abs(ret) < 1e-10, f"Expected ~0 return on suspended day, got {ret}"


# ═══════════════════════════════════════════════════════════
# Test: vol per-row None is not suspension (Fix 2)
# ═══════════════════════════════════════════════════════════


class TestVolNoneNotTreatedAsSuspended:
    """vol 逐行为 None(非整列缺失)时不应被视为停牌,慢/快路径须一致。

    慢路径 ``_apply_trade_constraints`` 的既有语义：只有
    ``vol is not None and float(vol) == 0.0`` 才判停牌，``vol is None``
    (逐行 null，区别于整列缺失时 ``_prepare_price_df`` 兜底填 1.0 的情形)
    不视为停牌。本测试把该语义钉死为两条路径的统一行为。
    """

    def test_slow_path_does_not_block_when_vol_is_none(self):
        """慢路径单元测试：vol=None 的整笔交易不应被判定为停牌。"""
        prices = _make_prices(
            n_days=2,
            overrides={(1, "000001.SZ"): {"vol": None, "close": 11.0}},
        )
        result = run_strategy_backtest(
            BuyOneStrategy(),
            _make_factors([(date(2024, 1, 1), "000001.SZ", 1.0)]),
            prices,
            config=_default_config(),
            cost_model=_zero_cost(),
        )
        trade = result.trades.filter(pl.col("trade_date") == date(2024, 1, 2)).row(0, named=True)
        assert trade["block_reason"] == ""
        assert trade["filled_delta_weight"] == pytest.approx(1.0)

    def test_fast_path_matches_slow_path_when_vol_is_none(self):
        """快速路径（PrecomputedWeightsStrategy）：vol=None 的判断须和慢路径一致。"""
        weights_by_date = {
            date(2024, 1, 1): pl.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [1.0]}),
        }
        prices = _make_prices(
            n_days=2,
            overrides={(1, "000001.SZ"): {"vol": None, "close": 11.0}},
        )
        factors = _make_factors([(date(2024, 1, 1), "000001.SZ", 1.0)])

        slow = run_strategy_backtest(
            PrecomputedWeightsStrategy(weights_by_date),
            factors,
            prices,
            config=_default_config(),
            cost_model=_zero_cost(),
        )
        fast = run_strategy_backtest(
            PrecomputedWeightsStrategy(weights_by_date),
            factors,
            prices,
            config=_default_config(),
            cost_model=_zero_cost(),
            collect_positions=False,
            collect_trades=False,
            include_context_positions=False,
        )

        exec_date = date(2024, 1, 2)
        slow_nav = slow.nav.filter(pl.col("trade_date") == exec_date)["nav"][0]
        fast_nav = fast.nav.filter(pl.col("trade_date") == exec_date)["nav"][0]
        # vol=None 不应阻断买入：000001.SZ 满仓买入 + open=10→close=11 (+10%)
        assert slow_nav == pytest.approx(1.1)
        assert fast_nav == pytest.approx(slow_nav), (
            f"快路径应和慢路径一致地不把 vol=None 当停牌处理，慢路径 nav={slow_nav}，"
            f"快路径 nav={fast_nav}"
        )


# ═══════════════════════════════════════════════════════════
# Test: Limit-up blocks buy but allows sell
# ═══════════════════════════════════════════════════════════


class TestLimitUpBlocking:
    """涨停板阻止买入但允许卖出。"""

    def test_limit_up_blocks_buy(self):
        """涨停开盘: delta > 0 被阻。"""
        price_map = {
            "000001.SZ": {
                "ts_code": "000001.SZ",
                "open": 10.98,
                "close": 10.98,
                "pre_close": 10.0,
                "pct_chg": 9.8,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            }
        }
        filled, reason = _apply_trade_constraints(
            code="000001.SZ",
            delta=0.5,
            price_map=price_map,
            portfolio_value=1_000_000.0,
            config=_default_config(),
        )
        assert filled == 0.0
        assert reason == "limit_up"

    def test_limit_up_allows_sell(self):
        """涨停开盘: delta < 0 允许。"""
        price_map = {
            "000001.SZ": {
                "ts_code": "000001.SZ",
                "open": 10.98,
                "close": 10.98,
                "pre_close": 10.0,
                "pct_chg": 9.8,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            }
        }
        filled, reason = _apply_trade_constraints(
            code="000001.SZ",
            delta=-0.5,
            price_map=price_map,
            portfolio_value=1_000_000.0,
            config=_default_config(),
        )
        assert filled == pytest.approx(-0.5)
        assert reason == ""

    def test_gem_limit_up_uses_20pct_threshold(self):
        """创业板涨停阈值为 19.8%。"""
        price_map = {
            "300001.SZ": {
                "ts_code": "300001.SZ",
                "open": 11.98,
                "close": 11.98,
                "pre_close": 10.0,
                "pct_chg": 19.8,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            }
        }
        filled, reason = _apply_trade_constraints(
            code="300001.SZ",
            delta=0.5,
            price_map=price_map,
            portfolio_value=1_000_000.0,
            config=_default_config(),
        )
        assert filled == 0.0
        assert reason == "limit_up"

    def test_gem_below_20pct_not_blocked(self):
        """创业板涨幅 9.8% 不触发涨停（阈值 19.8%）。"""
        price_map = {
            "300001.SZ": {
                "ts_code": "300001.SZ",
                "open": 10.98,
                "close": 10.98,
                "pre_close": 10.0,
                "pct_chg": 9.8,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            }
        }
        filled, reason = _apply_trade_constraints(
            code="300001.SZ",
            delta=0.5,
            price_map=price_map,
            portfolio_value=1_000_000.0,
            config=_default_config(),
        )
        assert filled == pytest.approx(0.5)
        assert reason == ""


# ═══════════════════════════════════════════════════════════
# Test: Fast path GEM limit-up float tolerance (Fix 1)
# ═══════════════════════════════════════════════════════════


class TestFastPathLimitUpTolerance:
    """快速路径涨跌停浮点容差与慢路径对称（Fix 1）。

    创业板 open=11.98 / pre_close=10.0 → opening_pct=(11.98/10-1)*100=19.7999...
    在快速路径里若不加 -1e-9 容差，19.7999... >= 19.8 = False，
    导致涨停买单被错误成交（虚高收益）。修复后阻断应生效，NAV 应 ≈ 1.0。
    """

    def test_gem_limit_up_blocks_buy_in_fast_path(self):
        """PrecomputedWeightsStrategy 快速路径：创业板浮点边界涨停阻断买入。"""
        weights_by_date = {
            date(2024, 1, 1): pl.DataFrame(
                {"ts_code": ["300001.SZ"], "target_weight": [1.0]}
            ),
        }
        # close=13.0 使 intraday_ret > 0：若买单成交则 NAV > 1.0，可与阻断情形(NAV≈1)区分
        prices = _make_prices(
            codes=["300001.SZ"],
            n_days=3,
            overrides={
                (1, "300001.SZ"): {
                    "open": 11.98,
                    "pre_close": 10.0,
                    "close": 13.0,
                },
            },
        )
        result = run_strategy_backtest(
            PrecomputedWeightsStrategy(weights_by_date),
            _make_factors([(date(2024, 1, 1), "300001.SZ", 1.0)]),
            prices,
            config=_default_config(),
            cost_model=_zero_cost(),
            collect_positions=False,
            collect_trades=False,
            include_context_positions=False,
        )

        # 执行日 2024-01-02 的 NAV:
        # - 涨停阻断生效 → filled=0 → 无仓位 → intraday_return=0 → NAV=1.0
        # - 涨停未阻断(BUG) → filled=1.0 → intraday_ret=(13/11.98-1)>0 → NAV>1.0
        exec_date = date(2024, 1, 2)
        nav_rows = result.nav.filter(pl.col("trade_date") == exec_date)
        assert not nav_rows.is_empty(), "执行日应有 NAV 记录"
        nav_val = nav_rows["nav"][0]
        assert nav_val == pytest.approx(1.0, abs=1e-6), (
            f"GEM 涨停买单应被快速路径阻断 (NAV≈1.0)，实际 NAV={nav_val:.8f}；"
            f"若 NAV>1 说明 fast path 缺少 -1e-9 浮点容差"
        )


# ═══════════════════════════════════════════════════════════
# Test: ST main board narrowed limit threshold (Fix 2)
# ═══════════════════════════════════════════════════════════


class TestSTBoardLimitInBacktest:
    """ST 主板涨跌停阈值收窄（4.8% 而非 9.8%）在慢路径/快路径回测中的接入测试。

    is_st / is_st_by_date 默认 False/None，行为与引入该参数前完全一致；本组
    测试验证传入 ST 标记后，约 +5.0% 涨幅（乘法/除法构造而非字面量）能被
    正确判定为涨停，而同样涨幅的非 ST 股票不受影响（主板非 ST 阈值 9.8%）。
    """

    def test_apply_trade_constraints_st_blocks_buy_at_5pct(self):
        """慢路径单元测试：_apply_trade_constraints(is_st=True) 阻断 5% 涨幅买入。"""
        open_price = 10.0 * 1.05  # 乘法构造，非字面量
        price_map = {
            "600001.SH": {
                "ts_code": "600001.SH",
                "open": open_price,
                "close": open_price,
                "pre_close": 10.0,
                "pct_chg": (open_price / 10.0 - 1.0) * 100,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            }
        }
        filled, reason = _apply_trade_constraints(
            code="600001.SH",
            delta=0.5,
            price_map=price_map,
            portfolio_value=1_000_000.0,
            config=_default_config(),
            is_st=True,
        )
        assert filled == 0.0
        assert reason == "limit_up"

    def test_apply_trade_constraints_non_st_same_5pct_not_blocked(self):
        """同样 5% 涨幅，is_st=False（默认）不应触发涨停（主板阈值 9.8%）。"""
        open_price = 10.0 * 1.05
        price_map = {
            "600001.SH": {
                "ts_code": "600001.SH",
                "open": open_price,
                "close": open_price,
                "pre_close": 10.0,
                "pct_chg": (open_price / 10.0 - 1.0) * 100,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            }
        }
        filled, reason = _apply_trade_constraints(
            code="600001.SH",
            delta=0.5,
            price_map=price_map,
            portfolio_value=1_000_000.0,
            config=_default_config(),
        )
        assert filled == pytest.approx(0.5)
        assert reason == ""

    def test_st_main_board_blocks_buy_in_slow_path_integration(self):
        """慢路径完整回测：run_strategy_backtest(is_st_by_date=...) 应阻断 ST 主板 5% 涨幅买入。"""
        open_px = 10.0 * 1.05
        prices = _make_prices(
            codes=["600001.SH"],
            n_days=3,
            overrides={
                (1, "600001.SH"): {"open": open_px, "pre_close": 10.0, "close": 13.0},
            },
        )
        result = run_strategy_backtest(
            BuyOneStrategy(code="600001.SH"),
            _make_factors([(date(2024, 1, 1), "600001.SH", 1.0)]),
            prices,
            config=_default_config(),
            cost_model=_zero_cost(),
            is_st_by_date={date(2024, 1, 2): {"600001.SH"}},
        )
        trade = result.trades.filter(pl.col("trade_date") == date(2024, 1, 2)).row(0, named=True)
        assert trade["filled_delta_weight"] == pytest.approx(0.0)
        assert trade["block_reason"] == "limit_up"

    def test_non_st_main_board_5pct_not_blocked_in_slow_path_integration(self):
        """慢路径完整回测：不传 is_st_by_date 时同样 5% 涨幅不应被阻断（向后兼容）。"""
        open_px = 10.0 * 1.05
        prices = _make_prices(
            codes=["600001.SH"],
            n_days=3,
            overrides={
                (1, "600001.SH"): {"open": open_px, "pre_close": 10.0, "close": 13.0},
            },
        )
        result = run_strategy_backtest(
            BuyOneStrategy(code="600001.SH"),
            _make_factors([(date(2024, 1, 1), "600001.SH", 1.0)]),
            prices,
            config=_default_config(),
            cost_model=_zero_cost(),
        )
        trade = result.trades.filter(pl.col("trade_date") == date(2024, 1, 2)).row(0, named=True)
        # BuyOneStrategy 目标权重恒为 1.0，首日 prev_weight=0.0 → delta=1.0（非阻断应全额成交）
        assert trade["filled_delta_weight"] == pytest.approx(1.0)
        assert trade["block_reason"] == ""

    def test_st_main_board_blocks_buy_in_fast_path(self):
        """快速路径（PrecomputedWeightsStrategy + is_st_by_date）：ST 主板 5% 涨幅应被阻断买入。"""
        weights_by_date = {
            date(2024, 1, 1): pl.DataFrame(
                {"ts_code": ["600001.SH"], "target_weight": [1.0]}
            ),
        }
        open_px = 10.0 * 1.05
        # close=13.0 使 intraday_ret > 0：若买单成交则 NAV > 1.0，可与阻断情形(NAV≈1)区分
        prices = _make_prices(
            codes=["600001.SH"],
            n_days=3,
            overrides={
                (1, "600001.SH"): {"open": open_px, "pre_close": 10.0, "close": 13.0},
            },
        )
        result = run_strategy_backtest(
            PrecomputedWeightsStrategy(weights_by_date),
            _make_factors([(date(2024, 1, 1), "600001.SH", 1.0)]),
            prices,
            config=_default_config(),
            cost_model=_zero_cost(),
            collect_positions=False,
            collect_trades=False,
            include_context_positions=False,
            is_st_by_date={date(2024, 1, 2): {"600001.SH"}},
        )

        exec_date = date(2024, 1, 2)
        nav_rows = result.nav.filter(pl.col("trade_date") == exec_date)
        assert not nav_rows.is_empty(), "执行日应有 NAV 记录"
        nav_val = nav_rows["nav"][0]
        assert nav_val == pytest.approx(1.0, abs=1e-6), (
            f"ST 主板 5% 涨幅应被快速路径阻断买入 (NAV≈1.0)，实际 NAV={nav_val:.8f}"
        )

    def test_non_st_main_board_5pct_not_blocked_in_fast_path(self):
        """快速路径：不传 is_st_by_date 时同样 5% 涨幅不应被阻断（主板阈值 9.8%，向后兼容）。"""
        weights_by_date = {
            date(2024, 1, 1): pl.DataFrame(
                {"ts_code": ["600001.SH"], "target_weight": [1.0]}
            ),
        }
        open_px = 10.0 * 1.05
        prices = _make_prices(
            codes=["600001.SH"],
            n_days=3,
            overrides={
                (1, "600001.SH"): {"open": open_px, "pre_close": 10.0, "close": 13.0},
            },
        )
        result = run_strategy_backtest(
            PrecomputedWeightsStrategy(weights_by_date),
            _make_factors([(date(2024, 1, 1), "600001.SH", 1.0)]),
            prices,
            config=_default_config(),
            cost_model=_zero_cost(),
            collect_positions=False,
            collect_trades=False,
            include_context_positions=False,
        )

        exec_date = date(2024, 1, 2)
        nav_rows = result.nav.filter(pl.col("trade_date") == exec_date)
        assert not nav_rows.is_empty()
        nav_val = nav_rows["nav"][0]
        assert nav_val > 1.0, "非 ST 5% 涨幅不应被阻断，应正常成交获得正收益"


# ═══════════════════════════════════════════════════════════
# Test: Limit-down blocks sell but allows buy
# ═══════════════════════════════════════════════════════════


class TestLimitDownBlocking:
    """跌停板阻止卖出但允许买入。"""

    def test_limit_down_blocks_sell(self):
        """跌停开盘: delta < 0 被阻。"""
        price_map = {
            "000001.SZ": {
                "ts_code": "000001.SZ",
                "open": 9.02,
                "close": 9.02,
                "pre_close": 10.0,
                "pct_chg": -9.8,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            }
        }
        filled, reason = _apply_trade_constraints(
            code="000001.SZ",
            delta=-0.5,
            price_map=price_map,
            portfolio_value=1_000_000.0,
            config=_default_config(),
        )
        assert filled == 0.0
        assert reason == "limit_down"

    def test_limit_down_allows_buy(self):
        """跌停开盘: delta > 0 允许。"""
        price_map = {
            "000001.SZ": {
                "ts_code": "000001.SZ",
                "open": 9.02,
                "close": 9.02,
                "pre_close": 10.0,
                "pct_chg": -9.8,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            }
        }
        filled, reason = _apply_trade_constraints(
            code="000001.SZ",
            delta=0.5,
            price_map=price_map,
            portfolio_value=1_000_000.0,
            config=_default_config(),
        )
        assert filled == pytest.approx(0.5)
        assert reason == ""


# ═══════════════════════════════════════════════════════════
# Test: ST stocks filtering by universe
# ═══════════════════════════════════════════════════════════


class TestSTFiltering:
    """ST / *ST / PT 股票通过 filter_st 被剔除。"""

    def test_filter_st_removes_st_stocks(self):
        from factorzen.core.universe import filter_st

        stocks = pl.DataFrame({
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "name": ["平安银行", "*ST远程", "PT南洋"],
        })
        result = filter_st(stocks, "20240101")
        assert result.height == 1
        assert result["ts_code"][0] == "000001.SZ"

    def test_filter_st_keeps_normal_stocks(self):
        from factorzen.core.universe import filter_st

        stocks = pl.DataFrame({
            "ts_code": ["000001.SZ", "600519.SH"],
            "name": ["平安银行", "贵州茅台"],
        })
        result = filter_st(stocks, "20240101")
        assert result.height == 2


# ═══════════════════════════════════════════════════════════
# Test: New listing stocks (<250 days) filtered
# ═══════════════════════════════════════════════════════════


class TestNewListingFiltering:
    """上市不足 250 个自然日的次新股被剔除。"""

    def test_filter_new_listing_removes_recent_ipo(self):
        from factorzen.core.universe import filter_new_listing

        stocks = pl.DataFrame({
            "ts_code": ["000001.SZ", "000002.SZ"],
            "list_date": [
                date(2023, 1, 1),   # >250 days → keep
                date(2024, 3, 1),   # <250 days → remove
            ],
        })
        result = filter_new_listing(stocks, "20240601", min_days=250)
        assert result.height == 1
        assert result["ts_code"][0] == "000001.SZ"

    def test_filter_new_listing_exact_boundary(self):
        from factorzen.core.universe import filter_new_listing

        target = date(2024, 6, 1)
        cutoff = target - timedelta(days=250)
        stocks = pl.DataFrame({
            "ts_code": ["exact.SZ"],
            "list_date": [cutoff],  # exactly 250 days → keep (<=)
        })
        result = filter_new_listing(stocks, "20240601", min_days=250)
        assert result.height == 1


# ═══════════════════════════════════════════════════════════
# Test: T+1 execution semantics
# ═══════════════════════════════════════════════════════════


class TestTPlus1Execution:
    """信号在 t 日生成，调仓在 t+1 日执行。"""

    def test_signal_date_before_execution_date(self):
        """Trade 发生在 signal date 的下一个交易日。"""
        prices = _make_prices(n_days=3)
        factors = _make_factors([(date(2024, 1, 1), "000001.SZ", 1.0)])

        result = run_strategy_backtest(
            BuyOneStrategy(),
            factors,
            prices,
            config=_default_config(),
            cost_model=_zero_cost(),
        )

        if not result.trades.is_empty():
            first_trade = result.trades.sort("trade_date").row(0, named=True)
            assert first_trade["trade_date"] == date(2024, 1, 2)

    def test_no_same_day_execution(self):
        """信号日本身不应有交易执行。"""
        prices = _make_prices(n_days=3)
        factors = _make_factors([(date(2024, 1, 1), "000001.SZ", 1.0)])

        result = run_strategy_backtest(
            BuyOneStrategy(),
            factors,
            prices,
            config=_default_config(),
            cost_model=_zero_cost(),
        )

        day1_trades = result.trades.filter(pl.col("trade_date") == date(2024, 1, 1))
        assert day1_trades.is_empty()

    def test_returns_start_on_execution_date(self):
        """收益序列从第一个执行日开始记录。"""
        prices = _make_prices(n_days=3)
        factors = _make_factors([(date(2024, 1, 1), "000001.SZ", 1.0)])

        result = run_strategy_backtest(
            BuyOneStrategy(),
            factors,
            prices,
            config=_default_config(),
            cost_model=_zero_cost(),
        )

        sorted_returns = result.returns.sort("trade_date")
        assert sorted_returns["trade_date"][0] == date(2024, 1, 2)


# ═══════════════════════════════════════════════════════════
# Test: Board limit helper
# ═══════════════════════════════════════════════════════════


class TestBoardLimit:
    """验证 _get_board_limit 按板块返回正确阈值。"""

    def test_main_board(self):
        assert _get_board_limit("000001.SZ") == pytest.approx(0.098)

    def test_gem_board(self):
        assert _get_board_limit("300001.SZ") == pytest.approx(0.198)

    def test_star_board(self):
        assert _get_board_limit("688001.SH") == pytest.approx(0.198)

    def test_bse_board(self):
        assert _get_board_limit("430001.BJ") == pytest.approx(0.298)


# ═══════════════════════════════════════════════════════════
# Test: Benchmark utilities
# ═══════════════════════════════════════════════════════════


class TestBenchmark:
    """验证 benchmark 工具的基本功能。"""

    def test_benchmark_step_records_timing(self):
        from factorzen.core.benchmark import BenchmarkReport, benchmark_step

        report = BenchmarkReport()

        @benchmark_step(report, "test_step")
        def dummy_work():
            total = 0
            for i in range(1000):
                total += i
            return total

        result = dummy_work()
        assert result == sum(range(1000))
        assert len(report.steps) == 1
        assert report.steps[0].name == "test_step"
        assert report.steps[0].elapsed_seconds > 0

    def test_benchmark_report_total_elapsed(self):
        from factorzen.core.benchmark import BenchmarkReport

        report = BenchmarkReport()
        report.add_step("a", 1.5, 100.0)
        report.add_step("b", 2.5, 200.0)
        assert report.total_elapsed == pytest.approx(4.0)

    def test_format_benchmark_report(self):
        from factorzen.core.benchmark import BenchmarkReport, format_benchmark_report

        report = BenchmarkReport()
        report.add_step("step_a", 1.234, 128.5)
        report.add_step("step_b", 0.567)

        output = format_benchmark_report(report)
        assert "steps" in output
        assert "total_elapsed" in output
        assert len(output["steps"]) == 2
        assert output["steps"][0]["name"] == "step_a"
        assert output["steps"][0]["peak_memory_mb"] == 128.5
        assert output["steps"][1]["peak_memory_mb"] is None
        assert output["total_elapsed"] == pytest.approx(1.801)

    def test_benchmark_step_preserves_exceptions(self):
        from factorzen.core.benchmark import BenchmarkReport, benchmark_step

        report = BenchmarkReport()

        @benchmark_step(report, "failing_step")
        def failing():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing()
        # Step timing should still be recorded even on exception
        assert len(report.steps) == 1
        assert report.steps[0].name == "failing_step"


# ═══════════════════════════════════════════════════════════
# Test: Suspended + limit combined edge cases
# ═══════════════════════════════════════════════════════════


class TestCombinedEdgeCases:
    """组合边界情况。"""

    def test_suspended_takes_priority_over_limit_up(self):
        """停牌优先于涨停判断。"""
        price_map = {
            "000001.SZ": {
                "ts_code": "000001.SZ",
                "open": 10.98,
                "close": 10.98,
                "pre_close": 10.0,
                "pct_chg": 9.8,
                "vol": 0.0,  # suspended
                "amount": 0.0,
            }
        }
        filled, reason = _apply_trade_constraints(
            code="000001.SZ",
            delta=0.5,
            price_map=price_map,
            portfolio_value=1_000_000.0,
            config=_default_config(),
        )
        assert filled == 0.0
        assert reason == "suspended"

    def test_normal_vol_not_blocked_as_suspended(self):
        """正常交易量的股票不被停牌逻辑阻断。"""
        price_map = {
            "000001.SZ": {
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            }
        }
        filled, reason = _apply_trade_constraints(
            code="000001.SZ",
            delta=0.5,
            price_map=price_map,
            portfolio_value=1_000_000.0,
            config=_default_config(),
        )
        assert filled == pytest.approx(0.5)
        assert reason == ""

    def test_zero_delta_returns_empty_reason(self):
        """零变动不应产生任何 block。"""
        price_map = {
            "000001.SZ": {
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 0.0,
                "amount": 0.0,
            }
        }
        filled, reason = _apply_trade_constraints(
            code="000001.SZ",
            delta=0.0,
            price_map=price_map,
            portfolio_value=1_000_000.0,
            config=_default_config(),
        )
        assert filled == 0.0
        assert reason == ""

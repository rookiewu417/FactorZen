"""快/慢路径对『signal 日在因子日历内但无预计算权重』的语义必须一致（D4）。

根因：signal 日 ∈ factor_by_date 但 ∉ weights_by_date 时，慢路径 generate_weights 返回
空权重 → 被当作目标全清仓（卖出、计成本），而快路径视为『无调仓指令』继续持有。二者 NAV/
换手分叉——研究/归因(慢路径)与模拟交易(快路径)对同一组预计算权重给出不同结果。
正确语义：非调仓日应持有（快路径行为），慢路径须对齐。
"""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from factorzen.daily.evaluation.backtest import (
    BacktestConfig,
    CostModel,
    PrecomputedWeightsStrategy,
    run_strategy_backtest,
)

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


def test_fast_and_slow_path_hold_on_missing_weights():
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


def test_explicit_empty_weights_flat_not_carry_fast_and_slow():
    """#5：weights_by_date 存在但空表 = 显式空仓 → 平到空；不得 carry 前仓。

    双路径必须一致：旧快路径把「空 indices」当「无信号」carry，慢路径则 flat。
    """
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


def test_missing_signal_date_still_carries_not_flat():
    """对照：sig_date 完全不在 weights_by_date → 仍 carry（#5 不得误伤缺失语义）。"""
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

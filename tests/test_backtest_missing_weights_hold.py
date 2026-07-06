"""快/慢路径对『signal 日在因子日历内但无预计算权重』的语义必须一致（D4）。

根因：signal 日 ∈ factor_by_date 但 ∉ weights_by_date 时，慢路径 generate_weights 返回
空权重 → 被当作目标全清仓（卖出、计成本），而快路径视为『无调仓指令』继续持有。二者 NAV/
换手分叉——研究/归因(慢路径)与模拟交易(快路径)对同一组预计算权重给出不同结果。
正确语义：非调仓日应持有（快路径行为），慢路径须对齐。
"""
from __future__ import annotations

from datetime import date

import polars as pl

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
    for di, d in enumerate(_DATES):
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

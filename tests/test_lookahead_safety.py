"""S0 防回归：验证策略回测不会使用同日收益。

新回测口径：
- t 日因子只能在 t+1 开盘调仓。
- 因子 = t 日收益时，不应预测 t+1 收益。
- 因子 = t+1 收益（人为未来泄漏）时，Sharpe 才会虚高。
"""

import numpy as np
import polars as pl

from daily.evaluation.backtest import run_stratified_backtest


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
    def test_same_day_return_factor_is_not_traded_same_day(self):
        """因子 = t 日收益时，t+1 执行后 Sharpe 应接近 0。"""
        same_day_factor, _, price_df = _make_synthetic_data()
        result = run_stratified_backtest(same_day_factor, price_df, n_groups=5)
        sharpe_same_day = result.summary_stats["long_short"]["sharpe"]
        assert abs(sharpe_same_day) < 2.0

    def test_future_return_factor_is_inflated(self):
        """因子 = t+1 收益是人为未来泄漏，Sharpe 应极高。"""
        _, leaked_factor, price_df = _make_synthetic_data()
        result = run_stratified_backtest(leaked_factor, price_df, n_groups=5)
        sharpe_leaked = result.summary_stats["long_short"]["sharpe"]
        assert sharpe_leaked > 5.0

    def test_future_leakage_gap_is_significant(self):
        """同日收益因子与未来泄漏因子的 Sharpe 差距应显著。"""
        same_day_factor, leaked_factor, price_df = _make_synthetic_data()
        r_same = run_stratified_backtest(same_day_factor, price_df, n_groups=5)
        r_leaked = run_stratified_backtest(leaked_factor, price_df, n_groups=5)
        gap = (
            r_leaked.summary_stats["long_short"]["sharpe"]
            - r_same.summary_stats["long_short"]["sharpe"]
        )
        assert gap >= 3.0

    def test_ret_definition_field(self):
        """BacktestResult.ret_definition 应记录新执行收益口径。"""
        same_day_factor, _, price_df = _make_synthetic_data()
        result = run_stratified_backtest(same_day_factor, price_df, n_groups=5)
        assert result.ret_definition == "open_to_close_with_overnight_carry"

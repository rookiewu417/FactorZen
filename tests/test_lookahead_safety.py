"""S0 防回归：验证 run_stratified_backtest 使用前向收益与同日收益时结果有显著差异。

构造方式：
- 用随机游走生成价格路径（收益 i.i.d.），避免独立价格隐含的均值回归偏差
- 因子 = 当日收益（与同日 ret 完全相同 → look-ahead）
- 用同日 ret 回测 → Sharpe 极高（因子即收益）
- 用 fwd_ret_1d 回测 → Sharpe 接近 0（未来收益独立）
"""

import numpy as np
import polars as pl

from daily.evaluation.backtest import run_stratified_backtest


def _make_synthetic_data(n_dates: int = 200, n_stocks: int = 100, seed: int = 42):
    """构造随机游走价格数据，因子 = 当日收益（人为制造 look-ahead）。"""
    rng = np.random.default_rng(seed)

    dates = [f"2024-{(i // 28 + 1):02d}-{(i % 28 + 1):02d}" for i in range(n_dates)]
    stocks = [f"{i:06d}.SZ" for i in range(n_stocks)]

    records = []
    # 每只股票独立随机游走
    for _i, s in enumerate(stocks):
        rets = rng.normal(0.0002, 0.02, n_dates)
        prices = np.cumprod(1 + rets)
        for j, d in enumerate(dates):
            records.append(
                {
                    "trade_date": d,
                    "ts_code": s,
                    "close": float(prices[j]),
                    "true_ret": float(rets[j]),
                }
            )

    df = pl.DataFrame(records).sort(["ts_code", "trade_date"])

    # 用实际收益构造前向收益（避免计算误差）
    df = df.with_columns(pl.col("true_ret").shift(-1).over("ts_code").alias("fwd_ret_1d")).filter(
        pl.col("fwd_ret_1d").is_not_null()
    )

    # 因子 = 当日收益（与 same_day_ret 完全相同，人为 look-ahead）
    factor_df = df.select(["trade_date", "ts_code", "true_ret"]).rename(
        {"true_ret": "factor_clean"}
    )
    same_day_ret = df.select(["trade_date", "ts_code", "true_ret"]).rename({"true_ret": "ret"})
    fwd_ret = df.select(["trade_date", "ts_code", "fwd_ret_1d"]).rename({"fwd_ret_1d": "ret"})

    return factor_df, same_day_ret, fwd_ret


class TestLookaheadSafety:
    def test_same_day_ret_sharpe_inflated(self):
        """因子 = 当日收益，用同日 ret 回测：多空 Sharpe 应极高（因子即收益，完美预测）。"""
        factor_df, same_day_ret, _ = _make_synthetic_data()
        result = run_stratified_backtest(factor_df, same_day_ret, n_groups=5)
        sharpe_same = result.summary_stats["long_short"]["sharpe"]
        assert sharpe_same > 5.0, (
            f"同日 ret Sharpe={sharpe_same:.2f} 应 >> 5（因子与收益完全相同，存在 look-ahead）"
        )

    def test_fwd_ret_sharpe_near_zero(self):
        """因子 = 当日收益，用 fwd_ret_1d 回测：fwd_ret 独立，Sharpe 应接近 0。"""
        factor_df, _, fwd_ret = _make_synthetic_data()
        result = run_stratified_backtest(factor_df, fwd_ret, n_groups=5)
        sharpe_fwd = result.summary_stats["long_short"]["sharpe"]
        # i.i.d. 收益下，随机因子年化 Sharpe 的期望标准差约 sqrt(252/N_dates) ≈ 1.1
        # 放宽到 2.0 避免偶发 false positive
        assert abs(sharpe_fwd) < 2.0, (
            f"fwd_ret Sharpe={sharpe_fwd:.2f} 应接近 0（独立 i.i.d. 收益，因子无预测力）"
        )

    def test_sharpe_gap_significant(self):
        """同日 ret 与 fwd_ret_1d 的 Sharpe 差距应 >= 3，说明 look-ahead 对结果有实质影响。"""
        factor_df, same_day_ret, fwd_ret = _make_synthetic_data()
        r_same = run_stratified_backtest(factor_df, same_day_ret, n_groups=5)
        r_fwd = run_stratified_backtest(factor_df, fwd_ret, n_groups=5)
        sharpe_same = r_same.summary_stats["long_short"]["sharpe"]
        sharpe_fwd = r_fwd.summary_stats["long_short"]["sharpe"]
        gap = sharpe_same - sharpe_fwd
        assert gap >= 3.0, (
            f"同日 ret Sharpe ({sharpe_same:.2f}) vs fwd_ret Sharpe ({sharpe_fwd:.2f}): "
            f"差距 {gap:.2f} < 3.0，look-ahead bias 未被有效检测"
        )

    def test_ret_definition_field(self):
        """BacktestResult.ret_definition 应记录为 'fwd_ret_1d'。"""
        factor_df, _, fwd_ret = _make_synthetic_data()
        result = run_stratified_backtest(factor_df, fwd_ret, n_groups=5)
        assert result.ret_definition == "fwd_ret_1d"

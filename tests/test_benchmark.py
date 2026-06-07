"""daily/evaluation/benchmark.py 的单元测试。

验证:
- BenchmarkResult 结构完整
- 超额收益数学一致性
- 统计指标方向正确
- 边界情况（零跟踪误差、空数据）
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from factorzen.daily.evaluation.benchmark import BenchmarkResult, compute_excess_return

# ── 辅助函数 ──────────────────────────────────────────────────────────


def _make_index_df(dates: list[str], seed: int = 42) -> pl.DataFrame:
    """合成基准指数 close 价格序列，用于 mock fetch_index_daily。"""
    rng = np.random.default_rng(seed)
    closes = np.cumprod(1 + rng.normal(0.0005, 0.01, len(dates)))
    return pl.DataFrame(
        {
            "trade_date": pl.Series(dates).str.strptime(pl.Date, "%Y-%m-%d"),
            "ts_code": ["000300.SH"] * len(dates),
            "close": closes,
        }
    )


def _make_strategy_nav(dates: list[str], seed: int = 99) -> pl.DataFrame:
    """合成策略日收益 DataFrame（net_return 列）。"""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0008, 0.012, len(dates))
    return pl.DataFrame(
        {
            "trade_date": dates,  # str format "YYYY-MM-DD"
            "net_return": rets,
            "nav": np.cumprod(1 + rets),
        }
    )


# ── 测试类 ────────────────────────────────────────────────────────────


class TestComputeExcessReturn(unittest.TestCase):
    """compute_excess_return 的单元测试，使用 mock 代替真实 Tushare 调用。"""

    def _dates(self, n: int = 40) -> list[str]:
        return [f"2026-01-{d + 1:02d}" if d < 31 else f"2026-02-{d - 30:02d}" for d in range(n)]

    @patch("factorzen.core.loader.fetch_index_daily")
    def test_preloaded_benchmark_skips_fetch(
        self, mock_fetch: unittest.mock.MagicMock
    ) -> None:
        dates = self._dates(10)

        result = compute_excess_return(
            _make_strategy_nav(dates),
            "000300.SH",
            "20260101",
            "20260110",
            benchmark_data=_make_index_df(dates),
        )

        self.assertGreater(result.daily.height, 0)
        mock_fetch.assert_not_called()

    @patch("factorzen.core.loader.fetch_index_daily")
    def test_basic_structure(self, mock_fetch: unittest.mock.MagicMock) -> None:
        """compute_excess_return 返回结构正确的 BenchmarkResult。"""
        dates = self._dates(40)
        mock_fetch.return_value = _make_index_df(dates)
        strategy_nav = _make_strategy_nav(dates)

        result = compute_excess_return(strategy_nav, "000300.SH", "20260101", "20260209")

        self.assertIsInstance(result, BenchmarkResult)
        # daily DataFrame 含必需列
        required_cols = {
            "trade_date",
            "strategy_ret",
            "benchmark_ret",
            "excess_ret",
            "strategy_nav",
            "benchmark_nav",
            "excess_nav",
        }
        self.assertTrue(required_cols.issubset(set(result.daily.columns)))
        self.assertGreater(result.daily.height, 0)

    @patch("factorzen.core.loader.fetch_index_daily")
    def test_excess_return_math(self, mock_fetch: unittest.mock.MagicMock) -> None:
        """超额收益 = 策略收益 - 基准收益，超额净值为超额收益的累积乘积。"""
        dates = self._dates(40)
        mock_fetch.return_value = _make_index_df(dates, seed=10)
        strategy_nav = _make_strategy_nav(dates, seed=20)

        result = compute_excess_return(strategy_nav, "000300.SH", "20260101", "20260209")

        df = result.daily
        strategy_ret = df["strategy_ret"].to_numpy()
        benchmark_ret = df["benchmark_ret"].to_numpy()
        excess_ret = df["excess_ret"].to_numpy()
        excess_nav = df["excess_nav"].to_numpy()

        # excess_ret = strategy_ret - benchmark_ret
        np.testing.assert_allclose(
            excess_ret,
            strategy_ret - benchmark_ret,
            atol=1e-10,
            err_msg="excess_ret != strategy_ret - benchmark_ret",
        )

        # excess_nav[i] = prod(1 + excess_ret[:i+1])
        expected_nav = np.cumprod(1 + excess_ret)
        np.testing.assert_allclose(
            excess_nav,
            expected_nav,
            atol=1e-10,
            err_msg="excess_nav does not match cumprod(1 + excess_ret)",
        )

    @patch("factorzen.core.loader.fetch_index_daily")
    def test_ann_excess_ret_direction(self, mock_fetch: unittest.mock.MagicMock) -> None:
        """策略持续跑赢基准时，ann_excess_ret > 0。"""
        dates = self._dates(40)
        # 基准收益极低（接近 0），策略收益明显正向
        rng = np.random.default_rng(77)
        low_closes = np.cumprod(1 + rng.normal(0.0, 0.001, len(dates)))
        index_df = pl.DataFrame(
            {
                "trade_date": pl.Series(dates).str.strptime(pl.Date, "%Y-%m-%d"),
                "ts_code": ["000300.SH"] * len(dates),
                "close": low_closes,
            }
        )
        mock_fetch.return_value = index_df

        # 策略每日收益固定为正（0.002），确保策略 > 基准
        strategy_nav = pl.DataFrame(
            {
                "trade_date": dates,
                "net_return": [0.002] * len(dates),
                "nav": np.cumprod([1.002] * len(dates)),
            }
        )

        result = compute_excess_return(strategy_nav, "000300.SH", "20260101", "20260209")

        self.assertGreater(result.ann_excess_ret, 0.0)

    @patch("factorzen.core.loader.fetch_index_daily")
    def test_tracking_error_nonnegative(self, mock_fetch: unittest.mock.MagicMock) -> None:
        """tracking_error >= 0 恒成立。"""
        dates = self._dates(40)
        mock_fetch.return_value = _make_index_df(dates, seed=5)
        strategy_nav = _make_strategy_nav(dates, seed=6)

        result = compute_excess_return(strategy_nav, "000300.SH", "20260101", "20260209")

        self.assertGreaterEqual(result.tracking_error, 0.0)

    @patch("factorzen.core.loader.fetch_index_daily")
    def test_excess_max_dd_nonpositive(self, mock_fetch: unittest.mock.MagicMock) -> None:
        """excess_max_dd <= 0 恒成立（最大回撤为非正数）。"""
        dates = self._dates(40)
        mock_fetch.return_value = _make_index_df(dates, seed=7)
        strategy_nav = _make_strategy_nav(dates, seed=8)

        result = compute_excess_return(strategy_nav, "000300.SH", "20260101", "20260209")

        self.assertLessEqual(result.excess_max_dd, 0.0)

    @patch("factorzen.core.loader.fetch_index_daily")
    def test_ir_zero_when_no_volatility(self, mock_fetch: unittest.mock.MagicMock) -> None:
        """策略与基准收益完全一致时，超额收益方差为 0，IR 应返回 0.0。"""
        dates = self._dates(40)
        index_df = _make_index_df(dates, seed=42)
        mock_fetch.return_value = index_df

        # 策略收益 = 基准收益（从 index_df close 反推）
        closes = index_df["close"].to_numpy()
        bm_rets = closes[1:] / closes[:-1] - 1
        # strategy dates 与 benchmark dates 对齐：策略需要有相同日期
        # benchmark 在函数内部会 drop_nulls("benchmark_ret") -> len=39
        # 我们提供与 index_df 等长的收益，但第一行计算 benchmark_ret 时 shift 会丢弃
        # 所以策略也使用全部 40 日期，内部 join 后对齐
        all_rets_for_strat = np.concatenate([[0.0], bm_rets])  # 对应 index_df 的 40 个日期
        strategy_nav = pl.DataFrame(
            {
                "trade_date": dates,
                "net_return": all_rets_for_strat,
                "nav": np.cumprod(1 + all_rets_for_strat),
            }
        )

        result = compute_excess_return(strategy_nav, "000300.SH", "20260101", "20260209")

        # tracking_error 应接近 0，IR 应为 0.0
        self.assertAlmostEqual(result.tracking_error, 0.0, places=8)
        self.assertAlmostEqual(result.information_ratio, 0.0, places=8)

    @patch("factorzen.core.loader.fetch_index_daily")
    def test_summary_string(self, mock_fetch: unittest.mock.MagicMock) -> None:
        """summary() 返回非空字符串且包含基准名称。"""
        dates = self._dates(40)
        mock_fetch.return_value = _make_index_df(dates)
        strategy_nav = _make_strategy_nav(dates)

        result = compute_excess_return(strategy_nav, "000300.SH", "20260101", "20260209")

        summary = result.summary()
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        # benchmark_name for "000300.SH" is "HS300" per BENCHMARK_INDICES
        self.assertIn(result.benchmark_name, summary)

    @patch("factorzen.core.loader.fetch_index_daily")
    def test_raises_on_empty_index_data(self, mock_fetch: unittest.mock.MagicMock) -> None:
        """fetch_index_daily 返回空 DataFrame 时，函数应抛出 ValueError。"""
        dates = self._dates(40)
        mock_fetch.return_value = pl.DataFrame(
            {
                "trade_date": pl.Series([], dtype=pl.Date),
                "ts_code": pl.Series([], dtype=pl.Utf8),
                "close": pl.Series([], dtype=pl.Float64),
            }
        )
        strategy_nav = _make_strategy_nav(dates)

        with self.assertRaises(ValueError):
            compute_excess_return(strategy_nav, "000300.SH", "20260101", "20260209")


# ── pytest 入口（同时支持 unittest discover）─────────────────────────

if __name__ == "__main__":
    unittest.main()

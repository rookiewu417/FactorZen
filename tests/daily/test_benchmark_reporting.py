"""
test_benchmark.py：daily/evaluation/benchmark.py 的单元测试。
test_report_persistence.py：_report_persistence / _report_direction 的离线单测。
"""

from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from factorzen.daily.evaluation.benchmark import BenchmarkResult, compute_excess_return
from factorzen.pipelines import _report_direction as direction
from factorzen.pipelines import _report_persistence as persist

# ==== 来自 test_benchmark.py ====
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

# ==== 来自 test_report_persistence.py ====
@pytest.fixture
def results():
    """构造一组最小但字段完整的评价结果对象。"""
    from factorzen.daily.evaluation.backtest import StrategyBacktestResult
    from factorzen.daily.evaluation.ic_analysis import ICAnalysisResult
    from factorzen.daily.evaluation.turnover import TurnoverResult

    clean_df = pl.DataFrame(
        {"trade_date": [date(2024, 1, 2)], "ts_code": ["000001.SZ"], "factor_clean": [1.0]}
    )
    ic_result = ICAnalysisResult(
        factor_name="momentum_20d",
        ic_mean=0.01,
        ic_std=0.02,
        ir=0.5,
        ic_positive_ratio=1.0,
        n_periods=1,
        ic_series=pl.DataFrame({"trade_date": [date(2024, 1, 2)], "ic": [0.01]}),
    )
    returns = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2)],
            "gross_return": [0.0],
            "cost": [0.0],
            "borrow_cost": [0.0],
            "net_return": [0.0],
            "nav": [1.0],
            "cash_weight": [1.0],
            "turnover": [0.0],
        }
    )
    bt_result = StrategyBacktestResult(
        factor_name="momentum_20d",
        strategy_name="quantile_long_short",
        n_groups=5,
        returns=returns,
        nav=returns.select(
            ["trade_date", "gross_return", "cost", "borrow_cost", "net_return", "nav", "cash_weight"]
        ),
        positions=pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "ts_code": pl.Utf8,
                "weight": pl.Float64,
                "market_value": pl.Float64,
            }
        ),
        trades=pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "ts_code": pl.Utf8,
                "prev_weight": pl.Float64,
                "target_weight": pl.Float64,
                "filled_delta_weight": pl.Float64,
                "turnover": pl.Float64,
                "cost": pl.Float64,
                "block_reason": pl.Utf8,
            }
        ),
        summary_stats={"portfolio": {"sharpe": 0.0}},
        config={"max_abs_weight": 0.1},
    )
    to_result = TurnoverResult(
        factor_name="momentum_20d",
        avg_turnover=0.1,
        daily_turnover=pl.DataFrame({"trade_date": [date(2024, 1, 2)], "turnover": [0.1]}),
        migration_matrix=pl.DataFrame({"from": [0], "to": [1], "count": [1]}),
    )
    return clean_df, ic_result, bt_result, to_result


@pytest.fixture
def tmp_dirs(tmp_path, monkeypatch):
    """把 persist 的输出目录重定向到 tmp_path。"""
    monkeypatch.setattr(persist, "daily_factor_output_dir", lambda f: tmp_path / "factors")
    monkeypatch.setattr(persist, "daily_result_output_dir", lambda f: tmp_path / "results")
    monkeypatch.setattr(persist, "daily_report_output_dir", lambda f: tmp_path / "reports")
    return tmp_path


def _save(results, **kw):
    clean_df, ic_result, bt_result, to_result = results
    persist._save_results(
        "momentum_20d", "20240101", "20240131", clean_df, ic_result, bt_result, to_result, **kw
    )


# ── 往返：save → load ───────────────────────────────────────


def test_save_load_round_trip(tmp_dirs, results):
    _save(results)
    loaded = persist._load_results("momentum_20d", "20240101", "20240131")
    assert loaded is not None
    _clean, ic, bt, to = loaded
    assert ic.factor_name == "momentum_20d"
    assert ic.ic_mean == pytest.approx(0.01)
    assert ic.n_periods == 1
    assert bt.factor_name == "momentum_20d"
    assert bt.n_groups == 5
    assert to.avg_turnover == pytest.approx(0.1)


def test_load_results_missing_meta_returns_none(tmp_dirs):
    assert persist._load_results("momentum_20d", "20240101", "20240131") is None


def test_load_results_missing_parquet_returns_none(tmp_dirs, results):
    _save(results)
    # 删除其中一个必需 parquet → 退回重新计算（None）
    (tmp_dirs / "results" / "momentum_20d_20240101_20240131_ic.parquet").unlink()
    assert persist._load_results("momentum_20d", "20240101", "20240131") is None


# ── walk-forward / direction 回读 ───────────────────────────


def test_load_walk_forward_summary_round_trip(tmp_dirs, results):
    summary = {"status": "ok", "n_folds": 3, "oos_sharpe_mean": 0.7}
    _save(results, walk_forward_summary=summary)
    assert persist._load_walk_forward_summary("momentum_20d", "20240101", "20240131") == summary


def test_load_walk_forward_summary_missing_returns_none(tmp_dirs):
    assert persist._load_walk_forward_summary("x", "20240101", "20240131") is None


def test_load_backtest_direction_round_trip(tmp_dirs, results):
    decision = {"direction": "reversed", "should_reverse": True, "reason": "neg IC"}
    _save(results, backtest_direction=decision)
    loaded = direction._load_backtest_direction("momentum_20d", "20240101", "20240131")
    assert loaded["direction"] == "reversed"
    assert loaded["should_reverse"] is True


def test_load_backtest_direction_missing_returns_none(tmp_dirs):
    assert direction._load_backtest_direction("x", "20240101", "20240131") is None


# ── _existing_report_outputs / _save_quality_report ─────────


def test_existing_report_outputs_lists_present_files(tmp_dirs, results):
    _save(results)
    persist._save_quality_report("momentum_20d", "20240101", "20240131", {"status": "ok"})
    outputs = persist._existing_report_outputs("momentum_20d", "20240101", "20240131")
    assert "meta" in outputs
    assert "quality_report" in outputs
    assert outputs["meta"].endswith("_meta.json")


def test_existing_report_outputs_empty_when_nothing_saved(tmp_dirs):
    assert persist._existing_report_outputs("x", "20240101", "20240131") == {}


def test_save_quality_report_writes_json(tmp_dirs):
    import json

    path = persist._save_quality_report(
        "momentum_20d", "20240101", "20240131", {"status": "warning", "warnings": ["w"]}
    )
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["status"] == "warning"


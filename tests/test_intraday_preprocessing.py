"""测试 IntradayPreprocessingPipeline — fill_missing_bars 与 clip_outliers。"""

import polars as pl
import polars.testing as pl_testing

from intraday.preprocessing.pipeline import (
    IntradayPreprocessingPipeline,
    clip_outliers,
    fill_missing_bars,
)

# ── fill_missing_bars ───────────────────────────────────────────────────────


class TestFillMissingBars:
    """验证 forward-fill 缺失 bar 的行为。"""

    def test_fill_null_within_group(self):
        """同股票内 null 值被前一 bar 的 factor_value 填充。"""
        df = pl.DataFrame(
            {
                "trade_time": [
                    "2026-05-14 09:30:00",
                    "2026-05-14 09:31:00",
                    "2026-05-14 09:32:00",
                ],
                "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                "factor_value": [1.0, None, 3.0],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        result = fill_missing_bars(df)
        expected = [1.0, 1.0, 3.0]
        assert result["factor_value"].to_list() == expected

    def test_fill_cross_group_boundary(self):
        """forward-fill 不应跨股票。"""
        df = pl.DataFrame(
            {
                "trade_time": [
                    "2026-05-14 09:30:00",
                    "2026-05-14 09:31:00",
                    "2026-05-14 09:30:00",
                ],
                "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ"],
                "factor_value": [1.0, None, None],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        result = fill_missing_bars(df)
        values = result["factor_value"].to_list()
        assert values[0] == 1.0
        assert values[1] == 1.0
        assert values[2] is None

    def test_leading_null_remains_null(self):
        """股票第一个 bar 为 null 时，forward_fill 无法填充（无先序值）。"""
        df = pl.DataFrame(
            {
                "trade_time": [
                    "2026-05-14 09:30:00",
                    "2026-05-14 09:31:00",
                ],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "factor_value": [None, 2.0],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        result = fill_missing_bars(df)
        values = result["factor_value"].to_list()
        assert values[0] is None
        assert values[1] == 2.0

    def test_retains_other_columns(self):
        """填充操作不应丢弃原有列。"""
        df = pl.DataFrame(
            {
                "trade_time": [
                    "2026-05-14 09:30:00",
                    "2026-05-14 09:31:00",
                ],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "factor_value": [1.0, None],
                "volume": [100, 200],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        result = fill_missing_bars(df)
        assert "volume" in result.columns
        assert result["volume"].to_list() == [100, 200]

    def test_all_present_no_change(self):
        """无缺失值时 DataFrame 不变。"""
        df = pl.DataFrame(
            {
                "trade_time": [
                    "2026-05-14 09:30:00",
                    "2026-05-14 09:31:00",
                ],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "factor_value": [1.0, 2.0],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        result = fill_missing_bars(df)
        pl_testing.assert_frame_equal(result, df)

    def test_fill_missing_bars_does_not_cross_trading_day_boundary(self):
        """forward-fill 不应跨交易日。"""
        df = pl.DataFrame(
            {
                "trade_time": [
                    "2024-01-02 15:00:00",
                    "2024-01-03 09:30:00",
                    "2024-01-03 09:31:00",
                ],
                "ts_code": ["000001.SZ"] * 3,
                "factor_value": [1.5, None, 2.0],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        result = fill_missing_bars(df)

        assert result["factor_value"].to_list() == [1.5, None, 2.0]


# ── clip_outliers ───────────────────────────────────────────────────────────


class TestClipOutliers:
    """验证分位数截尾行为。"""

    def test_clip_both_ends(self):
        """上下同时截尾：超出分位数界的值被 clamp。"""
        df = pl.DataFrame(
            {
                "trade_time": ["2026-05-14 09:30:00"] * 5,
                "ts_code": [f"00000{i}.SZ" for i in range(1, 6)],
                "factor_value": [-100.0, 1.0, 2.0, 3.0, 200.0],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        result = clip_outliers(df, lower_pct=0.0, upper_pct=60.0)
        values = result["factor_value"].to_list()
        assert -100.0 in values
        assert 200.0 not in values
        assert all(v <= 200.0 for v in values)

    def test_clip_lower_only(self):
        """仅截取下界：上界 100% 不起作用。"""
        df = pl.DataFrame(
            {
                "factor_value": [-100.0, 1.0, 2.0, 3.0, 10.0],
            }
        )
        result = clip_outliers(df, lower_pct=20.0, upper_pct=100.0)
        clipped = sorted(result["factor_value"].to_list())
        assert clipped[-1] == 10.0

    def test_clip_upper_only(self):
        """仅截取上界：下界 0% 不起作用。"""
        df = pl.DataFrame(
            {
                "factor_value": [-100.0, 1.0, 2.0, 3.0, 10.0],
            }
        )
        result = clip_outliers(df, lower_pct=0.0, upper_pct=80.0)
        clipped = sorted(result["factor_value"].to_list())
        assert clipped[0] == -100.0

    def test_default_bounds_no_clip_on_normal_data(self):
        """默认 1%/99% 分位数：正常数据不应被截。"""
        df = pl.DataFrame(
            {
                "factor_value": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        result = clip_outliers(df)
        pl_testing.assert_frame_equal(result, df)

    def test_clip_preserves_other_columns(self):
        """截尾不应丢弃原有列。"""
        df = pl.DataFrame(
            {
                "trade_time": ["2026-05-14 09:30:00"] * 3,
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "factor_value": [-50.0, 2.0, 50.0],
                "volume": [100, 200, 300],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        result = clip_outliers(df, lower_pct=33.0, upper_pct=67.0)
        assert "volume" in result.columns
        assert result["volume"].to_list() == [100, 200, 300]

    def test_single_value_no_clip(self):
        """单一值不触发截尾。"""
        df = pl.DataFrame({"factor_value": [42.0]})
        result = clip_outliers(df)
        assert result["factor_value"][0] == 42.0


# ── IntradayPreprocessingPipeline ───────────────────────────────────────────


class TestIntradayPreprocessingPipeline:
    """验证预处理管线的构造、配置和 run() 行为。"""

    def test_default_config(self):
        """默认配置：fill_missing 和 clip_outliers 均开启。"""
        pipe = IntradayPreprocessingPipeline()
        assert pipe.do_fill_missing is True
        assert pipe.do_clip_outliers is True
        assert pipe.clip_lower_pct == 1.0
        assert pipe.clip_upper_pct == 99.0

    def test_custom_config(self):
        """自定义分位数参数正确存储。"""
        pipe = IntradayPreprocessingPipeline(
            do_fill_missing=False,
            clip_lower_pct=5.0,
            clip_upper_pct=95.0,
        )
        assert pipe.do_fill_missing is False
        assert pipe.clip_lower_pct == 5.0
        assert pipe.clip_upper_pct == 95.0

    def test_run_produces_factor_clean(self):
        """run() 必须产出 factor_clean 列。"""
        df = pl.DataFrame(
            {
                "trade_time": [
                    "2026-05-14 09:30:00",
                    "2026-05-14 09:31:00",
                ],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "factor_value": [1.0, 2.0],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        result = IntradayPreprocessingPipeline().run(df)
        assert "factor_clean" in result.columns
        assert result["factor_clean"].to_list() == [1.0, 2.0]

    def test_run_with_missing_and_outliers(self):
        """同时处理缺失和异常值。"""
        df = pl.DataFrame(
            {
                "trade_time": [
                    "2026-05-14 09:30:00",
                    "2026-05-14 09:31:00",
                    "2026-05-14 09:32:00",
                ],
                "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                "factor_value": [1.0, None, 100.0],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        pipe = IntradayPreprocessingPipeline(clip_lower_pct=0.0, clip_upper_pct=50.0)
        result = pipe.run(df)
        assert "factor_clean" in result.columns
        assert result["factor_clean"].to_list() == [1.0, 1.0, 1.0]

    def test_run_skip_fill(self):
        """do_fill_missing=False 时跳过填充。"""
        df = pl.DataFrame(
            {
                "trade_time": ["2026-05-14 09:30:00", "2026-05-14 09:31:00"],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "factor_value": [1.0, None],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        pipe = IntradayPreprocessingPipeline(do_fill_missing=False)
        result = pipe.run(df)
        assert result["factor_clean"].to_list() == [1.0, None]

    def test_run_skip_clip(self):
        """do_clip_outliers=False 时跳过截尾。"""
        df = pl.DataFrame(
            {
                "trade_time": ["2026-05-14 09:30:00"],
                "ts_code": ["000001.SZ"],
                "factor_value": [999.0],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        pipe = IntradayPreprocessingPipeline(do_clip_outliers=False)
        result = pipe.run(df)
        assert result["factor_clean"][0] == 999.0

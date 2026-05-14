"""Tear Sheet 报告引擎测试。

验证:
- HTML 输出非空、包含因子名
- 基本 HTML 结构完整
- 6 面板标题存在
- base64 图表嵌入
- None 输入不崩溃
- 导入和模板文件存在
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
import pytest

from daily.evaluation.ic_analysis import ICAnalysisResult
from daily.evaluation.backtest import BacktestResult
from daily.evaluation.turnover import TurnoverResult
from daily.evaluation.advanced import (
    ICDecayResult,
    MonotonicityResult,
    RankAutocorrResult,
    SectorICResult,
    SizeICResult,
)
from reporting.tear_sheet import generate_tear_sheet


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_dates(n: int = 60) -> list:
    return [f"2025-{(i // 20 + 1):02d}-{(i % 20 + 1):02d}" for i in range(n)]


@pytest.fixture
def ic_result() -> ICAnalysisResult:
    dates = _make_dates()
    n = len(dates)
    rng = np.random.default_rng(42)
    ics = rng.normal(0.03, 0.08, n).tolist()
    return ICAnalysisResult(
        factor_name="test_factor",
        ic_mean=float(np.mean(ics)),
        ic_std=float(np.std(ics, ddof=1)),
        ir=float(np.mean(ics)) / float(np.std(ics, ddof=1)),
        ic_positive_ratio=float(np.mean(np.array(ics) > 0)),
        n_periods=n,
        ic_series=pl.DataFrame({"trade_date": dates, "ic": ics}),
        decay={1: 0.032, 5: 0.025, 10: 0.018, 20: 0.010},
        frequency="daily",
    )


@pytest.fixture
def bt_result() -> BacktestResult:
    dates = _make_dates()
    n_groups = 10
    rng = np.random.default_rng(43)

    records, nav_records, ls_ret = [], [], []
    for d in dates:
        day_rets = {}
        for g in range(n_groups):
            ret = rng.normal(0.0002 * (g - 4.5), 0.015)
            records.append({"trade_date": d, "group": g, "ret": ret})
            day_rets[g] = ret

    daily_returns = pl.DataFrame(records)

    # NAV by group
    for g in range(n_groups):
        g_rets = [r["ret"] for r in records if r["group"] == g]
        cum = np.cumprod(1 + np.array(g_rets))
        for i, d in enumerate(dates):
            nav_records.append({"trade_date": d, "group": g, "nav": float(cum[i])})

    # Long-short
    for i, d in enumerate(dates):
        day_rets_i = {r["group"]: r["ret"] for r in records if r["trade_date"] == d}
        long_ret = day_rets_i.get(n_groups - 1, 0)
        short_ret = day_rets_i.get(0, 0)
        ls_r = long_ret - short_ret
        ls_ret.append(ls_r)
    ls_cum = np.cumprod(1 + np.array(ls_ret))
    long_short_nav = pl.DataFrame({
        "trade_date": dates, "ret": ls_ret, "nav": ls_cum,
    })

    # Summary stats
    summary_stats = {}
    for g in range(n_groups):
        grets = np.array([r["ret"] for r in records if r["group"] == g])
        cum = np.cumprod(1 + grets)
        summary_stats[g] = {
            "ann_ret": float(np.mean(grets) * 252),
            "ann_vol": float(np.std(grets) * np.sqrt(252)),
            "sharpe": float(np.mean(grets) * 252 / (np.std(grets) * np.sqrt(252) + 1e-9)),
            "max_dd": float(np.min(cum / np.maximum.accumulate(cum) - 1)),
        }
    ls_arr = np.array(ls_ret)
    ls_cum_arr = np.cumprod(1 + ls_arr)
    summary_stats["long_short"] = {
        "ann_ret": float(np.mean(ls_arr) * 252),
        "ann_vol": float(np.std(ls_arr) * np.sqrt(252)),
        "sharpe": float(np.mean(ls_arr) * 252 / (np.std(ls_arr) * np.sqrt(252) + 1e-9)),
        "max_dd": float(np.min(ls_cum_arr / np.maximum.accumulate(ls_cum_arr) - 1)),
    }

    return BacktestResult(
        factor_name="test_factor",
        n_groups=n_groups,
        daily_returns=daily_returns,
        nav=pl.DataFrame(nav_records),
        long_short_nav=long_short_nav,
        summary_stats=summary_stats,
        frequency="daily",
    )


@pytest.fixture
def to_result() -> TurnoverResult:
    dates = _make_dates()
    n = len(dates)
    rng = np.random.default_rng(44)
    turnover_vals = rng.uniform(0.15, 0.35, n).tolist()
    return TurnoverResult(
        factor_name="test_factor",
        avg_turnover=float(np.mean(turnover_vals)),
        migration_matrix=pl.DataFrame(),
        daily_turnover=pl.DataFrame({"trade_date": dates, "turnover": turnover_vals}),
        frequency="daily",
    )


@pytest.fixture
def advanced_results() -> dict:
    return {
        "decay_results": [
            ICDecayResult(horizon=1, ic_mean=0.032, ic_std=0.08),
            ICDecayResult(horizon=5, ic_mean=0.025, ic_std=0.07),
            ICDecayResult(horizon=20, ic_mean=0.010, ic_std=0.06),
        ],
        "mono": MonotonicityResult(
            factor_name="test_factor",
            monotonicity_score=0.85,
            group_means=[-0.002, -0.001, 0.001, 0.003],
            direction="positive",
            ols_slope=0.0012,
        ),
        "autocorr": RankAutocorrResult(
            factor_name="test_factor",
            autocorr_values=[0.65],
            mean_autocorr=0.65,
            half_life_est=1.6,
            _lag_to_autocorr={1: 0.65},
        ),
        "sector": SectorICResult(
            factor_name="test_factor",
            sector_ic_df=pl.DataFrame({
                "sector": ["fin", "tech", "cons"],
                "ic": [0.028, 0.035, 0.022],
            }),
        ),
        "size": SizeICResult(
            factor_name="test_factor",
            buckets={"Large": 0.030, "Mid": 0.033, "Small": 0.025},
        ),
    }


# ── Tests: generate_tear_sheet ────────────────────────────────────────

class TestGenerateTearSheet:
    def test_basic_generation(self, ic_result, bt_result, to_result):
        """Smoke test: 生成 HTML 无错误。"""
        html = generate_tear_sheet(
            "momentum_20d", ic_result, bt_result, to_result,
            frequency="daily",
            date_range="2025-01-01 ~ 2025-05-13",
        )
        assert isinstance(html, str)
        assert len(html) > 1000

    def test_html_contains_key_elements(self, ic_result, bt_result, to_result):
        """HTML 包含预期的结构元素。"""
        html = generate_tear_sheet("momentum_20d", ic_result, bt_result, to_result)
        assert "<!DOCTYPE html>" in html
        assert "momentum_20d" in html
        assert "Overview" in html
        assert "Returns Analysis" in html
        assert "IC Analysis" in html
        assert "Turnover Analysis" in html
        assert "Risk Attribution" in html
        assert "Summary" in html
        assert "</html>" in html


    def test_generate_html_contains_factor_name(self, ic_result, bt_result, to_result):
        """生成 HTML 包含因子名称。"""
        html = generate_tear_sheet("momentum_20d", ic_result, bt_result, to_result)
        assert "<html" in html
        assert "momentum_20d" in html

    def test_html_size_under_5mb(self, ic_result, bt_result, to_result):
        """生成 HTML 小于 5MB。"""
        html = generate_tear_sheet("momentum_20d", ic_result, bt_result, to_result)
        size_bytes = len(html.encode("utf-8"))
        assert size_bytes < 5 * 1024 * 1024, f"HTML size {size_bytes} exceeds 5MB"
    def test_html_contains_chart_base64(self, ic_result, bt_result, to_result):
        """图表以 base64 嵌入 HTML。"""
        html = generate_tear_sheet(
            "test_factor", ic_result, bt_result, to_result,
            date_range="2025-01-01 ~ 2025-05-13",
        )
        assert "data:image/png;base64," in html

    def test_none_backtest(self, ic_result, to_result):
        """None backtest 优雅处理。"""
        html = generate_tear_sheet(
            "test_factor", ic_result, None, to_result,
            date_range="2025-01-01 ~ 2025-05-13",
        )
        assert isinstance(html, str)
        assert "No backtest data" in html

    def test_none_turnover(self, ic_result, bt_result):
        """None turnover 优雅处理。"""
        html = generate_tear_sheet(
            "test_factor", ic_result, bt_result, None,
            date_range="2025-01-01 ~ 2025-05-13",
        )
        assert isinstance(html, str)
        assert "No turnover data" in html

    def test_all_none_results(self):
        """全部 None 仍生成有效 HTML。"""
        html = generate_tear_sheet(
            "empty_factor", None, None, None,
            date_range="2025-01-01 ~ 2025-01-02",
        )
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "empty_factor" in html

    def test_summary_has_stars(self, ic_result, bt_result, to_result):
        """Summary 面板包含星级评级。"""
        html = generate_tear_sheet(
            "test_factor", ic_result, bt_result, to_result,
            date_range="2025-01-01 ~ 2025-05-13",
        )
        assert chr(9733) in html  # ★

    def test_different_frequency(self, ic_result, bt_result, to_result):
        """不同频率标签正确渲染。"""
        for freq in ["daily", "weekly", "monthly"]:
            html = generate_tear_sheet(
                f"test_{freq}", ic_result, bt_result, to_result,
                frequency=freq, date_range="2025-01-01 ~ 2025-05-13",
            )
            assert freq in html

    def test_different_factor_names(self, ic_result, bt_result, to_result):
        """多种因子名称正确渲染。"""
        names = ["momentum_20d", "value_ep", "My_Custom_Factor"]
        for name in names:
            html = generate_tear_sheet(
                name, ic_result, bt_result, to_result,
                date_range="2025-01-01 ~ 2025-05-13",
            )
            assert name in html

    def test_with_advanced_results(self, ic_result, bt_result, to_result, advanced_results):
        """高级评价结果被包含进报告。"""
        html = generate_tear_sheet(
            "momentum_20d", ic_result, bt_result, to_result,
            advanced_results=advanced_results,
            date_range="2025-01-01 ~ 2025-05-13",
        )
        assert "Monotonicity" in html or "monotonicity" in html.lower()

    def test_html_non_empty(self, ic_result, bt_result, to_result):
        """报告 HTML 应非空。"""
        html = generate_tear_sheet("test_factor", ic_result, bt_result, to_result)
        assert html is not None
        assert len(html) > 0


class TestTearSheetImports:
    """模块导入和模板文件测试。"""

    def test_tear_sheet_import(self):
        from reporting.tear_sheet import generate_tear_sheet
        assert callable(generate_tear_sheet)

    def test_template_dir_exists(self):
        template_dir = Path(__file__).resolve().parent.parent / "reporting" / "templates"
        assert template_dir.is_dir()

    def test_template_file_exists(self):
        template_file = Path(__file__).resolve().parent.parent / "reporting" / "templates" / "tear_sheet.html"
        assert template_file.is_file()

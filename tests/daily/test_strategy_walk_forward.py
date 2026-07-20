"""
test_strategy_registry.py：Tests for pluggable backtest strategy construction.
test_walk_forward_strategy.py：策略级 Walk-Forward 验证测试。
test_walk_forward.py：S3 防回归：验证 walk-forward IC 交叉验证。
test_rebalance_threshold.py：rebalance_threshold 功能测试：换手率低于阈值时跳过调仓。
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from textwrap import dedent

import numpy as np
import polars as pl
import pytest

from factorzen.daily.evaluation.backtest import (
    BacktestConfig,
    TopNLongOnlyStrategy,
    run_strategy_backtest,
)
from factorzen.daily.evaluation.ic_analysis import _compute_walk_forward_ic
from factorzen.daily.evaluation.walk_forward import (
    WalkForwardFoldResult,
    WalkForwardResult,
    WalkForwardSplitter,
    _compute_oos_max_dd,
    run_walk_forward,
)


# ==== 来自 test_strategy_registry.py ====
def test_builtin_strategy_registry_builds_supported_strategies():
    from factorzen.daily.evaluation.backtest import (
        FactorWeightedStrategy,
        OptimizerStrategy,
        QuantileLongShortStrategy,
        TopNLongOnlyStrategy,
    )
    from factorzen.daily.evaluation.strategy_registry import build_strategy

    topn = build_strategy("topn_long_only", {"top_n": 12})
    quantile = build_strategy("quantile_long_short", {"quantiles": 4})
    weighted = build_strategy(
        "factor_weighted",
        {"long_only": True, "gross_exposure": 1.0, "long_exposure": 0.8},
    )

    assert isinstance(topn, TopNLongOnlyStrategy)
    assert topn.n == 12
    assert isinstance(quantile, QuantileLongShortStrategy)
    assert quantile.n_groups == 4
    assert isinstance(weighted, FactorWeightedStrategy)
    assert weighted.long_only is True
    assert weighted.long_exposure == 0.8

    optimizer = build_strategy(
        "optimizer_strategy",
        {
            "optimizer": "mean_variance",
            "risk_aversion": 2.0,
            "lookback_days": 40,
            "cov_estimator": "ledoit_wolf",
            "long_only": True,
            "top_n": 80,
            "max_weight": 0.08,
            "gross_exposure": 1.0,
            "net_exposure": 1.0,
        },
    )

    assert isinstance(optimizer, OptimizerStrategy)
    assert optimizer.lookback_days == 40
    assert optimizer.cov_estimator == "ledoit_wolf"
    assert optimizer.long_only is True
    assert optimizer.top_n == 80
    assert optimizer.constraints.max_weight == 0.08


def test_strategy_registry_imports_custom_strategy_from_dotted_path(tmp_path, monkeypatch):
    module_path = tmp_path / "custom_strategy.py"
    module_path.write_text(
        dedent(
            """
            import polars as pl

            from factorzen.daily.evaluation.backtest import Strategy


            class CustomStrategy(Strategy):
                name = "custom"

                def __init__(self, multiplier: int) -> None:
                    self.multiplier = multiplier

                @classmethod
                def from_config(cls, config):
                    return cls(multiplier=config["multiplier"])

                def generate_weights(self, context):
                    return pl.DataFrame(
                        {"ts_code": [], "target_weight": []},
                        schema={"ts_code": pl.Utf8, "target_weight": pl.Float64},
                    )
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("custom_strategy", None)

    from factorzen.daily.evaluation.strategy_registry import build_strategy

    strategy = build_strategy("custom_strategy.CustomStrategy", {"multiplier": 3})

    assert strategy.name == "custom"
    assert strategy.multiplier == 3

# ==== 来自 test_walk_forward_strategy.py ====
# ── 测试夹具 ─────────────────────────────────────────────────────────────────


@pytest.fixture
def factor_df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n_dates, n_stocks = 300, 30
    start = date(2022, 1, 3)
    dates = [(start + timedelta(days=i)).isoformat() for i in range(n_dates)]
    records = []
    for d in dates:
        for s in range(n_stocks):
            records.append(
                {
                    "trade_date": d,
                    "ts_code": f"{s:06d}.SZ",
                    "factor_clean": float(rng.normal()),
                }
            )
    return pl.DataFrame(records)


@pytest.fixture
def price_df(factor_df: pl.DataFrame) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    codes = factor_df["ts_code"].unique().to_list()
    dates = factor_df["trade_date"].unique().sort().to_list()
    records = []
    for code in codes:
        price = 10.0
        for d in dates:
            price *= 1 + rng.normal(0.0005, 0.02)
            records.append(
                {
                    "trade_date": d,
                    "ts_code": code,
                    "close": price,
                    "open": price * 0.998,
                }
            )
    return pl.DataFrame(records)


# ── TestWalkForwardSplitter ──────────────────────────────────────────────────


class TestWalkForwardSplitter:
    def test_n_splits_formula(self):
        """n_splits(total_days) 应与 split(dates) 实际返回折数一致。"""
        splitter = WalkForwardSplitter(
            train_days=100, test_days=30, step_days=30, embargo_days=5
        )
        total_days = 250
        dates = [f"day_{i}" for i in range(total_days)]
        actual = len(splitter.split(dates))
        estimated = splitter.n_splits(total_days)
        assert estimated == actual, (
            f"n_splits({total_days})={estimated} 与 split 实际折数={actual} 不一致"
        )

    def test_embargo_prevents_leakage(self):
        """每折历史观察期末尾索引 + embargo_days <= 未来验证期首索引。"""
        splitter = WalkForwardSplitter(
            train_days=100, test_days=30, step_days=30, embargo_days=5
        )
        total_days = 250
        dates = list(range(total_days))
        folds = splitter.split(dates)
        assert len(folds) > 0, "应有至少一折"
        for train_dates, test_dates in folds:
            # 找最后一个历史观察日在 dates 中的索引
            train_end_val = train_dates[-1]
            test_start_val = test_dates[0]
            train_end_idx = dates.index(train_end_val)
            test_start_idx = dates.index(test_start_val)
            assert test_start_idx - train_end_idx >= splitter.embargo_days, (
                f"embargo 不足: train_end_idx={train_end_idx}, "
                f"test_start_idx={test_start_idx}"
            )

    def test_empty_when_too_short(self):
        """总日数不足时返回空列表，不崩溃。"""
        splitter = WalkForwardSplitter(
            train_days=200, test_days=50, step_days=50, embargo_days=10
        )
        # total_days < train_days + embargo_days + test_days
        dates = [f"day_{i}" for i in range(100)]
        result = splitter.split(dates)
        assert result == []

    def test_train_always_from_zero(self):
        """展开窗口：每折历史观察期从 dates[0] 开始。"""
        splitter = WalkForwardSplitter(
            train_days=80, test_days=20, step_days=20, embargo_days=5
        )
        dates = [f"day_{i}" for i in range(200)]
        folds = splitter.split(dates)
        assert len(folds) > 0
        for train_dates, _test_dates in folds:
            assert train_dates[0] == dates[0], (
                "展开窗口：历史观察期第一个日期应始终为 dates[0]"
            )


# ── TestRunWalkForward ───────────────────────────────────────────────────────


def test_oos_max_drawdown_includes_initial_nav():
    assert _compute_oos_max_dd([0.90]) == pytest.approx(-0.10)


class TestRunWalkForward:
    def _make_splitter(self) -> WalkForwardSplitter:
        return WalkForwardSplitter(
            train_days=100, test_days=30, step_days=30, embargo_days=5
        )

    def _strategy_factory(self, params: dict) -> object:
        from factorzen.daily.evaluation.backtest import QuantileLongShortStrategy

        return QuantileLongShortStrategy(n_groups=params.get("n_groups", 5))

    def test_oos_nav_starts_at_one(self, factor_df: pl.DataFrame, price_df: pl.DataFrame):
        """OOS 拼接净值序列的第一个值应接近 1.0（从初始净值 1.0 开始乘以 (1+ret)）。"""
        splitter = self._make_splitter()
        result = run_walk_forward(
            strategy_factory=self._strategy_factory,
            factor_df=factor_df,
            price_df=price_df,
            splitter=splitter,
            params={"n_groups": 5},
        )
        assert len(result.folds) > 0, "Expected at least one WF fold but got 0 — fixture may be too small"
        if result.folds and not result.oos_returns.is_empty():
            first_nav = result.oos_returns.sort("trade_date")["nav"][0]
            # 第一个 nav 应等于 1 + first_net_return，不必精确为 1.0
            first_ret = result.oos_returns.sort("trade_date")["net_return"][0]
            expected = 1.0 * (1.0 + first_ret)
            assert abs(float(first_nav) - float(expected)) < 1e-9

    def test_oos_returns_no_gaps(self, factor_df: pl.DataFrame, price_df: pl.DataFrame):
        """OOS 日期在不同折之间不应重叠（每个日期至多出现一次）。"""
        splitter = self._make_splitter()
        result = run_walk_forward(
            strategy_factory=self._strategy_factory,
            factor_df=factor_df,
            price_df=price_df,
            splitter=splitter,
            params={"n_groups": 5},
        )
        assert len(result.folds) > 0, "Expected at least one WF fold but got 0 — fixture may be too small"
        if result.oos_returns.is_empty():
            return
        dates = result.oos_returns["trade_date"].to_list()
        assert len(dates) == len(set(dates)), "OOS 日期存在重叠（跨折）"

    def test_stability_ratio_bounds(self, factor_df: pl.DataFrame, price_df: pl.DataFrame):
        """stability_ratio 不应为 NaN 或 Inf，即使 IS Sharpe 接近 0。"""
        splitter = self._make_splitter()
        result = run_walk_forward(
            strategy_factory=self._strategy_factory,
            factor_df=factor_df,
            price_df=price_df,
            splitter=splitter,
            params={"n_groups": 5},
        )
        assert np.isfinite(result.stability_ratio), (
            f"stability_ratio 应为有限数，got {result.stability_ratio}"
        )

    def test_result_structure(self, factor_df: pl.DataFrame, price_df: pl.DataFrame):
        """WalkForwardResult 应包含所有必需字段且类型正确。"""
        splitter = self._make_splitter()
        result = run_walk_forward(
            strategy_factory=self._strategy_factory,
            factor_df=factor_df,
            price_df=price_df,
            splitter=splitter,
            params={"n_groups": 5},
        )
        assert len(result.folds) > 0, "Expected at least one WF fold but got 0 — fixture may be too small"
        assert isinstance(result, WalkForwardResult)
        assert isinstance(result.folds, list)
        assert isinstance(result.oos_returns, pl.DataFrame)
        assert isinstance(result.is_sharpe_mean, float)
        assert isinstance(result.oos_sharpe_mean, float)
        assert isinstance(result.oos_sharpe_std, float)
        assert isinstance(result.oos_max_dd, float)
        assert isinstance(result.stability_ratio, float)

        # 检查必须列
        if not result.oos_returns.is_empty():
            cols = set(result.oos_returns.columns)
            assert "trade_date" in cols
            assert "net_return" in cols
            assert "fold_id" in cols
            assert "nav" in cols

        # 每折都是 WalkForwardFoldResult
        for fold in result.folds:
            assert isinstance(fold, WalkForwardFoldResult)
            assert isinstance(fold.fold_id, int)
            assert isinstance(fold.is_sharpe, float)
            assert isinstance(fold.oos_sharpe, float)
            assert isinstance(fold.oos_ann_ret, float)
            assert isinstance(fold.oos_max_dd, float)
            assert isinstance(fold.params, dict)


def _wf_strategy_factory(params: dict) -> object:
    from factorzen.daily.evaluation.backtest import QuantileLongShortStrategy

    return QuantileLongShortStrategy(n_groups=params.get("n_groups", 5))


def _fake_backtest_result() -> object:
    from types import SimpleNamespace

    return SimpleNamespace(
        summary_stats={"portfolio": {"sharpe": 0.1, "ann_ret": 0.0, "max_dd": 0.0}},
        returns=pl.DataFrame(schema={"trade_date": pl.Date, "net_return": pl.Float64}),
    )


def test_run_walk_forward_passes_is_st_by_date_to_backtest(
    factor_df: pl.DataFrame, price_df: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ST涨跌停容差接线：run_walk_forward 应基于 price_df 的 codes/trade_dates
    只构建一次 is_st_by_date，并传给 IS、OOS 两处 run_strategy_backtest 调用。
    """
    import factorzen.daily.evaluation.walk_forward as wf_mod

    calls: list[dict] = []

    def _fake_run_strategy_backtest(strategy, factor, price, cfg=None, **kwargs):
        calls.append(kwargs)
        return _fake_backtest_result()

    sentinel = {date(2022, 1, 3): {"000000.SZ"}}
    monkeypatch.setattr(wf_mod, "run_strategy_backtest", _fake_run_strategy_backtest)
    monkeypatch.setattr(wf_mod, "build_is_st_by_date", lambda codes, dates: sentinel)

    splitter = WalkForwardSplitter(train_days=100, test_days=30, step_days=30, embargo_days=5)
    run_walk_forward(
        strategy_factory=_wf_strategy_factory,
        factor_df=factor_df,
        price_df=price_df,
        splitter=splitter,
        params={"n_groups": 5},
    )

    assert calls, "run_strategy_backtest 应至少被调用一次（IS + OOS）"
    assert all(c.get("is_st_by_date") == sentinel for c in calls), (
        f"IS/OOS 调用都应收到相同的 is_st_by_date，实际: {[c.get('is_st_by_date') for c in calls]}"
    )


def test_run_walk_forward_search_passes_is_st_by_date_to_all_backtest_calls(
    factor_df: pl.DataFrame, price_df: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ST涨跌停容差接线：run_walk_forward_search 的三处 run_strategy_backtest
    调用（IS 全量缓存 / 逐折 IS 搜索 / OOS）都应收到同一份 is_st_by_date，
    只构建一次、不逐折/逐候选重复构建。reuse_is_backtests 的 True/False 两条
    分支各自覆盖不同的 IS 调用位置，OOS 调用位置两条分支都会覆盖。
    """
    import factorzen.daily.evaluation.walk_forward as wf_mod
    from factorzen.daily.evaluation.walk_forward import run_walk_forward_search

    calls: list[dict] = []

    def _fake_run_strategy_backtest(strategy, factor, price, cfg=None, **kwargs):
        calls.append(kwargs)
        return _fake_backtest_result()

    sentinel = {date(2022, 1, 3): {"000000.SZ"}}
    monkeypatch.setattr(wf_mod, "run_strategy_backtest", _fake_run_strategy_backtest)
    monkeypatch.setattr(wf_mod, "build_is_st_by_date", lambda codes, dates: sentinel)

    splitter = WalkForwardSplitter(train_days=100, test_days=30, step_days=30, embargo_days=5)

    for reuse in (False, True):
        calls.clear()
        run_walk_forward_search(
            strategy_factory=_wf_strategy_factory,
            factor_df=factor_df,
            price_df=price_df,
            splitter=splitter,
            param_candidates=[{"n_groups": 5}],
            reuse_is_backtests=reuse,
            parallel_workers=1,
        )
        assert calls, f"reuse_is_backtests={reuse} 时 run_strategy_backtest 应至少被调用一次"
        assert all(c.get("is_st_by_date") == sentinel for c in calls), (
            f"reuse_is_backtests={reuse} 时全部调用都应收到相同的 is_st_by_date，"
            f"实际: {[c.get('is_st_by_date') for c in calls]}"
        )

# ==== 来自 test_walk_forward.py ====
class TestWalkForwardIC:
    def test_returns_list_of_dicts(self):
        """返回值应为 list of dict，每个 dict 含 fold / train_ic / test_ic。"""
        ic = np.random.default_rng(0).normal(0.03, 0.08, 200)
        result = _compute_walk_forward_ic(ic, n_folds=5, embargo=5)
        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert "fold" in item
            assert "train_ic" in item
            assert "test_ic" in item

    def test_fold_count(self):
        """足够长的序列应返回至少 2 个、至多 n_folds 个结果（末折可能因数据不足跳过）。"""
        ic = np.random.default_rng(1).normal(0.02, 0.07, 300)
        result = _compute_walk_forward_ic(ic, n_folds=5, embargo=5)
        assert 2 <= len(result) <= 5

    def test_train_set_grows_over_folds(self):
        """每折的 train_ic 基于越来越长的历史（expanding window），fold 编号递增。"""
        ic = np.random.default_rng(2).normal(0.03, 0.08, 250)
        result = _compute_walk_forward_ic(ic, n_folds=5, embargo=5)
        folds = [r["fold"] for r in result]
        assert folds == sorted(folds), "fold 编号应递增"

    def test_too_short_returns_empty(self):
        """样本过少时返回空列表，不崩溃。"""
        ic = np.array([0.03, 0.02, 0.05])
        result = _compute_walk_forward_ic(ic, n_folds=5, embargo=5)
        assert result == []

    def test_embargo_prevents_leakage(self):
        """embargo > 0 时，test 序列开头与 train 末尾之间有间隔。"""
        # 构造一个特定序列：前半段全正，后半段全负，embargo=10
        ic = np.concatenate([np.ones(50) * 0.05, np.ones(50) * (-0.05)])
        result = _compute_walk_forward_ic(ic, n_folds=2, embargo=10)
        # 验证 test 的第一折 IC < train IC（后半段 IC 为负）
        if result:
            assert result[-1]["test_ic"] < result[-1]["train_ic"]

    def test_finite_values(self):
        """所有返回值应为有限浮点数。"""
        ic = np.random.default_rng(3).normal(0.02, 0.06, 200)
        result = _compute_walk_forward_ic(ic, n_folds=5, embargo=5)
        for r in result:
            for key in ("train_ic", "test_ic"):
                assert np.isfinite(r[key]), f"fold {r['fold']} {key}={r[key]} 含非有限值"

    def test_integrated_in_compute_rank_ic(self):
        """compute_rank_ic 返回的 ICAnalysisResult 包含 walk_forward_ic 字段。"""
        import numpy as np
        import polars as pl

        from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns, compute_rank_ic

        rng = np.random.default_rng(42)
        n_dates, n_stocks = 120, 50
        dates = [f"2024-{(i // 25 + 1):02d}-{(i % 25 + 1):02d}" for i in range(n_dates)]
        stocks = [f"{i:06d}.SZ" for i in range(n_stocks)]

        factor_rows, price_rows = [], []
        for d in dates:
            fv = rng.standard_normal(n_stocks)
            rets = rng.normal(0, 0.02, n_stocks)
            for i, s in enumerate(stocks):
                factor_rows.append({"trade_date": d, "ts_code": s, "factor_clean": float(fv[i])})
                price_rows.append({"trade_date": d, "ts_code": s, "ret": float(rets[i])})

        factor_df = pl.DataFrame(factor_rows).with_columns(
            pl.col("trade_date").str.strptime(pl.Date, "%Y-%m-%d")
        )
        price_df = pl.DataFrame(price_rows).with_columns(
            pl.col("trade_date").str.strptime(pl.Date, "%Y-%m-%d")
        )
        ret_df = compute_fwd_returns(price_df, horizons=[1, 5], ret_col="ret")

        result = compute_rank_ic(factor_df, ret_df, horizons=[1, 5])
        assert hasattr(result, "walk_forward_ic")
        assert isinstance(result.walk_forward_ic, list)

# ==== 来自 test_rebalance_threshold.py ====
# ──────────────────────────────────────────────────────────
# 测试夹具
# ──────────────────────────────────────────────────────────


def _make_fixtures(
    n_days: int = 40,
    n_stocks: int = 20,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """构造最小因子+价格数据（无依赖外部存储）。"""
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 3)
    dates = [(start + timedelta(days=i)).isoformat() for i in range(n_days)]

    factor_rows = []
    price_rows = []
    last_close = {f"00{s:04d}.SZ": 10.0 + s for s in range(n_stocks)}

    for d in dates:
        for s in range(n_stocks):
            ts = f"00{s:04d}.SZ"
            factor_rows.append({
                "trade_date": d,
                "ts_code": ts,
                "factor_clean": float(rng.standard_normal()),
            })
            open_price = last_close[ts]
            close_price = open_price * (1.0 + float(rng.uniform(-0.05, 0.05)))
            price_rows.append({
                "trade_date": d,
                "ts_code": ts,
                "open": open_price,
                "close": close_price,
                "pre_close": last_close[ts],
                "pct_chg": (close_price / last_close[ts] - 1.0) * 100,
                "vol": float(rng.uniform(1e6, 1e8)),
                "amount": float(rng.uniform(1e7, 1e9)),
            })
            last_close[ts] = close_price

    return pl.DataFrame(factor_rows), pl.DataFrame(price_rows)


# ──────────────────────────────────────────────────────────
# BacktestConfig rebalance_threshold 字段测试
# ──────────────────────────────────────────────────────────


def test_backtest_config_has_rebalance_threshold():
    """BacktestConfig 包含 rebalance_threshold 字段，默认 None。"""
    cfg = BacktestConfig()
    assert hasattr(cfg, "rebalance_threshold")
    assert cfg.rebalance_threshold is None


def test_backtest_config_rebalance_threshold_custom():
    """BacktestConfig 可自定义 rebalance_threshold。"""
    cfg = BacktestConfig(rebalance_threshold=0.5)
    assert cfg.rebalance_threshold == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────
# 高阈值 → 换手率应降低（几乎不调仓）
# ──────────────────────────────────────────────────────────


def test_high_threshold_reduces_turnover():
    """rebalance_threshold 很大时，几乎每期都跳过调仓，换手率应显著低于无阈值。"""
    factor_df, price_df = _make_fixtures()
    strategy = TopNLongOnlyStrategy(n=5)

    cfg_no_threshold = BacktestConfig(
        rebalance_threshold=None,
        max_participation_rate=1.0,
    )
    cfg_high_threshold = BacktestConfig(
        rebalance_threshold=100.0,  # 极大阈值，几乎永远不触发调仓
        max_participation_rate=1.0,
    )

    result_no = run_strategy_backtest(strategy, factor_df, price_df, cfg_no_threshold)
    result_high = run_strategy_backtest(strategy, factor_df, price_df, cfg_high_threshold)

    turnover_no = result_no.summary_stats["portfolio"]["avg_turnover"]
    turnover_high = result_high.summary_stats["portfolio"]["avg_turnover"]

    assert turnover_high <= turnover_no + 1e-6, (
        f"高阈值换手率 {turnover_high:.4f} 应 ≤ 无阈值换手率 {turnover_no:.4f}"
    )


def test_zero_threshold_matches_no_threshold():
    """rebalance_threshold=0 时，每期换手率 > 0 → 永不跳过，结果应与 None 完全相同。"""
    factor_df, price_df = _make_fixtures()
    strategy = TopNLongOnlyStrategy(n=5)

    cfg_none = BacktestConfig(rebalance_threshold=None, max_participation_rate=1.0)
    cfg_zero = BacktestConfig(rebalance_threshold=0.0, max_participation_rate=1.0)

    result_none = run_strategy_backtest(strategy, factor_df, price_df, cfg_none)
    result_zero = run_strategy_backtest(strategy, factor_df, price_df, cfg_zero)

    nav_none = result_none.nav["nav"].to_list()
    nav_zero = result_zero.nav["nav"].to_list()

    assert len(nav_none) == len(nav_zero)
    for a, b in zip(nav_none, nav_zero, strict=True):
        assert abs(a - b) < 1e-10, f"threshold=0 与 threshold=None 结果应一致: {a} vs {b}"


def test_result_structure_with_threshold():
    """带 rebalance_threshold 的回测结果结构完整。"""
    from factorzen.daily.evaluation.backtest import StrategyBacktestResult

    factor_df, price_df = _make_fixtures()
    strategy = TopNLongOnlyStrategy(n=5)
    cfg = BacktestConfig(rebalance_threshold=0.3, max_participation_rate=1.0)

    result = run_strategy_backtest(strategy, factor_df, price_df, cfg)

    assert isinstance(result, StrategyBacktestResult)
    assert "net_return" in result.returns.columns
    assert "nav" in result.nav.columns
    assert "portfolio" in result.summary_stats


def test_moderate_threshold_reduces_turnover():
    """适中阈值（0.3）时换手率不超过无阈值版本。"""
    factor_df, price_df = _make_fixtures()
    strategy = TopNLongOnlyStrategy(n=5)

    cfg_no = BacktestConfig(rebalance_threshold=None, max_participation_rate=1.0)
    cfg_mod = BacktestConfig(rebalance_threshold=0.3, max_participation_rate=1.0)

    result_no = run_strategy_backtest(strategy, factor_df, price_df, cfg_no)
    result_mod = run_strategy_backtest(strategy, factor_df, price_df, cfg_mod)

    to_no = result_no.summary_stats["portfolio"]["avg_turnover"]
    to_mod = result_mod.summary_stats["portfolio"]["avg_turnover"]

    # 有阈值时换手率应 ≤ 无阈值
    assert to_mod <= to_no + 1e-6, f"moderate threshold turnover {to_mod:.4f} > no threshold {to_no:.4f}"


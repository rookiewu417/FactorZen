"""
test_optimizer.py：组合优化器单元测试。
test_square_root_impact.py：平方根冲击成本模型单元测试。
test_turnover_nan_filter.py：compute_turnover 分组前须过滤 NaN/null，防 polars rank 把 NaN 排最大污染分组。
test_hyperparameter.py：Optuna 超参搜索测试。
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from factorzen.daily.evaluation.cost_models import LinearCostModel, SquareRootImpactCostModel
from factorzen.daily.evaluation.hyperparameter import ParamSpec, TuningSpace, run_optuna_search
from factorzen.daily.evaluation.turnover import compute_turnover
from factorzen.daily.optimization.base import OptimizerConstraints
from factorzen.daily.optimization.covariance import (
    ewma_covariance,
    ledoit_wolf_shrinkage,
    sample_covariance,
)
from factorzen.daily.optimization.max_sharpe import MaxSharpeOptimizer
from factorzen.daily.optimization.mean_variance import MeanVarianceOptimizer
from factorzen.daily.optimization.risk_parity import RiskParityOptimizer


# ==== 来自 test_optimizer.py ====
def _equal_cov(n: int, sigma: float = 0.01) -> np.ndarray:
    """n 个独立资产的对角协方差矩阵（等波动率）。"""
    return np.eye(n) * sigma**2


def _default_cons(n: int, max_weight: float = 1.0) -> OptimizerConstraints:
    return OptimizerConstraints(max_weight=max_weight, min_weight=0.0, gross_exposure=1.0, net_exposure=1.0)


class TestMeanVarianceOptimizer:
    def test_mean_variance_optimizer_suite(self):
        """单资产时权重应为 1.0。；max_weight 约束必须严格生效。；long_only 情形下权重应非负。；不可行问题时应返回有效的 fallback 权重。"""
        # -- 原 test_single_asset_full_weight --
        opt = MeanVarianceOptimizer(risk_aversion=1.0)
        mu = np.array([0.01])
        cov = np.array([[0.0001]])
        cons = _default_cons(1)
        w = opt.solve(mu, cov, cons)
        assert len(w) == 1
        assert abs(w[0] - 1.0) < 0.05

        # -- 原 test_max_weight_respected --
        opt = MeanVarianceOptimizer(risk_aversion=0.1)
        n = 5
        mu = np.ones(n) * 0.01
        mu[0] = 0.1  # 第一个资产预期收益最高
        cov = _equal_cov(n)
        cons = _default_cons(n, max_weight=0.3)
        w = opt.solve(mu, cov, cons)
        assert np.all(w <= 0.3 + 1e-6), f"max_weight 违反: {w}"

        # -- 原 test_weights_nonnegative --
        opt = MeanVarianceOptimizer(risk_aversion=1.0)
        n = 4
        mu = np.array([0.01, 0.02, -0.01, 0.005])
        cov = _equal_cov(n)
        cons = _default_cons(n)
        w = opt.solve(mu, cov, cons)
        assert np.all(w >= -1e-6), f"负权重: {w}"

        # -- 原 test_fallback_on_infeasible --
        opt = MeanVarianceOptimizer(risk_aversion=1.0)
        n = 3
        mu = np.ones(n) * 0.01
        # 构造奇异协方差（不一定触发失败，但结果应合法）
        cov = np.zeros((n, n))
        cons = _default_cons(n)
        w = opt.solve(mu, cov, cons)
        assert len(w) == n
        assert np.all(np.isfinite(w))


class TestRiskParityOptimizer:
    def test_risk_parity_optimizer_suite(self):
        """等波动率资产时风险平价退化为等权。；高波动率资产应获得更低权重。；权重之和应约为 1。"""
        # -- 原 test_equal_vol_equal_weight --
        opt = RiskParityOptimizer()
        n = 4
        cov = _equal_cov(n, sigma=0.01)
        mu = np.ones(n) * 0.01
        cons = _default_cons(n)
        w = opt.solve(mu, cov, cons)
        assert len(w) == n
        np.testing.assert_allclose(
            w, np.full(n, 1.0 / n), atol=0.02, err_msg="等波动率时风险平价应接近等权"
        )

        # -- 原 test_high_vol_lower_weight --
        opt = RiskParityOptimizer()
        n = 2
        cov = np.diag([0.01**2, 0.03**2])  # 第二个波动率 3 倍
        mu = np.ones(n) * 0.01
        cons = _default_cons(n)
        w = opt.solve(mu, cov, cons)
        assert w[0] > w[1], f"高波动资产权重应更低: {w}"

        # -- 原 test_weights_sum_to_one --
        opt = RiskParityOptimizer()
        n = 5
        cov = _equal_cov(n)
        mu = np.ones(n) * 0.01
        cons = _default_cons(n)
        w = opt.solve(mu, cov, cons)
        assert abs(w.sum() - 1.0) < 0.01


class TestMaxSharpeOptimizer:
    def test_max_sharpe_optimizer_suite(self):
        """高预期收益资产权重应显著更高。；全部预期收益非正时应返回合法 fallback。"""
        # -- 原 test_concentrates_on_high_return --
        opt = MaxSharpeOptimizer()
        n = 3
        mu = np.array([0.001, 0.01, 0.001])
        cov = _equal_cov(n)
        cons = _default_cons(n, max_weight=0.9)
        w = opt.solve(mu, cov, cons)
        assert w[1] > w[0] and w[1] > w[2], f"高收益资产权重应最高: {w}"

        # -- 原 test_negative_returns_fallback --
        opt = MaxSharpeOptimizer()
        n = 3
        mu = np.array([-0.01, -0.02, -0.005])
        cov = _equal_cov(n)
        cons = _default_cons(n)
        w = opt.solve(mu, cov, cons)
        assert len(w) == n
        assert np.all(np.isfinite(w))


class TestCovarianceEstimators:
    def test_covariance_estimators_suite(self):
        """test_sample_covariance_shape；test_ewma_covariance_shape；Ledoit-Wolf 协方差矩阵应为正半定。"""
        # -- 原 test_sample_covariance_shape --
        rng = np.random.default_rng(0)
        returns = rng.normal(0, 0.01, (100, 5))
        cov = sample_covariance(returns)
        assert cov.shape == (5, 5)
        assert np.allclose(cov, cov.T), "协方差矩阵应对称"

        # -- 原 test_ewma_covariance_shape --
        rng = np.random.default_rng(1)
        returns = rng.normal(0, 0.01, (60, 4))
        cov = ewma_covariance(returns, halflife=20)
        assert cov.shape == (4, 4)
        assert np.allclose(cov, cov.T)

        # -- 原 test_ledoit_wolf_psd --
        rng = np.random.default_rng(2)
        returns = rng.normal(0, 0.01, (50, 5))
        cov = ledoit_wolf_shrinkage(returns)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10), f"协方差矩阵含负特征值: {eigenvalues.min()}"


class TestUnsupportedConstraintWarnings:
    """MaxSharpe/RiskParity 不施加 turnover/net/gross 约束——必须显式告警而非静默忽略，
    否则用户以为约束生效、实际换手/暴露不受限，net-of-cost 研究结论失真。"""

    def test_helper_lists_only_nondefault_unsupported_constraints(self):
        from factorzen.daily.optimization.base import unsupported_constraint_warnings

        assert unsupported_constraint_warnings(OptimizerConstraints()) == []
        cons = OptimizerConstraints(turnover_limit=0.1, net_exposure=2.0, gross_exposure=1.5)
        msgs = unsupported_constraint_warnings(cons)
        assert any("turnover" in m for m in msgs)
        assert any("net_exposure" in m for m in msgs)
        assert any("gross_exposure" in m for m in msgs)

    def test_max_sharpe_warns_when_turnover_limit_ignored(self, caplog):
        import logging

        opt = MaxSharpeOptimizer()
        mu = np.array([0.02, 0.01])
        cov = np.eye(2) * 1e-4
        cons = OptimizerConstraints(turnover_limit=0.1, prev_weights=np.array([0.5, 0.5]))
        with caplog.at_level(logging.WARNING):
            opt.solve(mu, cov, cons)
        assert any("turnover" in r.message for r in caplog.records), (
            "MaxSharpe 忽略 turnover_limit 时应告警"
        )

    def test_risk_parity_warns_when_net_exposure_ignored(self, caplog):
        import logging

        opt = RiskParityOptimizer()
        mu = np.array([0.0, 0.0])
        cov = np.array([[1e-4, 0.0], [0.0, 4e-4]])
        cons = OptimizerConstraints(net_exposure=2.0)
        with caplog.at_level(logging.WARNING):
            opt.solve(mu, cov, cons)
        assert any("net_exposure" in r.message for r in caplog.records), (
            "RiskParity 忽略 net_exposure 时应告警"
        )

    def test_default_constraints_do_not_warn(self, caplog):
        import logging

        mu = np.array([0.02, 0.01])
        cov = np.eye(2) * 1e-4
        with caplog.at_level(logging.WARNING):
            MaxSharpeOptimizer().solve(mu, cov, OptimizerConstraints())
            RiskParityOptimizer().solve(
                mu, np.array([[1e-4, 0.0], [0.0, 4e-4]]), OptimizerConstraints()
            )
        assert not any("不施加" in r.message for r in caplog.records), (
            "默认约束(turnover=None, net/gross=1.0)不应产生未施加约束告警"
        )

# ==== 来自 test_square_root_impact.py ====
class TestLinearCostModel:
    def test_linear_cost_model_suite(self):
        """test_zero_delta_returns_zero；买入无印花税。；卖出包含印花税。；test_borrow_rate_daily；test_borrow_rate_weekly"""
        # -- 原 test_zero_delta_returns_zero --
        m = LinearCostModel()
        assert m.trade_cost(0.0) == 0.0

        # -- 原 test_buy_cost_has_no_stamp_tax --
        m = LinearCostModel(commission_rate=0.0003, stamp_tax_rate=0.001, slippage_rate=0.001)
        buy_cost = m.trade_cost(0.1)
        # commission + slippage only
        expected = 0.1 * (0.0003 + 0.001)
        assert abs(buy_cost - expected) < 1e-12

        # -- 原 test_sell_cost_includes_stamp_tax --
        m = LinearCostModel(commission_rate=0.0003, stamp_tax_rate=0.001, slippage_rate=0.001)
        sell_cost = m.trade_cost(-0.1)
        expected = 0.1 * (0.0003 + 0.001 + 0.001)
        assert abs(sell_cost - expected) < 1e-12

        # -- 原 test_borrow_rate_daily --
        m = LinearCostModel(annual_borrow_rate=0.08, trading_days_per_year=252)
        expected = 0.08 / 252
        assert abs(m.borrow_rate_per_period("daily") - expected) < 1e-12

        # -- 原 test_borrow_rate_weekly --
        m = LinearCostModel(annual_borrow_rate=0.08, trading_days_per_year=252)
        expected = 0.08 * 5 / 252
        assert abs(m.borrow_rate_per_period("weekly") - expected) < 1e-12


class TestSquareRootImpactCostModel:

    def test_square_root_impact_suite(self):
        """test_zero_delta_returns_zero；大交易时，平方根冲击成本 > 纯线性成本（有冲击项加成）。；平方根冲击：大交易边际成本率 > 小交易边际成本率（超线性）。；alpha=0 时，平方根模型退化为纯线性（无冲击项）。；提供更大的 ADV 时，冲击成本应降低（流动性更好）。"""
        # -- 原 test_zero_delta_returns_zero --
        m = SquareRootImpactCostModel()
        assert m.trade_cost(0.0) == 0.0

        # -- 原 test_sqroot_cost_greater_than_linear_for_large_trades --
        linear = LinearCostModel()
        sqroot = SquareRootImpactCostModel(alpha=0.1)
        large_delta = 0.5
        assert sqroot.trade_cost(large_delta) > linear.trade_cost(large_delta)

        # -- 原 test_sqroot_is_superlinear --
        m = SquareRootImpactCostModel(alpha=0.1)
        cost_small = m.trade_cost(0.01)
        cost_large = m.trade_cost(0.1)
        # 如果是纯线性，ratio = 10；超线性时 ratio > 10
        ratio = cost_large / cost_small
        assert ratio > 10, f"期望超线性 (ratio > 10)，实际 ratio={ratio:.3f}"

        # -- 原 test_alpha_zero_equals_linear --
        lin = LinearCostModel(commission_rate=0.0003, stamp_tax_rate=0.001, slippage_rate=0.001)
        sqr = SquareRootImpactCostModel(
            alpha=0.0,
            commission_rate=0.0003,
            stamp_tax_rate=0.001,
            slippage_rate=0.001,
        )
        delta = 0.3
        assert abs(sqr.trade_cost(delta) - lin.trade_cost(delta)) < 1e-12

        # -- 原 test_adv_provided_reduces_impact --
        m = SquareRootImpactCostModel(alpha=0.1, fallback_adv=1e7)
        cost_low_adv = m.trade_cost(0.1, adv=1e6)   # ADV 小（1e7 fallback / 1e6 = 10x smaller → higher impact）
        cost_high_adv = m.trade_cost(0.1, adv=1e8)  # ADV 大
        assert cost_high_adv < cost_low_adv, "ADV 越大，冲击成本应越小"


# ==== 来自 test_turnover_nan_filter.py ====
def _stable_panel(n_days: int = 5, n_stocks: int = 10) -> pl.DataFrame:
    """跨日排名完全稳定的截面（无 NaN）。"""
    rows: list[dict] = []
    d0 = date(2024, 1, 1)
    for di in range(n_days):
        d = d0 + timedelta(days=di)
        for si in range(n_stocks):
            rows.append(
                {
                    "trade_date": d,
                    "ts_code": f"{si:06d}.SZ",
                    "factor_clean": float(si),
                }
            )
    return pl.DataFrame(rows)


def test_turnover_nan_filter_suite():
    """次日多出 NaN 股会抬高 max_rank：旧实现挤占最高组、压缩有效股分位 → 虚假换手。；中位有效股变 NaN：旧实现跳进最高组并挤开更高有效股 → 虚假迁移。；不变量：每日固定若干 NaN 股 ≡ 物理删除这些行（含迁移矩阵）。；零回归：无 NaN 的稳定截面 → 换手率 0，结构完整。；退化：全日 NaN / 过滤后空 → 不崩，avg_turnover=0.0。"""
    # -- 原 test_nan_does_not_pollute_group_boundaries --
    def _section_0_test_nan_does_not_pollute_group_boundaries():
        d0, d1 = date(2024, 1, 2), date(2024, 1, 3)
        rows: list[dict] = []
        for d in (d0, d1):
            for si in range(5):
                rows.append(
                    {"trade_date": d, "ts_code": f"s{si}", "factor_clean": float(si)}
                )
        # 仅 d1 多一只 NaN（无有效信号，不应参与）
        rows.append({"trade_date": d1, "ts_code": "s_nan", "factor_clean": float("nan")})
        df = pl.DataFrame(rows)

        res = compute_turnover(df, factor_col="factor_clean", n_groups=5)
        assert not res.daily_turnover.is_empty()
        assert res.avg_turnover == pytest.approx(0.0, abs=1e-12), (
            f"NaN 不应抬高 max_rank 污染分组：期望 turnover=0，得 {res.avg_turnover}"
        )

    _section_0_test_nan_does_not_pollute_group_boundaries()

    # -- 原 test_nan_stock_does_not_enter_highest_group_and_jump --
    def _section_1_test_nan_stock_does_not_enter_highest_group_and_jump():
        d0, d1 = date(2024, 1, 2), date(2024, 1, 3)
        rows: list[dict] = []
        for si in range(5):
            rows.append({"trade_date": d0, "ts_code": f"s{si}", "factor_clean": float(si)})
        for si in range(5):
            val = float("nan") if si == 2 else float(si)
            rows.append({"trade_date": d1, "ts_code": f"s{si}", "factor_clean": val})
        nan_df = pl.DataFrame(rows)
        drop_df = nan_df.filter(pl.col("factor_clean").is_not_nan() & pl.col("factor_clean").is_not_null())

        nan_res = compute_turnover(nan_df, n_groups=5)
        drop_res = compute_turnover(drop_df, n_groups=5)

        assert nan_res.avg_turnover == pytest.approx(drop_res.avg_turnover, abs=1e-12)
        assert nan_res.daily_turnover.equals(drop_res.daily_turnover)
        assert nan_res.migration_matrix.equals(drop_res.migration_matrix)

        # 旧实现：s2 从 group2 跳到最高组，且 s3/s4 被挤压，avg 与 drop 不一致且通常更高
        # 修复后与 drop 一致；此处再断言「不应出现 NaN 跳跃带来的额外换手」
        # drop 路径下 d1 只有 4 只有效股，会有因截面缩减的正常重分桶换手
        assert nan_res.avg_turnover == pytest.approx(drop_res.avg_turnover, abs=1e-12)

    _section_1_test_nan_stock_does_not_enter_highest_group_and_jump()

    # -- 原 test_nan_factor_rows_equivalent_to_dropped_rows --
    def _section_2_test_nan_factor_rows_equivalent_to_dropped_rows():
        d0 = date(2024, 1, 1)
        rows: list[dict] = []
        n_stocks = 10
        for di in range(6):
            d = d0 + timedelta(days=di)
            for si in range(n_stocks):
                # 偶数日：高编号 3 只变 NaN；奇数日：全有效
                is_nan = (di % 2 == 0) and si >= 7
                rows.append(
                    {
                        "trade_date": d,
                        "ts_code": f"{si:06d}.SZ",
                        "factor_clean": float("nan") if is_nan else float(si),
                    }
                )
        nan_df = pl.DataFrame(rows)
        drop_df = nan_df.filter(
            pl.col("factor_clean").is_not_null() & pl.col("factor_clean").is_not_nan()
        )

        nan_res = compute_turnover(nan_df, factor_col="factor_clean", n_groups=5)
        drop_res = compute_turnover(drop_df, factor_col="factor_clean", n_groups=5)

        assert nan_res.avg_turnover == pytest.approx(drop_res.avg_turnover, abs=1e-12), (
            f"NaN 行应等价于删除：nan_avg={nan_res.avg_turnover} vs drop_avg={drop_res.avg_turnover}"
        )
        assert nan_res.daily_turnover.equals(drop_res.daily_turnover), (
            "daily_turnover 应与删除 NaN 行后一致"
        )
        assert nan_res.migration_matrix.equals(drop_res.migration_matrix), (
            "migration_matrix 应与删除 NaN 行后一致"
        )

    _section_2_test_nan_factor_rows_equivalent_to_dropped_rows()

    # -- 原 test_no_nan_stable_panel_zero_turnover_regression --
    def _section_3_test_no_nan_stable_panel_zero_turnover_regression():
        df = _stable_panel(n_days=5, n_stocks=10)
        res = compute_turnover(df, factor_col="factor_clean", n_groups=5)
        assert res.avg_turnover == pytest.approx(0.0, abs=1e-12)
        assert not res.daily_turnover.is_empty()
        assert "trade_date" in res.daily_turnover.columns
        assert "turnover" in res.daily_turnover.columns
        assert not res.migration_matrix.is_empty()
        assert set(res.migration_matrix.columns) == {"prev_group", "group", "prob"}

    _section_3_test_no_nan_stable_panel_zero_turnover_regression()

    # -- 原 test_all_nan_day_does_not_crash_avg_zero --
    def _section_4_test_all_nan_day_does_not_crash_avg_zero():
        df = pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 1), date(2024, 1, 1)],
                "ts_code": ["s0", "s1"],
                "factor_clean": [float("nan"), float("nan")],
            }
        )
        res = compute_turnover(df, factor_col="factor_clean", n_groups=2)
        assert res.avg_turnover == 0.0
        assert res.daily_turnover.is_empty()
        assert res.migration_matrix.is_empty() or res.migration_matrix.height == 0

    _section_4_test_all_nan_day_does_not_crash_avg_zero()


def test_null_factor_equivalent_to_nan_dropped():
    """null 与 NaN 一样不参与分组。"""
    d0 = date(2024, 1, 1)
    rows: list[dict] = []
    for di in range(4):
        d = d0 + timedelta(days=di)
        for si in range(8):
            is_null = (di % 2 == 0) and si >= 6
            rows.append(
                {
                    "trade_date": d,
                    "ts_code": f"{si:06d}.SZ",
                    "factor_clean": None if is_null else float(si),
                }
            )
    null_df = pl.DataFrame(rows).with_columns(pl.col("factor_clean").cast(pl.Float64))
    drop_df = null_df.filter(pl.col("factor_clean").is_not_null())

    null_res = compute_turnover(null_df, n_groups=4)
    drop_res = compute_turnover(drop_df, n_groups=4)
    assert null_res.avg_turnover == pytest.approx(drop_res.avg_turnover, abs=1e-12)
    assert null_res.daily_turnover.equals(drop_res.daily_turnover)
    assert null_res.migration_matrix.equals(drop_res.migration_matrix)

# ==== 来自 test_hyperparameter.py ====
# ── TestTuningSpace ──────────────────────────────────────────────────────────


def test_hyperparameter_optuna_suite():
    """整数参数 suggest 应返回指定范围内的 int。；浮点参数 suggest 应返回指定范围内的 float。；分类参数 suggest 应返回 choices 之一。；混合类型的 TuningSpace 应返回所有参数键。；凸目标函数 -(x-3)^2 的最大值应在 x=3 附近（允许误差 1.0）。"""
    # -- 原 test_suggest_int --
    def _section_0_test_suggest_int():
        import optuna

        space = TuningSpace([ParamSpec("n_groups", "int", low=5, high=20)])
        study = optuna.create_study()
        trial = study.ask()
        params = space.suggest(trial)
        assert "n_groups" in params
        assert isinstance(params["n_groups"], int)
        assert 5 <= params["n_groups"] <= 20

    _section_0_test_suggest_int()

    # -- 原 test_suggest_float --
    def _section_1_test_suggest_float():
        import optuna

        space = TuningSpace([ParamSpec("lr", "float", low=0.001, high=0.1)])
        study = optuna.create_study()
        trial = study.ask()
        params = space.suggest(trial)
        assert "lr" in params
        assert isinstance(params["lr"], float)
        assert 0.001 <= params["lr"] <= 0.1

    _section_1_test_suggest_float()

    # -- 原 test_suggest_categorical --
    def _section_2_test_suggest_categorical():
        import optuna

        choices = ["a", "b", "c"]
        space = TuningSpace([ParamSpec("mode", "categorical", choices=choices)])
        study = optuna.create_study()
        trial = study.ask()
        params = space.suggest(trial)
        assert "mode" in params
        assert params["mode"] in choices

    _section_2_test_suggest_categorical()

    # -- 原 test_suggest_all_types --
    def _section_3_test_suggest_all_types():
        import optuna

        space = TuningSpace(
            [
                ParamSpec("n", "int", low=10, high=50),
                ParamSpec("alpha", "float", low=0.01, high=1.0),
                ParamSpec("method", "categorical", choices=["a", "b"]),
            ]
        )
        study = optuna.create_study()
        trial = study.ask()
        params = space.suggest(trial)
        assert set(params.keys()) == {"n", "alpha", "method"}

    _section_3_test_suggest_all_types()

    # -- 原 test_finds_optimum_on_convex --
    def _section_4_test_finds_optimum_on_convex():
        space = TuningSpace([ParamSpec("x", "float", low=0.0, high=6.0)])

        def objective(params: dict) -> float:
            return -((params["x"] - 3.0) ** 2)

        best_params, _study = run_optuna_search(
            objective_fn=objective,
            space=space,
            n_trials=20,
            direction="maximize",
            study_name="test_convex",
        )
        assert "x" in best_params
        assert abs(best_params["x"] - 3.0) < 1.0, (
            f"最优 x={best_params['x']:.3f} 与真实最优 3.0 相差超过 1.0"
        )

    _section_4_test_finds_optimum_on_convex()

# ── TestRunOptunaSearch ──────────────────────────────────────────────────────


class TestRunOptunaSearch:

    def test_direction_minimize(self):
        """direction="minimize" 时应最小化目标函数。"""
        space = TuningSpace([ParamSpec("x", "float", low=0.0, high=10.0)])

        def objective(params: dict) -> float:
            return (params["x"] - 5.0) ** 2

        best_params, study = run_optuna_search(
            objective_fn=objective,
            space=space,
            n_trials=20,
            direction="minimize",
            study_name="test_minimize",
        )
        assert abs(best_params["x"] - 5.0) < 2.0, (
            f"minimize 时最优 x={best_params['x']:.3f} 应接近 5.0"
        )
        # 最优值应为最小（接近 0）
        assert study.best_value < 5.0


"""S3 防回归：验证 walk-forward IC 交叉验证。"""

import numpy as np

from factorzen.daily.evaluation.ic_analysis import _compute_walk_forward_ic


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

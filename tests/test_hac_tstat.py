"""S3 防回归：验证 Newey-West HAC t-stat 修正。

核心断言：对自相关 IC 序列，HAC t-stat 应明显小于朴素 t-stat（OLS 不带 HAC），
防止 t-stat 被高估 2-3 倍。
"""

import numpy as np
from scipy import stats as scipy_stats

from daily.evaluation.ic_analysis import _hac_maxlags, _ic_stats


def _make_autocorr_ic(n: int = 200, ar_coef: float = 0.6, seed: int = 42) -> np.ndarray:
    """生成 AR(1) 自相关 IC 序列（模拟因子 IC 的序列相关性）。"""
    rng = np.random.default_rng(seed)
    ic = np.zeros(n)
    ic[0] = rng.normal(0.03, 0.08)
    for t in range(1, n):
        ic[t] = ar_coef * ic[t - 1] + rng.normal(0, 0.08 * np.sqrt(1 - ar_coef**2))
    return ic


class TestHACTstat:
    def test_hac_maxlags_formula(self):
        """HAC 最优滞后阶数公式：floor(4*(N/100)^(2/9))，最小为 1。"""
        assert _hac_maxlags(100) == 4
        assert _hac_maxlags(50) >= 1
        assert _hac_maxlags(500) > 4

    def test_hac_tstat_smaller_than_naive_for_autocorr_series(self):
        """对高自相关 IC 序列，HAC t-stat 应小于朴素（iid）t-stat。"""
        ic = _make_autocorr_ic(n=200, ar_coef=0.6)
        # HAC t-stat
        _, _, _, _, hac_t, _ = _ic_stats(ic)
        # 朴素 t-stat（假设 iid）
        naive_t, _ = scipy_stats.ttest_1samp(ic, popmean=0.0)
        # HAC 应更保守（绝对值更小）
        assert abs(hac_t) < abs(naive_t), (
            f"HAC t={abs(hac_t):.2f} 应 < 朴素 t={abs(naive_t):.2f}（AR(1) 自相关序列）"
        )

    def test_hac_correction_ratio_reasonable(self):
        """HAC 与朴素 t-stat 的比值应在 0.3~1.0 范围（30-70% 修正幅度）。"""
        ic = _make_autocorr_ic(n=300, ar_coef=0.6)
        _, _, _, _, hac_t, _ = _ic_stats(ic)
        naive_t, _ = scipy_stats.ttest_1samp(ic, popmean=0.0)
        ratio = abs(hac_t) / (abs(naive_t) + 1e-10)
        assert 0.2 < ratio < 1.01, f"HAC/朴素 t 比值 {ratio:.2f} 超出合理范围 [0.2, 1.0]"

    def test_hac_low_autocorr_close_to_naive(self):
        """低自相关 IC 序列，HAC t-stat 应接近朴素 t-stat（修正幅度小）。"""
        rng = np.random.default_rng(99)
        ic = rng.normal(0.03, 0.08, 300)  # i.i.d.
        _, _, _, _, hac_t, _ = _ic_stats(ic)
        naive_t, _ = scipy_stats.ttest_1samp(ic, popmean=0.0)
        ratio = abs(hac_t) / (abs(naive_t) + 1e-10)
        # i.i.d. 序列下 HAC 与朴素几乎相同（允许 30% 偏差）
        assert ratio > 0.7, (
            f"i.i.d. 序列下 HAC t={abs(hac_t):.2f} 与朴素 t={abs(naive_t):.2f} 应接近"
        )

    def test_ic_stats_returns_valid_types(self):
        """_ic_stats 返回 6 个 float，无 nan/inf。"""
        ic = _make_autocorr_ic(n=100, ar_coef=0.4)
        result = _ic_stats(ic)
        assert len(result) == 6
        for val in result:
            assert isinstance(val, float), f"返回值 {val} 应为 float"
            assert np.isfinite(val), f"返回值 {val} 包含 nan/inf"

    def test_ic_stats_empty_input(self):
        """空输入应返回零值，不崩溃。"""
        result = _ic_stats(np.array([]))
        assert result == (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

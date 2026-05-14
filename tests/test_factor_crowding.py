"""测试因子拥挤度：衡量多个因子间的信号相似度。"""

import polars as pl

from daily.evaluation.advanced import (
    CrowdingResult,
    compute_factor_crowding,
)


def _make_factor_dict() -> dict[str, pl.DataFrame]:
    """构造多个因子数据的字典。"""
    stocks = [f"s{i}" for i in range(50)]
    base = pl.DataFrame({
        "trade_date": ["2026-01-05"] * 50,
        "ts_code": stocks,
    })
    # 因子 A 和 B 强相关（线性相关），因子 C 独立
    return {
        "momentum": base.with_columns(pl.lit(0.5).alias("factor_clean")),
        "value": base.with_columns(pl.Series("factor_clean", [i / 50 for i in range(50)])),
        "low_vol": base.with_columns(pl.Series("factor_clean", [i / 50 * (-1) for i in range(50)])),
    }


def test_crowding_returns_result_object():
    """compute_factor_crowding 返回 CrowdingResult。"""
    factor_dict = _make_factor_dict()
    result = compute_factor_crowding(factor_dict, factor_col="factor_clean")
    assert isinstance(result, CrowdingResult)


def test_crowding_has_corr_matrix():
    """CrowdingResult 包含相关性矩阵和因子名称列表。"""
    factor_dict = _make_factor_dict()
    result = compute_factor_crowding(factor_dict, factor_col="factor_clean")
    assert hasattr(result, "corr_matrix")
    assert hasattr(result, "factor_names")
    import numpy as np
    assert isinstance(result.corr_matrix, np.ndarray)
    assert result.corr_matrix.shape == (3, 3)


def test_crowding_diagonal_is_one():
    """相关性矩阵对角线为 1.0。"""
    factor_dict = _make_factor_dict()
    result = compute_factor_crowding(factor_dict, factor_col="factor_clean")
    n = len(result.factor_names)
    for i in range(n):
        assert abs(result.corr_matrix[i][i] - 1.0) < 1e-10


def test_crowding_has_crowding_score():
    """CrowdingResult 包含整体拥挤度评分。"""
    factor_dict = _make_factor_dict()
    result = compute_factor_crowding(factor_dict, factor_col="factor_clean")
    assert hasattr(result, "crowding_score")
    assert 0.0 <= result.crowding_score <= 1.0


def test_crowding_has_pairwise_df():
    """CrowdingResult 包含因子对级相关性 DataFrame。"""
    factor_dict = _make_factor_dict()
    result = compute_factor_crowding(factor_dict, factor_col="factor_clean")
    assert hasattr(result, "pairwise_corr")
    assert isinstance(result.pairwise_corr, pl.DataFrame)
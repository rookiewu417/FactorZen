"""协方差矩阵估计方法。"""
from __future__ import annotations

import numpy as np


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """样本协方差矩阵（无偏估计，ddof=1）。

    Args:
        returns: shape (T, N)，T期N个资产收益率。
    Returns:
        cov: shape (N, N)
    """
    return np.cov(returns, rowvar=False)


def ledoit_wolf_shrinkage(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf 线性收缩协方差矩阵估计（分析型公式）。

    收缩目标为球形矩阵（μI），按 Oracle 近似最优收缩强度收缩。
    使用 sklearn.covariance.ledoit_wolf 实现。
    如果 sklearn 不可用，回退到 sample_covariance。

    Args:
        returns: shape (T, N)
    Returns:
        cov: shape (N, N)
    """
    try:
        from sklearn.covariance import ledoit_wolf as sk_lw

        cov, _ = sk_lw(returns)
        return cov
    except ImportError:
        return sample_covariance(returns)


def ewma_covariance(returns: np.ndarray, halflife: int = 20) -> np.ndarray:
    """指数加权移动平均协方差（EWMA）。

    λ = 0.5^(1/halflife)，最近观测权重最大。

    Args:
        returns: shape (T, N)
        halflife: 半衰期（交易日数），默认 20。
    Returns:
        cov: shape (N, N)
    """
    T, _N = returns.shape
    lam = 0.5 ** (1.0 / halflife)
    weights = lam ** np.arange(T - 1, -1, -1)  # 越近越大
    weights /= weights.sum()
    mean_w = (weights[:, None] * returns).sum(axis=0)
    demeaned = returns - mean_w
    cov = (weights[:, None, None] * demeaned[:, :, None] * demeaned[:, None, :]).sum(axis=0)
    # 对称化防止浮点误差
    return (cov + cov.T) / 2

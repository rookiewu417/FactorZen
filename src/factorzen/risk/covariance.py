"""因子协方差矩阵与特质风险估计。

提供三种核心估计器：
1. estimate_factor_covariance — 指数加权 + Newey-West 调整的因子协方差
2. estimate_specific_risk — 收缩估计的特质风险（个股残差波动率）
3. eigenvector_adjustment — Monte Carlo 特征值调整（校正估计误差）
"""

from __future__ import annotations

import numpy as np

from factorzen.core.logger import get_logger

logger = get_logger(__name__)


def estimate_factor_covariance(
    factor_returns: np.ndarray,
    half_life: int = 90,
    nw_lags: int = 2,
) -> np.ndarray:
    """指数加权协方差 + Newey-West 自相关调整。

    计算流程：
    1. 指数加权去均值
    2. 加权样本协方差
    3. Newey-West 自相关修正（Bartlett 核权重）

    Args:
        factor_returns: shape (T, K)，T 期 K 个因子的收益率序列。
        half_life: 指数加权半衰期（交易日），默认 90。
        nw_lags: Newey-West 滞后阶数，默认 2。

    Returns:
        因子协方差矩阵，shape (K, K)，保证半正定。
    """
    T, K = factor_returns.shape

    if T < 2:
        logger.warning(f"因子收益序列过短 (T={T})，返回单位矩阵")
        return np.eye(K)

    # ── 1. 指数权重 ──────────────────────────────────────────────────────────
    lam = 0.5 ** (1.0 / half_life)
    raw_weights = lam ** np.arange(T - 1, -1, -1)  # 越近越大
    weights = raw_weights / raw_weights.sum()

    # ── 2. 加权均值 & 去均值 ─────────────────────────────────────────────────
    weighted_mean = (weights[:, None] * factor_returns).sum(axis=0)
    demeaned = factor_returns - weighted_mean

    # ── 3. 加权样本协方差 ────────────────────────────────────────────────────
    cov = np.zeros((K, K))
    for t in range(T):
        cov += weights[t] * np.outer(demeaned[t], demeaned[t])

    # ── 4. Newey-West 自相关修正 ─────────────────────────────────────────────
    for lag in range(1, nw_lags + 1):
        bartlett_weight = 1.0 - lag / (nw_lags + 1)
        gamma = np.zeros((K, K))
        for t in range(lag, T):
            w = min(weights[t], weights[t - lag])
            gamma += w * np.outer(demeaned[t], demeaned[t - lag])
        # 对称修正
        cov += bartlett_weight * (gamma + gamma.T)

    # ── 5. 对称化 & 半正定保证 ───────────────────────────────────────────────
    cov = (cov + cov.T) / 2.0

    # 确保半正定：特征值截断
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    cov = (cov + cov.T) / 2.0

    return cov


def estimate_specific_risk(
    residuals: np.ndarray,
    half_life: int = 90,
    shrinkage: float = 0.3,
) -> np.ndarray:
    """特质风险估计：指数加权波动率 + 贝叶斯收缩。

    计算流程：
    1. 每只股票独立计算指数加权残差方差（时序估计）
    2. 计算所有股票的截面均值方差（截面估计）
    3. 贝叶斯收缩：blend = (1 - shrinkage) * ts_var + shrinkage * cs_mean_var

    Args:
        residuals: shape (T, N)，T 期 N 只股票的回归残差。
        half_life: 指数加权半衰期，默认 90。
        shrinkage: 收缩强度 ∈ [0, 1]，默认 0.3。

    Returns:
        特质风险（标准差），shape (N,)。
    """
    T, N = residuals.shape

    if T < 2:
        logger.warning(f"残差序列过短 (T={T})，返回全 1 特质风险")
        return np.ones(N)

    # ── 1. 指数权重 ──────────────────────────────────────────────────────────
    lam = 0.5 ** (1.0 / half_life)
    raw_weights = lam ** np.arange(T - 1, -1, -1)
    weights = raw_weights / raw_weights.sum()

    # ── 2. 加权方差（时序估计）───────────────────────────────────────────────
    weighted_mean = (weights[:, None] * residuals).sum(axis=0)  # shape (N,)
    demeaned = residuals - weighted_mean
    ts_var = (weights[:, None] * demeaned**2).sum(axis=0)  # shape (N,)

    # ── 3. 截面均值方差 ─────────────────────────────────────────────────────
    cs_mean_var = np.mean(ts_var)

    # ── 4. 贝叶斯收缩 ─────────────────────────────────────────────────────
    blended_var = (1.0 - shrinkage) * ts_var + shrinkage * cs_mean_var

    # 方差不能为负
    blended_var = np.maximum(blended_var, 1e-12)

    return np.sqrt(blended_var)


def eigenvector_adjustment(
    cov: np.ndarray,
    n_simulations: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    """Monte Carlo 特征值调整。

    用于校正因子协方差矩阵的特征值估计偏差：
    1. 对原始协方差进行特征分解
    2. 从分解的分布中模拟多次样本协方差
    3. 计算每个特征值的偏差比率（模拟/真实）
    4. 用偏差比率调整原始特征值

    Args:
        cov: 原始因子协方差矩阵，shape (K, K)。
        n_simulations: Monte Carlo 模拟次数，默认 1000。
        seed: 随机种子。

    Returns:
        调整后的协方差矩阵，shape (K, K)。
    """
    K = cov.shape[0]
    if K == 0:
        return cov

    rng = np.random.default_rng(seed)

    # ── 1. 原始特征分解 ─────────────────────────────────────────────────────
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)

    # ── 2. 模拟 ─────────────────────────────────────────────────────────────
    # 用 T = K * 5 作为假设的样本量（经验值）
    T_sim = max(K * 5, 50)

    # 构造"真实"协方差
    L = eigvecs @ np.diag(np.sqrt(eigvals))

    sim_eigvals = np.zeros((n_simulations, K))
    for i in range(n_simulations):
        # 从 N(0, Σ) 生成样本
        Z = rng.standard_normal((T_sim, K))
        samples = Z @ L.T
        # 计算样本协方差
        sample_cov = np.cov(samples, rowvar=False)
        # 特征值
        sim_eigvals[i] = np.sort(np.linalg.eigvalsh(sample_cov))

    # ── 3. 计算偏差比率并调整 ────────────────────────────────────────────────
    mean_sim_eigvals = np.mean(sim_eigvals, axis=0)

    # 避免除零
    sorted_eigvals = np.sort(eigvals)
    adjustment_ratio = np.ones(K)
    for i in range(K):
        if mean_sim_eigvals[i] > 1e-15 and sorted_eigvals[i] > 1e-15:
            adjustment_ratio[i] = sorted_eigvals[i] / mean_sim_eigvals[i]

    # 调整原始特征值
    # eigvals 已从 eigh 排序（升序）
    adjusted_eigvals = eigvals * adjustment_ratio
    adjusted_eigvals = np.maximum(adjusted_eigvals, 0.0)

    # ── 4. 重构协方差矩阵 ───────────────────────────────────────────────────
    adjusted_cov = eigvecs @ np.diag(adjusted_eigvals) @ eigvecs.T
    adjusted_cov = (adjusted_cov + adjusted_cov.T) / 2.0

    return adjusted_cov

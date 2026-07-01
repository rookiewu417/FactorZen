"""MC3: crypto 风险模型（复用协方差/特质/MCR 数学 + crypto 风格因子/sector）。"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from factorzen.markets.base import RiskModel as RiskModelPort
from factorzen.markets.crypto.factors import CryptoFactorSet
from factorzen.markets.crypto.profile import build_crypto_profile
from factorzen.markets.crypto.risk import CryptoRiskModel, build_crypto_risk_model
from factorzen.markets.crypto.risk_factors import CRYPTO_STYLE_NAMES, CRYPTO_STYLE_REGISTRY
from factorzen.markets.crypto.sectors import build_sector_frame
from factorzen.risk.model import RiskModel
from tests.test_markets_crypto_mining import FakeCCXTBulk

_SECTORS = ["L1", "DeFi", "meme"]


def _daily_with_btc(n_other: int = 39, n_days: int = 65, seed: int = 5):
    rng = np.random.default_rng(seed)
    codes = ["BTCUSDT"] + [f"SYM{i:02d}USDT" for i in range(n_other)]
    start = date(2024, 1, 1)
    rows = []
    for i, code in enumerate(codes):
        price = 100.0 + i
        for d in range(n_days):
            price = max(1.0, price * (1 + rng.normal(0, 0.02)))
            vol = float(rng.uniform(50, 500))
            rows.append({
                "ts_code": code, "trade_date": start + timedelta(days=d),
                "open": price, "high": price * 1.01, "low": price * 0.99, "close": price,
                "vol": vol, "amount": price * vol,
                "funding_rate": float(rng.normal(0.0001, 0.0003)),
                "open_interest": float(rng.uniform(1e3, 5e3)),
            })
    daily = CryptoFactorSet().derived_columns(pl.DataFrame(rows))
    return daily, codes


# ── Port ──────────────────────────────────────────────────────────────────────
def test_crypto_risk_is_port():
    rm = CryptoRiskModel()
    assert isinstance(rm, RiskModelPort)


def test_style_factors_registry():
    rm = CryptoRiskModel()
    sf = rm.style_factors()
    assert set(sf) == {"size", "liquidity", "momentum", "volatility", "funding_carry", "btc_beta"}


def test_sector_classification_one_hot():
    rm = CryptoRiskModel()
    dummies = rm.sector_classification(["BTCUSDT", "UNIUSDT", "DOGEUSDT"], "20240101")
    cols = set(dummies.columns)
    assert "ind_L1" in cols and "ind_DeFi" in cols and "ind_meme" in cols
    # BTC=L1
    btc = dummies.filter(pl.col("ts_code") == "BTCUSDT")
    assert btc["ind_L1"][0] == 1.0


# ── 核心构建（含 BTC，btc_beta 有效）─────────────────────────────────────────────
def test_core_build_with_crypto_factors():
    daily, codes = _daily_with_btc()
    sector_map = {c: _SECTORS[i % 3] for i, c in enumerate(codes)}
    stocks = build_sector_frame(codes, sector_map)
    model = RiskModel(periods_per_year=365, cov_half_life=30, spec_half_life=30)
    res = model.build(
        daily, daily, stocks, "20240101", "20240310",
        style_registry=CRYPTO_STYLE_REGISTRY, style_names=CRYPTO_STYLE_NAMES,
        ret_col="ret_1d", ret_is_pct=False,
    )
    assert res.factor_names, "应产出因子暴露"
    # 含 crypto 风格因子 + sector one-hot
    assert {"size", "liquidity", "momentum", "volatility"} <= set(res.factor_names)
    assert any(n.startswith("ind_") for n in res.factor_names)
    # 因子协方差 PSD（复用 Newey-West 估计）
    eig = np.linalg.eigvalsh(res.factor_covariance)
    assert (eig >= -1e-8).all()
    # 特质风险非负有限
    assert np.all(np.isfinite(res.specific_risk)) and np.all(res.specific_risk >= 0)
    # predict/decompose（年化用 365）
    n = res.factor_exposures.n_stocks
    w = np.ones(n) / n
    total = model.predict_risk(w, res)
    assert np.isfinite(total) and total >= 0
    decomp = model.decompose_risk(w, res)
    assert np.isfinite(decomp["total_risk"])
    assert model.periods_per_year == 365


def test_annualization_uses_365_not_252():
    """同一日度协方差，crypto(365) 年化风险 > A股(252) 比例 √(365/252)。"""
    daily, codes = _daily_with_btc()
    stocks = build_sector_frame(codes, {c: _SECTORS[i % 3] for i, c in enumerate(codes)})
    kw = dict(style_registry=CRYPTO_STYLE_REGISTRY, style_names=CRYPTO_STYLE_NAMES,
              ret_col="ret_1d", ret_is_pct=False)
    m365 = RiskModel(periods_per_year=365, cov_half_life=30, spec_half_life=30)
    res = m365.build(daily, daily, stocks, "20240101", "20240310", **kw)
    m252 = RiskModel(periods_per_year=252, cov_half_life=30, spec_half_life=30)
    n = res.factor_exposures.n_stocks
    w = np.ones(n) / n
    r365 = m365.predict_risk(w, res)
    r252 = m252.predict_risk(w, res)
    if r252 > 0:
        assert abs(r365 / r252 - np.sqrt(365 / 252)) < 1e-6


# ── 端到端入口（FakeCCXTBulk，无 BTC → btc_beta 退化但仍构建）──────────────────────
def test_build_crypto_risk_model_end_to_end():
    fake = FakeCCXTBulk()
    profile = build_crypto_profile(client=fake)
    syms = fake.symbols
    sector_map = {c: _SECTORS[i % 3] for i, c in enumerate(syms)}
    model, res = build_crypto_risk_model(
        profile, syms, "20240101", "20240224", sector_map=sector_map,
        cov_half_life=30, spec_half_life=30,
    )
    assert res.factor_names
    assert {"size", "liquidity", "volatility"} <= set(res.factor_names)
    assert model.periods_per_year == 365
    eig = np.linalg.eigvalsh(res.factor_covariance)
    assert (eig >= -1e-8).all()

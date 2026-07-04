"""crypto RiskModel Port 实现 + 风险模型构建入口。

复用市场无关的协方差/特质风险/MCR 数学（risk/covariance.py, risk/model.py），
只替换风格因子集（crypto 版）与 sector 分类；年化用 365。
"""
from __future__ import annotations

from datetime import date

import polars as pl

from factorzen.markets.base import MarketProfile, RiskModel
from factorzen.markets.crypto.frequency import periods_per_year as _freq_ppy
from factorzen.markets.crypto.risk_factors import CRYPTO_STYLE_NAMES, CRYPTO_STYLE_REGISTRY
from factorzen.markets.crypto.sectors import build_sector_frame


class CryptoRiskModel(RiskModel):
    """crypto 风险因子集 + sector 分类（RiskModel Port）。"""

    def __init__(self, sector_map: dict[str, str] | None = None) -> None:
        self.sector_map = sector_map

    def style_factors(self) -> dict:
        return dict(CRYPTO_STYLE_REGISTRY)

    def sector_classification(self, symbols: list[str], d: date | str) -> pl.DataFrame:
        """标的 → sector one-hot（``ind_*`` 列，与消费方约定兼容）。d 目前为静态分类。"""
        from factorzen.risk.industry_factors import get_industry_dummies

        frame = build_sector_frame(symbols, self.sector_map)
        return get_industry_dummies(frame, industry_col="industry")


def build_crypto_risk_model(
    profile: MarketProfile,
    symbols: list[str],
    start: str,
    end: str,
    *,
    freq: str | None = None,
    sector_map: dict[str, str] | None = None,
    cov_half_life: int = 90,
    nw_lags: int = 2,
    spec_half_life: int = 90,
    spec_shrinkage: float = 0.3,
):
    """构建 crypto 多因子风险模型，返回 ``(RiskModel, RiskModelResult)``。

    RiskModel 实例持有 ``periods_per_year=365``，供后续 ``predict_risk``/``decompose_risk``。
    """
    from factorzen.markets.crypto.mining import build_crypto_daily
    from factorzen.risk.model import RiskModel as CoreRiskModel

    freq = freq or profile.base_freq
    provider = profile.provider
    assert hasattr(provider, "fetch_funding"), "build_crypto_risk_model 需 crypto profile(provider 缺 funding 扩展)"
    daily = build_crypto_daily(provider, symbols, start, end, freq)
    daily = profile.factors.derived_columns(daily)
    stocks = build_sector_frame(symbols, sector_map)
    model = CoreRiskModel(
        cov_half_life=cov_half_life,
        nw_lags=nw_lags,
        spec_half_life=spec_half_life,
        spec_shrinkage=spec_shrinkage,
        periods_per_year=int(_freq_ppy(freq)),
    )
    result = model.build(
        daily, daily, stocks, start, end,
        style_registry=CRYPTO_STYLE_REGISTRY,
        style_names=CRYPTO_STYLE_NAMES,
        ret_col="ret_1d",
        ret_is_pct=False,
    )
    return model, result

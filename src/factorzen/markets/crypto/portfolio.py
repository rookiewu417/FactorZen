"""crypto 组合构建入口：crypto 风险模型 → 优化(可做空/市场中性) → 归因(年化365)。

复用市场无关的 optimize_portfolio / run_portfolio pipeline；只把约束换成 crypto：
- 默认**市场中性做空**：budget=0(Σw=0 美元中性)、long_only=False、gross_limit(毛敞口上限)。
- sector 中性到 0（做空下可行，无需基准暴露；对比 A 股 long-only 必须中性到基准）。
"""
from __future__ import annotations

import numpy as np
import polars as pl

from factorzen.markets.base import MarketProfile
from factorzen.markets.crypto.risk import build_crypto_risk_model

_CRYPTO_PERIODS_PER_YEAR = 365


def _latest_factor_returns(risk_result) -> dict:
    fr = risk_result.factor_returns
    if fr is None or fr.is_empty():
        return {}
    last = fr.tail(1)
    return {n: float(last[n][0]) for n in risk_result.factor_names if n in last.columns}


def build_crypto_portfolio(
    profile: MarketProfile,
    alpha: pl.DataFrame,
    symbols: list[str],
    start: str,
    end: str,
    *,
    market_neutral: bool = True,
    w_max: float = 0.1,
    gross_limit: float = 1.0,
    risk_aversion: float = 1.0,
    sector_neutral: bool = True,
    out_dir: str = "workspace/portfolios",
    run_id: str | None = None,
    signal_date: str | None = None,
    sector_map: dict[str, str] | None = None,
) -> dict:
    """构建 crypto 组合（返回 run_portfolio 的落盘结果 dict）。

    ``alpha``: ``[ts_code, alpha]`` 截面（如 export_crypto_alpha 产出）。
    """
    from factorzen.pipelines.portfolio_build import run_portfolio

    _model, risk_result = build_crypto_risk_model(
        profile, symbols, start, end, sector_map=sector_map
    )
    codes = risk_result.factor_exposures.codes
    if not codes:
        raise RuntimeError("crypto 风险模型无暴露（数据不足），无法建仓")

    # α 对齐 codes（缺失填 0）
    amap = dict(zip(alpha["ts_code"].to_list(), alpha["alpha"].to_list(), strict=False))
    alpha_vec = np.array([float(amap.get(c, 0.0)) for c in codes])

    # sector 标签（Brinson 归因用）+ sector 中性列
    from factorzen.markets.crypto.sectors import sector_of
    sectors = [sector_of(c, sector_map) for c in codes]
    neutral_factors = (
        [n for n in risk_result.factor_names if n.startswith("ind_")] if sector_neutral else None
    )

    return run_portfolio(
        alpha_vec,
        risk_result,
        codes=codes,
        stock_returns=np.zeros(len(codes)),  # 建仓时点无持仓期收益
        sectors=sectors,
        factor_returns_latest=_latest_factor_returns(risk_result),
        neutral_factors=neutral_factors,
        bench_weights=None,           # 做空下 sector 中性到 0 可行
        risk_aversion=risk_aversion,
        w_max=w_max,
        long_only=not market_neutral,  # 市场中性 → 允许做空
        budget=0.0 if market_neutral else 1.0,
        gross_limit=gross_limit if market_neutral else None,
        periods_per_year=_CRYPTO_PERIODS_PER_YEAR,
        out_dir=out_dir,
        run_id=run_id,
        signal_date=signal_date,
    )

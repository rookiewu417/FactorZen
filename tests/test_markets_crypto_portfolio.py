"""MC4: crypto 组合优化(可做空/市场中性) + 归因(年化365)。"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from factorzen.markets.crypto.portfolio import build_crypto_portfolio
from factorzen.markets.crypto.profile import build_crypto_profile
from tests.test_markets_crypto_mining import FakeCCXTBulk

_SECTORS = ["L1", "DeFi", "meme"]


def _setup():
    fake = FakeCCXTBulk()
    profile = build_crypto_profile(client=fake)
    syms = fake.symbols
    sector_map = {c: _SECTORS[i % 3] for i, c in enumerate(syms)}
    # 合成 α：按序号给正负信号，确保优化非平凡
    rng = np.random.default_rng(0)
    alpha = pl.DataFrame({"ts_code": syms, "alpha": rng.normal(0, 1, len(syms))})
    return profile, syms, sector_map, alpha


def test_crypto_portfolio_market_neutral(tmp_path):
    profile, syms, sector_map, alpha = _setup()
    res = build_crypto_portfolio(
        profile, alpha, syms, "20240101", "20240224",
        market_neutral=True, w_max=0.15, gross_limit=1.0, risk_aversion=0.1,
        out_dir=str(tmp_path), run_id="cryp1", sector_map=sector_map,
    )
    assert res["status"] == "optimal"
    w = pl.read_parquet(tmp_path / "cryp1" / "weights.parquet")["target_weight"].to_numpy()
    # 市场中性：Σw ≈ 0
    assert abs(w.sum()) < 1e-5
    # 毛敞口 ≤ gross_limit（含数值容差）
    assert np.abs(w).sum() <= 1.0 + 1e-4
    # 做空存在：有负权重
    assert (w < -1e-6).any()
    # box 上下界
    assert w.max() <= 0.15 + 1e-6 and w.min() >= -0.15 - 1e-6


def test_crypto_portfolio_manifest_and_risk_summary(tmp_path):
    profile, syms, sector_map, alpha = _setup()
    build_crypto_portfolio(
        profile, alpha, syms, "20240101", "20240224",
        risk_aversion=0.1, out_dir=str(tmp_path), run_id="cryp2",
        signal_date="2024-02-24", sector_map=sector_map,
    )
    manifest = json.loads((tmp_path / "cryp2" / "manifest.json").read_text())
    assert manifest["signal_date"] == "2024-02-24"  # sim 消费的关键字段
    assert (tmp_path / "cryp2" / "risk_summary.csv").exists()
    assert (tmp_path / "cryp2" / "attribution.csv").exists()
    # 风险摘要非空（decompose_risk 年化365）
    rs = pl.read_csv(tmp_path / "cryp2" / "risk_summary.csv")
    assert rs.height > 0

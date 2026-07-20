"""test_markets_crypto_backtest.py：MC5: crypto 净值回测（换手成本 + funding 逐期计提 + 做空）。
test_markets_crypto_portfolio.py：MC4: crypto 组合优化(可做空/市场中性) + 归因(年化365)。
test_markets_crypto_risk.py：MC3: crypto 风险模型（复用协方差/特质/MCR 数学 + crypto 风格因子/sector）。
test_markets_crypto_intraday_backtest.py：intraday NAV 回测:手算 ground truth(收益/funding 逐 bar)+ 信号键上抛。
"""

from __future__ import annotations

import json
from datetime import (
    date,
    datetime,
    timedelta,
)

import numpy as np
import polars as pl
import pytest

from factorzen.markets.base import RiskModel as RiskModelPort
from factorzen.markets.crypto.backtest import (
    _coerce_signal_keys,
    run_crypto_simulation,
    simulate_crypto_nav,
)
from factorzen.markets.crypto.costs import CryptoCostModel
from factorzen.markets.crypto.factors import CryptoFactorSet
from factorzen.markets.crypto.portfolio import build_crypto_portfolio
from factorzen.markets.crypto.profile import build_crypto_profile
from factorzen.markets.crypto.risk import (
    CryptoRiskModel,
    build_crypto_risk_model,
)
from factorzen.markets.crypto.risk_factors import (
    CRYPTO_STYLE_NAMES,
    CRYPTO_STYLE_REGISTRY,
)
from factorzen.markets.crypto.sectors import build_sector_frame
from factorzen.risk.model import RiskModel
from tests.markets.test_crypto_mining import FakeCCXTBulk


# ==== 来自 test_markets_crypto_backtest.py ====
def test_simulate_crypto_nav_ground_truth():
    """手算 ground-truth：多空+换手成本+funding（多头付/空头收）。"""
    d0, d1, d2 = date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)
    daily = pl.DataFrame({
        "ts_code": ["A", "A", "A", "B", "B", "B"],
        "trade_date": [d0, d1, d2, d0, d1, d2],
        "close": [100.0, 110.0, 121.0, 50.0, 45.0, 45.0],
    })
    funding = pl.DataFrame({
        "ts_code": ["A", "A", "B", "B"],
        "trade_date": [d1, d2, d1, d2],
        "funding_rate": [0.001, 0.0, 0.0, 0.002],
    })
    weights = {d0: pl.DataFrame({"ts_code": ["A", "B"], "target_weight": [0.5, -0.5]})}
    out = simulate_crypto_nav(weights, daily, funding)  # default CryptoCostModel: taker+slip=0.001
    nav = out["nav"].sort("trade_date")
    assert nav.height == 3
    # d0：建仓 turnover=1.0 → cost=0.001，net=-0.001
    assert abs(nav["cost"][0] - 0.001) < 1e-12
    assert abs(nav["net_return"][0] + 0.001) < 1e-12
    # d1：gross=0.5*0.1+(-0.5)*(-0.1)=0.1；funding=0.5*0.001=0.0005；net=0.0995
    assert abs(nav["gross_return"][1] - 0.1) < 1e-12
    assert abs(nav["borrow_cost"][1] - 0.0005) < 1e-12
    assert abs(nav["net_return"][1] - 0.0995) < 1e-12
    assert abs(nav["cost"][1]) < 1e-12  # 无调仓无换手
    # d2：gross=0.05；funding=(-0.5)*0.002=-0.001(空头收)；net=0.051
    assert abs(nav["gross_return"][2] - 0.05) < 1e-12
    assert abs(nav["borrow_cost"][2] + 0.001) < 1e-12
    assert abs(nav["net_return"][2] - 0.051) < 1e-12
    # nav 递推：0.999 * 1.0995 * 1.051
    expected = 0.999 * 1.0995 * 1.051
    assert abs(nav["nav"][2] - expected) < 1e-9


def test_simulate_crypto_nav_metrics():
    d = [date(2024, 1, i) for i in range(1, 6)]
    daily = pl.DataFrame({
        "ts_code": ["A"] * 5, "trade_date": d,
        "close": [100.0, 101.0, 102.0, 103.0, 104.0],
    })
    weights = {d[0]: pl.DataFrame({"ts_code": ["A"], "target_weight": [1.0]})}
    out = simulate_crypto_nav(weights, daily)
    m = out["metrics"]
    assert set(m) >= {"ann_ret", "ann_vol", "sharpe", "max_dd", "avg_turnover", "total_cost"}
    assert m["ann_ret"] > 0  # 单调上涨
    assert np.isfinite(m["sharpe"])


def test_run_crypto_simulation_end_to_end(tmp_path):
    """组合建仓 → crypto 模拟交易 → nav.parquet + metrics.json。"""
    fake = FakeCCXTBulk()
    profile = build_crypto_profile(client=fake)
    syms = fake.symbols
    sector_map = {c: ["L1", "DeFi", "meme"][i % 3] for i, c in enumerate(syms)}
    rng = np.random.default_rng(1)
    alpha = pl.DataFrame({"ts_code": syms, "alpha": rng.normal(0, 1, len(syms))})
    pdir = tmp_path / "port"
    build_crypto_portfolio(
        profile, alpha, syms, "20240101", "20240131",
        risk_aversion=0.1, out_dir=str(pdir), run_id="p1",
        signal_date="2024-02-01", sector_map=sector_map,
    )
    res = run_crypto_simulation(
        [str(pdir / "p1")], profile, "20240201", "20240224",
        out_dir=str(tmp_path / "sim"), run_id="s1",
    )
    assert np.isfinite(res["sharpe"])
    nav = pl.read_parquet(tmp_path / "sim" / "s1" / "nav.parquet")
    assert {"trade_date", "gross_return", "cost", "borrow_cost", "net_return", "nav",
            "cash_weight"} <= set(nav.columns)
    assert nav.height >= 2
    manifest = json.loads((tmp_path / "sim" / "s1" / "manifest.json").read_text())
    assert manifest["market"] == "crypto"
    metrics = json.loads((tmp_path / "sim" / "s1" / "metrics.json").read_text())
    assert "sharpe" in metrics and "max_dd" in metrics

# ==== 来自 test_markets_crypto_portfolio.py ====
_SECTORS__portfolio = ["L1", "DeFi", "meme"]


def _setup():
    fake = FakeCCXTBulk()
    profile = build_crypto_profile(client=fake)
    syms = fake.symbols
    sector_map = {c: _SECTORS__portfolio[i % 3] for i, c in enumerate(syms)}
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

# ==== 来自 test_markets_crypto_risk.py ====
_SECTORS__risk = ["L1", "DeFi", "meme"]


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
    sector_map = {c: _SECTORS__risk[i % 3] for i, c in enumerate(codes)}
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
    stocks = build_sector_frame(codes, {c: _SECTORS__risk[i % 3] for i, c in enumerate(codes)})
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
    sector_map = {c: _SECTORS__risk[i % 3] for i, c in enumerate(syms)}
    model, res = build_crypto_risk_model(
        profile, syms, "20240101", "20240224", sector_map=sector_map,
        cov_half_life=30, spec_half_life=30,
    )
    assert res.factor_names
    assert {"size", "liquidity", "volatility"} <= set(res.factor_names)
    assert model.periods_per_year == 365
    eig = np.linalg.eigvalsh(res.factor_covariance)
    assert (eig >= -1e-8).all()

# ==== 来自 test_markets_crypto_intraday_backtest.py ====
def test_coerce_signal_keys_upcasts_date_for_intraday():
    w = pl.DataFrame({"ts_code": ["BTCUSDT"], "target_weight": [1.0]})
    out = _coerce_signal_keys({date(2026, 5, 1): w}, "1h")
    assert list(out.keys()) == [datetime(2026, 5, 1, 0, 0)]
    same = _coerce_signal_keys({date(2026, 5, 1): w}, "daily")
    assert list(same.keys()) == [date(2026, 5, 1)]  # daily 不动


def test_simulate_nav_hourly_ground_truth():
    # 单标的满仓多头,3 根 1h bar:100→110→99;第 2 根 bar 落 0.001 funding
    ts = [datetime(2026, 5, 1, h) for h in (0, 1, 2)]
    daily = pl.DataFrame({"ts_code": ["BTCUSDT"] * 3, "trade_date": ts,
                          "close": [100.0, 110.0, 99.0]})
    funding = pl.DataFrame({"ts_code": ["BTCUSDT"], "trade_date": [ts[1]],
                            "funding_rate": [0.001]})
    w = {ts[0]: pl.DataFrame({"ts_code": ["BTCUSDT"], "target_weight": [1.0]})}
    res = simulate_crypto_nav(w, daily, funding,
                              cost_model=CryptoCostModel(taker=0.0, slippage=0.0),
                              periods_per_year=8760)
    nets = res["nav"]["net_return"].to_list()
    # bar0=信号日无持仓;bar1: +10% - 0.001 funding = 0.099;bar2: -10%
    assert nets[0] == pytest.approx(0.0)
    assert nets[1] == pytest.approx(0.10 - 0.001)
    assert nets[2] == pytest.approx(-0.10)
    assert res["metrics"]["total_funding"] == pytest.approx(0.001)  # 仅 bar1 计提

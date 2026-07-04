"""MC5: crypto 净值回测（换手成本 + funding 逐期计提 + 做空）。"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import polars as pl

from factorzen.markets.crypto.backtest import run_crypto_simulation, simulate_crypto_nav
from factorzen.markets.crypto.portfolio import build_crypto_portfolio
from factorzen.markets.crypto.profile import build_crypto_profile
from tests.test_markets_crypto_mining import FakeCCXTBulk


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

"""test_crypto_e2e_pipeline.py：crypto 全链路端到端 smoke（离线）：挖掘→防过拟合→风险→组合→模拟交易→展示页。
test_crypto_intraday_e2e.py：crypto 15m 全链路端到端(离线 mini-lake):挖掘→验证→α→NAV。
test_markets_crypto_intraday_mining.py：15m 挖掘链:mini-lake → build_crypto_daily → run_session 离线全绿。
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import polars as pl
import pytest

from factorzen.markets.crypto.backtest import (
    run_crypto_simulation,
    simulate_crypto_nav,
)
from factorzen.markets.crypto.mining import (
    build_crypto_daily,
    export_crypto_alpha,
    run_crypto_mining,
    validate_crypto_expression,
)
from factorzen.markets.crypto.portfolio import build_crypto_portfolio
from factorzen.markets.crypto.profile import build_crypto_profile
from factorzen.reports.portfolio_report import generate_portfolio_report
from tests.markets.test_crypto_lake import make_mini_lake
from tests.markets.test_crypto_mining import FakeCCXTBulk

# ==== 来自 test_crypto_e2e_pipeline.py ====
_SECTORS = ["L1", "DeFi", "meme"]


def test_crypto_full_pipeline_offline(tmp_path):
    fake = FakeCCXTBulk()
    profile = build_crypto_profile(client=fake)
    syms = fake.symbols
    sector_map = {c: _SECTORS[i % 3] for i, c in enumerate(syms)}
    start, end = "20240101", "20240224"

    # 1) 挖掘 → 带 OOS/holdout/PBO 的候选
    mine = run_crypto_mining(profile, syms, start, end, n_trials=30, top_k=3, seed=5,
                             out_dir=str(tmp_path / "mine"))
    assert mine["candidates"], "阶段1 挖掘应产候选"
    expr = mine["candidates"][0]["expression"]

    # 2) 防过拟合验证
    rep = validate_crypto_expression(profile, expr, syms, start, end)
    assert 0.0 <= rep["dsr_p"] <= 1.0

    # 3) export-alpha 截面
    alpha = export_crypto_alpha(profile, expr, syms, start, end, date=end)
    assert alpha.columns == ["ts_code", "alpha"] and alpha.height >= 30

    # 4) 组合构建（市场中性做空 + crypto 风险模型 + 归因）
    port = build_crypto_portfolio(
        profile, alpha, syms, "20240101", "20240201",
        risk_aversion=0.1, out_dir=str(tmp_path / "port"), run_id="p",
        signal_date="2024-02-05", sector_map=sector_map,
    )
    assert port["status"] == "optimal", "阶段4 组合优化应可解"

    # 5) 模拟交易（funding + 做空 NAV 回测）
    sim = run_crypto_simulation(
        [str(tmp_path / "port" / "p")], profile, "20240205", "20240224",
        out_dir=str(tmp_path / "sim"), run_id="s",
    )
    assert np.isfinite(sim["sharpe"])
    metrics = json.loads((tmp_path / "sim" / "s" / "metrics.json").read_text())
    assert "total_funding" in metrics  # 阶段5→6 接口：资金费成本落盘

    # 6) 展示页（crypto 语境）
    attribution = pl.read_csv(tmp_path / "port" / "p" / "attribution.csv")
    risk_summary = pl.read_csv(tmp_path / "port" / "p" / "risk_summary.csv")
    html = generate_portfolio_report(
        None, metrics=metrics, attribution_df=attribution,
        risk_summary_df=risk_summary, market="crypto",
    )
    assert "USDT" in html and "365" in html and "资金费" in html

# ==== 来自 test_crypto_intraday_e2e.py ====
def test_full_chain_15m_offline(tmp_path):
    syms = [f"C{i:02d}USDT" for i in range(40)]  # ≥MIN_IC_SAMPLES(30)
    make_mini_lake(tmp_path, symbols=tuple(syms), days=(1, 2))
    profile = build_crypto_profile(lake_root=tmp_path)
    # 1) 挖掘
    res = run_crypto_mining(profile, syms, "20260501", "20260502",
                            n_trials=8, top_k=3, seed=7, freq="15m",
                            out_dir=str(tmp_path / "sessions"))
    assert res["candidates"], "15m 挖掘应产出候选"
    expr = res["candidates"][0]["expression"]
    # 2) 防过拟合验证
    rep = validate_crypto_expression(profile, expr, syms, "20260501", "20260502", freq="15m")
    assert set(rep) >= {"ic_mean", "ir", "dsr_p", "ci_lo", "ci_hi", "n"} and rep["n"] > 0
    # 3) 截面 α 导出(intraday: 当日 00:00 bar 截面)
    alpha = export_crypto_alpha(profile, expr, syms, "20260501", "20260502",
                                date="20260502", freq="15m")
    assert alpha.columns == ["ts_code", "alpha"]
    # 4) NAV 回测(2 标的等权多空,信号=首 bar)
    two = syms[:2]
    daily = profile.provider.fetch_bars(two, "20260501", "20260502", "15m")
    w = {datetime(2026, 5, 1, 0, 0): pl.DataFrame(
        {"ts_code": two, "target_weight": [0.5, -0.5]})}
    sim = simulate_crypto_nav(
        w, daily,
        profile.provider.fetch_funding(two, "20260501", "20260502", "15m"),
        cost_model=profile.costs, periods_per_year=35040)
    assert sim["nav"].height > 0 and "sharpe" in sim["metrics"]


@pytest.mark.skipif(os.environ.get("FZ_NETWORK_TESTS") != "1",
                    reason="真网 smoke:FZ_NETWORK_TESTS=1 手动触发,CI 跳过")
def test_real_vision_backfill_smoke(tmp_path):
    """2 标的 × 一个整月:真实下载 → 湖 → 15m bars 非空。"""
    from factorzen.markets.crypto.lake import CryptoLake
    from factorzen.markets.crypto.lake_provider import CryptoLakeProvider
    from factorzen.markets.crypto.vision import backfill
    lake = CryptoLake(tmp_path)
    backfill(lake, ["BTCUSDT", "ETHUSDT"], "20260501", "20260531")
    bars = CryptoLakeProvider(lake=lake).fetch_bars(
        ["BTCUSDT", "ETHUSDT"], "20260501", "20260531", "15m")
    assert bars.height > 2 * 2000  # 31 天 × 96 根/日 × 2 标的,允许零星缺口

# ==== 来自 test_markets_crypto_intraday_mining.py ====
def test_build_crypto_daily_15m(tmp_path):
    make_mini_lake(tmp_path)
    profile = build_crypto_profile(lake_root=tmp_path)
    daily = build_crypto_daily(profile.provider, ["BTCUSDT", "ETHUSDT"],
                               "20260501", "20260502", "15m")
    assert daily.schema["trade_date"] == pl.Datetime("us")
    # funding 只落在 00:00 bar;OI intraday 前向填充不留 0 空洞
    b = daily.filter(pl.col("ts_code") == "BTCUSDT").sort("trade_date")
    assert b["funding_rate"][0] == 0.0001 and b["funding_rate"][1] == 0.0
    assert 0.0 not in b["open_interest"].to_list()[1:]  # ffill 生效(首 bar 前无值仍可为 0)

"""crypto 全链路端到端 smoke（离线）：挖掘→防过拟合→风险→组合→模拟交易→展示页。

针对项目最痛的教训「单测各自绿 ≠ 拼起来能跑」——把 6 个阶段真正串起来跑一遍，
用同一个 crypto MarketProfile + 同一份 fake 交易所数据，验证跨阶段接口无 gap。
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from factorzen.markets.crypto.backtest import run_crypto_simulation
from factorzen.markets.crypto.mining import (
    export_crypto_alpha,
    run_crypto_mining,
    validate_crypto_expression,
)
from factorzen.markets.crypto.portfolio import build_crypto_portfolio
from factorzen.markets.crypto.profile import build_crypto_profile
from factorzen.reports.portfolio_report import generate_portfolio_report
from tests.test_markets_crypto_mining import FakeCCXTBulk

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

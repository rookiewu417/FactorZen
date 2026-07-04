"""crypto 15m 全链路端到端(离线 mini-lake):挖掘→验证→α→NAV。

防跨组件集成 gap(单测各自绿 ≠ 拼起来能跑)。真实网络 smoke 见文件末
network marker 用例(CI 自动跳过)。注:组合优化(portfolio build)的 intraday
risk 截面 dtype 未在本链验证(e2e 不走组合),intraday 组合为已知限制。
"""
import os
from datetime import datetime

import polars as pl
import pytest

from factorzen.markets.crypto.backtest import simulate_crypto_nav
from factorzen.markets.crypto.mining import (
    export_crypto_alpha,
    run_crypto_mining,
    validate_crypto_expression,
)
from factorzen.markets.crypto.profile import build_crypto_profile
from tests.test_markets_crypto_lake_provider import make_mini_lake


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

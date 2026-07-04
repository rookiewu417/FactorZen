"""15m 挖掘链:mini-lake → build_crypto_daily → run_session 离线全绿。"""
import polars as pl

from factorzen.markets.crypto.mining import build_crypto_daily, run_crypto_mining
from factorzen.markets.crypto.profile import build_crypto_profile
from tests.test_markets_crypto_lake_provider import make_mini_lake


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


def test_run_crypto_mining_15m_smoke(tmp_path):
    # ≥MIN_IC_SAMPLES(30) 个标的,否则 compute_rank_ic 跳过全部横截面 → 挖不出候选
    syms = [f"C{i:02d}USDT" for i in range(40)]
    make_mini_lake(tmp_path, symbols=tuple(syms))
    profile = build_crypto_profile(lake_root=tmp_path)
    res = run_crypto_mining(profile, syms, "20260501", "20260502",
                            n_trials=8, top_k=3, seed=7, freq="15m",
                            out_dir=str(tmp_path / "sessions"))
    assert len(res["candidates"]) >= 1

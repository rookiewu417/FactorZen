"""MC1 T3: run_session 吃 MarketProfile —— crypto 数据(无 close_adj)能挖掘，A 股默认不变。"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from factorzen.discovery.mining_session import run_session
from factorzen.markets.crypto.profile import build_crypto_profile


def _synthetic_crypto_daily(n_sym: int = 40, n_days: int = 55, seed: int = 7) -> pl.DataFrame:
    # 截面样本数需 ≥ MIN_IC_SAMPLES(30) 否则 compute_rank_ic 跳过该日 → IC 序列空
    rng = np.random.default_rng(seed)
    rows = []
    start = date(2024, 1, 1)
    for s in range(n_sym):
        code = f"SYM{s:02d}USDT"
        price = 100.0 + s
        for d in range(n_days):
            ret = rng.normal(0, 0.02)
            price = max(1.0, price * (1 + ret))
            vol = float(rng.uniform(50, 500))
            rows.append({
                "ts_code": code,
                "trade_date": start + timedelta(days=d),
                "open": price * (1 + rng.normal(0, 0.001)),
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "vol": vol,
                "amount": price * vol,
                "funding_rate": float(rng.normal(0.0001, 0.0002)),
                "open_interest": float(rng.uniform(1000, 5000)),
            })
    return pl.DataFrame(rows)


def test_run_session_crypto_profile(tmp_path):
    """crypto daily(无 close_adj) + crypto profile → 挖出带 holdout/DSR/PBO 的 candidates。"""
    daily = _synthetic_crypto_daily()
    profile = build_crypto_profile()
    result = run_session(
        daily,
        n_trials=40,
        top_k=5,
        seed=1,
        method="random",
        out_dir=str(tmp_path),
        profile=profile,
    )
    assert result["candidates"], "crypto 挖掘应产出至少一个候选"
    sess = tmp_path / "session_1_random"
    assert (sess / "candidates.csv").exists()
    cand = pl.read_csv(sess / "candidates.csv")
    # OOS + 防过拟合列齐全
    for col in ["holdout_ic", "dsr_pvalue", "pbo", "ic_train", "ir_train"]:
        assert col in cand.columns
    # 候选表达式只用 crypto 叶子(含 funding_rate/open_interest 可能出现)
    from factorzen.discovery.expression import feature_names, parse_expr
    crypto_leaves = set(profile.factors.leaf_features().keys())
    for expr in cand["expression"].to_list():
        assert feature_names(parse_expr(expr, crypto_leaves)) <= crypto_leaves


def test_run_session_ashare_default_unchanged(tmp_path):
    """A 股默认路径(profile=None)仍用 close_adj 派生，行为不变。"""
    rng = np.random.default_rng(3)
    rows = []
    start = date(2024, 1, 1)
    for s in range(10):
        code = f"00000{s}.SZ"
        price = 10.0 + s
        for d in range(50):
            price = max(1.0, price * (1 + rng.normal(0, 0.02)))
            vol = float(rng.uniform(1e5, 1e6))
            rows.append({
                "ts_code": code, "trade_date": start + timedelta(days=d),
                "open": price, "high": price * 1.01, "low": price * 0.99, "close": price,
                "open_adj": price, "high_adj": price * 1.01, "low_adj": price * 0.99,
                "close_adj": price, "vol": vol, "amount": price * vol,
                # 基本面叶子列(避免随机表达式引用 LEAF_FEATURES 中的 basic 叶子报 missing)
                "total_mv": price * 1e6, "circ_mv": price * 8e5, "pb": 1.5,
                "pe_ttm": 15.0, "ps_ttm": 3.0, "dv_ttm": 2.0,
            })
    daily = pl.DataFrame(rows)
    result = run_session(daily, n_trials=30, top_k=3, seed=2, out_dir=str(tmp_path))
    assert "candidates" in result
    assert (tmp_path / "session_2_random" / "candidates.csv").exists()

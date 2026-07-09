"""MC1 T3: run_session 吃 MarketProfile —— crypto 数据(无 close_adj)能挖掘，A 股默认不变。"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

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
    """A 股默认路径(profile=None)：真产候选、只用 A 股叶子、护栏列齐全。

    旧版只断言 `"candidates" in result`（字典必然有这个 key）与 CSV 文件存在，
    并跑在 **10 只股票**上——`_MIN_CROSS_SAMPLES=30` 把每个截面整天丢弃，IC 恒空。
    这条守卫是 CLAUDE.md 的「A 股零回归是底线」，判别力必须真实存在。
    """
    rng = np.random.default_rng(3)
    rows = []
    start_d = date(2024, 1, 1)
    for s_i in range(40):        # ≥ _MIN_CROSS_SAMPLES(=30)，否则 IC 序列为空
        code = f"{600000 + s_i:06d}.SH"
        price = 10.0 + s_i
        for d in range(120):
            prev_price = price
            price = max(1.0, price * (1 + rng.normal(0, 0.02)))
            vol = float(rng.uniform(1e5, 1e6))
            rows.append({
                "ts_code": code, "trade_date": start_d + timedelta(days=d),
                "pre_close": prev_price,
                "open": price, "high": price * 1.01, "low": price * 0.99, "close": price,
                "open_adj": price, "high_adj": price * 1.01, "low_adj": price * 0.99,
                "close_adj": price, "vol": vol, "amount": price * vol,
                "total_mv": price * 1e6, "circ_mv": price * 8e5, "pb": 1.5,
                "pe_ttm": 15.0, "ps_ttm": 3.0, "dv_ttm": 2.0,
            })
    daily = pl.DataFrame(rows)
    from factorzen.discovery.operators import BASIC_FEATURES

    daily = daily.with_columns([
        pl.lit(1.0).alias(c) for c in sorted(BASIC_FEATURES) if c not in daily.columns
    ])
    result = run_session(daily, n_trials=30, top_k=3, seed=2, out_dir=str(tmp_path))

    # 与 crypto profile 那条测试对称：断言**非空**，而非 `"candidates" in result`
    # ——后者是字典必然有的 key，恒真。
    assert result["candidates"], "A 股默认路径应产出至少一个候选"

    sess = tmp_path / "session_2_random"
    assert (sess / "candidates.csv").exists()
    cand = pl.read_csv(sess / "candidates.csv")
    assert cand.height > 0, "A 股默认路径应产出候选（IC 全空时这里会是 0 行）"

    # 护栏列齐全（与 crypto profile 那条测试对称）
    for col in ["holdout_ic", "dsr_pvalue", "pbo", "ic_train", "ir_train", "passed"]:
        assert col in cand.columns

    # IC 真的被算出来了——全 NaN 说明截面被整天丢弃
    ic = cand["ic_train"].to_list()
    assert any(v == v and v != 0.0 for v in ic), f"ic_train 全为 nan/0：{ic}"

    # 表达式只用 A 股叶子（profile=None 不该混入 crypto 叶子）
    from factorzen.discovery.expression import feature_names, parse_expr
    from factorzen.discovery.operators import LEAF_FEATURES

    ashare_leaves = set(LEAF_FEATURES.keys())
    for expr in cand["expression"].to_list():
        assert feature_names(parse_expr(expr)) <= ashare_leaves
    crypto_only = {"funding_rate", "open_interest"}
    assert not (set().union(*(feature_names(parse_expr(e))
                              for e in cand["expression"].to_list())) & crypto_only)


def test_ashare_default_derives_ret_1d_from_close_adj():
    """docstring 承诺的「仍用 close_adj 派生」必须真的被验证。

    造一个 close 与 close_adj 显著不同的帧（模拟除权）：`ret_1d` 必须由 close_adj 算出。
    旧测试通篇没碰过这件事。
    """
    from factorzen.discovery.derived import add_derived_columns

    rows = []
    for d in range(4):
        rows.append({
            "ts_code": "600000.SH", "trade_date": date(2024, 1, 1) + timedelta(days=d),
            "pre_close": 10.0, "open": 10.0, "high": 10.1, "low": 9.9,
            "close": 100.0 * (d + 1),          # 未复权价：乱跳
            "close_adj": 10.0 * (1.10 ** d),   # 复权价：每日 +10%
            "open_adj": 10.0, "high_adj": 10.1, "low_adj": 9.9,
            "vol": 1e5, "amount": 1e6,
        })
    out = add_derived_columns(pl.DataFrame(rows))
    ret = out["ret_1d"].to_list()

    assert ret[0] is None
    for v in ret[1:]:
        assert v == pytest.approx(0.10, abs=1e-9), (
            f"ret_1d={v}，应为 close_adj 的 10% 日涨幅；若由 close 派生会得到 1.0/0.5/…"
        )

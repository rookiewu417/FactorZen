"""MC2: 防过拟合护栏(bootstrap IC CI + DSR)在 crypto 上有效。"""
from __future__ import annotations

import math

from factorzen.discovery.scoring import ic_overfit_report
from factorzen.markets.crypto.mining import build_crypto_daily, validate_crypto_expression
from factorzen.markets.crypto.profile import build_crypto_profile
from tests.test_markets_crypto_mining import FakeCCXTBulk


def _profile_syms():
    fake = FakeCCXTBulk()
    return build_crypto_profile(client=fake), fake.symbols


def test_ic_overfit_report_market_agnostic():
    """ic_overfit_report 吃 factor_df+daily(任意市场)，产出 IC/IR/DSR/CI。"""
    profile, syms = _profile_syms()
    daily = build_crypto_daily(profile.provider, syms, "20240101", "20240224")
    daily = profile.factors.derived_columns(daily)
    factor_df = daily.select(["trade_date", "ts_code"]).with_columns(
        daily["ret_1d"].alias("factor_value")
    )
    rep = ic_overfit_report(factor_df, daily)
    assert set(rep) >= {"ic_mean", "ir", "dsr_p", "ci_lo", "ci_hi", "n"}
    assert rep["n"] > 0
    assert all(math.isfinite(rep[k]) for k in ("ic_mean", "ir", "dsr_p"))
    assert rep["ci_lo"] <= rep["ci_hi"]


def test_validate_crypto_expression():
    """crypto 单表达式防过拟合验证：bootstrap CI + DSR 在 crypto 上跑通。"""
    profile, syms = _profile_syms()
    rep = validate_crypto_expression(
        profile, "ts_mean(ret_1d, 5)", syms, "20240101", "20240224"
    )
    assert rep["n"] > 0
    assert math.isfinite(rep["ir"])
    assert math.isfinite(rep["dsr_p"])
    assert 0.0 <= rep["dsr_p"] <= 1.0

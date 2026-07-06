"""泛化回归守卫：所有注册因子的 required_data 必含 "daily"。

根因：daily_single 管线第 6 步无条件 `ctx.daily.collect()` 算前向收益（IC/回测都靠它），
而 FactorDataContext.daily 在 "daily" 未声明时直接 raise ValueError。任何因子只声明
["daily_basic"] 就会在评估到第 6 步时崩溃——size_style/value_style/liquidity_style 与月频
pe_ttm/pb/ep_ratio/bm_ratio 曾集体踩坑（与已修的 finance 因子 asset_growth/roe_ttm 同类）。

用注册表遍历做守卫，任何新因子漏声明 "daily" 都会在此失败，而非等真实数据评估时才炸。
"""
from __future__ import annotations

import factorzen.builtin_factors  # noqa: F401  触发因子注册
from factorzen.daily.factors.registry import get_factor, list_factors


def test_every_registered_factor_declares_daily():
    offenders = []
    for name in list_factors():
        factor = get_factor(name)
        required = getattr(factor, "required_data", None) or []
        if "daily" not in required:
            offenders.append((name, getattr(factor, "category", "?"), list(required)))
    assert not offenders, (
        "以下因子的 required_data 漏声明 'daily'，评估管线算前向收益时会 raise "
        "'daily data not declared'：" + "; ".join(f"{n}({c})={rd}" for n, c, rd in offenders)
    )


def test_valuation_factors_keep_daily_basic_and_add_daily():
    """这些估值/规模/流动性因子的 compute 确实读 daily_basic，故 daily 与 daily_basic 都要有。"""
    for name in ("size_style", "value_style", "liquidity_style", "pe_ttm", "pb",
                 "ep_ratio", "bm_ratio"):
        required = getattr(get_factor(name), "required_data", None) or []
        assert "daily" in required, f"{name} 需 ctx.daily 算前向收益"
        assert "daily_basic" in required, f"{name} compute 读 daily_basic，不应移除声明"

"""filter_liquidity 的 amount 单位回归测试。

根因：Tushare ``daily.amount`` 单位是**千元**，而 ``min_amount`` 文档语义是**元**（默认
10_000_000=1000万元）。直接 ``amount >= min_amount`` 把千元当元比，实际门槛变成 100 亿元，
intraday_default 股票池塌缩到几十只。修复后应按 1000万元 的真实意图过滤。
"""
from __future__ import annotations

import polars as pl


def _fake_daily(amounts_qy: dict[str, float]):
    """构造一天的 daily 帧，amount 单位=千元（Tushare 口径）。"""
    codes = list(amounts_qy)
    return pl.LazyFrame(
        {
            "ts_code": codes,
            "trade_date": [pl.date(2026, 6, 5)] * len(codes),
            "amount": [amounts_qy[c] for c in codes],
        }
    )


def test_filter_liquidity_uses_yuan_threshold(monkeypatch):
    import factorzen.core.storage as storage
    from factorzen.core.universe import filter_liquidity

    # A: 2000万元成交额 = 20_000 千元（应留）；B: 500万元 = 5_000 千元（应剔）
    amounts_qy = {"A.SZ": 20_000.0, "B.SZ": 5_000.0}
    monkeypatch.setattr(storage, "load_parquet", lambda *a, **k: _fake_daily(amounts_qy))

    stocks = pl.DataFrame({"ts_code": ["A.SZ", "B.SZ"], "industry": ["X", "Y"]})
    # 默认 min_amount=1000万元
    kept = filter_liquidity(stocks, "20260605")["ts_code"].to_list()

    assert "A.SZ" in kept, "2000万元成交额应通过 1000万元 门槛（修复前因单位错配被剔除）"
    assert "B.SZ" not in kept, "500万元成交额应被 1000万元 门槛剔除"


def test_filter_liquidity_realistic_market_not_collapsed(monkeypatch):
    """真实量级：中位数约 1.36亿元（≈135_762 千元）的市场不应被门槛几乎清空。"""
    import factorzen.core.storage as storage
    from factorzen.core.universe import filter_liquidity

    # 100 只股票，成交额 5000万~5亿元（=50_000~500_000 千元），全部远超 1000万元 门槛
    amounts_qy = {f"{i:06d}.SZ": 50_000.0 + i * 4500.0 for i in range(100)}
    monkeypatch.setattr(storage, "load_parquet", lambda *a, **k: _fake_daily(amounts_qy))

    stocks = pl.DataFrame({"ts_code": list(amounts_qy), "industry": ["X"] * 100})
    kept = filter_liquidity(stocks, "20260605")
    assert kept.height == 100, f"全部应通过，修复前会因 100亿元 假门槛只剩极少数（实得 {kept.height}）"

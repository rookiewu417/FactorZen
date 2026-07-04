# tests/test_discovery_factor.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import polars as pl


def _make_daily_lf(n_stocks=8, n_days=60, seed=42) -> pl.LazyFrame:
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for s in [f"{i:06d}.SH" for i in range(n_stocks)]:
        price = 10.0
        for day in days:
            price = float(max(price * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": day, "ts_code": s, "close": price,
                         "open": price, "high": price, "low": price,
                         "close_adj": price, "open_adj": price, "high_adj": price, "low_adj": price,
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
                         "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows).lazy()


@dataclass
class MockCtx:
    start: str = "20240301"
    end: str = "20240331"
    required_data: list = field(default_factory=lambda: ["daily", "daily_basic"])
    lookback_days: int = 30
    universe: list | None = None
    snapshot_mode: str = "daily"
    _daily: pl.LazyFrame | None = None
    _basic: pl.LazyFrame | None = None

    @property
    def daily(self) -> pl.LazyFrame:
        return self._daily

    @property
    def daily_basic(self) -> pl.LazyFrame:
        return self._basic if self._basic is not None else pl.DataFrame(
            {"trade_date": [], "ts_code": []}).lazy()


def test_expression_factor_matches_builtin_momentum():
    """pct_change(close, 20) 应与内置 momentum_20d 的 compute 输出一致。"""
    from factorzen.builtin_factors.daily.momentum import Momentum20D
    from factorzen.discovery.factor import ExpressionFactor

    lf = _make_daily_lf()
    ctx = MockCtx(_daily=lf)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builtin = Momentum20D().compute(ctx).sort(["trade_date", "ts_code"])

    mined = ExpressionFactor(expression="pct_change(close, 20)", mined_name="m20",
                             lookback_days=30).compute(ctx).sort(["trade_date", "ts_code"])

    j = builtin.join(mined, on=["trade_date", "ts_code"], suffix="_m")
    assert j.height > 0
    diff = (j["factor_value"] - j["factor_value_m"]).abs().max()
    assert diff is not None and diff < 1e-9


def test_suspended_rows_masked():
    """vol==0（停牌）行价量被置 null → 因子值被过滤；vol>0 行正常产出。
    用零阶表达式 close 使单行即可判别，避免窗口不足导致的 trivial 通过。"""
    from factorzen.discovery.factor import ExpressionFactor

    lf = _make_daily_lf()
    d = date(2024, 3, 15)
    extra = pl.DataFrame({
        "trade_date": [d, d], "ts_code": ["888888.SH", "999999.SH"],
        "close": [5.0, 5.0], "open": [5.0, 5.0], "high": [5.0, 5.0], "low": [5.0, 5.0],
        "close_adj": [5.0, 5.0], "open_adj": [5.0, 5.0], "high_adj": [5.0, 5.0], "low_adj": [5.0, 5.0],
        "amount": [1e6, 0.0], "vol": [1e5, 0.0],
    }).lazy()
    ctx = MockCtx(_daily=pl.concat([lf, extra]))
    out = ExpressionFactor(expression="close", mined_name="x", lookback_days=30).compute(ctx)
    # 停牌股(vol=0)该行被掩码 → 无输出
    sus = out.filter((pl.col("ts_code") == "999999.SH") & (pl.col("trade_date") == d))
    assert sus.height == 0
    # 正常股(vol>0)该行有 close=5.0
    ok = out.filter((pl.col("ts_code") == "888888.SH") & (pl.col("trade_date") == d))
    assert ok.height == 1 and abs(ok["factor_value"][0] - 5.0) < 1e-9


def test_expression_factor_is_valid_dailyfactor():
    from factorzen.daily.factors.base import DailyFactor
    from factorzen.discovery.factor import ExpressionFactor
    ef = ExpressionFactor(expression="ts_mean(close, 5)", mined_name="probe", lookback_days=40)
    assert isinstance(ef, DailyFactor)
    assert ef.name == "probe"
    assert ef.frequency == "daily"
    assert ef.category == "daily"
    assert ef.lookback_days == 40
    assert "daily" in ef.required_data


def test_ret_1d_correct_when_ctx_daily_rows_unsorted():
    """compute() 必须先排序(ts_code, trade_date)再派生依赖行序的 ret_1d(shift().over())。

    构造收盘价逐日单调上涨（每天 +1%）但行序被打乱（非 ts_code/trade_date 有序）的数据：
    若 compute() 在排序前就用 shift(1).over("ts_code") 算 ret_1d，会把同一只股票里
    乱序的「上一行」当成「前一交易日」，算出包含负值的错误结果；正确实现下，因为
    收盘价严格单调上涨，每只股票每天的 ret_1d 必须全部是同一个正值 0.01。"""
    from factorzen.discovery.factor import ExpressionFactor

    start = date(2024, 1, 2)
    n_days = 20
    days: list[date] = []
    d = start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)

    rows = []
    for s in ["000001.SH", "000002.SH", "000003.SH"]:
        price = 10.0
        for day in days:
            price *= 1.01  # 严格单调上涨：每天 +1%
            rows.append({
                "trade_date": day, "ts_code": s,
                "close": price, "open": price, "high": price, "low": price,
                "close_adj": price, "open_adj": price, "high_adj": price, "low_adj": price,
                "amount": 1e7, "vol": 1e5,
            })

    # 行序打乱（固定 seed 可复现）：不再是 (ts_code, trade_date) 有序
    daily_df = pl.DataFrame(rows).sample(fraction=1.0, shuffle=True, seed=7)
    per_stock = daily_df.filter(pl.col("ts_code") == "000001.SH")["trade_date"].to_list()
    assert per_stock != sorted(per_stock), "fixture 未真正打乱行序，测试无法复现 bug"

    ctx = MockCtx(_daily=daily_df.lazy(), start="20240101")
    out = ExpressionFactor(expression="ret_1d", mined_name="r1", lookback_days=5).compute(ctx)

    assert out.height > 0
    assert (out["factor_value"] > 0).all(), "收盘价逐日单调上涨，ret_1d 必须全部为正"
    assert (out["factor_value"] - 0.01).abs().max() < 1e-9

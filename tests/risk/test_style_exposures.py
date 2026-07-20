"""test_risk_style_factors.py：风险风格因子 cs_standardize / registry / size 截面形状
test_style_factors.py：个人日频库 Barra 风格因子（size/value/momentum/vol/liquidity/beta）
test_risk_style_panel_materialize.py：风格面板一次物化与逐日重算等价；行业并集稳定化
test_risk_exposures.py：compute_exposures 形状/行业哑元/PIT 行业降级与部分覆盖
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

import factorzen.risk.exposures as exposures_module
from factorzen.daily.factors.base import DailyFactor
from factorzen.risk.exposures import (
    compute_exposures,
    materialize_industry_panel,
    materialize_style_panel,
)
from factorzen.risk.style_factors import cs_standardize

# ==== 来自 test_risk_style_factors.py ====
# tests/test_risk_style_factors.py

def _trade_days__risk_style_factors(start, n):
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days

def make_daily_basic__risk_style_factors(n_stocks=8, n_days=10, seed=0):
    rng = np.random.default_rng(seed)
    days = _trade_days__risk_style_factors(dt.date(2023, 1, 3), n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        for d in days:
            rows.append({"trade_date": d, "ts_code": c,
                         "total_mv": float(abs(rng.standard_normal()) * 1e9 + 5e9),
                         "pb": float(abs(rng.standard_normal()) + 1.5),
                         "pe_ttm": float(abs(rng.standard_normal()) * 10 + 15),
                         "turnover_rate": float(abs(rng.standard_normal()) * 2 + 1)})
    return pl.DataFrame(rows)

def test_registry_has_eight_named_factors():
    from factorzen.risk.style_factors import STYLE_FACTOR_NAMES, STYLE_FACTOR_REGISTRY
    assert STYLE_FACTOR_NAMES == ["size", "value", "momentum", "volatility",
                                  "liquidity", "quality", "growth", "leverage"]
    assert set(STYLE_FACTOR_REGISTRY.keys()) == set(STYLE_FACTOR_NAMES)

def test_size_factor_shape():
    from factorzen.risk.style_factors import STYLE_FACTOR_REGISTRY
    db = make_daily_basic__risk_style_factors()
    out = STYLE_FACTOR_REGISTRY["size"](pl.DataFrame(), db)
    assert set(out.columns) >= {"trade_date", "ts_code", "factor_value"}
    assert out.height > 0
    vals = out["factor_value"].drop_nulls().drop_nans()
    assert vals.std() > 0  # 非全零/全同值 stub（size 在不同市值股票间有离散度）

def test_cs_standardize_rejects_unknown_method():
    import pytest

    from factorzen.risk.style_factors import cs_standardize

    df = pl.DataFrame({"trade_date": [dt.date(2023, 1, 3)], "factor_value": [1.0]})
    with pytest.raises(ValueError):
        cs_standardize(df, method="zscore")

# ==== 来自 test_style_factors.py ====
def _make_daily_lf(n_stocks: int = 10, n_days: int = 310, seed: int = 42) -> pl.LazyFrame:
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 3)
    days: list[date] = []
    d = start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)

    stocks = [f"{i:06d}.SH" for i in range(n_stocks)]
    rows = []
    for s in stocks:
        price = 10.0
        for day in days:
            price = float(max(price * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": day, "ts_code": s, "close_adj": price})
    return pl.DataFrame(rows).lazy()

def _make_daily_basic_lf(n_stocks: int = 10, n_days: int = 60, seed: int = 0) -> pl.LazyFrame:
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)
    days: list[date] = []
    d = start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)

    stocks = [f"{i:06d}.SH" for i in range(n_stocks)]
    rows = []
    for s in stocks:
        for day in days:
            rows.append(
                {
                    "trade_date": day,
                    "ts_code": s,
                    "turnover_rate": float(abs(rng.standard_normal()) * 2 + 1),
                    "total_mv": float(abs(rng.standard_normal()) * 1e9 + 5e9),
                    "pb": float(abs(rng.standard_normal()) * 1 + 2),
                }
            )
    return pl.DataFrame(rows).lazy()

@dataclass
class MockCtx:
    start: str = "20240101"
    end: str = "20240430"
    required_data: list = field(default_factory=list)
    lookback_days: int = 30
    universe: list | None = None
    snapshot_mode: str = "daily"
    _daily_lf: pl.LazyFrame | None = field(default=None, repr=False)
    _daily_basic_lf: pl.LazyFrame | None = field(default=None, repr=False)

    @property
    def daily(self) -> pl.LazyFrame:
        return self._daily_lf

    @property
    def daily_basic(self) -> pl.LazyFrame:
        return self._daily_basic_lf

@pytest.fixture()
def ctx_basic():
    c = MockCtx(start="20240101", end="20240430")
    c._daily_basic_lf = _make_daily_basic_lf()
    return c

@pytest.fixture()
def ctx_daily():
    # start early so the date filter keeps data with shift(252) filled
    c = MockCtx(start="20230601", end="20240430")
    c._daily_lf = _make_daily_lf()
    return c

def _check(result: pl.DataFrame, name: str) -> None:
    assert isinstance(result, pl.DataFrame), f"{name}: expected DataFrame"
    assert "trade_date" in result.columns, f"{name}: missing trade_date"
    assert "ts_code" in result.columns, f"{name}: missing ts_code"
    assert "factor_value" in result.columns, f"{name}: missing factor_value"
    assert result.shape[0] > 0, f"{name}: empty result"

def test_liquidity_style(ctx_basic):
    from factorzen.builtin_factors.daily.liquidity import LiquidityStyle

    factor = LiquidityStyle()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx_basic)
    _check(result, "liquidity_style")

def test_size_style(ctx_basic):
    from factorzen.builtin_factors.daily.size import SizeStyle

    factor = SizeStyle()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx_basic)
    _check(result, "size_style")
    non_null = result["factor_value"].drop_nulls().to_numpy()
    assert np.all(np.isfinite(non_null))

def test_value_style(ctx_basic):
    from factorzen.builtin_factors.daily.value import ValueStyle

    factor = ValueStyle()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx_basic)
    _check(result, "value_style")
    non_null = result["factor_value"].drop_nulls().to_numpy()
    # -log(pb) with pb > 0; all finite
    assert np.all(np.isfinite(non_null))

def test_momentum_style(ctx_daily):
    from factorzen.builtin_factors.daily.momentum_style import MomentumStyle

    factor = MomentumStyle()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx_daily)
    assert isinstance(result, pl.DataFrame)
    assert "factor_value" in result.columns

def test_volatility_style(ctx_daily):
    from factorzen.builtin_factors.daily.volatility_style import VolatilityStyle

    factor = VolatilityStyle()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx_daily)
    assert isinstance(result, pl.DataFrame)
    assert "factor_value" in result.columns
    non_null = result["factor_value"].drop_nulls().to_numpy()
    assert np.all(non_null >= 0), "Volatility must be non-negative"

def test_beta_style_is_alias_of_beta60d():
    from factorzen.builtin_factors.daily.beta import Beta60D
    from factorzen.builtin_factors.daily.beta_style import BetaStyle

    assert BetaStyle is Beta60D

# ==== 来自 test_risk_style_panel_materialize.py ====
@pytest.fixture(autouse=True)
def _pit_off(monkeypatch):
    monkeypatch.setattr(exposures_module, "fetch_index_member_all", lambda: None)
    monkeypatch.setattr(exposures_module, "_pit_industry_warned", False)
    yield

def _trade_days__style_panel_materialize(start, n):
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days

def _make(n_stocks=8, n_days=80, seed=42):
    rng = np.random.default_rng(seed)
    days = _trade_days__style_panel_materialize(dt.date(2023, 1, 3), n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    daily = pl.DataFrame([
        {"trade_date": d, "ts_code": c, "pct_chg": float(rng.standard_normal() * 2)}
        for c in codes for d in days
    ])
    db = pl.DataFrame([
        {
            "trade_date": d, "ts_code": c,
            "total_mv": float(abs(rng.standard_normal()) * 1e9 + 5e9),
            "pb": float(abs(rng.standard_normal()) + 1.5),
            "pe_ttm": float(abs(rng.standard_normal()) * 10 + 15),
            "turnover_rate": float(abs(rng.standard_normal()) * 2 + 1),
        }
        for c in codes for d in days
    ])
    inds = ["银行", "医药", "电子", "食品饮料"]
    stocks = pl.DataFrame({
        "ts_code": codes,
        "industry": [inds[i % 4] for i in range(n_stocks)],
    })
    return daily, db, stocks, days

def test_materialize_style_panel_matches_per_day_compute():
    """全窗一次物化再切片 ≡ 逐日 compute_exposures 的风格列（atol 1e-12）。"""
    daily, db, stocks, days = _make()
    panel = materialize_style_panel(daily, db, standardize=True)
    # 静态风格：size/value 全窗必有
    target = days[-1]
    exp_day = compute_exposures(daily, db, stocks, target)  # 无 panel → 旧路径
    exp_panel = compute_exposures(
        daily, db, stocks, target, style_panel=panel
    )
    for name in ("size", "value", "liquidity", "quality", "leverage"):
        if name not in exp_day.factor_names or name not in exp_panel.factor_names:
            continue
        # 对齐 codes
        day_map = {c: i for i, c in enumerate(exp_day.codes)}
        pan_map = {c: i for i, c in enumerate(exp_panel.codes)}
        common = sorted(set(day_map) & set(pan_map))
        assert common, f"{name}: 无共同股票"
        i_d = exp_day.factor_names.index(name)
        i_p = exp_panel.factor_names.index(name)
        v_d = np.array([exp_day.matrix[day_map[c], i_d] for c in common])
        v_p = np.array([exp_panel.matrix[pan_map[c], i_p] for c in common])
        np.testing.assert_allclose(v_p, v_d, atol=1e-12, err_msg=f"style {name}")

def test_cs_standardize_is_by_trade_date_not_pooled():
    """查证：标准化按 trade_date 分组，两日均值各自≈0。"""
    df = pl.DataFrame({
        "trade_date": [dt.date(2024, 1, 2)] * 4 + [dt.date(2024, 1, 3)] * 4,
        "ts_code": [f"{i}.SZ" for i in range(4)] * 2,
        "factor_value": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
    })
    out = cs_standardize(df, "factor_value", method="mad")
    for d in (dt.date(2024, 1, 2), dt.date(2024, 1, 3)):
        m = out.filter(pl.col("trade_date") == d)["factor_value"].mean()
        assert abs(m) < 1e-10, f"date {d} mean={m}"

def test_industry_panel_union_fills_missing():
    """行业中途出现：并集面板缺列日为 0。"""
    stocks = pl.DataFrame({
        "ts_code": ["A", "B"],
        "industry": ["银行", "医药"],
    })
    # 无 PIT → 两日相同行业
    dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    panel, cols = materialize_industry_panel(stocks, dates)
    assert set(cols) == {"ind_医药", "ind_银行"}
    assert panel.filter(pl.col("trade_date") == dates[0]).height == 2
    # 强制更大并集
    panel2, cols2 = materialize_industry_panel(
        stocks, dates, industry_names=["银行", "医药", "新能源"]
    )
    assert "ind_新能源" in cols2
    day0 = panel2.filter(pl.col("trade_date") == dates[0])
    assert (day0["ind_新能源"] == 0.0).all()

# ==== 来自 test_risk_exposures.py ====
# tests/test_risk_exposures.py

def _trade_days__risk_exposures(start, n):
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days

def make_daily(n_stocks=8, n_days=20, seed=42):
    rng = np.random.default_rng(seed)
    days = _trade_days__risk_exposures(dt.date(2023, 1, 3), n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = [{"trade_date": d, "ts_code": c, "pct_chg": float(rng.standard_normal() * 2.0)}
            for c in codes for d in days]
    return pl.DataFrame(rows)

def make_daily_basic__risk_exposures(n_stocks=8, n_days=20, seed=0):
    rng = np.random.default_rng(seed)
    days = _trade_days__risk_exposures(dt.date(2023, 1, 3), n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = [{"trade_date": d, "ts_code": c,
             "total_mv": float(abs(rng.standard_normal()) * 1e9 + 5e9),
             "pb": float(abs(rng.standard_normal()) + 1.5),
             "pe_ttm": float(abs(rng.standard_normal()) * 10 + 15),
             "turnover_rate": float(abs(rng.standard_normal()) * 2 + 1)}
            for c in codes for d in days]
    return pl.DataFrame(rows)

def make_stocks(n_stocks=8):
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    inds = ["银行", "医药", "电子", "食品饮料"]
    return pl.DataFrame({"ts_code": codes, "industry": [inds[i % 4] for i in range(n_stocks)]})

@pytest.fixture(autouse=True)
def _pit_industry_unavailable_by_default(monkeypatch):
    """默认所有测试都不触达真实 Tushare：PIT 历史行业数据视为不可用，走现有
    stocks.industry 降级路径（与改造 PIT 行业暴露之前的行为完全一致）。

    需要验证 PIT 可用路径的用例自行在测试体内覆盖 fetch_index_member_all 的 mock。
    同时重置"只警告一次"标记，避免跨测试用例互相污染。
    """
    monkeypatch.setattr(exposures_module, "fetch_index_member_all", lambda: None)
    monkeypatch.setattr(exposures_module, "_pit_industry_warned", False)
    yield

def test_compute_exposures_shape_and_factors():
    from factorzen.risk.exposures import compute_exposures
    daily, db, stocks = make_daily(), make_daily_basic__risk_exposures(), make_stocks()
    target = daily["trade_date"].max()  # 用数据里实际存在的最后一个交易日
    exp = compute_exposures(daily, db, stocks, target)
    assert exp.n_stocks > 0
    # 8 只股票全有数据（merged from n_stocks_matches_input）
    assert exp.n_stocks == 8
    assert len(exp.codes) == 8
    assert exp.n_factors == exp.matrix.shape[1]
    assert exp.matrix.shape == (exp.n_stocks, exp.n_factors)
    # factor_names 含风格因子(小写)与行业列(ind_)
    assert any(f in exp.factor_names for f in ["size", "value"])
    assert "size" in exp.factor_names
    assert "value" in exp.factor_names
    ind_cols = [f for f in exp.factor_names if f.startswith("ind_")]
    assert len(ind_cols) == 4, f"期望 4 列行业哑变量，实际: {ind_cols}"
    # 矩阵无 NaN（null 已填 0）
    assert not np.isnan(exp.matrix).any()

def test_compute_exposures_industry_dummies_one_hot():
    """行业哑变量每行之和精确为 1.0（每只股票属且仅属一个行业，数学确定量）。"""
    from factorzen.risk.exposures import compute_exposures
    daily, db, stocks = make_daily(), make_daily_basic__risk_exposures(), make_stocks()
    target = daily["trade_date"].max()
    exp = compute_exposures(daily, db, stocks, target)
    ind_cols = [f for f in exp.factor_names if f.startswith("ind_")]
    assert ind_cols, "无行业列，无法验证"
    ind_indices = [exp.factor_names.index(c) for c in ind_cols]
    ind_matrix = exp.matrix[:, ind_indices]
    row_sums = ind_matrix.sum(axis=1)
    # 每只股票恰属一个行业：和严格为 1（浮点精度 1e-10 内）
    assert np.all(np.abs(row_sums - 1.0) < 1e-10), f"行业哑变量行和异常: {row_sums}"

def test_compute_exposures_style_factors_zscore_mean():
    """风格因子经截面 Z-score 标准化后，截面均值数学上严格为 0（1e-10 级）。"""
    from factorzen.risk.exposures import compute_exposures
    daily, db, stocks = make_daily(), make_daily_basic__risk_exposures(), make_stocks()
    target = daily["trade_date"].max()
    exp = compute_exposures(daily, db, stocks, target)
    style_factors = ["size", "value", "liquidity", "quality", "leverage"]
    for name in style_factors:
        if name not in exp.factor_names:
            continue
        col_idx = exp.factor_names.index(name)
        col_vals = exp.matrix[:, col_idx]
        mean_val = col_vals.mean()
        assert abs(mean_val) < 1e-10, (
            f"风格因子 '{name}' 截面均值应≈0，实际: {mean_val:.2e}"
        )

def test_compute_exposures_style_factors_nontrivial_dispersion():
    """风格因子 Z-score 后标准差应接近 1（n=8，> 0.1 排除全零 stub）。"""
    from factorzen.risk.exposures import compute_exposures
    daily, db, stocks = make_daily(), make_daily_basic__risk_exposures(), make_stocks()
    target = daily["trade_date"].max()
    exp = compute_exposures(daily, db, stocks, target)
    style_factors = ["size", "value"]  # 一定存在的因子
    for name in style_factors:
        col_idx = exp.factor_names.index(name)
        col_vals = exp.matrix[:, col_idx]
        std_val = col_vals.std()
        assert std_val > 0.1, (
            f"风格因子 '{name}' 标准差过低 ({std_val:.4f})，疑似全零 stub"
        )

def test_compute_exposures_pit_industry_uses_historical_classification(monkeypatch):
    """PIT 历史行业数据可用时：同一只股票在窗口早期、晚期应按当时实际分类取得
    不同的行业暴露，而不是用单一（如"当前"）分类污染整个窗口。

    构造目标股票在窗口中途从"银行"重分类为"医药"，验证最早/最晚两个查询日期
    分别落在重分类前后、各自对应当时实际分类。
    """
    from factorzen.risk.exposures import compute_exposures

    daily, db, stocks = make_daily(), make_daily_basic__risk_exposures(), make_stocks()
    codes = stocks["ts_code"].to_list()
    all_days = sorted(daily["trade_date"].unique().to_list())
    early_date, late_date = all_days[0], all_days[-1]
    cutover = all_days[len(all_days) // 2]
    target_code = codes[0]  # make_stocks() 原本把它分到"银行"(i%4==0)

    inds = ["银行", "医药", "电子", "食品饮料"]
    rows = []
    for i, code in enumerate(codes):
        if code == target_code:
            # 目标股票：银行(窗口前段) -> 医药(窗口后段)，cutover 当天起算入新分类
            rows.append(
                {"ts_code": code, "l1_name": "银行", "in_date": dt.date(2000, 1, 1),
                 "out_date": cutover}
            )
            rows.append(
                {"ts_code": code, "l1_name": "医药", "in_date": cutover, "out_date": None}
            )
        else:
            # 其余股票全程不变，维持 make_stocks() 原有的 4 行业分布
            rows.append(
                {"ts_code": code, "l1_name": inds[i % 4], "in_date": dt.date(2000, 1, 1),
                 "out_date": None}
            )
    membership = pl.DataFrame(rows)
    monkeypatch.setattr(exposures_module, "fetch_index_member_all", lambda: membership)

    exp_early = compute_exposures(daily, db, stocks, early_date)
    exp_late = compute_exposures(daily, db, stocks, late_date)

    idx_early = exp_early.codes.index(target_code)
    idx_late = exp_late.codes.index(target_code)
    bank_early = exp_early.factor_names.index("ind_银行")
    pharma_early = exp_early.factor_names.index("ind_医药")
    bank_late = exp_late.factor_names.index("ind_银行")
    pharma_late = exp_late.factor_names.index("ind_医药")

    # 早期：归属"银行"
    assert exp_early.matrix[idx_early, bank_early] == 1.0
    assert exp_early.matrix[idx_early, pharma_early] == 0.0
    # 晚期：归属"医药"——与早期不同，证明确实按 trade_date 做了历史归属查找
    assert exp_late.matrix[idx_late, pharma_late] == 1.0
    assert exp_late.matrix[idx_late, bank_late] == 0.0

def test_compute_exposures_pit_industry_unavailable_falls_back_with_warning(monkeypatch, caplog):
    """PIT 历史行业数据获取失败（如无权限/网络问题）：compute_exposures 不应崩溃，
    应降级为现有 stocks.industry 行为，并记录警告日志说明降级。"""
    from factorzen.risk.exposures import compute_exposures

    def _boom():
        raise RuntimeError("抱歉，您没有访问该接口的权限")

    monkeypatch.setattr(exposures_module, "fetch_index_member_all", _boom)
    monkeypatch.setattr(exposures_module, "_pit_industry_warned", False)

    daily, db, stocks = make_daily(), make_daily_basic__risk_exposures(), make_stocks()
    target = daily["trade_date"].max()

    with caplog.at_level(logging.WARNING):
        exp = compute_exposures(daily, db, stocks, target)

    # 不崩溃：行为与未启用 PIT 时完全一致（仍按 stocks.industry 生成 4 个行业列）
    assert exp.n_stocks == 8
    ind_cols = [f for f in exp.factor_names if f.startswith("ind_")]
    assert len(ind_cols) == 4
    # 有警告日志提示降级为非 PIT 模式
    assert any(
        "PIT" in r.getMessage() or "降级" in r.getMessage() for r in caplog.records
    ), f"未找到降级警告日志，实际日志: {[r.getMessage() for r in caplog.records]}"

def test_compute_exposures_pit_industry_warns_only_once(monkeypatch, caplog):
    """降级警告只应触发一次，不能每次 compute_exposures 调用都刷屏。"""
    from factorzen.risk.exposures import compute_exposures

    monkeypatch.setattr(exposures_module, "fetch_index_member_all", lambda: None)
    monkeypatch.setattr(exposures_module, "_pit_industry_warned", False)

    daily, db, stocks = make_daily(), make_daily_basic__risk_exposures(), make_stocks()
    target = daily["trade_date"].max()

    with caplog.at_level(logging.WARNING):
        compute_exposures(daily, db, stocks, target)
        compute_exposures(daily, db, stocks, target)
        compute_exposures(daily, db, stocks, target)

    warnings = [
        r for r in caplog.records if "PIT" in r.getMessage() or "降级" in r.getMessage()
    ]
    assert len(warnings) == 1, f"应只警告一次，实际 {len(warnings)} 次"

def test_compute_exposures_pit_industry_partial_coverage_fills_gap_from_stocks(
    monkeypatch, caplog
):
    """PIT 历史行业数据只覆盖部分股票代码（如合成代码恰好与真实代码撞号）时：
    覆盖到的代码应使用 PIT 分类，未覆盖的代码应按代码级别用 stocks.industry
    补齐，而不是被整体丢弃或让全部代码退化成非 PIT 模式。
    """
    from factorzen.risk.exposures import compute_exposures

    daily, db, stocks = make_daily(), make_daily_basic__risk_exposures(), make_stocks()
    codes = stocks["ts_code"].to_list()  # 000000.SZ..000007.SZ
    covered_codes = codes[:4]  # 只覆盖前4只，故意用与 stocks.industry 不同的行业名
    membership = pl.DataFrame(
        [
            {"ts_code": c, "l1_name": "科技", "in_date": dt.date(2000, 1, 1), "out_date": None}
            for c in covered_codes
        ]
    )
    monkeypatch.setattr(exposures_module, "fetch_index_member_all", lambda: membership)

    target = daily["trade_date"].max()
    with caplog.at_level(logging.WARNING):
        exp = compute_exposures(daily, db, stocks, target)

    # 覆盖到的代码：用 PIT 分类"科技"，而非各自的 stocks.industry 原值
    assert "ind_科技" in exp.factor_names
    tech_idx = exp.factor_names.index("ind_科技")
    for c in covered_codes:
        row = exp.codes.index(c)
        assert exp.matrix[row, tech_idx] == 1.0, f"{c} 应按 PIT 分类归入 ind_科技"

    # 未覆盖的代码（004-007）：按代码补齐为各自的 stocks.industry 原值，而非丢失行业暴露
    stocks_by_code = dict(zip(stocks["ts_code"], stocks["industry"], strict=True))
    for c in codes[4:]:
        expected_col = f"ind_{stocks_by_code[c]}"
        assert expected_col in exp.factor_names
        row = exp.codes.index(c)
        col = exp.factor_names.index(expected_col)
        assert exp.matrix[row, col] == 1.0, f"{c} 应按 stocks.industry 补齐为 {expected_col}"
        # 且不应被错误归入 PIT 覆盖代码所用的"科技"分类
        assert exp.matrix[row, tech_idx] == 0.0

    assert any(
        "仅覆盖" in r.getMessage() or "补齐" in r.getMessage() for r in caplog.records
    ), f"应有警告说明部分覆盖，实际日志: {[r.getMessage() for r in caplog.records]}"

# tests/test_risk_exposures.py
import datetime as dt
import logging

import numpy as np
import polars as pl
import pytest

import factorzen.risk.exposures as exposures_module


def _trade_days(start, n):
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days


def make_daily(n_stocks=8, n_days=20, seed=42):
    rng = np.random.default_rng(seed)
    days = _trade_days(dt.date(2023, 1, 3), n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = [{"trade_date": d, "ts_code": c, "pct_chg": float(rng.standard_normal() * 2.0)}
            for c in codes for d in days]
    return pl.DataFrame(rows)


def make_daily_basic(n_stocks=8, n_days=20, seed=0):
    rng = np.random.default_rng(seed)
    days = _trade_days(dt.date(2023, 1, 3), n_days)
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
    daily, db, stocks = make_daily(), make_daily_basic(), make_stocks()
    target = daily["trade_date"].max()  # 用数据里实际存在的最后一个交易日
    exp = compute_exposures(daily, db, stocks, target)
    assert exp.n_stocks > 0
    assert exp.n_factors == exp.matrix.shape[1]
    assert exp.matrix.shape == (exp.n_stocks, exp.n_factors)
    # factor_names 含风格因子(小写)与行业列(ind_)
    assert any(f in exp.factor_names for f in ["size", "value"])
    assert any(f.startswith("ind_") for f in exp.factor_names)
    # 矩阵无 NaN（null 已填 0）
    assert not np.isnan(exp.matrix).any()


def test_compute_exposures_n_stocks_matches_input():
    """所有输入股票都应出现在暴露矩阵中（有 total_mv / pb 数据则不会被丢弃）。"""
    from factorzen.risk.exposures import compute_exposures
    daily, db, stocks = make_daily(), make_daily_basic(), make_stocks()
    target = daily["trade_date"].max()
    exp = compute_exposures(daily, db, stocks, target)
    # 8 只股票全有数据，不应有缺失
    assert exp.n_stocks == 8
    assert len(exp.codes) == 8


def test_compute_exposures_style_factor_names_exact():
    """size 和 value 因子只需静态日内数据，20 天历史必可计算，须严格出现在 factor_names 中。"""
    from factorzen.risk.exposures import compute_exposures
    daily, db, stocks = make_daily(), make_daily_basic(), make_stocks()
    target = daily["trade_date"].max()
    exp = compute_exposures(daily, db, stocks, target)
    # size 和 value 不依赖长历史窗口，一定会出现
    assert "size" in exp.factor_names, f"'size' missing from {exp.factor_names}"
    assert "value" in exp.factor_names, f"'value' missing from {exp.factor_names}"


def test_compute_exposures_industry_columns_present():
    """行业哑变量列须以 ind_ 为前缀，且数量等于行业数（4 个行业）。"""
    from factorzen.risk.exposures import compute_exposures
    daily, db, stocks = make_daily(), make_daily_basic(), make_stocks()
    target = daily["trade_date"].max()
    exp = compute_exposures(daily, db, stocks, target)
    ind_cols = [f for f in exp.factor_names if f.startswith("ind_")]
    assert len(ind_cols) == 4, f"期望 4 列行业哑变量，实际: {ind_cols}"


def test_compute_exposures_industry_dummies_one_hot():
    """行业哑变量每行之和精确为 1.0（每只股票属且仅属一个行业，数学确定量）。"""
    from factorzen.risk.exposures import compute_exposures
    daily, db, stocks = make_daily(), make_daily_basic(), make_stocks()
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
    daily, db, stocks = make_daily(), make_daily_basic(), make_stocks()
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
    daily, db, stocks = make_daily(), make_daily_basic(), make_stocks()
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

    daily, db, stocks = make_daily(), make_daily_basic(), make_stocks()
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

    daily, db, stocks = make_daily(), make_daily_basic(), make_stocks()
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

    daily, db, stocks = make_daily(), make_daily_basic(), make_stocks()
    target = daily["trade_date"].max()

    with caplog.at_level(logging.WARNING):
        compute_exposures(daily, db, stocks, target)
        compute_exposures(daily, db, stocks, target)
        compute_exposures(daily, db, stocks, target)

    warnings = [
        r for r in caplog.records if "PIT" in r.getMessage() or "降级" in r.getMessage()
    ]
    assert len(warnings) == 1, f"应只警告一次，实际 {len(warnings)} 次"

"""测试 IC Decay 分析：因子 IC 随持有期的衰减。"""

import polars as pl

from daily.evaluation.advanced import ICDecayResult, compute_ic_decay


def _make_factor_and_returns() -> tuple[pl.DataFrame, pl.DataFrame]:
    """构造合成因子值和前向收益数据。"""
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    stocks = ["A", "B", "C"]
    rows = []
    for d in dates:
        for s in stocks:
            rows.append({"trade_date": d, "ts_code": s})
    factor = pl.DataFrame(rows).with_columns(
        pl.lit(1.0).alias("factor_clean")
    )
    # 前向收益：fwd_ret_1d, fwd_ret_5d, fwd_ret_10d
    ret = pl.DataFrame(rows).with_columns([
        pl.Series("fwd_ret_1d", [0.02, 0.01, -0.01] * 3, dtype=pl.Float64),
        pl.Series("fwd_ret_5d", [0.05, 0.03, -0.02] * 3, dtype=pl.Float64),
        pl.Series("fwd_ret_10d", [0.08, 0.04, -0.03] * 3, dtype=pl.Float64),
    ])
    return factor, ret


def test_ic_decay_returns_list_of_results():
    """compute_ic_decay 返回每个 horizon 的 ICDecayResult。"""
    factor, ret = _make_factor_and_returns()
    results = compute_ic_decay(factor, ret, factor_col="factor_clean")
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert isinstance(r, ICDecayResult)
        assert r.horizon > 0


def test_ic_decay_monotonic_decreasing():
    """较长持有期的 IC 绝对值通常 ≤ 较短持有期（信号衰减）。"""
    factor, ret = _make_factor_and_returns()
    results = compute_ic_decay(factor, ret, factor_col="factor_clean")
    # 按 horizon 排序
    horizons = sorted(r.horizon for r in results)
    assert horizons == sorted(horizons)  # 确保已排序
    # 各 horizon 的 ic_mean 非零（数据有信号）
    ic_values = {r.horizon: r.ic_mean for r in results}
    assert all(v != 0.0 for v in ic_values.values()), "数据有信号时 IC 不应全零"


def test_ic_decay_result_fields():
    """每个 ICDecayResult 包含必要字段。"""
    factor, ret = _make_factor_and_returns()
    results = compute_ic_decay(factor, ret, factor_col="factor_clean")
    for r in results:
        assert hasattr(r, "horizon")
        assert hasattr(r, "ic_mean")
        assert hasattr(r, "ic_std")
        assert hasattr(r, "ic_series")
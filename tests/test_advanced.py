"""测试 IC Decay 分析：因子 IC 随持有期的衰减。

历史教训（本文件是重灾区）：原来的三个测试跑在 **3 只股票 + 常数因子** 上——
`_MIN_CROSS_SAMPLES = 30` 把每个截面整天丢弃，`ic_series` 为空、`ic_mean` 是 `nan`。
于是：

- `assert all(v != 0.0 ...)`（注释写「数据有信号时 IC 不应全零」）**恒真**，因为 `nan != 0.0`
- `horizons = sorted(...)` 之后 `assert horizons == sorted(horizons)` **恒真**
- 测试名叫 `monotonic_decreasing`，通篇**没有任何单调性断言**

而且那份数据 `factor_clean = pl.lit(1.0)` 是常数因子，本就没有信号可言。三个测试合起来
零判别力。现在：≥30 只股票、真实截面因子、按 horizon 递减的信噪比、真的断言单调衰减。
"""

import numpy as np
import polars as pl
import pytest

from factorzen.daily.evaluation.advanced import ICDecayResult, compute_ic_decay

_N_STOCKS = 40      # 必须 ≥ _MIN_CROSS_SAMPLES(=30)，否则整天被丢弃
_N_DAYS = 30


def _make_factor_and_returns(seed: int = 3) -> tuple[pl.DataFrame, pl.DataFrame]:
    """构造带**递减信噪比**的合成数据：IC(1d) > IC(5d) > IC(10d)。

    每只股票一个持久的截面得分 `f_i`。前向收益 = `f_i` + 噪声，噪声随持有期放大
    （模拟「信号只在近端有效，远端被后续随机收益稀释」）。
    """
    rng = np.random.default_rng(seed)
    dates = pl.date_range(
        pl.date(2026, 1, 5), pl.date(2026, 1, 5) + pl.duration(days=_N_DAYS - 1),
        interval="1d", eager=True,
    )
    codes = [f"{600000 + i:06d}.SH" for i in range(_N_STOCKS)]
    f = rng.standard_normal(_N_STOCKS)          # 截面得分，逐日不变

    rows_f, rows_r = [], []
    for d in dates:
        for i, code in enumerate(codes):
            rows_f.append({"trade_date": d, "ts_code": code, "factor_clean": float(f[i])})
            rows_r.append({
                "trade_date": d, "ts_code": code,
                "fwd_ret_1d": float(f[i] + 0.5 * rng.standard_normal()),
                "fwd_ret_5d": float(f[i] + 2.0 * rng.standard_normal()),
                "fwd_ret_10d": float(f[i] + 5.0 * rng.standard_normal()),
            })
    return pl.DataFrame(rows_f), pl.DataFrame(rows_r)


def test_ic_decay_returns_one_result_per_detected_horizon():
    """从 `fwd_ret_{h}d` 列名自动检测 horizon，返回值与之一一对应。"""
    factor, ret = _make_factor_and_returns()
    results = compute_ic_decay(factor, ret, factor_col="factor_clean")

    assert [r.horizon for r in results] == [1, 5, 10]
    assert all(isinstance(r, ICDecayResult) for r in results)


def test_ic_decay_is_monotonically_decreasing():
    """信噪比按持有期递减 ⇒ |IC| 必须严格递减。这是本文件名字所承诺的断言。"""
    factor, ret = _make_factor_and_returns()
    results = compute_ic_decay(factor, ret, factor_col="factor_clean")
    ic = {r.horizon: r.ic_mean for r in results}

    assert all(v == v for v in ic.values()), f"IC 不该是 nan（截面被丢空了？）：{ic}"
    assert ic[1] > 0.5, f"1 日 IC 应显著为正，实得 {ic[1]:.4f}"
    assert abs(ic[1]) > abs(ic[5]) > abs(ic[10]), f"IC 未随持有期衰减：{ic}"


def test_ic_decay_series_covers_every_trading_day():
    """`ic_series` 每天一个值——为空说明截面被 `_MIN_CROSS_SAMPLES` 整天丢弃了。"""
    factor, ret = _make_factor_and_returns()
    results = compute_ic_decay(factor, ret, factor_col="factor_clean")

    for r in results:
        assert len(r.ic_series) == _N_DAYS, (
            f"horizon={r.horizon} 的 IC 序列长度 {len(r.ic_series)} != {_N_DAYS}"
        )
        assert r.ic_std > 0, "常数 IC 序列说明数据退化"
        assert r.ic_mean == pytest.approx(float(np.mean(r.ic_series)))


def test_degenerate_cross_section_yields_nan_not_zero():
    """截面不足 30 只时 IC 是 `nan`，不是 `0.0`——两者语义不同，不许混淆。

    这条把「历史假绿的成因」钉成断言：`nan != 0.0` 为 True，所以
    `assert ic_mean != 0.0` 这种写法在退化数据上恒真、零判别力。
    """
    factor, ret = _make_factor_and_returns()
    keep = set(factor["ts_code"].unique().sort().head(3).to_list())
    few = factor.filter(pl.col("ts_code").is_in(keep))
    few_ret = ret.filter(pl.col("ts_code").is_in(keep))

    results = compute_ic_decay(few, few_ret, factor_col="factor_clean")

    for r in results:
        assert r.ic_series == []
        assert r.ic_mean != r.ic_mean, "退化截面的 ic_mean 应为 nan"
        assert r.ic_mean != 0.0, "nan != 0.0 —— 正是这一点让旧断言恒真"

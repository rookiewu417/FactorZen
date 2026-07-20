"""`_evaluate_oos` 的带成本净收益列。

**为什么加这两列**：2026-07-19 实测——库 120 上 lgbm 毛年化 +30.30%、换手 55.3%/日，
A 股 10bp/边下成本年化 27.9%，**吃掉毛 alpha 的 92%**，净仅 +2.44%；
四方法在现实成本下净收益全部为负或贴零。报告只列 IC/换手时，
**IC 最高的方法（lgbm）恰好换手也最高**，按 IC 选会系统性高估可部署性。

**测试设计**：本文件刻意**不用被测函数的输出去构造期望值**
（CLAUDE.md 反复踩的陷阱 #1：`C` 由 `A`、`B` 构造再断言 `C=f(A,B)` 恒真零判别力）。
所有期望值由**手工构造的面板独立算出**：面板被设计成每日 spread 与每期换手
都是事先知道的常数，故净收益有闭式解。
"""
from __future__ import annotations

import polars as pl
import pytest

from factorzen.research.combination.experiment import (
    _COST_PER_SIDE,
    _evaluate_oos,
    _top_bucket_turnover_series,
)

N_GROUPS = 5


def _panel(day_specs: list[list[tuple[str, float, float]]]):
    """day_specs[i] = 第 i 日的 [(ts_code, factor_value, ret), ...]。"""
    rows_f, rows_r = [], []
    for i, day in enumerate(day_specs):
        d = f"2024-01-{i + 1:02d}"
        for code, fv, rv in day:
            rows_f.append({"trade_date": d, "ts_code": code, "factor_value": fv})
            rows_r.append({"trade_date": d, "ts_code": code, "ret": rv})
    return pl.DataFrame(rows_f), pl.DataFrame(rows_r)


def _stable_day(d_idx: int, top_codes: list[str]):
    """10 只票：`top_codes` 是分数最高的 2 只（top 1/5 桶），收益 +1%；其余 0。

    分数与收益都写死 ⇒ 每日 spread = top均值 − bottom均值 = 0.01 − 0.0 = 0.01，
    与被测函数无关，可用作独立 ground-truth。
    """
    day = []
    others = [f"S{j}" for j in range(10) if f"S{j}" not in top_codes]
    for c in top_codes:
        day.append((c, 10.0, 0.01))
    # bottom 2 只分数最低、收益 0；中间 6 只分数居中、收益 0
    for rank, c in enumerate(others):
        day.append((c, float(rank), 0.0))
    return day


def test_zero_turnover_net_equals_gross():
    """持仓完全不变 ⇒ 换手 0 ⇒ 净 spread 必须**逐位等于**毛 spread。"""
    days = [_stable_day(i, ["S0", "S1"]) for i in range(6)]
    combined, ret_df = _panel(days)
    out = _evaluate_oos(combined, ret_df, n_groups=N_GROUPS)

    assert out["turnover"] == pytest.approx(0.0)
    assert out["net_spread_10bp"] == pytest.approx(out["top_bottom_spread"])
    # 独立 ground-truth：spread 由构造决定 = 0.01
    assert out["top_bottom_spread"] == pytest.approx(0.01, abs=1e-12)


def test_full_turnover_charges_expected_fee():
    """每期 top 桶**整桶换掉** ⇒ 换手 1.0 ⇒ 净 = 毛 − 4×1.0×10bp。

    期望值由构造独立给出（毛 0.01、换手 1.0），不引用被测函数的中间量。
    """
    tops = [["S0", "S1"], ["S2", "S3"], ["S4", "S5"], ["S6", "S7"]]
    combined, ret_df = _panel([_stable_day(i, t) for i, t in enumerate(tops)])
    out = _evaluate_oos(combined, ret_df, n_groups=N_GROUPS)

    assert out["turnover"] == pytest.approx(1.0)
    # 4 天：第 0 天不扣费（无前一期），后 3 天各扣 4×1.0×0.001
    fee_per_day = 4.0 * 1.0 * _COST_PER_SIDE
    expected = 0.01 - (3 * fee_per_day) / 4
    assert out["net_spread_10bp"] == pytest.approx(expected, abs=1e-12)
    assert out["net_spread_10bp"] < out["top_bottom_spread"]


def test_half_turnover_is_between():
    """换手 0.5（2 只里换 1 只）⇒ 费用恰为全换的一半。"""
    tops = [["S0", "S1"], ["S1", "S2"], ["S2", "S3"], ["S3", "S4"]]
    combined, ret_df = _panel([_stable_day(i, t) for i, t in enumerate(tops)])
    out = _evaluate_oos(combined, ret_df, n_groups=N_GROUPS)

    assert out["turnover"] == pytest.approx(0.5)
    fee_per_day = 4.0 * 0.5 * _COST_PER_SIDE
    expected = 0.01 - (3 * fee_per_day) / 4
    assert out["net_spread_10bp"] == pytest.approx(expected, abs=1e-12)


def test_turnover_series_matches_aggregate():
    """逐期序列的均值必须等于聚合版——两者是同一口径的两种取法。

    净收益必须逐期扣费再平均（换手与收益可能相关），
    但**均值口径**上二者应一致，此断言守住这个不变量。
    """
    import numpy as np

    from factorzen.research.combination.experiment import _top_bucket_turnover

    tops = [frozenset({"a", "b"}), frozenset({"b", "c"}),
            frozenset({"c", "d"}), frozenset({"c", "d"})]
    series = _top_bucket_turnover_series(tops)
    assert series == pytest.approx([0.5, 0.5, 0.0])
    assert float(np.mean(series)) == pytest.approx(_top_bucket_turnover(tops))


def test_empty_panel_has_net_keys():
    """空面板也须带上新键，否则消费方 `r['net_spread_10bp']` 会 KeyError。"""
    empty = pl.DataFrame(
        {"trade_date": [], "ts_code": [], "factor_value": []},
        schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8, "factor_value": pl.Float64},
    )
    ret_df = pl.DataFrame(
        {"trade_date": [], "ts_code": [], "ret": []},
        schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8, "ret": pl.Float64},
    )
    out = _evaluate_oos(empty, ret_df)
    assert out["net_spread_10bp"] == 0.0
    assert out["net_sharpe_10bp"] == 0.0


def test_net_sharpe_zero_when_no_variance():
    """净收益方差为 0（恒定）⇒ SR 定义为 0，不得抛除零或返 inf/nan。"""
    days = [_stable_day(i, ["S0", "S1"]) for i in range(6)]
    combined, ret_df = _panel(days)
    out = _evaluate_oos(combined, ret_df, n_groups=N_GROUPS)
    assert out["net_sharpe_10bp"] == 0.0

"""估权/应用拆分 + 滚动 OOS 组合器(含泄漏探针)的测试。"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from factorzen.research.combination.cv import PurgedWalkForwardCV
from factorzen.research.combination.methods import (
    apply_weights,
    equal_weight,
    estimate_ic_weights,
    ic_weighted,
)
from factorzen.research.combination.oos import combine_oos


def _panel(n_days=120, n_stocks=30, seed=0):
    """两因子面板:a 与 ret 强相关(IC 高),b 弱相关。"""
    rng = np.random.default_rng(seed)
    dates = [f"2025{1 + i // 28:02d}{1 + i % 28:02d}" for i in range(n_days)]
    ra, rb, rr = [], [], []
    for d in dates:
        fa = rng.standard_normal(n_stocks)
        fb = rng.standard_normal(n_stocks)
        ret = 0.6 * fa + 0.1 * fb + rng.standard_normal(n_stocks) * 0.5
        for s in range(n_stocks):
            c = f"{s:04d}.SZ"
            ra.append({"trade_date": d, "ts_code": c, "factor_value": float(fa[s])})
            rb.append({"trade_date": d, "ts_code": c, "factor_value": float(fb[s])})
            rr.append({"trade_date": d, "ts_code": c, "ret": float(ret[s])})
    return {"a": pl.DataFrame(ra), "b": pl.DataFrame(rb)}, pl.DataFrame(rr), dates


def _single_day(vals_a, vals_b):
    codes = [str(i) for i in range(len(vals_a))]
    dfa = pl.DataFrame({"trade_date": ["d"] * len(codes), "ts_code": codes, "factor_value": vals_a})
    dfb = pl.DataFrame({"trade_date": ["d"] * len(codes), "ts_code": codes, "factor_value": vals_b})
    return dfa, dfb


# ── 拆分:apply / estimate ───────────────────────────────
def test_apply_weights_ground_truth():
    dfa, dfb = _single_day([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    out = apply_weights({"a": dfa, "b": dfb}, {"a": 0.7, "b": 0.3}).sort("ts_code")
    # z_a=[-1,0,1] z_b=[1,0,-1] → 0.7*z_a+0.3*z_b = [-0.4,0,0.4]
    assert out["factor_value"].to_list() == pytest.approx([-0.4, 0.0, 0.4])


def test_equal_weight_is_mean():
    dfa, dfb = _single_day([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    out = equal_weight({"a": dfa, "b": dfb}).sort("ts_code")
    assert out["factor_value"].to_list() == pytest.approx([0.0, 0.0, 0.0])


def test_estimate_ic_weights_favors_high_ic():
    factor_dfs, ret_df, _ = _panel()
    w = estimate_ic_weights(factor_dfs, ret_df, ic_window=-1)
    assert w["a"] > w["b"]
    assert sum(w.values()) == pytest.approx(1.0)


def test_estimate_ic_weights_all_negative_degenerates_equal():
    _, ret_df, _ = _panel()
    # 因子 = -ret → IC=-1 全负 → max(0,ic)=0 → 退化等权
    neg = ret_df.rename({"ret": "factor_value"}).with_columns(
        (pl.col("factor_value") * -1).alias("factor_value")
    )
    w = estimate_ic_weights({"a": neg, "b": neg}, ret_df, ic_window=-1)
    assert w == pytest.approx({"a": 0.5, "b": 0.5})


def test_ic_weighted_equals_estimate_then_apply():
    """薄包装等价:ic_weighted == apply_weights(estimate_ic_weights)。"""
    factor_dfs, ret_df, _ = _panel()
    direct = ic_weighted(factor_dfs, ret_df, ic_window=60).sort(["trade_date", "ts_code"])
    w = estimate_ic_weights(factor_dfs, ret_df, ic_window=60)
    via = apply_weights(factor_dfs, w).sort(["trade_date", "ts_code"])
    assert_frame_equal(direct, via)


# ── OOS 组合器 ──────────────────────────────────────────
def test_combine_oos_covers_test_and_has_fold_id():
    factor_dfs, ret_df, dates = _panel(n_days=120)
    cv = PurgedWalkForwardCV(train_days=40, test_days=20, purge_days=5)
    out = combine_oos(factor_dfs, ret_df, cv, method="ic_weighted")
    assert set(out.columns) >= {"trade_date", "ts_code", "factor_value", "fold_id"}
    assert set(out["trade_date"].to_list()) == set(dates[40:120])


def test_combine_oos_no_lookahead():
    """泄漏探针:扰动 cutoff 之后的收益,cutoff 之前的 OOS 组合值必须逐行不变。"""
    factor_dfs, ret_df, dates = _panel(n_days=120)
    cv = PurgedWalkForwardCV(train_days=40, test_days=20, purge_days=5)
    base = combine_oos(factor_dfs, ret_df, cv, method="ic_weighted")
    cutoff = dates[79]
    tampered_ret = ret_df.with_columns(
        pl.when(pl.col("trade_date") > cutoff)
        .then(pl.col("ret") * -3.0)
        .otherwise(pl.col("ret"))
        .alias("ret")
    )
    tampered = combine_oos(factor_dfs, tampered_ret, cv, method="ic_weighted")
    b = base.filter(pl.col("trade_date") <= cutoff).sort(["trade_date", "ts_code"])
    t = tampered.filter(pl.col("trade_date") <= cutoff).sort(["trade_date", "ts_code"])
    assert_frame_equal(b, t)


def test_combine_oos_equal_weight_also_probes_clean():
    """等权同样过泄漏探针(天然 OOS,但走同一接口)。"""
    factor_dfs, ret_df, dates = _panel(n_days=120)
    cv = PurgedWalkForwardCV(train_days=40, test_days=20, purge_days=5)
    out = combine_oos(factor_dfs, ret_df, cv, method="equal_weight")
    assert set(out["trade_date"].to_list()) == set(dates[40:120])


def test_combine_oos_unknown_method_raises():
    factor_dfs, ret_df, _ = _panel(n_days=80)
    cv = PurgedWalkForwardCV(train_days=40, test_days=20, purge_days=5)
    with pytest.raises(ValueError):
        combine_oos(factor_dfs, ret_df, cv, method="nonsense")

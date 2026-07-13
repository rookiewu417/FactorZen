"""LightGBM 组合器的测试:可学习性 / 确定性 / 泄漏探针 / 缺值处理。"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from factorzen.research.combination.cv import PurgedWalkForwardCV
from factorzen.research.combination.models import build_panel, combine_lgbm


def _panel(n_days=200, n_stocks=50, seed=0):
    """ret = 0.8*fa - 0.4*fb + 噪声:fa 正贡献强、fb 负贡献。"""
    rng = np.random.default_rng(seed)
    dates = [f"2025{1 + i // 28:02d}{1 + i % 28:02d}" for i in range(n_days)]
    ra, rb, rr = [], [], []
    for d in dates:
        fa = rng.standard_normal(n_stocks)
        fb = rng.standard_normal(n_stocks)
        ret = 0.8 * fa - 0.4 * fb + rng.standard_normal(n_stocks) * 0.3
        for s in range(n_stocks):
            c = f"{s:04d}.SZ"
            ra.append({"trade_date": d, "ts_code": c, "factor_value": float(fa[s])})
            rb.append({"trade_date": d, "ts_code": c, "factor_value": float(fb[s])})
            rr.append({"trade_date": d, "ts_code": c, "ret": float(ret[s])})
    return {"fa": pl.DataFrame(ra), "fb": pl.DataFrame(rb)}, pl.DataFrame(rr), dates


def _oos_rank_ic(combined: pl.DataFrame, ret_df: pl.DataFrame) -> float:
    m = combined.join(
        ret_df.with_columns(pl.col("trade_date").cast(pl.Utf8)),
        on=["trade_date", "ts_code"],
        how="inner",
    )
    ics = []
    for _d, g in m.group_by("trade_date"):
        if len(g) < 10:
            continue
        f = g["factor_value"].to_numpy()
        r = g["ret"].to_numpy()
        fr = f.argsort().argsort().astype(float)
        rr = r.argsort().argsort().astype(float)
        ic = float(np.corrcoef(fr, rr)[0, 1])
        if np.isfinite(ic):
            ics.append(ic)
    return float(np.mean(ics))


def test_lgbm_learns_signal():
    factor_dfs, ret_df, _ = _panel()
    cv = PurgedWalkForwardCV(train_days=60, test_days=20, purge_days=5)
    out = combine_lgbm(factor_dfs, ret_df, cv, min_child_samples=20, n_estimators=80)
    assert _oos_rank_ic(out, ret_df) > 0.15


def test_lgbm_deterministic():
    factor_dfs, ret_df, _ = _panel(n_days=120, n_stocks=30)
    cv = PurgedWalkForwardCV(train_days=60, test_days=20, purge_days=5)
    a = combine_lgbm(factor_dfs, ret_df, cv, seed=7, n_estimators=50).sort(
        ["trade_date", "ts_code"]
    )
    b = combine_lgbm(factor_dfs, ret_df, cv, seed=7, n_estimators=50).sort(
        ["trade_date", "ts_code"]
    )
    assert_frame_equal(a, b)


def test_lgbm_no_lookahead():
    """泄漏探针:扰动 cutoff 后收益,cutoff 前 OOS 预测逐行不变。"""
    factor_dfs, ret_df, dates = _panel(n_days=120, n_stocks=30)
    cv = PurgedWalkForwardCV(train_days=60, test_days=20, purge_days=5)
    base = combine_lgbm(factor_dfs, ret_df, cv, seed=1, n_estimators=50)
    cutoff = dates[99]
    tampered_ret = ret_df.with_columns(
        pl.when(pl.col("trade_date") > cutoff)
        .then(pl.col("ret") * -3.0)
        .otherwise(pl.col("ret"))
        .alias("ret")
    )
    tampered = combine_lgbm(factor_dfs, tampered_ret, cv, seed=1, n_estimators=50)
    b = base.filter(pl.col("trade_date") <= cutoff).sort(["trade_date", "ts_code"])
    t = tampered.filter(pl.col("trade_date") <= cutoff).sort(["trade_date", "ts_code"])
    assert_frame_equal(b, t)


def test_build_panel_inner_join_and_ret():
    factor_dfs, ret_df, _ = _panel(n_days=30, n_stocks=20)
    panel = build_panel(factor_dfs, ret_df)
    assert set(panel.columns) >= {"trade_date", "ts_code", "fa", "fb", "ret"}
    assert panel.height > 0


@pytest.mark.filterwarnings("ignore:build_panel")
def test_lgbm_drops_all_null_factor_and_continues():
    """一个因子全缺 → 丢弃它、用其余因子照常组合(健壮性:不因坏因子崩整个 run)。"""
    factor_dfs, ret_df, _ = _panel(n_days=80, n_stocks=30)
    factor_dfs["fa"] = factor_dfs["fa"].with_columns(
        pl.lit(None, dtype=pl.Float64).alias("factor_value")
    )
    cv = PurgedWalkForwardCV(train_days=40, test_days=20, purge_days=5)
    out = combine_lgbm(factor_dfs, ret_df, cv, n_estimators=20)
    assert out.height > 0  # fb 仍在 → 正常产出

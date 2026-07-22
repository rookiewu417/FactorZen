"""S1：strategies → sim 桥离线闭环。

微型合成 daily + 指数 → 生成 weights 产物 → ``run_strategy_simulation`` →
断言 sim run_dir 落 nav.parquet / metrics.json，nav 有限且非全零。
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.strategies.momentum_rotation import generate_momentum_rotation_products
from factorzen.strategies.runner import run_strategy_simulation
from factorzen.strategies.trend_timing import generate_trend_timing_products


def _dates(n: int, start: date = date(2023, 1, 3)) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _fake_daily(dates: list[date], codes: list[str], *, seed: int = 0) -> pl.DataFrame:
    """合成日线：close 有小幅随机游走，保证 nav 非全零且有限。"""
    rng = np.random.default_rng(seed)
    rows = []
    for c in codes:
        px = 10.0
        for d in dates:
            ret = float(rng.normal(0.001, 0.01))
            open_px = px
            close = px * (1.0 + ret)
            rows.append(
                {
                    "trade_date": d,
                    "ts_code": c,
                    "open": open_px,
                    "high": max(open_px, close) * 1.01,
                    "low": min(open_px, close) * 0.99,
                    "close": close,
                    "pre_close": px,
                    "change": close - px,
                    "pct_chg": ret * 100.0,
                    "vol": 1e6,
                    "amount": 1e9,
                }
            )
            px = close
    return pl.DataFrame(rows)


def _assert_sim_nav(res: dict) -> None:
    run_dir = Path(res["run_dir"])
    assert (run_dir / "nav.parquet").exists(), "nav.parquet missing"
    assert (run_dir / "metrics.json").exists(), "metrics.json missing"
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    for k in ("ann_ret", "sharpe", "max_dd"):
        assert k in metrics, f"metrics.json missing {k}"
    nav = pl.read_parquet(run_dir / "nav.parquet")
    assert "nav" in nav.columns and nav.height > 0
    vals = nav["nav"].to_numpy()
    assert np.isfinite(vals).all(), "nav 含非有限值"
    assert not np.allclose(vals, 0.0), "nav 不应全零"
    assert res.get("run_dir")


def test_trend_timing_to_sim(tmp_path):
    """trend_timing 产物 → run_strategy_simulation 产出有效 nav。"""
    n = 20
    dates = _dates(n)
    codes = ["A.SZ", "B.SZ", "C.SZ"]
    # 指数单调上行 → risk-on（close > MA）
    idx = pl.DataFrame(
        {"trade_date": dates, "close": [10.0 + i * 0.5 for i in range(n)]}
    )
    daily = _fake_daily(dates, codes)
    # signal 不能落在末日（T+1 才执行）
    rebalance = [dates[8], dates[12]]

    def members(_code: str, _ds: str) -> list[str]:
        return codes

    run_dirs = generate_trend_timing_products(
        str(tmp_path / "tt_products"),
        idx,
        daily,
        rebalance,
        members_fn=members,
        ma_window=5,
        top_n=2,
        timing=True,
    )
    assert run_dirs
    # 至少一期非空权重
    non_empty = any(
        pl.read_parquet(Path(rd) / "weights.parquet").height > 0 for rd in run_dirs
    )
    assert non_empty, "risk-on 应至少产出一期非空权重"

    res = run_strategy_simulation(
        run_dirs,
        daily,
        out_dir=str(tmp_path / "sim"),
        run_id="tt_sim",
    )
    _assert_sim_nav(res)


def test_momentum_rotation_to_sim(tmp_path):
    """momentum_rotation 产物 → run_strategy_simulation 产出有效 nav。"""
    n = 20
    dates = _dates(n)
    codes = ["A1.SZ", "A2.SZ", "B1.SZ", "B2.SZ"]
    # IDXA 强动量，IDXB 弱
    idx_a = pl.DataFrame(
        {"trade_date": dates, "close": [10.0 + i * 0.3 for i in range(n)]}
    )
    idx_b = pl.DataFrame(
        {"trade_date": dates, "close": [10.0 + i * 0.05 for i in range(n)]}
    )
    daily = _fake_daily(dates, codes, seed=1)
    rebalance = [dates[10], dates[14]]

    def members(code: str, _ds: str) -> list[str]:
        return {"IDXA": ["A1.SZ", "A2.SZ"], "IDXB": ["B1.SZ", "B2.SZ"]}[code]

    run_dirs = generate_momentum_rotation_products(
        str(tmp_path / "mr_products"),
        {"IDXA": idx_a, "IDXB": idx_b},
        daily,
        rebalance,
        members_fn=members,
        lookback=5,
        top_n=2,
    )
    assert run_dirs
    non_empty = any(
        pl.read_parquet(Path(rd) / "weights.parquet").height > 0 for rd in run_dirs
    )
    assert non_empty, "正动量应至少产出一期非空权重"

    res = run_strategy_simulation(
        run_dirs,
        daily,
        out_dir=str(tmp_path / "sim"),
        run_id="mr_sim",
    )
    _assert_sim_nav(res)

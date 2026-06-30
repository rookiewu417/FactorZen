"""Smoke tests for sim/engine.py — TDD RED phase."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorzen.sim.engine import run_portfolio_simulation


def _write_portfolio_dir(tmp_path, run_id, codes, weights, sig_date):
    d = tmp_path / run_id
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "ts_code": codes,
            "target_weight": weights,
            "prev_weight": [0.0] * len(codes),
        }
    ).write_parquet(d / "weights.parquet")
    (d / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "signal_date": sig_date})
    )
    return str(d)


def _fake_daily(codes, start="20230101", end="20230228"):
    """构造 mock 日线数据（不连接真实数据源）。"""
    dates = pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 2, 28), "1d", eager=True)
    rng = np.random.default_rng(0)
    rows = []
    for c in codes:
        for dt in dates:
            rows.append(
                {
                    "trade_date": dt,
                    "ts_code": c,
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.5,
                    "close": 10.0,
                    "pre_close": 10.0,
                    "change": 0.0,
                    "pct_chg": float(rng.normal(0, 1)),
                    "vol": 1e6,
                    "amount": 1e7,
                }
            )
    return pl.DataFrame(rows)


def test_run_portfolio_simulation_produces_metrics(tmp_path: Path):
    """nav.parquet / metrics.json / manifest.json 落盘，返回 sharpe/max_dd/ann_ret。"""
    codes = ["000001.SZ", "000002.SZ"]
    p1 = _write_portfolio_dir(tmp_path, "p1", codes, [0.5, 0.5], "2023-01-10")
    daily = _fake_daily(codes)
    res = run_portfolio_simulation(
        [p1], daily, out_dir=str(tmp_path / "sim"), run_id="s1"
    )
    run_dir = Path(res["run_dir"])
    assert (run_dir / "nav.parquet").exists(), "nav.parquet missing"
    assert (run_dir / "metrics.json").exists(), "metrics.json missing"
    assert (run_dir / "manifest.json").exists(), "manifest.json missing"

    m = json.loads((run_dir / "metrics.json").read_text())
    # summary_stats["portfolio"] 包含这三个键
    for k in ["ann_ret", "sharpe", "max_dd"]:
        assert k in m, f"metrics.json missing key: {k}"

    assert "sharpe" in res, "返回 dict 缺少 sharpe"
    assert "max_dd" in res, "返回 dict 缺少 max_dd"
    assert "ann_ret" in res, "返回 dict 缺少 ann_ret"


def test_run_portfolio_simulation_multiple_signals(tmp_path: Path):
    """多个 signal_date 时仍能正常落盘。"""
    codes = ["000001.SZ", "000002.SZ"]
    p1 = _write_portfolio_dir(tmp_path, "r1", codes, [0.5, 0.5], "2023-01-10")
    p2 = _write_portfolio_dir(tmp_path, "r2", codes, [0.3, 0.7], "2023-02-01")
    daily = _fake_daily(codes)
    res = run_portfolio_simulation(
        [p1, p2], daily, out_dir=str(tmp_path / "sim2"), run_id="multi"
    )
    run_dir = Path(res["run_dir"])
    assert (run_dir / "nav.parquet").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["n_signals"] == 2


def test_run_portfolio_simulation_nav_is_parquet(tmp_path: Path):
    """nav.parquet 可被 polars 读取且含 nav 列。"""
    codes = ["000001.SZ", "000002.SZ"]
    p1 = _write_portfolio_dir(tmp_path, "px", codes, [0.5, 0.5], "2023-01-10")
    daily = _fake_daily(codes)
    res = run_portfolio_simulation(
        [p1], daily, out_dir=str(tmp_path / "sim3"), run_id="navtest"
    )
    nav_df = pl.read_parquet(Path(res["run_dir"]) / "nav.parquet")
    assert "nav" in nav_df.columns, f"nav 列缺失, 有: {nav_df.columns}"
    assert "trade_date" in nav_df.columns


def test_run_portfolio_simulation_warns_when_signal_date_after_trade_dates(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Fix 2: signal_date 晚于回测末日时，应发出 warning，不抛错。"""
    codes = ["000001.SZ"]
    # 信号日 2023-03-01 > 数据末日 2023-02-28 → 权重永不生效 → nav 为空 → 应 warning
    p1 = _write_portfolio_dir(tmp_path, "late", codes, [1.0], "2023-03-01")
    daily = _fake_daily(codes)  # 数据截止 2023-02-28
    with caplog.at_level(logging.WARNING, logger="factorzen.sim.engine"):
        run_portfolio_simulation(
            [p1], daily, out_dir=str(tmp_path / "sim_warn"), run_id="sw1"
        )
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "signal_date" in m and "调仓" in m for m in warning_messages
    ), f"未找到预期 warning，记录到的 warning: {warning_messages}"

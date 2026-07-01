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


# ── 代码评审修复回归测试 ───────────────────────────────────────────────────


def test_run_portfolio_simulation_charges_trading_cost_on_turnover(tmp_path: Path):
    """修复1：3 次大幅调仓（模拟高换手）后，nav.parquet 的 cost 列与 metrics.json
    的 total_cost 不应再恒为 0（此前 run_strategy_backtest 调用未传 cost_model，
    fast path 在 cost_model=None 时永远不计交易成本）。
    """
    codes = ["000001.SZ", "000002.SZ", "000003.SZ"]
    # 三次几乎全仓轮动（c0 → c1 → c2），制造大额换手
    p1 = _write_portfolio_dir(tmp_path, "c1", codes, [1.0, 0.0, 0.0], "2023-01-10")
    p2 = _write_portfolio_dir(tmp_path, "c2", codes, [0.0, 1.0, 0.0], "2023-01-20")
    p3 = _write_portfolio_dir(tmp_path, "c3", codes, [0.0, 0.0, 1.0], "2023-02-05")
    daily = _fake_daily(codes)
    res = run_portfolio_simulation(
        [p1, p2, p3], daily, out_dir=str(tmp_path / "sim_cost"), run_id="costcheck"
    )
    run_dir = Path(res["run_dir"])
    nav_df = pl.read_parquet(run_dir / "nav.parquet")
    assert "cost" in nav_df.columns
    assert float(nav_df["cost"].sum()) > 0, (
        "高换手调仓后 nav.parquet 的 cost 列仍恒为 0（cost_model 未生效）"
    )

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics.get("total_cost", 0.0) > 0, "metrics.json 的 total_cost 仍为 0"


def test_run_portfolio_simulation_passes_is_st_by_date_to_backtest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """修复4（ST涨跌停容差接线）：run_portfolio_simulation 应基于 daily 的
    codes/trade_dates 构建 is_st_by_date 并传给 run_strategy_backtest，而不是
    让其默认为 None——此前即使 backtest.py 已支持该参数，sim 从不构造真实值
    传入，ST 股票涨跌停判断永远退化为非 ST 阈值。
    """
    import factorzen.sim.engine as engine_mod

    codes = ["000001.SZ"]
    p1 = _write_portfolio_dir(tmp_path, "p1", codes, [1.0], "2023-01-10")
    daily = _fake_daily(codes)

    captured: dict = {}
    real_run = engine_mod.run_strategy_backtest

    def _capturing_run(*args, **kwargs):
        captured.update(kwargs)
        return real_run(*args, **kwargs)

    sentinel = {daily["trade_date"][0]: {"000001.SZ"}}
    monkeypatch.setattr(engine_mod, "run_strategy_backtest", _capturing_run)
    monkeypatch.setattr(engine_mod, "build_is_st_by_date", lambda codes, dates: sentinel)

    engine_mod.run_portfolio_simulation(
        [p1], daily, out_dir=str(tmp_path / "sim_st"), run_id="st1"
    )

    assert captured.get("is_st_by_date") == sentinel, (
        "run_strategy_backtest 应收到由 build_is_st_by_date 构建的 is_st_by_date，"
        f"实际收到: {captured.get('is_st_by_date')!r}"
    )


def test_load_weights_by_date_skips_non_optimal_status(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """修复2：manifest.status 非成功状态（如 infeasible 兜底全零持仓）时，
    该 run_dir 应被跳过、不进入 weights_by_date，且应有 warning 说明原因。
    """
    from datetime import date as _date

    from factorzen.sim.engine import _load_weights_by_date

    codes = ["000001.SZ"]
    good = _write_portfolio_dir(tmp_path, "good", codes, [1.0], "2023-01-10")

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    pl.DataFrame(
        {"ts_code": codes, "target_weight": [0.0], "prev_weight": [0.0]}
    ).write_parquet(bad_dir / "weights.parquet")
    (bad_dir / "manifest.json").write_text(
        json.dumps({"run_id": "bad", "signal_date": "2023-01-15", "status": "infeasible"})
    )

    with caplog.at_level(logging.WARNING, logger="factorzen.sim.engine"):
        out = _load_weights_by_date([good, str(bad_dir)])

    assert _date(2023, 1, 10) in out
    assert _date(2023, 1, 15) not in out, (
        "status=infeasible 的全零兜底持仓不应被当作有效清仓信号执行"
    )
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "infeasible" in m or "status" in m for m in warning_messages
    ), f"未找到跳过原因的 warning，记录到的 warning: {warning_messages}"


def test_load_weights_by_date_keeps_manifest_without_status_field(tmp_path: Path) -> None:
    """修复2 不应破坏向后兼容：manifest 完全没有 status 字段时（历史产物/旧版
    pipeline）应照常视为有效信号，而不是被新增的状态校验误伤。
    """
    from datetime import date as _date

    from factorzen.sim.engine import _load_weights_by_date

    codes = ["000001.SZ"]
    legacy = _write_portfolio_dir(tmp_path, "legacy", codes, [1.0], "2023-01-10")
    out = _load_weights_by_date([legacy])
    assert _date(2023, 1, 10) in out


def test_load_weights_by_date_warns_on_missing_signal_date(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """修复5：manifest 缺 signal_date 字段时应 warning 说明具体 run_dir，
    而不是静默 continue 跳过（此前完全没有任何日志）。
    """
    from factorzen.sim.engine import _load_weights_by_date

    codes = ["000001.SZ"]
    good = _write_portfolio_dir(tmp_path, "good", codes, [1.0], "2023-01-10")

    no_sig_dir = tmp_path / "no_sig"
    no_sig_dir.mkdir()
    pl.DataFrame(
        {"ts_code": codes, "target_weight": [1.0], "prev_weight": [0.0]}
    ).write_parquet(no_sig_dir / "weights.parquet")
    (no_sig_dir / "manifest.json").write_text(json.dumps({"run_id": "no_sig"}))

    with caplog.at_level(logging.WARNING, logger="factorzen.sim.engine"):
        out = _load_weights_by_date([good, str(no_sig_dir)])

    assert len(out) == 1, "缺 signal_date 的 run_dir 仍应被跳过（行为不变）"
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("no_sig" in m and "signal_date" in m for m in warning_messages), (
        f"manifest 缺 signal_date 时应 warning 说明具体 run_dir，实际: {warning_messages}"
    )


def test_load_weights_by_date_warns_on_signal_date_collision(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """修复5：多个 run_dir 撞同一 signal_date 时应 warning 说明发生了覆盖，
    而不是静默覆盖（此前完全没有任何日志）。选择规则本身不变：仍是遍历顺序
    中较晚的覆盖较早的。
    """
    from datetime import date as _date

    from factorzen.sim.engine import _load_weights_by_date

    codes = ["000001.SZ"]
    first = _write_portfolio_dir(tmp_path, "a_first", codes, [1.0], "2023-01-10")
    second = _write_portfolio_dir(tmp_path, "z_second", codes, [0.5], "2023-01-10")

    with caplog.at_level(logging.WARNING, logger="factorzen.sim.engine"):
        out = _load_weights_by_date([first, second])

    assert len(out) == 1
    assert out[_date(2023, 1, 10)]["target_weight"].to_list() == [0.5], (
        "撞键时行为不变：仍是遍历顺序中较晚的覆盖较早的"
    )
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "2023-01-10" in m and "a_first" in m and "z_second" in m for m in warning_messages
    ), f"撞键时应 warning 说明具体的两个 run_dir，实际: {warning_messages}"


def test_run_portfolio_simulation_warns_for_specific_stale_signal_among_valid_ones(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """修复3：N 个 portfolio_run_dirs 中只有最新一个 signal_date 过期、其余
    信号正常执行（更常见的真实场景）时，应针对那个过期信号单独告警，而不是
    因为整体 nav 非空就完全不告警。
    """
    codes = ["000001.SZ"]
    p1 = _write_portfolio_dir(tmp_path, "old1", codes, [1.0], "2023-01-10")
    p2 = _write_portfolio_dir(tmp_path, "old2", codes, [0.0], "2023-01-20")
    # 数据截止 2023-02-28，该信号晚于末日 → 永不生效，但前两个信号仍正常执行
    p_stale = _write_portfolio_dir(tmp_path, "stale", codes, [1.0], "2023-03-01")
    daily = _fake_daily(codes)

    with caplog.at_level(logging.WARNING, logger="factorzen.sim.engine"):
        res = run_portfolio_simulation(
            [p1, p2, p_stale],
            daily,
            out_dir=str(tmp_path / "sim_partial_stale"),
            run_id="ps1",
        )

    run_dir = Path(res["run_dir"])
    nav_df = pl.read_parquet(run_dir / "nav.parquet")
    assert not nav_df.is_empty(), "前两个有效信号应正常执行，nav 不应整体为空"

    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("2023-03-01" in m for m in warning_messages), (
        "应针对过期信号 2023-03-01 单独告警，而不是因 nav 整体非空而沉默；"
        f"记录到的 warning: {warning_messages}"
    )

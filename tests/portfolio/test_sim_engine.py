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

def test_sim_engine_core_suite(tmp_path, caplog):
    """nav.parquet / metrics.json / manifest.json 落盘，返回 sharpe/max_dd/ann_ret。；可复现铁律#3：sim manifest 须含 inputs/窗口/成本/配置/command/git_sha，；manifest 加厚不得改 nav/metrics 产物；JSON 可往返 loads。；Fix 2: signal_date 晚于回测末日时，应发出 warning，不抛错。；组合流权重可能 gross>2.0（杠杆/多空），sim 不应用 daily-research 默认；单票权重 > 1.0（杠杆集中/多空腿）也不应触发默认 max_abs_weight=1.0 校验崩溃。；放宽 gross/abs 上限后，NaN/inf 权重的数据损坏防线仍须保留（不能一起放掉）。；修复1：3 次大幅调仓（模拟高换手）后，nav.parquet 的 cost 列与 metrics.json；修复4（ST涨跌停容差接线）：run_portfolio_simulation 应基于 daily 的"""
    # -- 原 test_run_portfolio_simulation_produces_metrics --
    def _section_0_test_run_portfolio_simulation_produces_metrics(tmp_path):
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

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_run_portfolio_simulation_produces_metrics(_tp0)

    # -- 原 test_sim_manifest_reproducibility_fields --
    def _section_1_test_sim_manifest_reproducibility_fields(tmp_path):
        codes = ["000001.SZ", "000002.SZ"]
        p1 = _write_portfolio_dir(tmp_path, "p1", codes, [0.5, 0.5], "2023-01-10")
        p2 = _write_portfolio_dir(tmp_path, "p2", codes, [0.3, 0.7], "2023-02-01")
        daily = _fake_daily(codes)
        inputs = [p1, p2]
        res = run_portfolio_simulation(
            inputs, daily, out_dir=str(tmp_path / "sim_mf"), run_id="mf1"
        )
        run_dir = Path(res["run_dir"])
        raw = (run_dir / "manifest.json").read_text(encoding="utf-8")
        # JSON 合法：无不可序列化对象导致的落盘失败；可 loads 回来
        manifest = json.loads(raw)

        # 保留字段
        assert manifest["run_id"] == "mf1"
        assert manifest["n_signals"] == 2
        assert manifest.get("git_sha"), "git_sha 须非空"

        # 可复现字段
        assert manifest.get("inputs") == inputs, "inputs 须为 portfolio_run_dirs 列表"
        assert manifest.get("start") == "2023-01-10"
        assert manifest.get("end") == "2023-02-01"
        assert manifest.get("command") == "sim run"
        assert isinstance(manifest.get("cost_model"), dict) and manifest["cost_model"], (
            "cost_model 须为非空 dict（费率/印花/借券等）"
        )
        assert isinstance(manifest.get("config"), dict) and manifest["config"], (
            "config 须为非空 dict（BacktestConfig 关键字段）"
        )
        # n_exec_dates：有执行日时应为正
        assert isinstance(manifest.get("n_exec_dates"), int)
        assert manifest["n_exec_dates"] > 0

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_sim_manifest_reproducibility_fields(_tp1)

    # -- 原 test_sim_manifest_json_roundtrip_and_no_nav_regression --
    def _section_2_test_sim_manifest_json_roundtrip_and_no_nav_regression(tmp_path):
        codes = ["000001.SZ", "000002.SZ"]
        p1 = _write_portfolio_dir(tmp_path, "p1", codes, [0.5, 0.5], "2023-01-10")
        daily = _fake_daily(codes)
        res = run_portfolio_simulation(
            [p1], daily, out_dir=str(tmp_path / "sim_rt"), run_id="rt1"
        )
        run_dir = Path(res["run_dir"])

        # JSON 往返
        raw = (run_dir / "manifest.json").read_text(encoding="utf-8")
        m1 = json.loads(raw)
        m2 = json.loads(json.dumps(m1, ensure_ascii=False))
        assert m2 == m1

        # 零回归：返回值与 metrics/nav 仍可用
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        for k in ("ann_ret", "sharpe", "max_dd"):
            assert k in metrics
        assert res.get("sharpe") == metrics.get("sharpe")
        assert res.get("max_dd") == metrics.get("max_dd")
        assert res.get("ann_ret") == metrics.get("ann_ret")
        nav = pl.read_parquet(run_dir / "nav.parquet")
        assert "nav" in nav.columns and nav.height > 0

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_sim_manifest_json_roundtrip_and_no_nav_regression(_tp2)

    # -- 原 test_run_portfolio_simulation_warns_when_signal_date_after_trade_dates --
    def _section_3_test_run_portfolio_simulation_warns_when_signal_date_after_trade_dates(tmp_path, caplog):
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

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_run_portfolio_simulation_warns_when_signal_date_after_trade_dates(_tp3, caplog)

    # -- 原 test_run_portfolio_simulation_accepts_leveraged_portfolio_weights --
    def _section_4_test_run_portfolio_simulation_accepts_leveraged_portfolio_weights(tmp_path):
        codes = ["000001.SZ", "000002.SZ", "000003.SZ"]
        # gross=2.4（>2.0 默认上限），单票 0.8（<1.0）：杠杆多头组合
        p1 = _write_portfolio_dir(tmp_path, "lev", codes, [0.8, 0.8, 0.8], "2023-01-10")
        daily = _fake_daily(codes)
        res = run_portfolio_simulation(
            [p1], daily, out_dir=str(tmp_path / "sim_lev"), run_id="lev1"
        )
        nav_df = pl.read_parquet(Path(res["run_dir"]) / "nav.parquet")
        assert "nav" in nav_df.columns, "杠杆组合应正常产出 nav，而非被默认 gross 上限崩掉"

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_run_portfolio_simulation_accepts_leveraged_portfolio_weights(_tp4)

    # -- 原 test_run_portfolio_simulation_accepts_concentrated_weight_over_one --
    def _section_5_test_run_portfolio_simulation_accepts_concentrated_weight_over_one(tmp_path):
        codes = ["000001.SZ", "000002.SZ"]
        # 单票 1.5（>1.0 默认上限）
        p1 = _write_portfolio_dir(tmp_path, "conc", codes, [1.5, -0.5], "2023-01-10")
        daily = _fake_daily(codes)
        res = run_portfolio_simulation(
            [p1], daily, out_dir=str(tmp_path / "sim_conc"), run_id="conc1"
        )
        assert (Path(res["run_dir"]) / "nav.parquet").exists()

    _tp5 = tmp_path / "_s5"
    _tp5.mkdir(exist_ok=True)
    _section_5_test_run_portfolio_simulation_accepts_concentrated_weight_over_one(_tp5)

    # -- 原 test_run_portfolio_simulation_still_rejects_nonfinite_weights --
    def _section_6_test_run_portfolio_simulation_still_rejects_nonfinite_weights(tmp_path):
        codes = ["000001.SZ", "000002.SZ"]
        p1 = _write_portfolio_dir(tmp_path, "bad", codes, [float("nan"), 0.5], "2023-01-10")
        daily = _fake_daily(codes)
        with pytest.raises(ValueError, match="finite"):
            run_portfolio_simulation(
                [p1], daily, out_dir=str(tmp_path / "sim_bad"), run_id="bad1"
            )

    _tp6 = tmp_path / "_s6"
    _tp6.mkdir(exist_ok=True)
    _section_6_test_run_portfolio_simulation_still_rejects_nonfinite_weights(_tp6)

    # -- 原 test_run_portfolio_simulation_charges_trading_cost_on_turnover --
    def _section_7_test_run_portfolio_simulation_charges_trading_cost_on_turnover(tmp_path):
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

    _tp7 = tmp_path / "_s7"
    _tp7.mkdir(exist_ok=True)
    _section_7_test_run_portfolio_simulation_charges_trading_cost_on_turnover(_tp7)

    # -- 原 test_run_portfolio_simulation_passes_is_st_by_date_to_backtest --
    def _section_8_test_run_portfolio_simulation_passes_is_st_by_date_to_backtest(tmp_path, mp):
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
        mp.setattr(engine_mod, "run_strategy_backtest", _capturing_run)
        mp.setattr(engine_mod, "build_is_st_by_date", lambda codes, dates: sentinel)

        engine_mod.run_portfolio_simulation(
            [p1], daily, out_dir=str(tmp_path / "sim_st"), run_id="st1"
        )

        assert captured.get("is_st_by_date") == sentinel, (
            "run_strategy_backtest 应收到由 build_is_st_by_date 构建的 is_st_by_date，"
            f"实际收到: {captured.get('is_st_by_date')!r}"
        )

    _tp8 = tmp_path / "_s8"
    _tp8.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_8_test_run_portfolio_simulation_passes_is_st_by_date_to_backtest(_tp8, mp)


# ── 代码评审修复回归测试 ───────────────────────────────────────────────────


def test_load_weights_by_date_suite(tmp_path, caplog):
    """修复2：manifest.status 非成功状态（如 infeasible 兜底全零持仓）时，；修复2 不应破坏向后兼容：manifest 完全没有 status 字段时（历史产物/旧版；修复5：manifest 缺 signal_date 字段时应 warning 说明具体 run_dir，；修复5：多个 run_dir 撞同一 signal_date 时应 warning 说明发生了覆盖，；修复3：N 个 portfolio_run_dirs 中只有最新一个 signal_date 过期、其余"""
    # -- 原 test_load_weights_by_date_skips_non_optimal_status --
    def _section_0_test_load_weights_by_date_skips_non_optimal_status(tmp_path, caplog):
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

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_load_weights_by_date_skips_non_optimal_status(_tp0, caplog)

    # -- 原 test_load_weights_by_date_keeps_manifest_without_status_field --
    def _section_1_test_load_weights_by_date_keeps_manifest_without_status_field(tmp_path):
        from datetime import date as _date

        from factorzen.sim.engine import _load_weights_by_date

        codes = ["000001.SZ"]
        legacy = _write_portfolio_dir(tmp_path, "legacy", codes, [1.0], "2023-01-10")
        out = _load_weights_by_date([legacy])
        assert _date(2023, 1, 10) in out

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_load_weights_by_date_keeps_manifest_without_status_field(_tp1)

    # -- 原 test_load_weights_by_date_warns_on_missing_signal_date --
    def _section_2_test_load_weights_by_date_warns_on_missing_signal_date(tmp_path, caplog):
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

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_load_weights_by_date_warns_on_missing_signal_date(_tp2, caplog)

    # -- 原 test_load_weights_by_date_warns_on_signal_date_collision --
    def _section_3_test_load_weights_by_date_warns_on_signal_date_collision(tmp_path, caplog):
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

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_load_weights_by_date_warns_on_signal_date_collision(_tp3, caplog)

    # -- 原 test_run_portfolio_simulation_warns_for_specific_stale_signal_among_valid_ones --
    def _section_4_test_run_portfolio_simulation_warns_for_specific_stale_signal_among_valid_ones(tmp_path, caplog):
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

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_run_portfolio_simulation_warns_for_specific_stale_signal_among_valid_ones(_tp4, caplog)



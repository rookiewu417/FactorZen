"""
test_repro_infra.py：test_expr_registry_repro.py：tests/test_expr_registry_repro.py — registry 复现接线与 attach_expr_leaves。
test_calendar_data.py：test_calendar.py：common/calendar.py 单元测试(本地缓存 mock,不调用 Tushare)
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import logging
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import polars as pl
import pytest

import factorzen.core.calendar as cal_mod
import factorzen.core.data_ensure as data_ensure
from factorzen.core.feature_schema import INTRADAY_FEATURES
from factorzen.core.storage import save_parquet
from factorzen.core.timing import StageTimer
from factorzen.core.validation import require_columns
from factorzen.discovery.intraday_expr import (
    attach_expr_leaves,
    ensure_expr_panel,
    make_expr_spec,
    materialize_expr_features,
    register_expr_features,
)
from factorzen.research.combination.models import (
    _warn_incomplete,
    build_panel,
)


# ==== 来自 test_repro_infra.py ====
# ==== 来自 test_expr_registry_repro.py ====
def _seed_expr_cache(
    base: Path, src: Path, spec, *, start: str = "20240102", end: str = "20240102"
) -> pl.DataFrame:
    """稀疏测试源湖无法过默认 0.8 覆盖：先 min_bar_coverage=0 物化并落缓存。"""
    panel = materialize_expr_features(
        [spec],
        start,
        end,
        freq=spec.freq,
        source_dir=src,
        min_bar_coverage=0.0,
    )
    save_parquet(
        panel.select(["trade_date", "ts_code", spec.name]),
        data_type=f"exp/{spec.freq}/{spec.name}",
        date_col="trade_date",
        base_dir=base,
        mode="overwrite",
    )
    return panel

def _dt(h: int, m: int, day: int = 2) -> dt.datetime:
    return dt.datetime(2024, 1, day, h, m, 0)

def _sparse_minute() -> pl.DataFrame:
    rows = [
        (_dt(9, 30), 10.0, 10.0, 10.0, 10.0, 100, 1000.0),
        (_dt(9, 31), 10.0, 10.6, 10.0, 10.5, 200, 2100.0),
        (_dt(9, 40), 10.5, 10.5, 10.2, 10.2, 150, 1530.0),
        (_dt(10, 0), 10.2, 10.4, 10.1, 10.3, 400, 4120.0),
        (_dt(11, 30), 10.3, 10.5, 10.2, 10.4, 250, 2600.0),
        (_dt(13, 1), 10.4, 10.6, 10.3, 10.5, 300, 3150.0),
        (_dt(14, 30), 10.5, 10.7, 10.4, 10.6, 200, 2120.0),
        (_dt(14, 35), 10.6, 10.8, 10.5, 10.7, 350, 3745.0),
        (_dt(15, 0), 10.7, 10.9, 10.6, 10.8, 500, 5400.0),
    ]
    frames = []
    for code, scale in (("000001.SZ", 1.0), ("000002.SZ", 1.1)):
        frames.append(
            pl.DataFrame(
                {
                    "ts_code": [code] * len(rows),
                    "trade_time": pl.Series(
                        [r[0] for r in rows], dtype=pl.Datetime("us")
                    ),
                    "open": [r[1] * scale for r in rows],
                    "high": [r[2] * scale for r in rows],
                    "low": [r[3] * scale for r in rows],
                    "close": [r[4] * scale for r in rows],
                    "vol": pl.Series([r[5] for r in rows], dtype=pl.Int64),
                    "amount": [r[6] * scale for r in rows],
                }
            )
        )
    return pl.concat(frames)

def _write_src(tmp: Path) -> Path:
    src = tmp / "src"
    save_parquet(
        _sparse_minute(),
        data_type="minute_1min",
        date_col="trade_time",
        base_dir=src,
        mode="overwrite",
    )
    return src

def _builtin_panel(dates: list[str], codes: list[str]) -> pl.DataFrame:
    rows = []
    for code in codes:
        for j, d in enumerate(dates):
            r: dict = {
                "trade_date": dt.datetime.strptime(d, "%Y%m%d").date(),
                "ts_code": code,
            }
            for i, c in enumerate(sorted(INTRADAY_FEATURES)):
                r[c] = float(i + 1) + 0.01 * j
            rows.append(r)
    return pl.DataFrame(rows)

def _daily_frame__repro_infra(dates: list[str], codes: list[str]) -> pl.DataFrame:
    rows = []
    for code in codes:
        for j, d in enumerate(dates):
            rows.append(
                {
                    "trade_date": dt.datetime.strptime(d, "%Y%m%d").date(),
                    "ts_code": code,
                    "close": 10.0 + j,
                    "close_adj": 10.0 + j,
                    "open": 10.0,
                    "open_adj": 10.0,
                    "high": 11.0,
                    "high_adj": 11.0,
                    "low": 9.0,
                    "low_adj": 9.0,
                    "pre_close": 10.0,
                    "vol": 1e5,
                    "amount": 1e6,
                }
            )
    return pl.DataFrame(rows)

class TestAttachExprLeaves:
    def test_attach_expr_leaves_suite(self, tmp_path):
        """test_join_ok；test_missing_require_false_null_warn；test_missing_require_true_raises"""
        # -- 原 test_join_ok --
        _tp0 = tmp_path / "_s0"
        _tp0.mkdir(exist_ok=True)
        dates = ["20240102"]
        codes = ["000001.SZ", "000002.SZ"]
        daily = _daily_frame__repro_infra(dates, codes)
        # 先 join builtin 模拟 prepare 链
        from factorzen.daily.data.intraday import attach_intraday

        panel = _builtin_panel(dates, codes)
        daily = attach_intraday(daily, injected=panel, require=False)

        spec = make_expr_spec("div(amount, vol)", "mean", freq="5min")
        base = _tp0 / "feat"
        src = _write_src(_tp0)
        register_expr_features([spec], session="t", base_dir=base)
        _seed_expr_cache(base, src, spec)

        out = attach_expr_leaves(
            daily,
            [spec.name],
            require=False,
            base_dir=base,
            source_dir=src,
        )
        assert spec.name in out.columns
        assert out[spec.name].null_count() < out.height

        # -- 原 test_missing_require_false_null_warn --
        _tp1 = tmp_path / "_s1"
        _tp1.mkdir(exist_ok=True)
        daily = _daily_frame__repro_infra(["20240102"], ["000001.SZ"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = attach_expr_leaves(
                daily,
                ["ix_notexist"],
                require=False,
                base_dir=_tp1 / "empty_reg",
            )
        assert "ix_notexist" in out.columns
        assert out["ix_notexist"][0] is None
        assert any("ix_notexist" in str(x.message) or "未注册" in str(x.message) for x in w)

        # -- 原 test_missing_require_true_raises --
        _tp2 = tmp_path / "_s2"
        _tp2.mkdir(exist_ok=True)
        daily = _daily_frame__repro_infra(["20240102"], ["000001.SZ"])
        with pytest.raises(ValueError, match=r"未注册|ix_notexist"):
            attach_expr_leaves(
                daily,
                ["ix_notexist"],
                require=True,
                base_dir=_tp2 / "empty_reg",
            )


class TestExpressionFactorParity:
    def test_rank_ix_matches_direct(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """注册 ix 叶 → ExpressionFactor 经 attach 与直算逐值一致。"""
        from factorzen.discovery.derived import add_derived_columns
        from factorzen.discovery.expression import evaluate_materialized, parse_expr
        from factorzen.discovery.factor import ExpressionFactor
        from factorzen.discovery.operators import LEAF_FEATURES

        src = _write_src(tmp_path)
        base = tmp_path / "feat"
        spec = make_expr_spec("div(amount, vol)", "mean", freq="5min")
        register_expr_features([spec], session="parity", base_dir=base)

        monkeypatch.setattr(
            "factorzen.config.settings.INTRADAY_FEATURES_DIR", base
        )
        monkeypatch.setattr(
            "factorzen.discovery.intraday_expr.INTRADAY_FEATURES_DIR", base
        )
        monkeypatch.setattr(
            "factorzen.discovery.intraday_expr.DATA_RAW", src
        )

        # 预热缓存（稀疏帧需 min_bar_coverage=0）
        mat = _seed_expr_cache(base, src, spec)
        assert mat.height >= 1
        ep = ensure_expr_panel(
            spec.name, "20240102", "20240102", base_dir=base, source_dir=src
        )
        assert ep[spec.name].null_count() == 0

        dates = ["20240102"]
        codes = ["000001.SZ", "000002.SZ"]
        daily = _daily_frame__repro_infra(dates, codes)

        def _fake_attach_expr(frame, names, **kw):
            out = frame
            for name in names:
                part = ensure_expr_panel(
                    name, "20240102", "20240102", base_dir=base, source_dir=src
                )
                if name in out.columns:
                    out = out.drop(name)
                out = out.join(
                    part.select(["trade_date", "ts_code", name]),
                    on=["trade_date", "ts_code"],
                    how="left",
                )
            return out

        monkeypatch.setattr(
            "factorzen.discovery.factor.attach_expr_leaves", _fake_attach_expr
        )

        expr = f"rank({spec.name})"
        fac = ExpressionFactor(expr, mined_name="ix_rank")
        assert spec.name in fac._ix_leaves

        class _Ctx:
            start = "20240102"
            end = "20240102"

            @property
            def daily(self):
                return daily.lazy()

            @property
            def daily_basic(self):
                return pl.DataFrame(
                    {
                        "trade_date": [dt.date(2024, 1, 2)] * 2,
                        "ts_code": codes,
                        "circ_mv": [1e6, 1e6],
                    }
                ).lazy()

        out = fac.compute(_Ctx())
        assert out.height > 0

        attached = _fake_attach_expr(daily, [spec.name], require=True)
        prepped = add_derived_columns(attached.sort(["ts_code", "trade_date"]))
        leaf_map = {**LEAF_FEATURES, spec.name: spec.name}
        node = parse_expr(expr, leaf_map)
        direct = (
            prepped.with_columns(
                evaluate_materialized(node, prepped, leaf_map).alias("factor_value")
            )
            .filter(pl.col("trade_date") >= dt.date(2024, 1, 2))
            .select(["trade_date", "ts_code", "factor_value"])
            .filter(
                pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite()
            )
        )
        a = out.sort(["ts_code", "trade_date"])
        b = direct.sort(["ts_code", "trade_date"])
        assert a.height == b.height
        for va, vb in zip(
            a["factor_value"].to_list(), b["factor_value"].to_list(), strict=True
        ):
            assert va == pytest.approx(vb, abs=1e-9, nan_ok=True)

        # 与 materialize 直算列值一致（rank 前）
        joined = attached.join(
            mat.select(["trade_date", "ts_code", spec.name]).rename(
                {spec.name: "_direct"}
            ),
            on=["trade_date", "ts_code"],
            how="left",
        )
        for row in joined.iter_rows(named=True):
            if row[spec.name] is not None and row["_direct"] is not None:
                assert row[spec.name] == pytest.approx(row["_direct"], abs=1e-9)

class TestIntradayExprLeafNames:
    def test_collect_ix_names(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from factorzen.discovery.preparation import (
            expressions_need_intraday,
            intraday_expr_leaf_names,
        )

        base = tmp_path / "feat"
        spec = make_expr_spec("bar_ret", "std", freq="5min")
        register_expr_features([spec], session="n", base_dir=base)
        monkeypatch.setattr(
            "factorzen.config.settings.INTRADAY_FEATURES_DIR", base
        )
        monkeypatch.setattr(
            "factorzen.discovery.intraday_expr.INTRADAY_FEATURES_DIR", base
        )

        exprs = [f"rank({spec.name})", "ts_mean(close, 5)", "rank(i_rv)"]
        assert expressions_need_intraday(exprs) is True
        names = intraday_expr_leaf_names(exprs)
        assert names == [spec.name]

# ==== 来自 test_script_manifest_wrapping.py ====
def _single_manifest(experiments_dir):
    manifests = list(experiments_dir.glob("*/manifest.json"))
    assert len(manifests) == 1
    return json.loads(manifests[0].read_text(encoding="utf-8"))

def test_failure_manifest_partial_outputs_suite(tmp_path):
    """generate_report / daily backtest / daily eval 失败时 manifest 记 partial outputs。"""
    # -- 原 test_generate_report_failure_manifest_records_partial_outputs --
    def _section_0_test_generate_report_failure_manifest_records_partial_outputs(tmp_path, mp):
        from factorzen.core import experiment as exp_mod
        from factorzen.pipelines import _report_persistence as persist
        from factorzen.pipelines import generate_report as mod

        experiments_dir = tmp_path / "experiments"
        mp.setattr(exp_mod, "EXPERIMENTS_DIR", experiments_dir)
        # _meta_path / _existing_report_outputs 已拆到 _report_persistence，在该模块解析路径函数
        mp.setattr(
            persist, "daily_result_output_dir", lambda factor_name: tmp_path / "results"
        )
        mp.setattr(
            persist, "daily_report_output_dir", lambda factor_name: tmp_path / "reports"
        )
        mp.setattr(
            sys,
            "argv",
            [
                "generate_report.py",
                "--factor",
                "momentum_20d",
                "--start",
                "20240101",
                "--end",
                "20240131",
            ],
        )

        def fail_after_meta(args, effective_config, timer=None):
            meta_path = mod._meta_path(args.factor, args.start, args.end)
            meta_path.write_text("{}", encoding="utf-8")
            raise RuntimeError("report boom")

        mp.setattr(mod, "_run", fail_after_meta)

        with pytest.raises(SystemExit) as exc:
            mod.main()

        assert exc.value.code == 1
        manifest = _single_manifest(experiments_dir)
        assert manifest["status"] == "failure"
        assert manifest["error"] == "report boom"
        assert manifest["config"]["factor"] == "momentum_20d"
        assert manifest["outputs"]["meta"] == str(
            tmp_path / "results" / "momentum_20d_20240101_20240131_meta.json"
        )

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_generate_report_failure_manifest_records_partial_outputs(_tp0, mp)

    # -- 原 test_run_daily_failure_manifest_records_partial_outputs --
    def _section_1_test_run_daily_failure_manifest_records_partial_outputs(tmp_path, mp):
        from factorzen.core import experiment as exp_mod
        from factorzen.pipelines import daily_single as mod

        experiments_dir = tmp_path / "experiments"
        mp.setattr(exp_mod, "EXPERIMENTS_DIR", experiments_dir)
        mp.setattr(mod, "daily_factor_output_dir", lambda factor_name: tmp_path / "factors")
        mp.setattr(mod, "daily_result_output_dir", lambda factor_name: tmp_path / "results")
        mp.setattr(mod, "daily_report_output_dir", lambda factor_name: tmp_path / "reports")
        mp.setattr(
            sys,
            "argv",
            [
                "run_daily_single.py",
                "--factor",
                "momentum_20d",
                "--start",
                "20240101",
                "--end",
                "20240131",
            ],
        )

        def fail_after_quality(args, effective_config, timer=None):
            quality_path = mod.daily_result_output_dir(args.factor) / (
                f"{args.factor}_{args.start}_{args.end}_quality.json"
            )
            quality_path.parent.mkdir(parents=True, exist_ok=True)
            quality_path.write_text("{}", encoding="utf-8")
            raise RuntimeError("daily boom")

        # main() 默认 track="backtest" → 分派到 run_factor_backtest（双轨拆分后）
        mp.setattr(mod, "run_factor_backtest", fail_after_quality)

        with pytest.raises(SystemExit) as exc:
            mod.main()

        assert exc.value.code == 1
        manifest = _single_manifest(experiments_dir)
        assert manifest["status"] == "failure"
        assert manifest["error"] == "daily boom"
        assert manifest["config"]["factor"] == "momentum_20d"
        assert manifest["outputs"]["quality_report"] == str(
            tmp_path / "results" / "momentum_20d_20240101_20240131_quality.json"
        )

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_run_daily_failure_manifest_records_partial_outputs(_tp1, mp)

    # -- eval 轨失败 manifest --
    def _section_2_test_run_daily_eval_failure_manifest_records_partial_outputs(tmp_path, mp):
        from factorzen.core import experiment as exp_mod
        from factorzen.pipelines import daily_single as mod

        experiments_dir = tmp_path / "experiments"
        mp.setattr(exp_mod, "EXPERIMENTS_DIR", experiments_dir)
        mp.setattr(mod, "daily_factor_output_dir", lambda factor_name: tmp_path / "factors")
        mp.setattr(mod, "daily_result_output_dir", lambda factor_name: tmp_path / "results")
        mp.setattr(mod, "daily_report_output_dir", lambda factor_name: tmp_path / "reports")
        mp.setattr(
            sys,
            "argv",
            [
                "run_daily_single.py",
                "--factor",
                "momentum_20d",
                "--start",
                "20240101",
                "--end",
                "20240131",
            ],
        )

        def fail_after_quality(args, effective_config, timer=None):
            quality_path = mod.daily_result_output_dir(args.factor) / (
                f"{args.factor}_{args.start}_{args.end}_quality.json"
            )
            quality_path.parent.mkdir(parents=True, exist_ok=True)
            quality_path.write_text("{}", encoding="utf-8")
            raise RuntimeError("eval boom")

        mp.setattr(mod, "run_factor_eval", fail_after_quality)

        with pytest.raises(SystemExit) as exc:
            mod.main(track="eval")

        assert exc.value.code == 1
        manifest = _single_manifest(experiments_dir)
        assert manifest["status"] == "failure"
        assert manifest["error"] == "eval boom"
        assert manifest["config"]["factor"] == "momentum_20d"
        assert manifest["outputs"]["quality_report"] == str(
            tmp_path / "results" / "momentum_20d_20240101_20240131_quality.json"
        )

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_run_daily_eval_failure_manifest_records_partial_outputs(_tp2, mp)


# ==== 来自 test_seed.py ====
def test_seed_repro_suite():
    """固定种子两次采样结果相同。；同种子采样器产生相同建议。"""
    # -- 原 test_seed_reproducibility --
    def _section_0_test_seed_reproducibility():
        from factorzen.core.seed import set_global_seed

        set_global_seed(42)
        a = np.random.rand(5)
        set_global_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    _section_0_test_seed_reproducibility()

    # -- 原 test_get_optuna_sampler_reproducible --
    def _section_1_test_get_optuna_sampler_reproducible():
        import optuna

        from factorzen.core.seed import get_optuna_sampler
        sampler1 = get_optuna_sampler(42)
        sampler2 = get_optuna_sampler(42)
        study = optuna.create_study(sampler=sampler1)
        trial1 = study.ask({"x": optuna.distributions.FloatDistribution(0, 1)})
        study2 = optuna.create_study(sampler=sampler2)
        trial2 = study2.ask({"x": optuna.distributions.FloatDistribution(0, 1)})
        assert trial1.params == trial2.params

    _section_1_test_get_optuna_sampler_reproducible()


# ==== 来自 test_validation.py ====
def test_require_columns_suite():
    """test_require_columns_raises_listing_missing_and_actual；test_require_columns_does_not_flag_present_columns"""
    # -- 原 test_require_columns_raises_listing_missing_and_actual --
    def _section_0_test_require_columns_raises_listing_missing_and_actual():
        df = pl.DataFrame({"trade_date": ["20240101"], "close": [1.0]})
        with pytest.raises(ValueError) as exc:
            require_columns(df, ["trade_date", "ts_code", "factor"], context="compute_ic")
        msg = str(exc.value)
        assert "compute_ic" in msg
        assert "ts_code" in msg and "factor" in msg
        # 实际存在的列也应在错误信息中,便于排查
        assert "trade_date" in msg

    _section_0_test_require_columns_raises_listing_missing_and_actual()

    # -- 原 test_require_columns_does_not_flag_present_columns --
    def _section_1_test_require_columns_does_not_flag_present_columns():
        df = pl.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError) as exc:
            require_columns(df, ["a", "c"])
        assert "c" in str(exc.value)
        # 'a' 已存在,不应被列为缺失(出现在“缺少必需列 [...]”片段里)
        missing_part = str(exc.value).split("实际列")[0]
        assert "'a'" not in missing_part

    _section_1_test_require_columns_does_not_flag_present_columns()


# ==== 来自 test_timing.py ====
def test_stage_timer_suite(caplog):
    """test_stage_timer_accumulates_named_durations；test_stage_timer_logs_stage_name_and_duration；test_stage_timer_records_duration_even_on_exception"""
    # -- 原 test_stage_timer_accumulates_named_durations --
    def _section_0_test_stage_timer_accumulates_named_durations():
        timer = StageTimer()
        with timer.stage("ic"):
            pass
        with timer.stage("backtest"):
            pass
        assert set(timer.timings) == {"ic", "backtest"}
        assert all(isinstance(v, float) and v >= 0 for v in timer.timings.values())

    _section_0_test_stage_timer_accumulates_named_durations()

    # -- 原 test_stage_timer_logs_stage_name_and_duration --
    def _section_1_test_stage_timer_logs_stage_name_and_duration(caplog):
        timer = StageTimer()
        with caplog.at_level(logging.INFO), timer.stage("报告生成"):
            pass
        assert any("报告生成" in r.getMessage() and "耗时" in r.getMessage() for r in caplog.records)

    _section_1_test_stage_timer_logs_stage_name_and_duration(caplog)

    # -- 原 test_stage_timer_records_duration_even_on_exception --
    def _section_2_test_stage_timer_records_duration_even_on_exception():
        timer = StageTimer()
        with pytest.raises(ValueError), timer.stage("boom"):
            raise ValueError("x")
        assert "boom" in timer.timings

    _section_2_test_stage_timer_records_duration_even_on_exception()


# ==== 来自 test_calendar_data.py ====
# ==== 来自 test_calendar.py ====
def _make_mock_calendar(tmp_path: Path) -> pl.DataFrame:
    """生成 2024-01-01 ~ 2024-01-10 的模拟交易日历，工作日为交易日。"""
    from datetime import date, timedelta

    rows = []
    d = date(2024, 1, 1)
    for _ in range(10):
        rows.append(
            {
                "cal_date": d,
                "is_open": 0 if d.weekday() >= 5 else 1,
                "pretrade_date": "",
            }
        )
        d += timedelta(days=1)
    df = pl.DataFrame(rows)
    cal_file = tmp_path / "trade_cal.parquet"
    df.write_parquet(cal_file)
    return df

@pytest.fixture()
def mock_calendar(tmp_path, monkeypatch):
    """将 _CAL_FILE 和 _is_cache_valid 重定向到 tmp 目录。"""
    import factorzen.core.calendar as cal_mod

    cal_file = tmp_path / "trade_cal.parquet"
    _make_mock_calendar(tmp_path)

    monkeypatch.setattr(cal_mod, "_CAL_FILE", cal_file)

    def _always_valid():
        return True

    monkeypatch.setattr(cal_mod, "_is_cache_valid", _always_valid)
    return cal_mod

def test_trade_date_nav_suite(mock_calendar):
    """test_is_trade_date_weekday；test_is_trade_date_weekend；test_is_trade_date_string_input；test_prev_trade_date；test_next_trade_date；test_get_trade_calendar_filter"""
    # -- 原 test_is_trade_date_weekday --
    def _section_0_test_is_trade_date_weekday(mock_calendar):
        assert mock_calendar.is_trade_date(date(2024, 1, 2)) is True  # 周二

    _section_0_test_is_trade_date_weekday(mock_calendar)

    # -- 原 test_is_trade_date_weekend --
    def _section_1_test_is_trade_date_weekend(mock_calendar):
        assert mock_calendar.is_trade_date(date(2024, 1, 6)) is False  # 周六

    _section_1_test_is_trade_date_weekend(mock_calendar)

    # -- 原 test_is_trade_date_string_input --
    def _section_2_test_is_trade_date_string_input(mock_calendar):
        assert mock_calendar.is_trade_date("20240102") is True

    _section_2_test_is_trade_date_string_input(mock_calendar)

    # -- 原 test_prev_trade_date --
    def _section_3_test_prev_trade_date(mock_calendar):
        result = mock_calendar.prev_trade_date(date(2024, 1, 3), n=1)
        assert result == date(2024, 1, 2)

    _section_3_test_prev_trade_date(mock_calendar)

    # -- 原 test_next_trade_date --
    def _section_4_test_next_trade_date(mock_calendar):
        result = mock_calendar.next_trade_date(date(2024, 1, 5), n=1)
        assert result == date(2024, 1, 8)

    _section_4_test_next_trade_date(mock_calendar)

    # -- 原 test_get_trade_calendar_filter --
    def _section_5_test_get_trade_calendar_filter(mock_calendar):
        cal = mock_calendar.get_trade_calendar(start="20240103", end="20240105")
        assert cal.shape[0] == 3
        assert cal["cal_date"].min() == date(2024, 1, 3)
        assert cal["cal_date"].max() == date(2024, 1, 5)

    _section_5_test_get_trade_calendar_filter(mock_calendar)


# ==== 来自 test_calendar_extra.py ====
def _make_calendar(start: date, days: int) -> pl.DataFrame:
    """生成连续 days 天的日历，工作日 is_open=1，周末=0。"""
    rows = []
    d = start
    for _ in range(days):
        rows.append(
            {"cal_date": d, "is_open": 0 if d.weekday() >= 5 else 1, "pretrade_date": ""}
        )
        d += timedelta(days=1)
    return pl.DataFrame(rows)

@pytest.fixture
def long_calendar(tmp_path, monkeypatch):
    """2024-01-01 起 60 天日历，重定向 _CAL_FILE 并强制缓存有效。"""
    cal_file = tmp_path / "trade_cal.parquet"
    _make_calendar(date(2024, 1, 1), 60).write_parquet(cal_file)
    monkeypatch.setattr(cal_mod, "_CAL_FILE", cal_file)
    monkeypatch.setattr(cal_mod, "_is_cache_valid", lambda: True)
    return cal_mod

# ══════════════════════════════════════════════════════════
# _is_cache_valid
# ══════════════════════════════════════════════════════════

def test_cache_valid_suite(tmp_path):
    """test_cache_valid_missing_file；test_cache_valid_fresh_file；test_cache_valid_expired_file"""
    # -- 原 test_cache_valid_missing_file --
    def _section_0_test_cache_valid_missing_file(tmp_path, mp):
        mp.setattr(cal_mod, "_CAL_FILE", tmp_path / "nope.parquet")
        assert cal_mod._is_cache_valid() is False

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_cache_valid_missing_file(_tp0, mp)

    # -- 原 test_cache_valid_fresh_file --
    def _section_1_test_cache_valid_fresh_file(tmp_path, mp):
        f = tmp_path / "trade_cal.parquet"
        pl.DataFrame({"cal_date": [date(2024, 1, 1)], "is_open": [1]}).write_parquet(f)
        mp.setattr(cal_mod, "_CAL_FILE", f)
        assert cal_mod._is_cache_valid() is True

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_cache_valid_fresh_file(_tp1, mp)

    # -- 原 test_cache_valid_expired_file --
    def _section_2_test_cache_valid_expired_file(tmp_path, mp):
        f = tmp_path / "trade_cal.parquet"
        pl.DataFrame({"cal_date": [date(2024, 1, 1)], "is_open": [1]}).write_parquet(f)
        old = time.time() - 30 * 86400
        import os

        os.utime(f, (old, old))
        mp.setattr(cal_mod, "_CAL_FILE", f)
        assert cal_mod._is_cache_valid() is False

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_cache_valid_expired_file(_tp2, mp)


# ══════════════════════════════════════════════════════════
# _fetch_from_tushare / _load_calendar
# ══════════════════════════════════════════════════════════

def test_fetch_calendar_io_suite(tmp_path):
    """test_fetch_from_tushare_writes_cache；test_fetch_from_tushare_empty_raises；test_load_calendar_cache_miss_fetches"""
    # -- 原 test_fetch_from_tushare_writes_cache --
    def _section_0_test_fetch_from_tushare_writes_cache(tmp_path, mp):
        mp.setattr(cal_mod, "_CAL_FILE", tmp_path / "trade_cal.parquet")
        mp.setattr(cal_mod, "DATA_CACHE", tmp_path)
        mp.setattr(cal_mod, "ensure_token", lambda: "dummy")

        import tushare as ts

        fake_pro = MagicMock()
        fake_pro.trade_cal.return_value = pd.DataFrame(
            {"cal_date": ["20240102", "20240103"], "is_open": [1, 1]}
        )
        mp.setattr(ts, "set_token", lambda t: None)
        mp.setattr(ts, "pro_api", lambda: fake_pro)

        out = cal_mod._fetch_from_tushare()
        assert out["cal_date"].dtype == pl.Date
        assert out["is_open"].dtype == pl.Int8
        assert (tmp_path / "trade_cal.parquet").exists()

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_fetch_from_tushare_writes_cache(_tp0, mp)

    # -- 原 test_fetch_from_tushare_empty_raises --
    def _section_1_test_fetch_from_tushare_empty_raises(tmp_path, mp):
        mp.setattr(cal_mod, "_CAL_FILE", tmp_path / "trade_cal.parquet")
        mp.setattr(cal_mod, "DATA_CACHE", tmp_path)
        mp.setattr(cal_mod, "ensure_token", lambda: "dummy")

        import tushare as ts

        fake_pro = MagicMock()
        fake_pro.trade_cal.return_value = pd.DataFrame()
        mp.setattr(ts, "set_token", lambda t: None)
        mp.setattr(ts, "pro_api", lambda: fake_pro)

        with pytest.raises(RuntimeError, match="返回空数据"):
            cal_mod._fetch_from_tushare()

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_fetch_from_tushare_empty_raises(_tp1, mp)

    # -- 原 test_load_calendar_cache_miss_fetches --
    def _section_2_test_load_calendar_cache_miss_fetches(mp):
        mp.setattr(cal_mod, "_is_cache_valid", lambda: False)
        sentinel = pl.DataFrame({"cal_date": [date(2024, 1, 2)], "is_open": [1]})
        mp.setattr(cal_mod, "_fetch_from_tushare", lambda: sentinel)
        assert cal_mod._load_calendar().equals(sentinel)

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_load_calendar_cache_miss_fetches(mp)


# ══════════════════════════════════════════════════════════
# is_trade_date / prev / next 边界
# ══════════════════════════════════════════════════════════

def test_trade_date_edges_suite(long_calendar):
    """日历中不存在的日期 → False。；日历最早日之前没有足够交易日 → ValueError。；test_next_trade_date_insufficient_raises；从周一往后第 5 个交易日应跨周。；字符串入参应被解析为日期。；test_next_trade_date_string_input；test_get_trade_dates_excludes_weekends"""
    # -- 原 test_is_trade_date_unknown_date_false --
    def _section_0_test_is_trade_date_unknown_date_false(long_calendar):
        assert long_calendar.is_trade_date(date(2050, 1, 1)) is False

    _section_0_test_is_trade_date_unknown_date_false(long_calendar)

    # -- 原 test_prev_trade_date_insufficient_raises --
    def _section_1_test_prev_trade_date_insufficient_raises(long_calendar):
        with pytest.raises(ValueError, match="不足"):
            long_calendar.prev_trade_date(date(2024, 1, 2), n=99)

    _section_1_test_prev_trade_date_insufficient_raises(long_calendar)

    # -- 原 test_next_trade_date_insufficient_raises --
    def _section_2_test_next_trade_date_insufficient_raises(long_calendar):
        with pytest.raises(ValueError, match="不足"):
            long_calendar.next_trade_date(date(2024, 2, 28), n=99)

    _section_2_test_next_trade_date_insufficient_raises(long_calendar)

    # -- 原 test_next_trade_date_multi_step --
    def _section_3_test_next_trade_date_multi_step(long_calendar):
        result = long_calendar.next_trade_date(date(2024, 1, 1), n=5)  # 周一
        assert result.weekday() < 5  # 落在工作日

    _section_3_test_next_trade_date_multi_step(long_calendar)

    # -- 原 test_prev_trade_date_string_input --
    def _section_4_test_prev_trade_date_string_input(long_calendar):
        assert long_calendar.prev_trade_date("20240103", n=1) == date(2024, 1, 2)

    _section_4_test_prev_trade_date_string_input(long_calendar)

    # -- 原 test_next_trade_date_string_input --
    def _section_5_test_next_trade_date_string_input(long_calendar):
        assert long_calendar.next_trade_date("20240105", n=1) == date(2024, 1, 8)

    _section_5_test_next_trade_date_string_input(long_calendar)

    # -- 原 test_get_trade_dates_excludes_weekends --
    def _section_6_test_get_trade_dates_excludes_weekends(long_calendar):
        dates = long_calendar.get_trade_dates("20240101", "20240107")
        assert all(d.weekday() < 5 for d in dates)
        assert date(2024, 1, 6) not in dates  # 周六

    _section_6_test_get_trade_dates_excludes_weekends(long_calendar)


# ══════════════════════════════════════════════════════════
# get_trade_dates / 时段 / 周月快照
# ══════════════════════════════════════════════════════════


def test_get_trading_sessions():
    sessions = cal_mod.get_trading_sessions()
    assert len(sessions) == 2
    assert sessions[0][0].hour == 9 and sessions[0][0].minute == 30

def test_snapshot_dates_suite(long_calendar):
    """test_weekly_snapshot_takes_last_trade_day_per_week；test_weekly_snapshot_empty_when_no_trades；test_monthly_snapshot_takes_last_trade_day_per_month；test_monthly_snapshot_empty_when_no_trades"""
    # -- 原 test_weekly_snapshot_takes_last_trade_day_per_week --
    def _section_0_test_weekly_snapshot_takes_last_trade_day_per_week(long_calendar):
        snaps = long_calendar.get_weekly_snapshot_dates("20240101", "20240131")
        # 每个快照日应是其 ISO 周内最后一个交易日（通常为周五）
        assert snaps == sorted(snaps)
        assert all(d.weekday() <= 4 for d in snaps)
        # 第一周 (2024-01-01~05) 的快照应为周五 2024-01-05
        assert date(2024, 1, 5) in snaps

    _section_0_test_weekly_snapshot_takes_last_trade_day_per_week(long_calendar)

    # -- 原 test_weekly_snapshot_empty_when_no_trades --
    def _section_1_test_weekly_snapshot_empty_when_no_trades(mp):
        mp.setattr(cal_mod, "get_trade_dates", lambda s, e: [])
        assert cal_mod.get_weekly_snapshot_dates("20240101", "20240131") == []

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_weekly_snapshot_empty_when_no_trades(mp)

    # -- 原 test_monthly_snapshot_takes_last_trade_day_per_month --
    def _section_2_test_monthly_snapshot_takes_last_trade_day_per_month(long_calendar):
        snaps = cal_mod.get_monthly_snapshot_dates("20240101", "20240229")
        # 1 月最后交易日应为 2024-01-31（周三）
        assert date(2024, 1, 31) in snaps
        assert snaps == sorted(snaps)

    _section_2_test_monthly_snapshot_takes_last_trade_day_per_month(long_calendar)

    # -- 原 test_monthly_snapshot_empty_when_no_trades --
    def _section_3_test_monthly_snapshot_empty_when_no_trades(mp):
        mp.setattr(cal_mod, "get_trade_dates", lambda s, e: [])
        assert cal_mod.get_monthly_snapshot_dates("20240101", "20240131") == []

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_monthly_snapshot_empty_when_no_trades(mp)


# ==== 来自 test_data_ensure.py ====
def _daily_frame__calendar_data(trade_date: date | str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [trade_date],
            "ts_code": ["000001.SZ"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "pre_close": [10.0],
            "change": [0.5],
            "pct_chg": [5.0],
            "vol": [1000.0],
            "amount": [10000.0],
        }
    )

def _pd_daily(trade_date: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [trade_date],
            "ts_code": ["000001.SZ"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "pre_close": [10.0],
            "change": [0.5],
            "pct_chg": [5.0],
            "vol": [1000.0],
            "amount": [10000.0],
        }
    )

def test_audit_daily_like_reports_missing_trading_dates(tmp_path, monkeypatch):
    part_dir = tmp_path / "daily" / "year=2024" / "month=01"
    part_dir.mkdir(parents=True)
    _daily_frame__calendar_data(date(2024, 1, 2)).write_parquet(
        part_dir / "data.parquet",
    )
    monkeypatch.setattr(
        data_ensure,
        "get_trade_dates",
        lambda start, end: [date(2024, 1, 2), date(2024, 1, 3)],
    )

    result = data_ensure.audit_daily_like("daily", "20240102", "20240103", base_dir=tmp_path)

    assert result.ok is False
    assert result.present_dates == ["20240102"]
    assert result.missing_dates == ["20240103"]

def test_ensure_daily_happy_suite(tmp_path):
    """test_ensure_daily_fetches_only_missing_dates；test_ensure_daily_does_not_fetch_when_cache_is_complete；test_ensure_daily_run_uses_complete_local_cache_without_tushare"""
    # -- 原 test_ensure_daily_fetches_only_missing_dates --
    def _section_0_test_ensure_daily_fetches_only_missing_dates(tmp_path, mp):
        part_dir = tmp_path / "daily" / "year=2024" / "month=01"
        part_dir.mkdir(parents=True)
        _daily_frame__calendar_data(date(2024, 1, 2)).write_parquet(
            part_dir / "data.parquet",
        )
        mp.setattr(
            data_ensure,
            "get_trade_dates",
            lambda start, end: [date(2024, 1, 2), date(2024, 1, 3)],
        )
        pro = MagicMock()
        pro.daily.return_value = _pd_daily("20240103")
        mp.setattr(data_ensure, "init_tushare", lambda: pro)
        mp.setattr(data_ensure, "_retry", lambda func, **kwargs: func(**kwargs))

        result = data_ensure.ensure_daily("20240102", "20240103", base_dir=tmp_path)

        assert result.ok is True
        pro.daily.assert_called_once_with(trade_date="20240103")
        loaded = pl.scan_parquet(str(tmp_path / "daily" / "**/*.parquet")).collect()
        assert sorted(loaded["trade_date"].dt.strftime("%Y%m%d").unique().to_list()) == [
            "20240102",
            "20240103",
        ]

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_ensure_daily_fetches_only_missing_dates(_tp0, mp)

    # -- 原 test_ensure_daily_does_not_fetch_when_cache_is_complete --
    def _section_1_test_ensure_daily_does_not_fetch_when_cache_is_complete(tmp_path, mp):
        part_dir = tmp_path / "daily" / "year=2024" / "month=01"
        part_dir.mkdir(parents=True)
        _daily_frame__calendar_data(date(2024, 1, 2)).write_parquet(
            part_dir / "data.parquet",
        )
        mp.setattr(data_ensure, "get_trade_dates", lambda start, end: [date(2024, 1, 2)])
        pro = MagicMock()
        mp.setattr(data_ensure, "init_tushare", lambda: pro)

        result = data_ensure.ensure_daily("20240102", "20240102", base_dir=tmp_path)

        assert result.ok is True
        pro.daily.assert_not_called()

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_ensure_daily_does_not_fetch_when_cache_is_complete(_tp1, mp)

    # -- 原 test_ensure_daily_run_uses_complete_local_cache_without_tushare --
    def _section_2_test_ensure_daily_run_uses_complete_local_cache_without_tushare(tmp_path, mp):
        for data_type in ("daily", "adj_factor", "daily_basic"):
            part_dir = tmp_path / data_type / "year=2024" / "month=01"
            part_dir.mkdir(parents=True)
            frame = _daily_frame__calendar_data(date(2024, 1, 2))
            if data_type == "adj_factor":
                frame = frame.select(["ts_code", "trade_date"]).with_columns(
                    pl.lit(1.0).alias("adj_factor")
                )
            elif data_type == "daily_basic":
                frame = frame.select(["trade_date", "ts_code"]).with_columns(
                    pl.lit(1_000_000.0).alias("total_mv")
                )
            frame.write_parquet(part_dir / "data.parquet")

        mp.setattr(data_ensure, "get_trade_dates", lambda start, end: [date(2024, 1, 2)])
        mp.setattr(data_ensure, "DATA_RAW", tmp_path)

        def _unexpected_api_init():
            raise AssertionError("complete local cache must not call Tushare")

        mp.setattr(data_ensure, "init_tushare", _unexpected_api_init)

        result = data_ensure.ensure_data_for_daily_run(
            required_data=["daily"],
            start="20240102",
            end="20240102",
            needs_size_neutralization=True,
            strict=True,
        )

        assert set(result) == {"daily", "adj_factor", "daily_basic"}
        assert all(audit.ok for audit in result.values())

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_ensure_daily_run_uses_complete_local_cache_without_tushare(_tp2, mp)


def test_ensure_daily_repair_fail_suite(tmp_path):
    """test_ensure_daily_repairs_duplicate_keys_without_fetching；test_ensure_daily_raises_when_fetch_does_not_fill_gap；test_ensure_daily_persists_successful_fetches_before_later_failure"""
    # -- 原 test_ensure_daily_repairs_duplicate_keys_without_fetching --
    def _section_0_test_ensure_daily_repairs_duplicate_keys_without_fetching(tmp_path, mp):
        part_dir = tmp_path / "daily" / "year=2024" / "month=01"
        part_dir.mkdir(parents=True)
        duplicate = pl.concat(
            [
                _daily_frame__calendar_data(date(2024, 1, 2)),
                _daily_frame__calendar_data(date(2024, 1, 2)).with_columns(pl.lit(10.8).alias("close")),
            ]
        )
        duplicate.write_parquet(part_dir / "data.parquet")
        mp.setattr(data_ensure, "get_trade_dates", lambda start, end: [date(2024, 1, 2)])

        def _unexpected_api_init():
            raise AssertionError("duplicate repair must not call Tushare")

        mp.setattr(data_ensure, "init_tushare", _unexpected_api_init)

        result = data_ensure.ensure_daily("20240102", "20240102", base_dir=tmp_path)

        assert result.ok is True
        loaded = pl.read_parquet(part_dir / "data.parquet")
        assert loaded.height == 1
        assert loaded["close"][0] == 10.8

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_ensure_daily_repairs_duplicate_keys_without_fetching(_tp0, mp)

    # -- 原 test_ensure_daily_raises_when_fetch_does_not_fill_gap --
    def _section_1_test_ensure_daily_raises_when_fetch_does_not_fill_gap(tmp_path, mp):
        mp.setattr(data_ensure, "get_trade_dates", lambda start, end: [date(2024, 1, 2)])
        pro = MagicMock()
        pro.daily.return_value = pd.DataFrame()
        mp.setattr(data_ensure, "init_tushare", lambda: pro)
        mp.setattr(data_ensure, "_retry", lambda func, **kwargs: pd.DataFrame())

        with pytest.raises(data_ensure.DataEnsureError, match="daily still missing"):
            data_ensure.ensure_daily("20240102", "20240102", base_dir=tmp_path)

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_ensure_daily_raises_when_fetch_does_not_fill_gap(_tp1, mp)

    # -- 原 test_ensure_daily_persists_successful_fetches_before_later_failure --
    def _section_2_test_ensure_daily_persists_successful_fetches_before_later_failure(tmp_path, mp):
        trade_dates = [
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 1, 4),
        ]
        mp.setattr(data_ensure, "get_trade_dates", lambda start, end: trade_dates)
        mp.setattr(data_ensure, "FETCH_SAVE_BATCH_SIZE", 2)
        pro = MagicMock()
        mp.setattr(data_ensure, "init_tushare", lambda: pro)

        def fake_retry(_func, *, trade_date):
            if trade_date == "20240104":
                raise RuntimeError("network stopped")
            return _pd_daily(trade_date)

        mp.setattr(data_ensure, "_retry", fake_retry)

        with pytest.raises(RuntimeError, match="network stopped"):
            data_ensure.ensure_daily("20240102", "20240104", base_dir=tmp_path)

        after = data_ensure.audit_daily_like("daily", "20240102", "20240104", base_dir=tmp_path)
        assert after.present_dates == ["20240102", "20240103"]
        assert after.missing_dates == ["20240104"]

        retry_dates: list[str] = []

        def finish_retry(_func, *, trade_date):
            retry_dates.append(trade_date)
            return _pd_daily(trade_date)

        mp.setattr(data_ensure, "_retry", finish_retry)

        final = data_ensure.ensure_daily("20240102", "20240104", base_dir=tmp_path)

        assert retry_dates == ["20240104"]
        assert final.ok is True

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_ensure_daily_persists_successful_fetches_before_later_failure(_tp2, mp)


def test_ensure_index_daily_fetches_missing_range(tmp_path, monkeypatch):
    monkeypatch.setattr(
        data_ensure,
        "get_trade_dates",
        lambda start, end: [date(2024, 1, 2), date(2024, 1, 3)],
    )
    pro = MagicMock()
    pro.index_daily.return_value = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103"],
            "ts_code": ["000300.SH", "000300.SH"],
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "pre_close": [1.0, 1.0],
            "change": [0.0, 0.0],
            "pct_chg": [0.0, 0.0],
            "vol": [1.0, 1.0],
            "amount": [1.0, 1.0],
        }
    )
    monkeypatch.setattr(data_ensure, "init_tushare", lambda: pro)
    monkeypatch.setattr(data_ensure, "_retry", lambda func, **kwargs: func(**kwargs))

    result = data_ensure.ensure_index_daily("000300.SH", "20240102", "20240103", base_dir=tmp_path)

    assert result.ok is True
    pro.index_daily.assert_called_once_with(
        ts_code="000300.SH",
        start_date="20240102",
        end_date="20240103",
    )

# ==== 来自 test_smoke_data.py ====
# tools/ 不是包，按文件路径加载模块
_SPEC = importlib.util.spec_from_file_location(
    "smoke_data", Path(__file__).resolve().parents[2] / "tools" / "smoke_data.py"
)
assert _SPEC and _SPEC.loader
smoke_data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke_data)

def _audit(status: str, rows: int = 100, warnings=None, errors=None) -> dict:
    return {
        "status": status,
        "checks": {"total_rows": rows},
        "warnings": warnings or [],
        "errors": errors or [],
    }

# ── _worst_status ───────────────────────────────────────────

def test_worst_status_suite():
    """test_worst_status_error_dominates；test_worst_status_warning_over_ok；test_worst_status_all_ok；test_worst_status_empty_is_ok"""
    # -- 原 test_worst_status_error_dominates --
    def _section_0_test_worst_status_error_dominates():
        assert smoke_data._worst_status(["ok", "warning", "error"]) == "error"

    _section_0_test_worst_status_error_dominates()

    # -- 原 test_worst_status_warning_over_ok --
    def _section_1_test_worst_status_warning_over_ok():
        assert smoke_data._worst_status(["ok", "warning", "ok"]) == "warning"

    _section_1_test_worst_status_warning_over_ok()

    # -- 原 test_worst_status_all_ok --
    def _section_2_test_worst_status_all_ok():
        assert smoke_data._worst_status(["ok", "ok"]) == "ok"

    _section_2_test_worst_status_all_ok()

    # -- 原 test_worst_status_empty_is_ok --
    def _section_3_test_worst_status_empty_is_ok():
        assert smoke_data._worst_status([]) == "ok"

    _section_3_test_worst_status_empty_is_ok()


# ── run_audits ──────────────────────────────────────────────

def test_run_audits_calls_audit_per_type(monkeypatch):
    calls = []

    def fake(*, data_type, start, end, universe_codes=None):
        calls.append(data_type)
        return _audit("ok")

    monkeypatch.setattr(smoke_data, "build_raw_data_audit", fake)
    results = smoke_data.run_audits(["daily", "finance"], "20230101", "20231231")
    assert set(results) == {"daily", "finance"}
    assert calls == ["daily", "finance"]

# ── summarize → 退出码 ──────────────────────────────────────

def test_summarize_exit_codes_suite(capsys):
    """test_summarize_all_ok_returns_0；test_summarize_warning_returns_2；test_summarize_error_returns_1；test_summarize_connectivity_fail_is_error；test_summarize_skipped_connectivity"""
    # -- 原 test_summarize_all_ok_returns_0 --
    def _section_0_test_summarize_all_ok_returns_0(capsys):
        code = smoke_data.summarize((True, "ok"), {"daily": _audit("ok")})
        assert code == 0
        assert "OK" in capsys.readouterr().out

    _section_0_test_summarize_all_ok_returns_0(capsys)

    # -- 原 test_summarize_warning_returns_2 --
    def _section_1_test_summarize_warning_returns_2():
        code = smoke_data.summarize(
            (True, "ok"), {"daily": _audit("warning", warnings=["缺 3 天"])}
        )
        assert code == 2

    _section_1_test_summarize_warning_returns_2()

    # -- 原 test_summarize_error_returns_1 --
    def _section_2_test_summarize_error_returns_1():
        code = smoke_data.summarize(
            (True, "ok"), {"daily": _audit("error", errors=["分区为空"])}
        )
        assert code == 1

    _section_2_test_summarize_error_returns_1()

    # -- 原 test_summarize_connectivity_fail_is_error --
    def _section_3_test_summarize_connectivity_fail_is_error():
        code = smoke_data.summarize((False, "token 缺失"), {"daily": _audit("ok")})
        assert code == 1

    _section_3_test_summarize_connectivity_fail_is_error()

    # -- 原 test_summarize_skipped_connectivity --
    def _section_4_test_summarize_skipped_connectivity(capsys):
        code = smoke_data.summarize(None, {"daily": _audit("ok")})
        assert code == 0
        assert "跳过" in capsys.readouterr().out

    _section_4_test_summarize_skipped_connectivity(capsys)


# ── check_tushare_connectivity ──────────────────────────────

def _fake_pro():
    """init_tushare 桩：需有被 _retry 引用的 trade_cal 属性。"""
    return SimpleNamespace(trade_cal=lambda **kw: None)

def test_connectivity_suite():
    """test_connectivity_success；test_connectivity_empty_result；test_connectivity_exception"""
    # -- 原 test_connectivity_success --
    def _section_0_test_connectivity_success(mp):
        import factorzen.core.loader as loader_mod

        mp.setattr(loader_mod, "init_tushare", _fake_pro)

        class _DF:
            empty = False

            def __len__(self):
                return 5

        mp.setattr(loader_mod, "_retry", lambda fn, **kw: _DF())
        ok, msg = smoke_data.check_tushare_connectivity()
        assert ok and "正常" in msg

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_connectivity_success(mp)

    # -- 原 test_connectivity_empty_result --
    def _section_1_test_connectivity_empty_result(mp):
        import factorzen.core.loader as loader_mod

        mp.setattr(loader_mod, "init_tushare", _fake_pro)

        class _Empty:
            empty = True

        mp.setattr(loader_mod, "_retry", lambda fn, **kw: _Empty())
        ok, _ = smoke_data.check_tushare_connectivity()
        assert not ok

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_connectivity_empty_result(mp)

    # -- 原 test_connectivity_exception --
    def _section_2_test_connectivity_exception(mp):
        import factorzen.core.loader as loader_mod

        def _boom():
            raise RuntimeError("no token")

        mp.setattr(loader_mod, "init_tushare", _boom)
        ok, msg = smoke_data.check_tushare_connectivity()
        assert not ok and "失败" in msg

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_connectivity_exception(mp)


# ── main / argparse ─────────────────────────────────────────

def test_smoke_main_cli_suite(capsys):
    """--skip-tushare 不触发连通性检查，退出码由审计决定。；test_main_json_output；test_main_error_audit_exit_1；test_main_default_audits_all_three"""
    # -- 原 test_main_skip_tushare_offline --
    def _section_0_test_main_skip_tushare_offline(mp):
        mp.setattr(
            smoke_data, "build_raw_data_audit", lambda **kw: _audit("ok")
        )

        def _should_not_call():
            raise AssertionError("--skip-tushare 时不应检查连通性")

        mp.setattr(smoke_data, "check_tushare_connectivity", _should_not_call)
        code = smoke_data.main(["--skip-tushare", "--data-type", "daily"])
        assert code == 0

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_main_skip_tushare_offline(mp)

    # -- 原 test_main_json_output --
    def _section_1_test_main_json_output(mp, capsys):
        mp.setattr(smoke_data, "build_raw_data_audit", lambda **kw: _audit("ok"))
        mp.setattr(
            smoke_data, "check_tushare_connectivity", lambda: (True, "ok")
        )
        code = smoke_data.main(["--data-type", "daily", "--json"])
        out = capsys.readouterr().out
        assert code == 0
        assert '"exit_code": 0' in out

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_main_json_output(mp, capsys)

    # -- 原 test_main_error_audit_exit_1 --
    def _section_2_test_main_error_audit_exit_1(mp):
        mp.setattr(
            smoke_data, "build_raw_data_audit", lambda **kw: _audit("error", errors=["空"])
        )
        code = smoke_data.main(["--skip-tushare", "--data-type", "finance"])
        assert code == 1

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_main_error_audit_exit_1(mp)

    # -- 原 test_main_default_audits_all_three --
    def _section_3_test_main_default_audits_all_three(mp):
        seen = []
        mp.setattr(
            smoke_data,
            "build_raw_data_audit",
            lambda **kw: seen.append(kw["data_type"]) or _audit("ok"),
        )
        smoke_data.main(["--skip-tushare"])
        assert set(seen) == {"daily", "daily_basic", "finance"}

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_main_default_audits_all_three(mp)


# ==== 来自 test_build_panel_coverage_warn.py ====
def _feat(name: str, n: int, coverage: float, *, seed: int = 0) -> pl.DataFrame:
    """coverage = 非空比例；其余为 null。"""
    rng = np.random.default_rng(seed)
    n_ok = max(0, round(n * coverage))
    vals = [float(x) for x in rng.normal(size=n_ok)] + [None] * (n - n_ok)
    rng.shuffle(vals)
    return pl.DataFrame({
        "trade_date": [f"202001{i+1:02d}" for i in range(n)],
        "ts_code": ["000001.SZ"] * n,
        "factor_value": vals,
    })

def _ret(n: int) -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": [f"202001{i+1:02d}" for i in range(n)],
        "ts_code": ["000001.SZ"] * n,
        "ret": [0.01] * n,
    })

def test_panel_coverage_warn_suite():
    """一列 20% 覆盖 → warn，文案含该列名。；全列 ≥30% 非空 → 不 warn，即便行齐全率 <70%（回归旧恒真行为）。"""
    # -- 原 test_warn_when_one_column_below_30pct --
    def _section_0_test_warn_when_one_column_below_30pct():
        n = 100
        # 两列健康、一列 20%
        dfs = {
            "ok_a": _feat("ok_a", n, 0.9, seed=1),
            "ok_b": _feat("ok_b", n, 0.8, seed=2),
            "sparse_x": _feat("sparse_x", n, 0.20, seed=3),
        }
        # full join 后行齐全率几乎必 <70%（互补稀疏）——旧口径恒 warn
        panel = build_panel(dfs, _ret(n))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_incomplete(panel)
            msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
        assert msgs, "应触发逐列覆盖警告"
        joined = " ".join(msgs)
        assert "sparse_x" in joined
        assert "20%" in joined or "0.2" in joined or "20" in joined

    _section_0_test_warn_when_one_column_below_30pct()

    # -- 原 test_no_warn_when_all_cols_above_30pct_even_if_row_complete_low --
    def _section_1_test_no_warn_when_all_cols_above_30pct_even_if_row_complete_low():
        n = 100
        # 三列各 50% 但互补缺失 → 行齐全率很低，但 min 列覆盖 =50% ≥30%
        rng = np.random.default_rng(42)
        dates = [f"202001{i+1:02d}" for i in range(n)]
        codes = ["000001.SZ"] * n

        def col_half(seed: int) -> list:
            mask = rng.random(n) < 0.5
            return [float(rng.normal()) if m else None for m in mask]

        # 手工拼宽表走 _warn_incomplete（与 build_panel 同路径）
        from factorzen.research.combination.models import _factor_panel, _join_ret

        feat = {
            "f1": pl.DataFrame({
                "trade_date": dates, "ts_code": codes, "factor_value": col_half(1),
            }),
            "f2": pl.DataFrame({
                "trade_date": dates, "ts_code": codes, "factor_value": col_half(2),
            }),
            "f3": pl.DataFrame({
                "trade_date": dates, "ts_code": codes, "factor_value": col_half(3),
            }),
        }
        wide = _join_ret(_factor_panel(feat), _ret(n))
        # 确认行齐全率 <70%（旧口径会 warn）
        names = [c for c in wide.columns if c not in ("trade_date", "ts_code", "ret")]
        complete_pct = wide.drop_nulls(subset=names).height / wide.height
        assert complete_pct < 0.7, f"本测依赖行齐全率低，得到 {complete_pct}"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_incomplete(wide)
            cov_warns = [
                x for x in w
                if issubclass(x.category, UserWarning)
                and "build_panel" in str(x.message)
            ]
        assert cov_warns == [], f"全列≥30% 不应 warn: {[str(x.message) for x in cov_warns]}"

    _section_1_test_no_warn_when_all_cols_above_30pct_even_if_row_complete_low()



"""test_expr_registry_repro.py：tests/test_expr_registry_repro.py — registry 复现接线与 attach_expr_leaves。
test_script_manifest_wrapping.py：Tests for script-level experiment manifest wrappers.
test_seed.py：Tests for common.seed module.
test_validation.py：Tests for core.validation column-contract helper.
test_timing.py：Tests for core.timing.StageTimer (per-stage timing observability).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

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

def _daily_frame(dates: list[str], codes: list[str]) -> pl.DataFrame:
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
    def test_join_ok(self, tmp_path: Path) -> None:
        dates = ["20240102"]
        codes = ["000001.SZ", "000002.SZ"]
        daily = _daily_frame(dates, codes)
        # 先 join builtin 模拟 prepare 链
        from factorzen.daily.data.intraday import attach_intraday

        panel = _builtin_panel(dates, codes)
        daily = attach_intraday(daily, injected=panel, require=False)

        spec = make_expr_spec("div(amount, vol)", "mean", freq="5min")
        base = tmp_path / "feat"
        src = _write_src(tmp_path)
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

    def test_missing_require_false_null_warn(self, tmp_path: Path) -> None:
        daily = _daily_frame(["20240102"], ["000001.SZ"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = attach_expr_leaves(
                daily,
                ["ix_notexist"],
                require=False,
                base_dir=tmp_path / "empty_reg",
            )
        assert "ix_notexist" in out.columns
        assert out["ix_notexist"][0] is None
        assert any("ix_notexist" in str(x.message) or "未注册" in str(x.message) for x in w)

    def test_missing_require_true_raises(self, tmp_path: Path) -> None:
        daily = _daily_frame(["20240102"], ["000001.SZ"])
        with pytest.raises(ValueError, match=r"未注册|ix_notexist"):
            attach_expr_leaves(
                daily,
                ["ix_notexist"],
                require=True,
                base_dir=tmp_path / "empty_reg",
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
        daily = _daily_frame(dates, codes)

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

def test_generate_report_failure_manifest_records_partial_outputs(tmp_path, monkeypatch):
    from factorzen.core import experiment as exp_mod
    from factorzen.pipelines import _report_persistence as persist
    from factorzen.pipelines import generate_report as mod

    experiments_dir = tmp_path / "experiments"
    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", experiments_dir)
    # _meta_path / _existing_report_outputs 已拆到 _report_persistence，在该模块解析路径函数
    monkeypatch.setattr(
        persist, "daily_result_output_dir", lambda factor_name: tmp_path / "results"
    )
    monkeypatch.setattr(
        persist, "daily_report_output_dir", lambda factor_name: tmp_path / "reports"
    )
    monkeypatch.setattr(
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

    monkeypatch.setattr(mod, "_run", fail_after_meta)

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

def test_run_daily_failure_manifest_records_partial_outputs(tmp_path, monkeypatch):
    from factorzen.core import experiment as exp_mod
    from factorzen.pipelines import daily_single as mod

    experiments_dir = tmp_path / "experiments"
    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", experiments_dir)
    monkeypatch.setattr(mod, "daily_factor_output_dir", lambda factor_name: tmp_path / "factors")
    monkeypatch.setattr(mod, "daily_result_output_dir", lambda factor_name: tmp_path / "results")
    monkeypatch.setattr(mod, "daily_report_output_dir", lambda factor_name: tmp_path / "reports")
    monkeypatch.setattr(
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

    monkeypatch.setattr(mod, "_run", fail_after_quality)

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

# ==== 来自 test_seed.py ====
def test_seed_reproducibility():
    """固定种子两次采样结果相同。"""
    from factorzen.core.seed import set_global_seed

    set_global_seed(42)
    a = np.random.rand(5)
    set_global_seed(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)

def test_get_optuna_sampler_reproducible():
    """同种子采样器产生相同建议。"""
    import optuna

    from factorzen.core.seed import get_optuna_sampler
    sampler1 = get_optuna_sampler(42)
    sampler2 = get_optuna_sampler(42)
    study = optuna.create_study(sampler=sampler1)
    trial1 = study.ask({"x": optuna.distributions.FloatDistribution(0, 1)})
    study2 = optuna.create_study(sampler=sampler2)
    trial2 = study2.ask({"x": optuna.distributions.FloatDistribution(0, 1)})
    assert trial1.params == trial2.params

# ==== 来自 test_validation.py ====
def test_require_columns_raises_listing_missing_and_actual():
    df = pl.DataFrame({"trade_date": ["20240101"], "close": [1.0]})
    with pytest.raises(ValueError) as exc:
        require_columns(df, ["trade_date", "ts_code", "factor"], context="compute_ic")
    msg = str(exc.value)
    assert "compute_ic" in msg
    assert "ts_code" in msg and "factor" in msg
    # 实际存在的列也应在错误信息中,便于排查
    assert "trade_date" in msg

def test_require_columns_does_not_flag_present_columns():
    df = pl.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError) as exc:
        require_columns(df, ["a", "c"])
    assert "c" in str(exc.value)
    # 'a' 已存在,不应被列为缺失(出现在“缺少必需列 [...]”片段里)
    missing_part = str(exc.value).split("实际列")[0]
    assert "'a'" not in missing_part

# ==== 来自 test_timing.py ====
def test_stage_timer_accumulates_named_durations():
    timer = StageTimer()
    with timer.stage("ic"):
        pass
    with timer.stage("backtest"):
        pass
    assert set(timer.timings) == {"ic", "backtest"}
    assert all(isinstance(v, float) and v >= 0 for v in timer.timings.values())

def test_stage_timer_logs_stage_name_and_duration(caplog):
    timer = StageTimer()
    with caplog.at_level(logging.INFO), timer.stage("报告生成"):
        pass
    assert any("报告生成" in r.getMessage() and "耗时" in r.getMessage() for r in caplog.records)

def test_stage_timer_records_duration_even_on_exception():
    timer = StageTimer()
    with pytest.raises(ValueError), timer.stage("boom"):
        raise ValueError("x")
    assert "boom" in timer.timings

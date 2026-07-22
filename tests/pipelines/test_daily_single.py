"""合并自: test_daily_single.py, test_daily_single_config.py
目标: test_daily_single.py

--- 来源 test_daily_single.py ---
test_daily_single_helpers.py：daily_single.py 纯 helper 的离线单测：日期归一、产物存在性、默认 YAML 配置查找。
test_daily_single_metrics.py：daily_single._write_run_metrics 单测：sweep 读取的 IC + 主策略回测指标 JSON。
test_daily_single_set_flag.py：daily_single 的 --set 接线测试：经 --dry-run 路径（纯配置流，无需数据）验证生效。

--- 来源 test_daily_single_config.py ---
Tests for run_daily_single configuration merging.
"""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import date, datetime
from types import SimpleNamespace

import polars as pl
import pytest

from factorzen.pipelines import daily_single
from factorzen.pipelines import daily_single as ds
from factorzen.pipelines.daily_single import _write_run_metrics


# ==== 来自 test_daily_single.py ====
# ==== 来自 test_daily_single_helpers.py ====
def test_date_expr_suite():
    """test_date_expr_parses_dash_format；test_date_expr_parses_plain_format；test_date_expr_invalid_becomes_null"""
    # -- 原 test_date_expr_parses_dash_format --
    def _section_0_test_date_expr_parses_dash_format():
        df = pl.DataFrame({"trade_date": ["2024-01-02"]}).with_columns(ds._date_expr("trade_date"))
        assert df["trade_date"].item() == date(2024, 1, 2)

    _section_0_test_date_expr_parses_dash_format()

    # -- 原 test_date_expr_parses_plain_format --
    def _section_1_test_date_expr_parses_plain_format():
        df = pl.DataFrame({"trade_date": ["20240102"]}).with_columns(ds._date_expr("trade_date"))
        assert df["trade_date"].item() == date(2024, 1, 2)

    _section_1_test_date_expr_parses_plain_format()

    # -- 原 test_date_expr_invalid_becomes_null --
    def _section_2_test_date_expr_invalid_becomes_null():
        df = pl.DataFrame({"trade_date": ["garbage"]}).with_columns(ds._date_expr("trade_date"))
        assert df["trade_date"].item() is None

    _section_2_test_date_expr_invalid_becomes_null()


# ── _ensure_date_column ─────────────────────────────────────

def test_ensure_date_column_suite():
    """test_ensure_date_passthrough_when_already_date；test_ensure_date_from_datetime；test_ensure_date_from_utf8；test_ensure_date_missing_column_passthrough"""
    # -- 原 test_ensure_date_passthrough_when_already_date --
    def _section_0_test_ensure_date_passthrough_when_already_date():
        df = pl.DataFrame({"trade_date": [date(2024, 1, 2)]})
        assert ds._ensure_date_column(df, "trade_date")["trade_date"].dtype == pl.Date

    _section_0_test_ensure_date_passthrough_when_already_date()

    # -- 原 test_ensure_date_from_datetime --
    def _section_1_test_ensure_date_from_datetime():
        df = pl.DataFrame({"trade_date": [datetime(2024, 1, 2, 9, 30)]})
        out = ds._ensure_date_column(df, "trade_date")
        assert out["trade_date"].dtype == pl.Date
        assert out["trade_date"].item() == date(2024, 1, 2)

    _section_1_test_ensure_date_from_datetime()

    # -- 原 test_ensure_date_from_utf8 --
    def _section_2_test_ensure_date_from_utf8():
        df = pl.DataFrame({"trade_date": ["20240102"]})
        out = ds._ensure_date_column(df, "trade_date")
        assert out["trade_date"].item() == date(2024, 1, 2)

    _section_2_test_ensure_date_from_utf8()

    # -- 原 test_ensure_date_missing_column_passthrough --
    def _section_3_test_ensure_date_missing_column_passthrough():
        df = pl.DataFrame({"x": [1]})
        assert ds._ensure_date_column(df, "trade_date").equals(df)

    _section_3_test_ensure_date_missing_column_passthrough()


# ── _existing_run_outputs
# ── _existing_run_outputs ───────────────────────────────────

def test_existing_run_outputs_suite(tmp_path):
    """test_existing_run_outputs_lists_present；test_existing_run_outputs_empty_when_none"""
    # -- 原 test_existing_run_outputs_lists_present --
    def _section_0_test_existing_run_outputs_lists_present(tmp_path, mp):
        mp.setattr(ds, "daily_factor_output_dir", lambda f: tmp_path / "factors")
        mp.setattr(ds, "daily_result_output_dir", lambda f: tmp_path / "results")
        mp.setattr(ds, "daily_report_output_dir", lambda f: tmp_path / "reports")
        (tmp_path / "results").mkdir(parents=True)
        (tmp_path / "results" / "mom_20240101_20240131_ic.parquet").write_text("x")

        out = ds._existing_run_outputs("mom", "20240101", "20240131")
        assert set(out) == {"ic"}
        assert out["ic"].endswith("_ic.parquet")

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_existing_run_outputs_lists_present(_tp0, mp)

    # -- 原 test_existing_run_outputs_empty_when_none --
    def _section_1_test_existing_run_outputs_empty_when_none(tmp_path, mp):
        mp.setattr(ds, "daily_factor_output_dir", lambda f: tmp_path / "factors")
        mp.setattr(ds, "daily_result_output_dir", lambda f: tmp_path / "results")
        mp.setattr(ds, "daily_report_output_dir", lambda f: tmp_path / "reports")
        assert ds._existing_run_outputs("mom", "20240101", "20240131") == {}

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_existing_run_outputs_empty_when_none(_tp1, mp)


# ── _find_default_run_config_path ───────────────────────────

def _write_yaml(path, factor):
    path.write_text(f"factor: {factor}\nstart: '20230101'\nend: '20231231'\n", encoding="utf-8")

def test_find_default_run_config_suite(tmp_path):
    """test_find_config_missing_dir_returns_none；test_find_config_no_match_returns_none；test_find_config_exact_stem_match；test_find_config_single_factor_prefix_preferred；test_find_config_ambiguous_raises"""
    # -- 原 test_find_config_missing_dir_returns_none --
    def _section_0_test_find_config_missing_dir_returns_none(tmp_path):
        assert ds._find_default_run_config_path("mom", "daily", configs_root=tmp_path) is None

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_find_config_missing_dir_returns_none(_tp0)

    # -- 原 test_find_config_no_match_returns_none --
    def _section_1_test_find_config_no_match_returns_none(tmp_path):
        d = tmp_path / "daily"
        d.mkdir()
        _write_yaml(d / "other.yaml", "value")
        assert ds._find_default_run_config_path("mom", "daily", configs_root=tmp_path) is None

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_find_config_no_match_returns_none(_tp1)

    # -- 原 test_find_config_exact_stem_match --
    def _section_2_test_find_config_exact_stem_match(tmp_path):
        d = tmp_path / "daily"
        d.mkdir()
        _write_yaml(d / "mom.yaml", "mom")
        _write_yaml(d / "another_mom.yaml", "mom")
        result = ds._find_default_run_config_path("mom", "daily", configs_root=tmp_path)
        assert result.stem == "mom"

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_find_config_exact_stem_match(_tp2)

    # -- 原 test_find_config_single_factor_prefix_preferred --
    def _section_3_test_find_config_single_factor_prefix_preferred(tmp_path):
        d = tmp_path / "daily"
        d.mkdir()
        _write_yaml(d / "single_factor_mom.yaml", "mom")
        _write_yaml(d / "batch_mom.yaml", "mom")
        result = ds._find_default_run_config_path("mom", "daily", configs_root=tmp_path)
        assert result.name == "single_factor_mom.yaml"

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_find_config_single_factor_prefix_preferred(_tp3)

    # -- 原 test_find_config_ambiguous_raises --
    def _section_4_test_find_config_ambiguous_raises(tmp_path):
        d = tmp_path / "daily"
        d.mkdir()
        _write_yaml(d / "alpha_mom.yaml", "mom")
        _write_yaml(d / "beta_mom.yaml", "mom")
        with pytest.raises(ValueError, match="多个默认配置"):
            ds._find_default_run_config_path("mom", "daily", configs_root=tmp_path)

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_find_config_ambiguous_raises(_tp4)


# ==== 来自 test_daily_single_metrics.py ====
def _fake_ic_result():
    return SimpleNamespace(
        ic_mean=0.0397,
        ir=0.13,
        ic_tstat=1.95,
        ic_positive_ratio=0.55,
        n_periods=241,
    )

def _fake_bt_result(portfolio):
    return SimpleNamespace(summary_stats={"portfolio": portfolio, "long_short": portfolio})

def test_write_run_metrics_suite(tmp_path):
    """test_write_run_metrics_includes_ic_and_backtest；回测 summary 缺 portfolio 键时回测指标置 None，不抛异常。"""
    # -- 原 test_write_run_metrics_includes_ic_and_backtest --
    def _section_0_test_write_run_metrics_includes_ic_and_backtest(tmp_path):
        path = tmp_path / "metrics.json"
        _write_run_metrics(
            str(path),
            _fake_ic_result(),
            _fake_bt_result({"sharpe": -1.15, "ann_ret": -0.022, "avg_turnover": 0.51, "max_dd": -0.035}),
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["ic_mean"] == 0.0397
        assert data["ir"] == 0.13
        assert data["t"] == 1.95
        assert data["ic_pos"] == 0.55
        assert data["n"] == 241
        assert data["sharpe"] == -1.15
        assert data["ann_ret"] == -0.022
        assert data["avg_turnover"] == 0.51
        assert data["max_dd"] == -0.035

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_write_run_metrics_includes_ic_and_backtest(_tp0)

    # -- 原 test_write_run_metrics_tolerates_missing_portfolio --
    def _section_1_test_write_run_metrics_tolerates_missing_portfolio(tmp_path):
        path = tmp_path / "metrics.json"
        bad_bt = SimpleNamespace(summary_stats={})
        _write_run_metrics(str(path), _fake_ic_result(), bad_bt)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["ic_mean"] == 0.0397
        assert data["sharpe"] is None
        assert data["ann_ret"] is None

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_write_run_metrics_tolerates_missing_portfolio(_tp1)

    # -- eval 轨 bt_result=None --
    def _section_2_test_write_run_metrics_tolerates_none_bt(tmp_path):
        path = tmp_path / "metrics.json"
        _write_run_metrics(str(path), _fake_ic_result(), None)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["ic_mean"] == 0.0397
        assert data["sharpe"] is None
        assert data["max_dd"] is None

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_write_run_metrics_tolerates_none_bt(_tp2)


# ==== 来自 test_daily_single_set_flag.py ====
def _run_dry(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["daily_single", *argv, "--dry-run"])
    daily_single.main()

def test_set_override_dry_run_suite(capsys, tmp_path):
    """无 YAML + --set backtest.top_n=30 → 单一 topn_30 策略（不套用 4 策略默认套件）。；test_set_override_preprocessing；对照：无 --set 无 --config 时维持研究预设（quantile_ls_5 单策略）。；test_set_override_with_config；test_set_override_invalid_value_exits；test_set_override_malformed_exits"""
    # -- 原 test_set_override_no_config_bakes_topn --
    def _section_0_test_set_override_no_config_bakes_topn(mp, capsys):
        _run_dry(
            mp,
            ["--factor", "f", "--start", "20230101", "--end", "20231231", "--set", "backtest.top_n=30"],
        )
        bt = json.loads(capsys.readouterr().out)["config"]["backtest"]
        assert bt["top_n"] == 30
        assert len(bt["strategies"]) == 1
        assert bt["strategies"][0]["name"] == "topn_30"
        assert bt["strategies"][0]["params"] == {"top_n": 30}

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_set_override_no_config_bakes_topn(mp, capsys)

    # -- 原 test_set_override_preprocessing --
    def _section_1_test_set_override_preprocessing(mp, capsys):
        _run_dry(
            mp,
            [
                "--factor",
                "f",
                "--start",
                "20230101",
                "--end",
                "20231231",
                "--set",
                "preprocessing.neutralize=true",
                "--set",
                "preprocessing.normalizer=rank_normal",
            ],
        )
        pp = json.loads(capsys.readouterr().out)["config"]["preprocessing"]
        assert pp["neutralize"] is True
        assert pp["normalizer"] == "rank_normal"

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_set_override_preprocessing(mp, capsys)

    # -- 原 test_no_set_no_config_keeps_default_suite --
    def _section_2_test_no_set_no_config_keeps_default_suite(mp, capsys):
        _run_dry(mp, ["--factor", "f", "--start", "20230101", "--end", "20231231"])
        bt = json.loads(capsys.readouterr().out)["config"]["backtest"]
        assert [s["name"] for s in bt["strategies"]] == ["quantile_ls_5"]
        assert bt["primary"] == "quantile_ls_5"

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_no_set_no_config_keeps_default_suite(mp, capsys)

    # -- 原 test_set_override_with_config --
    def _section_3_test_set_override_with_config(mp, capsys, tmp_path):
        cfg = tmp_path / "base.yaml"
        cfg.write_text(
            "factor: f\nstart: '20230101'\nend: '20231231'\nbacktest:\n  top_n: 50\n",
            encoding="utf-8",
        )
        _run_dry(mp, ["--config", str(cfg), "--set", "backtest.top_n=20"])
        bt = json.loads(capsys.readouterr().out)["config"]["backtest"]
        assert bt["top_n"] == 20
        assert bt["strategies"][0]["name"] == "topn_20"

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_set_override_with_config(mp, capsys, _tp3)

    # -- 原 test_set_override_invalid_value_exits --
    def _section_4_test_set_override_invalid_value_exits(mp):
        mp.setattr(
            "sys.argv",
            [
                "daily_single",
                "--factor",
                "f",
                "--start",
                "20230101",
                "--end",
                "20231231",
                "--set",
                "preprocessing.normalizer=bogus",
                "--dry-run",
            ],
        )
        with pytest.raises(SystemExit) as exc:
            daily_single.main()
        assert exc.value.code == 2

    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_set_override_invalid_value_exits(mp)

    # -- 原 test_set_override_malformed_exits --
    def _section_5_test_set_override_malformed_exits(mp):
        mp.setattr(
            "sys.argv",
            [
                "daily_single",
                "--factor",
                "f",
                "--start",
                "20230101",
                "--end",
                "20231231",
                "--set",
                "backtest.top_n",  # 缺 '='
                "--dry-run",
            ],
        )
        with pytest.raises(SystemExit) as exc:
            daily_single.main()
        assert exc.value.code == 2

    with pytest.MonkeyPatch.context() as mp:
        _section_5_test_set_override_malformed_exits(mp)


# ==== 来自 test_daily_single_config.py ====
def _ns(**kw):
    base = dict(
        factor=None,
        start=None,
        end=None,
        universe=None,
        benchmark=None,
        seed=None,
    )
    base.update(kw)
    return Namespace(**base)

def test_build_forward_return_frame_suite():
    """close_adj 偏好 / close 回退 / 分股 partial adj；显式 exec_price_col 缺列 fail-loudly。"""
    # -- 原 test_build_forward_return_frame_prefers_adjusted_close --
    def _section_0_test_build_forward_return_frame_prefers_adjusted_close():
        from factorzen.pipelines.daily_single import _build_forward_return_frame

        daily = pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 2), date(2024, 1, 3)],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "close": [10.0, 5.0],
                "close_adj": [10.0, 10.0],
            }
        )

        ret_df = _build_forward_return_frame(daily)

        assert ret_df["ret"][1] == pytest.approx(0.0)
        assert ret_df["fwd_ret_1d"][0] == pytest.approx(0.0)

    _section_0_test_build_forward_return_frame_prefers_adjusted_close()

    # -- 原 test_build_forward_return_frame_falls_back_to_close_without_adjusted_close --
    def _section_1_test_build_forward_return_frame_falls_back_to_close_without_adjusted_close():
        from factorzen.pipelines.daily_single import _build_forward_return_frame

        daily = pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 2), date(2024, 1, 3)],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "close": [10.0, 5.0],
            }
        )

        ret_df = _build_forward_return_frame(daily)

        assert ret_df["ret"][1] == pytest.approx(-0.5)
        assert ret_df["fwd_ret_1d"][0] == pytest.approx(-0.5)

    _section_1_test_build_forward_return_frame_falls_back_to_close_without_adjusted_close()

    # -- 原 test_build_forward_return_frame_falls_back_per_stock_for_partial_adjusted_close --
    def _section_2_test_build_forward_return_frame_falls_back_per_stock_for_partial_adjusted_close():
        from factorzen.pipelines.daily_single import _build_forward_return_frame

        daily = pl.DataFrame(
            {
                "trade_date": [
                    date(2024, 1, 3),
                    date(2024, 1, 3),
                    date(2024, 1, 2),
                    date(2024, 1, 2),
                ],
                "ts_code": ["000002.SZ", "000001.SZ", "000002.SZ", "000001.SZ"],
                "close": [50.0, 5.0, 100.0, 10.0],
                "close_adj": [220.0, None, 200.0, 10.0],
            }
        )

        ret_df = _build_forward_return_frame(daily)
        stock_a = ret_df.filter(pl.col("ts_code") == "000001.SZ").sort("trade_date")
        stock_b = ret_df.filter(pl.col("ts_code") == "000002.SZ").sort("trade_date")

        assert stock_a["ret"][1] == pytest.approx(-0.5)
        assert stock_a["fwd_ret_1d"][0] == pytest.approx(-0.5)
        assert stock_b["ret"][1] == pytest.approx(0.1)
        assert stock_b["fwd_ret_1d"][0] == pytest.approx(0.1)

    _section_2_test_build_forward_return_frame_falls_back_per_stock_for_partial_adjusted_close()

    # -- 显式 exec_price_col 缺列必须 fail-loudly，禁止静默回退 close→close --
    def _section_3_test_exec_price_col_missing_raises():
        from factorzen.pipelines.daily_single import _build_forward_return_frame

        daily = pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 2), date(2024, 1, 3)],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "close": [10.0, 11.0],
                "close_adj": [10.0, 11.0],
            }
        )
        with pytest.raises(ValueError, match="exec_price_col"):
            _build_forward_return_frame(daily, exec_lag=1, exec_price_col="open_adj")

    _section_3_test_exec_price_col_missing_raises()


def test_exec_price_col_vs_price_col_path_parity():
    """两条等价路径的 fwd_ret 数值对拍，防双路径漂移。

    A) daily_single 实现：``price_col=open_adj, exec_price_col=None``
    B) 生产/文档口径：``price_col=close_adj, exec_price_col=open_adj``

    在合成价格帧上 ``exec_lag=1`` 时 ``fwd_ret_1d`` 必须逐位一致。
    """
    from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns

    # 两只股票 × 4 日；open 与 close 系统错位，足以暴露路径分歧
    daily = pl.DataFrame(
        {
            "trade_date": (
                [date(2024, 1, d) for d in (2, 3, 4, 5)] * 2
            ),
            "ts_code": ["A"] * 4 + ["B"] * 4,
            "close_adj": [10.0, 11.0, 12.0, 13.0, 20.0, 19.0, 21.0, 22.0],
            "open_adj": [9.5, 10.5, 11.5, 12.5, 19.5, 18.5, 20.5, 21.5],
        }
    ).sort(["ts_code", "trade_date"])

    # A：与 _build_forward_return_frame 可实现分支同构
    path_a = compute_fwd_returns(
        daily.select(["trade_date", "ts_code", "open_adj"]).with_columns(
            (pl.col("open_adj") / pl.col("open_adj").shift(1).over("ts_code") - 1).alias("ret")
        ),
        ret_col="ret",
        price_col="open_adj",
        exec_lag=1,
        exec_price_col=None,
        horizons=[1],
    )

    # B：生产文档口径（标签价 close_adj + 成交价 open_adj）
    path_b = compute_fwd_returns(
        daily.select(["trade_date", "ts_code", "close_adj", "open_adj"]).with_columns(
            (pl.col("close_adj") / pl.col("close_adj").shift(1).over("ts_code") - 1).alias("ret")
        ),
        ret_col="ret",
        price_col="close_adj",
        exec_lag=1,
        exec_price_col="open_adj",
        horizons=[1],
    )

    a = path_a.sort(["ts_code", "trade_date"])["fwd_ret_1d"].to_list()
    b = path_b.sort(["ts_code", "trade_date"])["fwd_ret_1d"].to_list()
    assert len(a) == len(b) == 8
    for i, (va, vb) in enumerate(zip(a, b, strict=True)):
        if va is None or (isinstance(va, float) and va != va):  # NaN
            assert vb is None or (isinstance(vb, float) and vb != vb), i
        else:
            assert va == pytest.approx(vb, abs=1e-12, rel=0), (i, va, vb)

    # 与 _build_forward_return_frame 接线一致
    from factorzen.pipelines.daily_single import _build_forward_return_frame

    via_helper = _build_forward_return_frame(
        daily, exec_lag=1, exec_price_col="open_adj",
    ).sort(["ts_code", "trade_date"])["fwd_ret_1d"].to_list()
    for i, (va, vh) in enumerate(zip(a, via_helper, strict=True)):
        if va is None or (isinstance(va, float) and va != va):
            assert vh is None or (isinstance(vh, float) and vh != vh), i
        else:
            assert va == pytest.approx(vh, abs=1e-12, rel=0), (i, va, vh)


def test_run_backtest_strategies_wiring_suite():
    """test_run_backtest_strategies_runs_each_configured_strategy；ST涨跌停容差接线：_run_backtest_strategies 应基于 daily 的"""
    # -- 原 test_run_backtest_strategies_runs_each_configured_strategy --
    def _section_0_test_run_backtest_strategies_runs_each_configured_strategy(mp):
        from types import SimpleNamespace

        from factorzen.config.research import RunConfig
        from factorzen.pipelines import daily_single as mod

        cfg = RunConfig(
            factor="x",
            start="20230101",
            end="20230131",
            backtest={
                "primary": "topn_5",
                "strategies": [
                    {"name": "topn_5", "type": "topn_long_only", "params": {"top_n": 5}},
                    {
                        "name": "quantile_ls_4",
                        "type": "quantile_long_short",
                        "params": {"quantiles": 4},
                    },
                ],
            },
        )
        calls = []

        def fake_run_strategy_backtest(strategy, *_args, **_kwargs):
            calls.append(strategy.name)
            return SimpleNamespace(strategy_name=strategy.name)

        mp.setattr(mod, "run_strategy_backtest", fake_run_strategy_backtest)
        mp.setattr(mod, "trim_backtest_to_first_trade", lambda result: result)

        primary, results = mod._run_backtest_strategies(
            cfg,
            pl.DataFrame(),
            pl.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [date(2023, 1, 3)]}),
            factor_name="x",
            frequency="daily",
        )

        assert calls == ["topn_5", "quantile_ls_4"]
        assert primary.strategy_name == "topn_5"
        assert list(results) == ["topn_5", "quantile_ls_4"]

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_run_backtest_strategies_runs_each_configured_strategy(mp)

    # -- 原 test_run_backtest_strategies_passes_is_st_by_date_to_backtest --
    def _section_1_test_run_backtest_strategies_passes_is_st_by_date_to_backtest(mp):
        from types import SimpleNamespace

        from factorzen.config.research import RunConfig
        from factorzen.pipelines import daily_single as mod

        cfg = RunConfig(
            factor="x",
            start="20230101",
            end="20230131",
            backtest={
                "primary": "topn_5",
                "strategies": [
                    {"name": "topn_5", "type": "topn_long_only", "params": {"top_n": 5}},
                ],
            },
        )
        captured: dict = {}

        def fake_run_strategy_backtest(strategy, *_args, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(strategy_name=strategy.name)

        sentinel = {date(2023, 1, 3): {"000001.SZ"}}
        mp.setattr(mod, "run_strategy_backtest", fake_run_strategy_backtest)
        mp.setattr(mod, "trim_backtest_to_first_trade", lambda result: result)
        mp.setattr(mod, "build_is_st_by_date", lambda codes, dates: sentinel)

        daily = pl.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [date(2023, 1, 3)]})
        mod._run_backtest_strategies(cfg, pl.DataFrame(), daily, factor_name="x", frequency="daily")

        assert captured.get("is_st_by_date") == sentinel, (
            "run_strategy_backtest 应收到由 build_is_st_by_date 构建的 is_st_by_date，"
            f"实际收到: {captured.get('is_st_by_date')!r}"
        )

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_run_backtest_strategies_passes_is_st_by_date_to_backtest(mp)


def test_merge_run_config_and_dry_run_suite():
    """test_merge_run_config_args_uses_yaml_for_missing_cli_values；test_merge_run_config_args_keeps_explicit_cli_values；test_merge_run_config_args_keeps_explicit_cli_benchmark；test_dry_run_payload_includes_effective_config_and_output_dir；防回归：合并后的 namespace / dry-run payload 不再含深度评估键。；test_effective_run_config_without_yaml_uses_quantile_ls_5"""
    # -- 原 test_merge_run_config_args_uses_yaml_for_missing_cli_values --
    def _section_0_test_merge_run_config_args_uses_yaml_for_missing_cli_values():
        from factorzen.config.research import RunConfig
        from factorzen.pipelines.daily_single import _merge_run_config_args

        args = _ns()
        cfg = RunConfig(
            factor="momentum_20d",
            start="20230101",
            end="20241231",
            universe="csi500",
            benchmark=None,
            seed=42,
        )

        merged = _merge_run_config_args(args, cfg)

        assert merged.factor == "momentum_20d"
        assert merged.start == "20230101"
        assert merged.end == "20241231"
        assert merged.universe == "csi500"
        assert merged.benchmark == "000905.SH"
        assert merged.seed == 42

    _section_0_test_merge_run_config_args_uses_yaml_for_missing_cli_values()

    # -- 原 test_merge_run_config_args_keeps_explicit_cli_values --
    def _section_1_test_merge_run_config_args_keeps_explicit_cli_values():
        from factorzen.config.research import RunConfig
        from factorzen.pipelines.daily_single import _merge_run_config_args

        args = _ns(
            factor="reversal_5d",
            start="20240101",
            end="20241231",
            universe="csi300",
            benchmark=None,
            seed=7,
        )
        cfg = RunConfig(
            factor="momentum_20d",
            start="20230101",
            end="20231231",
            universe="csi500",
            benchmark="000905.SH",
            seed=42,
        )

        merged = _merge_run_config_args(args, cfg)

        assert merged.factor == "reversal_5d"
        assert merged.start == "20240101"
        assert merged.end == "20241231"
        assert merged.universe == "csi300"
        assert merged.benchmark == "000905.SH"
        assert merged.seed == 7

    _section_1_test_merge_run_config_args_keeps_explicit_cli_values()

    # -- 原 test_merge_run_config_args_keeps_explicit_cli_benchmark --
    def _section_2_test_merge_run_config_args_keeps_explicit_cli_benchmark():
        from factorzen.config.research import RunConfig
        from factorzen.pipelines.daily_single import _merge_run_config_args

        args = _ns(benchmark="000852.SH")
        cfg = RunConfig(
            factor="momentum_20d",
            start="20230101",
            end="20231231",
            universe="csi500",
        )

        merged = _merge_run_config_args(args, cfg)

        assert merged.benchmark == "000852.SH"

    _section_2_test_merge_run_config_args_keeps_explicit_cli_benchmark()

    # -- 原 test_dry_run_payload_includes_effective_config_and_output_dir --
    def _section_3_test_dry_run_payload_includes_effective_config_and_output_dir():
        from factorzen.config.research import RunConfig
        from factorzen.pipelines.daily_single import _build_dry_run_payload

        cfg = RunConfig(
            factor="momentum_20d",
            start="20230101",
            end="20231231",
            universe="csi500",
            benchmark="000905.SH",
            backtest={"top_n": 25},
            walk_forward={"n_trials": 3},
        )

        payload = _build_dry_run_payload(cfg)

        assert payload["config"]["benchmark"] == "000905.SH"
        assert payload["config"]["backtest"]["top_n"] == 25
        assert payload["config"]["walk_forward"]["n_trials"] == 3
        assert payload["output_dir"].endswith("workspace/factor_evaluations/<run_id>")
        assert "execution" not in payload
        for banned in ("ic_method", "neutralized_ic", "event_study", "llm_explain", "llm_refresh"):
            assert banned not in payload["config"]

    _section_3_test_dry_run_payload_includes_effective_config_and_output_dir()

    # -- 原 test_merge_run_config_args_and_dry_run_drop_deep_eval_keys --
    def _section_4_test_merge_run_config_args_and_dry_run_drop_deep_eval_keys():
        from factorzen.pipelines.daily_single import (
            _build_dry_run_payload,
            _effective_run_config,
            _merge_run_config_args,
        )

        args = _ns(
            factor="momentum_20d",
            start="20240101",
            end="20240131",
        )
        # 旧 YAML 字段应被 ignore，不应出现在 args 上
        merged = _merge_run_config_args(args, None)
        for banned in ("ic_method", "neutralized_ic", "event_study", "llm_explain", "llm_refresh", "all"):
            assert not hasattr(merged, banned) or getattr(merged, banned, None) is None
            # 更严格：合并逻辑不应写入这些属性
            assert banned not in vars(merged)

        cfg = _effective_run_config(merged, None)
        dumped = cfg.model_dump()
        for banned in ("ic_method", "neutralized_ic", "event_study"):
            assert banned not in dumped

        payload = _build_dry_run_payload(cfg, args=merged)
        assert "execution" not in payload
        for banned in ("llm_explain", "llm_refresh", "ic_method", "neutralized_ic", "event_study"):
            assert banned not in payload.get("execution", {})
            assert banned not in payload["config"]

    _section_4_test_merge_run_config_args_and_dry_run_drop_deep_eval_keys()

    # -- 原 test_effective_run_config_without_yaml_uses_quantile_ls_5 --
    def _section_5_test_effective_run_config_without_yaml_uses_quantile_ls_5():
        from factorzen.pipelines.daily_single import _effective_run_config, _merge_run_config_args

        args = _ns(
            factor="momentum_20d",
            start="20240101",
            end="20240131",
        )

        merged = _merge_run_config_args(args, None)
        cfg = _effective_run_config(merged, None)

        assert cfg.universe == "csi500"
        assert cfg.benchmark == "000905.SH"
        assert cfg.seed == 42
        assert cfg.preprocessing.neutralize is True
        assert cfg.preprocessing.neutralize_by == "industry+size"
        assert cfg.backtest.primary == "quantile_ls_5"
        assert [spec.name for spec in cfg.backtest.strategy_specs] == ["quantile_ls_5"]
        assert cfg.backtest.strategy_specs[0].type == "quantile_long_short"
        assert cfg.backtest.strategy_specs[0].params == {"quantiles": 5}

    _section_5_test_effective_run_config_without_yaml_uses_quantile_ls_5()


def test_neutralization_and_ensure_data_suite():
    """test_preprocess_with_industry_neutralization_uses_universe_industry；test_load_daily_basic_for_neutralization_reads_ensured_cache；test_run_ensures_required_data_before_loading_universe"""
    # -- 原 test_preprocess_with_industry_neutralization_uses_universe_industry --
    def _section_0_test_preprocess_with_industry_neutralization_uses_universe_industry():
        from factorzen.config.research import RunConfig
        from factorzen.pipelines.daily_single import _preprocess_factor

        rows = []
        universe_rows = []
        for i in range(40):
            code = f"{i:06d}.SZ"
            industry = "银行" if i < 20 else "医药"
            value = 1.0 if industry == "银行" else -1.0
            rows.append({"trade_date": "2024-01-02", "ts_code": code, "factor_value": value})
            universe_rows.append({"ts_code": code, "industry": industry})

        cfg = RunConfig(
            factor="momentum_20d",
            start="20240101",
            end="20240131",
            preprocessing={"neutralize": True, "neutralize_by": "industry"},
        )

        clean = _preprocess_factor(
            pl.DataFrame(rows),
            cfg,
            universe=pl.DataFrame(universe_rows),
            daily_basic=None,
        )

        by_industry = (
            clean.join(pl.DataFrame(universe_rows), on="ts_code")
            .group_by("industry")
            .agg(pl.col("factor_clean").mean().alias("mean_factor"))
        )
        assert by_industry["mean_factor"].abs().max() < 1e-10

    _section_0_test_preprocess_with_industry_neutralization_uses_universe_industry()

    # -- 原 test_load_daily_basic_for_neutralization_reads_ensured_cache --
    def _section_1_test_load_daily_basic_for_neutralization_reads_ensured_cache(mp):
        import factorzen.pipelines.daily_single as run_mod

        expected = pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 2)],
                "ts_code": ["000001.SZ"],
                "total_mv": [100.0],
            }
        )
        calls: list[tuple[str, str, str]] = []

        class _LazyFrameStub:
            def collect(self):
                return expected

        def fake_load_parquet(data_type: str, *, start: str, end: str):
            calls.append((data_type, start, end))
            return _LazyFrameStub()

        mp.setattr(run_mod, "load_parquet", fake_load_parquet)

        result = run_mod._load_daily_basic_for_neutralization("20240102", "20240103")

        assert result.equals(expected)
        assert calls == [("daily_basic", "20240102", "20240103")]

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_load_daily_basic_for_neutralization_reads_ensured_cache(mp)

    # -- 原 test_run_ensures_required_data_before_loading_universe --
    def _section_2_test_run_ensures_required_data_before_loading_universe(mp):
        import factorzen.pipelines.daily_single as run_mod
        from factorzen.config.research import RunConfig

        calls: list[str] = []

        class DummyFactor:
            name = "dummy_factor"
            description = "dummy"
            required_data = ["daily"]
            lookback_days = 20

        def fake_ensure_data_for_daily_run(**kwargs):
            calls.append("ensure")
            assert kwargs["required_data"] == ["daily"]
            assert kwargs["start"] == "20240102"
            assert kwargs["end"] == "20240103"

        def fake_load_pit_membership(*args, **kwargs):
            calls.append("universe")
            raise RuntimeError("stop after data ensure")

        mp.setattr(run_mod, "get_factor", lambda name: DummyFactor)
        mp.setattr(run_mod, "get_trade_dates", lambda start, end: [date(2024, 1, 2)])
        mp.setattr(run_mod, "ensure_data_for_daily_run", fake_ensure_data_for_daily_run)
        mp.setattr(run_mod, "load_pit_membership", fake_load_pit_membership)

        args = Namespace(
            factor="dummy_factor",
            start="20240102",
            end="20240103",
            universe="csi300",
            frequency="daily",
            benchmark=None,
            seed=None,
        )

        with pytest.raises(RuntimeError, match="stop after data ensure"):
            run_mod._prepare_evaluation_inputs(
                args, RunConfig(factor="dummy_factor", start="20240102", end="20240103")
            )

        assert calls == ["ensure", "universe"]

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_run_ensures_required_data_before_loading_universe(mp)


# -- signal 落盘 --
def test_signal_backtest_artifacts_suite(tmp_path):
    """信号层回测落盘契约：_signal.json 含 ann_ret_gross；_signal_group_nav.parquet 列齐全。

    现有 suite 无完整 _run 形态，按约定退而测与 daily_single 11b 同构的落盘路径。
    """
    from datetime import date, timedelta

    from factorzen.daily.evaluation.signal_backtest import run_signal_backtest

    # 10 股 × 8 日，n_groups=5 足够有效分组
    n_days, n_stocks, n_groups = 8, 10, 5
    dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(n_days)]
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    f_rows, r_rows = [], []
    for di, d in enumerate(dates):
        for si, c in enumerate(codes):
            f_rows.append(
                {
                    "trade_date": d,
                    "ts_code": c,
                    "factor_clean": float(si) + 0.01 * di,
                }
            )
            r_rows.append(
                {
                    "trade_date": d,
                    "ts_code": c,
                    "fwd_ret_1d": 0.001 * si + 0.0001 * di,
                }
            )
    factor_df = pl.DataFrame(f_rows)
    ret_df = pl.DataFrame(r_rows)

    signal_result = run_signal_backtest(
        factor_df,
        ret_df,
        factor_col="factor_clean",
        n_groups=n_groups,
        frequency="daily",
        factor_name="dummy_factor",
        meta={"exec_lag": 1, "exec_price_col": "open_adj", "direction": "long"},
    )

    factor_name, start, end = "dummy_factor", "20240102", "20240109"
    result_dir = tmp_path / "results"
    result_dir.mkdir(parents=True)

    # 与 daily_single 11b 同构落盘
    signal_json_path = result_dir / f"{factor_name}_{start}_{end}_signal.json"
    signal_json_path.write_text(
        json.dumps(
            {
                "summary_stats": signal_result.summary_stats,
                "meta": signal_result.meta,
                "n_groups": signal_result.n_groups,
                "cost_bps": signal_result.cost_bps,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    signal_nav_path = result_dir / f"{factor_name}_{start}_{end}_signal_group_nav.parquet"
    signal_result.group_nav.write_parquet(str(signal_nav_path))

    assert signal_json_path.exists()
    payload = json.loads(signal_json_path.read_text(encoding="utf-8"))
    assert "ann_ret_gross" in payload["summary_stats"]["long_short"]
    assert payload["meta"]["return_basis"] == "gross_signal_level"
    assert payload["n_groups"] == n_groups

    nav = pl.read_parquet(str(signal_nav_path))
    assert {"trade_date", "group", "nav"}.issubset(set(nav.columns))
    assert not nav.is_empty()
    # 独立期望：5 组均应有净值序列
    assert set(nav["group"].to_list()) == set(range(n_groups))


# -- eval / backtest 双轨拆分产物 --
def test_eval_backtest_track_artifacts_suite(tmp_path):
    """eval 轨与 backtest 轨产物隔离：各自该落的在、不该跑的没跑。"""
    from datetime import date, timedelta

    from factorzen.config.research import RunConfig
    from factorzen.pipelines import daily_single as ds

    n_days, n_stocks = 6, 10
    dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(n_days)]
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    start, end = "20240102", "20240107"
    factor_name = "track_split_dummy"

    def _factor_rows():
        rows = []
        for di, d in enumerate(dates):
            for si, c in enumerate(codes):
                rows.append(
                    {
                        "trade_date": d,
                        "ts_code": c,
                        "factor_value": float(si) + 0.01 * di,
                    }
                )
        return pl.DataFrame(rows)

    def _daily_rows():
        rows = []
        for di, d in enumerate(dates):
            for si, c in enumerate(codes):
                px = 10.0 + 0.01 * si + 0.001 * di
                rows.append(
                    {
                        "trade_date": d,
                        "ts_code": c,
                        "close": px,
                        "close_adj": px,
                        "open": px,
                        "open_adj": px,
                        "high": px,
                        "low": px,
                        "vol": 1e5,
                        "amount": 1e6,
                    }
                )
        return pl.DataFrame(rows)

    mem = pl.DataFrame(
        [{"trade_date": d.strftime("%Y%m%d"), "ts_code": c} for d in dates for c in codes]
    )

    class DummyFactor:
        name = factor_name
        description = "track split"
        required_data = ["daily"]
        lookback_days = 1
        category = "test"

        def compute(self, ctx):
            return _factor_rows()

        def validate(self, df):
            return {"coverage": 1.0, "n_rows": df.height}

    class FakeCtx:
        def __init__(self, **kw):
            pass

        @property
        def daily(self):
            return _daily_rows().lazy()

    def _fake_ic(clean_df, ret_df, **kw):
        ic_series = pl.DataFrame(
            {
                "trade_date": dates,
                "ic": [0.02] * len(dates),
            }
        )
        return SimpleNamespace(
            ic_mean=0.02,
            ir=0.5,
            ic_tstat=2.0,
            ic_pvalue=0.05,
            n_periods=len(dates),
            ic_positive_ratio=0.6,
            ic_series=ic_series,
            factor_name=None,
            summary=lambda: "ic ok",
        )

    def _fake_turnover(df, **kw):
        return SimpleNamespace(
            avg_turnover=0.2,
            factor_name=None,
            summary=lambda: "to ok",
        )

    def _fake_signal(factor_df, ret_df, **kw):
        group_nav = pl.DataFrame(
            {
                "trade_date": dates * 2,
                "group": [0] * len(dates) + [1] * len(dates),
                "nav": [1.0] * (len(dates) * 2),
            }
        )
        return SimpleNamespace(
            summary_stats={"long_short": {"ann_ret_gross": 0.1}},
            meta={"return_basis": "gross_signal_level"},
            n_groups=5,
            cost_bps=0.0,
            group_nav=group_nav,
            summary=lambda: "signal ok",
        )

    def _fake_bt_strategies(config, clean_df, daily, **kw):
        nav = pl.DataFrame({"trade_date": dates, "nav": [1.0 + i * 0.001 for i in range(len(dates))]})
        rets = pl.DataFrame({"trade_date": dates, "ret": [0.001] * len(dates)})
        bt = SimpleNamespace(
            summary_stats={
                "portfolio": {
                    "sharpe": 0.5,
                    "ann_ret": 0.1,
                    "avg_turnover": 0.2,
                    "max_dd": -0.05,
                }
            },
            nav=nav,
            returns=rets,
            summary=lambda: "bt ok",
        )
        return bt, {"quantile_ls_5": bt}

    def _install_common_mocks(mp, out_root):
        mp.setattr(ds, "get_factor", lambda n: DummyFactor)
        mp.setattr(ds, "get_trade_dates", lambda s, e: dates)
        mp.setattr(ds, "ensure_data_for_daily_run", lambda **kw: None)
        mp.setattr(
            "factorzen.core.universe.get_universe_membership",
            lambda s, e, u: mem,
        )
        mp.setattr(
            ds,
            "get_universe",
            lambda d, u: pl.DataFrame({"ts_code": codes, "industry": ["银行"] * len(codes)}),
        )
        mp.setattr(ds, "FactorDataContext", FakeCtx)
        mp.setattr(ds, "compute_rank_ic", _fake_ic)
        mp.setattr(ds, "compute_turnover", _fake_turnover)
        mp.setattr(
            ds,
            "build_daily_quality_report",
            lambda **kw: {"warnings": [], "status": "ok"},
        )
        mp.setattr(
            ds,
            "_preprocess_factor",
            lambda factor_df, cfg, **kw: factor_df.with_columns(
                pl.col("factor_value").alias("factor_clean")
            ),
        )
        mp.setattr(ds, "daily_factor_output_dir", lambda name: out_root / "factors" / name)
        mp.setattr(ds, "daily_result_output_dir", lambda name: out_root / "results" / name)
        mp.setattr(ds, "daily_report_output_dir", lambda name: out_root / "reports" / name)
        mp.setattr(
            "factorzen.pipelines._report_persistence.daily_result_output_dir",
            lambda name: out_root / "results" / name,
        )
        mp.setattr(ds, "generate_tear_sheet", lambda *a, **k: "<html>ok</html>")
        mp.setattr(
            ds,
            "_compute_monotonicity_result",
            lambda *a, **k: None,
        )
        mp.setattr(
            ds,
            "_decide_backtest_direction",
            lambda ic: {"direction": "normal", "should_reverse": False, "reason": "test"},
        )
        mp.setattr(
            ds,
            "_apply_backtest_direction",
            lambda df, d: df,
        )

    args = SimpleNamespace(
        factor=factor_name,
        start=start,
        end=end,
        universe="csi300",
        frequency="daily",
        benchmark=None,
        seed=None,
        metrics_out=None,
        exec_lag=1,
        exec_price_col="open_adj",
    )
    cfg = RunConfig(factor=factor_name, start=start, end=end)

    # -- eval 轨 --
    def _section_0_eval_track(tmp_path, mp):
        out = tmp_path / "eval"
        _install_common_mocks(mp, out)
        signal_called = {"n": 0}
        bt_called = {"n": 0}
        wf_called = {"n": 0}

        def fake_signal(*a, **k):
            signal_called["n"] += 1
            return _fake_signal(*a, **k)

        def fake_bt(*a, **k):
            bt_called["n"] += 1
            return _fake_bt_strategies(*a, **k)

        def fake_wf(*a, **k):
            wf_called["n"] += 1
            return {"status": "ok", "n_folds": 1}, None

        mp.setattr(ds, "run_signal_backtest", fake_signal)
        mp.setattr(ds, "_run_backtest_strategies", fake_bt)
        mp.setattr(ds, "run_quantile_walk_forward_summary", fake_wf)

        outputs = ds.run_factor_eval(args, cfg)
        result_dir = out / "results" / factor_name
        report_dir = out / "reports" / factor_name
        prefix = f"{factor_name}_{start}_{end}"

        assert signal_called["n"] == 1
        assert bt_called["n"] == 0, "eval 轨不得跑策略日环"
        assert wf_called["n"] == 0, "eval 轨不得跑 walk-forward"
        assert (result_dir / f"{prefix}_signal.json").exists()
        assert (result_dir / f"{prefix}_signal_group_nav.parquet").exists()
        assert (result_dir / f"{prefix}_ic.parquet").exists()
        assert (report_dir / f"{prefix}_eval.html").exists()
        assert not (result_dir / f"{prefix}_walk_forward.json").exists()
        assert not (report_dir / f"{prefix}.html").exists()
        assert "signal" in outputs
        assert "walk_forward_summary" not in outputs

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_eval_track(_tp0, mp)

    # -- backtest 轨 --
    def _section_1_backtest_track(tmp_path, mp):
        out = tmp_path / "bt"
        _install_common_mocks(mp, out)
        signal_called = {"n": 0}
        bt_called = {"n": 0}
        wf_called = {"n": 0}

        def fake_signal(*a, **k):
            signal_called["n"] += 1
            return _fake_signal(*a, **k)

        def fake_bt(*a, **k):
            bt_called["n"] += 1
            return _fake_bt_strategies(*a, **k)

        def fake_wf(*a, **k):
            wf_called["n"] += 1
            return {"status": "disabled", "n_folds": 0}, None

        mp.setattr(ds, "run_signal_backtest", fake_signal)
        mp.setattr(ds, "_run_backtest_strategies", fake_bt)
        mp.setattr(ds, "run_quantile_walk_forward_summary", fake_wf)

        outputs = ds.run_factor_backtest(args, cfg)
        result_dir = out / "results" / factor_name
        report_dir = out / "reports" / factor_name
        prefix = f"{factor_name}_{start}_{end}"

        assert bt_called["n"] == 1
        assert signal_called["n"] == 0, "backtest 轨不得跑信号回测"
        assert (result_dir / f"{prefix}_walk_forward.json").exists()
        assert (result_dir / f"{prefix}_ic.parquet").exists()
        assert (report_dir / f"{prefix}.html").exists()
        assert not (result_dir / f"{prefix}_signal.json").exists()
        assert not (report_dir / f"{prefix}_eval.html").exists()
        assert "walk_forward_summary" in outputs
        assert "signal" not in outputs

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_backtest_track(_tp1, mp)


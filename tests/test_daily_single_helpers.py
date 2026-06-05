"""daily_single.py 纯 helper 的离线单测：日期归一、行业查表、中性化 IC 帧装配、
产物存在性、默认 YAML 配置查找。补 test_run_daily_single_config.py 未覆盖的辅助函数。
"""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from factorzen.pipelines import daily_single as ds

# ── _date_expr ──────────────────────────────────────────────


def test_date_expr_parses_dash_format():
    df = pl.DataFrame({"trade_date": ["2024-01-02"]}).with_columns(ds._date_expr("trade_date"))
    assert df["trade_date"].item() == date(2024, 1, 2)


def test_date_expr_parses_plain_format():
    df = pl.DataFrame({"trade_date": ["20240102"]}).with_columns(ds._date_expr("trade_date"))
    assert df["trade_date"].item() == date(2024, 1, 2)


def test_date_expr_invalid_becomes_null():
    df = pl.DataFrame({"trade_date": ["garbage"]}).with_columns(ds._date_expr("trade_date"))
    assert df["trade_date"].item() is None


# ── _ensure_date_column ─────────────────────────────────────


def test_ensure_date_passthrough_when_already_date():
    df = pl.DataFrame({"trade_date": [date(2024, 1, 2)]})
    assert ds._ensure_date_column(df, "trade_date")["trade_date"].dtype == pl.Date


def test_ensure_date_from_datetime():
    df = pl.DataFrame({"trade_date": [datetime(2024, 1, 2, 9, 30)]})
    out = ds._ensure_date_column(df, "trade_date")
    assert out["trade_date"].dtype == pl.Date
    assert out["trade_date"].item() == date(2024, 1, 2)


def test_ensure_date_from_utf8():
    df = pl.DataFrame({"trade_date": ["20240102"]})
    out = ds._ensure_date_column(df, "trade_date")
    assert out["trade_date"].item() == date(2024, 1, 2)


def test_ensure_date_missing_column_passthrough():
    df = pl.DataFrame({"x": [1]})
    assert ds._ensure_date_column(df, "trade_date").equals(df)


# ── _sector_lookup ──────────────────────────────────────────


def test_sector_lookup_renames_and_dedupes():
    universe = pl.DataFrame(
        {
            "ts_code": ["A", "A", "B", "C"],
            "industry": ["银行", "银行", "", None],
        }
    )
    out = ds._sector_lookup(universe)
    assert set(out.columns) == {"ts_code", "sector"}
    # A 去重保留，B(空)/C(null) 被剔除
    assert out["ts_code"].to_list() == ["A"]


def test_sector_lookup_empty_universe():
    out = ds._sector_lookup(pl.DataFrame(schema={"ts_code": pl.Utf8, "industry": pl.Utf8}))
    assert out.is_empty()
    assert set(out.columns) == {"ts_code", "sector"}


def test_sector_lookup_missing_industry_column():
    out = ds._sector_lookup(pl.DataFrame({"ts_code": ["A"]}))
    assert out.is_empty()


# ── _build_neutralized_ic_frame ─────────────────────────────


def _clean_df():
    return pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2)],
            "ts_code": ["A", "B"],
            "factor_clean": [1.0, 2.0],
        }
    )


def _ret_df():
    return pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2)],
            "ts_code": ["A", "B"],
            "fwd_ret_1d": [0.01, -0.02],
        }
    )


def test_neutralized_ic_frame_joins_industry_and_size():
    universe = pl.DataFrame({"ts_code": ["A", "B"], "industry": ["银行", "地产"]})
    daily_basic = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2)],
            "ts_code": ["A", "B"],
            "total_mv": [1e9, 2e9],
        }
    )
    frame = ds._build_neutralized_ic_frame(
        _clean_df(), _ret_df(), universe=universe, daily_basic=daily_basic
    )
    assert "ret_1d" in frame.columns
    assert "industry" in frame.columns
    assert "total_mv" in frame.columns
    assert frame.height == 2


def test_neutralized_ic_frame_without_industry_or_size():
    universe = pl.DataFrame({"ts_code": ["A", "B"]})  # 无 industry 列
    frame = ds._build_neutralized_ic_frame(
        _clean_df(), _ret_df(), universe=universe, daily_basic=None
    )
    assert "ret_1d" in frame.columns
    assert "industry" not in frame.columns
    assert "total_mv" not in frame.columns


# ── _existing_run_outputs ───────────────────────────────────


def test_existing_run_outputs_lists_present(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "daily_factor_output_dir", lambda f: tmp_path / "factors")
    monkeypatch.setattr(ds, "daily_result_output_dir", lambda f: tmp_path / "results")
    monkeypatch.setattr(ds, "daily_report_output_dir", lambda f: tmp_path / "reports")
    (tmp_path / "results").mkdir(parents=True)
    (tmp_path / "results" / "mom_20240101_20240131_ic.parquet").write_text("x")

    out = ds._existing_run_outputs("mom", "20240101", "20240131")
    assert set(out) == {"ic"}
    assert out["ic"].endswith("_ic.parquet")


def test_existing_run_outputs_empty_when_none(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "daily_factor_output_dir", lambda f: tmp_path / "factors")
    monkeypatch.setattr(ds, "daily_result_output_dir", lambda f: tmp_path / "results")
    monkeypatch.setattr(ds, "daily_report_output_dir", lambda f: tmp_path / "reports")
    assert ds._existing_run_outputs("mom", "20240101", "20240131") == {}


# ── _find_default_run_config_path ───────────────────────────


def _write_yaml(path, factor):
    path.write_text(f"factor: {factor}\nstart: '20230101'\nend: '20231231'\n", encoding="utf-8")


def test_find_config_missing_dir_returns_none(tmp_path):
    assert ds._find_default_run_config_path("mom", "daily", configs_root=tmp_path) is None


def test_find_config_no_match_returns_none(tmp_path):
    d = tmp_path / "daily"
    d.mkdir()
    _write_yaml(d / "other.yaml", "value")
    assert ds._find_default_run_config_path("mom", "daily", configs_root=tmp_path) is None


def test_find_config_exact_stem_match(tmp_path):
    d = tmp_path / "daily"
    d.mkdir()
    _write_yaml(d / "mom.yaml", "mom")
    _write_yaml(d / "another_mom.yaml", "mom")
    result = ds._find_default_run_config_path("mom", "daily", configs_root=tmp_path)
    assert result.stem == "mom"


def test_find_config_single_factor_prefix_preferred(tmp_path):
    d = tmp_path / "daily"
    d.mkdir()
    _write_yaml(d / "single_factor_mom.yaml", "mom")
    _write_yaml(d / "batch_mom.yaml", "mom")
    result = ds._find_default_run_config_path("mom", "daily", configs_root=tmp_path)
    assert result.name == "single_factor_mom.yaml"


def test_find_config_ambiguous_raises(tmp_path):
    d = tmp_path / "daily"
    d.mkdir()
    _write_yaml(d / "alpha_mom.yaml", "mom")
    _write_yaml(d / "beta_mom.yaml", "mom")
    with pytest.raises(ValueError, match="多个默认配置"):
        ds._find_default_run_config_path("mom", "daily", configs_root=tmp_path)

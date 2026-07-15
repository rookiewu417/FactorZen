"""attach_intraday：注入 join、缺面板 null+warn、require raise、out_meta、leaf_health。"""
from __future__ import annotations

import datetime as dt
import warnings

import polars as pl
import pytest

from factorzen.core.feature_schema import INTRADAY_FEATURES
from factorzen.daily.data.intraday import attach_intraday

_COLS = sorted(INTRADAY_FEATURES)


def _daily_date(dates: list[str], code: str = "000001.SZ") -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": [dt.datetime.strptime(d, "%Y%m%d").date() for d in dates],
        "ts_code": [code] * len(dates),
        "close": [10.0] * len(dates),
    })


def _daily_utf8(dates: list[str], code: str = "000001.SZ") -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": dates,
        "ts_code": [code] * len(dates),
        "close": [10.0] * len(dates),
    })


def _panel(dates: list[str], code: str = "000001.SZ", *, as_date: bool = True) -> pl.DataFrame:
    td = (
        [dt.datetime.strptime(d, "%Y%m%d").date() for d in dates]
        if as_date
        else list(dates)
    )
    data: dict = {"trade_date": td, "ts_code": [code] * len(dates)}
    for i, c in enumerate(_COLS):
        data[c] = [float(i + 1) + 0.1 * j for j in range(len(dates))]
    return pl.DataFrame(data)


def test_injected_join_date_dtype():
    daily = _daily_date(["20240102", "20240103"])
    panel = _panel(["20240102", "20240103"])
    # 显式固定 i_rv 便于断言
    panel = panel.with_columns(
        pl.when(pl.col("trade_date") == dt.date(2024, 1, 2))
        .then(0.42)
        .otherwise(0.43)
        .alias("i_rv")
    )
    out = attach_intraday(daily, injected=panel)
    for c in _COLS:
        assert c in out.columns
    by = {r["trade_date"]: r for r in out.iter_rows(named=True)}
    assert by[dt.date(2024, 1, 2)]["i_rv"] == pytest.approx(0.42)
    assert by[dt.date(2024, 1, 3)]["i_rv"] == pytest.approx(0.43)


def test_injected_join_utf8_daily():
    """daily trade_date 为 Utf8 时也能 left-join。"""
    daily = _daily_utf8(["20240102", "20240103"])
    panel = _panel(["20240102", "20240103"], as_date=True).with_columns(
        pl.lit(0.99).alias("i_rv")
    )
    out = attach_intraday(daily, injected=panel)
    assert out["trade_date"].dtype in (pl.Utf8, pl.String)
    assert out.filter(pl.col("trade_date") == "20240102")["i_rv"][0] == pytest.approx(0.99)


def test_missing_panel_require_false_nulls_and_warning():
    daily = _daily_date(["20240102"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = attach_intraday(daily, injected=pl.DataFrame(), require=False)
    assert any("intraday" in str(x.message).lower() or "i_*" in str(x.message)
               or "日内" in str(x.message) for x in w)
    for c in _COLS:
        assert c in out.columns
        assert out[c][0] is None


def test_missing_panel_require_true_raises_with_build_hint():
    daily = _daily_date(["20240102"])
    with pytest.raises(ValueError, match="intraday-features build"):
        attach_intraday(daily, injected=pl.DataFrame(), require=True)


def test_out_meta_filled():
    daily = _daily_date(["20240102", "20240103"])
    panel = _panel(["20240102", "20240103"])
    meta: dict = {}
    attach_intraday(daily, injected=panel, out_meta=meta)
    assert "intraday_panel" in meta
    ip = meta["intraday_panel"]
    assert ip["version"] == "v1"
    assert ip["freq"] == "5min"
    assert ip["coverage_start"] is not None
    assert ip["coverage_end"] is not None


def test_leaf_health_zero_coverage_on_null_i_leaves():
    """缺面板 require=False → 全 null 列；leaf_health 对 i_* 覆盖率 0。"""
    from factorzen.discovery.leaf_health import leaf_holdout_coverage

    # 扩截面以满足 min_cross 语义；i_* 全 null → 覆盖率仍 0
    rows = []
    for d in [dt.date(2024, 1, d) for d in range(2, 12)]:
        for i in range(40):
            rows.append({
                "trade_date": d,
                "ts_code": f"{i:06d}.SZ",
                "close_adj": 10.0,
            })
    frame = pl.DataFrame(rows)
    out = attach_intraday(frame, injected=pl.DataFrame(), require=False)
    hstart = dt.date(2024, 1, 7)
    cov = leaf_holdout_coverage(
        out, list(INTRADAY_FEATURES), hstart,
        leaf_map={k: k for k in INTRADAY_FEATURES},
        min_cross=30,
    )
    assert all(v == 0.0 for v in cov.values())

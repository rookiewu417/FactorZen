"""
test_calendar.py：common/calendar.py 单元测试(本地缓存 mock,不调用 Tushare)
test_calendar_extra.py：core/calendar.py 离线补测(缓存/拉取/周月快照/交易时段)
test_data_ensure.py：data ensure 相关测试
test_smoke_data.py：tools/smoke_data.py 离线单测(mock build_raw_data_audit 与连通性)
test_build_panel_coverage_warn.py：build_panel 覆盖警告改口径(逐列非空率)
"""

from __future__ import annotations

import importlib.util
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
from factorzen.research.combination.models import (
    LOW_FEATURE_COVERAGE_WARN,
    _warn_incomplete,
    build_panel,
)


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

def test_is_trade_date_weekday(mock_calendar):
    assert mock_calendar.is_trade_date(date(2024, 1, 2)) is True  # 周二

def test_is_trade_date_weekend(mock_calendar):
    assert mock_calendar.is_trade_date(date(2024, 1, 6)) is False  # 周六

def test_is_trade_date_string_input(mock_calendar):
    assert mock_calendar.is_trade_date("20240102") is True

def test_prev_trade_date(mock_calendar):
    # 2024-01-03（周三）的前一个交易日是 2024-01-02（周二）
    result = mock_calendar.prev_trade_date(date(2024, 1, 3), n=1)
    assert result == date(2024, 1, 2)

def test_next_trade_date(mock_calendar):
    # 2024-01-05（周五）的下一个交易日是 2024-01-08（周一，下一周）
    result = mock_calendar.next_trade_date(date(2024, 1, 5), n=1)
    assert result == date(2024, 1, 8)

def test_get_trade_calendar_filter(mock_calendar):
    cal = mock_calendar.get_trade_calendar(start="20240103", end="20240105")
    assert cal.shape[0] == 3
    assert cal["cal_date"].min() == date(2024, 1, 3)
    assert cal["cal_date"].max() == date(2024, 1, 5)

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

def test_cache_valid_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cal_mod, "_CAL_FILE", tmp_path / "nope.parquet")
    assert cal_mod._is_cache_valid() is False

def test_cache_valid_fresh_file(tmp_path, monkeypatch):
    f = tmp_path / "trade_cal.parquet"
    pl.DataFrame({"cal_date": [date(2024, 1, 1)], "is_open": [1]}).write_parquet(f)
    monkeypatch.setattr(cal_mod, "_CAL_FILE", f)
    assert cal_mod._is_cache_valid() is True

def test_cache_valid_expired_file(tmp_path, monkeypatch):
    f = tmp_path / "trade_cal.parquet"
    pl.DataFrame({"cal_date": [date(2024, 1, 1)], "is_open": [1]}).write_parquet(f)
    old = time.time() - 30 * 86400
    import os

    os.utime(f, (old, old))
    monkeypatch.setattr(cal_mod, "_CAL_FILE", f)
    assert cal_mod._is_cache_valid() is False

# ══════════════════════════════════════════════════════════
# _fetch_from_tushare / _load_calendar
# ══════════════════════════════════════════════════════════

def test_fetch_from_tushare_writes_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cal_mod, "_CAL_FILE", tmp_path / "trade_cal.parquet")
    monkeypatch.setattr(cal_mod, "DATA_CACHE", tmp_path)
    monkeypatch.setattr(cal_mod, "ensure_token", lambda: "dummy")

    import tushare as ts

    fake_pro = MagicMock()
    fake_pro.trade_cal.return_value = pd.DataFrame(
        {"cal_date": ["20240102", "20240103"], "is_open": [1, 1]}
    )
    monkeypatch.setattr(ts, "set_token", lambda t: None)
    monkeypatch.setattr(ts, "pro_api", lambda: fake_pro)

    out = cal_mod._fetch_from_tushare()
    assert out["cal_date"].dtype == pl.Date
    assert out["is_open"].dtype == pl.Int8
    assert (tmp_path / "trade_cal.parquet").exists()

def test_fetch_from_tushare_empty_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(cal_mod, "_CAL_FILE", tmp_path / "trade_cal.parquet")
    monkeypatch.setattr(cal_mod, "DATA_CACHE", tmp_path)
    monkeypatch.setattr(cal_mod, "ensure_token", lambda: "dummy")

    import tushare as ts

    fake_pro = MagicMock()
    fake_pro.trade_cal.return_value = pd.DataFrame()
    monkeypatch.setattr(ts, "set_token", lambda t: None)
    monkeypatch.setattr(ts, "pro_api", lambda: fake_pro)

    with pytest.raises(RuntimeError, match="返回空数据"):
        cal_mod._fetch_from_tushare()

def test_load_calendar_cache_miss_fetches(monkeypatch):
    monkeypatch.setattr(cal_mod, "_is_cache_valid", lambda: False)
    sentinel = pl.DataFrame({"cal_date": [date(2024, 1, 2)], "is_open": [1]})
    monkeypatch.setattr(cal_mod, "_fetch_from_tushare", lambda: sentinel)
    assert cal_mod._load_calendar().equals(sentinel)

# ══════════════════════════════════════════════════════════
# is_trade_date / prev / next 边界
# ══════════════════════════════════════════════════════════

def test_is_trade_date_unknown_date_false(long_calendar):
    """日历中不存在的日期 → False。"""
    assert long_calendar.is_trade_date(date(2050, 1, 1)) is False

def test_prev_trade_date_insufficient_raises(long_calendar):
    """日历最早日之前没有足够交易日 → ValueError。"""
    with pytest.raises(ValueError, match="不足"):
        long_calendar.prev_trade_date(date(2024, 1, 2), n=99)

def test_next_trade_date_insufficient_raises(long_calendar):
    with pytest.raises(ValueError, match="不足"):
        long_calendar.next_trade_date(date(2024, 2, 28), n=99)

def test_next_trade_date_multi_step(long_calendar):
    """从周一往后第 5 个交易日应跨周。"""
    result = long_calendar.next_trade_date(date(2024, 1, 1), n=5)  # 周一
    assert result.weekday() < 5  # 落在工作日

def test_prev_trade_date_string_input(long_calendar):
    """字符串入参应被解析为日期。"""
    assert long_calendar.prev_trade_date("20240103", n=1) == date(2024, 1, 2)

def test_next_trade_date_string_input(long_calendar):
    assert long_calendar.next_trade_date("20240105", n=1) == date(2024, 1, 8)

# ══════════════════════════════════════════════════════════
# get_trade_dates / 时段 / 周月快照
# ══════════════════════════════════════════════════════════

def test_get_trade_dates_excludes_weekends(long_calendar):
    dates = long_calendar.get_trade_dates("20240101", "20240107")
    assert all(d.weekday() < 5 for d in dates)
    assert date(2024, 1, 6) not in dates  # 周六

def test_get_trading_sessions():
    sessions = cal_mod.get_trading_sessions()
    assert len(sessions) == 2
    assert sessions[0][0].hour == 9 and sessions[0][0].minute == 30

def test_weekly_snapshot_takes_last_trade_day_per_week(long_calendar):
    snaps = long_calendar.get_weekly_snapshot_dates("20240101", "20240131")
    # 每个快照日应是其 ISO 周内最后一个交易日（通常为周五）
    assert snaps == sorted(snaps)
    assert all(d.weekday() <= 4 for d in snaps)
    # 第一周 (2024-01-01~05) 的快照应为周五 2024-01-05
    assert date(2024, 1, 5) in snaps

def test_weekly_snapshot_empty_when_no_trades(monkeypatch):
    monkeypatch.setattr(cal_mod, "get_trade_dates", lambda s, e: [])
    assert cal_mod.get_weekly_snapshot_dates("20240101", "20240131") == []

def test_monthly_snapshot_takes_last_trade_day_per_month(long_calendar):
    snaps = cal_mod.get_monthly_snapshot_dates("20240101", "20240229")
    # 1 月最后交易日应为 2024-01-31（周三）
    assert date(2024, 1, 31) in snaps
    assert snaps == sorted(snaps)

def test_monthly_snapshot_empty_when_no_trades(monkeypatch):
    monkeypatch.setattr(cal_mod, "get_trade_dates", lambda s, e: [])
    assert cal_mod.get_monthly_snapshot_dates("20240101", "20240131") == []

# ==== 来自 test_data_ensure.py ====
def _daily_frame(trade_date: date | str) -> pl.DataFrame:
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
    _daily_frame(date(2024, 1, 2)).write_parquet(
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

def test_ensure_daily_fetches_only_missing_dates(tmp_path, monkeypatch):
    part_dir = tmp_path / "daily" / "year=2024" / "month=01"
    part_dir.mkdir(parents=True)
    _daily_frame(date(2024, 1, 2)).write_parquet(
        part_dir / "data.parquet",
    )
    monkeypatch.setattr(
        data_ensure,
        "get_trade_dates",
        lambda start, end: [date(2024, 1, 2), date(2024, 1, 3)],
    )
    pro = MagicMock()
    pro.daily.return_value = _pd_daily("20240103")
    monkeypatch.setattr(data_ensure, "init_tushare", lambda: pro)
    monkeypatch.setattr(data_ensure, "_retry", lambda func, **kwargs: func(**kwargs))

    result = data_ensure.ensure_daily("20240102", "20240103", base_dir=tmp_path)

    assert result.ok is True
    pro.daily.assert_called_once_with(trade_date="20240103")
    loaded = pl.scan_parquet(str(tmp_path / "daily" / "**/*.parquet")).collect()
    assert sorted(loaded["trade_date"].dt.strftime("%Y%m%d").unique().to_list()) == [
        "20240102",
        "20240103",
    ]

def test_ensure_daily_does_not_fetch_when_cache_is_complete(tmp_path, monkeypatch):
    part_dir = tmp_path / "daily" / "year=2024" / "month=01"
    part_dir.mkdir(parents=True)
    _daily_frame(date(2024, 1, 2)).write_parquet(
        part_dir / "data.parquet",
    )
    monkeypatch.setattr(data_ensure, "get_trade_dates", lambda start, end: [date(2024, 1, 2)])
    pro = MagicMock()
    monkeypatch.setattr(data_ensure, "init_tushare", lambda: pro)

    result = data_ensure.ensure_daily("20240102", "20240102", base_dir=tmp_path)

    assert result.ok is True
    pro.daily.assert_not_called()

def test_ensure_daily_run_uses_complete_local_cache_without_tushare(tmp_path, monkeypatch):
    for data_type in ("daily", "adj_factor", "daily_basic"):
        part_dir = tmp_path / data_type / "year=2024" / "month=01"
        part_dir.mkdir(parents=True)
        frame = _daily_frame(date(2024, 1, 2))
        if data_type == "adj_factor":
            frame = frame.select(["ts_code", "trade_date"]).with_columns(
                pl.lit(1.0).alias("adj_factor")
            )
        elif data_type == "daily_basic":
            frame = frame.select(["trade_date", "ts_code"]).with_columns(
                pl.lit(1_000_000.0).alias("total_mv")
            )
        frame.write_parquet(part_dir / "data.parquet")

    monkeypatch.setattr(data_ensure, "get_trade_dates", lambda start, end: [date(2024, 1, 2)])
    monkeypatch.setattr(data_ensure, "DATA_RAW", tmp_path)

    def _unexpected_api_init():
        raise AssertionError("complete local cache must not call Tushare")

    monkeypatch.setattr(data_ensure, "init_tushare", _unexpected_api_init)

    result = data_ensure.ensure_data_for_daily_run(
        required_data=["daily"],
        start="20240102",
        end="20240102",
        needs_size_neutralization=True,
        strict=True,
    )

    assert set(result) == {"daily", "adj_factor", "daily_basic"}
    assert all(audit.ok for audit in result.values())

def test_ensure_daily_repairs_duplicate_keys_without_fetching(tmp_path, monkeypatch):
    part_dir = tmp_path / "daily" / "year=2024" / "month=01"
    part_dir.mkdir(parents=True)
    duplicate = pl.concat(
        [
            _daily_frame(date(2024, 1, 2)),
            _daily_frame(date(2024, 1, 2)).with_columns(pl.lit(10.8).alias("close")),
        ]
    )
    duplicate.write_parquet(part_dir / "data.parquet")
    monkeypatch.setattr(data_ensure, "get_trade_dates", lambda start, end: [date(2024, 1, 2)])

    def _unexpected_api_init():
        raise AssertionError("duplicate repair must not call Tushare")

    monkeypatch.setattr(data_ensure, "init_tushare", _unexpected_api_init)

    result = data_ensure.ensure_daily("20240102", "20240102", base_dir=tmp_path)

    assert result.ok is True
    loaded = pl.read_parquet(part_dir / "data.parquet")
    assert loaded.height == 1
    assert loaded["close"][0] == 10.8

def test_ensure_daily_raises_when_fetch_does_not_fill_gap(tmp_path, monkeypatch):
    monkeypatch.setattr(data_ensure, "get_trade_dates", lambda start, end: [date(2024, 1, 2)])
    pro = MagicMock()
    pro.daily.return_value = pd.DataFrame()
    monkeypatch.setattr(data_ensure, "init_tushare", lambda: pro)
    monkeypatch.setattr(data_ensure, "_retry", lambda func, **kwargs: pd.DataFrame())

    with pytest.raises(data_ensure.DataEnsureError, match="daily still missing"):
        data_ensure.ensure_daily("20240102", "20240102", base_dir=tmp_path)

def test_ensure_daily_persists_successful_fetches_before_later_failure(tmp_path, monkeypatch):
    trade_dates = [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
    ]
    monkeypatch.setattr(data_ensure, "get_trade_dates", lambda start, end: trade_dates)
    monkeypatch.setattr(data_ensure, "FETCH_SAVE_BATCH_SIZE", 2)
    pro = MagicMock()
    monkeypatch.setattr(data_ensure, "init_tushare", lambda: pro)

    def fake_retry(_func, *, trade_date):
        if trade_date == "20240104":
            raise RuntimeError("network stopped")
        return _pd_daily(trade_date)

    monkeypatch.setattr(data_ensure, "_retry", fake_retry)

    with pytest.raises(RuntimeError, match="network stopped"):
        data_ensure.ensure_daily("20240102", "20240104", base_dir=tmp_path)

    after = data_ensure.audit_daily_like("daily", "20240102", "20240104", base_dir=tmp_path)
    assert after.present_dates == ["20240102", "20240103"]
    assert after.missing_dates == ["20240104"]

    retry_dates: list[str] = []

    def finish_retry(_func, *, trade_date):
        retry_dates.append(trade_date)
        return _pd_daily(trade_date)

    monkeypatch.setattr(data_ensure, "_retry", finish_retry)

    final = data_ensure.ensure_daily("20240102", "20240104", base_dir=tmp_path)

    assert retry_dates == ["20240104"]
    assert final.ok is True

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

def test_worst_status_error_dominates():
    assert smoke_data._worst_status(["ok", "warning", "error"]) == "error"

def test_worst_status_warning_over_ok():
    assert smoke_data._worst_status(["ok", "warning", "ok"]) == "warning"

def test_worst_status_all_ok():
    assert smoke_data._worst_status(["ok", "ok"]) == "ok"

def test_worst_status_empty_is_ok():
    assert smoke_data._worst_status([]) == "ok"

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

def test_summarize_all_ok_returns_0(capsys):
    code = smoke_data.summarize((True, "ok"), {"daily": _audit("ok")})
    assert code == 0
    assert "OK" in capsys.readouterr().out

def test_summarize_warning_returns_2():
    code = smoke_data.summarize(
        (True, "ok"), {"daily": _audit("warning", warnings=["缺 3 天"])}
    )
    assert code == 2

def test_summarize_error_returns_1():
    code = smoke_data.summarize(
        (True, "ok"), {"daily": _audit("error", errors=["分区为空"])}
    )
    assert code == 1

def test_summarize_connectivity_fail_is_error():
    code = smoke_data.summarize((False, "token 缺失"), {"daily": _audit("ok")})
    assert code == 1

def test_summarize_skipped_connectivity(capsys):
    code = smoke_data.summarize(None, {"daily": _audit("ok")})
    assert code == 0
    assert "跳过" in capsys.readouterr().out

# ── check_tushare_connectivity ──────────────────────────────

def _fake_pro():
    """init_tushare 桩：需有被 _retry 引用的 trade_cal 属性。"""
    return SimpleNamespace(trade_cal=lambda **kw: None)

def test_connectivity_success(monkeypatch):
    import factorzen.core.loader as loader_mod

    monkeypatch.setattr(loader_mod, "init_tushare", _fake_pro)

    class _DF:
        empty = False

        def __len__(self):
            return 5

    monkeypatch.setattr(loader_mod, "_retry", lambda fn, **kw: _DF())
    ok, msg = smoke_data.check_tushare_connectivity()
    assert ok and "正常" in msg

def test_connectivity_empty_result(monkeypatch):
    import factorzen.core.loader as loader_mod

    monkeypatch.setattr(loader_mod, "init_tushare", _fake_pro)

    class _Empty:
        empty = True

    monkeypatch.setattr(loader_mod, "_retry", lambda fn, **kw: _Empty())
    ok, _ = smoke_data.check_tushare_connectivity()
    assert not ok

def test_connectivity_exception(monkeypatch):
    import factorzen.core.loader as loader_mod

    def _boom():
        raise RuntimeError("no token")

    monkeypatch.setattr(loader_mod, "init_tushare", _boom)
    ok, msg = smoke_data.check_tushare_connectivity()
    assert not ok and "失败" in msg

# ── main / argparse ─────────────────────────────────────────

def test_main_skip_tushare_offline(monkeypatch):
    """--skip-tushare 不触发连通性检查，退出码由审计决定。"""
    monkeypatch.setattr(
        smoke_data, "build_raw_data_audit", lambda **kw: _audit("ok")
    )

    def _should_not_call():
        raise AssertionError("--skip-tushare 时不应检查连通性")

    monkeypatch.setattr(smoke_data, "check_tushare_connectivity", _should_not_call)
    code = smoke_data.main(["--skip-tushare", "--data-type", "daily"])
    assert code == 0

def test_main_json_output(monkeypatch, capsys):
    monkeypatch.setattr(smoke_data, "build_raw_data_audit", lambda **kw: _audit("ok"))
    monkeypatch.setattr(
        smoke_data, "check_tushare_connectivity", lambda: (True, "ok")
    )
    code = smoke_data.main(["--data-type", "daily", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    assert '"exit_code": 0' in out

def test_main_error_audit_exit_1(monkeypatch):
    monkeypatch.setattr(
        smoke_data, "build_raw_data_audit", lambda **kw: _audit("error", errors=["空"])
    )
    code = smoke_data.main(["--skip-tushare", "--data-type", "finance"])
    assert code == 1

def test_main_default_audits_all_three(monkeypatch):
    seen = []
    monkeypatch.setattr(
        smoke_data,
        "build_raw_data_audit",
        lambda **kw: seen.append(kw["data_type"]) or _audit("ok"),
    )
    smoke_data.main(["--skip-tushare"])
    assert set(seen) == {"daily", "daily_basic", "finance"}

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

def test_low_feature_coverage_constant():
    assert LOW_FEATURE_COVERAGE_WARN == 0.30

def test_warn_when_one_column_below_30pct():
    """一列 20% 覆盖 → warn，文案含该列名。"""
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

def test_no_warn_when_all_cols_above_30pct_even_if_row_complete_low():
    """全列 ≥30% 非空 → 不 warn，即便行齐全率 <70%（回归旧恒真行为）。"""
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


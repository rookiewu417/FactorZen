from __future__ import annotations

import json

import pytest

from tools import download_tushare_lake as dl


def test_to_df_preserves_numeric_columns_and_fills_short_rows():
    frame = dl._to_df(["ts_code", "close"], [["000001.SZ", 10], ["000002.SZ"]])

    assert frame.columns == ["ts_code", "close"]
    assert frame["ts_code"].to_list() == ["000001.SZ", "000002.SZ"]
    assert frame["close"].to_list() == [10, None]


def test_lake_ledger_and_atomic_parquet_are_resumable(tmp_path):
    lake = dl.Lake(tmp_path)
    lake.mark("minute", "000001.SZ")
    rows = lake.write_parquet(
        "minute/1min/000001.SZ.parquet",
        ["ts_code", "trade_time", "close"],
        [["000001.SZ", "2024-01-02 09:31:00", 10.0]],
    )

    reloaded = dl.Lake(tmp_path)
    assert rows == 1
    assert reloaded.done_set("minute") == {"000001.SZ"}
    assert (tmp_path / "minute/1min/000001.SZ.parquet").is_file()
    assert not list(tmp_path.rglob("*.tmp"))


def test_api_call_uses_tushare_wire_contract(monkeypatch):
    captured: dict = {}

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"code": 0, "data": {"fields": ["x"], "items": [[1]]}}

    def fake_post(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return Response()

    monkeypatch.setattr(dl.RL, "wait", lambda: None)
    monkeypatch.setattr(dl.requests, "post", fake_post)
    monkeypatch.setattr(dl, "TOKEN", "test-token")

    assert dl.api_call("trade_cal", {"exchange": "SSE"}) == (["x"], [[1]])
    assert captured["json"] == {
        "api_name": "trade_cal",
        "token": "test-token",
        "params": {"exchange": "SSE"},
        "fields": "",
    }


def test_daily_quota_response_fails_closed_without_retry(monkeypatch):
    class Response:
        status_code = 200
        text = json.dumps({"code": -2001})

        @staticmethod
        def json():
            return {"code": -2001, "msg": "今日调用已达上限，明日再试"}

    monkeypatch.setattr(dl.RL, "wait", lambda: None)
    monkeypatch.setattr(dl.requests, "post", lambda *args, **kwargs: Response())

    with pytest.raises(dl.DailyCap):
        dl.api_call("stk_mins", {}, max_tries=6)

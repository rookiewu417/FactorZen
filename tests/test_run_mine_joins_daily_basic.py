"""run_mine 须把 daily_basic join 进 daily，否则 BASIC_FEATURES 叶子是死叶子（F5）。

根因：run_mine 声明 required_data=["daily","daily_basic"] 却只 collect ctx.daily 传给
run_session，从不 join daily_basic。搜索空间里 total_mv/pb/pe_ttm 等 10 个基本面叶子的
候选 compile 时 ColumnNotFound 被静默跳过——估值/换手类因子永远挖不出（=R9 第一步）。
"""
from __future__ import annotations

from datetime import date

import polars as pl


def test_run_mine_joins_daily_basic_into_frame(monkeypatch):
    import factorzen.daily.data.context as ctx_mod
    import factorzen.pipelines.factor_mine as fm

    d = [date(2024, 1, 1), date(2024, 1, 2)]
    daily = pl.DataFrame({
        "trade_date": d * 2,
        "ts_code": ["A.SZ", "A.SZ", "B.SZ", "B.SZ"],
        "close": [10.0, 11.0, 20.0, 21.0], "close_adj": [10.0, 11.0, 20.0, 21.0],
        "open": [10.0, 11.0, 20.0, 21.0], "high": [10.0, 11.0, 20.0, 21.0],
        "low": [10.0, 11.0, 20.0, 21.0], "vol": [1e5, 1e5, 1e5, 1e5],
        "amount": [1e6, 1e6, 1e6, 1e6],
    })
    basic = pl.DataFrame({
        "trade_date": d * 2,
        "ts_code": ["A.SZ", "A.SZ", "B.SZ", "B.SZ"],
        "total_mv": [5e5, 5e5, 8e5, 8e5], "pb": [1.5, 1.5, 2.0, 2.0],
    })

    class _FakeCtx:
        def __init__(self, **kw):
            pass

        @property
        def daily(self):
            return daily.lazy()

        @property
        def daily_basic(self):
            return basic.lazy()

    monkeypatch.setattr(ctx_mod, "FactorDataContext", _FakeCtx)

    captured: dict = {}

    def _fake_run_session(frame, **kw):
        captured["frame"] = frame
        return {"candidates": [], "session_dir": "x", "n_trials": 0, "n_scored": 0}

    monkeypatch.setattr(fm, "run_session", _fake_run_session)

    fm.run_mine(start="20240101", end="20240102", n_trials=1)

    cols = set(captured["frame"].columns)
    assert "total_mv" in cols and "pb" in cols, (
        f"run_mine 传给 run_session 的帧应含 daily_basic 基本面列（否则 BASIC_FEATURES 死叶子），实得 {cols}"
    )


def test_prepare_mining_daily_default_warmup_covers_search_space(monkeypatch):
    """默认预热前缀 = search_space_max_lookback()（覆盖搜索空间最大回看），不再是会误拒
    长窗口/深嵌套因子的旧默认 60。FactorDataContext 收到的 lookback_days 即证据。"""
    import factorzen.daily.data.context as ctx_mod
    import factorzen.pipelines.factor_mine as fm
    from factorzen.discovery.search.random_search import search_space_max_lookback

    captured: dict = {}
    empty = pl.DataFrame({"trade_date": [], "ts_code": []})

    class _FakeCtx:
        def __init__(self, **kw):
            captured["lookback_days"] = kw.get("lookback_days")

        @property
        def daily(self):
            return empty.lazy()

        @property
        def daily_basic(self):
            return empty.lazy()

    monkeypatch.setattr(ctx_mod, "FactorDataContext", _FakeCtx)

    fm.prepare_mining_daily("20240101", "20240201")

    assert captured["lookback_days"] == search_space_max_lookback()

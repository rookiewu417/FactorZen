"""股东户数叶子：ann_date PIT、期际环比、双路径一致、叶子注册。

语义：t 日可用 = ann_date <= t 的最新一期（end_date 最大）；
holder_num_chg 在源数据整理阶段按 (ts_code, end_date 升序) 期际环比，
随本期 ann_date 生效（与 fina *_yoy 同理，非 ts_* 日频差分）。
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from factorzen.daily.data.pit import attach_holders


def _holder() -> pl.DataFrame:
    """两期公告：end 0331/ann 0420 holder=10000；end 0630/ann 0815 holder=8000。

    环比 chg = (8000-10000)/10000 = -0.2，随 0815 公告生效。
    """
    return pl.DataFrame({
        "ts_code": ["000001.SZ", "000001.SZ"],
        "end_date": ["20200331", "20200630"],
        "ann_date": ["20200420", "20200815"],
        "holder_num": [10000.0, 8000.0],
    })


def _daily(dates: list[str], codes: list[str] | None = None) -> pl.DataFrame:
    if codes is None:
        codes = ["000001.SZ"]
    rows = []
    for code in codes:
        for d in dates:
            rows.append({
                "trade_date": dt.datetime.strptime(d, "%Y%m%d").date(),
                "ts_code": code,
                "close": 10.0,
            })
    return pl.DataFrame(rows)


def test_holder_no_future_leak_before_announcement():
    """t 在首期公告前 → holder_num / holder_num_chg 必须 null。"""
    out = attach_holders(_daily(["20200410"]), holder_df=_holder())
    row = out.filter(pl.col("trade_date") == dt.date(2020, 4, 10)).row(0, named=True)
    assert row["holder_num"] is None, "首期公告前泄漏 → 未来函数!"
    assert row["holder_num_chg"] is None


def test_holder_pit_uses_latest_announced_period():
    """公告间持有上一期；新公告日后切换；期间无 ffill 伪造（PIT 自然前向）。"""
    out = attach_holders(
        _daily(["20200410", "20200501", "20200810", "20200820"]),
        holder_df=_holder(),
    )
    by = {r["trade_date"]: r for r in out.iter_rows(named=True)}
    assert by[dt.date(2020, 4, 10)]["holder_num"] is None
    assert by[dt.date(2020, 5, 1)]["holder_num"] == 10000.0
    # 第二期 0815 才公告 → 0810 仍用第一期
    assert by[dt.date(2020, 8, 10)]["holder_num"] == 10000.0
    assert by[dt.date(2020, 8, 20)]["holder_num"] == 8000.0


def test_holder_num_chg_period_over_period_and_pit():
    """环比在期际算好：第二期 chg=-0.2 仅 0815 起可见；第一期无上期 → chg null。"""
    out = attach_holders(
        _daily(["20200501", "20200810", "20200820"]),
        holder_df=_holder(),
    )
    by = {r["trade_date"]: r for r in out.iter_rows(named=True)}
    # 第一期已公告、尚无上期 → chg null
    assert by[dt.date(2020, 5, 1)]["holder_num_chg"] is None
    # 第二期公告前不可见 chg
    assert by[dt.date(2020, 8, 10)]["holder_num_chg"] is None
    # 第二期公告后 chg = (8000-10000)/10000 = -0.2
    assert abs(by[dt.date(2020, 8, 20)]["holder_num_chg"] - (-0.2)) < 1e-12


def test_holder_missing_stock_null():
    """无股东户数数据的股票 → null（诚实缺测，不填 0）。"""
    out = attach_holders(
        _daily(["20200501"], codes=["000002.SZ"]),
        holder_df=_holder(),
    )
    assert out["holder_num"][0] is None
    assert out["holder_num_chg"][0] is None


def test_holder_empty_source_null_cols():
    out = attach_holders(_daily(["20200501"]), holder_df=pl.DataFrame())
    assert "holder_num" in out.columns and "holder_num_chg" in out.columns
    assert out["holder_num"][0] is None


def test_holder_mining_and_materialize_paths_value_identical(monkeypatch):
    """prepare_mining_daily 与 ExpressionFactor.compute 共用 attach_holders → 逐值一致。"""
    import factorzen.daily.data.context as ctx_mod
    import factorzen.daily.data.pit as pit_mod
    import factorzen.pipelines.factor_mine as fm
    from factorzen.discovery.factor import ExpressionFactor

    dates = [dt.date(2020, 5, 1), dt.date(2020, 8, 20)]
    daily = pl.DataFrame({
        "trade_date": dates,
        "ts_code": ["000001.SZ"] * 2,
        "close": [10.0, 11.0], "close_adj": [10.0, 11.0],
        "open": [10.0] * 2, "open_adj": [10.0] * 2,
        "high": [11.0] * 2, "high_adj": [11.0] * 2,
        "low": [9.0] * 2, "low_adj": [9.0] * 2,
        "pre_close": [10.0, 10.0],
        "vol": [1e5] * 2, "amount": [1e5] * 2,
    })
    basic = pl.DataFrame({
        "trade_date": dates,
        "ts_code": ["000001.SZ"] * 2,
        "circ_mv": [1e6, 1e6],
        "total_mv": [2e6, 2e6],
    })
    holder = _holder()

    class _FakeCtx:
        def __init__(self, **kw):
            self.start = kw.get("start", "20200501")
            self.end = kw.get("end", "20200820")

        @property
        def daily(self):
            return daily.lazy()

        @property
        def daily_basic(self):
            return basic.lazy()

    monkeypatch.setattr(ctx_mod, "FactorDataContext", _FakeCtx)

    real_attach = pit_mod.attach_holders

    def _attach_with_holder(d, holder_df=None):
        return real_attach(d, holder_df=holder if holder_df is None else holder_df)

    monkeypatch.setattr(pit_mod, "attach_holders", _attach_with_holder)
    monkeypatch.setattr(pit_mod, "attach_fundamentals", lambda d, fina_df=None: d)

    import factorzen.daily.data.flows as flows_mod
    monkeypatch.setattr(flows_mod, "attach_flows", lambda d, **kw: d)

    mined = fm.prepare_mining_daily("20200501", "20200820")
    mat_frame = _attach_with_holder(daily)

    for col in ("holder_num", "holder_num_chg"):
        a = mined.sort(["ts_code", "trade_date"])[col].to_list()
        b = mat_frame.sort(["ts_code", "trade_date"])[col].to_list()
        assert a == b, f"双路径 {col} 不一致: mine={a} mat={b}"

    class _Ctx:
        start = "20200501"
        end = "20200820"

        @property
        def daily(self):
            return daily.lazy()

        @property
        def daily_basic(self):
            return basic.lazy()

    # ExpressionFactor.compute 也走 attach_holders（已 patch）
    fac = ExpressionFactor("rank(holder_num)", mined_name="h_num")
    out = fac.compute(_Ctx())
    assert "factor_value" in out.columns


def test_holder_leaves_registered_and_parse():
    from factorzen.discovery.expression import feature_names, parse_expr
    from factorzen.discovery.operators import HOLDER_FEATURES, LEAF_FEATURES

    expected = {"holder_num", "holder_num_chg"}
    assert expected <= HOLDER_FEATURES
    for leaf in expected:
        assert leaf in LEAF_FEATURES
        assert leaf in feature_names(parse_expr(f"rank({leaf})"))


def test_prompt_mentions_holder_family():
    from factorzen.agents.roles.hypothesis import signal_families
    from factorzen.llm.generation import build_agent_messages

    fam = signal_families("ashare")
    assert "股东" in fam or "holder" in fam.lower()

    sys = build_agent_messages(["ts_mean"], ["close", "holder_num"], market="ashare")[0]["content"]
    assert "股东" in sys or "holder" in sys.lower()
    assert "ann_date" in sys or "公告" in sys or "PIT" in sys


def test_holder_leaves_pass_through_leaf_health():
    """无数据叶在 holdout 全 null → 被 leaf_health 摘除。"""
    from factorzen.discovery.leaf_health import filter_leaves_by_holdout_coverage
    from factorzen.discovery.operators import HOLDER_FEATURES

    days = [dt.date(2024, 1, d) for d in range(2, 22)]
    hstart = days[10]
    codes = [f"{i:06d}.SZ" for i in range(40)]
    rows = []
    for day in days:
        for c in codes:
            rows.append({
                "trade_date": day,
                "ts_code": c,
                "close_adj": 10.0,
                "holder_num": None if day >= hstart else 10000.0,
                "holder_num_chg": None if day >= hstart else -0.05,
            })
    df = pl.DataFrame(rows)
    leaves = ["close", "holder_num", "holder_num_chg"]
    leaf_map = {"close": "close_adj", "holder_num": "holder_num",
                "holder_num_chg": "holder_num_chg"}
    kept, excluded = filter_leaves_by_holdout_coverage(
        df, leaves, hstart, leaf_map=leaf_map, min_coverage=0.5, min_cross=30,
    )
    assert "holder_num" in excluded
    assert "holder_num_chg" in excluded
    assert "close" in kept
    assert HOLDER_FEATURES

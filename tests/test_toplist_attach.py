"""龙虎榜叶子：lag(1)、未上榜 fill 0、同日聚合、单位换算、双路径、leaf_health。

披露时点：t 日龙虎榜 t 日盘后（晚间）披露 → 保守 lag(1)。
未上榜 = 真实零事件（全市场事件筛选）→ join 后 fill 0（与两融非标的=null 相反）。
单位：top_list net_amount=万元、amount=千元 → 比前统一到元。
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from factorzen.daily.data.flows import attach_flows


def _daily(dates: list[str], codes: list[str] | None = None, *,
           amount: float = 1e5) -> pl.DataFrame:
    """amount 默认 1e5 千元 = 1e8 元。"""
    if codes is None:
        codes = ["000001.SZ"]
    rows = []
    for code in codes:
        for d in dates:
            rows.append({
                "trade_date": dt.datetime.strptime(d, "%Y%m%d").date(),
                "ts_code": code,
                "close": 10.0,
                "close_adj": 10.0,
                "open": 10.0, "high": 10.0, "low": 10.0,
                "vol": 1e5,
                "amount": amount,
                "circ_mv": 1e6,
            })
    return pl.DataFrame(rows)


def _top(rows: list[dict]) -> pl.DataFrame:
    """rows: ts_code, trade_date YYYYMMDD, net_amount(万元), amount(千元), reason?"""
    return pl.DataFrame({
        "ts_code": [r["ts_code"] for r in rows],
        "trade_date": [dt.datetime.strptime(r["trade_date"], "%Y%m%d").date() for r in rows],
        "net_amount": [r["net_amount"] for r in rows],
        "amount": [r["amount"] for r in rows],
        "reason": [r.get("reason", "涨幅偏离") for r in rows],
    })


def _inj(top: pl.DataFrame) -> dict:
    return {
        "moneyflow": pl.DataFrame(),
        "hk_hold": pl.DataFrame(),
        "margin_detail": pl.DataFrame(),
        "top_list": top,
    }


# ── A. lag / fill0 / 聚合 / 单位 ──────────────────────────────────────────────


def test_toplist_lag1_and_not_listed_fill_zero():
    """t 日拿到 t-1 上榜信息；未上榜日/股 fill 0（非 null）。"""
    top = _top([
        # 仅 01-02 上榜：net_amount=1000 万元, amount=1e5 千元
        # 比 = (1000*1e4)/(1e5*1e3) = 1e7/1e8 = 0.1
        {"ts_code": "000001.SZ", "trade_date": "20240102",
         "net_amount": 1000.0, "amount": 1e5},
    ])
    out = attach_flows(
        _daily(["20240102", "20240103", "20240104"], amount=1e5),
        injected=_inj(top),
    )
    by = {r["trade_date"]: r for r in out.iter_rows(named=True)}
    # 01-02：lag 后无 t-1 → 0（fill 0 后覆盖=100%）
    assert by[dt.date(2024, 1, 2)]["top_list_flag"] == 0.0
    assert by[dt.date(2024, 1, 2)]["top_list_net_buy"] == 0.0
    # 01-03：拿到 01-02 上榜
    assert by[dt.date(2024, 1, 3)]["top_list_flag"] == 1.0
    assert abs(by[dt.date(2024, 1, 3)]["top_list_net_buy"] - 0.1) < 1e-12
    # 01-04：昨日未上榜 → 0
    assert by[dt.date(2024, 1, 4)]["top_list_flag"] == 0.0
    assert by[dt.date(2024, 1, 4)]["top_list_net_buy"] == 0.0


def test_not_listed_stock_fill_zero_not_null():
    """从未上榜的股票：全 0，不是 null（与两融非标的=null 相反）。"""
    top = _top([
        {"ts_code": "000001.SZ", "trade_date": "20240102",
         "net_amount": 100.0, "amount": 1e4},
        {"ts_code": "000001.SZ", "trade_date": "20240103",
         "net_amount": 100.0, "amount": 1e4},
    ])
    out = attach_flows(
        _daily(["20240102", "20240103", "20240104"], codes=["000002.SZ"]),
        injected=_inj(top),
    )
    assert out["top_list_flag"].null_count() == 0
    assert out["top_list_net_buy"].null_count() == 0
    assert out["top_list_flag"].to_list() == [0.0, 0.0, 0.0]
    assert out["top_list_net_buy"].to_list() == [0.0, 0.0, 0.0]


def test_toplist_same_day_multi_reason_sum_net_amount():
    """同日多条上榜原因：net_amount 先 sum 再算比；amount 取 first（同股同日相同）。"""
    top = _top([
        {"ts_code": "000001.SZ", "trade_date": "20240102",
         "net_amount": 500.0, "amount": 1e5, "reason": "涨幅偏离"},
        {"ts_code": "000001.SZ", "trade_date": "20240102",
         "net_amount": 500.0, "amount": 1e5, "reason": "换手率"},
    ])
    out = attach_flows(
        _daily(["20240102", "20240103"], amount=1e5),
        injected=_inj(top),
    )
    row = out.filter(pl.col("trade_date") == dt.date(2024, 1, 3)).row(0, named=True)
    # sum net=1000 万元 → 1e7 元；amount 1e5 千元 → 1e8 元；比=0.1
    assert abs(row["top_list_net_buy"] - 0.1) < 1e-12
    assert row["top_list_flag"] == 1.0


def test_toplist_unit_scale_net_wan_amount_qian():
    """单位钉死：net_amount 万元×1e4、amount 千元×1e3。

    net=2000 万元=2e7 元；amount=5e4 千元=5e7 元 → 比=0.4。
    """
    top = _top([
        {"ts_code": "000001.SZ", "trade_date": "20240102",
         "net_amount": 2000.0, "amount": 5e4},
    ])
    out = attach_flows(
        _daily(["20240102", "20240103"]),
        injected=_inj(top),
    )
    row = out.filter(pl.col("trade_date") == dt.date(2024, 1, 3)).row(0, named=True)
    assert abs(row["top_list_net_buy"] - 0.4) < 1e-12


# ── B. 双路径 ────────────────────────────────────────────────────────────────


def test_toplist_mining_and_materialize_paths_value_identical(monkeypatch):
    import factorzen.daily.data.context as ctx_mod
    import factorzen.daily.data.flows as flows_mod
    import factorzen.pipelines.factor_mine as fm
    from factorzen.discovery.factor import ExpressionFactor

    dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)]
    daily = pl.DataFrame({
        "trade_date": dates,
        "ts_code": ["000001.SZ"] * 3,
        "close": [10.0, 11.0, 12.0], "close_adj": [10.0, 11.0, 12.0],
        "open": [10.0] * 3, "open_adj": [10.0] * 3,
        "high": [11.0] * 3, "high_adj": [11.0] * 3,
        "low": [9.0] * 3, "low_adj": [9.0] * 3,
        "pre_close": [10.0, 10.0, 11.0],
        "vol": [1e5] * 3, "amount": [1e5] * 3,
    })
    basic = pl.DataFrame({
        "trade_date": dates,
        "ts_code": ["000001.SZ"] * 3,
        "circ_mv": [1e6] * 3,
        "total_mv": [2e6] * 3,
    })
    top = _top([
        {"ts_code": "000001.SZ", "trade_date": "20240102",
         "net_amount": 1000.0, "amount": 1e5},
        {"ts_code": "000001.SZ", "trade_date": "20240103",
         "net_amount": 500.0, "amount": 1e5},
    ])

    class _FakeCtx:
        def __init__(self, **kw):
            self.start = kw.get("start", "20240102")
            self.end = kw.get("end", "20240104")

        @property
        def daily(self):
            return daily.lazy()

        @property
        def daily_basic(self):
            return basic.lazy()

    monkeypatch.setattr(ctx_mod, "FactorDataContext", _FakeCtx)

    real_attach = flows_mod.attach_flows

    def _attach_with_top(d, *, injected=None):
        inj = dict(injected or {})
        inj.setdefault("moneyflow", pl.DataFrame())
        inj.setdefault("hk_hold", pl.DataFrame())
        inj.setdefault("margin_detail", pl.DataFrame())
        inj.setdefault("top_list", top)
        return real_attach(d, injected=inj)

    monkeypatch.setattr(flows_mod, "attach_flows", _attach_with_top)

    import factorzen.daily.data.pit as pit_mod
    monkeypatch.setattr(pit_mod, "attach_fundamentals", lambda d, fina_df=None: d)
    monkeypatch.setattr(pit_mod, "attach_holders", lambda d, holder_df=None: d)

    mined = fm.prepare_mining_daily("20240102", "20240104")
    mat_frame = _attach_with_top(daily)

    for col in ("top_list_net_buy", "top_list_flag"):
        a = mined.sort(["ts_code", "trade_date"])[col].to_list()
        b = mat_frame.sort(["ts_code", "trade_date"])[col].to_list()
        assert a == b, f"双路径 {col} 不一致: mine={a} mat={b}"

    class _Ctx:
        start = "20240102"
        end = "20240104"

        @property
        def daily(self):
            return daily.lazy()

        @property
        def daily_basic(self):
            return basic.lazy()

    fac = ExpressionFactor("rank(top_list_flag)", mined_name="tl_flag")
    out = fac.compute(_Ctx())
    assert "factor_value" in out.columns


# ── C. 注册 / prompt / leaf_health ────────────────────────────────────────────


def test_toplist_leaves_registered_and_parse():
    from factorzen.discovery.expression import feature_names, parse_expr
    from factorzen.discovery.operators import FLOW_FEATURES, LEAF_FEATURES, TOPLIST_FEATURES

    expected = {"top_list_net_buy", "top_list_flag"}
    assert expected <= TOPLIST_FEATURES
    assert expected <= FLOW_FEATURES
    for leaf in expected:
        assert leaf in LEAF_FEATURES
        feats = feature_names(parse_expr(f"rank({leaf})"))
        assert leaf in feats
        assert feats & FLOW_FEATURES


def test_prompt_mentions_toplist_family():
    from factorzen.agents.roles.hypothesis import signal_families
    from factorzen.llm.generation import build_agent_messages

    fam = signal_families("ashare")
    assert "龙虎" in fam or "top_list" in fam.lower()

    sys = build_agent_messages(
        ["ts_mean"], ["close", "top_list_net_buy"], market="ashare"
    )[0]["content"]
    assert "龙虎" in sys or "top_list" in sys.lower()
    assert "lag" in sys.lower() or "盘后" in sys or "T+1" in sys or "t+1" in sys.lower()


def test_toplist_fill0_leaves_pass_leaf_health_full_coverage():
    """fill 0 后龙虎榜叶子 holdout 覆盖=100%，经 leaf_health 检查保留。"""
    from factorzen.discovery.leaf_health import filter_leaves_by_holdout_coverage
    from factorzen.discovery.operators import TOPLIST_FEATURES

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
                # fill 0 语义：全有值
                "top_list_flag": 0.0,
                "top_list_net_buy": 0.0,
            })
    df = pl.DataFrame(rows)
    leaves = ["close", "top_list_flag", "top_list_net_buy"]
    leaf_map = {
        "close": "close_adj",
        "top_list_flag": "top_list_flag",
        "top_list_net_buy": "top_list_net_buy",
    }
    kept, excluded = filter_leaves_by_holdout_coverage(
        df, leaves, hstart, leaf_map=leaf_map, min_coverage=0.5, min_cross=30,
    )
    assert "top_list_flag" in kept
    assert "top_list_net_buy" in kept
    assert "top_list_flag" not in excluded
    assert TOPLIST_FEATURES

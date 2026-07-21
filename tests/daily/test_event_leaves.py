"""业绩预告/快报事件叶：PIT、20 日窗、编码、装配、leaf_health。

对照 top_list 事件 fill-0 范式；PIT：ann_date 盘后可得 → t+1 交易日起生效。
"""
from __future__ import annotations

import datetime as dt

import polars as pl

# ── helpers ──────────────────────────────────────────────────────────────────


def _daily(
    dates: list[str],
    codes: list[str] | None = None,
) -> pl.DataFrame:
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
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "vol": 1e5,
                "amount": 1e5,
            })
    return pl.DataFrame(rows)


def _forecast(rows: list[dict]) -> pl.DataFrame:
    """rows keys: ts_code, ann_date(YYYYMMDD), type, p_change_min, p_change_max, end_date?"""
    return pl.DataFrame({
        "ts_code": [r["ts_code"] for r in rows],
        "ann_date": [
            dt.datetime.strptime(r["ann_date"], "%Y%m%d").date() for r in rows
        ],
        "end_date": [
            dt.datetime.strptime(r.get("end_date", "20191231"), "%Y%m%d").date()
            for r in rows
        ],
        "type": [r.get("type", "预增") for r in rows],
        "p_change_min": [r.get("p_change_min") for r in rows],
        "p_change_max": [r.get("p_change_max") for r in rows],
    })


def _express(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame({
        "ts_code": [r["ts_code"] for r in rows],
        "ann_date": [
            dt.datetime.strptime(r["ann_date"], "%Y%m%d").date() for r in rows
        ],
        "end_date": [
            dt.datetime.strptime(r.get("end_date", "20191231"), "%Y%m%d").date()
            for r in rows
        ],
        "yoy_net_profit": [r.get("yoy_net_profit") for r in rows],
        "n_income": [r.get("n_income") for r in rows],
    })


# 交易日日历：周一至周五 2024-01（跳过周末；1/1 元旦当非交易日测顺延）
# 2024-01-01 Mon 元旦休市；02 Tue 开市；… 05 Fri；06-07 周末；08 Mon …
_JAN_TD = [
    "20240102", "20240103", "20240104", "20240105",
    "20240108", "20240109", "20240110", "20240111", "20240112",
    "20240115", "20240116", "20240117", "20240118", "20240119",
    "20240122", "20240123", "20240124", "20240125", "20240126",
    "20240129", "20240130", "20240131",
]


# ── 编码 golden ──────────────────────────────────────────────────────────────


def test_fc_type_score_encoding_golden():
    """各 type 字符串 → 分值硬编码（来自真实 data distinct type）。"""
    from factorzen.daily.data.events import FC_TYPE_SCORE

    expected = {
        "预增": 2.0,
        "扭亏": 2.0,
        "略增": 1.0,
        "续盈": 1.0,
        "略减": -1.0,
        "续亏": -1.0,
        "预减": -2.0,
        "首亏": -2.0,
        "不确定": 0.0,
        "其他": 0.0,
    }
    for t, s in expected.items():
        assert FC_TYPE_SCORE[t] == s, f"{t}: got {FC_TYPE_SCORE.get(t)} want {s}"
    # 未知 type → 0
    from factorzen.daily.data.events import encode_fc_type

    assert encode_fc_type("未知类型") == 0.0
    assert encode_fc_type(None) == 0.0


# ── PIT ──────────────────────────────────────────────────────────────────────


def test_pit_ann_date_not_visible_same_day():
    """公告日 D 当日因子 = 0；D+1 交易日起 = 编码值。"""
    from factorzen.daily.data.events import attach_forecast

    # 公告 01-03（交易日），编码 预增=+2
    fc = _forecast([{
        "ts_code": "000001.SZ",
        "ann_date": "20240103",
        "type": "预增",
        "p_change_min": 50.0,
        "p_change_max": 100.0,
    }])
    daily = _daily(["20240102", "20240103", "20240104", "20240105"])
    out = attach_forecast(daily, forecast_df=fc)
    by = {r["trade_date"]: r for r in out.iter_rows(named=True)}

    assert by[dt.date(2024, 1, 2)]["fc_type_score"] == 0.0
    assert by[dt.date(2024, 1, 2)]["fc_flag"] == 0.0
    # 公告日当日不可见
    assert by[dt.date(2024, 1, 3)]["fc_type_score"] == 0.0
    assert by[dt.date(2024, 1, 3)]["fc_flag"] == 0.0
    assert by[dt.date(2024, 1, 3)]["fc_surprise"] == 0.0
    # t+1 生效
    assert by[dt.date(2024, 1, 4)]["fc_type_score"] == 2.0
    assert by[dt.date(2024, 1, 4)]["fc_flag"] == 1.0
    assert abs(by[dt.date(2024, 1, 4)]["fc_surprise"] - 0.75) < 1e-12  # (50+100)/2/100


def test_pit_non_trading_ann_effective_next_td():
    """非交易日公告：生效 = 严格下一交易日（周末披露 → 周一可见，不再多 lag）。

    语义：ann_date 当日盘后可得；非交易日无「当日盘」，披露已发生，
    下一交易日开盘即可用 → effective = first trade_date > ann_date。
    """
    from factorzen.daily.data.events import attach_forecast

    # 2024-01-06 周六公告，下一交易日 01-08 周一生效
    fc = _forecast([{
        "ts_code": "000001.SZ",
        "ann_date": "20240106",
        "type": "首亏",
        "p_change_min": -80.0,
        "p_change_max": -50.0,
    }])
    daily = _daily(["20240105", "20240108", "20240109"])
    out = attach_forecast(daily, forecast_df=fc)
    by = {r["trade_date"]: r for r in out.iter_rows(named=True)}

    assert by[dt.date(2024, 1, 5)]["fc_type_score"] == 0.0
    assert by[dt.date(2024, 1, 5)]["fc_flag"] == 0.0
    # 周一开盘可见
    assert by[dt.date(2024, 1, 8)]["fc_type_score"] == -2.0
    assert by[dt.date(2024, 1, 8)]["fc_flag"] == 1.0
    assert by[dt.date(2024, 1, 9)]["fc_type_score"] == -2.0


# ── 窗口 ─────────────────────────────────────────────────────────────────────


def test_window_20_trading_days_then_zero():
    """第 20 交易日仍有效、第 21 日归 0。"""
    from factorzen.daily.data.events import attach_forecast

    # 公告 01-02 → 生效 01-03；窗 = 01-03 起 20 个交易日
    fc = _forecast([{
        "ts_code": "000001.SZ",
        "ann_date": "20240102",
        "type": "略增",
        "p_change_min": 10.0,
        "p_change_max": 20.0,
    }])
    # 需要 ≥ 22 个交易日：生效日 + 20 窗 + 1 归零日
    dates = _JAN_TD  # 22 个交易日，01-02 是第一个
    daily = _daily(dates)
    out = attach_forecast(daily, forecast_df=fc)
    by = {r["trade_date"]: r for r in out.iter_rows(named=True)}

    # 01-02 公告日不可见
    assert by[dt.date(2024, 1, 2)]["fc_flag"] == 0.0
    # 生效日起 20 日
    # dates[0]=01-02 ann, dates[1]=01-03 effective (= index 1), window indices 1..20
    window_dates = [dt.datetime.strptime(d, "%Y%m%d").date() for d in dates[1:21]]
    assert len(window_dates) == 20
    for d in window_dates:
        assert by[d]["fc_flag"] == 1.0, f"{d} should be in window"
        assert by[d]["fc_type_score"] == 1.0
    # 第 21 交易日（dates[21]）归 0
    day21 = dt.datetime.strptime(dates[21], "%Y%m%d").date()
    assert by[day21]["fc_flag"] == 0.0
    assert by[day21]["fc_type_score"] == 0.0
    assert by[day21]["fc_surprise"] == 0.0


def test_overlapping_windows_take_latest_ann():
    """重叠窗取最新公告（as-of last）。"""
    from factorzen.daily.data.events import attach_forecast

    fc = _forecast([
        {
            "ts_code": "000001.SZ",
            "ann_date": "20240102",
            "type": "预增",  # +2
            "p_change_min": 50.0,
            "p_change_max": 50.0,
        },
        {
            "ts_code": "000001.SZ",
            "ann_date": "20240108",
            "type": "预减",  # -2
            "p_change_min": -30.0,
            "p_change_max": -30.0,
        },
    ])
    daily = _daily([
        "20240102", "20240103", "20240104", "20240105",
        "20240108", "20240109", "20240110",
    ])
    out = attach_forecast(daily, forecast_df=fc)
    by = {r["trade_date"]: r for r in out.iter_rows(named=True)}

    # 01-03..01-08: 第一公告生效（01-08 是第二公告日，尚不可见）
    assert by[dt.date(2024, 1, 3)]["fc_type_score"] == 2.0
    assert by[dt.date(2024, 1, 8)]["fc_type_score"] == 2.0
    # 01-09 起第二公告生效覆盖
    assert by[dt.date(2024, 1, 9)]["fc_type_score"] == -2.0
    assert by[dt.date(2024, 1, 9)]["fc_flag"] == 1.0
    assert abs(by[dt.date(2024, 1, 9)]["fc_surprise"] - (-0.30)) < 1e-12


# ── surprise 边角 ────────────────────────────────────────────────────────────


def test_fc_surprise_one_side_and_both_null():
    """仅一侧非空用该侧；两侧全空 → 0。"""
    from factorzen.daily.data.events import attach_forecast

    fc = _forecast([
        {
            "ts_code": "000001.SZ",
            "ann_date": "20240102",
            "type": "预增",
            "p_change_min": 40.0,
            "p_change_max": None,
        },
        {
            "ts_code": "000002.SZ",
            "ann_date": "20240102",
            "type": "不确定",
            "p_change_min": None,
            "p_change_max": None,
        },
        {
            "ts_code": "000003.SZ",
            "ann_date": "20240102",
            "type": "略减",
            "p_change_min": None,
            "p_change_max": -15.0,
        },
    ])
    daily = _daily(
        ["20240102", "20240103"],
        codes=["000001.SZ", "000002.SZ", "000003.SZ"],
    )
    out = attach_forecast(daily, forecast_df=fc)
    d3 = out.filter(pl.col("trade_date") == dt.date(2024, 1, 3))
    by = {r["ts_code"]: r for r in d3.iter_rows(named=True)}

    assert abs(by["000001.SZ"]["fc_surprise"] - 0.40) < 1e-12
    assert by["000002.SZ"]["fc_surprise"] == 0.0
    assert abs(by["000003.SZ"]["fc_surprise"] - (-0.15)) < 1e-12


# ── express ──────────────────────────────────────────────────────────────────


def test_express_yoy_pit_and_window():
    """express_yoy：PIT + 20 日窗 + /100。"""
    from factorzen.daily.data.events import attach_express

    ex = _express([{
        "ts_code": "000001.SZ",
        "ann_date": "20240103",
        "yoy_net_profit": 25.0,  # → 0.25 after /100
    }])
    daily = _daily(["20240102", "20240103", "20240104", "20240105"])
    out = attach_express(daily, express_df=ex)
    by = {r["trade_date"]: r for r in out.iter_rows(named=True)}

    assert by[dt.date(2024, 1, 3)]["express_yoy"] == 0.0  # 公告日不可见
    assert abs(by[dt.date(2024, 1, 4)]["express_yoy"] - 0.25) < 1e-12
    assert by[dt.date(2024, 1, 2)]["express_yoy"] == 0.0


def test_empty_source_all_null_not_zero():
    """空源 → 全 null（覆盖审计诚实），非全 0。"""
    from factorzen.daily.data.events import attach_express, attach_forecast

    daily = _daily(["20240102", "20240103"])
    out_fc = attach_forecast(daily, forecast_df=pl.DataFrame())
    out_ex = attach_express(daily, express_df=pl.DataFrame())
    for col in ("fc_type_score", "fc_surprise", "fc_flag"):
        assert out_fc[col].null_count() == out_fc.height
    assert out_ex["express_yoy"].null_count() == out_ex.height


# ── 注册 / parse / 双路径 ────────────────────────────────────────────────────


def test_event_leaves_registered_and_parse():
    from factorzen.discovery.expression import feature_names, parse_expr
    from factorzen.discovery.operators import (
        EVENT_FILL0_FEATURES,
        EXPRESS_FEATURES,
        FORECAST_FEATURES,
        LEAF_FEATURES,
    )

    fc = {"fc_type_score", "fc_surprise", "fc_flag"}
    ex = {"express_yoy"}
    assert fc <= FORECAST_FEATURES
    assert ex <= EXPRESS_FEATURES
    assert fc | ex <= EVENT_FILL0_FEATURES
    for leaf in fc | ex:
        assert leaf in LEAF_FEATURES
        feats = feature_names(parse_expr(f"rank({leaf})"))
        assert leaf in feats


def test_prompt_mentions_forecast_express_family():
    from factorzen.agents.roles.hypothesis import signal_families
    from factorzen.llm.generation import build_agent_messages

    fam = signal_families("ashare")
    assert "预告" in fam or "forecast" in fam.lower() or "fc_" in fam
    assert "快报" in fam or "express" in fam.lower()

    sys = build_agent_messages(
        ["ts_mean"], ["close", "fc_type_score", "express_yoy"], market="ashare"
    )[0]["content"]
    assert "fc_type_score" in sys or "预告" in sys
    assert "express" in sys.lower() or "快报" in sys
    assert "t+1" in sys.lower() or "T+1" in sys or "盘后" in sys or "公告" in sys


def test_leaf_health_event_fill0_source_audit_not_value_dist():
    """事件 fill-0：空源全 null → 被摘；有源 fill-0 → 不按稀疏非零误杀。"""
    from factorzen.discovery.leaf_health import (
        filter_leaves_by_holdout_coverage,
        leaf_holdout_coverage,
    )
    from factorzen.discovery.operators import EVENT_FILL0_FEATURES

    days = [dt.date(2024, 1, d) for d in range(2, 22)]
    hstart = days[10]
    codes = [f"{i:06d}.SZ" for i in range(40)]

    # 有源 fill-0：绝大多数为 0，仅极少非零——值分布 sparse，但源覆盖完整
    rows = []
    for day in days:
        for i, c in enumerate(codes):
            flag = 1.0 if (i == 0 and day == days[11]) else 0.0
            rows.append({
                "trade_date": day,
                "ts_code": c,
                "close_adj": 10.0,
                "fc_flag": flag,
                "fc_type_score": 2.0 if flag else 0.0,
                "fc_surprise": 0.5 if flag else 0.0,
                "express_yoy": 0.1 if flag else 0.0,
            })
    df = pl.DataFrame(rows)
    leaves = ["close", "fc_flag", "fc_type_score", "fc_surprise", "express_yoy"]
    leaf_map = {L: L if L != "close" else "close_adj" for L in leaves}
    leaf_map["close"] = "close_adj"

    kept, excluded = filter_leaves_by_holdout_coverage(
        df, leaves, hstart, leaf_map=leaf_map, min_coverage=0.5, min_cross=30,
    )
    for leaf in ("fc_flag", "fc_type_score", "fc_surprise", "express_yoy"):
        assert leaf in kept, f"{leaf} 不应因 fill-0 稀疏非零被摘"
        assert leaf not in excluded
    assert EVENT_FILL0_FEATURES

    # 空源（全 null Float64）→ coverage 0 → 被摘
    rows_null = []
    for day in days:
        for c in codes:
            rows_null.append({
                "trade_date": day,
                "ts_code": c,
                "fc_flag": None,
                "fc_type_score": None,
            })
    df_null = pl.DataFrame(rows_null).with_columns([
        pl.col("fc_flag").cast(pl.Float64),
        pl.col("fc_type_score").cast(pl.Float64),
    ])
    cov = leaf_holdout_coverage(
        df_null, ["fc_flag", "fc_type_score"], hstart,
        leaf_map={"fc_flag": "fc_flag", "fc_type_score": "fc_type_score"},
        min_cross=30,
    )
    assert cov["fc_flag"] == 0.0
    assert cov["fc_type_score"] == 0.0


def test_expression_factor_attaches_forecast_on_demand(monkeypatch):
    """ExpressionFactor 引用 fc_* 时走 attach（与 prepare 同函数）。"""
    from factorzen.discovery.factor import ExpressionFactor

    calls: list[str] = []

    def _fake_attach_forecast(daily, forecast_df=None):
        calls.append("forecast")
        return daily.with_columns(
            pl.lit(1.0).alias("fc_type_score"),
            pl.lit(0.0).alias("fc_surprise"),
            pl.lit(1.0).alias("fc_flag"),
        )

    def _fake_attach_express(daily, express_df=None):
        calls.append("express")
        return daily.with_columns(pl.lit(0.0).alias("express_yoy"))

    monkeypatch.setattr(
        "factorzen.daily.data.events.attach_forecast", _fake_attach_forecast
    )
    monkeypatch.setattr(
        "factorzen.daily.data.events.attach_express", _fake_attach_express
    )

    fac = ExpressionFactor("rank(fc_type_score)", mined_name="fc_ts")
    assert "fc_type_score" in fac._feats
    # 最小日线 + derived 需要的 pre_close
    daily = _daily(["20240102", "20240103", "20240104"]).with_columns(
        pl.lit(10.0).alias("pre_close"),
    )

    class _Ctx:
        start = "20240103"
        daily = type("LF", (), {"collect": staticmethod(lambda: daily)})()
        daily_basic = type("LF", (), {
            "collect": staticmethod(lambda: pl.DataFrame())
        })()

    out = fac.compute(_Ctx())
    assert "forecast" in calls
    assert out.height >= 1

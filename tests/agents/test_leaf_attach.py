"""合并自 agents 相关碎片测试（test_leaf_attach.py）。

test_margin_attach.py：两融（margin_detail）叶子：T+1 lag、单位换算、双路径一致、叶子注册
test_holder_attach.py：股东户数叶子：ann_date PIT、期际环比、双路径一致、叶子注册
test_toplist_attach.py：龙虎榜叶子：lag(1)、条件 fill 0、同日聚合、单位换算、双路径、leaf_health
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from factorzen.daily.data.flows import attach_flows
from factorzen.daily.data.pit import attach_holders


# ==== 来自 test_margin_attach.py ====
def _daily__margin(dates: list[str], codes: list[str] | None = None, *,
           circ_mv: float = 1e6, amount: float = 1e5) -> pl.DataFrame:
    """合成日线帧。circ_mv 默认 1e6 万元 = 1e10 元；amount 默认 1e5 千元 = 1e8 元。"""
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
                "amount": amount,       # 千元
                "circ_mv": circ_mv,     # 万元
            })
    return pl.DataFrame(rows)


def _margin(rows: list[dict]) -> pl.DataFrame:
    """rows: {ts_code, trade_date YYYYMMDD, rzye, rzmre, rqyl}；rzye/rzmre 单位元。"""
    return pl.DataFrame({
        "ts_code": [r["ts_code"] for r in rows],
        "trade_date": [dt.datetime.strptime(r["trade_date"], "%Y%m%d").date() for r in rows],
        "rzye": [r["rzye"] for r in rows],
        "rzmre": [r["rzmre"] for r in rows],
        "rqyl": [r["rqyl"] for r in rows],
    })


# ── A. lag / 覆盖 / 单位 ──────────────────────────────────────────────────────


def test_margin_attach_values_suite():
    """t 日行拿到 t-1 两融；首日（组内第一交易日）null。；非融资融券标的股：无 margin 行 → join 后全 null（诚实缺测，不填 0）。；单位防回归：rzye 元 / (circ_mv 万元 × 1e4) = 元/元。；比值必须**源日同日**计算后整体 lag——t 日拿到 rzmre(t-1)/amount(t-1)，"""
    # -- 原 test_margin_lag1_first_day_null_and_t_gets_t_minus_1 --
    def _section_0_test_margin_lag1_first_day_null_and_t_gets_t_minus_1():
        margin = _margin([
            {"ts_code": "000001.SZ", "trade_date": "20240102", "rzye": 1e9, "rzmre": 1e8, "rqyl": 100.0},
            {"ts_code": "000001.SZ", "trade_date": "20240103", "rzye": 2e9, "rzmre": 2e8, "rqyl": 200.0},
            {"ts_code": "000001.SZ", "trade_date": "20240104", "rzye": 3e9, "rzmre": 3e8, "rqyl": 300.0},
        ])
        out = attach_flows(
            _daily__margin(["20240102", "20240103", "20240104"]),
            injected={"moneyflow": pl.DataFrame(), "hk_hold": pl.DataFrame(), "margin_detail": margin},
        )
        by = {r["trade_date"]: r for r in out.iter_rows(named=True)}
        # 首日：无 t-1 → null
        assert by[dt.date(2024, 1, 2)]["margin_balance"] is None
        assert by[dt.date(2024, 1, 2)]["short_balance"] is None
        assert by[dt.date(2024, 1, 2)]["margin_ratio"] is None
        assert by[dt.date(2024, 1, 2)]["margin_buy_ratio"] is None
        # t=01-03 拿到 01-02 的 rzye=1e9 / rqyl=100
        assert by[dt.date(2024, 1, 3)]["margin_balance"] == 1e9
        assert by[dt.date(2024, 1, 3)]["short_balance"] == 100.0
        # t=01-04 拿到 01-03 的值
        assert by[dt.date(2024, 1, 4)]["margin_balance"] == 2e9
        assert by[dt.date(2024, 1, 4)]["short_balance"] == 200.0

    _section_0_test_margin_lag1_first_day_null_and_t_gets_t_minus_1()

    # -- 原 test_non_margin_stock_all_null --
    def _section_1_test_non_margin_stock_all_null():
        margin = _margin([
            {"ts_code": "000001.SZ", "trade_date": "20240102", "rzye": 1e9, "rzmre": 1e8, "rqyl": 50.0},
            {"ts_code": "000001.SZ", "trade_date": "20240103", "rzye": 1e9, "rzmre": 1e8, "rqyl": 50.0},
        ])
        # 000002.SZ 不在 margin 里
        out = attach_flows(
            _daily__margin(["20240102", "20240103"], codes=["000002.SZ"]),
            injected={"moneyflow": pl.DataFrame(), "hk_hold": pl.DataFrame(), "margin_detail": margin},
        )
        for col in ("margin_balance", "short_balance", "margin_ratio", "margin_buy_ratio"):
            assert out[col].null_count() == out.height, f"{col} 应对非标的全 null"

    _section_1_test_non_margin_stock_all_null()

    # -- 原 test_margin_ratio_unit_scale --
    def _section_2_test_margin_ratio_unit_scale():
        margin = _margin([
            {"ts_code": "000001.SZ", "trade_date": "20240102", "rzye": 1e9, "rzmre": 1e7, "rqyl": 1.0},
            {"ts_code": "000001.SZ", "trade_date": "20240103", "rzye": 1e9, "rzmre": 1e7, "rqyl": 1.0},
        ])
        # circ_mv=1e6 万元 → 1e10 元；amount=1e4 千元 → 1e7 元
        out = attach_flows(
            _daily__margin(["20240102", "20240103"], circ_mv=1e6, amount=1e4),
            injected={"moneyflow": pl.DataFrame(), "hk_hold": pl.DataFrame(), "margin_detail": margin},
        )
        row = out.filter(pl.col("trade_date") == dt.date(2024, 1, 3)).row(0, named=True)
        assert abs(row["margin_ratio"] - 0.1) < 1e-12
        # margin_buy_ratio: lag rzmre=1e7 元 / (1e4 千元 × 1e3 = 1e7 元) = 1.0
        assert abs(row["margin_buy_ratio"] - 1.0) < 1e-12

    _section_2_test_margin_ratio_unit_scale()

    # -- 原 test_margin_ratios_are_same_day_then_lagged --
    def _section_3_test_margin_ratios_are_same_day_then_lagged():
        margin = _margin([
            {"ts_code": "000001.SZ", "trade_date": "20240102", "rzye": 1e9, "rzmre": 1e7, "rqyl": 1.0},
            {"ts_code": "000001.SZ", "trade_date": "20240103", "rzye": 1e9, "rzmre": 1e7, "rqyl": 1.0},
        ])
        d1 = _daily__margin(["20240102"], amount=1e4)
        d2 = _daily__margin(["20240103"], amount=1e5)
        out = attach_flows(
            pl.concat([d1, d2]),
            injected={"moneyflow": pl.DataFrame(), "hk_hold": pl.DataFrame(), "margin_detail": margin},
        )
        row = out.filter(pl.col("trade_date") == dt.date(2024, 1, 3)).row(0, named=True)
        assert abs(row["margin_buy_ratio"] - 1.0) < 1e-12, \
            f"应为同日比 1.0(1e7/1e7)，误用当日分母会得 0.1；实得 {row['margin_buy_ratio']}"

    _section_3_test_margin_ratios_are_same_day_then_lagged()


# ── B. 双路径逐值一致 ────────────────────────────────────────────────────────


def test_leaf_dual_path_parity_suite(monkeypatch):
    """prepare_mining_daily 与 ExpressionFactor.compute 共用 attach_flows → 逐值一致。；prepare_mining_daily 与 ExpressionFactor.compute 共用 attach_holders → 逐值一致。；test_toplist_mining_and_materialize_paths_value_identical"""
    # -- 原 test_mining_and_materialize_paths_value_identical --
    def _section_0_test_mining_and_materialize_paths_value_identical(mp):
        from datetime import date

        import factorzen.daily.data.context as ctx_mod
        import factorzen.pipelines.factor_mine as fm
        from factorzen.discovery.factor import ExpressionFactor

        dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
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
            "circ_mv": [1e6, 1e6, 1e6],
            "total_mv": [2e6, 2e6, 2e6],
        })
        margin = _margin([
            {"ts_code": "000001.SZ", "trade_date": "20240102", "rzye": 1e9, "rzmre": 1e7, "rqyl": 10.0},
            {"ts_code": "000001.SZ", "trade_date": "20240103", "rzye": 2e9, "rzmre": 2e7, "rqyl": 20.0},
            {"ts_code": "000001.SZ", "trade_date": "20240104", "rzye": 3e9, "rzmre": 3e7, "rqyl": 30.0},
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

        mp.setattr(ctx_mod, "FactorDataContext", _FakeCtx)

        # 挖掘路径：prepare 会调 attach_flows；拦截 load 注入 margin
        import factorzen.daily.data.flows as flows_mod
        real_attach = flows_mod.attach_flows

        def _attach_with_margin(d, *, injected=None):
            inj = dict(injected or {})
            inj.setdefault("moneyflow", pl.DataFrame())
            inj.setdefault("hk_hold", pl.DataFrame())
            inj.setdefault("margin_detail", margin)
            return real_attach(d, injected=inj)

        mp.setattr(flows_mod, "attach_flows", _attach_with_margin)
        # prepare_mining_daily 从 factorzen.daily.data.flows import attach_flows —— 需补丁源模块
        mp.setattr("factorzen.pipelines.factor_mine.attach_flows", _attach_with_margin, raising=False)

        # prepare 内部 from factorzen.daily.data.flows import attach_flows 是局部 import，
        # patch flows_mod.attach_flows 即可覆盖。
        mined = fm.prepare_mining_daily("20240102", "20240104")

        # 物化路径
        class _Ctx:
            start = "20240102"
            end = "20240104"

            @property
            def daily(self):
                return daily.lazy()

            @property
            def daily_basic(self):
                return basic.lazy()

        # ExpressionFactor.compute 也走 attach_flows（已 patch）
        fac = ExpressionFactor("rank(margin_ratio)", mined_name="m_ratio")
        # 物化输出只有 factor_value；改用 attach 帧对齐叶子列
        mat_frame = _attach_with_margin(
            daily.join(basic, on=["trade_date", "ts_code"], how="left")
        )

        for col in ("margin_ratio", "margin_buy_ratio", "margin_balance", "short_balance"):
            a = mined.sort(["ts_code", "trade_date"])[col].to_list()
            b = mat_frame.sort(["ts_code", "trade_date"])[col].to_list()
            assert a == b, f"双路径 {col} 不一致: mine={a} mat={b}"

        # ExpressionFactor 能编译/跑通（不抛）
        out = fac.compute(_Ctx())
        assert "factor_value" in out.columns

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_mining_and_materialize_paths_value_identical(mp)

    # -- 原 test_holder_mining_and_materialize_paths_value_identical --
    def _section_1_test_holder_mining_and_materialize_paths_value_identical(mp):
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

        mp.setattr(ctx_mod, "FactorDataContext", _FakeCtx)

        real_attach = pit_mod.attach_holders

        def _attach_with_holder(d, holder_df=None):
            return real_attach(d, holder_df=holder if holder_df is None else holder_df)

        mp.setattr(pit_mod, "attach_holders", _attach_with_holder)
        mp.setattr(pit_mod, "attach_fundamentals", lambda d, fina_df=None: d)

        import factorzen.daily.data.flows as flows_mod
        mp.setattr(flows_mod, "attach_flows", lambda d, **kw: d)

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

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_holder_mining_and_materialize_paths_value_identical(mp)

    # -- 原 test_toplist_mining_and_materialize_paths_value_identical --
    def _section_2_test_toplist_mining_and_materialize_paths_value_identical(mp):
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

        mp.setattr(ctx_mod, "FactorDataContext", _FakeCtx)

        real_attach = flows_mod.attach_flows

        def _attach_with_top(d, *, injected=None):
            inj = dict(injected or {})
            inj.setdefault("moneyflow", pl.DataFrame())
            inj.setdefault("hk_hold", pl.DataFrame())
            inj.setdefault("margin_detail", pl.DataFrame())
            inj.setdefault("top_list", top)
            return real_attach(d, injected=inj)

        mp.setattr(flows_mod, "attach_flows", _attach_with_top)

        import factorzen.daily.data.pit as pit_mod
        mp.setattr(pit_mod, "attach_fundamentals", lambda d, fina_df=None: d)
        mp.setattr(pit_mod, "attach_holders", lambda d, holder_df=None: d)

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

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_toplist_mining_and_materialize_paths_value_identical(mp)


# ── C. 叶子注册 / prompt / leaf_health ────────────────────────────────────────


def test_margin_register_prompt_health_suite():
    """test_margin_leaves_registered_and_parse；generation / hypothesis 文案含两融描述（T+1 lag / 单位说明）。；新叶子自动经 leaf_health 覆盖检查（P1 全局：LEAF_FEATURES → filter）。"""
    # -- 原 test_margin_leaves_registered_and_parse --
    def _section_0_test_margin_leaves_registered_and_parse():
        from factorzen.discovery.expression import feature_names, parse_expr
        from factorzen.discovery.operators import FLOW_FEATURES, LEAF_FEATURES, MARGIN_FEATURES

        expected = {"margin_ratio", "margin_buy_ratio", "margin_balance", "short_balance"}
        assert expected <= MARGIN_FEATURES
        assert expected <= FLOW_FEATURES
        for leaf in expected:
            assert leaf in LEAF_FEATURES
            feats = feature_names(parse_expr(f"rank({leaf})"))
            assert leaf in feats
            assert feats & FLOW_FEATURES  # 触发物化路径 attach 门

    _section_0_test_margin_leaves_registered_and_parse()

    # -- 原 test_prompt_mentions_margin_family --
    def _section_1_test_prompt_mentions_margin_family():
        from factorzen.agents.roles.hypothesis import signal_families
        from factorzen.llm.generation import build_agent_messages

        fam = signal_families("ashare")
        assert "两融" in fam or "杠杆" in fam

        sys = build_agent_messages(["ts_mean"], ["close", "margin_ratio"], market="ashare")[0]["content"]
        assert "两融" in sys or "杠杆" in sys
        assert "lag" in sys.lower() or "T+1" in sys or "t+1" in sys.lower()

    _section_1_test_prompt_mentions_margin_family()

    # -- 原 test_margin_leaves_pass_through_leaf_health --
    def _section_2_test_margin_leaves_pass_through_leaf_health():
        from factorzen.discovery.leaf_health import filter_leaves_by_holdout_coverage
        from factorzen.discovery.operators import MARGIN_FEATURES

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
                    # holdout 段 margin_ratio 全 null → 死叶
                    "margin_ratio": None if day >= hstart else 0.05,
                    "margin_balance": None if day >= hstart else 1e9,
                })
        df = pl.DataFrame(rows)
        leaves = ["close", "margin_ratio", "margin_balance"]
        leaf_map = {"close": "close_adj", "margin_ratio": "margin_ratio",
                    "margin_balance": "margin_balance"}
        kept, excluded = filter_leaves_by_holdout_coverage(
            df, leaves, hstart, leaf_map=leaf_map, min_coverage=0.5, min_cross=30,
        )
        assert "margin_ratio" in excluded
        assert "margin_balance" in excluded
        assert "close" in kept
        # 证明 MARGIN 叶子在注册集中（leaf_health 上游用 LEAF_FEATURES.keys()）
        assert MARGIN_FEATURES

    _section_2_test_margin_leaves_pass_through_leaf_health()


# ==== 来自 test_holder_attach.py ====
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


def _daily__holder(dates: list[str], codes: list[str] | None = None) -> pl.DataFrame:
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


def test_holder_pit_suite():
    """t 在首期公告前 → holder_num / holder_num_chg 必须 null。；公告间持有上一期；新公告日后切换；期间无 ffill 伪造（PIT 自然前向）。；环比在期际算好：第二期 chg=-0.2 仅 0815 起可见；第一期无上期 → chg null。；无股东户数数据的股票 → null（诚实缺测，不填 0）。；test_holder_empty_source_null_cols"""
    # -- 原 test_holder_no_future_leak_before_announcement --
    def _section_0_test_holder_no_future_leak_before_announcement():
        out = attach_holders(_daily__holder(["20200410"]), holder_df=_holder())
        row = out.filter(pl.col("trade_date") == dt.date(2020, 4, 10)).row(0, named=True)
        assert row["holder_num"] is None, "首期公告前泄漏 → 未来函数!"
        assert row["holder_num_chg"] is None

    _section_0_test_holder_no_future_leak_before_announcement()

    # -- 原 test_holder_pit_uses_latest_announced_period --
    def _section_1_test_holder_pit_uses_latest_announced_period():
        out = attach_holders(
            _daily__holder(["20200410", "20200501", "20200810", "20200820"]),
            holder_df=_holder(),
        )
        by = {r["trade_date"]: r for r in out.iter_rows(named=True)}
        assert by[dt.date(2020, 4, 10)]["holder_num"] is None
        assert by[dt.date(2020, 5, 1)]["holder_num"] == 10000.0
        # 第二期 0815 才公告 → 0810 仍用第一期
        assert by[dt.date(2020, 8, 10)]["holder_num"] == 10000.0
        assert by[dt.date(2020, 8, 20)]["holder_num"] == 8000.0

    _section_1_test_holder_pit_uses_latest_announced_period()

    # -- 原 test_holder_num_chg_period_over_period_and_pit --
    def _section_2_test_holder_num_chg_period_over_period_and_pit():
        out = attach_holders(
            _daily__holder(["20200501", "20200810", "20200820"]),
            holder_df=_holder(),
        )
        by = {r["trade_date"]: r for r in out.iter_rows(named=True)}
        # 第一期已公告、尚无上期 → chg null
        assert by[dt.date(2020, 5, 1)]["holder_num_chg"] is None
        # 第二期公告前不可见 chg
        assert by[dt.date(2020, 8, 10)]["holder_num_chg"] is None
        # 第二期公告后 chg = (8000-10000)/10000 = -0.2
        assert abs(by[dt.date(2020, 8, 20)]["holder_num_chg"] - (-0.2)) < 1e-12

    _section_2_test_holder_num_chg_period_over_period_and_pit()

    # -- 原 test_holder_missing_stock_null --
    def _section_3_test_holder_missing_stock_null():
        out = attach_holders(
            _daily__holder(["20200501"], codes=["000002.SZ"]),
            holder_df=_holder(),
        )
        assert out["holder_num"][0] is None
        assert out["holder_num_chg"][0] is None

    _section_3_test_holder_missing_stock_null()

    # -- 原 test_holder_empty_source_null_cols --
    def _section_4_test_holder_empty_source_null_cols():
        out = attach_holders(_daily__holder(["20200501"]), holder_df=pl.DataFrame())
        assert "holder_num" in out.columns and "holder_num_chg" in out.columns
        assert out["holder_num"][0] is None

    _section_4_test_holder_empty_source_null_cols()


def test_holder_register_prompt_suite():
    """test_holder_leaves_registered_and_parse；test_prompt_mentions_holder_family"""
    # -- 原 test_holder_leaves_registered_and_parse --
    def _section_0_test_holder_leaves_registered_and_parse():
        from factorzen.discovery.expression import feature_names, parse_expr
        from factorzen.discovery.operators import HOLDER_FEATURES, LEAF_FEATURES

        expected = {"holder_num", "holder_num_chg"}
        assert expected <= HOLDER_FEATURES
        for leaf in expected:
            assert leaf in LEAF_FEATURES
            assert leaf in feature_names(parse_expr(f"rank({leaf})"))

    _section_0_test_holder_leaves_registered_and_parse()

    # -- 原 test_prompt_mentions_holder_family --
    def _section_1_test_prompt_mentions_holder_family():
        from factorzen.agents.roles.hypothesis import signal_families
        from factorzen.llm.generation import build_agent_messages

        fam = signal_families("ashare")
        assert "股东" in fam or "holder" in fam.lower()

        sys = build_agent_messages(["ts_mean"], ["close", "holder_num"], market="ashare")[0]["content"]
        assert "股东" in sys or "holder" in sys.lower()
        assert "ann_date" in sys or "公告" in sys or "PIT" in sys

    _section_1_test_prompt_mentions_holder_family()


# ==== 来自 test_toplist_attach.py ====
_TOPLIST_EMPTY_CODE = "__EMPTY__"


def _daily__toplist(dates: list[str], codes: list[str] | None = None, *,
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
        "net_amount": [r.get("net_amount") for r in rows],
        "amount": [r.get("amount") for r in rows],
        "reason": [r.get("reason", "涨幅偏离") for r in rows],
    })


def _inj(top: pl.DataFrame) -> dict:
    return {
        "moneyflow": pl.DataFrame(),
        "hk_hold": pl.DataFrame(),
        "margin_detail": pl.DataFrame(),
        "top_list": top,
    }


def _known_days(dates: list[str], listed: list[dict] | None = None) -> pl.DataFrame:
    """构造已知日集合：真实行 ∪ __EMPTY__ sentinel（模拟 fetch 已拉标记）。"""
    parts = []
    listed = listed or []
    listed_dates = {r["trade_date"] for r in listed}
    if listed:
        parts.append(_top(listed))
    sent_rows = []
    for d in dates:
        if d not in listed_dates:
            sent_rows.append({
                "ts_code": _TOPLIST_EMPTY_CODE,
                "trade_date": d,
                "net_amount": None,
                "amount": None,
                "reason": None,
            })
    if sent_rows:
        parts.append(_top(sent_rows))
    return pl.concat(parts) if parts else pl.DataFrame()


# ── A. lag / 条件 fill0 / 聚合 / 单位 ─────────────────────────────────────────


def test_toplist_values_suite():
    """t 日拿到 t-1 上榜信息；已知日未上榜 fill 0（非 null）。；已知日内从未上榜的股票：0，不是 null（与两融非标的=null 相反）。；条件 fill-0：已知日未上榜=0、sentinel 空日=0、未知日=null。；全空源（无数据文件）→ 全 null 而非全 0（覆盖审计诚实）。；同日多条上榜原因：net_amount 先 sum 再算比；amount 取 first（同股同日相同）。；单位钉死：net_amount 万元×1e4、amount 千元×1e3。"""
    # -- 原 test_toplist_lag1_and_not_listed_fill_zero --
    def _section_0_test_toplist_lag1_and_not_listed_fill_zero():
        top = _known_days(
            ["20240102", "20240103", "20240104"],
            listed=[{
                "ts_code": "000001.SZ", "trade_date": "20240102",
                "net_amount": 1000.0, "amount": 1e5,
            }],
        )
        out = attach_flows(
            _daily__toplist(["20240102", "20240103", "20240104"], amount=1e5),
            injected=_inj(top),
        )
        by = {r["trade_date"]: r for r in out.iter_rows(named=True)}
        # 01-02：lag 后无 t-1 → null（帧内无更早已知日）
        assert by[dt.date(2024, 1, 2)]["top_list_flag"] is None
        assert by[dt.date(2024, 1, 2)]["top_list_net_buy"] is None
        # 01-03：拿到 01-02 上榜
        assert by[dt.date(2024, 1, 3)]["top_list_flag"] == 1.0
        assert abs(by[dt.date(2024, 1, 3)]["top_list_net_buy"] - 0.1) < 1e-12
        # 01-04：昨日已知且未上榜 → 0
        assert by[dt.date(2024, 1, 4)]["top_list_flag"] == 0.0
        assert by[dt.date(2024, 1, 4)]["top_list_net_buy"] == 0.0

    _section_0_test_toplist_lag1_and_not_listed_fill_zero()

    # -- 原 test_not_listed_stock_fill_zero_not_null --
    def _section_1_test_not_listed_stock_fill_zero_not_null():
        top = _known_days(
            ["20240102", "20240103", "20240104"],
            listed=[
                {"ts_code": "000001.SZ", "trade_date": "20240102",
                 "net_amount": 100.0, "amount": 1e4},
                {"ts_code": "000001.SZ", "trade_date": "20240103",
                 "net_amount": 100.0, "amount": 1e4},
            ],
        )
        out = attach_flows(
            _daily__toplist(["20240102", "20240103", "20240104"], codes=["000002.SZ"]),
            injected=_inj(top),
        )
        by = {r["trade_date"]: r for r in out.iter_rows(named=True)}
        # 首日 lag 无前值 → null；其后已知日未上榜 → 0
        assert by[dt.date(2024, 1, 2)]["top_list_flag"] is None
        assert by[dt.date(2024, 1, 3)]["top_list_flag"] == 0.0
        assert by[dt.date(2024, 1, 4)]["top_list_flag"] == 0.0
        assert by[dt.date(2024, 1, 3)]["top_list_net_buy"] == 0.0
        assert by[dt.date(2024, 1, 4)]["top_list_net_buy"] == 0.0

    _section_1_test_not_listed_stock_fill_zero_not_null()

    # -- 原 test_toplist_conditional_fill0_unknown_day_null --
    def _section_2_test_toplist_conditional_fill0_unknown_day_null():
        top = pl.concat([
            _top([{
                "ts_code": "000001.SZ", "trade_date": "20240102",
                "net_amount": 1000.0, "amount": 1e5,
            }]),
            _top([{
                "ts_code": _TOPLIST_EMPTY_CODE, "trade_date": "20240103",
                "net_amount": None, "amount": None, "reason": None,
            }]),
        ])
        out = attach_flows(
            _daily__toplist(["20240102", "20240103", "20240104", "20240105"], amount=1e5),
            injected=_inj(top),
        )
        by = {r["trade_date"]: r for r in out.iter_rows(named=True)}
        # lag 后：t 日 = t-1 事件状态
        assert by[dt.date(2024, 1, 2)]["top_list_flag"] is None  # 无 t-1
        assert by[dt.date(2024, 1, 3)]["top_list_flag"] == 1.0   # t-1=01-02 上榜
        assert by[dt.date(2024, 1, 4)]["top_list_flag"] == 0.0   # t-1=01-03 sentinel 空日
        assert by[dt.date(2024, 1, 5)]["top_list_flag"] is None  # t-1=01-04 未知日
        assert by[dt.date(2024, 1, 5)]["top_list_net_buy"] is None

    _section_2_test_toplist_conditional_fill0_unknown_day_null()

    # -- 原 test_toplist_empty_source_all_null_not_zero --
    def _section_3_test_toplist_empty_source_all_null_not_zero():
        out = attach_flows(
            _daily__toplist(["20240102", "20240103"]),
            injected=_inj(pl.DataFrame()),
        )
        assert out["top_list_flag"].null_count() == out.height
        assert out["top_list_net_buy"].null_count() == out.height
        assert all(v is None for v in out["top_list_flag"].to_list())

    _section_3_test_toplist_empty_source_all_null_not_zero()

    # -- 原 test_toplist_same_day_multi_reason_sum_net_amount --
    def _section_4_test_toplist_same_day_multi_reason_sum_net_amount():
        top = _top([
            {"ts_code": "000001.SZ", "trade_date": "20240102",
             "net_amount": 500.0, "amount": 1e5, "reason": "涨幅偏离"},
            {"ts_code": "000001.SZ", "trade_date": "20240102",
             "net_amount": 500.0, "amount": 1e5, "reason": "换手率"},
        ])
        out = attach_flows(
            _daily__toplist(["20240102", "20240103"], amount=1e5),
            injected=_inj(top),
        )
        row = out.filter(pl.col("trade_date") == dt.date(2024, 1, 3)).row(0, named=True)
        # sum net=1000 万元 → 1e7 元；amount 1e5 千元 → 1e8 元；比=0.1
        assert abs(row["top_list_net_buy"] - 0.1) < 1e-12
        assert row["top_list_flag"] == 1.0

    _section_4_test_toplist_same_day_multi_reason_sum_net_amount()

    # -- 原 test_toplist_unit_scale_net_wan_amount_qian --
    def _section_5_test_toplist_unit_scale_net_wan_amount_qian():
        top = _top([
            {"ts_code": "000001.SZ", "trade_date": "20240102",
             "net_amount": 2000.0, "amount": 5e4},
        ])
        out = attach_flows(
            _daily__toplist(["20240102", "20240103"]),
            injected=_inj(top),
        )
        row = out.filter(pl.col("trade_date") == dt.date(2024, 1, 3)).row(0, named=True)
        assert abs(row["top_list_net_buy"] - 0.4) < 1e-12

    _section_5_test_toplist_unit_scale_net_wan_amount_qian()


# ── B. 双路径 ────────────────────────────────────────────────────────────────


# ── C. 注册 / prompt / leaf_health ────────────────────────────────────────────


def test_toplist_register_health_suite():
    """test_toplist_leaves_registered_and_parse；test_prompt_mentions_toplist_family；已知窗口内 fill 0 后龙虎榜叶子 holdout 覆盖=100%，经 leaf_health 检查保留。；部分覆盖帧（未拉取日为 null）→ leaf_health 给出 <100% 真实覆盖率。"""
    # -- 原 test_toplist_leaves_registered_and_parse --
    def _section_0_test_toplist_leaves_registered_and_parse():
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

    _section_0_test_toplist_leaves_registered_and_parse()

    # -- 原 test_prompt_mentions_toplist_family --
    def _section_1_test_prompt_mentions_toplist_family():
        from factorzen.agents.roles.hypothesis import signal_families
        from factorzen.llm.generation import build_agent_messages

        fam = signal_families("ashare")
        assert "龙虎" in fam or "top_list" in fam.lower()

        sys = build_agent_messages(
            ["ts_mean"], ["close", "top_list_net_buy"], market="ashare"
        )[0]["content"]
        assert "龙虎" in sys or "top_list" in sys.lower()
        assert "lag" in sys.lower() or "盘后" in sys or "T+1" in sys or "t+1" in sys.lower()

    _section_1_test_prompt_mentions_toplist_family()

    # -- 原 test_toplist_fill0_leaves_pass_leaf_health_full_coverage --
    def _section_2_test_toplist_fill0_leaves_pass_leaf_health_full_coverage():
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
                    # 已知日 fill 0 语义：全有值
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

    _section_2_test_toplist_fill0_leaves_pass_leaf_health_full_coverage()

    # -- 原 test_toplist_partial_coverage_leaf_health_sees_gaps --
    def _section_3_test_toplist_partial_coverage_leaf_health_sees_gaps():
        from factorzen.discovery.leaf_health import leaf_holdout_coverage

        # holdout 10 天：前 5 天有值(0)，后 5 天 null（未回补）
        days = [dt.date(2024, 1, d) for d in range(2, 12)]
        hstart = days[0]
        codes = [f"{i:06d}.SZ" for i in range(40)]
        rows = []
        for i, day in enumerate(days):
            val = 0.0 if i < 5 else None
            for c in codes:
                rows.append({
                    "trade_date": day,
                    "ts_code": c,
                    "top_list_flag": val,
                    "top_list_net_buy": val,
                })
        df = pl.DataFrame(rows)
        cov = leaf_holdout_coverage(
            df, ["top_list_flag", "top_list_net_buy"], hstart,
            leaf_map={"top_list_flag": "top_list_flag", "top_list_net_buy": "top_list_net_buy"},
            min_cross=30,
        )
        assert cov["top_list_flag"] == 0.5
        assert cov["top_list_net_buy"] == 0.5
        assert cov["top_list_flag"] < 1.0

    _section_3_test_toplist_partial_coverage_leaf_health_sees_gaps()



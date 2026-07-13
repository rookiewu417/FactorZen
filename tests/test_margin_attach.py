"""两融（margin_detail）叶子：T+1 lag、单位换算、双路径一致、叶子注册。

披露时点：T 日两融数据 T+1 早间披露 → t 日信号只能用 t-1 日两融；
lag 在 attach 层按 ts_code 组内交易日序 shift(1) 结构性完成。
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from factorzen.daily.data.flows import attach_flows


def _daily(dates: list[str], codes: list[str] | None = None, *,
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


def test_margin_lag1_first_day_null_and_t_gets_t_minus_1():
    """t 日行拿到 t-1 两融；首日（组内第一交易日）null。"""
    margin = _margin([
        {"ts_code": "000001.SZ", "trade_date": "20240102", "rzye": 1e9, "rzmre": 1e8, "rqyl": 100.0},
        {"ts_code": "000001.SZ", "trade_date": "20240103", "rzye": 2e9, "rzmre": 2e8, "rqyl": 200.0},
        {"ts_code": "000001.SZ", "trade_date": "20240104", "rzye": 3e9, "rzmre": 3e8, "rqyl": 300.0},
    ])
    out = attach_flows(
        _daily(["20240102", "20240103", "20240104"]),
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


def test_non_margin_stock_all_null():
    """非融资融券标的股：无 margin 行 → join 后全 null（诚实缺测，不填 0）。"""
    margin = _margin([
        {"ts_code": "000001.SZ", "trade_date": "20240102", "rzye": 1e9, "rzmre": 1e8, "rqyl": 50.0},
        {"ts_code": "000001.SZ", "trade_date": "20240103", "rzye": 1e9, "rzmre": 1e8, "rqyl": 50.0},
    ])
    # 000002.SZ 不在 margin 里
    out = attach_flows(
        _daily(["20240102", "20240103"], codes=["000002.SZ"]),
        injected={"moneyflow": pl.DataFrame(), "hk_hold": pl.DataFrame(), "margin_detail": margin},
    )
    for col in ("margin_balance", "short_balance", "margin_ratio", "margin_buy_ratio"):
        assert out[col].null_count() == out.height, f"{col} 应对非标的全 null"


def test_margin_ratio_unit_scale():
    """单位防回归：rzye 元 / (circ_mv 万元 × 1e4) = 元/元。

    1e9 元余额 / (1e6 万元 × 1e4 = 1e10 元市值) = 0.1。
    """
    margin = _margin([
        {"ts_code": "000001.SZ", "trade_date": "20240102", "rzye": 1e9, "rzmre": 1e7, "rqyl": 1.0},
        {"ts_code": "000001.SZ", "trade_date": "20240103", "rzye": 1e9, "rzmre": 1e7, "rqyl": 1.0},
    ])
    # circ_mv=1e6 万元 → 1e10 元；amount=1e4 千元 → 1e7 元
    out = attach_flows(
        _daily(["20240102", "20240103"], circ_mv=1e6, amount=1e4),
        injected={"moneyflow": pl.DataFrame(), "hk_hold": pl.DataFrame(), "margin_detail": margin},
    )
    row = out.filter(pl.col("trade_date") == dt.date(2024, 1, 3)).row(0, named=True)
    assert abs(row["margin_ratio"] - 0.1) < 1e-12
    # margin_buy_ratio: lag rzmre=1e7 元 / (1e4 千元 × 1e3 = 1e7 元) = 1.0
    assert abs(row["margin_buy_ratio"] - 1.0) < 1e-12


def test_margin_ratios_are_same_day_then_lagged():
    """比值必须**源日同日**计算后整体 lag——t 日拿到 rzmre(t-1)/amount(t-1)，
    而非跨日的 rzmre(t-1)/amount(t)（后者携带分母错日噪声）。

    构造 amount 逐日变化：01-02 amount=1e4 千元(=1e7 元)、01-03 amount=1e5 千元(=1e8 元)。
    t=01-03 的 margin_buy_ratio 应 = rzmre(01-02)/amount(01-02) = 1e7/1e7 = 1.0；
    若实现误用 amount(01-03) 会得 0.1。
    """
    margin = _margin([
        {"ts_code": "000001.SZ", "trade_date": "20240102", "rzye": 1e9, "rzmre": 1e7, "rqyl": 1.0},
        {"ts_code": "000001.SZ", "trade_date": "20240103", "rzye": 1e9, "rzmre": 1e7, "rqyl": 1.0},
    ])
    d1 = _daily(["20240102"], amount=1e4)
    d2 = _daily(["20240103"], amount=1e5)
    out = attach_flows(
        pl.concat([d1, d2]),
        injected={"moneyflow": pl.DataFrame(), "hk_hold": pl.DataFrame(), "margin_detail": margin},
    )
    row = out.filter(pl.col("trade_date") == dt.date(2024, 1, 3)).row(0, named=True)
    assert abs(row["margin_buy_ratio"] - 1.0) < 1e-12, \
        f"应为同日比 1.0(1e7/1e7)，误用当日分母会得 0.1；实得 {row['margin_buy_ratio']}"


def test_margin_buy_ratio_unit_scale():
    """rzmre 元 / (amount 千元 × 1e3)。2e8 元 / (1e5 千元 × 1e3 = 1e8 元) = 2.0。"""
    margin = _margin([
        {"ts_code": "000001.SZ", "trade_date": "20240102", "rzye": 1.0, "rzmre": 2e8, "rqyl": 0.0},
        {"ts_code": "000001.SZ", "trade_date": "20240103", "rzye": 1.0, "rzmre": 9e9, "rqyl": 0.0},
    ])
    out = attach_flows(
        _daily(["20240102", "20240103"], amount=1e5),
        injected={"moneyflow": pl.DataFrame(), "hk_hold": pl.DataFrame(), "margin_detail": margin},
    )
    row = out.filter(pl.col("trade_date") == dt.date(2024, 1, 3)).row(0, named=True)
    assert abs(row["margin_buy_ratio"] - 2.0) < 1e-12


# ── B. 双路径逐值一致 ────────────────────────────────────────────────────────


def test_mining_and_materialize_paths_value_identical(monkeypatch):
    """prepare_mining_daily 与 ExpressionFactor.compute 共用 attach_flows → 逐值一致。

    通过注入同一 margin 帧，比较两路径产出的 margin_ratio 列。
    """
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

    monkeypatch.setattr(ctx_mod, "FactorDataContext", _FakeCtx)

    # 挖掘路径：prepare 会调 attach_flows；拦截 load 注入 margin
    import factorzen.daily.data.flows as flows_mod
    real_attach = flows_mod.attach_flows

    def _attach_with_margin(d, *, injected=None):
        inj = dict(injected or {})
        inj.setdefault("moneyflow", pl.DataFrame())
        inj.setdefault("hk_hold", pl.DataFrame())
        inj.setdefault("margin_detail", margin)
        return real_attach(d, injected=inj)

    monkeypatch.setattr(flows_mod, "attach_flows", _attach_with_margin)
    # prepare_mining_daily 从 factorzen.daily.data.flows import attach_flows —— 需补丁源模块
    monkeypatch.setattr("factorzen.pipelines.factor_mine.attach_flows", _attach_with_margin, raising=False)

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


# ── C. 叶子注册 / prompt / leaf_health ────────────────────────────────────────


def test_margin_leaves_registered_and_parse():
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


def test_prompt_mentions_margin_family():
    """generation / hypothesis 文案含两融描述（T+1 lag / 单位说明）。"""
    from factorzen.agents.roles.hypothesis import signal_families
    from factorzen.llm.generation import build_agent_messages

    fam = signal_families("ashare")
    assert "两融" in fam or "杠杆" in fam

    sys = build_agent_messages(["ts_mean"], ["close", "margin_ratio"], market="ashare")[0]["content"]
    assert "两融" in sys or "杠杆" in sys
    assert "lag" in sys.lower() or "T+1" in sys or "t+1" in sys.lower()


def test_margin_leaves_pass_through_leaf_health():
    """新叶子自动经 leaf_health 覆盖检查（P1 全局：LEAF_FEATURES → filter）。

    构造 holdout 段 margin_ratio 全 null 的帧 → 该叶应被摘除。
    """
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

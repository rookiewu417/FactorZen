"""任务 H：命名 universe 逐日 PIT membership（消除期末成分幸存偏差）。

全 mock 离线，绝不真调 Tushare。
"""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

# ── 假交易日（1 月 3 个、2 月 3 个，均为工作日风格）──────────────────────
_JAN_DATES = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
_FEB_DATES = [date(2024, 2, 1), date(2024, 2, 2), date(2024, 2, 5)]
_ALL_TRADE_DATES = _JAN_DATES + _FEB_DATES

_JAN_STR = [d.strftime("%Y%m%d") for d in _JAN_DATES]
_FEB_STR = [d.strftime("%Y%m%d") for d in _FEB_DATES]


def _mock_trade_dates(start: str, end: str) -> list[date]:
    """按 [start, end] 截取假交易日。"""
    return [d for d in _ALL_TRADE_DATES if start <= d.strftime("%Y%m%d") <= end]


def _members_by_month(index_code: str, date_str: str) -> list[str]:
    """1 月 {A,B}、2 月 {B,C}；csi800 测试会并 300/500。"""
    ym = date_str[:6]
    if index_code == "000300.SH":
        if ym == "202401":
            return ["A.SZ", "B.SZ"]
        if ym == "202402":
            return ["B.SZ", "C.SZ"]
    if index_code == "000905.SH":
        if ym == "202401":
            return ["D.SZ"]
        if ym == "202402":
            return ["C.SZ", "E.SZ"]
    return []


@pytest.fixture
def patch_calendar_and_members(monkeypatch):
    """mock 交易日历 + 指数成分加载。"""
    monkeypatch.setattr(
        "factorzen.core.calendar.get_trade_dates", _mock_trade_dates
    )
    monkeypatch.setattr(
        "factorzen.core.universe._load_index_members", _members_by_month
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. 月度展开 + membership_hash
# ═══════════════════════════════════════════════════════════════════════════


def test_monthly_expand_and_hash_stability(patch_calendar_and_members):
    from factorzen.core.universe import get_universe_membership, membership_hash

    mem = get_universe_membership("20240102", "20240205", "csi300")
    assert mem.columns == ["trade_date", "ts_code"] or set(mem.columns) >= {
        "trade_date",
        "ts_code",
    }
    assert mem["trade_date"].dtype == pl.Utf8

    jan = mem.filter(pl.col("trade_date").is_in(_JAN_STR))
    feb = mem.filter(pl.col("trade_date").is_in(_FEB_STR))

    assert set(jan["ts_code"].unique().to_list()) == {"A.SZ", "B.SZ"}
    assert set(feb["ts_code"].unique().to_list()) == {"B.SZ", "C.SZ"}
    # 1 月交易日只有 A/B
    for d in _JAN_STR:
        codes = set(jan.filter(pl.col("trade_date") == d)["ts_code"].to_list())
        assert codes == {"A.SZ", "B.SZ"}
    for d in _FEB_STR:
        codes = set(feb.filter(pl.col("trade_date") == d)["ts_code"].to_list())
        assert codes == {"B.SZ", "C.SZ"}

    h1 = membership_hash(mem)
    h2 = membership_hash(mem)
    assert h1 == h2

    mem_shuffled = mem.sample(fraction=1.0, shuffle=True, seed=7)
    assert membership_hash(mem_shuffled) == h1

    mem_other = get_universe_membership("20240102", "20240205", "csi500")
    assert membership_hash(mem_other) != h1


# ═══════════════════════════════════════════════════════════════════════════
# 6a. csi800 并集
# ═══════════════════════════════════════════════════════════════════════════


def test_csi800_is_monthly_union(patch_calendar_and_members):
    from factorzen.core.universe import get_universe_membership

    mem = get_universe_membership("20240102", "20240205", "csi800")
    jan = mem.filter(pl.col("trade_date") == _JAN_STR[0])
    feb = mem.filter(pl.col("trade_date") == _FEB_STR[0])
    # 1 月：300={A,B} ∪ 500={D}
    assert set(jan["ts_code"].to_list()) == {"A.SZ", "B.SZ", "D.SZ"}
    # 2 月：300={B,C} ∪ 500={C,E}
    assert set(feb["ts_code"].to_list()) == {"B.SZ", "C.SZ", "E.SZ"}


# ═══════════════════════════════════════════════════════════════════════════
# 6b. 动态池 ValueError
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "name",
    ["daily_default", "intraday_default", "lft_default", "mft_default"],
)
def test_dynamic_universe_raises(name):
    from factorzen.core.universe import get_universe_membership

    with pytest.raises(ValueError, match=r"不支持|基础池|membership"):
        get_universe_membership("20240102", "20240205", name)


# ═══════════════════════════════════════════════════════════════════════════
# all_a：按 list_date / delist_date 展开
# ═══════════════════════════════════════════════════════════════════════════


def test_all_a_list_delist_expand(monkeypatch):
    from factorzen.core import universe as uni_mod

    # _membership_all_a 内 from calendar import get_trade_dates
    monkeypatch.setattr(
        "factorzen.core.calendar.get_trade_dates", _mock_trade_dates
    )
    basic = pl.DataFrame(
        {
            "ts_code": ["X.SZ", "Y.SZ", "Z.SZ"],
            "symbol": ["X", "Y", "Z"],
            "name": ["x", "y", "z"],
            "area": ["深圳"] * 3,
            "industry": ["银行"] * 3,
            "market": ["主板"] * 3,
            "list_date": [
                date(2020, 1, 1),
                date(2024, 2, 1),  # 2 月才上市
                date(2019, 1, 1),
            ],
            "delist_date": [
                None,
                None,
                date(2024, 1, 4),  # 1 月末退市（1/4 当天已退）
            ],
        }
    )
    monkeypatch.setattr(uni_mod, "get_stock_basic", lambda: basic)

    mem = uni_mod.get_universe_membership("20240102", "20240205", "all_a")
    x_dates = set(mem.filter(pl.col("ts_code") == "X.SZ")["trade_date"].to_list())
    y_dates = set(mem.filter(pl.col("ts_code") == "Y.SZ")["trade_date"].to_list())
    z_dates = set(mem.filter(pl.col("ts_code") == "Z.SZ")["trade_date"].to_list())

    assert x_dates == set(_JAN_STR + _FEB_STR)
    assert y_dates == set(_FEB_STR)  # 2 月 1 日起
    # delist_date 严格大于：2024-01-04 当天不在
    assert "20240104" not in z_dates
    assert "20240102" in z_dates and "20240103" in z_dates
    assert not any(d.startswith("202402") for d in z_dates)


# ═══════════════════════════════════════════════════════════════════════════
# prepare_mining_daily：调出 / 调入 / 连续性 / fail-closed
# ═══════════════════════════════════════════════════════════════════════════


def _synthetic_daily_frame() -> pl.DataFrame:
    """并集 {A,B,C} × 全部交易日（含预热日 2023-12-29）。"""
    warmup = [date(2023, 12, 29)]
    days = warmup + _ALL_TRADE_DATES
    codes = ["A.SZ", "B.SZ", "C.SZ"]
    rows = []
    for c in codes:
        for d in days:
            rows.append(
                {
                    "trade_date": d,
                    "ts_code": c,
                    "close": 10.0,
                    "close_adj": 10.0,
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "vol": 1e5,
                    "amount": 1e6,
                }
            )
    return pl.DataFrame(rows)


def _patch_prepare_stack(monkeypatch, daily: pl.DataFrame, *, end_universe=None):
    """mock FactorDataContext + attach_* + get_universe(期末) + calendar/members。"""
    import factorzen.daily.data.context as ctx_mod
    import factorzen.pipelines.factor_mine as fm

    monkeypatch.setattr(
        "factorzen.core.calendar.get_trade_dates", _mock_trade_dates
    )
    monkeypatch.setattr(
        "factorzen.core.universe._load_index_members", _members_by_month
    )

    class _FakeCtx:
        def __init__(self, **kw):
            self.kw = kw
            _FakeCtx.last_kw = kw

        @property
        def daily(self):
            uni = self.kw.get("universe")
            df = daily
            # 与 FactorDataContext 一致：空 list 假值 → 不过滤（all_a 空池=全市场）
            if uni:
                df = df.filter(pl.col("ts_code").is_in(list(uni)))
            return df.lazy()

        @property
        def daily_basic(self):
            return pl.DataFrame(
                {
                    "trade_date": pl.Series([], dtype=pl.Date),
                    "ts_code": pl.Series([], dtype=pl.Utf8),
                }
            ).lazy()

    _FakeCtx.last_kw = {}
    monkeypatch.setattr(ctx_mod, "FactorDataContext", _FakeCtx)

    # attach_* 在函数内 import，补丁源模块
    monkeypatch.setattr(
        "factorzen.daily.data.pit.attach_fundamentals", lambda d: d
    )
    monkeypatch.setattr("factorzen.daily.data.pit.attach_holders", lambda d: d)
    monkeypatch.setattr("factorzen.daily.data.flows.attach_flows", lambda d: d)

    if end_universe is not None:
        def _fake_get_universe(date_str, universe_name="all_a"):
            return pl.DataFrame({"ts_code": end_universe})

        monkeypatch.setattr(
            "factorzen.core.universe.get_universe", _fake_get_universe
        )

    return fm, _FakeCtx


def test_delist_from_index_keeps_jan_rows(monkeypatch):
    """调出反例：A 1 月在成分、2 月调出；期末快照=2 月不含 A。

    修复后：A 的 1 月行 in_universe=True，2 月 False；行仍保留。
    """
    daily = _synthetic_daily_frame()
    fm, FakeCtx = _patch_prepare_stack(
        monkeypatch, daily, end_universe=["B.SZ", "C.SZ"]
    )

    out = fm.prepare_mining_daily("20240102", "20240205", universe="csi300")

    assert "in_universe" in out.columns
    # 并集 = {A,B,C}（窗口内曾在成分内）
    assert set(FakeCtx.last_kw["universe"]) == {"A.SZ", "B.SZ", "C.SZ"}

    a = out.filter(pl.col("ts_code") == "A.SZ")
    a_jan = a.filter(pl.col("trade_date").is_in(_JAN_DATES))
    a_feb = a.filter(pl.col("trade_date").is_in(_FEB_DATES))
    assert a_jan.height == 3
    assert a_jan["in_universe"].all()
    assert a_feb.height == 3
    assert not a_feb["in_universe"].any()

    # 预热行保留且 in_universe=False
    warm = a.filter(pl.col("trade_date") == date(2023, 12, 29))
    assert warm.height == 1
    assert not warm["in_universe"].item()


def test_new_entrant_excluded_in_jan(monkeypatch):
    """调入反例：C 2 月才调入 → 1 月 in_universe=False。"""
    daily = _synthetic_daily_frame()
    fm, _ = _patch_prepare_stack(monkeypatch, daily)

    out = fm.prepare_mining_daily("20240102", "20240205", universe="csi300")
    c_jan = out.filter(
        (pl.col("ts_code") == "C.SZ") & pl.col("trade_date").is_in(_JAN_DATES)
    )
    c_feb = out.filter(
        (pl.col("ts_code") == "C.SZ") & pl.col("trade_date").is_in(_FEB_DATES)
    )
    assert not c_jan["in_universe"].any()
    assert c_feb["in_universe"].all()


def test_continuity_rows_preserved(monkeypatch):
    """并集股票的原始行（含非成分日/预热段）全部保留。"""
    daily = _synthetic_daily_frame()
    n_raw = daily.height  # 3 股 × 7 日
    fm, _ = _patch_prepare_stack(monkeypatch, daily)

    out = fm.prepare_mining_daily("20240102", "20240205", universe="csi300")
    assert out.height == n_raw
    # 仅标记不同
    assert out["in_universe"].dtype == pl.Boolean
    assert 0 < out["in_universe"].sum() < out.height


def test_membership_failure_fails_closed(monkeypatch):
    """命名指数 membership 构造抛异常 → fail closed，拒绝静态回退。"""
    daily = _synthetic_daily_frame()
    fm, _ = _patch_prepare_stack(
        monkeypatch, daily, end_universe=["B.SZ", "C.SZ"]
    )

    def _boom(*a, **k):
        raise RuntimeError("mock membership failure")

    monkeypatch.setattr(
        "factorzen.core.universe.get_universe_membership", _boom
    )

    with pytest.raises(ValueError, match=r"PIT membership|look-ahead|拒绝回退"):
        fm.prepare_mining_daily("20240102", "20240205", universe="csi300")


def test_membership_empty_named_index_fails_closed(monkeypatch):
    """命名指数 membership 返回空 → fail closed，拒绝 as-of 回退。"""
    daily = _synthetic_daily_frame()
    fm, _ = _patch_prepare_stack(
        monkeypatch, daily, end_universe=["B.SZ", "C.SZ"]
    )

    monkeypatch.setattr(
        "factorzen.core.universe.get_universe_membership",
        lambda *a, **k: pl.DataFrame(
            {
                "trade_date": pl.Series([], dtype=pl.Utf8),
                "ts_code": pl.Series([], dtype=pl.Utf8),
            }
        ),
    )

    with pytest.raises(ValueError, match=r"空|未回补|拒绝|as-of|PIT"):
        fm.prepare_mining_daily("20240102", "20240205", universe="csi300")


def test_all_a_empty_membership_still_succeeds(monkeypatch):
    """all_a 空 membership 视为全市场，不抛错、不走静态回退。"""
    daily = _synthetic_daily_frame()
    fm, FakeCtx = _patch_prepare_stack(monkeypatch, daily)

    monkeypatch.setattr(
        "factorzen.core.universe.get_universe_membership",
        lambda *a, **k: pl.DataFrame(
            {
                "trade_date": pl.Series([], dtype=pl.Utf8),
                "ts_code": pl.Series([], dtype=pl.Utf8),
            }
        ),
    )

    out_meta: dict = {}
    out = fm.prepare_mining_daily(
        "20240102", "20240205", universe="all_a", out_meta=out_meta
    )
    # all_a 空池：uni=[] 对 FactorDataContext 即不过滤；仍标 pit 并 attach in_universe
    assert FakeCtx.last_kw["universe"] == []
    assert out_meta["membership_mode"] == "pit"
    assert "in_universe" in out.columns
    assert out.height == daily.height
    # 空 membership → 全部 in_universe=False（attach 语义）
    assert not out["in_universe"].any()


# ═══════════════════════════════════════════════════════════════════════════
# 评估截面过滤（evaluation + mining_session）
# ═══════════════════════════════════════════════════════════════════════════


def test_eval_frame_filters_in_universe_false():
    """评估截面不含 in_universe=False 行。"""
    from factorzen.agents.evaluation import _factor_df_from_prepped
    from factorzen.discovery.expression import parse_expr

    days = _JAN_DATES + _FEB_DATES
    rows = []
    for c, in_u_jan in [("A.SZ", True), ("C.SZ", False)]:
        for d in days:
            in_u = in_u_jan if d in _JAN_DATES else (c == "C.SZ")
            rows.append(
                {
                    "trade_date": d,
                    "ts_code": c,
                    "close": 10.0,
                    "close_adj": 10.0,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "vol": 1e5,
                    "amount": 1e6,
                    "in_universe": in_u if d in _JAN_DATES else (c != "A.SZ"),
                }
            )
    # 简化：A 全程 True，C 全程 False
    prepped = pl.DataFrame(
        {
            "trade_date": days * 2,
            "ts_code": ["A.SZ"] * len(days) + ["C.SZ"] * len(days),
            "close": [10.0] * (len(days) * 2),
            "close_adj": [10.0] * (len(days) * 2),
            "open": [10.0] * (len(days) * 2),
            "high": [11.0] * (len(days) * 2),
            "low": [9.0] * (len(days) * 2),
            "vol": [1e5] * (len(days) * 2),
            "amount": [1e6] * (len(days) * 2),
            "in_universe": [True] * len(days) + [False] * len(days),
        }
    )
    node = parse_expr("close")
    fdf = _factor_df_from_prepped(node, prepped, eval_start=date(2024, 1, 2))
    assert set(fdf["ts_code"].unique().to_list()) == {"A.SZ"}
    assert "in_universe" not in fdf.columns


def test_eval_frame_no_in_universe_column_zero_regression():
    """无 in_universe 列时评估帧不过滤（零回归）。"""
    from factorzen.agents.evaluation import _factor_df_from_prepped
    from factorzen.discovery.expression import parse_expr

    days = _JAN_DATES
    prepped = pl.DataFrame(
        {
            "trade_date": days * 2,
            "ts_code": ["A.SZ"] * len(days) + ["C.SZ"] * len(days),
            "close": [10.0] * (len(days) * 2),
            "close_adj": [10.0] * (len(days) * 2),
            "open": [10.0] * (len(days) * 2),
            "high": [11.0] * (len(days) * 2),
            "low": [9.0] * (len(days) * 2),
            "vol": [1e5] * (len(days) * 2),
            "amount": [1e6] * (len(days) * 2),
        }
    )
    node = parse_expr("close")
    fdf = _factor_df_from_prepped(node, prepped, eval_start=date(2024, 1, 2))
    assert set(fdf["ts_code"].unique().to_list()) == {"A.SZ", "C.SZ"}


def test_m1_factor_values_filters_in_universe():
    """M1 路径 _factor_values 同样过滤 in_universe=False。"""
    from factorzen.discovery.expression import parse_expr
    from factorzen.discovery.mining_session import _factor_values

    days = _JAN_DATES
    daily = pl.DataFrame(
        {
            "trade_date": days * 2,
            "ts_code": ["A.SZ"] * len(days) + ["C.SZ"] * len(days),
            "close": [10.0] * (len(days) * 2),
            "close_adj": [10.0] * (len(days) * 2),
            "open": [10.0] * (len(days) * 2),
            "high": [11.0] * (len(days) * 2),
            "low": [9.0] * (len(days) * 2),
            "vol": [1e5] * (len(days) * 2),
            "amount": [1e6] * (len(days) * 2),
            "in_universe": [True] * len(days) + [False] * len(days),
        }
    )
    node = parse_expr("close")
    fdf = _factor_values(node, daily, eval_start="20240102")
    assert set(fdf["ts_code"].unique().to_list()) == {"A.SZ"}

"""基本面叶子 PIT 对齐:确保 attach_fundamentals 无未来函数(铁律#1)。"""
import datetime as dt

import polars as pl

from factorzen.daily.data.pit import attach_fundamentals


def _fina() -> pl.DataFrame:
    """两份报告:Q1(end 0331)0420 公告、Q2(end 0630)0815 公告——真实数据 ann/end 为 String。

    含全套质量/成长字段,验证扩充后的叶子一并 PIT 对齐。
    """
    return pl.DataFrame({
        "ts_code": ["000001.SZ", "000001.SZ"],
        "end_date": ["20200331", "20200630"],
        "ann_date": ["20200420", "20200815"],
        "roe": [10.0, 12.0], "roa": [1.0, 1.2],
        "grossprofit_margin": [40.0, 41.0], "netprofit_margin": [20.0, 21.0],
        "debt_to_assets": [50.0, 51.0],
        "or_yoy": [8.0, 9.0], "netprofit_yoy": [15.0, 16.0], "assets_yoy": [5.0, 6.0],
    })


def _daily(dates: list[str]) -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": [dt.datetime.strptime(d, "%Y%m%d").date() for d in dates],
        "ts_code": ["000001.SZ"] * len(dates),
        "close": [10.0] * len(dates),
    })


def test_no_future_leak_before_announcement():
    """t 日在 Q1 公告(0420)之前 → roe 必须是 null,绝不能把 0420 才公告的报告泄漏回 0410。"""
    out = attach_fundamentals(_daily(["20200410"]), fina_df=_fina())
    row = out.filter(pl.col("trade_date") == dt.date(2020, 4, 10))
    assert row["roe"][0] is None, "Q1 报告在公告日前泄漏 → 未来函数!"
    assert row["assets_yoy"][0] is None


def test_uses_latest_announced_report():
    """公告后取最新已公告报告:0420~0814 用 Q1(10.0);0815 起用 Q2(end 更大,12.0)。"""
    out = attach_fundamentals(_daily(["20200410", "20200501", "20200820"]), fina_df=_fina())
    by_date = {r["trade_date"]: r["roe"] for r in out.iter_rows(named=True)}
    assert by_date[dt.date(2020, 4, 10)] is None       # 公告前
    assert by_date[dt.date(2020, 5, 1)] == 10.0         # Q1 已公告
    assert by_date[dt.date(2020, 8, 20)] == 12.0        # Q2 已公告(end_date 更大)


def test_missing_finance_returns_daily_with_null_cols():
    """无 finance 数据(空帧)→ 原样返回但补齐 roe/assets_yoy 为 null(表达式引用不崩)。"""
    out = attach_fundamentals(_daily(["20200501"]), fina_df=pl.DataFrame())
    assert "roe" in out.columns and "assets_yoy" in out.columns
    assert out["roe"][0] is None


def test_expanded_fields_pit_aligned():
    """扩充的质量/成长字段(毛利率/营收增速等)与 roe 同套 PIT 对齐,公告后取最新报告。"""
    out = attach_fundamentals(_daily(["20200410", "20200820"]), fina_df=_fina())
    pre = out.filter(pl.col("trade_date") == dt.date(2020, 4, 10))
    post = out.filter(pl.col("trade_date") == dt.date(2020, 8, 20))
    for col in ("grossprofit_margin", "or_yoy", "netprofit_yoy", "debt_to_assets", "roa"):
        assert pre[col][0] is None, f"{col} 公告前泄漏 → 未来函数!"
    assert post["grossprofit_margin"][0] == 41.0   # Q2
    assert post["or_yoy"][0] == 9.0


def test_all_fundamental_leaves_registered_and_parse():
    """全套质量/成长叶子已注册且可解析(否则 LLM/搜索碰不到、prompt 广告了却用不了)。"""
    from factorzen.discovery.expression import feature_names, parse_expr
    from factorzen.discovery.operators import FUNDAMENTAL_FEATURES, LEAF_FEATURES
    expected = {"roe", "roa", "grossprofit_margin", "netprofit_margin", "debt_to_assets",
                "or_yoy", "netprofit_yoy", "assets_yoy"}
    assert expected <= FUNDAMENTAL_FEATURES
    for leaf in expected:
        assert leaf in LEAF_FEATURES, f"{leaf} 未注册为叶子"
        assert leaf in feature_names(parse_expr(f"rank({leaf})")), f"{leaf} 解析不出"

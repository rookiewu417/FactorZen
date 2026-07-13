"""资金流/北向日频叶子:attach_flows 按交易日 join,叶子注册,双路径门。"""
import datetime as dt

import polars as pl

from factorzen.daily.data.flows import attach_flows


def _daily(dates: list[str], code="000001.SZ") -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": [dt.datetime.strptime(d, "%Y%m%d").date() for d in dates],
        "ts_code": [code] * len(dates),
        "close": [10.0] * len(dates),
    })


def _mf() -> pl.DataFrame:
    return pl.DataFrame({
        "ts_code": ["000001.SZ", "000001.SZ"],
        "trade_date": [dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
        "net_mf_amount": [1234.5, -678.9],
    })


def _hk() -> pl.DataFrame:
    return pl.DataFrame({
        "ts_code": ["000001.SZ", "000001.SZ"],
        "trade_date": [dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
        "ratio": [3.5, 3.6],
    })


def test_flows_join_by_trade_date():
    """资金流/北向按 (trade_date, ts_code) 逐日 join;ratio 重命名为 north_ratio。"""
    out = attach_flows(_daily(["20240102", "20240103"]),
                       injected={"moneyflow": _mf(), "hk_hold": _hk()})
    by = {r["trade_date"]: r for r in out.iter_rows(named=True)}
    assert by[dt.date(2024, 1, 2)]["net_mf_amount"] == 1234.5
    assert by[dt.date(2024, 1, 3)]["net_mf_amount"] == -678.9
    assert by[dt.date(2024, 1, 2)]["north_ratio"] == 3.5      # ratio → north_ratio
    assert "ratio" not in out.columns


def test_missing_dates_get_null():
    """flow 数据缺某天 → 该天叶子为 null(不崩、不错配到别的日子)。"""
    out = attach_flows(_daily(["20240102", "20240110"]),
                       injected={"moneyflow": _mf(), "hk_hold": _hk()})
    by = {r["trade_date"]: r for r in out.iter_rows(named=True)}
    assert by[dt.date(2024, 1, 10)]["net_mf_amount"] is None   # 无数据日
    assert by[dt.date(2024, 1, 2)]["net_mf_amount"] == 1234.5


def test_missing_source_returns_null_cols():
    """无 flow 数据(注入空帧)→ 原样返回但补 net_mf_amount/north_ratio 为 null。

    注入空帧而非 {},避免回落读盘(真实 moneyflow 已缓存时 {} 会拿到真数据,测试就不 hermetic)。
    """
    out = attach_flows(_daily(["20240102"]),
                       injected={"moneyflow": pl.DataFrame(), "hk_hold": pl.DataFrame()})
    assert "net_mf_amount" in out.columns and "north_ratio" in out.columns
    assert out["net_mf_amount"][0] is None


def test_flow_leaves_registered_and_gate():
    """flow 叶子已注册、可解析,且触发 FLOW_FEATURES 门(物化路径会 attach)。"""
    from factorzen.discovery.expression import feature_names, parse_expr
    from factorzen.discovery.operators import FLOW_FEATURES, LEAF_FEATURES
    assert "net_mf_amount" in FLOW_FEATURES and "north_ratio" in FLOW_FEATURES
    for leaf in FLOW_FEATURES:
        assert leaf in LEAF_FEATURES
        feats = feature_names(parse_expr(f"rank({leaf})"))
        assert leaf in feats
        assert feats & FLOW_FEATURES   # 触发物化路径 attach 门

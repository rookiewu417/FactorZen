"""资金流/北向日频叶子:attach_flows 按交易日 join,叶子注册,双路径门。"""
import datetime as dt

import polars as pl
import pytest

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


def test_duplicate_source_rows_do_not_multiply_daily():
    """源帧含重复 (trade_date, ts_code) 时，left-join 不得成倍放行 daily。

    **实证根因（2026-07-19）**：`hk_hold` 在 2026-06-30 单日返回 4200 行/4061 只
    （正常日 ~950 只北向标的），139 只带重复记录；`HK_HOLD_COLS` 又把 `exchange`
    等区分列裁掉，事后无法分辨保留哪条（股本反推可证每对里恰有一条 ratio 与
    vol/total_share 不自洽）。这些重复行流进因子面板后，在组合层
    `_zscore_and_merge` 的链式 full join 下按 重复数^因子数 爆炸，实测打满 23GB。
    """
    daily = _daily(["20240102", "20240103"])
    hk_dup = pl.DataFrame({
        "ts_code": ["000001.SZ"] * 3,
        "trade_date": [dt.date(2024, 1, 2)] * 2 + [dt.date(2024, 1, 3)],
        "ratio": [1.2, 0.33, 0.5],   # 同键两条不同值：正是生产观测到的形态
        "vol": [102592076, 4970305, 1000],
    })
    with pytest.warns(UserWarning, match="重复"):
        out = attach_flows(daily, injected={"hk_hold": hk_dup})

    assert out.height == daily.height, f"daily 被放大到 {out.height} 行"
    assert out.select(["trade_date", "ts_code"]).unique().height == out.height
    # keep="first" → 取到 1.2 那条（确定性，不随运行变化）
    got = out.filter(pl.col("trade_date") == dt.date(2024, 1, 2))["north_ratio"][0]
    assert got == 1.2


def test_margin_does_not_square_amplify_dirty_daily():
    """daily 已被上游污染成 2 行/股时，_attach_margin 不得再平方放大成 4 行/股。

    生产链实测：hk_hold 重复让 daily 变 2 行/股 → `_attach_margin` 先从 daily 取
    同日分母（继承 2 行）再 join 回 daily ⇒ **2×2=4 行/股**。探针实测
    318 行 → 320 行（hk_hold）→ 324 行（margin），目标股票 1→2→4 行。
    """
    from factorzen.daily.data.flows import _attach_margin

    dirty = pl.DataFrame({
        "trade_date": [dt.date(2024, 1, 2)] * 2,
        "ts_code": ["000001.SZ"] * 2,
        "circ_mv": [1000.0, 1000.0],
        "amount": [500.0, 500.0],
    })
    margin = pl.DataFrame({
        "ts_code": ["000001.SZ"],
        "trade_date": [dt.date(2024, 1, 2)],
        "rzye": [1.0e8],
        "rzmre": [1.0e7],
        "rqyl": [1000.0],
    })
    with pytest.warns(UserWarning, match="重复"):
        out = _attach_margin(dirty, injected={"margin_detail": margin})
    # 入参本就脏(2 行)——契约是**不再放大**，而非替上游清洗
    assert out.height == 2, f"margin 把 2 行放大成了 {out.height} 行"

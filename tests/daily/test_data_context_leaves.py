"""
test_daily_data_context.py：daily/data/context.py 的离线单测。
test_flows_attach.py：资金流/北向日频叶子:attach_flows 按交易日 join,叶子注册,双路径门。
test_evaluation_contracts.py：评估入口的数据契约(列校验 fail-fast)测试。
test_factor_required_data_declares_daily.py：泛化回归守卫：所有注册因子的 required_data 必含 "daily"。
"""

from __future__ import annotations

import datetime as dt
from datetime import date

import polars as pl
import pytest

import factorzen.builtin_factors  # noqa: F401  触发因子注册
from factorzen.daily.data import context as ctx_mod
from factorzen.daily.data.context import FactorDataContext
from factorzen.daily.data.flows import attach_flows
from factorzen.daily.evaluation.backtest import _prepare_factor_df, _prepare_price_df
from factorzen.daily.evaluation.turnover import compute_turnover
from factorzen.daily.factors.registry import get_factor, list_factors


# ==== 来自 test_daily_data_context.py ====
def _daily_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_code": ["A", "A", "B"],
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 2)],
            "close": [10.0, 11.0, 20.0],
            "open": [9.0, 10.0, 19.0],
            "high": [10.5, 11.5, 20.5],
            "low": [8.5, 9.5, 18.5],
            "vol": [100.0, 200.0, 300.0],
        }
    )


def _adj_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_code": ["A", "A", "B"],
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 2)],
            "adj_factor": [2.0, 2.0, 1.0],
        }
    )


def _basic_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_code": ["A", "A", "B"],
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 2)],
            "pe": [15.0, 16.0, 30.0],
        }
    )


@pytest.fixture
def patched(monkeypatch):
    """重定向 prev_trade_date 与 load_parquet，提供 daily/daily_basic/adj 合成数据。"""
    monkeypatch.setattr(ctx_mod, "prev_trade_date", lambda d, n: date(2023, 12, 1))

    def fake_load(category, start=None, end=None):
        if category == "daily":
            return _daily_df().lazy()
        if category == "daily_basic":
            return _basic_df().lazy()
        if category == "adj_factor":
            return _adj_df().lazy()
        raise ValueError(f"未知 category: {category}")

    monkeypatch.setattr(ctx_mod, "load_parquet", fake_load)
    return monkeypatch


# ══════════════════════════════════════════════════════════
# expanded_start
# ══════════════════════════════════════════════════════════


def test_expanded_start_uses_prev_trade_date(patched):
    ctx = FactorDataContext(start="20240102", end="20240103", lookback_days=20)
    assert ctx.expanded_start == "20231201"


# ══════════════════════════════════════════════════════════
# daily 属性：复权 join / 回退 / universe / 缓存 / 未声明
# ══════════════════════════════════════════════════════════


def test_daily_applies_adj_factor(patched):
    ctx = FactorDataContext(start="20240102", end="20240103")
    df = ctx.daily.collect().sort(["ts_code", "trade_date"])
    # A 在 2024-01-02：close 10 × adj 2.0 = 20.0
    row = df.filter((pl.col("ts_code") == "A") & (pl.col("trade_date") == date(2024, 1, 2)))
    assert row["close_adj"].item() == 20.0
    assert "adj_factor" not in df.columns  # join 后已 drop


def test_daily_fallback_when_adj_missing(patched, monkeypatch):
    """adj_factor 未落盘（load 抛异常）时，close_adj 回退为原始价格。"""

    def fake_load(category, start=None, end=None):
        if category == "daily":
            return _daily_df().lazy()
        raise FileNotFoundError("adj_factor 未落盘")

    monkeypatch.setattr(ctx_mod, "load_parquet", fake_load)
    ctx = FactorDataContext(start="20240102", end="20240103")
    df = ctx.daily.collect()
    row = df.filter((pl.col("ts_code") == "A") & (pl.col("trade_date") == date(2024, 1, 2)))
    assert row["close_adj"].item() == 10.0  # 回退原值


def test_daily_filters_universe(patched):
    ctx = FactorDataContext(start="20240102", end="20240103", universe=["A"])
    df = ctx.daily.collect()
    assert set(df["ts_code"].to_list()) == {"A"}


def test_daily_not_declared_raises(patched):
    ctx = FactorDataContext(start="20240102", end="20240103", required_data=["daily_basic"])
    with pytest.raises(ValueError, match="daily data not declared"):
        _ = ctx.daily


def test_daily_lazy_cached(patched):
    ctx = FactorDataContext(start="20240102", end="20240103")
    assert ctx.daily is ctx.daily  # 第二次命中缓存，同一对象


# ══════════════════════════════════════════════════════════
# daily_basic
# ══════════════════════════════════════════════════════════


def test_daily_basic_loads(patched):
    ctx = FactorDataContext(
        start="20240102", end="20240103", required_data=["daily", "daily_basic"]
    )
    df = ctx.daily_basic.collect()
    assert "pe" in df.columns
    assert df.height == 3


def test_daily_basic_not_declared_raises(patched):
    ctx = FactorDataContext(start="20240102", end="20240103", required_data=["daily"])
    with pytest.raises(ValueError, match="daily_basic data not declared"):
        _ = ctx.daily_basic


def test_daily_basic_filters_universe(patched):
    ctx = FactorDataContext(
        start="20240102",
        end="20240103",
        required_data=["daily_basic"],
        universe=["B"],
    )
    df = ctx.daily_basic.collect()
    assert set(df["ts_code"].to_list()) == {"B"}


# ══════════════════════════════════════════════════════════
# snapshot_dates 三种模式
# ══════════════════════════════════════════════════════════


def test_daily_snapshot_downsample_suite(patched, monkeypatch):
    """test_snapshot_dates_daily_mode；test_snapshot_dates_weekly_mode；test_snapshot_dates_monthly_mode；test_weekly_downsamples_to_snapshot；test_monthly_downsamples_to_snapshot"""
    # -- 原 test_snapshot_dates_daily_mode --
    def _section_0_test_snapshot_dates_daily_mode(patched, mp):
        mp.setattr(
            "factorzen.core.calendar.get_trade_dates",
            lambda s, e: [date(2024, 1, 2), date(2024, 1, 3)],
        )
        ctx = FactorDataContext(start="20240102", end="20240103", snapshot_mode="daily")
        assert ctx.snapshot_dates == [date(2024, 1, 2), date(2024, 1, 3)]

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_snapshot_dates_daily_mode(patched, mp)

    # -- 原 test_snapshot_dates_weekly_mode --
    def _section_1_test_snapshot_dates_weekly_mode(patched, mp):
        mp.setattr(
            "factorzen.core.calendar.get_weekly_snapshot_dates",
            lambda s, e: [date(2024, 1, 3)],
        )
        ctx = FactorDataContext(start="20240102", end="20240103", snapshot_mode="weekly")
        assert ctx.snapshot_dates == [date(2024, 1, 3)]

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_snapshot_dates_weekly_mode(patched, mp)

    # -- 原 test_snapshot_dates_monthly_mode --
    def _section_2_test_snapshot_dates_monthly_mode(patched, mp):
        mp.setattr(
            "factorzen.core.calendar.get_monthly_snapshot_dates",
            lambda s, e: [date(2024, 1, 3)],
        )
        ctx = FactorDataContext(start="20240102", end="20240103", snapshot_mode="monthly")
        assert ctx.snapshot_dates == [date(2024, 1, 3)]

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_snapshot_dates_monthly_mode(patched, mp)

    # -- 原 test_weekly_downsamples_to_snapshot --
    def _section_3_test_weekly_downsamples_to_snapshot(patched, mp):
        mp.setattr(
            "factorzen.core.calendar.get_weekly_snapshot_dates",
            lambda s, e: [date(2024, 1, 3)],
        )
        ctx = FactorDataContext(start="20240102", end="20240103", snapshot_mode="weekly")
        df = ctx.weekly.collect()
        assert df["trade_date"].unique().to_list() == [date(2024, 1, 3)]
        assert ctx.weekly is ctx.weekly  # 缓存

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_weekly_downsamples_to_snapshot(patched, mp)

    # -- 原 test_monthly_downsamples_to_snapshot --
    def _section_4_test_monthly_downsamples_to_snapshot(patched, mp):
        mp.setattr(
            "factorzen.core.calendar.get_monthly_snapshot_dates",
            lambda s, e: [date(2024, 1, 2)],
        )
        ctx = FactorDataContext(start="20240102", end="20240103", snapshot_mode="monthly")
        df = ctx.monthly.collect()
        assert df["trade_date"].unique().to_list() == [date(2024, 1, 2)]

    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_monthly_downsamples_to_snapshot(patched, mp)


# ══════════════════════════════════════════════════════════
# 下采样属性：weekly / monthly / *_basic
# ══════════════════════════════════════════════════════════


def test_weekly_basic_downsamples(patched, monkeypatch):
    monkeypatch.setattr(
        "factorzen.core.calendar.get_weekly_snapshot_dates",
        lambda s, e: [date(2024, 1, 3)],
    )
    ctx = FactorDataContext(
        start="20240102",
        end="20240103",
        required_data=["daily_basic"],
        snapshot_mode="weekly",
    )
    df = ctx.weekly_basic.collect()
    assert df["trade_date"].unique().to_list() == [date(2024, 1, 3)]


def test_monthly_basic_downsamples(patched, monkeypatch):
    monkeypatch.setattr(
        "factorzen.core.calendar.get_monthly_snapshot_dates",
        lambda s, e: [date(2024, 1, 2)],
    )
    ctx = FactorDataContext(
        start="20240102",
        end="20240103",
        required_data=["daily_basic"],
        snapshot_mode="monthly",
    )
    df = ctx.monthly_basic.collect()
    assert df["trade_date"].unique().to_list() == [date(2024, 1, 2)]


# ══════════════════════════════════════════════════════════
# load_all
# ══════════════════════════════════════════════════════════


def test_load_all_daily_and_basic(patched, monkeypatch):
    monkeypatch.setattr(
        "factorzen.core.calendar.get_weekly_snapshot_dates",
        lambda s, e: [date(2024, 1, 3)],
    )
    ctx = FactorDataContext(
        start="20240102",
        end="20240103",
        required_data=["daily", "daily_basic"],
        snapshot_mode="weekly",
    )
    ctx.load_all()
    # 所有惰性缓存均已填充
    assert ctx._daily is not None
    assert ctx._daily_basic is not None
    assert ctx._weekly_snapshot is not None


def test_load_all_monthly_mode(patched, monkeypatch):
    monkeypatch.setattr(
        "factorzen.core.calendar.get_monthly_snapshot_dates",
        lambda s, e: [date(2024, 1, 2)],
    )
    ctx = FactorDataContext(
        start="20240102", end="20240103", required_data=["daily"], snapshot_mode="monthly"
    )
    ctx.load_all()
    assert ctx._monthly_snapshot is not None

# ==== 来自 test_flows_attach.py ====
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


def test_flows_attach_suite():
    """资金流/北向按 (trade_date, ts_code) 逐日 join;ratio 重命名为 north_ratio。；flow 数据缺某天 → 该天叶子为 null(不崩、不错配到别的日子)。；无 flow 数据(注入空帧)→ 原样返回但补 net_mf_amount/north_ratio 为 null。；flow 叶子已注册、可解析,且触发 FLOW_FEATURES 门(物化路径会 attach)。；源帧含重复 (trade_date, ts_code) 时，left-join 不得成倍放行 daily。；daily 已被上游污染成 2 行/股时，_attach_margin 不得再平方放大成 4 行/股。"""
    # -- 原 test_flows_join_by_trade_date --
    def _section_0_test_flows_join_by_trade_date():
        out = attach_flows(_daily(["20240102", "20240103"]),
                           injected={"moneyflow": _mf(), "hk_hold": _hk()})
        by = {r["trade_date"]: r for r in out.iter_rows(named=True)}
        assert by[dt.date(2024, 1, 2)]["net_mf_amount"] == 1234.5
        assert by[dt.date(2024, 1, 3)]["net_mf_amount"] == -678.9
        assert by[dt.date(2024, 1, 2)]["north_ratio"] == 3.5      # ratio → north_ratio
        assert "ratio" not in out.columns

    _section_0_test_flows_join_by_trade_date()

    # -- 原 test_missing_dates_get_null --
    def _section_1_test_missing_dates_get_null():
        out = attach_flows(_daily(["20240102", "20240110"]),
                           injected={"moneyflow": _mf(), "hk_hold": _hk()})
        by = {r["trade_date"]: r for r in out.iter_rows(named=True)}
        assert by[dt.date(2024, 1, 10)]["net_mf_amount"] is None   # 无数据日
        assert by[dt.date(2024, 1, 2)]["net_mf_amount"] == 1234.5

    _section_1_test_missing_dates_get_null()

    # -- 原 test_missing_source_returns_null_cols --
    def _section_2_test_missing_source_returns_null_cols():
        out = attach_flows(_daily(["20240102"]),
                           injected={"moneyflow": pl.DataFrame(), "hk_hold": pl.DataFrame()})
        assert "net_mf_amount" in out.columns and "north_ratio" in out.columns
        assert out["net_mf_amount"][0] is None

    _section_2_test_missing_source_returns_null_cols()

    # -- 原 test_flow_leaves_registered_and_gate --
    def _section_3_test_flow_leaves_registered_and_gate():
        from factorzen.discovery.expression import feature_names, parse_expr
        from factorzen.discovery.operators import FLOW_FEATURES, LEAF_FEATURES
        assert "net_mf_amount" in FLOW_FEATURES and "north_ratio" in FLOW_FEATURES
        for leaf in FLOW_FEATURES:
            assert leaf in LEAF_FEATURES
            feats = feature_names(parse_expr(f"rank({leaf})"))
            assert leaf in feats
            assert feats & FLOW_FEATURES   # 触发物化路径 attach 门

    _section_3_test_flow_leaves_registered_and_gate()

    # -- 原 test_duplicate_source_rows_do_not_multiply_daily --
    def _section_4_test_duplicate_source_rows_do_not_multiply_daily():
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

    _section_4_test_duplicate_source_rows_do_not_multiply_daily()

    # -- 原 test_margin_does_not_square_amplify_dirty_daily --
    def _section_5_test_margin_does_not_square_amplify_dirty_daily():
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

    _section_5_test_margin_does_not_square_amplify_dirty_daily()


# ==== 来自 test_evaluation_contracts.py ====
def test_evaluation_contracts_suite():
    """test_compute_turnover_raises_on_missing_factor_column；test_compute_turnover_raises_on_missing_key_columns；test_prepare_factor_df_error_lists_actual_columns；test_prepare_price_df_error_lists_actual_columns；test_every_registered_factor_declares_daily；这些估值/规模/流动性因子的 compute 确实读 daily_basic，故 daily 与 daily_basic 都要有。"""
    # -- 原 test_compute_turnover_raises_on_missing_factor_column --
    def _section_0_test_compute_turnover_raises_on_missing_factor_column():
        df = pl.DataFrame({"trade_date": ["20240101"], "ts_code": ["000001.SZ"]})
        with pytest.raises(ValueError) as exc:
            compute_turnover(df, factor_col="factor_clean")
        msg = str(exc.value)
        assert "factor_clean" in msg
        assert "实际列" in msg

    _section_0_test_compute_turnover_raises_on_missing_factor_column()

    # -- 原 test_compute_turnover_raises_on_missing_key_columns --
    def _section_1_test_compute_turnover_raises_on_missing_key_columns():
        df = pl.DataFrame({"factor_clean": [1.0]})
        with pytest.raises(ValueError) as exc:
            compute_turnover(df, factor_col="factor_clean")
        msg = str(exc.value)
        assert "trade_date" in msg and "ts_code" in msg

    _section_1_test_compute_turnover_raises_on_missing_key_columns()

    # -- 原 test_prepare_factor_df_error_lists_actual_columns --
    def _section_2_test_prepare_factor_df_error_lists_actual_columns():
        df = pl.DataFrame({"trade_date": ["20240101"], "ts_code": ["000001.SZ"]})
        with pytest.raises(ValueError) as exc:
            _prepare_factor_df(df, "factor_clean")
        msg = str(exc.value)
        assert "factor_clean" in msg
        assert "实际列" in msg

    _section_2_test_prepare_factor_df_error_lists_actual_columns()

    # -- 原 test_prepare_price_df_error_lists_actual_columns --
    def _section_3_test_prepare_price_df_error_lists_actual_columns():
        df = pl.DataFrame({"trade_date": ["20240101"], "ts_code": ["000001.SZ"]})
        with pytest.raises(ValueError) as exc:
            _prepare_price_df(df)
        msg = str(exc.value)
        assert "close" in msg
        assert "实际列" in msg

    _section_3_test_prepare_price_df_error_lists_actual_columns()

    # -- 原 test_every_registered_factor_declares_daily --
    def _section_4_test_every_registered_factor_declares_daily():
        offenders = []
        for name in list_factors():
            factor = get_factor(name)
            required = getattr(factor, "required_data", None) or []
            if "daily" not in required:
                offenders.append((name, getattr(factor, "category", "?"), list(required)))
        assert not offenders, (
            "以下因子的 required_data 漏声明 'daily'，评估管线算前向收益时会 raise "
            "'daily data not declared'：" + "; ".join(f"{n}({c})={rd}" for n, c, rd in offenders)
        )

    _section_4_test_every_registered_factor_declares_daily()

    # -- 原 test_valuation_factors_keep_daily_basic_and_add_daily --
    def _section_5_test_valuation_factors_keep_daily_basic_and_add_daily():
        for name in ("size_style", "value_style", "liquidity_style", "pe_ttm", "pb",
                     "ep_ratio", "bm_ratio"):
            required = getattr(get_factor(name), "required_data", None) or []
            assert "daily" in required, f"{name} 需 ctx.daily 算前向收益"
            assert "daily_basic" in required, f"{name} compute 读 daily_basic，不应移除声明"

    _section_5_test_valuation_factors_keep_daily_basic_and_add_daily()


# ==== 来自 test_factor_required_data_declares_daily.py ====



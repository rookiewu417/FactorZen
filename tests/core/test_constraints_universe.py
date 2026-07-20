"""
test_trade_constraints.py：test_trade_constraints_batch.py：Unit tests for vectorized apply_trade_constraints_batch (W3-B).
test_universe_rules.py：test_universe_board_limit.py：板块涨跌停阈值及 filter_limit 按板块细化测试。
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import polars as pl
import pytest

from factorzen.core.universe import _get_board_limit, filter_limit, get_universe
from factorzen.daily.evaluation.backtest import BacktestConfig
from factorzen.daily.evaluation.trade_constraints import (
    BLOCK_CAPACITY,
    BLOCK_INVALID_PORTFOLIO,
    BLOCK_LIMIT_DOWN,
    BLOCK_LIMIT_UP,
    BLOCK_MISSING_PRICE,
    BLOCK_OK,
    BLOCK_SUSPENDED,
    apply_trade_constraints,
    apply_trade_constraints_batch,
    block_reason_to_str,
    board_limit_pct_for_codes,
)


# ==== 来自 test_trade_constraints.py ====
# ==== 来自 test_trade_constraints_batch.py ====
def _batch(
    *,
    delta,
    open_px,
    pre_close,
    vol,
    adv,
    board_limits,
    portfolio_value=1e8,
    max_participation_rate=0.05,
    fallback_adv=None,
):
    return apply_trade_constraints_batch(
        delta=np.asarray(delta, dtype=float),
        open_px=np.asarray(open_px, dtype=float),
        pre_close=np.asarray(pre_close, dtype=float),
        vol=np.asarray(vol, dtype=float),
        adv=np.asarray(adv, dtype=float),
        board_limits=np.asarray(board_limits, dtype=float),
        portfolio_value=portfolio_value,
        max_participation_rate=max_participation_rate,
        fallback_adv=fallback_adv,
    )

def test_near_zero_delta_short_circuit():
    filled, reason = _batch(
        delta=[1e-15, 0.0],
        open_px=[10.0, 10.0],
        pre_close=[10.0, 10.0],
        vol=[1e6, 1e6],
        adv=[1e7, 1e7],
        board_limits=[9.8, 9.8],
    )
    assert np.allclose(filled, 0.0)
    assert reason.tolist() == [BLOCK_OK, BLOCK_OK]

def test_missing_price():
    filled, reason = _batch(
        delta=[0.1, 0.1, 0.1],
        open_px=[np.nan, 0.0, 10.0],
        pre_close=[10.0, 10.0, np.nan],
        vol=[1e6, 1e6, 1e6],
        adv=[1e7, 1e7, 1e7],
        board_limits=[9.8, 9.8, 9.8],
    )
    assert np.allclose(filled, 0.0)
    assert reason.tolist() == [BLOCK_MISSING_PRICE] * 3

def test_suspended_vol_zero_not_nan():
    # 大 ADV 避免 capacity 干扰；NaN vol ≠ 停牌
    filled, reason = _batch(
        delta=[0.1, -0.1, 0.1],
        open_px=[10.0, 10.0, 10.0],
        pre_close=[10.0, 10.0, 10.0],
        vol=[0.0, 0.0, np.nan],
        adv=[1e30, 1e30, 1e30],
        board_limits=[9.8, 9.8, 9.8],
        fallback_adv=None,
    )
    assert filled[0] == 0.0 and reason[0] == BLOCK_SUSPENDED
    assert filled[1] == 0.0 and reason[1] == BLOCK_SUSPENDED
    assert reason[2] == BLOCK_OK
    assert filled[2] == pytest.approx(0.1)

def test_limit_up_blocks_buy_allows_sell():
    # main board +9.9%
    filled, reason = _batch(
        delta=[0.1, -0.1],
        open_px=[10.99, 10.99],
        pre_close=[10.0, 10.0],
        vol=[1e6, 1e6],
        adv=[1e30, 1e30],
        board_limits=[9.8, 9.8],
    )
    assert filled[0] == 0.0 and reason[0] == BLOCK_LIMIT_UP
    assert filled[1] == pytest.approx(-0.1) and reason[1] == BLOCK_OK

def test_limit_down_blocks_sell_allows_buy():
    filled, reason = _batch(
        delta=[-0.1, 0.1],
        open_px=[9.01, 9.01],
        pre_close=[10.0, 10.0],
        vol=[1e6, 1e6],
        adv=[1e30, 1e30],
        board_limits=[9.8, 9.8],
    )
    assert filled[0] == 0.0 and reason[0] == BLOCK_LIMIT_DOWN
    assert filled[1] == pytest.approx(0.1) and reason[1] == BLOCK_OK

def test_gem_float_tolerance_limit_up():
    # open=11.98/pre=10 → 19.7999... must block at 19.8
    filled, reason = _batch(
        delta=[0.1],
        open_px=[11.98],
        pre_close=[10.0],
        vol=[1e6],
        adv=[1e30],
        board_limits=[19.8],
    )
    assert filled[0] == 0.0 and reason[0] == BLOCK_LIMIT_UP

def test_st_board_limit_switch():
    limits = board_limit_pct_for_codes(["600001.SH", "600001.SH"])
    limits_st = board_limit_pct_for_codes(["600001.SH", "600001.SH"], is_st=True)
    assert limits[0] == pytest.approx(9.8)
    assert limits_st[0] == pytest.approx(4.8)
    # +5% open: ST blocks, non-ST passes
    filled, reason = _batch(
        delta=[0.1, 0.1],
        open_px=[10.5, 10.5],
        pre_close=[10.0, 10.0],
        vol=[1e6, 1e6],
        adv=[1e30, 1e30],
        board_limits=[limits_st[0], limits[0]],
    )
    assert reason[0] == BLOCK_LIMIT_UP and filled[0] == 0.0
    assert reason[1] == BLOCK_OK and filled[1] == pytest.approx(0.1)

def test_fallback_adv_still_invalid_no_cap():
    # no adv, no fallback → full delta, no capacity
    filled, reason = _batch(
        delta=[0.2],
        open_px=[10.2],
        pre_close=[10.0],
        vol=[1e6],
        adv=[np.nan],
        board_limits=[9.8],
        portfolio_value=1e7,
        fallback_adv=None,
    )
    assert filled[0] == pytest.approx(0.2)
    assert reason[0] == BLOCK_OK

def test_capacity_caps_delta():
    # adv=1e7, rate=0.05 → max trade value 5e5; pv=1e7 → max_delta=0.05
    filled, reason = _batch(
        delta=[0.2, -0.2],
        open_px=[10.2, 10.2],
        pre_close=[10.0, 10.0],
        vol=[1e6, 1e6],
        adv=[1e7, 1e7],
        board_limits=[9.8, 9.8],
        portfolio_value=1e7,
        max_participation_rate=0.05,
        fallback_adv=None,
    )
    assert filled[0] == pytest.approx(0.05) and reason[0] == BLOCK_CAPACITY
    assert filled[1] == pytest.approx(-0.05) and reason[1] == BLOCK_CAPACITY

def test_portfolio_value_le_zero():
    filled, reason = _batch(
        delta=[0.1],
        open_px=[10.2],
        pre_close=[10.0],
        vol=[1e6],
        adv=[1e7],
        board_limits=[9.8],
        portfolio_value=0.0,
    )
    assert filled[0] == 0.0 and reason[0] == BLOCK_INVALID_PORTFOLIO

def test_scalar_wrapper_matches_batch():
    cfg = BacktestConfig(fallback_adv=10_000_000.0, max_participation_rate=0.05)
    price_map = {
        "000001.SZ": {"open": 10.2, "pre_close": 10.0, "vol": 1e6},
        "600001.SH": {"open": 10.5, "pre_close": 10.0, "vol": 0.0},
        "300001.SZ": {"open": 11.98, "pre_close": 10.0, "vol": 1e6},
    }
    codes = list(price_map)
    deltas = [0.1, -0.05, 0.2]
    for code, d in zip(codes, deltas, strict=True):
        sf, sr = apply_trade_constraints(
            code=code,
            delta=d,
            price_map=price_map,
            portfolio_value=1e8,
            config=cfg,
            adv=1e7,
            is_st=False,
        )
        # single-row batch
        rec = price_map[code]
        from factorzen.core.universe import _get_board_limit

        bf, br = apply_trade_constraints_batch(
            delta=np.array([d]),
            open_px=np.array([float(rec["open"])]),
            pre_close=np.array([float(rec["pre_close"])]),
            vol=np.array([float(rec["vol"])]),
            adv=np.array([1e7]),
            board_limits=np.array([_get_board_limit(code) * 100]),
            portfolio_value=1e8,
            max_participation_rate=0.05,
            fallback_adv=cfg.fallback_adv,
        )
        assert sf == pytest.approx(float(bf[0]))
        assert sr == block_reason_to_str(br)[0]

def test_block_reason_to_str_roundtrip():
    codes = np.array(
        [
            BLOCK_OK,
            BLOCK_MISSING_PRICE,
            BLOCK_SUSPENDED,
            BLOCK_LIMIT_UP,
            BLOCK_LIMIT_DOWN,
            BLOCK_CAPACITY,
            BLOCK_INVALID_PORTFOLIO,
        ],
        dtype=np.int8,
    )
    assert block_reason_to_str(codes) == [
        "",
        "missing_price",
        "suspended",
        "limit_up",
        "limit_down",
        "capacity",
        "invalid_portfolio_value",
    ]

# ==== 来自 test_filter_liquidity_unit.py ====
def _fake_daily(amounts_qy: dict[str, float]):
    """构造一天的 daily 帧，amount 单位=千元（Tushare 口径）。"""
    codes = list(amounts_qy)
    return pl.LazyFrame(
        {
            "ts_code": codes,
            "trade_date": [pl.date(2026, 6, 5)] * len(codes),
            "amount": [amounts_qy[c] for c in codes],
        }
    )

def test_filter_liquidity_uses_yuan_threshold(monkeypatch):
    import factorzen.core.storage as storage
    from factorzen.core.universe import filter_liquidity

    # A: 2000万元成交额 = 20_000 千元（应留）；B: 500万元 = 5_000 千元（应剔）
    amounts_qy = {"A.SZ": 20_000.0, "B.SZ": 5_000.0}
    monkeypatch.setattr(storage, "load_parquet", lambda *a, **k: _fake_daily(amounts_qy))

    stocks = pl.DataFrame({"ts_code": ["A.SZ", "B.SZ"], "industry": ["X", "Y"]})
    # 默认 min_amount=1000万元
    kept = filter_liquidity(stocks, "20260605")["ts_code"].to_list()

    assert "A.SZ" in kept, "2000万元成交额应通过 1000万元 门槛（修复前因单位错配被剔除）"
    assert "B.SZ" not in kept, "500万元成交额应被 1000万元 门槛剔除"

def test_filter_liquidity_realistic_market_not_collapsed(monkeypatch):
    """真实量级：中位数约 1.36亿元（≈135_762 千元）的市场不应被门槛几乎清空。"""
    import factorzen.core.storage as storage
    from factorzen.core.universe import filter_liquidity

    # 100 只股票，成交额 5000万~5亿元（=50_000~500_000 千元），全部远超 1000万元 门槛
    amounts_qy = {f"{i:06d}.SZ": 50_000.0 + i * 4500.0 for i in range(100)}
    monkeypatch.setattr(storage, "load_parquet", lambda *a, **k: _fake_daily(amounts_qy))

    stocks = pl.DataFrame({"ts_code": list(amounts_qy), "industry": ["X"] * 100})
    kept = filter_liquidity(stocks, "20260605")
    assert kept.height == 100, f"全部应通过，修复前会因 100亿元 假门槛只剩极少数（实得 {kept.height}）"

# ==== 来自 test_universe_rules.py ====
# ==== 来自 test_universe_board_limit.py ====
@pytest.fixture(autouse=True)
def _no_namechange_by_default(monkeypatch):
    """默认 namechange 不可用，filter_limit 统一走降级（按 name 字符串匹配）路径。

    universe.py 用 ``from factorzen.core.loader import fetch_namechange`` 在
    模块级绑定，须 patch ``factorzen.core.universe.fetch_namechange`` 才能
    生效（patch ``factorzen.core.loader.fetch_namechange`` 对已绑定的引用
    无效）。避免本机 .env 配了真实 token 时意外触发真实网络请求。
    """

    def _boom() -> pl.DataFrame:
        raise RuntimeError("namechange unavailable in offline tests")

    monkeypatch.setattr("factorzen.core.universe.fetch_namechange", _boom)

# ──────────────────────────────────────────────────────────
# _get_board_limit 单元测试
# ──────────────────────────────────────────────────────────

def test_chuang_ye_ban_limit_300():
    """创业板 300xxx → 19.8%。"""
    assert abs(_get_board_limit("300001.SZ") - 0.198) < 1e-6

def test_chuang_ye_ban_limit_301():
    """创业板 301xxx → 19.8%。"""
    assert abs(_get_board_limit("301001.SZ") - 0.198) < 1e-6

def test_ke_chuang_ban_limit_688():
    """科创板 688xxx → 19.8%。"""
    assert abs(_get_board_limit("688001.SH") - 0.198) < 1e-6

def test_ke_chuang_ban_limit_689():
    """科创板 689xxx → 19.8%。"""
    assert abs(_get_board_limit("689001.SH") - 0.198) < 1e-6

def test_bei_jiao_suo_limit():
    """北交所 .BJ 后缀 → 29.8%。"""
    assert abs(_get_board_limit("830001.BJ") - 0.298) < 1e-6

def test_main_board_limit_600():
    """主板 600xxx → 9.8%。"""
    assert abs(_get_board_limit("600001.SH") - 0.098) < 1e-6

def test_main_board_limit_case_insensitive():
    """大小写不敏感。"""
    assert abs(_get_board_limit("600001.sh") - 0.098) < 1e-6

# ──────────────────────────────────────────────────────────
# _get_board_limit(is_st=True) — ST 主板收窄阈值
# ──────────────────────────────────────────────────────────

def test_main_board_st_limit_is_4_8pct():
    """主板 ST/*ST 股票 is_st=True → 4.8%（5% 真实限额 - 0.2pp 容差）。"""
    assert abs(_get_board_limit("600001.SH", is_st=True) - 0.048) < 1e-6

def test_chuang_ye_ban_is_st_does_not_affect_limit():
    """创业板不受 is_st 影响（2020 年注册制改革后 ST 与非 ST 涨跌幅规则相同）。"""
    assert abs(_get_board_limit("300001.SZ", is_st=True) - 0.198) < 1e-6

def test_ke_chuang_ban_is_st_does_not_affect_limit():
    """科创板不受 is_st 影响（同上）。"""
    assert abs(_get_board_limit("688001.SH", is_st=True) - 0.198) < 1e-6

def test_bei_jiao_suo_is_st_does_not_affect_limit():
    """北交所不受 is_st 影响。"""
    assert abs(_get_board_limit("830001.BJ", is_st=True) - 0.298) < 1e-6

# ──────────────────────────────────────────────────────────
# filter_limit 纯 DataFrame 路径（不依赖日线存储）
#
# filter_limit 正常路径需要 load_parquet；
# 此处通过 monkeypatch 绕过，直接测试过滤逻辑。
# ──────────────────────────────────────────────────────────

def _make_daily(ts_code: str, pct_chg: float) -> pl.DataFrame:
    """构造仅含 ts_code 和 pct_chg 的最小日线 DataFrame。"""
    return pl.DataFrame({
        "ts_code": [ts_code],
        "pct_chg": [pct_chg],
        "vol": [1000.0],
        "amount": [1_000_000.0],
        "open": [10.0],
        "close": [10.0],
    })

def _make_stocks(ts_code: str) -> pl.DataFrame:
    return pl.DataFrame({
        "ts_code": [ts_code],
        "name": ["Test Stock"],
        "list_date": [None],
        "delist_date": [None],
    })

def test_filter_limit_allows_chuang_ye_195pct(monkeypatch):
    """创业板 19.5% 涨幅 < 19.8% 阈值，不应被过滤。"""

    ts_code = "300001.SZ"
    pct_chg = 19.5

    def fake_load(category, start=None, end=None):

        class LazyWrapper:
            def collect(self):
                return _make_daily(ts_code, pct_chg)

        return LazyWrapper()

    monkeypatch.setattr("factorzen.core.storage.load_parquet", fake_load)

    stocks = _make_stocks(ts_code)
    result = filter_limit(stocks, "20240101")
    assert len(result) == 1, f"创业板 19.5% 不应被过滤，但 result={result}"

def test_filter_limit_blocks_chuang_ye_198pct(monkeypatch):
    """创业板 19.8% 正好达到阈值，应被过滤（>= 而非 >）。"""
    ts_code = "300001.SZ"
    pct_chg = 19.8

    def fake_load(category, start=None, end=None):
        class LazyWrapper:
            def collect(self):
                return _make_daily(ts_code, pct_chg)

        return LazyWrapper()

    monkeypatch.setattr("factorzen.core.storage.load_parquet", fake_load)

    stocks = _make_stocks(ts_code)
    result = filter_limit(stocks, "20240101")
    assert len(result) == 0, f"创业板 19.8% 应被过滤，但 result={result}"

def test_filter_limit_blocks_main_board_10pct(monkeypatch):
    """主板 10% > 9.8% 阈值，应被过滤。"""
    ts_code = "600001.SH"
    pct_chg = 10.0

    def fake_load(category, start=None, end=None):
        class LazyWrapper:
            def collect(self):
                return _make_daily(ts_code, pct_chg)

        return LazyWrapper()

    monkeypatch.setattr("factorzen.core.storage.load_parquet", fake_load)

    stocks = _make_stocks(ts_code)
    result = filter_limit(stocks, "20240101")
    assert len(result) == 0, f"主板 10% 应被过滤，但 result={result}"

def test_filter_limit_allows_main_board_9pct(monkeypatch):
    """主板 9% < 9.8% 阈值，不应被过滤。"""
    ts_code = "600001.SH"
    pct_chg = 9.0

    def fake_load(category, start=None, end=None):
        class LazyWrapper:
            def collect(self):
                return _make_daily(ts_code, pct_chg)

        return LazyWrapper()

    monkeypatch.setattr("factorzen.core.storage.load_parquet", fake_load)

    stocks = _make_stocks(ts_code)
    result = filter_limit(stocks, "20240101")
    assert len(result) == 1, f"主板 9% 不应被过滤，但 result={result}"

def test_filter_limit_mixed_boards(monkeypatch):
    """主板 10% 被过滤，创业板 19.5% 保留，测试混合场景。"""
    daily_data = pl.DataFrame({
        "ts_code": ["600001.SH", "300001.SZ"],
        "pct_chg": [10.0, 19.5],
        "vol": [1000.0, 1000.0],
        "amount": [1_000_000.0, 1_000_000.0],
        "open": [10.0, 10.0],
        "close": [10.0, 10.0],
    })

    def fake_load(category, start=None, end=None):
        class LazyWrapper:
            def collect(self):
                return daily_data

        return LazyWrapper()

    monkeypatch.setattr("factorzen.core.storage.load_parquet", fake_load)

    stocks = pl.DataFrame({
        "ts_code": ["600001.SH", "300001.SZ"],
        "name": ["Main", "ChiNext"],
        "list_date": [None, None],
        "delist_date": [None, None],
    })
    result = filter_limit(stocks, "20240101")
    assert len(result) == 1
    assert result["ts_code"][0] == "300001.SZ"

# ──────────────────────────────────────────────────────────
# filter_limit — ST 主板收窄阈值（4.8%），经 namechange PIT 判断
# ──────────────────────────────────────────────────────────

def _namechange_st_df(ts_code: str, start_date: date = date(2024, 1, 1)) -> pl.DataFrame:
    """构造单只股票当前处于 ST 状态的 namechange 记录。"""
    return pl.DataFrame(
        {
            "ts_code": [ts_code],
            "name": ["ST测试股"],
            "start_date": [start_date],
            "end_date": [None],
            "ann_date": [start_date],
            "change_reason": ["ST"],
        }
    )

def test_filter_limit_st_main_board_5pct_blocked(monkeypatch):
    """主板 ST 股票涨幅约 +5.0%（除法构造而非字面量），namechange 标记 ST 后
    应被 filter_limit 判定涨停过滤（阈值 4.8%）。
    """
    ts_code = "600001.SH"
    pct_chg = (10.5 / 10.0 - 1.0) * 100  # ≈5.0，由除法构造

    def fake_load(category, start=None, end=None):
        class LazyWrapper:
            def collect(self):
                return _make_daily(ts_code, pct_chg)

        return LazyWrapper()

    monkeypatch.setattr("factorzen.core.storage.load_parquet", fake_load)
    monkeypatch.setattr(
        "factorzen.core.universe.fetch_namechange",
        lambda: _namechange_st_df(ts_code),
    )

    stocks = _make_stocks(ts_code)
    result = filter_limit(stocks, "20240101")
    assert len(result) == 0, f"ST 主板 5% 涨幅应被判定涨停过滤，实际 result={result}"

def test_filter_limit_non_st_5pct_not_blocked(monkeypatch):
    """同样约 +5.0% 涨幅，非 ST 主板不应被过滤（主板非 ST 阈值 9.8%）。"""
    ts_code = "600001.SH"
    pct_chg = (10.5 / 10.0 - 1.0) * 100  # ≈5.0，由除法构造

    def fake_load(category, start=None, end=None):
        class LazyWrapper:
            def collect(self):
                return _make_daily(ts_code, pct_chg)

        return LazyWrapper()

    monkeypatch.setattr("factorzen.core.storage.load_parquet", fake_load)
    # namechange 可用但无该代码的 ST 记录 → 判定为非 ST
    monkeypatch.setattr(
        "factorzen.core.universe.fetch_namechange",
        lambda: _namechange_st_df("000999.SZ"),
    )

    stocks = _make_stocks(ts_code)
    result = filter_limit(stocks, "20240101")
    assert len(result) == 1, f"非 ST 5% 涨幅不应被过滤，实际 result={result}"

# ==== 来自 test_universe_pit.py ====
@pytest.fixture
def synthetic_stock_basic(monkeypatch):
    """注入含退市股的合成股票基本信息。"""
    df = pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
            "symbol": ["000001", "000002", "000003", "000004"],
            "name": ["股票A", "股票B（已退市）", "股票C（未来上市）", "股票D（无退市日）"],
            "area": ["深圳"] * 4,
            "industry": ["银行"] * 4,
            "market": ["主板"] * 4,
            "list_date": [
                date(2005, 1, 1),  # 2005 上市，至今在市
                date(2010, 1, 1),  # 2010 上市，2023-12-31 退市
                date(2025, 1, 1),  # 2025 上市（基准日之后），不应出现
                date(2008, 1, 1),  # 2008 上市，无退市日（仍在市）
            ],
            "delist_date": [
                None,  # A：无退市日，仍在市
                date(2023, 12, 31),  # B：2023-12-31 退市
                None,  # C：未来上市
                None,  # D：无退市日，仍在市
            ],
        }
    )
    monkeypatch.setattr("factorzen.core.universe.get_stock_basic", lambda: df)
    return df

class TestUniversePIT:
    def test_all_a_excludes_delisted(self, synthetic_stock_basic):
        """基准日 2024-01-15：已于 2023-12-31 退市的 000002.SZ 不应出现。"""
        result = get_universe("20240115", "all_a")
        codes = result["ts_code"].to_list()
        assert "000002.SZ" not in codes, "退市股 000002.SZ 不应出现在 2024-01-15 的股票池"

    def test_all_a_excludes_future_listed(self, synthetic_stock_basic):
        """基准日 2024-01-15：2025 年上市的 000003.SZ 不应出现。"""
        result = get_universe("20240115", "all_a")
        codes = result["ts_code"].to_list()
        assert "000003.SZ" not in codes, "未上市股 000003.SZ 不应出现在 2024-01-15 的股票池"

    def test_all_a_includes_active_stocks(self, synthetic_stock_basic):
        """基准日 2024-01-15：2005 上市、仍在市的 000001.SZ 应出现。"""
        result = get_universe("20240115", "all_a")
        codes = result["ts_code"].to_list()
        assert "000001.SZ" in codes, "在市股 000001.SZ 应出现在 2024-01-15 的股票池"
        assert "000004.SZ" in codes, "在市股 000004.SZ 应出现在 2024-01-15 的股票池"

    def test_all_a_includes_stock_before_delist(self, synthetic_stock_basic):
        """基准日 2023-06-01：000002.SZ 尚未退市（2023-12-31 才退），应出现。"""
        result = get_universe("20230601", "all_a")
        codes = result["ts_code"].to_list()
        assert "000002.SZ" in codes, "尚未退市的 000002.SZ 应出现在 2023-06-01 的股票池"

    def test_all_a_excludes_stock_on_delist_date(self, synthetic_stock_basic):
        """基准日 2023-12-31（退市日当天）：000002.SZ 应已被排除（delist_date > date 严格大于）。"""
        result = get_universe("20231231", "all_a")
        codes = result["ts_code"].to_list()
        assert "000002.SZ" not in codes, (
            "退市当日 000002.SZ 不应出现在股票池（delist_date 严格大于）"
        )

    def test_pit_count_varies_by_date(self, synthetic_stock_basic):
        """不同日期的股票池大小应不同（PIT 过滤生效）。"""
        pre_delist = get_universe("20230601", "all_a")  # B 尚在市 → 3 只
        post_delist = get_universe("20240115", "all_a")  # B 已退市 → 2 只
        assert len(pre_delist) > len(post_delist), (
            f"2023-06-01 ({len(pre_delist)} 只) 应多于 2024-01-15 ({len(post_delist)} 只)"
        )

# ==== 来自 test_universe.py ====
# ── helpers ────────────────────────────────────────────────────────────────

needs_tushare = pytest.mark.skipif(
    not os.environ.get("TUSHARE_TOKEN"),
    reason="TUSHARE_TOKEN 未设置，跳过 Tushare 集成测试",
)

# 使用近期交易日，确保 Tushare 有数据
FIXTURE_DATE = "20260512"
FIXTURE_INDEX_CSI300 = "000300.SH"
FIXTURE_INDEX_CSI500 = "000905.SH"

# ── index members ──────────────────────────────────────────────────────────

@needs_tushare
def test_get_index_members_csi300():
    """CSI300 成分股应返回 200-350 只股票（而非全 A 股 ~5500 只）。"""
    result = get_universe(FIXTURE_DATE, "csi300")

    assert not result.is_empty(), "CSI300 不应为空"
    assert "ts_code" in result.columns
    assert "name" in result.columns

    count = result.height
    assert 200 <= count <= 350, f"CSI300 预期 200-350 只，实际 {count} 只"

@needs_tushare
def test_csi800_is_union():
    """CSI800 = CSI300 ∪ CSI500，去重后数量应 ≈ CSI300 + CSI500。"""
    csi300_codes = set(get_universe(FIXTURE_DATE, "csi300")["ts_code"].to_list())
    csi500_codes = set(get_universe(FIXTURE_DATE, "csi500")["ts_code"].to_list())
    csi800_codes = set(get_universe(FIXTURE_DATE, "csi800")["ts_code"].to_list())

    n800 = len(csi800_codes)

    # CSI800 应为 union 去重
    expected_union = csi300_codes | csi500_codes
    assert expected_union == csi800_codes, "CSI800 应为 CSI300 ∪ CSI500"

    assert n800 == len(expected_union), (
        f"CSI800({n800}) 应等于 union 去重结果({len(expected_union)})"
    )

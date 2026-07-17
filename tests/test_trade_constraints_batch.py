"""Unit tests for vectorized apply_trade_constraints_batch (W3-B)."""
from __future__ import annotations

import numpy as np
import pytest

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

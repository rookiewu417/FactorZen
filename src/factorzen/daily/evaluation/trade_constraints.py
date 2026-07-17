"""交易约束内核。

慢路径（``run_strategy_backtest``）、快路径（``_run_precomputed_weights_backtest_fast``）
与 ``execution/brokers/paper.py::PaperBroker`` 共用同一套约束语义。

向量化入口：``apply_trade_constraints_batch``（当日截面 numpy 数组 → filled_delta +
block_reason 码）。标量 ``apply_trade_constraints`` 是 batch 的单行包装，供
PaperBroker / 旧调用方使用，行为与历史一致。

涨跌停板块阈值：``board_limit_pct_for_codes`` 按 code 一次预计算（含 ST 变体），
快慢两侧共用，消除双路径漂移。
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from factorzen.core.universe import _get_board_limit

if TYPE_CHECKING:
    from factorzen.daily.evaluation.backtest import BacktestConfig

# block_reason 整型码（batch 输出）；映射回字符串枚举与旧口径一致
BLOCK_OK = 0
BLOCK_MISSING_PRICE = 1
BLOCK_SUSPENDED = 2
BLOCK_LIMIT_UP = 3
BLOCK_LIMIT_DOWN = 4
BLOCK_CAPACITY = 5
BLOCK_INVALID_PORTFOLIO = 6

BLOCK_REASON_STR: tuple[str, ...] = (
    "",
    "missing_price",
    "suspended",
    "limit_up",
    "limit_down",
    "capacity",
    "invalid_portfolio_value",
)


def block_reason_to_str(codes: np.ndarray) -> list[str]:
    """Map integer block-reason codes to legacy string enums."""
    table = BLOCK_REASON_STR
    # 快速路径：整段同质时避免 Python 循环
    flat = np.asarray(codes, dtype=np.int8).ravel()
    return [table[int(c)] for c in flat]


def board_limit_pct_for_codes(
    codes: Sequence[str],
    *,
    is_st: bool = False,
) -> np.ndarray:
    """按 code 预计算涨跌停阈值（**百分比**，如主板 9.8）。

    与 ``_get_board_limit(code, is_st) * 100`` 逐值一致；供快/慢路径一次装载复用，
    避免每股每日字符串解析。
    """
    n = len(codes)
    out = np.empty(n, dtype=np.float64)
    for i, code in enumerate(codes):
        out[i] = _get_board_limit(code, is_st=is_st) * 100.0
    return out


def apply_trade_constraints_batch(
    *,
    delta: np.ndarray,
    open_px: np.ndarray,
    pre_close: np.ndarray,
    vol: np.ndarray,
    adv: np.ndarray,
    board_limits: np.ndarray,
    portfolio_value: float,
    max_participation_rate: float,
    fallback_adv: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """向量化交易约束（当日截面）。

    Parameters
    ----------
    delta, open_px, pre_close, vol, adv, board_limits
        等长 1-d float 数组。``board_limits`` 为**当日有效**阈值（已含 ST 切换）。
        ``vol``：``isfinite(vol) & vol==0`` → 停牌；**NaN ≠ 停牌**。
        ``adv``：缺失/非正时填 ``fallback_adv``；仍无效则**不 cap**（放行 delta）。
    portfolio_value
        当日 ``open_nav * initial_capital``；``<=0`` 且 ADV 有效时全零
        （``invalid_portfolio_value``）。与标量内核一致：无有效 ADV 时不进 cap 分支。
    max_participation_rate, fallback_adv
        来自 ``BacktestConfig``。

    Returns
    -------
    filled_delta : np.ndarray[float64]
    block_reason : np.ndarray[int8]
        见模块级 ``BLOCK_*`` 常量；``BLOCK_OK`` 表示放行/无交易。
    """
    delta = np.asarray(delta, dtype=np.float64)
    n = delta.shape[0]
    open_px = np.asarray(open_px, dtype=np.float64)
    pre_close = np.asarray(pre_close, dtype=np.float64)
    vol = np.asarray(vol, dtype=np.float64)
    adv = np.asarray(adv, dtype=np.float64)
    board_limits = np.asarray(board_limits, dtype=np.float64)

    filled = np.zeros(n, dtype=np.float64)
    reason = np.zeros(n, dtype=np.int8)

    active = np.abs(delta) > 1e-12
    if not np.any(active):
        return filled, reason

    # 1) missing_price
    valid_price = (
        np.isfinite(open_px)
        & np.isfinite(pre_close)
        & (open_px > 0)
        & (pre_close > 0)
    )
    missing = active & ~valid_price
    reason[missing] = BLOCK_MISSING_PRICE

    # 2) suspended: vol is not None and vol==0; NaN ≠ suspended
    suspended = active & valid_price & np.isfinite(vol) & (vol == 0.0)
    reason[suspended] = BLOCK_SUSPENDED

    # 3) limit up/down (only if still candidate)
    candidate = active & valid_price & ~suspended
    opening_pct = np.zeros(n, dtype=np.float64)
    if np.any(candidate):
        opening_pct[candidate] = (
            open_px[candidate] / pre_close[candidate] - 1.0
        ) * 100.0

    # 浮点容差：创业板 open=11.98/pre=10 → 19.7999... >= 19.8 须判涨停
    limit_up = candidate & (delta > 0) & (opening_pct >= board_limits - 1e-9)
    limit_down = candidate & (delta < 0) & (opening_pct <= -board_limits + 1e-9)
    reason[limit_up] = BLOCK_LIMIT_UP
    reason[limit_down] = BLOCK_LIMIT_DOWN

    tradable = candidate & ~limit_up & ~limit_down
    filled[tradable] = delta[tradable]

    if not np.any(tradable):
        return filled, reason

    # 4) ADV fallback；仍无效 → 不 cap（保留 filled=delta, reason=OK）
    adv_eff = adv.copy()
    valid_adv = np.isfinite(adv_eff) & (adv_eff > 0)
    if (
        fallback_adv is not None
        and np.isfinite(float(fallback_adv))
        and float(fallback_adv) > 0
    ):
        adv_eff[~valid_adv] = float(fallback_adv)
        valid_adv = np.isfinite(adv_eff) & (adv_eff > 0)

    # 无有效 ADV 的 tradable：放行，reason 保持 OK
    need_cap = tradable & valid_adv
    if not np.any(need_cap):
        return filled, reason

    # 5) portfolio_value <= 0 → 仅对 need_cap 股全零（与标量：先过 ADV 再查 portfolio）
    if portfolio_value <= 0:
        filled[need_cap] = 0.0
        reason[need_cap] = BLOCK_INVALID_PORTFOLIO
        return filled, reason

    # 6) capacity
    max_delta = adv_eff[need_cap] * float(max_participation_rate) / float(portfolio_value)
    abs_d = np.abs(filled[need_cap])
    over = abs_d > max_delta + 1e-12
    if np.any(over):
        idx = np.flatnonzero(need_cap)
        over_idx = idx[over]
        filled[over_idx] = np.sign(filled[over_idx]) * max_delta[over]
        reason[over_idx] = BLOCK_CAPACITY

    return filled, reason


def apply_trade_constraints(
    *,
    code: str,
    delta: float,
    price_map: dict[str, dict[str, Any]],
    portfolio_value: float,
    config: BacktestConfig,
    adv: float | None = None,
    is_st: bool = False,
) -> tuple[float, str]:
    """单股约束（batch 核单行包装）。PaperBroker / 旧调用方入口；行为不变。"""
    if abs(delta) < 1e-12:
        return 0.0, ""

    rec = price_map.get(code)
    if rec is None:
        open_v = pre_v = vol_v = np.nan
    else:
        o = rec.get("open")
        p = rec.get("pre_close")
        open_v = float(o) if o is not None else np.nan
        pre_v = float(p) if p is not None else np.nan
        # vol is None → NaN（不视为停牌）；与标量旧语义 ``vol is not None and float(vol)==0`` 对齐
        v = rec.get("vol")
        vol_v = float(v) if v is not None else np.nan

    if code:
        board = _get_board_limit(code, is_st=is_st) * 100.0
    else:
        board = float(config.limit_up_pct)

    adv_v = float(adv) if adv is not None else np.nan

    filled, reasons = apply_trade_constraints_batch(
        delta=np.array([delta], dtype=np.float64),
        open_px=np.array([open_v], dtype=np.float64),
        pre_close=np.array([pre_v], dtype=np.float64),
        vol=np.array([vol_v], dtype=np.float64),
        adv=np.array([adv_v], dtype=np.float64),
        board_limits=np.array([board], dtype=np.float64),
        portfolio_value=float(portfolio_value),
        max_participation_rate=float(config.max_participation_rate),
        fallback_adv=config.fallback_adv,
    )
    return float(filled[0]), BLOCK_REASON_STR[int(reasons[0])]

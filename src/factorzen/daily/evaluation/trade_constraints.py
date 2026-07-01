"""交易约束内核。

从 backtest.py 抽取（纯搬运，行为不变）。batch backtest 的**慢路径**
（`_apply_trade_constraints` 是本模块 `apply_trade_constraints` 的别名/复用）与
`execution/brokers/paper.py::PaperBroker` 都 import 本函数，物理保证这两条链路
的涨跌停/停牌/容量逻辑只有一份。

注意：`_run_precomputed_weights_backtest_fast`（backtest.py，模拟交易 sim 走的
**快路径**）出于向量化性能考虑，仍保留一份独立的 numpy 实现（停牌/涨跌停/ADV
容量语义与本内核逐条对齐，靠注释手工保持 parity，历史遗留双路径）、并未 import
本函数。改本文件的约束语义时，须同步检查该快路径是否也需要同步修改；未来应收
敛为单一实现。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from factorzen.core.universe import _get_board_limit

if TYPE_CHECKING:
    from factorzen.daily.evaluation.backtest import BacktestConfig


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
    if abs(delta) < 1e-12:
        return 0.0, ""
    rec = price_map.get(code)
    if rec is None or rec.get("open") is None or rec.get("pre_close") is None:
        return 0.0, "missing_price"
    open_price = float(rec["open"])
    pre_close = float(rec["pre_close"])
    if (
        not np.isfinite(open_price)
        or not np.isfinite(pre_close)
        or open_price <= 0
        or pre_close <= 0
    ):
        return 0.0, "missing_price"

    # Check if stock is suspended (vol == 0)
    vol = rec.get("vol")
    if vol is not None and float(vol) == 0.0:
        return 0.0, "suspended"

    opening_pct = (open_price / pre_close - 1.0) * 100.0
    board_limit_pct = _get_board_limit(code, is_st=is_st) * 100 if code else config.limit_up_pct
    effective_limit_up = board_limit_pct
    effective_limit_down = -board_limit_pct
    # 浮点容差：防止 (11.98/10-1)*100=19.7999... >= 19.8 漏判
    if delta > 0 and opening_pct >= effective_limit_up - 1e-9:
        return 0.0, "limit_up"
    if delta < 0 and opening_pct <= effective_limit_down + 1e-9:
        return 0.0, "limit_down"

    adv_value = float(adv) if adv is not None else 0.0
    if not np.isfinite(adv_value) or adv_value <= 0:
        fallback_adv = config.fallback_adv
        adv_value = float(fallback_adv) if fallback_adv is not None else 0.0
    if not np.isfinite(adv_value) or adv_value <= 0:
        return delta, ""

    max_trade_value = adv_value * config.max_participation_rate
    if portfolio_value <= 0:
        return 0.0, "invalid_portfolio_value"
    max_delta = max_trade_value / portfolio_value
    if abs(delta) > max_delta + 1e-12:
        return float(np.sign(delta) * max_delta), "capacity"
    return delta, ""

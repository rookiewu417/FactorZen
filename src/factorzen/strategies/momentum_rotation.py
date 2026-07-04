"""非因子策略：宽基指数动量轮动。PIT 生成多期 weights 产物。

每个 rebalance_date ``T`` 处：

- 对每个候选宽基指数，算过去 ``lookback`` 个交易日的动量（``close(T)/close(T-lookback)-1``，
  只用 ``trade_date <= T`` 的收盘，PIT 无未来函数）。
- 取动量最强的指数；若最强也为负（全负动量），空仓（risk-off，全现金）。
- 持有该指数的 PIT 成分中按 ``trade_date <= T`` 20 日均成交额排名前 ``top_n`` 只，等权。
- 每期落盘 ``weights.parquet`` + ``manifest.json``（供 ``execution`` / ``sim`` 消费）。
"""
from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import polars as pl

_ADV_WINDOW = 20


def _default_members(index_code: str, date_str: str) -> list[str]:
    """默认成分来源：真实 Tushare 指数成分（``_load_index_members`` 要 YYYYMMDD）。"""
    from factorzen.core.universe import _load_index_members

    return _load_index_members(index_code, date_str.replace("-", ""))


def _momentum(idx_sorted: pl.DataFrame, T: date, lookback: int) -> float | None:
    """PIT 动量：只用 trade_date <= T 的收盘；历史不足 lookback+1 根则 None。"""
    hist = idx_sorted.filter(pl.col("trade_date") <= T)
    if hist.height <= lookback:
        return None
    closes = hist["close"].to_list()
    return closes[-1] / closes[-1 - lookback] - 1.0


def _topn_equal_weight(
    price_daily: pl.DataFrame, members: list[str], T: date, top_n: int
) -> dict[str, float]:
    """PIT 流动性：只用 trade_date <= T 的成交额，组内最近 20 日均值取前 top_n 等权。"""
    px = price_daily.filter(
        (pl.col("trade_date") <= T) & pl.col("ts_code").is_in(members)
    ).sort("trade_date")
    if px.is_empty():
        return {}
    liq = px.group_by("ts_code").agg(pl.col("amount").tail(_ADV_WINDOW).mean().alias("adv"))
    top = liq.sort("adv", descending=True).head(top_n)["ts_code"].to_list()
    if not top:
        return {}
    w = 1.0 / len(top)
    return {c: w for c in top}


def generate_momentum_rotation_products(
    out_dir: str,
    index_dailies: dict[str, pl.DataFrame],
    price_daily: pl.DataFrame,
    rebalance_dates: list[date],
    *,
    members_fn: Callable[[str, str], list[str]] | None = None,
    lookback: int = 126,
    top_n: int = 50,
) -> list[str]:
    """宽基指数动量轮动：PIT 生成多期 weights 产物。

    Parameters
    ----------
    index_dailies : dict[index_code, DataFrame(trade_date, close)]
        候选宽基指数日线（如 ``{"000300.SH": df1, "000905.SH": df2}``）。
    price_daily : DataFrame(trade_date, ts_code, amount, ...)
        个股日线，用于成分流动性排序与执行。
    lookback : int
        动量回看交易日数（默认 126 ≈ 6 个月）。
    top_n : int
        每期持有胜出指数成分的数量上限（等权）。

    Returns
    -------
    list[str]
        每个 rebalance_date 对应的 run_dir，顺序一致。
    """
    members_fn = members_fn or _default_members
    idx_sorted = {c: d.sort("trade_date") for c, d in index_dailies.items()}

    run_dirs: list[str] = []
    for T in rebalance_dates:
        moms = {c: _momentum(d, T, lookback) for c, d in idx_sorted.items()}
        valid = {c: m for c, m in moms.items() if m is not None}
        if not valid:
            weights: dict[str, float] = {}
        else:
            winner = max(valid, key=lambda c: valid[c])
            if valid[winner] <= 0.0:  # 全负动量 → 现金
                weights = {}
            else:
                weights = _topn_equal_weight(
                    price_daily, members_fn(winner, T.isoformat()), T, top_n
                )

        rd = Path(out_dir) / T.isoformat()
        rd.mkdir(parents=True, exist_ok=True)
        codes = list(weights)
        pl.DataFrame(
            {"ts_code": codes, "target_weight": [weights[c] for c in codes]},
            schema={"ts_code": pl.Utf8, "target_weight": pl.Float64},
        ).write_parquet(rd / "weights.parquet")
        (rd / "manifest.json").write_text(
            json.dumps({"signal_date": T.isoformat(), "status": "optimal"}, ensure_ascii=False)
        )
        run_dirs.append(str(rd))
    return run_dirs

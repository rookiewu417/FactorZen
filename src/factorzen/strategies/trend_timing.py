"""非因子策略：HS300 200日均线趋势择时 overlay。PIT 生成多期 weights 产物。

每个 rebalance_date ``T`` 处：

- 均线判定只使用 ``trade_date <= T`` 的指数收盘价（PIT，无未来函数）。
- risk-on（``close(T) > MA`` 或 ``timing=False`` 基线）时，按 ``trade_date <= T``
  的 20 日均成交额（ADV）对指数成分股排序，取前 ``top_n`` 等权持仓；
  risk-off 时空仓。
- 每期落盘 ``weights.parquet`` + ``manifest.json``（含 ``signal_date``、
  ``status``），供 ``sim`` 模拟交易 / ``execution`` 消费。
"""
from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import polars as pl

_ADV_WINDOW = 20


def _default_members(index_code: str, date_str: str) -> list[str]:
    """默认成分股来源：真实 Tushare 指数成分（按月缓存）。

    ``_load_index_members`` 要求 ``date_str`` 为 ``"YYYYMMDD"``，而本模块内部
    统一以 ISO 格式（``"YYYY-MM-DD"``）传递日期，这里做一次格式转换。
    """
    from factorzen.core.universe import _load_index_members

    return _load_index_members(index_code, date_str.replace("-", ""))


def _rebalance_weights(
    T: date,
    idx: pl.DataFrame,
    price_daily: pl.DataFrame,
    members_fn: Callable[[str, str], list[str]],
    *,
    index_code: str,
    ma_window: int,
    top_n: int,
    timing: bool,
) -> dict[str, float]:
    """计算单个 rebalance_date T 的目标权重（全程只使用 trade_date <= T 的数据）。"""
    hist = idx.filter(pl.col("trade_date") <= T)
    if hist.height < ma_window:
        # 历史不足 ma_window 根，无法判定趋势：保守 risk-off，空仓。
        return {}

    closes = hist["close"].to_list()
    ma = sum(closes[-ma_window:]) / ma_window
    close_T = closes[-1]
    risk_on = (not timing) or (close_T > ma)
    if not risk_on:
        return {}

    members = members_fn(index_code, T.isoformat())
    if not members:
        return {}

    # PIT 流动性：只用 trade_date <= T 的成交额，按 ts_code 分组取最近 20 个
    # 交易日的均值（预先按 trade_date 排序，保证组内 tail(20) 为时间上最近的
    # 若干条，而非任意顺序）。members_fn 返回的成分股若在 price_daily 中无
    # 任何行情记录，自然不会出现在 px / liq 中，等价于与 price_daily 取交集。
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


def generate_trend_timing_products(
    out_dir: str,
    index_daily: pl.DataFrame,
    price_daily: pl.DataFrame,
    rebalance_dates: list[date],
    *,
    members_fn: Callable[[str, str], list[str]] | None = None,
    index_code: str = "000300.SH",
    ma_window: int = 200,
    top_n: int = 50,
    timing: bool = True,
) -> list[str]:
    """HS300（或指定指数）均线趋势择时 overlay：PIT 生成多期 weights 产物。

    Parameters
    ----------
    out_dir : str
        产物根目录，每个 rebalance_date 落 ``out_dir/<T.isoformat()>/``。
    index_daily : pl.DataFrame
        指数日线，须含 ``trade_date``、``close``。
    price_daily : pl.DataFrame
        个股日线，须含 ``trade_date``、``ts_code``、``amount``。
    rebalance_dates : list[date]
        调仓日列表。
    members_fn : Callable[[str, str], list[str]], optional
        ``(index_code, date_str_iso) -> [ts_code, ...]`` 成分股来源，默认走
        ``core.universe._load_index_members``（真实 Tushare）。测试注入避免
        网络依赖。
    index_code : str
        指数代码，默认 ``"000300.SH"``（HS300）。
    ma_window : int
        均线窗口（交易日数），默认 200。
    top_n : int
        risk-on 时按 ADV 取的成分股数量上限。
    timing : bool
        是否启用择时；``False`` 为基线（始终满仓 top_n，不受均线信号影响）。

    Returns
    -------
    list[str]
        每个 rebalance_date 对应的 run_dir（与 rebalance_dates 一一对应，顺序一致）。
    """
    members_fn = members_fn or _default_members
    idx = index_daily.sort("trade_date")

    run_dirs: list[str] = []
    for T in rebalance_dates:
        weights = _rebalance_weights(
            T,
            idx,
            price_daily,
            members_fn,
            index_code=index_code,
            ma_window=ma_window,
            top_n=top_n,
            timing=timing,
        )

        rd = Path(out_dir) / T.isoformat()
        rd.mkdir(parents=True, exist_ok=True)

        codes = list(weights)
        pl.DataFrame(
            {"ts_code": codes, "target_weight": [weights[c] for c in codes]},
            schema={"ts_code": pl.Utf8, "target_weight": pl.Float64},
        ).write_parquet(rd / "weights.parquet")

        manifest = {"signal_date": T.isoformat(), "status": "optimal"}
        (rd / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

        run_dirs.append(str(rd))

    return run_dirs

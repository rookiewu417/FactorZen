"""PIT-safe market-data preparation shared by all discovery entry points."""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import polars as pl


def _attach_in_universe(
    daily: pl.DataFrame, membership: pl.DataFrame
) -> pl.DataFrame:
    """Attach daily PIT membership without removing warm-up rows."""
    if membership.is_empty():
        return daily.with_columns(pl.lit(False).alias("in_universe"))

    td_dtype = daily.schema.get("trade_date")
    mem = membership.select(["trade_date", "ts_code"]).unique()
    if td_dtype == pl.Date:
        mem = mem.with_columns(pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d"))
    elif td_dtype is not None and td_dtype != pl.Utf8:
        mem = mem.with_columns(
            pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").cast(td_dtype)
        )

    mem = mem.with_columns(pl.lit(True).alias("in_universe"))
    out = daily.join(mem, on=["trade_date", "ts_code"], how="left")
    return out.with_columns(pl.col("in_universe").fill_null(False))


def expressions_need_intraday(
    exprs: Iterable[str],
    leaf_map: dict[str, str] | None = None,
) -> bool:
    """任一表达式引用 ``INTRADAY_FEATURES`` 叶子 → True。

    parse 失败的表达式跳过（不抛），避免畸形候选阻塞装帧。
    """
    from factorzen.core.feature_schema import INTRADAY_FEATURES
    from factorzen.discovery.expression import feature_names, parse_expr

    for e in exprs:
        if not e:
            continue
        try:
            node = parse_expr(str(e), leaf_map)
        except Exception:
            continue
        if feature_names(node) & INTRADAY_FEATURES:
            return True
    return False


def prepare_mining_daily(
    start: str,
    end: str,
    universe: str | None = None,
    lookback_days: int | None = None,
    out_meta: dict | None = None,
    *,
    intraday: bool = False,
    intraday_freq: str = "5min",
    intraday_version: str = "v1",
) -> pl.DataFrame:
    """Build the canonical A-share frame used by every discovery path.

    The returned frame contains adjusted prices, daily-basic leaves, announcement-date
    aligned fundamentals, flow data, and PIT universe membership.  Warm-up rows remain
    in the frame for rolling operators but have ``in_universe=False``.

    Named universes fail closed if their daily membership cannot be constructed.  A
    static/as-of fallback would introduce survivorship bias and is intentionally absent.

    ``intraday=True`` 时在 attach 链末尾 join 日内特征面板（``require=True``）；
    默认 ``False`` 保持 A 股日频链路字节级零回归。
    """
    from factorzen.core.universe import get_universe_membership, membership_hash
    from factorzen.daily.data.context import FactorDataContext
    from factorzen.discovery.search.random_search import search_space_max_lookback

    if lookback_days is None:
        lookback_days = search_space_max_lookback()

    universe_codes: list[str] | None = None
    membership: pl.DataFrame | None = None
    membership_mode: str | None = None
    membership_hash_value: str | None = None
    membership_n_rows: int | None = None
    if universe:
        try:
            membership = get_universe_membership(start, end, universe)
        except Exception as exc:
            raise ValueError(
                f"universe={universe!r} 的逐日 PIT membership 构造失败"
                f"（{type(exc).__name__}: {exc}）；"
                "拒绝回退静态成分生成可入库产物（会引入 look-ahead+幸存偏差，"
                "违反 PIT 铁律）。请回补指数成分数据，或改用 --universe all_a。"
            ) from exc
        universe_codes = membership["ts_code"].unique().to_list()
        if not universe_codes and universe != "all_a":
            raise ValueError(
                f"universe={universe!r} 在 [{start},{end}] 的逐日 PIT membership 为空"
                "（成分数据未回补到该窗口）；拒绝 as-of/静态成分回退生成可入库产物"
                "（会引入 look-ahead+幸存偏差，违反 PIT 铁律）。"
                "请回补指数成分数据，或改用 --universe all_a。"
            )
        membership_mode = "pit"
        membership_hash_value = membership_hash(membership)
        membership_n_rows = int(membership.height)

    if out_meta is not None:
        out_meta.update(
            {
                "membership_mode": membership_mode,
                "membership_hash": membership_hash_value,
                "membership_n_rows": membership_n_rows,
                "universe": universe,
            }
        )

    context = FactorDataContext(
        start=start,
        end=end,
        required_data=["daily", "daily_basic"],
        lookback_days=lookback_days,
        universe=universe_codes,
    )
    daily = context.daily.collect()
    basic = context.daily_basic.collect()
    if not basic.is_empty():
        daily = daily.join(basic, on=["trade_date", "ts_code"], how="left")

    # Mining and materialisation deliberately share these attach functions so the
    # same expression has identical leaves on both paths.
    from factorzen.daily.data.flows import attach_flows
    from factorzen.daily.data.pit import attach_fundamentals, attach_holders

    daily = attach_fundamentals(daily)
    daily = attach_holders(daily)
    daily = attach_flows(daily)

    if intraday:
        from factorzen.daily.data.intraday import attach_intraday

        daily = attach_intraday(
            daily,
            freq=intraday_freq,
            version=intraday_version,
            require=True,
            out_meta=out_meta,
        )
        # 面板 coverage 起点晚于扩窗取数起点 → 短历史由 leaf 预算/warmup 门处理，只 warn
        if out_meta is not None:
            panel_meta = out_meta.get("intraday_panel") or {}
            cov_start = panel_meta.get("coverage_start")
            exp_start = context.expanded_start
            if (
                cov_start is not None
                and str(cov_start).replace("-", "")[:8]
                > str(exp_start).replace("-", "")[:8]
            ):
                warnings.warn(
                    f"日内特征面板 coverage 起点 {cov_start} 晚于扩窗取数起点 "
                    f"{exp_start}；短历史叶子由 leaf 预算/warmup 门自然处理。",
                    stacklevel=2,
                )

    if membership is not None:
        daily = _attach_in_universe(daily, membership)
    return daily

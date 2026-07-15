"""PIT-safe market-data preparation shared by all discovery entry points."""

from __future__ import annotations

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


def prepare_mining_daily(
    start: str,
    end: str,
    universe: str | None = None,
    lookback_days: int | None = None,
    out_meta: dict | None = None,
) -> pl.DataFrame:
    """Build the canonical A-share frame used by every discovery path.

    The returned frame contains adjusted prices, daily-basic leaves, announcement-date
    aligned fundamentals, flow data, and PIT universe membership.  Warm-up rows remain
    in the frame for rolling operators but have ``in_universe=False``.

    Named universes fail closed if their daily membership cannot be constructed.  A
    static/as-of fallback would introduce survivorship bias and is intentionally absent.
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

    if membership is not None:
        daily = _attach_in_universe(daily, membership)
    return daily

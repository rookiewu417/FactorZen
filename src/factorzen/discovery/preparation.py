"""PIT-safe market-data preparation shared by all discovery entry points."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence

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
    """任一表达式引用 ``INTRADAY_FEATURES`` 或 ``ix_*`` 叶子 → True。

    parse 失败的表达式跳过（不抛），避免畸形候选阻塞装帧。
    ``ix_*`` 在默认 leaf_map 中不可 parse 时，用词法前缀回退识别。
    """
    from factorzen.core.feature_schema import INTRADAY_FEATURES
    from factorzen.discovery.expression import feature_names, parse_expr

    for e in exprs:
        if not e:
            continue
        s = str(e)
        # ix_* 可能不在默认 LEAF_FEATURES 中 → parse 失败；词法回退
        if "ix_" in s:
            return True
        try:
            node = parse_expr(s, leaf_map)
        except Exception:
            continue
        if feature_names(node) & INTRADAY_FEATURES:
            return True
    return False


def intraday_expr_leaf_names(
    exprs: Iterable[str],
    leaf_map: dict[str, str] | None = None,
) -> list[str]:
    """收集表达式引用的 ``ix_*`` 叶名（稳定去重顺序）；parse 失败则跳过。"""
    from factorzen.discovery.expression import feature_names, parse_expr
    from factorzen.discovery.intraday_expr import load_expr_registry
    from factorzen.discovery.operators import LEAF_FEATURES

    # 扩展 leaf_map：合并已注册 ix 名，便于 parse
    lm = dict(leaf_map) if leaf_map is not None else None
    if lm is None:
        try:
            reg = load_expr_registry()
            lm = {**LEAF_FEATURES, **{n: n for n in reg}}
        except Exception:
            lm = None

    found: list[str] = []
    seen: set[str] = set()
    for e in exprs:
        if not e:
            continue
        try:
            node = parse_expr(str(e), lm)
        except Exception:
            continue
        for name in sorted(feature_names(node)):
            if name.startswith("ix_") and name not in seen:
                seen.add(name)
                found.append(name)
    return found


# daily_basic 死重列：源表有、但 BASIC_FEATURES 叶子不映射（挖掘链无消费）。
# 正式回测/风控若需 pe/ps/… 请 slim=False 或走非 mining 装配路径。
_DAILY_BASIC_DEAD_COLS = frozenset({
    "pe", "ps", "dv_ratio", "total_share", "free_share",
})
# daily raw 死重：挖掘 IC 链不消费；派生叶子要的是 OHLC+pre_close+vol+amount。
_DAILY_RAW_DEAD_COLS = frozenset({"change", "pct_chg"})

# P4c：全 A 级行数上 ts_code Utf8 每帧 ~0.3–0.5G；转 Categorical 显著省内存。
# 与 compact 池同哲学——大帧自动开，小帧/测试默认 off 零回归。
KEYS_CATEGORICAL_ROWS_THRESHOLD = 4_000_000


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
    intraday_expr_leaves: Sequence[str] | None = None,
    slim: bool = True,
    categorical_keys: bool | None = None,
) -> pl.DataFrame:
    """Build the canonical A-share frame used by every discovery path.

    The returned frame contains adjusted prices, daily-basic leaves, announcement-date
    aligned fundamentals, flow data, and PIT universe membership.  Warm-up rows remain
    in the frame for rolling operators but have ``in_universe=False``.

    Named universes fail closed if their daily membership cannot be constructed.  A
    static/as-of fallback would introduce survivorship bias and is intentionally absent.

    ``intraday=True`` 时在 attach 链末尾 join 日内特征面板（``require=True``）；
    默认 ``False`` 保持 A 股日频链路字节级零回归。

    ``slim``（默认 ``True``）：挖掘帧列白名单裁剪——
    - ``daily_basic`` join 前只 select 键 + ``BASIC_FEATURES`` 10 列
      （``feature_schema.BASIC_FEATURES``：pe_ttm/pb/ps_ttm/dv_ttm/total_mv/circ_mv/
      turnover_rate/turnover_rate_f/volume_ratio/float_share），剔除 pe/ps/dv_ratio/
      total_share/free_share 死重；
    - 出口 drop ``change``/``pct_chg``（daily raw 死重；保留 raw OHLC+pre_close+vol+
      amount，派生叶子 vwap/amplitude/… 依赖它们）。
    ``slim=False`` 逃生口：完整旧帧（全 basic + change/pct_chg），供对照/非挖掘复用。
    白名单依据：叶子 schema + 挖掘链 grep 消费面（见帧瘦身 mapping D2）。

    ``categorical_keys``：出口将 ``ts_code`` cast 为 ``pl.Categorical``（P4c）。
    - ``None``（默认）→ 仅当 ``slim`` 且 ``height >= KEYS_CATEGORICAL_ROWS_THRESHOLD`` 自动开；
    - ``True`` / ``False`` 显式开关（测试与逃生）。
    membership/basic join 在 cast **之前**完成（外部 Utf8 帧无需预先对齐）。
    跨帧 join 消费点须对小帧 cast 对齐（见 scoring._align_join_key）；落盘再转回 Utf8。
    """
    from factorzen.core.feature_schema import BASIC_FEATURES
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
        if slim:
            # 白名单：键 + BASIC 叶子列（以 schema 为准；源表缺列则跳过）
            basic_keep = ["trade_date", "ts_code", *[
                c for c in BASIC_FEATURES if c in basic.columns
            ]]
            basic = basic.select(basic_keep)
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

    if intraday_expr_leaves:
        from factorzen.discovery.intraday_expr import attach_expr_leaves

        daily = attach_expr_leaves(
            daily, list(intraday_expr_leaves), require=True,
        )

    if membership is not None:
        daily = _attach_in_universe(daily, membership)

    if slim:
        drop = [c for c in _DAILY_RAW_DEAD_COLS if c in daily.columns]
        # 双保险：若 slim=False 路径残留或上游误 join 了 basic 死重
        drop += [c for c in _DAILY_BASIC_DEAD_COLS if c in daily.columns]
        if drop:
            daily = daily.drop(drop)

    # P4c：键 Categorical（在全部 join 之后，避免 membership/basic Utf8 侧 SchemaError）
    use_cat = (
        bool(categorical_keys)
        if categorical_keys is not None
        else (slim and daily.height >= KEYS_CATEGORICAL_ROWS_THRESHOLD)
    )
    if use_cat and "ts_code" in daily.columns and daily.schema["ts_code"] != pl.Categorical:
        daily = daily.with_columns(pl.col("ts_code").cast(pl.Categorical))
    return daily

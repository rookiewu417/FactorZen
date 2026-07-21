"""业绩预告 / 业绩快报事件叶对齐。

事件窗 fill-0 语义（第一期，供挖掘使用）:
- ``ann_date`` 当日**盘后**可得 → **严格下一交易日**起生效（effective_date）
- 非交易日公告：下一交易日开盘即可见（无额外 lag）
- 生效日起 ``EVENT_WINDOW`` 个交易日有值，窗外 0
- 重叠窗：join_asof backward 取最新生效公告（as-of last）
- 空源 → 全 null（leaf_health 可见）；有源 → 窗外/无事件 fill 0

叶子:
- ``fc_type_score`` / ``fc_surprise`` / ``fc_flag`` ← forecast
- ``express_yoy`` ← 真同比 express.n_income/yoy_net_profit − 1（clip ±5;
  yoy_net_profit 是上年同期净利**额**非同比%，见 attach_express docstring）
"""
from __future__ import annotations

from datetime import date

import polars as pl

from factorzen.core.feature_schema import EXPRESS_FEATURES, FORECAST_FEATURES

# 生效日起有效交易日数（含生效日）
EVENT_WINDOW = 20

# 预告 type → 序数量（真实 data distinct；未知 → 0）
FC_TYPE_SCORE: dict[str, float] = {
    "预增": 2.0,
    "扭亏": 2.0,
    "略增": 1.0,
    "续盈": 1.0,
    "略减": -1.0,
    "续亏": -1.0,
    "预减": -2.0,
    "首亏": -2.0,
    "不确定": 0.0,
    "其他": 0.0,
}

_FC_LEAF_COLS = sorted(FORECAST_FEATURES)
_EX_LEAF_COLS = sorted(EXPRESS_FEATURES)


def encode_fc_type(type_str: str | None) -> float:
    """type 字符串 → 序数量；未知/None → 0。"""
    if type_str is None:
        return 0.0
    return float(FC_TYPE_SCORE.get(str(type_str), 0.0))


def _surprise_expr() -> pl.Expr:
    """(p_change_min + p_change_max) / 2 / 100；单边非空取该侧；全空 → 0。"""
    lo = pl.col("p_change_min")
    hi = pl.col("p_change_max")
    mid = (
        pl.when(lo.is_not_null() & hi.is_not_null())
        .then((lo + hi) / 2.0)
        .when(lo.is_not_null())
        .then(lo)
        .when(hi.is_not_null())
        .then(hi)
        .otherwise(0.0)
    )
    return (mid / 100.0).alias("fc_surprise")


def _align_ann_date(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or "ann_date" not in df.columns:
        return df
    if df["ann_date"].dtype == pl.Utf8:
        return df.with_columns(
            pl.col("ann_date").str.strptime(pl.Date, "%Y%m%d", strict=False)
        )
    return df


def _trade_calendar(daily: pl.DataFrame) -> list[date]:
    """面板交易日 + 向前垫 ``EVENT_WINDOW`` 个交易日（窗跨面板起点时仍可算 dist）。

    优先 ``core.calendar.get_trade_dates``；失败则退回面板内 unique trade_date。
    """
    panel = daily["trade_date"].unique().sort().to_list()
    if not panel:
        return []
    try:
        from factorzen.core.calendar import get_trade_dates, prev_trade_date

        pad_start = prev_trade_date(panel[0], n=EVENT_WINDOW)
        start_s = pad_start.strftime("%Y%m%d")
        end_s = panel[-1].strftime("%Y%m%d")
        full = get_trade_dates(start_s, end_s)
        if full:
            return full
    except Exception:
        pass
    return panel


def _map_effective_dates(
    ann_dates: list[date | None],
    calendar: list[date],
) -> list[date | None]:
    """ann_date → 严格大于 ann_date 的第一个交易日；无则 None。"""
    if not calendar:
        return [None] * len(ann_dates)
    out: list[date | None] = []
    n = len(calendar)
    for ann in ann_dates:
        if ann is None:
            out.append(None)
            continue
        # 线性/二分：第一个 > ann
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if calendar[mid] <= ann:
                lo = mid + 1
            else:
                hi = mid
        out.append(calendar[lo] if lo < n else None)
    return out


def _prepare_events_with_effective(
    events: pl.DataFrame,
    calendar: list[date],
    *,
    value_exprs: list[pl.Expr],
    sort_keys: list[str],
) -> pl.DataFrame:
    """过滤 null ann → 映射 effective_date → 去重 keep last。"""
    events = _align_ann_date(events)
    events = events.filter(pl.col("ann_date").is_not_null())
    if events.is_empty():
        return events

    ann_list = events["ann_date"].to_list()
    eff = _map_effective_dates(ann_list, calendar)
    events = events.with_columns(pl.Series("effective_date", eff, dtype=pl.Date))
    events = events.filter(pl.col("effective_date").is_not_null())
    if events.is_empty():
        return events

    events = events.with_columns(value_exprs)
    # 同 (ts_code, effective_date) 多条：按 ann_date 取最新
    events = (
        events.sort(["ts_code", "effective_date", "ann_date"])
        .unique(subset=["ts_code", "effective_date"], keep="last", maintain_order=True)
        .select(["ts_code", "effective_date", *sort_keys])
    )
    return events


def _expand_window_asof(
    daily: pl.DataFrame,
    events: pl.DataFrame,
    leaf_cols: list[str],
    calendar: list[date],
) -> pl.DataFrame:
    """join_asof 取最新生效事件，距生效日 ≥ EVENT_WINDOW 交易日则置 0。"""
    if events.is_empty():
        return daily.with_columns([
            pl.lit(None, dtype=pl.Float64).alias(c)
            for c in leaf_cols if c not in daily.columns
        ])

    # 交易日 → 序号
    cal_df = pl.DataFrame({
        "trade_date": calendar,
        "_td_idx": list(range(len(calendar))),
    }).with_columns(pl.col("trade_date").cast(pl.Date))

    ev = events.join(
        cal_df.rename({"trade_date": "effective_date", "_td_idx": "_eff_idx"}),
        on="effective_date",
        how="left",
    )
    left = (
        daily.select(["ts_code", "trade_date"])
        .unique()
        .join(cal_df, on="trade_date", how="left")
        .sort(["ts_code", "trade_date"])
    )
    right = ev.sort(["ts_code", "effective_date"])

    joined = left.join_asof(
        right,
        left_on="trade_date",
        right_on="effective_date",
        by="ts_code",
        strategy="backward",
        check_sortedness=False,
    )
    # 距生效日交易日偏移：0 = 生效日；[0, EVENT_WINDOW) 有效
    dist = pl.col("_td_idx") - pl.col("_eff_idx")
    in_win = (
        pl.col("effective_date").is_not_null()
        & pl.col("_eff_idx").is_not_null()
        & (dist >= 0)
        & (dist < EVENT_WINDOW)
    )
    # 有源 fill-0：窗外 / 无事件 → 0；窗内取事件值
    panel = joined.with_columns([
        pl.when(in_win).then(pl.col(c)).otherwise(0.0).alias(c)
        for c in leaf_cols
    ]).select(["ts_code", "trade_date", *leaf_cols])
    # 若 daily 已有同名列（重入），先丢再 join
    drop = [c for c in leaf_cols if c in daily.columns]
    if drop:
        daily = daily.drop(drop)
    return daily.join(panel, on=["ts_code", "trade_date"], how="left")


def attach_forecast(
    daily: pl.DataFrame,
    forecast_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """把业绩预告事件叶 join 进日线帧。

    ``forecast_df is None`` 时从 ``data/raw/forecast/`` 读取；
    空源 → 叶子列全 null；有源 → 窗外/无事件 fill 0。
    """
    cols = _FC_LEAF_COLS
    if daily.is_empty() or "trade_date" not in daily.columns:
        return daily
    if forecast_df is None:
        forecast_df = _load_forecast()
    if forecast_df is None or forecast_df.is_empty():
        return _ensure_cols(daily, cols)

    calendar = _trade_calendar(daily)
    # 保证源列存在
    need = ["ts_code", "ann_date", "type"]
    if any(c not in forecast_df.columns for c in need):
        return _ensure_cols(daily, cols)
    for c in ("p_change_min", "p_change_max"):
        if c not in forecast_df.columns:
            forecast_df = forecast_df.with_columns(
                pl.lit(None, dtype=pl.Float64).alias(c)
            )

    type_score = (
        pl.col("type")
        .replace_strict(FC_TYPE_SCORE, default=0.0, return_dtype=pl.Float64)
        .alias("fc_type_score")
    )
    events = _prepare_events_with_effective(
        forecast_df,
        calendar,
        value_exprs=[
            type_score,
            _surprise_expr(),
            pl.lit(1.0).alias("fc_flag"),
        ],
        sort_keys=["fc_type_score", "fc_surprise", "fc_flag"],
    )
    if events.is_empty():
        return _ensure_cols(daily, cols)

    daily = _expand_window_asof(daily, events, cols, calendar)
    return _ensure_cols(daily, cols)


def attach_express(
    daily: pl.DataFrame,
    express_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """把业绩快报 ``express_yoy`` join 进日线帧（真同比 n_income/yoy_net_profit−1，20 日窗）。

    tushare express 的 ``yoy_net_profit`` 是**上年同期修正后净利润(元,绝对额)**而非同比%
    (2026-07-20 实测中位 8.77e7;旧实现 /100 编码的是盈利规模,与 pe_ttm rank 相关 −0.77,
    子集 IC +0.05 混入估值效应不是干净 PEAD)。真同比 = n_income/yoy_net_profit − 1;
    分母 ≤0 时同比语义破碎(负基数增长率符号反转)→ 0;clip ±5(扭亏/微基数极值会统治
    mul/zscore 类算子,rank 不受影响但算子组合受)。
    """
    cols = _EX_LEAF_COLS
    if daily.is_empty() or "trade_date" not in daily.columns:
        return daily
    if express_df is None:
        express_df = _load_express()
    if express_df is None or express_df.is_empty():
        return _ensure_cols(daily, cols)

    calendar = _trade_calendar(daily)
    if any(c not in express_df.columns for c in ("ts_code", "ann_date")):
        return _ensure_cols(daily, cols)
    for c in ("yoy_net_profit", "n_income"):
        if c not in express_df.columns:
            express_df = express_df.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))

    yoy = (
        pl.when(
            pl.col("n_income").is_not_null()
            & pl.col("yoy_net_profit").is_not_null()
            & (pl.col("yoy_net_profit") > 0)
        )
        .then(
            (pl.col("n_income") / pl.col("yoy_net_profit") - 1.0).clip(-5.0, 5.0)
        )
        .otherwise(0.0)
        .alias("express_yoy")
    )
    events = _prepare_events_with_effective(
        express_df,
        calendar,
        value_exprs=[yoy],
        sort_keys=["express_yoy"],
    )
    if events.is_empty():
        return _ensure_cols(daily, cols)

    daily = _expand_window_asof(daily, events, cols, calendar)
    return _ensure_cols(daily, cols)


def _load_forecast() -> pl.DataFrame | None:
    return _load_event_part(
        "forecast",
        ["ts_code", "ann_date", "end_date", "type", "p_change_min", "p_change_max"],
    )


def _load_express() -> pl.DataFrame | None:
    return _load_event_part(
        "express",
        ["ts_code", "ann_date", "end_date", "yoy_net_profit", "n_income"],
    )


def _load_event_part(part: str, cols: list[str]) -> pl.DataFrame | None:
    from factorzen.core.storage import scan_parquet

    try:
        lf = scan_parquet(part)
        names = lf.collect_schema().names()
        have = [c for c in cols if c in names]
        if "ts_code" not in have or "ann_date" not in have:
            return None
        return lf.select(have).collect()
    except Exception:
        return None


def _ensure_cols(daily: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    missing = [c for c in cols if c not in daily.columns]
    if missing:
        daily = daily.with_columns([
            pl.lit(None, dtype=pl.Float64).alias(c) for c in missing
        ])
    return daily

"""Point-In-Time 财务数据对齐。确保月末调仓不使用未公告的财报。"""

from datetime import date

import polars as pl

from factorzen.core.feature_schema import FUNDAMENTAL_FEATURES, HOLDER_FEATURES


def pit_align(
    fina_df: pl.DataFrame,
    snapshot_dates: list[date],
) -> pl.DataFrame:
    """对财务数据做 Point-In-Time 对齐。

    对每个月频快照日期，找出每只股票「最新已公告」的财务报告——
    即 ann_date <= snapshot_date 中 end_date 最大的那条。

    同 ts_code 同 end_date 多条（更正公告）时的 tie-break：与旧实现一致——
    按 ``["ts_code", "end_date"]`` 降序 end_date 后 ``group_by().first()``，
    polars 稳定排序下同 end_date 取原相对顺序第一条。实现上以
    ``(end_date desc, 原行序 asc)`` 为优劣键，按 ann_date 做 running-best
    状态转移后 ``join_asof``，与逐快照日循环语义等价。

    Args:
        fina_df: 财务数据，必须含 ts_code, end_date(Date), ann_date(Date) 及指标列
        snapshot_dates: 月频快照日列表(升序)

    Returns:
        pl.DataFrame, 列: snapshot_date, ts_code, end_date, ann_date, 指标列
    """
    if fina_df.is_empty() or not snapshot_dates:
        return pl.DataFrame()

    # ann_date 在存储中为 String "YYYYMMDD"，比较前统一转成 Date
    if fina_df["ann_date"].dtype == pl.Utf8:
        fina_df = fina_df.with_columns(
            pl.col("ann_date").str.strptime(pl.Date, "%Y%m%d", strict=False)
        )

    # 过滤掉 ann_date 为 null 的记录（未公告的财报）
    fina_df = fina_df.filter(pl.col("ann_date").is_not_null())
    if fina_df.is_empty():
        return pl.DataFrame()

    # 原相对顺序：稳定排序 + group_by.first 的 tie-break 键
    fina = fina_df.with_row_index("_ord")

    # 优劣秩：end_date 越大越好，同 end_date 原行序越靠前越好 → rank 越小越好
    ranked = (
        fina.sort(["ts_code", "end_date", "_ord"], descending=[False, True, False])
        .with_columns(pl.int_range(pl.len()).over("ts_code").alias("_rank"))
    )

    # 按公告日推进 running-best：每当出现更优秩时记录一次状态转移
    events = ranked.sort(["ts_code", "ann_date", "_ord"]).with_columns(
        pl.col("_rank").cum_min().over("ts_code").alias("_best_rank")
    )
    states = events.filter(pl.col("_rank") == pl.col("_best_rank")).drop(
        ["_ord", "_rank", "_best_rank"]
    )
    if states.is_empty():
        return pl.DataFrame()

    # 快照 × 股票，一次 join_asof(ann_date <= snapshot_date) 取当前 best 行
    snaps = pl.DataFrame(
        {"snapshot_date": snapshot_dates},
        schema={"snapshot_date": pl.Date},
    )
    left = (
        states.select("ts_code")
        .unique()
        .join(snaps, how="cross")
        .sort(["ts_code", "snapshot_date"])
    )
    right = states.sort(["ts_code", "ann_date"])

    aligned = left.join_asof(
        right,
        left_on="snapshot_date",
        right_on="ann_date",
        by="ts_code",
        strategy="backward",
        check_sortedness=False,  # by 分组时 polars 无法校验 sortedness
    )
    # 无任何 ann_date <= snapshot 的股票-日 → 该日不输出
    aligned = aligned.filter(pl.col("ann_date").is_not_null())
    if aligned.is_empty():
        return pl.DataFrame()

    # 列序与旧实现大致一致：指标列在前，snapshot_date 在后（下游按名取值）
    out_cols = [c for c in fina_df.columns if c in aligned.columns]
    if "snapshot_date" not in out_cols:
        out_cols.append("snapshot_date")
    # 补齐 join 可能带出的其余列
    for c in aligned.columns:
        if c not in out_cols and c not in ("_ord", "_rank", "_best_rank"):
            out_cols.append(c)
    return aligned.select(out_cols)


# 挖掘/物化路径共用的基本面叶子——单一真源在 operators.FUNDAMENTAL_FEATURES，此处只排序取用
# （防「注册的叶子」与「attach 的列」漂移）。fina_indicator 字段名即叶子名。
def _fundamental_cols() -> list[str]:
    return sorted(FUNDAMENTAL_FEATURES)


def attach_fundamentals(
    daily: pl.DataFrame,
    fina_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """把**按公告日 PIT 对齐**的基本面(roe/assets_yoy)join 进日线帧，作为叶子列。

    对 ``daily`` 里每个交易日 t，取 ``ann_date <= t`` 中 end_date 最大的那份财报（复用
    `pit_align`，与月频内置因子同一套 PIT 语义，绝不漂移）。t 之后才公告的报告**不会**泄漏
    到 t（铁律#1 无未来函数，见 test_fundamentals_pit）。

    ``fina_df``：财务帧（列 ts_code/end_date/ann_date + 指标），``None`` 时从 finance
    parquet 读取（优先 ``finance_fina_indicator``——当前 fetch_finance 写此分区、含全套质量/成长
    字段；回落旧 ``finance`` 分区——只有 roe/assets_yoy）。缺数据 / 读取失败 → **原样返回
    daily**（离线/CI 不崩），缺的基本面列补 null（表达式引用到时得到 null 而非 KeyError）。

    挖掘(`prepare_mining_daily`)与物化(`ExpressionFactor.compute`)两条路都调它，保证
    同一因子在挖掘与回测里逐值一致。
    """
    cols = _fundamental_cols()
    if daily.is_empty() or "trade_date" not in daily.columns:
        return daily
    if fina_df is None:
        fina_df = _load_fina(cols)
    if fina_df is None or fina_df.is_empty():
        return _ensure_fundamental_cols(daily)

    snapshot_dates = daily["trade_date"].unique().sort().to_list()
    pit = pit_align(fina_df, snapshot_dates)
    if pit.is_empty():
        return _ensure_fundamental_cols(daily)

    have = [c for c in cols if c in pit.columns]
    pit = pit.select(["snapshot_date", "ts_code", *have]).rename({"snapshot_date": "trade_date"})
    return _ensure_fundamental_cols(daily.join(pit, on=["trade_date", "ts_code"], how="left"))


def _load_fina(cols: list[str]) -> pl.DataFrame | None:
    """从 finance parquet 读财务帧：优先 finance_fina_indicator(全字段)，回落 finance(旧, roe/assets_yoy)。"""
    from factorzen.core.storage import scan_parquet
    for part in ("finance_fina_indicator", "finance"):
        try:
            lf = scan_parquet(part).filter(pl.col("end_date").is_not_null())
            names = lf.collect_schema().names()
            have = [c for c in cols if c in names]
            if not have:
                continue
            return lf.select(["ts_code", "end_date", "ann_date", *have]).collect()
        except Exception:
            continue
    return None


def _ensure_fundamental_cols(daily: pl.DataFrame) -> pl.DataFrame:
    """补齐缺失的基本面列为 null——表达式引用到未成功 attach 的叶子时得到 null 而非崩溃。"""
    missing = [c for c in _fundamental_cols() if c not in daily.columns]
    if missing:
        daily = daily.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in missing])
    return daily


# ── 股东户数（低频 PIT，与 fina 同款 pit_align）──────────────────────────────


def _holder_cols() -> list[str]:
    return sorted(HOLDER_FEATURES)


def attach_holders(
    daily: pl.DataFrame,
    holder_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """把**按公告日 PIT 对齐**的股东户数 join 进日线帧。

    对 ``daily`` 每个交易日 t，取 ``ann_date <= t`` 中 end_date 最大的一期（复用
    `pit_align`，与 fina 同套 PIT 语义）。公告间隔内自然前向持有（PIT 结果，非 ffill）。

    叶子：
    - ``holder_num``：最新一期股东户数（户）
    - ``holder_num_chg``：相邻两期环比 ``(本期-上期)/上期``——在源数据整理阶段按
      (ts_code, end_date 升序) 算好，随本期 ann_date 生效。低频 PIT 上 ts_delta 会失真
      （平台期多为 0、只在公告日跳变），故变化率必须期际算（与 fina *_yoy 同理）。

    无数据股票 → null（诚实缺测）。``holder_df is None`` 时从 ``stk_holdernumber``
    parquet 读；缺数据/失败 → 原样返回并补 null 列。
    """
    cols = _holder_cols()
    if daily.is_empty() or "trade_date" not in daily.columns:
        return daily
    if holder_df is None:
        holder_df = _load_holder()
    if holder_df is None or holder_df.is_empty():
        return _ensure_holder_cols(daily)

    prepared = _prepare_holder_df(holder_df)
    if prepared.is_empty():
        return _ensure_holder_cols(daily)

    snapshot_dates = daily["trade_date"].unique().sort().to_list()
    pit = pit_align(prepared, snapshot_dates)
    if pit.is_empty():
        return _ensure_holder_cols(daily)

    have = [c for c in cols if c in pit.columns]
    pit = pit.select(["snapshot_date", "ts_code", *have]).rename({"snapshot_date": "trade_date"})
    return _ensure_holder_cols(daily.join(pit, on=["trade_date", "ts_code"], how="left"))


def _prepare_holder_df(holder_df: pl.DataFrame) -> pl.DataFrame:
    """源数据整理：规范化日期 + 按 (ts_code, end_date 升序) 算 holder_num_chg。"""
    df = holder_df
    # end_date 可能是 String（注入）或 Date（parquet）
    if "end_date" in df.columns and df["end_date"].dtype == pl.Utf8:
        df = df.with_columns(pl.col("end_date").str.strptime(pl.Date, "%Y%m%d", strict=False))
    if "ann_date" in df.columns and df["ann_date"].dtype == pl.Utf8:
        df = df.with_columns(pl.col("ann_date").str.strptime(pl.Date, "%Y%m%d", strict=False))
    if "holder_num" not in df.columns:
        return pl.DataFrame()
    # 期际环比：按 end_date 升序，(本期-上期)/上期；首期 null
    df = (
        df.filter(pl.col("end_date").is_not_null() & pl.col("holder_num").is_not_null())
        .sort(["ts_code", "end_date"])
        .with_columns(
            pl.when(pl.col("holder_num").shift(1).over("ts_code").abs() > 1e-12)
            .then(
                (pl.col("holder_num") - pl.col("holder_num").shift(1).over("ts_code"))
                / pl.col("holder_num").shift(1).over("ts_code")
            )
            .otherwise(None)
            .alias("holder_num_chg")
        )
    )
    return df


def _load_holder() -> pl.DataFrame | None:
    from factorzen.core.storage import scan_parquet
    try:
        lf = scan_parquet("stk_holdernumber").filter(pl.col("end_date").is_not_null())
        names = lf.collect_schema().names()
        need = ["ts_code", "end_date", "ann_date", "holder_num"]
        if not all(c in names for c in need):
            return None
        return lf.select(need).collect()
    except Exception:
        return None


def _ensure_holder_cols(daily: pl.DataFrame) -> pl.DataFrame:
    missing = [c for c in _holder_cols() if c not in daily.columns]
    if missing:
        daily = daily.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in missing])
    return daily

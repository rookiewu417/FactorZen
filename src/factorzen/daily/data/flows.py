"""日频信号（资金流 / 北向持股 / 两融 / 龙虎榜）对齐。

与基本面不同,资金流/北向本身就是**日频 point-in-time**,当日数据当日可得,无需
PIT 季度对齐(pit_align)。两融/龙虎榜例外（披露时点滞后）:
- **两融**：T 日两融数据 T+1 早间披露 → lag(1)；非标的 join 得 null（不填 0）。
- **龙虎榜**：t 日龙虎榜 t 日盘后（晚间）披露 → 保守 lag(1)；**条件 fill 0**：
  源表已知日（真实行 ∪ ``__EMPTY__`` sentinel）内未上榜 = 确定没上榜 → fill 0；
  **未拉取日保持 null**（没拉数据 ≠ 没上榜）。全空源 → 全 null（覆盖审计诚实）。

lag 均在 attach 层按 ts_code 组内交易日序 ``shift(1)`` 结构性完成,不靠表达式作者写 delay。

挖掘(`prepare_mining_daily`)与物化(`ExpressionFactor.compute`)两条路都调
`attach_flows`,保证同一因子逐值一致(陷阱#2)。

叶子有效期(2026-07 缓存审计):
- ``net_mf_amount``(moneyflow):2016-06 起逐日完整(≈244 交易日/年)。
- ``north_ratio``(hk_hold 北向持股占比):**有效起点 2016-06-29**(此前 API 返回空,非 bug);
  缓存存在 **2017-2018 空档**(需增量回补,Tushare 限速);2019+ 逐日完整(≈236 交易日/年,
  因港股通假期少于 A 股日历)。**2024-08-16 后港交所停止逐日披露个股北向持股**——缓存里
  2024-09 起覆盖股票数骤降(约 1/5)、连续重复值占比升高(2024上 0.245 → 2024下 0.380),属
  降频/部分披露。此后缺失日 join 得 null(``attach_flows`` 只 left-join,**不 ffill 伪造**,
  PIT 诚实);用 ``north_ratio`` 的因子在 2024-09 后有效截面变薄,评估时会自然缩样。
- 两融叶子(``margin_ratio``/``margin_buy_ratio``/``margin_balance``/``short_balance``):
  仅融资融券标的有数据(全 A 约一半;CSI300 覆盖通常 >90%);非标的 join 得 null(不填 0)。
  单位:rzye/rzmre=**元**;circ_mv=**万元**→比前 ×1e4;amount=**千元**→比前 ×1e3。
  变化率/滚动交给算子库(ts_*),叶子保持原子性。
- 龙虎榜叶子(``top_list_net_buy``/``top_list_flag``):全市场事件;已知日未上榜 fill 0,
  未拉取日 null → leaf_health 对事件叶子恢复视力。单位:net_amount=**万元**→×1e4;
  amount=**千元**→×1e3;同日多原因 sum net_amount。
"""
from __future__ import annotations

import polars as pl

# 叶子名 → (缓存分区, 源列名)。北向 ratio 重命名为 north_ratio,避免与通用名冲突。
# 两融/龙虎榜由专用 _attach_* 处理(需 lag/比值/fill0),不进此表。
_FLOW_SOURCES: dict[str, tuple[str, str]] = {
    "net_mf_amount": ("moneyflow", "net_mf_amount"),  # 主力净流入额(万元)
    "north_ratio": ("hk_hold", "ratio"),              # 北向持股占比(%)
}

# 两融源列(margin_detail)。rzye/rzmre 单位元;rqyl 单位股。
_MARGIN_SRC_COLS = ["rzye", "rzmre", "rqyl"]
# 龙虎榜源列。net_amount 万元; amount 千元(Tushare top_list 口径)。
_TOPLIST_SRC_COLS = ["net_amount", "amount"]
# circ_mv 万元→元; amount 千元→元; top_list net_amount 万元→元
_CIRC_MV_TO_YUAN = 1e4
_AMOUNT_TO_YUAN = 1e3
_NET_AMOUNT_TO_YUAN = 1e4  # top_list net_amount 万元
_TOPLIST_EMPTY_CODE = "__EMPTY__"  # fetch 空日 sentinel，attach 过滤


def attach_flows(daily: pl.DataFrame, *, injected: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """把资金流/北向/两融/龙虎榜日频信号按 (trade_date, ts_code) join 进日线帧,作为叶子列。

    缺数据 / 读取失败 → 原样返回,缺的叶子补 null(表达式引用到时得 null 而非 KeyError)。
    ``injected``:``{分区名: DataFrame}`` 供测试注入,绕过 parquet 读取。
    两融/龙虎榜在 join 前对源列做组内 lag(1)；龙虎榜对已知日条件 fill 0。
    """
    from factorzen.discovery.operators import FLOW_FEATURES, MARGIN_FEATURES, TOPLIST_FEATURES

    if daily.is_empty() or "trade_date" not in daily.columns:
        return daily
    injected = injected or {}
    # 仅处理 plain flow 源(两融/龙虎榜走专用路径)
    plain = sorted(FLOW_FEATURES - MARGIN_FEATURES - TOPLIST_FEATURES)
    by_part: dict[str, list[tuple[str, str]]] = {}
    for leaf in plain:
        part, col = _FLOW_SOURCES[leaf]
        by_part.setdefault(part, []).append((leaf, col))

    for part, pairs in by_part.items():
        src_cols = [c for _, c in pairs]
        df = injected.get(part)
        if df is None:
            df = _load_flow(part, src_cols)
        if df is None or df.is_empty():
            continue
        sel = df.select(["ts_code", "trade_date", *[c for c in src_cols if c in df.columns]])
        sel = sel.rename({col: leaf for leaf, col in pairs if col in sel.columns})
        sel = _align_trade_date(sel, daily)
        daily = daily.join(sel, on=["trade_date", "ts_code"], how="left")

    daily = _attach_margin(daily, injected=injected)
    daily = _attach_toplist(daily, injected=injected)
    return _ensure_flow_cols(daily)


def _attach_margin(daily: pl.DataFrame, *, injected: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """两融叶子:源日**同日**算比值 → 全部叶子整体 lag(1) → left-join。

    披露时点:T 日两融 T+1 早间披露 → t 日信号 = t-1 日的两融状态/参与度。
    比值在**源日同日**内完成(rzye(t')/circ_mv(t')、rzmre(t')/amount(t'))再 lag,
    避免「t-1 分子 / t 分母」的跨日噪声;分母取自 daily 帧同日行(join 后计算)。
    非融资标的无源行 → join 后 null(诚实缺测,不填 0)。
    单位:rzye/rzmre 元;circ_mv 万元(×1e4);amount 千元(×1e3)。
    """
    df = injected.get("margin_detail")
    if df is None:
        df = _load_flow("margin_detail", _MARGIN_SRC_COLS)
    if df is None or df.is_empty():
        return daily
    have = [c for c in _MARGIN_SRC_COLS if c in df.columns]
    if not have:
        return daily
    sel = df.select(["ts_code", "trade_date", *have])
    sel = _align_trade_date(sel, daily)

    # 同日分母:从 daily 帧取源日的 circ_mv/amount(仅取存在的列)
    denom_cols = [c for c in ("circ_mv", "amount") if c in daily.columns]
    if denom_cols:
        sel = sel.join(
            daily.select(["trade_date", "ts_code", *denom_cols]),
            on=["trade_date", "ts_code"], how="left",
        )

    exprs: list[pl.Expr] = []
    if "rzye" in have:
        exprs.append(pl.col("rzye").alias("margin_balance"))
    if "rqyl" in have:
        exprs.append(pl.col("rqyl").alias("short_balance"))
    # margin_ratio = 融资余额(元)/流通市值(元),同日:rzye(t')/(circ_mv(t')×1e4)
    if "rzye" in have and "circ_mv" in denom_cols:
        exprs.append(
            pl.when(pl.col("rzye").is_not_null()
                    & pl.col("circ_mv").is_not_null()
                    & (pl.col("circ_mv").abs() > 1e-12))
            .then(pl.col("rzye") / (pl.col("circ_mv") * _CIRC_MV_TO_YUAN))
            .otherwise(None).alias("margin_ratio")
        )
    # margin_buy_ratio = 融资买入额(元)/成交额(元),同日:rzmre(t')/(amount(t')×1e3)
    if "rzmre" in have and "amount" in denom_cols:
        exprs.append(
            pl.when(pl.col("rzmre").is_not_null()
                    & pl.col("amount").is_not_null()
                    & (pl.col("amount").abs() > 1e-12))
            .then(pl.col("rzmre") / (pl.col("amount") * _AMOUNT_TO_YUAN))
            .otherwise(None).alias("margin_buy_ratio")
        )
    sel = sel.with_columns(exprs)
    leaf_cols = [c for c in ("margin_balance", "short_balance",
                             "margin_ratio", "margin_buy_ratio") if c in sel.columns]
    if not leaf_cols:
        return daily
    # 全部叶子整体 lag(1):按 ts_code 组内交易日序 shift,首日 null(披露 T+1)
    sel = (sel.select(["ts_code", "trade_date", *leaf_cols])
              .sort(["ts_code", "trade_date"])
              .with_columns([pl.col(c).shift(1).over("ts_code") for c in leaf_cols]))
    return daily.join(sel, on=["trade_date", "ts_code"], how="left")


def _attach_toplist(daily: pl.DataFrame, *, injected: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """龙虎榜叶子:同日聚合 → 比值 → join daily → **条件 fill 0** → 整列 lag(1)。

    披露时点:t 日龙虎榜 t 日盘后(晚间)披露 → t 日信号只能用 t-1 上榜信息。

    条件 fill 0（事件叶子诚实缺测）:
    - 已知日集合 = 源表 distinct trade_date（真实行 ∪ ``__EMPTY__`` sentinel）
    - ``trade_date ∈ 已知日`` 且 join 缺失 → fill 0（确定没上榜）
    - ``trade_date ∉ 已知日`` → 保持 null（未拉取 ≠ 没上榜）
    - 全空源 → 全 null（覆盖审计诚实；leaf_health 可见缺口）
    与两融「非标的=null」不同:已知日内未上榜是真实零事件。

    单位(Tushare top_list):net_amount=**万元**→×1e4 元;amount=**千元**→×1e3 元。
    同日多条上榜原因:net_amount sum,amount first(同股同日成交额相同)。
    """
    df = injected.get("top_list")
    if df is None:
        # 保留 sentinel 行以便构建已知日集合
        df = _load_flow("top_list", _TOPLIST_SRC_COLS, keep_toplist_sentinel=True)
    leaf_cols = ["top_list_net_buy", "top_list_flag"]
    if df is None or df.is_empty():
        return daily.with_columns([
            pl.lit(None, dtype=pl.Float64).alias(c)
            for c in leaf_cols if c not in daily.columns
        ])

    df = _align_trade_date(df, daily)
    # 已知日 = 真实行 ∪ sentinel 空日（fetch 已拉标记）
    known_dates = df.select("trade_date").unique()

    # 过滤 sentinel 后再做事件聚合
    real = (
        df.filter(pl.col("ts_code") != _TOPLIST_EMPTY_CODE)
        if "ts_code" in df.columns else df
    )
    have = [c for c in _TOPLIST_SRC_COLS if c in real.columns]
    if not have or "net_amount" not in have:
        # 无事件列但可能有 sentinel 已知日 → 已知日 fill 0，未知日 null
        daily = daily.with_columns([
            pl.lit(None, dtype=pl.Float64).alias(c)
            for c in leaf_cols if c not in daily.columns
        ])
        known_set = known_dates["trade_date"].to_list()
        is_known = pl.col("trade_date").is_in(known_set)
        daily = daily.with_columns([
            pl.when(is_known).then(0.0).otherwise(pl.col(c)).alias(c)
            for c in leaf_cols
        ])
        return (
            daily.sort(["ts_code", "trade_date"])
            .with_columns([pl.col(c).shift(1).over("ts_code") for c in leaf_cols])
        )

    # 同日多原因聚合
    agg_exprs: list[pl.Expr] = [pl.col("net_amount").sum().alias("net_amount")]
    if "amount" in have:
        agg_exprs.append(pl.col("amount").first().alias("amount"))
    sel = real.group_by(["ts_code", "trade_date"]).agg(agg_exprs)

    # 比值: (net_amount 万元 × 1e4) / (amount 千元 × 1e3)
    if "amount" in sel.columns:
        sel = sel.with_columns(
            pl.when(pl.col("amount").is_not_null() & (pl.col("amount").abs() > 1e-12))
            .then(
                (pl.col("net_amount") * _NET_AMOUNT_TO_YUAN)
                / (pl.col("amount") * _AMOUNT_TO_YUAN)
            )
            .otherwise(None)
            .alias("top_list_net_buy"),
            pl.lit(1.0).alias("top_list_flag"),
        )
    else:
        sel = sel.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("top_list_net_buy"),
            pl.lit(1.0).alias("top_list_flag"),
        )

    sel = sel.select(["ts_code", "trade_date", *leaf_cols])
    # 1) left-join 事件 2) 已知日缺失 fill 0，未知日保持 null 3) lag(1)（首日/未知前值保持 null）
    daily = daily.join(sel, on=["trade_date", "ts_code"], how="left")
    known_set = known_dates["trade_date"].to_list()
    is_known = pl.col("trade_date").is_in(known_set)
    daily = daily.with_columns([
        pl.when(is_known & pl.col(c).is_null())
        .then(0.0)
        .otherwise(pl.col(c))
        .alias(c)
        for c in leaf_cols
    ])
    daily = (
        daily.sort(["ts_code", "trade_date"])
        .with_columns([pl.col(c).shift(1).over("ts_code") for c in leaf_cols])
    )
    return daily


def _load_flow(
    part: str,
    cols: list[str],
    *,
    keep_toplist_sentinel: bool = False,
) -> pl.DataFrame | None:
    from factorzen.core.storage import scan_parquet
    try:
        lf = scan_parquet(part)
        names = lf.collect_schema().names()
        have = [c for c in cols if c in names]
        if not have:
            return None
        out = lf.select(["ts_code", "trade_date", *have]).collect()
        # top_list：默认保留 sentinel 供已知日集合；其他分区无此开关
        if part == "top_list" and not keep_toplist_sentinel and "ts_code" in out.columns:
            out = out.filter(pl.col("ts_code") != _TOPLIST_EMPTY_CODE)
        return out
    except Exception:
        return None


def _align_trade_date(sel: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame:
    """把 flow 帧的 trade_date 类型对齐到 daily(通常都是 Date;注入的 String 转 Date)。"""
    if sel["trade_date"].dtype == daily["trade_date"].dtype:
        return sel
    if sel["trade_date"].dtype == pl.Utf8:
        return sel.with_columns(pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d", strict=False))
    return sel


def _ensure_flow_cols(daily: pl.DataFrame) -> pl.DataFrame:
    from factorzen.discovery.operators import FLOW_FEATURES
    missing = [c for c in sorted(FLOW_FEATURES) if c not in daily.columns]
    if missing:
        daily = daily.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in missing])
    return daily

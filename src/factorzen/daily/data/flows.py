"""日频信号（资金流 / 北向持股 / 两融）对齐。

与基本面不同,资金流/北向本身就是**日频 point-in-time**,当日数据当日可得,无需
PIT 季度对齐(pit_align)。两融例外：**T 日两融数据 T+1 早间披露**(交易所/券商惯例),
t 日信号只能用 t-1 日两融——``attach_flows`` 在 join 前按 ts_code 组内交易日序
``shift(1)`` 结构性完成 lag,不靠表达式作者记得写 delay。

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
"""
from __future__ import annotations

import polars as pl

# 叶子名 → (缓存分区, 源列名)。北向 ratio 重命名为 north_ratio,避免与通用名冲突。
# 两融叶子由 _attach_margin 单独处理(需 lag + 比值换算),不进此表。
_FLOW_SOURCES: dict[str, tuple[str, str]] = {
    "net_mf_amount": ("moneyflow", "net_mf_amount"),  # 主力净流入额(万元)
    "north_ratio": ("hk_hold", "ratio"),              # 北向持股占比(%)
}

# 两融源列(margin_detail)。rzye/rzmre 单位元;rqyl 单位股。
_MARGIN_SRC_COLS = ["rzye", "rzmre", "rqyl"]
# circ_mv 万元→元; amount 千元→元(与 daily / daily_basic 口径一致)
_CIRC_MV_TO_YUAN = 1e4
_AMOUNT_TO_YUAN = 1e3


def attach_flows(daily: pl.DataFrame, *, injected: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """把资金流/北向/两融日频信号按 (trade_date, ts_code) join 进日线帧,作为叶子列。

    缺数据 / 读取失败 → 原样返回,缺的叶子补 null(表达式引用到时得 null 而非 KeyError)。
    ``injected``:``{分区名: DataFrame}`` 供测试注入,绕过 parquet 读取。
    两融在 join 前对源列做组内 lag(1)(披露 T+1)。
    """
    from factorzen.discovery.operators import FLOW_FEATURES, MARGIN_FEATURES

    if daily.is_empty() or "trade_date" not in daily.columns:
        return daily
    injected = injected or {}
    # 仅处理非两融的 flow 源(两融走 _attach_margin)
    plain = sorted(FLOW_FEATURES - MARGIN_FEATURES)
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


def _load_flow(part: str, cols: list[str]) -> pl.DataFrame | None:
    from factorzen.core.storage import scan_parquet
    try:
        lf = scan_parquet(part)
        names = lf.collect_schema().names()
        have = [c for c in cols if c in names]
        if not have:
            return None
        return lf.select(["ts_code", "trade_date", *have]).collect()
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

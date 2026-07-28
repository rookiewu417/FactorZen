"""PIT-safe market-data preparation shared by all discovery entry points."""

from __future__ import annotations

import datetime as _dt
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


# ── P6：keep_leaves 列白名单 ────────────────────────────────────────────────
#
# 派生叶 → 它在**装配帧**里需要的 raw 列。单一真源是
# `discovery/derived.py:add_derived_columns`；改那边必须同步本表
# （守卫：tests/agents/test_frame_diet.py::test_p6_derived_deps_table_is_exact
#   逐条验充分性与必要性）。
DERIVED_LEAF_RAW_DEPS: dict[str, frozenset[str]] = {
    "vwap": frozenset({"amount", "vol"}),
    "log_vol": frozenset({"vol"}),
    "ret_1d": frozenset({"close_adj"}),
    "amplitude": frozenset({"high", "low", "pre_close"}),
    "intraday_ret": frozenset({"close", "open"}),
    "overnight_ret": frozenset({"open", "pre_close"}),
}

# 跨源叶依赖：某个叶的计算要用到**别的源**产出的列，裁掉分母不会报错、只会让该叶
# 静默消失或全 null（`flows._attach_margin` 的 denom_cols 空分支），故必须显式声明。
CROSS_SOURCE_LEAF_DEPS: dict[str, frozenset[str]] = {
    "margin_ratio": frozenset({"circ_mv"}),      # rzye / (circ_mv × 1e4)
    "margin_buy_ratio": frozenset({"amount"}),   # rzmre / (amount × 1e3)
}

# 无论白名单怎么写都**必须**保留的列。理由：装配帧的下游 `_preprocess_daily` →
# `add_derived_columns` 无条件引用这些 raw 列（缺列直接 ColumnNotFoundError）；
# 且 `_preprocess_daily` 在 ``*_adj`` 缺失时会拿**未复权**价补别名——那是静默的
# 值漂移，比崩掉更坏，所以复权价列一并锁死。
MINING_MANDATORY_COLS: frozenset[str] = frozenset({
    "trade_date", "ts_code", "in_universe",
    "open", "high", "low", "close", "pre_close", "vol", "amount",
    "open_adj", "high_adj", "low_adj", "close_adj",
})


def _trimmable_leaf_cols() -> frozenset[str]:
    """白名单可裁掉的叶列全集（出口兜底 select 只敢动这些列）。"""
    from factorzen.core.feature_schema import (
        BASIC_FEATURES,
        EXPRESS_FEATURES,
        FLOW_FEATURES,
        FORECAST_FEATURES,
        FUNDAMENTAL_FEATURES,
        HOLDER_FEATURES,
        INTRADAY_FEATURES,
    )

    return frozenset(
        set(BASIC_FEATURES)
        | set(FUNDAMENTAL_FEATURES)
        | set(HOLDER_FEATURES)
        | set(FLOW_FEATURES)
        | set(FORECAST_FEATURES)
        | set(EXPRESS_FEATURES)
        | set(INTRADAY_FEATURES)
    ) - MINING_MANDATORY_COLS


def resolve_keep_columns(
    keep_leaves: Iterable[str],
    leaf_map: dict[str, str] | None = None,
) -> set[str]:
    """叶名白名单 → 装配帧保留列集。

    ``keep_leaves`` 取 `expression.feature_names` 的输出口径（叶**名**，如 ``close`` /
    ``grossprofit_margin`` / ``i_rv``），经 ``leaf_map`` 映射到列名（``close`` → ``close_adj``）；
    映射外的名字（``ix_*``、直接给的列名）原样保留。

    返回列集 = 强制核心列 ∪ 白名单叶列 ∪ 派生叶 raw 依赖 ∪ 跨源叶依赖。
    **宁多勿漏**：少给一列会让叶子静默变 null（不报错），多给一列只是多占字节。
    """
    from factorzen.discovery.operators import LEAF_FEATURES

    lm = leaf_map if leaf_map is not None else LEAF_FEATURES
    keep = set(MINING_MANDATORY_COLS)
    for raw in keep_leaves:
        leaf = str(raw)
        keep.add(lm.get(leaf, leaf))
        keep |= DERIVED_LEAF_RAW_DEPS.get(leaf, frozenset())
        keep |= CROSS_SOURCE_LEAF_DEPS.get(leaf, frozenset())
    return keep


def _collect_daily_basic(
    context,
    *,
    slim: bool,
    keep_cols: set[str] | None,
) -> pl.DataFrame | None:
    """collect ``daily_basic``，列裁剪在 **lazy 链上**完成（parquet 投影下推）。

    返回 ``None`` = 与白名单交集为空 → 调用方整源跳过 join。
    ``slim=False and keep_cols is None`` 走原样 collect，逐字节零回归。
    """
    from factorzen.core.feature_schema import BASIC_FEATURES

    lf = context.daily_basic
    if not slim and keep_cols is None:
        return lf.collect()
    names = lf.collect_schema().names()
    if slim:
        cols = [c for c in sorted(BASIC_FEATURES) if c in names]
    else:
        cols = [c for c in names if c not in ("trade_date", "ts_code")]
    if keep_cols is not None:
        cols = [c for c in cols if c in keep_cols]
        if not cols:
            return None
    return lf.select(["trade_date", "ts_code", *cols]).collect()


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
    keep_leaves: Iterable[str] | None = None,
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

    ``keep_leaves``（默认 ``None`` = 不裁，完全零回归）：**本次真正会被求值的叶名集合**
    （`expression.feature_names` 口径）。给了就按 `resolve_keep_columns` 解析成列白名单，
    并在**源头**生效而不是出口补刀——每个 attach 源 join 前先裁到白名单交集，交集为空的
    源整个跳过（不 join、不补 null 列）。动机是全 A × 12.5 年 rebuild 的装配期 OOM：
    ~13M 行帧被逐步 eager join 加宽到 ~66 列，每次 join 瞬时双份，叠上源帧打满 23G 被
    OOM killer 杀（dmesg anon-rss 19.3G）。少 join 一个源既省常驻列、也省那一次全帧复制。
    强制核心列（`MINING_MANDATORY_COLS`）永不裁——下游 `_preprocess_daily` 无条件要它们。

    ``categorical_keys``：出口将 ``ts_code`` cast 为 ``pl.Categorical``（P4c）。
    - ``None``（默认）→ 仅当 ``slim`` 且 ``height >= KEYS_CATEGORICAL_ROWS_THRESHOLD`` 自动开；
    - ``True`` / ``False`` 显式开关（测试与逃生）。
    membership/basic join 在 cast **之前**完成（外部 Utf8 帧无需预先对齐）。
    跨帧 join 消费点须对小帧 cast 对齐（见 scoring._align_join_key）；落盘再转回 Utf8。
    """
    from factorzen.core.feature_schema import (
        EXPRESS_FEATURES,
        FLOW_FEATURES,
        FORECAST_FEATURES,
        FUNDAMENTAL_FEATURES,
        HOLDER_FEATURES,
        INTRADAY_FEATURES,
    )
    from factorzen.core.universe import get_universe_membership, membership_hash
    from factorzen.daily.data.context import FactorDataContext
    from factorzen.discovery.search.random_search import search_space_max_lookback

    if lookback_days is None:
        lookback_days = search_space_max_lookback()

    keep_cols: set[str] | None = (
        None if keep_leaves is None else resolve_keep_columns(keep_leaves)
    )

    def _src_keep(features: Iterable[str]) -> list[str] | None:
        """源叶集 ∩ 白名单。``None`` = 不裁（全要）；空 list = 该源整体跳过。"""
        if keep_cols is None:
            return None
        return sorted(set(features) & keep_cols)

    def _kw(keep: list[str] | None) -> dict:
        """``keep=None`` 时**不传该参数**——既有测试替身多为 ``lambda d: d``，零回归。"""
        return {} if keep is None else {"keep": keep}

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
    # daily raw：slim 时把死重 drop 推进 lazy 链，parquet 直接少读两列
    # （原实现是 collect 全部再在出口 drop，多付一次读+一份常驻）。
    daily_lf = context.daily
    if slim:
        _dead = [
            c for c in _DAILY_RAW_DEAD_COLS
            if c in daily_lf.collect_schema().names()
        ]
        if _dead:
            daily_lf = daily_lf.drop(_dead)
    daily = daily_lf.collect()
    del daily_lf

    basic = _collect_daily_basic(context, slim=slim, keep_cols=keep_cols)
    if basic is not None and not basic.is_empty():
        daily = daily.join(basic, on=["trade_date", "ts_code"], how="left")
    del basic

    # Mining and materialisation deliberately share these attach functions so the
    # same expression has identical leaves on both paths.
    from factorzen.daily.data.events import attach_express, attach_forecast
    from factorzen.daily.data.flows import attach_flows
    from factorzen.daily.data.pit import attach_fundamentals, attach_holders

    _fund = _src_keep(FUNDAMENTAL_FEATURES)
    if _fund is None or _fund:
        daily = attach_fundamentals(daily, **_kw(_fund))
    _hold = _src_keep(HOLDER_FEATURES)
    if _hold is None or _hold:
        daily = attach_holders(daily, **_kw(_hold))
    _flow = _src_keep(FLOW_FEATURES)
    if _flow is None or _flow:
        daily = attach_flows(daily, **_kw(_flow))
    _fc = _src_keep(FORECAST_FEATURES)
    if _fc is None or _fc:
        daily = attach_forecast(daily, **_kw(_fc))
    _ex = _src_keep(EXPRESS_FEATURES)
    if _ex is None or _ex:
        daily = attach_express(daily, **_kw(_ex))

    if intraday:
        from factorzen.daily.data.intraday import attach_intraday

        _ix = _src_keep(INTRADAY_FEATURES)
        if _ix is None or _ix:
            daily = attach_intraday(
                daily,
                freq=intraday_freq,
                version=intraday_version,
                require=True,
                out_meta=out_meta,
                **_kw(_ix),
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

    if keep_cols is not None:
        # 出口兜底：源头已按白名单裁过，这里只收尾 `ix_*` 之外的漏网叶列。
        # 只敢动 `_trimmable_leaf_cols()`（已知叶列）——未登记的列一律留着，
        # 免得把 ix_* / 未来新增的键列误裁掉（宁多勿漏）。
        trimmable = _trimmable_leaf_cols()
        sel = [c for c in daily.columns if c in keep_cols or c not in trimmable]
        if len(sel) < len(daily.columns):
            daily = daily.select(sel)

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
    _warn_short_warmup_prefix(daily, start, lookback_days, out_meta)
    return daily


def _warn_short_warmup_prefix(
    daily: pl.DataFrame, start: str, lookback_days: int, out_meta: dict | None,
) -> None:
    """扩窗预热是**请求**不是保证：盘上没有的历史变不出来，这里如实报账。

    `FactorDataContext.expanded_start` 向前要 ``lookback_days`` 个自然日，但当 ``start``
    落在数据湖左边界附近时（A股 ``data/raw/daily`` 最早 20160104），返回帧里
    ``trade_date < start`` 的行可能一根都没有。后果对**所有**消费方相同：
    `expression.warmup_bars_by_leaf` 对每个叶子都算出 ``have=0`` → 带 ts_*/delay 的
    表达式一律被 `warmup_shortfall` 判「预热不足」丢弃。

    挖掘路径（`fz mine` / `fz mine-team`）与 `factor-library rebuild` 共用本函数装帧，
    因此共用这条曝光面——挖掘一直没踩到只是因为它用近年窗口，不是因为它免疫。
    rebuild 侧另有 `backtest_window.clamp_eval_start_for_warmup` 把评估窗下界前推；
    挖掘侧不改窗口语义（候选是运行中现生成的，窗口由操作者定），只在这里出声。

    ``out_meta``（若给）回填 ``warmup_prefix`` 溯源字典，供调用方落 manifest。
    """
    if daily.is_empty() or "trade_date" not in daily.columns:
        return
    try:
        cut = _dt.datetime.strptime(str(start), "%Y%m%d").date()
    except ValueError:
        return
    dates = daily["trade_date"].unique().to_list()
    have = sum(1 for d in dates if d is not None and _as_date(d) < cut)
    if out_meta is not None:
        out_meta["warmup_prefix"] = {"requested_bars": int(lookback_days), "available_bars": have}
    if have >= lookback_days:
        return
    warnings.warn(
        f"预热前缀不足：{start} 之前装配帧只有 {have} 根 bar（请求 {lookback_days} 根）。"
        f"数据湖左边界晚于扩窗取数起点时无法补足；带 ts_*/delay 的表达式会被预热门"
        f"（warmup_shortfall）整条丢弃。请把窗口起点后移，或回补更早的历史数据。",
        stacklevel=3,
    )


def _as_date(v):
    """把 trade_date 单元格（date/datetime/str）统一成 ``date`` 以便比较。"""
    if isinstance(v, _dt.datetime):
        return v.date()
    if isinstance(v, _dt.date):
        return v
    return _dt.datetime.strptime(str(v).replace("-", "")[:8], "%Y%m%d").date()

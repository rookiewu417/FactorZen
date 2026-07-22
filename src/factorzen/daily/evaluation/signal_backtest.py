"""向量化因子研究信号回测（毛收益口径）。

与 ``backtest.py`` 日环撮合引擎分离：本模块只做分层/多空/IC 等信号层评价，
**不含** 停牌/涨跌停/T+1/冲击成本等可交易性约束。输出的是研究口径毛收益，
不可直接当可实现收益汇报。

前向收益由调用方经 ``compute_fwd_returns``（含 exec_lag/exec_price_col）预计算后
透传；本模块不调用 ``compute_fwd_returns``，遵守「四处同源」纪律。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from factorzen.config.constants import TRADING_DAYS_PER_YEAR
from factorzen.core.dates import with_iso_date
from factorzen.core.logger import get_logger
from factorzen.core.validation import require_columns
from factorzen.daily.evaluation.advanced.monotonicity import (
    MonotonicityResult,
    compute_monotonicity,
)
from factorzen.daily.evaluation.grouping import assign_quantile_groups
from factorzen.daily.evaluation.ic_analysis import ICAnalysisResult, compute_rank_ic
from factorzen.daily.evaluation.turnover import TurnoverResult, compute_turnover

_logger = get_logger(__name__)

# 年化基数单一真源(引擎 backtest.py 导入同一常量),避免两轨漂移
_TRADING_DAYS = TRADING_DAYS_PER_YEAR

_EMPTY_GROUP_RETURNS = pl.DataFrame(
    schema={
        "trade_date": pl.Utf8,
        "group": pl.Int32,
        "ret": pl.Float64,
        "n_stocks": pl.UInt32,
    }
)
_EMPTY_GROUP_NAV = pl.DataFrame(
    schema={"trade_date": pl.Utf8, "group": pl.Int32, "nav": pl.Float64}
)
_EMPTY_LS_RETURNS = pl.DataFrame(
    schema={
        "trade_date": pl.Utf8,
        "ls_ret_gross": pl.Float64,
        "ls_ret_net": pl.Float64,
        "ls_turnover": pl.Float64,
    }
)
_EMPTY_LS_NAV = pl.DataFrame(
    schema={"trade_date": pl.Utf8, "nav_gross": pl.Float64, "nav_net": pl.Float64}
)

_WARNING_LINE = (
    "⚠️ 信号层毛收益(不含约束/撮合),未经可交易性检验,不可直接当可实现收益汇报"
)


@dataclass
class SignalBacktestResult:
    """信号层回测结果（毛收益 / 研究口径）。"""

    factor_name: str
    n_groups: int
    cost_bps: float
    group_returns: pl.DataFrame  # trade_date, group, ret, n_stocks
    group_nav: pl.DataFrame  # trade_date, group, nav
    ls_returns: pl.DataFrame  # trade_date, ls_ret_gross, ls_ret_net, ls_turnover
    ls_nav: pl.DataFrame  # trade_date, nav_gross, nav_net
    ic: ICAnalysisResult
    monotonicity: MonotonicityResult
    turnover: TurnoverResult
    summary_stats: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        ls = self.summary_stats.get("long_short", {})
        ic = self.summary_stats.get("ic", {})
        lines = [
            f"SignalBacktest: {self.factor_name}  n_groups={self.n_groups}  "
            f"cost_bps={self.cost_bps}",
            f"  LS ann_ret_gross={ls.get('ann_ret_gross', 0.0):.4f}  "
            f"sharpe_gross={ls.get('sharpe_gross', 0.0):.2f}  "
            f"max_dd_gross={ls.get('max_dd_gross', 0.0):.2%}",
            f"  LS ann_ret_net={ls.get('ann_ret_net', 0.0):.4f}  "
            f"sharpe_net={ls.get('sharpe_net', 0.0):.2f}  "
            f"avg_turnover={ls.get('avg_turnover', 0.0):.4f}",
            f"  IC mean={ic.get('ic_mean', 0.0):.4f}  IR={ic.get('ir', 0.0):.2f}  "
            f"tstat={ic.get('tstat', 0.0):.2f}",
            f"  dropped_days={self.meta.get('dropped_days', 0)}",
            _WARNING_LINE,
        ]
        return "\n".join(lines)


def _empty_ic(factor_col: str, frequency: str) -> ICAnalysisResult:
    return ICAnalysisResult(
        factor_name=factor_col,
        ic_mean=0.0,
        ic_std=0.0,
        ir=0.0,
        ic_positive_ratio=0.0,
        n_periods=0,
        ic_series=pl.DataFrame(
            schema={"trade_date": pl.Utf8, "ic": pl.Float64}
        ),
        frequency=frequency,
    )


def _empty_turnover(frequency: str) -> TurnoverResult:
    return TurnoverResult(
        factor_name="",
        avg_turnover=0.0,
        migration_matrix=pl.DataFrame(
            schema={"prev_group": pl.Int32, "group": pl.Int32, "prob": pl.Float64}
        ),
        daily_turnover=pl.DataFrame(
            schema={"trade_date": pl.Utf8, "turnover": pl.Float64}
        ),
        frequency=frequency,
    )


def _zero_summary_stats() -> dict[str, Any]:
    return {
        "groups": {},
        "long_short": {
            "ann_ret_gross": 0.0,
            "sharpe_gross": 0.0,
            "max_dd_gross": 0.0,
            "ann_ret_net": 0.0,
            "sharpe_net": 0.0,
            "max_dd_net": 0.0,
            "avg_turnover": 0.0,
            # n_days=0 是「一天都没算出来」的判据:没有它,0.0 的年化与真实零 alpha
            # 不可区分(机器消费方读 signal.json 时尤其需要)
            "n_days": 0,
        },
        "ic": {"ic_mean": 0.0, "ir": 0.0, "tstat": 0.0, "n_periods": 0},
    }


def periods_per_year(frequency: str) -> float:
    """一年多少期。``--frequency`` 是可达 CLI 选项，硬编码 252 会把周频年化放大约
    4.8 倍、月频约 21 倍。日频复用 ``config.constants`` 的单一真源。
    """
    return {"daily": float(_TRADING_DAYS), "weekly": 52.0, "monthly": 12.0}.get(
        (frequency or "daily").lower(), float(_TRADING_DAYS)
    )


def _ann_ret_sharpe(rets: np.ndarray, ppy: float = float(_TRADING_DAYS)) -> tuple[float, float]:
    """年化收益与 Sharpe；近常数序列 std=0 → sharpe=0.0。

    ``ddof=0`` 与交易轨 ``backtest._summary_stats`` 一致——两轨 Sharpe 会被并排比较，
    口径必须同源（样本标准差 ddof=1 在 n 小时会让信号轨系统性偏低）。
    """
    valid = rets[np.isfinite(rets)]
    if len(valid) == 0:
        return 0.0, 0.0
    ann_ret = float(np.mean(valid) * ppy)
    std = float(np.std(valid)) if len(valid) > 1 else 0.0
    ann_vol = std * math.sqrt(ppy)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    return ann_ret, sharpe


def _max_drawdown(nav: np.ndarray) -> float:
    """净值相对历史峰值的最大回撤（负数或 0）。

    前置起点 1.0 与交易轨 ``backtest._summary_stats`` 的
    ``cum = concatenate([[1.0], cumprod(1+r)])`` 对齐：``nav`` 序列自身从 ``1+r0`` 起，
    不补起点就把**首日下跌**排除在回撤之外，结果系统性偏乐观
    （nav=[0.90,0.95,1.10] 会得 0.0，真实值 −10%）。
    """
    if nav.size == 0:
        return 0.0
    full = np.concatenate([[1.0], nav])
    peak = np.maximum.accumulate(full)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, full / peak - 1.0, 0.0)
    return float(np.min(dd))


def _leg_turnover(leg_members: pl.DataFrame) -> pl.DataFrame:
    """等权单腿换手：``0.5 * Σ|w_t - w_{t-1}|``；缺席视 w=0；首日=1.0。

    输入须含 trade_date, ts_code（仅该腿成分股）。
    """
    if leg_members.is_empty():
        return pl.DataFrame(schema={"trade_date": pl.Utf8, "leg_turnover": pl.Float64})

    sizes = leg_members.group_by("trade_date").agg(pl.len().alias("_n"))
    w = (
        leg_members.join(sizes, on="trade_date")
        .with_columns((1.0 / pl.col("_n").cast(pl.Float64)).alias("w"))
        .select(["trade_date", "ts_code", "w"])
    )
    dates = w.select("trade_date").unique().sort("trade_date").with_row_index("_di")
    w = w.join(dates, on="trade_date")

    curr = w.select(["_di", "trade_date", "ts_code", "w"])
    prev = w.select(
        (pl.col("_di") + 1).alias("_di"),
        pl.col("ts_code"),
        pl.col("w").alias("w_prev"),
    )
    joined = curr.join(prev, on=["_di", "ts_code"], how="full", coalesce=True)
    # full join 后 trade_date 仅来自 curr；用 di→date 表回填
    joined = (
        joined.drop("trade_date")
        .join(dates, on="_di")
        .with_columns(
            pl.col("w").fill_null(0.0),
            pl.col("w_prev").fill_null(0.0),
        )
    )
    to = (
        joined.group_by(["_di", "trade_date"])
        .agg(
            (0.5 * (pl.col("w") - pl.col("w_prev")).abs().sum()).alias("leg_turnover")
        )
        .sort("_di")
    )
    first_di = int(dates["_di"].min())  # type: ignore[arg-type]
    # 首日建仓：约定换手 1.0（full-join 缺 prev 只会得到 0.5）
    to = to.with_columns(
        pl.when(pl.col("_di") == first_di)
        .then(pl.lit(1.0))
        .otherwise(pl.col("leg_turnover"))
        .alias("leg_turnover")
    )
    return to.select(["trade_date", "leg_turnover"])


def _empty_result(
    *,
    factor_name: str,
    factor_col: str,
    n_groups: int,
    cost_bps: float,
    frequency: str,
    meta: dict[str, Any],
    ic: ICAnalysisResult | None = None,
    monotonicity: MonotonicityResult | None = None,
    turnover: TurnoverResult | None = None,
) -> SignalBacktestResult:
    return SignalBacktestResult(
        factor_name=factor_name or factor_col,
        n_groups=n_groups,
        cost_bps=cost_bps,
        group_returns=_EMPTY_GROUP_RETURNS.clear(),
        group_nav=_EMPTY_GROUP_NAV.clear(),
        ls_returns=_EMPTY_LS_RETURNS.clear(),
        ls_nav=_EMPTY_LS_NAV.clear(),
        ic=ic if ic is not None else _empty_ic(factor_col, frequency),
        monotonicity=monotonicity
        if monotonicity is not None
        else MonotonicityResult(factor_name=factor_col),
        turnover=turnover if turnover is not None else _empty_turnover(frequency),
        summary_stats=_zero_summary_stats(),
        meta=meta,
    )


def run_signal_backtest(
    factor_df: pl.DataFrame,
    fwd_returns: pl.DataFrame,
    *,
    factor_col: str = "factor_clean",
    n_groups: int = 5,
    cost_bps: float = 0.0,
    horizons: list[int] | None = None,
    frequency: str = "daily",
    factor_name: str = "",
    meta: dict | None = None,
) -> SignalBacktestResult:
    """纯向量化信号层回测（分层收益 / 多空 / IC / 换手粗成本）。

    Args:
        factor_df: 列 trade_date, ts_code, {factor_col}。
        fwd_returns: ``compute_fwd_returns`` 输出（至少含 fwd_ret_1d）；
            前向收益口径由调用方决定并透传，本函数不重算。
        factor_col: 因子列名。
        n_groups: 截面分位组数。
        cost_bps: 提示性单边成本（bp）；``ls_ret_net = gross − ls_turnover × cost_bps/1e4``。
            cost_bps=0 时 net 与 gross 逐位相等。非撮合引擎。
        horizons: 透传 ``compute_rank_ic``。
        frequency: 频率标签。
        factor_name: 展示用名称。
        meta: 附加元信息；会写入 ``return_basis`` 与 ``dropped_days``。

    Returns:
        SignalBacktestResult。join 后为空或全部日被剔除时返回空结构，不抛异常。

    计算语义:
        1. 双方 ``with_iso_date`` 后 inner join；(fwd_ret_1d 先 fill_nan(None) 再 drop null)。
        2. ``assign_quantile_groups``；当日有效样本 < n_groups 整日剔除。
        3. 组收益 = 组内等权 mean(fwd_ret_1d)。
        4. ls_ret_gross = mean(group=n_groups-1) − mean(group=0)。
        5. 每腿换手 0.5·Σ|Δw|（等权，缺席 w=0，首日=1.0）；
           ls_turnover = top + bottom；net = gross − turnover·cost_bps/1e4。
        6. NAV = cumprod(1+ret)，首日 nav=1+ret（无前置 1.0 行）。
    """
    out_meta: dict[str, Any] = dict(meta or {})
    # setdefault 而非直接赋值:调用方若已声明更具体的口径(如某条链路自定义的
    # 收益基准),不该被本模块无条件覆盖掉。
    out_meta.setdefault("return_basis", "gross_signal_level")
    out_meta.setdefault("dropped_days", 0)
    name = factor_name or factor_col

    require_columns(
        factor_df, ["trade_date", "ts_code", factor_col], context="run_signal_backtest"
    )
    require_columns(
        fwd_returns,
        ["trade_date", "ts_code", "fwd_ret_1d"],
        context="run_signal_backtest",
    )
    # 多空需要 top/bottom 两个不同的组;n_groups<2 时 top_g==0==bottom,
    # ls_ret 恒 0 而换手照算 → cost_bps>0 会凭空造出「稳定小亏」的假信号。
    if int(n_groups) < 2:
        raise ValueError(f"n_groups 必须 >=2(多空需要两个不同分组),收到 {n_groups}")

    # 日期形态必须先归一再喂给任何子计算:factor 侧 Date、ret 侧 ISO 字符串这类
    # 不一致会让 join 零命中,IC 返回 0.0 哨兵且无告警——正是 core/dates.py 记录的
    # 那条 live P0(admission_ic 恒 0.0)。归一化后两侧同形态。
    fdf = with_iso_date(factor_df)
    rdf = with_iso_date(fwd_returns)
    # NaN 传染防线：聚合跳过 null 但被 NaN 污染
    rdf = rdf.with_columns(pl.col("fwd_ret_1d").fill_nan(None))

    # (trade_date, ts_code) 重复会在 join 后按笛卡尔积放大：重复股在组均值与等权
    # 权重里被双计，n_stocks 也虚高。同仓 _zscore_and_merge 已有同类告警。
    for _frame, _label in ((fdf, "factor_df"), (rdf, "fwd_returns")):
        _dups = _frame.height - _frame.select(["trade_date", "ts_code"]).unique().height
        if _dups > 0:
            _logger.warning(
                "%s 含 %d 行重复 (trade_date, ts_code)，join 后会被重复计数，"
                "组均值/权重将失真——请先去重",
                _label,
                _dups,
            )

    # 辅助指标基于归一化后的输入（即使主路径退化也尽量给出 IC/换手/单调性）。
    # 只接 ValueError:polars 的 SchemaError 等不是 ValueError 子类,无差别吞掉会把
    # 响亮的报错降级成 0.0 哨兵,与「异常契约统一」相悖。
    try:
        ic_result = compute_rank_ic(
            fdf,
            rdf,
            factor_col=factor_col,
            horizons=horizons,
            frequency=frequency,
        )
    except ValueError as exc:
        _logger.warning("compute_rank_ic 失败,IC 置空: %s", exc)
        ic_result = _empty_ic(factor_col, frequency)

    try:
        to_result = compute_turnover(
            fdf, factor_col=factor_col, n_groups=n_groups, frequency=frequency
        )
    except ValueError as exc:
        _logger.warning("compute_turnover 失败,换手置空: %s", exc)
        to_result = _empty_turnover(frequency)

    merged = fdf.join(
        rdf.select(["trade_date", "ts_code", "fwd_ret_1d"]),
        on=["trade_date", "ts_code"],
        how="inner",
    ).filter(
        # is_finite 同时挡住 null/NaN/±inf:inf 会让 ls_ret 与 nav 全变 inf,
        # 而 _ann_ret_sharpe 用 isfinite 滤空后返回 0.0 —— 彻底损坏的序列
        # 会长得和「零 alpha」一模一样。全仓同类位置(ic_analysis/backtest)都用 is_finite。
        pl.col("fwd_ret_1d").is_finite() & pl.col(factor_col).is_finite()
    )

    if merged.is_empty():
        try:
            mono = compute_monotonicity(
                pl.DataFrame(
                    schema={
                        "trade_date": pl.Utf8,
                        "ts_code": pl.Utf8,
                        factor_col: pl.Float64,
                        "fwd_ret_1d": pl.Float64,
                    }
                ),
                factor_col=factor_col,
                ret_col="fwd_ret_1d",
                n_groups=n_groups,
            )
        except Exception:
            mono = MonotonicityResult(factor_name=factor_col)
        return _empty_result(
            factor_name=name,
            factor_col=factor_col,
            n_groups=n_groups,
            cost_bps=cost_bps,
            frequency=frequency,
            meta=out_meta,
            ic=ic_result,
            monotonicity=mono,
            turnover=to_result,
        )

    merged = merged.sort(["trade_date", "ts_code"])
    grouped = assign_quantile_groups(merged, factor_col=factor_col, n_groups=n_groups)

    # 当日有效样本数 < n_groups → 整日剔除。
    # 分母取**输入**的唯一日期数,不是 join+过滤后还剩行的日子——否则「当日 0 个
    # 有效样本」(全 NaN / join 零命中)这类整日消失的情况会记成 dropped_days=0,
    # 与「一天都没丢」不可区分。
    input_days = factor_df.select("trade_date").unique().height
    day_counts = grouped.group_by("trade_date").agg(pl.len().alias("_n_valid"))
    keep_days = day_counts.filter(pl.col("_n_valid") >= n_groups)
    out_meta["dropped_days"] = int(max(0, input_days - keep_days.height))

    grouped = grouped.join(
        keep_days.select("trade_date"), on="trade_date", how="inner"
    )

    if grouped.is_empty():
        mono = compute_monotonicity(
            merged, factor_col=factor_col, ret_col="fwd_ret_1d", n_groups=n_groups
        )
        return _empty_result(
            factor_name=name,
            factor_col=factor_col,
            n_groups=n_groups,
            cost_bps=cost_bps,
            frequency=frequency,
            meta=out_meta,
            ic=ic_result,
            monotonicity=mono,
            turnover=to_result,
        )

    # 组内等权收益
    group_returns = (
        grouped.group_by(["trade_date", "group"])
        .agg(
            pl.col("fwd_ret_1d").mean().alias("ret"),
            pl.len().cast(pl.UInt32).alias("n_stocks"),
        )
        .sort(["trade_date", "group"])
    )

    # 多空毛收益
    top_g = n_groups - 1
    top_ret = (
        group_returns.filter(pl.col("group") == top_g)
        .select(["trade_date", pl.col("ret").alias("_top")])
    )
    bot_ret = (
        group_returns.filter(pl.col("group") == 0)
        .select(["trade_date", pl.col("ret").alias("_bot")])
    )
    ls_base = (
        top_ret.join(bot_ret, on="trade_date", how="inner")
        .with_columns((pl.col("_top") - pl.col("_bot")).alias("ls_ret_gross"))
        .select(["trade_date", "ls_ret_gross"])
        .sort("trade_date")
    )

    # 两腿换手（等权权重）
    top_members = grouped.filter(pl.col("group") == top_g).select(
        ["trade_date", "ts_code"]
    )
    bot_members = grouped.filter(pl.col("group") == 0).select(
        ["trade_date", "ts_code"]
    )
    top_to = _leg_turnover(top_members).rename({"leg_turnover": "_top_to"})
    bot_to = _leg_turnover(bot_members).rename({"leg_turnover": "_bot_to"})

    ls_returns = (
        ls_base.join(top_to, on="trade_date", how="left")
        .join(bot_to, on="trade_date", how="left")
        .with_columns(
            pl.col("_top_to").fill_null(1.0),
            pl.col("_bot_to").fill_null(1.0),
        )
        .with_columns(
            (pl.col("_top_to") + pl.col("_bot_to")).alias("ls_turnover"),
        )
        .with_columns(
            (
                pl.col("ls_ret_gross")
                - pl.col("ls_turnover") * (cost_bps / 10000.0)
            ).alias("ls_ret_net")
        )
        .select(["trade_date", "ls_ret_gross", "ls_ret_net", "ls_turnover"])
        .sort("trade_date")
    )

    # 组 NAV：cumprod(1+ret)，首日 nav=1+ret（无前置 1.0 行）
    group_nav = (
        group_returns.sort(["group", "trade_date"])
        .with_columns((1.0 + pl.col("ret")).cum_prod().over("group").alias("nav"))
        .select(["trade_date", "group", "nav"])
        .sort(["trade_date", "group"])
    )

    # 多空 NAV
    ls_nav = (
        ls_returns.sort("trade_date")
        .with_columns(
            (1.0 + pl.col("ls_ret_gross")).cum_prod().alias("nav_gross"),
            (1.0 + pl.col("ls_ret_net")).cum_prod().alias("nav_net"),
        )
        .select(["trade_date", "nav_gross", "nav_net"])
    )

    # 单调性：用 join 后、分组前过滤后的帧
    mono_result = compute_monotonicity(
        grouped.select(["trade_date", "ts_code", factor_col, "fwd_ret_1d"]),
        factor_col=factor_col,
        ret_col="fwd_ret_1d",
        n_groups=n_groups,
    )

    # summary_stats（年化基数随 frequency，非硬编码 252）
    ppy = periods_per_year(frequency)
    out_meta["periods_per_year"] = ppy
    groups_stats: dict[int, dict[str, float]] = {}
    for g in range(n_groups):
        g_rets = group_returns.filter(pl.col("group") == g)["ret"].to_numpy()
        ar, sh = _ann_ret_sharpe(np.asarray(g_rets, dtype=float), ppy)
        groups_stats[g] = {"ann_ret": ar, "sharpe": sh}

    gross = ls_returns["ls_ret_gross"].to_numpy().astype(float)
    net = ls_returns["ls_ret_net"].to_numpy().astype(float)
    ar_g, sh_g = _ann_ret_sharpe(gross, ppy)
    ar_n, sh_n = _ann_ret_sharpe(net, ppy)
    nav_g = ls_nav["nav_gross"].to_numpy().astype(float)
    nav_n = ls_nav["nav_net"].to_numpy().astype(float)
    to_arr = ls_returns["ls_turnover"].to_numpy().astype(float)
    avg_to = float(np.mean(to_arr)) if to_arr.size else 0.0

    summary_stats: dict[str, Any] = {
        "groups": groups_stats,
        "long_short": {
            "ann_ret_gross": ar_g,
            "sharpe_gross": sh_g,
            "max_dd_gross": _max_drawdown(nav_g),
            "ann_ret_net": ar_n,
            "sharpe_net": sh_n,
            "max_dd_net": _max_drawdown(nav_n),
            "avg_turnover": avg_to,
            "n_days": int(ls_returns.height),
        },
        "ic": {
            "ic_mean": float(ic_result.ic_mean),
            "ir": float(ic_result.ir),
            "tstat": float(ic_result.ic_tstat),
            # 机器消费方(signal.json)据此区分「IC 算不出来」与「IC 真的接近 0」
            "n_periods": int(ic_result.n_periods),
        },
    }

    return SignalBacktestResult(
        factor_name=name,
        n_groups=n_groups,
        cost_bps=cost_bps,
        group_returns=group_returns,
        group_nav=group_nav,
        ls_returns=ls_returns,
        ls_nav=ls_nav,
        ic=ic_result,
        monotonicity=mono_result,
        turnover=to_result,
        summary_stats=summary_stats,
        meta=out_meta,
    )

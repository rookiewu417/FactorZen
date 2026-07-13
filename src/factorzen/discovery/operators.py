# src/factorzen/discovery/operators.py
"""算子库：每个算子是一个把子表达式（pl.Expr）组合成新 pl.Expr 的工厂。

约定（编译前提）：求值表已按 (ts_code, trade_date) 排序。
- 时序算子(ts)用 .over("ts_code")；截面算子(cs)用 .over("trade_date")；算术(arith)逐元素。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import polars as pl

# 叶子名 → 求值表中的列名。vwap/log_vol/ret_1d/amplitude/intraday_ret/overnight_ret 为派生列（ExpressionFactor 预计算）。
LEAF_FEATURES: dict[str, str] = {
    "close": "close_adj", "open": "open_adj", "high": "high_adj", "low": "low_adj",
    "vol": "vol", "amount": "amount", "vwap": "vwap", "log_vol": "log_vol", "ret_1d": "ret_1d",
    "amplitude": "amplitude", "intraday_ret": "intraday_ret", "overnight_ret": "overnight_ret",
    "total_mv": "total_mv", "circ_mv": "circ_mv", "pb": "pb", "pe_ttm": "pe_ttm",
    "ps_ttm": "ps_ttm", "dv_ttm": "dv_ttm",
    "turnover_rate": "turnover_rate", "turnover_rate_f": "turnover_rate_f",
    "volume_ratio": "volume_ratio", "float_share": "float_share",
    # 基本面（财报 fina_indicator，按公告日 PIT 对齐；与量价正交，供价值/质量/成长类因子）
    # 质量：roe/roa/毛利率/净利率/资产负债率(杠杆)
    "roe": "roe", "roa": "roa",
    "grossprofit_margin": "grossprofit_margin", "netprofit_margin": "netprofit_margin",
    "debt_to_assets": "debt_to_assets",
    # 成长：营收同比/净利同比/资产同比
    "or_yoy": "or_yoy", "netprofit_yoy": "netprofit_yoy", "assets_yoy": "assets_yoy",
    # 资金流/北向（日频 point-in-time，与量价/基本面均正交）
    "net_mf_amount": "net_mf_amount",  # 主力资金净流入额
    "north_ratio": "north_ratio",      # 北向持股占比
    # 两融/杠杆情绪（margin_detail；T 日数据 T+1 早间披露 → attach 内置 lag(1)；
    # rzye/rzmre 单位元；margin_ratio=rzye/(circ_mv×1e4)，margin_buy_ratio=rzmre/(amount×1e3)）
    "margin_ratio": "margin_ratio",           # 融资余额/流通市值（杠杆拥挤度）
    "margin_buy_ratio": "margin_buy_ratio",   # 融资买入额/成交额（杠杆资金参与度）
    "margin_balance": "margin_balance",       # 融资余额原值 rzye（元，已 lag）
    "short_balance": "short_balance",         # 融券余量原值 rqyl（股，已 lag）
    # 股东户数（stk_holdernumber；按 ann_date PIT；holder_num_chg 源侧期际环比，非 ts_*）
    "holder_num": "holder_num",               # 最新一期股东户数（户）
    "holder_num_chg": "holder_num_chg",       # 相邻两期环比 (本期-上期)/上期；随 ann_date 生效
    # 龙虎榜（top_list；t 日盘后披露 → attach lag(1)；未上榜=真实零事件 fill 0）
    # net_amount 万元→×1e4、amount 千元→×1e3，比前统一到元；同日多原因先 sum net_amount
    "top_list_net_buy": "top_list_net_buy",   # 昨日龙虎榜净买入额/成交额
    "top_list_flag": "top_list_flag",         # 昨日是否上榜 0/1
}
BASIC_FEATURES: set[str] = {
    "total_mv", "circ_mv", "pb", "pe_ttm", "ps_ttm", "dv_ttm",
    "turnover_rate", "turnover_rate_f", "volume_ratio", "float_share",
}
# 需 finance 数据 + PIT 对齐的基本面叶子。用了这些叶子的因子须先 attach_fundamentals，
# 否则回测/物化路径上它们全 null（双路径漂移，陷阱#2）。fina_indicator 字段名即叶子名。
FUNDAMENTAL_FEATURES: set[str] = {
    "roe", "roa", "grossprofit_margin", "netprofit_margin", "debt_to_assets",
    "or_yoy", "netprofit_yoy", "assets_yoy",
}
# 股东户数叶子（stk_holdernumber + ann_date PIT）。用了它们的因子须先 attach_holders。
HOLDER_FEATURES: set[str] = {
    "holder_num", "holder_num_chg",
}
# 两融叶子（margin_detail + T+1 lag）。子集于 FLOW_FEATURES：物化路径经 attach_flows 门接入。
MARGIN_FEATURES: set[str] = {
    "margin_ratio", "margin_buy_ratio", "margin_balance", "short_balance",
}
# 龙虎榜叶子（top_list + lag(1) + fill 0）。子集于 FLOW_FEATURES。
TOPLIST_FEATURES: set[str] = {
    "top_list_net_buy", "top_list_flag",
}
# 需资金流/北向/两融/龙虎榜数据的日频叶子。用了它们的因子须先 attach_flows，否则回测/物化路径上全 null。
FLOW_FEATURES: set[str] = {"net_mf_amount", "north_ratio"} | MARGIN_FEATURES | TOPLIST_FEATURES

_MIN = 3  # rolling 最小样本


def _safe_div(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    # is_finite() 显式排除 NaN 分母：polars 中 NaN.abs() > 1e-12 判 True，只查 abs 会让
    # NaN 分母穿透守卫、输出 a/NaN=NaN。
    return pl.when(b.is_finite() & (b.abs() > 1e-12)).then(a / b).otherwise(None)


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    category: Literal["ts", "cs", "arith"]
    arity: int
    has_window: bool
    build: Callable[[list[pl.Expr], int | None], pl.Expr]


def _ts(name, fn):  # window 时序算子
    return OperatorSpec(name, "ts", 1, True, lambda c, w: fn(c[0], w))


def _cs(name, fn):  # 截面算子
    return OperatorSpec(name, "cs", 1, False, lambda c, w: fn(c[0]))


def _ar(name, arity, fn):  # 算术算子
    return OperatorSpec(name, "arith", arity, False, lambda c, w: fn(*c))


def _ts2(name, fn):  # 双输入时序算子（arity 2, 有 window）
    return OperatorSpec(name, "ts", 2, True, lambda c, w: fn(c[0], c[1], w))


def _ts_corr(a: pl.Expr, b: pl.Expr, w: int | None) -> pl.Expr:
    ma = a.rolling_mean(w, min_samples=_MIN).over("ts_code")  # type: ignore[arg-type]
    mb = b.rolling_mean(w, min_samples=_MIN).over("ts_code")  # type: ignore[arg-type]
    mab = (a * b).rolling_mean(w, min_samples=_MIN).over("ts_code")  # type: ignore[arg-type]
    va = (a * a).rolling_mean(w, min_samples=_MIN).over("ts_code") - ma * ma  # type: ignore[arg-type]
    vb = (b * b).rolling_mean(w, min_samples=_MIN).over("ts_code") - mb * mb  # type: ignore[arg-type]
    cov = mab - ma * mb
    # clip(0) 吸收近常数窗口 E[x²]−E[x]² 的浮点微负（否则 va*vb<0 → sqrt=NaN，而
    # NaN>1e-12 判 True 会让守卫放行、输出 NaN）；再叠 is_finite 双保险。
    denom = (va.clip(0.0, None) * vb.clip(0.0, None)).sqrt()
    return (
        pl.when(denom.is_finite() & (denom > 1e-12))
        .then((cov / denom).clip(-1.0, 1.0))
        .otherwise(None)
    )


def _ts_cov(a: pl.Expr, b: pl.Expr, w: int | None) -> pl.Expr:
    ma = a.rolling_mean(w, min_samples=_MIN).over("ts_code")  # type: ignore[arg-type]
    mb = b.rolling_mean(w, min_samples=_MIN).over("ts_code")  # type: ignore[arg-type]
    mab = (a * b).rolling_mean(w, min_samples=_MIN).over("ts_code")  # type: ignore[arg-type]
    return mab - ma * mb


OPERATORS: dict[str, OperatorSpec] = {
    # ── 时序（.over("ts_code")）──
    "ts_mean": _ts("ts_mean", lambda x, w: x.rolling_mean(w, min_samples=_MIN).over("ts_code")),
    "ts_std":  _ts("ts_std",  lambda x, w: x.rolling_std(w, min_samples=_MIN).over("ts_code")),
    "ts_sum":  _ts("ts_sum",  lambda x, w: x.rolling_sum(w, min_samples=_MIN).over("ts_code")),
    "ts_min":  _ts("ts_min",  lambda x, w: x.rolling_min(w, min_samples=_MIN).over("ts_code")),
    "ts_max":  _ts("ts_max",  lambda x, w: x.rolling_max(w, min_samples=_MIN).over("ts_code")),
    # polars 1.41.2 原生 rolling_rank，标记 unstable(升级 polars 时需重验语义)。
    # method="average" 并列取均值排名；除以窗口内**实际非空样本数**归一化到 (0,1]——
    # 除以固定 w 会让历史不足 w 天的股票(warm-up/次新)ts_rank 上限只有 count/w，被系统性压低。
    "ts_rank": _ts("ts_rank", lambda x, w: _safe_div(
        x.rolling_rank(w, method="average", min_samples=_MIN).over("ts_code"),
        x.is_not_null().cast(pl.Int64).rolling_sum(w, min_samples=_MIN).over("ts_code"),
    )),
    "delay":   _ts("delay",   lambda x, w: x.shift(w).over("ts_code")),
    "delta":   _ts("delta",   lambda x, w: (x - x.shift(w)).over("ts_code")),
    "pct_change": _ts("pct_change", lambda x, w:
        (pl.when(x.shift(w) > 1e-12).then(x / x.shift(w) - 1.0).otherwise(None)).over("ts_code")),
    "ts_decay_linear": _ts("ts_decay_linear", lambda x, w:
        x.rolling_mean(w, min_samples=_MIN).over("ts_code")),  # MVP：等权近似线性衰减
    "ts_corr": _ts2("ts_corr", _ts_corr),
    "ts_cov": _ts2("ts_cov", _ts_cov),
    "ts_median": _ts("ts_median", lambda x, w:
        x.rolling_median(w, min_samples=_MIN).over("ts_code")),
    "ts_zscore": _ts("ts_zscore", lambda x, w: _safe_div(
        x - x.rolling_mean(w, min_samples=_MIN).over("ts_code"),
        x.rolling_std(w, min_samples=_MIN).over("ts_code"))),
    # polars 1.41.2 原生 rolling_skew，标记 unstable(升级 polars 时需重验语义)。
    # bias=True = 总体偏度(ddof=0)，对应现有 numpy ground-truth 测试口径。
    # fill_nan(None)：零方差窗口原生实现是 0/0 → NaN 而非 null，必须显式转 null，
    # 否则又是一次"NaN 泄漏"（polars 中 NaN > 阈值 判定为 True，下游 pl.when 防护会被绕过）。
    "ts_skew": _ts("ts_skew", lambda x, w:
        x.rolling_skew(w, bias=True, min_samples=_MIN).over("ts_code").fill_nan(None)),
    # ── 截面（.over("trade_date")）──
    # 分母用 x.count()（非空计数）而非 pl.len()（含 null 行）：x.rank() 只给非空值
    # 1..k 排名，若除以含 null 的总行数，归一化尺度会随当日 null 比例漂移（同样的
    # 非空排名，null 多的日被系统性压小），破坏截面可比性。
    "rank":  _cs("rank",  lambda x: (x.rank().over("trade_date") / (x.count().over("trade_date") + 1))),
    "zscore": _cs("zscore", lambda x:
        _safe_div(x - x.mean().over("trade_date"), x.std().over("trade_date"))),
    "scale": _cs("scale", lambda x: _safe_div(x, x.abs().sum().over("trade_date"))),
    # ── 算术 ──
    "add": _ar("add", 2, lambda a, b: a + b),
    "sub": _ar("sub", 2, lambda a, b: a - b),
    "mul": _ar("mul", 2, lambda a, b: a * b),
    "div": _ar("div", 2, lambda a, b: _safe_div(a, b)),
    "abs": _ar("abs", 1, lambda a: a.abs()),
    "log": _ar("log", 1, lambda a: pl.when(a > 0).then(a.log()).otherwise(None)),
    "sign": _ar("sign", 1, lambda a: a.sign()),
    "sqrt": _ar("sqrt", 1, lambda a: pl.when(a >= 0).then(a.sqrt()).otherwise(None)),
    "neg": _ar("neg", 1, lambda a: -a),
    "inv": _ar("inv", 1, lambda a: _safe_div(pl.lit(1.0), a)),
    "square": _ar("square", 1, lambda a: a * a),
    "max": _ar("max", 2, lambda a, b: pl.max_horizontal(a, b)),
    "min": _ar("min", 2, lambda a, b: pl.min_horizontal(a, b)),
}

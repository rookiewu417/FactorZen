# src/factorzen/discovery/scoring.py
"""候选因子快速评估：两段式中的「内循环」——只算 Rank IC/IR，不跑回测。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from factorzen.daily.evaluation.correlation import compute_factor_correlation
from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns, compute_rank_ic
from factorzen.daily.preprocessing.normalizer import cross_sectional_zscore
from factorzen.discovery.expression import Node
from factorzen.discovery.expression import complexity as _complexity


def _cut_literal(df: pl.DataFrame, yyyymmdd: str):
    """"YYYYMMDD" → 与 df.trade_date dtype 匹配的比较字面量(Date→date,Datetime→当日零点)。

    日频帧行为与旧 ``.date()`` 完全一致;intraday(Datetime 键)返回 datetime,
    避免 polars Datetime 列与 date 字面量比较的类型错误。
    """
    from datetime import datetime
    dt = datetime.strptime(yyyymmdd, "%Y%m%d")
    return dt if isinstance(df.schema["trade_date"], pl.Datetime) else dt.date()


@dataclass
class DataBundle:
    daily: pl.DataFrame
    fwd_returns: pl.DataFrame
    train_end: str  # "YYYYMMDD"，train 段含此日及之前

    @classmethod
    def build(cls, daily: pl.DataFrame, train_ratio: float = 0.7) -> DataBundle:
        daily = daily.sort(["ts_code", "trade_date"])
        fwd = compute_fwd_returns(daily, price_col="close_adj" if "close_adj" in daily.columns else "close")
        dates = sorted(daily["trade_date"].unique().to_list())
        cut = dates[min(int(len(dates) * train_ratio), len(dates) - 1)]
        train_end = cut.strftime("%Y%m%d") if hasattr(cut, "strftime") else str(cut)
        return cls(daily=daily, fwd_returns=fwd, train_end=train_end)

    def _segment_mask(self, df: pl.DataFrame, segment: str) -> pl.DataFrame:
        cut = _cut_literal(df, self.train_end)
        if segment == "train":
            return df.filter(pl.col("trade_date") <= cut)
        return df.filter(pl.col("trade_date") > cut)


def quick_fitness(factor_df: pl.DataFrame, bundle: DataBundle,
                  segment: Literal["train", "valid"] = "train") -> dict:
    """factor_df: [trade_date, ts_code, factor_value] → {ic_mean, ir, tstat, n}。

    ``tstat`` 为 IC 序列的 Newey-West HAC t 统计量（``compute_rank_ic`` 已算），
    仅当有效 IC 天数 >4 且 ic_std>0 时非零，天然惩罚低样本 —— 用作排序键可避免
    小样本 ic_std 虚低把 IR 撑爆的假象（见 score_candidate）。
    """
    seg = bundle._segment_mask(factor_df, segment)
    if seg.is_empty():
        return {"ic_mean": 0.0, "ir": 0.0, "tstat": 0.0, "n": 0}
    # 截面 zscore（cross_sectional_zscore 新增列 factor_value_z）
    clean = cross_sectional_zscore(seg, col="factor_value").rename({"factor_value_z": "factor_clean"})
    ret = bundle._segment_mask(bundle.fwd_returns, segment)
    res = compute_rank_ic(clean.select(["trade_date", "ts_code", "factor_clean"]),
                          ret, factor_col="factor_clean", frequency="daily")
    return {"ic_mean": res.ic_mean, "ir": res.ir, "tstat": res.ic_tstat, "n": res.n_periods}


# 去相关 |corr| 门槛的单一真源——session 池去相关、库级正交、upsert 默认共用。
DEFAULT_DECORR_THRESHOLD = 0.7

# 与 compute_factor_correlation 逐日门槛一致
_MIN_CORR_CROSS = 30


@dataclass(frozen=True)
class LibraryCorrPanel:
    """库池一次对齐的宽面板，供候选 vs 库逐对相关向量化。

    语义与 ``compute_factor_correlation`` 逐对路径一致（见 ``max_correlation_detail``）：
    - ``present``：polars 非 null（**含 float NaN**——NaN 会毒化该日 corrcoef）
    - 缺行 / null → ``present=False``（该 (date,stock) 不参与该对）
    - ``names`` 保持 pool 插入序（并列 max|corr| 取后出现者）
    """

    names: tuple[str, ...]
    dates: tuple  # sorted unique trade_date
    stocks: tuple  # sorted unique ts_code
    date_idx: dict
    stock_idx: dict
    values: np.ndarray  # (n_dates, n_stocks, n_factors) float64
    present: np.ndarray  # (n_dates, n_stocks, n_factors) bool


def _factor_col_name(df: pl.DataFrame) -> str:
    if "factor_value" in df.columns:
        return "factor_value"
    if "factor_clean" in df.columns:
        return "factor_clean"
    raise ValueError(
        f"因子帧须含 factor_value 或 factor_clean，实得列={list(df.columns)}"
    )


def build_library_corr_panel(
    pool: dict[str, pl.DataFrame] | None,
) -> LibraryCorrPanel | None:
    """把库池对齐成 (date × stock × k) 矩阵 + present 掩码；空/None → None。

    Session 级构建一次、整 session 复用。不改池因子数值，只做散射对齐。
    """
    if not pool:
        return None
    names = tuple(pool.keys())
    prepared: list[pl.DataFrame] = []
    all_dates: set = set()
    all_stocks: set = set()
    for name in names:
        df = pool[name]
        col = _factor_col_name(df)
        sub = df.select(
            ["trade_date", "ts_code", pl.col(col).alias("_v")]
        )
        all_dates.update(sub["trade_date"].to_list())
        all_stocks.update(sub["ts_code"].to_list())
        prepared.append(sub)

    dates = tuple(sorted(all_dates))
    stocks = tuple(sorted(all_stocks))
    date_idx = {d: i for i, d in enumerate(dates)}
    stock_idx = {s: i for i, s in enumerate(stocks)}
    n_d, n_s, n_f = len(dates), len(stocks), len(names)
    values = np.full((n_d, n_s, n_f), np.nan, dtype=np.float64)
    present = np.zeros((n_d, n_s, n_f), dtype=bool)

    for fi, sub in enumerate(prepared):
        if sub.is_empty():
            continue
        r = np.fromiter(
            (date_idx.get(d, -1) for d in sub["trade_date"].to_list()),
            dtype=np.int64,
            count=sub.height,
        )
        c = np.fromiter(
            (stock_idx.get(s, -1) for s in sub["ts_code"].to_list()),
            dtype=np.int64,
            count=sub.height,
        )
        # null → present False；float NaN → present True（毒化语义）
        is_null = sub["_v"].is_null().to_numpy()
        # to_numpy 把 null 也变成 NaN；用 fill_null 再取数，present 单独管
        arr = sub["_v"].fill_null(0.0).to_numpy().astype(np.float64, copy=False)
        # 真正的 float NaN 在 fill_null 后仍在
        keep = (r >= 0) & (c >= 0)
        r_k, c_k = r[keep], c[keep]
        values[r_k, c_k, fi] = arr[keep]
        present[r_k, c_k, fi] = ~is_null[keep]

    return LibraryCorrPanel(
        names=names,
        dates=dates,
        stocks=stocks,
        date_idx=date_idx,
        stock_idx=stock_idx,
        values=values,
        present=present,
    )


def _scatter_candidate_to_panel(
    factor_df: pl.DataFrame, panel: LibraryCorrPanel,
) -> tuple[np.ndarray, np.ndarray]:
    """候选散射到 panel 网格 → (values, present)，形状 (n_dates, n_stocks)。"""
    n_d, n_s = len(panel.dates), len(panel.stocks)
    vals = np.full((n_d, n_s), np.nan, dtype=np.float64)
    pres = np.zeros((n_d, n_s), dtype=bool)
    if factor_df.is_empty():
        return vals, pres
    col = _factor_col_name(factor_df)
    sub = factor_df.select(["trade_date", "ts_code", pl.col(col).alias("_v")])
    r = np.fromiter(
        (panel.date_idx.get(d, -1) for d in sub["trade_date"].to_list()),
        dtype=np.int64,
        count=sub.height,
    )
    c = np.fromiter(
        (panel.stock_idx.get(s, -1) for s in sub["ts_code"].to_list()),
        dtype=np.int64,
        count=sub.height,
    )
    is_null = sub["_v"].is_null().to_numpy()
    arr = sub["_v"].fill_null(0.0).to_numpy().astype(np.float64, copy=False)
    keep = (r >= 0) & (c >= 0)
    r_k, c_k = r[keep], c[keep]
    vals[r_k, c_k] = arr[keep]
    pres[r_k, c_k] = ~is_null[keep]
    return vals, pres


def _max_corr_detail_panel(
    factor_df: pl.DataFrame, panel: LibraryCorrPanel,
) -> tuple[float, str | None]:
    """矩阵化逐对相关：逐日向量化算全部库因子，语义对齐 compute_factor_correlation。"""
    if not panel.names:
        return 0.0, None
    cand_v, cand_p = _scatter_candidate_to_panel(factor_df, panel)
    n_d = len(panel.dates)
    n_f = len(panel.names)
    cum = np.zeros(n_f, dtype=np.float64)
    cnt = np.zeros(n_f, dtype=np.int64)

    for di in range(n_d):
        cp = cand_p[di]
        if not cp.any():
            continue
        c_row = cand_v[di]
        lp = panel.present[di]  # (n_s, n_f)
        lv = panel.values[di]  # (n_s, n_f)
        valid = cp[:, None] & lp  # (n_s, n_f)
        n = valid.sum(axis=0).astype(np.float64)  # (n_f,)
        enough = n >= _MIN_CORR_CROSS
        if not enough.any():
            continue
        # invalid → 0；valid 保留原值（含 NaN → 和式变 NaN → 该对日跳过）
        c_m = np.where(valid, c_row[:, None], 0.0)
        l_m = np.where(valid, lv, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sum_c = c_m.sum(axis=0)
            sum_l = l_m.sum(axis=0)
            sum_c2 = (c_m * c_m).sum(axis=0)
            sum_l2 = (l_m * l_m).sum(axis=0)
            sum_cl = (c_m * l_m).sum(axis=0)
            # Pearson ≡ np.corrcoef：|mean corr| 用 (nΣxy−ΣxΣy)/√(...)
            # std==0（ddof=0）⇔ n·Σx²−(Σx)² == 0
            den_c = n * sum_c2 - sum_c * sum_c
            den_l = n * sum_l2 - sum_l * sum_l
            num = n * sum_cl - sum_c * sum_l
            denom = np.sqrt(den_c * den_l)
            corr = num / denom
        # 幸存：样本够、方差非零、有限（NaN 毒化 / 0/0 均排除）
        ok = enough & (den_c > 0) & (den_l > 0) & np.isfinite(corr)
        cum[ok] += corr[ok]
        cnt[ok] += 1

    best = 0.0
    nearest: str | None = None
    for fi, name in enumerate(panel.names):
        if cnt[fi] <= 0:
            c = 0.0
        else:
            c = abs(float(cum[fi] / cnt[fi]))
        if c == c and c >= best:
            best, nearest = c, name
    return best, nearest


def max_correlation(
    factor_df: pl.DataFrame,
    pool: dict[str, pl.DataFrame],
    panel: LibraryCorrPanel | None = None,
) -> float:
    """factor_df 与 pool 中每个因子的截面相关性绝对值的最大值。pool 为空时返回 0。

    逐对(pairwise)计算：候选与池中**每个**因子单独算相关。这样一个退化的池因子
    (截面 std==0 / 不足 30 只 / NaN) 只会让它自己那一对得 0，不会污染其它对。
    历史 bug：把候选 + 全池一次性 inner-join 交给 compute_factor_correlation，任一
    池因子退化就 continue 丢整条截面 → count=0 → 所有真实高相关一起被抹成 0.0，
    数学等价簇因此逃过 0.7 去重门槛。不动 compute_factor_correlation（daily 报告仍用其语义）。

    ``panel``：可选预构建库面板；传入时走矩阵化路径（与逐对数值等价）。
    """
    return max_correlation_detail(factor_df, pool, panel=panel)[0]


def max_correlation_detail(
    factor_df: pl.DataFrame,
    pool: dict[str, pl.DataFrame],
    panel: LibraryCorrPanel | None = None,
) -> tuple[float, str | None]:
    """同 ``max_correlation``，额外返回最相近的 pool key（表达式）。pool 空 → (0.0, None)。

    ``panel`` 非 None 时走矩阵化路径，须由同一 ``pool`` 经 ``build_library_corr_panel`` 构建。
    """
    if not pool:
        return 0.0, None
    if panel is not None:
        return _max_corr_detail_panel(factor_df, panel)
    cand = (factor_df.rename({"factor_value": "factor_clean"})
            if "factor_value" in factor_df.columns else factor_df)
    best = 0.0
    nearest: str | None = None
    for name, df in pool.items():
        other = df.rename({"factor_value": "factor_clean"}) if "factor_value" in df.columns else df
        res = compute_factor_correlation({"__fz_cand__": cand, name: other}, factor_col="factor_clean")
        if len(res.factor_names) < 2:
            continue
        c = abs(float(res.corr_matrix[0][1]))  # [cand, other] 按插入序，[0][1]=候选对该因子
        if c == c and c >= best:  # 排除 NaN；并列取后出现者亦可
            best, nearest = c, name
    return best, nearest


def library_orthogonal_check(
    factor_df: pl.DataFrame,
    lib_pool: dict[str, pl.DataFrame] | None,
    *,
    threshold: float = DEFAULT_DECORR_THRESHOLD,
    panel: LibraryCorrPanel | None = None,
) -> tuple[bool, float, str | None]:
    """库相关度量：与库池 max|corr| 是否 ``>= threshold``。

    返回 ``(ok, max_corr_library, nearest_expr)``——``ok=True`` 当且仅当 max|corr| < threshold。
    ``lib_pool`` 空/None → 恒通过、corr=0（零回归）。

    **阈值由调用方按政策传入**（本函数只做度量 + 比较，不做硬拒/软信号语义）：
    - 硬拒重复：``threshold=DEFAULT_DUPLICATE_CORR``（0.95）
    - 快速通道/旧默认：``threshold=DEFAULT_DECORR_THRESHOLD``（0.7，向后兼容）
    M1 与 team/agent 双路径必须调本函数，禁止各自内联相关计算（架构守卫锁死）。

    ``panel``：可选 ``LibraryCorrPanel``（session 级构建一次）；不传则逐对原路径。
    """
    if not lib_pool:
        return True, 0.0, None
    mc, nearest = max_correlation_detail(factor_df, lib_pool, panel=panel)
    if mc >= threshold:
        return False, mc, nearest
    return True, mc, nearest


def score_candidate(factor_df: pl.DataFrame, node: Node, bundle: DataBundle,
                    pool: dict[str, pl.DataFrame], lam: float = 0.5,
                    gamma: float = 0.002) -> dict:
    train = quick_fitness(factor_df, bundle, "train")
    mc = max_correlation(factor_df, pool)
    cplx = _complexity(node)
    # 排序键用 t-stat 而非裸 IR：t-stat 自带 n>4 门槛（低样本→0），避免小样本 ic_std
    # 虚低把 IR 撑成假象（历史 rank1: ic≈2.4e-16 却 IR=14.68、n=7 排第一）。
    # ir_train 仍保留在结果里供 DSR / CSV 使用。
    tstat = train["tstat"]
    fitness = tstat - lam * mc - gamma * cplx
    return {"fitness": fitness, "ic_train": train["ic_mean"], "ir_train": train["ir"],
            "tstat_train": tstat, "max_corr": mc, "complexity": cplx, "n_train": train["n"]}


def ic_overfit_report(
    factor_df: pl.DataFrame, daily: pl.DataFrame, train_ratio: float = 1.0
) -> dict:
    """市场无关的单因子防过拟合报告：全样本 IC/IR + bootstrap IC 95%CI + DSR(N=1)。

    ``factor_df``: ``[trade_date, ts_code, factor_value]``；``daily`` 用于算前向收益。
    A 股 ``fz validate overfit`` 与 crypto 单表达式验证共用此路径（避免双实现）。
    """
    from factorzen.discovery.guardrails import DeflationBasis, deflated_pvalue
    from factorzen.validation.bootstrap import block_bootstrap_ic_ci

    bundle = DataBundle.build(daily, train_ratio=train_ratio)
    clean = cross_sectional_zscore(factor_df, col="factor_value").rename(
        {"factor_value_z": "factor_clean"}
    )
    ic_res = compute_rank_ic(
        clean.select(["trade_date", "ts_code", "factor_clean"]),
        bundle.fwd_returns, factor_col="factor_clean", frequency="daily",
    )
    ic_vals = ic_res.ic_series["ic"].drop_nulls().drop_nans().to_numpy()
    lo, hi = block_bootstrap_ic_ci(ic_vals)
    # 单因子验证：语义上不存在 trial 池，N=1 → expected_max_sharpe 返回 0（无 deflation）。
    # 仍走共享入口，使 deflated_sharpe 的导入收口在 guardrails.py 一处（架构守卫测试强制）。
    _dsr, p = deflated_pvalue(ic_res.ir, DeflationBasis(n_trials=1, sharpe_variance=1.0),
                              len(ic_vals))
    return {"ic_mean": float(ic_res.ic_mean), "ir": float(ic_res.ir),
            "dsr_p": float(p), "ci_lo": float(lo), "ci_hi": float(hi), "n": len(ic_vals)}

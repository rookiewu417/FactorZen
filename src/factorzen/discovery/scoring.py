# src/factorzen/discovery/scoring.py
"""候选因子快速评估：两段式中的「内循环」——只算 Rank IC/IR，不跑回测。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


def max_correlation(factor_df: pl.DataFrame, pool: dict[str, pl.DataFrame]) -> float:
    """factor_df 与 pool 中每个因子的截面相关性绝对值的最大值。pool 为空时返回 0。

    逐对(pairwise)计算：候选与池中**每个**因子单独算相关。这样一个退化的池因子
    (截面 std==0 / 不足 30 只 / NaN) 只会让它自己那一对得 0，不会污染其它对。
    历史 bug：把候选 + 全池一次性 inner-join 交给 compute_factor_correlation，任一
    池因子退化就 continue 丢整条截面 → count=0 → 所有真实高相关一起被抹成 0.0，
    数学等价簇因此逃过 0.7 去重门槛。不动 compute_factor_correlation（daily 报告仍用其语义）。
    """
    if not pool:
        return 0.0
    cand = (factor_df.rename({"factor_value": "factor_clean"})
            if "factor_value" in factor_df.columns else factor_df)
    best = 0.0
    for name, df in pool.items():
        other = df.rename({"factor_value": "factor_clean"}) if "factor_value" in df.columns else df
        res = compute_factor_correlation({"__fz_cand__": cand, name: other}, factor_col="factor_clean")
        if len(res.factor_names) < 2:
            continue
        c = abs(float(res.corr_matrix[0][1]))  # [cand, other] 按插入序，[0][1]=候选对该因子
        if c == c:  # 排除 NaN
            best = max(best, c)
    return best


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

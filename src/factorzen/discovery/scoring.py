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
        from datetime import datetime
        cut = datetime.strptime(self.train_end, "%Y%m%d").date()
        if segment == "train":
            return df.filter(pl.col("trade_date") <= cut)
        return df.filter(pl.col("trade_date") > cut)


def quick_fitness(factor_df: pl.DataFrame, bundle: DataBundle,
                  segment: Literal["train", "valid"] = "train") -> dict:
    """factor_df: [trade_date, ts_code, factor_value] → {ic_mean, ir, n}。"""
    seg = bundle._segment_mask(factor_df, segment)
    if seg.is_empty():
        return {"ic_mean": 0.0, "ir": 0.0, "n": 0}
    # 截面 zscore（cross_sectional_zscore 新增列 factor_value_z）
    clean = cross_sectional_zscore(seg, col="factor_value").rename({"factor_value_z": "factor_clean"})
    ret = bundle._segment_mask(bundle.fwd_returns, segment)
    res = compute_rank_ic(clean.select(["trade_date", "ts_code", "factor_clean"]),
                          ret, factor_col="factor_clean", frequency="daily")
    return {"ic_mean": res.ic_mean, "ir": res.ir, "n": res.n_periods}


def max_correlation(factor_df: pl.DataFrame, pool: dict[str, pl.DataFrame]) -> float:
    """factor_df 与 pool 中每个因子的截面相关性绝对值的最大值。pool 为空时返回 0。"""
    if not pool:
        return 0.0
    fd = {"__fz_cand__": factor_df.rename({"factor_value": "factor_clean"})
          if "factor_value" in factor_df.columns else factor_df}
    for name, df in pool.items():
        fd[name] = df.rename({"factor_value": "factor_clean"}) if "factor_value" in df.columns else df
    res = compute_factor_correlation(fd, factor_col="factor_clean")
    i = res.factor_names.index("__fz_cand__")
    corrs = [abs(res.corr_matrix[i][j]) for j in range(len(res.factor_names)) if j != i]
    return max(corrs) if corrs else 0.0


def score_candidate(factor_df: pl.DataFrame, node: Node, bundle: DataBundle,
                    pool: dict[str, pl.DataFrame], lam: float = 0.5,
                    gamma: float = 0.002) -> dict:
    train = quick_fitness(factor_df, bundle, "train")
    mc = max_correlation(factor_df, pool)
    cplx = _complexity(node)
    fitness = train["ir"] - lam * mc - gamma * cplx
    return {"fitness": fitness, "ic_train": train["ic_mean"], "ir_train": train["ir"],
            "max_corr": mc, "complexity": cplx, "n_train": train["n"]}

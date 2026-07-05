"""多因子组合方法的样本外(OOS)对比实验驱动。

在同一 PurgedWalkForwardCV 协议下评估 equal_weight/ic_weighted/max_ir/lgbm,
用统一口径(RankIC / ICIR / 分层多空 spread / 净值最大回撤)横向对比,落盘可复现
产物 + Markdown 报告。允许「ML 没赢」的诚实结论——这本身就是有价值的实验记录。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from factorzen.core.experiment import build_manifest_base, get_git_sha
from factorzen.research.combination.cv import PurgedWalkForwardCV
from factorzen.research.combination.importance import explain
from factorzen.research.combination.models import LGBMCombiner, build_panel, combine_lgbm
from factorzen.research.combination.oos import combine_oos

_LINEAR = {"equal_weight", "ic_weighted", "max_ir"}
_DEFAULT_METHODS = ["equal_weight", "ic_weighted", "max_ir", "lgbm"]


def _max_drawdown(cum: list[float]) -> float:
    peak = float("-inf")
    mdd = 0.0
    for v in cum:
        peak = max(peak, v)
        mdd = min(mdd, v - peak)
    return mdd


def _evaluate_oos(
    combined: pl.DataFrame, ret_df: pl.DataFrame, n_groups: int = 5
) -> dict[str, float]:
    """OOS 组合因子的统一评估:逐日 RankIC + 分层多空 spread,汇总指标。"""
    rdf = ret_df.with_columns(pl.col("trade_date").cast(pl.Utf8))
    m = combined.join(rdf, on=["trade_date", "ts_code"], how="inner")
    day_rows = []
    for _d, g in m.group_by("trade_date", maintain_order=True):
        if len(g) < n_groups * 2:
            continue
        f = g["factor_value"].to_numpy().astype(float)
        r = g["ret"].to_numpy().astype(float)
        if np.std(f) < 1e-12 or np.std(r) < 1e-12:
            continue
        fr = f.argsort().argsort().astype(float)
        rr = r.argsort().argsort().astype(float)
        ic = float(np.corrcoef(fr, rr)[0, 1])
        order = f.argsort()
        q = max(1, len(f) // n_groups)
        spread = float(r[order[-q:]].mean() - r[order[:q]].mean())
        if np.isfinite(ic):
            day_rows.append((str(_d[0]), ic, spread))
    if not day_rows:
        return {
            "rank_ic_mean": 0.0,
            "icir": 0.0,
            "ic_positive_ratio": 0.0,
            "top_bottom_spread": 0.0,
            "max_drawdown": 0.0,
            "n_periods": 0,
        }
    day_rows.sort(key=lambda x: x[0])
    ics = np.array([r[1] for r in day_rows])
    spreads = [r[2] for r in day_rows]
    cum, acc = [], 0.0
    for s in spreads:
        acc += s
        cum.append(acc)
    return {
        "rank_ic_mean": float(ics.mean()),
        "icir": float(ics.mean() / ics.std()) if ics.std() > 1e-12 else 0.0,
        "ic_positive_ratio": float((ics > 0).mean()),
        "top_bottom_spread": float(np.mean(spreads)),
        "max_drawdown": _max_drawdown(cum),
        "n_periods": len(day_rows),
    }


def _combine(
    method: str,
    factor_dfs: dict[str, pl.DataFrame],
    ret_df: pl.DataFrame,
    cv: PurgedWalkForwardCV,
    seed: int,
) -> pl.DataFrame:
    if method in _LINEAR:
        return combine_oos(factor_dfs, ret_df, cv, method=method)
    if method == "lgbm":
        return combine_lgbm(factor_dfs, ret_df, cv, seed=seed)
    raise ValueError(f"未知 method: {method}")


def run_combination_experiment(
    factor_dfs: dict[str, pl.DataFrame],
    ret_df: pl.DataFrame,
    *,
    cv: PurgedWalkForwardCV,
    methods: list[str] | None = None,
    seed: int = 0,
    out_dir: str = "workspace/combinations",
    run_id: str | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """跑四方法 OOS 对比,落盘 comparison/importance/report/manifest。"""
    methods = methods or list(_DEFAULT_METHODS)
    run_id = run_id or "combination"
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    importance_df: pl.DataFrame | None = None
    for method in methods:
        combined = _combine(method, factor_dfs, ret_df, cv, seed)
        combined.write_parquet(run_dir / f"combined_{method}.parquet")
        metrics = _evaluate_oos(combined, ret_df)
        rows.append({"method": method, **metrics})
        if method == "lgbm":
            panel = build_panel(factor_dfs, ret_df)
            names = list(factor_dfs.keys())
            model = LGBMCombiner(seed=seed)
            model.fit(panel.select(names), _rank_series(panel))
            importance_df = explain(model, panel.select(names))

    comparison = pl.DataFrame(rows)
    comparison.write_csv(run_dir / "comparison.csv")
    if importance_df is not None:
        importance_df.write_csv(run_dir / "importance.csv")

    manifest = build_manifest_base(list(command or []), {"seed": seed})
    manifest.update(
        {
            "git_sha": get_git_sha(),
            "run_id": run_id,
            "seed": seed,
            "methods": methods,
            "factors": list(factor_dfs.keys()),
            "cv": {
                "train_days": cv.train_days,
                "test_days": cv.test_days,
                "purge_days": cv.purge_days,
                "embargo_days": cv.embargo_days,
                "expanding": cv.expanding,
            },
        }
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(
        _render_report(run_id, comparison, importance_df), encoding="utf-8"
    )
    return {"run_dir": str(run_dir), "comparison": comparison}


def _rank_series(panel: pl.DataFrame) -> pl.Series:
    n = pl.col("ret").count().over("trade_date")
    return panel.with_columns(
        pl.when(n > 1)
        .then((pl.col("ret").rank().over("trade_date") - 1) / (n - 1) - 0.5)
        .otherwise(0.0)
        .alias("_y")
    )["_y"]


def _render_report(
    run_id: str, comparison: pl.DataFrame, importance_df: pl.DataFrame | None
) -> str:
    best = comparison.sort("rank_ic_mean", descending=True).row(0, named=True)
    lines = [
        f"# 多因子组合 OOS 对比 · {run_id}",
        "",
        "> 同一 purged & embargoed walk-forward CV 协议下的样本外对比。",
        "",
        "## 方法对比",
        "",
        "| 方法 | RankIC 均值 | ICIR | 多空 spread | 最大回撤 | IC>0 占比 | 期数 |",
        "|------|-----------|------|-----------|---------|----------|------|",
    ]
    for r in comparison.iter_rows(named=True):
        lines.append(
            f"| {r['method']} | {r['rank_ic_mean']:.4f} | {r['icir']:.3f} | "
            f"{r['top_bottom_spread']:.4%} | {r['max_drawdown']:.4%} | "
            f"{r['ic_positive_ratio']:.1%} | {r['n_periods']} |"
        )
    lines += [
        "",
        f"**最高 RankIC 方法:** `{best['method']}`(RankIC={best['rank_ic_mean']:.4f}, "
        f"ICIR={best['icir']:.3f})。",
        "",
        "> 注:RankIC 高不代表实盘更优,需结合换手/容量/稳健性综合判断;"
        "若 ML 未显著胜出,线性方法因更稳健/可解释而更可运营——诚实记录即结论。",
    ]
    if importance_df is not None:
        lines += ["", "## LightGBM 因子重要性", "", "| 因子 | 重要性 | 方法 |", "|------|-------|------|"]
        imp_sorted = importance_df.sort("importance", descending=True)
        for r in imp_sorted.iter_rows(named=True):
            lines.append(f"| {r['factor']} | {r['importance']:.4f} | {r['method']} |")
    return "\n".join(lines) + "\n"


__all__ = ["run_combination_experiment"]

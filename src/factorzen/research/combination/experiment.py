"""多因子组合方法的样本外(OOS)对比实验驱动。

在同一 PurgedWalkForwardCV 协议下评估 equal_weight/ic_weighted/max_ir/lgbm,
用统一口径(RankIC / ICIR / 分层多空 spread / 净值最大回撤)横向对比,落盘可复现
产物 + Markdown 报告。允许「ML 没赢」的诚实结论——这本身就是有价值的实验记录。
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from factorzen.config.settings import COMBINATIONS_DIR
from factorzen.core.experiment import build_manifest_base, get_git_sha
from factorzen.core.stats import spearman_avg_rank
from factorzen.research.combination.cv import PurgedWalkForwardCV
from factorzen.research.combination.importance import explain
from factorzen.research.combination.methods import (
    IcCache,
    build_ic_cache,
    pre_zscore_factors,
)
from factorzen.research.combination.models import LGBMCombiner, build_panel, combine_lgbm
from factorzen.research.combination.oos import combine_oos, drop_degenerate_factors

_LINEAR = {
    "equal_weight", "ic_weighted", "max_ir",
    # *_signed:允许负权(L1 归一化)的对应版本。加进默认对照是为了让
    # 「允许负权是否真的更好」由同协议 OOS 数据裁决,而不是靠先验假设。
    "ic_weighted_signed", "max_ir_signed",
}
# *_signed 不进默认对照:已实测在真实库上更差(见 methods._solve_max_ir_weights 的
# 对照表),留着白烧每次 run 的算力。要复检用 --methods 显式点名。
_DEFAULT_METHODS = ["equal_weight", "ic_weighted", "max_ir", "lgbm"]

# 净收益的成本假设：单边 10bp/单位成交名义额。
# A 股实际 ≈ 佣金 2.5bp + 印花税 10bp(仅卖出,摊双边 5bp) + 冲击 5~10bp ⇒ 10~15bp/边。
# 取 10bp 是**乐观**的一侧：若该档已为负，更贵的成本只会更差。
_COST_PER_SIDE = 10.0 / 1e4
_ANN_DAYS = 252


def _max_drawdown(cum: list[float]) -> float:
    peak = float("-inf")
    mdd = 0.0
    for v in cum:
        peak = max(peak, v)
        mdd = min(mdd, v - peak)
    return mdd


def _top_bucket_turnover(tops: list[frozenset[str]]) -> float:
    """相邻期 top 分位成分的变动率均值 ∈ [0,1]。

    **为什么这样定义**：组合因子是**截面打分**不是权重，没有「权重变动 L1」可算；
    而实际成本来自「按打分选股、每期换仓」，故直接量 top 桶的成分换手：
    ``|Tₜ \\ Tₜ₋₁| / |Tₜ|``。次序在桶内变动不算换手（不产生交易），
    这与用 rank 相关衡量不同——后者会把桶内洗牌也记成成本。

    不足 2 期 → 0.0（无相邻对可比，形态一致不抛）。
    """
    rates = _top_bucket_turnover_series(tops)
    return float(np.mean(rates)) if rates else 0.0


def _top_bucket_turnover_series(tops: list[frozenset[str]]) -> list[float]:
    """逐期换手率序列（``_top_bucket_turnover`` 的未聚合版）。

    净收益必须**逐日**扣费再求均值——先平均换手、再乘成本会丢掉
    「换手高的日子恰好收益也高/低」的协同，两者不等价。
    """
    if len(tops) < 2:
        return []
    return [
        len(cur - prev) / len(cur)
        for prev, cur in itertools.pairwise(tops)
        if cur
    ]


def _evaluate_oos(
    combined: pl.DataFrame, ret_df: pl.DataFrame, n_groups: int = 5
) -> dict[str, float]:
    """OOS 组合因子的统一评估:逐日 RankIC + 分层多空 spread + top 桶换手,汇总指标。

    RankIC 走 ``spearman_avg_rank``（与 lift_test / ic_analysis average-rank 主口径一致）。

    ``turnover``：报告一直写着「RankIC 高不代表实盘更优,需结合换手判断」却没算过，
    仅凭 IC 无法在 lgbm/线性方法之间下部署结论（ML 信号可能换手快得多）。

    ``net_spread_10bp`` / ``net_sharpe_10bp``：**带成本净收益**。
    2026-07-19 实测：库 120 上 lgbm 毛年化 +30.30%、换手 55.3%/日，
    A 股 10bp/边下成本年化 **27.9%**——**吃掉毛 alpha 的 92%**，净仅 +2.44%；
    四方法在现实成本下净收益全部为负或贴零。**只报 IC 会系统性误导**
    （IC 最高的 lgbm 换手也最高，两者的排序在扣费后可能反转）。

    成本口径：``spread`` 是 ``top均值 − bottom均值``，即 **1 份多头 + 1 份空头**。
    每腿每期换手 ``x`` ⇒ 该腿卖 ``x`` 买 ``x``、成交名义额 ``2x``；两腿合计 ``4x``。
    故 ``cost_t = 4 × turnover_t × cost_per_side``。
    **假设空头腿换手与多头腿相同**（只量了 top 桶）；A 股融券受限时实盘应看
    long-only 形态，届时成本约减半。
    """
    rdf = ret_df.with_columns(pl.col("trade_date").cast(pl.Utf8))
    m = combined.join(rdf, on=["trade_date", "ts_code"], how="inner")
    day_rows = []
    for _d, g in m.group_by("trade_date", maintain_order=True):
        if len(g) < n_groups * 2:
            continue
        f = g["factor_value"].to_numpy().astype(float)
        r = g["ret"].to_numpy().astype(float)
        ic = spearman_avg_rank(f, r)
        if ic is None:
            continue
        order = f.argsort()
        q = max(1, len(f) // n_groups)
        spread = float(r[order[-q:]].mean() - r[order[:q]].mean())
        codes = g["ts_code"].to_list()
        top = frozenset(codes[i] for i in order[-q:])
        day_rows.append((str(_d[0]), ic, spread, top))
    if not day_rows:
        return {
            "rank_ic_mean": 0.0,
            "icir": 0.0,
            "ic_positive_ratio": 0.0,
            "top_bottom_spread": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "net_spread_10bp": 0.0,
            "net_sharpe_10bp": 0.0,
            "n_periods": 0,
        }
    day_rows.sort(key=lambda x: x[0])
    ics = np.array([r[1] for r in day_rows])
    spreads = [r[2] for r in day_rows]
    cum, acc = [], 0.0
    for s in spreads:
        acc += s
        cum.append(acc)

    # 逐日扣费：第 0 期无前一期可比，不产生换手成本。
    # ⚠️ `_top_bucket_turnover_series` 会跳过空 top 桶，序列可能短于 len(spreads)-1；
    # 当前 `q = max(1, ...)` 保证桶非空，但那是**隐式不变量**——显式补齐，
    # 免得日后改动把长度差变成静默的广播错误。
    tops = [r[3] for r in day_rows]
    turn_series = _top_bucket_turnover_series(tops)
    fee_list = [0.0] + [4.0 * t * _COST_PER_SIDE for t in turn_series]
    fee_list += [0.0] * max(0, len(spreads) - len(fee_list))
    fees = np.array(fee_list[: len(spreads)])
    net = np.array(spreads) - fees
    net_sharpe = (
        float(net.mean() / net.std(ddof=1) * np.sqrt(_ANN_DAYS))
        if len(net) > 2 and net.std(ddof=1) > 1e-12
        else 0.0
    )
    return {
        "rank_ic_mean": float(ics.mean()),
        "icir": float(ics.mean() / ics.std()) if ics.std() > 1e-12 else 0.0,
        "ic_positive_ratio": float((ics > 0).mean()),
        "top_bottom_spread": float(np.mean(spreads)),
        "max_drawdown": _max_drawdown(cum),
        "turnover": float(np.mean(turn_series)) if turn_series else 0.0,
        "net_spread_10bp": float(net.mean()),
        "net_sharpe_10bp": net_sharpe,
        "n_periods": len(day_rows),
    }


def _combine(
    method: str,
    factor_dfs: dict[str, pl.DataFrame],
    ret_df: pl.DataFrame,
    cv: PurgedWalkForwardCV,
    seed: int,
    *,
    ic_cache: IcCache | None = None,
    z_factor_dfs: dict[str, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    if method in _LINEAR:
        return combine_oos(
            factor_dfs, ret_df, cv, method=method,
            ic_cache=ic_cache, z_factor_dfs=z_factor_dfs,
        )
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
    out_dir: str = str(COMBINATIONS_DIR),
    run_id: str | None = None,
    command: list[str] | None = None,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """跑四方法 OOS 对比,落盘 comparison/importance/report/manifest。

    ``extra_config``：调用方的运行参数（窗口/票池/选品口径等），合并进 manifest
    的 ``config``。**不传则 manifest 只有 seed，事后无法判断这次 run 覆盖了什么**
    ——2026-07-19 追查数据污染时就因此只能去读 combined parquet 反推窗口
    （CLAUDE.md：manifest 记全命令/窗口，漏了=假复现）。
    """
    methods = methods or list(_DEFAULT_METHODS)
    run_id = run_id or "combination"
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 跨方法共享预计算:IC 序列与截面 z-score 按日独立,全样本一次、各线性方法复用
    # (combine_oos 内部会对 None 按需自建,此处 hoist 只省重复,不改数值)。
    linear = [m for m in methods if m in _LINEAR]
    ic_cache = (
        build_ic_cache(drop_degenerate_factors(factor_dfs), ret_df)
        if any(m in ("ic_weighted", "max_ir") for m in linear) else None
    )
    z_factor_dfs = (
        pre_zscore_factors(drop_degenerate_factors(factor_dfs)) if linear else None
    )

    rows: list[dict[str, Any]] = []
    importance_df: pl.DataFrame | None = None
    for method in methods:
        combined = _combine(
            method, factor_dfs, ret_df, cv, seed,
            ic_cache=ic_cache, z_factor_dfs=z_factor_dfs,
        )
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

    manifest = build_manifest_base(
        list(command or []), {"seed": seed, **(extra_config or {})},
    )
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
    # 净 SR 最优未必与 RankIC 最优同一个方法——这正是加这列的理由
    _net_col = "net_sharpe_10bp" if "net_sharpe_10bp" in comparison.columns else "icir"
    best_net = comparison.sort(_net_col, descending=True).row(0, named=True)
    lines = [
        f"# 多因子组合 OOS 对比 · {run_id}",
        "",
        "> 同一 purged & embargoed walk-forward CV 协议下的样本外对比。",
        "",
        "## 方法对比",
        "",
        "| 方法 | RankIC 均值 | ICIR | 多空 spread | 换手 | **净 spread@10bp** | "
        "**净 SR@10bp** | 最大回撤 | IC>0 占比 | 期数 |",
        "|------|-----------|------|-----------|------|-----------|------|"
        "---------|----------|------|",
    ]
    for r in comparison.iter_rows(named=True):
        lines.append(
            f"| {r['method']} | {r['rank_ic_mean']:.4f} | {r['icir']:.3f} | "
            f"{r['top_bottom_spread']:.4%} | {r.get('turnover', 0.0):.1%} | "
            f"**{r.get('net_spread_10bp', 0.0):.4%}** | "
            f"**{r.get('net_sharpe_10bp', 0.0):.2f}** | "
            f"{r['max_drawdown']:.4%} | "
            f"{r['ic_positive_ratio']:.1%} | {r['n_periods']} |"
        )
    lines += [
        "",
        f"**最高 RankIC 方法:** `{best['method']}`(RankIC={best['rank_ic_mean']:.4f}, "
        f"ICIR={best['icir']:.3f}, 换手={best.get('turnover', 0.0):.1%}, "
        f"净 SR@10bp={best.get('net_sharpe_10bp', 0.0):.2f})。",
        f"**净 SR@10bp 最高方法:** `{best_net['method']}`"
        f"(净 SR={best_net.get('net_sharpe_10bp', 0.0):.2f}, "
        f"RankIC={best_net['rank_ic_mean']:.4f})"
        + ("——**与 RankIC 最优不是同一个方法**,按 IC 选会选错。"
           if best_net["method"] != best["method"] else "。"),
        "",
        "> 注:RankIC 高不代表实盘更优,需结合换手/容量/稳健性综合判断;"
        "若 ML 未显著胜出,线性方法因更稳健/可解释而更可运营——诚实记录即结论。",
        "",
        "> **净收益口径**:`净 spread = spread − 4 × 换手 × 10bp`(逐期扣费再平均)。"
        "spread 是 `top均值 − bottom均值`,即 1 份多头 + 1 份空头;每腿换手 x ⇒ "
        "卖 x 买 x、成交 2x,两腿合计 4x。**假设空头腿换手与多头腿相同**。"
        "10bp/边是**乐观**侧(A 股实际 10~15bp),该档已为负则更贵只会更差。"
        "A 股融券受限时实盘应看 long-only 形态,成本约减半。",
        "",
        "> ⚠️ **2026-07-19 实测**:库 120 上 lgbm 毛年化 +30.30%、换手 55.3%,"
        "10bp 下成本年化 **27.9%**——**吃掉毛 alpha 的 92%**,净仅 +2.44%;"
        "四方法在现实成本下净收益全部为负或贴零。**只看 IC 会系统性高估可部署性。**",
        "",
        "> **换手口径**:相邻期 **top 分位成分**的变动率均值 `|Tₜ\\Tₜ₋₁|/|Tₜ|`。"
        "组合因子是截面打分不是权重,故量成分换手而非权重 L1;桶内次序变动不计"
        "(不产生交易)。跨方法比较时:RankIC 相近而换手高者,成本后大概率更差。",
    ]
    if importance_df is not None:
        lines += ["", "## LightGBM 因子重要性", "", "| 因子 | 重要性 | 方法 |", "|------|-------|------|"]
        imp_sorted = importance_df.sort("importance", descending=True)
        for r in imp_sorted.iter_rows(named=True):
            lines.append(f"| {r['factor']} | {r['importance']:.4f} | {r['method']} |")
    return "\n".join(lines) + "\n"


__all__ = ["run_combination_experiment"]

# src/factorzen/discovery/mining_session.py
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.discovery.expression import compile_expr, parse_expr, to_expr_string
from factorzen.discovery.scoring import DataBundle, max_correlation, quick_fitness, score_candidate
from factorzen.discovery.search.random_search import RandomSearcher


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _factor_values(node, daily: pl.DataFrame) -> pl.DataFrame:
    df = daily.sort(["ts_code", "trade_date"]).with_columns(compile_expr(node).alias("factor_value"))
    return df.select(["trade_date", "ts_code", "factor_value"]).filter(
        pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())


def run_session(daily: pl.DataFrame, *, n_trials: int, top_k: int, seed: int,
                method: str = "random", train_ratio: float = 0.7,
                out_dir: str = "workspace/mining_sessions") -> dict:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    # 停牌掩码（与 ExpressionFactor 一致，保证挖掘内 IC 与 fz factor run 一致）+ 派生列
    daily = daily.sort(["ts_code", "trade_date"])
    _price = ["open", "high", "low", "close", "open_adj", "high_adj", "low_adj", "close_adj", "vol", "amount"]
    daily = daily.with_columns([
        pl.when(pl.col("vol") > 0).then(pl.col(c)).otherwise(None).alias(c)
        for c in _price if c in daily.columns
    ]).with_columns([
        (pl.col("amount") / pl.col("vol")).alias("vwap"),
        (pl.col("vol") + 1.0).log().alias("log_vol"),
    ]).with_columns(
        (pl.col("close_adj") / pl.col("close_adj").shift(1).over("ts_code") - 1.0).alias("ret_1d"))
    bundle = DataBundle.build(daily, train_ratio=train_ratio)

    # ── 按 method 选择候选节点列表 ─────────────────────────────────────
    if method == "genetic":
        from factorzen.discovery.search.genetic import GeneticSearcher
        gs = GeneticSearcher(rng, max_depth=3)
        cache: dict[str, float] = {}

        def _score(node):
            expr = to_expr_string(node)
            if expr in cache:
                return cache[expr]
            try:
                fdf = _factor_values(node, daily)
                val = score_candidate(fdf, node, bundle, pool={})["fitness"] if fdf.height >= 50 else -9.9
            except Exception:
                val = -9.9
            cache[expr] = val
            return val

        nodes = gs.evolve(_score, pop_size=max(20, n_trials // 5),
                          generations=max(3, n_trials // 40))
        candidate_nodes = nodes
    else:
        searcher = RandomSearcher(rng, max_depth=3)
        candidate_nodes = [searcher.propose() for _ in range(n_trials)]

    # ── 统一评分循环（random 与 genetic 共用）────────────────────────────
    scored: list[dict] = []
    seen: set[str] = set()
    n_errors = 0
    last_err: Exception | None = None
    for node in candidate_nodes:
        expr = to_expr_string(node)
        if expr in seen:
            continue
        seen.add(expr)
        try:
            fdf = _factor_values(node, daily)
            if fdf.height < 50:
                continue
            sc = score_candidate(fdf, node, bundle, pool={})
            if sc["n_train"] < 5:
                continue
            valid = quick_fitness(fdf, bundle, "valid")
            scored.append({"expression": expr, "ic_train": sc["ic_train"],
                           "ir_train": sc["ir_train"], "ic_valid": valid["ic_mean"],
                           "ir_valid": valid["ir"], "max_corr": sc["max_corr"],
                           "complexity": sc["complexity"], "fitness": sc["fitness"]})
        except Exception as e:
            n_errors += 1
            last_err = e
            continue
    if not scored and n_errors > 0:
        raise RuntimeError(
            f"run_session: 未产出任何有效候选，且 {n_errors} 次评分抛异常; last error: {last_err}"
        ) from last_err

    scored.sort(key=lambda d: d["fitness"], reverse=True)
    # 贪心去相关选 top-K：每个入选候选记录与「已选池」的真实 max_corr，过滤近重复
    selected = []
    selected_pool = {}  # expression -> factor_df
    for cand in scored:
        if len(selected) >= top_k:
            break
        try:
            fdf = _factor_values(parse_expr(cand["expression"]), daily)
        except Exception:
            continue
        mc = max_correlation(fdf, selected_pool)
        if mc < 0.7:
            cand = {**cand, "max_corr": round(float(mc), 4)}
            selected.append(cand)
            selected_pool[cand["expression"]] = fdf
    top = selected

    session_dir = Path(out_dir) / f"session_{seed}_{method}"
    session_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"rank": i + 1, **{k: c[k] for k in
             ["expression", "ic_train", "ir_train", "ic_valid", "ir_valid", "max_corr", "complexity"]}}
            for i, c in enumerate(top)]
    pl.DataFrame(rows).write_csv(session_dir / "candidates.csv") if rows else \
        (session_dir / "candidates.csv").write_text(
            "rank,expression,ic_train,ir_train,ic_valid,ir_valid,max_corr,complexity\n")
    manifest = {"seed": seed, "method": method, "n_trials": n_trials, "top_k": top_k,
                "train_end": bundle.train_end, "git_sha": _git_sha(),
                "duration_seconds": round(time.perf_counter() - t0, 3), "candidates": top,
                "reproduce_note": "导出因子在 exported/；复现需复制到 workspace/factors/daily/ 后 fz factor run <name> --set preprocessing.neutralize=false（IC parity）"}
    (session_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    from factorzen.discovery.export import export_candidate
    exported_dir = session_dir / "exported"
    for i, c in enumerate(top):
        export_candidate(c["expression"], f"mined_{seed}_{i+1}", str(exported_dir))
    return {"candidates": top, "n_trials": n_trials, "session_dir": str(session_dir)}

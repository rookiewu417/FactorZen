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
from factorzen.validation.deflated_sharpe import deflated_sharpe
from factorzen.validation.holdout import holdout_ic, split_holdout
from factorzen.validation.pbo import compute_pbo


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _factor_values(node, daily: pl.DataFrame, eval_start=None) -> pl.DataFrame:
    df = daily.sort(["ts_code", "trade_date"]).with_columns(compile_expr(node).alias("factor_value"))
    out = df.select(["trade_date", "ts_code", "factor_value"]).filter(
        pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
    if eval_start is not None:
        from datetime import datetime
        out = out.filter(pl.col("trade_date") >= datetime.strptime(eval_start, "%Y%m%d").date())
    return out


def _pool_pbo(scored: list, daily: pl.DataFrame, bundle, eval_start=None) -> float:
    """对 scored 候选（mining 段）构造日度 IC 矩阵跑 PBO；样本不足返回 nan。"""
    from factorzen.daily.evaluation.ic_analysis import compute_rank_ic
    from factorzen.daily.preprocessing.normalizer import cross_sectional_zscore
    series = []
    dates_ref = None
    for c in scored[:30]:  # 取 fitness 前 30 个候选，控制成本
        try:
            fdf = _factor_values(parse_expr(c["expression"]), daily, eval_start)
            clean = cross_sectional_zscore(fdf, col="factor_value").rename({"factor_value_z": "factor_clean"})
            ic_res = compute_rank_ic(clean.select(["trade_date", "ts_code", "factor_clean"]),
                                     bundle.fwd_returns, factor_col="factor_clean", frequency="daily")
            ser = ic_res.ic_series.sort("trade_date")
            if dates_ref is None:
                dates_ref = ser["trade_date"]
            ser = ser.join(pl.DataFrame({"trade_date": dates_ref}), on="trade_date", how="right").sort("trade_date")
            series.append(ser["ic"].fill_null(0.0).to_numpy())
        except Exception:
            continue
    if len(series) < 2:
        return float("nan")
    import numpy as _np
    return compute_pbo(_np.vstack(series), n_splits=10)


def run_session(daily: pl.DataFrame, *, n_trials: int, top_k: int, seed: int,
                method: str = "random", train_ratio: float = 0.7,
                holdout_ratio: float = 0.2,
                eval_start: str | None = None,
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

    # ── OOS holdout 永久隔离：挖掘只见 mining 段 ──
    mining_df, holdout_df, holdout_start = split_holdout(daily, holdout_ratio=holdout_ratio)
    daily = mining_df  # 后续挖掘全部只用 mining 段（DataBundle/搜索/去相关）
    bundle = DataBundle.build(daily, train_ratio=train_ratio)

    # eval_cache 提到 method 分支之前，供 genetic 统计真实评估数
    eval_cache: dict[str, float] = {}

    # ── 按 method 选择候选节点列表 ─────────────────────────────────────
    if method == "genetic":
        from factorzen.discovery.search.genetic import GeneticSearcher
        gs = GeneticSearcher(rng, max_depth=3)

        def _score(node):
            expr = to_expr_string(node)
            if expr in eval_cache:
                return eval_cache[expr]
            try:
                fdf = _factor_values(node, daily, eval_start)
                val = score_candidate(fdf, node, bundle, pool={})["fitness"] if fdf.height >= 50 else -9.9
            except Exception:
                val = -9.9
            eval_cache[expr] = val
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
            fdf = _factor_values(node, daily, eval_start)
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
    selected: list[dict] = []
    selected_pool: dict[str, pl.DataFrame] = {}  # expression -> factor_df
    for cand in scored:
        if len(selected) >= top_k:
            break
        try:
            fdf = _factor_values(parse_expr(cand["expression"]), daily, eval_start)
        except Exception:
            continue
        mc = max_correlation(fdf, selected_pool)
        if mc < 0.7:
            cand = {**cand, "max_corr": round(float(mc), 4)}
            selected.append(cand)
            selected_pool[cand["expression"]] = fdf
    top = selected

    # ── 护栏验收（holdout 只用一次）──
    from factorzen.validation.multiple_testing import TrialLedger
    # random=去重评估数；genetic=evolve 内评估的不同表达式数（eval_cache）
    eval_n = len(eval_cache) if method == "genetic" else len(seen)
    ledger = TrialLedger()
    ledger.record(eval_n)
    n_evaluated = ledger.n_trials

    if eval_start is not None:
        from datetime import datetime as _dt
        _es_date = _dt.strptime(eval_start, "%Y%m%d").date()
        n_obs_mining = daily.filter(pl.col("trade_date") >= _es_date)["trade_date"].n_unique()
    else:
        n_obs_mining = daily["trade_date"].n_unique()  # mining 段交易日数 ≈ IC 序列长度
    ir_pool = np.array([c["ir_train"] for c in scored]) if scored else np.array([0.0])
    sharpe_var = float(ir_pool.var()) if ir_pool.size > 1 else 1.0
    pbo = _pool_pbo(scored, daily, bundle, eval_start)  # 候选池(mining 段)日度 IC 矩阵 → PBO
    for c in top:
        node = parse_expr(c["expression"])
        fdf_hold = _factor_values(node, holdout_df)
        if fdf_hold.height >= 20:
            h_ic, _h_ir, (ci_lo, _ci_hi) = holdout_ic(fdf_hold, holdout_df)
        else:
            h_ic, ci_lo = float("nan"), float("nan")
        _dsr, p = deflated_sharpe(c["ir_train"], n_evaluated, n_obs_mining, sharpe_variance=sharpe_var)
        c["n_trials"] = n_evaluated
        c["pbo"] = round(pbo, 4) if pbo == pbo else float("nan")
        c["holdout_ic"] = round(float(h_ic), 4) if h_ic == h_ic else float("nan")
        c["dsr_pvalue"] = round(float(p), 4)
        c["ic_ci_low"] = round(float(ci_lo), 4) if ci_lo == ci_lo else float("nan")

    session_dir = Path(out_dir) / f"session_{seed}_{method}"
    session_dir.mkdir(parents=True, exist_ok=True)
    _cols = ["expression", "ic_train", "ir_train", "ic_valid", "ir_valid", "max_corr",
             "complexity", "holdout_ic", "dsr_pvalue", "pbo", "ic_ci_low"]
    rows = [{"rank": i + 1, "n_trials": n_evaluated, **{k: c.get(k) for k in _cols}} for i, c in enumerate(top)]
    pl.DataFrame(rows).write_csv(session_dir / "candidates.csv") if rows else \
        (session_dir / "candidates.csv").write_text("rank,n_trials," + ",".join(_cols) + "\n")
    manifest = {"seed": seed, "method": method, "n_trials": n_evaluated, "cli_n_trials": n_trials,
                "top_k": top_k, "train_end": bundle.train_end, "holdout_start": str(holdout_start),
                "git_sha": _git_sha(), "duration_seconds": round(time.perf_counter() - t0, 3),
                "candidates": top,
                "reproduce_note": "导出因子在 exported/；复现需复制到 workspace/factors/daily/ 后 fz factor run <name> --set preprocessing.neutralize=false（IC parity）"}
    (session_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    from factorzen.discovery.export import export_candidate
    exported_dir = session_dir / "exported"
    for i, c in enumerate(top):
        export_candidate(c["expression"], f"mined_{seed}_{i+1}", str(exported_dir))
    return {"candidates": top, "n_trials": n_evaluated, "session_dir": str(session_dir),
            "holdout_start": str(holdout_start), "mining_end": str(daily["trade_date"].max())}

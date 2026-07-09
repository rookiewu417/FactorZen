# src/factorzen/discovery/mining_session.py
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.core.experiment import get_git_sha
from factorzen.discovery.derived import add_derived_columns
from factorzen.discovery.expression import evaluate_materialized, parse_expr, to_expr_string
from factorzen.discovery.guardrails import (
    DeflationBasis,
    deflated_pvalue,
    guardrail_passed,
)
from factorzen.discovery.operators import LEAF_FEATURES
from factorzen.discovery.scoring import DataBundle, max_correlation, quick_fitness, score_candidate
from factorzen.discovery.search.random_search import RandomSearcher
from factorzen.validation.holdout import holdout_ic, split_holdout
from factorzen.validation.pbo import compute_pbo


def _factor_values(node, daily: pl.DataFrame, eval_start=None, leaf_map=None) -> pl.DataFrame:
    df = daily.sort(["ts_code", "trade_date"])
    df = df.with_columns(
        evaluate_materialized(node, df, leaf_map).alias("factor_value"))
    out = df.select(["trade_date", "ts_code", "factor_value"]).filter(
        pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
    if eval_start is not None:
        from factorzen.discovery.scoring import _cut_literal
        out = out.filter(pl.col("trade_date") >= _cut_literal(out, eval_start))
    return out


def _oos_adjusted_fitness(train_fitness: float, train_tstat: float, valid_tstat: float) -> float:
    """把 valid 段 OOS 一致性折进排序键：valid t-stat 与 train **反号**时按 ``|valid_tstat|`` 扣分。

    train 与 valid 的 t-stat 同尺度，扣 ``|valid_tstat|`` 是无 magic 系数、尺度一致、连续的降权，
    直接把「train 高但 valid 反号」的过拟合候选压到一致候选之后。valid 样本不足（HAC t-stat=0，
    n≤4）时不调整（保守，不足以判 OOS 一致性）。历史上 ic_valid/ir_valid 算了只写 CSV、不进选择。
    """
    if train_tstat != 0.0 and valid_tstat != 0.0 and (train_tstat > 0) != (valid_tstat > 0):
        return train_fitness - abs(valid_tstat)
    return train_fitness


def _guard_passed(c: dict, dsr_alpha: float = 0.05) -> bool:
    """防过拟合护栏软标记：DSR 显著(p<dsr_alpha) & holdout IC 与 train 同号 & holdout IC 95%CI 下界>0。

    任一指标缺失/NaN → 判否(保守)。护栏历史上「只算不判」——四个指标算出来只写进 CSV，
    候选入选只看 fitness 排序，过拟合垃圾照样导出。这里把它变成可被 leaderboard/export-alpha
    默认过滤的软标记(留 --all 逃生口)，不删候选、不破坏产物契约。
    """
    return guardrail_passed(
        ic_train=c.get("ic_train"),
        holdout_ic=c.get("holdout_ic"),
        dsr_pvalue=c.get("dsr_pvalue"),
        ci_low=c.get("ic_ci_low"),
        dsr_alpha=dsr_alpha,
    )


def _cross_section_variability(fdf: pl.DataFrame) -> float:
    """有截面变异的交易日占比 ∈ [0,1]。近常数因子(多数截面 std≈0)→接近 0。

    R7 退化过滤用：随机/遗传会大量产出截面恒定的表达式(常数、amplitude=high-low=0 等),
    它们无截面信号且会拖累去相关/护栏,应在打分前剔除。
    """
    if fdf.is_empty():
        return 0.0
    g = fdf.group_by("trade_date").agg(pl.col("factor_value").std().alias("s"))
    s = g["s"].fill_null(0.0).to_numpy()
    return float(np.mean(s > 1e-12)) if s.size else 0.0


def _rank_fingerprint(fdf: pl.DataFrame, n_dates: int = 4) -> str | None:
    """截面 rank 签名指纹(sha1)。单调(同向)变换的因子截面 rank 序完全一致 → 同指纹。

    R5 去重用:字符串去重挡不住 neg(amount)/sub(2,amount)/neg(abs(amount)) 这类数学等价簇
    (表达式串不同、rank IC 逐位相同)。对均匀取样的几个交易日取截面平均 rank(ties 稳健)哈希,
    同时并入该日的 ts_code 成员集 → 不同 universe 不会误并。**不做符号规范化**:X 与 −X 是
    相反方向的赌注,预打分阶段合并会有丢掉正确符号因子的风险;反向由 top-K 的 |corr| 门槛收尾。
    样本日不足(<2)返回 None(不去重)。
    """
    dates = fdf.select("trade_date").unique().sort("trade_date")["trade_date"].to_list()
    if len(dates) < 2:
        return None
    idx = sorted(set(np.linspace(0, len(dates) - 1, min(n_dates, len(dates))).round().astype(int).tolist()))
    h = hashlib.sha1()
    for i in idx:
        cross = fdf.filter(pl.col("trade_date") == dates[i]).sort("ts_code")
        h.update("|".join(cross["ts_code"].to_list()).encode())
        ranks = cross["factor_value"].rank(method="average").to_list()
        h.update((",".join(f"{float(x):.1f}" for x in ranks)).encode())
        h.update(b";")
    return h.hexdigest()


def _pool_pbo(scored: list, daily: pl.DataFrame, bundle, eval_start=None, leaf_map=None) -> float:
    """对 scored 候选（mining 段）构造日度 IC 矩阵跑 PBO；样本不足返回 nan。"""
    from factorzen.daily.evaluation.ic_analysis import compute_rank_ic
    from factorzen.daily.preprocessing.normalizer import cross_sectional_zscore
    series = []
    dates_ref = None
    for c in scored[:30]:  # 取 fitness 前 30 个候选，控制成本
        try:
            fdf = _factor_values(parse_expr(c["expression"], leaf_map), daily, eval_start, leaf_map)
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
                decorr_threshold: float = 0.7, min_n_train: int = 5,
                dsr_alpha: float = 0.05,
                eval_start: str | None = None,
                out_dir: str = "workspace/mining_sessions",
                profile=None, workers: int = 1) -> dict:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    daily = daily.sort(["ts_code", "trade_date"])
    # 叶子集/列映射/派生列由 MarketProfile.factors 注入（profile=None → A 股默认）。
    if profile is not None:
        leaf_map: dict[str, str] = profile.factors.leaf_features()
        leaves: list[str] | None = list(leaf_map.keys())
        daily = profile.factors.derived_columns(daily)
    else:
        leaf_map = LEAF_FEATURES
        leaves = None  # 搜索用 random_search 默认 A 股叶子
        # 停牌掩码（与 ExpressionFactor 一致）
        _price = ["open", "high", "low", "close", "open_adj", "high_adj", "low_adj",
                  "close_adj", "vol", "amount"]
        daily = daily.with_columns([
            pl.when(pl.col("vol") > 0).then(pl.col(c)).otherwise(None).alias(c)
            for c in _price if c in daily.columns
        ])
        # A 股派生列（与 factor.py 共用 add_derived_columns，消除双路径漂移
        # + amplitude/intraday_ret/overnight_ret 派生叶子）
        daily = add_derived_columns(daily)

    # ── OOS holdout 永久隔离：挖掘只见 mining 段 ──
    mining_df, holdout_df, holdout_start = split_holdout(daily, holdout_ratio=holdout_ratio)
    daily = mining_df  # 后续挖掘全部只用 mining 段（DataBundle/搜索/去相关）
    bundle = DataBundle.build(daily, train_ratio=train_ratio)

    # eval_cache 提到 method 分支之前，供 genetic 统计真实评估数
    eval_cache: dict[str, float] = {}
    # eval_ir：genetic 跨代评估过的每个唯一表达式的 train 段 IR，供 DSR 的 N 与
    # sharpe_variance 同源计算（见下方护栏验收处 F6）。
    eval_ir: dict[str, float] = {}

    # ── 按 method 选择候选节点列表 ─────────────────────────────────────
    if method == "genetic":
        from factorzen.discovery.search.genetic import GeneticSearcher
        gs = GeneticSearcher(rng, max_depth=3, leaves=leaves)

        def _score_one(node):
            try:
                fdf = _factor_values(node, daily, eval_start, leaf_map)
                if fdf.height < 50:
                    return -9.9
                sc = score_candidate(fdf, node, bundle, pool={})
                # 记录该表达式的 train IR，供 DSR 的 N 与 sharpe_var 同源（跨代全体评估）
                eval_ir[to_expr_string(node)] = float(sc["ir_train"])
                return sc["fitness"]
            except Exception:
                return -9.9

        def _score(node):
            expr = to_expr_string(node)
            if expr not in eval_cache:
                eval_cache[expr] = _score_one(node)
            return eval_cache[expr]

        def _score_many(nodes):
            # 批量预热缓存:去重未评估表达式,workers>1 时线程池并行(polars 求值释放 GIL);
            # 缓存键为表达式串、值只依赖表达式,填充顺序无关 → 与串行完全等价(确定性)。
            uniq: dict = {}
            for n in nodes:
                e = to_expr_string(n)
                if e not in eval_cache:
                    uniq.setdefault(e, n)
            if not uniq:
                return
            if workers > 1:
                from concurrent.futures import ThreadPoolExecutor

                items = list(uniq.items())
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    vals = list(ex.map(lambda it: _score_one(it[1]), items))
                for (e, _n), v in zip(items, vals, strict=True):
                    eval_cache[e] = v
            else:
                for e, n in uniq.items():
                    eval_cache[e] = _score_one(n)

        nodes = gs.evolve(_score, pop_size=max(20, n_trials // 5),
                          generations=max(3, n_trials // 40),
                          score_many=_score_many)
        candidate_nodes = nodes
    else:
        searcher = RandomSearcher(rng, max_depth=3, leaves=leaves)
        candidate_nodes = [searcher.propose() for _ in range(n_trials)]

    # ── 统一评分循环（random 与 genetic 共用）────────────────────────────
    scored: list[dict] = []
    seen: set[str] = set()
    seen_fp: set[str] = set()  # 截面 rank 指纹，合并数学等价/单调簇（R5）
    n_errors = 0
    last_err: Exception | None = None
    for node in candidate_nodes:
        expr = to_expr_string(node)
        if expr in seen:
            continue
        seen.add(expr)
        try:
            fdf = _factor_values(node, daily, eval_start, leaf_map)
            if fdf.height < 50:
                continue
            # R7 退化过滤：多数截面近常数的因子无信号，打分前剔除
            if _cross_section_variability(fdf) < 0.5:
                continue
            # R5 指纹去重：单调/符号等价簇（neg(amount)/2-amount/…）rank 序相同 → 只留首个
            fp = _rank_fingerprint(fdf)
            if fp is not None:
                if fp in seen_fp:
                    continue
                seen_fp.add(fp)
            sc = score_candidate(fdf, node, bundle, pool={})
            if sc["n_train"] < min_n_train:
                continue
            valid = quick_fitness(fdf, bundle, "valid")
            # R6：把 valid OOS 一致性折进排序键——valid 反号候选被降权（历史上算了不用）
            fitness = _oos_adjusted_fitness(sc["fitness"], sc["tstat_train"], valid["tstat"])
            scored.append({"expression": expr, "ic_train": sc["ic_train"],
                           "ir_train": sc["ir_train"], "ic_valid": valid["ic_mean"],
                           "ir_valid": valid["ir"], "max_corr": sc["max_corr"],
                           "complexity": sc["complexity"], "fitness": fitness,
                           "n_train": sc["n_train"]})
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
            fdf = _factor_values(parse_expr(cand["expression"], leaf_map), daily, eval_start, leaf_map)
        except Exception:
            continue
        mc = max_correlation(fdf, selected_pool)
        if mc < decorr_threshold:
            cand = {**cand, "max_corr": round(float(mc), 4)}
            selected.append(cand)
            selected_pool[cand["expression"]] = fdf
    top = selected

    # ── 护栏验收（holdout 只用一次）──
    from factorzen.validation.multiple_testing import TrialLedger
    # R8: DSR 的 N 与 sharpe_variance 必须同源（同一批 trial），否则 expected_max_sharpe
    # 的 deflation 基准不自洽。二者都取「真实评估过、拿到有效 Sharpe(IR) 的唯一表达式」population。
    # F6：random 路径这个 population 就是存活集 scored（候选即评估过的全部）；genetic 路径则是
    # 跨代 eval_ir——因为 elitism 使最终代最优即全程 argmax，选择实际发生在整个搜索空间上，而非
    # 仅最终代存活集 len(scored)≈pop_size；只数最终代会系统性低估 N、放松 DSR（passed 偏松，危险方向）。
    if method == "genetic" and eval_ir:
        ir_pool = list(eval_ir.values())
    else:
        ir_pool = [c["ir_train"] for c in scored] if scored else []
    # N 与 sharpe_variance 同源，且与 Agent 路径共用同一份配方（架构守卫测试禁止绕过）
    basis = DeflationBasis.from_ir_pool(ir_pool)
    ledger = TrialLedger()
    ledger.record(basis.n_trials)
    n_evaluated = ledger.n_trials
    pbo = _pool_pbo(scored, daily, bundle, eval_start, leaf_map)  # 候选池日度 IC 矩阵 → PBO
    for c in top:
        node = parse_expr(c["expression"], leaf_map)
        fdf_hold = _factor_values(node, holdout_df, leaf_map=leaf_map)
        if fdf_hold.height >= 20:
            h_ic, _h_ir, (ci_lo, _ci_hi) = holdout_ic(fdf_hold, holdout_df)
        else:
            h_ic, ci_lo = float("nan"), float("nan")
        # DSR 显著性检验须用该候选自己在 train 段的真实样本数(n_train)，
        # 不能用 mining 全段交易日数——后者比 train 段大约 1/train_ratio 倍，
        # 会系统性放大显著性（让候选看起来比实际更显著，危险方向）。
        _dsr, p = deflated_pvalue(c["ir_train"], basis, c["n_train"])
        c["n_trials"] = n_evaluated
        c["pbo"] = round(pbo, 4) if pbo == pbo else float("nan")
        c["holdout_ic"] = round(float(h_ic), 4) if h_ic == h_ic else float("nan")
        c["dsr_pvalue"] = round(float(p), 4)
        c["ic_ci_low"] = round(float(ci_lo), 4) if ci_lo == ci_lo else float("nan")
        # 护栏软标记：算完立刻判，供 leaderboard/export-alpha 默认过滤（--all 逃生口）
        c["passed"] = _guard_passed(c, dsr_alpha)

    session_dir = Path(out_dir) / f"session_{seed}_{method}"
    session_dir.mkdir(parents=True, exist_ok=True)
    _cols = ["expression", "ic_train", "ir_train", "ic_valid", "ir_valid", "max_corr",
             "complexity", "holdout_ic", "dsr_pvalue", "pbo", "ic_ci_low", "passed"]
    rows = [{"rank": i + 1, "n_trials": n_evaluated, **{k: c.get(k) for k in _cols}} for i, c in enumerate(top)]
    pl.DataFrame(rows).write_csv(session_dir / "candidates.csv") if rows else \
        (session_dir / "candidates.csv").write_text("rank,n_trials," + ",".join(_cols) + "\n")
    manifest = {"seed": seed, "method": method, "n_trials": n_evaluated, "cli_n_trials": n_trials,
                # deflation 门槛由 (n_trials, sharpe_variance) 共同决定，属可复现必要信息
                "sharpe_variance": basis.sharpe_variance,
                "top_k": top_k, "train_end": bundle.train_end, "holdout_start": str(holdout_start),
                "git_sha": get_git_sha(), "duration_seconds": round(time.perf_counter() - t0, 3),
                "candidates": top,
                "reproduce_note": "导出因子在 exported/；复现需复制到 workspace/factors/daily/ 后 fz factor run <name> --set preprocessing.neutralize=false（IC parity）"}
    (session_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    from factorzen.discovery.export import export_candidate
    exported_dir = session_dir / "exported"
    for i, c in enumerate(top):
        export_candidate(c["expression"], f"mined_{seed}_{i+1}", str(exported_dir))
    return {"candidates": top, "n_trials": n_evaluated, "n_scored": len(scored),
            "sharpe_variance": basis.sharpe_variance,
            "session_dir": str(session_dir),
            "holdout_start": str(holdout_start), "mining_end": str(daily["trade_date"].max())}

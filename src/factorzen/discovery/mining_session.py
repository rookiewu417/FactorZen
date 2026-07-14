# src/factorzen/discovery/mining_session.py
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.core.experiment import get_git_sha
from factorzen.discovery.derived import add_derived_columns
from factorzen.discovery.expression import (
    evaluate_materialized,
    parse_expr,
    to_expr_string,
    warmup_shortfall,
)
from factorzen.discovery.guardrails import (
    DEFAULT_DSR_ALPHA,
    DEFAULT_DUPLICATE_CORR,
    DEFAULT_GATE,
    DEFAULT_RESIDUAL_IC_FLOOR,
    REJECT_CATEGORY_LIFT_QUEUE,
    DeflationBasis,
    acceptance_reasons,
    deflated_pvalue,
    is_lift_queue_candidate,
)
from factorzen.discovery.leaf_health import (
    apply_leaf_exclusion,
    filter_leaves_by_holdout_coverage,
    log_excluded_leaves,
)
from factorzen.discovery.operators import LEAF_FEATURES
from factorzen.discovery.residual import (
    build_library_panel,
    compute_residual_ic,
    resolve_objective,
)
from factorzen.discovery.scoring import (
    DEFAULT_DECORR_THRESHOLD,
    DataBundle,
    library_orthogonal_check,
    max_correlation,
    quick_fitness,
    score_candidate,
)
from factorzen.discovery.search.random_search import RandomSearcher
from factorzen.validation.holdout import holdout_ic_result, split_holdout
from factorzen.validation.pbo import compute_pbo

_LOG = logging.getLogger(__name__)


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


def _underwarmed(node, daily: pl.DataFrame, eval_start, leaf_map=None) -> bool:
    """预热不足判定：某叶子 ``可用预热 < 自身 path-lookback`` → True（该表达式该拒）。

    与 agent `evaluate_expressions` 同一道共享门（`warmup_shortfall`）：逐叶对照
    `leaf_lookbacks`，避免『浅派生叶拖垮深 raw 叶路』的假拒绝，两条路判定天然一致，
    消除双路径漂移。M1 此前对超预热表达式不拒绝，让段首截断窗口噪声（`operators._MIN = 3`
    窗口不满照常出值）进 train IC。被拒的表达式在 random 循环 `continue`、genetic
    `_score_one` 提前 `return`，都在写 scored/eval_ir 之前，故不计入 DSR 的 N（与 agent 一致）。

    ``eval_start=None``（旧调用方）→ 不判（False），零回归。``daily`` 须派生列已物化
    （`run_session` 在 `add_derived_columns` 后调用）。``leaf_map`` 透传给 `warmup_shortfall`
    以支持 crypto 等非 A 股叶子映射。
    """
    if eval_start is None:
        return False
    from factorzen.discovery.scoring import _cut_literal
    return warmup_shortfall(node, daily, _cut_literal(daily, eval_start), leaf_map) is not None


def _oos_adjusted_fitness(train_fitness: float, train_tstat: float, valid_tstat: float) -> float:
    """把 valid 段 OOS 一致性折进排序键：valid t-stat 与 train **反号**时按 ``|valid_tstat|`` 扣分。

    train 与 valid 的 t-stat 同尺度，扣 ``|valid_tstat|`` 是无 magic 系数、尺度一致、连续的降权，
    直接把「train 高但 valid 反号」的过拟合候选压到一致候选之后。valid 样本不足（HAC t-stat=0，
    n≤4）时不调整（保守，不足以判 OOS 一致性）。历史上 ic_valid/ir_valid 算了只写 CSV、不进选择。
    """
    if train_tstat != 0.0 and valid_tstat != 0.0 and (train_tstat > 0) != (valid_tstat > 0):
        return train_fitness - abs(valid_tstat)
    return train_fitness


def _guard_reasons(c: dict, dsr_alpha: float = DEFAULT_DSR_ALPHA,
                   gate: str = DEFAULT_GATE, *,
                   objective: str = "raw") -> list[str]:
    """护栏未通过原因（空=入池）。session 内**唯一**的 acceptance_reasons 派发点。

    2026-07「因子库化」：默认 ``gate="library"`` —— 真(holdout 与 train 同号) + 有信号
    (|train_IC|≥floor)，**不含 DSR 单星显著性**（显著性挪到组合层 `fz combine run`）。
    ``gate="strict"`` 回到 DSR 显著+同号（松一档 alpha 0.10）。
    任一必需指标缺失/NaN → 判缺失(保守)。与 Agent `node_guardrails` **共用**
    `acceptance_reasons`（含 holdout_n_days 覆盖门）。

    ``objective="residual"``：用残差指标 + ``DEFAULT_RESIDUAL_IC_FLOOR`` 判定；
    裸 IC 字段仍在 ``c`` 里供报告。``"raw"``（默认本函数）喂裸 IC。
    """
    if objective == "residual":
        return acceptance_reasons(
            gate=gate,
            ic_train=c.get("residual_ic_train"),
            holdout_ic=c.get("residual_holdout_ic"),
            dsr_pvalue=c.get("dsr_pvalue"),
            ci_low=c.get("ic_ci_low"),
            dsr_alpha=dsr_alpha,
            ic_floor=DEFAULT_RESIDUAL_IC_FLOOR,
            holdout_n_days=c.get("n_residual_holdout_days"),
            reason_style="residual",
        )
    return acceptance_reasons(
        gate=gate,
        ic_train=c.get("ic_train"),
        holdout_ic=c.get("holdout_ic"),
        dsr_pvalue=c.get("dsr_pvalue"),
        ci_low=c.get("ic_ci_low"),
        dsr_alpha=dsr_alpha,
        holdout_n_days=c.get("n_holdout_days") if c.get("n_holdout_days") is not None
        else c.get("holdout_n_days"),
    )


def _guard_passed(c: dict, dsr_alpha: float = DEFAULT_DSR_ALPHA,
                  gate: str = DEFAULT_GATE, *,
                  objective: str = "raw") -> bool:
    """护栏软标记(passed)：``gate`` 口径下入池即 True。委托 `_guard_reasons`（单源）。"""
    return not _guard_reasons(c, dsr_alpha, gate, objective=objective)


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


def _library_upsert_session(top, *, seed, method, session_dir, warmup_daily, mining_daily,
                            leaf_map, profile, eval_start, decorr_threshold, library_root,
                            library_universe, horizon, out_dir) -> None:
    """M1 收尾把 passed 候选 upsert 进因子库。全 try/except 兜底，不拖垮挖掘产出。"""
    from datetime import date

    try:
        passed = [c for c in top if c.get("passed")]
        if not passed:
            return
        from factorzen.discovery import factor_library as _fl
        market = getattr(profile, "name", None) or "ashare"
        root = library_root or str(Path(out_dir).parent / "factor_library")
        _start = eval_start or warmup_daily["trade_date"].min().strftime("%Y%m%d")
        _end = warmup_daily["trade_date"].max().strftime("%Y%m%d")
        # 去相关用紧凑矩阵物化器（内存有界）：mining 段已含派生列，直接建网格。
        compact = _fl.make_compact_materializer(
            mining_daily.sort(["ts_code", "trade_date"]), leaf_map)

        _fl.upsert(
            market, passed, eval_window=(_start, _end), universe=library_universe,
            horizon=horizon, run_id=session_dir.name, session_dir=str(session_dir),
            git_sha=get_git_sha(), now=date.today().strftime("%Y-%m-%d"),
            decorr_threshold=decorr_threshold, compact_materialize=compact,
            leaf_map=leaf_map, root=root)
    except Exception as exc:  # 库写入失败不许影响挖掘产出（A股零回归底线）
        _LOG.warning("因子库 upsert 失败（不影响挖掘产出）: %s: %s", type(exc).__name__, exc)


def run_session(daily: pl.DataFrame, *, n_trials: int, top_k: int, seed: int,
                method: str = "random", train_ratio: float = 0.7,
                holdout_ratio: float = 0.2,
                decorr_threshold: float = DEFAULT_DECORR_THRESHOLD, min_n_train: int = 5,
                dsr_alpha: float = DEFAULT_DSR_ALPHA,
                eval_start: str | None = None,
                out_dir: str = "workspace/mining_sessions",
                profile=None, workers: int = 1,
                update_library: bool = True, library_root: str | None = None,
                library_universe: str | None = None, horizon: int = 1,
                library_orthogonal: bool = True,
                objective: str = "residual") -> dict:
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
    # 完整帧留作 holdout 段的扩窗预热前缀（滚动算子在 holdout 边界需要 mining 末尾的历史）。
    # PIT 安全：mining 整体早于 holdout，时序算子只向过去看；求值后裁剪到 >= holdout_start。
    warmup_daily = daily
    holdout_eval_start = holdout_start.strftime("%Y%m%d")

    # 开局摘死叶：holdout 有效截面覆盖不足的叶子移出本 session 搜索空间（不硬删 LEAF 定义）。
    # 此时 daily 已 derived_columns；与 Agent 的 _preprocess_daily 同序（价列别名在 A 股分支已处理）。
    _leaf_names = list(leaf_map.keys()) if leaves is None else list(leaves)
    _kept, excluded_leaves = filter_leaves_by_holdout_coverage(
        daily, _leaf_names, holdout_start, leaf_map=leaf_map,
    )
    log_excluded_leaves(excluded_leaves, prefix="mine-session")
    leaves, _filtered_map = apply_leaf_exclusion(_leaf_names, leaf_map, excluded_leaves)
    # leaf_map 可能被物化为子集；A 股默认 LEAF_FEATURES 常量本身不变。
    # 本函数内 leaf_map 恒非 None（A 股默认已落 LEAF_FEATURES），仅收窄 Optional 类型。
    leaf_map = _filtered_map if _filtered_map is not None else leaf_map

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
                if _underwarmed(node, daily, eval_start, leaf_map):
                    return -9.9   # 预热不足：与 agent 门一致，不评估、不进 eval_ir（不计入 N）
                fdf = _factor_values(node, daily, eval_start, leaf_map)
                if fdf.height < 50:
                    return -9.9
                sc = score_candidate(fdf, node, bundle, pool={})
                # 与 random 路径同一道门（见下方 `if sc["n_train"] < min_n_train: continue`）。
                # `quick_fitness` 对「求值后无任何有效截面」的表达式返回 sentinel ic=ir=0.0
                # （不是 nan），`DeflationBasis.from_ir_pool` 剔不掉有限值 0.0。放进 eval_ir 会
                # 同时膨胀 N 并压低经验方差，而 `expected_max_sharpe ∝ sqrt(var)` 使后者占优
                # → deflation 门槛系统性偏低（实测真实 csi300 genetic run：3.7% 死表达式 → sr0 -1.4%）。
                # n_train=0 的表达式没有可比较的 IR，永远不可能是 max，本就不该计入多重检验的 N。
                if sc["n_train"] < min_n_train:
                    return -9.9
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
            if _underwarmed(node, daily, eval_start, leaf_map):
                continue   # 预热不足：与 agent 门一致，不评估、不进 scored（不计入 N）
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

    # 库池：session 开始物化一次。
    # 残差目标需要 train+holdout 两段库因子 → 在完整 warmup 帧上物化；
    # 库相关门仍用 mining 段切片（selected 循环里用 mining fdf 对齐）。
    # 空库/关开关 → {}，行为与旧完全一致。
    lib_pool: dict[str, pl.DataFrame] = {}
    lib_pool_mining: dict[str, pl.DataFrame] = {}
    lib_root = library_root or str(Path(out_dir).parent / "factor_library")
    if library_orthogonal:
        try:
            from factorzen.discovery.factor_library import build_library_pool
            market = getattr(profile, "name", None) or "ashare"
            # 全窗（mining∪holdout）一次物化，残差 train/holdout 共用
            lib_pool = build_library_pool(market, warmup_daily, leaf_map, root=lib_root)
            # 库相关检查与 session 去相关同帧（mining）：按 mining 日期过滤
            if lib_pool:
                _mine_dates = set(daily["trade_date"].unique().to_list())
                lib_pool_mining = {
                    e: p.filter(pl.col("trade_date").is_in(list(_mine_dates)))
                    for e, p in lib_pool.items()
                }
                lib_pool_mining = {e: p for e, p in lib_pool_mining.items() if not p.is_empty()}
            else:
                lib_pool_mining = {}
        except Exception as exc:
            _LOG.warning("build_library_pool 失败，本 session 跳过库级正交: %s: %s",
                         type(exc).__name__, exc)
            lib_pool, lib_pool_mining = {}, {}

    lib_panel = build_library_panel(lib_pool)
    eff_objective = resolve_objective(objective, lib_panel is not None)

    # 贪心去相关选 top-K：先库**重复**硬门（corr>0.95），再与已选池去相关。
    # 库相关 (0.7, 0.95] 不硬拒——继续评估，由下方软 reason 挡快速通道、可入 lift 队列。
    # 共用 library_orthogonal_check；政策阈值在调用方（双路径与 team 一致）。
    selected: list[dict] = []
    selected_pool: dict[str, pl.DataFrame] = {}  # expression -> factor_df
    n_library_correlated_rejects = 0
    n_gray_zone = 0  # manifest 字段名兼容；语义= lift_queue 计数
    for cand in scored:
        if len(selected) >= top_k:
            break
        try:
            fdf = _factor_values(parse_expr(cand["expression"], leaf_map), daily, eval_start, leaf_map)
        except Exception:
            continue
        ok_lib, mc_lib, _nearest = library_orthogonal_check(
            fdf, lib_pool_mining or lib_pool, threshold=DEFAULT_DUPLICATE_CORR,
        )
        if not ok_lib:
            n_library_correlated_rejects += 1
            continue
        mc = max_correlation(fdf, selected_pool)
        if mc < decorr_threshold:
            cand = {**cand, "max_corr": round(float(mc), 4)}
            if lib_pool:
                cand["max_corr_library"] = round(float(mc_lib), 4)
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
    # holdout 前向收益（残差 holdout IC 复用，避免每候选重算）
    _hold_fwd = None
    if eff_objective == "residual" and lib_panel is not None:
        from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns
        _pc = "close_adj" if "close_adj" in holdout_df.columns else "close"
        _hold_fwd = compute_fwd_returns(
            holdout_df.sort(["ts_code", "trade_date"]), price_col=_pc,
        )
    for c in top:
        node = parse_expr(c["expression"], leaf_map)
        fdf_hold = _factor_values(node, warmup_daily, holdout_eval_start, leaf_map)
        hres = holdout_ic_result(fdf_hold, holdout_df)
        h_ic, ci_lo, n_h = hres.ic_mean, hres.ci[0], hres.n_days
        # DSR 显著性检验须用该候选自己在 train 段的真实样本数(n_train)，
        # 不能用 mining 全段交易日数——后者比 train 段大约 1/train_ratio 倍，
        # 会系统性放大显著性（让候选看起来比实际更显著，危险方向）。
        _dsr, p = deflated_pvalue(c["ir_train"], basis, c["n_train"])
        c["n_trials"] = n_evaluated
        c["pbo"] = round(pbo, 4) if pbo == pbo else float("nan")
        c["holdout_ic"] = round(float(h_ic), 4) if h_ic == h_ic else float("nan")
        c["n_holdout_days"] = int(n_h)
        c["dsr_pvalue"] = round(float(p), 4)
        c["ic_ci_low"] = round(float(ci_lo), 4) if ci_lo == ci_lo else float("nan")
        # 残差双指标（只对 top-K；裸 IC 已在上方）。_hold_fwd 非 None 与前置块同条件，
        # 显式收窄仅为 mypy。
        if eff_objective == "residual" and lib_panel is not None and _hold_fwd is not None:
            fdf_train = selected_pool.get(c["expression"])
            if fdf_train is None:
                fdf_train = _factor_values(node, daily, eval_start, leaf_map)
            r_tr = compute_residual_ic(fdf_train, lib_panel, bundle.fwd_returns)
            r_h = compute_residual_ic(fdf_hold, lib_panel, _hold_fwd)
            c["residual_ic_train"] = float(r_tr.ic_mean) if r_tr.ic_mean == r_tr.ic_mean else float("nan")
            c["residual_holdout_ic"] = float(r_h.ic_mean) if r_h.ic_mean == r_h.ic_mean else float("nan")
            c["n_residual_holdout_days"] = int(r_h.n_days)
        # 护栏软标记：算完立刻判，供 leaderboard/export-alpha 默认过滤（--all 逃生口）
        # residual 模式喂残差指标；与 Agent 共用 acceptance_reasons（经 _guard_reasons 单点）。
        # 库相关 (0.7, 0.95]：附加软 reason 挡快速通道（passed=False），不硬拒、不进 known_invalid。
        _reasons = _guard_reasons(c, dsr_alpha, objective=eff_objective)
        _mc_lib = c.get("max_corr_library")
        if _mc_lib is not None:
            try:
                if abs(float(_mc_lib)) >= DEFAULT_DECORR_THRESHOLD:
                    _reasons = [*_reasons, f"库相关持保留(corr={float(_mc_lib):.2f})"]
            except (TypeError, ValueError):
                pass
        c["passed"] = not _reasons
        if _reasons:
            c["reject_reason"] = "；".join(_reasons)
        # 第二通道：单因子门不过但可入 lift 队列 → 标记待组合裁决（挖掘内不跑 lift）。
        if not c["passed"] and is_lift_queue_candidate(c, objective=eff_objective):
            c["reject_category"] = REJECT_CATEGORY_LIFT_QUEUE
            prev = c.get("reject_reason") or ""
            c["reject_reason"] = prev + "(lift队列,待组合裁决)"
            n_gray_zone += 1

    session_dir = Path(out_dir) / f"session_{seed}_{method}"
    session_dir.mkdir(parents=True, exist_ok=True)
    _cols = ["expression", "ic_train", "ir_train", "ic_valid", "ir_valid", "max_corr",
             "complexity", "holdout_ic", "dsr_pvalue", "pbo", "ic_ci_low", "passed",
             "residual_ic_train", "residual_holdout_ic", "n_residual_holdout_days"]
    rows = [{"rank": i + 1, "n_trials": n_evaluated, **{k: c.get(k) for k in _cols}} for i, c in enumerate(top)]
    pl.DataFrame(rows).write_csv(session_dir / "candidates.csv") if rows else \
        (session_dir / "candidates.csv").write_text("rank,n_trials," + ",".join(_cols) + "\n")
    manifest = {"seed": seed, "method": method, "n_trials": n_evaluated, "cli_n_trials": n_trials,
                # deflation 门槛由 (n_trials, sharpe_variance) 共同决定，属可复现必要信息
                "sharpe_variance": basis.sharpe_variance,
                "top_k": top_k, "train_end": bundle.train_end, "holdout_start": str(holdout_start),
                "git_sha": get_git_sha(), "duration_seconds": round(time.perf_counter() - t0, 3),
                "candidates": top,
                "excluded_leaves": excluded_leaves,
                "library_pool_size": len(lib_pool),
                "n_library_correlated_rejects": n_library_correlated_rejects,
                "n_gray_zone": n_gray_zone,
                "objective": eff_objective,
                "reproduce_note": "导出因子在 exported/；复现需复制到 workspace/factors/daily/ 后 fz factor run <name> --set preprocessing.neutralize=false（IC parity）"}
    (session_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    from factorzen.discovery.export import export_candidate
    exported_dir = session_dir / "exported"
    for i, c in enumerate(top):
        export_candidate(c["expression"], f"mined_{seed}_{i+1}", str(exported_dir))

    # ── 自动维护因子库（M1 收尾 upsert）─────────────────────────────────────────
    # 只收 passed（library gate）者；市场从 profile.name 取（None→ashare）。库根默认由 out_dir
    # 推导（workspace/mining_sessions → workspace/factor_library；测试的 tmp out_dir 天然隔离）。
    # 整块 try/except 兜底：库写入是收尾副作用，绝不能拖垮挖掘产出（A股零回归底线）。
    if update_library:
        _library_upsert_session(
            top, seed=seed, method=method, session_dir=session_dir, warmup_daily=warmup_daily,
            mining_daily=daily, leaf_map=leaf_map, profile=profile, eval_start=eval_start,
            decorr_threshold=decorr_threshold, library_root=library_root,
            library_universe=library_universe, horizon=horizon, out_dir=out_dir)

    return {"candidates": top, "n_trials": n_evaluated, "n_scored": len(scored),
            "sharpe_variance": basis.sharpe_variance,
            "session_dir": str(session_dir),
            "holdout_start": str(holdout_start), "mining_end": str(daily["trade_date"].max()),
            "excluded_leaves": excluded_leaves}

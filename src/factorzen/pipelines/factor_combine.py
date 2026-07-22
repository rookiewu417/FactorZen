"""fz combine run 流水线:加载因子/收益 parquet → 四方法 OOS 对比实验。

因子 parquet 需含 [trade_date, ts_code, factor_value] 的**整段面板**(来源:因子评估产物
等含时间序列的因子面板);收益 parquet 需含 [trade_date, ts_code, ret](对齐到因子日的
前向收益)。因子名取文件名 stem。
注意:`fz mine export-alpha` 产物是 [ts_code, alpha] 的**单日截面**(缺 trade_date/
factor_value 列且只有一天),不能直接喂本流水线(walk-forward CV 需时间序列)。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from factorzen.config.settings import COMBINATIONS_DIR
from factorzen.research.combination.cv import PurgedWalkForwardCV
from factorzen.research.combination.experiment import run_combination_experiment


def run_factor_combination(
    *,
    factor_files: list[str],
    ret_file: str,
    train_days: int = 120,
    test_days: int = 20,
    purge_days: int = 5,
    embargo_days: int = 0,
    methods: list[str] | None = None,
    seed: int = 0,
    out_dir: str = str(COMBINATIONS_DIR),
    run_id: str | None = None,
    command: list[str] | None = None,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """从 parquet 加载因子/收益,跑 OOS 对比实验。

    ``extra_config`` 透传进 manifest 的 ``config``（窗口/票池/选品口径等可复现字段）。
    """
    factor_dfs: dict[str, pl.DataFrame] = {}
    for f in factor_files:
        name = Path(f).stem
        factor_dfs[name] = pl.read_parquet(f).select(
            ["trade_date", "ts_code", "factor_value"]
        )
    ret_df = pl.read_parquet(ret_file).select(["trade_date", "ts_code", "ret"])
    cv = PurgedWalkForwardCV(
        train_days=train_days,
        test_days=test_days,
        purge_days=purge_days,
        embargo_days=embargo_days,
    )
    return run_combination_experiment(
        factor_dfs,
        ret_df,
        cv=cv,
        methods=methods,
        seed=seed,
        out_dir=out_dir,
        run_id=run_id,
        command=command,
        extra_config=extra_config,
    )


def _load_session_candidates(session_dirs: list[str], *, passed_only: bool) -> list[dict]:
    """合并多 session 的 candidates.csv → 规范形去重（跨 run 同表达式只留一条，保 |holdout_ic|
    高者）→ 按 |holdout_ic| 降序返回 [{expression(规范形), holdout_ic}]。

    规范形 = ``to_expr_string(parse_expr(expr))``：抹平空格/等价书写差异，避免同一因子跨 run
    重复入组合、把有效 breadth 稀释成假象（IR≈IC·√breadth 里 breadth 必须正交）。
    """
    from pathlib import Path

    from factorzen.discovery.expression import parse_expr, to_expr_string

    best: dict[str, dict] = {}
    for sd in session_dirs:
        cand_csv = Path(sd) / "candidates.csv"
        if not cand_csv.exists():
            raise FileNotFoundError(f"找不到 {cand_csv}（需要挖掘 session 的因子库）")
        df = pl.read_csv(cand_csv)
        if passed_only and "passed" in df.columns:
            df = df.filter(pl.col("passed").cast(pl.Utf8).str.to_lowercase() == "true")
        has_ic = "holdout_ic" in df.columns
        for row in df.iter_rows(named=True):
            raw = row["expression"]
            try:
                canon = to_expr_string(parse_expr(str(raw)))
            except ValueError:
                canon = str(raw)              # 无法解析的原样保留（下游物化会再 try）
            ic = row.get("holdout_ic") if has_ic else None
            try:
                ic_val = float(ic) if ic is not None and str(ic) != "" else 0.0
            except (TypeError, ValueError):
                ic_val = 0.0
            prev = best.get(canon)
            if prev is None or abs(ic_val) > abs(prev["holdout_ic"]):
                best[canon] = {"expression": canon, "holdout_ic": ic_val}
    # |holdout_ic| 降序（贪心去相关按此序纳入：先纳信号强者，剔与之高相关的弱者）
    return sorted(best.values(), key=lambda r: -abs(r["holdout_ic"]))


def _greedy_decorrelate_reference(
    materialized: list[tuple[str, pl.DataFrame]], threshold: float
) -> tuple[list[tuple[str, pl.DataFrame]], list[dict]]:
    """旧版实现（``max_correlation`` 逐对 + 2x argmax 重算），仅 parity 测试引用。

    生产路径见 ``_greedy_decorrelate``（紧凑 float64 矩阵 + 一次算完 mc/partner）。
    """
    from factorzen.discovery.scoring import max_correlation

    kept: list[tuple[str, pl.DataFrame]] = []
    dropped: list[dict] = []
    for expr, fdf in materialized:
        pool = {e: d for e, d in kept}
        mc = max_correlation(fdf, pool)
        if kept and mc > threshold:
            # argmax 伙伴：逐个已纳入因子单算相关（复用 max_correlation 的单元素池语义）
            partner, best = None, -1.0
            for e, d in kept:
                c = max_correlation(fdf, {e: d})
                if c > best:
                    best, partner = c, e
            dropped.append({"identity": expr, "corr_with": partner, "corr": float(mc)})
            continue
        kept.append((expr, fdf))
    return kept, dropped


def _greedy_decorrelate(
    materialized: list[tuple[str, pl.DataFrame]], threshold: float
) -> tuple[list[tuple[str, pl.DataFrame]], list[dict]]:
    """按传入顺序（|holdout_ic| 降序）贪心纳入：与已纳入池 max|corr| > threshold 者剔除。

    加速：全部面板在**共享 date×stock 网格**（日期并集 × 股票并集）上一次转 float64
    紧凑矩阵，循环内调 ``_avg_cs_corr_matrices``（与 ``max_correlation`` 语义一致，
    见 ``test_compact_corr_parity_with_max_correlation``）。同一批逐对结果同时给出
    ``mc`` 与 argmax ``corr_with``，消灭旧版 2x 重算。

    **决策 parity 硬约束**：全日期（不截断）、float64、退化对 → 0.0；
    ``threshold=1.0`` 时 ``> 1.0`` 恒 False → 逃生口。kept 中 fdf 仍是**原面板**
    （下游写 parquet）；紧凑矩阵仅内部加速。
    """
    import numpy as np

    from factorzen.discovery.factor_library import _avg_cs_corr_matrices, _panel_to_compact

    if not materialized:
        return [], []

    # 共享网格：日期并集 × 股票并集（异质覆盖 → 缺位 NaN；全日期，不截断）
    dates: set = set()
    stocks: set = set()
    for _e, fdf in materialized:
        if fdf.height:
            dates.update(fdf["trade_date"].to_list())
            stocks.update(fdf["ts_code"].to_list())
    date_idx = {d: i for i, d in enumerate(sorted(dates))}
    stock_idx = {s: i for i, s in enumerate(sorted(stocks))}
    d_n, s_n = len(date_idx), len(stock_idx)

    mats: list[np.ndarray] = []
    for _e, fdf in materialized:
        if d_n == 0 or s_n == 0 or not fdf.height:
            mats.append(np.full((max(d_n, 1), max(s_n, 1)), np.nan, dtype=np.float64))
        else:
            mats.append(_panel_to_compact(
                fdf, date_idx, stock_idx, d_n, s_n, dtype=np.float64,
            ))

    kept: list[tuple[str, pl.DataFrame]] = []
    kept_mats: list[np.ndarray] = []
    kept_exprs: list[str] = []
    dropped: list[dict] = []

    for (expr, fdf), mat in zip(materialized, mats, strict=True):
        if not kept:
            kept.append((expr, fdf))
            kept_mats.append(mat)
            kept_exprs.append(expr)
            continue
        # 一次扫完：mc = max|corr|，partner = 严格 > 的首个 argmax（对齐 reference）
        partner, best = None, -1.0
        for e, km in zip(kept_exprs, kept_mats, strict=True):
            c = abs(_avg_cs_corr_matrices(mat, km))
            if c > best:
                best, partner = c, e
        mc = float(best) if best >= 0.0 else 0.0
        if mc > threshold:
            dropped.append({"identity": expr, "corr_with": partner, "corr": mc})
            continue
        kept.append((expr, fdf))
        kept_mats.append(mat)
        kept_exprs.append(expr)
    return kept, dropped


def combine_from_session(
    *,
    session_dir: str | None = None,
    session_dirs: list[str] | None = None,
    start: str,
    end: str,
    universe: str | None = None,
    horizon: int = 5,
    passed_only: bool = True,
    top_n: int | None = None,
    decorr_threshold: float = 0.7,
    out_dir: str = str(COMBINATIONS_DIR),
    train_days: int = 120,
    test_days: int = 20,
    purge_days: int = 5,
    embargo_days: int = 0,
    methods: list[str] | None = None,
    seed: int = 0,
    run_id: str | None = None,
) -> dict[str, Any]:
    """挖掘产的**因子库** → 组合层验收的端到端接线。

    从挖掘 session 的 ``candidates.csv`` 取表达式(默认只取 ``passed=True`` 的库因子)，在
    ``[start, end]`` × ``universe`` 上逐因子物化因子值 + `horizon` 日前向收益面板，喂给
    `run_factor_combination`(四方法 + PurgedWalkForwardCV OOS)。这是「不再专注单明星、
    构造因子库 → 组合」的最后一环:显著性/过拟合的把关在这里(组合级 OOS)，而非单因子 DSR。

    ``session_dirs``：多 session（``nargs="+"``），各 session 的 candidates.csv 合并 + 规范形去重
    （跨 run 同表达式只留一条）。``session_dir``：单 session 的向后兼容别名。
    ``decorr_threshold``：物化后、喂组合前按 ``|holdout_ic|`` 降序**贪心去相关**（复用
    `max_correlation`）：与已纳入因子相关性 > 阈值者剔除并记入返回的 ``dropped_correlated``；
    ``1.0`` 关闭（逃生口）。跨 run 合并时 `ts_rank(turnover_rate,20)`/`(...,21)` 这类近亲会重复
    入组合、塌缩有效 breadth，故合并层必须去相关（agent 路径的池内去相关只管单 run）。

    因子在含预热前缀的完整帧上求值、裁剪到 ``>= start``(扩窗预热，同挖掘路径)。
    库因子 < 2 个、可物化 < 2 个、或去相关后 < 2 个 → 报错(组合至少需两个)。
    """
    import tempfile
    from datetime import datetime
    from pathlib import Path

    from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns
    from factorzen.discovery.evaluation import _factor_df_from_prepped, _preprocess_daily
    from factorzen.discovery.expression import parse_expr
    from factorzen.pipelines.factor_mine import prepare_mining_daily

    dirs = list(session_dirs) if session_dirs else ([session_dir] if session_dir else [])
    if not dirs:
        raise ValueError("需要至少一个 session（session_dirs 或 session_dir）。")

    rows = _load_session_candidates(dirs, passed_only=passed_only)
    if top_n:
        rows = rows[:top_n]
    if len(rows) < 2:
        raise ValueError(
            f"因子库不足 2 个（得 {len(rows)}），无法组合；放宽 passed_only 或多挖一些因子。")

    # 库内/session 表达式若引用 i_* / ix_*，自动装日内面板（否则物化静默全 null）
    from factorzen.discovery.preparation import (
        expressions_need_intraday,
        intraday_expr_leaf_names,
    )
    _exprs = [str(r.get("expression") or "") for r in rows]
    need_intraday = expressions_need_intraday(_exprs)
    ix_leaves = intraday_expr_leaf_names(_exprs)
    daily = prepare_mining_daily(
        start, end, universe,
        intraday=need_intraday or bool(ix_leaves),
        intraday_expr_leaves=ix_leaves or None,
    )
    prepped = _preprocess_daily(daily)  # 预处理一次，逐因子复用
    start_date = datetime.strptime(start, "%Y%m%d").date()

    # 物化到内存（去相关需因子面板算相关性），再对存活者落 parquet 喂组合。
    materialized: list[tuple[str, pl.DataFrame]] = []
    for row in rows:
        e = row["expression"]
        try:
            fdf = _factor_df_from_prepped(parse_expr(e), prepped, eval_start=start_date)
        except Exception:
            continue
        materialized.append((e, fdf.select(["trade_date", "ts_code", "factor_value"])))
    if len(materialized) < 2:
        raise ValueError(f"可物化的库因子不足 2 个（得 {len(materialized)}）。")

    kept, dropped = _greedy_decorrelate(materialized, decorr_threshold)
    if len(kept) < 2:
        raise ValueError(
            f"去相关后库因子不足 2 个（得 {len(kept)}，剔除 {len(dropped)} 个高相关近亲）；"
            f"放宽 decorr_threshold 或多挖正交因子。")

    # 落盘：可读名（与 from-library 同款 default_name_for_expression；冲突加 _2）
    # factors_used 仍返回表达式列表（契约不变）
    from factorzen.discovery.factor_library import _normalize, default_name_for_expression

    work = Path(tempfile.mkdtemp(prefix="combine_mat_"))
    factor_files: list[str] = []
    used_names: set[str] = set()
    for e, fdf in kept:
        base = default_name_for_expression(_normalize(e))
        if base not in used_names:
            fname = base
        else:
            i = 2
            while f"{base}_{i}" in used_names:
                i += 1
            fname = f"{base}_{i}"
        used_names.add(fname)
        safe = fname.replace("/", "_").replace("\\", "_")
        p = work / f"{safe}.parquet"
        fdf.write_parquet(p)
        factor_files.append(str(p))

    price_col = "close_adj" if "close_adj" in daily.columns else "close"
    fwd = compute_fwd_returns(daily.sort(["ts_code", "trade_date"]), horizons=[horizon],
                              price_col=price_col)
    ret = (
        fwd.filter(pl.col("trade_date") >= start_date)
        .select(["trade_date", "ts_code", pl.col(f"fwd_ret_{horizon}d").alias("ret")])
        .filter(pl.col("ret").is_not_null())
    )
    ret_file = work / "ret.parquet"
    ret.write_parquet(ret_file)

    res = run_factor_combination(
        factor_files=factor_files, ret_file=str(ret_file),
        train_days=train_days, test_days=test_days, purge_days=purge_days,
        embargo_days=embargo_days, methods=methods, seed=seed,
        out_dir=out_dir, run_id=run_id, command=["combine", "from-session"],
    )
    res["factors_used"] = [e for e, _ in kept]
    res["dropped_correlated"] = dropped
    return res


def combine_from_library(
    *,
    market: str = "ashare",
    statuses: tuple[str, ...] = ("active",),
    library_root: str | None = None,
    start: str,
    end: str,
    universe: str | None = None,
    horizon: int = 5,
    top_n: int | None = None,
    decorr_threshold: float = 0.7,
    out_dir: str = str(COMBINATIONS_DIR),
    train_days: int = 120,
    test_days: int = 20,
    purge_days: int = 5,
    embargo_days: int = 0,
    methods: list[str] | None = None,
    seed: int = 0,
    run_id: str | None = None,
    no_store: bool = False,
    store_root: str | None = None,
) -> dict[str, Any]:
    """因子库登记簿 → 选品 → 物化 → 贪心去相关 → 四方法 OOS 组合。

    与 ``combine_from_session`` 同构的消费闭环，但数据源是 ``factor_library``
    （expression + python 统一登记簿），而非挖掘 session 的 candidates.csv。

    约束：
    - 选品按 ``|ic_train|`` 降序（与 ``build_library_pool`` 同序）；``top_n`` 截断
      必须记 ``truncated_from``（禁止静默 cap）。
    - python 型在组合路径 **fail-loudly**：``universe is None`` 且存在 python 记录
      → ``ValueError``（与库池「跳过+告警」不同——组合少一条腿直接改实验结论）。
    - 单记录物化失败跳过并记入 ``skipped_materialize``（与 from-session except-continue 同款）。
    - 落盘/组合报告用可读 ``name``，不用 ``factor_{i}``。
    - expression 型优先读 factor_store 物化 parquet（同源物化路径）；``no_store`` 强制重算。
    """
    import logging
    import tempfile
    from collections import Counter
    from datetime import datetime
    from pathlib import Path

    from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns
    from factorzen.discovery.evaluation import _factor_df_from_prepped, _preprocess_daily
    from factorzen.discovery.expression import parse_expr
    from factorzen.discovery.factor_library import (
        DEFAULT_ROOT,
        _is_python_record,
        _materialize_python_on_grid,
        _normalize,
        _pool_date_bounds,
        _python_name_from_expression,
        default_name_for_expression,
        is_python_identity,
        library_file_hash,
        load_library,
    )
    from factorzen.discovery.factor_store import (
        load_materialized_factor,
        store_root_for_library,
    )
    from factorzen.discovery.preparation import (
        expressions_need_intraday,
        intraday_expr_leaf_names,
    )
    from factorzen.pipelines.factor_mine import prepare_mining_daily

    _log = logging.getLogger(__name__)

    root = library_root if library_root is not None else DEFAULT_ROOT
    s_root = store_root if store_root is not None else store_root_for_library(root)
    recs = [r for r in load_library(market, root=root) if r.status in statuses]
    # 与 build_library_pool 同序：|ic_train| 降序，expression 作稳定 tie-break
    recs.sort(key=lambda r: (-abs(r.ic_train or 0.0), r.expression))

    truncated_from: int | None = None
    if top_n is not None and len(recs) > top_n:
        truncated_from = len(recs)
        recs = recs[:top_n]

    if len(recs) < 2:
        raise ValueError(
            f"因子库不足 2 个（得 {len(recs)}，statuses={statuses}）；"
            f"放宽 statuses 或先挖矿/lift-test 入库。"
        )

    # 组合选品不许静默缺腿：有 python 记录就必须给 universe
    if universe is None and any(_is_python_record(r) for r in recs):
        raise ValueError(
            "combine from-library：选品含 python 型因子时 universe 必填"
            "（组合路径 fail-loudly，不同于库池的跳过+告警）。"
        )

    # 可读名：r.name 优先；expression 缺省 mined_sha；python 剥哨兵。同名防御追加 _2
    used_names: set[str] = set()

    def _resolve_name(r) -> str:
        if _is_python_record(r):
            base = r.name or _python_name_from_expression(r.expression) or "python_factor"
        else:
            base = r.name or default_name_for_expression(_normalize(r.expression))
        if base not in used_names:
            used_names.add(base)
            return base
        i = 2
        while f"{base}_{i}" in used_names:
            i += 1
        n = f"{base}_{i}"
        used_names.add(n)
        return n

    named_recs: list[tuple[str, Any]] = [(_resolve_name(r), r) for r in recs]
    factors_status = {name: r.status for name, r in named_recs}

    # 日内自动装帧：与 from-session 同款，但跳过 py:: 哨兵（词法 ix_ 假阳性；
    # 一期 CLI lift-test 同款处理）
    _exprs = [
        str(r.expression or "")
        for _n, r in named_recs
        if not is_python_identity(str(r.expression or ""))
    ]
    need_intraday = expressions_need_intraday(_exprs)
    ix_leaves = intraday_expr_leaf_names(_exprs)
    # prepare + preprocess 不能省：前向收益 / python 因子 / store miss 重算都依赖
    daily = prepare_mining_daily(
        start, end, universe,
        intraday=need_intraday or bool(ix_leaves),
        intraday_expr_leaves=ix_leaves or None,
    )
    prepped = _preprocess_daily(daily)  # 预处理一次，逐因子复用
    start_date = datetime.strptime(start, "%Y%m%d").date()
    pool_start, pool_end = _pool_date_bounds(prepped)

    # 物化：expression 优先 store parquet → miss 则 _factor_df_from_prepped；
    # python 仍走 _materialize_python_on_grid
    materialized: list[tuple[str, pl.DataFrame]] = []
    skipped_materialize: list[str] = []
    store_hits: dict[str, dict[str, Any]] = {}
    store_misses: dict[str, str] = {}
    n_store_candidates = 0
    for name, r in named_recs:
        identity = str(r.expression or name)
        try:
            if _is_python_record(r):
                panel = _materialize_python_on_grid(
                    r, prepped,
                    market=market,
                    universe=universe,
                    python_materializer=None,
                    start=pool_start,
                    end=pool_end,
                )
                if panel is None:
                    skipped_materialize.append(identity)
                    continue
                fdf = (
                    panel.filter(pl.col("trade_date") >= start_date)
                    .select(["trade_date", "ts_code", "factor_value"])
                )
                if fdf.is_empty():
                    skipped_materialize.append(identity)
                    continue
            else:
                used_store = False
                if not no_store and universe is not None:
                    n_store_candidates += 1
                    fdf_store, miss_reason, hit_meta = load_materialized_factor(
                        r,
                        market=market,
                        root=s_root,
                        start=start,
                        end=end,
                        universe=universe,
                    )
                    if fdf_store is not None:
                        # 与重算路径一致：再滤 trade_date >= start_date
                        fdf = (
                            fdf_store.filter(pl.col("trade_date") >= start_date)
                            .select(["trade_date", "ts_code", "factor_value"])
                        )
                        if fdf.is_empty():
                            store_misses[name] = "empty_after_start_filter"
                        else:
                            used_store = True
                            store_hits[name] = hit_meta or {}
                    else:
                        store_misses[name] = miss_reason or "unknown"
                elif not no_store and universe is None:
                    n_store_candidates += 1
                    store_misses[name] = "no_universe"

                if not used_store:
                    fdf = _factor_df_from_prepped(
                        parse_expr(r.expression), prepped, eval_start=start_date,
                    )
                    fdf = fdf.select(["trade_date", "ts_code", "factor_value"])
        except Exception:
            skipped_materialize.append(identity)
            continue
        materialized.append((name, fdf))

    # store 命中汇总日志
    n_hits = len(store_hits)
    miss_dist = dict(Counter(store_misses.values()))
    _log.info(
        "store 命中 %s/%s(miss 原因分布: %s)",
        n_hits,
        n_store_candidates,
        miss_dist if miss_dist else "{}",
    )
    store_usage: dict[str, Any] = {
        "hits": store_hits,
        "misses": store_misses,
        "no_store": bool(no_store),
    }

    if len(materialized) < 2:
        raise ValueError(
            f"可物化的库因子不足 2 个（得 {len(materialized)}，"
            f"跳过 {len(skipped_materialize)}）。"
        )

    kept, dropped = _greedy_decorrelate(materialized, decorr_threshold)
    if len(kept) < 2:
        raise ValueError(
            f"去相关后库因子不足 2 个（得 {len(kept)}，剔除 {len(dropped)} 个高相关近亲）；"
            f"放宽 decorr_threshold 或多挖正交因子。"
        )

    # 落盘：文件名 = 可读 name（组合报告里取代 factor_{i}）
    work = Path(tempfile.mkdtemp(prefix="combine_lib_"))
    factor_files: list[str] = []
    for name, fdf in kept:
        # 文件名安全：去掉路径分隔符
        safe = name.replace("/", "_").replace("\\", "_")
        p = work / f"{safe}.parquet"
        fdf.write_parquet(p)
        factor_files.append(str(p))

    price_col = "close_adj" if "close_adj" in daily.columns else "close"
    fwd = compute_fwd_returns(daily.sort(["ts_code", "trade_date"]), horizons=[horizon],
                              price_col=price_col)
    ret = (
        fwd.filter(pl.col("trade_date") >= start_date)
        .select(["trade_date", "ts_code", pl.col(f"fwd_ret_{horizon}d").alias("ret")])
        .filter(pl.col("ret").is_not_null())
    )
    ret_file = work / "ret.parquet"
    ret.write_parquet(ret_file)

    res = run_factor_combination(
        factor_files=factor_files, ret_file=str(ret_file),
        train_days=train_days, test_days=test_days, purge_days=purge_days,
        embargo_days=embargo_days, methods=methods, seed=seed,
        out_dir=out_dir, run_id=run_id,
        # command 只有子命令名不足以复现——窗口/票池/选品口径全记进 config
        command=["combine", "from-library"],
        extra_config={
            "market": market,
            "statuses": list(statuses),
            "start": start,
            "end": end,
            "universe": universe,
            "horizon": horizon,
            "top_n": top_n,
            "decorr_threshold": decorr_threshold,
            "library_root": str(root),
            "library_hash": library_file_hash(market, root),
            "store_usage": store_usage,
        },
    )
    res["factors_used"] = [n for n, _ in kept]
    res["factors_status"] = factors_status
    res["dropped_correlated"] = dropped
    res["skipped_materialize"] = skipped_materialize
    res["library_hash"] = library_file_hash(market, root)
    res["market"] = market
    res["statuses"] = list(statuses)
    res["truncated_from"] = truncated_from
    res["store_usage"] = store_usage
    return res

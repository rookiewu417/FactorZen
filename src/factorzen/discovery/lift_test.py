"""组合增量 lift 实验：灰区/lift 队列候选加进库内 active 集，测 OOS RankIC 增量。

单因子库门语义不变；本模块是**后置第二通道**（试用/probation/active 入库裁决）。
挖掘内不跑 lift（保持挖掘快）；由 CLI ``fz factor-library lift-test`` 或
team session 末钩子批处理。

口径：
- 每日 OOS RankIC 与 ``combination.experiment._evaluate_oos`` 的逐日 spearman 一致；
- lift = 配对日 (cand_ic − base_ic) 均值；SE 用 block 均值的样本标准差 / √n_blocks
  （对冲 5 日前向收益重叠导致的日间自相关）。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl

from factorzen.core.stats import spearman_avg_rank
from factorzen.discovery.guardrails import (
    DEFAULT_LIFT_THRESHOLD,
    REJECT_CATEGORY_GRAY_ZONE,
)

_LOG = logging.getLogger(__name__)

DEFAULT_TOP_M = 10
DEFAULT_HORIZON = 5
DEFAULT_BLOCK_DAYS = 20

# TODO: 后续由 guardrails.REJECT_CATEGORY_LIFT_QUEUE 收口；旧 manifest 仍用 gray_zone。
LIFT_QUEUE_CATEGORY = "lift_queue"

_EXTRACT_CATEGORIES = frozenset({REJECT_CATEGORY_GRAY_ZONE, LIFT_QUEUE_CATEGORY})


def _rank_ic_key(c: dict) -> float:
    """按 |residual_ic_train| 优先、否则 |ic_train| 降序键。"""
    ric = c.get("residual_ic_train")
    if ric is not None and ric == ric:
        return abs(float(ric))
    ic = c.get("ic_train")
    if ic is not None and ic == ic:
        return abs(float(ic))
    return 0.0


def _daily_oos_rank_ic(
    combined: pl.DataFrame, ret_df: pl.DataFrame, n_groups: int = 5
) -> pl.DataFrame:
    """逐日 OOS RankIC 序列，口径对齐 ``_evaluate_oos`` 的 rank_ic_mean 分量。

    返回 ``[trade_date, ic]``（已按 trade_date 排序；无有效日时为空表）。
    分组守卫与 spearman 与 experiment._evaluate_oos 一致（core.stats average-rank）：
    - 日截面 len < n_groups*2 → 跳过
    - spearman_avg_rank 返回 None（n<2 / 常数列 / 非有限）→ 跳过
    """
    rdf = ret_df.with_columns(pl.col("trade_date").cast(pl.Utf8))
    m = combined.join(rdf, on=["trade_date", "ts_code"], how="inner")
    day_rows: list[tuple[str, float]] = []
    for _d, g in m.group_by("trade_date", maintain_order=True):
        if len(g) < n_groups * 2:
            continue
        f = g["factor_value"].to_numpy().astype(float)
        r = g["ret"].to_numpy().astype(float)
        ic = spearman_avg_rank(f, r)
        if ic is not None:
            day_rows.append((str(_d[0]), ic))
    if not day_rows:
        return pl.DataFrame(
            schema={"trade_date": pl.Utf8, "ic": pl.Float64},
        )
    day_rows.sort(key=lambda x: x[0])
    return pl.DataFrame(
        {"trade_date": [d for d, _ in day_rows], "ic": [ic for _, ic in day_rows]},
        schema={"trade_date": pl.Utf8, "ic": pl.Float64},
    )


def _mean_ic(daily: pl.DataFrame) -> float:
    """每日 IC 序列均值；空表/非数值（mypy: Series.mean 返回宽 union）→ 0.0。"""
    if daily.is_empty():
        return 0.0
    v = daily["ic"].mean()
    return float(v) if isinstance(v, (int, float)) else 0.0


def _oos_rank_ic(combined: pl.DataFrame, ret_df: pl.DataFrame) -> float:
    """每日序列均值；与 ``_evaluate_oos(... )['rank_ic_mean']`` 对齐。"""
    return _mean_ic(_daily_oos_rank_ic(combined, ret_df))


def paired_lift_stats(
    cand_daily: pl.DataFrame,
    base_daily: pl.DataFrame,
    block_days: int = DEFAULT_BLOCK_DAYS,
) -> dict[str, Any]:
    """配对日 lift + 块 SE + 半段稳定性。

    - 两序列按 trade_date inner join；diff = cand_ic − base_ic
    - 按时间序切连续 block（每块 ``block_days`` 交易日，尾块不足也算一块）
    - ``lift_se`` = std(块均值, ddof=1) / √n_blocks；n_blocks < 2 → None
    - 半段按**块数**二等分（奇数块中位块归前半）
    """
    empty: dict[str, Any] = {
        "lift": None,
        "lift_se": None,
        "n_blocks": 0,
        "n_days": 0,
        "lift_first_half": None,
        "lift_second_half": None,
    }
    if cand_daily is None or base_daily is None:
        return empty
    if cand_daily.is_empty() or base_daily.is_empty():
        return empty

    c = cand_daily.select(
        pl.col("trade_date").cast(pl.Utf8),
        pl.col("ic").alias("cand_ic"),
    )
    b = base_daily.select(
        pl.col("trade_date").cast(pl.Utf8),
        pl.col("ic").alias("base_ic"),
    )
    joined = c.join(b, on="trade_date", how="inner").sort("trade_date")
    if joined.is_empty():
        return empty

    diffs = (joined["cand_ic"] - joined["base_ic"]).to_numpy().astype(float)
    n_days = len(diffs)
    lift = float(np.mean(diffs))

    bd = max(1, int(block_days))
    block_means: list[float] = []
    block_slices: list[np.ndarray] = []
    for i in range(0, n_days, bd):
        chunk = diffs[i : i + bd]
        block_means.append(float(np.mean(chunk)))
        block_slices.append(chunk)
    n_blocks = len(block_means)

    if n_blocks < 2:
        lift_se: float | None = None
    else:
        lift_se = float(np.std(block_means, ddof=1) / np.sqrt(n_blocks))

    # 奇数块中位归前半：n=5 → mid=3；n=4 → mid=2；n=1 → mid=1
    mid = (n_blocks + 1) // 2
    first_chunks = block_slices[:mid]
    second_chunks = block_slices[mid:]
    if first_chunks:
        lift_first_half: float | None = float(
            np.mean(np.concatenate(first_chunks))
        )
    else:
        lift_first_half = None
    if second_chunks:
        lift_second_half: float | None = float(
            np.mean(np.concatenate(second_chunks))
        )
    else:
        lift_second_half = None

    return {
        "lift": lift,
        "lift_se": lift_se,
        "n_blocks": n_blocks,
        "n_days": n_days,
        "lift_first_half": lift_first_half,
        "lift_second_half": lift_second_half,
    }


def lift_admission(
    row: dict,
    *,
    threshold: float = DEFAULT_LIFT_THRESHOLD,
    se_mult: float = 1.0,
) -> str:
    """统一准入规则：返回 ``\"active\" | \"probation\" | \"reject\"``。

    - lift is None / 非有限 → reject
    - lift_se is None / 转换失败 / 非有限（NaN、±inf）→ reject
      （区间证据不完整，不再按 0 处理）
    - lift ≥ max(threshold, se_mult × lift_se) 且 lift_second_half > 0 → active
    - lift ≥ 同上门槛但 second_half 为 None 或 ≤ 0 → probation
    - 否则 reject

    finite lift_se（含 0.0）合法：bar = max(threshold, se_mult × se)。
    orchestrator / rebuild / CLI 三处共用此单一实现。
    """
    lift = row.get("lift")
    if lift is None:
        return "reject"
    try:
        lift_f = float(lift)
    except (TypeError, ValueError):
        return "reject"
    if not np.isfinite(lift_f):  # NaN/±inf 与 docstring 契约一致
        return "reject"

    se_raw = row.get("lift_se")
    if se_raw is None:
        return "reject"
    try:
        se_val = float(se_raw)
    except (TypeError, ValueError):
        return "reject"
    # SE 非有限 = 区间证据不完整 → reject（不按 0 退化）
    if not np.isfinite(se_val):
        return "reject"
    bar = max(float(threshold), float(se_mult) * se_val)
    if lift_f < bar:
        return "reject"

    sh = row.get("lift_second_half")
    if sh is not None:
        try:
            sh_f = float(sh)
        except (TypeError, ValueError):
            return "probation"
        if sh_f == sh_f and sh_f > 0:
            return "active"
    return "probation"


def _with_safe_feature_names(factor_dfs: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """键映射为安全特征名 f{i}（按插入序，确定性）。

    lgbm 不接受特征名含特殊 JSON 字符（括号/逗号等）——因子键是**真实表达式**时
    直接进 combine 会炸 `LightGBMError: Do not support special JSON characters in
    feature name`（线上事故）。仅影响 lgbm 内部特征名；lift 报告仍用真实表达式
    （调用方持有原键，映射不外泄）。"""
    return {f"f{i:03d}": df for i, df in enumerate(factor_dfs.values())}


def _empty_lift_fields() -> dict[str, Any]:
    return {
        "lift_se": None,
        "n_blocks": None,
        "lift_first_half": None,
        "lift_second_half": None,
    }


def run_lift_tests(
    gray_candidates: list[dict],
    *,
    market: str,
    daily: pl.DataFrame,
    leaf_map: dict[str, str] | None = None,
    library_root: str = "workspace/factor_library",
    cv_params: dict[str, Any] | None = None,
    top_m: int | None = None,
    threshold: float = DEFAULT_LIFT_THRESHOLD,
    seed: int = 0,
    active_factor_dfs: dict[str, pl.DataFrame] | None = None,
    ret_df: pl.DataFrame | None = None,
    materialize_candidate=None,
    combine_fn=None,
    horizon: int = DEFAULT_HORIZON,
    block_days: int = DEFAULT_BLOCK_DAYS,
) -> list[dict]:
    """对灰区/lift 队列候选跑 lgbm 组合 OOS lift 实验。

    - gray 按 |residual_ic_train|（缺则 |ic_train|）降序；``top_m=None`` 全测，
      否则截断（``truncated_from`` / n_selected 语义不变——**no silent caps**）。
    - 基线**只算一次**（含每日 IC 序列复用）：库内 active 集合 → combine → 每日 RankIC。
    - 每候选：active+candidate → 同 CV 同 seed → 配对日 lift + 块 SE + 半段。
    - ``lift ≥ threshold`` → passed（最终 active/probation/reject 见 ``lift_admission``）。
    - 逐候选 try/except：一个坏候选不崩整批。

    测试可注入 ``active_factor_dfs`` / ``ret_df`` / ``combine_fn`` / ``materialize_candidate``
    做离线 mock；生产路径走 ``build_library_pool`` + 表达式物化 + horizon 前向收益。
    """
    from factorzen.research.combination.cv import PurgedWalkForwardCV
    from factorzen.research.combination.models import combine_lgbm

    n_input = len(gray_candidates)
    ordered = sorted(gray_candidates, key=_rank_ic_key, reverse=True)
    if top_m is None:
        selected = list(ordered)
    else:
        selected = ordered[: max(0, int(top_m))]
    truncated = n_input - len(selected)

    cv_kw: dict[str, Any] = {
        "train_days": 120,
        "test_days": 20,
        "purge_days": 5,
        "embargo_days": 0,
        "expanding": True,
    }
    if cv_params:
        cv_kw.update(cv_params)
    cv = PurgedWalkForwardCV(**cv_kw)
    _combine = combine_fn or (
        lambda fds, rdf, c, **kw: combine_lgbm(fds, rdf, c, seed=seed, **kw)
    )

    meta = {
        "truncated_from": n_input if truncated else None,
        "n_selected": len(selected),
        "n_input": n_input,
        "threshold": threshold,
    }

    # ── 物化：active 面板 + 收益 ────────────────────────────────────────────
    if active_factor_dfs is None:
        from factorzen.discovery.factor_library import build_library_pool

        active_factor_dfs = build_library_pool(
            market, daily, leaf_map, root=library_root, statuses=("active",),
        )
    if not active_factor_dfs:
        _LOG.warning("lift_test: 库内无 active 因子，无法跑基线；全部判不过")
        return [
            {
                "expression": c.get("expression"),
                "lift": None,
                "baseline": None,
                "candidate_rank_ic": None,
                "passed": False,
                "error": "empty_active_library",
                **_empty_lift_fields(),
                **meta,
            }
            for c in selected
        ]

    if ret_df is None:
        ret_df = _build_ret_panel(daily, horizon=horizon)

    if materialize_candidate is None:
        materialize_candidate = _default_materializer(daily, leaf_map)

    # ── 基线一次（每日序列复用） ────────────────────────────────────────────
    try:
        base_combined = _combine(_with_safe_feature_names(active_factor_dfs), ret_df, cv)
        base_daily = _daily_oos_rank_ic(base_combined, ret_df)
        baseline = _mean_ic(base_daily)
    except Exception as exc:
        _LOG.warning("lift_test baseline 失败: %s: %s", type(exc).__name__, exc)
        return [
            {
                "expression": c.get("expression"),
                "lift": None,
                "baseline": None,
                "candidate_rank_ic": None,
                "passed": False,
                "error": f"baseline_failed:{type(exc).__name__}",
                **_empty_lift_fields(),
                **{k: v for k, v in meta.items() if k != "threshold"},
                "threshold": threshold,
            }
            for c in selected
        ]

    results: list[dict] = []
    for c in selected:
        expr = c.get("expression")
        row: dict[str, Any] = {
            "expression": expr,
            "lift": None,
            "baseline": baseline,
            "candidate_rank_ic": None,
            "passed": False,
            "error": None,
            **_empty_lift_fields(),
            **meta,
        }
        try:
            cand_df = materialize_candidate(expr) if expr else None
            if cand_df is None or (hasattr(cand_df, "is_empty") and cand_df.is_empty()):
                row["error"] = "materialize_failed"
                results.append(row)
                continue
            # 列规范：[trade_date, ts_code, factor_value]
            if "factor_value" not in cand_df.columns:
                row["error"] = "bad_panel_schema"
                results.append(row)
                continue
            pool = dict(active_factor_dfs)
            # 键用规范表达式串；若与 active 撞名则覆盖为候选自身（仍测「加它」）。
            # 进 combine 前统一映射安全特征名（候选按插入序恒为最后一个 f{n}）。
            key = str(expr)
            pool[key] = cand_df.select(["trade_date", "ts_code", "factor_value"])
            cand_combined = _combine(_with_safe_feature_names(pool), ret_df, cv)
            cand_daily = _daily_oos_rank_ic(cand_combined, ret_df)
            cand_ic = _mean_ic(cand_daily)
            stats = paired_lift_stats(cand_daily, base_daily, block_days=block_days)
            row["candidate_rank_ic"] = cand_ic
            row["lift"] = stats["lift"]
            row["lift_se"] = stats["lift_se"]
            row["n_blocks"] = stats["n_blocks"]
            row["lift_first_half"] = stats["lift_first_half"]
            row["lift_second_half"] = stats["lift_second_half"]
            lift_v = stats["lift"]
            row["passed"] = bool(
                lift_v is not None and float(lift_v) >= threshold
            )
        except Exception as exc:
            _LOG.warning(
                "lift_test candidate %r 失败: %s: %s",
                expr, type(exc).__name__, exc,
            )
            row["error"] = f"{type(exc).__name__}:{exc}"
        results.append(row)
    return results


def run_group_lift(
    queue: list[dict],
    *,
    market: str,
    daily: pl.DataFrame,
    leaf_map: dict[str, str] | None = None,
    library_root: str = "workspace/factor_library",
    cv_params: dict[str, Any] | None = None,
    top_m: int | None = None,  # 与 run_lift_tests 签名对齐；组测忽略截断
    threshold: float = DEFAULT_LIFT_THRESHOLD,
    seed: int = 0,
    active_factor_dfs: dict[str, pl.DataFrame] | None = None,
    ret_df: pl.DataFrame | None = None,
    materialize_candidate=None,
    combine_fn=None,
    horizon: int = DEFAULT_HORIZON,
    block_days: int = DEFAULT_BLOCK_DAYS,
) -> dict[str, Any]:
    """整组候选一次 combine vs 基线的配对 lift 统计。

    全部候选一起加进 active 池 combine 一次；逐候选物化失败的跳过并记
    ``skipped``；全部物化失败返回 error 行。同样走 ``_with_safe_feature_names``。
    """
    del top_m  # 组测全量；保留形参兼容注入签名
    from factorzen.research.combination.cv import PurgedWalkForwardCV
    from factorzen.research.combination.models import combine_lgbm

    cv_kw: dict[str, Any] = {
        "train_days": 120,
        "test_days": 20,
        "purge_days": 5,
        "embargo_days": 0,
        "expanding": True,
    }
    if cv_params:
        cv_kw.update(cv_params)
    cv = PurgedWalkForwardCV(**cv_kw)
    _combine = combine_fn or (
        lambda fds, rdf, c, **kw: combine_lgbm(fds, rdf, c, seed=seed, **kw)
    )

    def _err(msg: str, *, skipped: list | None = None, expressions: list | None = None):
        return {
            "lift": None,
            "lift_se": None,
            "n_blocks": 0,
            "n_days": 0,
            "lift_first_half": None,
            "lift_second_half": None,
            "n_candidates": 0,
            "expressions": list(expressions or []),
            "skipped": list(skipped or []),
            "baseline": None,
            "threshold": threshold,
            "error": msg,
        }

    if active_factor_dfs is None:
        from factorzen.discovery.factor_library import build_library_pool

        active_factor_dfs = build_library_pool(
            market, daily, leaf_map, root=library_root, statuses=("active",),
        )
    if not active_factor_dfs:
        return _err("empty_active_library")

    if ret_df is None:
        ret_df = _build_ret_panel(daily, horizon=horizon)

    if materialize_candidate is None:
        materialize_candidate = _default_materializer(daily, leaf_map)

    skipped: list[dict[str, Any]] = []
    expressions: list[str] = []
    pool = dict(active_factor_dfs)
    for c in queue:
        expr = c.get("expression")
        if not expr:
            skipped.append({"expression": expr, "error": "missing_expression"})
            continue
        try:
            cand_df = materialize_candidate(expr)
            if cand_df is None or (hasattr(cand_df, "is_empty") and cand_df.is_empty()):
                skipped.append({"expression": expr, "error": "materialize_failed"})
                continue
            if "factor_value" not in cand_df.columns:
                skipped.append({"expression": expr, "error": "bad_panel_schema"})
                continue
            pool[str(expr)] = cand_df.select(["trade_date", "ts_code", "factor_value"])
            expressions.append(str(expr))
        except Exception as exc:
            skipped.append({
                "expression": expr,
                "error": f"{type(exc).__name__}:{exc}",
            })

    if not expressions:
        return _err("all_candidates_materialize_failed", skipped=skipped)

    try:
        base_combined = _combine(_with_safe_feature_names(active_factor_dfs), ret_df, cv)
        base_daily = _daily_oos_rank_ic(base_combined, ret_df)
        baseline = _mean_ic(base_daily)
        group_combined = _combine(_with_safe_feature_names(pool), ret_df, cv)
        group_daily = _daily_oos_rank_ic(group_combined, ret_df)
        stats = paired_lift_stats(group_daily, base_daily, block_days=block_days)
    except Exception as exc:
        _LOG.warning("run_group_lift 失败: %s: %s", type(exc).__name__, exc)
        return _err(
            f"group_combine_failed:{type(exc).__name__}",
            skipped=skipped,
            expressions=expressions,
        )

    return {
        **stats,
        "n_candidates": len(expressions),
        "expressions": expressions,
        "skipped": skipped,
        "baseline": baseline,
        "threshold": threshold,
        "error": None,
    }


def _build_ret_panel(daily: pl.DataFrame, *, horizon: int = DEFAULT_HORIZON) -> pl.DataFrame:
    """horizon 日前向收益面板（复刻 factor_combine 口径）。"""
    from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns

    price_col = "close_adj" if "close_adj" in daily.columns else "close"
    fwd = compute_fwd_returns(
        daily.sort(["ts_code", "trade_date"]), horizons=[horizon], price_col=price_col,
    )
    return (
        fwd.select(["trade_date", "ts_code", pl.col(f"fwd_ret_{horizon}d").alias("ret")])
        .filter(pl.col("ret").is_not_null())
        .with_columns(pl.col("trade_date").cast(pl.Utf8))
    )


def _default_materializer(daily: pl.DataFrame, leaf_map: dict[str, str] | None):
    """表达式 → 因子面板（与 build_library_pool / _factor_df_from_prepped 同路径）。"""
    from factorzen.agents.evaluation import _factor_df_from_prepped, _preprocess_daily
    from factorzen.discovery.expression import parse_expr

    prepped = _preprocess_daily(daily).sort(["ts_code", "trade_date"])

    def _mat(expr: str) -> pl.DataFrame | None:
        try:
            node = parse_expr(expr, leaf_map)
            return _factor_df_from_prepped(node, prepped, leaf_map=leaf_map).select(
                ["trade_date", "ts_code", "factor_value"]
            )
        except Exception as exc:
            _LOG.debug("lift materialize %r: %s: %s", expr, type(exc).__name__, exc)
            return None

    return _mat


def extract_gray_candidates_from_manifest(manifest: dict) -> list[dict]:
    """从 mine_team / mine-agent / mining_session manifest 抽灰区/lift 队列候选。

    接受 ``reject_category`` 为 ``gray_zone``（旧）或 ``lift_queue``（新）。
    - team/agent：``attempts`` 里匹配 category
    - M1 session：``candidates`` 里同字段
    返回带 expression + 指标的 dict 列表（供 run_lift_tests / run_group_lift）。
    """
    out: list[dict] = []
    seen: set[str] = set()

    def _add(row: dict):
        expr = row.get("expression")
        if not expr or expr in seen:
            return
        if row.get("reject_category") not in _EXTRACT_CATEGORIES:
            return
        seen.add(expr)
        out.append(dict(row))

    for a in manifest.get("attempts") or []:
        if isinstance(a, dict):
            _add(a)
    for c in manifest.get("candidates") or []:
        if isinstance(c, dict):
            _add(c)
    return out


# 新语义别名：同一实现
extract_lift_queue_from_manifest = extract_gray_candidates_from_manifest


__all__ = [
    "DEFAULT_BLOCK_DAYS",
    "DEFAULT_HORIZON",
    "DEFAULT_TOP_M",
    "LIFT_QUEUE_CATEGORY",
    "extract_gray_candidates_from_manifest",
    "extract_lift_queue_from_manifest",
    "lift_admission",
    "paired_lift_stats",
    "run_group_lift",
    "run_lift_tests",
]

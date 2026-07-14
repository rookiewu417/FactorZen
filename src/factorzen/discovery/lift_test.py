"""组合增量 lift 实验：灰区候选加进库内 active 集，测 OOS RankIC 增量。

单因子库门语义不变；本模块是**后置第二通道**（试用/probation 入库裁决）。
挖掘内不跑 lift（保持挖掘快）；由 CLI ``fz factor-library lift-test`` 批处理。
"""
from __future__ import annotations

import logging
from typing import Any

import polars as pl

from factorzen.discovery.guardrails import DEFAULT_LIFT_THRESHOLD

_LOG = logging.getLogger(__name__)

DEFAULT_TOP_M = 10
DEFAULT_HORIZON = 5


def _rank_ic_key(c: dict) -> float:
    """按 |residual_ic_train| 优先、否则 |ic_train| 降序键。"""
    ric = c.get("residual_ic_train")
    if ric is not None and ric == ric:
        return abs(float(ric))
    ic = c.get("ic_train")
    if ic is not None and ic == ic:
        return abs(float(ic))
    return 0.0


def _oos_rank_ic(combined: pl.DataFrame, ret_df: pl.DataFrame) -> float:
    """复用 combination.experiment._evaluate_oos 的 RankIC 均值口径。"""
    from factorzen.research.combination.experiment import _evaluate_oos

    return float(_evaluate_oos(combined, ret_df).get("rank_ic_mean", 0.0))


def run_lift_tests(
    gray_candidates: list[dict],
    *,
    market: str,
    daily: pl.DataFrame,
    leaf_map: dict[str, str] | None = None,
    library_root: str = "workspace/factor_library",
    cv_params: dict[str, Any] | None = None,
    top_m: int = DEFAULT_TOP_M,
    threshold: float = DEFAULT_LIFT_THRESHOLD,
    seed: int = 0,
    active_factor_dfs: dict[str, pl.DataFrame] | None = None,
    ret_df: pl.DataFrame | None = None,
    materialize_candidate=None,
    combine_fn=None,
    horizon: int = DEFAULT_HORIZON,
) -> list[dict]:
    """对灰区候选跑 lgbm 组合 OOS lift 实验。

    - gray 按 |residual_ic_train|（缺则 |ic_train|）降序取 top_m（控成本；截断写进
      返回行的 ``truncated_from`` / 由调用方写 manifest——**no silent caps**）。
    - 基线**只算一次**：库内 active 集合 → combine_lgbm → OOS RankIC。
    - 每候选：active+candidate → 同 CV 同 seed → lift = cand − baseline。
    - ``lift ≥ threshold`` → passed。
    - 逐候选 try/except：一个坏候选不崩整批。

    测试可注入 ``active_factor_dfs`` / ``ret_df`` / ``combine_fn`` / ``materialize_candidate``
    做离线 mock；生产路径走 ``build_library_pool`` + 表达式物化 + horizon 前向收益。
    """
    from factorzen.research.combination.cv import PurgedWalkForwardCV
    from factorzen.research.combination.models import combine_lgbm

    n_input = len(gray_candidates)
    ordered = sorted(gray_candidates, key=_rank_ic_key, reverse=True)
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
                "truncated_from": n_input if truncated else None,
                "n_selected": len(selected),
                "n_input": n_input,
            }
            for c in selected
        ]

    if ret_df is None:
        ret_df = _build_ret_panel(daily, horizon=horizon)

    if materialize_candidate is None:
        materialize_candidate = _default_materializer(daily, leaf_map)

    # ── 基线一次 ────────────────────────────────────────────────────────────
    try:
        base_combined = _combine(dict(active_factor_dfs), ret_df, cv)
        baseline = _oos_rank_ic(base_combined, ret_df)
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
                "truncated_from": n_input if truncated else None,
                "n_selected": len(selected),
                "n_input": n_input,
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
            "truncated_from": n_input if truncated else None,
            "n_selected": len(selected),
            "n_input": n_input,
            "threshold": threshold,
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
            # 键用规范表达式串；若与 active 撞名则覆盖为候选自身（仍测「加它」）
            key = str(expr)
            pool[key] = cand_df.select(["trade_date", "ts_code", "factor_value"])
            cand_combined = _combine(pool, ret_df, cv)
            cand_ic = _oos_rank_ic(cand_combined, ret_df)
            lift = float(cand_ic) - float(baseline)
            row["candidate_rank_ic"] = cand_ic
            row["lift"] = lift
            row["passed"] = bool(lift >= threshold)
        except Exception as exc:
            _LOG.warning(
                "lift_test candidate %r 失败: %s: %s",
                expr, type(exc).__name__, exc,
            )
            row["error"] = f"{type(exc).__name__}:{exc}"
        results.append(row)
    return results


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
    """从 mine_team / mine-agent / mining_session manifest 抽 gray_zone 候选。

    - team/agent：``attempts`` 里 ``reject_category==gray_zone``
    - M1 session：``candidates`` 里同字段（或 attempt 形态）
    返回带 expression + 指标的 dict 列表（供 run_lift_tests）。
    """
    from factorzen.discovery.guardrails import REJECT_CATEGORY_GRAY_ZONE

    out: list[dict] = []
    seen: set[str] = set()

    def _add(row: dict):
        expr = row.get("expression")
        if not expr or expr in seen:
            return
        if row.get("reject_category") != REJECT_CATEGORY_GRAY_ZONE:
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


__all__ = [
    "DEFAULT_HORIZON",
    "DEFAULT_TOP_M",
    "extract_gray_candidates_from_manifest",
    "run_lift_tests",
]

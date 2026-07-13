# src/factorzen/discovery/leaf_health.py
"""叶子 holdout 覆盖健康检查：开局摘除 holdout 窗口实质死亡的叶子。

一日「有效截面」= 该日该列非空且有限值的股票数 ≥ ``min_cross``
（默认对齐 Rank IC 的 ``_MIN_CROSS_SAMPLES=30``）。覆盖率 = 有效截面日 / holdout 交易日总数。
NaN 与 null 均不计（先 ``fill_nan(None)``）。
"""
from __future__ import annotations

from typing import Any

import polars as pl

from factorzen.daily.evaluation.ic_analysis import _MIN_CROSS_SAMPLES
from factorzen.discovery.operators import LEAF_FEATURES

DEFAULT_LEAF_HOLDOUT_MIN_COVERAGE = 0.5


def leaf_holdout_coverage(
    daily: pl.DataFrame,
    leaves: list[str],
    holdout_start: Any,
    *,
    leaf_map: dict[str, str] | None = None,
    min_cross: int = _MIN_CROSS_SAMPLES,
) -> dict[str, float]:
    """每个 leaf 在 holdout 段的有效截面日覆盖率 ∈ [0, 1]。

    ``holdout_start``：与 ``trade_date`` 可比较的边界（date / datetime / 字面量）。
    列缺失 → 覆盖率 0.0。holdout 无交易日 → 全部 0.0。
    """
    lm = LEAF_FEATURES if leaf_map is None else leaf_map
    if daily.is_empty() or "trade_date" not in daily.columns:
        return {leaf: 0.0 for leaf in leaves}

    hold = daily.filter(pl.col("trade_date") >= holdout_start)
    n_hold_dates = hold["trade_date"].n_unique() if hold.height else 0
    if n_hold_dates == 0:
        return {leaf: 0.0 for leaf in leaves}

    out: dict[str, float] = {}
    for leaf in leaves:
        col = lm.get(leaf, leaf)
        # 映射列缺失时回退叶子名本身（测试帧常只有 close 无 close_adj；生产预处理会补别名）
        if col not in hold.columns:
            col = leaf if leaf in hold.columns else ""
        if not col or col not in hold.columns:
            out[leaf] = 0.0
            continue
        # NaN ≠ null：先 fill_nan 再判非空，与 IC 路径 is_finite 口径一致
        series = pl.col(col).fill_nan(None)
        per = (
            hold.group_by("trade_date")
            .agg(series.is_not_null().sum().alias("_n"))
        )
        n_ok = int(per.filter(pl.col("_n") >= min_cross).height)
        out[leaf] = n_ok / n_hold_dates
    return out


def filter_leaves_by_holdout_coverage(
    daily: pl.DataFrame,
    leaves: list[str],
    holdout_start: Any,
    *,
    leaf_map: dict[str, str] | None = None,
    min_coverage: float = DEFAULT_LEAF_HOLDOUT_MIN_COVERAGE,
    min_cross: int = _MIN_CROSS_SAMPLES,
) -> tuple[list[str], dict[str, float]]:
    """按 holdout 覆盖率过滤叶子。

    Returns:
        (kept_leaves, excluded: {leaf: coverage})
        覆盖率 < min_coverage 进 excluded；顺序与输入 leaves 一致。

    **Fail-open**：若全部叶子都低于阈值，说明是帧本身撑不起检查前提
    （小 universe 截面 < min_cross、holdout 无交易日等），而非个别叶子死亡——
    此时不摘任何叶子（摘光会让挖掘空转），交由下游逐候选的 holdout 覆盖门兜底。
    """
    cov = leaf_holdout_coverage(
        daily, leaves, holdout_start, leaf_map=leaf_map, min_cross=min_cross,
    )
    kept: list[str] = []
    excluded: dict[str, float] = {}
    for leaf in leaves:
        c = cov.get(leaf, 0.0)
        if c < min_coverage:
            excluded[leaf] = c
        else:
            kept.append(leaf)
    if leaves and not kept:
        print(f"[leaf-health] 全部 {len(leaves)} 个叶子 holdout 覆盖不足——"
              "视为帧不支持该检查(小截面/空 holdout)，fail-open 不摘叶", flush=True)
        return list(leaves), {}
    return kept, excluded


def apply_leaf_exclusion(
    leaves: list[str],
    leaf_map: dict[str, str] | None,
    excluded: dict[str, float],
) -> tuple[list[str], dict[str, str] | None]:
    """从叶子清单与映射中剔除 excluded；返回 (kept_names, filtered_map)。

    ``leaf_map is None``（A 股默认）时若有剔除，物化为 ``LEAF_FEATURES`` 子集，
    使 parse_expr 也不再接受死叶（与 prompt 一致）。
    """
    ban = set(excluded)
    kept = [L for L in leaves if L not in ban]
    if not ban:
        return kept, leaf_map
    base = LEAF_FEATURES if leaf_map is None else leaf_map
    return kept, {k: v for k, v in base.items() if k not in ban}


def log_excluded_leaves(excluded: dict[str, float], *, prefix: str = "leaf-health") -> None:
    if not excluded:
        return
    parts = [f"{k}={v:.2%}" for k, v in sorted(excluded.items(), key=lambda kv: kv[1])]
    msg = f"[{prefix}] holdout 覆盖不足，本 session 摘除叶子: {', '.join(parts)}"
    # 只 print 不再 _LOG.warning：CLI 已配置根 logger 时两者会重复输出同一行。
    print(msg, flush=True)

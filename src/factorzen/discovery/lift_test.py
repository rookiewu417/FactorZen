"""组合增量 lift 实验：灰区/lift 队列候选加进库内 active 集，测 OOS RankIC 增量。

单因子库门语义不变；本模块是**后置第二通道**（试用/probation/active 入库裁决）。
挖掘内不跑 lift（保持挖掘快）；由 CLI ``fz factor-library lift-test`` 或
team session 末钩子批处理。

口径：
- 每日 OOS RankIC 与 ``combination.experiment._evaluate_oos`` 的逐日 spearman 一致；
- lift = 配对日 (cand_ic − base_ic) 均值；SE 用 block 均值的样本标准差 / √n_blocks
  （对冲 5 日前向收益重叠导致的日间自相关）。
- **lift 专用 CV**（``DEFAULT_LIFT_CV``）：rolling ``train_days=250`` / ``test_days=40``
  / ``purge_days=5`` / ``expanding=False``，典型折数 ~30。相对旧默认
  （expanding 120d train / 20d test，~85 折）折数更少、后期训练集不再膨胀，
  **lift 数值与旧 expanding/20d 口径不可直接比**（阈值本待标定）。

评估上下文（``LiftEvalContext`` / ``make_lift_context``）：
- 统一对 daily **预处理恰好一次**（含 profile），baseline 物化与 candidate
  materializer 共用同一 ``prepped`` 帧——消除「active 未 prep / candidate 自 prep
  且不传 profile」的不对称。
- **评分窗 ≠ 建模窗**：combine / CV 仍在全帧滚动（walk-forward 可用窗前数据）；
  仅对日 IC 序列按 ``admission_start`` / ``admission_end`` 裁剪后做配对统计。
- ``admission_start=None``（默认）→ 不裁评分窗，向后兼容逃生口；
  ``admission_end=None`` → 裁到帧尾。
- ``horizon`` 显式写入 ctx 与结果行，不再隐式依赖 ``DEFAULT_HORIZON``。
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from factorzen.config.settings import FACTOR_LIBRARY_DIR
from factorzen.core.stats import spearman_avg_rank
from factorzen.discovery.guardrails import (
    DEFAULT_HOLDOUT_MIN_DAYS,
    DEFAULT_LIFT_THRESHOLD,
    REJECT_CATEGORY_GRAY_ZONE,
)

_LOG = logging.getLogger(__name__)

DEFAULT_TOP_M = 10
DEFAULT_HORIZON = 5
DEFAULT_BLOCK_DAYS = 20
# None → run_lift_tests 内自适应；显式 int（含 CLI --lift-workers）不走自适应。
DEFAULT_LIFT_WORKERS: int | None = None

# lift 专用 CV 单一真源：run_lift_tests / run_group_lift 共用；调用方 cv_params 覆盖语义不变。
# rolling 250d/40d → 折数 ~30；expanding=False 避免后期折训练集膨胀。
# 与旧 expanding 120/20 的 lift 数值不可直接比（阈值本待标定）。
DEFAULT_LIFT_CV: dict[str, Any] = {
    "train_days": 250,
    "test_days": 40,
    "purge_days": 5,
    "embargo_days": 0,
    "expanding": False,
}

# 每 worker 峰值约 3–5GB（build_panel 中间物化）；预留共享长表 1–2GB → 约 5GB/worker。
_LIFT_GB_PER_WORKER = 5
_LIFT_WORKERS_CAP = 4
_LIFT_WORKERS_FALLBACK = 2


def adaptive_lift_workers() -> int:
    """按可用内存自适应 lift 并发：``max(2, min(4, 可用内存GB // 5))``。

    依据：共享 base_panel 后每 worker 峰值约 0.5–1GB；2 并发在任意常见内存
    状况下都安全。Linux 用 ``SC_AVPHYS_PAGES * SC_PAGE_SIZE``；``sysconf`` 异常
    回退 2。显式 ``lift_workers=1`` 仍走纯串行（语义不变）。
    """
    try:
        avail = int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        avail_gb = avail / (1024.0 ** 3)
        return max(2, min(_LIFT_WORKERS_CAP, int(avail_gb // _LIFT_GB_PER_WORKER)))
    except (AttributeError, OSError, ValueError, TypeError):
        return _LIFT_WORKERS_FALLBACK


def resolve_lift_workers(lift_workers: int | None) -> int:
    """``None`` → 自适应并打日志；显式 int 原样使用（含 0/负数，由调用方 ``<=1`` 串行）。"""
    if lift_workers is None:
        w = adaptive_lift_workers()
        try:
            avail = int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
            avail_gb = avail / (1024.0 ** 3)
            reason = f"avail_mem≈{avail_gb:.1f}GB // {_LIFT_GB_PER_WORKER}, cap={_LIFT_WORKERS_CAP}"
        except (AttributeError, OSError, ValueError, TypeError):
            reason = f"sysconf 失败回退 {_LIFT_WORKERS_FALLBACK}"
        _LOG.info("lift_workers 自适应 → %d（%s）", w, reason)
        return w
    return int(lift_workers)


def group_gate_ok(
    group: dict,
    *,
    threshold: float,
    lift_se_mult: float,
) -> tuple[bool, float]:
    """组门判定：SE 有限 + lift ≥ max(threshold, se_mult×SE) + 无 error。

    返回 ``(ok, bar)``。SE 缺失/非有限 → 不过（与 lift_admission 同契约；
    不再把「无 SE」当「零方差」把 bar 退化为裸 threshold）。
    session 钩子与 CLI lift-test 共用本函数，单点语义。
    """
    g_lift = group.get("lift")
    g_se = group.get("lift_se")
    if isinstance(g_se, (int, float)) and math.isfinite(float(g_se)):
        se_finite, se_val = True, float(g_se)
    else:
        se_finite, se_val = False, 0.0
    bar = max(float(threshold), float(lift_se_mult) * se_val)
    ok = (
        se_finite
        and g_lift is not None
        and g_lift == g_lift
        and float(g_lift) >= bar
        and not group.get("error")
    )
    return ok, bar


def filter_candidates_by_coverage(
    candidates: list[dict],
    *,
    materialize_candidate: Callable[[str], Any],
    holdout_start: Any = None,
    min_days: int = DEFAULT_HOLDOUT_MIN_DAYS,
) -> tuple[list[dict], list[dict]]:
    """物化后按评分窗非空日数过滤低覆盖候选。

    与 session 末钩子原 1086–1126 逻辑一致：materialize 失败 / 空帧 / OOS 日数
    < ``min_days`` → dropped（带 expression + 原因码 + 实测日数）；其余进 kept。
    CLI 与钩子共用，避免外部 lift-test 路径烧低覆盖 LGBM。
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    for c in candidates:
        expr = c.get("expression")
        try:
            cdf = materialize_candidate(expr) if expr else None
        except Exception as exc:
            dropped.append({
                "expression": expr,
                "n_oos_days": 0,
                "error": f"materialize:{type(exc).__name__}",
            })
            continue
        if cdf is None or (hasattr(cdf, "is_empty") and cdf.is_empty()):
            dropped.append({
                "expression": expr,
                "n_oos_days": 0,
                "error": "materialize_failed",
            })
            continue
        try:
            oos = cdf
            if holdout_start is not None:
                # Date / Utf8 / YYYY-MM-DD 统一到可比较字符串，避免 SchemaError
                if oos.schema.get("trade_date") == pl.Date:
                    oos = oos.with_columns(
                        pl.col("trade_date").dt.strftime("%Y%m%d")
                    )
                else:
                    oos = oos.with_columns(pl.col("trade_date").cast(pl.Utf8))
                hs = holdout_start
                if hasattr(hs, "strftime"):
                    hs = hs.strftime("%Y%m%d")
                else:
                    hs = str(hs).replace("-", "")[:8]
                oos = oos.filter(pl.col("trade_date") >= str(hs))
            if "factor_value" in oos.columns:
                oos = oos.filter(pl.col("factor_value").is_not_null())
            n_oos = int(oos["trade_date"].n_unique()) if oos.height else 0
        except Exception:
            n_oos = 0
        if n_oos < min_days:
            dropped.append({
                "expression": expr,
                "n_oos_days": n_oos,
                "error": "holdout_coverage",
            })
            continue
        kept.append(c)
    return kept, dropped


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
    combined: pl.DataFrame,
    ret_df: pl.DataFrame,
    n_groups: int = 5,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pl.DataFrame:
    """逐日 OOS RankIC 序列，口径对齐 ``_evaluate_oos`` 的 rank_ic_mean 分量。

    返回 ``[trade_date, ic]``（已按 trade_date 排序；无有效日时为空表）。
    分组守卫与 spearman 与 experiment._evaluate_oos 一致（core.stats average-rank）：
    - 日截面 len < n_groups*2 → 跳过
    - spearman_avg_rank 返回 None（n<2 / 常数列 / 非有限）→ 跳过

    ``start`` / ``end``：评分窗闭区间裁剪（trade_date 字符串比较，YYYYMMDD 安全）；
    None 表示不裁该端。模型层（combine/CV）不经此裁剪——只影响返回的日 IC 序列。
    """
    # 两侧都 cast:candidate 物化面板 trade_date 可能是 pl.Date(prepped 帧原生),
    # ret 侧 Utf8——单侧 cast 会 SchemaError(2026-07-15 apply 全灭事故:38/38
    # "join keys don't match")。日期字符串 YYYYMMDD 序与 Date 序一致。
    rdf = ret_df.with_columns(pl.col("trade_date").cast(pl.Utf8))
    m = combined.with_columns(
        pl.col("trade_date").dt.strftime("%Y%m%d")
        if combined.schema.get("trade_date") == pl.Date
        else pl.col("trade_date").cast(pl.Utf8)
    )
    # P4c：combined.ts_code 可能 Categorical，ret 侧常为 Utf8 → 小帧 align 再 join
    if (
        "ts_code" in m.columns
        and "ts_code" in rdf.columns
        and m.schema["ts_code"] != rdf.schema["ts_code"]
    ):
        rdf = rdf.with_columns(pl.col("ts_code").cast(m.schema["ts_code"]))
    m = m.join(rdf, on=["trade_date", "ts_code"], how="inner")
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
    out = pl.DataFrame(
        {"trade_date": [d for d, _ in day_rows], "ic": [ic for _, ic in day_rows]},
        schema={"trade_date": pl.Utf8, "ic": pl.Float64},
    )
    if start is not None:
        out = out.filter(pl.col("trade_date") >= start)
    if end is not None:
        out = out.filter(pl.col("trade_date") <= end)
    return out


def _scored_bounds(
    cand_daily: pl.DataFrame, base_daily: pl.DataFrame,
) -> tuple[str | None, str | None]:
    """配对日实际 min/max trade_date；无有效配对 → (None, None)。"""
    if cand_daily is None or base_daily is None:
        return None, None
    if cand_daily.is_empty() or base_daily.is_empty():
        return None, None
    c = cand_daily.select(pl.col("trade_date").cast(pl.Utf8))
    b = base_daily.select(pl.col("trade_date").cast(pl.Utf8))
    joined = c.join(b, on="trade_date", how="inner")
    if joined.is_empty():
        return None, None
    dates = joined["trade_date"].sort()
    return str(dates[0]), str(dates[-1])


def _provenance_fields(
    *,
    admission_start: str | None,
    admission_end: str | None,
    scored_start: str | None,
    scored_end: str | None,
    horizon: int,
) -> dict[str, Any]:
    return {
        "admission_start": admission_start,
        "admission_end": admission_end,
        "scored_start": scored_start,
        "scored_end": scored_end,
        "horizon": horizon,
    }


def _baseline_hash(active_factor_dfs: Mapping[str, Any] | None) -> str | None:
    """active 表达式集合的稳定 hash（排序后 join；空池 → None）。"""
    if not active_factor_dfs:
        return None
    keys = ",".join(sorted(active_factor_dfs.keys()))
    return hashlib.sha256(keys.encode()).hexdigest()[:16]


def _lift_run_meta(
    *,
    n_input: int,
    n_selected: int,
    truncated: int,
    threshold: float,
    block_days: int,
    cv_kw: dict[str, Any],
    ctx: LiftEvalContext | None,
    market: str,
    baseline_hash: str | None = None,
) -> dict[str, Any]:
    """run_lift_tests 结果行共享的 meta/provenance（经 **meta 进每个 row）。

    不含 lift_se_mult：run_lift_tests 无 se_mult 入参，由 upsert 从调用方注入。
    """
    mkt = ctx.market if ctx is not None else market
    return {
        "truncated_from": n_input if truncated else None,
        "n_selected": n_selected,
        "n_input": n_input,
        "threshold": threshold,
        "block_days": int(block_days),
        "cv_train_days": int(cv_kw.get("train_days", 120)),
        "cv_test_days": int(cv_kw.get("test_days", 20)),
        "profile_name": ctx.profile_name if ctx is not None else None,
        # A 股默认 daily；其它市场不确定则 None（不加臆测）
        "frequency": "daily" if mkt == "ashare" else None,
        "baseline_hash": baseline_hash,
    }


@dataclass
class LiftEvalContext:
    """统一 lift 评估上下文：一次 prep、显式 horizon、可选 admission 评分窗。

    - ``prepped``：已预处理帧（可含预热前缀）；物化 / 建模用全帧。
    - ``admission_start`` / ``admission_end``：仅裁**评分**日 IC（None=不裁 / 至帧尾）。
    - ``profile_name``：provenance；profile 对象不进 ctx（防序列化坑）。
    - ``python_universe`` / ``python_market``：python 型（``py::``）候选物化口径；
      构建 materializer 时透传（expression 路径忽略）。
    """

    market: str
    prepped: pl.DataFrame
    leaf_map: dict[str, str] | None
    horizon: int
    admission_start: str | None
    admission_end: str | None
    library_root: str = str(FACTOR_LIBRARY_DIR)
    profile_name: str | None = None
    python_universe: str | None = None
    python_market: str = "ashare"


def make_lift_context(
    market: str,
    daily: pl.DataFrame,
    *,
    profile=None,
    leaf_map: dict[str, str] | None = None,
    horizon: int = DEFAULT_HORIZON,
    admission_start: str | None = None,
    admission_end: str | None = None,
    library_root: str = str(FACTOR_LIBRARY_DIR),
    prepped: pl.DataFrame | None = None,
    python_universe: str | None = None,
    python_market: str = "ashare",
) -> LiftEvalContext:
    """构造 ``LiftEvalContext``：``_preprocess_daily(daily, profile)`` 恰好一次并 sort。

    baseline 与 candidate 此后共用同一 ``prepped``（对称性的根）。

    ``prepped``：可选；session 已有同源 prep 帧时传入，跳过内部 ``_preprocess_daily``
    （须与 ``daily``/profile 同源——与 mine 评估帧同一契约；CLI 注释铁律）。
    传入帧会 sort(ts_code, trade_date) 以保证与默认路径一致。

    ``python_universe`` / ``python_market``：写入 ctx，供
    ``_materializer_from_prepped`` 在缺省 materializer 构建时透传。
    """
    from factorzen.discovery.evaluation import _preprocess_daily

    if prepped is None:
        prepped = _preprocess_daily(daily, profile).sort(["ts_code", "trade_date"])
    else:
        prepped = prepped.sort(["ts_code", "trade_date"])
    profile_name = getattr(profile, "name", None) if profile is not None else None
    return LiftEvalContext(
        market=market,
        prepped=prepped,
        leaf_map=leaf_map,
        horizon=int(horizon),
        admission_start=admission_start,
        admission_end=admission_end,
        library_root=library_root,
        profile_name=profile_name,
        python_universe=python_universe,
        python_market=python_market,
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

    # 日 diff 全零：候选 IC ≡ 基线（常见于候选列未进模型/全缺）。
    # SE=0 会像「高确信度零增量」；无信息时 SE 置 None，准入走 reject。
    if n_days > 0 and bool(np.all(diffs == 0)):
        mid0 = (n_blocks + 1) // 2
        first0 = block_slices[:mid0]
        second0 = block_slices[mid0:]
        return {
            "lift": 0.0,
            "lift_se": None,
            "n_blocks": n_blocks,
            "n_days": n_days,
            "lift_first_half": (
                float(np.mean(np.concatenate(first0))) if first0 else None
            ),
            "lift_second_half": (
                float(np.mean(np.concatenate(second0))) if second0 else None
            ),
        }

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


def _with_safe_feature_names(
    factor_dfs: Mapping[str, pl.DataFrame] | Any,
) -> dict[str, pl.DataFrame] | Any:
    """键映射为安全特征名 f{i}（按插入序，确定性）。

    lgbm 不接受特征名含特殊 JSON 字符（括号/逗号等）——因子键是**真实表达式**时
    直接进 combine 会炸 `LightGBMError: Do not support special JSON characters in
    feature name`（线上事故）。仅影响 lgbm 内部特征名；lift 报告仍用真实表达式
    （调用方持有原键，映射不外泄）。"""
    from factorzen.discovery.factor_library import CompactLibraryPool

    if isinstance(factor_dfs, CompactLibraryPool):
        return factor_dfs.with_safe_feature_names()  # type: ignore[return-value]
    return {f"f{i:03d}": df for i, df in enumerate(factor_dfs.values())}


def _is_degenerate_factor_df(df: pl.DataFrame | None) -> bool:
    """与 ``drop_degenerate_factors`` 同口径：空帧 / 无 factor_value / 全缺。"""
    if df is None or df.height == 0 or "factor_value" not in df.columns:
        return True
    return int(df["factor_value"].null_count()) >= df.height


def _safe_pool_with_new_factors(
    safe_active: dict[str, pl.DataFrame],
    new_factor_dfs: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """在已映射的基线安全名后**追加**新因子键（f{n}, f{n+1}, …）。

    禁止对「基线+候选」整表重映射：若候选表达式已在 active 中，整表重映射会
    得到与 base_panel **相同**的 f{{i}} 键集，共享路径 ``new_dfs`` 为空 →
    静默 lift=0（G2 全零行事故根因 b）。追加保证新列名永不落在 base 列集内。
    """
    from factorzen.discovery.factor_library import CompactLibraryPool

    if isinstance(safe_active, CompactLibraryPool):
        extras: dict[str, pl.DataFrame] = {}
        i = len(safe_active)
        for df in new_factor_dfs.values():
            extras[f"f{i:03d}"] = df
            i += 1
        return safe_active.with_extra_factors(extras)  # type: ignore[return-value]
    out = dict(safe_active)
    i = len(safe_active)
    for df in new_factor_dfs.values():
        out[f"f{i:03d}"] = df
        i += 1
    return out


def _empty_lift_fields() -> dict[str, Any]:
    return {
        "lift_se": None,
        "n_blocks": None,
        "lift_first_half": None,
        "lift_second_half": None,
        "admission_ic": None,  # 单因子 admission 窗 RankIC；错误行也有键，形态一致
    }


def _candidate_identity_fields(c: dict) -> dict[str, Any]:
    """从候选 dict 透传 kind/name/impl（有则原样拷入，无则不加键）。"""
    out: dict[str, Any] = {}
    for k in ("kind", "name", "impl"):
        if k in c:
            out[k] = c[k]
    return out


def _lgbm_lift_params(*, lift_workers: int) -> dict[str, Any]:
    """生产路径 LGBM 线程/确定性参数：并行/串行 parity（同 seed 可复现）。"""
    workers = max(1, int(lift_workers))
    n_cpu = os.cpu_count() or 1
    return {
        "num_threads": max(1, n_cpu // workers),
        "deterministic": True,
        "force_row_wise": True,
    }


def run_lift_tests(
    gray_candidates: list[dict],
    *,
    market: str,
    daily: pl.DataFrame,
    leaf_map: dict[str, str] | None = None,
    library_root: str = str(FACTOR_LIBRARY_DIR),
    cv_params: dict[str, Any] | None = None,
    top_m: int | None = None,
    threshold: float = DEFAULT_LIFT_THRESHOLD,
    seed: int = 0,
    active_factor_dfs: dict[str, pl.DataFrame] | Any | None = None,
    ret_df: pl.DataFrame | None = None,
    materialize_candidate=None,
    combine_fn=None,
    horizon: int | None = None,
    block_days: int = DEFAULT_BLOCK_DAYS,
    ctx: LiftEvalContext | None = None,
    lift_workers: int | None = DEFAULT_LIFT_WORKERS,
    base_daily: pl.DataFrame | None = None,
) -> list[dict]:
    """对灰区/lift 队列候选跑 lgbm 组合 OOS lift 实验。

    - gray 按 |residual_ic_train|（缺则 |ic_train|）降序；``top_m=None`` 全测，
      否则截断（``truncated_from`` / n_selected 语义不变——**no silent caps**）。
    - 基线**只算一次**（含每日 IC 序列复用）：库内 active 集合 → combine → 每日 RankIC。
      ``base_daily`` 非 None 时跳过基线 combine，直接复用注入的每日 IC 序列。
    - 每候选：active+candidate → 同 CV 同 seed → 配对日 lift + 块 SE + 半段。
    - ``lift ≥ threshold`` → passed（最终 active/probation/reject 见 ``lift_admission``）。
    - 逐候选 try/except：一个坏候选不崩整批。
    - ``lift_workers``：候选级线程并行。``None``（默认）→
      ``adaptive_lift_workers()``（``max(2, min(4, 可用内存GB//5))``，sysconf
      异常回退 2）；显式 int 不走自适应。``<=1`` 纯串行、**不**建
      ``ThreadPoolExecutor``（零回归约定同 ``_llm_map``）。生产路径每模型注入
      ``num_threads=cpu//workers`` + deterministic；注入的 ``combine_fn``（测试 mock）
      不强加 params。
    - **基线宽面板共享**（生产路径）：``build_panel(active)`` 一次，worker 只读
      引用；每候选在稳定 ``safe_active`` 后**追加**新安全名再
      ``combine_lgbm(..., base_panel=)`` 只 join 新列。注入 ``combine_fn`` 时
      关闭共享（mock 契约零回归）。候选全缺 → ``error=degenerate_candidate``
      （禁止静默 lift=0）。

    CV 默认见 ``DEFAULT_LIFT_CV``（rolling 250d/40d，~30 折）；``cv_params`` 覆盖。
    lift 数值与旧 expanding 120/20 口径不可直接比。

    ``ctx``（``LiftEvalContext``，可选）：
    - 缺省时从 ctx 派生 ``active_factor_dfs`` / ``ret_df`` / ``materialize_candidate`` /
      ``horizon``（**显式注入优先于 ctx**——现有 mock 契约不破）。
    - 评分窗：``_daily_oos_rank_ic`` 透传 ``ctx.admission_start/end``；
      combine/CV **不裁**（评分窗 ≠ 建模窗）。``admission_start=None`` → 不裁。
    - ``ctx=None`` → 现状路径零回归。

    测试可注入 ``active_factor_dfs`` / ``ret_df`` / ``combine_fn`` / ``materialize_candidate``
    / ``base_daily`` 做离线 mock；生产路径走 ``make_lift_context`` + ``build_library_pool``。
    """
    from factorzen.research.combination.cv import PurgedWalkForwardCV
    from factorzen.research.combination.models import build_panel, combine_lgbm

    n_input = len(gray_candidates)
    ordered = sorted(gray_candidates, key=_rank_ic_key, reverse=True)
    if top_m is None:
        selected = list(ordered)
    else:
        selected = ordered[: max(0, int(top_m))]
    truncated = n_input - len(selected)
    workers = resolve_lift_workers(lift_workers)

    # horizon / admission：显式优先，否则 ctx，再否则默认
    _horizon = (
        int(horizon) if horizon is not None
        else (ctx.horizon if ctx is not None else DEFAULT_HORIZON)
    )
    adm_start = ctx.admission_start if ctx is not None else None
    adm_end = ctx.admission_end if ctx is not None else None

    cv_kw: dict[str, Any] = dict(DEFAULT_LIFT_CV)
    if cv_params:
        cv_kw.update(cv_params)
    cv = PurgedWalkForwardCV(**cv_kw)
    # 注入 combine_fn（测试 mock）不强加 LGBM params；生产路径保并行/串行 parity
    # 与 base_panel 共享（mock 仍收全量 factor_dfs，契约不变）
    share_base_panel = combine_fn is None
    if combine_fn is not None:
        _combine = combine_fn
    else:
        _lgbm_params = _lgbm_lift_params(lift_workers=max(1, workers))

        def _combine(fds, rdf, c, **kw):
            return combine_lgbm(
                fds, rdf, c, seed=seed, params=_lgbm_params, **kw,
            )

    prov_empty = _provenance_fields(
        admission_start=adm_start,
        admission_end=adm_end,
        scored_start=None,
        scored_end=None,
        horizon=_horizon,
    )

    # ── 物化：active 面板 + 收益（ctx 派生；显式注入优先） ─────────────────
    if active_factor_dfs is None:
        from factorzen.discovery.factor_library import build_library_pool

        if ctx is not None:
            active_factor_dfs = build_library_pool(
                ctx.market, ctx.prepped, ctx.leaf_map,
                root=ctx.library_root, statuses=("active",),
            )
        else:
            active_factor_dfs = build_library_pool(
                market, daily, leaf_map, root=library_root, statuses=("active",),
            )

    # meta 含准入 provenance；baseline_hash 在 active 解析后一次算好复用
    meta = _lift_run_meta(
        n_input=n_input,
        n_selected=len(selected),
        truncated=truncated,
        threshold=threshold,
        block_days=block_days,
        cv_kw=cv_kw,
        ctx=ctx,
        market=market,
        baseline_hash=_baseline_hash(active_factor_dfs),
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
                **prov_empty,
                **_candidate_identity_fields(c),
            }
            for c in selected
        ]

    if ret_df is None:
        ret_src = ctx.prepped if ctx is not None else daily
        ret_df = _build_ret_panel(ret_src, horizon=_horizon)

    if materialize_candidate is None:
        if ctx is not None:
            materialize_candidate = _materializer_from_prepped(
                ctx.prepped, ctx.leaf_map,
                python_universe=ctx.python_universe,
                python_market=ctx.python_market,
            )
        else:
            materialize_candidate = _default_materializer(daily, leaf_map)

    # ── 基线宽面板（生产路径一次构建，worker 只读共享；mock 关闭） ───────
    safe_active = _with_safe_feature_names(active_factor_dfs)
    base_panel: pl.DataFrame | None = None
    if share_base_panel:
        try:
            base_panel = build_panel(safe_active, ret_df)
        except Exception as exc:
            _LOG.warning(
                "lift_test base_panel 构建失败，回退逐候选全量 build: %s: %s",
                type(exc).__name__, exc,
            )
            base_panel = None

    # ── 基线（base_daily 注入则跳过 combine；否则算一次并复用日序列） ─────
    try:
        if base_daily is None:
            if base_panel is not None:
                base_combined = _combine(
                    safe_active, ret_df, cv, base_panel=base_panel,
                )
            else:
                base_combined = _combine(safe_active, ret_df, cv)
            base_daily = _daily_oos_rank_ic(
                base_combined, ret_df, start=adm_start, end=adm_end,
            )
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
                **prov_empty,
                **_candidate_identity_fields(c),
            }
            for c in selected
        ]

    n_sel = len(selected)
    done_i = {"n": 0}

    def _eval_one(c: dict) -> dict[str, Any]:
        expr = c.get("expression")
        t0 = time.monotonic()
        row: dict[str, Any] = {
            "expression": expr,
            "lift": None,
            "baseline": baseline,
            "candidate_rank_ic": None,
            "passed": False,
            "error": None,
            "elapsed_s": None,
            **_empty_lift_fields(),
            **meta,
            **prov_empty,
            **_candidate_identity_fields(c),
        }
        try:
            cand_df = materialize_candidate(expr) if expr else None
            if cand_df is None or (hasattr(cand_df, "is_empty") and cand_df.is_empty()):
                row["error"] = "materialize_failed"
                return row
            # 列规范：[trade_date, ts_code, factor_value]
            if "factor_value" not in cand_df.columns:
                row["error"] = "bad_panel_schema"
                return row
            cand_sel = cand_df.select(["trade_date", "ts_code", "factor_value"])
            # 全缺/空候选：与 drop_degenerate 同口径 → 显式 error，禁止静默 lift=0
            if _is_degenerate_factor_df(cand_sel):
                row["error"] = "degenerate_candidate"
                return row
            # 单因子 admission 窗 RankIC（方向权威；≠ 组合 candidate_rank_ic）
            single_daily = _daily_oos_rank_ic(
                cand_sel, ret_df, start=adm_start, end=adm_end,
            )
            row["admission_ic"] = _mean_ic(single_daily)
            # 透传候选 provenance（审计用；方向权威仍是 admission_ic）
            row["ic_train"] = c.get("ic_train")
            row["residual_ic_train"] = c.get("residual_ic_train")
            # 共享路径：在 stable safe_active 后追加候选新键（永不与 base 列撞名）。
            # 非共享：整表重映射（旧路径）；键用表达式串，与 active 撞名则覆盖。
            if base_panel is not None:
                safe_pool = _safe_pool_with_new_factors(
                    safe_active, {str(expr): cand_sel},
                )
                cand_combined = _combine(
                    safe_pool, ret_df, cv, base_panel=base_panel,
                )
            else:
                from factorzen.discovery.factor_library import CompactLibraryPool

                if isinstance(active_factor_dfs, CompactLibraryPool) or isinstance(
                    safe_active, CompactLibraryPool,
                ):
                    safe_pool = _safe_pool_with_new_factors(
                        safe_active, {str(expr): cand_sel},
                    )
                else:
                    pool = dict(active_factor_dfs)
                    pool[str(expr)] = cand_sel
                    safe_pool = _with_safe_feature_names(pool)
                cand_combined = _combine(safe_pool, ret_df, cv)
            cand_daily = _daily_oos_rank_ic(
                cand_combined, ret_df, start=adm_start, end=adm_end,
            )
            cand_ic = _mean_ic(cand_daily)
            stats = paired_lift_stats(cand_daily, base_daily, block_days=block_days)
            scored_s, scored_e = _scored_bounds(cand_daily, base_daily)
            row["candidate_rank_ic"] = cand_ic
            row["lift"] = stats["lift"]
            row["lift_se"] = stats["lift_se"]
            row["n_blocks"] = stats["n_blocks"]
            row["lift_first_half"] = stats["lift_first_half"]
            row["lift_second_half"] = stats["lift_second_half"]
            row["scored_start"] = scored_s
            row["scored_end"] = scored_e
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
        finally:
            row["elapsed_s"] = float(time.monotonic() - t0)
            done_i["n"] += 1
            expr_s = (str(expr) if expr is not None else "")[:60]
            lift_s = row.get("lift")
            lift_fmt = f"{lift_s:.4f}" if isinstance(lift_s, (int, float)) and lift_s == lift_s else repr(lift_s)
            print(
                f"[lift {done_i['n']}/{n_sel}] {expr_s} "
                f"lift={lift_fmt} elapsed={row['elapsed_s']:.2f}s",
                flush=True,
            )
        return row

    # workers<=1：纯串行列表推导，不实例化 ThreadPoolExecutor（零回归同 _llm_map）
    if workers <= 1:
        return [_eval_one(c) for c in selected]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # map 按 selected 序装配（不用 as_completed）
        return list(pool.map(_eval_one, selected))


def run_group_lift(
    queue: list[dict],
    *,
    market: str,
    daily: pl.DataFrame,
    leaf_map: dict[str, str] | None = None,
    library_root: str = str(FACTOR_LIBRARY_DIR),
    cv_params: dict[str, Any] | None = None,
    top_m: int | None = None,  # 与 run_lift_tests 签名对齐；组测忽略截断
    threshold: float = DEFAULT_LIFT_THRESHOLD,
    seed: int = 0,
    active_factor_dfs: dict[str, pl.DataFrame] | Any | None = None,
    ret_df: pl.DataFrame | None = None,
    materialize_candidate=None,
    combine_fn=None,
    horizon: int | None = None,
    block_days: int = DEFAULT_BLOCK_DAYS,
    ctx: LiftEvalContext | None = None,
    base_daily: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """整组候选一次 combine vs 基线的配对 lift 统计。

    全部候选一起加进 active 池 combine 一次；逐候选物化失败的跳过并记
    ``skipped``；全部物化失败返回 error 行。同样走 ``_with_safe_feature_names``。

    ``base_daily`` 非 None 时跳过基线 combine，复用注入序列；成功路径结果含
    ``base_daily``（每日 IC 序列，供 ``run_lift_tests`` 复用，省 1 次 combine）。

    CV 默认见 ``DEFAULT_LIFT_CV``（与 ``run_lift_tests`` 同一真源）。

    ``ctx`` 语义同 ``run_lift_tests``：缺省参数从 ctx 派生（显式注入优先）；
    评分窗裁日 IC、建模全帧；``ctx=None`` 零回归。结果 dict 含
    ``admission_start/end`` / ``scored_start/end`` / ``horizon`` provenance。
    """
    del top_m  # 组测全量；保留形参兼容注入签名
    from factorzen.research.combination.cv import PurgedWalkForwardCV
    from factorzen.research.combination.models import build_panel, combine_lgbm

    _horizon = (
        int(horizon) if horizon is not None
        else (ctx.horizon if ctx is not None else DEFAULT_HORIZON)
    )
    adm_start = ctx.admission_start if ctx is not None else None
    adm_end = ctx.admission_end if ctx is not None else None

    cv_kw: dict[str, Any] = dict(DEFAULT_LIFT_CV)
    if cv_params:
        cv_kw.update(cv_params)
    cv = PurgedWalkForwardCV(**cv_kw)
    share_base_panel = combine_fn is None
    if combine_fn is not None:
        _combine = combine_fn
    else:
        # 组门单次 combine：workers=1 口径的线程数（整组不并行候选）
        _lgbm_params = _lgbm_lift_params(lift_workers=1)

        def _combine(fds, rdf, c, **kw):
            return combine_lgbm(
                fds, rdf, c, seed=seed, params=_lgbm_params, **kw,
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
            "base_daily": None,
            **_provenance_fields(
                admission_start=adm_start,
                admission_end=adm_end,
                scored_start=None,
                scored_end=None,
                horizon=_horizon,
            ),
        }

    if active_factor_dfs is None:
        from factorzen.discovery.factor_library import build_library_pool

        if ctx is not None:
            active_factor_dfs = build_library_pool(
                ctx.market, ctx.prepped, ctx.leaf_map,
                root=ctx.library_root, statuses=("active",),
            )
        else:
            active_factor_dfs = build_library_pool(
                market, daily, leaf_map, root=library_root, statuses=("active",),
            )
    if not active_factor_dfs:
        return _err("empty_active_library")

    if ret_df is None:
        ret_src = ctx.prepped if ctx is not None else daily
        ret_df = _build_ret_panel(ret_src, horizon=_horizon)

    if materialize_candidate is None:
        if ctx is not None:
            materialize_candidate = _materializer_from_prepped(
                ctx.prepped, ctx.leaf_map,
                python_universe=ctx.python_universe,
                python_market=ctx.python_market,
            )
        else:
            materialize_candidate = _default_materializer(daily, leaf_map)

    skipped: list[dict[str, Any]] = []
    expressions: list[str] = []
    # 新增长表因子（不 dict(compact) 以免复制键）
    new_long: dict[str, pl.DataFrame] = {}
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
            cand_sel = cand_df.select(["trade_date", "ts_code", "factor_value"])
            if _is_degenerate_factor_df(cand_sel):
                skipped.append({"expression": expr, "error": "degenerate_candidate"})
                continue
            new_long[str(expr)] = cand_sel
            expressions.append(str(expr))
        except Exception as exc:
            skipped.append({
                "expression": expr,
                "error": f"{type(exc).__name__}:{exc}",
            })

    if not expressions:
        return _err("all_candidates_materialize_failed", skipped=skipped)

    try:
        from factorzen.discovery.factor_library import CompactLibraryPool

        safe_active = _with_safe_feature_names(active_factor_dfs)
        base_panel: pl.DataFrame | None = None
        if share_base_panel:
            try:
                base_panel = build_panel(safe_active, ret_df)
            except Exception as exc:
                _LOG.warning(
                    "run_group_lift base_panel 构建失败，回退全量: %s: %s",
                    type(exc).__name__, exc,
                )
                base_panel = None

        if base_daily is None:
            if base_panel is not None:
                base_combined = _combine(
                    safe_active, ret_df, cv, base_panel=base_panel,
                )
            else:
                base_combined = _combine(safe_active, ret_df, cv)
            base_daily = _daily_oos_rank_ic(
                base_combined, ret_df, start=adm_start, end=adm_end,
            )
        baseline = _mean_ic(base_daily)

        # 新候选按 expressions 序；安全名追加，避免 dict(compact) 复制键
        new_only = {e: new_long[e] for e in expressions}
        if base_panel is not None:
            safe_pool = _safe_pool_with_new_factors(safe_active, new_only)
            group_combined = _combine(
                safe_pool, ret_df, cv, base_panel=base_panel,
            )
        elif isinstance(active_factor_dfs, CompactLibraryPool) or isinstance(
            safe_active, CompactLibraryPool,
        ):
            safe_pool = _safe_pool_with_new_factors(safe_active, new_only)
            group_combined = _combine(safe_pool, ret_df, cv)
        else:
            pool = dict(active_factor_dfs)
            pool.update(new_long)
            safe_pool = _with_safe_feature_names(pool)
            group_combined = _combine(safe_pool, ret_df, cv)
        group_daily = _daily_oos_rank_ic(
            group_combined, ret_df, start=adm_start, end=adm_end,
        )
        stats = paired_lift_stats(group_daily, base_daily, block_days=block_days)
        scored_s, scored_e = _scored_bounds(group_daily, base_daily)
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
        "base_daily": base_daily,
        **_provenance_fields(
            admission_start=adm_start,
            admission_end=adm_end,
            scored_start=scored_s,
            scored_end=scored_e,
            horizon=_horizon,
        ),
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


def _materializer_from_prepped(
    prepped: pl.DataFrame,
    leaf_map: dict[str, str] | None,
    *,
    python_universe: str | None = None,
    python_market: str = "ashare",
):
    """表达式 → 因子面板；接收**已 prep** 帧，不再二次预处理。

    python 型（``py::{name}``）复用 ``factor_library._materialize_python_on_grid``
    （materialize_python_panel + inner-join 到 prepped 网格）；expression 路径
    行为不变。坏候选 debug log + None，不崩整批。
    """
    from factorzen.discovery.evaluation import _factor_df_from_prepped
    from factorzen.discovery.expression import parse_expr
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _materialize_python_on_grid,
        _pool_date_bounds,
        _python_name_from_expression,
        is_python_identity,
    )

    _warned_no_universe = False
    start, end = _pool_date_bounds(prepped)

    def _mat(expr: str) -> pl.DataFrame | None:
        nonlocal _warned_no_universe
        try:
            if is_python_identity(expr):
                if not python_universe:
                    if not _warned_no_universe:
                        _LOG.warning(
                            "python 候选物化需要 python_universe（如 csi300）；"
                            "当前为空，跳过 py:: 候选",
                        )
                        _warned_no_universe = True
                    return None
                name = _python_name_from_expression(expr)
                if not name:
                    return None
                rec = FactorRecord(
                    expression=expr,
                    market=python_market,
                    kind="python",
                    name=name,
                    impl=name,
                )
                return _materialize_python_on_grid(
                    rec,
                    prepped,
                    market=python_market,
                    universe=python_universe,
                    python_materializer=None,
                    start=start,
                    end=end,
                )
            node = parse_expr(expr, leaf_map)
            return _factor_df_from_prepped(node, prepped, leaf_map=leaf_map).select(
                ["trade_date", "ts_code", "factor_value"]
            )
        except Exception as exc:
            _LOG.debug("lift materialize %r: %s: %s", expr, type(exc).__name__, exc)
            return None

    return _mat


def _default_materializer(
    daily: pl.DataFrame,
    leaf_map: dict[str, str] | None,
    *,
    python_universe: str | None = None,
    python_market: str = "ashare",
):
    """表达式 → 因子面板（与 build_library_pool / _factor_df_from_prepped 同路径）。

    内部 prep 后委托 ``_materializer_from_prepped``；签名/行为零回归。
    ``python_universe`` / ``python_market`` 透传到 python 分派。
    """
    from factorzen.discovery.evaluation import _preprocess_daily

    prepped = _preprocess_daily(daily).sort(["ts_code", "trade_date"])
    return _materializer_from_prepped(
        prepped, leaf_map,
        python_universe=python_universe,
        python_market=python_market,
    )


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
    "DEFAULT_LIFT_CV",
    "DEFAULT_LIFT_WORKERS",
    "DEFAULT_TOP_M",
    "LIFT_QUEUE_CATEGORY",
    "LiftEvalContext",
    "adaptive_lift_workers",
    "extract_gray_candidates_from_manifest",
    "extract_lift_queue_from_manifest",
    "filter_candidates_by_coverage",
    "group_gate_ok",
    "lift_admission",
    "make_lift_context",
    "paired_lift_stats",
    "resolve_lift_workers",
    "run_group_lift",
    "run_lift_tests",
]

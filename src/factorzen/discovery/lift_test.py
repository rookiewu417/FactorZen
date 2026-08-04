"""组合增量 lift 实验：灰区/lift 队列候选对库内 active 的残差 OOS RankIC 增量。

单因子库门语义不变；本模块是**后置第二通道**（试用/probation/active 入库裁决）。
挖掘内不跑 lift（保持挖掘快）；由 CLI ``fz factor-library lift-test`` 或
team session 末钩子批处理。

口径（``residual_ic_v1``，Frisch–Waugh）：
- 候选对库 active 因子逐日截面正交化（QR）；对**残差**算逐日 OOS RankIC；
- lift = 残差 IC 序列本身的均值（基线隐含在正交化里，**无**配对差 / 无基线 combine）；
- SE 用 block 均值样本标准差 / √n_blocks（``series_lift_stats`` 单序列内核）；
- ``candidate_rank_ic`` 与 ``lift`` **同源**（都是残差 IC 均值），不是两个独立量；
- ``admission_ic`` 仍是候选**裸** RankIC（方向权威，不换残差）；
- 生产组合为等权线性，残差口径比旧 lgbm walk-forward 更贴近生产。

评估上下文（``LiftEvalContext`` / ``make_lift_context``）：
- 统一对 daily **预处理恰好一次**（含 profile），库池物化与 candidate
  materializer 共用同一 ``prepped`` 帧。
- 评分窗：仅对日 IC 序列按 ``admission_start`` / ``admission_end`` 裁剪；
  投影本身不另开建模窗。
- ``admission_start=None``（默认）→ 不裁评分窗；``admission_end=None`` → 裁到帧尾。
- ``horizon`` 显式写入 ctx 与结果行，不再隐式依赖 ``DEFAULT_HORIZON``。
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from factorzen.config.settings import FACTOR_LIBRARY_DIR
from factorzen.core.dates import iso_date_str, with_iso_date
from factorzen.core.stats import spearman_avg_rank
from factorzen.discovery.guardrails import (
    DEFAULT_HOLDOUT_MIN_DAYS,
    DEFAULT_LIFT_THRESHOLD,
    OVERLAY_LIFT_THRESHOLD,
    REJECT_CATEGORY_GRAY_ZONE,
    SLEEVE_SUBSET_MIN_DAYS,
)

_LOG = logging.getLogger(__name__)

DEFAULT_TOP_M = 10
DEFAULT_HORIZON = 5
DEFAULT_BLOCK_DAYS = 20
# None → run_lift_tests 内自适应；显式 int（含 CLI --lift-workers）不走自适应。
DEFAULT_LIFT_WORKERS: int | None = None

# residual_ic_v1 口径阈值占位——阶段 D null 校准后回填，勿当已标定值用
DEFAULT_RESIDUAL_LIFT_THRESHOLD: float = 0.005

# sleeve overlay_v1（07-19 OOS 验证口径）：combined = main_z + w × sleeve_z
# w=0.25 = 扫参表保守中档（见 .artifacts/event-sleeve-overlay.md / sleeve3-progress）
DEFAULT_OVERLAY_W: float = 0.25
# 掩码有效日下限：与 sleeve 旁路 subset 日数守卫同源
OVERLAY_MIN_MASK_DAYS: int = SLEEVE_SUBSET_MIN_DAYS
# 单日事件股下限（与 event_sleeve_overlay m.sum() >= 3 一致）
OVERLAY_MIN_CROSS: int = 3

# 历史 lgbm walk-forward CV 常量（旧口径引用 / 对照）；residual 引擎不再使用。
DEFAULT_LIFT_CV: dict[str, Any] = {
    "train_days": 250,
    "test_days": 40,
    "purge_days": 5,
    "embargo_days": 0,
    "expanding": False,
}

# 5GB/worker 是**旧 lgbm 路径**（build_panel 中间物化 3–5GB）留下的保守值。
# residual_ic_v1 只做逐日 QR 投影，实际峰值远低于此——但**未实测前不下调**，
# 宁可少开并发（2026-07-18 宿主机刚因另一进程 23GB 触发内核 OOM）。
# 待阶段 D 真实批量跑出 RSS 曲线后再校准。
_LIFT_GB_PER_WORKER = 5
_LIFT_WORKERS_CAP = 4
_LIFT_WORKERS_FALLBACK = 2


def adaptive_lift_workers() -> int:
    """按可用内存自适应 lift 并发：``max(2, min(4, 可用内存GB // 5))``。

    Linux 用 ``SC_AVPHYS_PAGES * SC_PAGE_SIZE``；``sysconf`` 异常回退 2。
    显式 ``lift_workers=1`` 仍走纯串行（语义不变）。
    分母见 ``_LIFT_GB_PER_WORKER`` 上方说明（对残差路径偏保守，有意为之）。
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


# 组门短路豁免的批规模上限：K 超过它一律逐条测，不再全批连坐。
GROUP_GATE_SHORTCIRCUIT_MAX_K = 20


def group_gate_short_circuit(group: dict, *, n_candidates: int) -> bool:
    """组门不过时**是否**维持旧短路（跳过逐候选）。仅算力豁免，非裁决。

    R1（2026-07-28，``.artifacts/r1_lift_power/REPORT.md``）证明组门数学上零功效
    增益：``t_group = mean_i(δ_i)·√T/σ_u``，等权组合的 √K_eff 精确抵消——它检验的
    是**批内平均**增量，150 条批里 2 条真阳给出 t=0.15，永远不过。以组门一票否决
    整批（历史 88% 的拒绝来源）等于用批内平均否决个体，是**假阴性发生器**。

    故组门降级为筛选标记（``group_gate_weak``），默认继续逐条测。只保留一处算力
    豁免：``K ≤ GROUP_GATE_SHORTCIRCUIT_MAX_K`` **且** 组 t = lift/lift_se < 0——
    小批且方向明确为负时逐条翻案的概率可忽略，省算力。

    t 不可算（lift/SE 缺失、非有限、SE=0）→ **不短路**：缺证据不能当成"证据为负"
    搭便车省算力。
    """
    if int(n_candidates) > GROUP_GATE_SHORTCIRCUIT_MAX_K:
        return False
    g_lift, g_se = group.get("lift"), group.get("lift_se")
    if g_lift is None or g_se is None:
        return False
    try:
        lift_f, se_f = float(g_lift), float(g_se)
    except (TypeError, ValueError):
        return False
    if not (np.isfinite(lift_f) and np.isfinite(se_f)) or se_f <= 0.0:
        return False
    return bool(lift_f / se_f < 0.0)


def naked_ic_negative(row: dict) -> bool:
    """裸 IC（``admission_ic``）有限且 < 0。

    缺失 / 不可解析 / 非有限 → ``False``（**缺证据 ≠ 证据为负**；历史库记录与
    ``lift_null`` 零假设校准本就不带该字段）。

    单点语义：``lift_admission`` 的同号门降级判定与 ``upsert_lift_admissions``
    落盘 ``naked_ic_negative`` 共用本函数，避免两处各判一次而漂移。
    """
    raw = row.get("admission_ic")
    if raw is None:
        return False
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(val) and val < 0.0)


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

    ``start`` / ``end``：评分窗闭区间裁剪；两端与序列本身统一经
    ``core.dates`` 规范成 ISO，故传紧凑 ``YYYYMMDD`` 或 ``YYYY-MM-DD`` 等价。
    None 表示不裁该端。模型层（combine/CV）不经此裁剪——只影响返回的日 IC 序列。
    """
    # 两侧都规范成 ISO 再 join:candidate 物化面板 trade_date 常是 pl.Date
    # (prepped 帧原生),ret 侧被 _build_ret_panel cast 成 ISO Utf8。
    # 旧实现对 Date 走 "%Y%m%d" 而对 Utf8 走 cast,两侧形态不同 → join 零命中、
    # 不报错 → IC 序列静默变空 → _mean_ic 哨兵 0.0 写进库(2026-07-18 实证:
    # 库内 2 条 lift 轨记录 admission_ic 全为 0.0,致 forward_track 永判
    # missing_sign)。形态规范化收口到 core.dates 单一真源。
    rdf = with_iso_date(ret_df)
    m = with_iso_date(combined)
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
    # 窗界同样过 iso_date_str：调用方可能传紧凑 YYYYMMDD，而序列是 ISO——
    # 直接比会静默错行（"20260405" > "2026-04-10" 逐字符为真）
    start_iso = iso_date_str(start)
    end_iso = iso_date_str(end)
    if start_iso is not None:
        out = out.filter(pl.col("trade_date") >= start_iso)
    if end_iso is not None:
        out = out.filter(pl.col("trade_date") <= end_iso)
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
    ctx: LiftEvalContext | None,
    market: str,
    baseline_hash: str | None = None,
    seed: int = 0,
    n_lib_factors: int | None = None,
) -> dict[str, Any]:
    """run_lift_tests 结果行共享的 meta/provenance（经 **meta 进每个 row）。

    不含 lift_se_mult：run_lift_tests 无 se_mult 入参，由 upsert 从调用方注入。
    residual_ic_v1：``cv_train_days`` / ``cv_test_days`` 键保留（FactorRecord schema）
    但置 None；``lift_metric`` / ``n_lib_factors`` / ``seed`` 记账。
    """
    mkt = ctx.market if ctx is not None else market
    return {
        "truncated_from": n_input if truncated else None,
        "n_selected": n_selected,
        "n_input": n_input,
        "threshold": threshold,
        "block_days": int(block_days),
        "cv_train_days": None,
        "cv_test_days": None,
        "profile_name": ctx.profile_name if ctx is not None else None,
        # A 股默认 daily；其它市场不确定则 None（不加臆测）
        "frequency": "daily" if mkt == "ashare" else None,
        "baseline_hash": baseline_hash,
        "lift_metric": "residual_ic_v1",
        "n_lib_factors": n_lib_factors,
        "seed": int(seed),
    }


def _daily_series_bounds(
    daily: pl.DataFrame | None,
) -> tuple[str | None, str | None]:
    """单序列实际 min/max trade_date（ISO 字符串）；空 → (None, None)。"""
    if daily is None or daily.is_empty():
        return None, None
    dates = (
        daily.select(pl.col("trade_date").cast(pl.Utf8))
        .get_column("trade_date")
        .sort()
    )
    if dates.is_empty():
        return None, None
    return str(dates[0]), str(dates[-1])


# 共线投影后残差应数值≈0；z-score 前清零，避免 1e-15 噪声被放大成假信号
_RESIDUAL_NEAR_ZERO_ABS: float = 1e-10


def _zscore_long_panel(df: pl.DataFrame) -> pl.DataFrame:
    """长面板逐日截面 z-score；口径复用 ``residual._cs_zscore_null0``（ddof=1，非有限→0）。

    单日 ``max|val| < 1e-10`` 视为共线零残差，直接全 0（不 z-score 放大浮点噪声）。
    """
    from factorzen.discovery.residual import _cs_zscore_null0

    if df is None or df.is_empty():
        return pl.DataFrame(
            schema={
                "trade_date": pl.Utf8,
                "ts_code": pl.Utf8,
                "factor_value": pl.Float64,
            },
        )
    out_dates: list = []
    out_codes: list = []
    out_vals: list[float] = []
    for date, day_df in df.group_by("trade_date", maintain_order=True):
        d = date[0] if isinstance(date, tuple) else date
        codes = day_df["ts_code"].to_list()
        vals = day_df["factor_value"].to_numpy().astype(np.float64, copy=False)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0 or float(np.max(np.abs(finite))) < _RESIDUAL_NEAR_ZERO_ABS:
            z = np.zeros(vals.shape[0], dtype=np.float64)
        else:
            z = _cs_zscore_null0(vals)
        out_dates.extend([d] * len(codes))
        out_codes.extend(codes)
        out_vals.extend(float(v) for v in z)
    if not out_vals:
        return pl.DataFrame(
            schema={
                "trade_date": pl.Utf8,
                "ts_code": pl.Utf8,
                "factor_value": pl.Float64,
            },
        )
    return pl.DataFrame({
        "trade_date": out_dates,
        "ts_code": out_codes,
        "factor_value": out_vals,
    })


def _equal_weight_residual_combo(
    residual_panels: list[pl.DataFrame],
) -> pl.DataFrame:
    """组门组合分：各残差面板逐日 z-score 后等权平均。

    与生产 combine from-library 等权线性口径一致；z-score 见 ``_zscore_long_panel``
    （``residual._cs_zscore_null0``）。
    """
    empty = pl.DataFrame(
        schema={
            "trade_date": pl.Utf8,
            "ts_code": pl.Utf8,
            "factor_value": pl.Float64,
        },
    )
    zs = [
        _zscore_long_panel(p)
        for p in residual_panels
        if p is not None and not p.is_empty()
    ]
    zs = [z for z in zs if not z.is_empty()]
    if not zs:
        return empty
    stacked = pl.concat(zs, how="vertical_relaxed")
    return (
        stacked.group_by(["trade_date", "ts_code"], maintain_order=True)
        .agg(pl.col("factor_value").mean().alias("factor_value"))
    )


def _zero_ic_daily_from_factor_panel(
    factor_df: pl.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pl.DataFrame:
    """残差/组合分存在但 spearman 全跳过（典型：截面常数≈0）时，构造零 IC 日序列。

    共线候选经济含义 = 零增量；禁止落到 ``no_residual_days`` 静默误拒。
    日期经 ``core.dates.iso_date_str`` 规范化后再做 admission 窗过滤。
    """
    empty = pl.DataFrame(schema={"trade_date": pl.Utf8, "ic": pl.Float64})
    if factor_df is None or factor_df.is_empty():
        return empty
    raw_dates = factor_df.select(pl.col("trade_date")).unique().to_series().to_list()
    day_strs: list[str] = []
    for d in raw_dates:
        s = iso_date_str(d)
        if s is None:
            continue
        day_strs.append(s)
    if not day_strs:
        return empty
    day_strs = sorted(set(day_strs))
    start_iso = iso_date_str(start)
    end_iso = iso_date_str(end)
    if start_iso is not None:
        day_strs = [d for d in day_strs if d >= start_iso]
    if end_iso is not None:
        day_strs = [d for d in day_strs if d <= end_iso]
    if not day_strs:
        return empty
    return pl.DataFrame(
        {"trade_date": day_strs, "ic": [0.0] * len(day_strs)},
        schema={"trade_date": pl.Utf8, "ic": pl.Float64},
    )


class LiftBaselineCache:
    """跨 ``run_lift_tests`` / ``run_group_lift`` 调用复用的**基线侧**重设施。

    为什么需要：``factor_library.rebuild`` 的 lift 轨复审是**逐条记录**调
    ``run_lift_tests`` 的，每次都对同一个 active 池重建 ``build_library_panel``
    + ``ResidualProjector``（逐日 QR 全集）+ ``library_equal_weight_score_panel``、
    对同一 ``prepped`` 重建 ret 面板——全 A 2016-2026（10.7M 行 × 87 active）
    实测峰值随记录数反复出现，第 2 条记录时被内核 OOM-kill（23.2G）。
    ``fz factor-library lift-test`` 逐组 ``dataclasses.replace(base_ctx, ...)``
    共享同一实例（replace 原样传字段值），组门 ``run_group_lift`` 与逐条
    ``run_lift_tests`` 之间也复用；team session 末钩子单 ctx 喂三处同理。

    命中判定是**对象同一性**（``is``），不是值相等：缓存持强引用把键对象钉住
    （防 id 回收复用造成误命中），换了池/面板/帧一律重建。缓存只决定「算不算」，
    **不参与任何数值** —— lift 是单一准入裁决，任何口径漂移都不可接受。

    线程安全：``run_lift_tests`` 的候选级 ``ThreadPoolExecutor`` 只会并发触达
    ``scores_for``（sleeve 惰性路径），整锁保证只建一次。
    """

    # 六组槽：pool / panel / projector / scores / ret（各带自己的命中键）+ 一把整锁
    __slots__ = (
        "_lock",
        "_panel",
        "_panel_key",
        "_pool",
        "_pool_key",
        "_pool_src",
        "_proj",
        "_proj_key",
        "_ret",
        "_ret_key",
        "_ret_src",
        "_scores",
        "_scores_key",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pool_key: tuple | None = None
        self._pool_src: Any = None
        self._pool: Any = None
        self._panel_key: Any = None
        self._panel: Any = None
        self._proj_key: Any = None
        self._proj: Any = None
        self._scores_key: Any = None
        self._scores: pl.DataFrame | None = None
        self._ret_key: tuple | None = None
        self._ret_src: Any = None
        self._ret: pl.DataFrame | None = None

    def pool_for(self, src: Any, key: tuple, build: Callable[[Any], Any]) -> Any:
        """``build_library_pool(...)``，按（源帧同一性 + 市场/库根/statuses）缓存。

        为什么必须缓存**池本身**：``run_group_lift`` 与 ``run_lift_tests`` 各自
        `active_factor_dfs is None` 时会各建一个池对象，而 ``panel_for`` 命中判定是
        **对象同一性** —— 两个池 ⇒ 面板/QR 必然也各建一次。CLI lift-test 正是
        「组门紧接逐条」的调用序，全 A × 79 active 上这一份重复是 池 3.2G + 面板
        4.7G + Q，直接就是 2026-07-29 探针 exit=137 的死点。
        """
        with self._lock:
            if (
                self._pool_key is not None
                and self._pool_key == key
                and self._pool_src is not None
                and self._pool_src is src
            ):
                return self._pool
            # 先松手再建：旧池与新池不该在 build 期间共存
            self._pool_key = self._pool_src = self._pool = None
            pool = build(src)
            self._pool_key, self._pool_src, self._pool = key, src, pool
            return pool

    def panel_for(self, pool: Any, build: Callable[[Any], Any]) -> Any:
        """``build_library_panel(pool)``，按 pool 对象同一性缓存。"""
        with self._lock:
            if self._panel_key is not None and self._panel_key is pool:
                return self._panel
            # 换池重建时先释放旧面板：全 A 面板 ~4.7G，新旧共存过一次就是 OOM
            self._panel_key = self._panel = None
            panel = build(pool)
            self._panel_key, self._panel = pool, panel
            return panel

    def projector_for(self, panel: Any, build: Callable[[Any], Any]) -> Any:
        """``ResidualProjector(panel)``（逐日 QR），按 panel 对象同一性缓存。"""
        with self._lock:
            if self._proj_key is not None and self._proj_key is panel:
                return self._proj
            self._proj_key = self._proj = None
            proj = build(panel)
            self._proj_key, self._proj = panel, proj
            return proj

    def scores_for(self, panel: Any, build: Callable[[Any], Any]) -> pl.DataFrame:
        """``library_equal_weight_score_panel(panel)``，按 panel 对象同一性缓存。"""
        with self._lock:
            if self._scores_key is not None and self._scores_key is panel:
                assert self._scores is not None
                return self._scores
            scores = build(panel)
            self._scores_key, self._scores = panel, scores
            return scores

    def ret_for(
        self, src: Any, key: tuple, build: Callable[[Any], pl.DataFrame],
    ) -> pl.DataFrame:
        """``_build_ret_panel(src, ...)``，按（源帧同一性 + 口径标量）缓存。

        ``key`` = ``(horizon, exec_lag, exec_price_col)``——ret 面板随这三者变，
        同一 ctx 被 ``dataclasses.replace`` 换过口径时必须重建。
        """
        with self._lock:
            if (
                self._ret_key is not None
                and self._ret_key == key
                and self._ret_src is not None
                and self._ret_src is src
            ):
                assert self._ret is not None
                return self._ret
            ret = build(src)
            self._ret_key, self._ret_src, self._ret = key, src, ret
            return ret

    def clear(self) -> None:
        """评分结束后松手全部基线槽（池 / 面板 / QR / 等权分数 / ret）。

        CLI ``lift-test --apply`` 在 17 候选 + 79 active 全 A 上实测：评分收尾时
        MaxRSS 已 ~22G（机器 23G），若仍持有面板再走 W1 去相关，会在
        ``upsert_lift_admissions`` 前/中再次被 OOM-kill（exit=137，2026-07-29）。
        评分结果行是纯 dict，不再依赖这些槽——clear 后只留 ``prepped`` 供
        ``make_compact_materializer`` 做内存有界的 W1。
        """
        with self._lock:
            self._pool_key = self._pool_src = self._pool = None
            self._panel_key = self._panel = None
            self._proj_key = self._proj = None
            self._scores_key = self._scores = None
            self._ret_key = self._ret_src = self._ret = None


def _build_active_pool(
    *,
    ctx: Any,
    cache: LiftBaselineCache | None,
    market: str,
    daily: Any,
    leaf_map: dict[str, str] | None,
    library_root: str,
) -> Any:
    """``run_group_lift`` / ``run_lift_tests`` 共用的 active 池构建（走 ctx 缓存）。

    两处此前各写一遍同样的 `build_library_pool(...)`，是 CLAUDE.md 点名的双路径；
    合并后组门与逐条拿到的是**同一个池对象**，下游 ``panel_for`` 的同一性缓存才命中。
    """
    from factorzen.discovery.factor_library import build_library_pool

    if ctx is not None:
        _mkt, _root, _lm = ctx.market, ctx.library_root, ctx.leaf_map
        src = ctx.prepped
    else:
        _mkt, _root, _lm = market, library_root, leaf_map
        src = daily

    def _build(frame: Any) -> Any:
        return build_library_pool(
            _mkt, frame, _lm, root=_root, statuses=("active",),
        )

    if cache is None:
        return _build(src)
    return cache.pool_for(src, (_mkt, _root, ("active",)), _build)


def _panel_builder(build_library_panel: Callable[..., Any], *, consume: bool):
    """``panel_for`` 用的单参构建器；``consume`` 只在池由 lift 内部自建时为 True。"""

    def _build(pool: Any) -> Any:
        return build_library_panel(pool, consume=consume)

    return _build


@dataclass
class LiftEvalContext:
    """统一 lift 评估上下文：一次 prep、显式 horizon、可选 admission 评分窗。

    - ``prepped``：已预处理帧（可含预热前缀）；物化 / 建模用全帧。
    - ``admission_start`` / ``admission_end``：仅裁**评分**日 IC（None=不裁 / 至帧尾）。
    - ``profile_name``：provenance；profile 对象不进 ctx（防序列化坑）。
    - ``python_universe`` / ``python_market``：python 型（``py::``）候选物化口径；
      构建 materializer 时透传（expression 路径忽略）。
    - ``baseline_cache``：基线设施跨调用复用（见 ``LiftBaselineCache``）。不进
      ``__eq__``/``repr``；``dataclasses.replace`` 会把同一实例传给副本，
      故逐组 replace 的 CLI lift-test 也命中同一 panel/projector。
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
    # 成交口径：0 = t 日收盘成交（默认，向后兼容但**不可实现**）；
    # 1 + "open_adj" = t+1 开盘成交（可实现）。见 compute_fwd_returns docstring。
    exec_lag: int = 0
    exec_price_col: str | None = None
    baseline_cache: LiftBaselineCache = field(
        default_factory=LiftBaselineCache, compare=False, repr=False,
    )


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
    exec_lag: int = 0,
    exec_price_col: str | None = None,
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
        exec_lag=int(exec_lag),
        exec_price_col=exec_price_col,
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


def series_lift_stats(
    ic_daily: pl.DataFrame,
    block_days: int = DEFAULT_BLOCK_DAYS,
) -> dict[str, Any]:
    """单序列 lift 统计内核：均值 + 块 SE + 半段稳定性。

    输入 ``[trade_date (可 cast Utf8), ic (Float64)]`` 日序列；``lift`` = 序列均值。
    块切法 / SE / 半段 / 全零守卫与 ``paired_lift_stats`` 的 diff 路径同构：

    - 空帧 / None → 全 None、n_blocks=0、n_days=0
    - 先 ``sort("trade_date")``（cast Utf8 后字符串序）
    - ``lift_se`` = std(块均值, ddof=1) / √n_blocks；n_blocks < 2 → None
    - 半段按**块数**二等分（奇数块中位块归前半）
    - 全日 ic 全 0 → lift=0.0、SE=None、半段照算
    """
    empty: dict[str, Any] = {
        "lift": None,
        "lift_se": None,
        "n_blocks": 0,
        "n_days": 0,
        "lift_first_half": None,
        "lift_second_half": None,
    }
    if ic_daily is None or ic_daily.is_empty():
        return empty

    df = ic_daily.select(
        pl.col("trade_date").cast(pl.Utf8),
        pl.col("ic"),
    ).sort("trade_date")
    if df.is_empty():
        return empty

    diffs = df["ic"].to_numpy().astype(float)
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

    # 日序列全零：无信息增量。SE=0 会像「高确信度零」；无信息时 SE 置 None。
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

    统计内核见 ``series_lift_stats``（对 diff 序列调用）。
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

    diff_df = joined.select(
        pl.col("trade_date"),
        (pl.col("cand_ic") - pl.col("base_ic")).alias("ic"),
    )
    return series_lift_stats(diff_df, block_days=block_days)


def lift_admission(
    row: dict,
    *,
    threshold: float = DEFAULT_LIFT_THRESHOLD,
    se_mult: float = 1.0,
    require_positive_naked_ic: bool = True,
) -> str:
    """统一准入规则：返回 ``\"active\" | \"probation\" | \"reject\"``。

    - lift is None / 非有限 → reject
    - lift_se is None / 转换失败 / 非有限（NaN、±inf）→ reject
      （区间证据不完整，不再按 0 处理）
    - lift ≥ max(threshold, se_mult × lift_se) 且 lift_second_half > 0 → active
      （``row["overlay"]`` 为真且调用方未显式改门时，threshold 换
      ``OVERLAY_LIFT_THRESHOLD=3.35e-4``——overlay 口径量纲校准，见下方注释）
    - lift ≥ 同上门槛但 second_half 为 None 或 ≤ 0 → probation
    - **admission_ic（裸 IC）有限且 < 0 → 该候选封顶 probation**（同号门降级，见下）
    - 否则 reject

    finite lift_se（含 0.0）合法：bar = max(threshold, se_mult × se)。
    orchestrator / rebuild / CLI 三处共用此单一实现。

    **裸 IC 同号门：reject → 强制 probation（P0b，2026-07-28 R1 改造）**

    门的动机（未变）：准入判据用**残差** IC（``residual_ic_v1``，对库正交化后），
    而部署 ``combine_from_library`` 是**裸值等权**（z-score 后等权相加，无正交化）。
    等权无法表达负贡献，故裸 IC 为负的因子在等权部署下是拖累。

    但**「拒之门外」是过度处置**——R1（``.artifacts/r1_lift_power/REPORT.md``）
    量化了代价：对 ρ ≤ −0.15 且 δ ≤ 0.02 的**抑制变量型真阳，本门屏蔽率 48–100%**；
    历史回放显示现行门本可放行的 2 条真阳（``i_limit_up_*`` 涨停封板族，残差
    IC ≈ +0.0093、裸 IC ≈ 0/负）**全部**是被这道门杀的（回放 + 排除法闭合证明）。
    抑制变量符号反向是 W4 已证的**设计行为**，不是坏因子。

    故降级为：过 bar 的负裸 IC 候选**最高只能拿 probation**（进 forward-track，
    由真实 paper forward 证据裁决是否转 active），而不是直接出局。没过 bar 仍 reject。

    **与 2026-07-19 剔负实证的关系（口径差异，互不背书）**：那次实测（csi300
    2020-2026，库内 85 条 active 中 23 条裸 IC 为负，剔除后等权组合 rank_ic
    0.05601 → 0.06048、ICIR 0.2282 → 0.2416、配对 t = +5.338，n=1454）测的是
    **部署收益**——「已在库的负裸 IC 因子该不该进等权组合」；R1 测的是**准入功效**
    ——「负裸 IC 候选该不该有机会入库」。前者不蕴含后者：等权部署阶段可以继续排除
    它们（组合侧的事），准入阶段没有理由销毁证据。两条结论并存，原记录保留。
    **结论仍绑定「等权部署」前提**——若部署改用能表达负贡献的权重，本门须重估
    （signed 权重已于 2026-07-18 实测证伪，见 ``research/combination/methods``）。

    ``admission_ic`` 缺失/非有限 → **跳过**该门（与 lift_se 缺失即拒**有意不同**：
    SE 缺失算不出 bar，而裸 IC 缺失只是少一道门；历史库记录与 ``lift_null``
    零假设校准本就不带该字段，按缺失即拒会全部误杀）。
    ``require_positive_naked_ic=False`` 关门（语义不变：完全不看裸 IC，可拿 active），
    留对照/复检逃生口。
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
    # overlay 口径量纲校准：掩码仅覆盖少数交易日 × w=0.25，增量天然比全截面
    # 成员口径小一个量级；DEFAULT_LIFT_THRESHOLD=0.001 对其结构性过严。
    # 门取掩码内置换 null q95=3.35e-4（2026-07-21 校准，500 次，阴性对照双干净，
    # .artifacts/overlay-null-calib.md）。已知边界：单 session 校准；null 不覆盖
    # w 选择/多重 sleeve 叠加/挖掘选择偏差（q99=4.5e-4，用户裁决取 q95）。
    if row.get("overlay") and threshold == DEFAULT_LIFT_THRESHOLD:
        threshold = OVERLAY_LIFT_THRESHOLD
    bar = max(float(threshold), float(se_mult) * se_val)
    if lift_f < bar:
        return "reject"

    # 裸 IC 同号门（P0b）：负裸 IC **不再 reject**，改为封顶 probation——
    # 已过 bar 的抑制变量型真阳交给 forward-track 用真实 forward 证据裁决。
    # 缺失/不可解析/非有限一律跳过——缺证据 ≠ 证据为负。
    naked_negative = require_positive_naked_ic and naked_ic_negative(row)

    sh = row.get("lift_second_half")
    if sh is not None:
        try:
            sh_f = float(sh)
        except (TypeError, ValueError):
            return "probation"
        if sh_f == sh_f and sh_f > 0 and not naked_negative:
            return "active"
    return "probation"


def _is_degenerate_factor_df(df: pl.DataFrame | None) -> bool:
    """与 ``drop_degenerate_factors`` 同口径：空帧 / 无 factor_value / 全缺。"""
    if df is None or df.height == 0 or "factor_value" not in df.columns:
        return True
    return int(df["factor_value"].null_count()) >= df.height


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


def partition_lift_queue_by_sleeve(
    candidates: list[dict],
) -> tuple[list[dict], list[dict]]:
    """按 ``sleeve_candidate`` 拆稠密 / sleeve 两队列（CLI 与 session 同源）。

    sleeve 不走 residual 组门（见模块 docstring / sleeve3-progress）；稠密路径不变。
    """
    dense: list[dict] = []
    sleeve: list[dict] = []
    for c in candidates:
        if c.get("sleeve_candidate"):
            sleeve.append(c)
        else:
            dense.append(c)
    return dense, sleeve


def library_equal_weight_score_panel(panel: Any) -> pl.DataFrame:
    """库面板 → 等权主分数长表 ``[trade_date, ts_code, factor_value]``。

    与 07-19 ``event_sleeve_overlay``：``main_score = X.mean(axis=2)`` 同源
    （``LibraryPanel.X`` 已是逐日截面 z-score + null→0）。
    """
    empty = pl.DataFrame(
        schema={
            "trade_date": pl.Utf8,
            "ts_code": pl.Utf8,
            "factor_value": pl.Float64,
        },
    )
    if panel is None or getattr(panel, "k", 0) == 0:
        return empty
    X = np.asarray(panel.X)
    if X.ndim != 3 or X.shape[2] == 0:
        return empty
    # **不**整块升 f64：大面板 X 是 f32（见 residual.RESIDUAL_PANEL_F32_CELLS），
    # `np.asarray(X, dtype=np.float64)` 会当场复制一份全尺寸 f64——全 A
    # 2550×5800×79 就是 9.35G，2026-07-29 CLI 探针正是在 sleeve 分支第一次读
    # base_scores 时被这一句从 13.4G 顶到 23.5G 而 OOM-kill。
    # `mean(axis=2, dtype=np.float64)` 用 f64 累加、直接出 (n_d, n_s) f64，
    # 逐值与「先复制成 f64 再 mean」相同（每个 f32 精确升 f64，同一套 pairwise 求和）。
    main = X.mean(axis=2, dtype=np.float64)  # (n_d, n_s)
    n_d, n_s = main.shape
    # 展平；trade_date 统一 ISO 字符串，便于与 ret / cand join。
    # 键列走 gather 而不是逐格 append：(n_d × n_s) 个 python 对象在全 A 是 1.5e7 ×3 条
    # list，光对象头就 GB 级；gather 只多两条 int64 索引数组。
    date_iso = [
        (iso_date_str(d) if iso_date_str(d) is not None else str(d))
        for d in panel.dates
    ]
    code_str = [str(s) for s in panel.stocks]
    idx_d = np.repeat(np.arange(n_d, dtype=np.int64), n_s)
    idx_s = np.tile(np.arange(n_s, dtype=np.int64), n_d)
    return pl.DataFrame(
        {
            "trade_date": pl.Series("trade_date", date_iso, dtype=pl.Utf8).gather(idx_d),
            "ts_code": pl.Series("ts_code", code_str, dtype=pl.Utf8).gather(idx_s),
            "factor_value": pl.Series("factor_value", main.reshape(-1), dtype=pl.Float64),
        },
        schema={
            "trade_date": pl.Utf8,
            "ts_code": pl.Utf8,
            "factor_value": pl.Float64,
        },
    )


def apply_sleeve_overlay(
    base_scores: pl.DataFrame,
    cand_df: pl.DataFrame,
    *,
    w: float = DEFAULT_OVERLAY_W,
    min_cross: int = OVERLAY_MIN_CROSS,
) -> tuple[pl.DataFrame, int]:
    """``combined = main_z + w × sleeve_z``（sleeve 仅事件股非零）。

    事件股 = 候选 ``factor_value`` 有限且非零；日内事件子集 z-score 后叠
    ``w * sleeve_z``。事件股数 < ``min_cross`` 的日不叠（= 基线）。
    返回 ``(combined_long, n_mask_days)``；``n_mask_days`` = 实际发生叠加的日数。
    """
    empty = pl.DataFrame(
        schema={
            "trade_date": pl.Utf8,
            "ts_code": pl.Utf8,
            "factor_value": pl.Float64,
        },
    )
    if base_scores is None or base_scores.is_empty():
        return empty, 0
    if cand_df is None or cand_df.is_empty() or "factor_value" not in cand_df.columns:
        return (
            with_iso_date(base_scores).select(
                pl.col("trade_date").cast(pl.Utf8),
                pl.col("ts_code").cast(pl.Utf8),
                pl.col("factor_value").cast(pl.Float64),
            ),
            0,
        )

    base = with_iso_date(base_scores).select(
        pl.col("trade_date").cast(pl.Utf8),
        pl.col("ts_code").cast(pl.Utf8),
        pl.col("factor_value").cast(pl.Float64),
    )
    cand = with_iso_date(cand_df).select(
        pl.col("trade_date").cast(pl.Utf8),
        pl.col("ts_code").cast(pl.Utf8),
        pl.col("factor_value").cast(pl.Float64),
    )
    # 事件掩码：有限且 |v| > 0
    cand_ev = cand.filter(
        pl.col("factor_value").is_not_null()
        & pl.col("factor_value").is_finite()
        & (pl.col("factor_value").abs() > 0.0)
    )
    if cand_ev.is_empty():
        return base, 0

    from factorzen.discovery.residual import _cs_zscore_null0

    sleeve_rows: list[dict[str, Any]] = []
    n_mask_days = 0
    for _d, g in cand_ev.group_by("trade_date", maintain_order=True):
        d = _d[0] if isinstance(_d, tuple) else _d
        codes = g["ts_code"].to_list()
        vals = g["factor_value"].to_numpy().astype(np.float64, copy=False)
        if vals.size < int(min_cross):
            continue
        # 事件子集内 z-score（与 07-19 _zscore(raw[m]) 同构；ddof=1 走生产 _cs_zscore）
        z = _cs_zscore_null0(vals)
        # _cs_zscore_null0 对全截面 mask；此处 vals 已全有限，z 与 vals 对齐
        if int(np.isfinite(z).sum()) < int(min_cross):
            continue
        n_mask_days += 1
        for code, zv in zip(codes, z, strict=False):
            if not np.isfinite(zv) or zv == 0.0:
                continue
            sleeve_rows.append({
                "trade_date": str(d),
                "ts_code": str(code),
                "sleeve_z": float(zv),
            })

    if not sleeve_rows or n_mask_days == 0:
        return base, 0

    sleeve = pl.DataFrame(sleeve_rows)
    joined = base.join(sleeve, on=["trade_date", "ts_code"], how="left")
    combined = joined.with_columns(
        (
            pl.col("factor_value")
            + float(w) * pl.col("sleeve_z").fill_null(0.0)
        ).alias("factor_value")
    ).select(["trade_date", "ts_code", "factor_value"])
    return combined, n_mask_days


def _count_mask_days_in_window(
    cand_df: pl.DataFrame,
    *,
    start: str | None,
    end: str | None,
    min_cross: int = OVERLAY_MIN_CROSS,
) -> int:
    """admission 窗内事件日数（日事件股 ≥ min_cross）。"""
    if cand_df is None or cand_df.is_empty():
        return 0
    cand = with_iso_date(cand_df).select(
        pl.col("trade_date").cast(pl.Utf8),
        pl.col("factor_value").cast(pl.Float64),
    )
    start_iso = iso_date_str(start)
    end_iso = iso_date_str(end)
    if start_iso is not None:
        cand = cand.filter(pl.col("trade_date") >= start_iso)
    if end_iso is not None:
        cand = cand.filter(pl.col("trade_date") <= end_iso)
    cand = cand.filter(
        pl.col("factor_value").is_not_null()
        & pl.col("factor_value").is_finite()
        & (pl.col("factor_value").abs() > 0.0)
    )
    if cand.is_empty():
        return 0
    counts = cand.group_by("trade_date").len()
    return int(counts.filter(pl.col("len") >= int(min_cross)).height)


def _subset_factor_panel(cand_df: pl.DataFrame) -> pl.DataFrame:
    """保留事件子集行（有限非零）供 admission_ic / 子集 RankIC。"""
    if cand_df is None or cand_df.is_empty():
        return pl.DataFrame(
            schema={
                "trade_date": pl.Utf8,
                "ts_code": pl.Utf8,
                "factor_value": pl.Float64,
            },
        )
    return with_iso_date(cand_df).filter(
        pl.col("factor_value").is_not_null()
        & pl.col("factor_value").is_finite()
        & (pl.col("factor_value").abs() > 0.0)
    ).select(
        pl.col("trade_date").cast(pl.Utf8),
        pl.col("ts_code").cast(pl.Utf8),
        pl.col("factor_value").cast(pl.Float64),
    )


def _subset_mean_rank_ic(
    cand_df: pl.DataFrame,
    ret_df: pl.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
    min_cross: int = OVERLAY_MIN_CROSS,
) -> float:
    """掩码子集逐日 RankIC 均值（事件股日守卫 = ``min_cross``，非 n_groups*2）。

    全截面 ``_daily_oos_rank_ic`` 要求 ≥10 票，事件子集常只有数票——不能复用。
    """
    sub = _subset_factor_panel(cand_df)
    if sub.is_empty():
        return 0.0
    rdf = with_iso_date(ret_df)
    if (
        "ts_code" in sub.columns
        and "ts_code" in rdf.columns
        and sub.schema["ts_code"] != rdf.schema["ts_code"]
    ):
        rdf = rdf.with_columns(pl.col("ts_code").cast(sub.schema["ts_code"]))
    m = sub.join(rdf, on=["trade_date", "ts_code"], how="inner")
    start_iso = iso_date_str(start)
    end_iso = iso_date_str(end)
    if start_iso is not None:
        m = m.filter(pl.col("trade_date") >= start_iso)
    if end_iso is not None:
        m = m.filter(pl.col("trade_date") <= end_iso)
    ics: list[float] = []
    for _d, g in m.group_by("trade_date", maintain_order=True):
        if len(g) < int(min_cross):
            continue
        f = g["factor_value"].to_numpy().astype(float)
        r = g["ret"].to_numpy().astype(float)
        ic = spearman_avg_rank(f, r)
        if ic is not None and np.isfinite(ic):
            ics.append(float(ic))
    if not ics:
        return 0.0
    return float(np.mean(ics))


def run_lift_tests(
    gray_candidates: list[dict],
    *,
    market: str,
    daily: pl.DataFrame,
    leaf_map: dict[str, str] | None = None,
    library_root: str = str(FACTOR_LIBRARY_DIR),
    top_m: int | None = None,
    threshold: float = DEFAULT_RESIDUAL_LIFT_THRESHOLD,
    seed: int = 0,
    active_factor_dfs: dict[str, pl.DataFrame] | Any | None = None,
    ret_df: pl.DataFrame | None = None,
    materialize_candidate=None,
    horizon: int | None = None,
    block_days: int = DEFAULT_BLOCK_DAYS,
    ctx: LiftEvalContext | None = None,
    lift_workers: int | None = DEFAULT_LIFT_WORKERS,
) -> list[dict]:
    """对灰区/lift 队列候选跑 lift 实验。

    **稠密**（默认 / ``sleeve_candidate`` 假）：``residual_ic_v1`` 残差增量。

    - gray 按 |residual_ic_train|（缺则 |ic_train|）降序；``top_m=None`` 全测，
      否则截断（``truncated_from`` / n_selected 语义不变——**no silent caps**）。
    - 库 active → ``build_library_panel`` → ``ResidualProjector``（QR 一次，全批共用）。
    - 每候选：物化 → ``daily_residual_rank_ic`` → ``series_lift_stats``。
    - ``lift`` / ``candidate_rank_ic`` **同源**（残差 IC 均值）；``baseline=None``。
    - ``admission_ic`` = 候选**裸** RankIC（方向权威，不换残差）。

    **sleeve**（``sleeve_candidate=True``）：``overlay_v1`` 叠加增量（07-19 OOS 口径）。

    - 基线 = 库等权主分数逐日 RankIC；叠加 = ``main + w·sleeve_z``（仅事件股）；
    - ``lift`` = 配对日 IC 差均值（``paired_lift_stats``）；``baseline`` = 基线 IC 均值；
    - ``admission_ic`` = **掩码子集** RankIC（稀疏方向权威；全截面裸 IC 被 fill-0 稀释不可用）；
    - 掩码日 < ``OVERLAY_MIN_MASK_DAYS`` → ``error=insufficient_mask_days``。

    共用：

    - ``lift ≥ threshold`` → passed（最终 active/probation/reject 见 ``lift_admission``）。
    - 逐候选 try/except：一个坏候选不崩整批。
    - 残差序列空 → ``error=no_residual_days``、``lift=None``（**禁止静默 lift=0**）。
    - ``lift_workers``：候选级线程并行。``None``（默认）→
      ``adaptive_lift_workers()``；显式 int 不走自适应。``<=1`` 纯串行、**不**建
      ``ThreadPoolExecutor``。残差路径确定性，``seed`` 仅 meta 记账。

    ``ctx``（``LiftEvalContext``，可选）：
    - 缺省时从 ctx 派生 ``active_factor_dfs`` / ``ret_df`` / ``materialize_candidate`` /
      ``horizon``（**显式注入优先于 ctx**）。
    - 评分窗：残差日 IC 与裸 IC 均透传 ``ctx.admission_start/end``。

    测试可注入 ``active_factor_dfs`` / ``ret_df`` / ``materialize_candidate``；
    生产路径走 ``make_lift_context`` + ``build_library_pool``。
    """
    from factorzen.discovery.residual import (
        ResidualProjector,
        build_library_panel,
        daily_residual_rank_ic,
    )

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

    prov_empty = _provenance_fields(
        admission_start=adm_start,
        admission_end=adm_end,
        scored_start=None,
        scored_end=None,
        horizon=_horizon,
    )

    # ── 物化：active 面板 + 收益（ctx 派生；显式注入优先） ─────────────────
    # 基线设施走 ctx 缓存（逐条复审 / 组门紧接逐条时只建一次）；无 ctx 时原路径
    _cache = ctx.baseline_cache if ctx is not None else None
    # 内部自建池 = 本函数独占，此后只读键集（_baseline_hash）+ 建面板 ⇒ 可消费式建面板
    _pool_internal = active_factor_dfs is None
    if active_factor_dfs is None:
        active_factor_dfs = _build_active_pool(
            ctx=ctx, cache=_cache, market=market, daily=daily,
            leaf_map=leaf_map, library_root=library_root,
        )

    baseline_hash = _baseline_hash(active_factor_dfs)

    def _batch_error_rows(error: str, *, n_lib: int | None = None) -> list[dict]:
        meta_err = _lift_run_meta(
            n_input=n_input,
            n_selected=len(selected),
            truncated=truncated,
            threshold=threshold,
            block_days=block_days,
            ctx=ctx,
            market=market,
            baseline_hash=baseline_hash,
            seed=seed,
            n_lib_factors=n_lib,
        )
        return [
            {
                "expression": c.get("expression"),
                "lift": None,
                "baseline": None,
                "candidate_rank_ic": None,
                "passed": False,
                "error": error,
                **_empty_lift_fields(),
                **meta_err,
                **prov_empty,
                **_candidate_identity_fields(c),
            }
            for c in selected
        ]

    if not active_factor_dfs:
        _LOG.warning("lift_test: 库内无 active 因子，无法建残差投影；全部判不过")
        return _batch_error_rows("empty_active_library")

    _panel_build = _panel_builder(build_library_panel, consume=_pool_internal)
    panel = (
        _cache.panel_for(active_factor_dfs, _panel_build)
        if _cache is not None
        else _panel_build(active_factor_dfs)
    )
    if panel is None or panel.k == 0:
        _LOG.warning("lift_test: active 物化面板为空（empty_library_panel）")
        return _batch_error_rows("empty_library_panel", n_lib=0)

    projector = (
        _cache.projector_for(panel, ResidualProjector)
        if _cache is not None
        else ResidualProjector(panel)
    )
    n_lib_k = int(panel.k)
    meta = _lift_run_meta(
        n_input=n_input,
        n_selected=len(selected),
        truncated=truncated,
        threshold=threshold,
        block_days=block_days,
        ctx=ctx,
        market=market,
        baseline_hash=baseline_hash,
        seed=seed,
        n_lib_factors=n_lib_k,
    )
    # overlay 共用：等权基线分数 + 基线日 IC（admission 窗）。
    # **惰性**：它把 (n_dates × n_stocks) 展平成三条 Python list 再建长表
    # （全 A ≈ 1.2e7 行/次），而稠密 residual 分支从头到尾不读它——rebuild 逐条
    # 复审全是稠密路径，急建等于每条记录白付一次 GB 级构造。
    _scores_local: dict[str, pl.DataFrame] = {}

    def _base_scores() -> pl.DataFrame:
        if _cache is not None:
            return _cache.scores_for(panel, library_equal_weight_score_panel)
        if "v" not in _scores_local:
            _scores_local["v"] = library_equal_weight_score_panel(panel)
        return _scores_local["v"]

    if ret_df is None:
        ret_src = ctx.prepped if ctx is not None else daily
        _exec_lag = ctx.exec_lag if ctx is not None else 0
        _exec_col = ctx.exec_price_col if ctx is not None else None
        _ret_build = lambda src: _build_ret_panel(  # noqa: E731
            src, horizon=_horizon, exec_lag=_exec_lag, exec_price_col=_exec_col,
        )
        ret_df = (
            _cache.ret_for(ret_src, (_horizon, _exec_lag, _exec_col), _ret_build)
            if _cache is not None
            else _ret_build(ret_src)
        )

    if materialize_candidate is None:
        if ctx is not None:
            materialize_candidate = _materializer_from_prepped(
                ctx.prepped, ctx.leaf_map,
                python_universe=ctx.python_universe,
                python_market=ctx.python_market,
            )
        else:
            materialize_candidate = _default_materializer(daily, leaf_map)

    # 基线日 IC 懒缓存：仅 sleeve 批需要；稠密路径不读
    _base_daily_cache: dict[str, pl.DataFrame] = {}

    def _base_daily_ic() -> pl.DataFrame:
        key = f"{adm_start}|{adm_end}"
        if key not in _base_daily_cache:
            _base_daily_cache[key] = _daily_oos_rank_ic(
                _base_scores(), ret_df, start=adm_start, end=adm_end,
            )
        return _base_daily_cache[key]

    n_sel = len(selected)
    done_i = {"n": 0}

    def _eval_one(c: dict) -> dict[str, Any]:
        expr = c.get("expression")
        t0 = time.monotonic()
        is_sleeve = bool(c.get("sleeve_candidate"))
        row: dict[str, Any] = {
            "expression": expr,
            "lift": None,
            "baseline": None,
            "candidate_rank_ic": None,
            "passed": False,
            "error": None,
            "elapsed_s": None,
            **_empty_lift_fields(),
            **meta,
            **prov_empty,
            **_candidate_identity_fields(c),
        }
        if is_sleeve:
            row["overlay"] = True
            row["overlay_w"] = float(DEFAULT_OVERLAY_W)
            row["lift_metric"] = "overlay_v1"
            row["sleeve_candidate"] = True
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
            # 透传候选 provenance（审计用；方向权威仍是 admission_ic）
            row["ic_train"] = c.get("ic_train")
            row["residual_ic_train"] = c.get("residual_ic_train")

            if is_sleeve:
                n_mask = _count_mask_days_in_window(
                    cand_sel, start=adm_start, end=adm_end,
                )
                row["n_mask_days"] = n_mask
                if n_mask < int(OVERLAY_MIN_MASK_DAYS):
                    row["error"] = "insufficient_mask_days"
                    row["lift"] = None
                    return row
                # 掩码子集 RankIC = 方向权威（稀疏全截面裸 IC 被 fill-0 稀释）
                row["admission_ic"] = _subset_mean_rank_ic(
                    cand_sel, ret_df, start=adm_start, end=adm_end,
                )

                combined, n_mask_applied = apply_sleeve_overlay(
                    _base_scores(), cand_sel, w=float(DEFAULT_OVERLAY_W),
                )
                # 与守卫计数对齐（apply 全窗；评分仍裁 admission）
                row["n_mask_days"] = n_mask
                del n_mask_applied  # 全窗叠加日；admission 以 n_mask 为准

                base_daily = _base_daily_ic()
                cand_daily = _daily_oos_rank_ic(
                    combined, ret_df, start=adm_start, end=adm_end,
                )
                if base_daily.is_empty() or cand_daily.is_empty():
                    row["error"] = "no_overlay_days"
                    row["lift"] = None
                    return row
                stats = paired_lift_stats(
                    cand_daily, base_daily, block_days=block_days,
                )
                scored_s, scored_e = _scored_bounds(cand_daily, base_daily)
                lift_v = stats["lift"]
                row["baseline"] = _mean_ic(base_daily)
                row["candidate_rank_ic"] = _mean_ic(cand_daily)
                row["lift"] = lift_v
                row["lift_se"] = stats["lift_se"]
                row["n_blocks"] = stats["n_blocks"]
                row["lift_first_half"] = stats["lift_first_half"]
                row["lift_second_half"] = stats["lift_second_half"]
                row["scored_start"] = scored_s
                row["scored_end"] = scored_e
                row["passed"] = bool(
                    lift_v is not None and float(lift_v) >= threshold
                )
                return row

            # ── residual_ic_v1（稠密 / 非 sleeve）──────────────────────────
            # 单因子 admission 窗 RankIC（方向权威；≠ 残差 candidate_rank_ic）
            single_daily = _daily_oos_rank_ic(
                cand_sel, ret_df, start=adm_start, end=adm_end,
            )
            row["admission_ic"] = _mean_ic(single_daily)

            resid_daily = daily_residual_rank_ic(
                cand_sel,
                panel,
                ret_df,
                ret_col="ret",
                projector=projector,
                start=adm_start,
                end=adm_end,
            )
            if resid_daily.is_empty():
                # 区分：轴外/无有效日 → no_residual_days；
                # 共线残差≈0（spearman 跳过常数列）→ 零 IC 序列，lift=0（非错误）
                resid_panel = projector.residualize(cand_sel)
                if resid_panel is None or resid_panel.is_empty():
                    row["error"] = "no_residual_days"
                    row["lift"] = None
                    return row
                resid_daily = _zero_ic_daily_from_factor_panel(
                    resid_panel, start=adm_start, end=adm_end,
                )
                if resid_daily.is_empty():
                    row["error"] = "no_residual_days"
                    row["lift"] = None
                    return row

            stats = series_lift_stats(resid_daily, block_days=block_days)
            scored_s, scored_e = _daily_series_bounds(resid_daily)
            # residual_ic_v1：lift 与 candidate_rank_ic 同源（残差 IC 均值）
            lift_v = stats["lift"]
            row["candidate_rank_ic"] = lift_v
            row["lift"] = lift_v
            row["lift_se"] = stats["lift_se"]
            row["n_blocks"] = stats["n_blocks"]
            row["lift_first_half"] = stats["lift_first_half"]
            row["lift_second_half"] = stats["lift_second_half"]
            row["scored_start"] = scored_s
            row["scored_end"] = scored_e
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
            lift_fmt = (
                f"{lift_s:.4f}"
                if isinstance(lift_s, (int, float)) and lift_s == lift_s
                else repr(lift_s)
            )
            tag = "overlay" if row.get("overlay") else "resid"
            print(
                f"[lift {done_i['n']}/{n_sel}] ({tag}) {expr_s} "
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
    top_m: int | None = None,  # 与 run_lift_tests 签名对齐；组测忽略截断
    threshold: float = DEFAULT_RESIDUAL_LIFT_THRESHOLD,
    seed: int = 0,
    active_factor_dfs: dict[str, pl.DataFrame] | Any | None = None,
    ret_df: pl.DataFrame | None = None,
    materialize_candidate=None,
    horizon: int | None = None,
    block_days: int = DEFAULT_BLOCK_DAYS,
    ctx: LiftEvalContext | None = None,
) -> dict[str, Any]:
    """整组候选残差等权组合的增量 lift（``residual_ic_v1``）。

    组门：整批候选一起相对库投影；组内无增量则短路（调用方语义），本函数只出组统计。

    组增量定义：
    1. 各候选 ``projector.residualize``；
    2. 各残差面板逐日截面 z-score（``residual._cs_zscore_null0``）；
    3. 逐日等权平均成组合分；
    4. ``_daily_oos_rank_ic``（含 admission 窗）；
    5. ``series_lift_stats``。

    返回不含 ``base_daily``；含 ``lift_metric`` / ``n_lib_factors``。
    ``seed`` 仅记账（残差路径确定性）。
    """
    del top_m  # 组测全量；保留形参兼容注入签名
    del seed  # residual 确定性；形参保留给调用方对齐

    from factorzen.discovery.residual import ResidualProjector, build_library_panel

    _horizon = (
        int(horizon) if horizon is not None
        else (ctx.horizon if ctx is not None else DEFAULT_HORIZON)
    )
    adm_start = ctx.admission_start if ctx is not None else None
    adm_end = ctx.admission_end if ctx is not None else None

    def _err(
        msg: str,
        *,
        skipped: list | None = None,
        expressions: list | None = None,
        n_lib: int | None = None,
    ):
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
            "lift_metric": "residual_ic_v1",
            "n_lib_factors": n_lib,
            **_provenance_fields(
                admission_start=adm_start,
                admission_end=adm_end,
                scored_start=None,
                scored_end=None,
                horizon=_horizon,
            ),
        }

    # 与 run_lift_tests 共用同一 ctx 缓存：组门跑完紧接着逐条测，池/panel/QR/ret
    # 本就同源，重建一次全 A 是 GB 级纯重复（CLI lift-test 与 team 钩子都这样连着调）
    _cache = ctx.baseline_cache if ctx is not None else None
    _pool_internal = active_factor_dfs is None
    if active_factor_dfs is None:
        active_factor_dfs = _build_active_pool(
            ctx=ctx, cache=_cache, market=market, daily=daily,
            leaf_map=leaf_map, library_root=library_root,
        )
    if not active_factor_dfs:
        return _err("empty_active_library")

    _panel_build = _panel_builder(build_library_panel, consume=_pool_internal)
    panel = (
        _cache.panel_for(active_factor_dfs, _panel_build)
        if _cache is not None
        else _panel_build(active_factor_dfs)
    )
    if panel is None or panel.k == 0:
        return _err("empty_library_panel", n_lib=0)

    projector = (
        _cache.projector_for(panel, ResidualProjector)
        if _cache is not None
        else ResidualProjector(panel)
    )
    n_lib = int(panel.k)

    if ret_df is None:
        ret_src = ctx.prepped if ctx is not None else daily
        _exec_lag = ctx.exec_lag if ctx is not None else 0
        _exec_col = ctx.exec_price_col if ctx is not None else None
        _ret_build = lambda src: _build_ret_panel(  # noqa: E731
            src, horizon=_horizon, exec_lag=_exec_lag, exec_price_col=_exec_col,
        )
        ret_df = (
            _cache.ret_for(ret_src, (_horizon, _exec_lag, _exec_col), _ret_build)
            if _cache is not None
            else _ret_build(ret_src)
        )

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
    residual_panels: list[pl.DataFrame] = []
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
            resid = projector.residualize(cand_sel)
            if resid is None or resid.is_empty():
                skipped.append({"expression": expr, "error": "no_residual_days"})
                continue
            residual_panels.append(resid)
            expressions.append(str(expr))
        except Exception as exc:
            skipped.append({
                "expression": expr,
                "error": f"{type(exc).__name__}:{exc}",
            })

    if not expressions:
        return _err(
            "all_candidates_materialize_failed",
            skipped=skipped,
            n_lib=n_lib,
        )

    try:
        combo = _equal_weight_residual_combo(residual_panels)
        if combo.is_empty():
            return {
                **_err(
                    "no_residual_days",
                    skipped=skipped,
                    expressions=expressions,
                    n_lib=n_lib,
                ),
                "n_candidates": len(expressions),
                "expressions": expressions,
                "skipped": skipped,
            }
        group_daily = _daily_oos_rank_ic(
            combo, ret_df, start=adm_start, end=adm_end,
        )
        if group_daily.is_empty():
            # 组残差存在但 IC 全跳过（共线/常数组合分）→ 零增量，非错误
            group_daily = _zero_ic_daily_from_factor_panel(
                combo, start=adm_start, end=adm_end,
            )
            if group_daily.is_empty():
                return {
                    "lift": None,
                    "lift_se": None,
                    "n_blocks": 0,
                    "n_days": 0,
                    "lift_first_half": None,
                    "lift_second_half": None,
                    "n_candidates": len(expressions),
                    "expressions": expressions,
                    "skipped": skipped,
                    "baseline": None,
                    "threshold": threshold,
                    "error": "no_residual_days",
                    "lift_metric": "residual_ic_v1",
                    "n_lib_factors": n_lib,
                    **_provenance_fields(
                        admission_start=adm_start,
                        admission_end=adm_end,
                        scored_start=None,
                        scored_end=None,
                        horizon=_horizon,
                    ),
                }
        stats = series_lift_stats(group_daily, block_days=block_days)
        scored_s, scored_e = _daily_series_bounds(group_daily)
    except Exception as exc:
        _LOG.warning("run_group_lift 失败: %s: %s", type(exc).__name__, exc)
        return _err(
            f"group_residual_failed:{type(exc).__name__}",
            skipped=skipped,
            expressions=expressions,
            n_lib=n_lib,
        )

    return {
        **stats,
        "n_candidates": len(expressions),
        "expressions": expressions,
        "skipped": skipped,
        "baseline": None,
        "threshold": threshold,
        "error": None,
        "lift_metric": "residual_ic_v1",
        "n_lib_factors": n_lib,
        **_provenance_fields(
            admission_start=adm_start,
            admission_end=adm_end,
            scored_start=scored_s,
            scored_end=scored_e,
            horizon=_horizon,
        ),
    }


def _build_ret_panel(
    daily: pl.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    exec_lag: int = 0,
    exec_price_col: str | None = None,
) -> pl.DataFrame:
    """horizon 日前向收益面板（复刻 factor_combine 口径）。

    ``exec_lag`` / ``exec_price_col`` 透传给 ``compute_fwd_returns``，用于切到
    **可实现**口径。默认 ``exec_lag=0`` 逐位等价于旧行为（close→close，
    隐含「t 日收盘成交」）。

    ⚠️ 默认口径**系统性高估**可实现收益：实测 csi500 上 lgbm 组合 top 桶年化超额
    +35.20% 中隔夜段占 **100%**，而隔夜段不可交易（算信号需要 t 日收盘价）；
    切到 ``exec_lag=1, exec_price_col="open_adj"`` 后只剩 +10.08%。
    详见 ``compute_fwd_returns`` docstring 与 ``.artifacts/execution-timing.md``。
    """
    from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns

    price_col = "close_adj" if "close_adj" in daily.columns else "close"
    # 先投影再 sort：`sort` 是全量拷贝，而 compute_fwd_returns 只读键 + 价列
    # （输出也只保留 fwd_ret_*）。全宽 prepped 上原地 sort 会多造一份全宽副本，
    # 是 rebuild 复审期峰值的一大块。语义零变化。
    _keep = ["trade_date", "ts_code", price_col]
    if exec_price_col:
        _keep.append(exec_price_col)
    if "ret_1d" in daily.columns:
        _keep.append("ret_1d")
    _src = daily.select([c for c in dict.fromkeys(_keep) if c in daily.columns])
    fwd = compute_fwd_returns(
        _src.sort(["ts_code", "trade_date"]), horizons=[horizon], price_col=price_col,
        exec_lag=exec_lag, exec_price_col=exec_price_col,
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
    "DEFAULT_OVERLAY_W",
    "DEFAULT_RESIDUAL_LIFT_THRESHOLD",
    "DEFAULT_TOP_M",
    "GROUP_GATE_SHORTCIRCUIT_MAX_K",
    "LIFT_QUEUE_CATEGORY",
    "OVERLAY_MIN_CROSS",
    "OVERLAY_MIN_MASK_DAYS",
    "LiftEvalContext",
    "adaptive_lift_workers",
    "apply_sleeve_overlay",
    "extract_gray_candidates_from_manifest",
    "extract_lift_queue_from_manifest",
    "filter_candidates_by_coverage",
    "group_gate_ok",
    "group_gate_short_circuit",
    "library_equal_weight_score_panel",
    "lift_admission",
    "make_lift_context",
    "naked_ic_negative",
    "paired_lift_stats",
    "partition_lift_queue_by_sleeve",
    "resolve_lift_workers",
    "run_group_lift",
    "run_lift_tests",
    "series_lift_stats",
]

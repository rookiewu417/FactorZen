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
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from factorzen.config.settings import FACTOR_LIBRARY_DIR
from factorzen.core.dates import iso_date_str, with_iso_date
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

# residual_ic_v1 口径阈值占位——阶段 D null 校准后回填，勿当已标定值用
DEFAULT_RESIDUAL_LIFT_THRESHOLD: float = 0.005

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
    - **admission_ic（裸 IC）为负 → reject**（P1-① 同号门，见下）
    - lift ≥ max(threshold, se_mult × lift_se) 且 lift_second_half > 0 → active
    - lift ≥ 同上门槛但 second_half 为 None 或 ≤ 0 → probation
    - 否则 reject

    finite lift_se（含 0.0）合法：bar = max(threshold, se_mult × se)。
    orchestrator / rebuild / CLI 三处共用此单一实现。

    **裸 IC 同号门（P1-① 口径错配的解法 ②）**：准入判据用**残差** IC
    （``residual_ic_v1``，对库正交化后），而部署 ``combine_from_library`` 是
    **裸值等权**（z-score 后等权相加，无正交化）。等权无法表达负贡献，
    故裸 IC 为负的因子在部署时是纯拖累。

    实证（2026-07-19，csi300 2020-2026 全窗，同 CV）：库内 85 条 active 有
    23 条统一窗口裸 IC 为负；剔除后等权组合 rank_ic **0.05601 → 0.06048**、
    ICIR **0.2282 → 0.2416**，**配对 t = +5.338（n=1454）显著**。
    （两组共享 62 因子高度相关必须配对——独立样本 SE 是配对 SE 的 11 倍，
    用它得 t=+0.486「不显著」，结论相反。）

    代价已知并接受：W4 已证符号反向属**抑制变量效应**（设计行为，非 bug），
    本门会误杀部分真抑制变量；实测整体净收益为正。
    **结论绑定「等权部署」前提**——若部署改用能表达负贡献的权重，此门须重估
    （signed 权重已于 2026-07-18 实测证伪，见 ``research/combination/methods``）。

    ``admission_ic`` 缺失/非有限 → **跳过**该门（与 lift_se 缺失即拒**有意不同**：
    SE 缺失算不出 bar，而裸 IC 缺失只是少一道门；历史库记录与 ``lift_null``
    零假设校准本就不带该字段，按缺失即拒会全部误杀）。
    ``require_positive_naked_ic=False`` 关门，留对照/复检逃生口。
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

    # 裸 IC 同号门（P1-①）：部署是裸值等权，负裸 IC 因子是纯拖累。
    # 缺失/不可解析/非有限一律跳过——缺证据 ≠ 证据为负。
    if require_positive_naked_ic:
        naked_raw = row.get("admission_ic")
        if naked_raw is not None:
            try:
                naked = float(naked_raw)
            except (TypeError, ValueError):
                naked = float("nan")
            if np.isfinite(naked) and naked < 0.0:
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
    """对灰区/lift 队列候选跑残差增量 lift 实验（``residual_ic_v1``）。

    - gray 按 |residual_ic_train|（缺则 |ic_train|）降序；``top_m=None`` 全测，
      否则截断（``truncated_from`` / n_selected 语义不变——**no silent caps**）。
    - 库 active → ``build_library_panel`` → ``ResidualProjector``（QR 一次，全批共用）。
    - 每候选：物化 → ``daily_residual_rank_ic`` → ``series_lift_stats``。
    - ``lift`` / ``candidate_rank_ic`` **同源**（残差 IC 均值）；``baseline=None``。
    - ``admission_ic`` = 候选**裸** RankIC（方向权威，不换残差）。
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

    panel = build_library_panel(active_factor_dfs)
    if panel is None or panel.k == 0:
        _LOG.warning("lift_test: active 物化面板为空（empty_library_panel）")
        return _batch_error_rows("empty_library_panel", n_lib=0)

    projector = ResidualProjector(panel)
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
        n_lib_factors=int(panel.k),
    )

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

    n_sel = len(selected)
    done_i = {"n": 0}

    def _eval_one(c: dict) -> dict[str, Any]:
        expr = c.get("expression")
        t0 = time.monotonic()
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
            # 单因子 admission 窗 RankIC（方向权威；≠ 残差 candidate_rank_ic）
            single_daily = _daily_oos_rank_ic(
                cand_sel, ret_df, start=adm_start, end=adm_end,
            )
            row["admission_ic"] = _mean_ic(single_daily)
            # 透传候选 provenance（审计用；方向权威仍是 admission_ic）
            row["ic_train"] = c.get("ic_train")
            row["residual_ic_train"] = c.get("residual_ic_train")

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

    panel = build_library_panel(active_factor_dfs)
    if panel is None or panel.k == 0:
        return _err("empty_library_panel", n_lib=0)

    projector = ResidualProjector(panel)
    n_lib = int(panel.k)

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
    fwd = compute_fwd_returns(
        daily.sort(["ts_code", "trade_date"]), horizons=[horizon], price_col=price_col,
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
    "DEFAULT_RESIDUAL_LIFT_THRESHOLD",
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
    "series_lift_stats",
]

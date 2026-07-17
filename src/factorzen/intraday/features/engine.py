"""日内特征物化引擎：1min 湖 → 日频特征面板 + manifest。"""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import warnings
from calendar import monthrange
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.config.settings import DATA_RAW, INTRADAY_FEATURES_DIR
from factorzen.core.storage import load_parquet, partition_exists, save_parquet
from factorzen.intraday.bars_cache import load_or_build_bars
from factorzen.intraday.features.spec import (
    IntradayFeatureSpec,
    battery,
    battery_hash,
)
from factorzen.intraday.sessions import (
    ASHARE_BAR_FREQS,
    BAR_LABEL_CONVENTION,
    canonicalize_minute,
    normalize_freq,
    resample_intraday,
    session_bar_index,
)

_EPS = 1e-12
_KEYS = ["ts_code", "trade_date"]


@dataclass
class BuildReport:
    """``build_intraday_features`` 的构建摘要。"""

    months: list[str]
    rows: int
    n_stocks: int
    manifest_path: str


def _empty_panel(feature_names: Sequence[str]) -> pl.DataFrame:
    """空日频特征面板（正确 schema）。"""
    schema: dict[str, Any] = {
        "trade_date": pl.Date,
        "ts_code": pl.String,
    }
    for name in feature_names:
        schema[name] = pl.Float64
    return pl.DataFrame(schema=schema)


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """分母 ≤0 或 null → null 的除法。"""
    return (
        pl.when(den.is_not_null() & (den > 0))
        .then(num / den)
        .otherwise(None)
    )


def _smart_money_panel(work: pl.DataFrame) -> pl.DataFrame:
    """按 S=|r|/√vol 降序累计成交量，取前 20% 量（含跨阈桶）的 VWAP / 全日 VWAP。"""
    if work.is_empty():
        return pl.DataFrame(
            schema={
                "ts_code": pl.String,
                "trade_date": pl.Date,
                "i_smart_money": pl.Float64,
            }
        )

    day_stats = work.group_by(_KEYS).agg(
        pl.col("vol").sum().cast(pl.Float64).alias("_V"),
        pl.col("amount").sum().cast(pl.Float64).alias("_A"),
    )

    ranked = (
        work.filter(pl.col("_s").is_not_null() & (pl.col("vol") > 0))
        .sort(["ts_code", "trade_date", "_s"], descending=[False, False, True])
        .with_columns(pl.col("vol").cum_sum().over(_KEYS).alias("_cum_vol"))
        .join(day_stats, on=_KEYS, how="left")
        .with_columns(
            # 上一桶累计 < 0.2V → 仍在阈值内或为本根跨阈桶
            (
                (pl.col("_cum_vol") - pl.col("vol").cast(pl.Float64))
                < (0.2 * pl.col("_V"))
            ).alias("_is_smart")
        )
        .filter(pl.col("_is_smart"))
    )

    if ranked.is_empty():
        return day_stats.select(
            pl.col("ts_code"),
            pl.col("trade_date"),
            pl.lit(None, dtype=pl.Float64).alias("i_smart_money"),
        )

    smart = ranked.group_by(_KEYS).agg(
        pl.col("amount").sum().cast(pl.Float64).alias("_smart_a"),
        pl.col("vol").sum().cast(pl.Float64).alias("_smart_v"),
        pl.col("_V").first(),
        pl.col("_A").first(),
    )

    return smart.select(
        pl.col("ts_code"),
        pl.col("trade_date"),
        pl.when(
            (pl.col("_V") > 0)
            & (pl.col("_smart_v") > 0)
            & (pl.col("_A") > 0)
        )
        .then(
            (pl.col("_smart_a") / pl.col("_smart_v"))
            / (pl.col("_A") / pl.col("_V"))
        )
        .otherwise(None)
        .alias("i_smart_money"),
    )


def compute_day_panel(
    minute: pl.DataFrame,
    specs: Sequence[IntradayFeatureSpec],
    freq: str,
    *,
    min_bar_coverage: float = 0.8,
    bars: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """纯函数：1min 帧（或预物化 bars）→ 每股每日一行的日频特征面板。

    流程：``canonicalize_minute`` → ``resample_intraday(freq)`` → 特征计算。
    若传入 ``bars`` 则跳过前两步（供 bars 缓存命中路径）。

    覆盖守卫：当日有效桶数 < ``min_bar_coverage × bars_per_day`` 时，该
    ``(ts_code, 日)`` 全部特征置 null（行保留）。返回前 ``fill_nan(None)``。

    Args:
        minute: 原始 1 分钟 OHLCV 帧（``bars`` 给定时可传空帧）。
        specs: 特征规格（决定输出列名顺序）。
        freq: 计算频率。
        min_bar_coverage: 有效桶覆盖率门槛。
        bars: 可选预 resample 的 bar 帧（与 ``resample_intraday`` schema 一致）。

    Returns:
        列 ``[trade_date, ts_code, i_*×len(specs)]``，按 ``(trade_date, ts_code)`` 排序。
    """
    freq_n = normalize_freq(freq)
    k = ASHARE_BAR_FREQS[freq_n].minutes
    n_bars = ASHARE_BAR_FREQS[freq_n].bars_per_day
    k30 = 30 // k
    feature_names = [s.name for s in specs]

    if bars is None:
        if minute.is_empty():
            return _empty_panel(feature_names)
        # canonicalize 一次；resample 跳过重复过滤
        canon = canonicalize_minute(minute.lazy()).collect()
        if canon.is_empty():
            return _empty_panel(feature_names)
        bars = resample_intraday(canon, freq_n, already_canonical=True)

    if bars.is_empty():
        return _empty_panel(feature_names)

    # resample 行序无契约 → 此处 sort 一次供 shift/first/last
    work = (
        bars.with_columns(
            pl.col("trade_time").dt.date().alias("trade_date"),
            (session_bar_index("trade_time") // k).cast(pl.Int32).alias("_i"),
        )
        .sort(["ts_code", "trade_date", "trade_time"])
        .with_columns(
            pl.col("close").shift(1).over(_KEYS).alias("_pc"),
        )
        .with_columns(
            pl.when(pl.col("_pc").is_null())
            .then(pl.col("close") / pl.col("open") - 1.0)
            .otherwise(pl.col("close") / pl.col("_pc") - 1.0)
            .alias("_r"),
            pl.when(pl.col("_pc").is_null())
            .then(pl.col("close") - pl.col("open"))
            .otherwise(pl.col("close") - pl.col("_pc"))
            .alias("_delta"),
        )
        .with_columns(
            (
                (pl.col("vol") > 0)
                & pl.col("close").is_not_null()
                & pl.col("_r").is_not_null()
                & pl.col("_r").is_finite()
            ).alias("_valid"),
            pl.when(
                (pl.col("vol") > 0)
                & pl.col("_r").is_not_null()
                & pl.col("_r").is_finite()
            )
            .then(
                pl.col("_r").abs()
                / pl.col("vol").cast(pl.Float64).sqrt()
            )
            .otherwise(None)
            .alias("_s"),
            pl.when(pl.col("_r").is_not_null() & pl.col("_r").is_finite())
            .then(pl.col("_r"))
            .otherwise(None)
            .alias("_r_fin"),
        )
    )

    # —— 日聚合（不含 smart_money）——
    r2 = pl.col("_r_fin") ** 2
    r_valid = pl.when(pl.col("_valid")).then(pl.col("_r")).otherwise(None)
    r2_valid = r_valid**2

    agg = work.group_by(_KEYS).agg(
        # 覆盖
        pl.col("_valid").sum().cast(pl.Int32).alias("_n_valid"),
        pl.col("vol").sum().cast(pl.Float64).alias("_V"),
        pl.col("amount").sum().cast(pl.Float64).alias("_A"),
        pl.col("open").first().alias("_o_first"),
        pl.col("close").last().alias("_c_last"),
        # open30 / mid / close30 稳健 close
        pl.col("close")
        .filter(pl.col("_i") <= k30)
        .last()
        .alias("_c_open30"),
        pl.col("close")
        .filter(pl.col("_i") <= (n_bars - k30))
        .last()
        .alias("_c_pre_close30"),
        # 量份额
        pl.col("vol")
        .filter(pl.col("_i") <= k30)
        .sum()
        .cast(pl.Float64)
        .alias("_vol_open30"),
        pl.col("vol")
        .filter(pl.col("_i") > (n_bars - k30))
        .sum()
        .cast(pl.Float64)
        .alias("_vol_close30"),
        # RV 族（有限 r）
        r2.sum().alias("_sum_r2"),
        pl.when(pl.col("_r_fin") < 0).then(r2).otherwise(0.0).sum().alias("_sum_r2_down"),
        pl.when(pl.col("_r_fin") > 0).then(r2).otherwise(0.0).sum().alias("_sum_r2_up"),
        pl.col("_r_fin").abs().max().alias("_max_abs_r"),
        pl.col("_r_fin").abs().sum().alias("_sum_abs_r"),
        # 偏度/峰度（有效桶）
        r_valid.count().cast(pl.Float64).alias("_nv"),
        (r_valid**3).sum().alias("_sum_r3"),
        (r_valid**4).sum().alias("_sum_r4"),
        r2_valid.sum().alias("_sum_r2_v"),
        # 价量相关
        pl.corr(
            pl.when(pl.col("_valid")).then(pl.col("close")),
            pl.when(pl.col("_valid")).then(pl.col("vol").cast(pl.Float64)),
        ).alias("_pv_corr_raw"),
        # Amihud
        pl.when(pl.col("amount") > 0)
        .then(pl.col("_r_fin").abs() / pl.col("amount"))
        .otherwise(None)
        .mean()
        .alias("_amihud_raw"),
        # path
        pl.col("_delta").abs().sum().alias("_sum_abs_delta"),
    )

    # 熵：bar 级 p=vol/Σvol 再聚合（避免 group_by.agg 内 .over）
    vol_sum = work.group_by(_KEYS).agg(
        pl.col("vol").sum().cast(pl.Float64).alias("_V_e")
    )
    work_e = (
        work.join(vol_sum, on=_KEYS, how="left")
        .with_columns(
            pl.when((pl.col("vol") > 0) & (pl.col("_V_e") > 0))
            .then(pl.col("vol").cast(pl.Float64) / pl.col("_V_e"))
            .otherwise(None)
            .alias("_p"),
        )
        .with_columns(
            pl.when(pl.col("_p").is_not_null() & (pl.col("_p") > 0))
            .then(-(pl.col("_p") * pl.col("_p").log()))
            .otherwise(0.0)
            .alias("_plnp"),
        )
    )
    entropy_part = work_e.group_by(_KEYS).agg(
        pl.col("_plnp").sum().alias("_entropy_num"),
        pl.col("vol").filter(pl.col("vol") > 0).count().cast(pl.Float64).alias("_n_pos"),
    )

    agg = agg.join(entropy_part, on=_KEYS, how="left")

    smart = _smart_money_panel(work)

    panel = (
        agg.join(smart, on=_KEYS, how="left")
        .with_columns(
            # i_rv
            pl.when(pl.col("_sum_r2").is_not_null() & (pl.col("_sum_r2") >= 0))
            .then(pl.col("_sum_r2").sqrt())
            .otherwise(None)
            .alias("i_rv"),
            # i_rskew
            pl.when(pl.col("_sum_r2_v") > _EPS)
            .then(
                pl.col("_nv").sqrt()
                * pl.col("_sum_r3")
                / (pl.col("_sum_r2_v") ** 1.5)
            )
            .otherwise(None)
            .alias("i_rskew"),
            # i_rkurt
            pl.when(pl.col("_sum_r2_v") > _EPS)
            .then(
                pl.col("_nv")
                * pl.col("_sum_r4")
                / (pl.col("_sum_r2_v") ** 2)
            )
            .otherwise(None)
            .alias("i_rkurt"),
            # i_downvol_ratio
            pl.when(pl.col("_sum_r2") > _EPS)
            .then(pl.col("_sum_r2_down") / pl.col("_sum_r2"))
            .otherwise(None)
            .alias("i_downvol_ratio"),
            # i_updown_vol
            (
                ((pl.col("_sum_r2_up") + _EPS) / (pl.col("_sum_r2_down") + _EPS)).log()
            ).alias("i_updown_vol"),
            # session returns
            _safe_div(pl.col("_c_open30"), pl.col("_o_first")).sub(1.0).alias(
                "i_ret_open30"
            ),
            _safe_div(pl.col("_c_last"), pl.col("_c_pre_close30")).sub(1.0).alias(
                "i_ret_close30"
            ),
            _safe_div(pl.col("_c_pre_close30"), pl.col("_c_open30")).sub(1.0).alias(
                "i_ret_mid"
            ),
            # vwap dev
            pl.when(pl.col("_V") > 0)
            .then(pl.col("_c_last") / (pl.col("_A") / pl.col("_V")) - 1.0)
            .otherwise(None)
            .alias("i_vwap_dev"),
            # pv corr
            pl.when(pl.col("_n_valid") >= 10)
            .then(pl.col("_pv_corr_raw"))
            .otherwise(None)
            .alias("i_pv_corr"),
            # vol shares
            pl.when(pl.col("_V") > 0)
            .then(pl.col("_vol_open30") / pl.col("_V"))
            .otherwise(None)
            .alias("i_vol_open30_share"),
            pl.when(pl.col("_V") > 0)
            .then(pl.col("_vol_close30") / pl.col("_V"))
            .otherwise(None)
            .alias("i_vol_close30_share"),
            # entropy
            pl.when(pl.col("_n_pos") >= 2)
            .then(pl.col("_entropy_num") / pl.col("_n_pos").log())
            .otherwise(None)
            .alias("i_vol_entropy"),
            # amihud
            (pl.col("_amihud_raw") * 1e9).alias("i_amihud"),
            # path eff
            pl.when(pl.col("_sum_abs_delta") > _EPS)
            .then(
                (pl.col("_c_last") - pl.col("_o_first")).abs()
                / pl.col("_sum_abs_delta")
            )
            .otherwise(None)
            .alias("i_path_eff"),
            # max ret share
            pl.when(pl.col("_sum_abs_r") > _EPS)
            .then(pl.col("_max_abs_r") / pl.col("_sum_abs_r"))
            .otherwise(None)
            .alias("i_max_ret_share"),
            # coverage flag
            (
                pl.col("_n_valid").cast(pl.Float64)
                < (min_bar_coverage * float(n_bars))
            ).alias("_low_cov"),
        )
    )

    # 覆盖不足 → 全部特征 null（行保留）
    null_feats = [
        pl.when(pl.col("_low_cov")).then(None).otherwise(pl.col(name)).alias(name)
        for name in feature_names
    ]
    out = (
        panel.select(
            pl.col("trade_date").cast(pl.Date),
            pl.col("ts_code").cast(pl.String),
            *null_feats,
        )
        .with_columns([pl.col(c).fill_nan(None) for c in feature_names])
        .sort(["trade_date", "ts_code"])
    )
    return out


def _month_windows(start: str, end: str) -> list[tuple[str, str, str]]:
    """生成 ``[(YYYY-MM, month_start_YYYYMMDD, month_end_YYYYMMDD), ...]``。

    各月窗口与 ``[start, end]`` 求交。
    """
    s = datetime.strptime(start, "%Y%m%d").date()
    e = datetime.strptime(end, "%Y%m%d").date()
    if e < s:
        return []

    out: list[tuple[str, str, str]] = []
    y, m = s.year, s.month
    while True:
        first = date(y, m, 1)
        last = date(y, m, monthrange(y, m)[1])
        w0 = max(s, first)
        w1 = min(e, last)
        if w0 <= w1:
            label = f"{y:04d}-{m:02d}"
            out.append((label, w0.strftime("%Y%m%d"), w1.strftime("%Y%m%d")))
        if (y, m) == (e.year, e.month):
            break
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return out


def _git_sha_or_none() -> str | None:
    """复用仓库 ``get_git_sha``；不可用时返回 ``None``。"""
    try:
        from factorzen.core.experiment import get_git_sha

        sha = get_git_sha()
        if not sha or sha == "unknown":
            return None
        return sha
    except Exception:
        return None


def _manifest_path(out_dir: Path, version: str, freq: str) -> Path:
    return out_dir / version / freq / "manifest.json"


def read_manifest(
    *,
    version: str = "v1",
    freq: str = "5min",
    base_dir: Path | None = None,
) -> dict[str, Any] | None:
    """读取特征面板 manifest；不存在返回 ``None``。"""
    base = INTRADAY_FEATURES_DIR if base_dir is None else base_dir
    path = _manifest_path(base, version, normalize_freq(freq))
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _merge_coverage(
    existing: dict[str, Any] | None,
    start: str,
    end: str,
    months: list[str],
) -> dict[str, Any]:
    """合并 coverage：months 并集、start/end 取极值。"""
    if existing is None:
        return {"start": start, "end": end, "months": sorted(months)}
    cov = existing.get("coverage") or {}
    old_months = list(cov.get("months") or [])
    merged_months = sorted(set(old_months) | set(months))
    old_start = cov.get("start") or start
    old_end = cov.get("end") or end
    return {
        "start": min(str(old_start), start),
        "end": max(str(old_end), end),
        "months": merged_months,
    }


def _month_label_ym(label: str) -> tuple[int, int]:
    """``YYYY-MM`` → (year, month)。"""
    return int(label[:4]), int(label[5:7])


def _should_skip_month(
    label: str,
    *,
    data_type: str,
    out_dir: Path,
    existing: dict[str, Any] | None,
    bhash: str,
    force: bool,
) -> bool:
    """三条同时成立才跳过该月（缓存键完整性）。

    ① derived 分区存在且非空；② manifest coverage 覆盖该月；
    ③ manifest battery_hash 与当前电池一致。

    ``force=True`` 时永不跳过。hash 不一致时由入口 fail-loudly / overwrite
    处理，此处不跳过（与 overwrite 后全量重刷语义一致）。
    """
    if force:
        return False
    if existing is None or existing.get("battery_hash") != bhash:
        return False
    cov_months = list((existing.get("coverage") or {}).get("months") or [])
    if label not in cov_months:
        return False
    y, m = _month_label_ym(label)
    return partition_exists(data_type, y, m, base_dir=out_dir)


def _process_one_month(
    label: str,
    m_start: str,
    m_end: str,
    *,
    version: str,
    freq_n: str,
    source_dir: str,
    out_dir: str,
    data_type: str,
    codes: list[str] | None,
    min_bar_coverage: float,
    bars_cache_dir: str | None = None,
    use_bars_cache: bool = True,
) -> tuple[str, int, int] | None:
    """单月 load → compute → save。进程池 worker 入口（可 pickle）。

    分区写入互不相交（``year=/month=`` 独立目录）。返回
    ``(label, rows, n_stocks)``；空月 / 无源数据返回 ``None``。

    bars 层经 ``load_or_build_bars`` 读穿缓存（双消费方共享中间层）。
    """
    src = Path(source_dir)
    out = Path(out_dir)
    specs = battery(version=version, freq=freq_n)
    try:
        lf = load_parquet(
            "minute_1min",
            start=m_start,
            end=m_end,
            date_col="trade_time",
            base_dir=src,
        )
        minute = lf.collect()
    except Exception:
        return None

    if minute.is_empty():
        del minute
        gc.collect()
        return None

    # 全市场 bars 缓存（codes 过滤在 resample 之后，避免污染共享中间层）
    cache_path = Path(bars_cache_dir) if bars_cache_dir else None
    if use_bars_cache:
        bars = load_or_build_bars(
            label,
            freq_n,
            source_dir=src,
            cache_dir=cache_path,
            minute=minute,
        )
    else:
        from factorzen.intraday.bars_cache import build_bars_from_minute

        bars = build_bars_from_minute(minute, freq_n)
    del minute
    gc.collect()

    if codes is not None:
        bars = bars.filter(pl.col("ts_code").is_in(codes))
        if bars.is_empty():
            del bars
            gc.collect()
            return None

    panel = compute_day_panel(
        pl.DataFrame(schema={"ts_code": pl.String}),  # unused when bars given
        specs,
        freq_n,
        min_bar_coverage=min_bar_coverage,
        bars=bars,
    )
    del bars
    gc.collect()

    if panel.is_empty():
        del panel
        gc.collect()
        return None

    save_parquet(
        panel,
        data_type=data_type,
        date_col="trade_date",
        base_dir=out,
        mode="overwrite",
    )
    rows = panel.height
    n_stocks = panel["ts_code"].n_unique()
    del panel
    gc.collect()
    return (label, rows, int(n_stocks))


def build_intraday_features(
    start: str,
    end: str,
    *,
    freq: str = "5min",
    version: str = "v1",
    codes: list[str] | None = None,
    out_dir: Path | None = None,
    source_dir: Path | None = None,
    overwrite: bool = False,
    force: bool = False,
    workers: int = 1,
    min_bar_coverage: float = 0.8,
    bars_cache_dir: Path | None = None,
    use_bars_cache: bool = True,
) -> BuildReport:
    """逐月物化日内特征面板并写 manifest。

    对区间内每个自然月：
    ``load_parquet("minute_1min", ...)`` → bars 缓存读穿 → ``compute_day_panel`` →
    ``save_parquet(..., mode="overwrite")``；大帧显式释放防 OOM。

    **增量缺月跳过**（``force=False`` 时）：当 derived 分区非空、manifest
    coverage 已覆盖该月、且 ``battery_hash`` 一致时跳过该月重算。
    ``battery_hash`` 冲突时对齐既有语义：``overwrite=False`` fail-loudly；
    ``overwrite=True`` 全量重刷并重置 coverage。``force=True`` 忽略跳过判据
    全量重算（hash 守卫仍生效）。

    **月级并行**（``workers>1``）：``ProcessPoolExecutor`` 按月分发；分区
    写入在 worker 内（路径互不相交）；**manifest 合并/写入仅主进程串行**。

    不用 ``IntradayDataContext``（其 ``max_bars`` / ``expanded_start`` 对全市场
    物化是错误口径）。

    Args:
        start / end: ``YYYYMMDD`` 闭区间。
        freq: bar 频率。
        version: 电池版本。
        codes: 可选股票过滤。
        out_dir: 输出根目录，默认 ``INTRADAY_FEATURES_DIR``。
        source_dir: 1min 源湖根，默认 ``DATA_RAW``。
        overwrite: battery_hash 冲突时是否强制重写。
        force: 忽略增量跳过，全量重算已覆盖月。
        workers: 月级进程并行度，默认 1。单月峰值约 7.6 GiB；
            24 GiB 机器建议 2；``>2`` 会打警告。
        min_bar_coverage: 有效桶覆盖率门槛。
        bars_cache_dir: bars 中间层缓存根（默认 ``DATA_DERIVED``）。
        use_bars_cache: 是否走 ``load_or_build_bars``（默认 True）。

    Returns:
        ``BuildReport`` 摘要（``months`` 为本 run 实际重算的月，按标签排序）。

    Raises:
        ValueError: 已有 manifest 的 battery_hash 不匹配且 ``overwrite=False``；
            或 ``workers < 1``。
    """
    if workers < 1:
        raise ValueError(f"workers 必须 ≥ 1，得到 {workers}")
    if workers > 2:
        warnings.warn(
            f"workers={workers}：单月峰值 ~7.6 GiB，并行度>2 在 24 GiB 机器上易 OOM",
            UserWarning,
            stacklevel=2,
        )

    freq_n = normalize_freq(freq)
    specs = battery(version=version, freq=freq_n)
    bhash = battery_hash(specs)
    out = INTRADAY_FEATURES_DIR if out_dir is None else Path(out_dir)
    src = DATA_RAW if source_dir is None else Path(source_dir)
    # 自定义源湖且未显式指定 bars_cache_dir 时，不写共享 DATA_DERIVED
    if (
        use_bars_cache
        and bars_cache_dir is None
        and source_dir is not None
        and Path(source_dir).resolve() != DATA_RAW.resolve()
    ):
        use_bars_cache = False
    data_type = f"{version}/{freq_n}"
    mpath = _manifest_path(out, version, freq_n)

    existing = read_manifest(version=version, freq=freq_n, base_dir=out)
    if existing is not None and existing.get("battery_hash") != bhash and not overwrite:
        raise ValueError(
            f"manifest battery_hash 不匹配: 已有 {existing.get('battery_hash')!r}，"
            f"当前 {bhash!r}；请设 overwrite=True 以重写"
        )

    hash_ok = existing is not None and existing.get("battery_hash") == bhash
    windows = _month_windows(start, end)
    jobs: list[tuple[str, str, str]] = []
    skipped_months: list[str] = []

    for label, m_start, m_end in windows:
        if _should_skip_month(
            label,
            data_type=data_type,
            out_dir=out,
            existing=existing if hash_ok else None,
            bhash=bhash,
            force=force,
        ):
            skipped_months.append(label)
            continue
        jobs.append((label, m_start, m_end))

    month_results: list[tuple[str, int, int]] = []
    src_s = str(src)
    out_s = str(out)
    bars_cache_s = str(bars_cache_dir) if bars_cache_dir is not None else None

    if workers <= 1 or len(jobs) <= 1:
        for label, m_start, m_end in jobs:
            got = _process_one_month(
                label,
                m_start,
                m_end,
                version=version,
                freq_n=freq_n,
                source_dir=src_s,
                out_dir=out_s,
                data_type=data_type,
                codes=codes,
                min_bar_coverage=min_bar_coverage,
                bars_cache_dir=bars_cache_s,
                use_bars_cache=use_bars_cache,
            )
            if got is not None:
                month_results.append(got)
    else:
        # spawn 避免 fork 后与 polars/OpenMP 线程死锁；manifest 仅主进程写
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futs = {
                pool.submit(
                    _process_one_month,
                    label,
                    m_start,
                    m_end,
                    version=version,
                    freq_n=freq_n,
                    source_dir=src_s,
                    out_dir=out_s,
                    data_type=data_type,
                    codes=codes,
                    min_bar_coverage=min_bar_coverage,
                    bars_cache_dir=bars_cache_s,
                    use_bars_cache=use_bars_cache,
                ): label
                for label, m_start, m_end in jobs
            }
            for fut in as_completed(futs):
                got = fut.result()
                if got is not None:
                    month_results.append(got)

    month_results.sort(key=lambda t: t[0])
    processed_months = [t[0] for t in month_results]
    rows_total = sum(t[1] for t in month_results)
    n_stocks = max((t[2] for t in month_results), default=0)

    coverage = _merge_coverage(
        existing if hash_ok else None,
        start,
        end,
        processed_months,
    )
    # overwrite 且 hash 不匹配时：coverage 仅本 build
    if existing is not None and existing.get("battery_hash") != bhash and overwrite:
        coverage = {
            "start": start,
            "end": end,
            "months": sorted(processed_months),
        }

    # 本 build 无新月但 hash 匹配（含全量跳过）：保留旧 coverage / rows / n_stocks
    if not processed_months and hash_ok:
        coverage = _merge_coverage(existing, start, end, [])
        rows_total = int(existing.get("rows_total") or 0)  # type: ignore[union-attr]
        n_stocks = int(existing.get("n_stocks_last_build") or 0)  # type: ignore[union-attr]
    elif processed_months and hash_ok and skipped_months:
        # 部分补算：rows/n_stocks 在旧总量上叠加新月（新月此前不在 coverage）
        old_cov = set((existing.get("coverage") or {}).get("months") or [])  # type: ignore[union-attr]
        only_new = [m for m in processed_months if m not in old_cov]
        if only_new and len(only_new) == len(processed_months):
            rows_total = int(existing.get("rows_total") or 0) + rows_total  # type: ignore[union-attr]
            n_stocks = max(n_stocks, int(existing.get("n_stocks_last_build") or 0))  # type: ignore[union-attr]
        elif not only_new:
            # 全是 force 式重算已覆盖月：用本 run 行数；若混有 skip 则保留旧 n 的 max
            n_stocks = max(n_stocks, int(existing.get("n_stocks_last_build") or 0))  # type: ignore[union-attr]
            if skipped_months:
                # 重算子集 + 跳过其余：无法精确加总，保留 max(本 run, 旧)
                rows_total = max(rows_total, int(existing.get("rows_total") or 0))  # type: ignore[union-attr]

    payload: dict[str, Any] = {
        "version": version,
        "freq": freq_n,
        "battery_hash": bhash,
        "features": [
            {
                "name": s.name,
                "freq": s.freq,
                "formula": s.formula,
                "description": s.description,
            }
            for s in specs
        ],
        "coverage": coverage,
        "rows_total": rows_total,
        "n_stocks_last_build": n_stocks,
        "source": "minute_1min",
        "bar_label_convention": BAR_LABEL_CONVENTION,
        "session_policy": "regular_only_drop_after_1500",
        "units": {"vol": "share", "amount": "cny"},
        "git_sha": _git_sha_or_none(),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_manifest(mpath, payload)

    return BuildReport(
        months=processed_months,
        rows=rows_total,
        n_stocks=n_stocks,
        manifest_path=str(mpath),
    )


__all__ = [
    "BuildReport",
    "build_intraday_features",
    "compute_day_panel",
    "read_manifest",
]

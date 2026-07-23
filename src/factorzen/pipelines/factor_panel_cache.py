"""评估复用单一 store parquet：缺失则算并落库；覆盖不足则增量补行。

每个因子有且只有一份 ``factors/<market>/<name>/factor.parquet``（4 列）。
评估一律以它为因子值来源；**永不**用评估窗子集覆盖写这份文件。
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

_LOG = logging.getLogger(__name__)

_PANEL_COLS = ("trade_date", "ts_code", "factor_value", "factor_clean")


def _date_key(s: str | None) -> str:
    if not s:
        return ""
    return str(s).replace("-", "").replace("/", "")[:8]


def _to_date(s: str) -> date:
    k = _date_key(s)
    return date(int(k[:4]), int(k[4:6]), int(k[6:8]))


def _to_iso(d: date | str) -> str:
    if isinstance(d, date):
        return d.isoformat()
    k = _date_key(d)
    return f"{k[:4]}-{k[4:6]}-{k[6:8]}"


def _to_ymd8(s: str | date) -> str:
    if isinstance(s, date):
        return s.strftime("%Y%m%d")
    return _date_key(s)


def _prev_calendar_day(s: str) -> str:
    d = _to_date(s) - timedelta(days=1)
    return d.isoformat()


def _next_calendar_day(s: str) -> str:
    d = _to_date(s) + timedelta(days=1)
    return d.isoformat()


def _min_date_str(a: str, b: str) -> str:
    return a if _date_key(a) <= _date_key(b) else b


def _max_date_str(a: str, b: str) -> str:
    return a if _date_key(a) >= _date_key(b) else b


def _cast_panel(df: pl.DataFrame) -> pl.DataFrame:
    """统一 4 列 schema。"""
    missing = [c for c in _PANEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"panel missing columns: {missing}")
    out = df.select(list(_PANEL_COLS))
    return out.with_columns(
        pl.col("trade_date").cast(pl.Date),
        pl.col("ts_code").cast(pl.Utf8),
        pl.col("factor_value").cast(pl.Float64),
        pl.col("factor_clean").cast(pl.Float64),
    )


def _filter_window(df: pl.DataFrame, start: str, end: str) -> pl.DataFrame:
    s = _to_date(start)
    e = _to_date(end)
    return df.filter(
        (pl.col("trade_date") >= s) & (pl.col("trade_date") <= e)
    )


def _coverage_from_meta_or_parquet(
    meta: dict[str, Any] | None,
    pq_path: Path,
) -> tuple[str | None, str | None]:
    """覆盖区间以 meta.materialization.start/end 为准；meta 缺则用 parquet min/max。"""
    mat = (meta or {}).get("materialization") if meta else None
    if isinstance(mat, dict):
        cs = mat.get("start")
        ce = mat.get("end")
        if cs and ce:
            return str(cs), str(ce)
    if not pq_path.exists():
        return None, None
    try:
        df = pl.read_parquet(pq_path, columns=["trade_date"])
        if df.is_empty():
            return None, None
        mn = df["trade_date"].min()
        mx = df["trade_date"].max()
        if mn is None or mx is None:
            return None, None
        if isinstance(mn, datetime):
            mn_d: date = mn.date()
        elif isinstance(mn, date):
            mn_d = mn
        else:
            mn_d = _to_date(str(mn))
        if isinstance(mx, datetime):
            mx_d: date = mx.date()
        elif isinstance(mx, date):
            mx_d = mx
        else:
            mx_d = _to_date(str(mx))
        return _to_iso(mn_d), _to_iso(mx_d)
    except Exception as exc:
        _LOG.warning("read coverage from parquet failed %s: %s", pq_path, exc)
        return None, None


def _expression_stale(meta: dict[str, Any] | None, factor: Any) -> bool:
    """双方都非空且不一致 → 过期。"""
    meta_expr = (meta or {}).get("expression") if meta else None
    fac_expr = getattr(factor, "expression", None)
    return bool(meta_expr and fac_expr and str(meta_expr) != str(fac_expr))


def _factor_source_hash(factor: Any) -> str | None:
    """因子实现源文件的 sha256（python 因子失效判据）。

    动态生成类（expression 因子）拿不到源文件 → None（由 expression 门管失效）。
    哈希整个源文件而非单个类，改 factor.py 里的辅助函数同样触发失效。
    """
    try:
        import hashlib
        import inspect

        src_file = inspect.getsourcefile(type(factor))
        if not src_file:
            return None
        return hashlib.sha256(Path(src_file).read_bytes()).hexdigest()
    except Exception:
        return None


def _source_stale(meta: dict[str, Any] | None, cur_hash: str | None) -> bool:
    """meta 记过 source_hash 且与当前实现不一致 → 过期（改 factor.py 后不吃旧面板）。"""
    mat = (meta or {}).get("materialization") if meta else None
    stored = mat.get("source_hash") if isinstance(mat, dict) else None
    return bool(stored and cur_hash and stored != cur_hash)


def _compute_segment(
    factor: Any,
    seg_start: str,
    seg_end: str,
    *,
    benchmark: str | None,
) -> pl.DataFrame:
    """算一段 [seg_start, seg_end] 的 4 列面板（预热头已滤掉）。"""
    from factorzen.core.data_ensure import ensure_data_for_daily_run
    from factorzen.daily.data.context import FactorDataContext
    from factorzen.discovery.factor_store import finalize_factor_panel

    is_qlib = (
        getattr(factor, "category", "") == "qlib"
        or str(getattr(factor, "name", "")).startswith("qlib_")
    )
    ensure_data_for_daily_run(
        required_data=list(getattr(factor, "required_data", None) or ["daily"]),
        start=_to_ymd8(seg_start),
        end=_to_ymd8(seg_end),
        universe="all_a",
        benchmark=benchmark,
        needs_size_neutralization=False,
        is_qlib_factor=is_qlib,
    )
    ctx = FactorDataContext(
        start=_to_ymd8(seg_start),
        end=_to_ymd8(seg_end),
        required_data=list(getattr(factor, "required_data", None) or ["daily"]),
        lookback_days=int(getattr(factor, "lookback_days", 0) or 0),
        universe=None,
        snapshot_mode="daily",
    )
    panel = factor.compute(ctx)
    finalized = finalize_factor_panel(panel)
    # 过滤只保留段内行（预热头不得混入）
    return _filter_window(_cast_panel(finalized), seg_start, seg_end)


def _atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    import contextlib

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.write_parquet(tmp)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()
        raise


def is_daily_frequency(args: Any, factor: Any) -> bool:
    """面板缓存只支持日频口径；weekly/monthly（args 或因子自身）必须直算。

    daily_single 与 generate_report 共用此单点判定（双路径登记簿）。
    """
    args_freq = (getattr(args, "frequency", None) or "daily") == "daily"
    factor_freq = (getattr(factor, "frequency", None) or "daily") == "daily"
    return args_freq and factor_freq


def ensure_factor_store_panel(
    factor: Any,
    start: str,
    end: str,
    *,
    market: str = "ashare",
    root: str | None = None,
    benchmark: str | None = None,
) -> pl.DataFrame | None:
    """确保 store 面板覆盖请求窗；返回完整 4 列面板（非切片）。

    失败返回 None（调用方回落直算）；写盘用临时文件 + ``os.replace`` 原子替换。
    """
    # discovery import 一律放函数内（pipelines→discovery 反向依赖纪律）
    from factorzen.discovery import factor_store as fs
    from factorzen.discovery.factor_store import (
        DEFAULT_ROOT,
        FACTOR_PANEL_COLUMNS,
        STORE_MATERIALIZE_START,
        STORE_MATERIALIZE_UNIVERSE,
        asset_dir,
        store_materialize_end,
    )

    try:
        # 数据装配链（ensure_data/FactorDataContext）目前是 A 股专属；
        # 其他市场直接回落直算，绝不用 A 股数据算别的市场。
        if market != "ashare":
            _LOG.info(
                "ensure_factor_store_panel: market=%s 暂不支持缓存，回落直算", market
            )
            return None

        store_root = root if root is not None else DEFAULT_ROOT
        name = str(getattr(factor, "name", "") or "unknown")
        d = asset_dir(market, name, root=store_root)
        pq_path = d / "factor.parquet"
        meta_path = d / "meta.json"
        meta = fs._read_json(meta_path)

        mat_end = store_materialize_end()
        target_start = _min_date_str(start, STORE_MATERIALIZE_START)
        eff_end = (
            end if _date_key(end) <= _date_key(mat_end) else mat_end
        )

        source_hash = _factor_source_hash(factor)
        stale = _expression_stale(meta, factor) or _source_stale(meta, source_hash)
        cover_start, cover_end = _coverage_from_meta_or_parquet(meta, pq_path)
        has_panel = pq_path.exists() and not stale

        segments: list[tuple[str, str]] = []
        existing: pl.DataFrame | None = None
        head_ran = False

        if not has_panel or cover_start is None or cover_end is None:
            # 不存在或过期 → 一次性算全窗
            segments.append((target_start, mat_end))
            existing = None
            cover_start = None
            cover_end = None
        else:
            try:
                existing = _cast_panel(pl.read_parquet(pq_path))
            except Exception as exc:
                _LOG.warning(
                    "ensure_factor_store_panel: read existing failed %s: %s: %s",
                    pq_path,
                    type(exc).__name__,
                    exc,
                )
                segments.append((target_start, mat_end))
                existing = None
                cover_start = None
                cover_end = None
            else:
                # 请求窗完全在覆盖内 → 直接读返
                if (
                    _date_key(cover_start) <= _date_key(start)
                    and _date_key(cover_end) >= _date_key(eff_end)
                ):
                    _LOG.info(
                        "factor panel cache HIT %s/%s cover=%s~%s",
                        market,
                        name,
                        cover_start,
                        cover_end,
                    )
                    return existing

                # 补头：下界必须与 meta 将声称的 attempt_lo 同源（target_start），
                # 否则声称 [2016, start) 已覆盖但从未计算 → 后续评估静默缺头
                if _date_key(cover_start) > _date_key(start):
                    head_end = _prev_calendar_day(cover_start)
                    if _date_key(target_start) <= _date_key(head_end):
                        segments.append((target_start, head_end))
                        head_ran = True
                # 补尾
                if _date_key(cover_end) < _date_key(eff_end):
                    tail_start = _next_calendar_day(cover_end)
                    if _date_key(tail_start) <= _date_key(eff_end):
                        segments.append((tail_start, eff_end))

        if not segments and existing is not None:
            return existing

        new_parts: list[pl.DataFrame] = []
        for seg_s, seg_e in segments:
            if _date_key(seg_s) > _date_key(seg_e):
                continue
            _LOG.info(
                "factor panel compute segment %s/%s %s~%s",
                market,
                name,
                seg_s,
                seg_e,
            )
            part = _compute_segment(
                factor, seg_s, seg_e, benchmark=benchmark
            )
            if part is not None and not part.is_empty():
                new_parts.append(part)

        frames: list[pl.DataFrame] = []
        if existing is not None and not existing.is_empty():
            frames.append(existing)
        frames.extend(new_parts)

        if not frames:
            _LOG.warning(
                "ensure_factor_store_panel: empty result for %s/%s", market, name
            )
            return None

        merged = pl.concat(frames, how="vertical_relaxed")
        merged = _cast_panel(merged)
        # 旧行优先
        merged = (
            merged.unique(subset=["trade_date", "ts_code"], keep="first")
            .sort(["trade_date", "ts_code"])
        )

        # 更新 materialization 标记（只声称本轮真正 ensure 过的下界：
        # 头段没算就保留 cover_start，绝不 over-claim）
        if cover_start is None:
            # 全窗重算
            attempt_lo = target_start
            attempt_hi = mat_end
        else:
            attempt_lo = (
                _min_date_str(cover_start, target_start) if head_ran else cover_start
            )
            attempt_hi = _max_date_str(cover_end or eff_end, eff_end)

        expr = getattr(factor, "expression", None)
        if expr is None and meta:
            expr = meta.get("expression")

        materialization = {
            "start": _to_iso(attempt_lo),
            "end": _to_iso(attempt_hi),
            "universe": STORE_MATERIALIZE_UNIVERSE,
            "git_sha": fs._git_sha(),
            "n_rows": int(merged.height),
            "generated_at": fs._utc_now_iso(),
            "expression": expr,
            "columns": list(FACTOR_PANEL_COLUMNS),
            "source_hash": source_hash,
        }

        d.mkdir(parents=True, exist_ok=True)
        _atomic_write_parquet(merged, pq_path)

        new_meta = meta
        if new_meta is None:
            # 首创 meta 必须带 kind/expression：materialize_assets 靠 kind 选
            # fallback expression（缺了永久 errors），server 手写因子列表靠
            # kind=python + expression 非空才显示
            new_meta = {"name": name, "market": market}
            if expr is not None:
                new_meta["kind"] = "expression"
            else:
                from factorzen.discovery.factor_library import python_identity

                new_meta["kind"] = "python"
                new_meta["expression"] = python_identity(name)
        new_meta["materialization"] = materialization
        if expr is not None:
            new_meta["expression"] = expr
        fs._write_json(meta_path, new_meta)

        _LOG.info(
            "factor panel ensured %s/%s n_rows=%s cover=%s~%s path=%s",
            market,
            name,
            merged.height,
            materialization["start"],
            materialization["end"],
            pq_path,
        )
        return merged
    except Exception as exc:
        _LOG.warning(
            "ensure_factor_store_panel failed name=%s: %s: %s",
            getattr(factor, "name", "?"),
            type(exc).__name__,
            exc,
        )
        return None

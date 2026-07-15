"""日内特征面板 attach：把 ``i_*`` 日频叶子 join 进日线帧。

与 ``attach_flows`` / ``attach_fundamentals`` 同约定：
挖掘（``prepare_mining_daily``）与物化（``ExpressionFactor.compute``）共用本函数，
保证同一因子两条路逐值一致。

面板路径：``INTRADAY_FEATURES_DIR/{version}/{freq}/year=…/month=…/data.parquet``
（由 ``fz data intraday-features build`` 物化）。

叶子名单单源：``core.feature_schema.INTRADAY_FEATURES``（core 不依赖 intraday 层）。
"""
from __future__ import annotations

import warnings
from datetime import date, datetime
from typing import Any

import polars as pl

from factorzen.config.settings import INTRADAY_FEATURES_DIR
from factorzen.core.feature_schema import INTRADAY_FEATURES

# 稳定顺序（集合本身无序；join/补 null 用排序列表保证列序可复现）
_INTRADAY_COLS: list[str] = sorted(INTRADAY_FEATURES)

_BUILD_HINT = "fz data intraday-features build"


def attach_intraday(
    daily: pl.DataFrame,
    *,
    freq: str = "5min",
    version: str = "v1",
    injected: pl.DataFrame | None = None,
    require: bool = False,
    out_meta: dict | None = None,
) -> pl.DataFrame:
    """把日内特征面板按 ``(trade_date, ts_code)`` left-join 进日线帧。

    Args:
        daily: 日频帧，须含 ``trade_date`` / ``ts_code``。
        freq / version: 面板分区键（默认 ``v1@5min``）。
        injected: 测试注入面板（优先于读盘）；列至少含 ``trade_date``/``ts_code`` 与部分 ``i_*``。
        require: ``True`` 时面板缺失/为空 → ``ValueError``；``False`` 时补全 null 列并 warn。
        out_meta: 非 None 时回填 ``intraday_panel`` 溯源字典。

    Returns:
        含全部 17 个 ``i_*`` 列的日线帧（缺失列为 null）。
    """
    if daily.is_empty() or "trade_date" not in daily.columns:
        if out_meta is not None:
            out_meta["intraday_panel"] = _panel_meta(
                version, freq, battery_hash=None,
                coverage_start=None, coverage_end=None,
            )
        return _ensure_intraday_cols(daily)

    panel = injected
    if panel is None:
        panel = _load_panel(daily, freq=freq, version=version)

    if panel is None or panel.is_empty():
        if require:
            raise ValueError(
                f"日内特征面板缺失或为空（version={version!r} freq={freq!r}）；"
                f"请先运行: {_BUILD_HINT} --start ... --end ... "
                f"（或检查 {INTRADAY_FEATURES_DIR}/{version}/{freq}）"
            )
        warnings.warn(
            f"日内特征面板缺失或为空（version={version} freq={freq}），"
            f"i_* 列补 null；leaf_health 将摘除零覆盖叶子。"
            f"完整面板请运行: {_BUILD_HINT}",
            stacklevel=2,
        )
        if out_meta is not None:
            bhash, cov_s, cov_e = _read_manifest_fields(version, freq)
            out_meta["intraday_panel"] = _panel_meta(
                version, freq, battery_hash=bhash,
                coverage_start=cov_s, coverage_end=cov_e,
            )
        return _ensure_intraday_cols(daily)

    have = [c for c in _INTRADAY_COLS if c in panel.columns]
    sel = panel.select(["trade_date", "ts_code", *have])
    sel = _align_trade_date(sel, daily)
    # 已存在同名列时先 drop 再 join，避免 suffix 污染
    drop_cols = [c for c in have if c in daily.columns]
    if drop_cols:
        daily = daily.drop(drop_cols)
    daily = daily.join(sel, on=["trade_date", "ts_code"], how="left")
    daily = _ensure_intraday_cols(daily)

    if out_meta is not None:
        bhash, cov_s, cov_e = _read_manifest_fields(version, freq)
        # 注入路径无 manifest 时用面板日期极值作 coverage
        if cov_s is None or cov_e is None:
            cov_s2, cov_e2 = _frame_date_bounds(panel)
            cov_s = cov_s if cov_s is not None else cov_s2
            cov_e = cov_e if cov_e is not None else cov_e2
        out_meta["intraday_panel"] = _panel_meta(
            version, freq, battery_hash=bhash,
            coverage_start=cov_s, coverage_end=cov_e,
        )
    return daily


def _load_panel(
    daily: pl.DataFrame, *, freq: str, version: str,
) -> pl.DataFrame | None:
    """从 ``INTRADAY_FEATURES_DIR`` 按日频帧日期窗加载面板；失败返回 None。"""
    start_s, end_s = _frame_date_bounds(daily)
    if start_s is None or end_s is None:
        return None
    data_type = f"{version}/{freq}"
    try:
        from factorzen.core.storage import load_parquet

        lf = load_parquet(
            data_type,
            start=start_s,
            end=end_s,
            date_col="trade_date",
            base_dir=INTRADAY_FEATURES_DIR,
        )
        return lf.collect()
    except Exception:
        return None


def _frame_date_bounds(df: pl.DataFrame) -> tuple[str | None, str | None]:
    """帧内最早/最晚 trade_date → ``YYYYMMDD``；空帧 → (None, None)。"""
    if df.is_empty() or "trade_date" not in df.columns:
        return None, None
    col = df["trade_date"]
    try:
        mn, mx = col.min(), col.max()
    except Exception:
        return None, None
    return _to_yyyymmdd(mn), _to_yyyymmdd(mx)


def _to_yyyymmdd(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.strftime("%Y%m%d")
    if isinstance(v, date):
        return v.strftime("%Y%m%d")
    s = str(v).strip().replace("-", "")
    if len(s) >= 8 and s[:8].isdigit():
        return s[:8]
    return None


def _align_trade_date(sel: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame:
    """把面板 trade_date 类型对齐到 daily（Date / Utf8 双向兼容）。"""
    if "trade_date" not in sel.columns or "trade_date" not in daily.columns:
        return sel
    src_dt = sel["trade_date"].dtype
    tgt_dt = daily["trade_date"].dtype
    if src_dt == tgt_dt:
        return sel
    if tgt_dt == pl.Date and src_dt in (pl.Utf8, pl.String):
        return sel.with_columns(
            pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d", strict=False)
        )
    if tgt_dt in (pl.Utf8, pl.String) and src_dt == pl.Date:
        return sel.with_columns(pl.col("trade_date").dt.strftime("%Y%m%d"))
    # 其他形态尽力转字符串再对齐到 Date
    if tgt_dt == pl.Date:
        return sel.with_columns(
            pl.col("trade_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)
        )
    return sel


def _ensure_intraday_cols(daily: pl.DataFrame) -> pl.DataFrame:
    """补全缺失的 17 个 i_* 列为 Float64 null。"""
    missing = [c for c in _INTRADAY_COLS if c not in daily.columns]
    if missing:
        daily = daily.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(c) for c in missing]
        )
    return daily


def _read_manifest_fields(
    version: str, freq: str,
) -> tuple[str | None, str | None, str | None]:
    """读 manifest 的 battery_hash / coverage.start / coverage.end；读不到 → 全 None。

    直接读 JSON（不 import intraday.features），避免 daily→intraday 依赖环
    （架构守卫 test_top_level_package_dependency_graph_is_acyclic）；
    路径与 intraday.features.engine.read_manifest 落盘口径一致。
    """
    import json

    manifest_path = INTRADAY_FEATURES_DIR / version / freq / "manifest.json"
    try:
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None, None
    if not isinstance(man, dict):
        return None, None, None
    bhash = man.get("battery_hash")
    cov = man.get("coverage") or {}
    return (
        bhash if isinstance(bhash, str) else None,
        cov.get("start"),
        cov.get("end"),
    )


def _panel_meta(
    version: str,
    freq: str,
    *,
    battery_hash: str | None,
    coverage_start: str | None,
    coverage_end: str | None,
) -> dict[str, Any]:
    return {
        "version": version,
        "freq": freq,
        "battery_hash": battery_hash,
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
    }


__all__ = ["attach_intraday"]

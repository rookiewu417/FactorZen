"""A 股分钟 bar 口径审计 CLI：抽样交易日跑 census + reconcile + 标签推断。

用法::

    pixi run -- python tools/audit_minute_bars.py --start 20240101 --end 20240331
    pixi run -- python tools/audit_minute_bars.py --start 20240101 --end 20240131 --json

退出码：0=ok；1=error；2=仅 warning（无 error）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.config.settings import DATA_RAW, REPORTS_DIR
from factorzen.core.calendar import get_trade_dates
from factorzen.core.storage import load_parquet
from factorzen.intraday.audit import (
    infer_label_convention,
    reconcile_with_daily,
    timestamp_census,
)

_DEFAULT_CODES = ("000001.SZ", "600000.SH", "688981.SH", "300750.SZ")
_DEFAULT_OUT = REPORTS_DIR / "minute_audit"

# 倍率容差 1%
_VOL_TARGET = 100.0
_AMT_TARGET = 1000.0
_MULT_TOL = 0.01
_MATCH_WARN = 0.01  # open/close 不匹配率 >1% → warning


def _sample_trade_dates(start: str, end: str, n: int) -> list[date]:
    dates = get_trade_dates(start, end)
    if not dates:
        return []
    if len(dates) <= n:
        return dates
    if n == 1:
        return [dates[len(dates) // 2]]
    idxs = [round(i * (len(dates) - 1) / (n - 1)) for i in range(n)]
    # 去重保序
    seen: set[int] = set()
    out: list[date] = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(dates[i])
    return out


def _load_day(
    day: date,
    codes: list[str],
    *,
    base_dir: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    day_str = day.strftime("%Y%m%d")
    raw = DATA_RAW if base_dir is None else base_dir

    minute = (
        load_parquet(
            "minute_1min",
            start=day_str,
            end=day_str,
            date_col="trade_time",
            base_dir=raw,
        )
        .filter(pl.col("ts_code").is_in(codes))
        .collect()
    )
    daily = (
        load_parquet(
            "daily",
            start=day_str,
            end=day_str,
            date_col="trade_date",
            base_dir=raw,
        )
        .filter(pl.col("ts_code").is_in(codes))
        .select(
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "amount",
        )
        .collect()
    )
    # 确保 trade_date 为 Date
    if not daily.is_empty() and daily.schema["trade_date"] != pl.Date:
        daily = daily.with_columns(pl.col("trade_date").cast(pl.Date))
    return minute, daily


def _judge(
    labels: list[dict[str, object]],
    recon: pl.DataFrame,
) -> tuple[str, list[str], list[str]]:
    """返回 (status, errors, warnings)。"""
    errors: list[str] = []
    warnings: list[str] = []

    for lab in labels:
        if lab.get("label_convention") != "end":
            errors.append(
                f"label_convention={lab.get('label_convention')!r} != 'end' "
                f"(first={lab.get('first_time')}, last={lab.get('last_time')})"
            )

    if not recon.is_empty():
        vol_m = recon["vol_multiplier"].drop_nulls()
        amt_m = recon["amount_multiplier"].drop_nulls()
        if vol_m.len() > 0:
            med_v = float(vol_m.median())  # type: ignore[arg-type]
            if abs(med_v - _VOL_TARGET) / _VOL_TARGET > _MULT_TOL:
                errors.append(
                    f"vol_multiplier 中位 {med_v:.4g} 偏离 100 超 1%"
                )
        if amt_m.len() > 0:
            med_a = float(amt_m.median())  # type: ignore[arg-type]
            if abs(med_a - _AMT_TARGET) / _AMT_TARGET > _MULT_TOL:
                errors.append(
                    f"amount_multiplier 中位 {med_a:.4g} 偏离 1000 超 1%"
                )

        n = recon.height
        if n > 0:
            open_bad = recon.filter(~pl.col("open_match").fill_null(False)).height
            close_bad = recon.filter(~pl.col("close_match").fill_null(False)).height
            if open_bad / n > _MATCH_WARN:
                warnings.append(f"open 不匹配率 {open_bad / n:.2%} > 1%")
            if close_bad / n > _MATCH_WARN:
                warnings.append(f"close 不匹配率 {close_bad / n:.2%} > 1%")

    if errors:
        return "error", errors, warnings
    if warnings:
        return "warning", errors, warnings
    return "ok", errors, warnings


def run_audit(
    start: str,
    end: str,
    *,
    sample_days: int = 8,
    codes: list[str] | None = None,
) -> dict[str, Any]:
    """在窗口内均匀抽样交易日，聚合 census + reconcile + label。"""
    code_list = list(codes) if codes else list(_DEFAULT_CODES)
    sample = _sample_trade_dates(start, end, sample_days)

    day_reports: list[dict[str, Any]] = []
    all_recon: list[pl.DataFrame] = []
    labels: list[dict[str, object]] = []

    for day in sample:
        minute, daily = _load_day(day, code_list)
        census = timestamp_census(minute)
        recon = reconcile_with_daily(minute, daily)
        label = infer_label_convention(minute)
        labels.append(label)
        if not recon.is_empty():
            all_recon.append(recon)

        day_reports.append(
            {
                "trade_date": day.isoformat(),
                "n_minute_rows": minute.height,
                "n_daily_rows": daily.height,
                "label": label,
                "census": census.to_dicts() if not census.is_empty() else [],
                "reconcile": recon.to_dicts() if not recon.is_empty() else [],
            }
        )

    recon_all = (
        pl.concat(all_recon, how="vertical_relaxed") if all_recon else pl.DataFrame()
    )
    status, errors, warnings = _judge(labels, recon_all)

    summary: dict[str, Any] = {
        "vol_multiplier_median": None,
        "amount_multiplier_median": None,
        "open_match_rate": None,
        "close_match_rate": None,
        "n_reconcile_rows": recon_all.height if not recon_all.is_empty() else 0,
    }
    if not recon_all.is_empty():
        summary["vol_multiplier_median"] = float(
            recon_all["vol_multiplier"].drop_nulls().median()  # type: ignore[arg-type]
        )
        summary["amount_multiplier_median"] = float(
            recon_all["amount_multiplier"].drop_nulls().median()  # type: ignore[arg-type]
        )
        n = recon_all.height
        summary["open_match_rate"] = (
            recon_all.filter(pl.col("open_match").fill_null(False)).height / n
        )
        summary["close_match_rate"] = (
            recon_all.filter(pl.col("close_match").fill_null(False)).height / n
        )

    return {
        "start": start,
        "end": end,
        "sample_days": [d.isoformat() for d in sample],
        "codes": code_list,
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
        "days": day_reports,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A 股分钟 bar 口径审计")
    p.add_argument("--start", required=True, help="起始日期 YYYYMMDD")
    p.add_argument("--end", required=True, help="截止日期 YYYYMMDD")
    p.add_argument("--sample-days", type=int, default=8, help="均匀抽样交易日数")
    p.add_argument(
        "--codes",
        default=",".join(_DEFAULT_CODES),
        help="逗号分隔 ts_code 列表",
    )
    p.add_argument("--json", action="store_true", help="同时向 stdout 打印 JSON")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT,
        help="报告输出目录",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    codes = [c.strip() for c in args.codes.split(",") if c.strip()]

    report = run_audit(
        args.start,
        args.end,
        sample_days=args.sample_days,
        codes=codes,
    )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"minute_bars_audit_{ts}.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    status = report["status"]
    print("=" * 60)
    print("FactorZen 分钟 bar 口径审计")
    print("=" * 60)
    print(f"窗口: {args.start} ~ {args.end}")
    print(f"抽样日: {', '.join(report['sample_days'])}")
    print(f"标的: {', '.join(codes)}")
    s = report["summary"]
    print(
        f"vol× 中位={s['vol_multiplier_median']}  "
        f"amount× 中位={s['amount_multiplier_median']}  "
        f"open_match={s['open_match_rate']}  close_match={s['close_match_rate']}"
    )
    for e in report["errors"]:
        print(f"  · error:   {e}")
    for w in report["warnings"]:
        print(f"  · warning: {w}")
    print(f"报告: {out_path}")
    print("=" * 60)
    print(f"总体: {status.upper()}")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    return {"ok": 0, "error": 1, "warning": 2}[status]


if __name__ == "__main__":
    sys.exit(main())

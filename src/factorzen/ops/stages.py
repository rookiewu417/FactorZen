"""ops 每日链路的内置阶段。

每个 stage 签名统一为 ``(cfg, as_of, ctx) -> dict``:返回该阶段摘要写入 ctx,
失败抛 OpsStageError(或让底层异常如 DataEnsureError 冒泡,由 runner 统一捕获)。
阶段本身无状态,幂等由 runner 的 OpsState + SessionStore.has_date 保证。
"""
from __future__ import annotations

import glob
import subprocess
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from jinja2 import Environment, FileSystemLoader

from factorzen.core.data_audit import build_raw_data_audit
from factorzen.core.data_ensure import (
    ensure_adj_factor,
    ensure_daily,
    ensure_daily_basic,
    ensure_index_daily,
)
from factorzen.core.loader import fetch_daily, fetch_trade_cal
from factorzen.core.universe import get_universe
from factorzen.execution.drivers import run_daily_step
from factorzen.execution.store import SessionStore
from factorzen.ops.config import OpsConfig

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_ENV = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=True)


class OpsStageError(RuntimeError):
    """某阶段的业务失败(区别于底层异常)。runner 捕获后标 failed 并告警。"""

    def __init__(self, stage: str, msg: str) -> None:
        super().__init__(f"[{stage}] {msg}")
        self.stage = stage
        self.msg = msg


def _ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _window(cfg: OpsConfig, as_of: date) -> tuple[str, str]:
    """行情窗口 [as_of - lookback_days, as_of],YYYYMMDD(含 ADV 回看余量)。"""
    return _ymd(as_of - timedelta(days=cfg.lookback_days)), _ymd(as_of)


def stage_guard(cfg: OpsConfig, as_of: date, ctx: dict[str, Any]) -> dict[str, Any]:
    """交易日守卫:非交易日则整链短路成功退出。"""
    d = _ymd(as_of)
    cal = fetch_trade_cal(d, d)
    trading = cal.height > 0 and cal.filter(pl.col("is_open") == 1).height > 0
    return {"trading_day": trading}


def stage_data(cfg: OpsConfig, as_of: date, ctx: dict[str, Any]) -> dict[str, Any]:
    """数据补齐:日线/复权/估值/基准指数,strict=True(缺口补不齐则异常冒泡)。"""
    start, end = _window(cfg, as_of)
    return {
        "daily": ensure_daily(start, end).ok,
        "adj_factor": ensure_adj_factor(start, end).ok,
        "daily_basic": ensure_daily_basic(start, end).ok,
        "index_daily": ensure_index_daily(cfg.benchmark, start, end).ok,
    }


def stage_audit(cfg: OpsConfig, as_of: date, ctx: dict[str, Any]) -> dict[str, Any]:
    """数据质量门:按 audit_fail_on 级别拦截 error(或 warning)。"""
    start, end = _window(cfg, as_of)
    statuses: dict[str, Any] = {}
    problems: list[str] = []
    for dtype in cfg.audit_types:
        report = build_raw_data_audit(data_type=dtype, start=start, end=end)
        status = report.get("status", "error")
        statuses[dtype] = status
        blocked = status == "error" or (cfg.audit_fail_on == "warning" and status == "warning")
        if blocked:
            detail = report.get("errors") or report.get("warnings") or []
            problems.append(f"{dtype}:{status} {detail}")
    if problems:
        raise OpsStageError("audit", "数据质量门未通过: " + "; ".join(problems))
    return statuses


def stage_signal(cfg: OpsConfig, as_of: date, ctx: dict[str, Any]) -> dict[str, Any]:
    """信号生成:执行外部命令(None=跳过,直接消费已有 portfolio 产物)。"""
    if cfg.signal_command is None:
        return {"skipped": True}
    try:
        proc = subprocess.run(
            cfg.signal_command,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "")[-200:]
        raise OpsStageError("signal", f"信号命令失败(exit {exc.returncode}): {tail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise OpsStageError("signal", "信号命令超时(>3600s)") from exc
    return {"skipped": False, "stdout_tail": (proc.stdout or "")[-200:]}


def stage_live_step(cfg: OpsConfig, as_of: date, ctx: dict[str, Any]) -> dict[str, Any]:
    """纸面执行:拉行情→(可选)universe 过滤→按 glob 收组合产物→推进一个交易日。"""
    start, end = _window(cfg, as_of)
    daily = fetch_daily(start, end)
    if cfg.universe:
        stocks = get_universe(end, cfg.universe)
        daily = daily.filter(pl.col("ts_code").is_in(stocks["ts_code"].to_list()))
    run_dirs = sorted(glob.glob(cfg.portfolio_run_dirs_glob))
    if not run_dirs:
        raise OpsStageError("live_step", f"无匹配组合产物: {cfg.portfolio_run_dirs_glob}")
    config = {"initial_cash": cfg.initial_cash, "slippage_bps": cfg.slippage_bps}
    if not (Path(cfg.session_dir) / "manifest.json").exists():
        SessionStore(cfg.session_dir).init(config)
    return run_daily_step(cfg.session_dir, as_of, run_dirs, daily, config=config)


def stage_report(cfg: OpsConfig, as_of: date, ctx: dict[str, Any]) -> dict[str, Any]:
    """摘要:从会话账本取当日 NAV/成交,产出文本供通知与发布。"""
    records = SessionStore(cfg.session_dir).ledger_records()
    d_iso = as_of.isoformat()
    today = next((r for r in records if r["as_of_date"] == d_iso), None)
    if today is None:
        return {
            "summary_text": f"交易日 {d_iso}: 无执行记录(空目标/跳过)",
            "nav_after": None,
            "n_fills": 0,
        }
    nav_after = today["nav_after"]
    base = records[0]["nav_before"] if records else nav_after
    ret = (nav_after / base - 1.0) if base else 0.0
    n_orders = len(today["orders"])
    n_fills = len(today["fills"])
    text = (
        f"交易日 {d_iso}\n"
        f"NAV: {nav_after:,.0f} (期间 {ret:+.2%})\n"
        f"订单 {n_orders} 笔 · 成交 {n_fills} 笔"
    )
    return {"summary_text": text, "nav_after": nav_after, "n_fills": n_fills}


def _max_drawdown(navs: list[float]) -> float:
    """净值序列的最大回撤(<=0)。"""
    peak = float("-inf")
    mdd = 0.0
    for v in navs:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1.0)
    return mdd


def render_track_record(nav_df: pl.DataFrame, as_of: date) -> str:
    """把净值序列渲染为 track record 静态页 HTML。"""
    points = (
        [
            {"date": r["as_of_date"], "nav": float(r["nav_after"])}
            for r in nav_df.iter_rows(named=True)
        ]
        if nav_df.height
        else []
    )
    navs = [p["nav"] for p in points]
    latest = navs[-1] if navs else None
    first = navs[0] if navs else None
    total_return = (latest / first - 1.0) if (latest and first) else 0.0
    return _ENV.get_template("track_record.html").render(
        points=points,
        latest_nav=latest,
        total_return=total_return,
        max_drawdown=_max_drawdown(navs),
        n_days=len(points),
        as_of=as_of.isoformat(),
    )


def stage_publish(cfg: OpsConfig, as_of: date, ctx: dict[str, Any]) -> dict[str, Any]:
    """发布 track record 静态页(publish_enabled 才跑,否则跳过)。"""
    if not cfg.publish_enabled:
        return {"skipped": True}
    nav = SessionStore(cfg.session_dir).nav_frame()
    site = Path(cfg.publish_site_dir)
    site.mkdir(parents=True, exist_ok=True)
    out_path = site / "index.html"
    out_path.write_text(render_track_record(nav, as_of), encoding="utf-8")
    return {"skipped": False, "path": str(out_path)}

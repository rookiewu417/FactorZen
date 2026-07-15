"""ops 每日链路编排。

固定阶段序列,逐阶段幂等执行:已完成则跳过(重入),失败则标记+告警并返回 1,
全部完成则推送日报返回 0。guard 非交易日短路成功退出(且不落 done,重跑重新判断)。
调度完全外置(systemd/cron),本函数是无副作用的可重入编排核心。
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

from factorzen.core.logger import get_logger, setup_logging
from factorzen.ops.config import OpsConfig
from factorzen.ops.notify import Notifier, build_notifier
from factorzen.ops.stages import (
    stage_audit,
    stage_data,
    stage_guard,
    stage_intraday_features,
    stage_live_step,
    stage_publish,
    stage_report,
    stage_signal,
)
from factorzen.ops.state import OpsState

StageFn = Callable[[OpsConfig, date, dict[str, Any]], dict[str, Any]]

STAGES: list[tuple[str, StageFn]] = [
    ("guard", stage_guard),
    ("data", stage_data),
    ("audit", stage_audit),
    ("intraday_features", stage_intraday_features),
    ("signal", stage_signal),
    ("live_step", stage_live_step),
    ("report", stage_report),
    ("publish", stage_publish),
]


def run_ops_daily(
    cfg: OpsConfig, as_of: date, *, notifier: Notifier | None = None
) -> int:
    """执行一个交易日的无人值守链路。返回 0=成功/非交易日,1=某阶段失败。"""
    setup_logging()
    log = get_logger("factorzen.ops.runner")
    notifier = notifier or build_notifier(cfg)
    state = OpsState(cfg.state_dir, as_of)
    d_iso = as_of.isoformat()
    ctx: dict[str, Any] = {}

    for name, fn in STAGES:
        if state.is_done(name):
            log.info(f"[ops] 跳过已完成阶段: {name}")
            continue
        try:
            result = fn(cfg, as_of, ctx)
        except Exception as exc:  # 任意阶段异常都要落 state + 告警,不使全链路崩溃
            state.mark_failed(name, str(exc))
            log.error(f"[ops] 阶段 {name} 失败: {exc}")
            notifier.send(f"[FactorZen ops] {name} 失败 {d_iso}", str(exc), level="error")
            return 1
        ctx[name] = result
        # 非交易日短路:不落 done,重跑仍重新判断
        if name == "guard" and not result.get("trading_day", True):
            log.info(f"[ops] {d_iso} 非交易日,短路退出")
            return 0
        state.mark_done(name, detail=str(result)[:200])

    summary = ctx.get("report", {}).get("summary_text", "每日链路完成")
    notifier.send(f"[FactorZen ops] 日报 {d_iso}", summary, level="info")
    log.info(f"[ops] {d_iso} 全链路完成")
    return 0

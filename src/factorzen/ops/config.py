"""无人值守运营配置模型与加载。

OpsConfig 描述一次 `fz ops daily` 运行所需的全部参数:执行会话、组合产物来源、
数据窗口、质量门级别、通知方式与发布选项。配置经 YAML 驱动,``extra='forbid'``
保证拼写错误的字段被拒绝而非静默忽略(运维配置最忌静默失配)。
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class OpsConfig(BaseModel):
    """无人值守每日链路的配置。"""

    model_config = ConfigDict(extra="forbid")

    # ── 必填:执行会话与信号来源 ──
    session_dir: str
    portfolio_run_dirs_glob: str

    # ── 信号生成(可选外部命令;None=跳过,直接消费已有 portfolio 产物)──
    signal_command: list[str] | None = None

    # ── 数据窗口与基准 ──
    lookback_days: int = Field(90, gt=0)  # 零/负窗口无法取数,配置层直接拒绝
    benchmark: str = "000300.SH"
    universe: str | None = None

    # ── 数据质量门 ──
    audit_types: list[str] = Field(default_factory=lambda: ["daily", "daily_basic"])
    audit_fail_on: Literal["error", "warning"] = "error"

    # ── 纸面执行参数 ──
    initial_cash: float = Field(1_000_000.0, gt=0)  # 零/负本金无法纸面执行
    slippage_bps: float = Field(0.0, ge=0)  # 负滑点无经济意义(0 允许=零滑点对照)

    # ── 通知 ──
    notify_kind: Literal["webhook", "stdout"] = "stdout"
    notify_url_env: str = "FACTORZEN_NOTIFY_WEBHOOK"

    # ── 发布(track record 静态页)──
    publish_enabled: bool = False
    publish_site_dir: str = "workspace/ops/site"

    # ── 幂等状态 ──
    state_dir: str = "workspace/ops/state"


def load_ops_config(path: str | Path) -> OpsConfig:
    """从 YAML 文件加载并校验 OpsConfig。

    Args:
        path: YAML 配置文件路径(str 或 Path)。

    Returns:
        校验通过的 OpsConfig。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: YAML 顶层非映射,或字段校验失败(pydantic ValidationError 继承 ValueError)。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ops 配置文件不存在: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"ops 配置必须是 YAML 映射,得到 {type(data).__name__}: {p}")
    return OpsConfig(**data)

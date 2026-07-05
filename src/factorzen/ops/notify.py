"""ops 通知层。

可插拔 Notifier:把每日链路的日报与告警推给外部渠道。WebhookNotifier 零依赖
(urllib,兼容企业微信机器人/PushPlus 等),失败重试后**返回 False 而不抛异常**——
通知只是旁路,绝不能因推送失败炸掉主链路。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable

from factorzen.ops.config import OpsConfig


@runtime_checkable
class Notifier(Protocol):
    """通知发送接口。send 返回是否成功送达(失败不抛)。"""

    def send(self, title: str, content: str, *, level: str = "info") -> bool: ...


class StdoutNotifier:
    """打印到 stdout(本地开发/无 webhook 时的默认后端)。"""

    def send(self, title: str, content: str, *, level: str = "info") -> bool:
        print(f"[{level}] {title}\n{content}")
        return True


class WebhookNotifier:
    """POST JSON ``{title, content, level}`` 到 webhook。

    失败重试 ``max_retries`` 次(间隔 ``retry_delay`` 秒),仍失败则返回 False(不抛)。
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 10.0,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def send(self, title: str, content: str, *, level: str = "info") -> bool:
        payload = json.dumps(
            {"title": title, "content": content, "level": level}
        ).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        attempts = self.max_retries + 1
        for i in range(attempts):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    resp.read()
                return True
            except (urllib.error.URLError, TimeoutError, OSError):
                if i < attempts - 1:
                    time.sleep(self.retry_delay)
        return False


def build_notifier(cfg: OpsConfig) -> Notifier:
    """按配置构造 Notifier。

    webhook 模式但 URL 环境变量缺失时抛 RuntimeError——在启动期尽早暴露配置错,
    而非等到运行时才静默丢失告警。
    """
    if cfg.notify_kind == "stdout":
        return StdoutNotifier()
    url = os.environ.get(cfg.notify_url_env, "").strip()
    if not url:
        raise RuntimeError(
            f"notify_kind=webhook 但环境变量 {cfg.notify_url_env} 未设置"
        )
    return WebhookNotifier(url)

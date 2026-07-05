"""ops 通知层 Notifier 的测试(零依赖 webhook,失败不炸主链路)。"""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from factorzen.ops.config import OpsConfig
from factorzen.ops.notify import (
    StdoutNotifier,
    WebhookNotifier,
    build_notifier,
)


def test_stdout_notifier_returns_true(capsys):
    assert StdoutNotifier().send("hi", "body", level="warn") is True
    out = capsys.readouterr().out
    assert "hi" in out and "body" in out


def test_webhook_notifier_posts_json(monkeypatch):
    sent: dict = {}

    def fake_urlopen(req, timeout=None):
        sent["url"] = req.full_url
        sent["method"] = req.get_method()
        sent["body"] = json.loads(req.data.decode())
        return io.BytesIO(b"{}")

    monkeypatch.setattr("factorzen.ops.notify.urllib.request.urlopen", fake_urlopen)
    ok = WebhookNotifier("http://x/hook", retry_delay=0.0).send("t", "c", level="error")
    assert ok is True
    assert sent["url"] == "http://x/hook"
    assert sent["method"] == "POST"
    assert sent["body"] == {"title": "t", "content": "c", "level": "error"}


def test_webhook_notifier_retries_then_succeeds(monkeypatch):
    """前两次失败、第三次成功:重试机制生效,共 3 次尝试。"""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("down")
        return io.BytesIO(b"{}")

    monkeypatch.setattr("factorzen.ops.notify.urllib.request.urlopen", fake_urlopen)
    ok = WebhookNotifier("http://x/hook", max_retries=2, retry_delay=0.0).send("t", "c")
    assert ok is True
    assert calls["n"] == 3


def test_webhook_notifier_swallow_failure(monkeypatch):
    """全部失败:返回 False 而不抛(通知失败绝不能炸主链路)。"""

    def boom(req, timeout=None):
        raise urllib.error.URLError("down")

    monkeypatch.setattr("factorzen.ops.notify.urllib.request.urlopen", boom)
    assert WebhookNotifier("http://x/hook", max_retries=2, retry_delay=0.0).send("t", "c") is False


def test_build_notifier_stdout():
    cfg = OpsConfig(session_dir="s", portfolio_run_dirs_glob="g", notify_kind="stdout")
    assert isinstance(build_notifier(cfg), StdoutNotifier)


def test_build_notifier_webhook_with_env(monkeypatch):
    monkeypatch.setenv("FACTORZEN_NOTIFY_WEBHOOK", "http://hook")
    cfg = OpsConfig(session_dir="s", portfolio_run_dirs_glob="g", notify_kind="webhook")
    n = build_notifier(cfg)
    assert isinstance(n, WebhookNotifier)
    assert n.url == "http://hook"


def test_build_notifier_webhook_missing_env_raises(monkeypatch):
    """webhook 模式但 env 缺失:启动期尽早抛 RuntimeError,而非运行时静默。"""
    monkeypatch.delenv("FACTORZEN_NOTIFY_WEBHOOK", raising=False)
    cfg = OpsConfig(session_dir="s", portfolio_run_dirs_glob="g", notify_kind="webhook")
    with pytest.raises(RuntimeError):
        build_notifier(cfg)

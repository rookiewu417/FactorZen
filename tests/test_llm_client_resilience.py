# tests/test_llm_client_resilience.py
"""P1: LLM client 韧性 —— 重试、响应契约、上游锁定校验。

修复前：`client.py` 无任何重试；除 `node_critic` 的宽 `except Exception` 外，所有 LLM 调用点
都不 catch `LLMClientError`。多轮挖掘跑到第 N 轮遇到一次 429 → 异常一路冒泡 → 进程崩溃 →
前 N-1 轮的候选全部丢失（manifest 只在循环跑完后落盘）。在「无人值守运营」目标下，
例行网络抖动即静默全损。

另两条同族缺陷：
- `request_chat` 原样返回 `choices[0].message.content`，上游返回 `"content": null`
  （thinking 模式/网关常见）时返回 `None`，下游 `_extract_json(None)` 抛 **AttributeError**
  （非 ValueError），冒泡崩 session。
- 网关在响应体里回报了 `provider` / `is_fallback` / `model`，代码全部丢弃、从不校验。
  实测 aiping 网关确实尊重 `provider={"only":[...]}`（伪造 provider → HTTP 422），
  但一旦网关行为变化（fallback 生效、model 被替换），代码完全察觉不到。
"""
from __future__ import annotations

import io
import json
import logging
import urllib.error

import pytest

from factorzen.llm.client import LLMClientError, request_chat
from factorzen.llm.config import LLMConfig

_MSGS = [{"role": "user", "content": "hi"}]


def _cfg(**kw) -> LLMConfig:
    base = {
        "enabled": True,
        "base_url": "https://aiping.cn/api/v1",
        "api_key": "sk-super-secret-token",
        "model": "DeepSeek-V4-Pro",
        "provider": "DeepSeek",
    }
    base.update(kw)
    return LLMConfig(**base)


class _Resp:
    """最小 urlopen 返回值：支持 with 语法 + read()。"""

    def __init__(self, body: dict):
        self._raw = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._raw


def _ok_body(content: str = "好", provider: str = "DeepSeek",
             model: str = "DeepSeek-V4-Pro", is_fallback: bool = False) -> dict:
    body: dict = {
        "id": "x", "model": model, "is_fallback": is_fallback,
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }
    if provider is not None:
        body["provider"] = provider
    return body


def _http_error(code: int, body: bytes = b'{"msg":"boom"}') -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x/y", code, "err", {}, io.BytesIO(body))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """退避不真睡，测试跑得快。"""
    monkeypatch.setattr("factorzen.llm.client.time.sleep", lambda _s: None)


def _patch_urlopen(monkeypatch, side_effects):
    """side_effects: 每次调用弹出一个元素，是 Exception 就抛，否则当返回值。"""
    calls = {"n": 0}
    seq = list(side_effects)

    def fake(_req, timeout=None):
        calls["n"] += 1
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr("factorzen.llm.client.urllib.request.urlopen", fake)
    return calls


# ── 重试 ────────────────────────────────────────────────────────────────────


def test_retries_on_429_then_succeeds(monkeypatch):
    """429 是限流，重试即可恢复——不该让整轮挖掘全损。"""
    calls = _patch_urlopen(monkeypatch, [_http_error(429), _http_error(429), _Resp(_ok_body())])
    out = request_chat(_cfg(), _MSGS)
    assert out == "好"
    assert calls["n"] == 3, "429 应重试直到成功"


def test_retries_on_5xx_and_timeout(monkeypatch):
    calls = _patch_urlopen(monkeypatch, [_http_error(503), TimeoutError("slow"), _Resp(_ok_body())])
    assert request_chat(_cfg(), _MSGS) == "好"
    assert calls["n"] == 3


def test_does_not_retry_on_non_retryable_4xx(monkeypatch):
    """422「没有可用服务商」是配置错误，重试无意义且浪费配额——必须立刻失败。"""
    calls = _patch_urlopen(monkeypatch, [_http_error(422, b'{"msg":"no provider"}')])
    with pytest.raises(LLMClientError):
        request_chat(_cfg(), _MSGS)
    assert calls["n"] == 1, "不可重试错误不得重试"


def test_raises_llm_client_error_after_exhausting_retries(monkeypatch):
    _patch_urlopen(monkeypatch, [_http_error(429)] * 10)
    with pytest.raises(LLMClientError):
        request_chat(_cfg(max_retries=2), _MSGS)


def test_max_retries_zero_means_single_attempt(monkeypatch):
    calls = _patch_urlopen(monkeypatch, [_http_error(429)] * 3)
    with pytest.raises(LLMClientError):
        request_chat(_cfg(max_retries=0), _MSGS)
    assert calls["n"] == 1


# ── 响应契约 ────────────────────────────────────────────────────────────────


def test_null_content_raises_instead_of_returning_none(monkeypatch):
    """content=null 必须抛 LLMClientError（可预期），而非返回 None 让 _extract_json 抛 AttributeError。"""
    _patch_urlopen(monkeypatch, [_Resp(_ok_body(content=None))])  # type: ignore[arg-type]
    with pytest.raises(LLMClientError):
        request_chat(_cfg(max_retries=0), _MSGS)


def test_returns_str_content(monkeypatch):
    _patch_urlopen(monkeypatch, [_Resp(_ok_body(content="{\"a\":1}"))])
    assert request_chat(_cfg(), _MSGS) == '{"a":1}'


# ── 上游锁定校验 ────────────────────────────────────────────────────────────


def test_rejects_response_from_unexpected_provider(monkeypatch):
    """锁定 DeepSeek 却被路由到别家 → 必须拒绝，而不是静默接受。"""
    _patch_urlopen(monkeypatch, [_Resp(_ok_body(provider="OpenAI"))])
    with pytest.raises(LLMClientError, match="provider"):
        request_chat(_cfg(max_retries=0), _MSGS)


def test_rejects_fallback_response(monkeypatch):
    """网关 fallback 到备用上游 → 与「锁定」矛盾，必须拒绝。"""
    _patch_urlopen(monkeypatch, [_Resp(_ok_body(is_fallback=True))])
    with pytest.raises(LLMClientError, match="fallback"):
        request_chat(_cfg(max_retries=0), _MSGS)


def test_accepts_response_without_provider_field(monkeypatch):
    """非聚合网关（如 DeepSeek 官方）不回报 provider 字段 → 不应因此失败。"""
    body = _ok_body()
    body.pop("provider")
    body.pop("is_fallback")
    _patch_urlopen(monkeypatch, [_Resp(body)])
    assert request_chat(_cfg(), _MSGS) == "好"


def test_no_provider_pinned_is_allowed_but_unverified(monkeypatch):
    """未配置 provider 时不锁定、也不校验，但仍应正常返回（向后兼容）。"""
    _patch_urlopen(monkeypatch, [_Resp(_ok_body(provider="Whoever"))])
    assert request_chat(_cfg(provider=None), _MSGS) == "好"


def test_warns_once_when_provider_not_pinned(monkeypatch, caplog):
    """未锁定上游是静默降级——至少要告警，且只吵一次。"""
    import factorzen.llm.client as client_mod

    monkeypatch.setattr(client_mod, "_warned_provider_unpinned", False)
    _patch_urlopen(monkeypatch, [_Resp(_ok_body(provider="A")), _Resp(_ok_body(provider="B"))])

    with caplog.at_level(logging.WARNING, logger="factorzen.llm.client"):
        request_chat(_cfg(provider=None), _MSGS)
        request_chat(_cfg(provider=None), _MSGS)

    hits = [r for r in caplog.records if "FACTORZEN_LLM_PROVIDER 未配置" in r.getMessage()]
    assert len(hits) == 1, f"应恰好告警一次，实得 {len(hits)}"


# ── 凭据不泄漏 ──────────────────────────────────────────────────────────────


def test_api_key_never_leaks_into_exception(monkeypatch):
    _patch_urlopen(monkeypatch, [_http_error(422, b'{"msg":"bad"}')])
    with pytest.raises(LLMClientError) as ei:
        request_chat(_cfg(max_retries=0), _MSGS)
    assert "sk-super-secret-token" not in str(ei.value)

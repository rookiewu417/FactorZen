"""OpenAI SDK transport contract for AIPing streaming Chat Completions."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

import factorzen.llm.client as client_mod
from factorzen.llm.client import LLMClientError, request_chat
from factorzen.llm.config import LLMConfig

_MSGS = [{"role": "user", "content": "hi"}]


def _cfg(**kw) -> LLMConfig:
    base = {
        "enabled": True,
        "base_url": "https://www.aiping.cn/api/v1",
        "api_key": "sk-super-secret-token",
        "model": "DeepSeek-V4-Pro",
        "provider": "DeepSeek",
    }
    base.update(kw)
    return LLMConfig(**base)


def _chunk(
    content: str | None = None,
    *,
    reasoning: str | None = None,
    provider: str | None = None,
    is_fallback: bool | None = None,
    choices: bool = True,
):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta)] if choices else [],
        provider=provider,
        is_fallback=is_fallback,
    )


class _FakeCompletions:
    def __init__(self, result):
        self.result = result
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return iter(self.result)


class _FakeSDK:
    def __init__(self, result):
        self.completions = _FakeCompletions(result)
        self.chat = SimpleNamespace(completions=self.completions)


@pytest.fixture(autouse=True)
def _clear_client_cache():
    client_mod._openai_client.cache_clear()
    yield
    client_mod._openai_client.cache_clear()


def _install_fake(monkeypatch, result):
    sdk = _FakeSDK(result)
    init_calls: list[dict] = []

    def factory(**kwargs):
        init_calls.append(kwargs)
        return sdk

    monkeypatch.setattr(client_mod, "OpenAI", factory)
    return sdk, init_calls


def test_uses_openai_sdk_streaming_and_aiping_provider_only(monkeypatch):
    sdk, init_calls = _install_fake(
        monkeypatch,
        [
            _chunk(choices=False, provider="DeepSeek", is_fallback=False),
            _chunk(reasoning="internal reasoning"),
            _chunk('{"a":'),
            _chunk("1}"),
        ],
    )

    assert request_chat(_cfg(), _MSGS) == '{"a":1}'
    assert init_calls == [
        {
            "base_url": "https://www.aiping.cn/api/v1",
            "api_key": "sk-super-secret-token",
            "timeout": 30.0,
            "max_retries": 3,
        }
    ]
    call = sdk.completions.calls[0]
    assert call["stream"] is True
    assert call["extra_body"]["enable_thinking"] is False
    assert call["extra_body"]["provider"]["only"] == ["DeepSeek"]
    assert "response_format" not in call


def test_reuses_sdk_client_for_same_config(monkeypatch):
    _sdk, init_calls = _install_fake(monkeypatch, [_chunk("ok")])
    config = _cfg()

    assert request_chat(config, _MSGS) == "ok"
    assert request_chat(config, _MSGS) == "ok"
    assert len(init_calls) == 1


def test_full_chat_completions_url_is_normalized_for_sdk(monkeypatch):
    _sdk, init_calls = _install_fake(monkeypatch, [_chunk("ok")])

    request_chat(_cfg(base_url="https://www.aiping.cn/api/v1/chat/completions"), _MSGS)

    assert init_calls[0]["base_url"] == "https://www.aiping.cn/api/v1"


def test_empty_stream_content_raises(monkeypatch):
    _install_fake(monkeypatch, [_chunk(None), _chunk(reasoning="reasoning only")])

    with pytest.raises(LLMClientError, match=r"delta\.content"):
        request_chat(_cfg(), _MSGS)


def test_rejects_response_from_unexpected_provider(monkeypatch):
    _install_fake(monkeypatch, [_chunk("bad", provider="OpenAI")])

    with pytest.raises(LLMClientError, match="provider"):
        request_chat(_cfg(), _MSGS)


def test_rejects_fallback_response(monkeypatch):
    _install_fake(monkeypatch, [_chunk("bad", is_fallback=True)])

    with pytest.raises(LLMClientError, match="fallback"):
        request_chat(_cfg(), _MSGS)


def test_no_provider_pin_sends_empty_only_and_warns_once(monkeypatch, caplog):
    import factorzen.llm.client as module

    monkeypatch.setattr(module, "_warned_provider_unpinned", False)
    sdk, _ = _install_fake(monkeypatch, [_chunk("ok", provider="Whoever")])
    config = _cfg(provider=None)

    with caplog.at_level(logging.WARNING, logger="factorzen.llm.client"):
        assert request_chat(config, _MSGS) == "ok"
        assert request_chat(config, _MSGS) == "ok"

    assert sdk.completions.calls[0]["extra_body"]["provider"]["only"] == []
    hits = [r for r in caplog.records if "provider.only=[]" in r.getMessage()]
    assert len(hits) == 1


def test_sdk_status_error_is_wrapped_without_api_key(monkeypatch):
    request = httpx.Request("POST", "https://www.aiping.cn/api/v1/chat/completions")
    response = httpx.Response(422, request=request)
    exc = APIStatusError("bad request", response=response, body={"msg": "no provider"})
    _install_fake(monkeypatch, exc)

    with pytest.raises(LLMClientError) as caught:
        request_chat(_cfg(max_retries=0), _MSGS)

    assert "HTTP 422" in str(caught.value)
    assert "sk-super-secret-token" not in str(caught.value)


def test_sdk_connection_error_is_wrapped(monkeypatch):
    request = httpx.Request("POST", "https://www.aiping.cn/api/v1/chat/completions")
    _install_fake(monkeypatch, APIConnectionError(request=request))

    with pytest.raises(LLMClientError, match="APIConnectionError"):
        request_chat(_cfg(max_retries=0), _MSGS)

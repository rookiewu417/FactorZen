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
    client_mod._warned_provider_unpinned = False
    client_mod._warned_openai_provider_ignored = False
    yield
    client_mod._openai_client.cache_clear()
    client_mod._warned_provider_unpinned = False
    client_mod._warned_openai_provider_ignored = False


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


# ── openai flavor ────────────────────────────────────────────────────────────


def test_openai_flavor_skips_provider_chunk_validation(monkeypatch):
    """openai flavor 下 AIPing 扩展字段（provider/is_fallback）不得触发校验错误。"""
    _install_fake(
        monkeypatch,
        [_chunk("ok", provider="Whoever", is_fallback=True)],
    )

    assert request_chat(_cfg(flavor="openai", provider="DeepSeek"), _MSGS) == "ok"


def test_openai_flavor_warns_when_provider_configured(monkeypatch, caplog):
    monkeypatch.setattr(client_mod, "_warned_openai_provider_ignored", False)
    _install_fake(monkeypatch, [_chunk("ok")])

    with caplog.at_level(logging.WARNING, logger="factorzen.llm.client"):
        assert request_chat(_cfg(flavor="openai", provider="DeepSeek"), _MSGS) == "ok"

    hits = [r for r in caplog.records if "openai flavor" in r.getMessage() and "provider" in r.getMessage()]
    assert len(hits) >= 1


def test_openai_flavor_payload_has_no_extra_body_on_wire(monkeypatch):
    sdk, _ = _install_fake(monkeypatch, [_chunk("ok")])

    request_chat(_cfg(flavor="openai", provider=None, thinking="true"), _MSGS)

    call = sdk.completions.calls[0]
    assert "extra_body" not in call
    assert "temperature" not in call
    assert "max_tokens" not in call
    assert call["max_completion_tokens"] == 700
    assert call["stream"] is True


def _status_error(status: int, body, url: str = "http://localhost:8080/v1/chat/completions"):
    request = httpx.Request("POST", url)
    response = httpx.Response(status, request=request)
    return APIStatusError("bad request", response=response, body=body)


class _SequenceCompletions:
    """First create() returns/raises results[0], second results[1], ..."""

    def __init__(self, results):
        self.results = list(results)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.results:
            raise AssertionError("unexpected extra create() call")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return iter(result)


def _install_sequence(monkeypatch, results):
    completions = _SequenceCompletions(results)
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    init_calls: list[dict] = []

    def factory(**kwargs):
        init_calls.append(kwargs)
        return sdk

    monkeypatch.setattr(client_mod, "OpenAI", factory)
    return completions, init_calls


def test_openai_response_format_400_retries_once_without_format(monkeypatch, caplog):
    """openai + HTTP 400 提及 response_format → 重试一次去掉 response_format 成功。"""
    from factorzen.llm.client import request_llm_explanation

    # valid explanation JSON for request_llm_explanation
    ok_json = (
        '{"rating":"moderate","confidence":"medium",'
        '"factor_intuition":"s","evidence_assessment":"e",'
        '"risk_flags":["r"],"usage_suggestion":"u","next_steps":["n"]}'
    )
    completions, _ = _install_sequence(
        monkeypatch,
        [
            _status_error(
                400,
                {"error": {"message": "response_format is unsupported for this model"}},
            ),
            [_chunk(ok_json)],
        ],
    )

    with caplog.at_level(logging.WARNING, logger="factorzen.llm.client"):
        explanation = request_llm_explanation(
            _cfg(flavor="openai", provider=None, model="gpt-5.4"),
            _MSGS,
        )

    assert explanation.factor_intuition == "s"
    assert len(completions.calls) == 2
    assert completions.calls[0].get("response_format") == {"type": "json_object"}
    assert "response_format" not in completions.calls[1]
    assert any("response_format" in r.getMessage() for r in caplog.records)


def test_openai_response_format_400_retry_failure_raises_original(monkeypatch):
    """重试也失败 → 抛原错（首次 400）。"""
    from factorzen.llm.client import request_llm_explanation

    first = _status_error(400, {"error": {"message": "response_format unsupported"}})
    second = _status_error(500, {"error": {"message": "upstream boom"}})
    completions, _ = _install_sequence(monkeypatch, [first, second])

    with pytest.raises(LLMClientError, match="HTTP 400") as caught:
        request_llm_explanation(
            _cfg(flavor="openai", provider=None, model="gpt-5.4"),
            _MSGS,
        )

    assert len(completions.calls) == 2
    assert "response_format" in str(caught.value) or "HTTP 400" in str(caught.value)


def test_aiping_response_format_400_does_not_retry(monkeypatch):
    """aiping 路径对 response_format 400 不重试（行为不变）。"""
    from factorzen.llm.client import request_llm_explanation

    exc = _status_error(
        400,
        {"error": {"message": "response_format is unsupported"}},
        url="https://www.aiping.cn/api/v1/chat/completions",
    )
    completions, _ = _install_sequence(monkeypatch, [exc])

    with pytest.raises(LLMClientError, match="HTTP 400"):
        request_llm_explanation(_cfg(flavor="aiping"), _MSGS)

    assert len(completions.calls) == 1

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

    assert request_chat(_cfg(flavor="openai", provider="DeepSeek", stream=True), _MSGS) == "ok"


def test_openai_flavor_warns_when_provider_configured(monkeypatch, caplog):
    monkeypatch.setattr(client_mod, "_warned_openai_provider_ignored", False)
    _install_fake(monkeypatch, [_chunk("ok")])

    with caplog.at_level(logging.WARNING, logger="factorzen.llm.client"):
        assert request_chat(_cfg(flavor="openai", provider="DeepSeek", stream=True), _MSGS) == "ok"

    hits = [r for r in caplog.records if "openai flavor" in r.getMessage() and "provider" in r.getMessage()]
    assert len(hits) >= 1


def test_openai_flavor_payload_has_no_extra_body_on_wire(monkeypatch):
    sdk, _ = _install_fake(monkeypatch, [_chunk("ok")])

    request_chat(_cfg(flavor="openai", provider=None, thinking="true", stream=True), _MSGS)

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


# ── 传输层异常包装 + openai flavor 非流式（cockpit 断流事故回归）───────────────


class _BrokenStream:
    """迭代中途抛 httpx 传输异常（模拟网关长流式断流 incomplete chunked read）。"""

    def __iter__(self):
        yield _chunk("部分内容")
        import httpx

        raise httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body"
        )


def test_stream_transport_error_wrapped_as_llm_client_error(monkeypatch):
    """流迭代期的 httpx 传输异常必须包成 LLMClientError。

    真实事故：cockpit 网关长流式响应中途断流，RemoteProtocolError 穿透
    团队编排器的轮层容错（except LLMClientError）直接杀死整个挖掘 session。
    """
    sdk, _ = _install_fake(monkeypatch, [])
    monkeypatch.setattr(
        sdk.chat.completions, "create",
        lambda **kw: _BrokenStream(),
    )
    with pytest.raises(client_mod.LLMClientError, match="传输"):
        client_mod.request_chat(_cfg(), [{"role": "user", "content": "hi"}])


def _message_resp(content: str):
    """非流式 chat.completion 响应形状。"""
    msg = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_openai_flavor_defaults_to_non_streaming_and_reads_message(monkeypatch):
    """openai flavor 缺省非流式（本地网关对 chunked 长响应不可靠）：
    payload stream=False，内容从 choices[0].message.content 取。"""
    sdk, _ = _install_fake(monkeypatch, [])
    captured: list[dict] = []

    def create(**kw):
        captured.append(kw)
        return _message_resp("好的")

    monkeypatch.setattr(sdk.chat.completions, "create", create)
    cfg = _cfg(flavor="openai", provider=None)
    out = client_mod.request_chat(cfg, [{"role": "user", "content": "hi"}])
    assert out == "好的"
    assert captured[0]["stream"] is False


def test_openai_flavor_explicit_stream_true_still_streams(monkeypatch):
    """显式 stream=True 覆盖 flavor 缺省——openai flavor 仍可走流式。"""
    sdk, _ = _install_fake(monkeypatch, [_chunk("流式ok")])
    cfg = _cfg(flavor="openai", provider=None, stream=True)
    out = client_mod.request_chat(cfg, [{"role": "user", "content": "hi"}])
    assert out == "流式ok"
    assert sdk.chat.completions.calls[0]["stream"] is True


def test_empty_non_stream_content_raises(monkeypatch):
    sdk, _ = _install_fake(monkeypatch, [])
    monkeypatch.setattr(sdk.chat.completions, "create", lambda **kw: _message_resp(""))
    cfg = _cfg(flavor="openai", provider=None)
    with pytest.raises(client_mod.LLMClientError, match="content"):
        client_mod.request_chat(cfg, [{"role": "user", "content": "hi"}])

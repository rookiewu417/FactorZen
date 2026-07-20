"""
test_llm_client_resilience.py：OpenAI SDK transport contract for AIPing streaming Chat Completions
test_llm_explain_client.py：LLM explain 客户端 payload 构建与 flavor 分支测试
test_llm_explain_config.py：LLM explain 配置加载(env/profile/flavor/stream)测试
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

import factorzen.llm.client as client_mod
from factorzen.config.settings import ROOT
from factorzen.llm.client import LLMClientError, _build_payload, _provider_options, request_chat
from factorzen.llm.config import _DEFAULT_ENV_FILE, LLMConfig, load_llm_config

# ==== 来自 test_llm_client_resilience.py ====
_MSGS__llm_client_resilience = [{"role": "user", "content": "hi"}]

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

def test_aiping_sdk_client_suite(caplog):
    """test_uses_openai_sdk_streaming_and_aiping_provider_only；test_reuses_sdk_client_for_same_config；test_full_chat_completions_url_is_normalized_for_sdk；test_empty_stream_content_raises；test_rejects_response_from_unexpected_provider；test_rejects_fallback_response；test_no_provider_pin_sends_empty_only_and_warns_once；test_sdk_status_error_is_wrapped_without_api_key；test_sdk_connection_error_is_wrapped"""
    # -- 原 test_uses_openai_sdk_streaming_and_aiping_provider_only --
    def _section_0_test_uses_openai_sdk_streaming_and_aiping_provider_only(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        sdk, init_calls = _install_fake(
            mp,
            [
                _chunk(choices=False, provider="DeepSeek", is_fallback=False),
                _chunk(reasoning="internal reasoning"),
                _chunk('{"a":'),
                _chunk("1}"),
            ],
        )

        assert request_chat(_cfg(), _MSGS__llm_client_resilience) == '{"a":1}'
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

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_uses_openai_sdk_streaming_and_aiping_provider_only(mp)

    # -- 原 test_reuses_sdk_client_for_same_config --
    def _section_1_test_reuses_sdk_client_for_same_config(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        _sdk, init_calls = _install_fake(mp, [_chunk("ok")])
        config = _cfg()

        assert request_chat(config, _MSGS__llm_client_resilience) == "ok"
        assert request_chat(config, _MSGS__llm_client_resilience) == "ok"
        assert len(init_calls) == 1

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_reuses_sdk_client_for_same_config(mp)

    # -- 原 test_full_chat_completions_url_is_normalized_for_sdk --
    def _section_2_test_full_chat_completions_url_is_normalized_for_sdk(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        _sdk, init_calls = _install_fake(mp, [_chunk("ok")])

        request_chat(_cfg(base_url="https://www.aiping.cn/api/v1/chat/completions"), _MSGS__llm_client_resilience)

        assert init_calls[0]["base_url"] == "https://www.aiping.cn/api/v1"

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_full_chat_completions_url_is_normalized_for_sdk(mp)

    # -- 原 test_empty_stream_content_raises --
    def _section_3_test_empty_stream_content_raises(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        _install_fake(mp, [_chunk(None), _chunk(reasoning="reasoning only")])

        with pytest.raises(LLMClientError, match=r"delta\.content"):
            request_chat(_cfg(), _MSGS__llm_client_resilience)

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_empty_stream_content_raises(mp)

    # -- 原 test_rejects_response_from_unexpected_provider --
    def _section_4_test_rejects_response_from_unexpected_provider(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        _install_fake(mp, [_chunk("bad", provider="OpenAI")])

        with pytest.raises(LLMClientError, match="provider"):
            request_chat(_cfg(), _MSGS__llm_client_resilience)

    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_rejects_response_from_unexpected_provider(mp)

    # -- 原 test_rejects_fallback_response --
    def _section_5_test_rejects_fallback_response(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        _install_fake(mp, [_chunk("bad", is_fallback=True)])

        with pytest.raises(LLMClientError, match="fallback"):
            request_chat(_cfg(), _MSGS__llm_client_resilience)

    with pytest.MonkeyPatch.context() as mp:
        _section_5_test_rejects_fallback_response(mp)

    # -- 原 test_no_provider_pin_sends_empty_only_and_warns_once --
    def _section_6_test_no_provider_pin_sends_empty_only_and_warns_once(mp, caplog):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        caplog.clear()
        import factorzen.llm.client as module

        mp.setattr(module, "_warned_provider_unpinned", False)
        sdk, _ = _install_fake(mp, [_chunk("ok", provider="Whoever")])
        config = _cfg(provider=None)

        with caplog.at_level(logging.WARNING, logger="factorzen.llm.client"):
            assert request_chat(config, _MSGS__llm_client_resilience) == "ok"
            assert request_chat(config, _MSGS__llm_client_resilience) == "ok"

        assert sdk.completions.calls[0]["extra_body"]["provider"]["only"] == []
        hits = [r for r in caplog.records if "provider.only=[]" in r.getMessage()]
        assert len(hits) == 1

    with pytest.MonkeyPatch.context() as mp:
        _section_6_test_no_provider_pin_sends_empty_only_and_warns_once(mp, caplog)

    # -- 原 test_sdk_status_error_is_wrapped_without_api_key --
    def _section_7_test_sdk_status_error_is_wrapped_without_api_key(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        request = httpx.Request("POST", "https://www.aiping.cn/api/v1/chat/completions")
        response = httpx.Response(422, request=request)
        exc = APIStatusError("bad request", response=response, body={"msg": "no provider"})
        _install_fake(mp, exc)

        with pytest.raises(LLMClientError) as caught:
            request_chat(_cfg(max_retries=0), _MSGS__llm_client_resilience)

        assert "HTTP 422" in str(caught.value)
        assert "sk-super-secret-token" not in str(caught.value)

    with pytest.MonkeyPatch.context() as mp:
        _section_7_test_sdk_status_error_is_wrapped_without_api_key(mp)

    # -- 原 test_sdk_connection_error_is_wrapped --
    def _section_8_test_sdk_connection_error_is_wrapped(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        request = httpx.Request("POST", "https://www.aiping.cn/api/v1/chat/completions")
        _install_fake(mp, APIConnectionError(request=request))

        with pytest.raises(LLMClientError, match="APIConnectionError"):
            request_chat(_cfg(max_retries=0), _MSGS__llm_client_resilience)

    with pytest.MonkeyPatch.context() as mp:
        _section_8_test_sdk_connection_error_is_wrapped(mp)


# ── openai flavor ────────────────────────────────────────────────────────────

def test_openai_flavor_transport_suite(caplog):
    """openai flavor 下 AIPing 扩展字段（provider/is_fallback）不得触发校验错误。；test_openai_flavor_warns_when_provider_configured；test_openai_flavor_payload_has_no_extra_body_on_wire；流迭代期的 httpx 传输异常必须包成 LLMClientError。；openai flavor 缺省非流式（本地网关对 chunked 长响应不可靠）：；显式 stream=True 覆盖 flavor 缺省——openai flavor 仍可走流式。；test_empty_non_stream_content_raises"""
    # -- 原 test_openai_flavor_skips_provider_chunk_validation --
    def _section_0_test_openai_flavor_skips_provider_chunk_validation(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        _install_fake(
            mp,
            [_chunk("ok", provider="Whoever", is_fallback=True)],
        )

        assert request_chat(_cfg(flavor="openai", provider="DeepSeek", stream=True), _MSGS__llm_client_resilience) == "ok"

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_openai_flavor_skips_provider_chunk_validation(mp)

    # -- 原 test_openai_flavor_warns_when_provider_configured --
    def _section_1_test_openai_flavor_warns_when_provider_configured(mp, caplog):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        caplog.clear()
        mp.setattr(client_mod, "_warned_openai_provider_ignored", False)
        _install_fake(mp, [_chunk("ok")])

        with caplog.at_level(logging.WARNING, logger="factorzen.llm.client"):
            assert request_chat(_cfg(flavor="openai", provider="DeepSeek", stream=True), _MSGS__llm_client_resilience) == "ok"

        hits = [r for r in caplog.records if "openai flavor" in r.getMessage() and "provider" in r.getMessage()]
        assert len(hits) >= 1

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_openai_flavor_warns_when_provider_configured(mp, caplog)

    # -- 原 test_openai_flavor_payload_has_no_extra_body_on_wire --
    def _section_2_test_openai_flavor_payload_has_no_extra_body_on_wire(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        sdk, _ = _install_fake(mp, [_chunk("ok")])

        request_chat(_cfg(flavor="openai", provider=None, thinking="true", stream=True), _MSGS__llm_client_resilience)

        call = sdk.completions.calls[0]
        assert "extra_body" not in call
        assert "temperature" not in call
        assert "max_tokens" not in call
        assert call["max_completion_tokens"] == 700
        assert call["stream"] is True

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_openai_flavor_payload_has_no_extra_body_on_wire(mp)

    # -- 原 test_stream_transport_error_wrapped_as_llm_client_error --
    def _section_3_test_stream_transport_error_wrapped_as_llm_client_error(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        sdk, _ = _install_fake(mp, [])
        mp.setattr(
            sdk.chat.completions, "create",
            lambda **kw: _BrokenStream(),
        )
        with pytest.raises(client_mod.LLMClientError, match="传输"):
            client_mod.request_chat(_cfg(), [{"role": "user", "content": "hi"}])

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_stream_transport_error_wrapped_as_llm_client_error(mp)

    # -- 原 test_openai_flavor_defaults_to_non_streaming_and_reads_message --
    def _section_4_test_openai_flavor_defaults_to_non_streaming_and_reads_message(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        sdk, _ = _install_fake(mp, [])
        captured: list[dict] = []

        def create(**kw):
            captured.append(kw)
            return _message_resp("好的")

        mp.setattr(sdk.chat.completions, "create", create)
        cfg = _cfg(flavor="openai", provider=None)
        out = client_mod.request_chat(cfg, [{"role": "user", "content": "hi"}])
        assert out == "好的"
        assert captured[0]["stream"] is False

    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_openai_flavor_defaults_to_non_streaming_and_reads_message(mp)

    # -- 原 test_openai_flavor_explicit_stream_true_still_streams --
    def _section_5_test_openai_flavor_explicit_stream_true_still_streams(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        sdk, _ = _install_fake(mp, [_chunk("流式ok")])
        cfg = _cfg(flavor="openai", provider=None, stream=True)
        out = client_mod.request_chat(cfg, [{"role": "user", "content": "hi"}])
        assert out == "流式ok"
        assert sdk.chat.completions.calls[0]["stream"] is True

    with pytest.MonkeyPatch.context() as mp:
        _section_5_test_openai_flavor_explicit_stream_true_still_streams(mp)

    # -- 原 test_empty_non_stream_content_raises --
    def _section_6_test_empty_non_stream_content_raises(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        sdk, _ = _install_fake(mp, [])
        mp.setattr(sdk.chat.completions, "create", lambda **kw: _message_resp(""))
        cfg = _cfg(flavor="openai", provider=None)
        with pytest.raises(client_mod.LLMClientError, match="content"):
            client_mod.request_chat(cfg, [{"role": "user", "content": "hi"}])

    with pytest.MonkeyPatch.context() as mp:
        _section_6_test_empty_non_stream_content_raises(mp)


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


def _message_resp(content: str):
    """非流式 chat.completion 响应形状。"""
    msg = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


# ==== 来自 test_llm_explain_client.py ====
_MSGS__llm_explain_client = [{"role": "user", "content": "hi"}]

def test_build_payload_suite():
    """test_build_payload_translates_thinking_toggle_for_aiping；test_build_payload_pins_provider_when_configured；test_build_payload_sends_empty_provider_only_when_not_configured；flavor 缺省 aiping：payload 与改前逐键一致（含 extra_body 完整结构）。；openai flavor：max_completion_tokens、无 max_tokens、无 temperature、无 extra_body。；test_openai_flavor_payload_omits_response_format_when_disabled"""
    # -- 原 test_build_payload_translates_thinking_toggle_for_aiping --
    def _section_0_test_build_payload_translates_thinking_toggle_for_aiping():
        config = LLMConfig(
            enabled=True,
            base_url="https://api.deepseek.com",
            api_key="secret",
            model="deepseek-v4-flash",
            thinking="disabled",
        )

        payload = _build_payload(config, _MSGS__llm_explain_client)

        assert payload["model"] == "deepseek-v4-flash"
        assert payload["stream"] is True
        assert payload["extra_body"]["enable_thinking"] is False

    _section_0_test_build_payload_translates_thinking_toggle_for_aiping()

    # -- 原 test_build_payload_pins_provider_when_configured --
    def _section_1_test_build_payload_pins_provider_when_configured():
        config = LLMConfig(
            enabled=True,
            base_url="https://aiping.cn/api/v1",
            api_key="secret",
            model="DeepSeek-V4-Pro",
            provider="DeepSeek",
        )

        payload = _build_payload(config, _MSGS__llm_explain_client)

        assert payload["extra_body"]["provider"]["only"] == ["DeepSeek"]

    _section_1_test_build_payload_pins_provider_when_configured()

    # -- 原 test_build_payload_sends_empty_provider_only_when_not_configured --
    def _section_2_test_build_payload_sends_empty_provider_only_when_not_configured():
        config = LLMConfig(
            enabled=True,
            base_url="https://api.deepseek.com",
            api_key="secret",
            model="deepseek-v4-flash",
        )

        payload = _build_payload(config, _MSGS__llm_explain_client)

        assert payload["extra_body"]["provider"]["only"] == []

    _section_2_test_build_payload_sends_empty_provider_only_when_not_configured()

    # -- 原 test_aiping_payload_golden_byte_identical_shape --
    def _section_3_test_aiping_payload_golden_byte_identical_shape():
        config = LLMConfig(
            enabled=True,
            base_url="https://aiping.cn/api/v1",
            api_key="secret",
            model="DeepSeek-V4-Pro",
            provider="DeepSeek",
            thinking="true",
            temperature=0.2,
            max_tokens=700,
        )

        payload = _build_payload(config, _MSGS__llm_explain_client)

        assert payload == {
            "model": "DeepSeek-V4-Pro",
            "messages": _MSGS__llm_explain_client,
            "temperature": 0.2,
            "max_tokens": 700,
            "stream": True,
            "extra_body": {
                "enable_thinking": True,
                "provider": _provider_options(config),
            },
            "response_format": {"type": "json_object"},
        }
        # 明确钉死 AIPing 扩展字段集合，防止悄悄漂移
        assert set(payload["extra_body"]) == {"enable_thinking", "provider"}
        assert set(payload["extra_body"]["provider"]) == {
            "only",
            "order",
            "sort",
            "input_price_range",
            "output_price_range",
            "input_length_range",
            "output_length_range",
            "throughput_range",
            "latency_range",
        }
        assert "max_completion_tokens" not in payload

    _section_3_test_aiping_payload_golden_byte_identical_shape()

    # -- 原 test_openai_flavor_payload_uses_max_completion_tokens_no_temperature_no_extra_body --
    def _section_4_test_openai_flavor_payload_uses_max_completion_tokens_no_temperature_no_extra_body():
        config = LLMConfig(
            enabled=True,
            base_url="http://localhost:8080/v1",
            api_key="secret",
            model="gpt-5.4",
            max_tokens=900,
            temperature=0.7,  # 配置可有，payload 不得发出
            thinking="true",
            provider="DeepSeek",  # 配置可有，openai 不得进 payload
            flavor="openai",
            profile="sub2api",
        )

        payload = _build_payload(config, _MSGS__llm_explain_client)

        assert payload == {
            "model": "gpt-5.4",
            "messages": _MSGS__llm_explain_client,
            "max_completion_tokens": 900,
            # openai flavor 缺省非流式（本地网关 chunked 长响应不可靠;STREAM=true 可覆盖）
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        assert "max_tokens" not in payload
        assert "temperature" not in payload
        assert "extra_body" not in payload

    _section_4_test_openai_flavor_payload_uses_max_completion_tokens_no_temperature_no_extra_body()

    # -- 原 test_openai_flavor_payload_omits_response_format_when_disabled --
    def _section_5_test_openai_flavor_payload_omits_response_format_when_disabled():
        config = LLMConfig(
            enabled=True,
            base_url="http://localhost:8080/v1",
            api_key="secret",
            model="gpt-5.4",
            flavor="openai",
        )

        payload = _build_payload(config, _MSGS__llm_explain_client, include_response_format=False)

        assert "response_format" not in payload
        assert payload["stream"] is False
        assert payload["max_completion_tokens"] == 700

    _section_5_test_openai_flavor_payload_omits_response_format_when_disabled()


# ==== 来自 test_llm_explain_config.py ====
def _clear_llm_env(monkeypatch):
    """Clear flat + common profile LLM env vars so tests are hermetic."""
    for name in (
        "FACTORZEN_LLM_PROFILE",
        "FACTORZEN_LLM_FLAVOR",
        "FACTORZEN_LLM_ENABLED",
        "FACTORZEN_LLM_BASE_URL",
        "FACTORZEN_LLM_API_KEY",
        "FACTORZEN_LLM_MODEL",
        "FACTORZEN_LLM_TIMEOUT_SECONDS",
        "FACTORZEN_LLM_MAX_TOKENS",
        "FACTORZEN_LLM_MAX_RETRIES",
        "FACTORZEN_LLM_THINKING",
        "FACTORZEN_LLM_PROVIDER",
        "FACTORZEN_LLM_SUB2API_FLAVOR",
        "FACTORZEN_LLM_SUB2API_BASE_URL",
        "FACTORZEN_LLM_SUB2API_API_KEY",
        "FACTORZEN_LLM_SUB2API_MODEL",
        "FACTORZEN_LLM_SUB2API_TIMEOUT_SECONDS",
        "FACTORZEN_LLM_SUB2API_MAX_TOKENS",
        "FACTORZEN_LLM_SUB2API_MAX_RETRIES",
        "FACTORZEN_LLM_SUB2API_THINKING",
        "FACTORZEN_LLM_SUB2API_PROVIDER",
        "FACTORZEN_LLM_SUB2API_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)

def test_llm_config_load_suite(tmp_path):
    """test_default_env_file_is_project_root_env；test_config_is_disabled_without_explicit_flag；test_config_requires_complete_openai_compatible_settings；test_config_can_be_explicitly_enabled_but_not_ready；test_config_reads_project_env_file_when_process_env_missing；test_config_reads_optional_thinking_mode；test_config_reads_optional_provider_pin"""
    # -- 原 test_default_env_file_is_project_root_env --
    def _section_0_test_default_env_file_is_project_root_env():
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        assert _DEFAULT_ENV_FILE == ROOT / ".env"

    _section_0_test_default_env_file_is_project_root_env()

    # -- 原 test_config_is_disabled_without_explicit_flag --
    def _section_1_test_config_is_disabled_without_explicit_flag(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        _clear_llm_env(mp)

        config = load_llm_config(enabled=False, env_file=None)

        assert config.enabled is False
        assert config.is_ready is False

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_config_is_disabled_without_explicit_flag(mp)

    # -- 原 test_config_requires_complete_openai_compatible_settings --
    def _section_2_test_config_requires_complete_openai_compatible_settings(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        _clear_llm_env(mp)
        mp.setenv("FACTORZEN_LLM_BASE_URL", "https://api.example.com/v1")
        mp.setenv("FACTORZEN_LLM_API_KEY", "secret")
        mp.setenv("FACTORZEN_LLM_MODEL", "example-model")

        config = load_llm_config(enabled=True, env_file=None)

        assert config.enabled is True
        assert config.is_ready is True
        assert config.chat_completions_url == "https://api.example.com/v1/chat/completions"

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_config_requires_complete_openai_compatible_settings(mp)

    # -- 原 test_config_can_be_explicitly_enabled_but_not_ready --
    def _section_3_test_config_can_be_explicitly_enabled_but_not_ready(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        _clear_llm_env(mp)
        mp.setenv("FACTORZEN_LLM_BASE_URL", "https://api.example.com/v1")
        mp.setenv("FACTORZEN_LLM_MODEL", "example-model")

        config = load_llm_config(enabled=True, env_file=None)

        assert config.enabled is True
        assert config.is_ready is False

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_config_can_be_explicitly_enabled_but_not_ready(mp)

    # -- 原 test_config_reads_project_env_file_when_process_env_missing --
    def _section_4_test_config_reads_project_env_file_when_process_env_missing(tmp_path, mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        _clear_llm_env(mp)
        env_file = tmp_path / ".env"
        env_file.write_text(
            "\n".join(
                [
                    "FACTORZEN_LLM_BASE_URL=https://api.deepseek.com",
                    "FACTORZEN_LLM_API_KEY=secret",
                    "FACTORZEN_LLM_MODEL=deepseek-v4-pro",
                ]
            ),
            encoding="utf-8",
        )

        config = load_llm_config(enabled=True, env_file=env_file)

        assert config.is_ready is True
        assert config.model == "deepseek-v4-pro"

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_config_reads_project_env_file_when_process_env_missing(_tp4, mp)

    # -- 原 test_config_reads_optional_thinking_mode --
    def _section_5_test_config_reads_optional_thinking_mode(tmp_path, mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        _clear_llm_env(mp)
        env_file = tmp_path / ".env"
        env_file.write_text(
            "\n".join(
                [
                    "FACTORZEN_LLM_BASE_URL=https://api.deepseek.com",
                    "FACTORZEN_LLM_API_KEY=secret",
                    "FACTORZEN_LLM_MODEL=deepseek-v4-flash",
                    "FACTORZEN_LLM_THINKING=disabled",
                ]
            ),
            encoding="utf-8",
        )

        config = load_llm_config(enabled=True, env_file=env_file)

        assert config.thinking == "disabled"

    _tp5 = tmp_path / "_s5"
    _tp5.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_5_test_config_reads_optional_thinking_mode(_tp5, mp)

    # -- 原 test_config_reads_optional_provider_pin --
    def _section_6_test_config_reads_optional_provider_pin(tmp_path, mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        _clear_llm_env(mp)
        env_file = tmp_path / ".env"
        env_file.write_text(
            "\n".join(
                [
                    "FACTORZEN_LLM_BASE_URL=https://aiping.cn/api/v1",
                    "FACTORZEN_LLM_API_KEY=secret",
                    "FACTORZEN_LLM_MODEL=DeepSeek-V4-Pro",
                    "FACTORZEN_LLM_PROVIDER=DeepSeek",
                ]
            ),
            encoding="utf-8",
        )

        config = load_llm_config(enabled=True, env_file=env_file)

        assert config.provider == "DeepSeek"

    _tp6 = tmp_path / "_s6"
    _tp6.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_6_test_config_reads_optional_provider_pin(_tp6, mp)


# ── dual profile / flavor ─────────────────────────────────────────────────────

def test_llm_profile_suite(tmp_path):
    """PROFILE 未设 → 只读平铺变量；flavor 缺省 aiping；profile=None。；PROFILE 设后：profile 变量优先；缺项回退平铺。；load_llm_config(profile=...) 显式覆盖 FACTORZEN_LLM_PROFILE。；test_invalid_flavor_raises_value_error；PROFILE / profile 字段也可从 .env 文件读取（env 优先于文件）。；STREAM 未设 → None → stream_enabled 按 flavor 缺省（aiping=True/openai=False）。；test_stream_env_override_and_invalid_raises"""
    # -- 原 test_profile_unset_defaults_match_flat_layout_zero_regression --
    def _section_0_test_profile_unset_defaults_match_flat_layout_zero_regression(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        _clear_llm_env(mp)
        mp.setenv("FACTORZEN_LLM_BASE_URL", "https://aiping.example/v1")
        mp.setenv("FACTORZEN_LLM_API_KEY", "secret-flat")
        mp.setenv("FACTORZEN_LLM_MODEL", "DeepSeek-V4-Pro")
        mp.setenv("FACTORZEN_LLM_PROVIDER", "DeepSeek")
        mp.setenv("FACTORZEN_LLM_THINKING", "true")
        mp.setenv("FACTORZEN_LLM_TIMEOUT_SECONDS", "45")
        mp.setenv("FACTORZEN_LLM_MAX_TOKENS", "900")
        mp.setenv("FACTORZEN_LLM_MAX_RETRIES", "2")

        config = load_llm_config(enabled=True, env_file=None)

        assert config.profile is None
        assert config.flavor == "aiping"
        assert config.base_url == "https://aiping.example/v1"
        assert config.api_key == "secret-flat"
        assert config.model == "DeepSeek-V4-Pro"
        assert config.provider == "DeepSeek"
        assert config.thinking == "true"
        assert config.timeout_seconds == 45.0
        assert config.max_tokens == 900
        assert config.max_retries == 2
        assert config.is_ready is True

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_profile_unset_defaults_match_flat_layout_zero_regression(mp)

    # -- 原 test_profile_set_prefers_profile_vars_and_falls_back_to_flat --
    def _section_1_test_profile_set_prefers_profile_vars_and_falls_back_to_flat(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        _clear_llm_env(mp)
        mp.setenv("FACTORZEN_LLM_PROFILE", "sub2api")
        # flat defaults (AIPing)
        mp.setenv("FACTORZEN_LLM_BASE_URL", "https://aiping.example/v1")
        mp.setenv("FACTORZEN_LLM_API_KEY", "secret-flat")
        mp.setenv("FACTORZEN_LLM_MODEL", "DeepSeek-V4-Pro")
        mp.setenv("FACTORZEN_LLM_PROVIDER", "DeepSeek")
        mp.setenv("FACTORZEN_LLM_TIMEOUT_SECONDS", "30")
        mp.setenv("FACTORZEN_LLM_MAX_TOKENS", "700")
        # profile overrides only some fields
        mp.setenv("FACTORZEN_LLM_SUB2API_BASE_URL", "http://localhost:8080/v1")
        mp.setenv("FACTORZEN_LLM_SUB2API_API_KEY", "secret-sub2api")
        mp.setenv("FACTORZEN_LLM_SUB2API_MODEL", "gpt-5.4")
        mp.setenv("FACTORZEN_LLM_SUB2API_FLAVOR", "openai")
        # MAX_TOKENS / TIMEOUT / PROVIDER 不设 profile 前缀 → 回退平铺

        config = load_llm_config(enabled=True, env_file=None)

        assert config.profile == "sub2api"
        assert config.flavor == "openai"
        assert config.base_url == "http://localhost:8080/v1"
        assert config.api_key == "secret-sub2api"
        assert config.model == "gpt-5.4"
        assert config.provider == "DeepSeek"  # flat fallback
        assert config.timeout_seconds == 30.0
        assert config.max_tokens == 700

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_profile_set_prefers_profile_vars_and_falls_back_to_flat(mp)

    # -- 原 test_explicit_profile_arg_overrides_env --
    def _section_2_test_explicit_profile_arg_overrides_env(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        _clear_llm_env(mp)
        mp.setenv("FACTORZEN_LLM_PROFILE", "ignored")
        mp.setenv("FACTORZEN_LLM_BASE_URL", "https://flat.example/v1")
        mp.setenv("FACTORZEN_LLM_API_KEY", "secret-flat")
        mp.setenv("FACTORZEN_LLM_MODEL", "flat-model")
        mp.setenv("FACTORZEN_LLM_SUB2API_BASE_URL", "http://localhost:8080/v1")
        mp.setenv("FACTORZEN_LLM_SUB2API_API_KEY", "secret-sub2api")
        mp.setenv("FACTORZEN_LLM_SUB2API_MODEL", "gpt-5.4")
        mp.setenv("FACTORZEN_LLM_SUB2API_FLAVOR", "openai")

        config = load_llm_config(enabled=True, env_file=None, profile="sub2api")

        assert config.profile == "sub2api"
        assert config.flavor == "openai"
        assert config.model == "gpt-5.4"
        assert config.base_url == "http://localhost:8080/v1"

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_explicit_profile_arg_overrides_env(mp)

    # -- 原 test_invalid_flavor_raises_value_error --
    def _section_3_test_invalid_flavor_raises_value_error(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        _clear_llm_env(mp)
        mp.setenv("FACTORZEN_LLM_BASE_URL", "https://x/v1")
        mp.setenv("FACTORZEN_LLM_API_KEY", "secret")
        mp.setenv("FACTORZEN_LLM_MODEL", "m")
        mp.setenv("FACTORZEN_LLM_FLAVOR", "claude")

        import pytest

        with pytest.raises(ValueError, match="flavor"):
            load_llm_config(enabled=True, env_file=None)

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_invalid_flavor_raises_value_error(mp)

    # -- 原 test_profile_flavor_from_env_file --
    def _section_4_test_profile_flavor_from_env_file(tmp_path, mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        _clear_llm_env(mp)
        env_file = tmp_path / ".env"
        env_file.write_text(
            "\n".join(
                [
                    "FACTORZEN_LLM_PROFILE=sub2api",
                    "FACTORZEN_LLM_BASE_URL=https://flat.example/v1",
                    "FACTORZEN_LLM_API_KEY=secret-flat",
                    "FACTORZEN_LLM_MODEL=flat-model",
                    "FACTORZEN_LLM_SUB2API_BASE_URL=http://localhost:8080/v1",
                    "FACTORZEN_LLM_SUB2API_API_KEY=secret-sub2api",
                    "FACTORZEN_LLM_SUB2API_MODEL=gpt-5.4",
                    "FACTORZEN_LLM_SUB2API_FLAVOR=openai",
                ]
            ),
            encoding="utf-8",
        )

        config = load_llm_config(enabled=True, env_file=env_file)

        assert config.profile == "sub2api"
        assert config.flavor == "openai"
        assert config.model == "gpt-5.4"

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_profile_flavor_from_env_file(_tp4, mp)

    # -- 原 test_stream_default_none_follows_flavor --
    def _section_5_test_stream_default_none_follows_flavor(mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        for key in ("FACTORZEN_LLM_STREAM", "FACTORZEN_LLM_PROFILE"):
            mp.delenv(key, raising=False)
        config = load_llm_config(enabled=True, env_file=None)
        assert config.stream is None
        assert config.stream_enabled is True  # aiping 缺省
        from dataclasses import replace
        assert replace(config, flavor="openai").stream_enabled is False

    with pytest.MonkeyPatch.context() as mp:
        _section_5_test_stream_default_none_follows_flavor(mp)

    # -- 原 test_stream_env_override_and_invalid_raises --
    def _section_6_test_stream_env_override_and_invalid_raises(tmp_path, mp):
        client_mod._openai_client.cache_clear()
        client_mod._warned_provider_unpinned = False
        client_mod._warned_openai_provider_ignored = False
        for key in ("FACTORZEN_LLM_STREAM", "FACTORZEN_LLM_PROFILE"):
            mp.delenv(key, raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text(
            "FACTORZEN_LLM_PROFILE=gw\n"
            "FACTORZEN_LLM_GW_FLAVOR=openai\n"
            "FACTORZEN_LLM_GW_STREAM=true\n"
        )
        config = load_llm_config(enabled=True, env_file=env_file)
        assert config.stream is True and config.stream_enabled is True  # 显式覆盖 flavor 缺省

        env_file.write_text("FACTORZEN_LLM_STREAM=maybe\n")
        import pytest as _pytest
        with _pytest.raises(ValueError, match="STREAM"):
            load_llm_config(enabled=True, env_file=env_file)

    _tp6 = tmp_path / "_s6"
    _tp6.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_6_test_stream_env_override_and_invalid_raises(_tp6, mp)



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

def test_reuses_sdk_client_for_same_config(monkeypatch):
    _sdk, init_calls = _install_fake(monkeypatch, [_chunk("ok")])
    config = _cfg()

    assert request_chat(config, _MSGS__llm_client_resilience) == "ok"
    assert request_chat(config, _MSGS__llm_client_resilience) == "ok"
    assert len(init_calls) == 1

def test_full_chat_completions_url_is_normalized_for_sdk(monkeypatch):
    _sdk, init_calls = _install_fake(monkeypatch, [_chunk("ok")])

    request_chat(_cfg(base_url="https://www.aiping.cn/api/v1/chat/completions"), _MSGS__llm_client_resilience)

    assert init_calls[0]["base_url"] == "https://www.aiping.cn/api/v1"

def test_empty_stream_content_raises(monkeypatch):
    _install_fake(monkeypatch, [_chunk(None), _chunk(reasoning="reasoning only")])

    with pytest.raises(LLMClientError, match=r"delta\.content"):
        request_chat(_cfg(), _MSGS__llm_client_resilience)

def test_rejects_response_from_unexpected_provider(monkeypatch):
    _install_fake(monkeypatch, [_chunk("bad", provider="OpenAI")])

    with pytest.raises(LLMClientError, match="provider"):
        request_chat(_cfg(), _MSGS__llm_client_resilience)

def test_rejects_fallback_response(monkeypatch):
    _install_fake(monkeypatch, [_chunk("bad", is_fallback=True)])

    with pytest.raises(LLMClientError, match="fallback"):
        request_chat(_cfg(), _MSGS__llm_client_resilience)

def test_no_provider_pin_sends_empty_only_and_warns_once(monkeypatch, caplog):
    import factorzen.llm.client as module

    monkeypatch.setattr(module, "_warned_provider_unpinned", False)
    sdk, _ = _install_fake(monkeypatch, [_chunk("ok", provider="Whoever")])
    config = _cfg(provider=None)

    with caplog.at_level(logging.WARNING, logger="factorzen.llm.client"):
        assert request_chat(config, _MSGS__llm_client_resilience) == "ok"
        assert request_chat(config, _MSGS__llm_client_resilience) == "ok"

    assert sdk.completions.calls[0]["extra_body"]["provider"]["only"] == []
    hits = [r for r in caplog.records if "provider.only=[]" in r.getMessage()]
    assert len(hits) == 1

def test_sdk_status_error_is_wrapped_without_api_key(monkeypatch):
    request = httpx.Request("POST", "https://www.aiping.cn/api/v1/chat/completions")
    response = httpx.Response(422, request=request)
    exc = APIStatusError("bad request", response=response, body={"msg": "no provider"})
    _install_fake(monkeypatch, exc)

    with pytest.raises(LLMClientError) as caught:
        request_chat(_cfg(max_retries=0), _MSGS__llm_client_resilience)

    assert "HTTP 422" in str(caught.value)
    assert "sk-super-secret-token" not in str(caught.value)

def test_sdk_connection_error_is_wrapped(monkeypatch):
    request = httpx.Request("POST", "https://www.aiping.cn/api/v1/chat/completions")
    _install_fake(monkeypatch, APIConnectionError(request=request))

    with pytest.raises(LLMClientError, match="APIConnectionError"):
        request_chat(_cfg(max_retries=0), _MSGS__llm_client_resilience)

# ── openai flavor ────────────────────────────────────────────────────────────

def test_openai_flavor_skips_provider_chunk_validation(monkeypatch):
    """openai flavor 下 AIPing 扩展字段（provider/is_fallback）不得触发校验错误。"""
    _install_fake(
        monkeypatch,
        [_chunk("ok", provider="Whoever", is_fallback=True)],
    )

    assert request_chat(_cfg(flavor="openai", provider="DeepSeek", stream=True), _MSGS__llm_client_resilience) == "ok"

def test_openai_flavor_warns_when_provider_configured(monkeypatch, caplog):
    monkeypatch.setattr(client_mod, "_warned_openai_provider_ignored", False)
    _install_fake(monkeypatch, [_chunk("ok")])

    with caplog.at_level(logging.WARNING, logger="factorzen.llm.client"):
        assert request_chat(_cfg(flavor="openai", provider="DeepSeek", stream=True), _MSGS__llm_client_resilience) == "ok"

    hits = [r for r in caplog.records if "openai flavor" in r.getMessage() and "provider" in r.getMessage()]
    assert len(hits) >= 1

def test_openai_flavor_payload_has_no_extra_body_on_wire(monkeypatch):
    sdk, _ = _install_fake(monkeypatch, [_chunk("ok")])

    request_chat(_cfg(flavor="openai", provider=None, thinking="true", stream=True), _MSGS__llm_client_resilience)

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

# ==== 来自 test_llm_explain_client.py ====
_MSGS__llm_explain_client = [{"role": "user", "content": "hi"}]

def test_build_payload_translates_thinking_toggle_for_aiping():
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

def test_build_payload_pins_provider_when_configured():
    config = LLMConfig(
        enabled=True,
        base_url="https://aiping.cn/api/v1",
        api_key="secret",
        model="DeepSeek-V4-Pro",
        provider="DeepSeek",
    )

    payload = _build_payload(config, _MSGS__llm_explain_client)

    assert payload["extra_body"]["provider"]["only"] == ["DeepSeek"]

def test_build_payload_sends_empty_provider_only_when_not_configured():
    config = LLMConfig(
        enabled=True,
        base_url="https://api.deepseek.com",
        api_key="secret",
        model="deepseek-v4-flash",
    )

    payload = _build_payload(config, _MSGS__llm_explain_client)

    assert payload["extra_body"]["provider"]["only"] == []

def test_aiping_payload_golden_byte_identical_shape():
    """flavor 缺省 aiping：payload 与改前逐键一致（含 extra_body 完整结构）。"""
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

def test_openai_flavor_payload_uses_max_completion_tokens_no_temperature_no_extra_body():
    """openai flavor：max_completion_tokens、无 max_tokens、无 temperature、无 extra_body。"""
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

def test_openai_flavor_payload_omits_response_format_when_disabled():
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

def test_default_env_file_is_project_root_env():
    assert _DEFAULT_ENV_FILE == ROOT / ".env"

def test_config_is_disabled_without_explicit_flag(monkeypatch):
    _clear_llm_env(monkeypatch)

    config = load_llm_config(enabled=False, env_file=None)

    assert config.enabled is False
    assert config.is_ready is False

def test_config_requires_complete_openai_compatible_settings(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("FACTORZEN_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("FACTORZEN_LLM_API_KEY", "secret")
    monkeypatch.setenv("FACTORZEN_LLM_MODEL", "example-model")

    config = load_llm_config(enabled=True, env_file=None)

    assert config.enabled is True
    assert config.is_ready is True
    assert config.chat_completions_url == "https://api.example.com/v1/chat/completions"

def test_config_can_be_explicitly_enabled_but_not_ready(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("FACTORZEN_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("FACTORZEN_LLM_MODEL", "example-model")

    config = load_llm_config(enabled=True, env_file=None)

    assert config.enabled is True
    assert config.is_ready is False

def test_config_reads_project_env_file_when_process_env_missing(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
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

def test_config_reads_optional_thinking_mode(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
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

def test_config_reads_optional_provider_pin(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
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

# ── dual profile / flavor ─────────────────────────────────────────────────────

def test_profile_unset_defaults_match_flat_layout_zero_regression(monkeypatch):
    """PROFILE 未设 → 只读平铺变量；flavor 缺省 aiping；profile=None。"""
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("FACTORZEN_LLM_BASE_URL", "https://aiping.example/v1")
    monkeypatch.setenv("FACTORZEN_LLM_API_KEY", "secret-flat")
    monkeypatch.setenv("FACTORZEN_LLM_MODEL", "DeepSeek-V4-Pro")
    monkeypatch.setenv("FACTORZEN_LLM_PROVIDER", "DeepSeek")
    monkeypatch.setenv("FACTORZEN_LLM_THINKING", "true")
    monkeypatch.setenv("FACTORZEN_LLM_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("FACTORZEN_LLM_MAX_TOKENS", "900")
    monkeypatch.setenv("FACTORZEN_LLM_MAX_RETRIES", "2")

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

def test_profile_set_prefers_profile_vars_and_falls_back_to_flat(monkeypatch):
    """PROFILE 设后：profile 变量优先；缺项回退平铺。"""
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("FACTORZEN_LLM_PROFILE", "sub2api")
    # flat defaults (AIPing)
    monkeypatch.setenv("FACTORZEN_LLM_BASE_URL", "https://aiping.example/v1")
    monkeypatch.setenv("FACTORZEN_LLM_API_KEY", "secret-flat")
    monkeypatch.setenv("FACTORZEN_LLM_MODEL", "DeepSeek-V4-Pro")
    monkeypatch.setenv("FACTORZEN_LLM_PROVIDER", "DeepSeek")
    monkeypatch.setenv("FACTORZEN_LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("FACTORZEN_LLM_MAX_TOKENS", "700")
    # profile overrides only some fields
    monkeypatch.setenv("FACTORZEN_LLM_SUB2API_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("FACTORZEN_LLM_SUB2API_API_KEY", "secret-sub2api")
    monkeypatch.setenv("FACTORZEN_LLM_SUB2API_MODEL", "gpt-5.4")
    monkeypatch.setenv("FACTORZEN_LLM_SUB2API_FLAVOR", "openai")
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

def test_explicit_profile_arg_overrides_env(monkeypatch):
    """load_llm_config(profile=...) 显式覆盖 FACTORZEN_LLM_PROFILE。"""
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("FACTORZEN_LLM_PROFILE", "ignored")
    monkeypatch.setenv("FACTORZEN_LLM_BASE_URL", "https://flat.example/v1")
    monkeypatch.setenv("FACTORZEN_LLM_API_KEY", "secret-flat")
    monkeypatch.setenv("FACTORZEN_LLM_MODEL", "flat-model")
    monkeypatch.setenv("FACTORZEN_LLM_SUB2API_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("FACTORZEN_LLM_SUB2API_API_KEY", "secret-sub2api")
    monkeypatch.setenv("FACTORZEN_LLM_SUB2API_MODEL", "gpt-5.4")
    monkeypatch.setenv("FACTORZEN_LLM_SUB2API_FLAVOR", "openai")

    config = load_llm_config(enabled=True, env_file=None, profile="sub2api")

    assert config.profile == "sub2api"
    assert config.flavor == "openai"
    assert config.model == "gpt-5.4"
    assert config.base_url == "http://localhost:8080/v1"

def test_invalid_flavor_raises_value_error(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("FACTORZEN_LLM_BASE_URL", "https://x/v1")
    monkeypatch.setenv("FACTORZEN_LLM_API_KEY", "secret")
    monkeypatch.setenv("FACTORZEN_LLM_MODEL", "m")
    monkeypatch.setenv("FACTORZEN_LLM_FLAVOR", "claude")

    import pytest

    with pytest.raises(ValueError, match="flavor"):
        load_llm_config(enabled=True, env_file=None)

def test_profile_flavor_from_env_file(tmp_path, monkeypatch):
    """PROFILE / profile 字段也可从 .env 文件读取（env 优先于文件）。"""
    _clear_llm_env(monkeypatch)
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

def test_stream_default_none_follows_flavor(monkeypatch):
    """STREAM 未设 → None → stream_enabled 按 flavor 缺省（aiping=True/openai=False）。"""
    for key in ("FACTORZEN_LLM_STREAM", "FACTORZEN_LLM_PROFILE"):
        monkeypatch.delenv(key, raising=False)
    config = load_llm_config(enabled=True, env_file=None)
    assert config.stream is None
    assert config.stream_enabled is True  # aiping 缺省
    from dataclasses import replace
    assert replace(config, flavor="openai").stream_enabled is False

def test_stream_env_override_and_invalid_raises(tmp_path, monkeypatch):
    for key in ("FACTORZEN_LLM_STREAM", "FACTORZEN_LLM_PROFILE"):
        monkeypatch.delenv(key, raising=False)
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


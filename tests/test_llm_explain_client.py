from factorzen.llm.client import _build_payload, _provider_options
from factorzen.llm.config import LLMConfig

_MSGS = [{"role": "user", "content": "hi"}]


def test_build_payload_translates_thinking_toggle_for_aiping():
    config = LLMConfig(
        enabled=True,
        base_url="https://api.deepseek.com",
        api_key="secret",
        model="deepseek-v4-flash",
        thinking="disabled",
    )

    payload = _build_payload(config, _MSGS)

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

    payload = _build_payload(config, _MSGS)

    assert payload["extra_body"]["provider"]["only"] == ["DeepSeek"]


def test_build_payload_sends_empty_provider_only_when_not_configured():
    config = LLMConfig(
        enabled=True,
        base_url="https://api.deepseek.com",
        api_key="secret",
        model="deepseek-v4-flash",
    )

    payload = _build_payload(config, _MSGS)

    assert payload["extra_body"]["provider"]["only"] == []


def test_build_payload_enables_aiping_thinking_for_true_values():
    config = LLMConfig(
        enabled=True,
        base_url="https://www.aiping.cn/api/v1",
        api_key="secret",
        model="DeepSeek-V4-Pro",
        thinking="true",
    )

    payload = _build_payload(config, _MSGS)

    assert payload["extra_body"]["enable_thinking"] is True


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

    payload = _build_payload(config, _MSGS)

    assert payload == {
        "model": "DeepSeek-V4-Pro",
        "messages": _MSGS,
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

    payload = _build_payload(config, _MSGS)

    assert payload == {
        "model": "gpt-5.4",
        "messages": _MSGS,
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

    payload = _build_payload(config, _MSGS, include_response_format=False)

    assert "response_format" not in payload
    assert payload["stream"] is False
    assert payload["max_completion_tokens"] == 700

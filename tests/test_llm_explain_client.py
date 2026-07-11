from factorzen.llm.client import _build_payload
from factorzen.llm.config import LLMConfig


def test_build_payload_translates_thinking_toggle_for_aiping():
    config = LLMConfig(
        enabled=True,
        base_url="https://api.deepseek.com",
        api_key="secret",
        model="deepseek-v4-flash",
        thinking="disabled",
    )

    payload = _build_payload(config, [{"role": "user", "content": "hi"}])

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

    payload = _build_payload(config, [{"role": "user", "content": "hi"}])

    assert payload["extra_body"]["provider"]["only"] == ["DeepSeek"]


def test_build_payload_sends_empty_provider_only_when_not_configured():
    config = LLMConfig(
        enabled=True,
        base_url="https://api.deepseek.com",
        api_key="secret",
        model="deepseek-v4-flash",
    )

    payload = _build_payload(config, [{"role": "user", "content": "hi"}])

    assert payload["extra_body"]["provider"]["only"] == []


def test_build_payload_enables_aiping_thinking_for_true_values():
    config = LLMConfig(
        enabled=True,
        base_url="https://www.aiping.cn/api/v1",
        api_key="secret",
        model="DeepSeek-V4-Pro",
        thinking="true",
    )

    payload = _build_payload(config, [{"role": "user", "content": "hi"}])

    assert payload["extra_body"]["enable_thinking"] is True

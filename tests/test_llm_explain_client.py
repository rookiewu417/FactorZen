from factorzen.llm.client import _build_payload
from factorzen.llm.config import LLMConfig


def test_build_payload_includes_thinking_toggle_when_configured():
    config = LLMConfig(
        enabled=True,
        base_url="https://api.deepseek.com",
        api_key="secret",
        model="deepseek-v4-flash",
        thinking="disabled",
    )

    payload = _build_payload(config, [{"role": "user", "content": "hi"}])

    assert payload["model"] == "deepseek-v4-flash"
    assert payload["thinking"] == {"type": "disabled"}


def test_build_payload_pins_provider_when_configured():
    config = LLMConfig(
        enabled=True,
        base_url="https://aiping.cn/api/v1",
        api_key="secret",
        model="DeepSeek-V4-Pro",
        provider="DeepSeek",
    )

    payload = _build_payload(config, [{"role": "user", "content": "hi"}])

    assert payload["provider"] == {"only": ["DeepSeek"]}


def test_build_payload_omits_provider_when_not_configured():
    config = LLMConfig(
        enabled=True,
        base_url="https://api.deepseek.com",
        api_key="secret",
        model="deepseek-v4-flash",
    )

    payload = _build_payload(config, [{"role": "user", "content": "hi"}])

    assert "provider" not in payload

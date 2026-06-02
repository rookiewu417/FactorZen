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

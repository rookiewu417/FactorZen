"""Minimal OpenAI-compatible chat completions client."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from factorzen.llm.config import LLMConfig
from factorzen.llm.schema import LLMExplanation, parse_llm_explanation


class LLMClientError(RuntimeError):
    pass


def _build_payload(
    config: LLMConfig,
    messages: list[dict[str, str]],
    *,
    include_response_format: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if include_response_format:
        payload["response_format"] = {"type": "json_object"}
    if config.thinking:
        payload["thinking"] = {"type": config.thinking}
    if config.provider:
        # 聚合网关（如 AIPing）的上游路由锁定：只允许指定上游，避免被路由到其它模型。
        payload["provider"] = {"only": [config.provider]}
    return payload


def request_llm_explanation(
    config: LLMConfig,
    messages: list[dict[str, str]],
) -> LLMExplanation:
    """Call an OpenAI-compatible chat completions endpoint."""

    if not config.is_ready:
        raise LLMClientError("LLM config is not ready")

    payload = _build_payload(config, messages)
    request = urllib.request.Request(
        config.chat_completions_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            response_payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LLMClientError(str(exc)) from exc

    try:
        content = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMClientError("LLM response missing choices[0].message.content") from exc

    explanation = parse_llm_explanation(content)
    if explanation is None:
        raise LLMClientError("LLM response is not a valid explanation JSON")
    return explanation


def request_chat(config: LLMConfig, messages: list[dict[str, str]]) -> str:
    """通用 chat 请求：返回 choices[0].message.content 原始字符串。
    与 request_llm_explanation 的区别：不强制 response_format、不绑定 schema。"""
    if not config.is_ready:
        raise LLMClientError("LLM config is not ready")

    payload = _build_payload(config, messages, include_response_format=False)
    request = urllib.request.Request(
        config.chat_completions_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            response_payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LLMClientError(str(exc)) from exc

    try:
        return response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMClientError("chat 响应缺少 choices[0].message.content") from exc

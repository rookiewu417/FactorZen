"""OpenAI SDK client for AIPing-compatible streaming Chat Completions."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from openai import APIError, APIStatusError, OpenAI

from factorzen.llm.config import LLMConfig
from factorzen.llm.schema import LLMExplanation, parse_llm_explanation

_LOG = logging.getLogger(__name__)

_warned_provider_unpinned = False


class LLMClientError(RuntimeError):
    pass


def _provider_options(config: LLMConfig) -> dict[str, Any]:
    """Build AIPing's complete provider-routing object.

    ``only`` is the important safety boundary: a configured provider becomes a
    one-element allow-list; without a configured provider AIPing receives an
    explicit empty list and may route freely.
    """
    return {
        "only": [config.provider] if config.provider else [],
        "order": [],
        "sort": None,
        "input_price_range": [],
        "output_price_range": [],
        "input_length_range": [],
        "output_length_range": [],
        "throughput_range": [],
        "latency_range": [],
    }


def _build_payload(
    config: LLMConfig,
    messages: list[dict[str, str]],
    *,
    include_response_format: bool = True,
) -> dict[str, Any]:
    """Build keyword arguments for ``chat.completions.create``."""
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": True,
        "extra_body": {
            "enable_thinking": config.thinking_enabled,
            "provider": _provider_options(config),
        },
    }
    if include_response_format:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _warn_if_provider_unpinned(config: LLMConfig) -> None:
    """Warn once when ``provider.only`` is empty and routing is unrestricted."""
    global _warned_provider_unpinned
    if not config.provider and not _warned_provider_unpinned:
        _warned_provider_unpinned = True
        _LOG.warning(
            "FACTORZEN_LLM_PROVIDER 未配置：AIPing provider.only=[]，实际上游由网关路由决定。"
            "LLM 挖掘的结果强依赖模型，建议显式配置。"
        )


@lru_cache(maxsize=8)
def _openai_client(config: LLMConfig) -> OpenAI:
    """Create and reuse one SDK client per immutable LLM configuration."""
    return OpenAI(
        base_url=config.sdk_base_url,
        api_key=config.api_key,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )


def _extra_field(obj: Any, name: str) -> Any:
    """Read AIPing extension fields retained by the SDK's Pydantic models."""
    value = getattr(obj, name, None)
    if value is not None:
        return value
    extra = getattr(obj, "model_extra", None)
    return extra.get(name) if isinstance(extra, dict) else None


def _validate_gateway_chunk(config: LLMConfig, chunk: Any) -> None:
    """Reject a streamed chunk that proves the provider pin was violated."""
    if not config.provider:
        return
    actual = _extra_field(chunk, "provider")
    if actual is not None and actual != config.provider:
        raise LLMClientError(f"上游 provider 不符：锁定 {config.provider}，实得 {actual}")
    if _extra_field(chunk, "is_fallback"):
        raise LLMClientError(
            f"上游 fallback 生效（is_fallback=true），与 provider={config.provider} 锁定矛盾"
        )


def _error_body(exc: APIStatusError) -> str:
    """Render a short SDK error body without request headers or credentials."""
    body = getattr(exc, "body", None)
    if body is None:
        return ""
    try:
        return json.dumps(body, ensure_ascii=False)[:200]
    except (TypeError, ValueError):
        return str(body)[:200]


def _stream_content(
    config: LLMConfig,
    messages: list[dict[str, str]],
    *,
    include_response_format: bool,
) -> str:
    """Run one streaming completion and concatenate assistant content chunks.

    AIPing may expose ``delta.reasoning_content``.  FactorZen deliberately does
    not mix it into the returned string because Agent callers expect the final
    ``content`` to be parseable JSON.
    """
    _warn_if_provider_unpinned(config)
    try:
        stream = _openai_client(config).chat.completions.create(
            **_build_payload(
                config,
                messages,
                include_response_format=include_response_format,
            )
        )
        parts: list[str] = []
        for chunk in stream:
            _validate_gateway_chunk(config, chunk)
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if isinstance(content, str):
                parts.append(content)
    except LLMClientError:
        raise
    except APIStatusError as exc:
        raise LLMClientError(f"HTTP {exc.status_code}: {_error_body(exc)}") from exc
    except APIError as exc:
        raise LLMClientError(f"LLM SDK 请求失败: {type(exc).__name__}: {exc}") from exc

    content = "".join(parts)
    if not content:
        raise LLMClientError("LLM 流式响应缺少 choices[0].delta.content")
    return content


def request_llm_explanation(
    config: LLMConfig,
    messages: list[dict[str, str]],
) -> LLMExplanation:
    """Generate and validate one structured factor explanation."""
    if not config.is_ready:
        raise LLMClientError("LLM config is not ready")

    content = _stream_content(config, messages, include_response_format=True)
    explanation = parse_llm_explanation(content)
    if explanation is None:
        raise LLMClientError("LLM response is not a valid explanation JSON")
    return explanation


def request_chat(config: LLMConfig, messages: list[dict[str, str]]) -> str:
    """Return concatenated content from an AIPing streaming chat completion."""
    if not config.is_ready:
        raise LLMClientError("LLM config is not ready")
    return _stream_content(config, messages, include_response_format=False)

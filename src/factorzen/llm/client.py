"""OpenAI SDK client for AIPing / OpenAI-compatible streaming Chat Completions."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

import httpx
from openai import APIError, APIStatusError, OpenAI

from factorzen.llm.config import LLMConfig
from factorzen.llm.schema import LLMExplanation, parse_llm_explanation

_LOG = logging.getLogger(__name__)

_warned_provider_unpinned = False
_warned_openai_provider_ignored = False


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
    """Build keyword arguments for ``chat.completions.create``.

    ``flavor="aiping"`` keeps the historical AIPing payload (provider pin +
    enable_thinking).  ``flavor="openai"`` targets GPT-5.x / o-series gateways:
    ``max_completion_tokens`` instead of ``max_tokens``, no temperature, no
    AIPing ``extra_body``.
    """
    if config.flavor == "openai":
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "max_completion_tokens": config.max_tokens,
            # 缺省非流式：本地兼容网关对 chunked 长响应不可靠（实测 cockpit 长流式
            # 中途断流 RemoteProtocolError）；FACTORZEN_LLM_*_STREAM=true 可覆盖。
            "stream": config.stream_enabled,
        }
        if include_response_format:
            payload["response_format"] = {"type": "json_object"}
        return payload

    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": config.stream_enabled,
        "extra_body": {
            "enable_thinking": config.thinking_enabled,
            "provider": _provider_options(config),
        },
    }
    if include_response_format:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _warn_if_provider_unpinned(config: LLMConfig) -> None:
    """Warn once when ``provider.only`` is empty and routing is unrestricted.

    OpenAI-compatible gateways have no AIPing provider pin; if a provider is
    still configured under ``flavor=openai``, log that it is ignored.
    """
    global _warned_provider_unpinned, _warned_openai_provider_ignored
    if config.flavor == "openai":
        if config.provider and not _warned_openai_provider_ignored:
            _warned_openai_provider_ignored = True
            _LOG.warning(
                "openai flavor 忽略 provider=%s（OpenAI 兼容网关无 AIPing provider 路由）",
                config.provider,
            )
        return
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
    """Reject a streamed chunk that proves the provider pin was violated.

    Only meaningful for AIPing (``flavor=aiping``).  OpenAI-compatible gateways
    do not emit ``provider`` / ``is_fallback`` extension fields.
    """
    if config.flavor == "openai":
        return
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


def _mentions_response_format(exc: APIStatusError) -> bool:
    """True when the 400 body suggests ``response_format`` is unsupported."""
    text = _error_body(exc).lower()
    if "response_format" in text:
        return True
    # Some gateways only say "unsupported" / "not supported" about the format.
    return "unsupported" in text and "format" in text


def _consume_stream(config: LLMConfig, stream: Any) -> str:
    """Concatenate assistant ``delta.content`` from a streaming response."""
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
    content = "".join(parts)
    if not content:
        raise LLMClientError("LLM 流式响应缺少 choices[0].delta.content")
    return content


def _extract_message_content(response: Any) -> str:
    """非流式响应：取 ``choices[0].message.content``。"""
    choices = getattr(response, "choices", None)
    message = getattr(choices[0], "message", None) if choices else None
    content = getattr(message, "content", None) if message is not None else None
    if not isinstance(content, str) or not content:
        raise LLMClientError("LLM 非流式响应缺少 choices[0].message.content")
    return content


def _extract_content(config: LLMConfig, result: Any) -> str:
    """按流式开关消费 create() 的返回。"""
    if config.stream_enabled:
        return _consume_stream(config, result)
    return _extract_message_content(result)


def _create_stream(
    config: LLMConfig,
    messages: list[dict[str, str]],
    *,
    include_response_format: bool,
) -> Any:
    return _openai_client(config).chat.completions.create(
        **_build_payload(
            config,
            messages,
            include_response_format=include_response_format,
        )
    )


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

    For ``flavor=openai`` only: if the first request fails with HTTP 400 and the
    body mentions ``response_format`` / unsupported format, retry once without
    ``response_format``.  AIPing behaviour is unchanged.
    """
    _warn_if_provider_unpinned(config)
    try:
        stream = _create_stream(
            config, messages, include_response_format=include_response_format
        )
        return _extract_content(config, stream)
    except LLMClientError:
        raise
    except httpx.HTTPError as exc:
        # 传输层异常（断流/超时/连接重置）必须包成 LLMClientError——否则穿透
        # 团队编排器的轮层容错杀死整个挖掘 session（cockpit 断流真实事故）。
        raise LLMClientError(f"LLM 传输失败: {type(exc).__name__}: {exc}") from exc
    except APIStatusError as exc:
        can_retry = (
            config.flavor == "openai"
            and include_response_format
            and exc.status_code == 400
            and _mentions_response_format(exc)
        )
        if can_retry:
            _LOG.warning(
                "openai flavor: response_format 被上游拒绝 (HTTP 400)，"
                "去掉 response_format 重试一次"
            )
            try:
                stream = _create_stream(
                    config, messages, include_response_format=False
                )
                return _extract_content(config, stream)
            except LLMClientError:
                raise
            except httpx.HTTPError as retry_exc:
                raise LLMClientError(
                    f"LLM 传输失败: {type(retry_exc).__name__}: {retry_exc}"
                ) from retry_exc
            except APIStatusError:
                # 重试也失败 → 抛原错（首次 400），便于对照上游语义
                raise LLMClientError(f"HTTP {exc.status_code}: {_error_body(exc)}") from exc
            except APIError as retry_api_err:
                raise LLMClientError(
                    f"HTTP {exc.status_code}: {_error_body(exc)}"
                ) from retry_api_err
        raise LLMClientError(f"HTTP {exc.status_code}: {_error_body(exc)}") from exc
    except APIError as exc:
        raise LLMClientError(f"LLM SDK 请求失败: {type(exc).__name__}: {exc}") from exc


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
    """Return concatenated content from a streaming chat completion."""
    if not config.is_ready:
        raise LLMClientError("LLM config is not ready")
    return _stream_content(config, messages, include_response_format=False)

"""OpenAI SDK client for AIPing / OpenAI-compatible streaming Chat Completions."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

import httpx
from openai import APIError, APIStatusError, OpenAI

from factorzen.llm.config import LLMConfig

_LOG = logging.getLogger(__name__)

# 兼容旧测试重置；aiping 未 pin 不再告警，此标志不再写入。
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
    enable_thinking + optional reasoning_effort).  ``flavor="openai"`` targets
    GPT-5.x / o-series gateways: ``max_completion_tokens`` instead of
    ``max_tokens``, no temperature, no AIPing ``extra_body``; if
    ``reasoning_effort`` is set (and not ``none``) it is sent top-level for
    gateways that understand OpenAI-style effort.
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
        # 仅 thinking 开启时带 effort，避免 cockpit 等不识字段的网关 400。
        effort = (config.reasoning_effort or "").strip().lower()
        if config.thinking_enabled and effort and effort != "none":
            payload["reasoning_effort"] = effort
        return payload

    extra_body: dict[str, Any] = {
        "enable_thinking": config.thinking_enabled,
        "provider": _provider_options(config),
    }
    # thinking 开启时带 reasoning_effort（aiping 实测透传；关 thinking 时 effort 无效）
    effort = (config.reasoning_effort or "").strip().lower()
    if config.thinking_enabled and effort and effort != "none":
        extra_body["reasoning_effort"] = effort

    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": config.stream_enabled,
        "extra_body": extra_body,
    }
    if include_response_format:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _warn_if_provider_unpinned(config: LLMConfig) -> None:
    """Optional diagnostics for provider pin usage.

    AIPing 默认不再要求 ``FACTORZEN_LLM_PROVIDER``（模型名即可路由，如
    DeepSeek-V4-Flash-0731）；空 pin 是正常路径，不再告警。

    OpenAI-compatible gateways have no AIPing provider pin; if a provider is
    still configured under ``flavor=openai``, log that it is ignored.
    若仍显式配置了 provider，请求侧会写入 ``provider.only`` 并在响应侧校验。
    """
    global _warned_openai_provider_ignored
    if config.flavor == "openai":
        if config.provider and not _warned_openai_provider_ignored:
            _warned_openai_provider_ignored = True
            _LOG.warning(
                "openai flavor 忽略 provider=%s（OpenAI 兼容网关无 AIPing provider 路由）",
                config.provider,
            )
        return
    # aiping：未 pin 不再警告；显式 pin 仍走 extra_body.provider.only + chunk 校验。


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
    # aiping 2026-07 起流式 chunk 的 provider 恒为占位符 "-"（未标注实际上游），
    # 不是路由违规——与字段缺失（None）同等对待。路由锁定仍由请求侧
    # provider.only 允许列表承担；is_fallback 校验独立生效，不受此白名单影响。
    if actual in ("-", ""):
        actual = None
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


def _delta_reasoning(delta: Any) -> str | None:
    """AIPing / DeepSeek thinking：``delta.reasoning_content`` 或 model_extra。"""
    if delta is None:
        return None
    rc = getattr(delta, "reasoning_content", None)
    if isinstance(rc, str) and rc:
        return rc
    extra = getattr(delta, "model_extra", None)
    if isinstance(extra, dict):
        rc = extra.get("reasoning_content")
        if isinstance(rc, str) and rc:
            return rc
    return None


def _extract_jsonish_payload(text: str) -> str | None:
    """从思考链/杂文本里抠最后一段可解析 JSON 对象或数组（无 JSON 则 None）。"""
    if not text or not text.strip():
        return None
    s = text.strip()
    # 整段已是 JSON
    try:
        json.loads(s)
        return s
    except (TypeError, ValueError):
        pass
    # 取最后一个平衡的 {...} 或 [...]
    for open_c, close_c in (("{", "}"), ("[", "]")):
        end = s.rfind(close_c)
        if end < 0:
            continue
        depth = 0
        for i in range(end, -1, -1):
            ch = s[i]
            if ch == close_c:
                depth += 1
            elif ch == open_c:
                depth -= 1
                if depth == 0:
                    cand = s[i : end + 1]
                    try:
                        json.loads(cand)
                        return cand
                    except (TypeError, ValueError):
                        break
    return None


def _consume_stream(config: LLMConfig, stream: Any) -> str:
    """Concatenate assistant ``delta.content`` from a streaming response.

    DeepSeek V4 / aiping 在 ``enable_thinking`` 时常见**只流 reasoning_content、
    content 为空或极晚到达**。Agent 需要可 parse 的最终 content：优先 ``content``；
    若 content 为空则从 ``reasoning_content`` 中抽取最后一段 JSON（不把整段思考
    链当答案，避免污染解析器）。
    """
    parts: list[str] = []
    reasoning_parts: list[str] = []
    for chunk in stream:
        _validate_gateway_chunk(config, chunk)
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None) if delta is not None else None
        if isinstance(content, str):
            parts.append(content)
        rc = _delta_reasoning(delta)
        if rc:
            reasoning_parts.append(rc)
    content = "".join(parts)
    if content:
        return content
    # thinking-only 流：答案有时只出现在 reasoning 里
    reasoning = "".join(reasoning_parts)
    if reasoning:
        extracted = _extract_jsonish_payload(reasoning)
        if extracted:
            _LOG.warning(
                "流式 content 为空，已从 reasoning_content 抽取 JSON（len_reason=%d）",
                len(reasoning),
            )
            return extracted
    raise LLMClientError(
        "LLM 流式响应缺少 choices[0].delta.content"
        + (f"（reasoning 有 {len(reasoning)} 字符但无可用 JSON）" if reasoning else "")
    )


def _extract_message_content(response: Any) -> str:
    """非流式响应：取 ``choices[0].message.content``；空则尝试 reasoning 抽 JSON。"""
    choices = getattr(response, "choices", None)
    message = getattr(choices[0], "message", None) if choices else None
    content = getattr(message, "content", None) if message is not None else None
    if isinstance(content, str) and content:
        return content
    # 非流式 thinking：content 空、reasoning 有答案
    rc = None
    if message is not None:
        rc = getattr(message, "reasoning_content", None)
        if not rc:
            extra = getattr(message, "model_extra", None)
            if isinstance(extra, dict):
                rc = extra.get("reasoning_content")
    if isinstance(rc, str) and rc:
        extracted = _extract_jsonish_payload(rc)
        if extracted:
            _LOG.warning(
                "非流式 content 为空，已从 message.reasoning_content 抽取 JSON"
            )
            return extracted
    raise LLMClientError("LLM 非流式响应缺少 choices[0].message.content")


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
    """Run one chat completion and return assistant content (stream or not).

    AIPing thinking 流常把可解析答案放在 ``reasoning_content``；``_consume_stream``
    会优先 content、再从 reasoning 抽 JSON。

    容错：
    - ``flavor=openai`` + HTTP 400 且 body 提到 response_format → 去 format 重试一次
    - 流式失败且当前为 stream=True → **自动非流式重试一次**（aiping 空 content /
      「暂无可用服务商」抖动时，非流式往往更稳）
    """
    _warn_if_provider_unpinned(config)

    def _once(cfg: LLMConfig, *, with_response_format: bool) -> str:
        result = _create_stream(
            cfg, messages, include_response_format=with_response_format
        )
        return _extract_content(cfg, result)

    try:
        return _once(config, with_response_format=include_response_format)
    except LLMClientError as first_err:
        # 流式空 content / 抽 JSON 失败 → 非流式再试一次
        if config.stream_enabled:
            from dataclasses import replace

            _LOG.warning(
                "流式完成失败，改非流式重试一次: %s", first_err,
            )
            try:
                return _once(
                    replace(config, stream=False),
                    with_response_format=include_response_format,
                )
            except LLMClientError:
                raise first_err from None
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
                return _once(config, with_response_format=False)
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
        # aiping 抖动：流式 HTTP 错 → 非流式再试
        if config.stream_enabled and config.flavor == "aiping":
            from dataclasses import replace

            _LOG.warning(
                "aiping 流式 HTTP %s，改非流式重试: %s",
                exc.status_code, _error_body(exc)[:120],
            )
            try:
                return _once(
                    replace(config, stream=False),
                    with_response_format=include_response_format,
                )
            except Exception:
                pass
        raise LLMClientError(f"HTTP {exc.status_code}: {_error_body(exc)}") from exc
    except APIError as exc:
        msg = str(exc)
        # aiping「暂无可用服务商 / 所有服务商都调用失败」→ 非流式再试一次
        if config.stream_enabled and (
            "暂无可用" in msg or "服务商" in msg or "调用失败" in msg
        ):
            from dataclasses import replace

            _LOG.warning("上游服务商抖动，改非流式重试: %s", msg[:160])
            try:
                return _once(
                    replace(config, stream=False),
                    with_response_format=include_response_format,
                )
            except Exception:
                pass
        raise LLMClientError(f"LLM SDK 请求失败: {type(exc).__name__}: {exc}") from exc


def request_chat(config: LLMConfig, messages: list[dict[str, str]]) -> str:
    """Return concatenated content from a streaming chat completion."""
    if not config.is_ready:
        raise LLMClientError("LLM config is not ready")
    return _stream_content(config, messages, include_response_format=False)

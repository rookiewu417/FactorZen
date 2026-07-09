"""Minimal OpenAI-compatible chat completions client."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from factorzen.llm.config import LLMConfig
from factorzen.llm.schema import LLMExplanation, parse_llm_explanation

_LOG = logging.getLogger(__name__)

# 限流与服务端瞬时故障可重试；其余 4xx（如 422「没有可用服务商」）是配置错误，
# 重试只会浪费配额并拖长失败路径。
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

_warned_provider_unpinned = False


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


def _warn_if_provider_unpinned(config: LLMConfig) -> None:
    """未锁定上游时告警一次——否则实际服务商由网关决定，且响应无从校验。"""
    global _warned_provider_unpinned
    if not config.provider and not _warned_provider_unpinned:
        _warned_provider_unpinned = True
        _LOG.warning(
            "FACTORZEN_LLM_PROVIDER 未配置：不锁定上游服务商，实际由网关路由决定且不做校验。"
            "LLM 挖掘的结果强依赖模型，建议显式配置。"
        )


def _error_body(exc: urllib.error.HTTPError) -> str:
    """读取错误响应体（截断）。服务端返回的内容，不含本地凭据。"""
    try:
        return exc.read().decode("utf-8", errors="replace")[:200]
    except Exception:  # pragma: no cover - fp 已被消费/关闭
        return ""


def _post(config: LLMConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """POST 到 chat completions，返回解析后的响应体。

    可重试错误（429/5xx/网络故障/响应非 JSON）做指数退避重试；不可重试的 HTTP 状态立即抛。
    异常消息只携带 URL、状态码与服务端响应体，**绝不包含 Authorization header 或 api_key**。
    """
    attempts = max(0, config.max_retries) + 1
    last: Exception | None = None

    for i in range(attempts):
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
                body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                return body
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRYABLE_STATUS:
                raise LLMClientError(f"HTTP {exc.code}: {_error_body(exc)}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc

        if i < attempts - 1:
            backoff = config.retry_backoff_seconds * (2**i)
            _LOG.warning("LLM 请求失败（%s），%.1fs 后第 %d/%d 次重试",
                         type(last).__name__, backoff, i + 1, attempts - 1)
            time.sleep(backoff)

    raise LLMClientError(
        f"LLM 请求失败，已重试 {attempts - 1} 次仍不可恢复: {type(last).__name__}: {last}"
    ) from last


def _content_of(config: LLMConfig, body: dict[str, Any]) -> str:
    """从响应体取 content，并校验上游服务商与 fallback 状态。

    网关（如 AIPing）在响应里回报 ``provider`` / ``is_fallback`` / ``model``。配置了 provider
    锁定却拿到别家上游、或 fallback 生效，都与「锁定」矛盾——静默接受意味着后续因子实际由
    未知模型挖出，且事后无从追溯。只在字段存在时校验，非聚合网关（不回报该字段）不受影响。
    """
    if config.provider:
        actual = body.get("provider")
        if actual is not None and actual != config.provider:
            raise LLMClientError(
                f"上游 provider 不符：锁定 {config.provider}，实得 {actual}"
            )
        if body.get("is_fallback"):
            raise LLMClientError(
                f"上游 fallback 生效（is_fallback=true），与 provider={config.provider} 锁定矛盾"
            )

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMClientError("LLM 响应缺少 choices[0].message.content") from exc

    if not isinstance(content, str):
        # thinking 模式/部分网关会回 "content": null，只在 reasoning 字段里给内容。
        # 原样返回 None 会让下游 _extract_json(None) 抛 AttributeError（非可预期异常）。
        raise LLMClientError(
            f"LLM 响应 content 不是字符串（实得 {type(content).__name__}），疑似 content=null"
        )
    return content


def request_llm_explanation(
    config: LLMConfig,
    messages: list[dict[str, str]],
) -> LLMExplanation:
    """Call an OpenAI-compatible chat completions endpoint."""

    if not config.is_ready:
        raise LLMClientError("LLM config is not ready")

    _warn_if_provider_unpinned(config)
    body = _post(config, _build_payload(config, messages))
    content = _content_of(config, body)

    explanation = parse_llm_explanation(content)
    if explanation is None:
        raise LLMClientError("LLM response is not a valid explanation JSON")
    return explanation


def request_chat(config: LLMConfig, messages: list[dict[str, str]]) -> str:
    """通用 chat 请求：返回 choices[0].message.content 原始字符串。
    与 request_llm_explanation 的区别：不强制 response_format、不绑定 schema。"""
    if not config.is_ready:
        raise LLMClientError("LLM config is not ready")

    _warn_if_provider_unpinned(config)
    body = _post(config, _build_payload(config, messages, include_response_format=False))
    return _content_of(config, body)

"""Configuration for the OpenAI SDK backed AIPing LLM client."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from factorzen.config.settings import ROOT

_DEFAULT_ENV_FILE = ROOT / ".env"


def _read_env_file(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("FACTORZEN_LLM_"):
            values[key] = value.strip().strip('"').strip("'")
    return values


def _get_setting(name: str, file_values: dict[str, str]) -> str | None:
    return os.getenv(name) or file_values.get(name)


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool
    base_url: str | None
    api_key: str | None
    model: str | None
    timeout_seconds: float = 30.0
    temperature: float = 0.2
    max_tokens: int = 700
    thinking: str | None = None
    provider: str | None = None
    # 交给 OpenAI SDK 的有限重试次数；挖掘是多轮长循环，一次限流不该让整轮结果全损。
    max_retries: int = 3

    @property
    def is_ready(self) -> bool:
        return bool(self.enabled and self.base_url and self.api_key and self.model)

    @property
    def chat_completions_url(self) -> str:
        if not self.base_url:
            return ""
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    @property
    def sdk_base_url(self) -> str:
        """Return the API root expected by ``OpenAI(base_url=...)``.

        Older FactorZen configs were also allowed to contain the full
        ``/chat/completions`` endpoint.  Keep accepting those configs while the
        SDK itself appends the resource path.
        """
        if not self.base_url:
            return ""
        base = self.base_url.rstrip("/")
        suffix = "/chat/completions"
        if base.endswith(suffix):
            base = base[: -len(suffix)]
        return base

    @property
    def thinking_enabled(self) -> bool:
        """Translate the legacy string toggle to AIPing's boolean option."""
        if not self.thinking:
            return False
        return self.thinking.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def load_llm_config(*, enabled: bool, env_file: Path | None = _DEFAULT_ENV_FILE) -> LLMConfig:
    """Load optional LLM config.

    The feature is off unless the caller explicitly passes ``enabled=True``.
    ``FACTORZEN_LLM_ENABLED=false`` can still force-disable it.
    """

    file_values = _read_env_file(env_file)
    enabled_raw = _get_setting("FACTORZEN_LLM_ENABLED", file_values)
    final_enabled = enabled
    if enabled_raw is not None:
        final_enabled = final_enabled and enabled_raw.strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
    timeout_raw = _get_setting("FACTORZEN_LLM_TIMEOUT_SECONDS", file_values) or "30"
    max_tokens_raw = _get_setting("FACTORZEN_LLM_MAX_TOKENS", file_values) or "700"
    max_retries_raw = _get_setting("FACTORZEN_LLM_MAX_RETRIES", file_values) or "3"
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = 30.0
    try:
        max_tokens = int(max_tokens_raw)
    except ValueError:
        max_tokens = 700
    try:
        max_retries = max(0, int(max_retries_raw))
    except ValueError:
        max_retries = 3
    return LLMConfig(
        enabled=final_enabled,
        base_url=_get_setting("FACTORZEN_LLM_BASE_URL", file_values),
        api_key=_get_setting("FACTORZEN_LLM_API_KEY", file_values),
        model=_get_setting("FACTORZEN_LLM_MODEL", file_values),
        timeout_seconds=timeout,
        max_tokens=max_tokens,
        thinking=_get_setting("FACTORZEN_LLM_THINKING", file_values),
        provider=_get_setting("FACTORZEN_LLM_PROVIDER", file_values),
        max_retries=max_retries,
    )

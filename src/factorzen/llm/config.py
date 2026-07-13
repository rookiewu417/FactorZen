"""Configuration for the OpenAI SDK backed LLM client.

Supports two independent upstream profiles that can be switched at runtime:

```
# 平铺=默认 profile(AIPing/DeepSeek)：FACTORZEN_LLM_BASE_URL=...
#              FACTORZEN_LLM_API_KEY=sk-...
#              FACTORZEN_LLM_MODEL=DeepSeek-V4-Pro
#              FACTORZEN_LLM_FLAVOR=aiping   # 缺省即 aiping
#
# 第二 profile：FACTORZEN_LLM_SUB2API_BASE_URL=http://localhost:8080/v1
#              FACTORZEN_LLM_SUB2API_API_KEY=sk-...
#              FACTORZEN_LLM_SUB2API_MODEL=gpt-5.4
#              FACTORZEN_LLM_SUB2API_FLAVOR=openai
#
# 切换：FACTORZEN_LLM_PROFILE=sub2api（未设=平铺默认）
```

Profile-scoped keys take the form ``FACTORZEN_LLM_<PROFILE大写>_<FIELD>`` and
fall back to the flat ``FACTORZEN_LLM_<FIELD>`` when unset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from factorzen.config.settings import ROOT

_DEFAULT_ENV_FILE = ROOT / ".env"
_VALID_FLAVORS = frozenset({"aiping", "openai"})


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


def _get_profile_setting(
    field: str,
    profile: str | None,
    file_values: dict[str, str],
) -> str | None:
    """Read ``FACTORZEN_LLM_<PROFILE>_<FIELD>`` first, then flat fallback."""
    if profile:
        prefixed = f"FACTORZEN_LLM_{profile.upper()}_{field}"
        value = _get_setting(prefixed, file_values)
        if value is not None:
            return value
    return _get_setting(f"FACTORZEN_LLM_{field}", file_values)


def _resolve_profile(
    *,
    profile: str | None,
    file_values: dict[str, str],
) -> str | None:
    """Explicit ``profile=`` wins; else env/file ``FACTORZEN_LLM_PROFILE``; empty → None."""
    if profile is not None:
        cleaned = profile.strip()
        return cleaned or None
    raw = _get_setting("FACTORZEN_LLM_PROFILE", file_values)
    if raw is None:
        return None
    cleaned = raw.strip()
    return cleaned or None


def _resolve_flavor(raw: str | None) -> str:
    flavor = (raw or "aiping").strip().lower()
    if flavor not in _VALID_FLAVORS:
        raise ValueError(
            f"非法 LLM flavor={raw!r}；允许值: {sorted(_VALID_FLAVORS)}"
        )
    return flavor


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
    # 上游适配风格：aiping（默认，含 provider 路由 / enable_thinking）| openai（兼容网关）
    flavor: str = "aiping"
    # 当前选用的配置 profile 名（审计用；None = 平铺默认）
    profile: str | None = None

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


def load_llm_config(
    *,
    enabled: bool,
    env_file: Path | None = _DEFAULT_ENV_FILE,
    profile: str | None = None,
) -> LLMConfig:
    """Load optional LLM config.

    The feature is off unless the caller explicitly passes ``enabled=True``.
    ``FACTORZEN_LLM_ENABLED=false`` can still force-disable it.

    When ``FACTORZEN_LLM_PROFILE`` (or the ``profile=`` argument) is set, each
    field prefers ``FACTORZEN_LLM_<PROFILE>_<FIELD>`` and falls back to the flat
    ``FACTORZEN_LLM_<FIELD>``.  Unset profile keeps the historical flat-only
    layout (zero regression).
    """

    file_values = _read_env_file(env_file)
    active_profile = _resolve_profile(profile=profile, file_values=file_values)

    def setting(field: str) -> str | None:
        return _get_profile_setting(field, active_profile, file_values)

    enabled_raw = setting("ENABLED")
    final_enabled = enabled
    if enabled_raw is not None:
        final_enabled = final_enabled and enabled_raw.strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
    timeout_raw = setting("TIMEOUT_SECONDS") or "30"
    max_tokens_raw = setting("MAX_TOKENS") or "700"
    max_retries_raw = setting("MAX_RETRIES") or "3"
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

    flavor = _resolve_flavor(setting("FLAVOR"))

    return LLMConfig(
        enabled=final_enabled,
        base_url=setting("BASE_URL"),
        api_key=setting("API_KEY"),
        model=setting("MODEL"),
        timeout_seconds=timeout,
        max_tokens=max_tokens,
        thinking=setting("THINKING"),
        provider=setting("PROVIDER"),
        max_retries=max_retries,
        flavor=flavor,
        profile=active_profile,
    )

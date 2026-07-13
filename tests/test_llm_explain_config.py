from factorzen.config.settings import ROOT
from factorzen.llm.config import _DEFAULT_ENV_FILE, load_llm_config


def _clear_llm_env(monkeypatch):
    """Clear flat + common profile LLM env vars so tests are hermetic."""
    for name in (
        "FACTORZEN_LLM_PROFILE",
        "FACTORZEN_LLM_FLAVOR",
        "FACTORZEN_LLM_ENABLED",
        "FACTORZEN_LLM_BASE_URL",
        "FACTORZEN_LLM_API_KEY",
        "FACTORZEN_LLM_MODEL",
        "FACTORZEN_LLM_TIMEOUT_SECONDS",
        "FACTORZEN_LLM_MAX_TOKENS",
        "FACTORZEN_LLM_MAX_RETRIES",
        "FACTORZEN_LLM_THINKING",
        "FACTORZEN_LLM_PROVIDER",
        "FACTORZEN_LLM_SUB2API_FLAVOR",
        "FACTORZEN_LLM_SUB2API_BASE_URL",
        "FACTORZEN_LLM_SUB2API_API_KEY",
        "FACTORZEN_LLM_SUB2API_MODEL",
        "FACTORZEN_LLM_SUB2API_TIMEOUT_SECONDS",
        "FACTORZEN_LLM_SUB2API_MAX_TOKENS",
        "FACTORZEN_LLM_SUB2API_MAX_RETRIES",
        "FACTORZEN_LLM_SUB2API_THINKING",
        "FACTORZEN_LLM_SUB2API_PROVIDER",
        "FACTORZEN_LLM_SUB2API_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_env_file_is_project_root_env():
    assert _DEFAULT_ENV_FILE == ROOT / ".env"


def test_config_is_disabled_without_explicit_flag(monkeypatch):
    _clear_llm_env(monkeypatch)

    config = load_llm_config(enabled=False, env_file=None)

    assert config.enabled is False
    assert config.is_ready is False


def test_config_requires_complete_openai_compatible_settings(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("FACTORZEN_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("FACTORZEN_LLM_API_KEY", "secret")
    monkeypatch.setenv("FACTORZEN_LLM_MODEL", "example-model")

    config = load_llm_config(enabled=True, env_file=None)

    assert config.enabled is True
    assert config.is_ready is True
    assert config.chat_completions_url == "https://api.example.com/v1/chat/completions"


def test_config_can_be_explicitly_enabled_but_not_ready(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("FACTORZEN_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("FACTORZEN_LLM_MODEL", "example-model")

    config = load_llm_config(enabled=True, env_file=None)

    assert config.enabled is True
    assert config.is_ready is False


def test_config_reads_project_env_file_when_process_env_missing(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "FACTORZEN_LLM_BASE_URL=https://api.deepseek.com",
                "FACTORZEN_LLM_API_KEY=secret",
                "FACTORZEN_LLM_MODEL=deepseek-v4-pro",
            ]
        ),
        encoding="utf-8",
    )

    config = load_llm_config(enabled=True, env_file=env_file)

    assert config.is_ready is True
    assert config.model == "deepseek-v4-pro"


def test_config_reads_optional_thinking_mode(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "FACTORZEN_LLM_BASE_URL=https://api.deepseek.com",
                "FACTORZEN_LLM_API_KEY=secret",
                "FACTORZEN_LLM_MODEL=deepseek-v4-flash",
                "FACTORZEN_LLM_THINKING=disabled",
            ]
        ),
        encoding="utf-8",
    )

    config = load_llm_config(enabled=True, env_file=env_file)

    assert config.thinking == "disabled"


def test_config_reads_optional_provider_pin(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "FACTORZEN_LLM_BASE_URL=https://aiping.cn/api/v1",
                "FACTORZEN_LLM_API_KEY=secret",
                "FACTORZEN_LLM_MODEL=DeepSeek-V4-Pro",
                "FACTORZEN_LLM_PROVIDER=DeepSeek",
            ]
        ),
        encoding="utf-8",
    )

    config = load_llm_config(enabled=True, env_file=env_file)

    assert config.provider == "DeepSeek"


# ── dual profile / flavor ─────────────────────────────────────────────────────


def test_profile_unset_defaults_match_flat_layout_zero_regression(monkeypatch):
    """PROFILE 未设 → 只读平铺变量；flavor 缺省 aiping；profile=None。"""
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("FACTORZEN_LLM_BASE_URL", "https://aiping.example/v1")
    monkeypatch.setenv("FACTORZEN_LLM_API_KEY", "secret-flat")
    monkeypatch.setenv("FACTORZEN_LLM_MODEL", "DeepSeek-V4-Pro")
    monkeypatch.setenv("FACTORZEN_LLM_PROVIDER", "DeepSeek")
    monkeypatch.setenv("FACTORZEN_LLM_THINKING", "true")
    monkeypatch.setenv("FACTORZEN_LLM_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("FACTORZEN_LLM_MAX_TOKENS", "900")
    monkeypatch.setenv("FACTORZEN_LLM_MAX_RETRIES", "2")

    config = load_llm_config(enabled=True, env_file=None)

    assert config.profile is None
    assert config.flavor == "aiping"
    assert config.base_url == "https://aiping.example/v1"
    assert config.api_key == "secret-flat"
    assert config.model == "DeepSeek-V4-Pro"
    assert config.provider == "DeepSeek"
    assert config.thinking == "true"
    assert config.timeout_seconds == 45.0
    assert config.max_tokens == 900
    assert config.max_retries == 2
    assert config.is_ready is True


def test_profile_set_prefers_profile_vars_and_falls_back_to_flat(monkeypatch):
    """PROFILE 设后：profile 变量优先；缺项回退平铺。"""
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("FACTORZEN_LLM_PROFILE", "sub2api")
    # flat defaults (AIPing)
    monkeypatch.setenv("FACTORZEN_LLM_BASE_URL", "https://aiping.example/v1")
    monkeypatch.setenv("FACTORZEN_LLM_API_KEY", "secret-flat")
    monkeypatch.setenv("FACTORZEN_LLM_MODEL", "DeepSeek-V4-Pro")
    monkeypatch.setenv("FACTORZEN_LLM_PROVIDER", "DeepSeek")
    monkeypatch.setenv("FACTORZEN_LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("FACTORZEN_LLM_MAX_TOKENS", "700")
    # profile overrides only some fields
    monkeypatch.setenv("FACTORZEN_LLM_SUB2API_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("FACTORZEN_LLM_SUB2API_API_KEY", "secret-sub2api")
    monkeypatch.setenv("FACTORZEN_LLM_SUB2API_MODEL", "gpt-5.4")
    monkeypatch.setenv("FACTORZEN_LLM_SUB2API_FLAVOR", "openai")
    # MAX_TOKENS / TIMEOUT / PROVIDER 不设 profile 前缀 → 回退平铺

    config = load_llm_config(enabled=True, env_file=None)

    assert config.profile == "sub2api"
    assert config.flavor == "openai"
    assert config.base_url == "http://localhost:8080/v1"
    assert config.api_key == "secret-sub2api"
    assert config.model == "gpt-5.4"
    assert config.provider == "DeepSeek"  # flat fallback
    assert config.timeout_seconds == 30.0
    assert config.max_tokens == 700


def test_explicit_profile_arg_overrides_env(monkeypatch):
    """load_llm_config(profile=...) 显式覆盖 FACTORZEN_LLM_PROFILE。"""
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("FACTORZEN_LLM_PROFILE", "ignored")
    monkeypatch.setenv("FACTORZEN_LLM_BASE_URL", "https://flat.example/v1")
    monkeypatch.setenv("FACTORZEN_LLM_API_KEY", "secret-flat")
    monkeypatch.setenv("FACTORZEN_LLM_MODEL", "flat-model")
    monkeypatch.setenv("FACTORZEN_LLM_SUB2API_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("FACTORZEN_LLM_SUB2API_API_KEY", "secret-sub2api")
    monkeypatch.setenv("FACTORZEN_LLM_SUB2API_MODEL", "gpt-5.4")
    monkeypatch.setenv("FACTORZEN_LLM_SUB2API_FLAVOR", "openai")

    config = load_llm_config(enabled=True, env_file=None, profile="sub2api")

    assert config.profile == "sub2api"
    assert config.flavor == "openai"
    assert config.model == "gpt-5.4"
    assert config.base_url == "http://localhost:8080/v1"


def test_flavor_default_is_aiping_when_unset(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("FACTORZEN_LLM_BASE_URL", "https://aiping.example/v1")
    monkeypatch.setenv("FACTORZEN_LLM_API_KEY", "secret")
    monkeypatch.setenv("FACTORZEN_LLM_MODEL", "m")

    config = load_llm_config(enabled=True, env_file=None)

    assert config.flavor == "aiping"


def test_invalid_flavor_raises_value_error(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("FACTORZEN_LLM_BASE_URL", "https://x/v1")
    monkeypatch.setenv("FACTORZEN_LLM_API_KEY", "secret")
    monkeypatch.setenv("FACTORZEN_LLM_MODEL", "m")
    monkeypatch.setenv("FACTORZEN_LLM_FLAVOR", "claude")

    import pytest

    with pytest.raises(ValueError, match="flavor"):
        load_llm_config(enabled=True, env_file=None)


def test_profile_flavor_from_env_file(tmp_path, monkeypatch):
    """PROFILE / profile 字段也可从 .env 文件读取（env 优先于文件）。"""
    _clear_llm_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "FACTORZEN_LLM_PROFILE=sub2api",
                "FACTORZEN_LLM_BASE_URL=https://flat.example/v1",
                "FACTORZEN_LLM_API_KEY=secret-flat",
                "FACTORZEN_LLM_MODEL=flat-model",
                "FACTORZEN_LLM_SUB2API_BASE_URL=http://localhost:8080/v1",
                "FACTORZEN_LLM_SUB2API_API_KEY=secret-sub2api",
                "FACTORZEN_LLM_SUB2API_MODEL=gpt-5.4",
                "FACTORZEN_LLM_SUB2API_FLAVOR=openai",
            ]
        ),
        encoding="utf-8",
    )

    config = load_llm_config(enabled=True, env_file=env_file)

    assert config.profile == "sub2api"
    assert config.flavor == "openai"
    assert config.model == "gpt-5.4"

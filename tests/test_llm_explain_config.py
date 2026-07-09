from factorzen.config.settings import ROOT
from factorzen.llm.config import _DEFAULT_ENV_FILE, load_llm_config


def test_default_env_file_is_project_root_env():
    assert _DEFAULT_ENV_FILE == ROOT / ".env"


def test_config_is_disabled_without_explicit_flag(monkeypatch):
    monkeypatch.delenv("FACTORZEN_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("FACTORZEN_LLM_API_KEY", raising=False)
    monkeypatch.delenv("FACTORZEN_LLM_MODEL", raising=False)

    config = load_llm_config(enabled=False, env_file=None)

    assert config.enabled is False
    assert config.is_ready is False


def test_config_requires_complete_openai_compatible_settings(monkeypatch):
    monkeypatch.setenv("FACTORZEN_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("FACTORZEN_LLM_API_KEY", "secret")
    monkeypatch.setenv("FACTORZEN_LLM_MODEL", "example-model")

    config = load_llm_config(enabled=True, env_file=None)

    assert config.enabled is True
    assert config.is_ready is True
    assert config.chat_completions_url == "https://api.example.com/v1/chat/completions"


def test_config_can_be_explicitly_enabled_but_not_ready(monkeypatch):
    monkeypatch.setenv("FACTORZEN_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.delenv("FACTORZEN_LLM_API_KEY", raising=False)
    monkeypatch.setenv("FACTORZEN_LLM_MODEL", "example-model")

    config = load_llm_config(enabled=True, env_file=None)

    assert config.enabled is True
    assert config.is_ready is False


def test_config_reads_project_env_file_when_process_env_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("FACTORZEN_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("FACTORZEN_LLM_API_KEY", raising=False)
    monkeypatch.delenv("FACTORZEN_LLM_MODEL", raising=False)
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
    monkeypatch.delenv("FACTORZEN_LLM_THINKING", raising=False)
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
    monkeypatch.delenv("FACTORZEN_LLM_PROVIDER", raising=False)
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

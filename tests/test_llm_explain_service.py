from pathlib import Path

from factorzen.llm.schema import LLMExplanation
from factorzen.llm.service import generate_llm_explanation


def test_service_does_not_call_request_when_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("FACTORZEN_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("FACTORZEN_LLM_API_KEY", raising=False)
    monkeypatch.delenv("FACTORZEN_LLM_MODEL", raising=False)
    called = False

    def request_fn(_config, _messages):
        nonlocal called
        called = True
        raise AssertionError("request should not be called")

    explanation, path = generate_llm_explanation(
        enabled=False,
        refresh=False,
        cache_dir=tmp_path,
        factor_name="momentum_20d",
        factor_description="",
        start="20250101",
        end="20260513",
        frequency="daily",
        date_range="2025-01-01 ~ 2026-05-13",
        universe="csi300",
        ic_result=None,
        bt_result=None,
        to_result=None,
        env_file=None,
        request_fn=request_fn,
    )

    assert explanation is None
    assert path is None
    assert called is False


def test_service_skips_when_config_incomplete(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTORZEN_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.delenv("FACTORZEN_LLM_API_KEY", raising=False)
    monkeypatch.setenv("FACTORZEN_LLM_MODEL", "example-model")

    explanation, path = generate_llm_explanation(
        enabled=True,
        refresh=False,
        cache_dir=tmp_path,
        factor_name="momentum_20d",
        factor_description="",
        start="20250101",
        end="20260513",
        frequency="daily",
        date_range="2025-01-01 ~ 2026-05-13",
        universe="csi300",
        ic_result=None,
        bt_result=None,
        to_result=None,
        env_file=None,
    )

    assert explanation is None
    assert path is None


def test_service_uses_cache_before_request(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTORZEN_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("FACTORZEN_LLM_API_KEY", "secret")
    monkeypatch.setenv("FACTORZEN_LLM_MODEL", "example-model")
    cached = LLMExplanation(
        rating="weak",
        confidence="low",
        factor_intuition="动量因子刻画近期趋势。",
        evidence_assessment="证据较弱。",
        risk_flags=["统计显著性不足。"],
        usage_suggestion="仅适合继续观察。",
        next_steps=["扩大样本区间"],
    )

    def request_fn(_config, _messages):
        raise AssertionError("request should not be called when cache exists")

    first, first_path = generate_llm_explanation(
        enabled=True,
        refresh=False,
        cache_dir=tmp_path,
        factor_name="momentum_20d",
        factor_description="",
        start="20250101",
        end="20260513",
        frequency="daily",
        date_range="2025-01-01 ~ 2026-05-13",
        universe="csi300",
        ic_result=None,
        bt_result=None,
        to_result=None,
        env_file=None,
        request_fn=lambda _config, _messages: cached,
    )

    assert first == cached
    assert isinstance(first_path, Path)

    second, second_path = generate_llm_explanation(
        enabled=True,
        refresh=False,
        cache_dir=tmp_path,
        factor_name="momentum_20d",
        factor_description="",
        start="20250101",
        end="20260513",
        frequency="daily",
        date_range="2025-01-01 ~ 2026-05-13",
        universe="csi300",
        ic_result=None,
        bt_result=None,
        to_result=None,
        env_file=None,
        request_fn=request_fn,
    )

    assert second == cached
    assert second_path == first_path

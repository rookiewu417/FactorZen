from factorzen.llm.cache import cache_key, load_cached_explanation, save_cached_explanation
from factorzen.llm.schema import LLMExplanation


def test_cache_key_changes_when_snapshot_changes():
    first = cache_key(
        factor_name="momentum_20d",
        start="20250101",
        end="20260513",
        model="model-a",
        prompt_version="v1",
        snapshot={"ic": {"mean": 0.02}},
    )
    second = cache_key(
        factor_name="momentum_20d",
        start="20250101",
        end="20260513",
        model="model-a",
        prompt_version="v1",
        snapshot={"ic": {"mean": 0.03}},
    )

    assert first != second


def test_save_and_load_cached_explanation(tmp_path):
    explanation = LLMExplanation(
        rating="weak",
        confidence="low",
        factor_intuition="短期反转刻画近期过度反应。",
        evidence_assessment="IC 较弱，统计证据不足。",
        risk_flags=["样本期数不足。"],
        usage_suggestion="暂不作为核心信号。",
        next_steps=["扩大样本区间"],
    )
    key = "abc123"

    path = save_cached_explanation(tmp_path, key, explanation)
    loaded = load_cached_explanation(path)

    assert path.exists()
    assert loaded == explanation

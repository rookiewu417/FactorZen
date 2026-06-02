from factorzen.llm.schema import LLMExplanation, parse_llm_explanation


def test_parse_llm_explanation_accepts_compact_json():
    raw = """
    {
      "rating": "moderate",
      "confidence": "medium",
      "factor_intuition": "动量因子刻画近期强弱延续。",
      "evidence_assessment": "IC 为正但显著性一般，样本外仍需观察。",
      "risk_flags": ["换手率偏高，交易成本可能侵蚀收益。"],
      "usage_suggestion": "适合继续研究，不应单独作为交易信号。",
      "next_steps": ["检查行业中性后表现", "扩大样本区间"]
    }
    """

    explanation = parse_llm_explanation(raw)

    assert isinstance(explanation, LLMExplanation)
    assert explanation.rating == "moderate"
    assert explanation.confidence == "medium"
    assert len(explanation.risk_flags) == 1


def test_parse_llm_explanation_rejects_invalid_json():
    assert parse_llm_explanation("not-json") is None


def test_parse_llm_explanation_rejects_missing_required_fields():
    assert parse_llm_explanation('{"rating": "strong"}') is None


def test_parse_llm_explanation_accepts_json_inside_code_fence():
    raw = """
    ```json
    {
      "rating": "weak",
      "confidence": "low",
      "factor_intuition": "动量因子刻画近期趋势。",
      "evidence_assessment": "IC 显著为负，原方向不支持。",
      "risk_flags": ["需要考虑反向使用。"],
      "usage_suggestion": "仅适合作为反向信号继续研究。",
      "next_steps": ["检查更长样本"]
    }
    ```
    """

    explanation = parse_llm_explanation(raw)

    assert explanation is not None
    assert explanation.rating == "weak"

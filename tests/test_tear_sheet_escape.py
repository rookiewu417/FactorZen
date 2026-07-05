"""tear_sheet 的 LLM 解读文本须 HTML 转义。

summary_html 在模板里经 `{{ summary_html | safe }}` 渲染（绕过 jinja2 autoescape），
其中 LLM 输出的 evidence/usage/rating 是外部不可信文本，未转义会造成 HTML/JS 注入。
"""
from __future__ import annotations


def test_generate_summary_text_escapes_llm_html():
    from factorzen.reports.tear_sheet import _generate_summary_text

    llm = {
        "evidence_assessment": "<script>alert('xss')</script>",
        "usage_suggestion": "<img src=x onerror=alert(1)>",
        "rating": "A",
        "confidence": "high",
    }
    out = _generate_summary_text("f", {}, llm)
    assert "<script>" not in out, "LLM evidence 中的 <script> 应被转义"
    assert "&lt;script&gt;" in out, "转义后应出现 &lt;script&gt;"
    assert "<img src=x" not in out, "LLM usage 中的 <img onerror> 应被转义"

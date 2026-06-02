"""Prompt construction for compact factor explanations."""

from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "你是量化因子研究报告助手。只能基于用户提供的结构化指标解释，"
    "不得编造未提供的数据，不给投资建议，只给研究建议。"
    "输出必须是严格 JSON。"
)


def build_messages(snapshot: dict[str, Any]) -> list[dict[str, str]]:
    payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    user_prompt = (
        "基于以下因子评估摘要，生成低字数但完整的中文研究解读。"
        "字段要求：rating 只能是 strong/moderate/weak/invalid；confidence 只能是 high/medium/low；"
        "factor_intuition 80字以内；evidence_assessment 120字以内；"
        "risk_flags 最多5条，每条40字以内；usage_suggestion 100字以内；"
        "next_steps 最多4条。若统计证据弱、样本外不一致或换手高，必须降低置信度。"
        f"\n摘要JSON：{payload}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

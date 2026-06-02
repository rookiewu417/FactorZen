"""On-disk cache for LLM factor explanations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from factorzen.llm.schema import LLMExplanation, explanation_from_dict


def cache_key(
    *,
    factor_name: str,
    start: str,
    end: str,
    model: str,
    prompt_version: str,
    snapshot: dict[str, Any],
) -> str:
    payload = {
        "factor_name": factor_name,
        "start": start,
        "end": end,
        "model": model,
        "prompt_version": prompt_version,
        "snapshot": snapshot,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return f"{factor_name}_{start}_{end}_{digest}"


def save_cached_explanation(cache_dir: Path, key: str, explanation: LLMExplanation) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}_llm_explanation.json"
    path.write_text(
        json.dumps(explanation.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_cached_explanation(path: Path) -> LLMExplanation | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return explanation_from_dict(data)

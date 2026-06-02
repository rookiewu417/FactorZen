"""High-level orchestration for optional LLM factor explanations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from factorzen.core.logger import get_logger
from factorzen.llm.cache import cache_key, load_cached_explanation, save_cached_explanation
from factorzen.llm.client import LLMClientError, request_llm_explanation
from factorzen.llm.config import _DEFAULT_ENV_FILE, load_llm_config
from factorzen.llm.prompt import PROMPT_VERSION, build_messages
from factorzen.llm.schema import LLMExplanation
from factorzen.llm.snapshot import build_factor_snapshot

logger = get_logger(__name__)

RequestFn = Callable[[Any, list[dict[str, str]]], LLMExplanation]


def generate_llm_explanation(
    *,
    enabled: bool,
    refresh: bool,
    cache_dir: Path,
    factor_name: str,
    factor_description: str | None,
    start: str,
    end: str,
    frequency: str,
    date_range: str,
    universe: str,
    ic_result: Any,
    bt_result: Any,
    to_result: Any,
    walk_forward_summary: dict[str, Any] | None = None,
    quality_report: dict[str, Any] | None = None,
    backtest_direction: dict[str, Any] | None = None,
    env_file: Path | None = _DEFAULT_ENV_FILE,
    request_fn: RequestFn = request_llm_explanation,
) -> tuple[LLMExplanation | None, Path | None]:
    """Generate or load an LLM explanation when explicitly enabled."""

    config = load_llm_config(enabled=enabled, env_file=env_file)
    if not config.enabled:
        return None, None
    if not config.is_ready:
        logger.info("LLM explanation skipped: FACTORZEN_LLM_* config is incomplete")
        return None, None

    snapshot = build_factor_snapshot(
        factor_name=factor_name,
        factor_description=factor_description,
        frequency=frequency,
        date_range=date_range,
        universe=universe,
        ic_result=ic_result,
        bt_result=bt_result,
        to_result=to_result,
        walk_forward_summary=walk_forward_summary,
        quality_report=quality_report,
        backtest_direction=backtest_direction,
    )
    key = cache_key(
        factor_name=factor_name,
        start=start,
        end=end,
        model=config.model or "",
        prompt_version=PROMPT_VERSION,
        snapshot=snapshot,
    )
    cache_path = cache_dir / f"{key}_llm_explanation.json"
    if not refresh:
        cached = load_cached_explanation(cache_path)
        if cached is not None:
            logger.info(f"LLM explanation loaded from cache: {cache_path}")
            return cached, cache_path

    try:
        explanation = request_fn(config, build_messages(snapshot))
    except LLMClientError as exc:
        logger.warning(f"LLM explanation skipped: {exc}")
        return None, None

    path = save_cached_explanation(cache_dir, key, explanation)
    logger.info(f"LLM explanation saved: {path}")
    return explanation, path

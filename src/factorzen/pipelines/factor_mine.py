# src/factorzen/pipelines/factor_mine.py
"""fz mine 的 pipeline 入口：拉数据 → run_session。"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from factorzen.discovery.guardrails import DEFAULT_DSR_ALPHA
from factorzen.discovery.mining_session import run_session
from factorzen.discovery.preparation import prepare_mining_daily

_LOG = logging.getLogger(__name__)

def _inject_membership_into_session_manifest(
    session_dir: str | None, prep_meta: dict
) -> None:
    """run_session 无 manifest 注入口：在 run_mine 层读-补-写 membership 溯源字段。

    侵入最小：不改 mining_session 签名；manifest 文件不存在时静默跳过
    （测试 mock run_session 常只回 session_dir 字符串）。
    """
    if not session_dir or not prep_meta:
        return
    path = Path(session_dir) / "manifest.json"
    if not path.is_file():
        return
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("无法读取 session manifest 以注入 membership：%s", exc)
        return
    for key in (
        "membership_mode",
        "membership_hash",
        "membership_n_rows",
        "universe",
    ):
        if key in prep_meta:
            manifest[key] = prep_meta[key]
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_mine(*, start: str, end: str, universe: str | None = None,
             n_trials: int = 200, top_k: int = 10, seed: int = 42,
             method: str = "random", holdout_ratio: float = 0.2,
             train_ratio: float = 0.7, decorr_threshold: float = 0.7,
             min_n_train: int = 5, dsr_alpha: float = DEFAULT_DSR_ALPHA,
             workers: int = 1, update_library: bool = True,
             library_orthogonal: bool = True,
             objective: str = "residual") -> dict:
    prep_meta: dict = {}
    daily = prepare_mining_daily(start, end, universe, out_meta=prep_meta)
    # 收尾自动 upsert 因子库（--no-library 关）；库根由 run_session 从 out_dir 推导
    # （workspace/mining_sessions → workspace/factor_library）。universe 落进记录溯源。
    # library_orthogonal：搜索期避开库内已覆盖方向（--no-library-orthogonal 关）。
    # objective：残差/裸 IC 挖掘目标（库空时 residual 自动退化 raw）。
    result = run_session(daily, n_trials=n_trials, top_k=top_k, seed=seed, method=method,
                         holdout_ratio=holdout_ratio, train_ratio=train_ratio,
                         decorr_threshold=decorr_threshold, min_n_train=min_n_train,
                         dsr_alpha=dsr_alpha, eval_start=start, workers=workers,
                         update_library=update_library, library_universe=universe,
                         library_orthogonal=library_orthogonal, objective=objective)
    # run_session 无 manifest_extra：读-补-写 membership_*（可复现铁律）
    _inject_membership_into_session_manifest(result.get("session_dir"), prep_meta)
    return result

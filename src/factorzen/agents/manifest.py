"""Agent session manifest：把假设/表达式/分数/候选/参数全程落盘（可审计、可复现）。"""
from __future__ import annotations

import json
from pathlib import Path

from factorzen.core.experiment import get_git_sha


def json_safe_float(x: float | None) -> float | None:
    """nan/inf → None。

    `json.dumps` 默认把 nan 写成裸 `NaN`，那**不是合法 JSON**：Python 的 json.loads 宽容地
    接受，但标准解析器（其它语言、jq、前端）会直接失败。manifest 是跨工具消费的产物。
    `pool_pbo` 在候选 <2 时正常返回 nan，因此这条路径是常态而非异常。
    """
    if x is None or x != x or x in (float("inf"), float("-inf")):
        return None
    return float(x)


def _sanitize(obj):
    """递归把树里所有 nan/inf 变成 None（attempts[].ir_train、candidates[].dsr 都可能是 nan）。"""
    if isinstance(obj, float):
        return json_safe_float(obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def dump_manifest(manifest: dict, path: Path) -> None:
    """写 manifest：先递归清洗 nan/inf，再以 ``allow_nan=False`` 兜底。

    兜底的意义：日后谁加了 _sanitize 覆盖不到的浮点类型，这里会立刻抛，
    而不是悄悄写出一个别的语言读不了的 manifest。
    """
    path.write_text(
        json.dumps(_sanitize(manifest), ensure_ascii=False, indent=2, allow_nan=False)
    )


def write_session_manifest(
    result, *, out_dir: str, run_id: str, params: dict, partial: bool = False
) -> Path:
    """落 session manifest。

    ``partial=True`` 表示这是轮末的增量快照——挖掘尚未跑完，进程若在此后崩溃，
    留在磁盘上的就是它。消费方据此区分「跑完的结果」与「崩溃现场」。
    """
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state = result.state
    manifest = {
        "run_id": run_id, "seed": state.seed, "n_trials": result.n_trials,
        # deflation 基准的尺度。与 n_trials 一起，才够复算出候选的 dsr_pvalue
        # （`expected_max_sharpe ∝ sqrt(sharpe_variance)`）。partial 快照写 null——
        # 那时还没有最终 basis。`deflation_two_sided` 说明 effective_trials = 2×n_trials。
        "sharpe_variance": json_safe_float(getattr(result, "sharpe_variance", float("nan"))),
        "deflation_two_sided": True,
        "iterations": state.iteration, "params": params,
        "partial": partial,
        "pbo": json_safe_float(state.pbo),
        "attempts": [a.__dict__ for a in state.attempts],
        "candidates": result.candidates,
        "git_sha": get_git_sha(),
    }
    path = run_dir / "manifest.json"
    dump_manifest(manifest, path)
    return path

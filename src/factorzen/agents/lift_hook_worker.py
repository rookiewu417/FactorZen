"""mine team session 末 lift 钩子的**子进程执行体**。

由 ``team_orchestrator._lift_hook_via_subprocess`` 以
``python -m factorzen.agents.lift_hook_worker <payload.json> <out.json>`` 拉起。

分工：
- 主进程负责队列收集/去重、把 session 的 ``prepped`` 帧写成 Arrow IPC、写 payload，
  然后**松开自己的大帧**再 spawn；
- 本模块只做「读帧 → 调 ``_run_lift_hook_core`` → 写结果 JSON」，
  基线库池 / 面板 / 逐日 QR 全部在这个干净进程里现建现丢。

裁决实现是主进程与子进程**共用**的 ``_run_lift_hook_core`` —— 钩子不再自己搭一套
基线设施，CLAUDE.md 双路径表里「team session 末 lift 钩子 ▸ rebuild · lift-test --apply」
那一行在物理上合一。

业务异常在这里就地转成 ``lift_error``（与进程内路径逐字同格式）并**正常退出 0**，
让主进程拿到与旧行为一致的 meta；只有崩溃 / 被 OOM-kill 才让主进程走降级出口。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_USAGE = "usage: python -m factorzen.agents.lift_hook_worker <payload.json> <out.json>"


def run_payload(payload: dict) -> dict:
    """payload → 钩子 meta（与进程内 ``_session_end_auto_lift`` 返回同结构）。"""
    import polars as pl

    from factorzen.agents.experiment_index import ExperimentIndex
    from factorzen.agents.team_orchestrator import (
        _empty_lift_meta,
        _lift_hook_error_meta,
        _run_lift_hook_core,
    )

    queue = list(payload.get("queue") or [])
    meta = _empty_lift_meta(n_lift_queue=len(queue))
    try:
        prepped = pl.read_ipc(payload["prepped_path"])
        index_path = payload.get("index_path")
        index = ExperimentIndex(index_path) if index_path else None
        return _run_lift_hook_core(
            queue, meta,
            # ctx 一给，`daily` 在 run_group_lift / run_lift_tests 里就只是签名占位；
            # 传同一份帧避免任何「两个帧」的歧义
            daily=prepped,
            prepped=prepped,
            holdout_start=payload.get("holdout_start"),
            holdout_end=payload.get("holdout_end"),
            profile=None,  # 帧已 prep 过，子进程不重 prep ⇒ 不需要 profile 对象
            profile_name=payload.get("profile_name"),
            leaf_map=payload.get("leaf_map"),
            market=payload["market"],
            library_root=payload["library_root"],
            seed=int(payload.get("seed") or 0),
            horizon=int(payload["horizon"]),
            lift_se_mult=float(payload.get("lift_se_mult") or 1.0),
            lift_workers=payload.get("lift_workers"),
            data_window=payload.get("data_window"),
            materialize_candidate=None,
            active_factor_dfs=None,
            ret_df=None,
            run_id=payload.get("run_id"),
            index=index,
            exec_price_col=payload.get("exec_price_col"),
        )
    except Exception as exc:
        return _lift_hook_error_meta(meta, exc)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:]) if argv is None else list(argv)
    if len(args) != 2:
        print(_USAGE, file=sys.stderr)
        return 2
    payload = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    meta = run_payload(payload)
    Path(args[1]).write_text(
        json.dumps(meta, ensure_ascii=False, default=str), encoding="utf-8",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - 进程入口
    raise SystemExit(main())

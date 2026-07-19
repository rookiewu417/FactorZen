"""CLI lift-test 路径共用 mock：覆盖门 + 组门放行，避免 mock 帧物化全灭。

W0-fix-2 / W2b 后 CLI 顺序为 top_m → coverage → group_gate → run_lift_tests；
旧测试只 mock run_lift_tests 会在 coverage 步被真实 materializer 清空队列。
"""
from __future__ import annotations


def patch_cli_lift_pre_gates(monkeypatch, *, group_lift: float = 0.01, group_se: float = 0.001):
    """filter 全放行 + 组门过 + resolve_lift_workers 透传。"""
    import factorzen.discovery.lift_test as lt_mod

    monkeypatch.setattr(
        lt_mod,
        "filter_candidates_by_coverage",
        lambda cands, **k: (list(cands), []),
    )
    monkeypatch.setattr(
        lt_mod,
        "run_group_lift",
        lambda queue, **k: {
            "lift": group_lift,
            "lift_se": group_se,
            "error": None,
            "lift_metric": "residual_ic_v1",
        },
    )
    monkeypatch.setattr(
        lt_mod,
        "resolve_lift_workers",
        lambda w: 2 if w is None else int(w),
    )

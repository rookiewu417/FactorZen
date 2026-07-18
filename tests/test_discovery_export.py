from __future__ import annotations

from pathlib import Path


def _write_candidates_csv(tmp_path, *, with_passed=True):
    import polars as pl
    rows = [
        {"rank": 1, "expression": "close", "passed": True},
        {"rank": 2, "expression": "neg(close)", "passed": False},
    ]
    if not with_passed:
        rows = [{k: v for k, v in r.items() if k != "passed"} for r in rows]
    d = tmp_path / "sess"
    d.mkdir()
    pl.DataFrame(rows).write_csv(d / "candidates.csv")
    return str(d)


def test_read_candidate_require_passed_rejects_unpassed(tmp_path: Path):
    """R1：require_passed=True 时，请求未过护栏的 rank 报错并提示 --all；过的正常返回。"""
    import pytest

    from factorzen.discovery.export import read_candidate_expression
    sess = _write_candidates_csv(tmp_path)
    assert read_candidate_expression(sess, rank=1, require_passed=True) == "close"       # 过
    with pytest.raises(ValueError, match="--all"):
        read_candidate_expression(sess, rank=2, require_passed=True)                     # 未过 → 拒
    assert read_candidate_expression(sess, rank=2, require_passed=False) == "neg(close)"  # 逃生口


def test_read_candidate_backward_compat_no_passed_column(tmp_path: Path):
    """老 session 无 passed 列时 require_passed 不生效（不破坏向后兼容）。"""
    from factorzen.discovery.export import read_candidate_expression
    sess = _write_candidates_csv(tmp_path, with_passed=False)
    assert read_candidate_expression(sess, rank=2, require_passed=True) == "neg(close)"


# render_factor_file / export_candidate / exported/*.py 桥已废除（Batch 2）；
# lookback 契约见 test_export_lookback.py → lookback_for_expression。

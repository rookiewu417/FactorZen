# tests/test_holdout_coverage_guard.py
"""P1+P2：holdout 覆盖守卫 + 同号门修正 + 叶子健康检查 + known_invalid 卫生。

期望（独立构造，不读被测函数输出当期望）：
- 稀疏 holdout（n_days 不足）→ 死因含「覆盖不足」，不含「反号」；train 正/负均拒。
- 覆盖充足 + 真反号 → 仍报「反号」。
- holdout 精确 0 且天数够 → 「无信号」，不叫反号。
- 叶健康：holdout 段全 null 的叶被摘，健康叶保留；NaN 与 null 等价剔除。
- 双路径共用 acceptance_reasons（AST 架构守卫）。
- known_invalid 过滤 coverage 失败记录。
"""
from __future__ import annotations

import ast
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# ── A. library / acceptance 门 ──────────────────────────────────────────────


def test_sparse_holdout_positive_train_is_coverage_not_sign_flip():
    from factorzen.discovery.guardrails import library_reasons

    reasons = library_reasons(
        ic_train=0.05, holdout_ic=0.0, holdout_n_days=0, holdout_min_days=60,
    )
    assert any("覆盖不足" in r for r in reasons), reasons
    assert not any("反号" in r for r in reasons), reasons
    assert reasons  # 必须拒绝


def test_sparse_holdout_negative_train_also_blocked():
    """修非对称漏洞：train<0 + holdout 无数据 也不得通过（round 8 假过关）。"""
    from factorzen.discovery.guardrails import library_reasons

    reasons = library_reasons(
        ic_train=-0.05, holdout_ic=0.0, holdout_n_days=5, holdout_min_days=60,
    )
    assert any("覆盖不足" in r for r in reasons), reasons
    assert not any("反号" in r for r in reasons), reasons
    # 明确不得是空列表（不得通过）
    assert len(reasons) >= 1


def test_sufficient_coverage_true_sign_flip_still_reported():
    from factorzen.discovery.guardrails import library_reasons

    reasons = library_reasons(
        ic_train=0.05, holdout_ic=-0.03, holdout_n_days=100, holdout_min_days=60,
    )
    assert any("反号" in r for r in reasons), reasons
    assert not any("覆盖不足" in r for r in reasons), reasons


def test_holdout_exact_zero_with_enough_days_is_no_signal_not_flip():
    from factorzen.discovery.guardrails import library_reasons

    reasons = library_reasons(
        ic_train=0.05, holdout_ic=0.0, holdout_n_days=100, holdout_min_days=60,
    )
    assert any("无信号" in r for r in reasons), reasons
    assert not any("反号" in r for r in reasons), reasons


def test_same_sign_nonzero_passes_library_when_strong():
    from factorzen.discovery.guardrails import library_reasons

    assert library_reasons(
        ic_train=0.05, holdout_ic=0.04, holdout_n_days=100,
    ) == []
    assert library_reasons(
        ic_train=-0.05, holdout_ic=-0.04, holdout_n_days=100,
    ) == []


def test_acceptance_reasons_forwards_holdout_n_days():
    """统一入口必须把 n_days 传进 library 门（非恒真：n_days 不足时 library 拒、不传则可能不同）。"""
    from factorzen.discovery.guardrails import acceptance_reasons, library_reasons

    with_days = acceptance_reasons(
        gate="library", ic_train=0.05, holdout_ic=0.0, holdout_n_days=3, holdout_min_days=60,
    )
    direct = library_reasons(
        ic_train=0.05, holdout_ic=0.0, holdout_n_days=3, holdout_min_days=60,
    )
    assert with_days == direct
    assert any("覆盖不足" in r for r in with_days)


def test_strict_gate_also_blocks_insufficient_coverage():
    from factorzen.discovery.guardrails import guardrail_reasons

    reasons = guardrail_reasons(
        ic_train=0.05, holdout_ic=0.04, dsr_pvalue=0.01,
        holdout_n_days=10, holdout_min_days=60,
    )
    assert any("覆盖不足" in r for r in reasons), reasons


# ── holdout_ic_result 携带 n_days ───────────────────────────────────────────


def _daily_panel(n_stocks=40, n_days=120, seed=1):
    rng = np.random.default_rng(seed)
    start = dt.date(2024, 1, 2)
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for s in [f"{i:06d}.SH" for i in range(n_stocks)]:
        p = 10.0
        for day in days:
            p = float(max(p * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({
                "trade_date": day, "ts_code": s, "close": p, "close_adj": p,
                "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4),
            })
    return pl.DataFrame(rows)


def test_holdout_ic_result_empty_factor_has_zero_n_days():
    from factorzen.validation.holdout import holdout_ic_result, split_holdout

    daily = _daily_panel()
    _, holdout, _ = split_holdout(daily, holdout_ratio=0.2)
    empty = pl.DataFrame({
        "trade_date": pl.Series([], dtype=pl.Date),
        "ts_code": pl.Series([], dtype=pl.Utf8),
        "factor_value": pl.Series([], dtype=pl.Float64),
    })
    res = holdout_ic_result(empty, holdout)
    assert res.n_days == 0
    # 旧 3-tuple API 仍可用
    from factorzen.validation.holdout import holdout_ic
    triple = holdout_ic(empty, holdout)
    assert len(triple) == 3


def test_holdout_ic_result_dense_factor_has_positive_n_days():
    from factorzen.validation.holdout import holdout_ic_result, split_holdout

    daily = _daily_panel(n_days=200)
    _, holdout, _ = split_holdout(daily, holdout_ratio=0.2)
    fac = (
        holdout.sort(["ts_code", "trade_date"])
        .with_columns(
            (pl.col("close_adj").shift(-1).over("ts_code") / pl.col("close_adj") - 1.0)
            .alias("factor_value")
        )
        .select(["trade_date", "ts_code", "factor_value"])
        .drop_nulls()
    )
    res = holdout_ic_result(fac, holdout)
    assert res.n_days >= 20
    assert res.ic_mean > 0.05


# ── B. 叶子健康检查 ────────────────────────────────────────────────────────


def _leaf_frame_with_dead_leaf():
    """合成帧：close 全日有值；dead_leaf 仅 mining 有值、holdout 全 null；nan_leaf 在 holdout 为 NaN。"""
    days = [dt.date(2024, 1, 2) + dt.timedelta(days=i) for i in range(20)]  # 含周末简化：用连续日
    # 用 20 个交易日构造：前 12 mining，后 8 holdout
    codes = [f"{i:06d}.SH" for i in range(40)]
    rows = []
    for day in days:
        for c in codes:
            rows.append({
                "trade_date": day,
                "ts_code": c,
                "close_adj": 10.0 + (hash(c) % 7),
                "dead_leaf": 1.0 if day < days[12] else None,
                "nan_leaf": 1.0 if day < days[12] else float("nan"),
                "healthy": float((hash((c, day.isoformat())) % 50) + 1),
            })
    return pl.DataFrame(rows), days[12]


def test_leaf_holdout_coverage_drops_null_and_nan_leaves():
    from factorzen.discovery.leaf_health import (
        filter_leaves_by_holdout_coverage,
        leaf_holdout_coverage,
    )

    df, hstart = _leaf_frame_with_dead_leaf()
    leaf_map = {
        "close": "close_adj",
        "dead": "dead_leaf",
        "nanleaf": "nan_leaf",
        "healthy": "healthy",
    }
    cov = leaf_holdout_coverage(
        df, list(leaf_map.keys()), hstart, leaf_map=leaf_map, min_cross=30,
    )
    # dead/nan：holdout 有效截面日 = 0 → 覆盖率 0
    assert cov["dead"] == 0.0
    assert cov["nanleaf"] == 0.0
    # healthy / close：holdout 每日 40 只 ≥30 → 覆盖率 1
    assert cov["healthy"] == pytest.approx(1.0)
    assert cov["close"] == pytest.approx(1.0)

    kept, excluded = filter_leaves_by_holdout_coverage(
        df, list(leaf_map.keys()), hstart, leaf_map=leaf_map,
        min_coverage=0.5, min_cross=30,
    )
    assert "dead" in excluded and "nanleaf" in excluded
    assert "healthy" in kept and "close" in kept
    assert "dead" not in kept


def test_leaf_filter_fails_open_when_all_leaves_below_threshold():
    """全叶子低于阈值 = 帧撑不起检查前提（如小 universe 截面 < min_cross）→ fail-open 不摘叶。

    真实场景：crypto top-N≈30 小池、单测合成小帧。摘光叶子会让 Hypothesis 空转，
    而逐候选的 holdout 覆盖门仍在下游兜底，fail-open 不损失安全性。
    """
    from factorzen.discovery.leaf_health import filter_leaves_by_holdout_coverage

    df, hstart = _leaf_frame_with_dead_leaf()  # 截面 40 只
    leaf_map = {"close": "close_adj", "healthy": "healthy"}
    # min_cross=50 > 截面 40 → 所有叶子覆盖率 0 → 触发 fail-open
    kept, excluded = filter_leaves_by_holdout_coverage(
        df, list(leaf_map.keys()), hstart, leaf_map=leaf_map,
        min_coverage=0.5, min_cross=50,
    )
    assert kept == ["close", "healthy"]
    assert excluded == {}


# ── C. known_invalid 过滤 coverage 失败 ─────────────────────────────────────


def test_known_invalid_excludes_holdout_coverage_failures(tmp_path: Path):
    from factorzen.agents.experiment_index import ExperimentIndex

    idx = ExperimentIndex(str(tmp_path / "idx.jsonl"))
    idx.append([
        {
            "expression": "ts_mean(north_ratio, 5)",
            "ic_train": 0.02,
            "passed": False,
            "compile_ok": True,
            "reject_category": "holdout_coverage",
            "reject_reason": "holdout覆盖不足(days=0/需60)",
        },
        {
            "expression": "rank(vol)",
            "ic_train": 0.001,
            "passed": False,
            "compile_ok": True,
            "reject_reason": "train_IC 太弱(|0.0010|<0.015)",
        },
    ])
    inv = idx.known_invalid(k=10)
    assert "rank(vol)" in inv
    assert "ts_mean(north_ratio, 5)" not in inv
    assert not any("north_ratio" in e for e in inv)


# ── Critic 输入含 n_holdout_days ────────────────────────────────────────────


def test_critique_prompt_includes_n_holdout_days():
    from factorzen.agents.roles.critic import critique

    seen: list[str] = []

    def fake_llm(messages):
        seen.append(messages[-1]["content"])
        return '{"verdict":"keep","reason":"ok"}'

    critique(
        {
            "expression": "rank(close)",
            "hypothesis": "h",
            "ic_train": 0.05,
            "holdout_ic": 0.04,
            "n_holdout_days": 12,
            "dsr": 0.5,
            "dsr_pvalue": 0.2,
        },
        fake_llm,
    )
    assert seen, "LLM 应被调用"
    assert "n_holdout_days" in seen[0] or "holdout 有效天数" in seen[0]
    assert "12" in seen[0]


# ── 双路径架构守卫 ──────────────────────────────────────────────────────────


def test_dual_path_guardrails_share_acceptance_reasons():
    """nodes.py 与 mining_session.py 必须调用共享 acceptance_reasons，禁止各自复制判定。"""
    root = Path(__file__).resolve().parents[1] / "src" / "factorzen"
    paths = {
        "nodes": root / "agents" / "nodes.py",
        "mining_session": root / "discovery" / "mining_session.py",
    }
    for name, path in paths.items():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and (
                (isinstance(n.func, ast.Name) and n.func.id == "acceptance_reasons")
                or (isinstance(n.func, ast.Attribute) and n.func.attr == "acceptance_reasons")
            )
        ]
        assert calls, f"{name} 必须调用 acceptance_reasons，不得自写护栏判定"

    # 禁止在两处内联「holdout 反号」字符串拼接（应来自 guardrails）
    for name, path in paths.items():
        src = path.read_text(encoding="utf-8-sig")
        assert "holdout 反号" not in src, f"{name} 不得内联反号文案（应走共享 guardrails）"
        assert "holdout覆盖不足" not in src, f"{name} 不得内联覆盖不足文案"

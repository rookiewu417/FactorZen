"""S1：挖掘目标残差化——对库增量 IC。

覆盖：
1. 合成正交候选 residual_ic ≈ raw_ic → 放行
2. 合成冗余候选 residual_ic ≈ 0 → 残差IC太弱
3. 混合候选 residual < raw 但 > floor → 放行 + 双指标落盘
4. 空库 → 自动退化 raw，零回归
5. 日守卫：有效行 < k+10 → 跳过；全日不足 → NaN+n_days=0 → 覆盖门
6. PIT 结构守卫：单日残差签名 + 跨日污染反例
7. 双路径共用 compute_residual_ic（架构守卫）
8. CLI --objective 透传
"""
from __future__ import annotations

import ast
import datetime as dt
import inspect
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "factorzen"


# ── 合成工具 ────────────────────────────────────────────────────────────────


def _dates(n: int = 80) -> list[dt.date]:
    days, d = [], dt.date(2021, 1, 4)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days


def _codes(n: int = 50) -> list[str]:
    return [f"{600000 + i:06d}.SH" for i in range(n)]


def _panel_from_matrix(
    M: np.ndarray, dates: list, codes: list, *, name: str = "factor_value",
) -> pl.DataFrame:
    """M: (n_dates, n_stocks) → long panel。"""
    rows = []
    for i, d in enumerate(dates):
        for j, c in enumerate(codes):
            rows.append({"trade_date": d, "ts_code": c, name: float(M[i, j])})
    return pl.DataFrame(rows)


def _fwd_from_signal(signal: np.ndarray, dates: list, codes: list, rng, noise=0.5):
    """用 signal + 噪声构造 fwd_ret_1d，使 raw Spearman 可控且非平凡。"""
    ret = signal + rng.normal(0, noise, size=signal.shape)
    return _panel_from_matrix(ret, dates, codes, name="fwd_ret_1d")


def _raw_rank_ic(factor: pl.DataFrame, fwd: pl.DataFrame) -> float:
    from factorzen.daily.evaluation.ic_analysis import compute_rank_ic
    from factorzen.daily.preprocessing.normalizer import cross_sectional_zscore

    clean = cross_sectional_zscore(factor, col="factor_value").rename(
        {"factor_value_z": "factor_clean"}
    )
    res = compute_rank_ic(
        clean.select(["trade_date", "ts_code", "factor_clean"]),
        fwd, factor_col="factor_clean", frequency="daily",
    )
    return float(res.ic_mean)


# ── 1. 正交候选 ─────────────────────────────────────────────────────────────


def test_orthogonal_candidate_residual_ic_near_raw():
    """候选 = 独立 alpha + 噪声 → residual_ic ≈ raw_ic（差 < 0.3×|raw|）。"""
    from factorzen.discovery.residual import build_library_panel, compute_residual_ic

    rng = np.random.default_rng(11)
    dates, codes = _dates(60), _codes(50)
    # 库因子：与收益弱相关的方向
    lib_sig = rng.normal(0, 1, size=(len(dates), len(codes)))
    # 独立 alpha：与收益强相关，与库无关
    alpha = rng.normal(0, 1, size=(len(dates), len(codes)))
    fwd = _fwd_from_signal(alpha, dates, codes, rng, noise=0.3)
    lib_pool = {
        "lib_f1": _panel_from_matrix(lib_sig, dates, codes),
        "lib_f2": _panel_from_matrix(rng.normal(0, 1, size=lib_sig.shape), dates, codes),
    }
    cand = _panel_from_matrix(alpha + rng.normal(0, 0.05, size=alpha.shape), dates, codes)
    panel = build_library_panel(lib_pool)
    assert panel is not None and panel.k == 2
    raw = _raw_rank_ic(cand, fwd)
    res = compute_residual_ic(cand, panel, fwd)
    assert res.n_days > 0 and res.ic_mean == res.ic_mean
    assert abs(res.ic_mean - raw) < 0.3 * abs(raw) + 1e-6, (
        f"正交候选 residual={res.ic_mean:.4f} raw={raw:.4f}"
    )


# ── 2. 冗余候选 ─────────────────────────────────────────────────────────────


def test_redundant_candidate_residual_ic_near_zero():
    """候选 = 库线性组合 + 微噪声（raw 强）→ residual ≈ 0。"""
    from factorzen.discovery.residual import build_library_panel, compute_residual_ic

    rng = np.random.default_rng(22)
    dates, codes = _dates(60), _codes(50)
    f1 = rng.normal(0, 1, size=(len(dates), len(codes)))
    f2 = rng.normal(0, 1, size=f1.shape)
    combo = 0.6 * f1 + 0.4 * f2
    cand = combo + rng.normal(0, 0.01, size=f1.shape)
    # 收益跟 combo 走 → raw IC 强
    fwd = _fwd_from_signal(combo, dates, codes, rng, noise=0.2)
    lib_pool = {
        "lib_f1": _panel_from_matrix(f1, dates, codes),
        "lib_f2": _panel_from_matrix(f2, dates, codes),
    }
    panel = build_library_panel(lib_pool)
    raw = _raw_rank_ic(_panel_from_matrix(cand, dates, codes), fwd)
    res = compute_residual_ic(_panel_from_matrix(cand, dates, codes), panel, fwd)
    assert abs(raw) > 0.15, f"构造失败：raw 应强，得 {raw}"
    assert abs(res.ic_mean) < 0.05, f"冗余残差应≈0，得 residual={res.ic_mean:.4f}"


# ── 3. 混合候选 ─────────────────────────────────────────────────────────────


def test_mixed_candidate_residual_between_raw_and_floor():
    """0.7×库 + 0.3×独立 alpha → residual 显著 < raw 且 > floor。"""
    from factorzen.discovery.guardrails import DEFAULT_RESIDUAL_IC_FLOOR
    from factorzen.discovery.residual import build_library_panel, compute_residual_ic

    rng = np.random.default_rng(33)
    dates, codes = _dates(70), _codes(50)
    f1 = rng.normal(0, 1, size=(len(dates), len(codes)))
    alpha = rng.normal(0, 1, size=f1.shape)
    cand = 0.7 * f1 + 0.3 * alpha
    # 收益跟整候选走 → raw 强；残差应保留 alpha 分量
    fwd = _fwd_from_signal(cand, dates, codes, rng, noise=0.25)
    lib_pool = {"lib_f1": _panel_from_matrix(f1, dates, codes)}
    panel = build_library_panel(lib_pool)
    fdf = _panel_from_matrix(cand, dates, codes)
    raw = _raw_rank_ic(fdf, fwd)
    res = compute_residual_ic(fdf, panel, fwd)
    assert res.n_days > 0
    assert abs(res.ic_mean) < abs(raw) - 0.02, (
        f"混合：residual 应显著 < raw；r={res.ic_mean:.4f} raw={raw:.4f}"
    )
    assert abs(res.ic_mean) > DEFAULT_RESIDUAL_IC_FLOOR, (
        f"混合：residual 应 > floor；r={res.ic_mean:.4f}"
    )


# ── 4. 空库退化 ─────────────────────────────────────────────────────────────


def test_empty_library_resolves_to_raw():
    from factorzen.discovery.residual import build_library_panel, resolve_objective

    assert resolve_objective("residual", lib_nonempty=False) == "raw"
    assert resolve_objective("raw", lib_nonempty=True) == "raw"
    assert resolve_objective("residual", lib_nonempty=True) == "residual"
    assert resolve_objective(None, lib_nonempty=True) == "residual"
    assert resolve_objective(None, lib_nonempty=False) == "raw"
    assert build_library_panel({}) is None
    assert build_library_panel(None) is None


def test_empty_library_node_guardrails_zero_regression(tmp_path, monkeypatch):
    """空库 + objective=residual → 行为与 raw 一致（无 residual 字段强制门）。"""
    from factorzen.agents.nodes import node_guardrails
    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import HoldoutICResult
    from factorzen.validation.multiple_testing import TrialLedger

    rng = np.random.default_rng(1)
    dates, codes = _dates(40), _codes(35)
    # 构造简单 daily 帧
    rows = []
    for c in codes:
        base = 10.0
        for i, d in enumerate(dates):
            px = base * (1 + 0.001 * i) + rng.normal(0, 0.05)
            rows.append({
                "trade_date": d, "ts_code": c,
                "close": px, "open": px, "high": px * 1.01, "low": px * 0.99,
                "close_adj": px, "open_adj": px, "high_adj": px * 1.01, "low_adj": px * 0.99,
                "pre_close": px, "vol": 1e6, "amount": 1e7,
            })
    daily = pl.DataFrame(rows)
    bundle = DataBundle.build(daily)
    monkeypatch.setattr(
        "factorzen.validation.holdout.holdout_ic_result",
        lambda fdf, hdf: HoldoutICResult(0.05, 0.5, (0.01, 0.09), n_days=100),
    )
    state = AgentState(seed=1)
    state.attempts.append(AttemptRecord(
        iteration=0, hypothesis="h", expression="rank(close)",
        compile_ok=True, ic_train=0.05, passed_guardrails=False,
        critic_verdict=None, error=None, ir_train=0.4, turnover=0.3, n_train=80,
    ))
    node_guardrails(
        state, daily=daily, holdout_df=daily, bundle=bundle,
        ledger=TrialLedger(), top_k=3, lib_pool={}, objective="residual",
    )
    assert len(state.candidates) == 1
    c = state.candidates[0]
    # 空库 residual 退化：不应以残差字段作为准入门槛
    assert c["holdout_ic"] == pytest.approx(0.05)
    assert "residual_ic_train" not in c or c.get("residual_ic_train") is None


# ── 5. 日守卫 ───────────────────────────────────────────────────────────────


def test_day_guard_skips_thin_cross_section():
    """某日候选有效行 < k+10 → 不进序列；全日不足 → NaN + n_days=0。"""
    from factorzen.discovery.residual import (
        _day_min_samples,
        build_library_panel,
        compute_residual_ic,
    )

    dates = _dates(10)
    codes = _codes(40)
    k = 5
    # k+10 = 15；只给每只股票 8 个有效值 → 全日跳过
    rng = np.random.default_rng(5)
    lib_pool = {}
    for j in range(k):
        M = rng.normal(0, 1, size=(len(dates), len(codes)))
        lib_pool[f"f{j}"] = _panel_from_matrix(M, dates, codes)
    # 候选只在前 8 只股票有值
    thin = np.full((len(dates), len(codes)), np.nan)
    thin[:, :8] = rng.normal(0, 1, size=(len(dates), 8))
    cand = _panel_from_matrix(thin, dates, codes)
    # drop nan rows for panel (factor_value filter)
    cand = cand.filter(pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
    fwd_M = rng.normal(0, 1, size=(len(dates), len(codes)))
    fwd = _panel_from_matrix(fwd_M, dates, codes, name="fwd_ret_1d")
    panel = build_library_panel(lib_pool)
    assert _day_min_samples(k) == max(30, k + 10)
    res = compute_residual_ic(cand, panel, fwd)
    assert res.n_days == 0
    assert res.ic_mean != res.ic_mean  # NaN


def test_day_guard_coverage_reason_text():
    """n_days=0 走覆盖门 → 死因含覆盖不足（残差口径）。"""
    from factorzen.discovery.guardrails import (
        DEFAULT_HOLDOUT_MIN_DAYS,
        DEFAULT_RESIDUAL_IC_FLOOR,
        acceptance_reasons,
    )

    reasons = acceptance_reasons(
        gate="library",
        ic_train=0.05,
        holdout_ic=float("nan"),
        ic_floor=DEFAULT_RESIDUAL_IC_FLOOR,
        holdout_n_days=0,
        holdout_min_days=DEFAULT_HOLDOUT_MIN_DAYS,
        reason_style="residual",
    )
    assert any("覆盖不足" in r for r in reasons)


def test_residual_weak_and_sign_reason_text():
    from factorzen.discovery.guardrails import (
        DEFAULT_RESIDUAL_IC_FLOOR,
        acceptance_reasons,
    )

    weak = acceptance_reasons(
        gate="library", ic_train=0.001, holdout_ic=0.001,
        ic_floor=DEFAULT_RESIDUAL_IC_FLOOR, holdout_n_days=100,
        reason_style="residual",
    )
    assert any("残差IC太弱" in r for r in weak)

    flip = acceptance_reasons(
        gate="library", ic_train=0.05, holdout_ic=-0.03,
        ic_floor=DEFAULT_RESIDUAL_IC_FLOOR, holdout_n_days=100,
        reason_style="residual",
    )
    assert any("残差holdout反号" in r for r in flip)


# ── 6. PIT 结构守卫 ─────────────────────────────────────────────────────────


def test_residualize_cross_section_is_single_day_only():
    """签名只接受 1D y + 2D X（单日截面），无 date 维。"""
    from factorzen.discovery.residual import residualize_cross_section

    sig = inspect.signature(residualize_cross_section)
    params = list(sig.parameters)
    assert params == ["y", "X"], f"禁止扩展跨日参数，实得 {params}"
    y = np.array([1.0, 2.0, 3.0, 4.0])
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    r = residualize_cross_section(y, X)
    assert r.shape == (4,)
    # 完美共线 + 截距 → 残差≈0
    assert np.allclose(r, 0.0, atol=1e-8)


def test_pit_no_cross_day_fit_counterexample():
    """跨日污染反例：改 day1 库截面不改变 day0 残差（证明无跨日联合拟合）。

    注意：z-score 对仿射变换不变，污染必须改变截面**形状**（非常数平移/缩放）。
    """
    from factorzen.discovery.residual import (
        build_library_panel,
        residualize_cross_section,
    )

    rng = np.random.default_rng(99)
    dates = _dates(2)
    codes = _codes(40)
    X0 = rng.normal(0, 1, size=(len(codes),))
    X1 = rng.normal(0, 1, size=(len(codes),))
    # 改变截面形状（置换 + 非线性），而非 X1+c（z-score 后不变）
    X1_polluted = np.sin(X1) + rng.normal(0, 0.5, size=X1.shape)
    y0 = 0.5 * X0 + rng.normal(0, 0.1, size=len(codes))
    y1 = 0.5 * X1 + rng.normal(0, 0.1, size=len(codes))

    # 单日：用错误日的 X 拟合 y → 残差不同（跨日错误做法可观测）
    r0_a = residualize_cross_section(y0, X0.reshape(-1, 1))
    r0_wrong = residualize_cross_section(y0, X1.reshape(-1, 1))
    assert not np.allclose(r0_a, r0_wrong, atol=1e-6), (
        "反例失效：跨日 X 居然得到相同残差"
    )

    lib_a = {
        "f": pl.DataFrame({
            "trade_date": [dates[0]] * len(codes) + [dates[1]] * len(codes),
            "ts_code": codes + codes,
            "factor_value": list(X0) + list(X1),
        })
    }
    lib_b = {
        "f": pl.DataFrame({
            "trade_date": [dates[0]] * len(codes) + [dates[1]] * len(codes),
            "ts_code": codes + codes,
            "factor_value": list(X0) + list(X1_polluted),  # 只污染 day1 形状
        })
    }
    p_a = build_library_panel(lib_a)
    p_b = build_library_panel(lib_b)
    assert p_a is not None and p_b is not None
    assert np.allclose(p_a.X[0], p_b.X[0], atol=1e-9)
    assert not np.allclose(p_a.X[1], p_b.X[1], atol=1e-6)
    # day0 残差向量必须相同（逐日独立）
    si = np.arange(len(codes))
    r_a = residualize_cross_section(y0, p_a.X[0, si, :])
    r_b = residualize_cross_section(y0, p_b.X[0, si, :])
    assert np.allclose(r_a, r_b, atol=1e-9)
    # 且 y1 未参与 day0 拟合（结构断言）
    _ = y1  # 显式：day1 候选不进入 day0 residualize


# ── 7. 双路径架构守卫 ───────────────────────────────────────────────────────


def test_residual_shared_function_architecture_guard():
    """M1 与 team 必须调用同一 residual 入口。"""
    shared = {"compute_residual_ic", "build_library_panel", "resolve_objective"}
    for rel in ("agents/nodes.py", "discovery/mining_session.py"):
        tree = ast.parse((_SRC / rel).read_text(encoding="utf-8-sig"))
        called = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name):
                    called.add(n.func.id)
                elif isinstance(n.func, ast.Attribute):
                    called.add(n.func.attr)
        assert called & shared, (
            f"{rel} 未调用共享残差函数 {shared}；实得 ∩={called & shared}"
        )


# ── 8. CLI 透传 ─────────────────────────────────────────────────────────────


def test_cli_objective_flag_on_three_mine_commands():
    from factorzen.cli.main import build_parser

    parser = build_parser()
    for cmd in ("search", "agent", "team"):
        args = parser.parse_args(
            ["mine", cmd, "--start", "20240101", "--end", "20240601",
             "--objective", "raw"]
        )
        assert args.objective == "raw"
        args2 = parser.parse_args(
            ["mine", cmd, "--start", "20240101", "--end", "20240601"]
        )
        assert args2.objective == "residual"


def test_cli_objective_wired_to_run_session_signature():
    """capability↔wiring：run_session / run_team_agent / run_llm_agent 接收 objective。"""
    from factorzen.agents.orchestrator import run_llm_agent
    from factorzen.agents.team_orchestrator import run_team_agent
    from factorzen.discovery.mining_session import run_session

    for fn in (run_session, run_team_agent, run_llm_agent):
        params = inspect.signature(fn).parameters
        assert "objective" in params, f"{fn.__name__} 缺 objective 参数"
        # 默认 residual
        default = params["objective"].default
        assert default in ("residual", None) or default == "residual"


def test_default_residual_ic_floor_constant():
    from factorzen.discovery.guardrails import (
        DEFAULT_IC_FLOOR,
        DEFAULT_RESIDUAL_IC_FLOOR,
    )
    assert DEFAULT_RESIDUAL_IC_FLOOR == 0.010
    assert DEFAULT_RESIDUAL_IC_FLOOR < DEFAULT_IC_FLOOR


# ── 9. 冗余候选在护栏层被残差门拒绝（集成）────────────────────────────────


def test_node_guardrails_rejects_redundant_with_residual_reason(tmp_path, monkeypatch):
    """corr 0.3~0.7 带冗余：过库相关门，但残差 IC 太弱 → 残差IC太弱 拒绝。

    corr>0.7 由库门先拦（另测）；本用例锁定 S1 的核心增量区。
    """
    from factorzen.agents.nodes import node_guardrails
    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.discovery.residual import ResidualICResult
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import HoldoutICResult
    from factorzen.validation.multiple_testing import TrialLedger

    rng = np.random.default_rng(7)
    dates, codes = _dates(90), _codes(40)
    f1 = rng.normal(0, 1, size=(len(dates), len(codes)))
    # 中等相关冗余：0.5*lib + 独立噪声 → max|corr|≈0.5 < 0.7，残差≈噪声
    noise = rng.normal(0, 1, size=f1.shape)
    cand_M = 0.5 * f1 + 0.5 * noise
    rows = []
    for j, c in enumerate(codes):
        for i, d in enumerate(dates):
            px = 10.0 + f1[i, j]
            rows.append({
                "trade_date": d, "ts_code": c,
                "close": px, "open": px, "high": px * 1.01, "low": px * 0.99,
                "close_adj": px, "open_adj": px, "high_adj": px * 1.01, "low_adj": px * 0.99,
                "pre_close": px, "vol": 1e6 + abs(f1[i, j]) * 1e4, "amount": 1e7,
            })
    daily = pl.DataFrame(rows)
    bundle = DataBundle.build(daily)

    lib_fdf = _panel_from_matrix(f1, dates, codes)
    cand_fdf = _panel_from_matrix(cand_M, dates, codes)
    lib_pool = {"lib_signal": lib_fdf}

    monkeypatch.setattr(
        "factorzen.validation.holdout.holdout_ic_result",
        lambda fdf, hdf: HoldoutICResult(0.08, 0.6, (0.02, 0.12), n_days=100),
    )
    # 库相关固定 <0.7，迫使路径走到残差门
    monkeypatch.setattr(
        "factorzen.discovery.scoring.library_orthogonal_check",
        lambda fdf, pool, threshold=0.7: (True, 0.45, "lib_signal"),
    )
    # 残差 IC 固定过弱（独立构造，防恒真）
    monkeypatch.setattr(
        "factorzen.discovery.residual.compute_residual_ic",
        lambda *a, **k: ResidualICResult(0.001, 80),
    )

    state = AgentState(seed=1)
    state.attempts.append(AttemptRecord(
        iteration=0, hypothesis="h", expression="rank(close)",
        compile_ok=True, ic_train=0.08, passed_guardrails=False,
        critic_verdict=None, error=None, ir_train=0.5, turnover=0.2, n_train=80,
    ))
    monkeypatch.setattr(
        "factorzen.discovery.evaluation._factor_df_from_prepped",
        lambda *a, **k: cand_fdf,
    )
    monkeypatch.setattr(
        "factorzen.discovery.evaluation._preprocess_daily",
        lambda df, profile=None: df,
    )

    node_guardrails(
        state, daily=daily, holdout_df=daily, bundle=bundle,
        ledger=TrialLedger(), top_k=3, lib_pool=lib_pool, objective="residual",
    )
    assert state.candidates == []
    rejected = state.attempts[0]
    assert rejected.reject_reason and "残差IC太弱" in rejected.reject_reason

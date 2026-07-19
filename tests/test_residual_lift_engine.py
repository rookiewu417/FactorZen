"""residual_ic_v1 引擎契约：run_lift_tests / run_group_lift 残差增量口径。

期望值独立手算或经 daily_residual_rank_ic + series_lift_stats 旁路重算；
不拿被测函数的中间产物反推。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

# ── 合成面板（≤100 股 × ≤300 日）───────────────────────────────────────────


def _dates(n_days: int, start: date = date(2024, 1, 2)) -> list[str]:
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return days


def _codes(n: int) -> list[str]:
    return [f"{i:04d}.SZ" for i in range(n)]


def _long_panel(
    dates: list[str],
    codes: list[str],
    M: np.ndarray,
    *,
    col: str = "factor_value",
) -> pl.DataFrame:
    """M: (n_dates, n_stocks) → long [trade_date, ts_code, col]."""
    rows = []
    for i, d in enumerate(dates):
        for j, c in enumerate(codes):
            rows.append({"trade_date": d, "ts_code": c, col: float(M[i, j])})
    return pl.DataFrame(rows)


def _synth_lib_cand_ret(
    *,
    n_days: int = 60,
    n_stocks: int = 60,
    seed: int = 0,
    mode: str = "orthogonal_signal",
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame, pl.DataFrame]:
    """构造 2 库因子 + 1 候选 + 收益。

    mode:
    - orthogonal_signal: 候选 = 正交噪声 + 与 ret 强相关的分量
    - collinear: 候选 = 2*f0 + 3（严格线性变换库因子）
    - collinear_large_scale: 同 collinear 但整体 ×1e7（模拟未归一化的大量级表达式，
      如千元计的 amount / 万元计的 total_mv 组合）。残差仍是舍入噪声，但其**绝对**
      std 会越过 ``spearman_avg_rank`` 的 1e-12 绝对守卫。
    - ortho_noise: 候选与库正交、与 ret 无关
    - linear_combo_plus_noise: 候选 = a*f0+b*f1 + 正交噪声（端到端一致性）
    """
    rng = np.random.default_rng(seed)
    dates = _dates(n_days)
    codes = _codes(n_stocks)
    # 库因子（近似正交）
    f0 = rng.standard_normal((n_days, n_stocks))
    f1 = rng.standard_normal((n_days, n_stocks))
    # Gram-Schmidt 让 f1 对 f0 日截面近似正交（降低共线）
    for di in range(n_days):
        a, b = f0[di], f1[di]
        proj = (np.dot(b, a) / (np.dot(a, a) + 1e-12)) * a
        f1[di] = b - proj

    noise = rng.standard_normal((n_days, n_stocks))
    # 再对库正交化 noise
    for di in range(n_days):
        y = noise[di]
        X = np.column_stack([np.ones(n_stocks), f0[di], f1[di]])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        noise[di] = y - X @ beta

    signal = rng.standard_normal((n_days, n_stocks))
    for di in range(n_days):
        y = signal[di]
        X = np.column_stack([np.ones(n_stocks), f0[di], f1[di]])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        signal[di] = y - X @ beta

    if mode == "collinear":
        cand_M = 2.0 * f0 + 3.0
        ret_M = 0.5 * f0 + 0.3 * f1 + 0.2 * rng.standard_normal((n_days, n_stocks))
    elif mode == "collinear_large_scale":
        # 非整系数 + 大量级：残差相对量级仍是 ~1e-16，但绝对 std 越过 1e-12 守卫
        cand_M = (2.7183 * f0 - 1.4142 * f1 + 3.1416) * 1e7
        ret_M = 0.5 * f0 + 0.3 * f1 + 0.2 * rng.standard_normal((n_days, n_stocks))
    elif mode == "orthogonal_signal":
        cand_M = signal
        ret_M = 0.8 * signal + 0.15 * rng.standard_normal((n_days, n_stocks))
    elif mode == "ortho_noise":
        cand_M = noise
        ret_M = 0.5 * f0 + 0.3 * f1 + 0.2 * rng.standard_normal((n_days, n_stocks))
    elif mode == "linear_combo_plus_noise":
        cand_M = 0.4 * f0 + 0.3 * f1 + noise
        ret_M = 0.7 * noise + 0.2 * rng.standard_normal((n_days, n_stocks))
    else:
        raise ValueError(mode)

    active = {
        "lib_f0": _long_panel(dates, codes, f0),
        "lib_f1": _long_panel(dates, codes, f1),
    }
    cand = _long_panel(dates, codes, cand_M)
    ret = _long_panel(dates, codes, ret_M, col="ret")
    return active, cand, ret


def _independent_residual_mean_ic(
    cand: pl.DataFrame,
    active: dict[str, pl.DataFrame],
    ret: pl.DataFrame,
    *,
    block_days: int = 20,
) -> tuple[float, dict]:
    """**接线 parity**（非 ground-truth）：手工串起 run_lift_tests 内部同一条链。

    判别力边界要说清楚：本函数调用的正是生产的
    ``build_library_panel`` / ``daily_residual_rank_ic`` / ``series_lift_stats``，
    因此它只能证明 ``run_lift_tests`` **把参数传对了**（ret_col、评分窗、projector、
    block_days），**不能**证明残差数学本身正确——那三个函数一起错，本断言照样绿。

    残差数学的独立验算在
    ``tests/test_series_lift_stats.py::test_daily_residual_rank_ic_matches_independent_lstsq``
    （测试内自写 numpy lstsq + spearman，不经生产残差路径）。两处合起来才是完整覆盖。
    """
    from factorzen.discovery.lift_test import series_lift_stats
    from factorzen.discovery.residual import (
        ResidualProjector,
        build_library_panel,
        daily_residual_rank_ic,
    )

    panel = build_library_panel(active)
    assert panel is not None and panel.k > 0
    proj = ResidualProjector(panel)
    daily = daily_residual_rank_ic(
        cand, panel, ret, ret_col="ret", projector=proj,
    )
    assert not daily.is_empty()
    stats = series_lift_stats(daily, block_days=block_days)
    return float(stats["lift"]), stats


# ── 1. 端到端小样本 ──────────────────────────────────────────────────────────


def test_run_lift_tests_e2e_matches_independent_residual_ic():
    """lift 与独立旁路残差 IC 均值一致；lift_metric / baseline 契约。"""
    from factorzen.discovery.lift_test import run_lift_tests

    active, cand, ret = _synth_lib_cand_ret(
        n_days=60, n_stocks=60, seed=7, mode="linear_combo_plus_noise",
    )
    expected_lift, _ = _independent_residual_mean_ic(cand, active, ret, block_days=20)

    rows = run_lift_tests(
        [{"expression": "cand_mix", "residual_ic_train": 0.02}],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        lift_workers=1,
        block_days=20,
        threshold=-1.0,
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["error"] is None, r
    assert r["baseline"] is None
    assert r.get("lift_metric") == "residual_ic_v1"
    assert r["lift"] is not None
    assert abs(float(r["lift"]) - expected_lift) < 1e-9
    # 新口径：candidate_rank_ic 与 lift 同源（残差 IC 均值）
    assert abs(float(r["candidate_rank_ic"]) - float(r["lift"])) < 1e-12
    assert r.get("n_lib_factors") == 2
    assert r.get("cv_train_days") is None
    assert r.get("cv_test_days") is None


# ── 2. 共线 → 零增量 ────────────────────────────────────────────────────────


def test_collinear_candidate_near_zero_lift():
    """候选 = 2*f0+3 → 残差≈0 → |lift| < 1e-6，且**必须被拒**。

    经济含义才是重点：被库完全张成的候选零增量，绝不能入库。
    lift≈0 只是中间量——真正的契约是 ``lift_admission`` 判 reject
    （全零序列 SE=None → reject，见 ``series_lift_stats`` 全零守卫）。
    """
    from factorzen.discovery.lift_test import lift_admission, run_lift_tests

    active, cand, ret = _synth_lib_cand_ret(
        n_days=60, n_stocks=60, seed=11, mode="collinear",
    )
    rows = run_lift_tests(
        [{"expression": "col_cand"}],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        lift_workers=1,
        threshold=-1.0,
    )
    r = rows[0]
    assert r["error"] is None, r
    assert r["lift"] is not None
    assert abs(float(r["lift"])) < 1e-6, r
    # 阈值传 -1.0 时 passed 会为 True——所以 passed 不是准入契约，
    # lift_admission 才是。共线候选必须拒（否则「数量取胜」会被冗余因子灌水）。
    assert lift_admission(r, threshold=0.005, se_mult=1.0) == "reject", r


def test_large_magnitude_collinear_candidate_rejected():
    """大量级共线候选：残差是舍入噪声，绝不能被判为增量。

    回归锚（2026-07-19 实测的准入穿透）：``spearman_avg_rank`` 的退化守卫是
    **绝对**阈值 ``std < 1e-12``，而残差是否退化取决于它**相对**原值的比例。
    候选量级放大 1e7 后，同一条零增量候选的舍入残差（相对量级 ~1e-16）绝对 std
    越过 1e-12，Spearman 在纯噪声上算出 60 天日 IC，得 lift=0.0188（阈值 18 倍）、
    lift_admission=active——纯浮点噪声准入入库。

    与 ``test_collinear_candidate_near_zero_lift`` 是同一经济情形的两个数值分支，
    契约必须一致：零增量 → 拒。
    """
    from factorzen.discovery.lift_test import lift_admission, run_lift_tests

    active, cand, ret = _synth_lib_cand_ret(
        n_days=60, n_stocks=60, seed=11, mode="collinear_large_scale",
    )
    rows = run_lift_tests(
        [{"expression": "big_col_cand"}],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        lift_workers=1,
        threshold=-1.0,
    )
    r = rows[0]
    # 经济含义与小量级分支同：零增量必拒
    assert lift_admission(r, threshold=0.001, se_mult=1.0) == "reject", r
    # 且 lift 本身不得把舍入噪声报成增量
    assert r["lift"] is None or abs(float(r["lift"])) < 1e-6, r


def test_degenerate_guard_scale_invariant():
    """同一候选整体缩放不改变残差 IC 结论——退化判据必须是尺度不变的。

    这是上面那条穿透的**根因层**断言：小量级已被 1e-12 绝对守卫挡住，
    放大后就该同样被挡。用两个量级跑同一逻辑候选做对拍。
    """
    from factorzen.discovery.residual import (
        ResidualProjector,
        build_library_panel,
        daily_residual_rank_ic,
    )

    active, cand, ret = _synth_lib_cand_ret(
        n_days=40, n_stocks=60, seed=3, mode="collinear_large_scale",
    )
    panel = build_library_panel(active)
    proj = ResidualProjector.from_panel(panel)

    big = daily_residual_rank_ic(cand, panel, ret, ret_col="ret", projector=proj)
    small = daily_residual_rank_ic(
        cand.with_columns(pl.col("factor_value") / 1e7),
        panel, ret, ret_col="ret", projector=proj,
    )
    # 缩放不改变「无有效残差日」这一结论
    assert big.height == small.height, (big.height, small.height)
    assert big.is_empty(), f"大量级共线候选不应产出残差 IC 日，实得 {big.height} 天"


# ── 3. 正交强信号 → 正增量 ──────────────────────────────────────────────────


def test_orthogonal_strong_signal_positive_lift():
    """候选与库正交且与收益强相关 → lift > 0.05。"""
    from factorzen.discovery.lift_test import run_lift_tests

    active, cand, ret = _synth_lib_cand_ret(
        n_days=60, n_stocks=60, seed=22, mode="orthogonal_signal",
    )
    rows = run_lift_tests(
        [{"expression": "sig_cand"}],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        lift_workers=1,
        threshold=0.0,
    )
    r = rows[0]
    assert r["error"] is None, r
    assert r["lift"] is not None
    assert float(r["lift"]) > 0.05, r
    assert r["passed"] is True


# ── 4. no_residual_days ─────────────────────────────────────────────────────


def test_no_residual_days_when_ts_codes_outside_library():
    """候选 ts_code 全在库轴外 → error=no_residual_days 且 lift is None。"""
    from factorzen.discovery.lift_test import run_lift_tests

    active, _cand, ret = _synth_lib_cand_ret(
        n_days=40, n_stocks=40, seed=3, mode="orthogonal_signal",
    )
    dates = _dates(40)
    # 库外股票码
    foreign = [f"9{i:03d}.SH" for i in range(40)]
    M = np.random.default_rng(0).standard_normal((40, 40))
    cand_out = _long_panel(dates, foreign, M)

    rows = run_lift_tests(
        [{"expression": "out_of_axis"}],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand_out,
        lift_workers=1,
    )
    r = rows[0]
    assert r["error"] == "no_residual_days", r
    assert r["lift"] is None  # 不得为 0（历史事故：空序列静默写 0.0）


# ── 5. empty_library_panel ──────────────────────────────────────────────────


def test_empty_library_panel_error():
    """active 非空但物化不出面板 → empty_library_panel。"""
    from factorzen.discovery.lift_test import run_lift_tests

    empty_df = pl.DataFrame(
        schema={
            "trade_date": pl.Utf8,
            "ts_code": pl.Utf8,
            "factor_value": pl.Float64,
        },
    )
    # 非空 dict，值为空帧
    active = {"ghost_f": empty_df}
    dates = _dates(20)
    codes = _codes(40)
    M = np.zeros((20, 40))
    cand = _long_panel(dates, codes, M)
    ret = _long_panel(dates, codes, M, col="ret")

    rows = run_lift_tests(
        [{"expression": "c0"}],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        lift_workers=1,
    )
    assert len(rows) == 1
    assert rows[0]["error"] == "empty_library_panel"
    assert rows[0]["lift"] is None
    assert rows[0]["passed"] is False


# ── 6. 组门口径 ─────────────────────────────────────────────────────────────


def test_run_group_lift_collinear_vs_signal():
    """组内全共线 → 组 lift≈0；含正交强信号 → 组 lift>0。"""
    from factorzen.discovery.lift_test import run_group_lift

    active, cand_col, ret = _synth_lib_cand_ret(
        n_days=60, n_stocks=60, seed=5, mode="collinear",
    )
    # 第二个共线候选：3*f1 - 1
    f1 = active["lib_f1"]
    cand_col2 = f1.with_columns((pl.col("factor_value") * 3.0 - 1.0).alias("factor_value"))

    mats = {"c_col": cand_col, "c_col2": cand_col2}
    out_col = run_group_lift(
        [{"expression": "c_col"}, {"expression": "c_col2"}],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: mats[e],
        threshold=-1.0,
    )
    assert out_col["error"] is None, out_col
    assert out_col["lift"] is not None
    assert abs(float(out_col["lift"])) < 1e-5, out_col
    assert out_col.get("lift_metric") == "residual_ic_v1"
    assert "base_daily" not in out_col
    assert out_col["baseline"] is None
    assert out_col.get("n_lib_factors") == 2

    # 含正交强信号
    _, cand_sig, ret_sig = _synth_lib_cand_ret(
        n_days=60, n_stocks=60, seed=22, mode="orthogonal_signal",
    )
    # 用同一 active 轴；signal 候选独立合成，需对齐日期/股票
    active_s, cand_sig, ret_sig = _synth_lib_cand_ret(
        n_days=60, n_stocks=60, seed=22, mode="orthogonal_signal",
    )
    # 再造一个共线候选（相对 active_s）
    f0 = active_s["lib_f0"]
    cand_col_s = f0.with_columns((pl.col("factor_value") * 2.0 + 3.0).alias("factor_value"))
    mats2 = {"c_sig": cand_sig, "c_col": cand_col_s}
    out_sig = run_group_lift(
        [{"expression": "c_sig"}, {"expression": "c_col"}],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active_s,
        ret_df=ret_sig,
        materialize_candidate=lambda e: mats2[e],
        threshold=0.0,
    )
    assert out_sig["error"] is None, out_sig
    assert out_sig["lift"] is not None
    assert float(out_sig["lift"]) > 0.0, out_sig


# ── 7. 并行/串行 parity ─────────────────────────────────────────────────────


def test_parallel_serial_bit_identical():
    """lift_workers=1 与 3 结果逐位相等（残差路径确定性）。"""
    from factorzen.discovery.lift_test import run_lift_tests

    active, cand_a, ret = _synth_lib_cand_ret(
        n_days=50, n_stocks=50, seed=9, mode="orthogonal_signal",
    )
    _, cand_b, _ = _synth_lib_cand_ret(
        n_days=50, n_stocks=50, seed=10, mode="linear_combo_plus_noise",
    )
    _, cand_c, _ = _synth_lib_cand_ret(
        n_days=50, n_stocks=50, seed=11, mode="collinear",
    )
    mats = {"a": cand_a, "b": cand_b, "c": cand_c}
    grays = [
        {"expression": "a", "residual_ic_train": 0.03},
        {"expression": "b", "residual_ic_train": 0.02},
        {"expression": "c", "residual_ic_train": 0.01},
    ]
    common = dict(
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: mats[e],
        block_days=10,
        threshold=0.0,
        seed=0,
    )
    serial = run_lift_tests(grays, lift_workers=1, **common)
    parallel = run_lift_tests(grays, lift_workers=3, **common)
    assert len(serial) == len(parallel) == 3
    for a, b in zip(serial, parallel, strict=True):
        assert a["expression"] == b["expression"]
        assert a["lift"] == b["lift"]
        assert a["lift_se"] == b["lift_se"]
        assert a["baseline"] == b["baseline"]
        assert a["candidate_rank_ic"] == b["candidate_rank_ic"]
        assert a["passed"] == b["passed"]
        assert a["error"] == b["error"]
        assert a["n_blocks"] == b["n_blocks"]
        assert a["lift_first_half"] == b["lift_first_half"]
        assert a["lift_second_half"] == b["lift_second_half"]
        assert a["admission_ic"] == b["admission_ic"]
        assert a.get("lift_metric") == b.get("lift_metric") == "residual_ic_v1"

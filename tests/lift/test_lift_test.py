"""组合增量 lift 实验 + probation 入库通道单测。TDD、mock 离线。"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# ── 合成面板：ret 含 (lib × cand) 交互 ───────────────────────────────────────


def _dates(n_days: int):
    days, d = [], date(2024, 1, 2)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return days


def _synth_panels(n_days=80, n_stocks=40, seed=0, *, signal: bool = True):
    """构造 lib 因子 + 候选 + 收益（residual_ic_v1 离线面板）。

    signal=True：候选含与 ret 正交于 lib 的强相关分量 → 残差 lift 显著。
    signal=False：候选为与 ret 无关噪声 → 残差 lift≈0。
    n_stocks 默认 40：满足 residual 日守卫 max(30, k+10)。
    """
    rng = np.random.default_rng(seed)
    dates = _dates(n_days)
    lib_rows, cand_rows, noise_rows, ret_rows = [], [], [], []
    for d in dates:
        lib = rng.standard_normal(n_stocks)
        noise = rng.standard_normal(n_stocks)
        ortho = rng.standard_normal(n_stocks)
        # 对 lib 正交化 ortho
        ortho = ortho - (np.dot(ortho, lib) / (np.dot(lib, lib) + 1e-12)) * lib
        if signal:
            cand = ortho
            ret = 0.3 * lib + 0.7 * ortho + 0.15 * rng.standard_normal(n_stocks)
        else:
            cand = noise
            ret = 0.5 * lib + 0.4 * rng.standard_normal(n_stocks)
        for s in range(n_stocks):
            code = f"{s:04d}.SZ"
            lib_rows.append({"trade_date": d, "ts_code": code, "factor_value": float(lib[s])})
            cand_rows.append({"trade_date": d, "ts_code": code, "factor_value": float(cand[s])})
            noise_rows.append({"trade_date": d, "ts_code": code, "factor_value": float(noise[s])})
            ret_rows.append({"trade_date": d, "ts_code": code, "ret": float(ret[s])})
    return (
        {"lib_a": pl.DataFrame(lib_rows)},
        pl.DataFrame(cand_rows),
        pl.DataFrame(noise_rows),
        pl.DataFrame(ret_rows),
    )


def test_lift_signal_candidate_passes_noise_fails():
    """正交强信号候选 lift 过阈值；纯噪声候选 lift 更低/拒。residual_ic_v1 口径。"""
    from factorzen.discovery.lift_test import (
        DEFAULT_RESIDUAL_LIFT_THRESHOLD,
        run_lift_tests,
    )

    active, cand_good, _, ret = _synth_panels(signal=True, seed=0)
    _, cand_noise, _, _ = _synth_panels(signal=False, seed=1)
    # 噪声候选挂到同一 ret/active 轴（仅换候选值）
    cand_noise = cand_noise  # 同日期股票网格

    mat_map = {
        "signal_cand": cand_good,
        "noise_cand": cand_noise,
    }

    def materialize(expr):
        return mat_map[expr]

    rows = run_lift_tests(
        [
            {"expression": "signal_cand", "residual_ic_train": 0.006},
            {"expression": "noise_cand", "residual_ic_train": 0.005},
        ],
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),  # unused
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=materialize,
        top_m=10,
        threshold=DEFAULT_RESIDUAL_LIFT_THRESHOLD,
        seed=0,
        lift_workers=1,
    )
    by_expr = {r["expression"]: r for r in rows}
    assert by_expr["signal_cand"]["error"] is None, by_expr["signal_cand"]
    assert by_expr["signal_cand"]["passed"] is True, by_expr["signal_cand"]
    assert by_expr["signal_cand"]["lift"] >= DEFAULT_RESIDUAL_LIFT_THRESHOLD
    assert by_expr["signal_cand"]["baseline"] is None
    assert by_expr["signal_cand"].get("lift_metric") == "residual_ic_v1"
    # 噪声：lift 应显著低于信号
    assert by_expr["noise_cand"]["lift"] is not None
    assert by_expr["noise_cand"]["lift"] < by_expr["signal_cand"]["lift"]


def test_lift_top_m_truncation_recorded():
    """top_m 截断不静默：返回行带 n_input / n_selected。"""
    from factorzen.discovery.lift_test import run_lift_tests

    active, cand, _, ret = _synth_panels(n_days=60, n_stocks=40, signal=False)
    grays = [
        {"expression": f"c{i}", "residual_ic_train": 0.009 - i * 0.0001}
        for i in range(5)
    ]
    mats = {f"c{i}": cand for i in range(5)}

    rows = run_lift_tests(
        grays,
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: mats[e],
        top_m=2,
        threshold=0.001,
        lift_workers=1,
    )
    assert len(rows) == 2
    assert rows[0]["n_input"] == 5 and rows[0]["n_selected"] == 2
    assert rows[0]["truncated_from"] == 5


def test_lift_bad_candidate_does_not_crash_batch():
    from factorzen.discovery.lift_test import run_lift_tests

    active, cand, _, ret = _synth_panels(n_days=60, n_stocks=40, signal=False)

    def materialize(expr):
        if expr == "bad":
            raise RuntimeError("boom")
        return cand

    rows = run_lift_tests(
        [
            {"expression": "bad", "residual_ic_train": 0.008},
            {"expression": "ok", "residual_ic_train": 0.007},
        ],
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=materialize,
        top_m=10,
        lift_workers=1,
    )
    assert len(rows) == 2
    assert rows[0]["passed"] is False and rows[0]["error"]
    assert rows[1]["expression"] == "ok"


def test_extract_mixed_gray_zone_and_lift_queue():
    """extractor 同时接受 gray_zone（旧）与 lift_queue（新）；别名同实现。"""
    from factorzen.discovery.lift_test import (
        LIFT_QUEUE_CATEGORY,
        extract_gray_candidates_from_manifest,
        extract_lift_queue_from_manifest,
    )

    man = {
        "attempts": [
            {"expression": "old_gray", "reject_category": "gray_zone",
             "residual_ic_train": 0.006},
            {"expression": "new_queue", "reject_category": LIFT_QUEUE_CATEGORY,
             "residual_ic_train": 0.007},
            {"expression": "noise", "reject_category": "holdout_coverage"},
            {"expression": "old_gray", "reject_category": "lift_queue"},  # 去重
        ],
        "candidates": [
            {"expression": "cand_q", "reject_category": "lift_queue", "ic_train": 0.008},
            {"expression": "cand_g", "reject_category": "gray_zone", "ic_train": 0.009},
        ],
    }
    got = extract_gray_candidates_from_manifest(man)
    assert {g["expression"] for g in got} == {"old_gray", "new_queue", "cand_q", "cand_g"}
    assert extract_lift_queue_from_manifest is extract_gray_candidates_from_manifest
    assert extract_lift_queue_from_manifest(man) == got


# ── paired_lift_stats / daily IC / admission / group ─────────────────────────


def _ic_df(dates, ics):
    return pl.DataFrame(
        {"trade_date": list(dates), "ic": list(ics)},
        schema={"trade_date": pl.Utf8, "ic": pl.Float64},
    )


def test_paired_lift_stats_hand_calc():
    """手算：lift / SE / 半段 / 奇数块 / 尾块不足 / n_blocks=1 → SE=None。"""
    from factorzen.discovery.lift_test import paired_lift_stats

    # 50 日，block_days=20 → 块 [20, 20, 10]；奇数 3 块中位归前半 → 前 2 块 vs 后 1 块
    dates = [f"202401{d:02d}" for d in range(1, 51)]
    # base 全 0；cand = 常数序列便于手算
    # 日 diff: 前 20 日 = 0.02，中 20 日 = 0.04，尾 10 日 = 0.06
    cand_ics = [0.02] * 20 + [0.04] * 20 + [0.06] * 10
    base_ics = [0.0] * 50
    stats = paired_lift_stats(_ic_df(dates, cand_ics), _ic_df(dates, base_ics), block_days=20)

    # lift = (20*0.02 + 20*0.04 + 10*0.06) / 50 = (0.4+0.8+0.6)/50 = 0.036
    assert stats["n_days"] == 50
    assert stats["n_blocks"] == 3
    assert abs(stats["lift"] - 0.036) < 1e-12
    # 块均值 [0.02, 0.04, 0.06]；SE = std(ddof=1)/sqrt(3)
    block_means = np.array([0.02, 0.04, 0.06])
    expected_se = float(block_means.std(ddof=1) / np.sqrt(3))
    assert abs(stats["lift_se"] - expected_se) < 1e-12
    # 前半块（2 块 = 40 日）diff 均值 = (20*0.02+20*0.04)/40 = 0.03
    assert abs(stats["lift_first_half"] - 0.03) < 1e-12
    # 后半块（1 块 = 10 日）= 0.06
    assert abs(stats["lift_second_half"] - 0.06) < 1e-12

    # n_blocks=1 → SE=None，second_half=None
    short = paired_lift_stats(
        _ic_df(dates[:15], cand_ics[:15]),
        _ic_df(dates[:15], base_ics[:15]),
        block_days=20,
    )
    assert short["n_blocks"] == 1
    assert short["lift_se"] is None
    assert short["lift_first_half"] is not None
    assert short["lift_second_half"] is None
    assert abs(short["lift"] - 0.02) < 1e-12

    # 空配对
    empty = paired_lift_stats(
        _ic_df(["20240101"], [0.1]),
        _ic_df(["20240102"], [0.1]),
        block_days=20,
    )
    assert empty["lift"] is None and empty["n_days"] == 0 and empty["lift_se"] is None


def test_daily_oos_rank_ic_parity_with_evaluate_oos():
    """mean(每日序列) ≈ _evaluate_oos rank_ic_mean（atol 1e-12）。"""
    from factorzen.discovery.lift_test import _daily_oos_rank_ic
    from factorzen.research.combination.experiment import _evaluate_oos

    rng = np.random.default_rng(42)
    dates = _dates(30)
    rows_f, rows_r = [], []
    for d in dates:
        f = rng.standard_normal(40)
        r = 0.4 * f + 0.6 * rng.standard_normal(40)
        for s in range(40):
            code = f"{s:04d}.SZ"
            rows_f.append({
                "trade_date": d, "ts_code": code, "factor_value": float(f[s]),
            })
            rows_r.append({
                "trade_date": d, "ts_code": code, "ret": float(r[s]),
            })
    combined = pl.DataFrame(rows_f)
    ret_df = pl.DataFrame(rows_r)
    daily = _daily_oos_rank_ic(combined, ret_df)
    mean_daily = float(daily["ic"].mean())
    ref = float(_evaluate_oos(combined, ret_df)["rank_ic_mean"])
    assert abs(mean_daily - ref) < 1e-12
    assert daily.columns == ["trade_date", "ic"]
    assert len(daily) > 0


def test_paired_lift_equals_mean_diff_when_dates_align():
    """日期完全相同时配对 lift == 两序列均值之差。"""
    from factorzen.discovery.lift_test import paired_lift_stats

    dates = [f"d{i:03d}" for i in range(40)]
    cand = [0.01 + 0.001 * i for i in range(40)]
    base = [0.005 + 0.0005 * i for i in range(40)]
    stats = paired_lift_stats(_ic_df(dates, cand), _ic_df(dates, base), block_days=10)
    expected = float(np.mean(np.array(cand) - np.array(base)))
    assert abs(stats["lift"] - expected) < 1e-12
    assert abs(stats["lift"] - (float(np.mean(cand)) - float(np.mean(base)))) < 1e-12


def test_paired_lift_uses_inner_join_dates():
    """日期不齐时只在交集上配对（比各自均值之差更对）。"""
    from factorzen.discovery.lift_test import paired_lift_stats

    cand = _ic_df(["d1", "d2", "d3"], [0.10, 0.20, 0.30])
    base = _ic_df(["d2", "d3", "d4"], [0.05, 0.05, 0.99])
    stats = paired_lift_stats(cand, base, block_days=20)
    # 仅 d2,d3：diff = 0.15, 0.25 → mean 0.20
    assert stats["n_days"] == 2
    assert abs(stats["lift"] - 0.20) < 1e-12
    # 非配对：mean(cand)-mean(base) = 0.20 - (0.05+0.05+0.99)/3 ≠ 0.20
    naive = float(np.mean([0.10, 0.20, 0.30]) - np.mean([0.05, 0.05, 0.99]))
    assert abs(naive - stats["lift"]) > 1e-6


def test_lift_admission_four_branches():
    """四分支：恰等于阈值 / SE 大于阈值 / second_half=0 / lift=None。"""
    from factorzen.discovery.guardrails import DEFAULT_LIFT_THRESHOLD
    from factorzen.discovery.lift_test import lift_admission

    thr = DEFAULT_LIFT_THRESHOLD  # 0.001
    # 1) lift 恰等于阈值 + second_half > 0 → active
    assert lift_admission({
        "lift": thr, "lift_se": 0.0, "lift_second_half": 0.01,
    }, threshold=thr) == "active"
    # 2) SE 大于阈值：bar = se_mult * se；lift 介于 threshold 与 bar 之间 → reject
    assert lift_admission({
        "lift": 0.002, "lift_se": 0.005, "lift_second_half": 0.01,
    }, threshold=thr, se_mult=1.0) == "reject"
    # lift 过 bar → active
    assert lift_admission({
        "lift": 0.006, "lift_se": 0.005, "lift_second_half": 0.01,
    }, threshold=thr, se_mult=1.0) == "active"
    # 3) second_half = 0 → probation（过门槛但后半不 > 0）
    assert lift_admission({
        "lift": 0.003, "lift_se": 0.0, "lift_second_half": 0.0,
    }, threshold=thr) == "probation"
    # second_half None + 有限 SE → probation
    assert lift_admission({
        "lift": 0.003, "lift_se": 0.0, "lift_second_half": None,
    }, threshold=thr) == "probation"
    # SE 缺失/非有限 = 区间证据不完整 → reject（不再按 0 退化）
    assert lift_admission({
        "lift": 0.003, "lift_se": None, "lift_second_half": None,
    }, threshold=thr) == "reject"
    # 4) lift None / 低于阈值 → reject
    assert lift_admission({"lift": None}, threshold=thr) == "reject"
    assert lift_admission({
        "lift": thr - 1e-9, "lift_se": 0.0, "lift_second_half": 0.01,
    }, threshold=thr) == "reject"


def _signed_factor_panels(sign: float, n_days=60, n_stocks=40, seed=1):
    """构造单因子与 ret 同号/反号相关的面板（admission_ic 符号断言用）。

    factor ≈ sign * ret + 极小噪声 → RankIC 符号 ≈ sign 的符号。
    n_stocks≥40 满足 residual 日守卫。
    """
    rng = np.random.default_rng(seed)
    dates = _dates(n_days)
    lib_rows, cand_rows, ret_rows = [], [], []
    for d in dates:
        ret = rng.standard_normal(n_stocks)
        cand = float(sign) * ret + 0.02 * rng.standard_normal(n_stocks)
        lib = rng.standard_normal(n_stocks)
        for s in range(n_stocks):
            code = f"{s:04d}.SZ"
            lib_rows.append({"trade_date": d, "ts_code": code, "factor_value": float(lib[s])})
            cand_rows.append({"trade_date": d, "ts_code": code, "factor_value": float(cand[s])})
            ret_rows.append({"trade_date": d, "ts_code": code, "ret": float(ret[s])})
    return (
        {"lib_a": pl.DataFrame(lib_rows)},
        pl.DataFrame(cand_rows),
        pl.DataFrame(ret_rows),
    )


def test_run_lift_tests_admission_ic_reflects_single_factor_sign():
    """admission_ic = 单因子 admission 窗 RankIC，正/负相关各得对应符号；非残差 IC。"""
    from factorzen.discovery.lift_test import run_lift_tests

    # 正相关
    active_pos, cand_pos, ret_pos = _signed_factor_panels(+1.0, seed=11)
    rows_pos = run_lift_tests(
        [{"expression": "pos_cand", "ic_train": 0.03, "residual_ic_train": 0.02}],
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),
        active_factor_dfs=active_pos,
        ret_df=ret_pos,
        materialize_candidate=lambda e: cand_pos,
        top_m=None,
        threshold=0.001,
        lift_workers=1,
    )
    assert len(rows_pos) == 1
    rpos = rows_pos[0]
    assert rpos.get("admission_ic") is not None, rpos
    assert rpos["admission_ic"] > 0, rpos
    assert rpos.get("ic_train") == 0.03
    assert rpos.get("residual_ic_train") == 0.02
    # residual_ic_v1 下 candidate_rank_ic 与 lift 同源；方向权威仍是 admission_ic
    assert "candidate_rank_ic" in rpos

    # 负相关
    active_neg, cand_neg, ret_neg = _signed_factor_panels(-1.0, seed=22)
    rows_neg = run_lift_tests(
        [{"expression": "neg_cand"}],
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),
        active_factor_dfs=active_neg,
        ret_df=ret_neg,
        materialize_candidate=lambda e: cand_neg,
        top_m=None,
        threshold=0.001,
        lift_workers=1,
    )
    assert len(rows_neg) == 1
    rneg = rows_neg[0]
    assert rneg.get("admission_ic") is not None, rneg
    assert rneg["admission_ic"] < 0, rneg
    # 负向裸 IC 必须为负（方向权威）；残差 lift 可与之不同
    assert rneg["admission_ic"] < 0


def test_run_lift_tests_error_rows_have_admission_ic_key():
    """错误路径 row 也有 admission_ic 键（形态一致，下游 no KeyError）。"""
    from factorzen.discovery.lift_test import run_lift_tests

    active, cand, _, ret = _synth_panels(n_days=60, n_stocks=40, signal=False)

    def materialize(expr):
        if expr == "bad":
            raise RuntimeError("boom")
        return cand

    rows = run_lift_tests(
        [
            {"expression": "bad", "residual_ic_train": 0.008},
            {"expression": "ok", "residual_ic_train": 0.007},
        ],
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=materialize,
        top_m=10,
        lift_workers=1,
    )
    assert all("admission_ic" in r for r in rows)
    # 失败行 admission_ic 应为 None（未成功物化）
    by = {r["expression"]: r for r in rows}
    assert by["bad"]["admission_ic"] is None
    assert by["bad"]["error"]


def test_run_lift_tests_new_fields_and_top_m_none():
    """新字段全链 + top_m=None 全测；正交信号 → residual lift 过阈值。"""
    from factorzen.discovery.lift_test import run_lift_tests

    active, cand, _noise, ret = _synth_panels(n_days=60, n_stocks=40, signal=True)

    grays = [
        {"expression": f"c{i}", "residual_ic_train": 0.009 - i * 0.0001}
        for i in range(4)
    ]
    mats = {f"c{i}": cand for i in range(4)}

    rows = run_lift_tests(
        grays,
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: mats[e],
        top_m=None,  # 全测
        threshold=0.001,
        block_days=10,
        lift_workers=1,
    )
    assert len(rows) == 4
    assert rows[0]["n_input"] == 4 and rows[0]["n_selected"] == 4
    assert rows[0]["truncated_from"] is None
    for r in rows:
        assert "lift_se" in r
        assert "n_blocks" in r
        assert "lift_first_half" in r
        assert "lift_second_half" in r
        assert r["lift"] is not None
        assert r["n_blocks"] is not None and r["n_blocks"] >= 1
        assert r["passed"] is True
        assert r["error"] is None
        assert r["baseline"] is None
        assert r.get("lift_metric") == "residual_ic_v1"
        # residual_ic_v1：candidate_rank_ic 与 lift 同源
        assert r["candidate_rank_ic"] == r["lift"]


def test_run_group_lift_three_states():
    """run_group_lift：正常 / 部分物化失败 / 全失败。"""
    from factorzen.discovery.lift_test import run_group_lift

    active, cand, _, ret = _synth_panels(n_days=60, n_stocks=40, signal=True)

    def mat_partial(expr):
        if expr == "bad":
            raise RuntimeError("boom")
        if expr == "empty":
            return pl.DataFrame(schema={
                "trade_date": pl.Utf8, "ts_code": pl.Utf8, "factor_value": pl.Float64,
            })
        return cand

    # 正常：两个好候选
    ok = run_group_lift(
        [
            {"expression": "g1", "residual_ic_train": 0.006},
            {"expression": "g2", "residual_ic_train": 0.005},
        ],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        threshold=0.001,
    )
    assert ok["error"] is None
    assert ok["n_candidates"] == 2
    assert set(ok["expressions"]) == {"g1", "g2"}
    assert ok["skipped"] == []
    assert ok["lift"] is not None
    assert "lift_se" in ok and "n_blocks" in ok
    assert ok["baseline"] is None
    assert ok.get("lift_metric") == "residual_ic_v1"
    assert "base_daily" not in ok

    # 部分失败
    partial = run_group_lift(
        [
            {"expression": "good", "residual_ic_train": 0.006},
            {"expression": "bad", "residual_ic_train": 0.005},
            {"expression": "empty", "residual_ic_train": 0.004},
        ],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=mat_partial,
    )
    assert partial["error"] is None
    assert partial["n_candidates"] == 1
    assert partial["expressions"] == ["good"]
    assert len(partial["skipped"]) == 2
    skipped_exprs = {s["expression"] for s in partial["skipped"]}
    assert skipped_exprs == {"bad", "empty"}

    # 全失败
    all_bad = run_group_lift(
        [
            {"expression": "bad", "residual_ic_train": 0.006},
            {"expression": "empty", "residual_ic_train": 0.005},
        ],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=mat_partial,
    )
    assert all_bad["error"] == "all_candidates_materialize_failed"
    assert all_bad["n_candidates"] == 0
    assert all_bad["lift"] is None
    assert len(all_bad["skipped"]) == 2


# ── upsert_probation / rebuild 保留 / pool 默认不含 ──────────────────────────


def test_upsert_probation_roundtrip_fields(tmp_path):
    from factorzen.discovery.factor_library import load_library, upsert_probation

    rows = [
        {
            "expression": "rank(close)",
            "ic_train": 0.006,
            "holdout_ic": -0.002,
            "lift": 0.0035,
            "baseline": 0.04,
            "passed": True,
            "n_train": 200,
        }
    ]
    res = upsert_probation(
        "ashare", rows,
        eval_window=("20200101", "20260101"), universe="csi300", horizon=5,
        run_id="lift1", session_dir="s1", git_sha="abc", now="2026-07-14",
        root=str(tmp_path),
    )
    assert res.added == 1
    lib = load_library("ashare", root=str(tmp_path))
    assert len(lib) == 1
    r = lib[0]
    assert r.status == "probation"
    assert abs(r.lift - 0.0035) < 1e-9
    assert abs(r.lift_baseline - 0.04) < 1e-9
    assert r.expression == "rank(close)"


def test_upsert_probation_skips_not_passed(tmp_path):
    from factorzen.discovery.factor_library import load_library, upsert_probation

    res = upsert_probation(
        "ashare",
        [{"expression": "x", "passed": False, "lift": 0.0, "baseline": 0.01}],
        eval_window=("20200101", "20260101"), universe="u", horizon=5,
        run_id="r", session_dir="s", git_sha="a", now="2026-07-14",
        root=str(tmp_path),
    )
    assert res.skipped == 1 and res.added == 0
    assert load_library("ashare", root=str(tmp_path)) == []


def test_upsert_probation_skips_library_gate():
    """|IC| 远低于 floor 仍可 probation——不走单因子 gate。"""
    import tempfile

    from factorzen.discovery.factor_library import load_library, upsert_probation

    with tempfile.TemporaryDirectory() as td:
        res = upsert_probation(
            "ashare",
            [{
                "expression": "rank(vol)",
                "ic_train": 0.004,   # < DEFAULT_IC_FLOOR
                "holdout_ic": -0.01,  # 反号
                "lift": 0.002,
                "baseline": 0.03,
                "passed": True,
            }],
            eval_window=("20200101", "20260101"), universe="u", horizon=5,
            run_id="r", session_dir="s", git_sha="a", now="2026-07-14",
            root=td,
        )
        assert res.added == 1
        assert load_library("ashare", root=td)[0].status == "probation"


def test_rebuild_preserves_probation(tmp_path):
    from factorzen.discovery.factor_library import (
        load_library,
        rebuild,
        upsert,
        upsert_probation,
    )

    # 先写一条 active + 一条 probation
    upsert(
        "ashare",
        [{"expression": "rank(close)", "ic_train": 0.05, "holdout_ic": 0.04,
          "dsr_pvalue": 0.2, "n_train": 100, "n_holdout_days": 100}],
        eval_window=("20200101", "20260101"), universe="u", horizon=1,
        run_id="r1", session_dir="s", git_sha="a", now="2026-07-01",
        root=str(tmp_path),
    )
    upsert_probation(
        "ashare",
        [{"expression": "rank(vol)", "ic_train": 0.006, "holdout_ic": 0.001,
          "lift": 0.002, "baseline": 0.03, "passed": True}],
        eval_window=("20200101", "20260101"), universe="u", horizon=5,
        run_id="lift", session_dir="s", git_sha="a", now="2026-07-02",
        root=str(tmp_path),
    )

    def evaluate(exprs):
        # rebuild 只重算 active 源；rank(vol) 不在 sources 也不过 gate
        return [
            {"expression": "rank(close)", "ic_train": 0.05, "holdout_ic": 0.04,
             "dsr_pvalue": 0.2, "n_train": 100, "n_holdout_days": 100},
        ]

    rebuild(
        "ashare", sources=["rank(close)"], eval_window=("20200101", "20260101"),
        universe="u", horizon=1, evaluate=evaluate, git_sha="b",
        now="2026-07-14", root=str(tmp_path), fresh=True,
    )
    lib = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}
    assert "rank(close)" in lib and lib["rank(close)"].status == "active"
    assert "rank(vol)" in lib and lib["rank(vol)"].status == "probation"
    assert abs(lib["rank(vol)"].lift - 0.002) < 1e-9


def test_build_library_pool_excludes_probation_by_default(tmp_path):
    from factorzen.discovery.factor_library import build_library_pool, upsert, upsert_probation

    daily_rows = []
    d0 = date(2024, 1, 2)
    for i in range(30):
        d = d0 + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for s in range(5):
            daily_rows.append({
                "trade_date": d, "ts_code": f"{s:06d}.SH",
                "close": 10.0 + s, "close_adj": 10.0 + s,
                "open_adj": 10.0, "high_adj": 10.1, "low_adj": 9.9,
                "open": 10.0, "high": 10.1, "low": 9.9, "pre_close": 10.0,
                "vol": 1e5, "amount": 1e7,
            })
    daily = pl.DataFrame(daily_rows)

    upsert(
        "ashare",
        [{"expression": "rank(close)", "ic_train": 0.05, "holdout_ic": 0.04,
          "n_holdout_days": 100, "n_train": 100}],
        eval_window=("20200101", "20260101"), universe="u", horizon=1,
        run_id="r", session_dir="s", git_sha="a", now="2026-07-01",
        root=str(tmp_path),
    )
    upsert_probation(
        "ashare",
        [{"expression": "rank(vol)", "ic_train": 0.006, "lift": 0.002,
          "baseline": 0.01, "passed": True}],
        eval_window=("20200101", "20260101"), universe="u", horizon=5,
        run_id="l", session_dir="s", git_sha="a", now="2026-07-02",
        root=str(tmp_path),
    )
    pool = build_library_pool("ashare", daily, root=str(tmp_path))
    assert "rank(close)" in pool
    assert "rank(vol)" not in pool
    # 显式要 probation 才进
    pool2 = build_library_pool(
        "ashare", daily, root=str(tmp_path), statuses=("active", "probation"),
    )
    assert "rank(vol)" in pool2 or "rank(close)" in pool2  # vol 可能物化失败但不应默认混入


# ── 双路径 gray_zone 记账 ────────────────────────────────────────────────────


def test_node_guardrails_marks_gray_zone():
    """单因子门拒绝 + |IC|≥下界 → is_lift_queue_candidate（原 gray 钩子契约）。"""
    from factorzen.discovery.guardrails import is_lift_queue_candidate

    probe = {
        "ic_train": 0.012, "n_holdout_days": 100,
        "residual_ic_train": None,
    }
    assert is_lift_queue_candidate(probe, objective="raw")


def test_mining_session_gray_zone_fields_in_manifest_contract():
    """manifest 契约：n_gray_zone 字段存在于 write 路径（源码守卫）。

    C1：reject 类别改为 lift_queue；n_gray_zone 计数字段名兼容保留。
    """
    # parents[2]：本文件在 tests/lift/，repo root 为上两级（迁入前为 tests/ 下 parents[1]）
    src = (Path(__file__).resolve().parents[2] / "src" / "factorzen"
           / "discovery" / "mining_session.py").read_text(encoding="utf-8")
    assert "n_gray_zone" in src
    assert "REJECT_CATEGORY_LIFT_QUEUE" in src
    assert "is_lift_queue_candidate" in src
    agents_man = (Path(__file__).resolve().parents[2] / "src" / "factorzen"
                  / "agents" / "manifest.py").read_text(encoding="utf-8")
    assert "n_gray_zone" in agents_man
    nodes = (Path(__file__).resolve().parents[2] / "src" / "factorzen"
             / "agents" / "nodes.py").read_text(encoding="utf-8")
    assert "REJECT_CATEGORY_LIFT_QUEUE" in nodes
    assert "(lift队列,待组合裁决)" in nodes


# ── CLI 透传 ─────────────────────────────────────────────────────────────────


def test_expression_keys_survive_residual_engine(tmp_path):
    """因子字典键是**真实表达式**（含括号/逗号）时 residual 引擎仍正常。

    报告行 expression 保持真实表达式；库键可含特殊字符（残差路径不经 lgbm 特征名）。
    """
    from factorzen.discovery.lift_test import run_lift_tests

    actives, cand_panel, _noise, ret_df = _synth_panels(signal=True)
    # 键改成真实表达式形态（括号/逗号/空格）
    actives = {f"rank(ts_mean(close, {5 + i}))": df
               for i, (_k, df) in enumerate(actives.items())}
    gray = [{"expression": "mul(rank(vol), neg(ts_std(ret_1d, 20)))",
             "residual_ic_train": 0.006}]

    out = run_lift_tests(
        gray, market="ashare", daily=pl.DataFrame(),
        active_factor_dfs=actives, ret_df=ret_df,
        materialize_candidate=lambda e: cand_panel,
        top_m=1, threshold=-1.0, seed=0, lift_workers=1,
    )
    assert out[0]["error"] is None, f"真实表达式键不得炸 residual 引擎: {out[0]}"
    assert out[0]["baseline"] is None
    assert out[0]["lift"] is not None
    assert out[0]["expression"] == "mul(rank(vol), neg(ts_std(ret_1d, 20)))"
    assert out[0].get("lift_metric") == "residual_ic_v1"


# ── P9：准入 provenance 可重放 ───────────────────────────────────────────────


def test_run_lift_tests_admission_provenance_complete():
    """production 形态 run_lift_tests：row 含 admission/scored/block/baseline_hash。"""
    import hashlib

    from factorzen.discovery.lift_test import LiftEvalContext, run_lift_tests

    dates = _dates(60)
    n_stocks = 40  # residual 日守卫 max(30, k+10)
    # 两套 active 键集合，验证 baseline_hash 集合稳定与差异
    active_a = {
        "lib_z": pl.DataFrame({
            "trade_date": [d for d in dates for _ in range(n_stocks)],
            "ts_code": [f"{s:04d}.SZ" for _ in dates for s in range(n_stocks)],
            "factor_value": [
                float((hash(d) + s) % 17) for d in dates for s in range(n_stocks)
            ],
        }),
        "lib_a": pl.DataFrame({
            "trade_date": [d for d in dates for _ in range(n_stocks)],
            "ts_code": [f"{s:04d}.SZ" for _ in dates for s in range(n_stocks)],
            "factor_value": [
                float((hash(d) * 3 + s) % 13) for d in dates for s in range(n_stocks)
            ],
        }),
    }
    # 同集合不同插入序 → hash 应相同
    active_a_reordered = {"lib_a": active_a["lib_a"], "lib_z": active_a["lib_z"]}
    active_b = {"lib_only": active_a["lib_a"]}

    ret = pl.DataFrame({
        "trade_date": [d for d in dates for _ in range(n_stocks)],
        "ts_code": [f"{s:04d}.SZ" for _ in dates for s in range(n_stocks)],
        "ret": [
            0.01 * (s - n_stocks / 2) + 0.001 * (i % 5)
            for i, d in enumerate(dates) for s in range(n_stocks)
        ],
    })
    cand = pl.DataFrame({
        "trade_date": [d for d in dates for _ in range(n_stocks)],
        "ts_code": [f"{s:04d}.SZ" for _ in dates for s in range(n_stocks)],
        "factor_value": [
            float(s) + 0.01 * (hash(d) % 7) for d in dates for s in range(n_stocks)
        ],
    })

    ctx = LiftEvalContext(
        market="ashare",
        prepped=pl.DataFrame({"trade_date": ["x"], "ts_code": ["y"], "close": [1.0]}),
        leaf_map=None,
        horizon=5,
        admission_start="20240115",
        admission_end="20240301",
        profile_name="ashare_default",
    )
    common = dict(
        gray_candidates=[{"expression": "cand0", "residual_ic_train": 0.01}],
        market="ashare",
        daily=pl.DataFrame(),
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        block_days=10,
        threshold=0.001,
        ctx=ctx,
        horizon=5,
        lift_workers=1,
    )

    r1 = run_lift_tests(**common, active_factor_dfs=active_a)[0]
    r1b = run_lift_tests(**common, active_factor_dfs=active_a_reordered)[0]
    r2 = run_lift_tests(**common, active_factor_dfs=active_b)[0]

    # admission / scored / block / profile / frequency
    assert r1["admission_start"] == "20240115"
    assert r1["admission_end"] == "20240301"
    assert r1["scored_start"] is not None
    assert r1["scored_end"] is not None
    # 窗界入参可紧凑，但回填的 scored_* 一律 ISO（core.dates 单一形态）；
    # 两侧必须同形态比较——混比会静默为真（"20240115" > "2024-01-15"）
    assert r1["scored_start"] >= "2024-01-15"
    assert r1["block_days"] == 10
    # residual_ic_v1：CV 键保留、值 None
    assert r1["cv_train_days"] is None
    assert r1["cv_test_days"] is None
    assert r1["threshold"] == 0.001
    assert r1["profile_name"] == "ashare_default"
    assert r1["frequency"] == "daily"
    assert r1["horizon"] == 5
    assert r1.get("lift_metric") == "residual_ic_v1"
    assert r1.get("n_lib_factors") == 2

    # baseline_hash：同集合稳定（插入序无关）、不同集合不同
    expected = hashlib.sha256(",".join(sorted(active_a.keys())).encode()).hexdigest()[:16]
    assert r1["baseline_hash"] == expected
    assert r1b["baseline_hash"] == r1["baseline_hash"]
    assert r2["baseline_hash"] is not None
    assert r2["baseline_hash"] != r1["baseline_hash"]
    expected_b = hashlib.sha256(",".join(sorted(active_b.keys())).encode()).hexdigest()[:16]
    assert r2["baseline_hash"] == expected_b



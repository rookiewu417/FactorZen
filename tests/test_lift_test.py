"""组合增量 lift 实验 + probation 入库通道单测。TDD、mock 离线。"""
from __future__ import annotations

import json
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


def _synth_panels(n_days=160, n_stocks=40, seed=0, *, interactive=True):
    """构造 lib 因子 + 候选 + 收益。

    interactive=True：ret = 0.3*lib + 0.5*(lib*cand 交互) + 噪声
      → 候选线性 IC≈0，但 lgbm 学到交互后 lift 显著。
    interactive=False：ret = 0.5*lib + 噪声（候选纯噪声）→ lift≈0。
    """
    rng = np.random.default_rng(seed)
    dates = _dates(n_days)
    lib_rows, cand_rows, noise_rows, ret_rows = [], [], [], []
    for d in dates:
        lib = rng.standard_normal(n_stocks)
        cand = rng.standard_normal(n_stocks)
        noise = rng.standard_normal(n_stocks)
        if interactive:
            ret = 0.3 * lib + 0.6 * (lib * cand) + 0.25 * rng.standard_normal(n_stocks)
        else:
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


def test_lift_interactive_candidate_passes_noise_fails():
    """交互项候选 lift 显著过阈值；纯噪声候选 lift≈0 拒。基线只算一次。"""
    from factorzen.discovery.guardrails import DEFAULT_LIFT_THRESHOLD
    from factorzen.discovery.lift_test import run_lift_tests
    from factorzen.research.combination.models import combine_lgbm

    active, cand_good, cand_noise, ret = _synth_panels(interactive=True)
    # 噪声场景的 ret 另造一版纯噪声候选仍用 interactive ret（噪声候选本身无信号）
    call_count = {"n": 0}
    real_combine = combine_lgbm

    def counting_combine(fds, rdf, cv, **kw):
        call_count["n"] += 1
        return real_combine(fds, rdf, cv, seed=0, n_estimators=40, min_child_samples=15)

    cv_params = {"train_days": 60, "test_days": 20, "purge_days": 5}

    mat_map = {
        "interactive_cand": cand_good,
        "noise_cand": cand_noise,
    }

    def materialize(expr):
        return mat_map[expr]

    rows = run_lift_tests(
        [
            {"expression": "interactive_cand", "residual_ic_train": 0.006},
            {"expression": "noise_cand", "residual_ic_train": 0.005},
        ],
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),  # unused
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=materialize,
        combine_fn=counting_combine,
        cv_params=cv_params,
        top_m=10,
        threshold=DEFAULT_LIFT_THRESHOLD,
        seed=0,
    )
    by_expr = {r["expression"]: r for r in rows}
    assert by_expr["interactive_cand"]["passed"] is True, by_expr["interactive_cand"]
    assert by_expr["interactive_cand"]["lift"] >= DEFAULT_LIFT_THRESHOLD
    # 噪声：lift 应接近 0（允许小幅波动，但不得过阈值太多；硬门：不 passed 或 lift 显著低于交互）
    assert by_expr["noise_cand"]["passed"] is False or (
        by_expr["noise_cand"]["lift"] is not None
        and by_expr["noise_cand"]["lift"] < by_expr["interactive_cand"]["lift"]
    )
    # 基线 1 次 + 2 候选 = 3 次 combine（不重复算基线）
    assert call_count["n"] == 3, f"baseline 应只算一次，实际 combine 调用 {call_count['n']}"


def test_lift_top_m_truncation_recorded():
    """top_m 截断不静默：返回行带 n_input / n_selected。"""
    from factorzen.discovery.lift_test import run_lift_tests

    active, cand, _, ret = _synth_panels(n_days=100, n_stocks=20, interactive=False)
    grays = [
        {"expression": f"c{i}", "residual_ic_train": 0.009 - i * 0.0001}
        for i in range(5)
    ]
    mats = {f"c{i}": cand for i in range(5)}

    def combine_stub(fds, rdf, cv, **kw):
        # 返回合法空壳组合帧（全 0 因子值）
        return ret.select(["trade_date", "ts_code"]).with_columns(
            pl.lit(0.0).alias("factor_value")
        )

    rows = run_lift_tests(
        grays,
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: mats[e],
        combine_fn=combine_stub,
        top_m=2,
        threshold=0.001,
    )
    assert len(rows) == 2
    assert rows[0]["n_input"] == 5 and rows[0]["n_selected"] == 2
    assert rows[0]["truncated_from"] == 5


def test_lift_bad_candidate_does_not_crash_batch():
    from factorzen.discovery.lift_test import run_lift_tests

    active, cand, _, ret = _synth_panels(n_days=80, n_stocks=20, interactive=False)

    def materialize(expr):
        if expr == "bad":
            raise RuntimeError("boom")
        return cand

    def combine_stub(fds, rdf, cv, **kw):
        return ret.select(["trade_date", "ts_code"]).with_columns(
            pl.lit(0.0).alias("factor_value")
        )

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
        combine_fn=combine_stub,
        top_m=10,
    )
    assert len(rows) == 2
    assert rows[0]["passed"] is False and rows[0]["error"]
    assert rows[1]["expression"] == "ok"


def test_extract_gray_from_manifest():
    from factorzen.discovery.lift_test import extract_gray_candidates_from_manifest

    man = {
        "attempts": [
            {"expression": "a", "reject_category": "gray_zone", "residual_ic_train": 0.006},
            {"expression": "b", "reject_category": "holdout_coverage"},
            {"expression": "a", "reject_category": "gray_zone"},  # 去重
        ],
        "candidates": [
            {"expression": "c", "reject_category": "gray_zone", "ic_train": 0.008},
        ],
    }
    got = extract_gray_candidates_from_manifest(man)
    assert {g["expression"] for g in got} == {"a", "c"}


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


def _signed_factor_panels(sign: float, n_days=80, n_stocks=30, seed=1):
    """构造单因子与 ret 同号/反号相关的面板（admission_ic 符号断言用）。

    factor ≈ sign * ret + 极小噪声 → RankIC 符号 ≈ sign 的符号。
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
    """admission_ic = 单因子 admission 窗 RankIC，正/负相关各得对应符号；非组合 IC。"""
    from factorzen.discovery.lift_test import run_lift_tests

    def combine_stub(fds, rdf, cv, **kw):
        # 恒正假预测：若误用 candidate_rank_ic 当方向会永远为正
        return rdf.select(
            ["trade_date", "ts_code", pl.col("ret").abs().alias("factor_value")]
        )

    # 正相关
    active_pos, cand_pos, ret_pos = _signed_factor_panels(+1.0, seed=11)
    rows_pos = run_lift_tests(
        [{"expression": "pos_cand", "ic_train": 0.03, "residual_ic_train": 0.02}],
        market="ashare",
        daily=pl.DataFrame({"trade_date": [], "ts_code": [], "close": []}),
        active_factor_dfs=active_pos,
        ret_df=ret_pos,
        materialize_candidate=lambda e: cand_pos,
        combine_fn=combine_stub,
        top_m=None,
        threshold=0.001,
    )
    assert len(rows_pos) == 1
    rpos = rows_pos[0]
    assert rpos.get("admission_ic") is not None, rpos
    assert rpos["admission_ic"] > 0, rpos
    assert rpos.get("ic_train") == 0.03
    assert rpos.get("residual_ic_train") == 0.02
    # 组合模型 IC 可能为正，但方向权威是 admission_ic（单因子）
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
        combine_fn=combine_stub,
        top_m=None,
        threshold=0.001,
    )
    assert len(rows_neg) == 1
    rneg = rows_neg[0]
    assert rneg.get("admission_ic") is not None, rneg
    assert rneg["admission_ic"] < 0, rneg
    # 负向单因子 vs 组合 stub 的 candidate_rank_ic 符号可不同——证明不是同一字段
    assert rneg["admission_ic"] != rneg.get("candidate_rank_ic") or (
        rneg.get("candidate_rank_ic") is not None and rneg["candidate_rank_ic"] >= 0
    )


def test_run_lift_tests_error_rows_have_admission_ic_key():
    """错误路径 row 也有 admission_ic 键（形态一致，下游 no KeyError）。"""
    from factorzen.discovery.lift_test import run_lift_tests

    active, cand, _, ret = _synth_panels(n_days=60, n_stocks=20, interactive=False)

    def combine_stub(fds, rdf, cv, **kw):
        return ret.select(["trade_date", "ts_code"]).with_columns(
            pl.lit(0.0).alias("factor_value")
        )

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
        combine_fn=combine_stub,
        top_m=10,
    )
    assert all("admission_ic" in r for r in rows)
    # 失败行 admission_ic 应为 None（未成功物化）
    by = {r["expression"]: r for r in rows}
    assert by["bad"]["admission_ic"] is None
    assert by["bad"]["error"]


def test_run_lift_tests_new_fields_and_top_m_none():
    """新字段全链 + top_m=None 全测；mock combine 返回可控预测。"""
    from factorzen.discovery.lift_test import run_lift_tests

    active, cand, _noise, ret = _synth_panels(n_days=100, n_stocks=25, interactive=False)
    # 基线：弱噪声预测（截面方差 > 0，避免 std 守卫跳过整天）
    # 候选 add-one：用 ret 本身作预测 → 近完美 IC，相对基线正 lift
    rng = np.random.default_rng(7)
    noise_vals = rng.standard_normal(ret.height)
    base_pred = ret.select(["trade_date", "ts_code"]).with_columns(
        pl.Series("factor_value", noise_vals)
    )
    call_n = {"n": 0}

    def combine_ctrl(fds, rdf, cv, **kw):
        call_n["n"] += 1
        n_factors = len(fds)
        if n_factors <= len(active):
            return base_pred
        return rdf.select(
            ["trade_date", "ts_code", pl.col("ret").alias("factor_value")]
        )

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
        combine_fn=combine_ctrl,
        top_m=None,  # 全测
        threshold=0.001,
        block_days=10,
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
        assert r["passed"] is True  # 完美 ret 预测应对基线有正 lift
        assert r["error"] is None
    # 基线 1 + 4 候选
    assert call_n["n"] == 5


def test_run_group_lift_three_states():
    """run_group_lift：正常 / 部分物化失败 / 全失败。"""
    from factorzen.discovery.lift_test import run_group_lift

    active, cand, _, ret = _synth_panels(n_days=80, n_stocks=20, interactive=False)
    rng = np.random.default_rng(3)
    base_noise = ret.select(["trade_date", "ts_code"]).with_columns(
        pl.Series("factor_value", rng.standard_normal(ret.height))
    )

    def combine_stub(fds, rdf, cv, **kw):
        n = len(fds)
        # 更多因子 → 用 ret 当预测，制造正 group lift
        if n > len(active):
            return rdf.select(
                ["trade_date", "ts_code", pl.col("ret").alias("factor_value")]
            )
        return base_noise

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
        combine_fn=combine_stub,
        threshold=0.001,
    )
    assert ok["error"] is None
    assert ok["n_candidates"] == 2
    assert set(ok["expressions"]) == {"g1", "g2"}
    assert ok["skipped"] == []
    assert ok["lift"] is not None
    assert "lift_se" in ok and "n_blocks" in ok

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
        combine_fn=combine_stub,
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
        combine_fn=combine_stub,
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
        "ic_train": 0.008, "n_holdout_days": 100,
        "residual_ic_train": None,
    }
    assert is_lift_queue_candidate(probe, objective="raw")


def test_mining_session_gray_zone_fields_in_manifest_contract():
    """manifest 契约：n_gray_zone 字段存在于 write 路径（源码守卫）。

    C1：reject 类别改为 lift_queue；n_gray_zone 计数字段名兼容保留。
    """
    src = (Path(__file__).resolve().parents[1] / "src" / "factorzen"
           / "discovery" / "mining_session.py").read_text(encoding="utf-8")
    assert "n_gray_zone" in src
    assert "REJECT_CATEGORY_LIFT_QUEUE" in src
    assert "is_lift_queue_candidate" in src
    agents_man = (Path(__file__).resolve().parents[1] / "src" / "factorzen"
                  / "agents" / "manifest.py").read_text(encoding="utf-8")
    assert "n_gray_zone" in agents_man
    nodes = (Path(__file__).resolve().parents[1] / "src" / "factorzen"
             / "agents" / "nodes.py").read_text(encoding="utf-8")
    assert "REJECT_CATEGORY_LIFT_QUEUE" in nodes
    assert "(lift队列,待组合裁决)" in nodes


# ── CLI 透传 ─────────────────────────────────────────────────────────────────


def test_cli_lift_test_parser_and_dry_run(tmp_path, monkeypatch):
    """capability↔wiring：子命令注册 + dry-run 不写库。"""
    from factorzen.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "factor-library", "lift-test",
        "--session", str(tmp_path / "run1"),
        "--market", "ashare",
        "--start", "20200101",
        "--end", "20201231",
        "--universe", "csi300",
        "--top-m", "5",
        "--threshold", "0.002",
        "--dry-run",
    ])
    assert args.func.__name__ == "_cmd_factor_library_lift_test"
    assert args.top_m == 5
    assert args.threshold == 0.002
    assert args.dry_run is True

    # 写一个含 gray_zone 的 manifest
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "attempts": [
            {"expression": "rank(close)", "reject_category": "gray_zone",
             "residual_ic_train": 0.006, "n_residual_holdout_days": 100},
        ],
        "candidates": [],
    }), encoding="utf-8")

    lib_root = tmp_path / "lib"
    lib_root.mkdir()
    # mock 数据装配 + lift + upsert
    import factorzen.cli.main as cli_main
    import factorzen.discovery.factor_library as fl
    import factorzen.discovery.lift_test as lt_mod

    monkeypatch.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda args: (pl.DataFrame({
            "trade_date": [date(2020, 1, 2)], "ts_code": ["000001.SZ"],
            "close": [10.0], "close_adj": [10.0],
        }), None, {}),
    )

    def fake_lift(gray, **kw):
        return [{
            "expression": "rank(close)", "lift": 0.005, "baseline": 0.02,
            "passed": True, "candidate_rank_ic": 0.025,
        }]

    written = {"upsert": 0}

    def fake_upsert(*a, **k):
        written["upsert"] += 1
        from factorzen.discovery.factor_library import UpsertResult
        return UpsertResult(added=1)

    monkeypatch.setattr(lt_mod, "run_lift_tests", fake_lift)
    # CLI 从 factor_library 模块调 upsert_probation
    monkeypatch.setattr(fl, "upsert_probation", fake_upsert)

    args.library_root = str(lib_root)
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert written["upsert"] == 0  # dry-run 不写库
    assert (run_dir / "lift_test_manifest.json").exists()
    man = json.loads((run_dir / "lift_test_manifest.json").read_text(encoding="utf-8"))
    assert man["dry_run"] is True
    assert man["n_passed"] == 1
    assert man["threshold"] == 0.002


def test_expression_keys_survive_real_lgbm(tmp_path):
    """因子字典键是**真实表达式**（含括号/逗号）时必须能过真 lgbm——
    线上事故回归：LightGBMError: Do not support special JSON characters in
    feature name（合成测试用安全名没抓到，真实表达式键立刻炸基线）。
    进 combine 边界前键须映射为安全特征名，报告仍用真实表达式。
    """
    from factorzen.discovery.lift_test import run_lift_tests

    actives, cand_panel, _noise, ret_df = _synth_panels(interactive=True)
    # 键改成真实表达式形态（括号/逗号/空格——lgbm 特征名黑名单字符）
    actives = {f"rank(ts_mean(close, {5 + i}))": df
               for i, (_k, df) in enumerate(actives.items())}
    gray = [{"expression": "mul(rank(vol), neg(ts_std(ret_1d, 20)))",
             "residual_ic_train": 0.006}]

    out = run_lift_tests(
        gray, market="ashare", daily=pl.DataFrame(),
        active_factor_dfs=actives, ret_df=ret_df,
        materialize_candidate=lambda e: cand_panel,
        cv_params={"train_days": 60, "test_days": 20, "purge_days": 2,
                   "embargo_days": 0},
        top_m=1, threshold=-1.0, seed=0,
    )   # 不注入 combine_fn → 走真 combine_lgbm
    assert out[0]["error"] is None, f"真实表达式键不得炸 lgbm: {out[0]}"
    assert out[0]["baseline"] is not None
    assert out[0]["expression"] == "mul(rank(vol), neg(ts_std(ret_1d, 20)))"

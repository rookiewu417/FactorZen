"""稀疏因子子集 IC + sleeve 护栏旁路（TDD）。

设计契约（定死）:
- 非零覆盖率 < 0.20 → 稀疏；稠密路径不受影响
- subset IC = 仅非零样本逐日 RankIC；当日 n<30 跳过
- 旁路: |subset_ic_train|≥0.03 且 holdout 同号 且 n_days_train≥40 → lift_queue + sleeve_candidate
- 不直接 passed；lift 层零改
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from factorzen.discovery.evaluation import (
    SPARSE_COVERAGE_THRESHOLD,
    _nonzero_coverage,
    compute_subset_rank_ic,
)
from factorzen.discovery.guardrails import (
    REJECT_CATEGORY_LIFT_QUEUE,
    SLEEVE_SUBSET_IC_FLOOR,
    SLEEVE_SUBSET_MIN_DAYS,
    is_sleeve_lift_candidate,
)

# ── helpers: 合成面板 ──────────────────────────────────────────────────────

def _dates(n: int, start: date = date(2022, 1, 3)) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _panel(
    n_days: int,
    n_stocks: int,
    *,
    nz_per_day: int,
    perfect_ic: bool = True,
    seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """构造 factor_df + fwd_returns。

    每日前 ``nz_per_day`` 只非零；非零子集上 factor 与 ret 完全同序（perfect_ic）
    或打乱（perfect_ic=False）。其余为 0。
    """
    rng = np.random.default_rng(seed)
    days = _dates(n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    f_rows: list[dict] = []
    r_rows: list[dict] = []
    for dd in days:
        # 非零票：固定前 nz 只，便于 golden
        nz_codes = codes[:nz_per_day]
        ranks = np.arange(1, nz_per_day + 1, dtype=float)
        if perfect_ic:
            rets = ranks.copy()
        else:
            rets = rng.permutation(ranks)
        for j, c in enumerate(codes):
            if c in nz_codes:
                fv = float(ranks[j])
                rv = float(rets[j])
            else:
                fv = 0.0
                rv = float(rng.standard_normal() * 0.01)
            f_rows.append({"trade_date": dd, "ts_code": c, "factor_value": fv})
            r_rows.append({"trade_date": dd, "ts_code": c, "fwd_ret_1d": rv})
    return pl.DataFrame(f_rows), pl.DataFrame(r_rows)


# ── 1. 非零覆盖率 ──────────────────────────────────────────────────────────

def test_nonzero_coverage_sparse_and_dense():
    fdf, _ = _panel(10, 100, nz_per_day=5)  # 5%
    cov = _nonzero_coverage(fdf)
    assert cov == pytest.approx(0.05, abs=1e-9)
    assert cov < SPARSE_COVERAGE_THRESHOLD

    fdf_d, _ = _panel(10, 100, nz_per_day=50)  # 50%
    cov_d = _nonzero_coverage(fdf_d)
    assert cov_d == pytest.approx(0.50, abs=1e-9)
    assert cov_d >= SPARSE_COVERAGE_THRESHOLD


def test_nonzero_coverage_empty_is_zero():
    empty = pl.DataFrame(
        schema={"trade_date": pl.Date, "ts_code": pl.Utf8, "factor_value": pl.Float64}
    )
    assert _nonzero_coverage(empty) == 0.0


# ── 2. 子集 IC golden（小面板硬编码期望）──────────────────────────────────

def test_subset_rank_ic_perfect_correlation_golden():
    """非零子集上 factor 与 ret 同序 → 每日 RankIC=1 → mean=1；n_days=全部日。"""
    n_days, n_stocks, nz = 12, 80, 40  # coverage=0.5 但本测只看 subset IC 数值
    fdf, ret = _panel(n_days, n_stocks, nz_per_day=nz, perfect_ic=True)
    ic_mean, n = compute_subset_rank_ic(fdf, ret, min_samples=30)
    assert n == n_days
    assert ic_mean == pytest.approx(1.0, abs=1e-9)


def test_subset_rank_ic_anti_correlation_golden():
    """非零子集 factor 升序、ret 降序 → RankIC=-1。"""
    n_days, nz = 8, 40
    days = _dates(n_days)
    codes = [f"{i:06d}.SZ" for i in range(100)]
    f_rows, r_rows = [], []
    for dd in days:
        for j, c in enumerate(codes):
            if j < nz:
                f_rows.append({"trade_date": dd, "ts_code": c, "factor_value": float(j + 1)})
                r_rows.append({"trade_date": dd, "ts_code": c, "fwd_ret_1d": float(nz - j)})
            else:
                f_rows.append({"trade_date": dd, "ts_code": c, "factor_value": 0.0})
                r_rows.append({"trade_date": dd, "ts_code": c, "fwd_ret_1d": 0.0})
    ic_mean, n = compute_subset_rank_ic(pl.DataFrame(f_rows), pl.DataFrame(r_rows), min_samples=30)
    assert n == n_days
    assert ic_mean == pytest.approx(-1.0, abs=1e-9)


def test_subset_rank_ic_skips_days_below_min_samples():
    """当日子集 n<30 整日跳过；全部日不足 → n_days=0, ic=None。"""
    # 每日子集仅 10 只非零 < 30
    fdf, ret = _panel(20, 100, nz_per_day=10, perfect_ic=True)
    ic_mean, n = compute_subset_rank_ic(fdf, ret, min_samples=30)
    assert n == 0
    assert ic_mean is None


def test_subset_rank_ic_mixed_width_days():
    """部分日 n≥30 计入，部分不足跳过。"""
    days = _dates(6)
    codes = [f"{i:06d}.SZ" for i in range(100)]
    f_rows, r_rows = [], []
    for k, dd in enumerate(days):
        nz = 40 if k < 4 else 10  # 前 4 日够，后 2 日不够
        for j, c in enumerate(codes):
            if j < nz:
                f_rows.append({"trade_date": dd, "ts_code": c, "factor_value": float(j + 1)})
                r_rows.append({"trade_date": dd, "ts_code": c, "fwd_ret_1d": float(j + 1)})
            else:
                f_rows.append({"trade_date": dd, "ts_code": c, "factor_value": 0.0})
                r_rows.append({"trade_date": dd, "ts_code": c, "fwd_ret_1d": 0.0})
    ic_mean, n = compute_subset_rank_ic(pl.DataFrame(f_rows), pl.DataFrame(r_rows), min_samples=30)
    assert n == 4
    assert ic_mean == pytest.approx(1.0, abs=1e-9)


# ── 3. 旁路判定 is_sleeve_lift_candidate ───────────────────────────────────

def _sleeve_cand(**kw):
    base = {
        "is_sparse": True,
        "subset_ic_train": 0.05,
        "subset_ic_holdout": 0.04,
        "subset_n_days_train": 50,
    }
    base.update(kw)
    return base


def test_sleeve_gate_passes_strong_sparse_same_sign():
    assert is_sleeve_lift_candidate(_sleeve_cand()) is True


def test_sleeve_gate_rejects_dense():
    assert is_sleeve_lift_candidate(_sleeve_cand(is_sparse=False)) is False


def test_sleeve_gate_rejects_weak_subset_ic():
    assert is_sleeve_lift_candidate(
        _sleeve_cand(subset_ic_train=0.02)  # < 0.03
    ) is False
    assert SLEEVE_SUBSET_IC_FLOOR == 0.03


def test_sleeve_gate_rejects_sign_flip():
    assert is_sleeve_lift_candidate(
        _sleeve_cand(subset_ic_train=0.05, subset_ic_holdout=-0.04)
    ) is False


def test_sleeve_gate_rejects_insufficient_n_days():
    assert is_sleeve_lift_candidate(
        _sleeve_cand(subset_n_days_train=39)
    ) is False
    assert SLEEVE_SUBSET_MIN_DAYS == 40
    # 边界：恰 40 过
    assert is_sleeve_lift_candidate(
        _sleeve_cand(subset_n_days_train=40)
    ) is True


def test_sleeve_gate_rejects_zero_or_missing_holdout():
    assert is_sleeve_lift_candidate(_sleeve_cand(subset_ic_holdout=0.0)) is False
    assert is_sleeve_lift_candidate(_sleeve_cand(subset_ic_holdout=None)) is False
    assert is_sleeve_lift_candidate(_sleeve_cand(subset_ic_train=None)) is False


def test_sleeve_gate_switch_off():
    assert is_sleeve_lift_candidate(_sleeve_cand(), sleeve_gate=False) is False


def test_sleeve_gate_negative_same_sign():
    """负向事件 alpha 同号也可旁路。"""
    assert is_sleeve_lift_candidate(
        _sleeve_cand(subset_ic_train=-0.05, subset_ic_holdout=-0.03)
    ) is True


def test_sleeve_gate_floor_boundary_abs():
    assert is_sleeve_lift_candidate(
        _sleeve_cand(subset_ic_train=0.03, subset_ic_holdout=0.01)
    ) is True
    assert is_sleeve_lift_candidate(
        _sleeve_cand(subset_ic_train=0.0299, subset_ic_holdout=0.01)
    ) is False


# ── 4. 合成稀疏：全截面弱 vs 子集强 + 旁路语义表 ───────────────────────────

def test_synthetic_sparse_full_section_diluted_subset_strong():
    """5% 覆盖：子集 perfect IC；全截面因大量 0 稀释后 |IC| 明显变弱。"""
    from factorzen.daily.evaluation.ic_analysis import _rank_ic_by_date

    n_days, n_stocks, nz = 50, 200, 10  # 5% coverage; nz=10 < 30 → subset n_days 会 0
    # 改用 nz=35, stocks=200 → coverage 17.5% < 20%，子集 n≥30
    n_days, n_stocks, nz = 50, 200, 35
    fdf, ret = _panel(n_days, n_stocks, nz_per_day=nz, perfect_ic=True, seed=7)
    cov = _nonzero_coverage(fdf)
    assert cov < SPARSE_COVERAGE_THRESHOLD
    assert cov == pytest.approx(35 / 200, abs=1e-9)

    sub_ic, sub_n = compute_subset_rank_ic(fdf, ret, min_samples=30)
    assert sub_n == n_days
    assert sub_ic == pytest.approx(1.0, abs=1e-9)

    # 全截面 RankIC（含零）相对子集被稀释（大量并列 0 拉开秩距）
    joined = fdf.join(ret, on=["trade_date", "ts_code"], how="inner")
    full = _rank_ic_by_date(joined, "factor_value", "fwd_ret_1d", min_samples=30)
    full_mean = float(full["ic"].mean()) if full.height else 0.0
    assert abs(full_mean) < abs(sub_ic) - 0.2  # 相对子集显著稀释
    assert abs(full_mean) < 0.85


def test_bypass_decision_table_lift_queue_not_passed():
    """修后语义：主门不过 + 旁路过 → reject_category=lift_queue，passed 仍 False。"""
    cand = _sleeve_cand()
    passed_main = False  # 全截面门不过
    assert passed_main is False
    assert is_sleeve_lift_candidate(cand) is True
    # 调用方契约（与 node_guardrails 同形态）
    reject_category = None
    sleeve_candidate = False
    if not passed_main and is_sleeve_lift_candidate(cand):
        reject_category = REJECT_CATEGORY_LIFT_QUEUE
        sleeve_candidate = True
    assert reject_category == REJECT_CATEGORY_LIFT_QUEUE
    assert sleeve_candidate is True
    assert passed_main is False  # 绝不直接 passed


def test_dense_path_unaffected_by_sleeve_gate():
    """稠密因子：is_sparse=False → 旁路永不触发，与 sleeve_gate 开关无关。"""
    dense = {
        "is_sparse": False,
        "subset_ic_train": 0.10,  # 即便误填强子集也不得旁路
        "subset_ic_holdout": 0.10,
        "subset_n_days_train": 100,
    }
    assert is_sleeve_lift_candidate(dense, sleeve_gate=True) is False
    assert is_sleeve_lift_candidate(dense, sleeve_gate=False) is False


# ── 5. evaluate_expressions 稠密路径字段 + 关/开门对照 ─────────────────────

def test_evaluate_expressions_dense_has_is_sparse_false():
    """稠密表达式：is_sparse=False，主 IC 仍算；subset 字段为 None。"""
    import datetime as dt

    from factorzen.discovery.evaluation import evaluate_expressions
    from factorzen.discovery.scoring import DataBundle

    rng = np.random.default_rng(1)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < 80:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(40)]
    rows = []
    for c in codes:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px,
                "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
            })
    daily = pl.DataFrame(rows)
    bundle = DataBundle.build(daily, train_ratio=0.7)
    res = evaluate_expressions(["rank(close)"], daily, bundle)
    assert len(res) == 1
    r = res[0]
    assert r["compile_ok"] is True
    assert r["ic_train"] is not None
    assert r["is_sparse"] is False
    assert r["subset_ic_train"] is None
    assert r["subset_n_days_train"] is None
    assert r["nonzero_coverage"] is not None
    assert r["nonzero_coverage"] >= SPARSE_COVERAGE_THRESHOLD


def test_node_guardrails_sleeve_bypass_and_switch_off():
    """合成稀疏 attempt：主门不过 + 子集达标 → lift_queue + sleeve_candidate；关旁路则否。

    ic_train 故意压到 library floor 之下，保证主门不过；子集字段预填达标。
    """
    import datetime as dt

    from factorzen.agents.nodes import node_guardrails
    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import split_holdout
    from factorzen.validation.multiple_testing import TrialLedger

    rng = np.random.default_rng(11)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < 160:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(40)]
    rows = []
    for c in codes:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px,
                "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
            })
    daily = pl.DataFrame(rows)
    mining_df, holdout_df, _ = split_holdout(daily, holdout_ratio=0.2)
    bundle = DataBundle.build(mining_df)

    def _mk_state(*, sparse: bool = True):
        s = AgentState(seed=1)
        # |ic_train| << library floor → 主门必不过；子集预填旁路条件
        s.attempts.append(AttemptRecord(
            iteration=0, hypothesis="event", expression="rank(close)",
            compile_ok=True, ic_train=0.001, passed_guardrails=False,
            critic_verdict=None, error=None, ir_train=0.02, n_train=80,
            is_sparse=sparse,
            subset_ic_train=0.06 if sparse else None,
            subset_n_days_train=55 if sparse else None,
            subset_ic_holdout=0.05 if sparse else None,
            subset_n_days_holdout=40 if sparse else None,
            nonzero_coverage=0.04 if sparse else 0.9,
        ))
        return s

    # 开门 + 稀疏：lift_queue + sleeve_candidate，且不 passed
    s_on = _mk_state(sparse=True)
    node_guardrails(
        s_on, daily=mining_df, holdout_df=holdout_df, bundle=bundle,
        ledger=TrialLedger(), top_k=5, objective="raw", sleeve_gate=True,
    )
    a_on = s_on.attempts[0]
    assert a_on.passed_guardrails is False
    assert a_on.reject_category == REJECT_CATEGORY_LIFT_QUEUE
    assert a_on.sleeve_candidate is True
    assert a_on.reject_reason is not None
    assert "sleeve" in a_on.reject_reason

    # 关旁路：不得打 sleeve_candidate（可能仍因全截面 gray floor 进 lift_queue）
    s_off = _mk_state(sparse=True)
    node_guardrails(
        s_off, daily=mining_df, holdout_df=holdout_df, bundle=bundle,
        ledger=TrialLedger(), top_k=5, objective="raw", sleeve_gate=False,
    )
    a_off = s_off.attempts[0]
    assert a_off.sleeve_candidate is False
    assert a_off.passed_guardrails is False

    # 稠密对照：同弱 IC、无稀疏标记 → 不得 sleeve
    s_dense = _mk_state(sparse=False)
    node_guardrails(
        s_dense, daily=mining_df, holdout_df=holdout_df, bundle=bundle,
        ledger=TrialLedger(), top_k=5, objective="raw", sleeve_gate=True,
    )
    assert s_dense.attempts[0].sleeve_candidate is False


def test_cli_no_sleeve_gate_flag():
    """fz mine team --no-sleeve-gate 解析为 no_sleeve_gate=True。"""
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args([
        "mine", "team",
        "--start", "20200101", "--end", "20201231",
        "--no-sleeve-gate",
    ])
    assert getattr(args, "no_sleeve_gate", False) is True

    args_default = p.parse_args([
        "mine", "team",
        "--start", "20200101", "--end", "20201231",
    ])
    assert getattr(args_default, "no_sleeve_gate", False) is False


def test_librarian_records_sleeve_fields(tmp_path):
    """experiment_index 写入 subset_* / sleeve_candidate；缺字段容忍（不强制所有行都有）。"""
    from factorzen.agents.experiment_index import ExperimentIndex
    from factorzen.agents.roles.librarian import record
    from factorzen.agents.state import AttemptRecord

    idx = ExperimentIndex(str(tmp_path / "idx.jsonl"))
    attempts = [
        AttemptRecord(
            0, "h", "rank(close)", True, 0.002, False, None, None,
            is_sparse=True, subset_ic_train=0.05, subset_n_days_train=50,
            subset_ic_holdout=0.04, subset_n_days_holdout=30,
            sleeve_candidate=True, nonzero_coverage=0.05,
            reject_category=REJECT_CATEGORY_LIFT_QUEUE,
        ),
        AttemptRecord(
            0, "h2", "rank(vol)", True, 0.03, True, "keep", None,
        ),
    ]
    record(idx, attempts, run_id="t1")
    rows = idx.load()
    sparse_row = next(r for r in rows if r["expression"] == "rank(close)")
    assert sparse_row["is_sparse"] is True
    assert sparse_row["sleeve_candidate"] is True
    assert sparse_row["subset_ic_train"] == 0.05
    assert sparse_row["subset_n_days_train"] == 50
    dense_row = next(r for r in rows if r["expression"] == "rank(vol)")
    assert "sleeve_candidate" not in dense_row
    assert dense_row.get("is_sparse") is None or dense_row.get("is_sparse") is not True


# ── 6. 二期：事件掩码子集评估 ───────────────────────────────────────────────

def test_event_mask_leaves_registered():
    """EVENT_MASK_LEAVES 覆盖预告/快报/龙虎榜六叶。"""
    from factorzen.core.feature_schema import EVENT_MASK_LEAVES

    expected = {
        "fc_type_score", "fc_surprise", "fc_flag",
        "express_yoy", "top_list_flag", "top_list_net_buy",
    }
    assert set(EVENT_MASK_LEAVES) == expected


def test_build_event_leaf_mask_union_of_nonzero_leaves():
    """掩码 = 交集叶原值非零且非 null 的 (ts_code, trade_date) 并集。"""
    from factorzen.discovery.evaluation import build_event_leaf_mask

    days = _dates(3)
    codes = [f"{i:06d}.SZ" for i in range(10)]
    rows = []
    for di, dd in enumerate(days):
        for j, c in enumerate(codes):
            # 日0: 前 1 只有 fc_flag；日1: 另 1 只有 express_yoy；日2: 无事件
            fc = 1.0 if (di == 0 and j == 0) else 0.0
            ey = 2.0 if (di == 1 and j == 1) else 0.0
            rows.append({
                "trade_date": dd, "ts_code": c,
                "fc_flag": fc, "express_yoy": ey,
            })
    panel = pl.DataFrame(rows)
    mask = build_event_leaf_mask(panel, ["fc_flag", "express_yoy"])
    keys = set(zip(
        mask["trade_date"].to_list(), mask["ts_code"].to_list(), strict=True,
    ))
    assert (days[0], "000000.SZ") in keys
    assert (days[1], "000001.SZ") in keys
    assert len(keys) == 2


def test_masked_subset_rank_ic_golden_independent_of_value_coverage():
    """表达式值覆盖率 1.0 时，子集 IC 仍按叶原值掩码算（perfect → 1.0）。

    事件覆盖 40%（≥30 截面）；表达式值全非零（模拟 ts_rank 包装后覆盖 1.0）。
    掩码内 factor/ret 完美同序 → RankIC mean=1；值稀疏通道会被掩码外噪声稀释。
    """
    from factorzen.discovery.evaluation import (
        build_event_leaf_mask,
        compute_subset_rank_ic,
    )

    n_days, n_stocks, nz = 12, 100, 40
    days = _dates(n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    f_rows, r_rows, p_rows = [], [], []
    for dd in days:
        for j, c in enumerate(codes):
            on_event = j < nz
            if on_event:
                fv, rv, flag = float(j + 1), float(j + 1), 1.0
            else:
                fv, rv, flag = float(1000 + j), float((j * 7) % 97), 0.0
            f_rows.append({"trade_date": dd, "ts_code": c, "factor_value": fv})
            r_rows.append({"trade_date": dd, "ts_code": c, "fwd_ret_1d": rv})
            p_rows.append({"trade_date": dd, "ts_code": c, "fc_flag": flag})
    fdf = pl.DataFrame(f_rows)
    ret = pl.DataFrame(r_rows)
    panel = pl.DataFrame(p_rows)
    assert _nonzero_coverage(fdf) == pytest.approx(1.0)
    assert _nonzero_coverage(fdf) >= SPARSE_COVERAGE_THRESHOLD

    mask = build_event_leaf_mask(panel, ["fc_flag"])
    assert mask.height == n_days * nz

    ic_m, n_m = compute_subset_rank_ic(fdf, ret, min_samples=30, mask_keys=mask)
    assert n_m == n_days
    assert ic_m == pytest.approx(1.0, abs=1e-9)

    # 无掩码的值子集（全非零）= 全截面，被噪声稀释
    ic_v, n_v = compute_subset_rank_ic(fdf, ret, min_samples=30)
    assert n_v == n_days
    assert abs(ic_v) < 0.95


def test_is_sleeve_lift_candidate_mask_trigger_without_is_sparse():
    """掩码触发（subset_mask_leaves 非空）即使 is_sparse=False 也可旁路。"""
    cand = {
        "is_sparse": False,  # 值覆盖被包装成 1.0
        "subset_mask_leaves": ["fc_flag"],
        "subset_ic_train": 0.05,
        "subset_ic_holdout": 0.04,
        "subset_n_days_train": 50,
    }
    assert is_sleeve_lift_candidate(cand) is True
    # 无掩码且不稀疏 → 仍拒
    assert is_sleeve_lift_candidate({
        **cand, "subset_mask_leaves": None,
    }) is False
    assert is_sleeve_lift_candidate({
        **cand, "subset_mask_leaves": [],
    }) is False


def test_evaluate_expressions_event_mask_vs_dense_non_event():
    """含事件叶 → 掩码触发 + subset_mask_leaves；不含事件叶 → 掩码字段空、稠密不变。

    事件覆盖 5%（值稀疏期盲点）；ts_rank 包装后值覆盖 → 1.0；掩码仍触发。
    掩码日截面 n=5 < 30 → subset_n_days 可为 0，但 subset_mask_leaves 必须非空。
    """
    import datetime as dt

    from factorzen.discovery.evaluation import evaluate_expressions
    from factorzen.discovery.scoring import DataBundle

    rng = np.random.default_rng(42)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < 80:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    n_stocks = 100
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    # 每日前 5 只有事件（5%）；乘 close 再 ts_rank → 值覆盖 1.0
    nz = 5
    rows = []
    for c_i, c in enumerate(codes):
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            on = c_i < nz
            rows.append({
                "trade_date": dd, "ts_code": c,
                "close": px,
                "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
                "fc_flag": 1.0 if on else 0.0,
                "express_yoy": (0.1 * (c_i + 1)) if on else 0.0,
            })
    daily = pl.DataFrame(rows)
    bundle = DataBundle.build(daily, train_ratio=0.7)

    # 事件叶 + 包装：ts_rank 后值覆盖趋近 1.0，一期 is_sparse 会盲
    event_expr = "ts_rank(mul(express_yoy, close), 5)"
    res_e = evaluate_expressions([event_expr], daily, bundle)
    assert len(res_e) == 1
    re = res_e[0]
    assert re["compile_ok"] is True, re.get("error")
    assert re["ic_train"] is not None
    assert re["subset_mask_leaves"] is not None
    assert set(re["subset_mask_leaves"]) == {"express_yoy"}
    # 掩码触发：subset_n_days_train 有定义（可为 0：日截面 <30）
    assert re.get("subset_n_days_train") is not None
    # 值覆盖被包装抬高 → 一期 is_sparse 不触发
    assert re["nonzero_coverage"] is not None
    assert re["nonzero_coverage"] >= SPARSE_COVERAGE_THRESHOLD
    assert re["is_sparse"] is False

    # 无事件叶：掩码不触发
    dense_expr = "rank(close)"
    res_d = evaluate_expressions([dense_expr], daily, bundle)
    rd = res_d[0]
    assert rd["compile_ok"] is True
    assert rd.get("subset_mask_leaves") in (None, [])
    assert rd["is_sparse"] is False
    assert rd["subset_ic_train"] is None


def test_strong_mask_subset_enters_lift_queue_semantics():
    """强掩码子集 + 主门不过 → lift_queue + sleeve_candidate（与一期旁路同形态）。"""
    cand = {
        "is_sparse": False,
        "subset_mask_leaves": ["express_yoy"],
        "subset_ic_train": 0.06,
        "subset_ic_holdout": 0.05,
        "subset_n_days_train": 55,
    }
    assert is_sleeve_lift_candidate(cand) is True
    reject_category = None
    sleeve_candidate = False
    passed_main = False
    if not passed_main and is_sleeve_lift_candidate(cand):
        reject_category = REJECT_CATEGORY_LIFT_QUEUE
        sleeve_candidate = True
    assert reject_category == REJECT_CATEGORY_LIFT_QUEUE
    assert sleeve_candidate is True
    assert passed_main is False


# ── 7. 二期：死因 ic_too_weak ───────────────────────────────────────────────

def test_classify_reject_category_ic_too_weak():
    from factorzen.discovery.guardrails import (
        REJECT_CATEGORY_HOLDOUT_COVERAGE,
        REJECT_CATEGORY_IC_TOO_WEAK,
        classify_reject_category,
    )

    assert classify_reject_category(
        ["残差IC太弱(|0.0020|<0.010)"]
    ) == REJECT_CATEGORY_IC_TOO_WEAK
    assert classify_reject_category(
        ["train_IC 太弱(|0.0010|<0.015)"]
    ) == REJECT_CATEGORY_IC_TOO_WEAK
    # 反号 / 无信号 → None（方向证据，维持进 known_invalid）
    assert classify_reject_category(
        ["holdout 反号(train=0.0200/holdout=-0.0100)"]
    ) is None
    assert classify_reject_category(
        ["残差holdout反号(train=0.02/holdout=-0.01)"]
    ) is None
    assert classify_reject_category(
        ["holdout无信号(train=0.0200/holdout=0.0000)"]
    ) is None
    # 覆盖不足优先于太弱
    assert classify_reject_category(
        ["train_IC 太弱(|0.0010|<0.015)", "holdout覆盖不足(days=10/需60)"]
    ) == REJECT_CATEGORY_HOLDOUT_COVERAGE
    # 反号优先于太弱（有方向证据 → 仍可 known_invalid）
    assert classify_reject_category(
        ["train_IC 太弱(|0.0010|<0.015)", "holdout 反号(train=0.0010/holdout=-0.0020)"]
    ) is None

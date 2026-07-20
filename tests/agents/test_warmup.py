"""
test_warmup_parity.py：agent 挖掘路径的预热口径：train 与 holdout 必须走同一条裁剪路径。
test_warmup_holdout.py：合并自 agents 相关碎片测试（test_warmup_holdout.py）。
"""

from __future__ import annotations

import ast
import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorzen.discovery.evaluation import _node_to_factor_df
from factorzen.discovery.expression import (
    leaf_warmup_budgets,
    parse_expr,
    warmup_shortfall,
)
from factorzen.validation.holdout import split_holdout


# ==== 来自 test_warmup_parity.py ====
def _synthetic_daily(n_days: int = 120, n_codes: int = 40, start=dt.date(2020, 1, 1)) -> pl.DataFrame:
    """确定性合成帧：close 单调可预测，无随机性，便于 ground-truth 断言。"""
    dates = [start + dt.timedelta(days=i) for i in range(n_days)]
    rows = []
    for d_i, d in enumerate(dates):
        for c_i in range(n_codes):
            close = 10.0 + d_i * 0.1 + c_i
            open_ = 10.0 + d_i * 0.1 + c_i
            high = 11.0 + d_i * 0.1 + c_i
            low = 9.0 + d_i * 0.1 + c_i
            rows.append({
                "trade_date": d,
                "ts_code": f"{c_i:06d}.SZ",
                "close": close,
                "open": open_,
                "high": high,
                "low": low,
                # M1 的 `_factor_values` 不像 agent 路径的 `_preprocess_daily` 那样在缺失时
                # 自动补 *_adj（见 LEAF_FEATURES: "close"→"close_adj"）——直接调用它（跨路径
                # 一致性测试）必须显式提供，否则 `evaluate_materialized` 找不到列。
                "close_adj": close,
                "open_adj": open_,
                "high_adj": high,
                "low_adj": low,
                "vol": 1000.0 + c_i,
                "amount": 5000.0 + c_i,
            })
    return pl.DataFrame(rows)


def test_train_holdout_warmup_path_suite(tmp_path):
    """test_node_to_factor_df_clips_both_bounds；mining_df / holdout_df / bundle 必须建立在 [eval_start, end] 上，预热段只做求值前缀。；train 段 IC 只能算在 [eval_start, eval_end] 内——预热段绝不进 IC 序列。；跨路径一致性：M1 的 run_session(eval_start=) 与 agent 的 evaluate_expressions(eval_start=)；eval_end 单传（无 eval_start）会静默跳过下界裁剪与预热门——必须早失败，而不是悄悄放行。；session eval_start=None（旧调用者）时，滚动表达式必须照常求值，；test_agent_holdout_values_match_full_frame_ground_truth"""
    # -- 原 test_node_to_factor_df_clips_both_bounds --
    def _section_0_test_node_to_factor_df_clips_both_bounds():
        daily = _synthetic_daily(n_days=100)
        node = parse_expr("ts_mean(close, 5)")
        lo, hi = dt.date(2020, 2, 1), dt.date(2020, 3, 1)

        out = _node_to_factor_df(node, daily, eval_start=lo, eval_end=hi)

        assert out["trade_date"].min() == lo
        assert out["trade_date"].max() == hi

    _section_0_test_node_to_factor_df_clips_both_bounds()

    # -- 原 test_split_happens_after_clipping_to_eval_start --
    def _section_1_test_split_happens_after_clipping_to_eval_start():
        from factorzen.agents.team_orchestrator import _prepare_segments

        full = _synthetic_daily(n_days=120)
        mining_df, holdout_df, holdout_start = _prepare_segments(
            full, eval_start="20200201", holdout_ratio=0.2)

        assert mining_df["trade_date"].min() == dt.date(2020, 2, 1)
        assert holdout_df["trade_date"].min() == holdout_start
        assert mining_df["trade_date"].max() < holdout_start

    _section_1_test_split_happens_after_clipping_to_eval_start()

    # -- 原 test_train_ic_dates_exclude_warmup_segment --
    def _section_2_test_train_ic_dates_exclude_warmup_segment():
        from factorzen.discovery.evaluation import evaluate_expressions
        from factorzen.discovery.scoring import DataBundle

        full = _synthetic_daily(n_days=120)                     # 含预热段的完整帧
        eval_start = dt.date(2020, 2, 1)
        bundle = DataBundle.build(full)                          # 生产真实拓扑：预热段仍在 bundle 里
        train_end = dt.datetime.strptime(bundle.train_end, "%Y%m%d").date()

        res = evaluate_expressions(["ts_mean(close, 5)"], full, bundle,
                                   eval_start=eval_start, eval_end=train_end)[0]

        assert res["compile_ok"] is True
        n_cal = full.filter(
            (pl.col("trade_date") >= eval_start) & (pl.col("trade_date") <= train_end)
        )["trade_date"].n_unique()
        assert 0 < res["n_train"] <= n_cal

    _section_2_test_train_ic_dates_exclude_warmup_segment()

    # -- 原 test_m1_and_agent_paths_agree_on_train_ic_days --
    def _section_3_test_m1_and_agent_paths_agree_on_train_ic_days():
        from factorzen.agents.team_orchestrator import _prepare_segments
        from factorzen.discovery.evaluation import evaluate_expressions
        from factorzen.discovery.mining_session import _factor_values
        from factorzen.discovery.scoring import DataBundle, quick_fitness

        full = _synthetic_daily(n_days=120)
        eval_start_s, eval_start = "20200201", dt.date(2020, 2, 1)
        mining_df, _, _ = _prepare_segments(full, eval_start=eval_start_s, holdout_ratio=0.2)
        bundle = DataBundle.build(mining_df)
        train_end = dt.datetime.strptime(bundle.train_end, "%Y%m%d").date()

        expr = "ts_mean(close, 5)"
        agent_n = evaluate_expressions([expr], full, bundle,
                                       eval_start=eval_start, eval_end=train_end)[0]["n_train"]
        # `_factor_values`（M1 路径）的 eval_start 是 "YYYYMMDD" 字符串（`_cut_literal` 契约，
        # 与 `run_session` 一致），不是 date——与 agent 路径 `evaluate_expressions` 的 date 契约不同。
        m1_fdf = _factor_values(parse_expr(expr), full, eval_start=eval_start_s)
        m1_n = int(quick_fitness(m1_fdf, bundle, segment="train")["n"])

        assert agent_n == m1_n

    _section_3_test_m1_and_agent_paths_agree_on_train_ic_days()

    # -- 原 test_eval_end_without_eval_start_raises --
    def _section_4_test_eval_end_without_eval_start_raises():
        from factorzen.discovery.evaluation import evaluate_expressions

        full = _synthetic_daily(n_days=120)
        with pytest.raises(ValueError, match="eval_start"):
            evaluate_expressions(["ts_mean(close, 5)"], full, _bundle_for(full),
                                 eval_start=None, eval_end=dt.date(2020, 3, 1))

    _section_4_test_eval_end_without_eval_start_raises()

    # -- 原 test_eval_start_none_still_evaluates_rolling_expressions --
    def _section_5_test_eval_start_none_still_evaluates_rolling_expressions(tmp_path):
        from factorzen.agents.team_orchestrator import run_team_agent

        full = _synthetic_daily(n_days=120)
        expr = "ts_mean(close, 20)"
        result = run_team_agent(
            full, _scripted_team_fixed_expr(expr), n_rounds=1, seed=1,
            index_path=str(tmp_path / "idx.jsonl"), heal_rounds=0,
        )

        rolling = [a for a in result.state.attempts if a.expression == expr]
        assert rolling, f"{expr} 应至少被评估一次；实际 attempts={[a.expression for a in result.state.attempts]}"
        assert rolling[0].ic_train is not None, (
            f"eval_start=None 时 {expr} 被判『预热不足』或求值失败: error={rolling[0].error}"
        )
        assert rolling[0].n_train is not None and rolling[0].n_train > 0

    _tp5 = tmp_path / "_s5"
    _tp5.mkdir(exist_ok=True)
    _section_5_test_eval_start_none_still_evaluates_rolling_expressions(_tp5)

    # -- 原 test_agent_holdout_values_match_full_frame_ground_truth --
    def _section_6_test_agent_holdout_values_match_full_frame_ground_truth():
        from factorzen.discovery.evaluation import _node_to_factor_df
        from factorzen.discovery.expression import parse_expr

        daily = _daily()
        _mining, _holdout, hstart = split_holdout(daily, holdout_ratio=0.2)
        node = parse_expr(_ROLLING)

        truth = _truth_holdout_values(lambda df: _node_to_factor_df(node, df), daily, hstart)
        warmed = _node_to_factor_df(node, daily, eval_start=hstart).sort(["ts_code", "trade_date"])

        assert warmed.height == truth.height, "预热后 holdout 行数应与 ground truth 一致"
        got = warmed["factor_value"].to_numpy()
        want = truth["factor_value"].to_numpy()
        assert np.allclose(got, want), "扩窗预热的因子值必须与「全样本算完再切」逐值相同"

    _section_6_test_agent_holdout_values_match_full_frame_ground_truth()


def test_warmup_bars_parity_suite():
    """预热段有交易日、但该叶子全 null → 可用预热必须是 0，不能按交易日数报充足。；NaN 预热单元格（非 null）不算可用历史——polars 里 NaN 不是 null。；两个当前只靠代码自证、无断言守护的边界情形。；warmup_bars 属 discovery 共享层（M1 与 agent 双路径共用），且用传入的 leaf_map；leaf_lookbacks:每个叶子沿『根→该叶』路径的窗口累加,各叶分别给出。；核心复现:深 raw 叶路(close, need=20, 恰好有 20 根预热)不该被浅派生叶"""
    # -- 原 test_warmup_bars_counts_nonnull_history_per_leaf --
    def _section_0_test_warmup_bars_counts_nonnull_history_per_leaf():
        from factorzen.discovery.evaluation import _preprocess_daily
        from factorzen.discovery.expression import warmup_bars

        daily = _synthetic_daily(n_days=100)
        cutoff = dt.date(2020, 2, 1)
        # dv_ttm：cutoff 之前全 null，之后有值
        daily = daily.with_columns(
            pl.when(pl.col("trade_date") >= cutoff).then(pl.lit(2.0)).otherwise(None).alias("dv_ttm")
        )
        # close_adj 显式覆盖：cutoff 之前全 null，之后等于 close —— 与 close 本身（全非空）区分开
        daily = daily.with_columns(
            pl.when(pl.col("trade_date") >= cutoff).then(pl.col("close")).otherwise(None).alias("close_adj")
        )
        prepped = _preprocess_daily(daily)

        assert warmup_bars(parse_expr("dv_ttm"), prepped, cutoff) == 0
        # "close" 叶子经 LEAF_FEATURES 映射到 close_adj，预热段全 null → 必须是 0，
        # 即便原始 "close" 列在预热段全非空（漏查映射的实现会误报满预热天数）。
        assert warmup_bars(parse_expr("close"), prepped, cutoff) == 0
        # open 在预热段有值（未被覆盖）→ 预热 bar 数 = cutoff 之前的交易日数
        n_before = daily.filter(pl.col("trade_date") < cutoff)["trade_date"].n_unique()
        assert warmup_bars(parse_expr("open"), prepped, cutoff) == n_before
        # 混合表达式取各叶子最小值 → 被 dv_ttm 拉到 0（open 满预热，min 必须仍是 0）
        assert warmup_bars(parse_expr("add(open, dv_ttm)"), prepped, cutoff) == 0

    _section_0_test_warmup_bars_counts_nonnull_history_per_leaf()

    # -- 原 test_warmup_bars_excludes_nan_not_just_null --
    def _section_1_test_warmup_bars_excludes_nan_not_just_null():
        from factorzen.discovery.evaluation import _preprocess_daily
        from factorzen.discovery.expression import warmup_bars

        daily = _synthetic_daily(n_days=100)
        cutoff = dt.date(2020, 2, 1)
        # dv_ttm：预热段全是 NaN（非 null），cutoff 之后正常有值
        daily = daily.with_columns(
            pl.when(pl.col("trade_date") >= cutoff)
            .then(pl.lit(2.0))
            .otherwise(pl.lit(float("nan")))
            .alias("dv_ttm")
        )
        prepped = _preprocess_daily(daily)
        # 确认预热段确实是非空的 NaN，而不是 null（否则这条测试测的是别的东西）
        warm = prepped.filter(pl.col("trade_date") < cutoff)
        assert warm["dv_ttm"].null_count() == 0
        assert warm["dv_ttm"].is_nan().all()

        assert warmup_bars(parse_expr("dv_ttm"), prepped, cutoff) == 0

    _section_1_test_warmup_bars_excludes_nan_not_just_null()

    # -- 原 test_warmup_bars_absent_column_and_constant_expr --
    def _section_2_test_warmup_bars_absent_column_and_constant_expr():
        from factorzen.discovery.evaluation import _preprocess_daily
        from factorzen.discovery.expression import warmup_bars

        daily = _synthetic_daily(n_days=100)
        cutoff = dt.date(2020, 2, 1)
        prepped = _preprocess_daily(daily)

        # 叶子映射到的列在帧里完全不存在（如未拉取 daily_basic）→ 视为零预热，不报错
        assert "turnover_rate_f" not in prepped.columns
        assert warmup_bars(parse_expr("turnover_rate_f"), prepped, cutoff) == 0

        # 纯常数表达式无叶子 → 预热 bar 数直接是预热段的交易日数
        n_before = daily.filter(pl.col("trade_date") < cutoff)["trade_date"].n_unique()
        assert warmup_bars(parse_expr("1.0"), prepped, cutoff) == n_before

    _section_2_test_warmup_bars_absent_column_and_constant_expr()

    # -- 原 test_warmup_bars_importable_from_discovery_and_respects_leaf_map --
    def _section_3_test_warmup_bars_importable_from_discovery_and_respects_leaf_map():
        from factorzen.discovery.expression import warmup_bars

        daily = _synthetic_daily(n_days=100)
        cutoff = dt.date(2020, 2, 1)
        # close_adj 预热段全 null、vol 全非空 —— 让默认映射与 leaf_map 覆盖产生不同结果
        prepped = daily.with_columns(   # close 叶子只用 close_adj/vol，无需派生列
            pl.when(pl.col("trade_date") >= cutoff).then(pl.col("close")).otherwise(None).alias("close_adj")
        )
        node = parse_expr("close")
        n_before = daily.filter(pl.col("trade_date") < cutoff)["trade_date"].n_unique()

        # 默认 LEAF_FEATURES：close→close_adj（预热段全 null）→ 0
        assert warmup_bars(node, prepped, cutoff) == 0
        # 传 leaf_map 覆盖：close→vol（预热段全非空）→ n_before
        assert warmup_bars(node, prepped, cutoff, leaf_map={"close": "vol"}) == n_before

    _section_3_test_warmup_bars_importable_from_discovery_and_respects_leaf_map()

    # -- 原 test_leaf_lookbacks_are_per_leaf_not_global_max --
    def _section_4_test_leaf_lookbacks_are_per_leaf_not_global_max():
        from factorzen.discovery.expression import leaf_lookbacks

        node = parse_expr("mul(ts_mean(close, 20), ts_sum(ret_1d, 5))")
        assert leaf_lookbacks(node) == {"close": 20, "ret_1d": 5}

        deep = parse_expr("mul(ts_zscore(delta(div(close, pe_ttm), 60), 120), ts_sum(ret_1d, 20))")
        assert leaf_lookbacks(deep) == {"close": 180, "pe_ttm": 180, "ret_1d": 20}

        # 同叶多次出现取最大:close 一处 need=25、一处 need=3 → 25
        both = parse_expr("add(ts_mean(delta(close, 5), 20), ts_mean(close, 3))")
        assert leaf_lookbacks(both)["close"] == 25

    _section_4_test_leaf_lookbacks_are_per_leaf_not_global_max()

    # -- 原 test_warmup_shortfall_not_dragged_by_shallow_derived_leaf --
    def _section_5_test_warmup_shortfall_not_dragged_by_shallow_derived_leaf():
        from factorzen.discovery.evaluation import _preprocess_daily
        from factorzen.discovery.expression import (
            required_lookback,
            warmup_bars,
            warmup_shortfall,
        )

        prepped = _preprocess_daily(_synthetic_daily(n_days=100))
        cutoff = dt.date(2020, 1, 21)                     # 前 20 个交易日作预热
        node = parse_expr("mul(ts_mean(close, 20), ts_sum(ret_1d, 5))")

        # 前提:旧口径 min(close=20, ret_1d=19)=19 < required=20 → 旧门会假拒
        assert warmup_bars(node, prepped, cutoff) == 19
        assert required_lookback(node) == 20
        # 且 close 单叶其实够(20≥20)——所以这是假拒绝,不是真欠预热
        assert warmup_bars(parse_expr("ts_mean(close, 20)"), prepped, cutoff) == 20

        # 新逐叶门:close 20≥20、ret_1d 19≥5 → 无欠预热 → None
        assert warmup_shortfall(node, prepped, cutoff) is None

    _section_5_test_warmup_shortfall_not_dragged_by_shallow_derived_leaf()


def _bundle_for(sample: pl.DataFrame):
    from factorzen.discovery.scoring import DataBundle
    return DataBundle.build(sample)


def test_insufficient_warmup_expression_is_rejected_not_silently_noisy():
    """预热不足的表达式必须出声拒绝（ic/ir=None），而不是发窗口不满的噪声值。

    反例保护：operators._MIN = 3 意味着 250 日窗口只要 3 个观测就出值，
    静默通过时它会带着噪声 IC 进入 DSR 的 IR 池。
    """
    from factorzen.discovery.evaluation import evaluate_expressions

    full = _synthetic_daily(n_days=120)
    eval_start = dt.date(2020, 2, 1)          # 预热段仅 31 个交易日
    sample = full.filter(pl.col("trade_date") >= eval_start)
    bundle = _bundle_for(sample)
    train_end = dt.datetime.strptime(bundle.train_end, "%Y%m%d").date()

    res = evaluate_expressions(["ts_mean(close, 250)"], full, bundle,
                               eval_start=eval_start, eval_end=train_end)[0]

    assert res["compile_ok"] is True
    assert res["ic_train"] is None and res["ir_train"] is None
    assert res["n_train"] == 0
    assert "预热不足" in res["error"]


# ── Task 1.4: orchestrator 先裁样本再 split（train/holdout/PBO 三处口径统一）─────


# ── 假拒绝修复:逐叶 path-lookback,浅派生叶不得拖垮深 raw 路 ──────────────────
#
# smoke 照出:门用『全叶最小 warmup』对比『最深路径 max lookback』——跨叶错配。
# `mul(ts_zscore(delta(div(close,pe_ttm),60),120), ts_sum(ret_1d,20))` 的深路
# (close/pe_ttm, need=180)明明有 ≥180 根预热,却被只需 20 的派生叶 ret_1d(少 1
# 根 warmup,=179)拖成 min=179 < 180 → 假拒。正确判定必须逐叶:每个叶子只需填满
# 它上方的窗口。下面用可控合成帧复刻该拓扑(deep need == close 可用预热,ret_1d 少 1)。


def test_warmup_shortfall_flags_genuinely_underwarmed_leaf():
    """反向守卫(防修过头):某叶真的够不着自身 need 时必须报欠预热,返回最欠的叶。"""
    from factorzen.discovery.evaluation import _preprocess_daily
    from factorzen.discovery.expression import warmup_shortfall

    prepped = _preprocess_daily(_synthetic_daily(n_days=100))
    cutoff = dt.date(2020, 1, 11)                     # 只有 10 个预热交易日
    sf = warmup_shortfall(parse_expr("ts_mean(close, 20)"), prepped, cutoff)
    assert sf is not None
    leaf, need, have = sf
    assert leaf == "close" and need == 20 and have == 10


def test_m1_underwarmed_false_for_mixed_depth_expr():
    """M1 路径(mining_session._underwarmed):mixed 表达式不再假拒,真欠仍拒。"""
    from factorzen.discovery.evaluation import _preprocess_daily
    from factorzen.discovery.mining_session import _underwarmed

    prepped = _preprocess_daily(_synthetic_daily(n_days=100))
    mixed = parse_expr("mul(ts_mean(close, 20), ts_sum(ret_1d, 5))")
    # eval_start 是 "YYYYMMDD" 字符串(_cut_literal 契约,与 run_session 一致)
    assert _underwarmed(mixed, prepped, "20200121") is False
    assert _underwarmed(parse_expr("ts_mean(close, 20)"), prepped, "20200111") is True


def test_agent_evaluate_no_false_warmup_rejection_for_mixed_expr():
    """agent 路径(evaluate_expressions):mixed 表达式不得被判『预热不足』。

    与 M1 用同一道共享门(warmup_shortfall),两条路对同一 mixed 拓扑判定一致。
    """
    from factorzen.discovery.evaluation import evaluate_expressions
    from factorzen.discovery.scoring import DataBundle

    full = _synthetic_daily(n_days=120)
    eval_start = dt.date(2020, 1, 21)                 # 20 个预热交易日,恰卡 close need=20
    bundle = DataBundle.build(full)
    train_end = dt.datetime.strptime(bundle.train_end, "%Y%m%d").date()

    res = evaluate_expressions(["mul(ts_mean(close, 20), ts_sum(ret_1d, 5))"],
                               full, bundle, eval_start=eval_start, eval_end=train_end)[0]
    assert res["compile_ok"] is True
    assert "预热不足" not in (res.get("error") or ""), res


def _scripted_team_fixed_expr(expr: str):
    """Hypothesis→Coder→Critic(keep) 固定表达式的一轮脚本，循环复用（同 test_team_orchestrator.py）。"""
    import json as _json

    hyp = _json.dumps({"hypotheses": ["动量"]})
    code = _json.dumps({"expressions": [expr]})
    crit = _json.dumps({"verdict": "keep", "reason": "ok"})
    seq = [hyp, code, crit] * 10
    i = {"k": 0}

    def fn(messages):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v

    return fn


# ==== 来自 test_warmup_holdout.py ====
# ==== 来自 test_holdout_warmup.py ====
_SRC = Path(__file__).resolve().parents[2] / "src" / "factorzen"


def _daily(n_stocks: int = 40, n_days: int = 260, seed: int = 5) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2021, 1, 4)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{600000 + i:06d}.SH" for i in range(n_stocks)]:
        px = rng.uniform(8, 15)
        for dd in days:
            px = float(max(px * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": dd, "ts_code": c,
                         "close": px, "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                         "close_adj": px, "open_adj": px * 0.99,
                         "high_adj": px * 1.01, "low_adj": px * 0.98, "pre_close": px,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


_ROLLING = "ts_mean(close, 20)"


def _truth_holdout_values(evaluator, daily: pl.DataFrame, holdout_start) -> pl.DataFrame:
    """ground truth：在完整帧上求值，再切出 holdout 段。"""
    full = evaluator(daily)
    return full.filter(pl.col("trade_date") >= holdout_start).sort(["ts_code", "trade_date"])


# ── Agent 路径 ──────────────────────────────────────────────────────────────


def test_agent_holdout_without_warmup_is_biased_at_the_boundary():
    """判别性前置：不预热确实产生偏差——否则本文件的修复无意义。"""
    from factorzen.discovery.evaluation import _node_to_factor_df
    from factorzen.discovery.expression import parse_expr

    daily = _daily()
    _mining, holdout_df, hstart = split_holdout(daily, holdout_ratio=0.2)
    node = parse_expr(_ROLLING)

    truth = _truth_holdout_values(lambda df: _node_to_factor_df(node, df), daily, hstart)
    naive = _node_to_factor_df(node, holdout_df).sort(["ts_code", "trade_date"])

    joined = truth.join(naive, on=["trade_date", "ts_code"], how="inner", suffix="_naive")
    diff = np.abs(joined["factor_value"].to_numpy() - joined["factor_value_naive"].to_numpy())
    assert diff.max() > 1e-6, "若无偏差，说明测试数据/算子选得不对，修复将无从验证"


def test_agent_holdout_warmup_leaks_no_future_information():
    """PIT：holdout 段的因子值不得依赖 holdout_start 之后的数据。

    做法——把 holdout 段**之后**的价格全部改掉，重算，holdout 首日的值必须不变。
    （若求值用了未来数据，改动会渗回来。）
    """
    from factorzen.discovery.evaluation import _node_to_factor_df
    from factorzen.discovery.expression import parse_expr

    daily = _daily()
    _mining, _holdout, hstart = split_holdout(daily, holdout_ratio=0.2)
    node = parse_expr(_ROLLING)

    dates = sorted(daily["trade_date"].unique().to_list())
    later = dates[dates.index(hstart) + 5]          # holdout 内部靠后的某天
    tampered = daily.with_columns(
        pl.when(pl.col("trade_date") >= later).then(pl.col("close") * 3.0)
        .otherwise(pl.col("close")).alias("close")
    )

    base = _node_to_factor_df(node, daily, eval_start=hstart)
    tamp = _node_to_factor_df(node, tampered, eval_start=hstart)
    first_day = base.filter(pl.col("trade_date") == hstart).sort("ts_code")
    first_day_t = tamp.filter(pl.col("trade_date") == hstart).sort("ts_code")

    assert np.allclose(first_day["factor_value"].to_numpy(),
                       first_day_t["factor_value"].to_numpy()), \
        "篡改 holdout 后段数据改变了 holdout 首日的因子值 —— 存在未来函数"


def test_agent_preprocess_pre_close_uses_prior_session_when_warmed():
    """`pre_close` 在只喂 holdout 帧时被 fill_null 成当日 close；预热后应取 mining 末日 close。"""
    from factorzen.discovery.evaluation import _preprocess_daily

    daily = _daily()
    mining, holdout_df, hstart = split_holdout(daily, holdout_ratio=0.2)
    code = daily["ts_code"][0]

    prev_close = (mining.filter(pl.col("ts_code") == code)
                  .sort("trade_date")["close"].to_list()[-1])

    naive = _preprocess_daily(holdout_df.drop("pre_close"))
    warmed = _preprocess_daily(daily.drop("pre_close"))

    def _pc(df):
        return (df.filter((pl.col("ts_code") == code) & (pl.col("trade_date") == hstart))
                ["pre_close"].to_list()[0])

    assert _pc(warmed) == pytest.approx(prev_close), "预热后应取上一交易日收盘"
    assert _pc(naive) != pytest.approx(prev_close), "判别性前置：不预热时确实取错"


# ── M1 路径（双路径一致地错 → 两侧都要修）────────────────────────────────────


def test_holdout_warmup_window_suite(tmp_path, monkeypatch):
    """test_m1_holdout_values_match_full_frame_ground_truth；集成：`run_session` 对 holdout 求值时必须传完整帧 + eval_start，而非已切片的 holdout_df。；`warmup_daily=None` 缺省会**回退到不预热的旧行为**。；`warmup_daily` 必须是**完整帧**（mining + holdout），不是 mining 切片。；test_team_orchestrator_passes_the_full_frame_not_the_mining_slice；M1 与 Agent 在 holdout 段的因子值必须逐值相同——双路径登记簿。"""
    # -- 原 test_m1_holdout_values_match_full_frame_ground_truth --
    def _section_0_test_m1_holdout_values_match_full_frame_ground_truth():
        from factorzen.discovery.expression import parse_expr
        from factorzen.discovery.mining_session import _factor_values

        daily = _daily()
        _mining, _holdout, hstart = split_holdout(daily, holdout_ratio=0.2)
        node = parse_expr(_ROLLING)

        truth = _truth_holdout_values(lambda df: _factor_values(node, df), daily, hstart)
        warmed = _factor_values(node, daily, eval_start=hstart.strftime("%Y%m%d")).sort(
            ["ts_code", "trade_date"])

        assert warmed.height == truth.height
        assert np.allclose(warmed["factor_value"].to_numpy(), truth["factor_value"].to_numpy())

    _section_0_test_m1_holdout_values_match_full_frame_ground_truth()

    # -- 原 test_m1_run_session_warms_up_holdout --
    def _section_1_test_m1_run_session_warms_up_holdout(tmp_path, monkeypatch):
        from factorzen.discovery import mining_session as ms

        seen: list[dict] = []
        real = ms._factor_values

        def spy(node, daily, eval_start=None, leaf_map=None):
            seen.append({"rows": daily.height, "eval_start": eval_start})
            return real(node, daily, eval_start, leaf_map)

        monkeypatch.setattr(ms, "_factor_values", spy)
        ms.run_session(_daily(), n_trials=20, top_k=3, seed=3, method="random",
                       holdout_ratio=0.2, out_dir=str(tmp_path))

        holdout_calls = [c for c in seen if c["eval_start"] is not None]
        assert holdout_calls, "holdout 求值必须带 eval_start（扩窗预热后裁剪）"

    monkeypatch.undo()
    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_m1_run_session_warms_up_holdout(_tp1, monkeypatch)

    # -- 原 test_every_production_caller_passes_warmup_daily --
    def _section_2_test_every_production_caller_passes_warmup_daily():
        offenders: list[str] = []
        for path in _SRC.rglob("*.py"):
            if path.name == "nodes.py":          # 定义处
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for n in ast.walk(tree):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                        and n.func.id == "node_guardrails"
                        and not any(kw.arg == "warmup_daily" for kw in n.keywords)):
                    offenders.append(f"{path.relative_to(_SRC).as_posix()}:{n.lineno}")

        assert not offenders, (
            "这些 node_guardrails 调用漏传 warmup_daily，将静默退回「holdout 不预热」的旧行为："
            f"{offenders}"
        )

    monkeypatch.undo()
    _section_2_test_every_production_caller_passes_warmup_daily()

    # -- 原 test_single_agent_orchestrator_passes_the_full_frame_not_the_mining_slice --
    def _section_3_test_single_agent_orchestrator_passes_the_full_frame_not_the_mining_slice(monkeypatch):
        import factorzen.agents.orchestrator as orch
        from factorzen.agents.orchestrator import run_llm_agent

        seen: dict = {}
        monkeypatch.setattr(orch, "node_guardrails", _spy_guardrails(seen))
        run_llm_agent(_daily(), _fake_llm(), n_rounds=1, seed=1, heal_rounds=0)

        assert seen["warmup"] == seen["mining"] + seen["holdout"], (
            f"warmup_daily 应是完整帧（{seen['mining']}+{seen['holdout']} 行），"
            f"实得 {seen['warmup']} 行"
        )

    monkeypatch.undo()
    _section_3_test_single_agent_orchestrator_passes_the_full_frame_not_the_mining_slice(monkeypatch)

    # -- 原 test_team_orchestrator_passes_the_full_frame_not_the_mining_slice --
    def _section_4_test_team_orchestrator_passes_the_full_frame_not_the_mining_slice(tmp_path, monkeypatch):
        import factorzen.agents.team_orchestrator as team
        from factorzen.agents.team_orchestrator import run_team_agent

        seen: dict = {}
        monkeypatch.setattr(team, "node_guardrails", _spy_guardrails(seen))
        run_team_agent(_daily(), _fake_llm(), n_rounds=1, seed=1,
                       index_path=str(tmp_path / "i.jsonl"), heal_rounds=0)

        assert seen["warmup"] == seen["mining"] + seen["holdout"], (
            f"warmup_daily 应是完整帧，实得 {seen['warmup']} 行"
        )

    monkeypatch.undo()
    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_team_orchestrator_passes_the_full_frame_not_the_mining_slice(_tp4, monkeypatch)

    # -- 原 test_both_paths_produce_identical_holdout_values --
    def _section_5_test_both_paths_produce_identical_holdout_values():
        from factorzen.discovery.evaluation import _node_to_factor_df
        from factorzen.discovery.expression import parse_expr
        from factorzen.discovery.mining_session import _factor_values

        daily = _daily()
        _mining, _holdout, hstart = split_holdout(daily, holdout_ratio=0.2)
        node = parse_expr(_ROLLING)

        a = _node_to_factor_df(node, daily, eval_start=hstart).sort(["ts_code", "trade_date"])
        m = _factor_values(node, daily, eval_start=hstart.strftime("%Y%m%d")).sort(
            ["ts_code", "trade_date"])

        assert a.height == m.height
        assert np.allclose(a["factor_value"].to_numpy(), m["factor_value"].to_numpy())

    monkeypatch.undo()
    _section_5_test_both_paths_produce_identical_holdout_values()


# ── 架构守卫：默认值不许成为「静默不修」的藏身处 ──────────────────────────────


def _spy_guardrails(seen: dict):
    def fake(state, *, daily, holdout_df, bundle, ledger, top_k=5, dsr_alpha=0.05,
             warmup_daily=None, eval_start=None, **_kwargs):
        seen["warmup"] = None if warmup_daily is None else warmup_daily.height
        seen["mining"] = daily.height
        seen["holdout"] = holdout_df.height
        return state
    return fake


def _fake_llm():
    import json as _json
    st = {"round": -1}

    def fn(messages):
        system = messages[0]["content"]
        if "consistent" in system:
            return _json.dumps({"consistent": True, "reason": "ok"})
        if "verdict" in system:
            return _json.dumps({"verdict": "keep", "reason": "ok"})
        if '"expressions"' in system and '"hypothesis"' not in system:
            return _json.dumps({"expressions": ["ts_mean(close,5)"]})
        if '"hypotheses"' in system:
            return _json.dumps({"hypotheses": ["动量"]})
        st["round"] += 1
        return _json.dumps({"hypothesis": "h", "expressions": [f"ts_mean(close,{5 + st['round']})"],
                            "rationale": "r"})
    return fn


# ── 两条路径口径一致 ────────────────────────────────────────────────────────


# ==== 来自 test_leaf_warmup_budgets.py ====
def _workdays(anchor: dt.date, count: int, *, forward: bool) -> list[dt.date]:
    out: list[dt.date] = []
    d = anchor
    step = dt.timedelta(days=1 if forward else -1)
    while len(out) < count:
        if d.weekday() < 5:
            out.append(d)
        d += step
    return out


def _frame_with_north_ratio(n_warm: int = 100, n_after: int = 40,
                            eval_start: dt.date = dt.date(2022, 1, 3)):
    """两只股票的帧：north_ratio 在 eval_start 前恰好 n_warm 个非空交易日。"""
    before = sorted(_workdays(eval_start - dt.timedelta(days=1), n_warm, forward=False))
    after = _workdays(eval_start, n_after, forward=True)
    days = before + after
    rows = []
    for c in ("000001.SZ", "000002.SZ"):
        for i, d in enumerate(days):
            rows.append({"trade_date": d, "ts_code": c,
                         "close": 10.0 + i * 0.1, "north_ratio": float(i)})
    return pl.DataFrame(rows), eval_start


# ── B4.1 一致性（关键）─────────────────────────────────────────────────────────

def test_leaf_warmup_budget_suite(tmp_path):
    """test_budget_missing_column_is_zero；随机窗口下逐一对照：budget[leaf] 恒等于 shortfall 的 have（或充分时的 have）。；test_build_agent_messages_none_is_byte_identical；budgets 为空 dict（无短历史叶子）→ 不加文案，避免 prompt 膨胀。；test_syntax_prompt_none_is_byte_identical；双路径登记簿：两侧预算文案共用同一 fragment，杜绝漂移。；预热不足的表达式经 revise_from_error 回灌 → 修正版被评估；且**只回灌一轮**："""
    # -- 原 test_budget_missing_column_is_zero --
    def _section_0_test_budget_missing_column_is_zero():
        prepped, es = _frame_with_north_ratio(n_warm=30)
        # or_yoy 列不在帧里 → 预算 0（与 warmup_bars_by_leaf 列缺失记 0 一致）
        budgets = leaf_warmup_budgets(prepped, es, ["north_ratio", "or_yoy"])
        assert budgets["or_yoy"] == 0
        assert budgets["north_ratio"] == 30

    _section_0_test_budget_missing_column_is_zero()

    # -- 原 test_budget_matches_warmup_shortfall_across_random_windows --
    def _section_1_test_budget_matches_warmup_shortfall_across_random_windows():
        prepped, es = _frame_with_north_ratio(n_warm=120)
        have_budget = leaf_warmup_budgets(prepped, es, ["north_ratio"])["north_ratio"]
        for w in (10, 50, 119, 120, 121, 300):
            sf = warmup_shortfall(parse_expr(f"ts_mean(north_ratio, {w})"), prepped, es)
            if w > have_budget:
                assert sf is not None and sf[2] == have_budget
            else:
                assert sf is None  # have 足够，无缺口

    _section_1_test_budget_matches_warmup_shortfall_across_random_windows()

    # -- 原 test_build_agent_messages_none_is_byte_identical --
    def _section_2_test_build_agent_messages_none_is_byte_identical():
        from factorzen.llm.generation import build_agent_messages
        base = build_agent_messages(["ts_mean", "rank"], ["close", "north_ratio"])
        with_none = build_agent_messages(["ts_mean", "rank"], ["close", "north_ratio"],
                                         leaf_budgets=None)
        assert base == with_none
        assert "历史较短" not in base[0]["content"]  # 零回归：无预算时不加预算文案

    _section_2_test_build_agent_messages_none_is_byte_identical()

    # -- 原 test_build_agent_messages_empty_budget_no_text --
    def _section_3_test_build_agent_messages_empty_budget_no_text():
        from factorzen.llm.generation import build_agent_messages
        base = build_agent_messages(["ts_mean"], ["close"])
        empty = build_agent_messages(["ts_mean"], ["close"], leaf_budgets={})
        assert base == empty

    _section_3_test_build_agent_messages_empty_budget_no_text()

    # -- 原 test_syntax_prompt_none_is_byte_identical --
    def _section_4_test_syntax_prompt_none_is_byte_identical():
        from factorzen.agents.roles.coder import _syntax_prompt
        assert _syntax_prompt() == _syntax_prompt(leaf_budgets=None)
        assert "历史较短" not in _syntax_prompt()

    _section_4_test_syntax_prompt_none_is_byte_identical()

    # -- 原 test_budget_hint_shared_between_two_paths --
    def _section_5_test_budget_hint_shared_between_two_paths():
        from factorzen.agents.roles.coder import _syntax_prompt
        from factorzen.llm.generation import build_agent_messages, format_leaf_budget_hint
        b = {"north_ratio": 238, "or_yoy": 424}
        hint = format_leaf_budget_hint(b)
        assert hint  # 非空
        assert hint in build_agent_messages(["rank"], ["close"], leaf_budgets=b)[0]["content"]
        assert hint in _syntax_prompt(leaf_budgets=b)

    _section_5_test_budget_hint_shared_between_two_paths()

    # -- 原 test_warmup_error_refeed_one_round --
    def _section_6_test_warmup_error_refeed_one_round(tmp_path):
        from factorzen.agents.team_orchestrator import run_team_agent

        daily, eval_start = _mock_daily_with_north_ratio()
        calls: list[str] = []

        hyp = json.dumps({"hypotheses": ["北向持股占比高的股票未来收益更高"]})
        # 预热 ~60 根：50+50=100 > 60 必预热不足；单字面量 50 ≤ 预算不被钳
        bad = json.dumps({"expressions": ["ts_mean(ts_mean(north_ratio, 50), 50)"]})
        # 修正版 45+45=90 > 60 仍预热不足
        revised = json.dumps({"expressions": ["ts_mean(ts_mean(north_ratio, 45), 45)"]})
        keep = json.dumps({"verdict": "keep", "reason": "ok"})

        def fn(messages):
            text = "\n".join(m["content"] for m in messages)
            calls.append(text)
            if "诊断信息" in text:            # revise_from_error（回灌）
                return revised
            if "翻译成" in text:              # write_expressions
                return bad
            if "风控审计员" in text:          # critic
                return keep
            return hyp                        # propose_hypotheses

        res = run_team_agent(daily, fn, n_rounds=1, seed=7, heal_rounds=0,
                             index_path=str(tmp_path / "e.jsonl"), eval_start=eval_start)

        refeed_calls = [t for t in calls if "诊断信息" in t]
        assert len(refeed_calls) == 1, f"应恰好回灌一轮，实得 {len(refeed_calls)} 次"
        exprs_seen = {a.expression for a in res.state.attempts}
        assert "ts_mean(ts_mean(north_ratio, 50), 50)" in exprs_seen, \
            "原始预热不足表达式应被评估并落 attempt"
        assert "ts_mean(ts_mean(north_ratio, 45), 45)" in exprs_seen, "修正版应被评估（回灌一轮）"

    _tp6 = tmp_path / "_s6"
    _tp6.mkdir(exist_ok=True)
    _section_6_test_warmup_error_refeed_one_round(_tp6)


# ── B4.2 build_agent_messages（单 agent 路径）─────────────────────────────────


# ── B4.3 coder._syntax_prompt（team 路径）+ 双路径 parity ──────────────────────


# ── B4.4 预热错误回灌（只回灌一轮）────────────────────────────────────────────
def _mock_daily_with_north_ratio(n_days=250, n_stocks=20, eval_start_idx=60, seed=3):
    """含预热前缀的帧：north_ratio 全程有值但预热段仅 eval_start_idx 天。"""
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2021, 1, 4)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    eval_start = days[eval_start_idx]
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        px = 10.0
        for i, dd in enumerate(days):
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
                         "north_ratio": float(abs(rng.standard_normal()) + i * 0.01)})
    return pl.DataFrame(rows), eval_start.strftime("%Y%m%d")


    # 只回灌一轮：修正版仍预热不足，但不再触发第二次 revise_from_error（上面已断言 ==1）


def test_oversized_window_literal_clamped_no_refeed(tmp_path: Path):
    """W5b：单个超预算窗口字面量被本地钳制 → 评估正常、不触发 refeed LLM。"""
    from factorzen.agents.team_orchestrator import run_team_agent

    daily, eval_start = _mock_daily_with_north_ratio()
    calls: list[str] = []

    hyp = json.dumps({"hypotheses": ["北向"]})
    bad = json.dumps({"expressions": ["ts_mean(north_ratio, 999)"]})  # 超预算 → 被钳
    keep = json.dumps({"verdict": "keep", "reason": "ok"})

    def fn(messages):
        text = "\n".join(m["content"] for m in messages)
        calls.append(text)
        if "诊断信息" in text:
            return json.dumps({"expressions": []})
        if "翻译成" in text:
            return bad
        if "风控审计员" in text:
            return keep
        return hyp

    res = run_team_agent(daily, fn, n_rounds=1, seed=7, heal_rounds=0,
                         index_path=str(tmp_path / "e.jsonl"), eval_start=eval_start)

    assert not [t for t in calls if "诊断信息" in t], "钳制后不该再触发 refeed"
    assert res.rounds_log[0].get("n_window_clamped", 0) >= 1, "rounds_log 应记钳制次数"
    exprs_seen = {a.expression for a in res.state.attempts}
    assert any(e.startswith("ts_mean(north_ratio, ") and "999" not in e for e in exprs_seen), \
        f"应评估钳后表达式(窗口<999): {exprs_seen}"

# ==== 来自 test_holdout_coverage_guard.py ====
# ── A. library / acceptance 门 ──────────────────────────────────────────────


def test_holdout_coverage_guard_suite():
    """test_sparse_holdout_positive_train_is_coverage_not_sign_flip；修非对称漏洞：train<0 + holdout 无数据 也不得通过（round 8 假过关）。；test_sufficient_coverage_true_sign_flip_still_reported；test_holdout_exact_zero_with_enough_days_is_no_signal_not_flip；test_same_sign_nonzero_passes_library_when_strong；统一入口必须把 n_days 传进 library 门（非恒真：n_days 不足时 library 拒、不传则可能不同）。；test_strict_gate_also_blocks_insufficient_coverage"""
    # -- 原 test_sparse_holdout_positive_train_is_coverage_not_sign_flip --
    def _section_0_test_sparse_holdout_positive_train_is_coverage_not_sign_flip():
        from factorzen.discovery.guardrails import library_reasons

        reasons = library_reasons(
            ic_train=0.05, holdout_ic=0.0, holdout_n_days=0, holdout_min_days=60,
        )
        assert any("覆盖不足" in r for r in reasons), reasons
        assert not any("反号" in r for r in reasons), reasons
        assert reasons  # 必须拒绝

    _section_0_test_sparse_holdout_positive_train_is_coverage_not_sign_flip()

    # -- 原 test_sparse_holdout_negative_train_also_blocked --
    def _section_1_test_sparse_holdout_negative_train_also_blocked():
        from factorzen.discovery.guardrails import library_reasons

        reasons = library_reasons(
            ic_train=-0.05, holdout_ic=0.0, holdout_n_days=5, holdout_min_days=60,
        )
        assert any("覆盖不足" in r for r in reasons), reasons
        assert not any("反号" in r for r in reasons), reasons
        # 明确不得是空列表（不得通过）
        assert len(reasons) >= 1

    _section_1_test_sparse_holdout_negative_train_also_blocked()

    # -- 原 test_sufficient_coverage_true_sign_flip_still_reported --
    def _section_2_test_sufficient_coverage_true_sign_flip_still_reported():
        from factorzen.discovery.guardrails import library_reasons

        reasons = library_reasons(
            ic_train=0.05, holdout_ic=-0.03, holdout_n_days=100, holdout_min_days=60,
        )
        assert any("反号" in r for r in reasons), reasons
        assert not any("覆盖不足" in r for r in reasons), reasons

    _section_2_test_sufficient_coverage_true_sign_flip_still_reported()

    # -- 原 test_holdout_exact_zero_with_enough_days_is_no_signal_not_flip --
    def _section_3_test_holdout_exact_zero_with_enough_days_is_no_signal_not_flip():
        from factorzen.discovery.guardrails import library_reasons

        reasons = library_reasons(
            ic_train=0.05, holdout_ic=0.0, holdout_n_days=100, holdout_min_days=60,
        )
        assert any("无信号" in r for r in reasons), reasons
        assert not any("反号" in r for r in reasons), reasons

    _section_3_test_holdout_exact_zero_with_enough_days_is_no_signal_not_flip()

    # -- 原 test_same_sign_nonzero_passes_library_when_strong --
    def _section_4_test_same_sign_nonzero_passes_library_when_strong():
        from factorzen.discovery.guardrails import library_reasons

        assert library_reasons(
            ic_train=0.05, holdout_ic=0.04, holdout_n_days=100,
        ) == []
        assert library_reasons(
            ic_train=-0.05, holdout_ic=-0.04, holdout_n_days=100,
        ) == []

    _section_4_test_same_sign_nonzero_passes_library_when_strong()

    # -- 原 test_acceptance_reasons_forwards_holdout_n_days --
    def _section_5_test_acceptance_reasons_forwards_holdout_n_days():
        from factorzen.discovery.guardrails import acceptance_reasons, library_reasons

        with_days = acceptance_reasons(
            gate="library", ic_train=0.05, holdout_ic=0.0, holdout_n_days=3, holdout_min_days=60,
        )
        direct = library_reasons(
            ic_train=0.05, holdout_ic=0.0, holdout_n_days=3, holdout_min_days=60,
        )
        assert with_days == direct
        assert any("覆盖不足" in r for r in with_days)

    _section_5_test_acceptance_reasons_forwards_holdout_n_days()

    # -- 原 test_strict_gate_also_blocks_insufficient_coverage --
    def _section_6_test_strict_gate_also_blocks_insufficient_coverage():
        from factorzen.discovery.guardrails import guardrail_reasons

        reasons = guardrail_reasons(
            ic_train=0.05, holdout_ic=0.04, dsr_pvalue=0.01,
            holdout_n_days=10, holdout_min_days=60,
        )
        assert any("覆盖不足" in r for r in reasons), reasons

    _section_6_test_strict_gate_also_blocks_insufficient_coverage()


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
    root = Path(__file__).resolve().parents[2] / "src" / "factorzen"
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

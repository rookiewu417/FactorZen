"""agent 挖掘路径的预热口径：train 与 holdout 必须走同一条裁剪路径。"""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from factorzen.discovery.evaluation import _node_to_factor_df
from factorzen.discovery.expression import parse_expr


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


def test_node_to_factor_df_clips_both_bounds():
    daily = _synthetic_daily(n_days=100)
    node = parse_expr("ts_mean(close, 5)")
    lo, hi = dt.date(2020, 2, 1), dt.date(2020, 3, 1)

    out = _node_to_factor_df(node, daily, eval_start=lo, eval_end=hi)

    assert out["trade_date"].min() == lo
    assert out["trade_date"].max() == hi


def test_warmup_bars_counts_nonnull_history_per_leaf():
    """预热段有交易日、但该叶子全 null → 可用预热必须是 0，不能按交易日数报充足。

    复现 daily_basic 缺 2019 的真实情况：帧里有 60 个预热交易日，dv_ttm 全 null。

    同时验证 `LEAF_FEATURES` 映射真的被使用：合成帧显式给 `close_adj` 赋值，
    使其与 `close` 在预热段真正不同（`close_adj` cutoff 前全 null，`close` 全非空）。
    `_preprocess_daily` 只在 `close_adj` 缺失时才用 `close` 补齐（见其
    `if adj not in df.columns` 分支），显式提供的列会被保留，不会被覆盖。
    若实现漏查 `LEAF_FEATURES`（直接用叶子名 "close" 找列），会读到非空的 `close`
    列而不是全 null 的 `close_adj`，从而错误地报出满预热天数——这里必须是 0。

    `open`（未显式覆盖，`open_adj` 由 `_preprocess_daily` 等价复制自 `open`，预热段全非空）
    保留作为 min-vs-max-across-leaves 回归的判别项：与 dv_ttm（预热恒 0）组合，
    只有正确取 `min()` 才会得到 0；若误用 `max()`，会被 open 的满预热天数掩盖。
    """
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


def test_warmup_bars_absent_column_and_constant_expr():
    """两个当前只靠代码自证、无断言守护的边界情形。"""
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


def test_warmup_bars_excludes_nan_not_just_null():
    """NaN 预热单元格（非 null）不算可用历史——polars 里 NaN 不是 null。

    直接构造 NaN（不经 derived.py 的除法），使断言独立于 ret_1d 缺零分母守卫的行为。
    """
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


def _bundle_for(sample: pl.DataFrame):
    from factorzen.discovery.scoring import DataBundle
    return DataBundle.build(sample)


def test_warmup_bars_importable_from_discovery_and_respects_leaf_map():
    """warmup_bars 属 discovery 共享层（M1 与 agent 双路径共用），且用传入的 leaf_map
    映射叶子→列，不硬用 LEAF_FEATURES——crypto profile 的 leaf_map 才能正确判预热。"""
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


def test_train_ic_dates_exclude_warmup_segment():
    """train 段 IC 只能算在 [eval_start, eval_end] 内——预热段绝不进 IC 序列。

    ground-truth：train 段有效 IC 天数 <= 该区间的日历交易日数；
    若漏裁预热段，n_train 会超过这个上界。

    bundle 必须建在**含预热段的完整帧**上，与生产真实拓扑一致
    （`orchestrator.py`：`mining_df, holdout_df, _ = split_holdout(daily, ...)`；
    `bundle = DataBundle.build(mining_df)`——`mining_df` 不会被再裁到 eval_start，
    预热段仍在 `bundle.fwd_returns` 里）。若改用 `DataBundle.build(sample)`
    （sample 已按 eval_start 裁过），`bundle.fwd_returns` 本身就没有预热段日期，
    `quick_fitness` 里 `compute_rank_ic` 与之 join 时会隐式丢光预热行——不论
    `evaluate_expressions` 内部有没有裁剪，n_train 都被钉在同一个数，断言两边
    都过，零判别力（已用 bug-injection 探针验证，见 task-1.3-report.md）。
    """
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


def test_eval_end_without_eval_start_raises():
    """eval_end 单传（无 eval_start）会静默跳过下界裁剪与预热门——必须早失败，而不是悄悄放行。"""
    from factorzen.discovery.evaluation import evaluate_expressions

    full = _synthetic_daily(n_days=120)
    with pytest.raises(ValueError, match="eval_start"):
        evaluate_expressions(["ts_mean(close, 5)"], full, _bundle_for(full),
                             eval_start=None, eval_end=dt.date(2020, 3, 1))


# ── Task 1.4: orchestrator 先裁样本再 split（train/holdout/PBO 三处口径统一）─────


def test_split_happens_after_clipping_to_eval_start():
    """mining_df / holdout_df / bundle 必须建立在 [eval_start, end] 上，预热段只做求值前缀。"""
    from factorzen.agents.team_orchestrator import _prepare_segments

    full = _synthetic_daily(n_days=120)
    mining_df, holdout_df, holdout_start = _prepare_segments(
        full, eval_start="20200201", holdout_ratio=0.2)

    assert mining_df["trade_date"].min() == dt.date(2020, 2, 1)
    assert holdout_df["trade_date"].min() == holdout_start
    assert mining_df["trade_date"].max() < holdout_start


def test_m1_and_agent_paths_agree_on_train_ic_days():
    """跨路径一致性：M1 的 run_session(eval_start=) 与 agent 的 evaluate_expressions(eval_start=)
    对同一表达式、同一帧，train 段有效 IC 天数必须相等。"""
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


# ── 假拒绝修复:逐叶 path-lookback,浅派生叶不得拖垮深 raw 路 ──────────────────
#
# smoke 照出:门用『全叶最小 warmup』对比『最深路径 max lookback』——跨叶错配。
# `mul(ts_zscore(delta(div(close,pe_ttm),60),120), ts_sum(ret_1d,20))` 的深路
# (close/pe_ttm, need=180)明明有 ≥180 根预热,却被只需 20 的派生叶 ret_1d(少 1
# 根 warmup,=179)拖成 min=179 < 180 → 假拒。正确判定必须逐叶:每个叶子只需填满
# 它上方的窗口。下面用可控合成帧复刻该拓扑(deep need == close 可用预热,ret_1d 少 1)。


def test_leaf_lookbacks_are_per_leaf_not_global_max():
    """leaf_lookbacks:每个叶子沿『根→该叶』路径的窗口累加,各叶分别给出。

    与 required_lookback(只给最深路径全局最大)不同:ret_1d 只在 ts_sum(...,20) 下,
    need=20,不因 close 那条 180 深路被抬高。ground-truth 手算。
    """
    from factorzen.discovery.expression import leaf_lookbacks

    node = parse_expr("mul(ts_mean(close, 20), ts_sum(ret_1d, 5))")
    assert leaf_lookbacks(node) == {"close": 20, "ret_1d": 5}

    deep = parse_expr("mul(ts_zscore(delta(div(close, pe_ttm), 60), 120), ts_sum(ret_1d, 20))")
    assert leaf_lookbacks(deep) == {"close": 180, "pe_ttm": 180, "ret_1d": 20}

    # 同叶多次出现取最大:close 一处 need=25、一处 need=3 → 25
    both = parse_expr("add(ts_mean(delta(close, 5), 20), ts_mean(close, 3))")
    assert leaf_lookbacks(both)["close"] == 25


def test_warmup_shortfall_not_dragged_by_shallow_derived_leaf():
    """核心复现:深 raw 叶路(close, need=20, 恰好有 20 根预热)不该被浅派生叶
    (ret_1d, need=5 但只有 19 根 warmup)拖成假拒绝。

    前提用真实数值钉死『旧 min-vs-max 会假拒』的拓扑,再断言新逐叶门放行。
    """
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


def test_eval_start_none_still_evaluates_rolling_expressions(tmp_path):
    """session eval_start=None（旧调用者）时，滚动表达式必须照常求值，
    绝不能因空预热前缀被误判『预热不足』而拒绝——这是把无条件传 mining_df.min()
    误当 eval_start 会踩的回归（实测：ts_mean(close,20) 会变成 ic=None）。

    走真实 run_team_agent 端到端 orchestrator wiring（不传 eval_start，等价旧调用方
    `fz mine team` 升级前的行为），而不是手工拼装『已经正确』的 None 参数——后者绕过了
    `_run_one_round` 内部的 None-gating 分支逻辑本身，对该分支写错没有判别力。
    若实现把 `eval_start=mining_df["trade_date"].min()` 无条件传给
    `evaluate_expressions`（不管 session 级 eval_start 是否为 None），预热门会把
    `mining_df` 起点当成裁剪下界，warmup_bars 在其之前找不到任何交易日 → 0 根可用预热，
    `ts_mean(close, 20)` 需要 20 根历史会被判『预热不足』拒绝，ic_train 变 None——
    这里的断言就会失败。
    """
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

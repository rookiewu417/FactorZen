"""agent 挖掘路径的预热口径：train 与 holdout 必须走同一条裁剪路径。"""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from factorzen.agents.evaluation import _node_to_factor_df
from factorzen.discovery.expression import parse_expr


def _synthetic_daily(n_days: int = 120, n_codes: int = 40, start=dt.date(2020, 1, 1)) -> pl.DataFrame:
    """确定性合成帧：close 单调可预测，无随机性，便于 ground-truth 断言。"""
    dates = [start + dt.timedelta(days=i) for i in range(n_days)]
    rows = []
    for d_i, d in enumerate(dates):
        for c_i in range(n_codes):
            rows.append({
                "trade_date": d,
                "ts_code": f"{c_i:06d}.SZ",
                "close": 10.0 + d_i * 0.1 + c_i,
                "open": 10.0 + d_i * 0.1 + c_i,
                "high": 11.0 + d_i * 0.1 + c_i,
                "low": 9.0 + d_i * 0.1 + c_i,
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
    from factorzen.agents.evaluation import _preprocess_daily, warmup_bars

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
    from factorzen.agents.evaluation import _preprocess_daily, warmup_bars

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
    from factorzen.agents.evaluation import _preprocess_daily, warmup_bars

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
    from factorzen.agents.evaluation import evaluate_expressions
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
    from factorzen.agents.evaluation import evaluate_expressions

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
    from factorzen.agents.evaluation import evaluate_expressions

    full = _synthetic_daily(n_days=120)
    with pytest.raises(ValueError, match="eval_start"):
        evaluate_expressions(["ts_mean(close, 5)"], full, _bundle_for(full),
                             eval_start=None, eval_end=dt.date(2020, 3, 1))

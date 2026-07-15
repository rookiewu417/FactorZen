# tests/test_holdout_warmup.py
"""holdout 段的滚动算子必须扩窗预热——否则边界处发出的是**截断窗口的偏差值**。

`split_holdout` 是纯时间切片，holdout_df 前面没有 warmup 前缀。两条挖掘路径都直接在
holdout_df 上求值：

    M1     : `_factor_values(node, holdout_df, leaf_map=leaf_map)`        (eval_start=None)
    Agent  : `_node_to_factor_df(node, holdout_df)`

于是 `ts_mean(close, 20)` 在 holdout 前 ~20 天用**不足 20 天**的截断窗口计算——既不从 train
借前缀预热，也不丢弃这些天，而是把偏差值直接喂进 `holdout_ic`/CI，扭曲护栏验收。
这是**双路径一致地错**（不是漂移），修要两侧一起。

## 为什么用 mining 段预热是 PIT 合法的

mining 段整体早于 holdout 段。时序算子的滚动窗口只向**过去**看，holdout 首日用到的是
mining 末尾的数据——那是 ≤t 的信息。截面算子（`rank`/`zscore` 的 `.over("trade_date")`）
逐日独立，在完整帧上算与只在 holdout 帧上算逐值相同。求值后再裁剪到 `>= holdout_start`，
保证不泄漏任何 holdout 之后的信息。

**ground truth = 在完整帧上求值、再切出 holdout 段。** 本文件以此为判据。

顺带修好一个隐藏缺陷：`_preprocess_daily` 用 `close.shift(1).over("ts_code").fill_null(close)`
造 `pre_close`。只喂 holdout 帧时，holdout **首日的 pre_close 被填成它自己的 close**；
喂完整帧时它正确地取到 mining 末日的 close。
"""
from __future__ import annotations

import ast
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorzen.validation.holdout import split_holdout

_SRC = Path(__file__).resolve().parents[1] / "src" / "factorzen"


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


def test_agent_holdout_values_match_full_frame_ground_truth():
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


def test_m1_holdout_values_match_full_frame_ground_truth():
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


def test_m1_run_session_warms_up_holdout(tmp_path, monkeypatch):
    """集成：`run_session` 对 holdout 求值时必须传完整帧 + eval_start，而非已切片的 holdout_df。"""
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


# ── 架构守卫：默认值不许成为「静默不修」的藏身处 ──────────────────────────────


def test_every_production_caller_passes_warmup_daily():
    """`warmup_daily=None` 缺省会**回退到不预热的旧行为**。

    这是个陷阱：将来新增的调用方只要不传它，就静默地带着 holdout 边界偏差跑，而 CI 全绿。
    （本仓库的头号缺陷模式：修一处漏一处。）此处静态断言 src 下每个 `node_guardrails(...)`
    调用都显式传了 `warmup_daily`。

    保留缺省值而非改成必填，是因为单测常以「daily == holdout_df」的合成帧直接调用它；
    但生产代码没有这个借口。
    """
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


def test_single_agent_orchestrator_passes_the_full_frame_not_the_mining_slice(monkeypatch):
    """`warmup_daily` 必须是**完整帧**（mining + holdout），不是 mining 切片。

    ast 守卫只验参数**存在**。若传成 `mining_df`，holdout 求值会把它裁剪到
    `>= holdout_start` —— 结果为空 —— 候选**静默归零**，而没有任何东西抓得到。
    这是双路径登记簿点名的那类隐患，值得一个直接断言。
    """
    import factorzen.agents.orchestrator as orch
    from factorzen.agents.orchestrator import run_llm_agent

    seen: dict = {}
    monkeypatch.setattr(orch, "node_guardrails", _spy_guardrails(seen))
    run_llm_agent(_daily(), _fake_llm(), n_rounds=1, seed=1, heal_rounds=0)

    assert seen["warmup"] == seen["mining"] + seen["holdout"], (
        f"warmup_daily 应是完整帧（{seen['mining']}+{seen['holdout']} 行），"
        f"实得 {seen['warmup']} 行"
    )


def test_team_orchestrator_passes_the_full_frame_not_the_mining_slice(tmp_path, monkeypatch):
    import factorzen.agents.team_orchestrator as team
    from factorzen.agents.team_orchestrator import run_team_agent

    seen: dict = {}
    monkeypatch.setattr(team, "node_guardrails", _spy_guardrails(seen))
    run_team_agent(_daily(), _fake_llm(), n_rounds=1, seed=1,
                   index_path=str(tmp_path / "i.jsonl"), heal_rounds=0)

    assert seen["warmup"] == seen["mining"] + seen["holdout"], (
        f"warmup_daily 应是完整帧，实得 {seen['warmup']} 行"
    )


# ── 两条路径口径一致 ────────────────────────────────────────────────────────


def test_both_paths_produce_identical_holdout_values():
    """M1 与 Agent 在 holdout 段的因子值必须逐值相同——双路径登记簿。"""
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

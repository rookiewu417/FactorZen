# tests/test_agent_pipeline.py
import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.discovery.expression import parse_expr, to_expr_string
from factorzen.pipelines.factor_mine_agent import run_agent_mine
from factorzen.pipelines.factor_mine_team import run_team_mine


def _mock_daily(n_stocks=40, n_days=180, seed=1):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


def _scripted_llm():
    prop = json.dumps({"hypothesis": "动量", "expressions": ["ts_mean(close,5)"], "rationale": "r"})
    sem = json.dumps({"consistent": True, "reason": "ok"})
    crit = json.dumps({"verdict": "keep", "reason": "ok"})
    seq = [prop, sem, crit] * 50
    i = {"k": 0}
    def fn(messages):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v
    return fn


def _scripted_team(expr: str):
    """team 路径固定表达式 stub：Hypothesis→Coder→Critic(keep) 循环。"""
    hyp = json.dumps({"hypotheses": ["动量"]})
    code = json.dumps({"expressions": [expr]})
    crit = json.dumps({"verdict": "keep", "reason": "ok"})
    seq = [hyp, code, crit] * 50
    i = {"k": 0}
    def fn(messages):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v
    return fn


def _n_train_from_manifest(run_dir: str, expr: str) -> int:
    """从落盘 manifest 的 attempts 里取某表达式的 train 段有效 IC 天数。"""
    m = json.loads((Path(run_dir) / "manifest.json").read_text())
    norm = to_expr_string(parse_expr(expr))
    xs = [a["n_train"] for a in m["attempts"] if a["expression"] == norm and a["n_train"]]
    assert xs, f"{norm} 未被评估或 n_train 为空: {[a['expression'] for a in m['attempts']]}"
    return xs[0]


def test_run_agent_mine_writes_manifest(tmp_path: Path):
    daily = _mock_daily()
    res = run_agent_mine(daily, n_rounds=2, seed=42, out_dir=str(tmp_path),
                         llm_fn=_scripted_llm(), run_id="t1", export=False)
    run_dir = Path(res["run_dir"])
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "candidates.csv").exists()   # 兼容 fz mine leaderboard
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["n_trials"] >= 1
    assert res["n_trials"] == m["n_trials"]


def test_run_agent_mine_forwards_eval_start_clipping_warmup(tmp_path: Path):
    """接线漂移回归：pipeline `run_agent_mine` 必须把 eval_start 透传给 `run_llm_agent`。

    否则生产 `fz mine agent` 里 warmup-parity 修复完全失效——`daily` 由
    `prepare_mining_daily` 带 60 天预热前缀，不透传 eval_start 时整帧（含预热段）
    随 `split_holdout` 进 mining_df/DataBundle，预热噪声照旧灌进 train IC。

    判别力（纯行为，读落盘 manifest 的 attempts）：同一含预热前缀的完整帧，
    透传 eval_start 后 train 段裁到 eval_start，有效 IC 天数严格少于不透传
    （不透传时 train 段从帧首起、覆盖预热段）。签名断言无判别力，故从 pipeline
    最外层出发、以 n_train 的可观测差异为准。
    """
    daily = _mock_daily(n_days=180, seed=7)
    dates = sorted(set(daily["trade_date"].to_list()))
    eval_start = dates[40]                          # 前 40 个交易日作预热前缀
    expr = "ts_mean(close,5)"

    clip = run_agent_mine(daily, n_rounds=1, seed=42, out_dir=str(tmp_path / "clip"),
                          llm_fn=_scripted_llm(), run_id="clip", export=False,
                          eval_start=eval_start.strftime("%Y%m%d"))
    noclip = run_agent_mine(daily, n_rounds=1, seed=42, out_dir=str(tmp_path / "noclip"),
                            llm_fn=_scripted_llm(), run_id="noclip", export=False)

    n_clip = _n_train_from_manifest(clip["run_dir"], expr)
    n_noclip = _n_train_from_manifest(noclip["run_dir"], expr)
    assert n_clip < n_noclip, (
        f"eval_start 未透传/未裁预热段：clip n_train={n_clip} 应 < noclip n_train={n_noclip}")


def test_run_team_mine_forwards_eval_start_clipping_warmup(tmp_path: Path):
    """接线漂移回归：pipeline `run_team_mine` 必须把 eval_start 透传给 `run_team_agent`。

    与 agent 单路径同理：不透传时生产 `fz mine team` 的 warmup-parity 修复失效。
    判别力同上：透传后 train 段有效 IC 天数严格少于不透传。
    """
    daily = _mock_daily(n_days=180, seed=7)
    dates = sorted(set(daily["trade_date"].to_list()))
    eval_start = dates[40]
    expr = "ts_mean(close, 5)"

    clip = run_team_mine(daily, n_rounds=1, seed=42, index_path=str(tmp_path / "i1.jsonl"),
                         out_dir=str(tmp_path / "clip"), llm_fn=_scripted_team(expr),
                         run_id="clip", export=False, heal_rounds=0,
                         eval_start=eval_start.strftime("%Y%m%d"))
    noclip = run_team_mine(daily, n_rounds=1, seed=42, index_path=str(tmp_path / "i2.jsonl"),
                           out_dir=str(tmp_path / "noclip"), llm_fn=_scripted_team(expr),
                           run_id="noclip", export=False, heal_rounds=0)

    n_clip = _n_train_from_manifest(clip["run_dir"], expr)
    n_noclip = _n_train_from_manifest(noclip["run_dir"], expr)
    assert n_clip < n_noclip, (
        f"eval_start 未透传/未裁预热段：clip n_train={n_clip} 应 < noclip n_train={n_noclip}")

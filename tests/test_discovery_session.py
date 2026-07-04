from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl


def _daily(seed=3, n_stocks=40, n_days=120):
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for s in [f"{i:06d}.SH" for i in range(n_stocks)]:
        p = 10.0
        for day in days:
            p = float(max(p * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": day, "ts_code": s, "close": p, "close_adj": p,
                         "open_adj": p, "high_adj": p, "low_adj": p, "open": p, "high": p, "low": p,
                         "amount": 1e7, "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows)


def test_factor_values_eval_start_trims():
    from factorzen.discovery.expression import parse_expr
    from factorzen.discovery.mining_session import _factor_values
    daily = _daily()
    dates = sorted(daily["trade_date"].unique().to_list())
    cutoff = dates[len(dates) // 2]
    es = cutoff.strftime("%Y%m%d")
    out = _factor_values(parse_expr("close"), daily, eval_start=es)
    assert out["trade_date"].min() >= cutoff


def test_session_runs_and_writes_artifacts(tmp_path: Path):
    from factorzen.discovery.mining_session import run_session
    res = run_session(_daily(), n_trials=20, top_k=5, seed=42,
                      method="random", out_dir=str(tmp_path))
    session_dir = Path(res["session_dir"])
    assert (session_dir / "candidates.csv").exists()
    assert (session_dir / "manifest.json").exists()
    assert 0 < len(res["candidates"]) <= 5
    manifest = json.loads((session_dir / "manifest.json").read_text())
    assert manifest["cli_n_trials"] == 20
    assert manifest["seed"] == 42
    for c in res["candidates"]:
        assert c["max_corr"] < 0.7  # 贪心去相关保证：top-K 互不近重复，max_corr 是真实测量


def test_session_reproducible_same_seed(tmp_path: Path):
    from factorzen.discovery.mining_session import run_session
    a = run_session(_daily(), n_trials=20, top_k=5, seed=7, out_dir=str(tmp_path / "a"))
    b = run_session(_daily(), n_trials=20, top_k=5, seed=7, out_dir=str(tmp_path / "b"))
    expr_a = [c["expression"] for c in a["candidates"]]
    expr_b = [c["expression"] for c in b["candidates"]]
    assert expr_a == expr_b


def test_session_has_guard_metrics_and_holdout_isolated(tmp_path):
    from factorzen.discovery.mining_session import run_session
    res = run_session(_daily(n_stocks=40, n_days=150), n_trials=30, top_k=5, seed=42,
                      method="random", holdout_ratio=0.2, out_dir=str(tmp_path))
    assert 0 < len(res["candidates"]) <= 5
    for c in res["candidates"]:
        # 护栏指标齐全
        for key in ("n_trials", "pbo", "holdout_ic", "dsr_pvalue", "ic_ci_low"):
            assert key in c
        assert c["n_trials"] > 0          # 真实评估数（非 CLI n_trials 摆设）
        assert 0.0 <= c["pbo"] <= 1.0 or c["pbo"] != c["pbo"]  # [0,1] 或 nan
    # holdout 永久隔离：挖掘期数据严格早于 holdout（删除 daily=mining_df 会让此断言失败）
    assert res["mining_end"] < res["holdout_start"]


def test_deflated_sharpe_train_n_vs_mining_window_n_flips_significance():
    """数值对照：DSR 显著性检验必须用候选自己的 train 段样本数(n_train)，不能用
    mining 全段交易日数(n_obs_mining)——后者系统性偏大（约 1.43x：500/350），且放大
    方向是让候选看起来比实际更显著（危险方向）。固定 sharpe/n_trials/sharpe_variance，
    分别用「正确的 n_train=350」和「错误的 n_obs_mining=500」算 DSR，断言两者的
    显著性结论（p<0.05 与否）相反。"""
    from factorzen.validation.deflated_sharpe import deflated_sharpe
    sharpe, n_trials, sharpe_var = 0.14, 30, 0.001
    _dsr_correct, p_correct = deflated_sharpe(sharpe, n_trials, 350, sharpe_variance=sharpe_var)
    _dsr_wrong, p_wrong = deflated_sharpe(sharpe, n_trials, 500, sharpe_variance=sharpe_var)
    assert p_correct > 0.05  # 正确：用 train 段真实样本数 → 不显著
    assert p_wrong < 0.05  # 错误：用放大的 mining 全段样本数 → 假显著（危险方向）


def test_session_dsr_uses_candidate_own_train_n(tmp_path, monkeypatch):
    """集成测试：run_session 内对每个候选调用 deflated_sharpe() 时，传入的样本数
    必须是该候选自己在 train 段的真实样本数(c["n_train"])，而不是退化为所有候选共用
    的全局 mining 段交易日数。用 monkeypatch 拦截 deflated_sharpe 的调用参数核对。"""
    import factorzen.discovery.mining_session as ms_mod
    from factorzen.validation.holdout import split_holdout

    daily = _daily(n_stocks=40, n_days=150)
    # 独立重算旧 bug 会传入的「mining 段全局交易日数」，不依赖 mining_session 内部实现
    sorted_daily = daily.sort(["ts_code", "trade_date"])
    mining_df, _holdout_df, _holdout_start = split_holdout(sorted_daily, holdout_ratio=0.2)
    legacy_n_obs_mining = mining_df["trade_date"].n_unique()

    calls: list[int] = []
    real_dsr = ms_mod.deflated_sharpe

    def _spy_deflated_sharpe(sharpe, n_trials, n_obs, **kwargs):
        calls.append(n_obs)
        return real_dsr(sharpe, n_trials, n_obs, **kwargs)

    monkeypatch.setattr(ms_mod, "deflated_sharpe", _spy_deflated_sharpe)

    res = ms_mod.run_session(daily, n_trials=30, top_k=5, seed=42,
                             method="random", holdout_ratio=0.2, out_dir=str(tmp_path))

    assert calls, "deflated_sharpe 应至少被调用一次"
    assert len(calls) == len(res["candidates"])
    for n_obs_used, c in zip(calls, res["candidates"], strict=True):
        assert "n_train" in c
        assert n_obs_used == c["n_train"]          # 用的是候选自己的 train 段样本数
        assert n_obs_used < legacy_n_obs_mining     # 不是放大过的 mining 全段样本数（旧 bug）

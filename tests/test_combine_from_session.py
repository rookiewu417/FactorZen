"""mine→combine 端到端接线:因子库 session → 物化 → 四方法 OOS 组合。"""
import datetime as dt

import numpy as np
import polars as pl
import pytest

from factorzen.pipelines import factor_combine


def _daily(n_stocks=40, n_days=200, seed=1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2023, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for i in range(n_stocks):
        c, px = f"{i:06d}.SZ", 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "close_adj": px,
                         "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


def _session(tmp_path, exprs):
    sess = tmp_path / "sess"
    sess.mkdir()
    pl.DataFrame({"rank": list(range(1, len(exprs) + 1)),
                  "expression": exprs, "passed": [True] * len(exprs)}
                 ).write_csv(sess / "candidates.csv")
    return str(sess)


def test_combine_from_session_end_to_end(tmp_path, monkeypatch):
    """因子库(≥2 因子)→ 物化 + 收益面板 + 四方法 OOS 对比,返回 comparison。"""
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily",
                        lambda *a, **k: _daily())
    session = _session(tmp_path, ["rank(close)", "ts_mean(vol,5)", "neg(rank(ts_std(close,10)))"])
    res = factor_combine.combine_from_session(
        session_dir=session, start="20230103", end="20231231", universe=None,
        horizon=5, train_days=60, test_days=15, out_dir=str(tmp_path / "out"))
    comp = res["comparison"]
    methods = set(comp["method"].to_list())
    assert {"equal_weight", "ic_weighted", "max_ir"} <= methods   # 至少线性三法都跑了
    assert comp.height >= 3


def test_combine_from_session_needs_two_factors(tmp_path, monkeypatch):
    """因子库不足 2 个 → 明确报错(组合至少需两个,不静默产垃圾)。"""
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", lambda *a, **k: _daily())
    session = _session(tmp_path, ["rank(close)"])
    with pytest.raises(ValueError, match="不足 2 个"):
        factor_combine.combine_from_session(
            session_dir=session, start="20230103", end="20231231", out_dir=str(tmp_path / "o"))


def test_combine_from_session_passed_only_filters(tmp_path, monkeypatch):
    """默认只取 passed=True 的库因子;过滤后不足 2 个则报错。"""
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", lambda *a, **k: _daily())
    sess = tmp_path / "s2"
    sess.mkdir()
    pl.DataFrame({"rank": [1, 2, 3], "expression": ["rank(close)", "rank(vol)", "rank(high)"],
                  "passed": [True, False, False]}).write_csv(sess / "candidates.csv")
    with pytest.raises(ValueError, match="不足 2 个"):
        factor_combine.combine_from_session(
            session_dir=str(sess), start="20230103", end="20231231", out_dir=str(tmp_path / "o"))


# ── 任务 C：多 session 合并去重 + 贪心去相关 ──────────────────────────────────
def _session_with_ic(tmp_path, name, rows):
    """rows: list[(expression, holdout_ic)] → 写含 holdout_ic 列的 candidates.csv。"""
    sess = tmp_path / name
    sess.mkdir()
    pl.DataFrame({"rank": list(range(1, len(rows) + 1)),
                  "expression": [e for e, _ in rows],
                  "holdout_ic": [ic for _, ic in rows],
                  "passed": [True] * len(rows)}).write_csv(sess / "candidates.csv")
    return str(sess)


def test_combine_merges_and_dedups_across_sessions(tmp_path, monkeypatch):
    """两个 session 各含同一表达式（规范形相同）→ 合并后只出现一次。"""
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", lambda *a, **k: _daily())
    s1 = _session_with_ic(tmp_path, "s1", [("rank(close)", 0.05), ("ts_mean(vol,5)", 0.04)])
    # rank( close ) 空格差异 → parse_expr 规范化后与 rank(close) 相同
    s2 = _session_with_ic(tmp_path, "s2", [("rank( close )", 0.05),
                                           ("neg(rank(ts_std(close,10)))", 0.03)])
    res = factor_combine.combine_from_session(
        session_dirs=[s1, s2], start="20230103", end="20231231", horizon=5,
        train_days=60, test_days=15, decorr_threshold=1.0, out_dir=str(tmp_path / "o"))
    used = res["factors_used"]
    assert used.count("rank(close)") == 1, f"规范形重复未去重: {used}"
    assert "ts_mean(vol, 5)" in used and "neg(rank(ts_std(close, 10)))" in used


def test_combine_decorr_drops_near_duplicate(tmp_path, monkeypatch):
    """构造高相关对（ts_mean(close,20) 与 ts_mean(close,21)）→ 仅 |holdout_ic| 高者存活，
    被剔者记入 dropped_correlated。"""
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", lambda *a, **k: _daily())
    sess = _session_with_ic(tmp_path, "s", [
        ("ts_mean(close,20)", 0.08),   # |ic| 高 → 存活
        ("ts_mean(close,21)", 0.03),   # 与上近亲 → 被剔
        ("rank(neg(vol))", 0.05),      # 独立 → 存活
    ])
    res = factor_combine.combine_from_session(
        session_dirs=[sess], start="20230103", end="20231231", horizon=5,
        train_days=60, test_days=15, decorr_threshold=0.7, out_dir=str(tmp_path / "o"))
    dropped = [d["identity"] for d in res["dropped_correlated"]]
    assert "ts_mean(close, 21)" in dropped, f"高相关近亲未被剔: {res['dropped_correlated']}"
    assert "ts_mean(close, 20)" not in dropped, "|holdout_ic| 高者不应被剔"
    used = res["factors_used"]
    assert "ts_mean(close, 20)" in used and "ts_mean(close, 21)" not in used


def test_combine_decorr_threshold_one_keeps_all(tmp_path, monkeypatch):
    """decorr_threshold=1.0 → 逃生口，无剔除。"""
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", lambda *a, **k: _daily())
    sess = _session_with_ic(tmp_path, "s", [("ts_mean(close,20)", 0.08), ("ts_mean(close,21)", 0.03)])
    res = factor_combine.combine_from_session(
        session_dirs=[sess], start="20230103", end="20231231", horizon=5,
        train_days=60, test_days=15, decorr_threshold=1.0, out_dir=str(tmp_path / "o"))
    assert res["dropped_correlated"] == []
    assert len(res["factors_used"]) == 2


def test_combine_decorr_below_two_errors(tmp_path, monkeypatch):
    """去相关后 < 2 个因子 → 报错（组合至少需两个）。"""
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", lambda *a, **k: _daily())
    sess = _session_with_ic(tmp_path, "s", [("ts_mean(close,20)", 0.08), ("ts_mean(close,21)", 0.03)])
    with pytest.raises(ValueError, match="不足 2 个"):
        factor_combine.combine_from_session(
            session_dirs=[sess], start="20230103", end="20231231", horizon=5,
            train_days=60, test_days=15, decorr_threshold=0.7, out_dir=str(tmp_path / "o"))

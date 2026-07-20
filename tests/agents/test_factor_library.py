"""因子库登记系统（分市场·全信息·自动维护）单测。TDD、mock 离线。"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

# ── 构造工具 ────────────────────────────────────────────────────────────────

def _cand(expr, *, ic_train=0.05, holdout_ic=0.04, dsr_pvalue=0.2, ir_train=0.4,
          n_train=200, **extra):
    """一个「候选 dict」，字段口径与 mining_session/agents 候选一致。"""
    d = {"expression": expr, "ic_train": ic_train, "holdout_ic": holdout_ic,
         "dsr_pvalue": dsr_pvalue, "ir_train": ir_train, "n_train": n_train}
    d.update(extra)
    return d


def _panel(vals_per_stock, n_days=6, n_stocks=None):
    """[trade_date, ts_code, factor_value]：每股取 vals_per_stock[i]，每日相同。"""
    n_stocks = n_stocks if n_stocks is not None else len(vals_per_stock)
    rows = []
    for d in range(n_days):
        dt = date(2024, 1, 2) + timedelta(days=d)
        for i in range(n_stocks):
            rows.append({"trade_date": dt, "ts_code": f"{i:06d}.SH",
                         "factor_value": float(vals_per_stock[i])})
    return pl.DataFrame(rows)


# ── schema round-trip ────────────────────────────────────────────────────────

def test_factor_record_roundtrip():
    from factorzen.discovery.factor_library import FactorRecord
    r = FactorRecord(expression="rank(close)", market="ashare", ic_train=0.05,
                     holdout_ic=0.04, status="active", added_at="2026-07-12",
                     updated_at="2026-07-12", n_train=100)
    d = r.to_dict()
    assert d["expression"] == "rank(close)"
    r2 = FactorRecord.from_dict(d)
    assert r2 == r
    # 未知/缺失字段容忍（向前兼容）
    r3 = FactorRecord.from_dict({"expression": "x", "market": "crypto", "extra_unknown": 1})
    assert r3.expression == "x" and r3.market == "crypto" and r3.status == "active"


# ── upsert：新增/更新时间戳 + gate ───────────────────────────────────────────

def test_upsert_gate_suite(tmp_path):
    """test_upsert_new_sets_added_at；test_upsert_duplicate_updates_and_preserves_added_at；规范形去重：'add(close, 1)' 与 'add(close, 1.0)'（整型/浮点字面量 + 空白）归一化同 → 只一条。；不过 library gate 的被跳过：holdout 反号 / |IC| 太弱。；holdout 覆盖稀薄（如北向季末残留仅 5 天）的候选不得入库。；覆盖充足（n_holdout_days ≥ 阈值）且真+有信号 → 正常入库（别把守卫修成一刀切）。；旧调用方候选不带 n_holdout_days → 跳过覆盖门（零回归；与 guardrails None 语义一致）。；门槛复用：upsert 的入库判定与 acceptance_reasons(gate='library') 一致（非恒真："""
    # -- 原 test_upsert_new_sets_added_at --
    def _section_0_test_upsert_new_sets_added_at(tmp_path):
        from factorzen.discovery.factor_library import load_library, upsert
        res = upsert("ashare", [_cand("rank(close)")], eval_window=("20200101", "20260101"),
                     universe="csi300", horizon=1, run_id="r1", session_dir="s1",
                     git_sha="abc", now="2026-07-12", root=str(tmp_path))
        assert res.added == 1 and res.updated == 0
        lib = load_library("ashare", root=str(tmp_path))
        assert len(lib) == 1
        assert lib[0].added_at == "2026-07-12" and lib[0].updated_at == "2026-07-12"
        assert lib[0].eval_start == "20200101" and lib[0].eval_end == "20260101"
        assert lib[0].universe == "csi300" and lib[0].source_run_id == "r1"

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_upsert_new_sets_added_at(_tp0)

    # -- 原 test_upsert_duplicate_updates_and_preserves_added_at --
    def _section_1_test_upsert_duplicate_updates_and_preserves_added_at(tmp_path):
        from factorzen.discovery.factor_library import load_library, upsert
        upsert("ashare", [_cand("rank(close)", ic_train=0.05)],
               eval_window=("20200101", "20260101"), universe="u", horizon=1,
               run_id="r1", session_dir="s1", git_sha="a", now="2026-07-01", root=str(tmp_path))
        res2 = upsert("ashare", [_cand("rank(close)", ic_train=0.08)],
                      eval_window=("20200101", "20260101"), universe="u", horizon=1,
                      run_id="r2", session_dir="s2", git_sha="b", now="2026-07-12", root=str(tmp_path))
        assert res2.added == 0 and res2.updated == 1
        lib = load_library("ashare", root=str(tmp_path))
        assert len(lib) == 1
        assert lib[0].added_at == "2026-07-01"          # 保留原入库日
        assert lib[0].updated_at == "2026-07-12"        # 刷新更新日
        assert abs(lib[0].ic_train - 0.08) < 1e-9       # 指标已更新

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_upsert_duplicate_updates_and_preserves_added_at(_tp1)

    # -- 原 test_upsert_normalizes_expression_as_dedup_key --
    def _section_2_test_upsert_normalizes_expression_as_dedup_key(tmp_path):
        from factorzen.discovery.factor_library import load_library, upsert
        upsert("ashare", [_cand("add(close, 1)")], eval_window=("20200101", "20260101"),
               universe="u", horizon=1, run_id="r1", session_dir="s1", git_sha="a",
               now="2026-07-01", root=str(tmp_path))
        upsert("ashare", [_cand("add( close , 1.0 )")], eval_window=("20200101", "20260101"),
               universe="u", horizon=1, run_id="r2", session_dir="s2", git_sha="b",
               now="2026-07-12", root=str(tmp_path))
        lib = load_library("ashare", root=str(tmp_path))
        assert len(lib) == 1                             # 同一因子只登记一次
        assert lib[0].expression == "add(close, 1.0)"    # 存的是规范形

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_upsert_normalizes_expression_as_dedup_key(_tp2)

    # -- 原 test_upsert_skips_failing_library_gate --
    def _section_3_test_upsert_skips_failing_library_gate(tmp_path):
        from factorzen.discovery.factor_library import load_library, upsert
        cands = [
            _cand("rank(close)", ic_train=0.05, holdout_ic=0.04),      # 真+有信号 → 入库
            _cand("rank(open)", ic_train=0.05, holdout_ic=-0.04),      # holdout 反号 → 跳过
            _cand("rank(high)", ic_train=0.006, holdout_ic=0.005),     # |IC| 太弱 → 跳过
        ]
        res = upsert("ashare", cands, eval_window=("20200101", "20260101"), universe="u",
                     horizon=1, run_id="r1", session_dir="s1", git_sha="a",
                     now="2026-07-12", root=str(tmp_path))
        assert res.added == 1 and res.skipped == 2
        lib = load_library("ashare", root=str(tmp_path))
        assert [r.expression for r in lib] == ["rank(close)"]

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_upsert_skips_failing_library_gate(_tp3)

    # -- 原 test_upsert_blocks_thin_holdout_coverage --
    def _section_4_test_upsert_blocks_thin_holdout_coverage(tmp_path):
        from factorzen.discovery.factor_library import load_library, upsert
        res = upsert("ashare", [_cand("div(vol, amount)", n_holdout_days=5)],
                     eval_window=("20200101", "20260101"), universe="csi300", horizon=1,
                     run_id="r1", session_dir="s1", git_sha="a", now="2026-07-13",
                     root=str(tmp_path))
        assert res.skipped == 1 and res.added == 0
        assert load_library("ashare", root=str(tmp_path)) == []

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_upsert_blocks_thin_holdout_coverage(_tp4)

    # -- 原 test_upsert_admits_adequate_holdout_coverage --
    def _section_5_test_upsert_admits_adequate_holdout_coverage(tmp_path):
        from factorzen.discovery.factor_library import load_library, upsert
        res = upsert("ashare", [_cand("rank(close)", n_holdout_days=291)],
                     eval_window=("20200101", "20260101"), universe="csi300", horizon=1,
                     run_id="r1", session_dir="s1", git_sha="a", now="2026-07-13",
                     root=str(tmp_path))
        assert res.added == 1
        assert [r.expression for r in load_library("ashare", root=str(tmp_path))] == ["rank(close)"]

    _tp5 = tmp_path / "_s5"
    _tp5.mkdir(exist_ok=True)
    _section_5_test_upsert_admits_adequate_holdout_coverage(_tp5)

    # -- 原 test_upsert_without_n_holdout_days_keeps_old_behavior --
    def _section_6_test_upsert_without_n_holdout_days_keeps_old_behavior(tmp_path):
        from factorzen.discovery.factor_library import upsert
        res = upsert("ashare", [_cand("rank(close)")],
                     eval_window=("20200101", "20260101"), universe="csi300", horizon=1,
                     run_id="r1", session_dir="s1", git_sha="a", now="2026-07-13",
                     root=str(tmp_path))
        assert res.added == 1

    _tp6 = tmp_path / "_s6"
    _tp6.mkdir(exist_ok=True)
    _section_6_test_upsert_without_n_holdout_days_keeps_old_behavior(_tp6)

    # -- 原 test_upsert_gate_parity_with_acceptance_reasons --
    def _section_7_test_upsert_gate_parity_with_acceptance_reasons(tmp_path):
        from factorzen.discovery.factor_library import upsert
        from factorzen.discovery.guardrails import acceptance_reasons
        cases = [
            _cand("f0", ic_train=0.05, holdout_ic=0.04),
            _cand("f1", ic_train=-0.05, holdout_ic=-0.04),      # 同号反转 → 入
            _cand("f2", ic_train=0.05, holdout_ic=-0.04),       # 反号 → 不入
            _cand("f3", ic_train=0.006, holdout_ic=0.006),      # 太弱 → 不入
            _cand("f4", ic_train=float("nan"), holdout_ic=0.04),  # NaN → 不入
        ]
        res = upsert("crypto", cases, eval_window=("20210101", "20260101"), universe="perp",
                     horizon=1, run_id="r", session_dir="s", git_sha="a", now="2026-07-12",
                     root=str(tmp_path))
        accepted = {r.expression for r in res.records}
        for c in cases:
            should = not acceptance_reasons(gate="library", ic_train=c["ic_train"],
                                            holdout_ic=c["holdout_ic"], dsr_pvalue=c["dsr_pvalue"])
            assert (c["expression"] in accepted) == should, c["expression"]

    _tp7 = tmp_path / "_s7"
    _tp7.mkdir(exist_ok=True)
    _section_7_test_upsert_gate_parity_with_acceptance_reasons(_tp7)


# ── 去相关（方案 A：仍收录但打标记）──────────────────────────────────────────

def test_decorrelation_upsert_suite(tmp_path):
    """test_decorrelation_marks_correlated_but_keeps；test_decorrelation_threshold_one_disables；test_decorrelation_no_materialize_all_active"""
    # -- 原 test_decorrelation_marks_correlated_but_keeps --
    def _section_0_test_decorrelation_marks_correlated_but_keeps(tmp_path):
        from factorzen.discovery.factor_library import load_library, upsert
        base = [((i * 37) % 40) + 0.5 for i in range(40)]
        panels = {
            "rank(close)": _panel(base),
            "rank(open)": _panel([x * 3.0 + 7.0 for x in base]),   # 单调变换 → 高相关
            "rank(high)": _panel([((i * 11) % 40) + 0.5 for i in range(40)]),  # 不同序 → 低相关
        }

        def materialize(expr):
            return panels.get(expr)

        cands = [_cand("rank(close)"), _cand("rank(open)"), _cand("rank(high)")]
        res = upsert("ashare", cands, eval_window=("20200101", "20260101"), universe="u",
                     horizon=1, run_id="r", session_dir="s", git_sha="a", now="2026-07-12",
                     materialize=materialize, decorr_threshold=0.7, root=str(tmp_path))
        lib = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}
        assert len(lib) == 3                                       # 方案 A：全部收录
        assert lib["rank(close)"].status == "active"               # 首个 → active
        assert lib["rank(open)"].status == "correlated"            # 与 close 高相关 → 标记
        assert lib["rank(open)"].correlated_with == "rank(close)"
        assert lib["rank(open)"].max_corr_in_lib > 0.7
        assert lib["rank(high)"].status == "active"                # 低相关 → active
        assert res.correlated == 1

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_decorrelation_marks_correlated_but_keeps(_tp0)

    # -- 原 test_decorrelation_threshold_one_disables --
    def _section_1_test_decorrelation_threshold_one_disables(tmp_path):
        from factorzen.discovery.factor_library import load_library, upsert
        base = [((i * 37) % 40) + 0.5 for i in range(40)]
        panels = {"a": _panel(base), "b": _panel([x * 3.0 + 7.0 for x in base])}
        cands = [_cand("a"), _cand("b")]
        upsert("ashare", cands, eval_window=("20200101", "20260101"), universe="u", horizon=1,
               run_id="r", session_dir="s", git_sha="a", now="2026-07-12",
               materialize=lambda e: panels.get(e), decorr_threshold=1.0, root=str(tmp_path))
        lib = load_library("ashare", root=str(tmp_path))
        assert all(r.status == "active" for r in lib)              # 阈值 1.0 关闭去相关

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_decorrelation_threshold_one_disables(_tp1)

    # -- 原 test_decorrelation_no_materialize_all_active --
    def _section_2_test_decorrelation_no_materialize_all_active(tmp_path):
        from factorzen.discovery.factor_library import load_library, upsert
        upsert("ashare", [_cand("a"), _cand("b")], eval_window=("20200101", "20260101"),
               universe="u", horizon=1, run_id="r", session_dir="s", git_sha="a",
               now="2026-07-12", root=str(tmp_path))    # materialize=None
        lib = load_library("ashare", root=str(tmp_path))
        assert all(r.status == "active" for r in lib)

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_decorrelation_no_materialize_all_active(_tp2)


# ── default_window ───────────────────────────────────────────────────────────

def test_default_window_suite(monkeypatch):
    """test_default_window_end_is_latest_start_back_years；test_default_window_today_caps_end；test_default_window_crypto_floor；test_default_window_raises_when_cache_missing"""
    # -- 原 test_default_window_end_is_latest_start_back_years --
    def _section_0_test_default_window_end_is_latest_start_back_years(mp):
        import factorzen.discovery.backtest_window as bw
        mp.setattr(bw, "latest_data_date", lambda m: date(2026, 6, 30))
        start, end = bw.default_window("ashare", years=6)
        assert end == "20260630"
        assert start == "20200630"

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_default_window_end_is_latest_start_back_years(mp)

    # -- 原 test_default_window_today_caps_end --
    def _section_1_test_default_window_today_caps_end(mp):
        import factorzen.discovery.backtest_window as bw
        mp.setattr(bw, "latest_data_date", lambda m: date(2026, 6, 30))
        start, end = bw.default_window("ashare", years=6, today=date(2025, 1, 15))
        assert end == "20250115"                          # today 更早 → 封顶
        assert start == "20190115"

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_default_window_today_caps_end(mp)

    # -- 原 test_default_window_crypto_floor --
    def _section_2_test_default_window_crypto_floor(mp):
        import factorzen.discovery.backtest_window as bw
        mp.setattr(bw, "latest_data_date", lambda m: date(2026, 6, 30))
        start, end = bw.default_window("crypto", years=6)
        assert start == "20210101"                         # crypto 起点下限 20210101
        assert end == "20260630"

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_default_window_crypto_floor(mp)

    # -- 原 test_default_window_raises_when_cache_missing --
    def _section_3_test_default_window_raises_when_cache_missing(mp):
        import factorzen.discovery.backtest_window as bw
        mp.setattr(bw, "latest_data_date", lambda m: None)
        with pytest.raises(ValueError):
            bw.default_window("ashare")

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_default_window_raises_when_cache_missing(mp)


def test_latest_data_date_scans_partitions(tmp_path, monkeypatch):
    """真实探测：Hive 分区 parquet 取最新分区的 trade_date 最大值；crypto 读 manifest.json。"""
    import json as _json

    import factorzen.discovery.backtest_window as bw

    # A股风格分区：两个月，最新分区 2026-06 内含到 6/30
    root = tmp_path / "daily"
    for ym, mx in [("2026/month=05", date(2026, 5, 30)), ("2026/month=06", date(2026, 6, 30))]:
        d = root / f"year={ym}"
        d.mkdir(parents=True)
        pl.DataFrame({"trade_date": [date(2026, 6, 1), mx], "ts_code": ["a", "b"]}).write_parquet(
            d / "data.parquet")
    monkeypatch.setattr(bw, "_ASHARE_DAILY_ROOT", str(root))
    assert bw.latest_data_date("ashare") == date(2026, 6, 30)

    # crypto manifest
    lake = tmp_path / "lake"
    lake.mkdir()
    (lake / "manifest.json").write_text(_json.dumps({"start": "20210101", "end": "20260415"}))
    monkeypatch.setattr(bw, "_CRYPTO_LAKE_ROOT", str(lake))
    assert bw.latest_data_date("crypto") == date(2026, 4, 15)


# ── render markdown ──────────────────────────────────────────────────────────

def test_render_markdown_suite(tmp_path):
    """test_render_markdown_has_stats_and_table；test_render_markdown_empty_library_no_crash"""
    # -- 原 test_render_markdown_has_stats_and_table --
    def _section_0_test_render_markdown_has_stats_and_table(tmp_path):
        from factorzen.discovery.factor_library import render_markdown, upsert
        base = [((i * 37) % 40) + 0.5 for i in range(40)]
        panels = {"rank(close)": _panel(base), "rank(open)": _panel([x * 3 + 7 for x in base])}
        upsert("ashare", [_cand("rank(close)", holdout_ic=0.06), _cand("rank(open)", holdout_ic=0.03)],
               eval_window=("20200101", "20260101"), universe="csi300", horizon=1, run_id="r",
               session_dir="s", git_sha="a", now="2026-07-12",
               materialize=lambda e: panels.get(e), root=str(tmp_path))
        md = render_markdown("ashare", root=str(tmp_path))
        assert (Path(tmp_path) / "ashare.md").exists()
        assert "active" in md and "correlated" in md
        assert "rank(close)" in md and "rank(open)" in md
        # 表格按 holdout_ic 降序：close(0.06) 行在 open(0.03) 行之前
        assert md.index("rank(close)") < md.index("rank(open)")
        # summary.md 跨市场总览刷新
        assert (Path(tmp_path) / "summary.md").exists()

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_render_markdown_has_stats_and_table(_tp0)

    # -- 原 test_render_markdown_empty_library_no_crash --
    def _section_1_test_render_markdown_empty_library_no_crash(tmp_path):
        from factorzen.discovery.factor_library import render_markdown
        md = render_markdown("futures", root=str(tmp_path))
        assert isinstance(md, str)
        assert (Path(tmp_path) / "futures.md").exists()

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_render_markdown_empty_library_no_crash(_tp1)


# ── rebuild（mock 评估）──────────────────────────────────────────────────────

def test_rebuild_evaluates_in_window_and_upserts(tmp_path):
    """rebuild：mock 候选源 + mock 评估 → 在给定窗口重算并 upsert 合格者。"""
    from factorzen.discovery.factor_library import load_library, rebuild
    sources = ["rank(close)", "rank(open)", "close + 1"]
    seen_window = {}

    def evaluate(exprs):
        seen_window["exprs"] = list(exprs)
        # 返回候选 dict：close 过 gate、open 反号不过
        return [
            _cand("rank(close)", ic_train=0.05, holdout_ic=0.04),
            _cand("rank(open)", ic_train=0.05, holdout_ic=-0.04),
            _cand("close + 1", ic_train=0.03, holdout_ic=0.02),
        ]

    res = rebuild("ashare", sources=sources, eval_window=("20200101", "20260101"),
                  universe="csi300", horizon=1, evaluate=evaluate, git_sha="abc",
                  now="2026-07-12", root=str(tmp_path))
    assert res.added == 2 and res.skipped == 1        # open 反号被跳过
    lib = {r.expression for r in load_library("ashare", root=str(tmp_path))}
    assert "rank(close)" in lib and "rank(open)" not in lib
    # 评估拿到去重后的唯一表达式集
    assert len(seen_window["exprs"]) == 3
    # rebuild manifest 落盘（窗口/源/git_sha 可复现）
    assert (Path(tmp_path) / "rebuild_ashare_manifest.json").exists()


# ── 自动接入：M1 run_session 收尾 upsert ─────────────────────────────────────

def _mining_daily(seed=3, n_stocks=40, n_days=150):
    import numpy as np
    rng = np.random.default_rng(seed)
    days, d = [], date(2024, 1, 2)
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
                         "open_adj": p, "high_adj": p, "low_adj": p, "open": p, "high": p,
                         "low": p, "pre_close": p, "amount": 1e7,
                         "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows)


def test_session_library_upsert_suite(tmp_path):
    """M1 收尾自动 upsert：passed 候选进库（library gate 复用）；库文件+md 落盘。；test_run_session_no_library_flag_skips；M5/M6 收尾自动 upsert（与 M1 双路径配对）：最终 passed 候选进库；--no-library 不进。"""
    # -- 原 test_run_session_auto_upserts_passed_into_library --
    def _section_0_test_run_session_auto_upserts_passed_into_library(tmp_path):
        from factorzen.discovery.factor_library import load_library
        from factorzen.discovery.mining_session import run_session
        lib_root = str(tmp_path / "lib")
        res = run_session(_mining_daily(), n_trials=40, top_k=5, seed=42, method="random",
                          holdout_ratio=0.2, out_dir=str(tmp_path / "sess"),
                          library_root=lib_root, library_universe="csi300")
        passed = [c for c in res["candidates"] if c.get("passed")]
        lib = load_library("ashare", root=lib_root)
        lib_exprs = {r.expression for r in lib}
        from factorzen.discovery.expression import parse_expr, to_expr_string
        for c in passed:
            assert to_expr_string(parse_expr(c["expression"])) in lib_exprs
        if passed:
            assert (Path(lib_root) / "ashare.jsonl").exists()
            assert (Path(lib_root) / "ashare.md").exists()
            assert all(r.universe == "csi300" for r in lib)

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_run_session_auto_upserts_passed_into_library(_tp0)

    # -- 原 test_run_session_no_library_flag_skips --
    def _section_1_test_run_session_no_library_flag_skips(tmp_path):
        from factorzen.discovery.factor_library import load_library
        from factorzen.discovery.mining_session import run_session
        lib_root = str(tmp_path / "lib")
        run_session(_mining_daily(), n_trials=40, top_k=5, seed=42, method="random",
                    holdout_ratio=0.2, out_dir=str(tmp_path / "sess"),
                    update_library=False, library_root=lib_root)
        assert load_library("ashare", root=lib_root) == []          # 关开关 → 不写库

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_run_session_no_library_flag_skips(_tp1)

    # -- 原 test_run_team_agent_auto_upserts_into_library --
    def _section_2_test_run_team_agent_auto_upserts_into_library(tmp_path):
        from factorzen.agents.team_orchestrator import run_team_agent
        from factorzen.discovery.factor_library import load_library
        daily = _mining_daily(n_days=180)
        lib_root = str(tmp_path / "lib")
        run_team_agent(daily, _scripted_team(), n_rounds=2, seed=42,
                       index_path=str(tmp_path / "e.jsonl"), library_root=lib_root,
                       data_window={"start": "20240102", "end": "20240930",
                                    "universe": "csi300", "market": "ashare"})
        lib = load_library("ashare", root=lib_root)
        # 脚本 ts_mean(close,5) 若过 library gate 则进库；进库者窗口/市场正确
        for r in lib:
            assert r.market == "ashare" and r.eval_start == "20240102"

        lib_root2 = str(tmp_path / "lib2")
        run_team_agent(daily, _scripted_team(), n_rounds=2, seed=42,
                       index_path=str(tmp_path / "e2.jsonl"), update_library=False,
                       library_root=lib_root2)
        assert load_library("ashare", root=lib_root2) == []

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_run_team_agent_auto_upserts_into_library(_tp2)


# ── 自动接入：M5/M6 run_team_agent 收尾 upsert ───────────────────────────────

def _scripted_team():
    hyp = __import__("json").dumps({"hypotheses": ["动量"]})
    code = __import__("json").dumps({"expressions": ["ts_mean(close,5)"]})
    crit = __import__("json").dumps({"verdict": "keep", "reason": "ok"})
    seq = [hyp, code, crit] * 50
    i = {"k": 0}

    def fn(messages):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v

    return fn


# ── rebuild 真实路径：候选源收集 + 统一窗口评估器（offline mock 数据）─────────────

def test_collect_source_expressions_filters_by_market(tmp_path):
    import json as J

    from factorzen.discovery.factor_library import collect_source_expressions
    mt = tmp_path / "mine_team"
    mt.mkdir()
    (mt / "experiment_index.jsonl").write_text(
        J.dumps({"expression": "rank(close)", "data_window": {"market": "ashare"}}) + "\n"
        + J.dumps({"expression": "funding_rate", "data_window": {"market": "crypto"}}) + "\n")
    (mt / "experiment_index_crypto.jsonl").write_text(
        J.dumps({"expression": "ts_mean(funding_rate, 5)"}) + "\n")
    run = mt / "20260101_team_1_2r"
    run.mkdir()
    (run / "manifest.json").write_text(
        J.dumps({"params": {"market": "ashare"}, "candidates": [{"expression": "rank(vol)"}]}))
    ms = tmp_path / "mining_sessions" / "session_1_random"
    ms.mkdir(parents=True)
    pl.DataFrame({"rank": [1], "expression": ["ts_std(close, 10)"]}).write_csv(ms / "candidates.csv")

    a = collect_source_expressions("ashare", mine_team_root=str(mt),
                                   mining_sessions_root=str(tmp_path / "mining_sessions"))
    assert "rank(close)" in a and "rank(vol)" in a and "ts_std(close, 10)" in a
    assert "funding_rate" not in a                    # crypto 记录被剔出 ashare
    c = collect_source_expressions("crypto", mine_team_root=str(mt),
                                   mining_sessions_root=str(tmp_path / "mining_sessions"))
    assert "funding_rate" in c and "ts_mean(funding_rate, 5)" in c
    assert "rank(close)" not in c                      # ashare 记录被剔出 crypto
    assert "ts_std(close, 10)" in c                    # mining_sessions 无归属 → 两市场都收


def test_build_library_evaluator_produces_metrics(tmp_path):
    """真实评估器路径（offline mock A股数据）：evaluate 返回带全指标的候选 dict，不崩。"""
    import numpy as np

    from factorzen.discovery.factor_library import DEFAULT_DECORR_MAX_DATES, build_library_evaluator
    daily = _mining_daily(n_days=150)
    dates = sorted(daily["trade_date"].unique().to_list())
    eval_start = dates[30].strftime("%Y%m%d")          # 30 天预热前缀
    evaluate, compact = build_library_evaluator(daily, eval_start=eval_start)
    rows = evaluate(["rank(close)", "ts_mean(close, 5)", "funding_rate"])  # 末者 A股 parse 失败被剔
    exprs = {r["expression"] for r in rows}
    assert "rank(close)" in exprs
    for r in rows:
        for k in ("ic_train", "ir_train", "holdout_ic", "dsr_pvalue", "n_train",
                  "n_holdout_days"):
            assert k in r
        assert r["n_train"] > 0
        # 覆盖门燃料：健康表达式的 holdout 有效 IC 天数必须为正且不超过 holdout 段长度
        assert 0 < r["n_holdout_days"] <= 150
    # 去相关物化器给出紧凑 float32 (date×stock) 矩阵（丢掉 ts_code 字符串列，内存有界）
    m = compact("rank(close)")
    assert isinstance(m, np.ndarray) and m.ndim == 2 and m.dtype == np.float32
    assert m.shape[0] <= DEFAULT_DECORR_MAX_DATES      # 日期维封顶
    assert m.shape[1] == 40                            # 40 只股票
    assert compact("funding_rate") is None             # A股无此叶子 → None


def test_compact_corr_parity_with_max_correlation():
    """parity：紧凑矩阵逐对相关 `_avg_cs_corr_matrices` 与既有 `max_correlation` 语义一致
    （逐日截面 Pearson 跨日平均），杜绝去相关语义漂移。"""
    from factorzen.discovery.factor_library import _avg_cs_corr_matrices, _panel_to_compact
    from factorzen.discovery.scoring import max_correlation
    base = [((i * 37) % 40) + 0.5 for i in range(40)]
    pa = _panel(base)
    pb = _panel([x * 3.0 + 7.0 for x in base])          # 单调变换 → 高相关
    pc = _panel([((i * 11) % 40) + 0.5 for i in range(40)])  # 不同序 → 低相关
    dates = sorted(set(pa["trade_date"].to_list()))
    stocks = sorted(set(pa["ts_code"].to_list()))
    di = {d: i for i, d in enumerate(dates)}
    si = {s: i for i, s in enumerate(stocks)}
    D, S = len(dates), len(stocks)
    ma = _panel_to_compact(pa, di, si, D, S)
    mb = _panel_to_compact(pb, di, si, D, S)
    mc = _panel_to_compact(pc, di, si, D, S)
    # 与 max_correlation（逐对，真源）逐对核对
    assert abs(abs(_avg_cs_corr_matrices(ma, mb)) - max_correlation(pa, {"b": pb})) < 1e-4
    assert abs(abs(_avg_cs_corr_matrices(ma, mc)) - max_correlation(pa, {"c": pc})) < 1e-4
    assert abs(_avg_cs_corr_matrices(ma, mb)) > 0.9    # 单调变换高相关
    assert abs(_avg_cs_corr_matrices(ma, mc)) < 0.7    # 不同序低相关


def test_evaluate_batches_candidates_not_all_at_once(monkeypatch):
    """内存有界回归：evaluate 分批调 evaluate_expressions，绝不一次性把全部候选塞进去。"""
    import factorzen.discovery.evaluation as _ev
    from factorzen.discovery.factor_library import build_library_evaluator
    batch_sizes: list[int] = []
    real = _ev.evaluate_expressions

    def spy(expr_strs, *a, **k):
        batch_sizes.append(len(expr_strs))
        return real(expr_strs, *a, **k)

    monkeypatch.setattr(_ev, "evaluate_expressions", spy)
    daily = _mining_daily(n_days=150)
    dates = sorted(daily["trade_date"].unique().to_list())
    eval_start = dates[30].strftime("%Y%m%d")
    evaluate, _compact = build_library_evaluator(daily, eval_start=eval_start, batch_size=16)
    exprs = [f"add(close, {i})" for i in range(40)]     # 40 个不同的合法表达式
    evaluate(exprs)
    assert len(batch_sizes) >= 3                        # 40/16 → 至少 3 批
    assert max(batch_sizes) <= 16                       # 每批不超过 batch_size（不一次性全塞）
    assert sum(batch_sizes) == 40                       # 覆盖全部候选


def test_rebuild_real_evaluator_runs_and_writes_manifest(tmp_path):
    """rebuild 全真实路径（mock 数据）跑通不崩、落 manifest；入库者为合法 FactorRecord。"""
    from factorzen.discovery.factor_library import (
        FactorRecord,
        build_library_evaluator,
        load_library,
        rebuild,
    )
    daily = _mining_daily(n_days=150)
    dates = sorted(daily["trade_date"].unique().to_list())
    eval_start = dates[30].strftime("%Y%m%d")
    end = dates[-1].strftime("%Y%m%d")
    evaluate, compact = build_library_evaluator(daily, eval_start=eval_start)
    rebuild("ashare", sources=["rank(close)", "ts_mean(close, 5)", "funding_rate"],
            eval_window=(eval_start, end), universe=None, horizon=1, evaluate=evaluate,
            compact_materialize=compact, git_sha="x", now="2026-07-12", root=str(tmp_path))
    assert (Path(tmp_path) / "rebuild_ashare_manifest.json").exists()
    for r in load_library("ashare", root=str(tmp_path)):
        assert isinstance(r, FactorRecord)
        assert r.eval_start == eval_start and r.eval_end == end


def test_rebuild_skips_lookahead_sources(tmp_path):
    """P0：前视源（负窗口）在 rebuild 评估时 parse 抛 ValueError 被跳过，绝不入库。"""
    from factorzen.discovery.expression import is_lookahead_expr
    from factorzen.discovery.factor_library import build_library_evaluator, load_library, rebuild
    daily = _mining_daily(n_days=150)
    dates = sorted(daily["trade_date"].unique().to_list())
    eval_start = dates[30].strftime("%Y%m%d")
    end = dates[-1].strftime("%Y%m%d")
    evaluate, compact = build_library_evaluator(daily, eval_start=eval_start)
    sources = ["rank(close)", "ts_sum(delay(ret_1d, -1), 60)", "delta(close, -5)"]
    rebuild("ashare", sources=sources, eval_window=(eval_start, end), universe=None,
            horizon=1, evaluate=evaluate, compact_materialize=compact, git_sha="x",
            now="2026-07-12", root=str(tmp_path))
    lib_exprs = [r.expression for r in load_library("ashare", root=str(tmp_path))]
    assert not any(is_lookahead_expr(e) for e in lib_exprs), f"前视因子入库: {lib_exprs}"
    # 前视源不进候选，但干净的 rank(close) 若过 gate 可进（不强制，随机游走数据 gate 不定）


def test_rebuild_is_fresh_drops_stale_records(tmp_path):
    """rebuild 从零重建（权威）：旧库里不在本次结果的记录（如 P0 前视残留）被清掉，不残留。"""
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        rebuild,
    )
    stale = FactorRecord(expression="ts_sum(delay(ret_1d, -1.0), 60)", market="ashare",
                         ic_train=0.09, holdout_ic=0.09, status="active",
                         added_at="2026-07-01", updated_at="2026-07-01")
    _save_library("ashare", [stale], root=str(tmp_path))
    assert len(load_library("ashare", root=str(tmp_path))) == 1

    def evaluate(exprs):
        return [_cand("rank(close)", ic_train=0.05, holdout_ic=0.04)]

    rebuild("ashare", sources=["rank(close)"], eval_window=("20200101", "20260101"),
            universe=None, horizon=1, evaluate=evaluate, git_sha="x", now="2026-07-12",
            root=str(tmp_path))
    lib = [r.expression for r in load_library("ashare", root=str(tmp_path))]
    assert lib == ["rank(close)"]                              # 只剩本次重算结果
    assert not any("delay(ret_1d, -1" in e for e in lib)       # 前视残留被清

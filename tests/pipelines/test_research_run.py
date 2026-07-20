"""test_pipelines_research_run.py：无 module docstring 的测试。
test_experiment.py：Tests for common.experiment module.
test_factor_sweep.py：factor_sweep 纯逻辑单测：网格展开、注入式编排排序、表格/CSV 渲染、失败容错。
test_team_pipeline.py：无 module docstring 的测试。
"""

from __future__ import annotations

import datetime as dt
import json
import math
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from factorzen.pipelines.factor_mine_team import run_team_mine
from factorzen.pipelines.factor_sweep import (
    expand_grid,
    format_sweep_csv,
    format_sweep_table,
    run_sweep,
)


# ==== 来自 test_pipelines_research_run.py ====
def test_rebalance_dates_skips_warmup_and_last():
    from factorzen.pipelines.research_run import _rebalance_dates
    dates = list(range(10))  # 用 int 代交易日，验证切片逻辑
    # warmup=2 → usable=dates[2:-1]=[2..8]，step 2 → [2,4,6,8]
    assert _rebalance_dates(dates, rebalance_days=2, warmup=2) == [2, 4, 6, 8]
    # 交易日不足 warmup+1 → 空
    assert _rebalance_dates(dates, rebalance_days=2, warmup=20) == []

def test_rebalance_dates_rejects_bad_step():
    from factorzen.pipelines.research_run import _rebalance_dates
    with pytest.raises(ValueError):
        _rebalance_dates(list(range(10)), rebalance_days=0, warmup=0)

def test_select_passed_expression_picks_head_passed():
    from factorzen.pipelines.research_run import _select_passed_expression
    cands = [
        {"expression": "ts_mean(close, 5)", "passed": False},
        {"expression": "close", "passed": True},   # 头部 passed
        {"expression": "vol", "passed": True},
    ]
    assert _select_passed_expression(cands) == "close"

def test_select_passed_expression_raises_when_none_passed():
    from factorzen.pipelines.research_run import _select_passed_expression
    with pytest.raises(RuntimeError, match="passed"):
        _select_passed_expression([{"expression": "close", "passed": False}])

def test_alpha_file_for_date_writes_ts_code_alpha(tmp_path):
    from factorzen.pipelines.research_run import _alpha_file_for_date
    d = date(2024, 3, 1)
    panel = pl.DataFrame({
        "trade_date": [d, d, date(2024, 3, 2)],
        "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ"],
        "factor_value": [0.1, float("inf"), 0.3],  # inf 应被过滤
    })
    out = _alpha_file_for_date(panel, d, tmp_path / "a.parquet")
    got = pl.read_parquet(out)
    assert got.columns == ["ts_code", "alpha"]
    assert got.height == 1 and got["ts_code"][0] == "000001.SZ"  # inf 那行被剔除

# ── 全链路 mock e2e：monkeypatch 所有数据密集接缝，验证编排 glue ─────────────
def _fake_trade_dates(n=10):
    return [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]

@pytest.fixture
def patched_stages(monkeypatch, tmp_path):
    """把 run_research 内部所有数据密集调用替换成注入合成数据的 fake，捕获调用参数。"""
    calls: dict = {"portfolio": [], "sim": [], "report": 0, "fetch_daily_starts": []}
    tdates = _fake_trade_dates(10)
    codes = ["000001.SZ", "000002.SZ", "000003.SZ"]

    def fake_run_mine(**kw):
        calls["mine"] = kw
        sess = tmp_path / "mining" / "session_42_random"
        sess.mkdir(parents=True, exist_ok=True)
        return {"session_dir": str(sess),
                "candidates": [{"expression": "ts_mean(close, 5)", "passed": False},
                               {"expression": "close", "passed": True},
                               {"expression": "vol", "passed": True}]}

    def fake_get_universe(d, name):
        return pl.DataFrame({"ts_code": codes, "industry": ["银行", "地产", "科技"]})

    def fake_get_membership(start, end, name):
        # 稳定成分：每日全 codes（与旧 end-snapshot 等价，零回归既有 e2e）
        rows = [
            {"trade_date": d.strftime("%Y%m%d"), "ts_code": c}
            for d in tdates
            for c in codes
        ]
        return pl.DataFrame(rows)

    class FakeCtx:
        def __init__(self, **kw): self.kw = kw

    class FakeExprFactor:
        def __init__(self, expression=None, **kw): self.expression = expression

        def compute(self, ctx):
            rows = []
            for i, d in enumerate(tdates):
                for j, c in enumerate(codes):
                    rows.append({"trade_date": d, "ts_code": c, "factor_value": float(i + j) * 0.01})
            return pl.DataFrame(rows)

    def fake_fetch_daily(start, end):
        calls["fetch_daily_starts"].append(start)
        rows = [{"trade_date": d, "ts_code": c, "close": 10.0 + i}
                for i, d in enumerate(tdates) for c in codes]
        return pl.DataFrame(rows)

    def fake_fetch_daily_basic(start, end):
        return pl.DataFrame({"trade_date": [tdates[0]], "ts_code": [codes[0]]})

    class FakeRiskModel:
        def build(self, daily, daily_basic, stocks, start, end, **kwargs):
            # kwargs: style_panel / industry_panel / industry_names（W3 复用）
            return SimpleNamespace(
                factor_exposures=SimpleNamespace(codes=codes),
                factor_names=["size", "ind_银行", "ind_地产"])

    def fake_run_portfolio(alpha, risk_result, **kw):
        calls["portfolio"].append(kw)
        run_dir = Path(kw["out_dir"]) / kw["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"ts_code": codes, "target_weight": [0.4, 0.3, 0.3],
                      "prev_weight": [0.0, 0.0, 0.0]}).write_parquet(run_dir / "weights.parquet")
        pl.DataFrame({"type": ["x"], "key": ["y"], "value": [1.0]}).write_csv(run_dir / "attribution.csv")
        pl.DataFrame({"metric": ["te"], "value": [0.01]}).write_csv(run_dir / "risk_summary.csv")
        (run_dir / "manifest.json").write_text(
            json.dumps({"signal_date": kw["signal_date"], "status": "optimal"}))
        return {"run_dir": str(run_dir), "status": "optimal", "n_holdings": 3}

    def fake_run_sim(dirs, daily, *, out_dir, run_id, cost_model=None):
        calls["sim"].append({"dirs": dirs, "out_dir": out_dir, "run_id": run_id})
        run_dir = Path(out_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"trade_date": tdates[:3], "net_return": [0.01, -0.02, 0.03],
                      "nav": [1.01, 0.99, 1.02]}).write_parquet(run_dir / "nav.parquet")
        (run_dir / "metrics.json").write_text(json.dumps({"sharpe": 1.2, "ann_ret": 0.15}))
        return {"run_dir": str(run_dir), "sharpe": 1.2, "ann_ret": 0.15}

    def fake_report(**kw):
        calls["report"] += 1
        return "<html>dashboard</html>"

    monkeypatch.setattr("factorzen.pipelines.factor_mine.run_mine", fake_run_mine)
    monkeypatch.setattr("factorzen.core.universe.get_universe", fake_get_universe)
    monkeypatch.setattr(
        "factorzen.pipelines.daily_single.get_universe", fake_get_universe
    )
    monkeypatch.setattr(
        "factorzen.core.universe.get_universe_membership", fake_get_membership
    )
    monkeypatch.setattr("factorzen.daily.data.context.FactorDataContext", FakeCtx)
    monkeypatch.setattr("factorzen.discovery.factor.ExpressionFactor", FakeExprFactor)
    monkeypatch.setattr("factorzen.core.loader.fetch_daily", fake_fetch_daily)
    monkeypatch.setattr("factorzen.core.loader.fetch_daily_basic", fake_fetch_daily_basic)
    monkeypatch.setattr("factorzen.risk.model.RiskModel", FakeRiskModel)
    monkeypatch.setattr("factorzen.pipelines.portfolio_build.run_portfolio", fake_run_portfolio)
    monkeypatch.setattr("factorzen.sim.engine.run_portfolio_simulation", fake_run_sim)
    monkeypatch.setattr("factorzen.reports.portfolio_report.generate_portfolio_report", fake_report)
    return calls

def test_run_research_end_to_end_wiring(patched_stages, tmp_path):
    """全链路：mine→选头部 passed→循环 build→sim→report，run_id 贯穿、格式桥、产物落盘。"""
    from factorzen.pipelines.research_run import run_research
    calls = patched_stages
    res = run_research(start="20240101", end="20240110", universe=None,
                       n_trials=10, method="random", seed=42, top_k=5,
                       rebalance_days=2, warmup=2, out_root=str(tmp_path))

    # run_id 贯穿
    assert res["run_id"] == "research_42_random"
    # 选的是头部 passed 因子（close），非 rank1 未过护栏的 ts_mean
    assert res["expression"] == "close"
    # 10 交易日, warmup=2 → usable=[2..8], step2 → 4 个调仓日
    assert res["n_rebalances"] == 4
    assert len(calls["portfolio"]) == 4
    # 每次 build：out_dir 共享 = portfolios/<rid>, run_id=日期串, signal_date=ISO
    for c in calls["portfolio"]:
        assert c["out_dir"].endswith(f"portfolios/{res['run_id']}")
        assert len(c["run_id"]) == 8 and c["run_id"].isdigit()      # YYYYMMDD
        assert c["signal_date"].count("-") == 2                      # ISO
    # sim 一次，run_id=rid，喂全部调仓 run_dir
    assert len(calls["sim"]) == 1
    assert calls["sim"][0]["run_id"] == res["run_id"]
    assert len(calls["sim"][0]["dirs"]) == 4
    # report 生成并落盘
    assert calls["report"] == 1
    html = Path(res["report_html"])
    assert html.exists() and html.name == f"portfolio_{res['run_id']}.html"
    assert res["sharpe"] == 1.2
    # 顶层可复现 manifest 落盘
    assert (tmp_path / "research" / res["run_id"] / "manifest.json").exists()
    # L1：风险模型走 lookback 预热（load_risk_inputs），即有一次 fetch_daily 的 start
    # 早于研究区间起点（否则窗口首日滚动风格因子退化，与 portfolio build 双路径漂移）
    assert any(s < "20240101" for s in calls["fetch_daily_starts"]), (
        f"风险模型应带 lookback 预热(start<20240101)，实得 {calls['fetch_daily_starts']}"
    )

def test_run_research_threads_prev_weights_for_turnover(patched_stages, tmp_path):
    """L2：多期 build 须把上一期权重作为 prev_weights 传下去，否则 --turnover 静默失效。"""
    from factorzen.pipelines.research_run import run_research
    calls = patched_stages
    run_research(start="20240101", end="20240110", n_trials=10, seed=42,
                 rebalance_days=2, warmup=2, turnover=0.5, out_root=str(tmp_path))
    ports = calls["portfolio"]
    assert len(ports) >= 2
    assert ports[0].get("prev_weights") is None                  # 首期无上一期
    for c in ports[1:]:
        assert c.get("prev_weights") is not None                 # 后续期传上期权重
        assert c.get("turnover_budget") == 0.5

def test_run_research_custom_run_id_threads_everywhere(patched_stages, tmp_path):
    from factorzen.pipelines.research_run import run_research
    calls = patched_stages
    res = run_research(start="20240101", end="20240110", n_trials=10, seed=7,
                       rebalance_days=3, warmup=2, run_id="myrun", out_root=str(tmp_path))
    assert res["run_id"] == "myrun"
    assert calls["sim"][0]["run_id"] == "myrun"
    assert all(c["out_dir"].endswith("portfolios/myrun") for c in calls["portfolio"])
    assert Path(res["report_html"]).name == "portfolio_myrun.html"

def test_run_research_passes_intraday_to_mine_and_manifest(patched_stages, tmp_path):
    """``--intraday-leaves`` 透传：run_mine 收到 intraday 参数，manifest.config 可复现落盘。"""
    from factorzen.pipelines.research_run import run_research

    calls = patched_stages
    res = run_research(
        start="20240101", end="20240110", n_trials=10, seed=42,
        rebalance_days=2, warmup=2, out_root=str(tmp_path),
        intraday=True, intraday_freq="15min",
    )
    # 挖掘阶段把 i_* 纳入搜索空间
    assert calls["mine"]["intraday"] is True
    assert calls["mine"]["intraday_freq"] == "15min"
    # 默认可复现：manifest config 记 flag（漏记=假复现）
    manifest = json.loads(
        (tmp_path / "research" / res["run_id"] / "manifest.json").read_text(encoding="utf-8")
    )
    cfg = manifest["config"]
    assert cfg["intraday"] is True
    assert cfg["intraday_freq"] == "15min"

def test_run_research_intraday_defaults_off(patched_stages, tmp_path):
    """默认关日内叶子（零回归）：run_mine 收 intraday=False，manifest 仍记字段。"""
    from factorzen.pipelines.research_run import run_research

    calls = patched_stages
    res = run_research(
        start="20240101", end="20240110", n_trials=10, seed=42,
        rebalance_days=2, warmup=2, out_root=str(tmp_path),
    )
    assert calls["mine"]["intraday"] is False
    assert calls["mine"]["intraday_freq"] == "5min"
    manifest = json.loads(
        (tmp_path / "research" / res["run_id"] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["config"]["intraday"] is False
    assert manifest["config"]["intraday_freq"] == "5min"

# ==== 来自 test_experiment.py ====
def test_experiment_success(tmp_path, monkeypatch):
    from factorzen.core import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    from factorzen.config.research import RunConfig

    cfg = RunConfig(factor="momentum_20d", start="20230101", end="20241231")

    with exp_mod.run_experiment(cfg, run_id="test_run") as exp_dir:
        pass  # success

    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert manifest["status"] == "success"
    assert manifest["config"]["factor"] == "momentum_20d"
    assert manifest["end_ts"] is not None
    assert "git_sha" in manifest

def test_auto_run_id_includes_factor_name(tmp_path, monkeypatch):
    from factorzen.config.research import RunConfig
    from factorzen.core import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    cfg = RunConfig(factor="momentum_12_1", start="20230101", end="20241231")

    with exp_mod.run_experiment(cfg) as exp_dir:
        pass

    manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
    assert exp_dir.name.startswith("momentum_12_1_")
    assert manifest["run_id"] == exp_dir.name

def test_experiment_failure(tmp_path, monkeypatch):
    from factorzen.core import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    from factorzen.config.research import RunConfig

    cfg = RunConfig(factor="x", start="20230101", end="20241231")

    with pytest.raises(ValueError), exp_mod.run_experiment(cfg, run_id="fail_run") as exp_dir:
        raise ValueError("test error")

    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert manifest["status"] == "failure"
    assert "test error" in manifest["error"]

def test_experiment_index_created_on_success(tmp_path, monkeypatch):
    """成功 run 后，experiment_index.jsonl 被创建并含正确字段。"""
    import json as _json

    from factorzen.config.research import RunConfig
    from factorzen.core import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    cfg = RunConfig(factor="reversal_5d", start="20240101", end="20241231")

    with exp_mod.run_experiment(cfg, run_id="idx_run") as _exp_dir:
        pass

    index_path = tmp_path / "experiments" / "experiment_index.jsonl"
    assert index_path.exists(), "experiment_index.jsonl 应被创建"
    entry = _json.loads(index_path.read_text(encoding="utf-8").strip())
    assert entry["run_id"] == "idx_run"
    assert entry["factor"] == "reversal_5d"
    assert entry["status"] == "success"
    assert "manifest_path" in entry

def test_experiment_index_appends_multiple_runs(tmp_path, monkeypatch):
    """两次 run 各 append 一行，JSONL 共两行。"""
    import json as _json

    from factorzen.config.research import RunConfig
    from factorzen.core import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")

    for run_id, factor in [("run_a", "momentum_20d"), ("run_b", "reversal_5d")]:
        cfg = RunConfig(factor=factor, start="20240101", end="20241231")
        with exp_mod.run_experiment(cfg, run_id=run_id):
            pass

    index_path = tmp_path / "experiments" / "experiment_index.jsonl"
    lines = [_json.loads(ln) for ln in index_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    factors = {e["factor"] for e in lines}
    assert factors == {"momentum_20d", "reversal_5d"}

def test_experiment_index_records_failure_status(tmp_path, monkeypatch):
    """失败 run 的状态也被记录到索引。"""
    import json as _json

    from factorzen.config.research import RunConfig
    from factorzen.core import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    cfg = RunConfig(factor="x", start="20240101", end="20241231")

    with pytest.raises(RuntimeError), exp_mod.run_experiment(cfg, run_id="fail_idx"):
        raise RuntimeError("boom")

    index_path = tmp_path / "experiments" / "experiment_index.jsonl"
    entry = _json.loads(index_path.read_text(encoding="utf-8").strip())
    assert entry["status"] == "failure"

def test_experiment_records_reproducibility_metadata(tmp_path, monkeypatch):
    from factorzen.config.research import RunConfig
    from factorzen.core import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(exp_mod, "_get_git_dirty", lambda: True)
    monkeypatch.setattr(exp_mod, "_get_pixi_lock_hash", lambda: "abc123")
    cfg = RunConfig(factor="momentum_20d", start="20230101", end="20241231")

    with exp_mod.run_experiment(
        cfg,
        run_id="metadata_run",
        command=["python", "factorzen.pipelines.daily_single"],
    ) as exp_dir:
        exp_mod.record_experiment_output(exp_dir, "quality_report", "quality.json")

    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert manifest["git_dirty"] is True
    assert manifest["pixi_lock_sha256"] == "abc123"
    assert manifest["command"] == ["python", "factorzen.pipelines.daily_single"]
    assert manifest["outputs"]["quality_report"] == "quality.json"

def test_record_experiment_metadata_survives_run_finalization(tmp_path, monkeypatch):
    from factorzen.config.research import RunConfig
    from factorzen.core import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    cfg = RunConfig(factor="x", start="20230101", end="20241231")

    with exp_mod.run_experiment(cfg, run_id="meta_run") as exp_dir:
        exp_mod.record_experiment_metadata(exp_dir, "stage_timings", {"ic": 1.2, "backtest": 3.4})

    manifest = json.loads((exp_dir / "manifest.json").read_text())
    # 运行期写入的顶层元数据不应被 run_experiment 的 finally 覆盖丢失
    assert manifest["stage_timings"] == {"ic": 1.2, "backtest": 3.4}
    assert manifest["status"] == "success"
    assert manifest["end_ts"] is not None

def test_manifest_records_duration_seconds(tmp_path, monkeypatch):
    from factorzen.config.research import RunConfig
    from factorzen.core import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    cfg = RunConfig(factor="momentum_20d", start="20230101", end="20241231")

    with exp_mod.run_experiment(cfg, run_id="dur_run"):
        pass

    manifest = json.loads((tmp_path / "experiments" / "dur_run" / "manifest.json").read_text())
    assert "duration_seconds" in manifest
    assert isinstance(manifest["duration_seconds"], (int, float))
    assert manifest["duration_seconds"] >= 0

def test_experiment_warns_when_git_dirty(tmp_path, monkeypatch, caplog):
    import logging

    from factorzen.config.research import RunConfig
    from factorzen.core import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(exp_mod, "_get_git_dirty", lambda: True)
    cfg = RunConfig(factor="x", start="20230101", end="20241231")

    with caplog.at_level(logging.WARNING), exp_mod.run_experiment(cfg, run_id="dirty_warn_run"):
        pass

    assert any("git_dirty" in r.getMessage() for r in caplog.records)

def test_experiment_does_not_warn_when_clean(tmp_path, monkeypatch, caplog):
    import logging

    from factorzen.config.research import RunConfig
    from factorzen.core import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(exp_mod, "_get_git_dirty", lambda: False)
    cfg = RunConfig(factor="x", start="20230101", end="20241231")

    with caplog.at_level(logging.WARNING), exp_mod.run_experiment(cfg, run_id="clean_run"):
        pass

    assert not any("git_dirty" in r.getMessage() for r in caplog.records)

def test_build_manifest_base_returns_reproducibility_fields(monkeypatch):
    """build_manifest_base() 是可被其它 pipeline（risk_build/portfolio_build）复用的基础字段构造器。"""
    from factorzen.core import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "get_git_sha", lambda: "deadbeef")
    monkeypatch.setattr(exp_mod, "_get_git_dirty", lambda: False)
    monkeypatch.setattr(exp_mod, "_get_pixi_lock_hash", lambda: "lockhash123")

    base = exp_mod.build_manifest_base(
        ["python", "-m", "factorzen.cli.main", "risk", "build"],
        {"start": "20230101", "end": "20231231"},
    )

    assert base["schema_version"] == "1"
    assert base["git_sha"] == "deadbeef"
    assert base["git_dirty"] is False
    assert base["pixi_lock_sha256"] == "lockhash123"
    assert base["command"] == ["python", "-m", "factorzen.cli.main", "risk", "build"]
    assert base["config"] == {"start": "20230101", "end": "20231231"}
    assert base.get("start_ts")

def test_build_manifest_base_accepts_plain_dict_config(monkeypatch):
    """非 RunConfig 调用方（如 risk_build/portfolio_build）可直接传 dict 作为 config。"""
    from factorzen.core import experiment as exp_mod

    base = exp_mod.build_manifest_base(None, {"cov_half_life": 90, "nw_lags": 2})

    assert base["command"] is None
    assert base["config"] == {"cov_half_life": 90, "nw_lags": 2}

def test_build_manifest_base_used_by_run_experiment_unchanged(tmp_path, monkeypatch):
    """run_experiment() 重构为复用 build_manifest_base 后，对外行为（字段集合/取值）保持不变。"""
    from factorzen.config.research import RunConfig
    from factorzen.core import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    cfg = RunConfig(factor="momentum_20d", start="20230101", end="20241231")

    with exp_mod.run_experiment(cfg, run_id="base_reuse_run", command=["fz", "daily-single"]) as exp_dir:
        pass

    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == "1"
    assert manifest["run_id"] == "base_reuse_run"
    assert manifest["command"] == ["fz", "daily-single"]
    assert manifest["config"]["factor"] == "momentum_20d"
    assert isinstance(manifest["git_dirty"], bool)
    assert isinstance(manifest["pixi_lock_sha256"], str)

# ==== 来自 test_factor_sweep.py ====
def test_expand_grid_cartesian_product():
    combos = expand_grid(["backtest.top_n=30,50", "preprocessing.normalizer=zscore,rank_normal"])
    assert combos == [
        ["backtest.top_n=30", "preprocessing.normalizer=zscore"],
        ["backtest.top_n=30", "preprocessing.normalizer=rank_normal"],
        ["backtest.top_n=50", "preprocessing.normalizer=zscore"],
        ["backtest.top_n=50", "preprocessing.normalizer=rank_normal"],
    ]

def test_expand_grid_single_dim():
    assert expand_grid(["backtest.top_n=30,50,100"]) == [
        ["backtest.top_n=30"],
        ["backtest.top_n=50"],
        ["backtest.top_n=100"],
    ]

def test_expand_grid_empty():
    assert expand_grid([]) == []

def test_expand_grid_strips_whitespace():
    assert expand_grid(["backtest.top_n = 30, 50 "]) == [["backtest.top_n=30"], ["backtest.top_n=50"]]

@pytest.mark.parametrize("token", ["backtest.top_n", "=30,50", "backtest.top_n="])
def test_expand_grid_rejects_bad_tokens(token):
    with pytest.raises(ValueError):
        expand_grid([token])

def test_run_sweep_collects_and_sorts_by_ir():
    metrics = {
        "backtest.top_n=30": {"ic_mean": 0.04, "ir": 0.18},
        "backtest.top_n=50": {"ic_mean": 0.05, "ir": 0.12},
        "backtest.top_n=100": {"ic_mean": 0.03, "ir": 0.20},
    }

    def fake_runner(overrides):
        return metrics[overrides[0]]

    rows = run_sweep(["backtest.top_n=30,50,100"], fake_runner, sort_by="ir")
    assert [r["overrides"][0] for r in rows] == [
        "backtest.top_n=100",  # ir 0.20
        "backtest.top_n=30",  # ir 0.18
        "backtest.top_n=50",  # ir 0.12
    ]

def test_run_sweep_sort_by_backtest_metric():
    """top_n 维度只影响回测：按 sharpe 排序才有区分度。"""
    metrics = {
        "backtest.top_n=20": {"ir": 0.1, "sharpe": -1.15},
        "backtest.top_n=50": {"ir": 0.1, "sharpe": -1.21},
    }

    def fake_runner(overrides):
        return metrics[overrides[0]]

    rows = run_sweep(["backtest.top_n=20,50"], fake_runner, sort_by="sharpe")
    assert [r["overrides"][0] for r in rows] == ["backtest.top_n=20", "backtest.top_n=50"]

def test_run_sweep_applies_extra_overrides():
    seen = []

    def fake_runner(overrides):
        seen.append(overrides)
        return {"ir": 1.0}

    run_sweep(["backtest.top_n=30"], fake_runner, extra_overrides=["preprocessing.neutralize=true"])
    assert seen == [["preprocessing.neutralize=true", "backtest.top_n=30"]]

def test_run_sweep_tolerates_runner_failure():
    def flaky_runner(overrides):
        if overrides[0].endswith("50"):
            raise RuntimeError("数据不足")
        return {"ir": 0.3}

    rows = run_sweep(["backtest.top_n=30,50"], flaky_runner, sort_by="ir")
    # 成功组排前，失败组（-inf）排后并带 error
    assert rows[0]["overrides"] == ["backtest.top_n=30"]
    assert rows[0]["ir"] == 0.3
    assert rows[1]["error"] == "数据不足"
    assert "ir" not in rows[1]

def test_run_sweep_tolerates_runner_systemexit():
    # daily_single.main() 内部错误统一走 sys.exit(1/2) = SystemExit（BaseException，
    # 非 Exception），会逃逸 except Exception 而中止整个 sweep、丢掉前面已跑组合。
    # run_sweep 契约是"单组合失败不中断"，须一并捕获 SystemExit。
    def flaky_runner(overrides):
        if overrides[0].endswith("50"):
            raise SystemExit(2)  # 模拟 daily_single.main() sys.exit(2)
        return {"ir": 0.3}

    rows = run_sweep(["backtest.top_n=30,50"], flaky_runner, sort_by="ir")
    assert len(rows) == 2
    assert rows[0]["overrides"] == ["backtest.top_n=30"]
    assert rows[0]["ir"] == 0.3
    assert "error" in rows[1]
    assert "ir" not in rows[1]

def test_run_sweep_nan_sorts_last():
    def runner(overrides):
        return {"ir": float("nan")} if overrides[0].endswith("50") else {"ir": 0.1}

    rows = run_sweep(["backtest.top_n=50,30"], runner, sort_by="ir")
    assert rows[0]["overrides"] == ["backtest.top_n=30"]
    assert math.isnan(rows[1]["ir"])

def test_format_sweep_table_has_headers_and_rows():
    rows = [
        {
            "overrides": ["backtest.top_n=30"],
            "ic_mean": 0.04,
            "ir": 0.18,
            "sharpe": -1.15,
            "ann_ret": -0.022,
            "avg_turnover": 0.51,
            "n": 24,
        },
    ]
    table = format_sweep_table(rows)
    assert "top_n" in table  # 短名表头
    assert "ir" in table
    assert "sharpe" in table  # 回测指标列
    assert "0.1800" in table  # ir 4 位小数
    assert "-1.1500" in table  # sharpe
    assert "30" in table

def test_format_sweep_table_empty():
    assert format_sweep_table([]) == "(空 sweep)"

def test_format_sweep_table_shows_error():
    rows = [{"overrides": ["backtest.top_n=50"], "error": "数据不足"}]
    assert "数据不足" in format_sweep_table(rows)

def test_format_sweep_csv_roundtrip():
    rows = [
        {
            "overrides": ["backtest.top_n=30"],
            "ic_mean": 0.04,
            "ir": 0.18,
            "sharpe": -1.15,
            "ann_ret": -0.022,
            "avg_turnover": 0.51,
            "n": 24,
        },
    ]
    csv_text = format_sweep_csv(rows)
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("backtest.top_n,")
    assert "sharpe" in lines[0]
    assert "0.18" in lines[1]
    assert "30" in lines[1]

def test_format_sweep_csv_empty():
    assert format_sweep_csv([]) == ""

# ==== 来自 test_team_pipeline.py ====
def _mock_daily(n_stocks=20, n_days=180, seed=1):
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

def _scripted_team():
    seq = [json.dumps({"hypotheses": ["动量"]}), json.dumps({"expressions": ["ts_mean(close,5)"]}),
           json.dumps({"verdict": "keep", "reason": "ok"})] * 50
    i = {"k": 0}

    def fn(messages):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v

    return fn

def test_run_team_mine_writes_team_manifest(tmp_path: Path):
    daily = _mock_daily()
    res = run_team_mine(daily, n_rounds=2, seed=42, out_dir=str(tmp_path),
                        index_path=str(tmp_path / "e.jsonl"), llm_fn=_scripted_team(),
                        run_id="t1", export=False)
    run_dir = Path(res["run_dir"])
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "candidates.csv").exists()
    m = json.loads((run_dir / "manifest.json").read_text())
    assert "rounds_log" in m and "roles" in m       # team manifest 角色决策可审计
    assert res["n_trials"] == m["n_trials"]


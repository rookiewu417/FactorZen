from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest


# ── 纯逻辑单测 ─────────────────────────────────────────────────────────────
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
        def build(self, daily, daily_basic, stocks, start, end):
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

# tests/test_agent_cli.py
"""Parser smoke tests for `fz mine agent` CLI subcommand."""
from __future__ import annotations


def test_parser_has_mine_agent():
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        ["mine", "agent", "--start", "20220101", "--end", "20231231",
         "--iterations", "5", "--seed", "42"]
    )
    assert args.command == "mine"
    assert args.mine_command == "agent"
    assert args.start == "20220101"
    assert args.end == "20231231"
    assert args.iterations == 5
    assert args.seed == 42
    assert callable(args.func)


def test_parser_mine_agent_defaults():
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        ["mine", "agent", "--start", "20220101", "--end", "20231231"]
    )
    assert args.iterations == 5
    assert args.top_k == 5
    assert args.seed == 42
    assert args.human_review is False
    assert args.universe is None


def test_parser_mine_agent_human_review():
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        ["mine", "agent", "--start", "20220101", "--end", "20231231", "--human-review"]
    )
    assert args.human_review is True


def test_cmd_mine_agent_forwards_args_to_run_agent_mine(monkeypatch, capsys):
    """`fz mine agent`（无 --universe）应原样转发 daily + CLI 参数给 run_agent_mine。"""
    import polars as pl

    from factorzen.cli import main as cli

    fake_daily = pl.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20220101"]})

    prep_calls: list[tuple] = []

    def fake_prepare(start, end, universe=None, lookback_days=None, **kw):
        prep_calls.append((start, end, universe))
        return fake_daily

    run_calls: list[dict[str, object]] = []

    def fake_run_agent_mine(daily, *, n_rounds, seed, top_k, human_review, **_):
        run_calls.append(
            {
                "daily": daily,
                "n_rounds": n_rounds,
                "seed": seed,
                "top_k": top_k,
                "human_review": human_review,
            }
        )
        return {
            "n_candidates": 4,
            "n_trials": 30,
            "run_dir": "workspace/mine_agent/agent_42_5r",
        }

    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", fake_prepare)
    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine_agent.run_agent_mine", fake_run_agent_mine
    )

    rc = cli.main(
        [
            "mine",
            "agent",
            "--start",
            "20220101",
            "--end",
            "20231231",
            "--iterations",
            "3",
            "--top-k",
            "4",
            "--seed",
            "99",
            "--human-review",
        ]
    )

    assert rc == 0
    assert prep_calls == [("20220101", "20231231", None)]  # 无 --universe
    assert len(run_calls) == 1
    call = run_calls[0]
    assert call["daily"] is fake_daily  # prepare_mining_daily 结果原样转发
    assert call["n_rounds"] == 3
    assert call["seed"] == 99
    assert call["top_k"] == 4
    assert call["human_review"] is True

    out = capsys.readouterr().out
    assert out.splitlines()[0] == (
        "[mine-agent] 候选 4 个 / N=30 → workspace/mine_agent/agent_42_5r"
    )


def test_cmd_mine_agent_forwards_eval_start(monkeypatch):
    """`fz mine agent` 必须把挖掘窗口 --start 作为 eval_start 透传给 run_agent_mine。

    `prepare_mining_daily(start, ...)` 带 lookback 预热前缀；缺了 eval_start，该前缀会被
    `split_holdout` 当训练数据，warmup-parity 修复对生产 `fz mine agent` 完全失效。
    """
    import polars as pl

    from factorzen.cli import main as cli

    fake_daily = pl.DataFrame({"ts_code": ["000001.SZ"]})
    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine.prepare_mining_daily",
        lambda start, end, universe=None, lookback_days=None, **kw: fake_daily,
    )
    captured: dict[str, object] = {}

    def fake_run_agent_mine(daily, *, n_rounds, seed, top_k, human_review,
                            eval_start=None, **_):
        captured["eval_start"] = eval_start
        return {"n_candidates": 0, "n_trials": 0, "run_dir": "x"}

    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine_agent.run_agent_mine", fake_run_agent_mine
    )

    rc = cli.main(["mine", "agent", "--start", "20220101", "--end", "20231231"])
    assert rc == 0
    assert captured["eval_start"] == "20220101"


def test_cmd_mine_agent_provisions_longer_warmup_prefix_for_llm(monkeypatch):
    """`fz mine agent` 必须给 prepare_mining_daily 传更长的预热前缀 lookback_days。

    LLM 窗口无搜索空间上界（实测 structured 爱提 250/252 日长窗因子，required_lookback
    270-315），用 search_space_max_lookback（=180，只覆盖随机搜索 windows≤60）会把这些
    因子（正确地）判欠预热、永远评估不到。故 agent 路前缀须 = AGENT_WARMUP_LOOKBACK(>180)。
    """
    import polars as pl

    from factorzen.cli import main as cli
    from factorzen.config.constants import AGENT_WARMUP_LOOKBACK
    from factorzen.discovery.search.random_search import search_space_max_lookback

    fake_daily = pl.DataFrame({"ts_code": ["000001.SZ"]})
    captured: dict[str, object] = {}

    def fake_prepare(start, end, universe=None, lookback_days=None, **kw):
        captured["lookback_days"] = lookback_days
        return fake_daily

    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", fake_prepare)
    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine_agent.run_agent_mine",
        lambda daily, **kw: {"n_candidates": 0, "n_trials": 0, "run_dir": "x"},
    )

    rc = cli.main(["mine", "agent", "--start", "20220101", "--end", "20231231"])
    assert rc == 0
    assert captured["lookback_days"] == AGENT_WARMUP_LOOKBACK
    assert search_space_max_lookback() < AGENT_WARMUP_LOOKBACK


def test_cmd_mine_agent_passes_universe_to_prepare(monkeypatch):
    """`fz mine agent --universe` 应把 universe 透传给 prepare_mining_daily（其内部经
    FactorDataContext 按 universe 过滤 + 提供复权价/daily_basic）。"""
    import polars as pl

    from factorzen.cli import main as cli

    fake_daily = pl.DataFrame({"ts_code": ["000001.SZ", "000003.SZ"]})

    prep_calls: list[tuple] = []

    def fake_prepare(start, end, universe=None, lookback_days=None, **kw):
        prep_calls.append((start, end, universe))
        return fake_daily

    captured: dict[str, object] = {}

    def fake_run_agent_mine(daily, *, n_rounds, seed, top_k, human_review, **_):
        captured["daily"] = daily
        return {"n_candidates": 0, "n_trials": 0, "run_dir": "workspace/mine_agent/x"}

    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", fake_prepare)
    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine_agent.run_agent_mine", fake_run_agent_mine
    )

    rc = cli.main(
        [
            "mine",
            "agent",
            "--start",
            "20220101",
            "--end",
            "20231231",
            "--universe",
            "csi500",
        ]
    )

    assert rc == 0
    assert prep_calls == [("20220101", "20231231", "csi500")]
    assert captured["daily"] is fake_daily

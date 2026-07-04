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

    fetch_calls: list[tuple[str, str]] = []

    def fake_fetch_daily(start, end):
        fetch_calls.append((start, end))
        return fake_daily

    run_calls: list[dict[str, object]] = []

    def fake_run_agent_mine(daily, *, n_rounds, seed, top_k, human_review):
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

    monkeypatch.setattr("factorzen.core.loader.fetch_daily", fake_fetch_daily)
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
    assert fetch_calls == [("20220101", "20231231")]
    assert len(run_calls) == 1
    call = run_calls[0]
    assert call["daily"] is fake_daily  # 未传 --universe，daily 原样转发、不过滤
    assert call["n_rounds"] == 3
    assert call["seed"] == 99
    assert call["top_k"] == 4
    assert call["human_review"] is True

    out = capsys.readouterr().out
    assert out.splitlines()[0] == (
        "[mine-agent] 候选 4 个 / N=30 → workspace/mine_agent/agent_42_5r"
    )


def test_cmd_mine_agent_filters_daily_by_universe(monkeypatch):
    """`fz mine agent --universe` 应先 get_universe 再按 ts_code 过滤 daily。"""
    import polars as pl

    from factorzen.cli import main as cli

    fake_daily = pl.DataFrame({"ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"]})

    def fake_fetch_daily(start, end):
        return fake_daily

    universe_calls: list[tuple[str, str]] = []

    def fake_get_universe(date_str, universe_name):
        universe_calls.append((date_str, universe_name))
        return pl.DataFrame({"ts_code": ["000001.SZ", "000003.SZ"]})

    captured: dict[str, object] = {}

    def fake_run_agent_mine(daily, *, n_rounds, seed, top_k, human_review):
        captured["daily"] = daily
        return {"n_candidates": 0, "n_trials": 0, "run_dir": "workspace/mine_agent/x"}

    monkeypatch.setattr("factorzen.core.loader.fetch_daily", fake_fetch_daily)
    monkeypatch.setattr("factorzen.core.universe.get_universe", fake_get_universe)
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
    assert universe_calls == [("20231231", "csi500")]
    assert sorted(captured["daily"]["ts_code"].to_list()) == ["000001.SZ", "000003.SZ"]

"""Parser smoke tests for `fz mine team` CLI subcommand."""

from __future__ import annotations


def test_parser_has_mine_team() -> None:
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        ["mine", "team", "--start", "20220101", "--end", "20231231",
         "--iterations", "5", "--seed", "42"]
    )
    assert args.command == "mine"
    assert args.mine_command == "team"
    assert args.start == "20220101"
    assert args.iterations == 5
    assert args.seed == 42
    assert callable(args.func)


def test_parser_mine_team_index_default() -> None:
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        ["mine", "team", "--start", "20220101", "--end", "20231231"]
    )
    assert args.index_path.endswith(".jsonl")  # 长期记忆默认路径


def test_cmd_mine_team_forwards_args_to_run_team_mine(monkeypatch, capsys) -> None:
    """`fz mine team`（无 --universe）应原样转发 daily + CLI 参数给 run_team_mine。"""
    import polars as pl

    from factorzen.cli import main as cli

    fake_daily = pl.DataFrame({"ts_code": ["600000.SH"]})

    prep_calls: list[tuple] = []

    def fake_prepare(start, end, universe=None, lookback_days=None, **kw):
        prep_calls.append((start, end, universe))
        return fake_daily

    run_calls: list[dict[str, object]] = []

    def fake_run_team_mine(daily, *, n_rounds, seed, top_k, index_path, **_):
        run_calls.append(
            {
                "daily": daily,
                "n_rounds": n_rounds,
                "seed": seed,
                "top_k": top_k,
                "index_path": index_path,
            }
        )
        return {
            "n_candidates": 6,
            "n_trials": 50,
            "run_dir": "workspace/mine_team/team_1_2r",
        }

    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", fake_prepare)
    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine_team.run_team_mine", fake_run_team_mine
    )

    rc = cli.main(
        [
            "mine",
            "team",
            "--start",
            "20220101",
            "--end",
            "20231231",
            "--iterations",
            "2",
            "--top-k",
            "6",
            "--seed",
            "1",
            "--index-path",
            "workspace/mine_team/custom_index.jsonl",
        ]
    )

    assert rc == 0
    assert prep_calls == [("20220101", "20231231", None)]  # 无 --universe
    assert len(run_calls) == 1
    call = run_calls[0]
    assert call["daily"] is fake_daily  # prepare_mining_daily 结果原样转发
    assert call["n_rounds"] == 2
    assert call["seed"] == 1
    assert call["top_k"] == 6
    assert call["index_path"] == "workspace/mine_team/custom_index.jsonl"

    out = capsys.readouterr().out
    assert out.splitlines()[0] == (
        "[mine-team] 候选 6 个 / N=50 → workspace/mine_team/team_1_2r"
    )


def test_cmd_mine_team_forwards_eval_start(monkeypatch) -> None:
    """`fz mine team` 必须把挖掘窗口 --start 作为 eval_start 透传给 run_team_mine。

    与 agent 单路径同理：缺了 eval_start，`prepare_mining_daily` 的预热前缀会被
    `split_holdout` 当训练数据，warmup-parity 修复对生产 `fz mine team` 失效。
    """
    import polars as pl

    from factorzen.cli import main as cli

    fake_daily = pl.DataFrame({"ts_code": ["600000.SH"]})
    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine.prepare_mining_daily",
        lambda start, end, universe=None, lookback_days=None, **kw: fake_daily,
    )
    captured: dict[str, object] = {}

    def fake_run_team_mine(daily, *, n_rounds, seed, top_k, index_path,
                           eval_start=None, **_):
        captured["eval_start"] = eval_start
        return {"n_candidates": 0, "n_trials": 0, "run_dir": "x"}

    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine_team.run_team_mine", fake_run_team_mine
    )

    rc = cli.main(
        ["mine", "team", "--start", "20220101", "--end", "20231231",
         "--index-path", "workspace/mine_team/custom_index.jsonl"]
    )
    assert rc == 0
    assert captured["eval_start"] == "20220101"


def test_cmd_mine_team_provisions_longer_warmup_prefix_for_llm(monkeypatch) -> None:
    """`fz mine team` 必须给 prepare_mining_daily 传更长的预热前缀 lookback_days。

    与 agent 单路径同理：structured LLM 爱提 250/252 日长窗因子（required_lookback 270-315），
    用 search_space_max_lookback（=180）会把它们（正确地）判欠预热、永远评估不到。
    """
    import polars as pl

    from factorzen.cli import main as cli
    from factorzen.config.constants import AGENT_WARMUP_LOOKBACK
    from factorzen.discovery.search.random_search import search_space_max_lookback

    fake_daily = pl.DataFrame({"ts_code": ["600000.SH"]})
    captured: dict[str, object] = {}

    def fake_prepare(start, end, universe=None, lookback_days=None, **kw):
        captured["lookback_days"] = lookback_days
        return fake_daily

    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", fake_prepare)
    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine_team.run_team_mine",
        lambda daily, **kw: {"n_candidates": 0, "n_trials": 0, "run_dir": "x"},
    )

    rc = cli.main(["mine", "team", "--start", "20220101", "--end", "20231231",
                   "--index-path", "workspace/mine_team/custom_index.jsonl"])
    assert rc == 0
    assert captured["lookback_days"] == AGENT_WARMUP_LOOKBACK
    assert search_space_max_lookback() < AGENT_WARMUP_LOOKBACK


def test_cmd_mine_team_passes_universe_to_prepare(monkeypatch) -> None:
    """`fz mine team --universe` 应把 universe 透传给 prepare_mining_daily。"""
    import polars as pl

    from factorzen.cli import main as cli

    fake_daily = pl.DataFrame({"ts_code": ["000002.SZ"]})

    prep_calls: list[tuple] = []

    def fake_prepare(start, end, universe=None, lookback_days=None, **kw):
        prep_calls.append((start, end, universe))
        return fake_daily

    captured: dict[str, object] = {}

    def fake_run_team_mine(daily, *, n_rounds, seed, top_k, index_path, **_):
        captured["daily"] = daily
        return {"n_candidates": 0, "n_trials": 0, "run_dir": "workspace/mine_team/x"}

    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", fake_prepare)
    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine_team.run_team_mine", fake_run_team_mine
    )

    rc = cli.main(
        [
            "mine",
            "team",
            "--start",
            "20220101",
            "--end",
            "20231231",
            "--universe",
            "csi300",
        ]
    )

    assert rc == 0
    assert prep_calls == [("20220101", "20231231", "csi300")]
    assert captured["daily"] is fake_daily

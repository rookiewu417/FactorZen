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

    fetch_calls: list[tuple[str, str]] = []

    def fake_fetch_daily(start, end):
        fetch_calls.append((start, end))
        return fake_daily

    run_calls: list[dict[str, object]] = []

    def fake_run_team_mine(daily, *, n_rounds, seed, top_k, index_path):
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

    monkeypatch.setattr("factorzen.core.loader.fetch_daily", fake_fetch_daily)
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
    assert fetch_calls == [("20220101", "20231231")]
    assert len(run_calls) == 1
    call = run_calls[0]
    assert call["daily"] is fake_daily  # 未传 --universe，daily 原样转发、不过滤
    assert call["n_rounds"] == 2
    assert call["seed"] == 1
    assert call["top_k"] == 6
    assert call["index_path"] == "workspace/mine_team/custom_index.jsonl"

    out = capsys.readouterr().out
    assert out.splitlines()[0] == (
        "[mine-team] 候选 6 个 / N=50 → workspace/mine_team/team_1_2r"
    )


def test_cmd_mine_team_filters_daily_by_universe(monkeypatch) -> None:
    """`fz mine team --universe` 应先 get_universe 再按 ts_code 过滤 daily。"""
    import polars as pl

    from factorzen.cli import main as cli

    fake_daily = pl.DataFrame({"ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"]})

    def fake_fetch_daily(start, end):
        return fake_daily

    universe_calls: list[tuple[str, str]] = []

    def fake_get_universe(date_str, universe_name):
        universe_calls.append((date_str, universe_name))
        return pl.DataFrame({"ts_code": ["000002.SZ"]})

    captured: dict[str, object] = {}

    def fake_run_team_mine(daily, *, n_rounds, seed, top_k, index_path):
        captured["daily"] = daily
        return {"n_candidates": 0, "n_trials": 0, "run_dir": "workspace/mine_team/x"}

    monkeypatch.setattr("factorzen.core.loader.fetch_daily", fake_fetch_daily)
    monkeypatch.setattr("factorzen.core.universe.get_universe", fake_get_universe)
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
    assert universe_calls == [("20231231", "csi300")]
    assert captured["daily"]["ts_code"].to_list() == ["000002.SZ"]

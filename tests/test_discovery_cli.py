from __future__ import annotations


def test_parser_has_mine_search():
    from factorzen.cli.main import build_parser
    parser = build_parser()
    args = parser.parse_args(["mine", "search", "--start", "20240101", "--end", "20240601"])
    assert args.command == "mine"
    assert args.mine_command == "search"
    assert args.start == "20240101"
    assert callable(args.func)


def test_parser_has_mine_leaderboard():
    from factorzen.cli.main import build_parser
    parser = build_parser()
    args = parser.parse_args(["mine", "leaderboard", "some/dir"])
    assert args.mine_command == "leaderboard"
    assert args.session_dir == "some/dir"


def test_leaderboard_prints_csv(tmp_path, capsys):
    import argparse

    from factorzen.cli.main import _cmd_mine_leaderboard
    (tmp_path / "candidates.csv").write_text("rank,expression\n1,rank(close)\n")
    rc = _cmd_mine_leaderboard(argparse.Namespace(session_dir=str(tmp_path)))
    assert rc == 0
    assert "rank(close)" in capsys.readouterr().out


def test_cmd_mine_search_forwards_args_to_run_mine(monkeypatch, capsys, tmp_path):
    """`fz mine search` 应把 CLI 参数原样转发给 factor_mine.run_mine。"""
    from factorzen.cli import main as cli

    captured: dict[str, object] = {}

    def fake_run_mine(*, start, end, universe, n_trials, top_k, seed, method, workers=1):
        captured.update(
            start=start,
            end=end,
            universe=universe,
            n_trials=n_trials,
            top_k=top_k,
            seed=seed,
            method=method,
            workers=workers,
        )
        return {"session_dir": str(tmp_path / "session-1"), "candidates": [1, 2, 3]}

    monkeypatch.setattr("factorzen.pipelines.factor_mine.run_mine", fake_run_mine)

    rc = cli.main(
        [
            "mine",
            "search",
            "--start",
            "20230101",
            "--end",
            "20230601",
            "--universe",
            "csi500",
            "--method",
            "genetic",
            "--trials",
            "50",
            "--top-k",
            "5",
            "--seed",
            "7",
        ]
    )

    assert rc == 0
    assert captured == {
        "start": "20230101",
        "end": "20230601",
        "universe": "csi500",
        "n_trials": 50,
        "top_k": 5,
        "seed": 7,
        "method": "genetic",
        "workers": 1,
    }

    sd = str(tmp_path / "session-1")
    out = capsys.readouterr().out
    assert out.splitlines()[0] == f"[mine] 完成：3 个候选 → {sd}"


def test_cmd_mine_export_alpha_forwards_args_and_prints_summary(monkeypatch, tmp_path, capsys):
    """`fz mine export-alpha` 应转发 session/rank/date/universe/lookback/out，并落地产物。"""
    from pathlib import Path

    import polars as pl

    from factorzen.cli import main as cli

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    out_path = tmp_path / "alpha.parquet"

    read_calls: list[tuple[str, int]] = []

    def fake_read_candidate_expression(session, rank):
        read_calls.append((session, rank))
        return "rank(close_adj)"

    universe_calls: list[tuple[str, str]] = []

    def fake_get_universe(date_str, universe_name):
        universe_calls.append((date_str, universe_name))
        return pl.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"]})

    export_calls: list[tuple[str, object, str, str]] = []

    def fake_export_alpha_cross_section(expression, ctx, date, out_path_arg):
        export_calls.append((expression, ctx, date, out_path_arg))
        df = pl.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"], "alpha": [0.1, -0.2]})
        out = Path(out_path_arg)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out)
        return out

    monkeypatch.setattr(
        "factorzen.discovery.export.read_candidate_expression",
        fake_read_candidate_expression,
    )
    monkeypatch.setattr("factorzen.core.universe.get_universe", fake_get_universe)
    monkeypatch.setattr(
        "factorzen.discovery.export.export_alpha_cross_section",
        fake_export_alpha_cross_section,
    )

    rc = cli.main(
        [
            "mine",
            "export-alpha",
            "--session",
            str(session_dir),
            "--rank",
            "2",
            "--date",
            "20230615",
            "--universe",
            "csi300",
            "--lookback",
            "30",
            "--out",
            str(out_path),
        ]
    )

    assert rc == 0
    assert read_calls == [(str(session_dir), 2)]
    assert universe_calls == [("20230615", "csi300")]
    assert len(export_calls) == 1
    expr, ctx, date, out_arg = export_calls[0]
    assert expr == "rank(close_adj)"
    assert date == "20230615"
    assert out_arg == str(out_path)
    # ctx 为真实 FactorDataContext（构造不触发 I/O），校验 CLI 参数被正确转发进去
    assert ctx.start == "20230615"
    assert ctx.end == "20230615"
    assert ctx.lookback_days == 30
    assert ctx.universe == ["000001.SZ", "000002.SZ"]
    assert ctx.required_data == ["daily", "daily_basic"]

    out_text = capsys.readouterr().out
    assert out_text.splitlines()[0] == (
        f"[mine] export-alpha: rank=2 expr='rank(close_adj)' date=20230615 "
        f"→ {out_path} (2 只股票)"
    )

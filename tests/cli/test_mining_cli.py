"""test_agent_cli.py：Parser smoke tests for `fz mine agent` CLI subcommand.
test_team_cli.py：Parser smoke tests for `fz mine team` CLI subcommand.
test_discovery_cli.py：fz mine search/leaderboard/export-alpha CLI parser 与 leaderboard 输出冒烟。
"""


from __future__ import annotations

# ==== 来自 test_agent_cli.py ====
import pytest


def test_parser_mine_agent_team_suite():
    """test_parser_has_mine_agent；test_parser_mine_agent_defaults；test_parser_has_mine_team；test_parser_mine_team_index_default"""
    # -- 原 test_parser_has_mine_agent --
    def _section_0_test_parser_has_mine_agent():
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

    _section_0_test_parser_has_mine_agent()

    # -- 原 test_parser_mine_agent_defaults --
    def _section_1_test_parser_mine_agent_defaults():
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

    _section_1_test_parser_mine_agent_defaults()

    # -- 原 test_parser_has_mine_team --
    def _section_2_test_parser_has_mine_team():
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

    _section_2_test_parser_has_mine_team()

    # -- 原 test_parser_mine_team_index_default --
    def _section_3_test_parser_mine_team_index_default():
        from factorzen.cli.main import build_parser

        p = build_parser()
        args = p.parse_args(
            ["mine", "team", "--start", "20220101", "--end", "20231231"]
        )
        assert args.index_path.endswith(".jsonl")  # 长期记忆默认路径

    _section_3_test_parser_mine_team_index_default()


def test_cmd_mine_agent_wiring_suite(capsys):
    """`fz mine agent`（无 --universe）应原样转发 daily + CLI 参数给 run_agent_mine。；`fz mine agent` 必须把挖掘窗口 --start 作为 eval_start 透传给 run_agent_mine。；`fz mine agent` 必须给 prepare_mining_daily 传更长的预热前缀 lookback_days。；`fz mine agent --universe` 应把 universe 透传给 prepare_mining_daily（其内部经"""
    # -- 原 test_cmd_mine_agent_forwards_args_to_run_agent_mine --
    def _section_0_test_cmd_mine_agent_forwards_args_to_run_agent_mine(mp, capsys):
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

        mp.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", fake_prepare)
        mp.setattr(
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

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_cmd_mine_agent_forwards_args_to_run_agent_mine(mp, capsys)

    # -- 原 test_cmd_mine_agent_forwards_eval_start --
    def _section_1_test_cmd_mine_agent_forwards_eval_start(mp):
        import polars as pl

        from factorzen.cli import main as cli

        fake_daily = pl.DataFrame({"ts_code": ["000001.SZ"]})
        mp.setattr(
            "factorzen.pipelines.factor_mine.prepare_mining_daily",
            lambda start, end, universe=None, lookback_days=None, **kw: fake_daily,
        )
        captured: dict[str, object] = {}

        def fake_run_agent_mine(daily, *, n_rounds, seed, top_k, human_review,
                                eval_start=None, **_):
            captured["eval_start"] = eval_start
            return {"n_candidates": 0, "n_trials": 0, "run_dir": "x"}

        mp.setattr(
            "factorzen.pipelines.factor_mine_agent.run_agent_mine", fake_run_agent_mine
        )

        rc = cli.main(["mine", "agent", "--start", "20220101", "--end", "20231231"])
        assert rc == 0
        assert captured["eval_start"] == "20220101"

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_cmd_mine_agent_forwards_eval_start(mp)

    # -- 原 test_cmd_mine_agent_provisions_longer_warmup_prefix_for_llm --
    def _section_2_test_cmd_mine_agent_provisions_longer_warmup_prefix_for_llm(mp):
        import polars as pl

        from factorzen.cli import main as cli
        from factorzen.config.constants import AGENT_WARMUP_LOOKBACK
        from factorzen.discovery.search.random_search import search_space_max_lookback

        fake_daily = pl.DataFrame({"ts_code": ["000001.SZ"]})
        captured: dict[str, object] = {}

        def fake_prepare(start, end, universe=None, lookback_days=None, **kw):
            captured["lookback_days"] = lookback_days
            return fake_daily

        mp.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", fake_prepare)
        mp.setattr(
            "factorzen.pipelines.factor_mine_agent.run_agent_mine",
            lambda daily, **kw: {"n_candidates": 0, "n_trials": 0, "run_dir": "x"},
        )

        rc = cli.main(["mine", "agent", "--start", "20220101", "--end", "20231231"])
        assert rc == 0
        assert captured["lookback_days"] == AGENT_WARMUP_LOOKBACK
        assert search_space_max_lookback() < AGENT_WARMUP_LOOKBACK

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_cmd_mine_agent_provisions_longer_warmup_prefix_for_llm(mp)

    # -- 原 test_cmd_mine_agent_passes_universe_to_prepare --
    def _section_3_test_cmd_mine_agent_passes_universe_to_prepare(mp):
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

        mp.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", fake_prepare)
        mp.setattr(
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

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_cmd_mine_agent_passes_universe_to_prepare(mp)


# ==== 来自 test_team_cli.py ====


def test_cmd_mine_team_wiring_suite(capsys):
    """`fz mine team`（无 --universe）应原样转发 daily + CLI 参数给 run_team_mine。；`fz mine team` 必须把挖掘窗口 --start 作为 eval_start 透传给 run_team_mine。；`fz mine team` 必须给 prepare_mining_daily 传更长的预热前缀 lookback_days。；`fz mine team --universe` 应把 universe 透传给 prepare_mining_daily。"""
    # -- 原 test_cmd_mine_team_forwards_args_to_run_team_mine --
    def _section_0_test_cmd_mine_team_forwards_args_to_run_team_mine(mp, capsys):
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

        mp.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", fake_prepare)
        mp.setattr(
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

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_cmd_mine_team_forwards_args_to_run_team_mine(mp, capsys)

    # -- 原 test_cmd_mine_team_forwards_eval_start --
    def _section_1_test_cmd_mine_team_forwards_eval_start(mp):
        import polars as pl

        from factorzen.cli import main as cli

        fake_daily = pl.DataFrame({"ts_code": ["600000.SH"]})
        mp.setattr(
            "factorzen.pipelines.factor_mine.prepare_mining_daily",
            lambda start, end, universe=None, lookback_days=None, **kw: fake_daily,
        )
        captured: dict[str, object] = {}

        def fake_run_team_mine(daily, *, n_rounds, seed, top_k, index_path,
                               eval_start=None, **_):
            captured["eval_start"] = eval_start
            return {"n_candidates": 0, "n_trials": 0, "run_dir": "x"}

        mp.setattr(
            "factorzen.pipelines.factor_mine_team.run_team_mine", fake_run_team_mine
        )

        rc = cli.main(
            ["mine", "team", "--start", "20220101", "--end", "20231231",
             "--index-path", "workspace/mine_team/custom_index.jsonl"]
        )
        assert rc == 0
        assert captured["eval_start"] == "20220101"

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_cmd_mine_team_forwards_eval_start(mp)

    # -- 原 test_cmd_mine_team_provisions_longer_warmup_prefix_for_llm --
    def _section_2_test_cmd_mine_team_provisions_longer_warmup_prefix_for_llm(mp):
        import polars as pl

        from factorzen.cli import main as cli
        from factorzen.config.constants import AGENT_WARMUP_LOOKBACK
        from factorzen.discovery.search.random_search import search_space_max_lookback

        fake_daily = pl.DataFrame({"ts_code": ["600000.SH"]})
        captured: dict[str, object] = {}

        def fake_prepare(start, end, universe=None, lookback_days=None, **kw):
            captured["lookback_days"] = lookback_days
            return fake_daily

        mp.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", fake_prepare)
        mp.setattr(
            "factorzen.pipelines.factor_mine_team.run_team_mine",
            lambda daily, **kw: {"n_candidates": 0, "n_trials": 0, "run_dir": "x"},
        )

        rc = cli.main(["mine", "team", "--start", "20220101", "--end", "20231231",
                       "--index-path", "workspace/mine_team/custom_index.jsonl"])
        assert rc == 0
        assert captured["lookback_days"] == AGENT_WARMUP_LOOKBACK
        assert search_space_max_lookback() < AGENT_WARMUP_LOOKBACK

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_cmd_mine_team_provisions_longer_warmup_prefix_for_llm(mp)

    # -- 原 test_cmd_mine_team_passes_universe_to_prepare --
    def _section_3_test_cmd_mine_team_passes_universe_to_prepare(mp):
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

        mp.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", fake_prepare)
        mp.setattr(
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

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_cmd_mine_team_passes_universe_to_prepare(mp)


# ==== 来自 test_discovery_cli.py ====


def test_mine_leaderboard_suite(tmp_path, capsys):
    """test_parser_has_mine_leaderboard；test_leaderboard_prints_csv"""
    # -- 原 test_parser_has_mine_leaderboard --
    def _section_0_test_parser_has_mine_leaderboard():
        from factorzen.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["mine", "leaderboard", "some/dir"])
        assert args.mine_command == "leaderboard"
        assert args.session_dir == "some/dir"

    _section_0_test_parser_has_mine_leaderboard()

    # -- 原 test_leaderboard_prints_csv --
    def _section_1_test_leaderboard_prints_csv(tmp_path, capsys):
        import argparse

        from factorzen.cli.main import _cmd_mine_leaderboard
        (tmp_path / "candidates.csv").write_text("rank,expression\n1,rank(close)\n")
        rc = _cmd_mine_leaderboard(argparse.Namespace(session_dir=str(tmp_path)))
        assert rc == 0
        assert "rank(close)" in capsys.readouterr().out

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_leaderboard_prints_csv(_tp1, capsys)


def test_cmd_mine_search_forwards_args_to_run_mine(monkeypatch, capsys, tmp_path):
    """`fz mine search` 应把 CLI 参数原样转发给 factor_mine.run_mine。"""
    from factorzen.cli import main as cli

    captured: dict[str, object] = {}

    def fake_run_mine(*, start, end, universe, n_trials, top_k, seed, method, workers=1,
                      holdout_ratio=0.2, train_ratio=0.7, decorr_threshold=0.7,
                      min_n_train=5, dsr_alpha=0.05, update_library=True,
                      library_orthogonal=True, objective="residual",
                      intraday=False, intraday_freq="5min",
                      intraday_expr_leaves=None,
                      exec_lag=0, exec_price_col=None):
        captured.update(
            start=start,
            end=end,
            universe=universe,
            n_trials=n_trials,
            top_k=top_k,
            seed=seed,
            method=method,
            workers=workers,
            holdout_ratio=holdout_ratio,
            train_ratio=train_ratio,
            decorr_threshold=decorr_threshold,
            min_n_train=min_n_train,
            dsr_alpha=dsr_alpha,
            update_library=update_library,
            library_orthogonal=library_orthogonal,
            objective=objective,
            intraday=intraday,
            intraday_freq=intraday_freq,
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
            "--holdout-ratio",
            "0.25",
            "--train-ratio",
            "0.6",
            "--decorr-threshold",
            "0.8",
            "--min-n-train",
            "8",
            "--dsr-alpha",
            "0.1",
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
        "holdout_ratio": 0.25,
        "train_ratio": 0.6,
        "decorr_threshold": 0.8,
        "min_n_train": 8,
        "dsr_alpha": 0.1,
        "update_library": True,          # 默认开（未传 --no-library）
        "library_orthogonal": True,      # 默认开（未传 --no-library-orthogonal）
        "objective": "residual",         # 默认残差目标（未传 --objective）
        "intraday": False,               # 默认关（未传 --intraday-leaves）
        "intraday_freq": "5min",         # 默认频率（未传 --intraday-freq）
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

    def fake_read_candidate_expression(session, rank, require_passed=False):
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

"""CLI ``factor-library lift-test`` 批处理路径：候选面板与挖掘装配同源 + top_m 全测。

事故回归（20260714_074426_team_701_15r）：
- session 内钩子 ``top_m=None`` 全测；CLI 静默 top_m=10 只测 10/38；
- 事件叶子（top_list_* / margin_* / holder_*）若装配漂移 → 候选近乎全空 →
  ``build_panel: 仅 2% 行因子齐全`` → lift 噪声。

本文件 mock 离线，CI 无 token 可跑。从 CLI parser 最外层出发（不用 inspect.signature）。
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl


def _trading_dates(n: int, start: date = date(2024, 1, 2)) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _sparse_event_daily(n_days: int = 40, n_stocks: int = 8) -> pl.DataFrame:
    """合成日线 + 稀疏龙虎榜叶子（条件 fill-0 语义的已 attach 结果）。

    - 大部分行 top_list_flag=0（已知日未上榜）
    - 少数行 =1（事件）
    - 首日 null（lag 后无前值）——与 attach_flows 真实语义一致
    """
    dates = _trading_dates(n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for di, d in enumerate(dates):
        for si, code in enumerate(codes):
            # 稀疏事件：约 1/8 股票 × 每隔 7 日有 flag=1
            is_event = (di > 0) and (di % 7 == 0) and (si % 4 == 0)
            if di == 0:
                flag, net = None, None
            else:
                flag = 1.0 if is_event else 0.0
                net = 0.15 if is_event else 0.0
            rows.append({
                "trade_date": d,
                "ts_code": code,
                "close": 10.0 + si * 0.1,
                "close_adj": 10.0 + si * 0.1,
                "open": 10.0, "high": 10.5, "low": 9.5,
                "open_adj": 10.0, "high_adj": 10.5, "low_adj": 9.5,
                "vol": 1e5, "amount": 1e5,
                "circ_mv": 1e6, "total_mv": 2e6,
                "turnover_rate": 1.0,
                "top_list_flag": flag,
                "top_list_net_buy": net,
                "margin_buy_ratio": 0.01 if di > 0 else None,
                "holder_num_chg": 0.0 if di > 0 else None,
            })
    return pl.DataFrame(rows)


def _write_lift_queue_session(tmp_path: Path, expressions: list[str]) -> Path:
    run_dir = tmp_path / "run_session"
    run_dir.mkdir()
    attempts = [
        {
            "expression": e,
            "reject_category": "lift_queue",
            "residual_ic_train": 0.01 - i * 0.0001,
            "ic_train": 0.001,
            "n_residual_holdout_days": 100,
        }
        for i, e in enumerate(expressions)
    ]
    (run_dir / "manifest.json").write_text(
        json.dumps({"attempts": attempts, "candidates": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_dir


def _mining_path_nonzero_rows(daily: pl.DataFrame, expr: str) -> int:
    """挖掘路径口径：preprocess → factor_df_from_prepped 非空行数。"""
    from factorzen.discovery.evaluation import _factor_df_from_prepped, _preprocess_daily
    from factorzen.discovery.expression import parse_expr

    prepped = _preprocess_daily(daily)
    node = parse_expr(expr)
    panel = _factor_df_from_prepped(node, prepped)
    return int(panel.height)


# ── a. 稀疏事件叶子：CLI 物化覆盖 = 挖掘装配物化覆盖 ─────────────────────────


def test_cli_lift_sparse_event_panel_matches_mining_path(tmp_path, monkeypatch, capsys):
    """从 parser 最外层进 lift-test；候选面板非空行数与挖掘路径一致。

    同时断言 CLI 调用的是 ``_prepare_agent_mining_data``（与 mine team/agent 同源，
    不许另起 loader）。
    """
    import factorzen.cli.main as cli_main
    import factorzen.discovery.lift_test as lt_mod
    from factorzen.cli.main import build_parser
    from factorzen.discovery.lift_test import _materializer_from_prepped

    expr = "mul(delay(top_list_flag, 1), max(top_list_net_buy, 0.0))"
    daily = _sparse_event_daily()
    n_mining = _mining_path_nonzero_rows(daily, expr)
    assert n_mining > 0, "合成帧应产生非空候选面板（否则测试本身无效）"

    run_dir = _write_lift_queue_session(tmp_path, [expr])
    prep_calls: list = []

    def tracking_prepare(args):
        prep_calls.append({
            "start": getattr(args, "start", None),
            "end": getattr(args, "end", None),
            "universe": getattr(args, "universe", None),
        })
        return daily, None, {"membership_mode": None}

    monkeypatch.setattr(cli_main, "_prepare_agent_mining_data", tracking_prepare)

    captured: dict = {}

    def capturing_run_lift_tests(gray, **kw):
        captured["n_gray"] = len(gray)
        captured["top_m"] = kw.get("top_m")
        ctx = kw.get("ctx")
        assert ctx is not None, "CLI 必须构造 LiftEvalContext"
        # 与 lift 内部默认 materializer 同路径：prepped 上物化
        mat = _materializer_from_prepped(ctx.prepped, ctx.leaf_map)
        panel = mat(expr)
        assert panel is not None and not panel.is_empty()
        captured["n_cli"] = int(panel.height)
        # 叶子列必须仍在 prepped 上（装配未丢事件叶子）
        for col in ("top_list_flag", "top_list_net_buy"):
            assert col in ctx.prepped.columns, f"prepped 缺叶子 {col}"
        return [{
            "expression": expr,
            "lift": 0.0,
            "baseline": 0.01,
            "passed": False,
            "candidate_rank_ic": 0.01,
            "n_input": len(gray),
            "n_selected": len(gray),
        }]

    from tests._cli_lift_mocks import patch_cli_lift_pre_gates
    patch_cli_lift_pre_gates(monkeypatch)
    monkeypatch.setattr(lt_mod, "run_lift_tests", capturing_run_lift_tests)

    # parser 最外层（禁止 inspect.signature）
    args = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--market", "ashare",
        "--start", "20240102",
        "--end", "20240301",
        "--universe", "csi300",
        "--dry-run",
        "--library-root", str(tmp_path / "lib"),
        "--top-m", "0",  # 全测逃生口（默认 20）
    ])
    assert args.func.__name__ == "_cmd_factor_library_lift_test"
    assert args.top_m == 0

    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert len(prep_calls) == 1, "必须经 _prepare_agent_mining_data 装配"
    assert captured.get("n_cli") == n_mining, (
        f"CLI 物化非空行 {captured.get('n_cli')} ≠ 挖掘路径 {n_mining}"
    )
    assert captured.get("top_m") is None


def test_cli_and_mine_team_share_prepare_fn(monkeypatch, tmp_path):
    """mine team 与 lift-test 共用同一 ``_prepare_agent_mining_data`` 绑定。

    从 parser 最外层分别进两条命令，断言命中同一 tracking 替身（同源、非复制第二份）。
    """
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser

    daily = _sparse_event_daily(n_days=5, n_stocks=2)
    hits: list[str] = []

    def tracking_prepare(args):
        hits.append(getattr(args, "factor_library_command", None)
                    or getattr(args, "mine_command", None)
                    or "unknown")
        return daily, None, {}

    monkeypatch.setattr(cli_main, "_prepare_agent_mining_data", tracking_prepare)

    # lift-test
    run_dir = _write_lift_queue_session(tmp_path, ["rank(close)"])
    import factorzen.discovery.lift_test as lt_mod
    from tests._cli_lift_mocks import patch_cli_lift_pre_gates

    patch_cli_lift_pre_gates(monkeypatch)
    monkeypatch.setattr(
        lt_mod, "run_lift_tests",
        lambda gray, **kw: [{
            "expression": "rank(close)", "lift": None, "baseline": None,
            "passed": False,
        }],
    )
    args_lt = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--start", "20240102", "--end", "20240301",
        "--library-root", str(tmp_path / "lib"),
        "--dry-run",
    ])
    assert cli_main._cmd_factor_library_lift_test(args_lt) == 0

    # mine team（短路 run_team_mine，只验证装配入口）
    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine_team.run_team_mine",
        lambda *a, **k: {"n_candidates": 0, "n_trials": 0, "run_dir": str(tmp_path)},
    )
    args_mt = build_parser().parse_args([
        "mine", "team", "--start", "20240102", "--end", "20240301",
    ])
    assert cli_main._cmd_mine_team(args_mt) == 0

    assert len(hits) == 2, f"两条路径都应调用同源 prepare，got {hits}"


# ── b. top_m 默认 20 截断 + --top-m 0 全测 + 显式截断告警 ────────────────────


def test_cli_lift_top_m_default_truncates_to_20(tmp_path, monkeypatch, capsys):
    """默认 top_m=20：38 候选截到 20 进 run_lift_tests，stderr 截断告警。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.lift_test as lt_mod
    from factorzen.cli.main import build_parser
    from tests._cli_lift_mocks import patch_cli_lift_pre_gates

    # 唯一 expression 串（去重后仍 38 个）；mock lift 不物化
    exprs = [f"add(rank(close), {i}.0)" for i in range(38)]
    run_dir = _write_lift_queue_session(tmp_path, exprs)

    monkeypatch.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda args: (_sparse_event_daily(n_days=5, n_stocks=2), None, {}),
    )
    patch_cli_lift_pre_gates(monkeypatch)
    seen: dict = {}

    def fake_lift(gray, **kw):
        seen["n"] = len(gray)
        seen["top_m"] = kw.get("top_m")
        return [
            {
                "expression": g["expression"], "lift": -0.0001, "baseline": 0.01,
                "passed": False, "n_input": len(gray),
                "n_selected": len(gray),
            }
            for g in gray
        ]

    monkeypatch.setattr(lt_mod, "run_lift_tests", fake_lift)

    args = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--start", "20240102", "--end", "20240301",
        "--library-root", str(tmp_path / "lib"),
        "--dry-run",
    ])
    assert args.top_m == 20
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert seen["top_m"] is None  # CLI 已截断
    assert seen["n"] == 20

    err = capsys.readouterr().err
    assert "--top-m=20" in err
    assert "truncated_from=38" in err


def test_cli_lift_top_m_explicit_truncates_with_warning(tmp_path, monkeypatch, capsys):
    """显式 ``--top-m 10``：截断并打印告警行（no silent caps）。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.lift_test as lt_mod
    from factorzen.cli.main import build_parser
    from tests._cli_lift_mocks import patch_cli_lift_pre_gates

    exprs = [f"add(rank(close), {i}.0)" for i in range(38)]
    run_dir = _write_lift_queue_session(tmp_path, exprs)

    monkeypatch.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda args: (_sparse_event_daily(n_days=5, n_stocks=2), None, {}),
    )
    patch_cli_lift_pre_gates(monkeypatch)
    seen: dict = {}

    def fake_lift(gray, **kw):
        seen["top_m"] = kw.get("top_m")
        seen["n_input"] = len(gray)
        return [
            {
                "expression": g["expression"], "lift": -0.0001, "baseline": 0.01,
                "passed": False, "n_input": len(gray), "n_selected": len(gray),
            }
            for g in gray
        ]

    monkeypatch.setattr(lt_mod, "run_lift_tests", fake_lift)

    args = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--start", "20240102", "--end", "20240301",
        "--library-root", str(tmp_path / "lib"),
        "--top-m", "10",
        "--dry-run",
    ])
    assert args.top_m == 10
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert seen["top_m"] is None  # CLI 已截断，run_lift_tests 收 kept
    assert seen["n_input"] == 10

    err = capsys.readouterr().err
    assert "--top-m=10" in err
    assert "截断" in err
    assert "truncated_from=38" in err


def test_cli_lift_parser_top_m_default_is_20():
    """parser 契约：不传 --top-m → args.top_m == 20；--top-m 0 为全测逃生口。"""
    from factorzen.cli.main import build_parser

    args = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", "workspace/x",
        "--start", "20200101", "--end", "20201231",
    ])
    assert args.top_m == 20
    args0 = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", "workspace/x",
        "--start", "20200101", "--end", "20201231",
        "--top-m", "0",
    ])
    assert args0.top_m == 0

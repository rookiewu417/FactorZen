"""合并自: test_replay_smoke.py, test_daily_driver.py, test_live_cli_smoke.py, test_pit_sizing_and_st.py
目标: test_live_drivers.py

--- 来源 test_replay_smoke.py ---
(无原 docstring)

--- 来源 test_daily_driver.py ---
(无原 docstring)

--- 来源 test_live_cli_smoke.py ---
fz live init/step/status/report：CLI 路由 + 离线端到端 smoke。

- test_init_step_report_pipeline：底层 driver/attribution 直接跑通(init→多日
  step→report)，断言 attribution.json 落盘 + 关键字段。
- test_live_cli_parser_routes_new_subcommands：确认新增 4 子命令能被
  build_parser() 解析并挂到正确的 func，且不干扰既有 replay/其他顶层命令。

--- 来源 test_pit_sizing_and_st.py ---
PIT 修复：#2 定量用 pre_close（非执行日 close）；#6 逐日 ST 收窄涨跌停。

旧行为：
- drivers 用 m["close"] 作 ref_price → 前视（执行日收盘价决策时未知）
- PaperBroker 不传 is_st → ST 股用主板 9.8% 宽阈值
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl

from factorzen.execution.attribution import build_attribution_report
from factorzen.execution.broker import Order, round_lot
from factorzen.execution.brokers.paper import PaperBroker
from factorzen.execution.drivers import run_daily_step, run_replay
from factorzen.execution.store import SessionStore


# ==== 来自 test_replay_smoke.py ====
def _write_portfolio_run(dir_: Path, sig: date, code: str, w: float) -> str:
    dir_.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"ts_code": [code], "target_weight": [w]}).write_parquet(dir_ / "weights.parquet")
    (dir_ / "manifest.json").write_text(json.dumps(
        {"signal_date": sig.isoformat(), "status": "optimal"}))
    return str(dir_)

def _daily__replay_smoke(codes, dates):
    rows = []
    for d in dates:
        for c in codes:
            rows.append({"trade_date": d, "ts_code": c, "open": 10.0,
                         "pre_close": 10.0, "close": 10.0, "vol": 1e6})
    return pl.DataFrame(rows)

def test_replay_binds_adv_capacity_constraint(tmp_path: Path):
    """容量约束须真的在 replay 路径生效：day1 提供成交额历史用于算 day2 的
    trailing ADV（_precompute_adv_20d_by_date 对当日 shift(1)，故 day1 自身无
    ADV，day2 才有）。day2 信号从 0 → 满仓 1.0，若 ADV 未接入，delta=1.0 不会
    被 capacity 截断，买入股数将接近 100000 股（满仓）；若 ADV 正确接入，
    max_participation_rate=0.05 * adv=200000 / portfolio_value=1e6 = 0.01，
    截断后买入应恰好是 1000 股（独立手算 ground truth，非恒真）。
    """
    code = "A.SZ"
    dates = [date(2026, 1, 5), date(2026, 1, 6)]
    rows = [
        {"trade_date": d, "ts_code": code, "open": 10.0, "pre_close": 10.0,
         "close": 10.0, "vol": 1_000_000.0, "amount": 200_000.0}
        for d in dates
    ]
    daily = pl.DataFrame(rows)
    # day1 发满仓信号，次日 day2 执行(s<d)；day1 参与行情驱动为 day2 的 ADV 提供
    # 历史 amount。day1 自身无更早信号→不下单。
    rd = _write_portfolio_run(tmp_path / "pf", dates[0], code, 1.0)
    run_replay(
        session_dir=tmp_path / "sess", portfolio_run_dirs=[rd],
        daily=daily, initial_cash=1_000_000.0,
        from_date=dates[0], to_date=dates[-1], seed=0,
    )
    ledger = pl.read_parquet(tmp_path / "sess" / "ledger.parquet")
    day2 = ledger.filter(pl.col("as_of_date") == dates[1].isoformat())
    assert day2.height == 1
    payload = json.loads(day2["payload"][0])
    fills = payload["fills"]
    assert len(fills) == 1, fills
    assert fills[0]["filled_volume"] == 1000, fills  # ground truth: 手算见上

def test_replay_is_idempotent_on_rerun(tmp_path: Path):
    dates = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    daily = _daily__replay_smoke(["A.SZ"], dates)
    rd = _write_portfolio_run(tmp_path / "pf", date(2026, 1, 5), "A.SZ", 0.5)
    kwargs = dict(
        session_dir=tmp_path / "sess", portfolio_run_dirs=[rd],
        daily=daily, initial_cash=1_000_000.0,
        from_date=dates[0], to_date=dates[-1], seed=0,
    )
    out1 = run_replay(**kwargs)
    nav1 = pl.read_parquet(tmp_path / "sess" / "nav.parquet")
    out2 = run_replay(**kwargs)  # 重跑同一 session_dir
    nav2 = pl.read_parquet(tmp_path / "sess" / "nav.parquet")
    assert nav2.height == nav1.height  # 不翻倍/不追加重复日期行
    assert out2["n_steps"] == 0  # 第二次全部日期已记录，跳过不重复下单
    assert out1["n_steps"] >= 1

# ==== 来自 test_daily_driver.py ====
def _pf__daily_driver(dir_, sig, code, w):
    dir_.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"ts_code": [code], "target_weight": [w]}).write_parquet(dir_ / "weights.parquet")
    (dir_ / "manifest.json").write_text(
        json.dumps({"signal_date": sig.isoformat(), "status": "optimal"})
    )
    return str(dir_)

def _daily__daily_driver(dates, code):
    return pl.DataFrame(
        [
            {
                "trade_date": d,
                "ts_code": code,
                "open": 10.0,
                "pre_close": 10.0,
                "close": 10.0,
                "vol": 1e8,
                "amount": 1e9,
            }
            for d in dates
        ]
    )

def test_daily_step_idempotent_and_resumes(tmp_path: Path):
    d1, d2 = date(2026, 1, 5), date(2026, 1, 6)
    daily = _daily__daily_driver([d1, d2], "A.SZ")
    rd = _pf__daily_driver(tmp_path / "pf", date(2026, 1, 2), "A.SZ", 0.5)
    cfg = {"initial_cash": 1_000_000.0, "slippage_bps": 0.0}
    s = SessionStore(tmp_path / "sess")
    s.init({"broker": "paper", **cfg})
    r1 = run_daily_step(tmp_path / "sess", d1, [rd], daily, config=cfg)
    assert not r1["skipped"] and r1["n_fills"] >= 1
    # 幂等：同日再跑跳过
    r1b = run_daily_step(tmp_path / "sess", d1, [rd], daily, config=cfg)
    assert r1b["skipped"]
    # resume：新进程语义——第二天 step 从磁盘 load_state 续跑
    r2 = run_daily_step(tmp_path / "sess", d2, [rd], daily, config=cfg)
    assert not r2["skipped"]
    assert SessionStore(tmp_path / "sess").nav_frame().height == 2  # 只两行(d1,d2)

def test_daily_step_resume_nav_matches_single_shot_replay(tmp_path: Path) -> None:
    # 回归 Fix5：spec 承诺"daily 驱动跨两日 step 的 NAV == 连续两日 run_replay"。
    # 逐日"每天起一个新进程 + load_state 续跑"的 run_daily_step 序列，与一次性
    # run_replay 跑同一段历史，nav 序列必须数值相等（同 weights/同 daily/同
    # broker 逻辑，只是驱动方式不同：分次落盘续跑 vs 单进程内存循环）。
    d1, d2, d3 = date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)
    daily = _daily__daily_driver([d1, d2, d3], "A.SZ")
    rd = _pf__daily_driver(tmp_path / "pf", date(2026, 1, 2), "A.SZ", 0.5)
    cfg = {"initial_cash": 1_000_000.0, "slippage_bps": 0.0}

    sess_replay = tmp_path / "sess_replay"
    run_replay(
        session_dir=sess_replay,
        portfolio_run_dirs=[rd],
        daily=daily,
        initial_cash=cfg["initial_cash"],
        from_date=d1,
        to_date=d3,
        seed=0,
    )
    nav_replay = (
        SessionStore(sess_replay).nav_frame().sort("as_of_date")["nav_after"].to_list()
    )

    sess_daily = tmp_path / "sess_daily"
    SessionStore(sess_daily).init({"broker": "paper", **cfg})
    for d in (d1, d2, d3):
        # 每次调用视为独立进程：不复用 broker 实例，全靠 load_state 续跑。
        run_daily_step(sess_daily, d, [rd], daily, config=cfg)
    nav_daily = (
        SessionStore(sess_daily).nav_frame().sort("as_of_date")["nav_after"].to_list()
    )

    assert len(nav_replay) == len(nav_daily) == 3
    for replay_v, daily_v in zip(nav_replay, nav_daily, strict=True):
        assert abs(replay_v - daily_v) < 1e-6

def _daily_var(dates_prices, code):
    """带价格变化的 daily（区分「续跑重建持仓」vs「从空仓重来」的 nav）。"""
    return pl.DataFrame(
        [
            {"trade_date": d, "ts_code": code, "open": p, "pre_close": p,
             "close": p, "vol": 1e8, "amount": 1e9}
            for d, p in dates_prices
        ]
    )

def test_replay_resume_extends_window_matches_single_shot(tmp_path: Path) -> None:
    """扩窗/崩溃恢复：在已有部分 ledger 的 session 上再 run_replay（延长 --to），
    必须 load_state 重建 broker，否则续跑日从空仓 initial_cash 起、nav 错。
    扩窗 replay 的 nav 序列须与一次性 replay 相同。"""
    d1, d2, d3 = date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)
    daily = _daily_var([(d1, 10.0), (d2, 11.0), (d3, 12.0)], "A.SZ")  # 价格逐日上涨
    rd = _pf__daily_driver(tmp_path / "pf", date(2026, 1, 2), "A.SZ", 0.5)

    full = tmp_path / "full"
    run_replay(session_dir=full, portfolio_run_dirs=[rd], daily=daily,
               initial_cash=1_000_000.0, from_date=d1, to_date=d3, seed=0)
    nav_full = SessionStore(full).nav_frame().sort("as_of_date")["nav_after"].to_list()

    resume = tmp_path / "resume"
    run_replay(session_dir=resume, portfolio_run_dirs=[rd], daily=daily,
               initial_cash=1_000_000.0, from_date=d1, to_date=d2, seed=0)
    run_replay(session_dir=resume, portfolio_run_dirs=[rd], daily=daily,
               initial_cash=1_000_000.0, from_date=d1, to_date=d3, seed=0)  # 扩窗续跑
    nav_resume = SessionStore(resume).nav_frame().sort("as_of_date")["nav_after"].to_list()

    assert len(nav_full) == len(nav_resume) == 3
    for a, b in zip(nav_full, nav_resume, strict=True):
        assert abs(a - b) < 1e-6, f"扩窗 replay nav 须等于一次性 replay：{nav_full} vs {nav_resume}"

def test_replay_state_resumable_by_daily_step(tmp_path: Path) -> None:
    """run_replay 落的 state.json 须是可续跑态 {cash,pos,order_seq}，使后续
    run_daily_step 能 load_state 续跑，而非因显示视图 float(dict) 抛 TypeError。"""
    d1, d2 = date(2026, 1, 5), date(2026, 1, 6)
    daily = _daily_var([(d1, 10.0), (d2, 11.0)], "A.SZ")
    rd = _pf__daily_driver(tmp_path / "pf", date(2026, 1, 2), "A.SZ", 0.5)
    sess = tmp_path / "sess"
    run_replay(session_dir=sess, portfolio_run_dirs=[rd], daily=daily,
               initial_cash=1_000_000.0, from_date=d1, to_date=d1, seed=0)

    st = SessionStore(sess).load_state()
    assert {"cash", "pos", "order_seq"}.issubset(st), f"replay 应落可续跑态，实际 {set(st)}"

    # 后续 fz live step 续跑不应因显示视图 float(dict) 崩溃
    r = run_daily_step(sess, d2, [rd], daily, config={"initial_cash": 1_000_000.0})
    assert not r["skipped"] and r["nav_after"] is not None

def test_signal_executes_next_trading_day_not_same_day(tmp_path: Path) -> None:
    """E1：signal_date=组合数据截止日(用当日收盘算权重)，须在**次一交易日**执行，
    与 sim(trade_dates[i-1])对齐；同日执行=用当日收盘在当日开盘成交的未来函数。"""
    sig, d_next = date(2026, 1, 5), date(2026, 1, 6)
    daily = _daily__daily_driver([sig, d_next], "A.SZ")
    rd = _pf__daily_driver(tmp_path / "pf", sig, "A.SZ", 0.5)  # 信号日 = sig
    cfg = {"initial_cash": 1_000_000.0, "slippage_bps": 0.0}
    sess = tmp_path / "sess"
    SessionStore(sess).init({"broker": "paper", **cfg})

    # 信号日当天 step：不应成交（信号次日才生效）
    r_sig = run_daily_step(sess, sig, [rd], daily, config=cfg)
    assert r_sig["n_fills"] == 0, "信号当日不应成交（未来函数）"

    # 次一交易日 step：应成交
    r_next = run_daily_step(sess, d_next, [rd], daily, config=cfg)
    assert r_next["n_fills"] >= 1, "信号应于次一交易日执行"

# ==== 来自 test_live_cli_smoke.py ====
def test_init_step_report_pipeline(tmp_path: Path) -> None:
    dates = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    daily = pl.DataFrame(
        [
            {
                "trade_date": d,
                "ts_code": "A.SZ",
                "open": 10.1,
                "pre_close": 10.0,
                "close": 10.0,
                "vol": 1e8,
                "amount": 1e9,
            }
            for d in dates
        ]
    )
    pf = tmp_path / "pf"
    pf.mkdir()
    pl.DataFrame({"ts_code": ["A.SZ"], "target_weight": [0.5]}).write_parquet(
        pf / "weights.parquet"
    )
    # 信号早于首个执行日 → dates 三天都执行(s<d，次日执行)
    (pf / "manifest.json").write_text(
        json.dumps({"signal_date": date(2026, 1, 2).isoformat(), "status": "optimal"})
    )
    cfg = {"initial_cash": 1_000_000.0, "slippage_bps": 0.0}
    sess = tmp_path / "sess"
    SessionStore(sess).init({"broker": "paper", **cfg})
    for d in dates:
        run_daily_step(sess, d, [str(pf)], daily, config=cfg)
    rep = build_attribution_report(sess, [str(pf)], daily, initial_cash=1_000_000.0)
    assert (sess / "attribution.json").exists()
    assert rep["n_days"] == 3
    assert "cost_bps" in rep and "residual_bps" in rep


def test_live_cli_parser_routes_new_subcommands() -> None:
    from factorzen.cli.main import (
        _cmd_live_init,
        _cmd_live_report,
        _cmd_live_status,
        _cmd_live_step,
        build_parser,
    )

    parser = build_parser()

    init_args = parser.parse_args(
        ["live", "init", "--session-dir", "workspace/execution/sess1"]
    )
    assert init_args.func is _cmd_live_init
    assert init_args.session_dir == "workspace/execution/sess1"
    assert init_args.initial_cash == 1_000_000.0
    assert init_args.broker == "paper"

    step_args = parser.parse_args(
        [
            "live",
            "step",
            "--session-dir",
            "workspace/execution/sess1",
            "--date",
            "20260105",
            "--portfolio-run-dir",
            "workspace/portfolios/run1",
            "--start",
            "20251201",
            "--end",
            "20260105",
        ]
    )
    assert step_args.func is _cmd_live_step
    assert step_args.portfolio_run_dirs == ["workspace/portfolios/run1"]

    status_args = parser.parse_args(
        ["live", "status", "--session-dir", "workspace/execution/sess1"]
    )
    assert status_args.func is _cmd_live_status

    report_args = parser.parse_args(
        [
            "live",
            "report",
            "--session-dir",
            "workspace/execution/sess1",
            "--portfolio-run-dir",
            "workspace/portfolios/run1",
            "--start",
            "20251201",
            "--end",
            "20260105",
        ]
    )
    assert report_args.func is _cmd_live_report

    # replay(M1)与其余顶层命令不受影响，仍可正常解析。
    replay_args = parser.parse_args(
        [
            "live",
            "replay",
            "--session-dir",
            "workspace/execution/sess1",
            "--portfolio-run-dir",
            "workspace/portfolios/run1",
            "--start",
            "20251201",
            "--end",
            "20260105",
        ]
    )
    assert replay_args.broker == "paper"

    sim_show_args = parser.parse_args(["sim", "show", "--sim-dir", "workspace/sim/run1"])
    assert sim_show_args.sim_dir == "workspace/sim/run1"


def test_live_status_handles_resumable_state_shape(tmp_path: Path, capsys) -> None:
    # run_daily_step 落的是"可续跑态" broker.state() = {cash: float, pos, order_seq}。
    from factorzen.cli.main import _cmd_live_status, build_parser

    sess = tmp_path / "sess"
    cfg = {"initial_cash": 1_000_000.0}
    SessionStore(sess).init({"broker": "paper", **cfg})
    d0 = date(2026, 1, 5)
    daily = pl.DataFrame(
        [
            {
                "trade_date": d0,
                "ts_code": "A.SZ",
                "open": 10.0,
                "pre_close": 10.0,
                "close": 10.0,
                "vol": 1e8,
                "amount": 1e9,
            }
        ]
    )
    pf = tmp_path / "pf"
    pf.mkdir()
    pl.DataFrame({"ts_code": ["A.SZ"], "target_weight": [0.5]}).write_parquet(
        pf / "weights.parquet"
    )
    # 信号早于 d0 → 次日 d0 执行(s<d)，产生持仓
    (pf / "manifest.json").write_text(
        json.dumps({"signal_date": date(2026, 1, 2).isoformat(), "status": "optimal"})
    )
    run_daily_step(sess, d0, [str(pf)], daily, config=cfg)

    args = build_parser().parse_args(["live", "status", "--session-dir", str(sess)])
    rc = _cmd_live_status(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "持仓数=1" in out
    # 现金应是数字（可续跑态 cash 直接是 float），不能把 dict 原样打出来
    cash_field = out.split("现金=")[1].split(" ")[0]
    assert "{" not in cash_field
    float(cash_field)  # 不抛异常即说明是个可解析的数字


def test_live_status_handles_legacy_display_view_state(tmp_path: Path, capsys) -> None:
    # legacy 兼容：旧 session 的 state.json 可能是 step() 的"显示视图"
    # {positions: {...}, cash: {available,total_asset,market_value}}——旧版 run_replay
    # 曾落这种格式（现已改落可续跑态，见 test_replay_state_resumable_by_daily_step），
    # 但历史 session 仍需能读。_cmd_live_status 须解析它、不误报：cash 从 dict 取数值、
    # 持仓从 "positions" 键取（而非可续跑态的 "pos"）。直接构造显示视图以锁死该兼容分支。
    from factorzen.cli.main import _cmd_live_status, build_parser

    sess = tmp_path / "sess"
    SessionStore(sess).init({"broker": "paper", "initial_cash": 1_000_000.0})
    (sess / "state.json").write_text(
        json.dumps(
            {
                "positions": {
                    "A.SZ": {"ts_code": "A.SZ", "volume": 50000,
                             "can_use_volume": 50000, "avg_cost": 10.0}
                },
                "cash": {"available": 500000.0, "total_asset": 1000000.0,
                         "market_value": 500000.0},
            }
        )
    )
    pl.DataFrame(
        {"as_of_date": [date(2026, 1, 5).isoformat()], "nav_after": [1_000_000.0]}
    ).write_parquet(sess / "nav.parquet")

    args = build_parser().parse_args(["live", "status", "--session-dir", str(sess)])
    rc = _cmd_live_status(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "持仓数=1" in out  # 从 "positions" 键取，而非误报 0
    cash_field = out.split("现金=")[1].split(" ")[0]
    assert "{" not in cash_field  # 显示视图 cash 是 dict，应被解成数值而非原样打印
    float(cash_field)

# ==== 来自 test_pit_sizing_and_st.py ====
def _pf__pit_sizing_and_st(dir_: Path, sig: date, code: str, w: float) -> str:
    dir_.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"ts_code": [code], "target_weight": [w]}).write_parquet(dir_ / "weights.parquet")
    (dir_ / "manifest.json").write_text(
        json.dumps({"signal_date": sig.isoformat(), "status": "optimal"})
    )
    return str(dir_)


# ── #2 pre_close 定量 ──────────────────────────────────────────────


def test_daily_step_sizes_target_shares_with_pre_close_not_close(tmp_path: Path) -> None:
    """执行日 pre_close 与 close 明显不同时，目标股数须按 pre_close 算（非 close）。

    pre_close=10、close=13、w=0.5、nav=1e6：
      pre_close → round_lot(0.5*1e6/10)=50000
      close     → round_lot(0.5*1e6/13)=38400
    旧实现用 close → 红。
    """
    sig, exec_d = date(2026, 1, 5), date(2026, 1, 6)
    code = "A.SZ"
    pre_close, close_px, open_px = 10.0, 13.0, 10.0
    daily = pl.DataFrame(
        [
            {
                "trade_date": sig,
                "ts_code": code,
                "open": 10.0,
                "pre_close": 10.0,
                "close": 10.0,
                "vol": 1e8,
                "amount": 1e9,
            },
            {
                "trade_date": exec_d,
                "ts_code": code,
                "open": open_px,
                "pre_close": pre_close,
                "close": close_px,
                "vol": 1e8,
                "amount": 1e9,
            },
        ]
    )
    rd = _pf__pit_sizing_and_st(tmp_path / "pf", sig, code, 0.5)
    cfg = {"initial_cash": 1_000_000.0, "slippage_bps": 0.0}
    sess = tmp_path / "sess"
    SessionStore(sess).init({"broker": "paper", **cfg})

    r = run_daily_step(sess, exec_d, [rd], daily, config=cfg)
    assert not r["skipped"] and r["n_fills"] >= 1

    recs = SessionStore(sess).ledger_records()
    assert len(recs) == 1
    orders = recs[0]["orders"]
    buy = next(o for o in orders if o["ts_code"] == code and o["side"] == "buy")

    expected_pre = round_lot(0.5 * 1_000_000.0 / pre_close)  # 50000
    expected_close = round_lot(0.5 * 1_000_000.0 / close_px)  # 38400
    assert expected_pre != expected_close  # 判别力：两价必须拉开
    assert buy["volume"] == expected_pre, (
        f"定量须用 pre_close={pre_close} → {expected_pre}，"
        f"勿用 close={close_px} → {expected_close}；实际 volume={buy['volume']}"
    )


# ── #6 ST 收窄涨跌停 ──────────────────────────────────────────────


def test_paper_broker_st_limit_up_rejects_buy_when_is_st() -> None:
    """主板 ST 阈值 4.8%：开盘 +6% 对 ST 已涨停拒买；非 ST 未涨停应成交。

    旧实现 PaperBroker 不传 is_st → 恒用 9.8% → 6% 不拒 → 红。
    """
    # open/pre_close = 10.6/10 → +6%，落在 (4.8%, 9.8%) 之间
    open_px, pre_close = 10.6, 10.0
    code = "600001.SH"  # 主板

    # ST：应收窄并拒
    b_st = PaperBroker(initial_cash=1_000_000.0)
    b_st.advance_to(
        date(2026, 1, 5),
        {
            code: {
                "open": open_px,
                "pre_close": pre_close,
                "close": open_px,
                "vol": 1e6,
                "adv": 1e12,
                "is_st": True,
            }
        },
    )
    acks_st = b_st.place_orders([Order(code, "buy", 1000, "market", None)])
    assert not acks_st[0].accepted and acks_st[0].reason == "limit_up", (
        f"ST 股 +6% 应判涨停拒买，实际 accepted={acks_st[0].accepted} reason={acks_st[0].reason}"
    )

    # 非 ST（缺 is_st 或 False）：宽阈值，应成交
    b_ns = PaperBroker(initial_cash=1_000_000.0)
    b_ns.advance_to(
        date(2026, 1, 5),
        {
            code: {
                "open": open_px,
                "pre_close": pre_close,
                "close": open_px,
                "vol": 1e6,
                "adv": 1e12,
                "is_st": False,
            }
        },
    )
    acks_ns = b_ns.place_orders([Order(code, "buy", 1000, "market", None)])
    assert acks_ns[0].accepted, (
        f"非 ST +6% 不应拒，实际 reason={acks_ns[0].reason}"
    )


def test_daily_step_passes_is_st_from_build_is_st_by_date(
    tmp_path: Path, monkeypatch
) -> None:
    """drivers 须构造 is_st_by_date 并写入 market entry，使 broker 对 ST 收窄阈值。

    monkeypatch build_is_st_by_date 把执行日标为 ST；开盘 +6% → limit_up 拒买。
    旧实现不构造/不传 is_st → 买单成交 → 红。
    """
    sig, exec_d = date(2026, 1, 5), date(2026, 1, 6)
    code = "600001.SH"
    open_px, pre_close = 10.6, 10.0  # +6%
    daily = pl.DataFrame(
        [
            {
                "trade_date": sig,
                "ts_code": code,
                "open": 10.0,
                "pre_close": 10.0,
                "close": 10.0,
                "vol": 1e8,
                "amount": 1e9,
            },
            {
                "trade_date": exec_d,
                "ts_code": code,
                "open": open_px,
                "pre_close": pre_close,
                "close": open_px,
                "vol": 1e8,
                "amount": 1e9,
            },
        ]
    )
    # 仅执行日将该股标为 ST
    monkeypatch.setattr(
        "factorzen.execution.drivers.build_is_st_by_date",
        lambda codes, dates: {exec_d: {code}},
    )
    rd = _pf__pit_sizing_and_st(tmp_path / "pf", sig, code, 0.5)
    cfg = {"initial_cash": 1_000_000.0, "slippage_bps": 0.0}
    sess = tmp_path / "sess"
    SessionStore(sess).init({"broker": "paper", **cfg})

    r = run_daily_step(sess, exec_d, [rd], daily, config=cfg)
    assert not r["skipped"]
    # 买单被 ST 涨停拒 → 无成交（或 acks 含 limit_up）
    recs = SessionStore(sess).ledger_records()
    assert len(recs) == 1
    acks = recs[0]["acks"]
    assert any(not a["accepted"] and a["reason"] == "limit_up" for a in acks), (
        f"ST +6% 应 limit_up 拒买；acks={acks}, fills={recs[0]['fills']}"
    )
    assert r["n_fills"] == 0


import json
import logging
from datetime import date

import polars as pl

from factorzen.execution.attribution import build_attribution_report
from factorzen.execution.drivers import run_replay
from factorzen.execution.store import SessionStore


def _pf(dir_, sig, code, w):
    dir_.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"ts_code": [code], "target_weight": [w]}).write_parquet(dir_ / "weights.parquet")
    (dir_ / "manifest.json").write_text(json.dumps({"signal_date": sig.isoformat(), "status": "optimal"}))
    return str(dir_)

def _daily(rows):  # rows: list of dict(trade_date,ts_code,open,pre_close,close,vol,amount)
    return pl.DataFrame(rows)

def test_exec_attribution_scenarios_suite(tmp_path, caplog):
    """test_slippage_only_scenario_residual_near_zero；test_suspended_scenario_missed_notional_and_positive_residual；test_ideal_nav_aligned_to_exec_window_ignores_warmup_rows；test_partial_fill_shortfall_attributed_by_reason；test_build_attribution_report_backward_compat_no_acks_does_not_crash；test_build_attribution_report_warns_when_daily_does_not_cover_exec_date；理想孪生(_ideal_nav)遇到空目标信号(risk-off)必须与 real 一致地清仓，；撮合滑点(fill 成交价相对参考价 close 的偏离)必须进 slippage 桶，而非静默；乱序落盘的 ledger（如乱序 daily step）——real_nav 必须按 as_of_date 排序对齐"""
    # -- 原 test_slippage_only_scenario_residual_near_zero --
    def _section_0_test_slippage_only_scenario_residual_near_zero(tmp_path):
        dates = [date(2026,1,5), date(2026,1,6)]
        rows = []
        for d in dates:
            rows.append({"trade_date": d, "ts_code":"A.SZ", "open":10.1, "pre_close":10.0,
                         "close":10.0, "vol":1e8, "amount":1e9})
        daily = _daily(rows)
        rd = _pf(tmp_path/"pf", dates[0], "A.SZ", 0.5)
        run_replay(session_dir=tmp_path/"sess", portfolio_run_dirs=[rd], daily=daily,
                   initial_cash=1_000_000.0, from_date=dates[0], to_date=dates[-1], seed=0)
        rep = build_attribution_report(tmp_path/"sess", [rd], daily, initial_cash=1_000_000.0)
        assert sum(v["count"] for v in rep["missed_by_reason"].values()) == 0   # 无未成交
        assert rep["cost_bps"] > 0 and rep["slippage_bps"] != 0
        # 有成交 → 换手/成交笔数非零（换手接入归因）
        assert rep["ann_turnover"] > 0 and rep["n_fills"] > 0
        # residual = total_gap - cost - slippage 应接近 0（纯滑点+成本场景，无未成交/时点差）。
        # 收紧为「相对成本+滑点规模的 5%」而非宽松的「< 成本+滑点之和」，后者对任意
        # residual < 2×(cost+slip) 都会通过、没有判别力；这里 5% 阈值取自实测 residual
        # 量级（约 -0.17bps，相对 cost+slip 合计约 6746bps 是 0.0025% 量级），留了近
        # 20 倍裕度防浮点抖动，但仍能在 residual 真被打破（如量级到几十/几百 bps）时报警。
        assert abs(rep["residual_bps"]) < 0.05 * (abs(rep["cost_bps"]) + abs(rep["slippage_bps"]))

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_slippage_only_scenario_residual_near_zero(_tp0)

    # -- 原 test_suspended_scenario_missed_notional_and_positive_residual --
    def _section_1_test_suspended_scenario_missed_notional_and_positive_residual(tmp_path):
        dates = [date(2026,1,5), date(2026,1,6)]
        daily = _daily([
            {"trade_date": dates[0], "ts_code": "A.SZ", "open": 10.0, "pre_close": 10.0,
             "close": 10.0, "vol": 0.0, "amount": 0.0},  # 停牌
            {"trade_date": dates[1], "ts_code": "A.SZ", "open": 10.5, "pre_close": 10.0,
             "close": 10.5, "vol": 1e8, "amount": 1e9},  # 复牌，缺口 +5%
        ])
        rd = _pf(tmp_path/"pf", date(2026, 1, 2), "A.SZ", 0.5)  # 信号早于 d1，次日(d1)执行
        run_replay(session_dir=tmp_path/"sess", portfolio_run_dirs=[rd], daily=daily,
                   initial_cash=1_000_000.0, from_date=dates[0], to_date=dates[-1], seed=0)
        rep = build_attribution_report(tmp_path/"sess", [rd], daily, initial_cash=1_000_000.0)
        assert rep["missed_by_reason"]["suspended"]["count"] >= 1
        assert rep["missed_by_reason"]["suspended"]["notional"] > 0
        # 理想（frictionless，day1 已全额建仓吃满缺口）vs 真实（day1 停牌 0
        # 成交、day2 追价踏空缺口）应有显著非零总缺口
        assert rep["ideal"]["ann_ret"] != rep["real"]["ann_ret"]
        assert rep["ideal"]["ann_ret"] > rep["real"]["ann_ret"]

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_suspended_scenario_missed_notional_and_positive_residual(_tp1)

    # -- 原 test_ideal_nav_aligned_to_exec_window_ignores_warmup_rows --
    def _section_2_test_ideal_nav_aligned_to_exec_window_ignores_warmup_rows(tmp_path):
        e1, e2, e3 = date(2026, 1, 28), date(2026, 1, 29), date(2026, 1, 30)
        w1, w2 = date(2026, 1, 20), date(2026, 1, 21)

        def _rows(with_warmup: bool) -> list[dict]:
            rows = []
            if with_warmup:
                # 预热行用完全不同量级的价格（50/60 而非 10 附近），若 bug 复现（把
                # 这些未执行日也计入 ideal_nav）会把 ideal 指标带偏得非常明显。
                rows.append({"trade_date": w1, "ts_code": "A.SZ", "open": 50.0, "pre_close": 50.0,
                             "close": 50.0, "vol": 1e8, "amount": 1e9})
                rows.append({"trade_date": w2, "ts_code": "A.SZ", "open": 60.0, "pre_close": 50.0,
                             "close": 60.0, "vol": 1e8, "amount": 1e9})
            rows.append({"trade_date": e1, "ts_code": "A.SZ", "open": 10.0, "pre_close": 10.0,
                         "close": 10.0, "vol": 1e8, "amount": 1e9})
            rows.append({"trade_date": e2, "ts_code": "A.SZ", "open": 10.4, "pre_close": 10.0,
                         "close": 10.4, "vol": 1e8, "amount": 1e9})  # +4% 缺口（远离涨停阈值）
            rows.append({"trade_date": e3, "ts_code": "A.SZ", "open": 10.4, "pre_close": 10.4,
                         "close": 10.4, "vol": 1e8, "amount": 1e9})  # 持平
            return rows

        def _run(with_warmup: bool, sess_name: str) -> dict:
            daily = _daily(_rows(with_warmup))
            rd = _pf(tmp_path / f"pf_{sess_name}", date(2026, 1, 2), "A.SZ", 0.5)  # 早于 e1，e1 起执行
            run_replay(session_dir=tmp_path / sess_name, portfolio_run_dirs=[rd], daily=daily,
                       initial_cash=1_000_000.0, from_date=e1, to_date=e3, seed=0)
            return build_attribution_report(tmp_path / sess_name, [rd], daily, initial_cash=1_000_000.0)

        rep_no_warmup = _run(False, "sess_no_warmup")
        rep_with_warmup = _run(True, "sess_with_warmup")

        # 独立手算 ground-truth（仅 3 个执行日，frictionless 按 close 全额零成本、
        # 每日按当前 nav 再平衡到 0.5 权重）：
        #   day e1: nav 1,000,000 建仓 50,000 股@10.0 → nav_after 1,000,000（frictionless
        #           无成本，估值/成交同价，nav 不变）
        #   day e2: 价格→10.4（+4%），nav_before = 500,000(现金) + 50,000*10.4 = 1,020,000，
        #           再平衡到 0.5*1,020,000/10.4=49,038→整手 49,000 股，卖出 1,000 股@10.4，
        #           nav_after 仍 1,020,000（frictionless 无损耗）
        #   day e3: 价格持平 10.4，目标股数不变（delta=0，无交易），nav_after 1,020,000
        # nav 序列 [1,000,000, 1,000,000, 1,020,000, 1,020,000]
        # rets = [0, 0.02, 0] → ann_ret = mean(rets)*252 = (0.02/3)*252 = 1.68
        assert rep_no_warmup["n_days"] == 3
        assert abs(rep_no_warmup["ideal"]["ann_ret"] - 1.68) < 1e-9
        assert rep_no_warmup["ideal"]["max_dd"] == 0.0
        # 核心不变性：预热行完全不应泄漏进 ideal 指标——有/无预热行两次独立 run 的
        # ideal 一致（bug 未修时二者不等：预热日被当空仓稀释，ann_ret 会被拉低到
        # 约 1.008 而非 1.68）
        assert rep_with_warmup["n_days"] == 3
        assert rep_with_warmup["ideal"] == rep_no_warmup["ideal"]

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_ideal_nav_aligned_to_exec_window_ignores_warmup_rows(_tp2)

    # -- 原 test_partial_fill_shortfall_attributed_by_reason --
    def _section_3_test_partial_fill_shortfall_attributed_by_reason(tmp_path):
        d0 = date(2026, 1, 5)
        daily = _daily([{"trade_date": d0, "ts_code": "A.SZ", "open": 100.0, "pre_close": 100.0,
                         "close": 100.0, "vol": 1e8, "amount": 1e9}])
        rd = _pf(tmp_path / "pf", date(2026, 1, 2), "A.SZ", 1.0)  # 早于 d0，d0 执行
        run_replay(session_dir=tmp_path / "sess", portfolio_run_dirs=[rd], daily=daily,
                   initial_cash=1_000_000.0, from_date=d0, to_date=d0, seed=0)
        rep = build_attribution_report(tmp_path / "sess", [rd], daily, initial_cash=1_000_000.0)
        assert rep["missed_by_reason"]["insufficient_cash"]["count"] >= 1
        # 手算 notional = shortfall(100 股) × close(100.0) = 10,000
        assert abs(rep["missed_by_reason"]["insufficient_cash"]["notional"] - 10_000.0) < 1e-6

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_partial_fill_shortfall_attributed_by_reason(_tp3)

    # -- 原 test_build_attribution_report_backward_compat_no_acks_does_not_crash --
    def _section_4_test_build_attribution_report_backward_compat_no_acks_does_not_crash(tmp_path):
        d0 = date(2026, 1, 5)
        daily = _daily(
            [
                {
                    "trade_date": d0,
                    "ts_code": "A.SZ",
                    "open": 10.1,
                    "pre_close": 10.0,
                    "close": 10.0,
                    "vol": 1e8,
                    "amount": 1e9,
                }
            ]
        )
        rd = _pf(tmp_path / "pf", d0, "A.SZ", 0.5)
        sess = tmp_path / "sess"
        SessionStore(sess).init({"broker": "paper", "initial_cash": 1_000_000.0})

        orders = [
            {"ts_code": "A.SZ", "side": "buy", "volume": 100, "price_type": "market", "price": None}
        ]
        fills = [
            {
                "order_id": "paper-1",
                "ts_code": "A.SZ",
                "side": "buy",
                "filled_volume": 100,
                "price": 10.1,
                "cost": 2.5,
                "ts": d0.isoformat(),
            }
        ]
        row = {
            "as_of_date": d0.isoformat(),
            "nav_before": 1_000_000.0,
            "nav_after": 999_000.0,
            "payload": json.dumps({"orders": orders, "fills": fills}),  # 无 acks 键
        }
        pl.DataFrame([row]).write_parquet(sess / "ledger.parquet")
        pl.DataFrame([{"as_of_date": d0.isoformat(), "nav_after": 999_000.0}]).write_parquet(
            sess / "nav.parquet"
        )

        rep = build_attribution_report(sess, [rd], daily, initial_cash=1_000_000.0)
        assert rep["missed_by_reason"] == {}
        # 成本来自 fills，不依赖 acks，应仍正确算出（100 股 * 2.5 折算年化 bps）
        assert rep["cost_bps"] != 0

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_build_attribution_report_backward_compat_no_acks_does_not_crash(_tp4)

    # -- 原 test_build_attribution_report_warns_when_daily_does_not_cover_exec_date --
    def _section_5_test_build_attribution_report_warns_when_daily_does_not_cover_exec_date(tmp_path, caplog):
        d1, d2 = date(2026, 1, 5), date(2026, 1, 6)
        daily_full = _daily(
            [
                {
                    "trade_date": d1,
                    "ts_code": "A.SZ",
                    "open": 10.1,
                    "pre_close": 10.0,
                    "close": 10.0,
                    "vol": 1e8,
                    "amount": 1e9,
                },
                {
                    "trade_date": d2,
                    "ts_code": "A.SZ",
                    "open": 10.2,
                    "pre_close": 10.0,
                    "close": 10.1,
                    "vol": 1e8,
                    "amount": 1e9,
                },
            ]
        )
        rd = _pf(tmp_path / "pf", d1, "A.SZ", 0.5)
        run_replay(
            session_dir=tmp_path / "sess",
            portfolio_run_dirs=[rd],
            daily=daily_full,
            initial_cash=1_000_000.0,
            from_date=d1,
            to_date=d2,
            seed=0,
        )
        # 调用方传入比 session 实际执行窗口更窄的 daily（只覆盖 d1，缺 d2）
        daily_narrow = daily_full.filter(pl.col("trade_date") == d1)
        with caplog.at_level(logging.WARNING):
            build_attribution_report(
                tmp_path / "sess", [rd], daily_narrow, initial_cash=1_000_000.0
            )
        assert any(d2.isoformat() in rec.message for rec in caplog.records)

    _tp5 = tmp_path / "_s5"
    _tp5.mkdir(exist_ok=True)
    _section_5_test_build_attribution_report_warns_when_daily_does_not_cover_exec_date(_tp5, caplog)

    # -- 原 test_ideal_nav_liquidates_on_empty_target --
    def _section_6_test_ideal_nav_liquidates_on_empty_target(tmp_path):
        from factorzen.execution.attribution import _ideal_nav

        d1, d2, d3 = date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)
        # d3 价格翻倍：若 ideal 错误地持有到 d3 会使 nav 暴涨，清仓则不受影响。
        daily = _daily([
            {"trade_date": d1, "ts_code": "A.SZ", "open": 10.0, "pre_close": 10.0,
             "close": 10.0, "vol": 1e8, "amount": 1e9},
            {"trade_date": d2, "ts_code": "A.SZ", "open": 10.0, "pre_close": 10.0,
             "close": 10.0, "vol": 1e8, "amount": 1e9},
            {"trade_date": d3, "ts_code": "A.SZ", "open": 20.0, "pre_close": 10.0,
             "close": 20.0, "vol": 1e8, "amount": 1e9},
        ])
        buy = _pf(tmp_path / "buy", d1, "A.SZ", 0.9)
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        pl.DataFrame(
            {"ts_code": [], "target_weight": []},
            schema={"ts_code": pl.Utf8, "target_weight": pl.Float64},
        ).write_parquet(empty_dir / "weights.parquet")
        (empty_dir / "manifest.json").write_text(
            json.dumps({"signal_date": d2.isoformat(), "status": "optimal"})
        )

        nav = _ideal_nav([buy, str(empty_dir)], daily, 1_000_000.0, [d1, d2, d3])
        # nav = [init, d1, d2, d3]；d2 空目标 → 清仓 → 全现金；d3 价格翻倍但已清仓 → nav 不变。
        assert len(nav) == 4
        assert abs(nav[3] - nav[2]) < 1.0, f"空目标清仓后 d3 涨幅不应影响 ideal nav: {nav}"

    _tp6 = tmp_path / "_s6"
    _tp6.mkdir(exist_ok=True)
    _section_6_test_ideal_nav_liquidates_on_empty_target(_tp6)

    # -- 原 test_broker_slippage_enters_slippage_bucket_not_residual --
    def _section_7_test_broker_slippage_enters_slippage_bucket_not_residual(tmp_path):
        d0 = date(2026, 1, 5)
        # open=10.2、close=10.0；fill 成交价 10.5（含 broker 滑点，偏离 open）。
        daily = _daily([{"trade_date": d0, "ts_code": "A.SZ", "open": 10.2, "pre_close": 10.0,
                         "close": 10.0, "vol": 1e8, "amount": 1e9}])
        sess = tmp_path / "sess"
        SessionStore(sess).init({"broker": "paper", "initial_cash": 1_000_000.0})
        orders = [{"ts_code": "A.SZ", "side": "buy", "volume": 100, "price_type": "market", "price": None}]
        fills = [{"order_id": "paper-1", "ts_code": "A.SZ", "side": "buy", "filled_volume": 100,
                  "price": 10.5, "cost": 0.0, "ts": d0.isoformat()}]
        row = {"as_of_date": d0.isoformat(), "nav_before": 1_000_000.0, "nav_after": 998_950.0,
               "payload": json.dumps({"orders": orders, "fills": fills})}  # 无 acks → 跳过 missed
        pl.DataFrame([row]).write_parquet(sess / "ledger.parquet")
        rd = _pf(tmp_path / "pf", d0, "A.SZ", 0.5)

        rep = build_attribution_report(sess, [rd], daily, initial_cash=1_000_000.0)
        # slip_sum = filled(100) × (fill_price 10.5 − close 10.0) × sign(buy +1) = 50
        # slip_bps = 50 / 1e6 × 1e4 × (252/1) = 126.0（基于 open 会错成 20 → 50.4）
        assert abs(rep["slippage_bps"] - 126.0) < 0.5, (
            f"滑点桶应基于成交价 fill_price，实际 {rep['slippage_bps']}"
        )

    _tp7 = tmp_path / "_s7"
    _tp7.mkdir(exist_ok=True)
    _section_7_test_broker_slippage_enters_slippage_bucket_not_residual(_tp7)

    # -- 原 test_real_nav_aligned_by_date_when_ledger_out_of_order --
    def _section_8_test_real_nav_aligned_by_date_when_ledger_out_of_order(tmp_path):
        d1, d2 = date(2026, 1, 5), date(2026, 1, 6)
        daily = _daily([
            {"trade_date": d1, "ts_code": "A.SZ", "open": 10.0, "pre_close": 10.0,
             "close": 10.0, "vol": 1e8, "amount": 1e9},
            {"trade_date": d2, "ts_code": "A.SZ", "open": 11.0, "pre_close": 10.0,
             "close": 11.0, "vol": 1e8, "amount": 1e9},
        ])
        sess = tmp_path / "sess"
        SessionStore(sess).init({"broker": "paper", "initial_cash": 1_000_000.0})
        empty_payload = json.dumps({"orders": [], "acks": [], "fills": []})
        # 乱序落盘：d2 行在前、d1 行在后。nav 从 d1(1.0e6) 到 d2(1.05e6) 单调上升。
        rows = [
            {"as_of_date": d2.isoformat(), "nav_before": 1_000_000.0, "nav_after": 1_050_000.0,
             "payload": empty_payload},
            {"as_of_date": d1.isoformat(), "nav_before": 1_000_000.0, "nav_after": 1_000_000.0,
             "payload": empty_payload},
        ]
        pl.DataFrame(rows).write_parquet(sess / "ledger.parquet")
        rd = _pf(tmp_path / "pf", d1, "A.SZ", 0.5)

        rep = build_attribution_report(sess, [rd], daily, initial_cash=1_000_000.0)
        # 正确排序后 real_nav=[1e6, 1e6, 1.05e6] 单调不减 → max_dd=0；乱序则 1.05e6→1e6 出现回撤。
        assert rep["real"]["max_dd"] == 0.0, (
            f"real_nav 应按日期对齐、无虚假回撤，实际 max_dd={rep['real']['max_dd']}"
        )

    _tp8 = tmp_path / "_s8"
    _tp8.mkdir(exist_ok=True)
    _section_8_test_real_nav_aligned_by_date_when_ledger_out_of_order(_tp8)



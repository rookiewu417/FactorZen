"""合并自: test_metrics.py, test_momentum_rotation.py, test_trend_timing.py, test_trend_timing_smoke.py
目标: test_strategies.py

--- 来源 test_metrics.py ---
(无原 docstring)

--- 来源 test_momentum_rotation.py ---
(无原 docstring)

--- 来源 test_trend_timing.py ---
(无原 docstring)

--- 来源 test_trend_timing_smoke.py ---
择时 vs 基线端到端离线 smoke：run_trend_timing_experiment 跑通两套实验，
且 risk-off 段策略确实降仓/空仓（而基线始终满仓），非恒真——独立构造场景，
用 SessionStore.load_state() 的持仓做跨函数验证，不依赖生成产物本身的字段。
"""

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from factorzen.execution.store import SessionStore
from factorzen.strategies.metrics import (
    _metrics_from_nav,
    format_metrics_table,
    session_metrics,
)
from factorzen.strategies.momentum_rotation import generate_momentum_rotation_products
from factorzen.strategies.runner import run_trend_timing_experiment
from factorzen.strategies.trend_timing import generate_trend_timing_products


# ==== 来自 test_metrics.py ====
def test_metrics_format_suite(tmp_path):
    """test_metrics_from_nav_hand_computed；test_session_metrics_turnover_and_cost_hand_computed；test_format_metrics_table_contains_labels_and_values"""
    # -- 原 test_metrics_from_nav_hand_computed --
    def _section_0_test_metrics_from_nav_hand_computed():
        m = _metrics_from_nav([100.0, 110.0, 99.0])
        assert abs(m["total_return"] - (-0.01)) < 1e-9    # 99/100-1
        assert abs(m["ann_ret"] - 0.0) < 1e-9             # mean(0.1,-0.1)*252=0
        assert abs(m["max_dd"] - (-0.1)) < 1e-9           # 0.99/1.1-1
        assert abs(m["win_rate"] - 0.5) < 1e-9            # 1/2 天为正
        assert m["n_days"] == 2

    _section_0_test_metrics_from_nav_hand_computed()

    # -- 原 test_session_metrics_turnover_and_cost_hand_computed --
    def _section_1_test_session_metrics_turnover_and_cost_hand_computed(tmp_path):
        s = SessionStore(tmp_path / "sess")
        s.init({"broker": "paper", "initial_cash": 1_000_000.0})
        # 一天：买 1000 股 @10, 成本 5 元 → 成交额 10000, cost 5
        fills = [{"order_id": "o1", "ts_code": "X.SZ", "side": "buy",
                  "filled_volume": 1000, "price": 10.0, "cost": 5.0, "ts": "2026-01-05"}]
        s.append(_rec("2026-01-05", 1_000_000.0, 1_000_000.0, fills))

        m = session_metrics(str(tmp_path / "sess"), 1_000_000.0)
        assert m["n_fills"] == 1
        assert abs(m["total_cost"] - 5.0) < 1e-9
        assert abs(m["total_cost_bps"] - 0.05) < 1e-9      # 5/1e6*1e4
        assert m["ann_turnover"] > 0                       # 有成交 → 换手非零
        assert "sharpe" in m and "calmar" in m            # 净值类指标也在

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_session_metrics_turnover_and_cost_hand_computed(_tp1)

    # -- 原 test_format_metrics_table_contains_labels_and_values --
    def _section_2_test_format_metrics_table_contains_labels_and_values():
        a = _metrics_from_nav([100.0, 110.0, 121.0])
        a.update({"ann_turnover": 3.0, "total_cost_bps": 12.5, "n_fills": 42})
        t = format_metrics_table({"策略": a, "基线": a})
        assert "年化收益" in t and "年化换手(双边)" in t and "Calmar" in t
        assert "策略" in t and "基线" in t

    _section_2_test_format_metrics_table_contains_labels_and_values()


def _rec(d, nav_before, nav_after, fills):
    return {"as_of_date": d, "nav_before": nav_before, "nav_after": nav_after,
            "broker_state": {"cash": nav_after, "pos": {}, "order_seq": len(fills)},
            "orders": [], "acks": [], "fills": fills}


# ==== 来自 test_momentum_rotation.py ====
def _idx__momentum_rotation(closes: list[float], start=date(2026, 1, 1)):
    return pl.DataFrame(
        [{"trade_date": start + timedelta(days=i), "close": c} for i, c in enumerate(closes)]
    )


def _price__momentum_rotation(dates, codes, amount=1e9):
    return pl.DataFrame(
        [{"trade_date": d, "ts_code": c, "open": 10.0, "pre_close": 10.0,
          "close": 10.0, "vol": 1e8, "amount": amount} for d in dates for c in codes]
    )


def _members(code, date_str):  # 注入避网络：两指数各自成分
    return {"IDXA": ["A1.SZ", "A2.SZ"], "IDXB": ["B1.SZ", "B2.SZ"]}[code]


def test_rotation_strategy_suite():
    """test_rotation_picks_stronger_index；test_all_negative_momentum_goes_cash；test_pit_no_lookahead"""
    # -- 原 test_rotation_picks_stronger_index --
    def _section_0_test_rotation_picks_stronger_index():
        closes_a = [10.0] * 5 + [12.0]   # lookback=5: 12/10-1=+20%
        closes_b = [10.0] * 5 + [10.5]   # +5%
        T = date(2026, 1, 6)
        dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(6)]
        price = _price__momentum_rotation(dates, ["A1.SZ", "A2.SZ", "B1.SZ", "B2.SZ"])
        dirs = generate_momentum_rotation_products(
            "/tmp/_mr_a", {"IDXA": _idx__momentum_rotation(closes_a), "IDXB": _idx__momentum_rotation(closes_b)}, price, [T],
            members_fn=_members, lookback=5, top_n=2)
        held = set(pl.read_parquet(Path(dirs[0]) / "weights.parquet")["ts_code"].to_list())
        assert held == {"A1.SZ", "A2.SZ"}, f"应持强者 IDXA 成分, 实际 {held}"

    _section_0_test_rotation_picks_stronger_index()

    # -- 原 test_all_negative_momentum_goes_cash --
    def _section_1_test_all_negative_momentum_goes_cash():
        closes_a = [10.0] * 5 + [9.0]    # -10%
        closes_b = [10.0] * 5 + [9.5]    # -5%
        T = date(2026, 1, 6)
        dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(6)]
        price = _price__momentum_rotation(dates, ["A1.SZ", "A2.SZ", "B1.SZ", "B2.SZ"])
        dirs = generate_momentum_rotation_products(
            "/tmp/_mr_b", {"IDXA": _idx__momentum_rotation(closes_a), "IDXB": _idx__momentum_rotation(closes_b)}, price, [T],
            members_fn=_members, lookback=5, top_n=2)
        assert pl.read_parquet(Path(dirs[0]) / "weights.parquet").height == 0, "全负动量应空仓"

    _section_1_test_all_negative_momentum_goes_cash()

    # -- 原 test_pit_no_lookahead --
    def _section_2_test_pit_no_lookahead():
        T = date(2026, 1, 6)
        dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(8)]
        closes_a = [10.0] * 5 + [12.0, 12.0, 12.0]  # 到 T(idx5) +20%
        closes_b = [10.0] * 5 + [10.5, 100.0, 200.0]  # T 处 +5%, 但 T 之后暴涨
        price = _price__momentum_rotation(dates, ["A1.SZ", "A2.SZ", "B1.SZ", "B2.SZ"])
        dirs = generate_momentum_rotation_products(
            "/tmp/_mr_c", {"IDXA": _idx__momentum_rotation(closes_a), "IDXB": _idx__momentum_rotation(closes_b)}, price, [T],
            members_fn=_members, lookback=5, top_n=2)
        held = set(pl.read_parquet(Path(dirs[0]) / "weights.parquet")["ts_code"].to_list())
        assert held == {"A1.SZ", "A2.SZ"}, "T 处应仍选 IDXA; 若受 T 之后 IDXB 暴涨影响=泄漏未来"

    _section_2_test_pit_no_lookahead()


# ==== 来自 test_trend_timing.py ====
def _idx__trend_timing(rows):  # rows: (date, close)
    return pl.DataFrame([{"trade_date": d, "close": c} for d, c in rows])


def _price__trend_timing(dates, codes, amount=1e9):
    return pl.DataFrame(
        [
            {
                "trade_date": d,
                "ts_code": c,
                "open": 10.0,
                "pre_close": 10.0,
                "close": 10.0,
                "vol": 1e8,
                "amount": amount,
            }
            for d in dates
            for c in codes
        ]
    )


def _fake_members__trend_timing(code, date_str):  # 注入,避免网络
    return ["A.SZ", "B.SZ", "C.SZ"]


def test_ma_riskon_baseline_suite(tmp_path):
    """test_pit_no_lookahead_ma；test_risk_on_equal_weight_topn；test_baseline_always_full；test_strategy_vs_baseline_experiment"""
    # -- 原 test_pit_no_lookahead_ma --
    def _section_0_test_pit_no_lookahead_ma(tmp_path):
        dates = [date(2026, 1, d) for d in range(5, 12)]
        idx = _idx__trend_timing([(dates[i], 10.0) for i in range(6)] + [(dates[6], 100.0)])  # 最后一天暴涨
        price = _price__trend_timing(dates, ["A.SZ", "B.SZ", "C.SZ"])
        # 在 T=dates[5] 调仓, ma_window=3: MA=mean(close[≤T].tail(3))=10, close(T)=10 → not >MA → risk-off
        dirs = generate_trend_timing_products(
            str(tmp_path / "s"),
            idx,
            price,
            [dates[5]],
            members_fn=_fake_members__trend_timing,
            ma_window=3,
            top_n=3,
            timing=True,
        )
        w = pl.read_parquet(Path(dirs[0]) / "weights.parquet")
        assert w.height == 0, "T 处 close==MA 未站上 → 应 risk-off 空仓; 若受 T 之后暴涨影响则泄漏"

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_pit_no_lookahead_ma(_tp0)

    # -- 原 test_risk_on_equal_weight_topn --
    def _section_1_test_risk_on_equal_weight_topn(tmp_path):
        dates = [date(2026, 1, d) for d in range(5, 12)]
        # 让 close 明显站上 MA
        idx = _idx__trend_timing([(dates[i], 10.0) for i in range(6)] + [(dates[6], 10.0)])
        idx = idx.with_columns(
            pl.when(pl.col("trade_date") == dates[5]).then(20.0).otherwise(pl.col("close")).alias("close")
        )
        price = _price__trend_timing(dates, ["A.SZ", "B.SZ", "C.SZ"])
        dirs = generate_trend_timing_products(
            str(tmp_path / "s"),
            idx,
            price,
            [dates[5]],
            members_fn=_fake_members__trend_timing,
            ma_window=3,
            top_n=2,
            timing=True,
        )
        w = pl.read_parquet(Path(dirs[0]) / "weights.parquet")
        assert w.height == 2 and abs(w["target_weight"][0] - 0.5) < 1e-9  # top2 各 1/2

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_risk_on_equal_weight_topn(_tp1)

    # -- 原 test_baseline_always_full --
    def _section_2_test_baseline_always_full(tmp_path):
        dates = [date(2026, 1, d) for d in range(5, 9)]
        idx = _idx__trend_timing([(d, 5.0) for d in dates])  # 一直低于任何 MA
        price = _price__trend_timing(dates, ["A.SZ", "B.SZ", "C.SZ"])
        dirs = generate_trend_timing_products(
            str(tmp_path / "s"),
            idx,
            price,
            [dates[2]],
            members_fn=_fake_members__trend_timing,
            ma_window=2,
            top_n=3,
            timing=False,
        )  # 基线
        w = pl.read_parquet(Path(dirs[0]) / "weights.parquet")
        assert w.height == 3, "基线 timing=False 应始终满仓,无视信号"

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_baseline_always_full(_tp2)

    # -- 原 test_strategy_vs_baseline_experiment --
    def _section_3_test_strategy_vs_baseline_experiment(tmp_path):
        ma_window = 3
        dates = _dates(10)
        codes = ["A.SZ", "B.SZ", "C.SZ"]
        price = _price__trend_timing_smoke(dates, codes)

        # 指数：前段单调上行(站上 MA, risk-on)，最后 3 天骤跌到远低于近期 MA(risk-off)。
        closes = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 5.0, 5.0, 5.0]
        idx = _idx__trend_timing_smoke(dates, closes)

        t_on = dates[4]  # close=18, MA(tail3<=t_on)=mean(14,16,18)=16 → 18>16 risk-on
        # t_off 放在 dates[8]（而非最后一天 dates[9]）：signal 次一交易日才执行（s<d，与 sim
        # 对齐），放最后一天则其清仓信号无执行日、永不生效。dates[8] 的清仓于 dates[9] 执行。
        t_off = dates[8]  # close=5, MA(tail3<=t_off)=mean(22,5,5)=10.67 → 5<10.67 risk-off
        rebalance_dates = [t_on, t_off]

        out = run_trend_timing_experiment(
            str(tmp_path / "exp"),
            idx,
            price,
            rebalance_dates,
            initial_cash=1_000_000.0,
            from_date=dates[0],
            to_date=dates[-1],
            members_fn=_fake_members__trend_timing_smoke,
            ma_window=ma_window,
            top_n=3,
        )

        assert set(out) == {"strategy", "baseline"}
        for label in ("strategy", "baseline"):
            session_dir = Path(out[label]["session_dir"])
            assert (session_dir / "nav.parquet").exists(), f"{label} 应产 nav.parquet"
            metrics = out[label]["metrics"]
            assert {"ann_ret", "sharpe", "max_dd"} <= set(metrics)

        # risk-off 段(t_off 当天清仓)：策略应清空持仓；基线(timing=False)始终满仓，
        # 不受均线信号影响 —— 两者用同一份行情/成分股/资金,唯一差异是 timing 开关,
        # 若引擎/生成器未正确接入择时信号,这条断言会失败(非恒真)。
        strat_held = _held(SessionStore(out["strategy"]["session_dir"]).load_state())
        base_held = _held(SessionStore(out["baseline"]["session_dir"]).load_state())
        assert strat_held == {}, f"择时 risk-off 段末应空仓, 实际 {strat_held}"
        assert base_held != {}, "基线应始终满仓(不受择时信号影响)"

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_strategy_vs_baseline_experiment(_tp3)


# ==== 来自 test_trend_timing_smoke.py ====
def _dates(n: int, start: date = date(2026, 1, 5)) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _idx__trend_timing_smoke(dates: list[date], closes: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"trade_date": dates, "close": closes})


def _price__trend_timing_smoke(dates: list[date], codes: list[str], amount: float = 1e9) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "trade_date": d,
                "ts_code": c,
                "open": 10.0,
                "pre_close": 10.0,
                "close": 10.0,
                "vol": 1e8,
                "amount": amount,
            }
            for d in dates
            for c in codes
        ]
    )


def _fake_members__trend_timing_smoke(code: str, date_str: str) -> list[str]:  # 注入,避免网络
    return ["A.SZ", "B.SZ", "C.SZ"]


def _held(state: dict | None) -> dict:
    if not state:
        return {}
    pos = state.get("pos", state.get("positions", {}))
    return {
        c: p
        for c, p in pos.items()
        if (p.get("volume", 0) if isinstance(p, dict) else 0) > 0
    }



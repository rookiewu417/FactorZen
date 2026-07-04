from pathlib import Path

from factorzen.execution.store import SessionStore
from factorzen.strategies.metrics import (
    _metrics_from_nav,
    format_metrics_table,
    session_metrics,
)


def test_metrics_from_nav_hand_computed():
    # navs=[100,110,99] → rets=[+0.1,-0.1]
    m = _metrics_from_nav([100.0, 110.0, 99.0])
    assert abs(m["total_return"] - (-0.01)) < 1e-9    # 99/100-1
    assert abs(m["ann_ret"] - 0.0) < 1e-9             # mean(0.1,-0.1)*252=0
    assert abs(m["max_dd"] - (-0.1)) < 1e-9           # 0.99/1.1-1
    assert abs(m["win_rate"] - 0.5) < 1e-9            # 1/2 天为正
    assert m["n_days"] == 2


def _rec(d, nav_before, nav_after, fills):
    return {"as_of_date": d, "nav_before": nav_before, "nav_after": nav_after,
            "broker_state": {"cash": nav_after, "pos": {}, "order_seq": len(fills)},
            "orders": [], "acks": [], "fills": fills}


def test_session_metrics_turnover_and_cost_hand_computed(tmp_path: Path):
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


def test_format_metrics_table_contains_labels_and_values():
    a = _metrics_from_nav([100.0, 110.0, 121.0])
    a.update({"ann_turnover": 3.0, "total_cost_bps": 12.5, "n_fills": 42})
    t = format_metrics_table({"策略": a, "基线": a})
    assert "年化收益" in t and "年化换手(双边)" in t and "Calmar" in t
    assert "策略" in t and "基线" in t

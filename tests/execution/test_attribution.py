import json
import logging
from datetime import date
from pathlib import Path

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

def test_slippage_only_scenario_residual_near_zero(tmp_path: Path):
    # 单票、无停牌无涨跌停、open≠close → 纯滑点+成本，missed=0，residual≈0
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

def test_suspended_scenario_missed_notional_and_positive_residual(tmp_path: Path):
    # 独立手算 ground-truth：day1 停牌(vol=0) → 真实 0 成交、理想(frictionless)
    # 仍按 close=10.0 全额买入。day2 复牌且低开高走缺口(open=10.5，较 day1
    # close +5%，无涨停)：理想因 day1 已建仓，吃到这段缺口收益；真实 day1
    # 未成交、day2 才追价买入(price=10.5)，错过了这段缺口——这才是真实的
    # "停牌导致踏空"经济含义，而非同日 buy@close/mark@close 的恒等 0。
    # 手算：ideal day2 nav ~1,025,000（多头吃满 5% 缺口），real day2 nav
    # ~1,000,000-手续费（day2 才追价，无缺口收益）；两者 ann_ret 应显著不同。
    dates = [date(2026,1,5), date(2026,1,6)]
    daily = _daily([
        {"trade_date": dates[0], "ts_code": "A.SZ", "open": 10.0, "pre_close": 10.0,
         "close": 10.0, "vol": 0.0, "amount": 0.0},  # 停牌
        {"trade_date": dates[1], "ts_code": "A.SZ", "open": 10.5, "pre_close": 10.0,
         "close": 10.5, "vol": 1e8, "amount": 1e9},  # 复牌，缺口 +5%
    ])
    rd = _pf(tmp_path/"pf", dates[0], "A.SZ", 0.5)
    run_replay(session_dir=tmp_path/"sess", portfolio_run_dirs=[rd], daily=daily,
               initial_cash=1_000_000.0, from_date=dates[0], to_date=dates[-1], seed=0)
    rep = build_attribution_report(tmp_path/"sess", [rd], daily, initial_cash=1_000_000.0)
    assert rep["missed_by_reason"]["suspended"]["count"] >= 1
    assert rep["missed_by_reason"]["suspended"]["notional"] > 0
    # 理想（frictionless，day1 已全额建仓吃满缺口）vs 真实（day1 停牌 0
    # 成交、day2 追价踏空缺口）应有显著非零总缺口
    assert rep["ideal"]["ann_ret"] != rep["real"]["ann_ret"]
    assert rep["ideal"]["ann_ret"] > rep["real"]["ann_ret"]

def test_ideal_nav_aligned_to_exec_window_ignores_warmup_rows(tmp_path: Path):
    # daily 里混入 ledger 从未执行的「预热日」（如 ADV 预热/from_date 过滤掉的
    # 历史行）；_ideal_nav 必须只在 ledger 实际执行的 3 天窗口内建仓/估值，不能
    # 因为 daily 全量多出的预热日而被稀释。用两次独立 run（daily 有/无预热行，
    # 其余完全一致）做不变性验证：加预热行不应改变 ideal 指标一个比特。
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
        rd = _pf(tmp_path / f"pf_{sess_name}", e1, "A.SZ", 0.5)
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

def test_partial_fill_shortfall_attributed_by_reason(tmp_path: Path):
    # 独立手算 ground-truth：满仓单票(weight=1.0)，close=open=100.0，
    # initial_cash=1,000,000 → 目标 10,000 股（整手，无 lot 截断）。但买入需扣
    # 单边成本(佣金2.5bp+滑点5bp=7.5bp)，可用现金只够 floor(1,000,000/(100*1.00075)
    # /100)*100 = 9,900 股 → accepted=True 但 reason=insufficient_cash 部分成交，
    # shortfall=100 股。旧代码 `if ack.get("accepted"): continue` 会把这笔部分成
    # 交整单跳过、完全不归因；修复后应按 reason 归 missed_by_reason。
    d0 = date(2026, 1, 5)
    daily = _daily([{"trade_date": d0, "ts_code": "A.SZ", "open": 100.0, "pre_close": 100.0,
                     "close": 100.0, "vol": 1e8, "amount": 1e9}])
    rd = _pf(tmp_path / "pf", d0, "A.SZ", 1.0)
    run_replay(session_dir=tmp_path / "sess", portfolio_run_dirs=[rd], daily=daily,
               initial_cash=1_000_000.0, from_date=d0, to_date=d0, seed=0)
    rep = build_attribution_report(tmp_path / "sess", [rd], daily, initial_cash=1_000_000.0)
    assert rep["missed_by_reason"]["insufficient_cash"]["count"] >= 1
    # 手算 notional = shortfall(100 股) × close(100.0) = 10,000
    assert abs(rep["missed_by_reason"]["insufficient_cash"]["notional"] - 10_000.0) < 1e-6


def test_build_attribution_report_backward_compat_no_acks_does_not_crash(tmp_path: Path):
    # 手造旧形状 ledger.parquet：payload 只有 {orders, fills}，无 acks（旧版本
    # SessionStore.append 引入 acks 字段之前落的账），orders 非空。
    # 回归 Fix1：`for od, ack in zip(r["orders"], r["acks"], strict=True)` 在
    # acks=[] 但 orders 非空时会因长度不一致抛 ValueError，违反"向后兼容旧
    # ledger"的 spec。修复后应正常返回、不抛异常；成本/滑点来自 fills，不受
    # acks 缺失影响，仍应照常算出非零值；该行的 missed_by_reason 因缺 reason
    # 无法归因，允许为空。
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


def test_build_attribution_report_warns_when_daily_does_not_cover_exec_date(
    tmp_path: Path, caplog
) -> None:
    # 回归 Fix2：_ideal_nav 迭代 ledger 的 exec_dates，但价格取自调用方传入的
    # daily；若 daily 窗口比 session 实际执行窗口窄（如真实停牌导致某日无行/
    # 窗口配置过窄），_market_of_day 对该日返回空，估值/建仓静默按 0 处理，
    # ideal/桶结果静默失真。修复后 build_attribution_report 应在开头检测到
    # 缺口并 logging.warning（研究可信度优先，不能静默）。
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

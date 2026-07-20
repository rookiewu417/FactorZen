"""可实现成交口径：`compute_fwd_returns` 的 `exec_lag` / `exec_price_col`。

**为什么加这两个参数**：默认口径 `close[t+h]/close[t] − 1` 隐含「**t 日收盘成交**」，
但因子信号需要 t 日收盘数据才算得出——拿收盘价算信号、再用那个收盘价成交，**不可实现**。
项目铁律本是「t 日算 → t+1 执行」。

2026-07-19 实测：csi500 上 lgbm 组合 top 桶年化超额 +35.20% 中**隔夜段占 100%**
（日内段仅 +0.05%）；切到可实现口径后只剩 +10.08%。⇒ 默认口径系统性高估。

**测试设计**：期望值全部由**手工构造的价格序列独立算出**，不引用被测函数的中间量
（CLAUDE.md 反复踩的陷阱 #1：`C` 由 `A`、`B` 构造再断言 `C=f(A,B)` 恒真零判别力）。
"""
from __future__ import annotations

import polars as pl
import pytest

from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns


def _px(closes: list[float], opens: list[float] | None = None) -> pl.DataFrame:
    """单只票的价格序列，日期递增。"""
    n = len(closes)
    d = {
        "ts_code": ["X"] * n,
        "trade_date": [f"2024-01-{i + 1:02d}" for i in range(n)],
        "close": closes,
    }
    if opens is not None:
        d["open"] = opens
    return pl.DataFrame(d)


def test_default_unchanged_close_to_close():
    """默认（exec_lag=0）必须逐位等于 close[t+h]/close[t] − 1。

    价格取 100/110/121（每步 +10%）⇒ h=1 的前两个值都必须恰好是 0.10。
    这个期望值由构造决定，与被测函数无关。
    """
    df = compute_fwd_returns(_px([100.0, 110.0, 121.0]), horizons=[1])
    got = df["fwd_ret_1d"].to_list()
    assert got[0] == pytest.approx(0.10, abs=1e-12)
    assert got[1] == pytest.approx(0.10, abs=1e-12)
    assert got[2] is None  # 末日无前向价


def test_exec_lag_shifts_entry_and_exit():
    """exec_lag=1 ⇒ 用 price[t+2]/price[t+1] − 1。

    close = 100/110/121/133.1（每步 +10%）：
    - t=0 的 h=1 应是 121/110 − 1 = 0.10（而非 110/100）
    构造上每步都是 +10%，无法区分——**故意换成非等比序列**：
    close = 100/200/210/420 ⇒ t=0 的 exec_lag=1 值 = 210/200 − 1 = **0.05**
    （对比默认 exec_lag=0 的 200/100 − 1 = 1.00），两者显著不同才有判别力。
    """
    px = _px([100.0, 200.0, 210.0, 420.0])
    base = compute_fwd_returns(px, horizons=[1])["fwd_ret_1d"].to_list()
    lag1 = compute_fwd_returns(px, horizons=[1], exec_lag=1)["fwd_ret_1d"].to_list()

    assert base[0] == pytest.approx(1.00, abs=1e-12)   # 200/100 − 1
    assert lag1[0] == pytest.approx(0.05, abs=1e-12)   # 210/200 − 1
    assert lag1[1] == pytest.approx(1.00, abs=1e-12)   # 420/210 − 1
    assert lag1[2] is None and lag1[3] is None         # 尾部越界


def test_exec_price_col_uses_open():
    """exec_price_col='open' ⇒ 完全走 open 列，close 不参与。

    open = 10/20/25/50，close 故意设成毫不相关的常数 999 ——
    若实现误用了 close，结果会全是 0，测试立刻失败。
    """
    px = _px([999.0, 999.0, 999.0, 999.0], opens=[10.0, 20.0, 25.0, 50.0])
    got = compute_fwd_returns(
        px, horizons=[1], exec_lag=1, exec_price_col="open")["fwd_ret_1d"].to_list()
    assert got[0] == pytest.approx(0.25, abs=1e-12)   # open[2]/open[1] = 25/20
    assert got[1] == pytest.approx(1.00, abs=1e-12)   # open[3]/open[2] = 50/25


def test_horizon_and_lag_compose():
    """h=2 且 exec_lag=1 ⇒ price[t+3]/price[t+1] − 1。"""
    px = _px([1.0, 2.0, 4.0, 6.0, 12.0])
    got = compute_fwd_returns(px, horizons=[2], exec_lag=1)["fwd_ret_2d"].to_list()
    assert got[0] == pytest.approx(2.00, abs=1e-12)   # 6/2 − 1
    assert got[1] == pytest.approx(2.00, abs=1e-12)   # 12/4 − 1


def test_per_code_isolation():
    """shift 必须按 ts_code 分组——跨股票串价会污染边界日。"""
    df = pl.DataFrame({
        "ts_code": ["A", "A", "B", "B"],
        "trade_date": ["2024-01-01", "2024-01-02"] * 2,
        "close": [10.0, 20.0, 100.0, 300.0],
    })
    got = compute_fwd_returns(df, horizons=[1]).sort(["ts_code", "trade_date"])
    v = got["fwd_ret_1d"].to_list()
    assert v[0] == pytest.approx(1.0)   # A: 20/10
    assert v[1] is None                 # A 末日，**不得**借用 B 的价格
    assert v[2] == pytest.approx(2.0)   # B: 300/100
    assert v[3] is None


def test_ret_col_fallback_respects_lag():
    """无价格列时走单日收益复利，exec_lag 须跳过前 lag 步。

    ret = [0.5, 0.1, 0.2, ...]：h=1 且 lag=1 ⇒ 取 ret[t+2] 而非 ret[t+1]。
    """
    df = pl.DataFrame({
        "ts_code": ["X"] * 4,
        "trade_date": [f"2024-01-{i + 1:02d}" for i in range(4)],
        "ret_1d": [0.5, 0.1, 0.2, 0.3],
    })
    got = compute_fwd_returns(df, horizons=[1], exec_lag=1)["fwd_ret_1d"].to_list()
    assert got[0] == pytest.approx(0.2, abs=1e-12)   # ret[2]
    assert got[1] == pytest.approx(0.3, abs=1e-12)   # ret[3]


def test_invalid_args_raise():
    """负 exec_lag、不存在的 exec_price_col 必须显式报错，不得静默。"""
    px = _px([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="exec_lag"):
        compute_fwd_returns(px, horizons=[1], exec_lag=-1)
    with pytest.raises(ValueError, match="exec_price_col"):
        compute_fwd_returns(px, horizons=[1], exec_price_col="nope")


def test_lift_ret_panel_threads_exec_args():
    """`_build_ret_panel` 必须把两个参数透传下去，而不是吞掉。

    用 open 列与 close 列**取值完全不同**的构造，若未透传则结果会等于 close 版。
    """
    from factorzen.discovery.lift_test import _build_ret_panel

    df = pl.DataFrame({
        "ts_code": ["X"] * 4,
        "trade_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "close": [100.0, 100.0, 100.0, 100.0],   # close 恒定 ⇒ close 口径必为 0
        "open_adj": [10.0, 20.0, 25.0, 50.0],
    })
    base = _build_ret_panel(df, horizon=1)
    assert all(v == pytest.approx(0.0) for v in base["ret"].to_list())

    got = _build_ret_panel(df, horizon=1, exec_lag=1, exec_price_col="open_adj")
    vals = got["ret"].to_list()
    assert vals[0] == pytest.approx(0.25, abs=1e-12)   # 25/20
    assert vals[1] == pytest.approx(1.00, abs=1e-12)   # 50/25


# ── 接线层：参数必须真的一路传到底 ────────────────────────────────
# 记忆库有一条老坑「能力层↔接线层漂移」：功能实现完 + 单测绿，但 pipeline
# 没透传，用户用不了。且 `inspect.signature` 断言零判别力——必须从外层出发
# 用**可观测的数值差异**验证。

def test_databundle_threads_exec_args():
    """`DataBundle.build` 必须透传，否则护栏仍在评不可实现的收益。

    close 恒定 ⇒ close 口径的 fwd 必为 0；open 变化 ⇒ 传对了才非 0。
    """
    from factorzen.discovery.scoring import DataBundle

    df = pl.DataFrame({
        "ts_code": ["X"] * 6,
        "trade_date": [f"2024-01-{i + 1:02d}" for i in range(6)],
        "close": [100.0] * 6,
        "close_adj": [100.0] * 6,
        "open_adj": [10.0, 20.0, 25.0, 50.0, 60.0, 70.0],
    })
    base = DataBundle.build(df)
    assert all(v == pytest.approx(0.0)
               for v in base.fwd_returns["fwd_ret_1d"].drop_nulls().to_list())

    got = DataBundle.build(df, exec_lag=1, exec_price_col="open_adj")
    v = got.fwd_returns["fwd_ret_1d"].to_list()
    assert v[0] == pytest.approx(0.25, abs=1e-12)   # open[2]/open[1] = 25/20
    assert v[1] == pytest.approx(1.00, abs=1e-12)   # open[3]/open[2] = 50/25


def test_lift_context_carries_exec_args():
    """`make_lift_context` 必须把口径写进 ctx，供 lift 裁决读取。"""
    from factorzen.discovery.lift_test import make_lift_context

    df = pl.DataFrame({
        "ts_code": ["X", "X"],
        "trade_date": ["2024-01-01", "2024-01-02"],
        "close": [1.0, 2.0],
    })
    d = make_lift_context("ashare", df, prepped=df)
    assert d.exec_lag == 0 and d.exec_price_col is None      # 默认不变

    c = make_lift_context("ashare", df, prepped=df,
                          exec_lag=1, exec_price_col="open_adj")
    assert c.exec_lag == 1 and c.exec_price_col == "open_adj"



def test_session_end_auto_lift_accepts_exec_args():
    """`_session_end_auto_lift` 必须接住口径——它是 lift 裁决的入口。

    ⚠️ 这条来自一次真实漏洞：`make_lift_context` 的调用点**不在**
    `run_team_agent` 里而在本函数，最初误以为同作用域，被 ruff F821 抓到。
    若该函数恰好有同名局部变量，就会**静默传错值**——准入用一个口径、
    lift 裁决用另一个，而测试全绿。故显式钉住签名。
    """
    import inspect

    from factorzen.agents.team_orchestrator import _session_end_auto_lift

    sig = inspect.signature(_session_end_auto_lift)
    assert "exec_lag" in sig.parameters
    assert "exec_price_col" in sig.parameters
    assert sig.parameters["exec_lag"].default == 0
    assert sig.parameters["exec_price_col"].default is None


def test_cli_parser_exposes_exec_flags():
    """`fz mine team` 必须暴露两个旗标，且默认值 = 历史行为。

    从**最外层**（parser）出发验证，这是「能力层↔接线层漂移」的正确测法。
    """
    from factorzen.cli.parser import build_parser

    class _Stub:
        def __getattr__(self, _n):
            return lambda *a, **k: 0

    p = build_parser(_Stub())
    ns = p.parse_args(["mine", "team", "--start", "20200101", "--end", "20201231"])
    assert ns.exec_lag == 0
    assert ns.exec_price_col is None

    ns2 = p.parse_args([
        "mine", "team", "--start", "20200101", "--end", "20201231",
        "--exec-lag", "1", "--exec-price-col", "open_adj",
    ])
    assert ns2.exec_lag == 1
    assert ns2.exec_price_col == "open_adj"

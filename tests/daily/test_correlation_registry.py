"""
test_factor_correlation.py：test_factor_correlation.py：因子相关性矩阵测试。
test_factor_registry.py：test_discovery_factor.py：因子 discovery 相关测试
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from factorzen.daily.evaluation.advanced import (
    compute_factor_crowding,
)
from factorzen.daily.factors.base import DailyFactor
from factorzen.llm.generation import (
    build_agent_messages,
    generate_factor_proposal,
    semantic_check,
)


# ==== 来自 test_factor_correlation.py ====
def _make_factor_df(vals_fn, n: int = 50, n_dates: int = 10, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        dt = (date(2023, 1, 3) + timedelta(days=d)).isoformat()
        vals = rng.standard_normal(n)
        for s in range(n):
            rows.append(
                {
                    "trade_date": dt,
                    "ts_code": f"{s:06d}.SZ",
                    "factor_clean": float(vals_fn(vals, s)),
                }
            )
    return pl.DataFrame(rows)


def test_self_correlation_is_one():
    from factorzen.daily.evaluation.advanced import compute_factor_correlation

    rng = np.random.default_rng(42)
    n = 50
    rows = []
    for d in range(10):
        dt = (date(2023, 1, 3) + timedelta(days=d)).isoformat()
        vals = rng.standard_normal(n)
        for s in range(n):
            rows.append({"trade_date": dt, "ts_code": f"{s:06d}.SZ", "factor_clean": float(vals[s])})
    df = pl.DataFrame(rows)
    corr = compute_factor_correlation({"A": df, "B": df})
    # A vs A = 1, B vs B = 1, and A vs B should be 1 (same data)
    assert abs(corr.filter(pl.col("factor") == "A")["A"][0] - 1.0) < 1e-6
    assert abs(corr.filter(pl.col("factor") == "B")["B"][0] - 1.0) < 1e-6


def test_opposite_factor_correlation_is_negative():
    from factorzen.daily.evaluation.advanced import compute_factor_correlation

    rng = np.random.default_rng(0)
    n = 50
    rows_pos, rows_neg = [], []
    for d in range(10):
        dt = (date(2023, 1, 3) + timedelta(days=d)).isoformat()
        vals = rng.standard_normal(n)
        for s in range(n):
            rows_pos.append(
                {"trade_date": dt, "ts_code": f"{s:06d}.SZ", "factor_clean": float(vals[s])}
            )
            rows_neg.append(
                {"trade_date": dt, "ts_code": f"{s:06d}.SZ", "factor_clean": float(-vals[s])}
            )
    df_pos = pl.DataFrame(rows_pos)
    df_neg = pl.DataFrame(rows_neg)
    corr = compute_factor_correlation({"pos": df_pos, "neg": df_neg})
    corr_val = corr.filter(pl.col("factor") == "pos")["neg"][0]
    assert corr_val < -0.9



def test_single_factor_returns_identity__factor_correlation():
    """单因子应返回 1x1 矩阵，对角线为 1。"""
    from factorzen.daily.evaluation.advanced import compute_factor_correlation

    rng = np.random.default_rng(2)
    n = 20
    rows = []
    for d in range(5):
        dt = (date(2023, 1, 3) + timedelta(days=d)).isoformat()
        vals = rng.standard_normal(n)
        for s in range(n):
            rows.append({"trade_date": dt, "ts_code": f"{s:06d}.SZ", "factor_clean": float(vals[s])})
    df = pl.DataFrame(rows)
    result = compute_factor_correlation({"only": df})
    assert result.filter(pl.col("factor") == "only")["only"][0] == 1.0

# ==== 来自 test_factor_correlation_module.py ====
def _make_df(n: int = 50, n_dates: int = 10, seed: int = 42, negate: bool = False) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        dt = (date(2023, 1, 3) + timedelta(days=d)).isoformat()
        vals = rng.standard_normal(n)
        for s in range(n):
            v = float(-vals[s] if negate else vals[s])
            rows.append({"trade_date": dt, "ts_code": f"{s:06d}.SZ", "factor_clean": v})
    return pl.DataFrame(rows)


def test_single_factor_returns_identity__factor_correlation_module():
    from factorzen.daily.evaluation.correlation import CorrelationResult, compute_factor_correlation

    df = _make_df(n=50, n_dates=5)
    result = compute_factor_correlation({"A": df})
    assert isinstance(result, CorrelationResult)
    assert result.factor_names == ["A"]
    assert result.corr_matrix.shape == (1, 1)
    assert result.corr_matrix[0][0] == 1.0


def test_identical_factors_have_correlation_one():
    from factorzen.daily.evaluation.correlation import compute_factor_correlation

    df = _make_df(n=50, n_dates=10)
    result = compute_factor_correlation({"X": df, "Y": df})
    assert abs(result.corr_matrix[0][1] - 1.0) < 1e-6
    assert abs(result.corr_matrix[1][0] - 1.0) < 1e-6


def test_anti_correlated_factors():
    from factorzen.daily.evaluation.correlation import compute_factor_correlation

    df_pos = _make_df(n=50, n_dates=10, seed=7)
    df_neg = _make_df(n=50, n_dates=10, seed=7, negate=True)
    result = compute_factor_correlation({"pos": df_pos, "neg": df_neg})
    assert result.corr_matrix[0][1] < -0.9


def test_sparse_dates_skipped():
    """Dates with fewer than 30 stocks are skipped; diagonal is still 1."""
    from factorzen.daily.evaluation.correlation import compute_factor_correlation

    # Only 10 stocks — every date will be skipped
    df = _make_df(n=10, n_dates=5)
    result = compute_factor_correlation({"A": df, "B": df})
    assert result.corr_matrix[0][0] == 1.0
    assert result.corr_matrix[1][1] == 1.0


def test_zero_variance_factor_skipped():
    """A constant factor has zero std → that date is skipped; diagonal still 1."""
    from factorzen.daily.evaluation.correlation import compute_factor_correlation

    n = 50
    rows_const = [
        {"trade_date": "2023-01-03", "ts_code": f"{i:06d}.SZ", "factor_clean": 1.0}
        for i in range(n)
    ]
    rng = np.random.default_rng(9)
    rows_rand = [
        {"trade_date": "2023-01-03", "ts_code": f"{i:06d}.SZ", "factor_clean": float(rng.standard_normal())}
        for i in range(n)
    ]
    result = compute_factor_correlation(
        {"const": pl.DataFrame(rows_const), "rand": pl.DataFrame(rows_rand)}
    )
    assert result.corr_matrix[0][0] == 1.0
    assert result.corr_matrix[1][1] == 1.0


def test_non_overlapping_stocks_returns_identity():
    """Inner join on non-overlapping ts_code → empty merged → identity matrix."""
    from factorzen.daily.evaluation.correlation import compute_factor_correlation

    n = 50
    rows_a = [{"trade_date": "2023-01-03", "ts_code": f"A{i:05d}.SZ", "factor_clean": float(i)} for i in range(n)]
    rows_b = [{"trade_date": "2023-01-03", "ts_code": f"B{i:05d}.SZ", "factor_clean": float(i)} for i in range(n)]
    result = compute_factor_correlation({"A": pl.DataFrame(rows_a), "B": pl.DataFrame(rows_b)})
    assert result.corr_matrix[0][0] == 1.0
    assert result.corr_matrix[1][1] == 1.0


def test_diagonal_is_always_one():
    """Diagonal elements must be exactly 1 regardless of off-diagonal values."""
    from factorzen.daily.evaluation.correlation import compute_factor_correlation

    df1 = _make_df(n=60, n_dates=10, seed=1)
    df2 = _make_df(n=60, n_dates=10, seed=2)
    df3 = _make_df(n=60, n_dates=10, seed=3)
    result = compute_factor_correlation({"f1": df1, "f2": df2, "f3": df3})
    assert result.corr_matrix.shape == (3, 3)
    for i in range(3):
        assert result.corr_matrix[i][i] == pytest.approx(1.0)


def test_summary_contains_factor_names_and_values():
    from factorzen.daily.evaluation.correlation import CorrelationResult

    result = CorrelationResult(
        factor_names=["alpha", "beta"],
        corr_matrix=np.array([[1.0, 0.42], [0.42, 1.0]]),
    )
    summary = result.summary()
    assert "alpha" in summary
    assert "beta" in summary
    assert "0.420" in summary


# ==== 来自 test_factor_crowding.py ====
def _make_factor_dict() -> dict[str, pl.DataFrame]:
    """构造多个因子数据的字典。"""
    stocks = [f"s{i}" for i in range(50)]
    base = pl.DataFrame(
        {
            "trade_date": ["2026-01-05"] * 50,
            "ts_code": stocks,
        }
    )
    # 因子 A 和 B 强相关（线性相关），因子 C 独立
    return {
        "momentum": base.with_columns(pl.lit(0.5).alias("factor_clean")),
        "value": base.with_columns(pl.Series("factor_clean", [i / 50 for i in range(50)])),
        "low_vol": base.with_columns(pl.Series("factor_clean", [i / 50 * (-1) for i in range(50)])),
    }




def test_crowding_diagonal_is_one():
    """相关性矩阵对角线为 1.0。"""
    factor_dict = _make_factor_dict()
    result = compute_factor_crowding(factor_dict, factor_col="factor_clean")
    n = len(result.factor_names)
    for i in range(n):
        assert abs(result.corr_matrix[i][i] - 1.0) < 1e-10




# ==== 来自 test_factor_registry.py ====
# ==== 来自 test_discovery_factor.py ====
def _make_daily_lf(n_stocks=8, n_days=60, seed=42) -> pl.LazyFrame:
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for s in [f"{i:06d}.SH" for i in range(n_stocks)]:
        price = 10.0
        for day in days:
            price = float(max(price * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": day, "ts_code": s, "close": price,
                         "open": price, "high": price, "low": price, "pre_close": price,
                         "close_adj": price, "open_adj": price, "high_adj": price, "low_adj": price,
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
                         "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows).lazy()

@dataclass
class MockCtx:
    start: str = "20240301"
    end: str = "20240331"
    required_data: list = field(default_factory=lambda: ["daily", "daily_basic"])
    lookback_days: int = 30
    universe: list | None = None
    snapshot_mode: str = "daily"
    _daily: pl.LazyFrame | None = None
    _basic: pl.LazyFrame | None = None

    @property
    def daily(self) -> pl.LazyFrame:
        return self._daily

    @property
    def daily_basic(self) -> pl.LazyFrame:
        return self._basic if self._basic is not None else pl.DataFrame(
            {"trade_date": [], "ts_code": []}).lazy()

def test_expression_factor_matches_builtin_momentum():
    """pct_change(close, 20) 应与内置 momentum_20d 的 compute 输出一致。"""
    from factorzen.builtin_factors.daily.momentum import Momentum20D
    from factorzen.discovery.factor import ExpressionFactor

    lf = _make_daily_lf()
    ctx = MockCtx(_daily=lf)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builtin = Momentum20D().compute(ctx).sort(["trade_date", "ts_code"])

    mined = ExpressionFactor(expression="pct_change(close, 20)", mined_name="m20",
                             lookback_days=30).compute(ctx).sort(["trade_date", "ts_code"])

    j = builtin.join(mined, on=["trade_date", "ts_code"], suffix="_m")
    assert j.height > 0
    diff = (j["factor_value"] - j["factor_value_m"]).abs().max()
    assert diff is not None and diff < 1e-9

def test_suspended_rows_masked():
    """vol==0（停牌）行价量被置 null → 因子值被过滤；vol>0 行正常产出。
    用零阶表达式 close 使单行即可判别，避免窗口不足导致的 trivial 通过。"""
    from factorzen.discovery.factor import ExpressionFactor

    lf = _make_daily_lf()
    d = date(2024, 3, 15)
    extra = pl.DataFrame({
        "trade_date": [d, d], "ts_code": ["888888.SH", "999999.SH"],
        "close": [5.0, 5.0], "open": [5.0, 5.0], "high": [5.0, 5.0], "low": [5.0, 5.0],
        "pre_close": [5.0, 5.0],
        "close_adj": [5.0, 5.0], "open_adj": [5.0, 5.0], "high_adj": [5.0, 5.0], "low_adj": [5.0, 5.0],
        "amount": [1e6, 0.0], "vol": [1e5, 0.0],
    }).lazy()
    ctx = MockCtx(_daily=pl.concat([lf, extra]))
    out = ExpressionFactor(expression="close", mined_name="x", lookback_days=30).compute(ctx)
    # 停牌股(vol=0)该行被掩码 → 无输出
    sus = out.filter((pl.col("ts_code") == "999999.SH") & (pl.col("trade_date") == d))
    assert sus.height == 0
    # 正常股(vol>0)该行有 close=5.0
    ok = out.filter((pl.col("ts_code") == "888888.SH") & (pl.col("trade_date") == d))
    assert ok.height == 1 and abs(ok["factor_value"][0] - 5.0) < 1e-9

def test_ret_1d_correct_when_ctx_daily_rows_unsorted():
    """compute() 必须先排序(ts_code, trade_date)再派生依赖行序的 ret_1d(shift().over())。

    构造收盘价逐日单调上涨（每天 +1%）但行序被打乱（非 ts_code/trade_date 有序）的数据：
    若 compute() 在排序前就用 shift(1).over("ts_code") 算 ret_1d，会把同一只股票里
    乱序的「上一行」当成「前一交易日」，算出包含负值的错误结果；正确实现下，因为
    收盘价严格单调上涨，每只股票每天的 ret_1d 必须全部是同一个正值 0.01。"""
    from factorzen.discovery.factor import ExpressionFactor

    start = date(2024, 1, 2)
    n_days = 20
    days: list[date] = []
    d = start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)

    rows = []
    for s in ["000001.SH", "000002.SH", "000003.SH"]:
        price = 10.0
        for day in days:
            price *= 1.01  # 严格单调上涨：每天 +1%
            rows.append({
                "trade_date": day, "ts_code": s,
                "close": price, "open": price, "high": price, "low": price, "pre_close": price,
                "close_adj": price, "open_adj": price, "high_adj": price, "low_adj": price,
                "amount": 1e7, "vol": 1e5,
            })

    # 行序打乱（固定 seed 可复现）：不再是 (ts_code, trade_date) 有序
    daily_df = pl.DataFrame(rows).sample(fraction=1.0, shuffle=True, seed=7)
    per_stock = daily_df.filter(pl.col("ts_code") == "000001.SH")["trade_date"].to_list()
    assert per_stock != sorted(per_stock), "fixture 未真正打乱行序，测试无法复现 bug"

    ctx = MockCtx(_daily=daily_df.lazy(), start="20240101")
    out = ExpressionFactor(expression="ret_1d", mined_name="r1", lookback_days=5).compute(ctx)

    assert out.height > 0
    assert (out["factor_value"] > 0).all(), "收盘价逐日单调上涨，ret_1d 必须全部为正"
    assert (out["factor_value"] - 0.01).abs().max() < 1e-9

# ==== 来自 test_factor_class_attr_declaration.py ====
class _PlainDeclared(DailyFactor):
    """按 workspace/factors/*/TEMPLATE.md 教的写法声明——无注解的类属性。"""

    name = "plain_declared_probe"
    category = "weekly"
    frequency = "weekly"
    lookback_days = 30
    description = "探针因子"

    def compute(self, ctx: object) -> pl.DataFrame:  # pragma: no cover - 不求值
        return pl.DataFrame()

def test_plain_class_attrs_survive_instantiation():
    probe = _PlainDeclared()
    assert probe.lookback_days == 30, "子类声明的 lookback_days 被基类默认值覆盖"
    assert probe.frequency == "weekly", "子类声明的 frequency 被基类默认值覆盖"
    assert probe.category == "weekly"
    assert probe.name == "plain_declared_probe"

def test_no_daily_factor_loses_its_declaration():
    """全量守卫：任何内置日频因子的类声明都不得在实例化时丢失。"""
    from factorzen.daily.factors.registry import get_factor, list_factors

    drifted: list[str] = []
    for name in list_factors():
        cls = get_factor(name)
        if not (isinstance(cls, type) and issubclass(cls, DailyFactor)):
            continue
        try:
            inst = cls()
        except TypeError:  # 需要构造参数的因子不在本守卫范围
            continue
        for attr in ("lookback_days", "frequency", "category"):
            declared = getattr(cls, attr, None)
            actual = getattr(inst, attr, None)
            if declared != actual:
                drifted.append(f"{cls.__name__}.{attr}: 声明 {declared!r} → 实例 {actual!r}")

    assert not drifted, "以下因子的类属性声明在实例化时丢失:\n" + "\n".join(drifted)

# ==== 来自 test_finance_factor_required_data.py ====
def test_finance_monthly_factors_declare_finance_and_daily():
    from factorzen.builtin_factors.monthly.asset_growth import AssetGrowthMonthly
    from factorzen.builtin_factors.monthly.profitability import RoeYtdMonthly

    for cls in (AssetGrowthMonthly, RoeYtdMonthly):
        rd = cls.required_data
        assert "finance" in rd, f"{cls.name} compute 读 finance parquet，应声明 finance"
        assert "daily" in rd, f"{cls.name} pipeline 需 ctx.daily 算前向收益，应声明 daily"
        assert "daily_basic" not in rd, f"{cls.name} 从不读 daily_basic，不应声明"

def test_roe_factor_honestly_labeled_ytd_not_ttm():
    """诚实标注：该因子是 YTD 累计口径，不应叫 roe_ttm 或在 description 声称 TTM。"""
    from factorzen.builtin_factors.monthly.profitability import RoeYtdMonthly

    assert RoeYtdMonthly.name == "roe_ytd"
    assert "TTM" not in RoeYtdMonthly.description or "非 TTM" in RoeYtdMonthly.description

# ==== 来自 test_agent_generation.py ====
class FakeLLM:
    """确定性 LLMFn：按调用顺序返回预设字符串。"""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self._responses.pop(0) if self._responses else "{}"

def test_generate_factor_proposal_parses_json():
    raw = json.dumps(
        {
            "hypothesis": "低换手反转",
            "expressions": ["rank(close)", "ts_mean(vol,5)"],
            "rationale": "...",
        }
    )
    llm = FakeLLM([raw])
    props = generate_factor_proposal([{"role": "user", "content": "x"}], llm, n_hypotheses=1)
    assert len(props) == 1
    assert props[0].hypothesis == "低换手反转"
    assert props[0].expressions == ["rank(close)", "ts_mean(vol,5)"]

def test_generate_factor_proposal_tolerates_garbage():
    # 非 JSON → 返回空列表（降级，不抛）
    llm = FakeLLM(["这不是 JSON"])
    props = generate_factor_proposal([{"role": "user", "content": "x"}], llm)
    assert props == []

def test_generate_extracts_json_substring():
    # JSON 嵌在自然语言里 → 提取首个 {...}
    raw = '好的，这是我的提议：{"hypothesis":"h","expressions":["rank(close)"],"rationale":"r"} 完毕'
    llm = FakeLLM([raw])
    props = generate_factor_proposal([{"role": "user", "content": "x"}], llm)
    assert props and props[0].expressions == ["rank(close)"]

def test_semantic_check_yes_no():
    llm = FakeLLM(
        [
            json.dumps({"consistent": True, "reason": "对齐"}),
            json.dumps({"consistent": False, "reason": "表达式与假设无关"}),
        ]
    )
    ok1, _ = semantic_check("动量", "ts_mean(close,20)", llm)
    ok2, reason2 = semantic_check("动量", "rank(pb)", llm)
    assert ok1 is True and ok2 is False and reason2

def test_build_agent_messages_lists_ops_and_leaves():
    msgs = build_agent_messages(
        op_names=["ts_mean", "rank", "div"],
        leaf_names=["close", "vol", "pb"],
        feedback="上轮 IC 偏低",
        negatives=["rank(close)"],
    )
    blob = " ".join(m["content"] for m in msgs)
    assert "ts_mean" in blob and "close" in blob  # 算子/特征清单进 prompt
    assert "rank(close)" in blob  # Negative RAG 负例进 prompt
    assert any(m["role"] == "system" for m in msgs)


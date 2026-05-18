# Phase 5: Intraday 真实分钟数据路径打通

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 打通 intraday 真实分钟数据路径：拉取 Tushare 分钟线 → 计算前向收益 → 分层回测 → 完整 Tear Sheet 报告，与 daily 管线对齐。

**Architecture:** 修复 `fetch_minute` 逐股缓存检测逻辑，新增批量拉取脚本；在 `intraday/evaluation/` 补充前向收益计算和日内回测（将日内因子聚合到日频后复用 `run_stratified_backtest`）；新增第一个真实因子 VWAP 偏离度；最后更新 `run_intraday_single.py` 接入完整评估流。

**Tech Stack:** polars, tushare `pro.stk_mins`, 复用 `daily/evaluation/backtest.BacktestResult` + `run_stratified_backtest`，复用 `reporting/tear_sheet.generate_tear_sheet`

---

## 文件清单

| 文件 | 动作 | 说明 |
|---|---|---|
| `common/loader.py` | 修改 | 修复 `fetch_minute` 逐股缓存检测（第 326 行）|
| `scripts/fetch_minute_data.py` | 新建 | 批量拉取 CSI300/全量分钟线的 CLI 脚本 |
| `intraday/evaluation/returns.py` | 新建 | `compute_intraday_fwd_returns()` |
| `intraday/evaluation/backtest.py` | 新建 | `run_intraday_backtest()` |
| `intraday/factors/technical/__init__.py` | 新建 | 空 |
| `intraday/factors/technical/vwap_deviation.py` | 新建 | `VwapDeviation` 因子 |
| `intraday/factors/registry.py` | 修改 | 扫描 `technical` 子包 |
| `scripts/run_intraday_single.py` | 修改 | 接入 backtest + turnover + 完整 tearsheet |
| `tests/test_intraday_returns.py` | 新建 | 前向收益计算测试 |
| `tests/test_intraday_backtest.py` | 新建 | 日内回测测试 |
| `tests/test_intraday_vwap_factor.py` | 新建 | VWAP 因子测试 |

---

## Task 1: 修复 fetch_minute 逐股缓存检测

**问题：** `loader.py:326` 的 `partition_exists("minute", year, month)` 只检查分区文件是否存在，不检查当前 ts_code 是否已在分区中。第一只股票写入后，后续所有股票都会被跳过。

**Files:**
- Modify: `common/loader.py:326-329`
- Test: `tests/test_storage.py`（已有，验证 append 逻辑）

- [ ] **Step 1: 确认 bug 存在**

```bash
cd E:\code\量化研究\因子研究
pixi run python -c "
from common.storage import partition_exists, DATA_RAW
# 创建假的 minute 分区，模拟 bug
import polars as pl; from pathlib import Path
p = DATA_RAW / 'minute' / 'year=2025' / 'month=01'; p.mkdir(parents=True, exist_ok=True)
pl.DataFrame({'ts_code': ['000001.SZ'], 'trade_time': [None], 'close': [1.0]}).write_parquet(p / 'data.parquet')
print('partition_exists:', partition_exists('minute', 2025, 1))  # 应为 True
# 清理
import shutil; shutil.rmtree(str(DATA_RAW / 'minute'), ignore_errors=True)
"
```

Expected: `partition_exists: True` — 这意味着第二只股票会被跳过

- [ ] **Step 2: 修改 `common/loader.py`**

将 `fetch_minute` 内 `partition_exists` 的 skip 逻辑（第 326-329 行）替换为：

```python
        # 已有分区时，检查该 ts_code 是否已在其中（逐股追加场景）
        if partition_exists("minute", year, month):
            from config.settings import DATA_RAW as _DR
            _fp = _DR / "minute" / f"year={year}" / f"month={month:02d}" / "data.parquet"
            _existing_codes = pl.read_parquet(_fp, columns=["ts_code"])["ts_code"].unique().to_list()
            if ts_code in _existing_codes:
                logger.info(f"[minute] {ts_code} {year}-{month:02d} 已缓存，跳过")
                current = next_month
                continue
```

- [ ] **Step 3: 运行现有测试，确保无回退**

```bash
cd E:\code\量化研究\因子研究 && pixi run test
```

Expected: `133 passed`

- [ ] **Step 4: Commit**

```bash
git add common/loader.py
git commit -m "fix: fetch_minute per-ts_code cache check to support batch append"
```

---

## Task 2: 批量分钟数据拉取脚本

**Files:**
- Create: `scripts/fetch_minute_data.py`

- [ ] **Step 1: 创建脚本**

```python
"""批量拉取 CSI300（或自定义）股票分钟线数据。

用法:
  pixi run python scripts/fetch_minute_data.py --start 20260101 --end 20260516
  pixi run python scripts/fetch_minute_data.py --start 20260101 --end 20260516 --universe csi300 --freq 1min
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.loader import fetch_minute
from common.logger import get_logger, setup_logging
from common.universe import get_universe

setup_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="批量拉取分钟线数据")
    parser.add_argument("--start", required=True, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", required=True, help="截止日期 YYYYMMDD")
    parser.add_argument("--universe", default="csi300", help="股票池名称（csi300 / csi800）")
    parser.add_argument("--freq", default="1min", help="分钟频率：1min / 5min / 15min / 30min / 60min")
    parser.add_argument("--delay", type=float, default=0.5, help="每只股票间隔秒数（防限流）")
    args = parser.parse_args()

    universe = get_universe(args.end, args.universe)
    if universe.is_empty():
        logger.error(f"股票池为空: {args.universe}")
        sys.exit(1)
    ts_codes = universe["ts_code"].to_list()
    logger.info(f"股票池 {args.universe}: {len(ts_codes)} 只，准备拉取 {args.freq} 分钟线 {args.start}~{args.end}")

    ok, fail = 0, 0
    for i, ts_code in enumerate(ts_codes, 1):
        logger.info(f"[{i}/{len(ts_codes)}] {ts_code}")
        try:
            fetch_minute(ts_code, args.freq, args.start, args.end)
            ok += 1
        except Exception as e:
            logger.error(f"  {ts_code} 失败: {e}")
            fail += 1
        if args.delay > 0 and i < len(ts_codes):
            time.sleep(args.delay)

    logger.info(f"完成: 成功 {ok} 只，失败 {fail} 只")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 冒烟测试（不需要 Tushare，只验证导入和帮助）**

```bash
cd E:\code\量化研究\因子研究 && pixi run python scripts/fetch_minute_data.py --help
```

Expected: 打印参数说明，无报错

- [ ] **Step 3: Commit**

```bash
git add scripts/fetch_minute_data.py
git commit -m "feat: add fetch_minute_data.py batch pull script for intraday data"
```

---

## Task 3: 日内前向收益计算

**Files:**
- Create: `intraday/evaluation/returns.py`
- Create: `tests/test_intraday_returns.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_intraday_returns.py`：

```python
"""tests/test_intraday_returns.py"""
import polars as pl
import pytest

from intraday.evaluation.returns import compute_intraday_fwd_returns


def _make_minute_df() -> pl.DataFrame:
    """3 只股票，每只 10 根 bar。"""
    import datetime
    rows = []
    base_time = datetime.datetime(2026, 5, 16, 9, 30, 0)
    for ts in ["000001.SZ", "000002.SZ", "000003.SZ"]:
        for i in range(10):
            rows.append({
                "ts_code": ts,
                "trade_time": base_time + datetime.timedelta(minutes=i),
                "close": 10.0 + i * 0.1,
            })
    return pl.DataFrame(rows).with_columns(
        pl.col("trade_time").cast(pl.Datetime)
    )


def test_fwd_returns_columns():
    df = compute_intraday_fwd_returns(_make_minute_df(), periods=[1, 5])
    assert "fwd_ret_1bar" in df.columns
    assert "fwd_ret_5bar" in df.columns


def test_fwd_ret_1bar_last_row_is_null():
    df = compute_intraday_fwd_returns(_make_minute_df(), periods=[1])
    last_rows = df.filter(pl.col("ts_code") == "000001.SZ").sort("trade_time").tail(1)
    assert last_rows["fwd_ret_1bar"][0] is None


def test_fwd_ret_1bar_value():
    df = compute_intraday_fwd_returns(_make_minute_df(), periods=[1])
    row = df.filter(
        (pl.col("ts_code") == "000001.SZ") & (pl.col("trade_time").dt.minute() == 30)
    )
    expected = (10.1 - 10.0) / 10.0
    assert abs(row["fwd_ret_1bar"][0] - expected) < 1e-9


def test_no_cross_stock_leakage():
    df = compute_intraday_fwd_returns(_make_minute_df(), periods=[1])
    # 每只股票的最后一根 bar 前向收益应为 null（不借用下一只股票数据）
    for ts in ["000001.SZ", "000002.SZ", "000003.SZ"]:
        last_val = (
            df.filter(pl.col("ts_code") == ts)
            .sort("trade_time")
            .tail(1)["fwd_ret_1bar"][0]
        )
        assert last_val is None, f"{ts} 最后一行应为 null，实为 {last_val}"
```

- [ ] **Step 2: 运行确认失败**

```bash
cd E:\code\量化研究\因子研究 && pixi run python -m pytest tests/test_intraday_returns.py -v
```

Expected: `ImportError: cannot import name 'compute_intraday_fwd_returns'`

- [ ] **Step 3: 实现 `intraday/evaluation/returns.py`**

```python
"""intraday/evaluation/returns.py — 日内前向收益计算。"""

from __future__ import annotations

import polars as pl


def compute_intraday_fwd_returns(
    minute_df: pl.DataFrame,
    periods: list[int] | None = None,
    close_col: str = "close",
    time_col: str = "trade_time",
    code_col: str = "ts_code",
) -> pl.DataFrame:
    """计算分钟级前向收益。

    Args:
        minute_df: 含 trade_time、ts_code、close 的分钟线 DataFrame。
        periods: 前向 bar 数列表，默认 [1, 5, 15, 60]。
        close_col: 价格列名。
        time_col: 时间列名。
        code_col: 股票代码列名。

    Returns:
        原 DataFrame 追加 fwd_ret_{N}bar 列（末尾 N 行每股为 null）。
    """
    if periods is None:
        periods = [1, 5, 15, 60]

    df = minute_df.sort([code_col, time_col])
    for n in periods:
        future_close = pl.col(close_col).shift(-n).over(code_col)
        df = df.with_columns(
            (future_close / pl.col(close_col) - 1).alias(f"fwd_ret_{n}bar")
        )
    return df
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd E:\code\量化研究\因子研究 && pixi run python -m pytest tests/test_intraday_returns.py -v
```

Expected: `4 passed`

- [ ] **Step 5: 更新 `intraday/evaluation/__init__.py`（如有需要）并运行全量测试**

```bash
cd E:\code\量化研究\因子研究 && pixi run test
```

Expected: `≥133 passed`

- [ ] **Step 6: Commit**

```bash
git add intraday/evaluation/returns.py tests/test_intraday_returns.py
git commit -m "feat: add compute_intraday_fwd_returns for intraday forward return calculation"
```

---

## Task 4: 日内分层回测

**策略：** 将日内因子值聚合到日频（取每日收盘前最后一个有效因子值），然后复用 `daily/evaluation/backtest.run_stratified_backtest` 做日频分层回测。

**Files:**
- Create: `intraday/evaluation/backtest.py`
- Create: `tests/test_intraday_backtest.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_intraday_backtest.py`：

```python
"""tests/test_intraday_backtest.py"""
import datetime
import polars as pl
import pytest

from intraday.evaluation.backtest import (
    aggregate_intraday_factor,
    run_intraday_backtest,
)
from daily.evaluation.backtest import BacktestResult


def _make_minute_factor(n_stocks: int = 5, n_days: int = 10, bars_per_day: int = 20) -> pl.DataFrame:
    """合成分钟级因子数据：n_stocks 只股票 × n_days 天 × bars_per_day 根 bar。"""
    import random
    random.seed(42)
    rows = []
    for day in range(n_days):
        trade_date = (datetime.date(2026, 1, 2) + datetime.timedelta(days=day)).strftime("%Y%m%d")
        base_time = datetime.datetime(2026, 1, 2 + day, 9, 30)
        for ts in [f"00000{i}.SZ" for i in range(1, n_stocks + 1)]:
            for b in range(bars_per_day):
                rows.append({
                    "trade_date": trade_date,
                    "trade_time": base_time + datetime.timedelta(minutes=b),
                    "ts_code": ts,
                    "factor_clean": random.gauss(0, 1),
                })
    return pl.DataFrame(rows).with_columns(
        pl.col("trade_time").cast(pl.Datetime),
        pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d"),
    )


def _make_daily_ret(n_stocks: int = 5, n_days: int = 10) -> pl.DataFrame:
    import random
    random.seed(0)
    rows = []
    for day in range(n_days):
        trade_date = (datetime.date(2026, 1, 2) + datetime.timedelta(days=day)).strftime("%Y%m%d")
        for ts in [f"00000{i}.SZ" for i in range(1, n_stocks + 1)]:
            rows.append({"trade_date": trade_date, "ts_code": ts, "ret": random.gauss(0, 0.02)})
    return pl.DataFrame(rows).with_columns(
        pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d")
    )


def test_aggregate_returns_daily_rows():
    minute_factor = _make_minute_factor()
    daily = aggregate_intraday_factor(minute_factor)
    assert "trade_date" in daily.columns
    assert "ts_code" in daily.columns
    assert "factor_clean" in daily.columns
    # 每天每只股票只有 1 行
    assert len(daily) == len(minute_factor["trade_date"].unique()) * len(minute_factor["ts_code"].unique())


def test_run_intraday_backtest_returns_result():
    minute_factor = _make_minute_factor()
    daily_ret = _make_daily_ret()
    result = run_intraday_backtest(minute_factor, daily_ret, n_groups=5)
    assert isinstance(result, BacktestResult)
    assert result.n_groups == 5
    assert not result.nav.is_empty()


def test_run_intraday_backtest_has_long_short():
    minute_factor = _make_minute_factor()
    daily_ret = _make_daily_ret()
    result = run_intraday_backtest(minute_factor, daily_ret, n_groups=5)
    assert "long_short" in result.summary_stats
```

- [ ] **Step 2: 运行确认失败**

```bash
cd E:\code\量化研究\因子研究 && pixi run python -m pytest tests/test_intraday_backtest.py -v
```

Expected: `ImportError`

- [ ] **Step 3: 实现 `intraday/evaluation/backtest.py`**

```python
"""intraday/evaluation/backtest.py — 日内因子分层回测（聚合到日频后复用 daily 回测框架）。"""

from __future__ import annotations

import polars as pl

from daily.evaluation.backtest import BacktestResult, run_stratified_backtest


def aggregate_intraday_factor(
    minute_factor: pl.DataFrame,
    factor_col: str = "factor_clean",
    time_col: str = "trade_time",
    date_col: str = "trade_date",
    code_col: str = "ts_code",
) -> pl.DataFrame:
    """将分钟级因子聚合到日频（取每日最后一个有效因子值）。

    Args:
        minute_factor: 含 trade_time/trade_date、ts_code、factor_col 的分钟 DataFrame。
        factor_col: 因子列名。
        time_col: 时间列名。
        date_col: 日期列名（若已存在）或从 time_col 提取。
        code_col: 股票代码列名。

    Returns:
        日频 DataFrame，列：trade_date, ts_code, {factor_col}。
    """
    df = minute_factor.sort([code_col, time_col])

    # 若没有 trade_date 列，从 trade_time 提取
    if date_col not in df.columns:
        df = df.with_columns(pl.col(time_col).dt.date().alias(date_col))

    # 每日每股取最后一个非 null 因子值
    return (
        df.filter(pl.col(factor_col).is_not_null())
        .group_by([date_col, code_col])
        .agg(pl.col(factor_col).last())
        .sort([date_col, code_col])
    )


def run_intraday_backtest(
    minute_factor: pl.DataFrame,
    daily_ret: pl.DataFrame,
    factor_col: str = "factor_clean",
    n_groups: int = 10,
    factor_name: str = "",
) -> BacktestResult:
    """日内因子分层回测。

    将分钟因子聚合到日频后，对齐日频收益进行分层回测。

    Args:
        minute_factor: 分钟级因子 DataFrame，含 trade_time/trade_date、ts_code、{factor_col}。
        daily_ret: 日频收益 DataFrame，含 trade_date、ts_code、ret。
        factor_col: 因子列名。
        n_groups: 分组数。
        factor_name: 因子名称（显示用）。

    Returns:
        BacktestResult（复用 daily 框架）。
    """
    daily_factor = aggregate_intraday_factor(minute_factor, factor_col=factor_col)
    return run_stratified_backtest(
        daily_factor,
        daily_ret,
        factor_col=factor_col,
        n_groups=n_groups,
        factor_name=factor_name,
    )
```

- [ ] **Step 4: 运行测试**

```bash
cd E:\code\量化研究\因子研究 && pixi run python -m pytest tests/test_intraday_backtest.py -v
```

Expected: `3 passed`

- [ ] **Step 5: 运行全量测试**

```bash
cd E:\code\量化研究\因子研究 && pixi run test
```

Expected: `≥137 passed`

- [ ] **Step 6: Commit**

```bash
git add intraday/evaluation/backtest.py tests/test_intraday_backtest.py
git commit -m "feat: add intraday backtest via daily-aggregation approach"
```

---

## Task 5: VWAP 偏离度因子

**因子定义：** `factor = (close - vwap) / vwap`，其中 `vwap = cumsum(amount) / cumsum(vol)`（当日内累计均价，每根 bar 更新）。

**Files:**
- Create: `intraday/factors/technical/__init__.py`
- Create: `intraday/factors/technical/vwap_deviation.py`
- Modify: `intraday/factors/registry.py`（添加 `technical` 扫描包）
- Create: `tests/test_intraday_vwap_factor.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_intraday_vwap_factor.py`：

```python
"""tests/test_intraday_vwap_factor.py"""
import datetime
import polars as pl
import pytest

from intraday.factors.technical.vwap_deviation import VwapDeviation
from intraday.data.context import MFTDataContext


def _make_ctx_with_data(df: pl.DataFrame):
    """构造一个 MFTDataContext，其 minute 属性返回指定 LazyFrame。"""
    import unittest.mock as mock
    ctx = mock.MagicMock(spec=MFTDataContext)
    ctx.minute = df.lazy()
    return ctx


def _make_minute_df(n_bars: int = 20) -> pl.DataFrame:
    base = datetime.datetime(2026, 5, 16, 9, 30)
    rows = []
    for ts in ["000001.SZ", "000002.SZ"]:
        for i in range(n_bars):
            rows.append({
                "ts_code": ts,
                "trade_time": base + datetime.timedelta(minutes=i),
                "close": 10.0 + i * 0.05,
                "vol": 1000.0 + i * 10,
                "amount": (10.0 + i * 0.05) * (1000.0 + i * 10),
            })
    return pl.DataFrame(rows).with_columns(pl.col("trade_time").cast(pl.Datetime))


def test_vwap_deviation_columns():
    factor = VwapDeviation()
    ctx = _make_ctx_with_data(_make_minute_df())
    result = factor.compute(ctx)
    assert "trade_time" in result.columns
    assert "ts_code" in result.columns
    assert "factor_value" in result.columns


def test_vwap_deviation_no_cross_stock():
    """不同股票的 VWAP 不互相污染。"""
    factor = VwapDeviation()
    ctx = _make_ctx_with_data(_make_minute_df())
    result = factor.compute(ctx)
    assert set(result["ts_code"].unique().to_list()) == {"000001.SZ", "000002.SZ"}


def test_vwap_deviation_first_bar_near_zero():
    """第一根 bar 的 close == vwap（因为 cumsum 只有当前 bar），偏离为 0。"""
    factor = VwapDeviation()
    ctx = _make_ctx_with_data(_make_minute_df())
    result = factor.compute(ctx)
    first_bar = (
        result.filter(pl.col("ts_code") == "000001.SZ")
        .sort("trade_time")
        .head(1)["factor_value"][0]
    )
    assert abs(first_bar) < 1e-9


def test_vwap_deviation_registered():
    from intraday.factors.registry import get_factor
    factor_cls = get_factor("vwap_deviation")
    assert factor_cls is VwapDeviation
```

- [ ] **Step 2: 运行确认失败**

```bash
cd E:\code\量化研究\因子研究 && pixi run python -m pytest tests/test_intraday_vwap_factor.py -v
```

Expected: `ImportError`

- [ ] **Step 3: 创建 `intraday/factors/technical/__init__.py`**

```python
# intraday/factors/technical/__init__.py
```

- [ ] **Step 4: 实现 `intraday/factors/technical/vwap_deviation.py`**

```python
"""intraday/factors/technical/vwap_deviation.py — VWAP 偏离度因子。

定义：factor_value = (close - vwap) / vwap
其中 vwap = cumsum(amount) / cumsum(vol)（当日内从开市起累计）。

直觉：
- factor > 0：当前价高于日内均价，短期均值回归预期为负向信号（反转因子）
- factor < 0：当前价低于日内均价，做多候选
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from intraday.data.context import MFTDataContext
from intraday.factors.base import MFTFactor


@dataclass
class VwapDeviation(MFTFactor):
    name: str = "vwap_deviation"
    description: str = "当前价格与日内 VWAP 的偏离度"
    bar_size: str = "1min"
    lookback_bars: int = 0  # 无需额外 lookback，仅当日数据
    required_data: list[str] = field(default_factory=lambda: ["minute"])

    def compute(self, ctx: MFTDataContext) -> pl.DataFrame:
        df = ctx.minute.collect()
        if df.is_empty():
            return pl.DataFrame(
                schema={"trade_time": pl.Datetime, "ts_code": pl.Utf8, "factor_value": pl.Float64}
            )

        df = df.sort(["ts_code", "trade_time"])

        # 当日内累计 amount / vol = VWAP（每根 bar 末尾更新）
        df = df.with_columns([
            pl.col("amount").cum_sum().over(["ts_code", pl.col("trade_time").dt.date()]).alias("_cum_amount"),
            pl.col("vol").cum_sum().over(["ts_code", pl.col("trade_time").dt.date()]).alias("_cum_vol"),
        ]).with_columns(
            (pl.col("_cum_amount") / pl.col("_cum_vol")).alias("_vwap")
        ).with_columns(
            ((pl.col("close") - pl.col("_vwap")) / pl.col("_vwap")).alias("factor_value")
        ).select(["trade_time", "ts_code", "factor_value"])

        return df.filter(pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
```

- [ ] **Step 5: 修改 `intraday/factors/registry.py`，添加 technical 扫描包**

当前文件中找到 `scan_packages` 参数，追加 `"intraday.factors.technical"`：

```python
_registry = FactorRegistry(
    base_cls=MFTFactor,
    scan_packages=["intraday.factors.demo", "intraday.factors.technical"],
)
```

- [ ] **Step 6: 运行测试**

```bash
cd E:\code\量化研究\因子研究 && pixi run python -m pytest tests/test_intraday_vwap_factor.py -v
```

Expected: `4 passed`

- [ ] **Step 7: 运行全量测试**

```bash
cd E:\code\量化研究\因子研究 && pixi run test
```

Expected: `≥141 passed`

- [ ] **Step 8: Commit**

```bash
git add intraday/factors/technical/__init__.py intraday/factors/technical/vwap_deviation.py intraday/factors/registry.py tests/test_intraday_vwap_factor.py
git commit -m "feat: add VwapDeviation intraday factor + technical factor package"
```

---

## Task 6: 更新 run_intraday_single.py — 完整评估流

**改动目标：**
1. 真实数据路径：接入 `compute_intraday_fwd_returns` + IC 分析
2. 新增回测步骤：接入 `run_intraday_backtest` + `compute_turnover`（聚合日频后）
3. 持久化中间结果：parquet 落盘
4. 报告：复用 `reporting/tear_sheet.generate_tear_sheet`（需要适配 `IntradayICResult → ICAnalysisResult` 字段映射）

**Files:**
- Modify: `scripts/run_intraday_single.py`

- [ ] **Step 1: 读当前文件完整内容（了解现有 `_load_real_data` 和 `_render_html` 实现）**

```bash
cd E:\code\量化研究\因子研究 && head -100 scripts/run_intraday_single.py
```

- [ ] **Step 2: 更新脚本**

将 `scripts/run_intraday_single.py` 的 `main()` 修改为完整流水线（保留 `--demo` 路径，新增真实数据完整路径）：

```python
    # ── 7. 日内前向收益 ──（--demo 路径已内联；真实路径需重新加载 minute 数据）
    if not args.demo:
        from intraday.evaluation.returns import compute_intraday_fwd_returns
        minute_df = ctx.minute.collect()
        minute_with_ret = compute_intraday_fwd_returns(minute_df, periods=[1, 5])
    else:
        # demo 路径：合成数据已有 fwd_ret_1bar
        minute_with_ret = factor_df  # _make_demo_data() 返回已含前向收益

    # ── 8. IC 分析（已有）──
    ic_result = compute_intraday_rank_ic(
        factor_df=clean_df,
        ret_df=minute_with_ret,
        factor_col="factor_clean",
        ret_col="fwd_ret_1bar",
    )
    ic_result_obj = ic_result  # IntradayICResult

    # ── 9. 分层回测 ──
    from intraday.evaluation.backtest import run_intraday_backtest
    from daily.evaluation.ic_analysis import compute_fwd_returns as _daily_fwd
    from common.loader import fetch_daily
    from common.storage import load_parquet as _load_pq

    bt_result = None
    try:
        fetch_daily(args.start, args.end)
        daily_raw = _load_pq("daily", start=args.start, end=args.end).collect()
        if not daily_raw.is_empty():
            import polars as pl as _pl
            daily_ret = daily_raw.select(["trade_date", "ts_code", "close"]).sort(["ts_code", "trade_date"])
            daily_ret = daily_ret.with_columns(
                (_pl.col("close") / _pl.col("close").shift(1).over("ts_code") - 1).alias("ret")
            ).select(["trade_date", "ts_code", "ret"])
            bt_result = run_intraday_backtest(
                clean_df, daily_ret, n_groups=5, factor_name=factor.name
            )
            logger.info(f"\n{bt_result.summary()}")
    except Exception as e:
        logger.warning(f"回测失败（可跳过）: {e}")

    # ── 10. 换手率（聚合日频后）──
    to_result = None
    try:
        from intraday.evaluation.backtest import aggregate_intraday_factor
        from daily.evaluation.turnover import compute_turnover
        daily_factor_df = aggregate_intraday_factor(clean_df)
        to_result = compute_turnover(daily_factor_df, factor_col="factor_clean", frequency="daily")
        to_result.factor_name = factor.name
        logger.info(f"\n{to_result.summary()}")
    except Exception as e:
        logger.warning(f"换手率计算失败（可跳过）: {e}")

    # ── 11. 适配 ICAnalysisResult 接口 → generate_tear_sheet ──
    from daily.evaluation.ic_analysis import ICAnalysisResult
    import polars as pl
    adapted_ic = ICAnalysisResult(
        factor_name=factor.name,
        ic_mean=ic_result_obj.ic_mean,
        ic_std=ic_result_obj.ic_std,
        ir=ic_result_obj.ir,
        ic_positive_ratio=ic_result_obj.ic_positive_ratio,
        n_periods=ic_result_obj.n_periods,
        ic_series=ic_result_obj.daily_ic.rename({"ic_mean": "ic"}),
        frequency="daily",
    )
```

> **注意：** 以上为核心逻辑片段。将 Step 2 实际执行时，读取完整文件后做**最小侵入式修改**，不破坏现有 `--demo` 路径和 HTML 落盘逻辑。

- [ ] **Step 3: 语法验证**

```bash
cd E:\code\量化研究\因子研究 && pixi run python -c "import scripts.run_intraday_single" 2>&1 | head -5
```

Expected: 无 SyntaxError（可能有 ImportError，因为数据依赖）

- [ ] **Step 4: Demo 模式冒烟**

```bash
cd E:\code\量化研究\因子研究 && pixi run python scripts/run_intraday_single.py --factor momentum_1min --start 20260401 --end 20260430 --demo
```

Expected: 无崩溃，`output/intraday/reports/` 下有 HTML 文件

- [ ] **Step 5: 全量测试（确保原有测试不回退）**

```bash
cd E:\code\量化研究\因子研究 && pixi run test
```

Expected: `≥141 passed`

- [ ] **Step 6: Commit**

```bash
git add scripts/run_intraday_single.py
git commit -m "feat: run_intraday_single full pipeline (backtest + turnover + tearsheet)"
```

---

## Task 7: 拉取真实数据并端到端验证

> **前置：** 需要 Tushare token，且 `data/raw/minute/` 当前为空。

- [ ] **Step 1: 拉取 1 个月真实分钟数据（小规模测试）**

以 CSI300 中的 5 只股票为例先测试（避免耗尽 API 限额）：

```bash
cd E:\code\量化研究\因子研究
pixi run python -c "
from common.loader import fetch_minute
for code in ['000001.SZ', '000002.SZ', '000858.SZ', '600519.SH', '601318.SH']:
    print(f'Fetching {code}...')
    df = fetch_minute(code, '1min', '20260401', '20260430')
    print(f'  {code}: {len(df)} bars')
"
```

Expected: 每只股票约 4000-5000 行（每日约 241 根 1min bar）

- [ ] **Step 2: 验证 `data/raw/minute/` 有数据**

```bash
ls "E:\code\量化研究\因子研究\data\raw\minute"
```

Expected: 出现 `year=2026` 目录

- [ ] **Step 3: 运行真实数据 IC 评估（momentum_1min，CSI300）**

```bash
cd E:\code\量化研究\因子研究
pixi run python scripts/run_intraday_single.py --factor momentum_1min --start 20260401 --end 20260430 --universe csi300
```

Expected: 日志显示 IC 计算完成，`output/intraday/reports/` 下有 HTML

- [ ] **Step 4: 运行 VWAP 偏离度因子**

```bash
cd E:\code\量化研究\因子研究
pixi run python scripts/run_intraday_single.py --factor vwap_deviation --start 20260401 --end 20260430 --universe csi300
```

Expected: 日志显示因子计算完成，HTML 报告生成

- [ ] **Step 5: 批量拉取全量 CSI300（可选，耗时约 30-60 分钟）**

```bash
cd E:\code\量化研究\因子研究
pixi run python scripts/fetch_minute_data.py --start 20260101 --end 20260516 --universe csi300 --freq 1min
```

Expected: 日志显示逐只股票拉取进度，失败只数尽量为 0

- [ ] **Step 6: 确认 grep 无遗留引用问题**

```bash
grep -rn "OUTPUT_LFT\|OUTPUT_MFT\|run_lft_\|lft_default" "E:\code\量化研究\因子研究" --include="*.py" | grep -v __pycache__ | grep -v .sisyphus
```

Expected: 无输出（已在 Phase 4 清理）

---

## 验证检查单

- [ ] `pixi run test` → ≥141 通过，0 失败，crowding 无 RuntimeWarning
- [ ] `data/raw/minute/` 非空（≥1 个分区文件）
- [ ] `output/intraday/reports/momentum_1min_*.html` 存在且可在浏览器打开
- [ ] `output/intraday/reports/vwap_deviation_*.html` 存在
- [ ] `grep -rn "OUTPUT_LFT\|lft_default" . --include="*.py" | grep -v __pycache__` 无输出

---

## 范围外（下一期议题）

- C. `daily/combination/` 多因子合成（IC 加权/等权/PCA）
- D. 实际因子库扩充（按你研究方向继续）
- intraday 高级评价（IC Decay、Regime IC、统计显著性）
- intraday 报告模板专项优化（时段分析展示）

# M2 · 防过拟合护栏 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 M1 挖掘流水线套上针对数据窥探的统计护栏——永久隔离的 OOS holdout + 多重检验记账 + Deflated Sharpe Ratio + PBO + bootstrap 置信区间——让挖出的因子附带可信度证据。

**Architecture:** 新增独立纯统计库 `src/factorzen/validation/`（DSR/PBO/bootstrap/记账/holdout，无 M1 依赖），再由 M1 的 `mining_session` 反向调用：挖掘只见 mining 段（holdout 永久不可见），top-K 选定后在 holdout 段做一次护栏验收。护栏基于 IC 序列（挖掘原生产出），非每候选跑回测。

**Tech Stack:** Python 3.10–3.12 · numpy · scipy 1.17（`scipy.stats.norm`）· polars · pytest · 复用 M1 `discovery/` 与 `daily/evaluation/ic_analysis`。

## Global Constraints

- **护栏统计基础是 IC 序列**（非回测 Sharpe）：PBO 用候选 × 日度 IC 矩阵；DSR 用 IR + IC 偏度峰度；bootstrap 用 IC 序列。
- **DSR 用挖掘表现的 mining/train 段 IR + 真实试验数 N**；**PBO 用 mining 候选池 IC 矩阵**；**bootstrap 用 holdout IC 序列**——三者各司其职。
- **OOS holdout 软隔离**：挖掘（DataBundle/搜索/去相关）只见 mining 段；holdout 段不传入挖掘循环；测试断言挖掘期最大日期 < holdout_start。
- **多重检验 N**：random 路径 = 去重后真正评估数 `len(seen)`；genetic 路径 = evolve 评估的不同表达式数 `len(cache)`（修正 M1 deferred）。
- **IC 序列取值**：`ic_series["ic"].drop_nulls().drop_nans().to_numpy()`（`ic_series` 是 `pl.DataFrame` 列 `trade_date, ic`）。
- **环境**：`pixi run pytest`；scipy 1.17.1 可用（`from scipy.stats import norm`）；polars 1.41.2（`pl.len()` 非 `pl.count()`）。
- **测试纪律**：纯 mock（`np.random.default_rng(seed)`），无磁盘 IO、无网络；CI 离线。
- **提交前自查**：`pixi run ruff check src/factorzen/validation/ src/factorzen/discovery/` → 0 errors。
- **提交规范**：conventional commits；作者 `rookiewu417 <1007372080@qq.com>`（`git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit`）。
- **每个 task 只 `git add` 自己的文件**（工作区有无关的 M0/M3 未提交改动；**绝不** `git add -A`）。
- **范围外**：White Reality Check / Hansen SPA、regime 稳定性、参数高原、硬隔离 holdout、接入非挖掘的 `fz factor run`。

---

## File Structure

| 文件 | 职责 | Task |
|---|---|---|
| `src/factorzen/validation/__init__.py` | 包导出 | 1 |
| `src/factorzen/validation/multiple_testing.py` | `TrialLedger`：记真实评估数 N | 1 |
| `src/factorzen/validation/bootstrap.py` | `block_bootstrap_ic_ci`：IC 均值置信区间 | 2 |
| `src/factorzen/validation/deflated_sharpe.py` | `expected_max_sharpe` + `deflated_sharpe` | 3 |
| `src/factorzen/validation/pbo.py` | `compute_pbo`：CSCV 回测过拟合概率 | 4 |
| `src/factorzen/validation/holdout.py` | `split_holdout` + `holdout_evaluate` | 5 |
| `src/factorzen/discovery/mining_session.py` | 接回：holdout 切分 + top-K 护栏验收 + N 记账 | 6 |
| `src/factorzen/pipelines/factor_mine.py` | `run_mine` 增 `holdout_ratio` | 6 |
| `src/factorzen/cli/main.py` | `fz validate overfit` + leaderboard 增列 | 7 |
| `tests/test_validation_multiple_testing.py` | TrialLedger | 1 |
| `tests/test_validation_bootstrap.py` | bootstrap CI 构造验证 | 2 |
| `tests/test_validation_deflated_sharpe.py` | DSR 构造验证 + N 单调性 | 3 |
| `tests/test_validation_pbo.py` | PBO 构造验证 + 对称性 | 4 |
| `tests/test_validation_holdout.py` | 切分 + 隔离校验 | 5 |
| `tests/test_discovery_session.py`（扩展） | holdout 隔离 + 护栏指标 | 6 |
| `tests/test_validation_cli.py` | validate overfit + leaderboard 列 | 7 |

---

## Task 1: 多重检验记账（multiple_testing.py）

**Files:**
- Create: `src/factorzen/validation/__init__.py`, `src/factorzen/validation/multiple_testing.py`
- Test: `tests/test_validation_multiple_testing.py`

**Interfaces:**
- Produces: `@dataclass TrialLedger`，字段 `n_trials: int = 0`；方法 `record(self, k: int = 1) -> None`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_validation_multiple_testing.py
def test_trial_ledger_accumulates():
    from factorzen.validation.multiple_testing import TrialLedger
    led = TrialLedger()
    assert led.n_trials == 0
    led.record()
    led.record(5)
    assert led.n_trials == 6


def test_trial_ledger_default_zero():
    from factorzen.validation.multiple_testing import TrialLedger
    assert TrialLedger().n_trials == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_validation_multiple_testing.py -v`
Expected: FAIL（`ModuleNotFoundError: factorzen.validation`）

- [ ] **Step 3: 实现**

```python
# src/factorzen/validation/__init__.py
"""防过拟合护栏：DSR / PBO / bootstrap / 多重检验记账 / holdout 隔离。"""

# src/factorzen/validation/multiple_testing.py
"""多重检验记账：记录挖掘过程真实评估的候选数 N。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrialLedger:
    """累加真实评估候选数；该 N 喂给 DSR 并在报告中标注「从 N 个候选选出」。"""

    n_trials: int = 0

    def record(self, k: int = 1) -> None:
        self.n_trials += k
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_validation_multiple_testing.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
pixi run ruff check src/factorzen/validation/
git add src/factorzen/validation/__init__.py src/factorzen/validation/multiple_testing.py tests/test_validation_multiple_testing.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(validation): TrialLedger 多重检验记账"
```

---

## Task 2: bootstrap 置信区间（bootstrap.py）

**Files:**
- Create: `src/factorzen/validation/bootstrap.py`
- Test: `tests/test_validation_bootstrap.py`

**Interfaces:**
- Produces: `block_bootstrap_ic_ci(ic_series, block_size=10, n_boot=1000, alpha=0.05, seed=42) -> tuple[float, float]`（`ic_series`: 1D array-like of daily IC；返回 (ci_low, ci_high) of mean IC）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_validation_bootstrap.py
import numpy as np


def test_positive_ic_ci_above_zero():
    from factorzen.validation.bootstrap import block_bootstrap_ic_ci
    rng = np.random.default_rng(0)
    ic = rng.normal(0.05, 0.02, 250)  # 明显正 IC
    lo, hi = block_bootstrap_ic_ci(ic, seed=1)
    assert lo > 0 and hi > lo


def test_noise_ic_ci_straddles_zero():
    from factorzen.validation.bootstrap import block_bootstrap_ic_ci
    rng = np.random.default_rng(0)
    ic = rng.normal(0.0, 0.05, 250)  # 噪声 IC
    lo, hi = block_bootstrap_ic_ci(ic, seed=1)
    assert lo < 0 < hi


def test_too_short_returns_nan():
    from factorzen.validation.bootstrap import block_bootstrap_ic_ci
    lo, hi = block_bootstrap_ic_ci(np.array([0.1, 0.2]), block_size=10)
    assert np.isnan(lo) and np.isnan(hi)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_validation_bootstrap.py -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 实现**

```python
# src/factorzen/validation/bootstrap.py
"""IC 序列的 moving block bootstrap 置信区间（保留时序自相关）。"""
from __future__ import annotations

import numpy as np


def block_bootstrap_ic_ci(
    ic_series,
    block_size: int = 10,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    ic = np.asarray(ic_series, dtype=float)
    ic = ic[~np.isnan(ic)]
    n = ic.size
    if n < block_size or n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([ic[s : s + block_size] for s in starts])[:n]
        means[b] = sample.mean()
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_validation_bootstrap.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
pixi run ruff check src/factorzen/validation/
git add src/factorzen/validation/bootstrap.py tests/test_validation_bootstrap.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(validation): block bootstrap IC 置信区间"
```

---

## Task 3: Deflated Sharpe Ratio（deflated_sharpe.py）

**Files:**
- Create: `src/factorzen/validation/deflated_sharpe.py`
- Test: `tests/test_validation_deflated_sharpe.py`

**Interfaces:**
- Produces:
  - `expected_max_sharpe(sharpe_variance: float, n_trials: int) -> float`
  - `deflated_sharpe(sharpe, n_trials, n_obs, skew=0.0, kurt=3.0, sharpe_variance=None) -> tuple[float, float]`（返回 (dsr, pvalue)；`dsr` = PSR(期望最大 Sharpe)，`pvalue = 1 - dsr`）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_validation_deflated_sharpe.py
import numpy as np


def test_strong_sharpe_significant():
    from factorzen.validation.deflated_sharpe import deflated_sharpe
    # 高 IR、长样本、少试验 → 应显著
    dsr, p = deflated_sharpe(sharpe=0.15, n_trials=5, n_obs=500, sharpe_variance=0.0025)
    assert dsr > 0.95 and p < 0.05


def test_noise_sharpe_not_significant():
    from factorzen.validation.deflated_sharpe import deflated_sharpe
    # IR≈0 → 不显著
    dsr, p = deflated_sharpe(sharpe=0.0, n_trials=100, n_obs=500, sharpe_variance=0.0025)
    assert p > 0.05


def test_more_trials_tightens():
    from factorzen.validation.deflated_sharpe import deflated_sharpe
    # 同样观测 Sharpe，更多试验 → DSR 下降（多重检验收紧）
    dsr_few, _ = deflated_sharpe(0.12, n_trials=5, n_obs=500, sharpe_variance=0.0025)
    dsr_many, _ = deflated_sharpe(0.12, n_trials=1000, n_obs=500, sharpe_variance=0.0025)
    assert dsr_many < dsr_few


def test_expected_max_sharpe_grows_with_trials():
    from factorzen.validation.deflated_sharpe import expected_max_sharpe
    assert expected_max_sharpe(0.0025, 1000) > expected_max_sharpe(0.0025, 10)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_validation_deflated_sharpe.py -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 实现**

```python
# src/factorzen/validation/deflated_sharpe.py
"""Deflated Sharpe Ratio（Bailey & López de Prado 2014）。

挖掘 = 多重检验：从 N 个候选里选最优，观测 Sharpe 被夸大。DSR 用「期望最大
Sharpe」作为 deflation 基准，评估观测 Sharpe 扣除多重检验后是否仍显著。
本项目以因子 IC 的 IR 作为「Sharpe」、IC 序列长度作为样本数。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

_EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(sharpe_variance: float, n_trials: int) -> float:
    """N 次独立试验下、零假设期望最大 Sharpe（deflation 基准）。"""
    if n_trials < 2 or sharpe_variance <= 0:
        return 0.0
    e = np.e
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float(np.sqrt(sharpe_variance) * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))


def deflated_sharpe(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    sharpe_variance: float | None = None,
) -> tuple[float, float]:
    """返回 (dsr, pvalue)。dsr=PSR(期望最大 Sharpe)，pvalue=1-dsr，<0.05 视为显著。"""
    if n_obs < 2:
        return (0.0, 1.0)
    if sharpe_variance is None:
        sharpe_variance = 1.0 / n_obs  # H0 下 per-period Sharpe 的方差近似 1/T
    sr0 = expected_max_sharpe(sharpe_variance, n_trials)
    denom = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe**2
    if denom <= 0:
        return (0.0, 1.0)
    z = (sharpe - sr0) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    dsr = float(norm.cdf(z))
    return (dsr, 1.0 - dsr)
```

> 注：`sharpe_variance` 由调用方传 mining 候选池所有 IR 的方差（反映搜索空间离散度）；缺省退化为 `1/n_obs`。

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_validation_deflated_sharpe.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
pixi run ruff check src/factorzen/validation/
git add src/factorzen/validation/deflated_sharpe.py tests/test_validation_deflated_sharpe.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(validation): Deflated Sharpe Ratio"
```

---

## Task 4: PBO（pbo.py，CSCV）

**Files:**
- Create: `src/factorzen/validation/pbo.py`
- Test: `tests/test_validation_pbo.py`

**Interfaces:**
- Produces: `compute_pbo(perf_matrix: np.ndarray, n_splits: int = 10) -> float`（`perf_matrix` shape `(n_candidates, n_periods)`，每行一个候选的日度 IC；返回 PBO ∈ [0,1]）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_validation_pbo.py
import numpy as np


def test_pbo_noise_near_half():
    """纯噪声候选池：IS 最优在 OOS 无优势 → PBO ≈ 0.5。"""
    from factorzen.validation.pbo import compute_pbo
    rng = np.random.default_rng(0)
    perf = rng.normal(0, 1, (20, 200))
    pbo = compute_pbo(perf, n_splits=10)
    assert 0.3 < pbo < 0.7


def test_pbo_one_dominant_low():
    """一个候选全程显著最优 → IS 最优 = OOS 最优 → PBO 低。"""
    from factorzen.validation.pbo import compute_pbo
    rng = np.random.default_rng(0)
    perf = rng.normal(0, 1, (20, 200))
    perf[0] += 3.0  # 候选0 全程领先
    pbo = compute_pbo(perf, n_splits=10)
    assert pbo < 0.2


def test_pbo_too_small_returns_nan():
    from factorzen.validation.pbo import compute_pbo
    assert np.isnan(compute_pbo(np.zeros((1, 100)), n_splits=10))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_validation_pbo.py -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 实现**

```python
# src/factorzen/validation/pbo.py
"""PBO（Probability of Backtest Overfitting）via CSCV（López de Prado 2016）。

把时间分 S 块，枚举所有 S/2 块为 IS、其余为 OOS 的对称划分；每种里在 IS
选最优候选，看它在 OOS 的相对秩。PBO = IS 最优在 OOS 落后半区的频率。
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def compute_pbo(perf_matrix: np.ndarray, n_splits: int = 10) -> float:
    perf = np.asarray(perf_matrix, dtype=float)
    n_cand, n_periods = perf.shape
    if n_cand < 2 or n_periods < n_splits or n_splits % 2 != 0:
        return float("nan")
    block = n_periods // n_splits
    # 每块每候选的平均表现 → (n_splits, n_cand)
    block_means = np.array([perf[:, i * block : (i + 1) * block].mean(axis=1) for i in range(n_splits)])
    half = n_splits // 2
    logits = []
    for is_idx in combinations(range(n_splits), half):
        oos_idx = [i for i in range(n_splits) if i not in is_idx]
        is_perf = block_means[list(is_idx)].mean(axis=0)
        oos_perf = block_means[oos_idx].mean(axis=0)
        best = int(np.argmax(is_perf))
        # best 在 OOS 的相对秩 ∈ (0,1)
        rank = float((oos_perf <= oos_perf[best]).sum())  # 含自身
        rel = rank / (n_cand + 1)
        rel = min(max(rel, 1e-6), 1 - 1e-6)
        logits.append(np.log(rel / (1 - rel)))
    logits = np.asarray(logits)
    return float((logits <= 0).mean())
```

> 注：`n_splits=10` → `C(10,5)=252` 组合，可控；`n_splits` 须为偶数。`rel` 越接近 1 表示 best 在 OOS 仍领先（logit>0）；logit≤0 即落后半区。

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_validation_pbo.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
pixi run ruff check src/factorzen/validation/
git add src/factorzen/validation/pbo.py tests/test_validation_pbo.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(validation): PBO (CSCV) 回测过拟合概率"
```

---

## Task 5: holdout 切分与隔离（holdout.py）

**Files:**
- Create: `src/factorzen/validation/holdout.py`
- Test: `tests/test_validation_holdout.py`

**Interfaces:**
- Consumes: M1 `quick_fitness` / `DataBundle`（Task 6 用）；`block_bootstrap_ic_ci`（Task 2）
- Produces:
  - `split_holdout(daily: pl.DataFrame, holdout_ratio: float = 0.2) -> tuple[pl.DataFrame, pl.DataFrame, "datetime.date"]`（返回 mining_df, holdout_df, holdout_start）
  - `holdout_ic(factor_df: pl.DataFrame, holdout_df: pl.DataFrame) -> tuple[float, float, tuple[float, float]]`（返回 (ic_mean, ir, ci) on holdout）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_validation_holdout.py
import numpy as np
import polars as pl
from datetime import date, timedelta


def _daily(n_stocks=20, n_days=200, seed=1):
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for s in [f"{i:06d}.SH" for i in range(n_stocks)]:
        p = 10.0
        for day in days:
            p = float(max(p * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": day, "ts_code": s, "close": p, "close_adj": p,
                         "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows)


def test_split_holdout_disjoint_and_isolated():
    from factorzen.validation.holdout import split_holdout
    daily = _daily()
    mining, holdout, hstart = split_holdout(daily, holdout_ratio=0.2)
    # 隔离：mining 全部 < holdout_start ≤ holdout 全部
    assert mining["trade_date"].max() < hstart
    assert holdout["trade_date"].min() >= hstart
    # holdout 约占 20%
    frac = holdout["trade_date"].n_unique() / daily["trade_date"].n_unique()
    assert 0.15 < frac < 0.25


def test_holdout_ic_runs():
    from factorzen.validation.holdout import split_holdout, holdout_ic
    daily = _daily()
    mining, holdout, _ = split_holdout(daily, holdout_ratio=0.2)
    # 用「次日收益」当因子 → holdout IC 应为正
    fac = holdout.sort(["ts_code", "trade_date"]).with_columns(
        (pl.col("close_adj").shift(-1).over("ts_code") / pl.col("close_adj") - 1.0).alias("factor_value")
    ).select(["trade_date", "ts_code", "factor_value"]).drop_nulls()
    ic_mean, ir, (lo, hi) = holdout_ic(fac, holdout)
    assert ic_mean > 0.05 and lo <= hi
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_validation_holdout.py -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 实现**

```python
# src/factorzen/validation/holdout.py
"""OOS holdout 时间切分（软隔离）+ holdout 段 IC 验收。"""
from __future__ import annotations

import polars as pl

from factorzen.discovery.scoring import DataBundle, quick_fitness
from factorzen.validation.bootstrap import block_bootstrap_ic_ci


def split_holdout(daily: pl.DataFrame, holdout_ratio: float = 0.2):
    """按交易日时间序，最后 holdout_ratio 比例为 holdout；其余为 mining 段。"""
    dates = sorted(daily["trade_date"].unique().to_list())
    cut = int(len(dates) * (1.0 - holdout_ratio))
    cut = min(max(cut, 1), len(dates) - 1)
    holdout_start = dates[cut]
    mining_df = daily.filter(pl.col("trade_date") < holdout_start)
    holdout_df = daily.filter(pl.col("trade_date") >= holdout_start)
    return mining_df, holdout_df, holdout_start


def holdout_ic(factor_df: pl.DataFrame, holdout_df: pl.DataFrame):
    """top-K 候选因子值在 holdout 段算 (ic_mean, ir, bootstrap_ci)。"""
    bundle = DataBundle.build(holdout_df, train_ratio=1.0)  # 整个 holdout 当评估段
    res = quick_fitness(factor_df, bundle, "train")  # train_ratio=1.0 → 全段在 train
    # 取 IC 序列做 bootstrap：复用 quick_fitness 内部不暴露序列，这里直接重算
    from factorzen.daily.evaluation.ic_analysis import compute_rank_ic
    from factorzen.daily.preprocessing.normalizer import cross_sectional_zscore
    clean = cross_sectional_zscore(factor_df, col="factor_value").rename({"factor_value_z": "factor_clean"})
    ic_res = compute_rank_ic(clean.select(["trade_date", "ts_code", "factor_clean"]),
                             bundle.fwd_returns, factor_col="factor_clean", frequency="daily")
    ic_vals = ic_res.ic_series["ic"].drop_nulls().drop_nans().to_numpy()
    ci = block_bootstrap_ic_ci(ic_vals)
    return (res["ic_mean"], res["ir"], ci)
```

> 注：`DataBundle.build(holdout_df, train_ratio=1.0)` 让整个 holdout 段落在 train，`quick_fitness(..., "train")` 即对全 holdout 算 IC。Task 6 据此做 top-K 验收。

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_validation_holdout.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
pixi run ruff check src/factorzen/validation/ src/factorzen/discovery/
git add src/factorzen/validation/holdout.py tests/test_validation_holdout.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(validation): holdout 时间切分 + holdout IC 验收"
```

---

## Task 6: 接回挖掘流程（mining_session.py + factor_mine.py）

**Files:**
- Modify: `src/factorzen/discovery/mining_session.py`, `src/factorzen/pipelines/factor_mine.py`
- Test: `tests/test_discovery_session.py`（扩展）

**Interfaces:**
- Consumes: `split_holdout`, `holdout_ic`（Task 5）；`compute_pbo`（Task 4）；`deflated_sharpe`（Task 3）；`TrialLedger`（Task 1）
- Produces: `run_session(..., holdout_ratio: float = 0.2)` 返回的每个 candidate dict 增加 `n_trials`、`pbo`、`holdout_ic`、`dsr_pvalue`、`ic_ci_low`

- [ ] **Step 1: 写失败测试（扩展 session 测试）**

```python
# tests/test_discovery_session.py （追加）
def test_session_has_guard_metrics_and_holdout_isolated(tmp_path):
    from factorzen.discovery.mining_session import run_session
    res = run_session(_daily(), n_trials=30, top_k=5, seed=42,
                      method="random", holdout_ratio=0.2, out_dir=str(tmp_path))
    assert 0 < len(res["candidates"]) <= 5
    for c in res["candidates"]:
        # 护栏指标齐全
        for key in ("n_trials", "pbo", "holdout_ic", "dsr_pvalue", "ic_ci_low"):
            assert key in c
        assert c["n_trials"] > 0          # 真实评估数（非 CLI n_trials 摆设）
        assert 0.0 <= c["pbo"] <= 1.0 or c["pbo"] != c["pbo"]  # [0,1] 或 nan
```

（`_daily` 复用本文件已有 helper，n_days≥120 保证 mining/holdout 都够样本。）

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_discovery_session.py::test_session_has_guard_metrics_and_holdout_isolated -v`
Expected: FAIL（`run_session() got unexpected keyword 'holdout_ratio'` 或缺指标）

- [ ] **Step 3: 修改 run_session**

在 `mining_session.py` 顶部 import 补：
```python
from factorzen.validation.deflated_sharpe import deflated_sharpe
from factorzen.validation.holdout import holdout_ic, split_holdout
from factorzen.validation.pbo import compute_pbo
```

`run_session` 签名增 `holdout_ratio: float = 0.2`。在停牌掩码+派生列之后、`DataBundle.build` 之前插入 holdout 切分（挖掘只见 mining 段）：
```python
    # ── OOS holdout 永久隔离：挖掘只见 mining 段 ──
    mining_df, holdout_df, holdout_start = split_holdout(daily, holdout_ratio=holdout_ratio)
    daily = mining_df  # 后续挖掘全部只用 mining 段（DataBundle/搜索/去相关）
    bundle = DataBundle.build(daily, train_ratio=train_ratio)
```

把 `manifest["n_trials"]` 与每候选的 `n_trials` 改为**真实评估数**。先把现有 genetic 分支里的局部 `cache: dict[str, float] = {}` **提到 method 分支之前**声明为 `eval_cache: dict[str, float] = {}`，genetic 的 `_score` 闭包改用 `eval_cache`（这样评分后仍可见）。在统一评分循环后用 `TrialLedger` 记账：
```python
    from factorzen.validation.multiple_testing import TrialLedger
    # random=去重评估数；genetic=evolve 内评估的不同表达式数（eval_cache）
    eval_n = len(eval_cache) if method == "genetic" else len(seen)
    ledger = TrialLedger()
    ledger.record(eval_n)
    n_evaluated = ledger.n_trials
```

在 `top = selected` 之后、写产物之前，插入护栏验收（只用 holdout 一次）：
```python
    # ── 护栏验收（holdout 只用一次）──
    n_obs_mining = daily["trade_date"].n_unique()  # mining 段交易日数 ≈ IC 序列长度
    ir_pool = np.array([c["ir_train"] for c in scored]) if scored else np.array([0.0])
    sharpe_var = float(ir_pool.var()) if ir_pool.size > 1 else 1.0
    pbo = _pool_pbo(scored, daily, bundle)  # 候选池(mining 段)日度 IC 矩阵 → PBO
    for c in top:
        node = parse_expr(c["expression"])
        fdf_hold = _factor_values(node, holdout_df)
        if fdf_hold.height >= 20:
            h_ic, _h_ir, (ci_lo, _ci_hi) = holdout_ic(fdf_hold, holdout_df)
        else:
            h_ic, ci_lo = float("nan"), float("nan")
        _dsr, p = deflated_sharpe(c["ir_train"], n_evaluated, n_obs_mining, sharpe_variance=sharpe_var)
        c["n_trials"] = n_evaluated
        c["pbo"] = round(pbo, 4) if pbo == pbo else float("nan")
        c["holdout_ic"] = round(float(h_ic), 4) if h_ic == h_ic else float("nan")
        c["dsr_pvalue"] = round(float(p), 4)
        c["ic_ci_low"] = round(float(ci_lo), 4) if ci_lo == ci_lo else float("nan")
```

新增辅助函数（构造候选池 IC 矩阵跑 PBO）——放在 `run_session` 上方：
```python
def _pool_pbo(scored: list, daily: pl.DataFrame, bundle) -> float:
    """对 scored 候选（mining 段）构造日度 IC 矩阵跑 PBO；样本不足返回 nan。"""
    from factorzen.daily.evaluation.ic_analysis import compute_rank_ic
    from factorzen.daily.preprocessing.normalizer import cross_sectional_zscore
    series = []
    dates_ref = None
    for c in scored[:30]:  # 取 fitness 前 30 个候选，控制成本
        try:
            fdf = _factor_values(parse_expr(c["expression"]), daily)
            clean = cross_sectional_zscore(fdf, col="factor_value").rename({"factor_value_z": "factor_clean"})
            ic_res = compute_rank_ic(clean.select(["trade_date", "ts_code", "factor_clean"]),
                                     bundle.fwd_returns, factor_col="factor_clean", frequency="daily")
            ser = ic_res.ic_series.sort("trade_date")
            if dates_ref is None:
                dates_ref = ser["trade_date"]
            ser = ser.join(pl.DataFrame({"trade_date": dates_ref}), on="trade_date", how="right").sort("trade_date")
            series.append(ser["ic"].fill_null(0.0).to_numpy())
        except Exception:
            continue
    if len(series) < 2:
        return float("nan")
    import numpy as _np
    return compute_pbo(_np.vstack(series), n_splits=10)
```

更新 candidates.csv 列与空表头、manifest（`n_trials` 改 `n_evaluated`）：
```python
    _cols = ["expression", "ic_train", "ir_train", "ic_valid", "ir_valid", "max_corr",
             "complexity", "holdout_ic", "dsr_pvalue", "pbo", "ic_ci_low"]
    rows = [{"rank": i + 1, "n_trials": n_evaluated, **{k: c.get(k) for k in _cols}} for i, c in enumerate(top)]
    pl.DataFrame(rows).write_csv(session_dir / "candidates.csv") if rows else \
        (session_dir / "candidates.csv").write_text("rank,n_trials," + ",".join(_cols) + "\n")
    manifest = {"seed": seed, "method": method, "n_trials": n_evaluated, "cli_n_trials": n_trials,
                "top_k": top_k, "train_end": bundle.train_end, "holdout_start": str(holdout_start),
                "git_sha": _git_sha(), "duration_seconds": round(time.perf_counter() - t0, 3),
                "candidates": top,
                "reproduce_note": "导出因子在 exported/；复现需复制到 workspace/factors/daily/ 后 fz factor run <name> --set preprocessing.neutralize=false（IC parity）"}
```

`factor_mine.py` 的 `run_mine` 增 `holdout_ratio: float = 0.2` 参数并透传给 `run_session`。

> 实现要点：DSR 的 `n_obs` 用 mining 段交易日数 `n_obs_mining`（≈ IC 序列长度）；genetic 分支把局部 `cache` 提为外层 `eval_cache` 以正确统计真实评估数 N；保留 M1 已有的「全失败 raise」逻辑不动。

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_discovery_session.py -v`
Expected: PASS（含新护栏测试 + 原有 session 测试不破坏）

- [ ] **Step 5: 提交**

```bash
pixi run ruff check src/factorzen/discovery/ src/factorzen/validation/ src/factorzen/pipelines/factor_mine.py
git add src/factorzen/discovery/mining_session.py src/factorzen/pipelines/factor_mine.py tests/test_discovery_session.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(discovery): 挖掘接入 holdout 隔离 + PBO/DSR/CI 护栏验收 + 真实 N 记账"
```

---

## Task 7: CLI（fz validate overfit + leaderboard 增列）

**Files:**
- Modify: `src/factorzen/cli/main.py`
- Test: `tests/test_validation_cli.py`

**Interfaces:**
- Consumes: `deflated_sharpe`, `block_bootstrap_ic_ci`（单因子验证）；`build_parser`（现有）
- Produces: `fz validate overfit <factor> --start --end`；`_cmd_validate_overfit(args) -> int`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_validation_cli.py
def test_parser_has_validate_overfit():
    from factorzen.cli.main import build_parser
    p = build_parser()
    args = p.parse_args(["validate", "overfit", "momentum_12_1", "--start", "20230101", "--end", "20240101"])
    assert args.command == "validate"
    assert args.validate_command == "overfit"
    assert args.factor == "momentum_12_1"
    assert callable(args.func)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_validation_cli.py -v`
Expected: FAIL（`AttributeError`）

- [ ] **Step 3: 接入 CLI**

在 `build_parser()` 里（`mine` 组之后）加顶层 `validate` 组：
```python
    validate = sub.add_parser("validate", help="Overfitting / robustness checks")
    validate_sub = validate.add_subparsers(dest="validate_command", required=True)
    vo = validate_sub.add_parser("overfit", help="Deflated Sharpe + bootstrap CI for one factor")
    vo.add_argument("factor")
    vo.add_argument("--start", required=True)
    vo.add_argument("--end", required=True)
    vo.add_argument("--universe", default=None)
    vo.set_defaults(func=_cmd_validate_overfit)
```

模块顶层加（直调式，仿 `_cmd_factor_sweep`）：
```python
def _cmd_validate_overfit(args: argparse.Namespace) -> int:
    from factorzen.daily.data.context import FactorDataContext
    from factorzen.daily.evaluation.ic_analysis import compute_rank_ic
    from factorzen.daily.factors.registry import get_factor
    from factorzen.daily.preprocessing.normalizer import cross_sectional_zscore
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.bootstrap import block_bootstrap_ic_ci
    from factorzen.validation.deflated_sharpe import deflated_sharpe

    factor = get_factor(args.factor)()
    ctx = FactorDataContext(start=args.start, end=args.end,
                            required_data=["daily", "daily_basic"], lookback_days=getattr(factor, "lookback_days", 60))
    fdf = factor.compute(ctx)
    bundle = DataBundle.build(ctx.daily.collect(), train_ratio=1.0)
    clean = cross_sectional_zscore(fdf.rename({"factor_value": "factor_value"}), col="factor_value").rename(
        {"factor_value_z": "factor_clean"})
    ic_res = compute_rank_ic(clean.select(["trade_date", "ts_code", "factor_clean"]),
                             bundle.fwd_returns, factor_col="factor_clean", frequency="daily")
    ic_vals = ic_res.ic_series["ic"].drop_nulls().drop_nans().to_numpy()
    lo, hi = block_bootstrap_ic_ci(ic_vals)
    dsr, p = deflated_sharpe(ic_res.ir, n_trials=1, n_obs=len(ic_vals))  # 单因子：N=1
    print(f"[validate] {args.factor}: IC={ic_res.ic_mean:.4f} IR={ic_res.ir:.4f} "
          f"DSR_p={p:.4f} IC_95%CI=[{lo:.4f},{hi:.4f}]")
    print("[validate] 注：单因子 N=1（无多重检验扣减）；PBO 仅适用候选池，此处略。")
    return 0
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_validation_cli.py -v`
Expected: PASS

- [ ] **Step 5: 全量质量门 + 提交**

```bash
pixi run pytest tests/test_validation_*.py tests/test_discovery_*.py -q
pixi run ruff check src/factorzen/validation/ src/factorzen/discovery/
git add src/factorzen/cli/main.py tests/test_validation_cli.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(validation): fz validate overfit CLI"
```

---

## 收尾验收（全部 task 完成后）

- [ ] `pixi run pytest tests/test_validation_*.py tests/test_discovery_*.py -q` 全绿
- [ ] `pixi run ruff check src/factorzen/validation/ src/factorzen/discovery/` 0 errors
- [ ] holdout 隔离测试通过（挖掘期不可见 holdout）
- [ ] leaderboard/manifest 含 n_trials(N) / pbo / holdout_ic / dsr_pvalue / ic_ci_low
- [ ] `fz validate overfit <factor>` 对单因子输出 DSR + CI
- [ ] 手动 smoke（需本地数据）：`pixi run fz mine search --start 20230101 --end 20241231 --universe csi500 --trials 200 --top-k 10` → 观察护栏指标
- [ ] 更新 `docs/superpowers/plans/2026-06-30-m2-overfit-guards.md` 追加完成记录

---

*M2 完成后，挖掘流水线即具备「可信」属性（永久隔离 holdout + PBO/DSR/CI/记账），可支撑 M5 Agent 挖掘。*

---

## 实现完成记录（2026-06-30）

M2 按本计划 7 个 task 全部实现，subagent 驱动执行，通过 opus 整分支 final review（+ C1/I1/I2 fix）。

**成果**：`src/factorzen/validation/`（164 行源 + 测试，11 commits，54 测试全绿，ruff 0 errors）。
TrialLedger + bootstrap + Deflated Sharpe + PBO(CSCV) + holdout(永久隔离) → 接回 M1 挖掘：挖掘只见 mining 段，top-K 在 holdout 段护栏验收，leaderboard/manifest 增列 `n_trials/pbo/holdout_ic/dsr_pvalue/ic_ci_low`；`fz validate overfit` CLI。

**opus final review 验证**：holdout 隔离真闭合（无 seam 泄漏，隔离测试反向验证有效）；IC 口径一致（holdout_ic/mining 都用 `compute_rank_ic` min_samples=30）；护栏统计正确且**非摆设**（噪声→PBO 高/DSR 不显著，构造验证）；N 记账真实（genetic=`len(eval_cache)`）。

**final fix 兑现**：C1 `DataBundle` ratio=1.0 防越界（修 validate 崩溃）；I1 挖掘 IC 对齐 start 窗口（**兑现 M1+M2 deferred 的 IC parity**）；I2 validate 传 universe。

**deferred（非阻塞）**：bootstrap 死代码；PBO nan 分支测试/丢弃期注释；holdout_ratio 边界测试；集成路径噪声 flag 的端到端断言；sharpe_var=1.0 magic default。

**下一步**：M3 Barra 风险模型（`risk/` 已有未提交工作待收口接入），或 M5 Agent 挖掘（建立在 M1+M2 可信流水线上）。

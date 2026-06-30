# M3 · 风险模型收口 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给已写好但未提交、零测试、未接入主线的 Barra 风险模型（`src/factorzen/risk/`）补上 5 个模块的构造验证测试 + `fz risk build` CLI + 轻量风险报告，并提交入库。

**Architecture:** risk 模块代码已存在（不重写）；本计划只新增测试 + `pipelines/risk_build.py` 编排 + CLI 接入。测试全部纯 mock（risk 因子函数直接吃 polars DataFrame）。risk 源码本身在 Task 1 一并入库。

**Tech Stack:** Python 3.10–3.12 · numpy · polars · statsmodels（截面回归）· pytest · 复用 `core/loader`、`core/universe`、`daily/data/context`。

## Global Constraints

- **risk 源码当前未提交**（`src/factorzen/risk/` 6 文件，1198 行）；Task 1 把它与第一个测试一并 `git add` 入库。
- **现有 `tests/test_style_factors.py` 与 risk 无关**（它测 `builtin_factors.daily` 的 DailyFactor 类）；M3 给 `risk/` 写**全新**测试，不复用那套 ctx mock。risk 因子函数直接吃 `pl.DataFrame`（非 LazyFrame/ctx）。
- **测试 mock**：`trade_date` 用 `pl.Date`；`make_daily` 需 `pct_chg` 列（momentum/volatility 用，momentum 需 ≥252 期）；`make_daily_basic` 需 `total_mv/pb/pe_ttm/turnover_rate`；`make_stocks` 需 `ts_code/industry`。
- **`turnover_rate` 坑**：`loader.fetch_daily_basic` 不落该列 → liquidity 风格因子静默返回空。测试要覆盖 liquidity 必须手动加 `turnover_rate`；CLI 真实跑数时 liquidity 缺（可接受，报告标注）。
- **`weights` 对齐**：`predict_risk`/`decompose_risk` 不按 ts_code 重排，`weights[i]` 对应 `result.factor_exposures.codes[i]`。
- **`decompose_risk` 返回含行业键**：除 `total_risk/factor_risk/specific_risk` 外，每个风格名与每个 `ind_XXX` 行业名各一个键。
- **风险守恒断言**：`factor_risk² + specific_risk² ≈ total_risk²`（方差可加；用于 test_risk_model）。
- **测试暴露 risk 现有 bug**：顺手修并加回归断言（评审拍板）。
- **环境**：`pixi run pytest`；polars 1.41.2（`pl.len()` 非 `pl.count()`）；statsmodels 可用。
- **提交前自查**：`pixi run ruff check src/factorzen/risk/ <改的测试文件>` → 0 errors。
- **提交规范**：conventional commits；作者 `rookiewu417 <1007372080@qq.com>`；每个 task 只 `git add` 自己的文件（工作区有无关 M0 未提交改动，**绝不** `-A`）。
- **范围外**：组合优化（M4）、完整 HTML 报告、改写 risk 算法（仅修 bug）。

---

## File Structure

| 文件 | 职责 | Task |
|---|---|---|
| `src/factorzen/risk/*.py`（已存在，入库） | Barra 风险模型源码 | 1 |
| `tests/test_risk_industry.py` | get_industry_dummies / names | 1 |
| `tests/test_risk_style_factors.py` | cs_standardize + 因子函数 + registry | 2 |
| `tests/test_risk_exposures.py` | compute_exposures + ExposureMatrix | 3 |
| `tests/test_risk_covariance.py` | 协方差/特质风险/eigenvector | 4 |
| `tests/test_risk_model.py` | predict_risk/decompose_risk（手搓）+ build 端到端 | 5 |
| `src/factorzen/pipelines/risk_build.py` | `run_risk_build`：拉数据 build 落产物 + 报告 | 6 |
| `tests/test_risk_build_pipeline.py` | run_risk_build smoke（mock 数据） | 6 |
| `src/factorzen/cli/main.py` | `fz risk build` 接入 | 7 |
| `tests/test_risk_cli.py` | parser + handler smoke | 7 |

**共用 mock helper**：Task 1 在 `tests/test_risk_industry.py` 写 `_trade_days`/`make_daily`/`make_daily_basic`/`make_stocks`；后续测试各自复制所需的（或从该文件 import）。为简单，**每个测试文件自带所需 helper**（避免跨文件 import 依赖）。

---

## Task 1: risk 源码入库 + 行业因子测试

**Files:**
- Commit (existing): `src/factorzen/risk/__init__.py`, `industry_factors.py`, `exposures.py`, `covariance.py`, `style_factors.py`, `model.py`
- Test: `tests/test_risk_industry.py`

**Interfaces:**
- Consumes: `get_industry_dummies(stocks, industry_col="industry") -> pl.DataFrame`（输出 `ts_code` + `ind_{行业}` 列，0/1）；`get_industry_names(stocks, industry_col="industry") -> list[str]`（裸行业名，升序）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_risk_industry.py
import polars as pl


def make_stocks(n_stocks=8):
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    industries = ["银行", "医药", "电子", "食品饮料"]
    return pl.DataFrame({
        "ts_code": codes,
        "industry": [industries[i % len(industries)] for i in range(n_stocks)],
    })


def test_industry_dummies_one_hot_per_stock():
    from factorzen.risk.industry_factors import get_industry_dummies
    dummies = get_industry_dummies(make_stocks())
    ind_cols = [c for c in dummies.columns if c.startswith("ind_")]
    assert len(ind_cols) == 4  # 4 个唯一行业
    # 每只股票恰属一个行业：ind_* 列之和 == 1
    row_sums = dummies.select(ind_cols).sum_horizontal()
    assert row_sums.to_list() == [1.0] * dummies.height


def test_industry_names_sorted_bare():
    from factorzen.risk.industry_factors import get_industry_names
    names = get_industry_names(make_stocks())
    assert names == sorted(names)
    assert set(names) == {"银行", "医药", "电子", "食品饮料"}
    assert all(not n.startswith("ind_") for n in names)  # 裸名，无前缀


def test_industry_dummies_missing_col_raises():
    from factorzen.risk.industry_factors import get_industry_dummies
    import pytest
    with pytest.raises(ValueError):
        get_industry_dummies(pl.DataFrame({"ts_code": ["000001.SZ"]}))
```

- [ ] **Step 2: 跑测试确认通过（代码已存在）**

Run: `pixi run pytest tests/test_risk_industry.py -v`
Expected: PASS（3 passed）—— risk 源码已实现，测试应直接通过。若失败则暴露真实 bug，修复 `industry_factors.py` 并记录。

- [ ] **Step 3: ruff 自查**

Run: `pixi run ruff check src/factorzen/risk/ tests/test_risk_industry.py`
Expected: 0 errors（若 risk 源码有 lint，顺手修）

- [ ] **Step 4: 提交（risk 源码 + 行业测试一并入库）**

```bash
git add src/factorzen/risk/__init__.py src/factorzen/risk/industry_factors.py src/factorzen/risk/exposures.py src/factorzen/risk/covariance.py src/factorzen/risk/style_factors.py src/factorzen/risk/model.py tests/test_risk_industry.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(risk): Barra 风险模型入库 + 行业因子测试"
```

---

## Task 2: 风格因子测试（style_factors.py）

**Files:**
- Test: `tests/test_risk_style_factors.py`

**Interfaces:**
- Consumes: `STYLE_FACTOR_NAMES`（8 个有序名）；`STYLE_FACTOR_REGISTRY`（dict name→fn）；`cs_standardize(df, factor_col="factor_value", method="mad") -> pl.DataFrame`；各因子 `fn(daily_data, daily_basic) -> pl.DataFrame[trade_date, ts_code, factor_value]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_risk_style_factors.py
import datetime as dt
import numpy as np
import polars as pl


def _trade_days(start, n):
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days


def make_daily_basic(n_stocks=8, n_days=10, seed=0):
    rng = np.random.default_rng(seed)
    days = _trade_days(dt.date(2023, 1, 3), n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        for d in days:
            rows.append({"trade_date": d, "ts_code": c,
                         "total_mv": float(abs(rng.standard_normal()) * 1e9 + 5e9),
                         "pb": float(abs(rng.standard_normal()) + 1.5),
                         "pe_ttm": float(abs(rng.standard_normal()) * 10 + 15),
                         "turnover_rate": float(abs(rng.standard_normal()) * 2 + 1)})
    return pl.DataFrame(rows)


def test_registry_has_eight_named_factors():
    from factorzen.risk.style_factors import STYLE_FACTOR_NAMES, STYLE_FACTOR_REGISTRY
    assert STYLE_FACTOR_NAMES == ["size", "value", "momentum", "volatility",
                                  "liquidity", "quality", "growth", "leverage"]
    assert set(STYLE_FACTOR_REGISTRY.keys()) == set(STYLE_FACTOR_NAMES)


def test_size_factor_shape():
    from factorzen.risk.style_factors import STYLE_FACTOR_REGISTRY
    db = make_daily_basic()
    out = STYLE_FACTOR_REGISTRY["size"](pl.DataFrame(), db)
    assert set(out.columns) >= {"trade_date", "ts_code", "factor_value"}
    assert out.height > 0


def test_cs_standardize_zero_mean_per_date():
    from factorzen.risk.style_factors import cs_standardize
    # 构造单日截面，标准化后均值≈0
    df = pl.DataFrame({"trade_date": [dt.date(2023, 1, 3)] * 30,
                       "ts_code": [f"{i:06d}.SZ" for i in range(30)],
                       "factor_value": np.random.default_rng(1).standard_normal(30) * 5 + 100})
    std = cs_standardize(df, factor_col="factor_value", method="mad")
    assert abs(std["factor_value"].mean()) < 0.5  # MAD-z 后截面均值接近 0


def test_cs_standardize_rejects_unknown_method():
    from factorzen.risk.style_factors import cs_standardize
    import pytest
    df = pl.DataFrame({"trade_date": [dt.date(2023, 1, 3)], "factor_value": [1.0]})
    with pytest.raises(ValueError):
        cs_standardize(df, method="zscore")
```

- [ ] **Step 2: 跑测试确认通过**

Run: `pixi run pytest tests/test_risk_style_factors.py -v`
Expected: PASS（4 passed）。若 `test_size_factor_shape` 失败（size 因子签名不符），按真实签名调整测试或修 bug。

- [ ] **Step 3: ruff + 提交**

```bash
pixi run ruff check src/factorzen/risk/ tests/test_risk_style_factors.py
git add tests/test_risk_style_factors.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "test(risk): 风格因子 + cs_standardize 测试"
```

---

## Task 3: 暴露矩阵测试（exposures.py）

**Files:**
- Test: `tests/test_risk_exposures.py`

**Interfaces:**
- Consumes: `compute_exposures(daily_data, daily_basic, stocks, trade_date) -> ExposureMatrix`；`ExposureMatrix(codes: list[str], factor_names: list[str], matrix: np.ndarray)`，`.n_stocks`/`.n_factors` property

- [ ] **Step 1: 写失败测试**

```python
# tests/test_risk_exposures.py
import datetime as dt
import numpy as np
import polars as pl


def _trade_days(start, n):
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days


def make_daily(n_stocks=8, n_days=20, seed=42):
    rng = np.random.default_rng(seed)
    days = _trade_days(dt.date(2023, 1, 3), n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = [{"trade_date": d, "ts_code": c, "pct_chg": float(rng.standard_normal() * 2.0)}
            for c in codes for d in days]
    return pl.DataFrame(rows)


def make_daily_basic(n_stocks=8, n_days=20, seed=0):
    rng = np.random.default_rng(seed)
    days = _trade_days(dt.date(2023, 1, 3), n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = [{"trade_date": d, "ts_code": c,
             "total_mv": float(abs(rng.standard_normal()) * 1e9 + 5e9),
             "pb": float(abs(rng.standard_normal()) + 1.5),
             "pe_ttm": float(abs(rng.standard_normal()) * 10 + 15),
             "turnover_rate": float(abs(rng.standard_normal()) * 2 + 1)}
            for c in codes for d in days]
    return pl.DataFrame(rows)


def make_stocks(n_stocks=8):
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    inds = ["银行", "医药", "电子", "食品饮料"]
    return pl.DataFrame({"ts_code": codes, "industry": [inds[i % 4] for i in range(n_stocks)]})


def test_compute_exposures_shape_and_factors():
    from factorzen.risk.exposures import compute_exposures
    daily, db, stocks = make_daily(), make_daily_basic(), make_stocks()
    target = daily["trade_date"].max()  # 用数据里实际存在的最后一个交易日
    exp = compute_exposures(daily, db, stocks, target)
    assert exp.n_stocks > 0
    assert exp.n_factors == exp.matrix.shape[1]
    assert exp.matrix.shape == (exp.n_stocks, exp.n_factors)
    # factor_names 含风格因子(小写)与行业列(ind_)
    assert any(f in exp.factor_names for f in ["size", "value"])
    assert any(f.startswith("ind_") for f in exp.factor_names)
    # 矩阵无 NaN（null 已填 0）
    assert not np.isnan(exp.matrix).any()
```

- [ ] **Step 2: 跑测试确认通过**

Run: `pixi run pytest tests/test_risk_exposures.py -v`
Expected: PASS。若失败（如 target 日无数据），用 `daily["trade_date"].max()` 确保有数据；若暴露真实 bug 则修 `exposures.py`。

- [ ] **Step 3: ruff + 提交**

```bash
pixi run ruff check src/factorzen/risk/ tests/test_risk_exposures.py
git add tests/test_risk_exposures.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "test(risk): compute_exposures + ExposureMatrix 测试"
```

---

## Task 4: 协方差/特质风险测试（covariance.py）

**Files:**
- Test: `tests/test_risk_covariance.py`

**Interfaces:**
- Consumes: `estimate_factor_covariance(factor_returns(T,K), half_life=90, nw_lags=2) -> (K,K)`；`estimate_specific_risk(residuals(T,N), half_life=90, shrinkage=0.3) -> (N,)`；`eigenvector_adjustment(cov(K,K), n_simulations=1000, seed=None) -> (K,K)`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_risk_covariance.py
import numpy as np


def test_factor_covariance_symmetric_psd():
    from factorzen.risk.covariance import estimate_factor_covariance
    rng = np.random.default_rng(0)
    fr = rng.standard_normal((120, 5))  # (T=120, K=5)
    cov = estimate_factor_covariance(fr, half_life=60, nw_lags=2)
    assert cov.shape == (5, 5)
    assert np.allclose(cov, cov.T, atol=1e-10)            # 对称
    assert np.linalg.eigvalsh(cov).min() >= -1e-8         # 半正定


def test_specific_risk_positive():
    from factorzen.risk.covariance import estimate_specific_risk
    rng = np.random.default_rng(0)
    resid = rng.standard_normal((120, 8))  # (T=120, N=8)
    sr = estimate_specific_risk(resid, half_life=60, shrinkage=0.3)
    assert sr.shape == (8,)
    assert (sr > 0).all()                                 # 特质风险全正


def test_eigenvector_adjustment_symmetric_same_shape():
    from factorzen.risk.covariance import eigenvector_adjustment
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 4))
    cov = a @ a.T  # 半正定对称
    adj = eigenvector_adjustment(cov, n_simulations=200, seed=1)
    assert adj.shape == (4, 4)
    assert np.allclose(adj, adj.T, atol=1e-8)


def test_covariance_too_short_returns_identity():
    from factorzen.risk.covariance import estimate_factor_covariance
    cov = estimate_factor_covariance(np.zeros((1, 3)), half_life=60)
    assert cov.shape == (3, 3)
```

- [ ] **Step 2: 跑测试确认通过**

Run: `pixi run pytest tests/test_risk_covariance.py -v`
Expected: PASS（4 passed）。若协方差非半正定 → 真实 bug，修 `covariance.py` 的特征值截断并加回归断言。

- [ ] **Step 3: ruff + 提交**

```bash
pixi run ruff check src/factorzen/risk/ tests/test_risk_covariance.py
git add tests/test_risk_covariance.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "test(risk): 因子协方差/特质风险/eigenvector 性质测试"
```

---

## Task 5: 风险模型测试（model.py：predict/decompose 手搓 + build 端到端）

**Files:**
- Test: `tests/test_risk_model.py`

**Interfaces:**
- Consumes: `RiskModel(...).build/predict_risk/decompose_risk`；`RiskModelResult(factor_exposures, factor_covariance, specific_risk, factor_returns, r_squared, factor_names)`；`ExposureMatrix(codes, factor_names, matrix)`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_risk_model.py
import datetime as dt
import math
import numpy as np
import polars as pl

from factorzen.risk.exposures import ExposureMatrix
from factorzen.risk.model import RiskModel, RiskModelResult


def _toy_result():
    """手搓一个 RiskModelResult，绕开截面回归，做确定性 predict/decompose 验证。"""
    codes = ["A", "B", "C"]
    factor_names = ["size", "value"]
    X = np.array([[1.0, 0.5], [0.8, -0.3], [-0.2, 1.1]])  # (3 stocks, 2 factors)
    F = np.array([[0.04, 0.01], [0.01, 0.09]])             # (2,2) 因子协方差
    D = np.array([0.10, 0.15, 0.20])                       # (3,) 特质风险（std）
    exp = ExposureMatrix(codes=codes, factor_names=factor_names, matrix=X)
    return RiskModelResult(factor_exposures=exp, factor_covariance=F,
                           specific_risk=D, factor_names=factor_names)


def test_predict_risk_positive():
    result = _toy_result()
    w = np.array([0.5, 0.3, 0.2])
    risk = RiskModel().predict_risk(w, result)
    assert risk > 0


def test_decompose_risk_variance_conservation():
    """factor_risk² + specific_risk² ≈ total_risk²（方差可加）。"""
    result = _toy_result()
    w = np.array([0.5, 0.3, 0.2])
    d = RiskModel().decompose_risk(w, result)
    assert {"total_risk", "factor_risk", "specific_risk"} <= set(d)
    assert math.isclose(d["factor_risk"]**2 + d["specific_risk"]**2,
                        d["total_risk"]**2, rel_tol=1e-9)
    # 每个因子名都有一个贡献键
    assert "size" in d and "value" in d


def test_build_end_to_end_r_squared_in_range():
    """端到端 build（mock 数据，n_days≥280 让 momentum 有值）→ R²∈[0,1]。"""
    rng = np.random.default_rng(7)
    days, d = [], dt.date(2023, 1, 3)
    while len(days) < 290:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(12)]
    daily = pl.DataFrame([{"trade_date": dd, "ts_code": c, "pct_chg": float(rng.standard_normal() * 2)}
                          for c in codes for dd in days])
    db = pl.DataFrame([{"trade_date": dd, "ts_code": c,
                        "total_mv": float(abs(rng.standard_normal()) * 1e9 + 5e9),
                        "pb": float(abs(rng.standard_normal()) + 1.5),
                        "pe_ttm": float(abs(rng.standard_normal()) * 10 + 15)}
                       for c in codes for dd in days])
    stocks = pl.DataFrame({"ts_code": codes,
                           "industry": [["银行", "医药", "电子"][i % 3] for i in range(12)]})
    start = days[260].strftime("%Y%m%d")
    end = days[-1].strftime("%Y%m%d")
    result = RiskModel().build(daily, db, stocks, start, end)
    assert 0.0 <= result.r_squared <= 1.0
    assert result.factor_covariance.shape[0] == result.factor_covariance.shape[1]
    assert len(result.factor_names) > 0
```

- [ ] **Step 2: 跑测试确认通过**

Run: `pixi run pytest tests/test_risk_model.py -v`
Expected: PASS（3 passed）。若风险守恒断言失败 → `decompose_risk` 真实 bug，修 `model.py` 并保留断言。

- [ ] **Step 3: ruff + 提交**

```bash
pixi run ruff check src/factorzen/risk/ tests/test_risk_model.py
git add tests/test_risk_model.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "test(risk): RiskModel predict/decompose（风险守恒）+ build 端到端"
```

---

## Task 6: 风险构建 pipeline + 轻量报告（risk_build.py）

**Files:**
- Create: `src/factorzen/pipelines/risk_build.py`
- Test: `tests/test_risk_build_pipeline.py`

**Interfaces:**
- Consumes: `RiskModel`（Task 1-5）
- Produces: `run_risk_build(daily, daily_basic, stocks, start, end, *, out_dir, cov_half_life=90, nw_lags=2, spec_half_life=90, spec_shrinkage=0.3, run_id=None) -> dict`（含 `run_dir`, `r_squared`, `factor_names`）；落 `exposures.parquet`/`factor_covariance.parquet`/`specific_risk.parquet`/`factor_returns.parquet`/`risk_summary.csv`/`manifest.json`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_risk_build_pipeline.py
import datetime as dt
import json
from pathlib import Path
import numpy as np
import polars as pl


def _mock(n_stocks=12, n_days=290, seed=3):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2023, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    daily = pl.DataFrame([{"trade_date": dd, "ts_code": c, "pct_chg": float(rng.standard_normal() * 2)}
                          for c in codes for dd in days])
    db = pl.DataFrame([{"trade_date": dd, "ts_code": c,
                        "total_mv": float(abs(rng.standard_normal()) * 1e9 + 5e9),
                        "pb": float(abs(rng.standard_normal()) + 1.5),
                        "pe_ttm": float(abs(rng.standard_normal()) * 10 + 15)}
                       for c in codes for dd in days])
    stocks = pl.DataFrame({"ts_code": codes,
                           "industry": [["银行", "医药", "电子"][i % 3] for i in range(n_stocks)]})
    return daily, db, stocks, days[260].strftime("%Y%m%d"), days[-1].strftime("%Y%m%d")


def test_run_risk_build_writes_artifacts(tmp_path: Path):
    from factorzen.pipelines.risk_build import run_risk_build
    daily, db, stocks, start, end = _mock()
    res = run_risk_build(daily, db, stocks, start, end, out_dir=str(tmp_path), run_id="t1")
    run_dir = Path(res["run_dir"])
    for f in ["exposures.parquet", "factor_covariance.parquet", "specific_risk.parquet",
              "factor_returns.parquet", "risk_summary.csv", "manifest.json"]:
        assert (run_dir / f).exists(), f
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert 0.0 <= manifest["r_squared"] <= 1.0
    assert "factor_names" in manifest
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_risk_build_pipeline.py -v`
Expected: FAIL（`ModuleNotFoundError: factorzen.pipelines.risk_build`）

- [ ] **Step 3: 实现 risk_build.py**

```python
# src/factorzen/pipelines/risk_build.py
"""风险模型构建 pipeline：build → 落产物 + 轻量风险报告。"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.risk import RiskModel


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_risk_build(daily, daily_basic, stocks, start, end, *, out_dir="workspace/risk_models",
                   cov_half_life=90, nw_lags=2, spec_half_life=90, spec_shrinkage=0.3,
                   run_id=None) -> dict:
    t0 = time.perf_counter()
    model = RiskModel(cov_half_life=cov_half_life, nw_lags=nw_lags,
                      spec_half_life=spec_half_life, spec_shrinkage=spec_shrinkage)
    result = model.build(daily, daily_basic, stocks, start, end)

    rid = run_id or f"risk_{start}_{end}"
    run_dir = Path(out_dir) / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    names = result.factor_names
    exp = result.factor_exposures
    # exposures.parquet
    if exp.n_stocks > 0:
        exp_df = pl.DataFrame({"ts_code": exp.codes}).hstack(
            pl.DataFrame(exp.matrix, schema=names))
    else:
        exp_df = pl.DataFrame({"ts_code": []})
    exp_df.write_parquet(run_dir / "exposures.parquet")
    # factor_covariance.parquet
    cov = result.factor_covariance
    cov_df = pl.DataFrame(cov, schema=names) if cov.size else pl.DataFrame()
    cov_df.write_parquet(run_dir / "factor_covariance.parquet")
    # specific_risk.parquet
    sr = result.specific_risk
    sr_df = pl.DataFrame({"ts_code": exp.codes, "specific_risk": sr.tolist()}) \
        if exp.n_stocks and sr.size else pl.DataFrame({"ts_code": [], "specific_risk": []})
    sr_df.write_parquet(run_dir / "specific_risk.parquet")
    # factor_returns.parquet
    result.factor_returns.write_parquet(run_dir / "factor_returns.parquet")

    # ── 轻量报告 risk_summary.csv ──
    factor_vol = np.sqrt(np.clip(np.diag(cov), 0, None)) if cov.size else np.array([])
    summary_rows = [{"factor": n, "factor_vol": float(factor_vol[i])} for i, n in enumerate(names)] \
        if factor_vol.size else []
    pl.DataFrame(summary_rows if summary_rows else {"factor": [], "factor_vol": []}) \
        .write_csv(run_dir / "risk_summary.csv")

    # 等权组合风险分解示例
    decomp = {}
    if exp.n_stocks > 0:
        w = np.full(exp.n_stocks, 1.0 / exp.n_stocks)
        decomp = model.decompose_risk(w, result)

    manifest = {"run_id": rid, "start": start, "end": end, "universe_size": exp.n_stocks,
                "cov_half_life": cov_half_life, "nw_lags": nw_lags,
                "spec_half_life": spec_half_life, "spec_shrinkage": spec_shrinkage,
                "r_squared": result.r_squared, "factor_names": names,
                "specific_risk_mean": float(sr.mean()) if sr.size else 0.0,
                "equal_weight_decomp": {k: round(v, 6) for k, v in decomp.items()},
                "git_sha": _git_sha(), "duration_seconds": round(time.perf_counter() - t0, 3)}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    return {"run_dir": str(run_dir), "r_squared": result.r_squared, "factor_names": names}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_risk_build_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: ruff + 提交**

```bash
pixi run ruff check src/factorzen/pipelines/risk_build.py tests/test_risk_build_pipeline.py
git add src/factorzen/pipelines/risk_build.py tests/test_risk_build_pipeline.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(risk): run_risk_build pipeline + 轻量风险报告"
```

---

## Task 7: CLI `fz risk build`

**Files:**
- Modify: `src/factorzen/cli/main.py`
- Test: `tests/test_risk_cli.py`

**Interfaces:**
- Consumes: `run_risk_build`（Task 6）；`build_parser`（现有）；`get_universe`/`loader.fetch_daily`/`fetch_daily_basic`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_risk_cli.py
def test_parser_has_risk_build():
    from factorzen.cli.main import build_parser
    p = build_parser()
    args = p.parse_args(["risk", "build", "--start", "20230101", "--end", "20241231",
                         "--universe", "csi500"])
    assert args.command == "risk"
    assert args.risk_command == "build"
    assert args.start == "20230101"
    assert args.universe == "csi500"
    assert callable(args.func)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_risk_cli.py -v`
Expected: FAIL（`AttributeError` / `risk_command`）

- [ ] **Step 3: 接入 CLI**

在 `build_parser()` 里（`mine` 组之后、`return parser` 前）加顶层 `risk` 组：
```python
    # ── fz risk ──（顶层命令组）
    risk = sub.add_parser("risk", help="Risk model workflows")
    risk_sub = risk.add_subparsers(dest="risk_command", required=True)
    r_build = risk_sub.add_parser("build", help="Build Barra risk model")
    r_build.add_argument("--start", required=True, help="Start date YYYYMMDD")
    r_build.add_argument("--end", required=True, help="End date YYYYMMDD")
    r_build.add_argument("--universe", default="all_a", help="Universe name")
    r_build.add_argument("--cov-half-life", type=int, default=90, dest="cov_half_life")
    r_build.add_argument("--nw-lags", type=int, default=2, dest="nw_lags")
    r_build.set_defaults(func=_cmd_risk_build)
```

模块顶层加（直调式，仿 `_cmd_factor_sweep`）：
```python
def _cmd_risk_build(args: argparse.Namespace) -> int:
    import polars as pl  # 局部 import，仿其它 _cmd 的延迟 import 惯例
    from factorzen.core import loader
    from factorzen.core.universe import get_universe
    from factorzen.pipelines.risk_build import run_risk_build
    stocks = get_universe(args.end, args.universe)  # 含 industry 列
    uni = stocks["ts_code"].to_list()
    daily = loader.fetch_daily(args.start, args.end).filter(pl.col("ts_code").is_in(uni))
    daily_basic = loader.fetch_daily_basic(args.start, args.end).filter(pl.col("ts_code").is_in(uni))
    res = run_risk_build(daily, daily_basic, stocks, args.start, args.end,
                         cov_half_life=args.cov_half_life, nw_lags=args.nw_lags)
    print(f"[risk] factors={len(res['factor_names'])} R2={res['r_squared']:.4f} → {res['run_dir']}")
    return 0
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_risk_cli.py -v`
Expected: PASS

- [ ] **Step 5: 全量质量门 + 提交**

```bash
pixi run pytest tests/test_risk_*.py -q
pixi run ruff check src/factorzen/risk/ src/factorzen/pipelines/risk_build.py tests/test_risk_*.py
git add src/factorzen/cli/main.py tests/test_risk_cli.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(risk): fz risk build CLI"
```

---

## 收尾验收（全部 task 完成后）

- [ ] `pixi run pytest tests/test_risk_*.py -q` 全绿（industry/style/exposures/covariance/model/pipeline/cli）
- [ ] `pixi run ruff check src/factorzen/risk/ src/factorzen/pipelines/risk_build.py tests/test_risk_*.py` 0 errors
- [ ] 协方差半正定 / 风险分解守恒 / R²∈[0,1] 等性质有断言保护
- [ ] 手动 smoke（需本地数据）：`pixi run fz risk build --start 20230101 --end 20241231 --universe csi500` → 产出 6 个产物 + manifest
- [ ] 测试暴露的 risk bug 已修并有回归断言
- [ ] `git status --short` 干净（risk 相关已入库，未带其它 M0 未提交改动）
- [ ] 更新 README「核心能力」表 + 本 plan 追加完成记录

---

*M3 收口后，Barra 风险模型有测试保护、能从 `fz risk build` 跑出、有可读摘要，成为可展示成果，并为 M4 组合优化铺路。*

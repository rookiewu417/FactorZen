# Factor Research Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current factor research framework reproducible and research-correct by fixing forward-return semantics, intraday session leakage, the combination CLI, type checking, CI, and documentation drift.

**Architecture:** Keep the existing package boundaries: shared data and storage stay under `common/`, daily factor evaluation stays under `daily/evaluation/`, intraday handling stays under `intraday/`, and CLI behavior stays under `scripts/`. The plan favors narrow fixes and regression tests over broad refactors.

**Tech Stack:** Python 3.10-3.12, Polars, NumPy, statsmodels, pytest, ruff, mypy, pixi, GitHub Actions.

---

## Current Baseline

Run these commands before starting and save the output in the task notes:

```powershell
pixi run test
pixi run lint
pixi run typecheck
git status --short --branch
```

Expected baseline:
- `pixi run test`: `189 passed`
- `pixi run lint`: `All checks passed!`
- `pixi run typecheck`: fails with 7 errors in `daily/preprocessing/neutralizer.py`, `daily/evaluation/turnover.py`, `daily/evaluation/backtest.py`, and `daily/evaluation/advanced.py`
- `git status`: dirty working tree with existing user changes; do not revert unrelated changes

## File Structure

- Modify `daily/evaluation/ic_analysis.py`: define daily forward returns as cumulative future holding-period returns.
- Modify `tests/test_walk_forward.py`: keep compatibility for ret-only test data.
- Create `tests/test_fwd_returns.py`: regression tests for one-day and multi-day forward-return semantics.
- Modify `intraday/evaluation/returns.py`: compute forward returns within each stock and trading date.
- Modify `intraday/preprocessing/pipeline.py`: forward-fill only within each stock and trading date.
- Modify `tests/test_intraday_returns.py`: add cross-day boundary regression test.
- Modify `tests/test_intraday_preprocessing.py`: add no-cross-day-fill regression test.
- Modify `scripts/run_combination.py`: instantiate factor classes correctly, remove `sys.path` mutation, and build daily returns before `compute_fwd_returns`.
- Modify `tests/test_combination.py`: add smoke coverage for combination return preparation and factor instantiation.
- Modify `daily/preprocessing/neutralizer.py`: make design matrices consistently two-dimensional.
- Modify `daily/evaluation/turnover.py`: handle empty turnover means safely.
- Modify `daily/evaluation/backtest.py`: type `summary_stats` as mixed int/string keys.
- Modify `daily/evaluation/advanced.py`: avoid unsafe regex match access.
- Modify `pixi.toml`: add Linux platform support for GitHub Actions.
- Update `pixi.lock`: regenerate after adding Linux platform.
- Modify `.github/workflows/ci.yml`: keep lint, typecheck, and tests as required checks after pixi can solve on Linux.
- Modify `README.md`: align project scope with current code, especially combination status and factor list.
- Modify `daily/combination/README.md`: document combination as experimental, in-sample research tooling unless walk-forward weights are added later.

---

### Task 1: Correct Daily Forward-Return Semantics

**Files:**
- Modify: `daily/evaluation/ic_analysis.py:21-44`
- Modify: `tests/test_walk_forward.py:61-84`
- Create: `tests/test_fwd_returns.py`

- [ ] **Step 1: Add failing daily forward-return tests**

Create `tests/test_fwd_returns.py`:

```python
from datetime import date

import polars as pl
import pytest

from daily.evaluation.ic_analysis import compute_fwd_returns


def test_fwd_ret_1d_uses_next_close_over_current_close():
    df = pl.DataFrame({
        "trade_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        "ts_code": ["000001.SZ"] * 3,
        "close": [100.0, 110.0, 121.0],
    }).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("ret")
    )

    out = compute_fwd_returns(df, horizons=[1], ret_col="ret")

    assert out["fwd_ret_1d"].to_list() == pytest.approx([0.10, 0.10, None])


def test_fwd_ret_5d_is_cumulative_holding_period_return():
    closes = [100.0, 101.0, 103.0, 106.0, 110.0, 115.0, 121.0]
    df = pl.DataFrame({
        "trade_date": [
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 1, 4),
            date(2024, 1, 5),
            date(2024, 1, 8),
            date(2024, 1, 9),
            date(2024, 1, 10),
        ],
        "ts_code": ["000001.SZ"] * len(closes),
        "close": closes,
    }).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("ret")
    )

    out = compute_fwd_returns(df, horizons=[5], ret_col="ret")

    assert out["fwd_ret_5d"][0] == pytest.approx(115.0 / 100.0 - 1.0)
    assert out["fwd_ret_5d"][1] == pytest.approx(121.0 / 101.0 - 1.0)
    assert out["fwd_ret_5d"].to_list()[-5:] == [None, None, None, None, None]


def test_fwd_returns_compound_from_ret_when_close_is_absent():
    df = pl.DataFrame({
        "trade_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)],
        "ts_code": ["000001.SZ"] * 4,
        "ret": [0.0, 0.10, 0.20, -0.05],
    })

    out = compute_fwd_returns(df, horizons=[2], ret_col="ret")

    assert out["fwd_ret_2d"][0] == pytest.approx((1.10 * 1.20) - 1.0)
    assert out["fwd_ret_2d"][1] == pytest.approx((1.20 * 0.95) - 1.0)
    assert out["fwd_ret_2d"].to_list()[-2:] == [None, None]
```

- [ ] **Step 2: Run the new tests and confirm the semantic failure**

Run:

```powershell
pixi run pytest tests/test_fwd_returns.py -v
```

Expected: `test_fwd_ret_5d_is_cumulative_holding_period_return` fails because current `fwd_ret_5d` is a shifted one-day return, not a five-day holding-period return.

- [ ] **Step 3: Replace `compute_fwd_returns` implementation**

In `daily/evaluation/ic_analysis.py`, replace `compute_fwd_returns()` with:

```python
def compute_fwd_returns(
    price_df: pl.DataFrame,
    horizons: list[int] | None = None,
    ret_col: str = "ret_1d",
    price_col: str = "close",
    code_col: str = "ts_code",
    date_col: str = "trade_date",
) -> pl.DataFrame:
    """Precompute forward holding-period returns.

    If ``price_col`` exists, fwd_ret_{h}d is close[t+h] / close[t] - 1.
    If only ``ret_col`` exists, fwd_ret_{h}d compounds returns from t+1 through t+h.
    """
    if horizons is None:
        horizons = [1, 5, 10, 20]

    df = price_df.sort([code_col, date_col])
    for h in horizons:
        if price_col in df.columns:
            future_price = pl.col(price_col).shift(-h).over(code_col)
            df = df.with_columns(
                (future_price / pl.col(price_col) - 1.0).alias(f"fwd_ret_{h}d")
            )
        else:
            compounded = pl.lit(1.0)
            for step in range(1, h + 1):
                compounded = compounded * (1.0 + pl.col(ret_col).shift(-step).over(code_col))
            df = df.with_columns((compounded - 1.0).alias(f"fwd_ret_{h}d"))
    return df
```

- [ ] **Step 4: Keep walk-forward synthetic test compatible**

In `tests/test_walk_forward.py`, update the synthetic `price_rows.append(...)` block to include a `close` path:

```python
close = 100.0
for d in dates:
    fv = rng.standard_normal(n_stocks)
    rets = rng.normal(0, 0.02, n_stocks)
    for i, s in enumerate(stocks):
        close_i = close * float(np.prod(1.0 + rets[: i + 1]))
        factor_rows.append({"trade_date": d, "ts_code": s, "factor_clean": float(fv[i])})
        price_rows.append({"trade_date": d, "ts_code": s, "ret": float(rets[i]), "close": close_i})
```

If this local block is awkward because the test loops by date and stock, use the ret-only fallback test above as the compatibility guarantee and leave `test_walk_forward.py` unchanged.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
pixi run pytest tests/test_fwd_returns.py tests/test_walk_forward.py tests/test_advanced.py -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 1**

```powershell
git add daily/evaluation/ic_analysis.py tests/test_fwd_returns.py tests/test_walk_forward.py
git commit -m "fix: compute cumulative forward holding returns"
```

---

### Task 2: Prevent Intraday Cross-Day Leakage

**Files:**
- Modify: `intraday/evaluation/returns.py:8-36`
- Modify: `intraday/preprocessing/pipeline.py:46-65`
- Modify: `tests/test_intraday_returns.py`
- Modify: `tests/test_intraday_preprocessing.py`

- [ ] **Step 1: Add failing intraday forward-return boundary test**

Append to `tests/test_intraday_returns.py`:

```python
from datetime import datetime

import polars as pl

from intraday.evaluation.returns import compute_intraday_fwd_returns


def test_fwd_return_does_not_cross_trading_day_boundary():
    df = pl.DataFrame({
        "trade_time": [
            datetime(2024, 1, 2, 14, 59),
            datetime(2024, 1, 2, 15, 0),
            datetime(2024, 1, 3, 9, 30),
            datetime(2024, 1, 3, 9, 31),
        ],
        "ts_code": ["000001.SZ"] * 4,
        "close": [100.0, 101.0, 200.0, 202.0],
    })

    out = compute_intraday_fwd_returns(df, periods=[1])

    assert out["fwd_ret_1bar"].to_list() == [0.01, None, 0.01, None]
```

- [ ] **Step 2: Add failing intraday fill boundary test**

Append to `tests/test_intraday_preprocessing.py`:

```python
from datetime import datetime

import polars as pl

from intraday.preprocessing.pipeline import fill_missing_bars


def test_fill_missing_bars_does_not_cross_trading_day_boundary():
    df = pl.DataFrame({
        "trade_time": [
            datetime(2024, 1, 2, 15, 0),
            datetime(2024, 1, 3, 9, 30),
            datetime(2024, 1, 3, 9, 31),
        ],
        "ts_code": ["000001.SZ"] * 3,
        "factor_value": [1.5, None, 2.0],
    })

    out = fill_missing_bars(df)

    assert out["factor_value"].to_list() == [1.5, None, 2.0]
```

- [ ] **Step 3: Run focused tests and confirm failures**

Run:

```powershell
pixi run pytest tests/test_intraday_returns.py tests/test_intraday_preprocessing.py -v
```

Expected: the two new boundary tests fail with cross-day leakage.

- [ ] **Step 4: Fix intraday forward returns**

In `intraday/evaluation/returns.py`, update `compute_intraday_fwd_returns()`:

```python
def compute_intraday_fwd_returns(
    minute_df: pl.DataFrame,
    periods: list[int] | None = None,
    close_col: str = "close",
    time_col: str = "trade_time",
    code_col: str = "ts_code",
) -> pl.DataFrame:
    """计算分钟级前向收益；不会跨交易日取下一根 bar。"""
    if periods is None:
        periods = [1, 5, 15, 60]

    helper_col = "_trade_date_for_fwd_ret"
    df = minute_df.sort([code_col, time_col]).with_columns(
        pl.col(time_col).dt.date().alias(helper_col)
    )
    group_keys = [code_col, helper_col]

    for n in periods:
        future_close = pl.col(close_col).shift(-n).over(group_keys)
        df = df.with_columns(
            (future_close / pl.col(close_col) - 1.0).alias(f"fwd_ret_{n}bar")
        )
    return df.drop(helper_col)
```

- [ ] **Step 5: Fix intraday forward fill**

In `intraday/preprocessing/pipeline.py`, replace `fill_missing_bars()` with:

```python
def fill_missing_bars(
    df: pl.DataFrame,
    time_col: str = "trade_time",
    group_col: str = "ts_code",
) -> pl.DataFrame:
    """Forward-fill 缺失的分钟 bar 因子值，但不跨交易日填充。"""
    helper_col = "_trade_date_for_fill"
    return (
        df.sort([group_col, time_col])
        .with_columns(pl.col(time_col).dt.date().alias(helper_col))
        .with_columns(pl.col("factor_value").forward_fill().over([group_col, helper_col]))
        .drop(helper_col)
    )
```

- [ ] **Step 6: Run intraday focused tests**

Run:

```powershell
pixi run pytest tests/test_intraday_returns.py tests/test_intraday_preprocessing.py tests/test_intraday_evaluation.py tests/test_intraday_backtest.py -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 2**

```powershell
git add intraday/evaluation/returns.py intraday/preprocessing/pipeline.py tests/test_intraday_returns.py tests/test_intraday_preprocessing.py
git commit -m "fix: keep intraday calculations within trading day"
```

---

### Task 3: Repair Combination CLI and Add Smoke Coverage

**Files:**
- Modify: `scripts/run_combination.py`
- Modify: `tests/test_combination.py`

- [ ] **Step 1: Add focused tests for combination helpers**

Add these helper tests to `tests/test_combination.py`:

```python
from datetime import date

from scripts.run_combination import _instantiate_factor, _prepare_return_frame


class _DummyFactor:
    required_data = ["daily"]
    lookback_days = 3


def test_instantiate_factor_builds_instance_from_registry_class():
    factor = _instantiate_factor("dummy", registry_getter=lambda _name: _DummyFactor)

    assert isinstance(factor, _DummyFactor)
    assert factor.required_data == ["daily"]
    assert factor.lookback_days == 3


def test_prepare_return_frame_adds_ret_and_forward_returns():
    price_df = pl.DataFrame({
        "trade_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        "ts_code": ["000001.SZ"] * 3,
        "close": [100.0, 110.0, 121.0],
    })

    out = _prepare_return_frame(price_df, horizons=[1])

    assert "ret" in out.columns
    assert "fwd_ret_1d" in out.columns
    assert out["ret"].to_list() == [None, 0.10, 0.10]
    assert out["fwd_ret_1d"].to_list() == [0.10, 0.10, None]
```

- [ ] **Step 2: Run focused combination tests and confirm failure**

Run:

```powershell
pixi run pytest tests/test_combination.py -v
```

Expected: import or attribute failure because `_instantiate_factor` and `_prepare_return_frame` do not exist yet.

- [ ] **Step 3: Refactor `scripts/run_combination.py` helpers**

Remove these imports from `scripts/run_combination.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
```

Add helper functions after imports:

```python
def _instantiate_factor(fname: str, registry_getter=get_factor):
    factor_cls = registry_getter(fname)
    return factor_cls()


def _prepare_return_frame(price_df: pl.DataFrame, horizons: list[int] | None = None) -> pl.DataFrame:
    ret_df = (
        price_df
        .select(["trade_date", "ts_code", "close"])
        .sort(["ts_code", "trade_date"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("ret")
        )
    )
    return compute_fwd_returns(ret_df, horizons=horizons, ret_col="ret")
```

- [ ] **Step 4: Fix factor instantiation in `main()`**

Replace the factor loop body in `scripts/run_combination.py` with:

```python
    for fname in args.factors:
        try:
            factor = _instantiate_factor(fname)
        except KeyError:
            print(f"[错误] 未知因子: {fname}")
            sys.exit(1)

        ctx = FactorDataContext(
            start=args.start,
            end=args.end,
            required_data=factor.required_data,
            lookback_days=factor.lookback_days,
        )
        raw = factor.compute(ctx)
        processed = quick_preprocess(raw)
        factor_dfs[fname] = processed.rename({"factor_clean": "factor_value"}).select(
            ["trade_date", "ts_code", "factor_value"]
        )
        print(f"  {fname}: {len(factor_dfs[fname])} 行")
```

Keep `import sys` because the script still uses `sys.exit(1)`.

- [ ] **Step 5: Fix return preparation in `main()`**

Replace:

```python
    ret_df = compute_fwd_returns(price_df, horizons=[1, 5], ret_col="ret")
```

with:

```python
    ret_df = _prepare_return_frame(price_df, horizons=[1, 5])
```

- [ ] **Step 6: Run focused tests**

Run:

```powershell
pixi run pytest tests/test_combination.py -v
pixi run ruff check scripts/run_combination.py tests/test_combination.py
```

Expected: tests pass and ruff reports no issues.

- [ ] **Step 7: Commit Task 3**

```powershell
git add scripts/run_combination.py tests/test_combination.py
git commit -m "fix: repair combination CLI data preparation"
```

---

### Task 4: Make Type Checking and CI Reproducible

**Files:**
- Modify: `daily/preprocessing/neutralizer.py`
- Modify: `daily/evaluation/turnover.py`
- Modify: `daily/evaluation/backtest.py`
- Modify: `daily/evaluation/advanced.py`
- Modify: `pixi.toml`
- Modify: `pixi.lock`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Reproduce type errors**

Run:

```powershell
pixi run typecheck
```

Expected: current mypy failures in the four daily modules.

- [ ] **Step 2: Fix neutralizer matrix typing**

In both `neutralize_ols()` and `neutralize_by_styles()`, use two-dimensional matrices consistently:

```python
X_parts: list[np.ndarray] = [np.ones((len(codes), 1), dtype=float)]
```

Keep existing dummy matrices as two-dimensional arrays and replace:

```python
X = np.column_stack(X_parts)
```

with:

```python
X = np.hstack(X_parts)
```

Use this same pattern in the style-neutralization branch where `sv.reshape(-1, 1)` is appended.

- [ ] **Step 3: Fix empty turnover mean typing**

In `daily/evaluation/turnover.py`, replace:

```python
    avg_turnover = float(daily_turnover["turnover"].mean())
```

with:

```python
    mean_turnover = daily_turnover["turnover"].mean() if not daily_turnover.is_empty() else 0.0
    avg_turnover = float(mean_turnover or 0.0)
```

- [ ] **Step 4: Fix backtest summary stats typing**

In `daily/evaluation/backtest.py`, change the dataclass field:

```python
    summary_stats: dict[int | str, dict[str, float]]
```

Change the local declaration:

```python
    summary_stats: dict[int | str, dict[str, float]] = {}
```

Change `_group_stats` signature:

```python
    def _group_stats(rets: np.ndarray) -> dict[str, float]:
```

- [ ] **Step 5: Fix unsafe regex match in IC decay**

In `daily/evaluation/advanced.py`, replace the horizon auto-detection block with:

```python
    if horizons is None:
        detected: list[int] = []
        for c in daily_ret.columns:
            match = re.fullmatch(r"fwd_ret_(\d+)d", c)
            if match is not None:
                detected.append(int(match.group(1)))
        horizons = sorted(detected)
```

- [ ] **Step 6: Run typecheck**

Run:

```powershell
pixi run typecheck
```

Expected: success.

- [ ] **Step 7: Add Linux platform for CI**

Run:

```powershell
pixi project platform add linux-64
pixi lock
```

Expected: `pixi.toml` includes both `win-64` and `linux-64`, and `pixi.lock` is updated.

- [ ] **Step 8: Keep CI quality gates strict**

Confirm `.github/workflows/ci.yml` keeps these steps in order:

```yaml
      - name: Lint
        run: pixi run lint

      - name: Type check
        run: pixi run typecheck

      - name: Test
        env:
          TUSHARE_TOKEN: ${{ secrets.TUSHARE_TOKEN }}
        run: pixi run test
```

If CI still cannot solve on Ubuntu after adding `linux-64`, change `runs-on` to `windows-latest` instead of removing typecheck.

- [ ] **Step 9: Run local full verification**

Run:

```powershell
pixi run lint
pixi run typecheck
pixi run test
```

Expected: all three pass.

- [ ] **Step 10: Commit Task 4**

```powershell
git add daily/preprocessing/neutralizer.py daily/evaluation/turnover.py daily/evaluation/backtest.py daily/evaluation/advanced.py pixi.toml pixi.lock .github/workflows/ci.yml
git commit -m "fix: restore typecheck and CI solvability"
```

---

### Task 5: Align Documentation With Current Scope

**Files:**
- Modify: `README.md`
- Modify: `daily/combination/README.md`
- Optionally modify: `tick/README.md`

- [ ] **Step 1: Update README project status**

In `README.md`, replace the opening status sentence with:

```markdown
> **当前阶段**：核心主线是**单因子研究**——因子计算、预处理、IC/回测评估、Tear Sheet 报告生成。`daily/combination/` 已提供实验性多因子合成工具，用于研究对比，不作为当前生产组合优化模块。
```

- [ ] **Step 2: Update factor list**

In `README.md`, ensure the daily list includes `momentum_12_1`:

```markdown
| `momentum_12_1` | 动量 | Jegadeesh-Titman 12-1 动量，剔除最近 1 个月反转效应 |
```

Keep `momentum_20d` but mark it as legacy:

```markdown
| `momentum_20d` | 动量 | 20 日价格动量；保留兼容，研究上建议优先使用 `momentum_12_1` |
```

- [ ] **Step 3: Update combination section**

In `README.md`, replace the old “不实现” combination wording with:

```markdown
- **`daily/combination/`**：实验性多因子合成（等权、IC 加权、Max-IR），用于研究阶段对比。当前权重估计仍是 in-sample 口径，不能直接解释为生产可交易组合优化。
```

- [ ] **Step 4: Update `daily/combination/README.md` risk label**

At the top of `daily/combination/README.md`, add:

```markdown
> 状态：实验性研究工具。`ic_weighted` 和 `max_ir` 当前使用样本内 IC 估计权重，适合方法对比和候选因子筛选，不应用作无偏的样本外组合表现。
```

- [ ] **Step 5: Run documentation-adjacent checks**

Run:

```powershell
pixi run ruff check README.md daily/combination/README.md
pixi run pytest tests/test_reporting.py tests/test_combination.py -v
```

Expected: ruff ignores Markdown or reports no Python issues; selected tests pass.

- [ ] **Step 6: Commit Task 5**

```powershell
git add README.md daily/combination/README.md tick/README.md
git commit -m "docs: align scope with current factor tooling"
```

---

### Task 6: Clean Migration State Without Losing History

**Files:**
- Review: `.sisyphus/**`
- Review: `docs/archive/**`
- Review: deleted legacy scripts and tests from `git status`
- Modify only if needed: `.gitignore`

- [ ] **Step 1: Confirm archive copy exists before staging `.sisyphus` deletions**

Run:

```powershell
git ls-files .sisyphus
git ls-files -o --exclude-standard docs/archive
```

Expected: deleted `.sisyphus` files have corresponding archived files under `docs/archive/`.

- [ ] **Step 2: Confirm legacy script replacements**

Run:

```powershell
Test-Path scripts/run_lft_single.py
Test-Path scripts/run_daily_single.py
Test-Path scripts/run_lft_compare.py
Test-Path scripts/run_daily_compare.py
```

Expected: old `run_lft_*` paths are absent and new `run_daily_*` paths exist.

- [ ] **Step 3: Confirm old MFT tests were renamed, not lost**

Run:

```powershell
Test-Path tests/test_mft_data_context.py
Test-Path tests/test_intraday_data_context.py
Test-Path tests/test_mft_demo.py
Test-Path tests/test_intraday_demo.py
Test-Path tests/test_mft_factor_base.py
Test-Path tests/test_intraday_factor_base.py
Test-Path tests/test_mft_preprocessing.py
Test-Path tests/test_intraday_preprocessing.py
```

Expected: old `test_mft_*` paths are absent and new `test_intraday_*` paths exist.

- [ ] **Step 4: Stage migration cleanup intentionally**

If Steps 1-3 match expectations, stage the migration:

```powershell
git add -A docs/archive .sisyphus scripts/run_lft_single.py scripts/run_lft_compare.py scripts/run_daily_single.py scripts/run_daily_compare.py tests/test_mft_data_context.py tests/test_mft_demo.py tests/test_mft_factor_base.py tests/test_mft_preprocessing.py tests/test_intraday_data_context.py tests/test_intraday_demo.py tests/test_intraday_factor_base.py tests/test_intraday_preprocessing.py
```

- [ ] **Step 5: Review staged diff**

Run:

```powershell
git diff --cached --stat
git diff --cached --name-status
```

Expected: archive move/rename intent is clear; no raw data or output files are staged.

- [ ] **Step 6: Commit Task 6**

```powershell
git commit -m "chore: archive legacy planning artifacts"
```

---

### Task 7: Final Verification and Release Readiness

**Files:**
- No direct file changes unless a verification failure exposes a defect.

- [ ] **Step 1: Run formatting**

```powershell
pixi run format
```

Expected: files are formatted. If files change, inspect with `git diff`.

- [ ] **Step 2: Run full local gates**

```powershell
pixi run lint
pixi run typecheck
pixi run test
```

Expected: all pass.

- [ ] **Step 3: Run package import smoke**

```powershell
pixi run smoke
pixi run python -c "from daily.factors.registry import list_factors; print(list_factors())"
pixi run python -c "from intraday.factors.registry import list_factors; print(list_factors())"
```

Expected: `smoke` prints `ok`; daily list includes `momentum_12_1`; intraday list includes `momentum_1min` and `vwap_deviation`.

- [ ] **Step 4: Check working tree**

```powershell
git status --short --branch
```

Expected: only intended plan/document files or no changes remain. If uncommitted changes remain, classify them as planned follow-up, user-owned changes, or stage them into the relevant commit.

- [ ] **Step 5: Commit final formatting changes if any**

```powershell
git add -A
git commit -m "chore: format remediation changes"
```

Run this commit only if Step 1 changed files not already committed in prior tasks.

---

## Self-Review

- Spec coverage: daily forward-return semantics are covered by Task 1; intraday leakage by Task 2; broken combination CLI by Task 3; mypy and CI by Task 4; documentation drift by Task 5; migration hygiene by Task 6; final verification by Task 7.
- Placeholder scan: no task uses deferred placeholder wording; each code change has concrete target files and code snippets.
- Type consistency: `compute_fwd_returns()` keeps existing callers compatible through the `ret_col` fallback while adding `price_col`; mixed `summary_stats` keys are typed as `dict[int | str, dict[str, float]]`; intraday helper columns are dropped before returning.

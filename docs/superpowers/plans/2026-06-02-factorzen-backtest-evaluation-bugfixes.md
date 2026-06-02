# FactorZen Backtest Evaluation Bugfixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove future-information leakage and fix critical return/NAV/statistic formulas in FactorZen's daily backtest and evaluation pipeline.

**Architecture:** Keep the public API stable where possible. Make execution-time data availability explicit in `daily/evaluation/backtest.py`, keep return labels in `pipelines/daily_single.py`, and keep summary/statistic fixes local to their owning modules. Tests are written first for each behavior so every accounting decision is executable.

**Tech Stack:** Python 3.12, Polars, NumPy, pytest, pixi. Do not call global `python`, `python3`, or `pip`; use `pixi run ...`.

---

## File Structure

- Modify: `src/factorzen/daily/evaluation/backtest.py`
  - Owns next-open execution, trade constraints, position drift, NAV, and portfolio summary stats.
- Modify: `src/factorzen/pipelines/daily_single.py`
  - Builds forward-return labels used by IC and advanced evaluation.
- Modify: `src/factorzen/daily/evaluation/walk_forward.py`
  - Computes stitched OOS NAV and OOS max drawdown.
- Test: `tests/test_strategy_backtest.py`
  - Add execution-leakage and portfolio-accounting regression tests.
- Test: `tests/test_backtest.py`
  - Align NAV-start expectation with the repaired NAV series.
- Test: `tests/test_run_daily_single_config.py`
  - Add adjusted-close label regression test.
- Test: `tests/test_walk_forward_strategy.py`
  - Add OOS drawdown base-NAV regression test.

## Guardrails

- Before any commit, set:

```powershell
git config user.name rookiewu417
git config user.email 1007372080@qq.com
```

- After every commit, verify:

```powershell
git log -1 --format='%an <%ae>'
```

Expected:

```text
rookiewu417 <1007372080@qq.com>
```

- Do not run training. If any training command is introduced later, run `nvidia-smi` first.
- Keep changes surgical. Do not reorganize the repository or clean unrelated deleted legacy paths.

---

### Task 1: Baseline Reproduction

**Files:**
- No code changes.

- [ ] **Step 1: Record workspace status**

Run:

```powershell
git status --short
```

Expected: large existing dirty worktree is allowed. Do not revert unrelated files.

- [ ] **Step 2: Reproduce the targeted current failure**

Run:

```powershell
pixi run pytest tests\test_backtest.py::test_nav_starts_near_one -q
```

Expected before fixes: FAIL because current `result.nav` starts after the first return period, not at base NAV `1.0`.

- [ ] **Step 3: Run the existing high-risk subset**

Run:

```powershell
pixi run pytest tests\test_lookahead_safety.py tests\test_backtest.py tests\test_strategy_backtest.py tests\test_fwd_returns.py tests\test_intraday_evaluation.py tests\test_intraday_returns.py -q
```

Expected before fixes: one known failure in `tests/test_backtest.py::test_nav_starts_near_one`.

---

### Task 2: Remove Future Data From Next-Open Trade Constraints

**Files:**
- Modify: `tests/test_strategy_backtest.py`
- Modify: `src/factorzen/daily/evaluation/backtest.py`

- [ ] **Step 1: Add regression tests for close-based limit leakage and same-day amount leakage**

Append these tests after `test_capacity_constraint_partially_fills_trade` in `tests/test_strategy_backtest.py`:

```python
def test_next_open_buy_is_not_blocked_by_same_day_close_limit():
    prices = pl.DataFrame(
        [
            {
                "trade_date": date(2024, 1, 1),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
            {
                "trade_date": date(2024, 1, 2),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 11.0,
                "pre_close": 10.0,
                "pct_chg": 10.0,
                "vol": 1000.0,
                "amount": 100_000_000.0,
            },
        ]
    )

    result = run_strategy_backtest(
        BuyOneStrategy(),
        _factor(),
        prices,
        config=BacktestConfig(initial_capital=1_000_000, max_participation_rate=1.0),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )

    trade = result.trades.sort("trade_date").row(0, named=True)
    assert trade["filled_delta_weight"] == pytest.approx(1.0)
    assert trade["block_reason"] == ""


def test_next_open_buy_is_blocked_when_open_is_at_limit_up():
    prices = pl.DataFrame(
        [
            {
                "trade_date": date(2024, 1, 1),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
            {
                "trade_date": date(2024, 1, 2),
                "ts_code": "000001.SZ",
                "open": 11.0,
                "close": 11.0,
                "pre_close": 10.0,
                "pct_chg": 10.0,
                "vol": 1000.0,
                "amount": 100_000_000.0,
            },
        ]
    )

    result = run_strategy_backtest(
        BuyOneStrategy(),
        _factor(),
        prices,
        config=BacktestConfig(initial_capital=1_000_000, max_participation_rate=1.0),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )

    trade = result.trades.sort("trade_date").row(0, named=True)
    assert trade["filled_delta_weight"] == pytest.approx(0.0)
    assert trade["block_reason"] == "limit_up"


def test_capacity_uses_trailing_adv_not_execution_day_amount():
    prices = pl.DataFrame(
        [
            {
                "trade_date": date(2024, 1, 1),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 100.0,
            },
            {
                "trade_date": date(2024, 1, 2),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 100_000_000.0,
            },
        ]
    )

    result = run_strategy_backtest(
        BuyOneStrategy(),
        _factor(),
        prices,
        config=BacktestConfig(initial_capital=100.0, max_participation_rate=0.1),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )

    trade = result.trades.sort("trade_date").row(0, named=True)
    assert trade["filled_delta_weight"] == pytest.approx(0.1)
    assert trade["block_reason"] == "capacity"
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```powershell
pixi run pytest tests\test_strategy_backtest.py::test_next_open_buy_is_not_blocked_by_same_day_close_limit tests\test_strategy_backtest.py::test_next_open_buy_is_blocked_when_open_is_at_limit_up tests\test_strategy_backtest.py::test_capacity_uses_trailing_adv_not_execution_day_amount -q
```

Expected before implementation: at least the first and third tests fail.

- [ ] **Step 3: Change `_apply_trade_constraints` to use open-time information and trailing ADV**

In `src/factorzen/daily/evaluation/backtest.py`, change the call site inside `run_strategy_backtest`:

```python
filled_delta, reason = _apply_trade_constraints(
    code=code,
    delta=target_weight - prev_weight,
    price_map=price_map,
    portfolio_value=nav_value * cfg.initial_capital,
    config=cfg,
    adv=adv_20d.get(code),
)
```

Replace `_apply_trade_constraints` with:

```python
def _apply_trade_constraints(
    *,
    code: str,
    delta: float,
    price_map: dict[str, dict[str, Any]],
    portfolio_value: float,
    config: BacktestConfig,
    adv: float | None = None,
) -> tuple[float, str]:
    if abs(delta) < 1e-12:
        return 0.0, ""

    rec = price_map.get(code)
    if rec is None or rec.get("open") is None or rec.get("pre_close") is None:
        return 0.0, "missing_price"

    open_price = float(rec["open"])
    pre_close = float(rec["pre_close"])
    if not np.isfinite(open_price) or not np.isfinite(pre_close) or open_price <= 0 or pre_close <= 0:
        return 0.0, "missing_price"

    opening_pct = (open_price / pre_close - 1.0) * 100.0
    board_limit_pct = _get_board_limit(code) * 100 if code else config.limit_up_pct
    if delta > 0 and opening_pct >= board_limit_pct:
        return 0.0, "limit_up"
    if delta < 0 and opening_pct <= -board_limit_pct:
        return 0.0, "limit_down"

    liquidity_base = float(adv) if adv is not None and adv > 0 else None
    if liquidity_base is None and config.fallback_adv is not None and config.fallback_adv > 0:
        liquidity_base = float(config.fallback_adv)
    if liquidity_base is not None:
        max_trade_value = liquidity_base * config.max_participation_rate
        if portfolio_value <= 0:
            return 0.0, "invalid_portfolio_value"
        max_delta = max_trade_value / portfolio_value
        if abs(delta) > max_delta + 1e-12:
            return float(np.sign(delta) * max_delta), "capacity"

    return delta, ""
```

Note: this intentionally removes full-day `pct_chg`, `vol`, and execution-day `amount` from next-open execution constraints.

- [ ] **Step 4: Update suspension-specific tests to model missing open price**

In `tests/test_strategy_backtest.py`, update `test_suspended_stock_blocks_trade` so it passes `open=None` rather than `day2_vol=0.0`, because daily full-session `vol` is not known at next-open decision time.

Use this local price fixture in the test:

```python
prices = _prices().with_columns(
    pl.when(pl.col("trade_date") == date(2024, 1, 2))
    .then(None)
    .otherwise(pl.col("open"))
    .alias("open")
)
result = run_strategy_backtest(BuyOneStrategy(), _factor(), prices)
```

Keep the assertion:

```python
assert trade["block_reason"] == "missing_price"
```

- [ ] **Step 5: Run the execution constraint test file**

Run:

```powershell
pixi run pytest tests\test_strategy_backtest.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src\factorzen\daily\evaluation\backtest.py tests\test_strategy_backtest.py
git commit -m "fix: remove future data from next-open constraints"
git log -1 --format='%an <%ae>'
```

Expected author:

```text
rookiewu417 <1007372080@qq.com>
```

---

### Task 3: Fix Portfolio Return Compounding And Position Drift

**Files:**
- Modify: `tests/test_strategy_backtest.py`
- Modify: `src/factorzen/daily/evaluation/backtest.py`

- [ ] **Step 1: Add regression tests for compounding and close-weight drift**

Append these tests to `tests/test_strategy_backtest.py`:

```python
class EqualTwoStockStrategy(Strategy):
    name = "equal_two_stock"

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "target_weight": [0.5, 0.5],
            }
        )


def test_overnight_and_intraday_returns_are_compounded():
    prices = pl.DataFrame(
        [
            {
                "trade_date": date(2024, 1, 1),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
            {
                "trade_date": date(2024, 1, 2),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 11.0,
                "pre_close": 10.0,
                "pct_chg": 10.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
            {
                "trade_date": date(2024, 1, 3),
                "ts_code": "000001.SZ",
                "open": 12.1,
                "close": 13.31,
                "pre_close": 11.0,
                "pct_chg": 21.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
        ]
    )
    factors = _factor(
        [
            (date(2024, 1, 1), "000001.SZ", 1.0),
            (date(2024, 1, 2), "000001.SZ", 1.0),
        ]
    )

    result = run_strategy_backtest(
        BuyOneStrategy(),
        factors,
        prices,
        config=BacktestConfig(initial_capital=1_000_000, max_participation_rate=1.0),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )

    day3 = result.returns.filter(pl.col("trade_date") == date(2024, 1, 3)).row(0, named=True)
    assert day3["gross_return"] == pytest.approx((1.10 * 1.10) - 1.0)
    assert day3["net_return"] == pytest.approx((1.10 * 1.10) - 1.0)


def test_positions_are_recorded_as_close_weights_after_intraday_drift():
    prices = pl.DataFrame(
        [
            {
                "trade_date": date(2024, 1, 1),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
            {
                "trade_date": date(2024, 1, 1),
                "ts_code": "000002.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
            {
                "trade_date": date(2024, 1, 2),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 11.0,
                "pre_close": 10.0,
                "pct_chg": 10.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
            {
                "trade_date": date(2024, 1, 2),
                "ts_code": "000002.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
        ]
    )
    factors = pl.DataFrame(
        [
            {"trade_date": date(2024, 1, 1), "ts_code": "000001.SZ", "factor_clean": 1.0},
            {"trade_date": date(2024, 1, 1), "ts_code": "000002.SZ", "factor_clean": 1.0},
        ]
    )

    result = run_strategy_backtest(
        EqualTwoStockStrategy(),
        factors,
        prices,
        config=BacktestConfig(initial_capital=1_000_000, max_participation_rate=1.0),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )

    positions = result.positions.filter(pl.col("trade_date") == date(2024, 1, 2))
    weights = dict(zip(positions["ts_code"].to_list(), positions["weight"].to_list(), strict=True))
    assert weights["000001.SZ"] == pytest.approx(0.55 / 1.05)
    assert weights["000002.SZ"] == pytest.approx(0.50 / 1.05)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```powershell
pixi run pytest tests\test_strategy_backtest.py::test_overnight_and_intraday_returns_are_compounded tests\test_strategy_backtest.py::test_positions_are_recorded_as_close_weights_after_intraday_drift -q
```

Expected before implementation: both tests fail.

- [ ] **Step 3: Generalize `_drift_weights`**

In `src/factorzen/daily/evaluation/backtest.py`, replace `_drift_weights` with:

```python
def _drift_weights(
    weights: dict[str, float],
    price_map: dict[str, dict[str, Any]],
    portfolio_return: float,
    return_col: str = "overnight_ret",
) -> dict[str, float]:
    denom = 1.0 + portfolio_return
    if abs(denom) < 1e-12:
        return dict(weights)
    drifted = {}
    for code, weight in weights.items():
        rec = price_map.get(code)
        asset_ret = float(rec[return_col]) if rec is not None and rec.get(return_col) is not None else 0.0
        next_weight = weight * (1.0 + asset_ret) / denom
        if abs(next_weight) >= 1e-12:
            drifted[code] = next_weight
    return drifted
```

- [ ] **Step 4: Compound daily return and carry close weights into the next day**

In `run_strategy_backtest`, replace the block from intraday return through position recording with:

```python
intraday_return = _weighted_return(next_weights, price_map, "intraday_ret")
gross_return = (1.0 + overnight_return) * (1.0 + intraday_return) - 1.0

borrow_cost = 0.0
if cost_model is not None:
    short_exposure = sum(abs(w) for w in next_weights.values() if w < 0)
    borrow_cost = short_exposure * cost_model.borrow_rate_per_period(cfg.frequency)
net_return = gross_return - trade_cost - borrow_cost

close_weights = _drift_weights(
    next_weights,
    price_map,
    intraday_return,
    return_col="intraday_ret",
)
if abs(1.0 + net_return) >= 1e-12:
    exposure_scale = (1.0 + gross_return) / (1.0 + net_return)
    close_weights = {
        code: weight * exposure_scale
        for code, weight in close_weights.items()
        if abs(weight * exposure_scale) >= 1e-12
    }

nav_value *= 1.0 + net_return
cash_weight = 1.0 - sum(close_weights.values())
if has_started:
    nav_rows.append(
        {
            "trade_date": execution_date,
            "gross_return": gross_return,
            "cost": trade_cost,
            "borrow_cost": borrow_cost,
            "net_return": net_return,
            "nav": nav_value,
            "cash_weight": cash_weight,
            "turnover": turnover,
        }
    )
    for code, weight in sorted(close_weights.items()):
        position_rows.append(
            {
                "trade_date": execution_date,
                "ts_code": code,
                "weight": weight,
                "market_value": weight * nav_value * cfg.initial_capital,
            }
        )
current_weights = close_weights
```

Remove the old trailing `current_weights = next_weights`.

- [ ] **Step 5: Run strategy backtest tests**

Run:

```powershell
pixi run pytest tests\test_strategy_backtest.py tests\test_backtest_costs.py tests\test_rebalance_threshold.py -q
```

Expected: PASS, except tests that intentionally assert old NAV semantics and are handled in Task 4.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src\factorzen\daily\evaluation\backtest.py tests\test_strategy_backtest.py
git commit -m "fix: compound returns and drift close weights"
git log -1 --format='%an <%ae>'
```

Expected author:

```text
rookiewu417 <1007372080@qq.com>
```

---

### Task 4: Normalize NAV And Max Drawdown Semantics

**Files:**
- Modify: `tests/test_backtest.py`
- Modify: `tests/test_strategy_backtest.py`
- Modify: `tests/test_walk_forward_strategy.py`
- Modify: `src/factorzen/daily/evaluation/backtest.py`
- Modify: `src/factorzen/daily/evaluation/walk_forward.py`

- [ ] **Step 1: Add summary-stat max-drawdown regression test**

In `tests/test_strategy_backtest.py`, import `_summary_stats` from `factorzen.daily.evaluation.backtest`, then add:

```python
def test_summary_max_drawdown_includes_initial_nav():
    returns = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2)],
            "gross_return": [-0.10],
            "cost": [0.0],
            "borrow_cost": [0.0],
            "net_return": [-0.10],
            "nav": [0.90],
            "cash_weight": [0.0],
            "turnover": [0.0],
        }
    )
    trades = pl.DataFrame(
        schema={
            "trade_date": pl.Date,
            "ts_code": pl.Utf8,
            "prev_weight": pl.Float64,
            "target_weight": pl.Float64,
            "filled_delta_weight": pl.Float64,
            "turnover": pl.Float64,
            "cost": pl.Float64,
            "block_reason": pl.Utf8,
        }
    )

    stats = _summary_stats(returns, trades)

    assert stats["portfolio"]["max_dd"] == pytest.approx(-0.10)
```

- [ ] **Step 2: Update NAV-start expectations**

Update `tests/test_strategy_backtest.py::test_next_open_execution_starts_when_prior_signal_is_available` to assert separate `returns` and `nav` semantics:

```python
returns = result.returns.sort("trade_date")
nav = result.nav.sort("trade_date")
assert returns["trade_date"][0] == date(2024, 1, 2)
assert returns["nav"][0] == pytest.approx(1.1)
assert nav["trade_date"][0] == date(2024, 1, 1)
assert nav["nav"][0] == pytest.approx(1.0)
assert nav["trade_date"][1] == date(2024, 1, 2)
assert nav["nav"][1] == pytest.approx(1.1)
```

Keep `tests/test_backtest.py::test_nav_starts_near_one` expecting `first_nav == 1.0`.

- [ ] **Step 3: Add walk-forward OOS max-drawdown regression test**

In `tests/test_walk_forward_strategy.py`, import `_compute_oos_max_dd` from `factorzen.daily.evaluation.walk_forward`, then add:

```python
def test_oos_max_drawdown_includes_initial_nav():
    assert _compute_oos_max_dd([0.90]) == pytest.approx(-0.10)
```

- [ ] **Step 4: Run the new tests to verify they fail**

Run:

```powershell
pixi run pytest tests\test_strategy_backtest.py::test_summary_max_drawdown_includes_initial_nav tests\test_backtest.py::test_nav_starts_near_one tests\test_walk_forward_strategy.py::test_oos_max_drawdown_includes_initial_nav -q
```

Expected before implementation: failures in NAV start and max drawdown.

- [ ] **Step 5: Build `nav` with an explicit base row while leaving `returns` as period returns**

In `run_strategy_backtest`, after constructing `returns`, replace the current `nav = returns.select(...)` with:

```python
if returns.is_empty():
    nav = returns.select(
        ["trade_date", "gross_return", "cost", "borrow_cost", "net_return", "nav", "cash_weight"]
    )
else:
    first_return_date = returns.sort("trade_date")["trade_date"][0]
    first_signal_date = trade_dates[trade_dates.index(first_return_date) - 1]
    base_nav = pl.DataFrame(
        [
            {
                "trade_date": first_signal_date,
                "gross_return": 0.0,
                "cost": 0.0,
                "borrow_cost": 0.0,
                "net_return": 0.0,
                "nav": 1.0,
                "cash_weight": 1.0,
            }
        ],
        schema={
            "trade_date": pl.Date,
            "gross_return": pl.Float64,
            "cost": pl.Float64,
            "borrow_cost": pl.Float64,
            "net_return": pl.Float64,
            "nav": pl.Float64,
            "cash_weight": pl.Float64,
        },
    )
    nav = pl.concat(
        [
            base_nav,
            returns.select(
                [
                    "trade_date",
                    "gross_return",
                    "cost",
                    "borrow_cost",
                    "net_return",
                    "nav",
                    "cash_weight",
                ]
            ),
        ],
        how="vertical",
    )
```

- [ ] **Step 6: Include base NAV in summary max drawdown**

In `_summary_stats`, replace:

```python
cum = np.cumprod(1 + valid)
max_dd = float(np.min(cum / np.maximum.accumulate(cum) - 1))
```

with:

```python
cum = np.concatenate([[1.0], np.cumprod(1 + valid)])
max_dd = float(np.min(cum / np.maximum.accumulate(cum) - 1))
```

- [ ] **Step 7: Include base NAV in walk-forward OOS max drawdown**

In `src/factorzen/daily/evaluation/walk_forward.py`, replace `_compute_oos_max_dd` with:

```python
def _compute_oos_max_dd(nav_series: list[float]) -> float:
    """Compute max drawdown including the initial base NAV of 1.0."""
    if not nav_series:
        return 0.0
    arr = np.concatenate([[1.0], np.array(nav_series, dtype=float)])
    running_max = np.maximum.accumulate(arr)
    dd = arr / running_max - 1.0
    return float(np.min(dd))
```

- [ ] **Step 8: Run NAV/stat tests**

Run:

```powershell
pixi run pytest tests\test_backtest.py tests\test_strategy_backtest.py tests\test_walk_forward_strategy.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

Run:

```powershell
git add src\factorzen\daily\evaluation\backtest.py src\factorzen\daily\evaluation\walk_forward.py tests\test_backtest.py tests\test_strategy_backtest.py tests\test_walk_forward_strategy.py
git commit -m "fix: normalize nav and drawdown baselines"
git log -1 --format='%an <%ae>'
```

Expected author:

```text
rookiewu417 <1007372080@qq.com>
```

---

### Task 5: Use Adjusted Close For IC Forward-Return Labels

**Files:**
- Modify: `tests/test_run_daily_single_config.py`
- Modify: `src/factorzen/pipelines/daily_single.py`

- [ ] **Step 1: Add a focused helper regression test**

Add this test to `tests/test_run_daily_single_config.py`:

```python
def test_build_forward_return_frame_prefers_adjusted_close():
    from factorzen.pipelines.daily_single import _build_forward_return_frame

    daily = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3)],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "close": [10.0, 5.0],
            "close_adj": [10.0, 10.0],
        }
    )

    ret_df = _build_forward_return_frame(daily)

    assert ret_df["ret"][1] == pytest.approx(0.0)
    assert ret_df["fwd_ret_1d"][0] == pytest.approx(0.0)
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```powershell
pixi run pytest tests\test_run_daily_single_config.py::test_build_forward_return_frame_prefers_adjusted_close -q
```

Expected before implementation: FAIL with import error for `_build_forward_return_frame`.

- [ ] **Step 3: Extract the forward-return helper**

In `src/factorzen/pipelines/daily_single.py`, add this helper near `_build_advanced_results`:

```python
def _build_forward_return_frame(daily: pl.DataFrame) -> pl.DataFrame:
    """Build IC forward-return labels, preferring adjusted close when available."""
    price_col = "close_adj" if "close_adj" in daily.columns else "close"
    ret_df = daily.select(["trade_date", "ts_code", price_col]).sort(["ts_code", "trade_date"])
    ret_df = ret_df.with_columns(
        (pl.col(price_col) / pl.col(price_col).shift(1).over("ts_code") - 1).alias("ret")
    )
    return compute_fwd_returns(ret_df, ret_col="ret", price_col=price_col)
```

- [ ] **Step 4: Use the helper in `_run`**

Replace:

```python
ret_df = daily.select(["trade_date", "ts_code", "close"]).sort(["ts_code", "trade_date"])
ret_df = ret_df.with_columns(
    (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1).alias("ret")
)
ret_df = compute_fwd_returns(ret_df, ret_col="ret")
```

with:

```python
ret_df = _build_forward_return_frame(daily)
```

- [ ] **Step 5: Run return-label and pipeline config tests**

Run:

```powershell
pixi run pytest tests\test_fwd_returns.py tests\test_run_daily_single_config.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src\factorzen\pipelines\daily_single.py tests\test_run_daily_single_config.py
git commit -m "fix: use adjusted close for forward return labels"
git log -1 --format='%an <%ae>'
```

Expected author:

```text
rookiewu417 <1007372080@qq.com>
```

---

### Task 6: Regression Sweep

**Files:**
- No new code unless a targeted failure reveals a direct regression from Tasks 2-5.

- [ ] **Step 1: Run core backtest/evaluation subset**

Run:

```powershell
pixi run pytest tests\test_lookahead_safety.py tests\test_backtest.py tests\test_backtest_costs.py tests\test_strategy_backtest.py tests\test_fwd_returns.py tests\test_run_daily_single_config.py tests\test_walk_forward_strategy.py tests\test_walk_forward_summary.py tests\test_benchmark.py tests\test_intraday_evaluation.py tests\test_intraday_returns.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full suite**

Run:

```powershell
pixi run pytest
```

Expected: PASS. If this exceeds local timeout, rerun failing or incomplete chunks by directory and record the last completed chunk.

- [ ] **Step 3: Run lint**

Run:

```powershell
pixi run lint
```

Expected: PASS.

- [ ] **Step 4: Inspect diff for unrelated churn**

Run:

```powershell
git diff --stat
git diff -- src\factorzen\daily\evaluation\backtest.py src\factorzen\daily\evaluation\walk_forward.py src\factorzen\pipelines\daily_single.py tests\test_backtest.py tests\test_strategy_backtest.py tests\test_run_daily_single_config.py tests\test_walk_forward_strategy.py
```

Expected: only planned files changed; no unrelated formatting-only edits.

- [ ] **Step 5: Final commit if Task 6 required any extra fixes**

Only if new fixes were required:

```powershell
git add <changed-files>
git commit -m "fix: stabilize backtest evaluation regressions"
git log -1 --format='%an <%ae>'
```

Expected author:

```text
rookiewu417 <1007372080@qq.com>
```

---

## Success Criteria

- Next-open execution no longer uses execution-day close-based `pct_chg` or execution-day full-session `amount`.
- Capacity constraints use trailing ADV or configured fallback, not same-day future liquidity.
- Daily gross return compounds overnight and intraday legs.
- Positions recorded at date close reflect intraday drift and are used as next day's starting holdings.
- NAV series includes an explicit base `1.0` row; returns remain period-return rows.
- Backtest and walk-forward max drawdown include initial NAV `1.0`.
- IC forward-return labels prefer `close_adj` when present.
- Targeted and full regression suites pass or any infrastructure timeout is documented with completed chunks.

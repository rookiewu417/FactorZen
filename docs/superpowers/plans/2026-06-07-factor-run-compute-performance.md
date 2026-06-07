# Factor Run Compute Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce pure compute time for daily factor backtests and walk-forward validation without changing reported portfolio returns or summary metrics.

**Architecture:** Keep the public factor pipeline unchanged and optimize the internal backtest/walk-forward execution path. The work adds behavior-preserving fast paths for summary-only validation, precomputes deterministic TopN targets, reuses IS backtests across expanding walk-forward folds, and parallelizes independent OOS folds when deterministic seeding is not requested.

**Tech Stack:** Python, Polars, NumPy, pytest, pixi.

---

### Task 1: Baseline And Guardrails

**Files:**
- Modify: `tests/test_strategy_backtest.py`
- Modify: `tests/test_walk_forward_summary.py`

- [x] Add tests proving lightweight backtests keep `returns`, `nav`, and `summary_stats` identical while omitting heavy `positions` and `trades`.
- [x] Add tests proving precomputed TopN weights produce the same backtest output as `TopNLongOnlyStrategy`.
- [x] Add tests proving optimized walk-forward summary matches the legacy sequential path for deterministic fixtures.
- [x] Run: `/home/ivenwu/.pixi/bin/pixi run pytest tests/test_strategy_backtest.py tests/test_walk_forward_summary.py -q`

### Task 2: Lightweight Backtest Output

**Files:**
- Modify: `src/factorzen/daily/evaluation/backtest.py`

- [x] Add keyword-only options to `run_strategy_backtest`: `collect_positions`, `collect_trades`, and `include_context_positions`.
- [x] Keep defaults as `True` so report and public callers preserve existing behavior.
- [x] Use empty schema DataFrames when collection is disabled.
- [x] Keep turnover, cost, returns, nav, and summary calculations unchanged.
- [x] Run the strategy backtest test file.

### Task 3: Precomputed TopN Targets

**Files:**
- Modify: `src/factorzen/daily/evaluation/backtest.py`

- [x] Add `precompute_top_n_weights(factor_df, top_n, factor_col)` using one Polars pass over all dates.
- [x] Add `PrecomputedWeightsStrategy` that returns the precomputed target weights for `context.signal_date`.
- [x] Keep `TopNLongOnlyStrategy` unchanged for public compatibility.
- [x] Use the precomputed strategy only where the caller explicitly opts in.
- [x] Run the strategy backtest test file.

### Task 4: Walk-Forward Reuse And Range Slicing

**Files:**
- Modify: `src/factorzen/daily/evaluation/walk_forward.py`
- Modify: `src/factorzen/daily/evaluation/walk_forward_summary.py`

- [x] Replace `is_in(set(...))` fold filters with continuous date range filters.
- [x] Add optional IS reuse: run each candidate once on the full window, then compute fold IS Sharpe from the candidate return prefix.
- [x] Keep generic `run_walk_forward_search` default compatible; enable IS reuse from `run_quantile_walk_forward_summary`.
- [x] Use summary-only backtests for walk-forward IS and OOS.
- [x] Run walk-forward summary tests.

### Task 5: Parallel OOS Folds

**Files:**
- Modify: `src/factorzen/daily/evaluation/walk_forward.py`

- [x] Add optional `parallel_workers`.
- [x] Use `ThreadPoolExecutor` for independent OOS folds when `seed is None` and workers > 1.
- [x] Preserve fold ordering by sorting completed results by `fold_id`.
- [x] Force sequential execution when `seed` is set to avoid global RNG races.
- [x] Run walk-forward summary tests.

### Task 6: TopN Summary Fast Path

**Files:**
- Modify: `src/factorzen/daily/evaluation/backtest.py`
- Modify: `src/factorzen/daily/evaluation/walk_forward.py`

- [x] Add a NumPy-backed summary-only fast path for `PrecomputedWeightsStrategy`.
- [x] Activate it only when positions/trades/context positions are disabled and the cost model is `None` or the built-in `CostModel`.
- [x] Fall back to the generic engine for unsupported strategies or cost models.
- [x] Prove equivalence against the generic engine with tests.
- [x] Run the strategy and walk-forward tests.

### Task 7: End-To-End Verification

**Files:**
- No direct code files.

- [x] Run targeted tests:
  `/home/ivenwu/.pixi/bin/pixi run pytest tests/test_strategy_backtest.py tests/test_backtest_costs.py tests/test_walk_forward_summary.py tests/test_run_daily_single_config.py -q`
- [x] Run the real factor command:
  `/home/ivenwu/.pixi/bin/pixi run fz factor run --config workspace/configs/daily/volume_return_corr_20d.yaml`
- [x] Compare stage timing against the previous baseline: total about 58s, main backtest about 8.5s, walk-forward about 35.6s, report about 3.9s.

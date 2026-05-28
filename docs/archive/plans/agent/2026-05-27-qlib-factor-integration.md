# Qlib Factor Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated qlib daily factor package that can expose qlib Alpha158 and Alpha360 features through FactorZen's existing factor registry.

**Architecture:** Add `pyqlib` as a project dependency, create `daily/factors/qlib/`, and wrap qlib handlers behind `DailyFactor` classes that return `trade_date`, `ts_code`, and `factor_value`. Tests patch qlib imports and data loading so they verify FactorZen integration without requiring a local qlib data bundle.

**Tech Stack:** Python, pixi, pyqlib, polars, pytest.

---

### Task 1: Lock the Public API With Tests

**Files:**
- Modify: `tests/test_daily_factors.py`

- [ ] Add tests that assert `qlib_alpha158` and `qlib_alpha360` are discovered by `daily.factors.registry.list_factors()`.
- [ ] Add tests that patch qlib handler output and assert each wrapper returns exactly `trade_date`, `ts_code`, and `factor_value`.
- [ ] Run `pixi run pytest tests/test_daily_factors.py -q`; expected result before implementation is failure because the qlib package is not registered yet.

### Task 2: Add qlib Dependency and Package

**Files:**
- Modify: `pyproject.toml`
- Modify: `pixi.toml`
- Create: `daily/factors/qlib/__init__.py`
- Create: `daily/factors/qlib/handler.py`
- Create: `daily/factors/qlib/README.md`
- Modify: `daily/factors/registry.py`

- [ ] Add `pyqlib` to both dependency declarations.
- [ ] Add a focused helper that initializes qlib only when compute runs and converts qlib's pandas output to FactorZen's polars schema.
- [ ] Add two factor classes, `QlibAlpha158` and `QlibAlpha360`, with names `qlib_alpha158` and `qlib_alpha360`.
- [ ] Add `daily.factors.qlib` to the registry scan list.

### Task 3: Verify

**Files:**
- No new files.

- [ ] Run `pixi run pytest tests/test_daily_factors.py -q`; expected result is pass.
- [ ] Run a registry smoke command with `pixi run python -c "from daily.factors.registry import list_factors; print([f for f in list_factors() if f.startswith('qlib_')])"`; expected output includes `qlib_alpha158` and `qlib_alpha360`.
- [ ] If dependency resolution changes `pixi.lock` or `uv.lock`, leave those changes in place and report them.

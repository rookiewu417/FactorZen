# FactorZen Structure Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize FactorZen into a clear `src/factorzen` framework package plus a `workspace` research area with per-run outputs.

**Architecture:** Framework code moves under `src/factorzen`; user-authored factors and experiment configs live under `workspace`; generated artifacts are grouped by run under `workspace/factor_evaluations`. Existing data and output artifacts are preserved, while legacy scripts move under `tools/legacy`.

**Tech Stack:** Python 3.10+, pixi, hatchling, polars, pytest, ruff, mypy.

---

### Task 1: Prepare Skeleton

**Files:**
- Create: `src/factorzen/__init__.py`
- Create: `workspace/factors/{daily,weekly,monthly,intraday}/__init__.py`
- Create: `workspace/configs/{daily,intraday}/.gitkeep`
- Create: `workspace/factor_evaluations/.gitkeep`
- Create: `workspace/notebooks/.gitkeep`
- Create: `tools/legacy/.gitkeep`

- [ ] Create the directories without touching `data/` or existing `output/`.
- [ ] Add minimal `__init__.py` files where Python imports require packages.

### Task 2: Move Framework Packages

**Files:**
- Move the old shared package into `src/factorzen/core/`, except run path helpers become `src/factorzen/experiments/`
- Move: `config/` -> `src/factorzen/config/`
- Move: `daily/` -> `src/factorzen/daily/` initially, then keep public imports stable through mechanical import updates
- Move: `intraday/` -> `src/factorzen/intraday/`
- Move: `reporting/` -> `src/factorzen/reports/`
- Move: `automation/` -> `src/factorzen/automation/`
- Move: `research/` -> `src/factorzen/research/`
- Move: `llm_explain/` -> `src/factorzen/llm/`
- Move: `scripts/` -> `tools/legacy/scripts/`

- [ ] Use non-destructive moves only.
- [ ] Preserve all files and directory contents.

### Task 3: Update Imports

**Files:**
- Modify: `src/factorzen/**/*.py`
- Modify: `tests/**/*.py`
- Modify: `benchmarks/*.py`
- Modify: `tools/legacy/*.py`

- [ ] Replace old imports with `factorzen.*` imports.
- [ ] Keep tests pointing at the migrated modules.
- [ ] Avoid compatibility stubs unless needed by CLI ergonomics.

### Task 4: Add Unified CLI and Run Paths

**Files:**
- Create: `src/factorzen/cli/main.py`
- Create: `src/factorzen/experiments/run_paths.py`
- Modify: daily single-factor pipeline to write artifacts into `workspace/factor_evaluations/{run_id}/`.

- [ ] Implement `fz factor list`.
- [ ] Implement `fz factor test ...` by delegating to the migrated daily pipeline.
- [ ] Implement `fz report open RUN_ID` as a report path locator.
- [ ] Keep legacy scripts callable from `tools/legacy`.

### Task 5: Packaging and Docs

**Files:**
- Modify: `pyproject.toml`
- Modify: `pixi.toml`
- Modify: `README.md`
- Create: `docs/architecture.md`
- Create: `docs/factor-authoring.md`
- Create: `docs/runbook.md`

- [ ] Switch hatch build to source layout.
- [ ] Add `fz` console script.
- [ ] Update pixi tasks to use `python -m factorzen.cli.main` or `fz`.
- [ ] Document new structure and day-to-day workflow.

### Task 6: Verification

- [ ] Run targeted tests for registry, config loader, output paths, and script wrapping.
- [ ] Run `pixi run test` if time permits.
- [ ] Run `pixi run lint` or at least import smoke tests.


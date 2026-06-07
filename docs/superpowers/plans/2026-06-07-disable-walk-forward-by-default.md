# Disable Walk-Forward By Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Disable strategy walk-forward evaluation by default while preserving explicit opt-in through YAML or `--set`.

**Architecture:** Add an `enabled` flag to `WalkForwardConfig` with a default of `false`. Both report pipelines check the validated config before invoking the existing walk-forward implementation and emit a serializable disabled summary when skipped.

**Tech Stack:** Python, Pydantic v2, pytest, YAML, Ruff

---

### Task 1: Configuration Default

**Files:**
- Modify: `src/factorzen/core/config_loader.py`
- Test: `tests/test_config_loader.py`

- [ ] Add tests asserting `WalkForwardConfig.enabled` defaults to `False` and accepts `True`.
- [ ] Run `pixi run pytest tests/test_config_loader.py -q` and verify the new tests fail.
- [ ] Add `enabled: bool = False` and keep the built-in research preset disabled.
- [ ] Re-run the tests and verify they pass.

### Task 2: Pipeline Gating

**Files:**
- Modify: `src/factorzen/pipelines/daily_single.py`
- Modify: `src/factorzen/pipelines/generate_report.py`
- Test: `tests/test_run_daily_single_config.py`
- Test: `tests/test_generate_report_persistence.py`

- [ ] Add tests that disabled walk-forward returns `{"status": "disabled", "n_folds": 0}` without calling the runner.
- [ ] Add tests that `enabled: true` still calls the runner.
- [ ] Run the targeted tests and verify they fail before implementation.
- [ ] Gate each pipeline call on `effective_config.walk_forward.enabled`.
- [ ] Re-run the targeted tests and verify they pass.

### Task 3: YAML Defaults

**Files:**
- Modify: `workspace/configs/daily/daily_factor_template.yaml`
- Modify: `workspace/configs/daily/volume_return_corr_20d.yaml`

- [ ] Add `enabled: false` to both walk-forward sections.
- [ ] Verify `pixi run fz factor run volume_return_corr_20d --dry-run` reports `walk_forward.enabled=false`.
- [ ] Verify `--set walk_forward.enabled=true` reports `walk_forward.enabled=true`.

### Task 4: Verification

**Files:**
- Verify all modified source and test files.

- [ ] Run Ruff format and check.
- [ ] Run targeted configuration and pipeline tests.
- [ ] Run a short factor evaluation and confirm no Walk-forward timer or computation appears by default.

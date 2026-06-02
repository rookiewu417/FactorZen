# Clean Legacy Directories Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all legacy directories from the active project: `tools/legacy/`, `output/`, and `docs/archive/`.

**Architecture:** First move every still-needed behavior behind supported package modules or `workspace/factor_evaluations/`, then update tests, tasks, and docs to stop referencing legacy paths. Only after references are gone, delete the legacy directories and verify with repository search plus the test suite.

**Tech Stack:** Python package under `src/factorzen`, Pixi task runner, pytest, ruff, coverage.

---

## Scope And Success Criteria

Legacy directories to remove:

- `tools/legacy/`
- `output/`
- `docs/archive/`

Success criteria:

- `git ls-files tools/legacy docs/archive` returns no files.
- `Test-Path output` is false after local cleanup, or the directory is absent except for user-restored local artifacts.
- `rg -n "tools/legacy|tools\\\\legacy|docs/archive|docs\\\\archive|output/|output\\\\" -S .` returns no active references outside the cleanup plan itself and release notes that intentionally document history.
- `pixi run pytest tests/ -v` passes.
- `pixi run ruff check .` passes.
- `pixi run coverage` or its replacement passes without calling `tools/legacy/scripts/run_coverage.py`.

## Files To Modify

- Modify: `pixi.toml`
  - Replace `coverage = "python tools/legacy/scripts/run_coverage.py"` with a non-legacy command.
  - Replace `smoke-data = "python tools/legacy/scripts/smoke_data.py"` with a package CLI or remove it if no supported equivalent exists.
- Modify: `src/factorzen/config/settings.py`
  - Remove `SCRIPTS_DIR = ROOT / "tools" / "legacy" / "scripts"` or replace it with a non-legacy path only if still used.
- Modify: `src/factorzen/pipelines/generate_report.py`
  - Stop writing compatibility copies to `output/daily/**`.
  - Keep canonical artifacts under `workspace/factor_evaluations/{run_id}/`.
- Modify: `src/factorzen/automation/state.py`
  - Move default automation state from `output/automation/runs.jsonl` to `workspace/factor_evaluations/automation/runs.jsonl` or another active workspace path.
- Modify: `tests/test_combination.py`
  - Stop importing helpers from `tools.legacy.scripts.run_combination`.
  - Move the helpers into supported package code, or test the public supported API directly.
- Modify: `tests/test_run_qlib_batch_reports.py`
  - Remove or rewrite the test so it targets a supported package module rather than `tools.legacy.scripts.run_qlib_batch_reports`.
- Modify: `README.md`
  - Remove instructions that say `output/` is retained.
  - Remove examples that call `tools/legacy/scripts/*.py`.
  - State that `workspace/factor_evaluations/` is the only supported run output location.
- Modify: `docs/architecture.md`, `docs/project-explanation.md`, `docs/runbook.md`, `docs/README.md`, `docs/evolution-plan-2026.md`
  - Remove legacy directory references.
  - Replace old script examples with `pixi run fz ...`, `pixi run daily --config ...`, or `pixi run report ...`.
- Modify: `src/factorzen/research/combination/README.md`
  - Replace `tools/legacy/scripts/run_combination.py` example with a supported CLI or package-module command.
- Delete: `tools/legacy/**`
- Delete: `docs/archive/**`
- Delete local ignored directory: `output/**`

## Task 1: Baseline Inventory

**Files:**
- Read: repository search results

- [ ] **Step 1: Capture current legacy references**

Run:

```powershell
rg -n "tools/legacy|tools\\legacy|docs/archive|docs\\archive|output/|output\\|legacy|archive" -S .
```

Expected: references appear in README/docs, `pixi.toml`, `pyproject.toml`, `src/factorzen/config/settings.py`, `src/factorzen/pipelines/generate_report.py`, `src/factorzen/automation/state.py`, and tests importing `tools.legacy`.

- [ ] **Step 2: Capture tracked files under legacy directories**

Run:

```powershell
git ls-files tools/legacy docs/archive
```

Expected: tracked files are listed. These are candidates for deletion after replacements are complete.

- [ ] **Step 3: Capture ignored local `output/` contents**

Run:

```powershell
Get-ChildItem -Force -Recurse -LiteralPath output | Select-Object FullName, Length, LastWriteTime
```

Expected: either output artifacts are listed, or PowerShell reports that `output` does not exist. Do not delete yet.

## Task 2: Replace Legacy Pixi Tasks

**Files:**
- Modify: `pixi.toml`

- [ ] **Step 1: Replace `coverage` task**

Change:

```toml
coverage   = "python tools/legacy/scripts/run_coverage.py"
```

to:

```toml
coverage   = "coverage run -m pytest tests/ && coverage report"
```

- [ ] **Step 2: Replace or remove `smoke-data` task**

If `tools/legacy/scripts/smoke_data.py` only checks Tushare connectivity and local data availability, replace it with a supported CLI before deleting the legacy script. If no supported equivalent exists, remove this task from `pixi.toml` and document the supported data checks in `docs/runbook.md`.

Preferred final form:

```toml
smoke-data = "python -m factorzen.cli.main data smoke"
```

Only use this form after adding and testing the `data smoke` subcommand. If not adding that command in this cleanup, delete the `smoke-data` task instead.

- [ ] **Step 3: Verify pixi task references**

Run:

```powershell
rg -n "tools/legacy|tools\\legacy" pixi.toml
```

Expected: no matches.

## Task 3: Remove Legacy Script Path Constant

**Files:**
- Modify: `src/factorzen/config/settings.py`
- Test: existing test suite

- [ ] **Step 1: Check whether `SCRIPTS_DIR` is used**

Run:

```powershell
rg -n "SCRIPTS_DIR" src tests
```

Expected: declaration appears in `src/factorzen/config/settings.py`; no active callers should remain.

- [ ] **Step 2: Remove the constant**

Delete this line from `src/factorzen/config/settings.py`:

```python
SCRIPTS_DIR = ROOT / "tools" / "legacy" / "scripts"
```

- [ ] **Step 3: Verify no reference remains**

Run:

```powershell
rg -n "SCRIPTS_DIR|tools/legacy|tools\\legacy" src/factorzen/config/settings.py src tests
```

Expected: no `SCRIPTS_DIR` or `tools/legacy` references from active code.

## Task 4: Stop Writing Compatibility Copies To `output/`

**Files:**
- Modify: `src/factorzen/pipelines/generate_report.py`
- Test: `tests/test_generate_report_persistence.py`, `tests/test_output_paths.py`, `tests/test_workspace_layout.py`

- [ ] **Step 1: Inspect current output writes**

Run:

```powershell
rg -n "output|daily/results|daily/reports|workspace/factor_evaluations|legacy" src/factorzen/pipelines/generate_report.py tests/test_generate_report_persistence.py tests/test_output_paths.py tests/test_workspace_layout.py
```

Expected: `generate_report.py` writes canonical artifacts to `workspace/factor_evaluations` and compatibility artifacts to `output/daily`.

- [ ] **Step 2: Update tests to require only `workspace/factor_evaluations` artifacts**

Change expectations so report generation asserts these files under `workspace/factor_evaluations/{run_id}/`:

```text
report.html
manifest.json
factor.parquet
ic.parquet
quality.json
walk_forward.json
```

Remove assertions that expect files under:

```text
output/daily/**
```

- [ ] **Step 3: Remove compatibility write code**

In `src/factorzen/pipelines/generate_report.py`, remove the function or block whose docstring says:

```python
"""灏嗗洜瀛?DataFrame 鍜岃瘎浠风粨鏋滆惤鐩樺埌 output/daily/銆?""
```

Also remove the log message:

```python
logger.info(f"涓棿缁撴灉宸茶惤鐩? output/daily/results/{prefix}_*.parquet")
```

Keep the canonical `workspace/factor_evaluations/{run_id}` persistence path intact.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
pixi run pytest tests/test_generate_report_persistence.py tests/test_output_paths.py tests/test_workspace_layout.py -v
```

Expected: all selected tests pass.

## Task 5: Move Automation State Out Of `output/`

**Files:**
- Modify: `src/factorzen/automation/state.py`
- Test: `tests/test_automation_state.py`

- [ ] **Step 1: Change the default state path**

Change:

```python
STATE_FILE = Path("output/automation/runs.jsonl")
```

to:

```python
STATE_FILE = Path("workspace/factor_evaluations/automation/runs.jsonl")
```

- [ ] **Step 2: Run focused test**

Run:

```powershell
pixi run pytest tests/test_automation_state.py -v
```

Expected: all tests pass. If a test hard-codes `output/automation`, update it to assert `workspace/factor_evaluations/automation`.

## Task 6: Replace Tests That Import Legacy Scripts

**Files:**
- Modify: `tests/test_combination.py`
- Modify: `tests/test_run_qlib_batch_reports.py`
- Modify or create supported package modules if needed under `src/factorzen/research/combination/` or `src/factorzen/cli/`

- [ ] **Step 1: Inspect imported legacy helpers**

Run:

```powershell
Get-Content -LiteralPath tests/test_combination.py
Get-Content -LiteralPath tests/test_run_qlib_batch_reports.py
Get-Content -LiteralPath tools/legacy/scripts/run_combination.py
Get-Content -LiteralPath tools/legacy/scripts/run_qlib_batch_reports.py
```

Expected: tests import helper functions from legacy scripts.

- [ ] **Step 2: Move still-needed helper behavior into package code**

For combination helpers, prefer a supported module under:

```text
src/factorzen/research/combination/pipeline.py
```

or an existing package function if it already exists. Tests should import from `factorzen.research.combination`, not from `tools.legacy`.

For Qlib batch report argument parsing, either:

- move `_namespace` into a supported CLI module if the behavior is still supported, or
- delete the test if Qlib batch report scripting is intentionally removed with the legacy directory.

- [ ] **Step 3: Verify no tests import `tools.legacy`**

Run:

```powershell
rg -n "tools\\.legacy|tools/legacy|tools\\legacy" tests src
```

Expected: no matches.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
pixi run pytest tests/test_combination.py tests/test_run_qlib_batch_reports.py -v
```

Expected: all remaining selected tests pass. If `tests/test_run_qlib_batch_reports.py` is deleted because the feature is removed, this command should omit the deleted file and the deletion should be visible in `git status`.

## Task 7: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/project-explanation.md`
- Modify: `docs/runbook.md`
- Modify: `docs/README.md`
- Modify: `docs/evolution-plan-2026.md`
- Modify: `src/factorzen/research/combination/README.md`

- [ ] **Step 1: Replace old output language**

Replace claims that `output/` is retained for compatibility with:

```text
`workspace/factor_evaluations/{run_id}/` is the supported location for run artifacts.
```

- [ ] **Step 2: Replace legacy script examples**

Replace examples like:

```powershell
pixi run python tools/legacy/scripts/run_daily_compare.py --factors momentum_20d,reversal_5d --start 20250101 --end 20260513
```

with supported commands. Use the closest existing command:

```powershell
pixi run fz factor test momentum_20d --start 20250101 --end 20260513
pixi run daily --config workspace/configs/daily/daily_factor_template.yaml
pixi run report --factor momentum_20d --start 20250101 --end 20260513
```

If no supported replacement exists for an old script, remove the old example rather than documenting deleted behavior.

- [ ] **Step 3: Remove docs archive index**

Delete documentation that only points to `docs/archive/**`, including `docs/README.md` archive bullet list.

- [ ] **Step 4: Verify docs references**

Run:

```powershell
rg -n "tools/legacy|tools\\legacy|docs/archive|docs\\archive|output/|output\\" README.md docs src/factorzen/research/combination/README.md
```

Expected: no active references, except release notes or this plan if intentionally retained.

## Task 8: Delete Legacy Directories

**Files:**
- Delete: `tools/legacy/**`
- Delete: `docs/archive/**`
- Delete local ignored directory: `output/**`

- [ ] **Step 1: Delete tracked legacy directories**

Run:

```powershell
Remove-Item -LiteralPath tools/legacy -Recurse -Force
Remove-Item -LiteralPath docs/archive -Recurse -Force
```

Expected: directories are removed from the working tree.

- [ ] **Step 2: Delete local ignored output directory**

Before deletion, verify the target path is inside the repository:

```powershell
(Resolve-Path .).Path
(Resolve-Path output).Path
```

Expected: `output` resolves under the repository root.

Then run:

```powershell
Remove-Item -LiteralPath output -Recurse -Force
```

Expected: `Test-Path output` returns `False`.

- [ ] **Step 3: Verify deleted tracked files**

Run:

```powershell
git status --short
git ls-files tools/legacy docs/archive
```

Expected: `git status --short` shows deletions under `tools/legacy` and `docs/archive`; `git ls-files tools/legacy docs/archive` still lists tracked paths until staging, then lists no paths after staging.

## Task 9: Update Tooling Exclusions

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`

- [ ] **Step 1: Remove obsolete coverage omissions**

In `pyproject.toml`, remove these entries from `[tool.coverage.run].omit`:

```toml
"tools/legacy/*",
"docs/archive/*",
```

Keep:

```toml
"tests/*",
```

- [ ] **Step 2: Keep `output/` ignored only if needed**

If the project should reject all future `output/` usage, remove `output/` from `.gitignore` and rely on tests/docs to keep artifacts in `workspace/factor_evaluations/`.

If backward-compatible local junk might still be produced by old user commands outside the repo, keep `output/` ignored but do not recreate it.

Preferred cleanup stance for "娓呯悊鎵€鏈夋棫鐩綍": remove `output/` from `.gitignore` so accidental recreation is visible in `git status`.

- [ ] **Step 3: Verify config references**

Run:

```powershell
rg -n "tools/legacy|tools\\legacy|docs/archive|docs\\archive|output/|output\\" pyproject.toml .gitignore
```

Expected: no matches if `.gitignore` also removes `output/`.

## Task 10: Full Verification

**Files:**
- Read: entire repository

- [ ] **Step 1: Search for old directory references**

Run:

```powershell
rg -n "tools/legacy|tools\\legacy|docs/archive|docs\\archive|output/|output\\" -S .
```

Expected: no active references outside this plan and any deliberately retained release notes. If release notes are considered active docs, remove those references too.

- [ ] **Step 2: Run lint**

Run:

```powershell
pixi run ruff check .
```

Expected: pass.

- [ ] **Step 3: Run tests**

Run:

```powershell
pixi run pytest tests/ -v
```

Expected: pass.

- [ ] **Step 4: Run coverage command**

Run:

```powershell
pixi run coverage
```

Expected: pass, and the command does not reference `tools/legacy`.

- [ ] **Step 5: Confirm no output directory was recreated**

Run:

```powershell
Test-Path output
```

Expected:

```text
False
```

## Task 11: Stage And Commit

**Files:**
- Stage all cleanup changes

- [ ] **Step 1: Review changed files**

Run:

```powershell
git status --short
git diff --stat
```

Expected: changes are limited to replacing legacy references, deleting legacy directories, and updating tests/docs/tooling.

- [ ] **Step 2: Stage changes**

Run:

```powershell
git add -A
git status --short
```

Expected: deleted legacy files and modified active files are staged.

- [ ] **Step 3: Commit with required identity**

Run:

```powershell
git -c user.name="rookiewu417" -c user.email="1007372080@qq.com" commit -m "chore: remove legacy directories"
```

Expected: commit succeeds.

- [ ] **Step 4: Verify commit author**

Run:

```powershell
git log -1 --format='%an <%ae>'
```

Expected:

```text
rookiewu417 <1007372080@qq.com>
```

If the output differs, amend immediately:

```powershell
git -c user.name="rookiewu417" -c user.email="1007372080@qq.com" commit --amend --no-edit --reset-author
git log -1 --format='%an <%ae>'
```

Expected:

```text
rookiewu417 <1007372080@qq.com>
```

## Self-Review

- Spec coverage: the plan removes all three old directories identified in the request: `tools/legacy/`, `output/`, and `docs/archive/`.
- Reference coverage: known active references from `pixi.toml`, `pyproject.toml`, docs, tests, `settings.py`, `generate_report.py`, and `automation/state.py` are assigned to tasks.
- Risk: deleting ignored `output/` removes local run artifacts. The plan includes an inventory step before deletion so the executor can confirm what will be removed.


# Report Semantics UI Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade FactorZen daily factor reports so strategy semantics, multi-strategy comparisons, missing-data states, and visual hierarchy are clear enough for serious factor research review.

**Architecture:** Keep the current Jinja-based static HTML report pipeline. Add a thin report semantic layer inside `src/factorzen/reports/tear_sheet.py` that prepares display-ready strategy pages, status notices, and dashboard values before rendering. Update `src/factorzen/reports/templates/tear_sheet.html` to consume those semantics without changing the pipeline call sites.

**Tech Stack:** Python 3.12, Polars, Jinja2, pytest, ruff, static HTML/CSS/vanilla JavaScript.

---

### Task 1: Strategy Return Semantics

**Files:**
- Modify: `src/factorzen/reports/tear_sheet.py`
- Modify: `src/factorzen/reports/templates/tear_sheet.html`
- Test: `tests/test_reporting.py`

- [ ] **Step 1: Write failing tests for long-only vs long-short labels**

Add tests to `tests/test_reporting.py`:

```python
def test_long_only_strategy_does_not_show_long_short_label(self, ic_result, bt_result, to_result):
    long_only = replace(
        bt_result,
        strategy_name="topn_long_only",
        summary_stats={
            "portfolio": {
                "ann_ret": 0.08,
                "ann_vol": 0.16,
                "sharpe": 0.50,
                "max_dd": -0.08,
                "avg_turnover": 0.25,
                "total_cost": 0.01,
                "ann_turnover": 63.0,
            }
        },
        config={
            "strategy_type": "topn_long_only",
            "cost_model": "linear",
            "max_abs_weight": 0.1,
        },
    )
    html = generate_tear_sheet(
        "test_factor",
        ic_result,
        long_only,
        to_result,
        strategy_results={"topn_50": long_only},
        primary_strategy="topn_50",
    )

    assert "组合收益" in html
    assert "多空组合" not in html
    assert "<td>L/S</td>" not in html
    assert "多头 TopN" in html


def test_long_short_strategy_shows_long_short_label(self, ic_result, bt_result, to_result):
    long_short = replace(
        bt_result,
        strategy_name="quantile_long_short",
        config={
            "strategy_type": "quantile_long_short",
            "cost_model": "linear",
            "max_abs_weight": 0.1,
        },
    )
    html = generate_tear_sheet(
        "test_factor",
        ic_result,
        long_short,
        to_result,
        strategy_results={"quantile_ls_5": long_short},
        primary_strategy="quantile_ls_5",
    )

    assert "多空组合" in html
    assert "分位数组合多空" in html
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pixi run pytest tests/test_reporting.py::TestGenerateTearSheet::test_long_only_strategy_does_not_show_long_short_label tests/test_reporting.py::TestGenerateTearSheet::test_long_short_strategy_shows_long_short_label -q
```

Expected: FAIL because strategy semantic labels are not prepared yet.

- [ ] **Step 3: Add strategy semantic helpers**

In `src/factorzen/reports/tear_sheet.py`, add helpers near `_strategy_portfolio_stats`:

```python
def _infer_strategy_type(name: str, bt_result: Any) -> str:
    config = _safe_attr(bt_result, "config", {}) or {}
    configured = config.get("strategy_type") if isinstance(config, dict) else None
    raw = configured or _safe_attr(bt_result, "strategy_name", name) or name
    return str(raw)


def _is_long_short_strategy(strategy_type: str, stats: dict[str, Any]) -> bool:
    lowered = strategy_type.lower()
    if "long_short" in lowered or lowered.endswith("_ls") or "factor_weighted" in lowered:
        return True
    group_keys = [key for key in stats if isinstance(key, int)]
    return bool(group_keys and "long_short" in stats)


def _strategy_exposure_label(strategy_type: str, is_long_short: bool) -> str:
    lowered = strategy_type.lower()
    if "topn" in lowered or "long_only" in lowered:
        return "多头 TopN"
    if "quantile" in lowered and is_long_short:
        return "分位数组合多空"
    if "factor_weighted" in lowered and is_long_short:
        return "因子加权多空"
    if "optimizer" in lowered:
        return "优化组合"
    return "多空" if is_long_short else "组合"


def _strategy_constraints(bt_result: Any) -> dict[str, Any]:
    config = _safe_attr(bt_result, "config", {}) or {}
    if not isinstance(config, dict):
        return {}
    keys = ("cost_model", "max_abs_weight", "rebalance_threshold", "alpha", "fallback_adv")
    return {key: config.get(key) for key in keys if config.get(key) is not None}
```

Update `_build_strategy_pages` so every page includes:

```python
stats = _safe_attr(result, "summary_stats", {}) or {}
strategy_type = _infer_strategy_type(name, result)
is_long_short = _is_long_short_strategy(strategy_type, stats)
portfolio_stats = stats.get("portfolio") or (stats.get("long_short") if is_long_short else {}) or {}
long_short_stats = stats.get("long_short") if is_long_short else None
```

Add fields to the page dict:

```python
"strategy_type": strategy_type,
"exposure_label": _strategy_exposure_label(strategy_type, is_long_short),
"is_long_short": is_long_short,
"return_label": "多空组合" if is_long_short else "组合收益",
"portfolio_stats": portfolio_stats,
"long_short_stats": long_short_stats,
"constraints": _strategy_constraints(result),
```

- [ ] **Step 4: Render semantic labels in the strategy template**

In `src/factorzen/reports/templates/tear_sheet.html`, replace the strategy section heading with:

```html
<div class="strategy-header">
  <div>
    <h3>{{ strategy.name }}</h3>
    <p class="note">{{ strategy.exposure_label }} | {{ strategy.return_label }}</p>
  </div>
  <div class="strategy-badges">
    {% if strategy.is_primary %}<span class="badge badge-primary">主策略</span>{% endif %}
    <span class="badge">{{ strategy.exposure_label }}</span>
  </div>
</div>
```

Change the backtest stats table header from `分组` to `收益口径`, and change `_build_bt_summary_table` so `long_short` rows use `多空组合` instead of `L/S`.

- [ ] **Step 5: Run tests to verify pass**

Run:

```powershell
pixi run pytest tests/test_reporting.py::TestGenerateTearSheet::test_long_only_strategy_does_not_show_long_short_label tests/test_reporting.py::TestGenerateTearSheet::test_long_short_strategy_shows_long_short_label -q
```

Expected: PASS.

---

### Task 2: Cross-Strategy Comparison Completeness

**Files:**
- Modify: `src/factorzen/reports/templates/tear_sheet.html`
- Modify: `src/factorzen/reports/tear_sheet.py`
- Test: `tests/test_reporting.py`

- [ ] **Step 1: Write failing test for comparison columns**

Add test:

```python
def test_cross_strategy_table_explains_direction_cost_and_turnover(self, ic_result, bt_result, to_result):
    topn = replace(
        bt_result,
        strategy_name="topn_long_only",
        summary_stats={
            "portfolio": {
                "ann_ret": 0.08,
                "ann_vol": 0.16,
                "sharpe": 0.50,
                "max_dd": -0.08,
                "avg_turnover": 0.25,
                "total_cost": 0.01,
                "ann_turnover": 63.0,
            }
        },
        config={"strategy_type": "topn_long_only", "cost_model": "linear"},
    )
    quantile = replace(
        bt_result,
        strategy_name="quantile_long_short",
        config={"strategy_type": "quantile_long_short", "cost_model": "linear"},
    )

    html = generate_tear_sheet(
        "test_factor",
        ic_result,
        topn,
        to_result,
        strategy_results={"topn_50": topn, "quantile_ls_5": quantile},
        primary_strategy="topn_50",
    )

    for heading in ("策略方向", "收益口径", "平均换手", "交易成本", "成本模型"):
        assert heading in html
    assert "多头 TopN" in html
    assert "分位数组合多空" in html
    assert "linear" in html
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
pixi run pytest tests/test_reporting.py::TestGenerateTearSheet::test_cross_strategy_table_explains_direction_cost_and_turnover -q
```

Expected: FAIL because the current comparison table lacks these columns.

- [ ] **Step 3: Add formatted strategy metrics**

Add helper:

```python
def _strategy_stat(stats: dict[str, Any], key: str, default: float = 0.0) -> float:
    return _num(stats.get(key), default)
```

Add page fields:

```python
"avg_turnover": _strategy_stat(portfolio_stats, "avg_turnover"),
"total_cost": _strategy_stat(portfolio_stats, "total_cost"),
"cost_model": (_strategy_constraints(result).get("cost_model") or "未指定"),
```

- [ ] **Step 4: Expand cross-strategy table**

Change table header to:

```html
<tr>
  <th>策略</th><th>策略方向</th><th>收益口径</th><th>年化收益</th><th>年化波动</th>
  <th>Sharpe</th><th>最大回撤</th><th>平均换手</th><th>交易成本</th><th>成本模型</th>
</tr>
```

Add row cells for `exposure_label`, `return_label`, `avg_turnover`, `total_cost`, and `cost_model`.

- [ ] **Step 5: Run test to verify pass**

Run:

```powershell
pixi run pytest tests/test_reporting.py::TestGenerateTearSheet::test_cross_strategy_table_explains_direction_cost_and_turnover -q
```

Expected: PASS.

---

### Task 3: Dashboard and Status Callouts

**Files:**
- Modify: `src/factorzen/reports/templates/tear_sheet.html`
- Modify: `src/factorzen/reports/tear_sheet.py`
- Test: `tests/test_reporting.py`

- [ ] **Step 1: Write failing test for dashboard/status semantics**

Add test:

```python
def test_overview_dashboard_contains_quality_and_primary_strategy_context(self, ic_result, bt_result, to_result):
    html = generate_tear_sheet(
        "test_factor",
        ic_result,
        bt_result,
        to_result,
        date_range="20250101 ~ 20250331",
        strategy_results={"quantile_ls_5": bt_result},
        primary_strategy="quantile_ls_5",
    )

    assert "研究仪表盘" in html
    assert "主策略" in html
    assert "有效区间" in html
    assert "样本期数" in html
    assert "数据质量" in html
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
pixi run pytest tests/test_reporting.py::TestGenerateTearSheet::test_overview_dashboard_contains_quality_and_primary_strategy_context -q
```

Expected: FAIL because the dashboard section does not exist.

- [ ] **Step 3: Prepare dashboard fields**

In `generate_tear_sheet`, compute:

```python
primary_page = next((page for page in strategy_pages if page["is_primary"]), strategy_pages[0] if strategy_pages else None)
dashboard = {
    "primary_strategy": primary_page["name"] if primary_page else metrics.get("bt_strategy_name", "未运行"),
    "primary_exposure": primary_page["exposure_label"] if primary_page else "未运行",
    "effective_range": (
        f"{metrics.get('effective_start')} ~ {metrics.get('effective_end')}"
        if metrics.get("effective_start") and metrics.get("effective_end")
        else "未计算"
    ),
    "sample_periods": metrics.get("n_periods", 0),
    "data_quality": "需关注" if warnings else "正常",
}
```

Pass `dashboard=dashboard` into the template.

- [ ] **Step 4: Render dashboard cards**

In the overview panel before `综合评估`, add:

```html
<h2>研究仪表盘</h2>
<div class="dashboard-grid">
  <div class="dashboard-card"><div class="label">主策略</div><div class="value">{{ dashboard.primary_strategy }}</div><div class="subvalue">{{ dashboard.primary_exposure }}</div></div>
  <div class="dashboard-card"><div class="label">有效区间</div><div class="value value-small">{{ dashboard.effective_range }}</div></div>
  <div class="dashboard-card"><div class="label">样本期数</div><div class="value">{{ dashboard.sample_periods }}</div></div>
  <div class="dashboard-card"><div class="label">数据质量</div><div class="value">{{ dashboard.data_quality }}</div></div>
</div>
```

- [ ] **Step 5: Run test to verify pass**

Run:

```powershell
pixi run pytest tests/test_reporting.py::TestGenerateTearSheet::test_overview_dashboard_contains_quality_and_primary_strategy_context -q
```

Expected: PASS.

---

### Task 4: Visual Polish for Professional Report Layout

**Files:**
- Modify: `src/factorzen/reports/templates/tear_sheet.html`
- Test: `tests/test_reporting.py`

- [ ] **Step 1: Write failing test for new CSS hooks**

Add test:

```python
def test_report_contains_professional_layout_css_hooks(self, ic_result, bt_result, to_result):
    html = generate_tear_sheet("test_factor", ic_result, bt_result, to_result)

    for css_hook in (
        ".dashboard-grid",
        ".dashboard-card",
        ".badge",
        ".status-callout",
        ".strategy-header",
        ".table-wrap",
    ):
        assert css_hook in html
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
pixi run pytest tests/test_reporting.py::TestGenerateTearSheet::test_report_contains_professional_layout_css_hooks -q
```

Expected: FAIL because those CSS hooks are missing.

- [ ] **Step 3: Add CSS hooks**

Add CSS to `tear_sheet.html`:

```css
.dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 12px 0 16px; }
.dashboard-card { background: #f8fafc; border: 1px solid #dbe4ee; border-radius: 8px; padding: 12px 14px; }
.dashboard-card .label { font-size: 12px; color: #64748b; }
.dashboard-card .value { font-size: 19px; font-weight: 700; color: #1f2d3d; margin-top: 3px; }
.dashboard-card .value-small { font-size: 14px; line-height: 1.4; }
.dashboard-card .subvalue { color: #64748b; font-size: 12px; margin-top: 2px; }
.badge { display: inline-flex; align-items: center; border: 1px solid #cbd5e1; border-radius: 999px; padding: 2px 8px; font-size: 12px; color: #334155; background: #f8fafc; }
.badge-primary { color: #075985; background: #e0f2fe; border-color: #7dd3fc; }
.status-callout { border: 1px solid #dbe4ee; border-left: 4px solid #64748b; background: #f8fafc; border-radius: 6px; padding: 10px 12px; margin: 10px 0; color: #475569; }
.strategy-header { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; margin: 14px 0 10px; }
.strategy-header h3 { font-size: 15px; color: #34495e; margin: 0 0 4px; }
.strategy-badges { display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }
.table-wrap { overflow-x: auto; margin: 10px 0; }
```

- [ ] **Step 4: Wrap wide tables**

Wrap cross-strategy and strategy stats tables with:

```html
<div class="table-wrap">
  <table>...</table>
</div>
```

- [ ] **Step 5: Run test to verify pass**

Run:

```powershell
pixi run pytest tests/test_reporting.py::TestGenerateTearSheet::test_report_contains_professional_layout_css_hooks -q
```

Expected: PASS.

---

### Task 5: Verification and Report Regeneration

**Files:**
- Verify: `src/factorzen/reports/tear_sheet.py`
- Verify: `src/factorzen/reports/templates/tear_sheet.html`
- Verify: `tests/test_reporting.py`
- Output: `workspace/runs/artifacts/daily/reports/momentum_20d_20230101_20230131.html`

- [ ] **Step 1: Run reporting tests**

Run:

```powershell
pixi run pytest tests/test_reporting.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run focused lint**

Run:

```powershell
pixi run ruff check src/factorzen/reports/tear_sheet.py tests/test_reporting.py
```

Expected: all checks pass.

- [ ] **Step 3: Regenerate the sample report**

Run:

```powershell
pixi run fz report build momentum_20d --start 20230101 --end 20230131 --all
```

Expected: report generation succeeds and writes `workspace/runs/artifacts/daily/reports/momentum_20d_20230101_20230131.html`.

- [ ] **Step 4: Inspect generated HTML for semantic markers**

Run:

```powershell
rg -n "研究仪表盘|策略方向|多头 TopN|分位数组合多空|多空组合|组合收益|dashboard-grid|strategy-header" workspace\runs\artifacts\daily\reports\momentum_20d_20230101_20230131.html
```

Expected: all semantic markers appear.


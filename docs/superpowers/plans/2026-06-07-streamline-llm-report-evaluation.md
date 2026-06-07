# Streamline LLM Report Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove duplicated LLM judgments from FactorZen reports so deterministic rules remain the sole source of ratings, risks, gaps, and next actions, while the optional LLM contributes only factor intuition and non-obvious cross-metric interpretation.

**Architecture:** Keep the existing single-call, content-addressed LLM service and shared daily-results cache. Narrow the structured LLM output from seven judgment fields to two explanatory fields, exclude non-semantic direction prose from the snapshot hash, render the explanation once in a clearly subordinate report section, and require explicit `--llm-explain` opt-in in every pipeline.

**Tech Stack:** Python 3.10+, dataclasses, Pydantic configuration, Jinja2 HTML templates, pytest, pixi, Ruff, mypy.

---

## Scope And Decisions

### Target behavior

- Rule-based report code remains the only owner of:
  - factor score and star rating;
  - evidence strength;
  - risk and rating caps;
  - research verdict and action;
  - evidence gaps and next steps.
- The LLM returns exactly two required text fields:
  - `factor_intuition`: economic or behavioral intuition that does not repeat the scorecard;
  - `cross_metric_analysis`: non-obvious agreement or tension across IC, OOS, backtest, turnover, quality, and direction evidence.
- The report renders the two LLM fields once under `大模型补充解读`.
- `--all` and no-YAML defaults do not enable the LLM.
- Only explicit `--llm-explain` enables an API request. `--llm-refresh` only controls cache bypass after explicit enablement.
- Prompt version changes from `v1` to `v2`; old cached JSON remains on disk but is not reused.
- Normal-direction reason wording does not affect cache identity. A real direction change from normal to reversed still changes the cache key.
- Default output budget drops from 700 to 400 tokens.

### Non-goals

- Do not change deterministic report scoring thresholds or research verdict rules.
- Do not change backtest direction selection behavior.
- Do not add a second LLM call, model ensemble, self-critique, or LLM-as-judge pass.
- Do not migrate or delete archived `v1` LLM artifacts.
- Do not add LLM settings to `RunConfig`; CLI opt-in remains separate execution metadata.
- Do not call a live LLM during tests.

### Corrected baseline finding

`daily_single` and `generate_report` already use the same cache directory returned by `daily_result_output_dir()`. The cache issue to fix is not directory duplication: it is avoidable hash divergence caused by including free-form direction `reason` text in the snapshot.

## File Map

### LLM contract and prompt

- Modify `src/factorzen/llm/schema.py`
  - Keep the public `LLMExplanation` name.
  - Replace judgment fields with `factor_intuition` and `cross_metric_analysis`.
- Modify `src/factorzen/llm/prompt.py`
  - Upgrade to `PROMPT_VERSION = "v2"`.
  - Request only the two explanatory fields and explicitly forbid scoring or recommendations.
- Modify `tests/test_llm_explain_schema.py`
  - Define the new accepted and rejected JSON contract.
- Create `tests/test_llm_explain_prompt.py`
  - Lock the prompt version, required fields, and forbidden judgment fields.

### Snapshot, cache, and output budget

- Modify `src/factorzen/llm/snapshot.py`
  - Hash only the actual `reversed` direction state, not free-form reason text.
- Modify `src/factorzen/llm/config.py`
  - Reduce the default and invalid-value fallback token limit to 400.
- Modify `tests/test_llm_explain_snapshot.py`
  - Verify reason wording does not alter snapshot identity.
- Modify `tests/test_llm_explain_cache.py`
  - Use the new schema and verify prompt-version cache invalidation.
- Modify `tests/test_llm_explain_service.py`
  - Use the new schema while retaining the one-request/cache-reuse contract.
- Modify `tests/test_llm_explain_config.py`
  - Verify the new default output budget.

### Report rendering

- Modify `src/factorzen/reports/tear_sheet.py`
  - Remove LLM content from `_generate_summary_text`.
  - Validate and prepare only the two explanatory fields.
- Modify `src/factorzen/reports/templates/tear_sheet.html`
  - Remove LLM rating, confidence, risk, recommendation, and next-step UI.
  - Render the two fields once under a subordinate heading.
- Modify `tests/test_reporting.py`
  - Verify one-time rendering and deterministic-summary isolation.

### Explicit opt-in

- Modify `src/factorzen/pipelines/daily_single.py`
  - Stop enabling LLM for `--all` and built-in no-YAML defaults.
  - Update CLI help.
- Modify `src/factorzen/pipelines/_report_config.py`
  - Stop enabling LLM for `--all`.
- Modify `tests/test_run_daily_single_config.py`
  - Verify deep/default presets keep LLM disabled unless explicit.
- Modify `tests/test_generate_report_persistence.py`
  - Verify report `--all` keeps LLM disabled unless explicit.

### User-facing documentation

- Modify `.env.example`
  - Document 400-token default.
- Modify `README.md`
  - Remove claims that no-YAML runs enable LLM.
- Modify `docs/runbook.md`
  - Make explicit opt-in statements internally consistent.
- Modify `docs/project-explanation.md`
  - Describe the LLM as an optional explanatory supplement, not an evaluator.
- Modify `CHANGELOG.md`
  - Record report de-duplication and explicit opt-in behavior under `Unreleased`.

## Execution Prerequisite

The current worktree contains unrelated uncommitted changes, including `daily_single.py` and its tests. Before implementing:

1. Preserve those changes in their intended branch or commit.
2. Use `superpowers:using-git-worktrees` to create an isolated implementation worktree from the intended base.
3. Do not reset, discard, or overwrite existing user changes.
4. Run all commands through `pixi`; no GPU pre-flight is needed because this work performs no training.

### Task 1: Narrow The LLM Output Contract

**Files:**
- Modify: `tests/test_llm_explain_schema.py`
- Create: `tests/test_llm_explain_prompt.py`
- Modify: `src/factorzen/llm/schema.py`
- Modify: `src/factorzen/llm/prompt.py`

- [ ] **Step 1: Replace schema tests with the two-field contract**

Replace the accepted examples and assertions in `tests/test_llm_explain_schema.py` with:

```python
from factorzen.llm.schema import LLMExplanation, parse_llm_explanation


def test_parse_llm_explanation_accepts_explanatory_json():
    raw = """
    {
      "factor_intuition": "量价相关性可能刻画趋势确认与拥挤交易之间的关系。",
      "cross_metric_analysis": "全样本 IC 显著但样本外减弱，说明统计存在性强于可迁移性。"
    }
    """

    explanation = parse_llm_explanation(raw)

    assert explanation == LLMExplanation(
        factor_intuition="量价相关性可能刻画趋势确认与拥挤交易之间的关系。",
        cross_metric_analysis="全样本 IC 显著但样本外减弱，说明统计存在性强于可迁移性。",
    )


def test_parse_llm_explanation_rejects_invalid_json():
    assert parse_llm_explanation("not-json") is None


def test_parse_llm_explanation_rejects_missing_required_fields():
    assert parse_llm_explanation('{"factor_intuition": "只有一个字段"}') is None


def test_parse_llm_explanation_rejects_legacy_judgment_shape():
    raw = """
    {
      "rating": "weak",
      "confidence": "low",
      "factor_intuition": "旧格式",
      "evidence_assessment": "旧格式",
      "risk_flags": [],
      "usage_suggestion": "旧格式",
      "next_steps": []
    }
    """

    assert parse_llm_explanation(raw) is None


def test_parse_llm_explanation_rejects_extra_judgment_fields():
    raw = """
    {
      "factor_intuition": "短期反转可能来自价格过度反应。",
      "cross_metric_analysis": "IC 与回测方向一致。",
      "rating": "strong"
    }
    """

    assert parse_llm_explanation(raw) is None


def test_parse_llm_explanation_accepts_json_inside_code_fence():
    raw = """
    ```json
    {
      "factor_intuition": "短期反转可能来自价格过度反应。",
      "cross_metric_analysis": "IC 与回测方向一致，但高换手削弱了可交易性。"
    }
    ```
    """

    explanation = parse_llm_explanation(raw)

    assert explanation is not None
    assert explanation.cross_metric_analysis.endswith("可交易性。")
```

- [ ] **Step 2: Add prompt contract tests**

Create `tests/test_llm_explain_prompt.py`:

```python
from factorzen.llm.prompt import PROMPT_VERSION, build_messages


def test_prompt_v2_requests_only_explanatory_fields():
    messages = build_messages({"ic": {"mean": 0.02}})
    combined = "\n".join(message["content"] for message in messages)

    assert PROMPT_VERSION == "v2"
    assert "factor_intuition" in combined
    assert "cross_metric_analysis" in combined
    for forbidden in (
        '"rating"',
        '"confidence"',
        '"risk_flags"',
        '"usage_suggestion"',
        '"next_steps"',
    ):
        assert forbidden not in combined


def test_prompt_forbids_replacing_rule_based_judgments():
    messages = build_messages({"ic": {"mean": 0.02}})
    combined = "\n".join(message["content"] for message in messages)

    assert "不得给出评级、置信度、风险清单、使用建议或下一步动作" in combined
    assert "不要逐项复述输入指标" in combined
```

- [ ] **Step 3: Run the new contract tests and verify failure**

Run:

```bash
pixi run pytest tests/test_llm_explain_schema.py tests/test_llm_explain_prompt.py -v
```

Expected: failures because `LLMExplanation` still requires seven fields, the old shape is accepted, and `PROMPT_VERSION` is still `v1`.

- [ ] **Step 4: Replace the schema fields and parser**

In `src/factorzen/llm/schema.py`, remove `Literal`, `Rating`, `Confidence`, `_as_short_list`, and the seven-field parsing logic. Keep `parse_llm_explanation()` and `explanation_from_dict()` entry points, but use:

```python
@dataclass(frozen=True)
class LLMExplanation:
    factor_intuition: str
    cross_metric_analysis: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _required_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _from_dict(data: dict[str, Any]) -> LLMExplanation | None:
    if set(data) != {"factor_intuition", "cross_metric_analysis"}:
        return None
    factor_intuition = _required_str(data.get("factor_intuition"))
    cross_metric_analysis = _required_str(data.get("cross_metric_analysis"))
    if factor_intuition is None or cross_metric_analysis is None:
        return None
    return LLMExplanation(
        factor_intuition=factor_intuition,
        cross_metric_analysis=cross_metric_analysis,
    )
```

- [ ] **Step 5: Replace the prompt with the v2 explanatory prompt**

In `src/factorzen/llm/prompt.py`, set `PROMPT_VERSION = "v2"` and replace the prompt text with:

```python
SYSTEM_PROMPT = (
    "你是量化因子研究报告的补充解释助手。只能基于用户提供的结构化指标解释，"
    "不得编造未提供的数据，不得给出投资建议。报告的评分、评级、风险、缺口和下一步"
    "已经由确定性规则生成，你不得重新判断或覆盖这些结论。输出必须是严格 JSON。"
)


def build_messages(snapshot: dict[str, Any]) -> list[dict[str, str]]:
    payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    user_prompt = (
        "基于以下因子评估摘要，输出两个简短中文字段："
        '"factor_intuition" 说明因子的经济、行为或市场微观结构直觉，100字以内；'
        '"cross_metric_analysis" 只指出 IC、样本外、回测、换手、数据质量和方向证据之间'
        "非显然的一致性或矛盾，180字以内。不要逐项复述输入指标。"
        "不得给出评级、置信度、风险清单、使用建议或下一步动作。"
        f"\n摘要JSON：{payload}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
```

- [ ] **Step 6: Run contract tests and verify pass**

Run:

```bash
pixi run pytest tests/test_llm_explain_schema.py tests/test_llm_explain_prompt.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit the contract change**

```bash
git add src/factorzen/llm/schema.py src/factorzen/llm/prompt.py tests/test_llm_explain_schema.py tests/test_llm_explain_prompt.py
git -c user.name='rookiewu417' -c user.email='1007372080@qq.com' commit -m "refactor: narrow llm report contract"
git log -1 --format='%an <%ae>'
```

Expected identity: `rookiewu417 <1007372080@qq.com>`.

### Task 2: Stabilize Cache Semantics And Reduce Output Budget

**Files:**
- Modify: `tests/test_llm_explain_snapshot.py`
- Modify: `tests/test_llm_explain_cache.py`
- Modify: `tests/test_llm_explain_service.py`
- Modify: `tests/test_llm_explain_config.py`
- Modify: `src/factorzen/llm/snapshot.py`
- Modify: `src/factorzen/llm/config.py`

- [ ] **Step 1: Update cached explanation fixtures**

In `tests/test_llm_explain_cache.py` and `tests/test_llm_explain_service.py`, replace every seven-field `LLMExplanation(...)` fixture with:

```python
LLMExplanation(
    factor_intuition="短期反转刻画近期价格过度反应。",
    cross_metric_analysis="IC 较弱且样本外衰减，回测收益不足以提供额外支持。",
)
```

- [ ] **Step 2: Add prompt-version invalidation coverage**

Append to `tests/test_llm_explain_cache.py`:

```python
def test_cache_key_changes_when_prompt_version_changes():
    kwargs = {
        "factor_name": "momentum_20d",
        "start": "20250101",
        "end": "20260513",
        "model": "model-a",
        "snapshot": {"ic": {"mean": 0.02}},
    }

    assert cache_key(prompt_version="v1", **kwargs) != cache_key(
        prompt_version="v2", **kwargs
    )
```

- [ ] **Step 3: Add stable direction snapshot coverage**

Append to `tests/test_llm_explain_snapshot.py`, reusing simple `None` metric inputs:

```python
def test_snapshot_ignores_direction_reason_wording():
    common = {
        "factor_name": "momentum_20d",
        "factor_description": "20日价格动量",
        "frequency": "daily",
        "date_range": "2025-01-01 ~ 2026-05-13",
        "universe": "csi300",
        "ic_result": None,
        "bt_result": None,
        "to_result": None,
    }

    first = build_factor_snapshot(
        **common,
        backtest_direction={"direction": "normal", "reason": "IC非负，保持原方向"},
    )
    second = build_factor_snapshot(
        **common,
        backtest_direction={"direction": "normal", "reason": "configured direction"},
    )
    reversed_snapshot = build_factor_snapshot(
        **common,
        backtest_direction={"direction": "reversed", "reason": "negative IC"},
    )

    assert first["direction"] == {"reversed": False}
    assert first == second
    assert first != reversed_snapshot
```

- [ ] **Step 4: Add default token-budget coverage**

Append to `tests/test_llm_explain_config.py`:

```python
def test_config_uses_compact_default_token_budget(monkeypatch):
    monkeypatch.delenv("FACTORZEN_LLM_MAX_TOKENS", raising=False)

    config = load_llm_config(enabled=False, env_file=None)

    assert config.max_tokens == 400


def test_config_invalid_token_budget_falls_back_to_compact_default(monkeypatch):
    monkeypatch.setenv("FACTORZEN_LLM_MAX_TOKENS", "invalid")

    config = load_llm_config(enabled=False, env_file=None)

    assert config.max_tokens == 400
```

- [ ] **Step 5: Run snapshot, cache, service, and config tests and verify failure**

Run:

```bash
pixi run pytest tests/test_llm_explain_snapshot.py tests/test_llm_explain_cache.py tests/test_llm_explain_service.py tests/test_llm_explain_config.py -v
```

Expected: direction snapshot and 400-token assertions fail before implementation; updated schema fixtures may also expose any remaining legacy-field usage.

- [ ] **Step 6: Remove direction reason from the snapshot**

In `src/factorzen/llm/snapshot.py`, replace the direction object with:

```python
"direction": {
    "reversed": direction.get("direction") == "reversed",
},
```

Do not remove `backtest_direction` from the service signature; the actual normal/reversed state remains part of cache identity.

- [ ] **Step 7: Reduce the default token budget**

In `src/factorzen/llm/config.py`:

```python
max_tokens: int = 400
```

and:

```python
max_tokens_raw = _get_setting("FACTORZEN_LLM_MAX_TOKENS", file_values) or "400"
```

and in the invalid integer fallback:

```python
except ValueError:
    max_tokens = 400
```

- [ ] **Step 8: Run focused LLM tests and verify pass**

Run:

```bash
pixi run pytest tests/test_llm_explain_snapshot.py tests/test_llm_explain_cache.py tests/test_llm_explain_service.py tests/test_llm_explain_config.py tests/test_llm_explain_client.py -v
```

Expected: all tests pass; no network request is made.

- [ ] **Step 9: Commit cache and budget changes**

```bash
git add src/factorzen/llm/snapshot.py src/factorzen/llm/config.py tests/test_llm_explain_snapshot.py tests/test_llm_explain_cache.py tests/test_llm_explain_service.py tests/test_llm_explain_config.py
git -c user.name='rookiewu417' -c user.email='1007372080@qq.com' commit -m "fix: stabilize llm explanation cache"
git log -1 --format='%an <%ae>'
```

Expected identity: `rookiewu417 <1007372080@qq.com>`.

### Task 3: Remove LLM Judgment Duplication From The Report

**Files:**
- Modify: `tests/test_reporting.py`
- Modify: `src/factorzen/reports/tear_sheet.py`
- Modify: `src/factorzen/reports/templates/tear_sheet.html`

- [ ] **Step 1: Replace the LLM rendering test fixture and assertions**

Change `test_html_contains_llm_explanation_when_provided` in `tests/test_reporting.py` to:

```python
def test_html_contains_llm_supplement_once_when_provided(
    self, ic_result, bt_result, to_result
):
    html = generate_tear_sheet(
        "momentum_20d",
        ic_result,
        bt_result,
        to_result,
        llm_explanation={
            "factor_intuition": "动量因子可能刻画价格趋势的行为延续。",
            "cross_metric_analysis": "IC 为正但样本外减弱，统计证据强于迁移证据。",
        },
    )

    assert html.count("大模型补充解读") == 1
    assert html.count("动量因子可能刻画价格趋势的行为延续。") == 1
    assert html.count("IC 为正但样本外减弱，统计证据强于迁移证据。") == 1
    assert "结论标签" not in html
    assert "置信度" not in html
    assert "LLM 综合结论" not in html
    assert "LLM 使用建议" not in html
```

- [ ] **Step 2: Replace the summary duplication test**

Replace `test_summary_uses_llm_explanation_when_provided` with:

```python
def test_rule_based_summary_does_not_embed_llm_text(
    self, ic_result, bt_result, to_result
):
    llm_text = "LLM_ONLY_CROSS_METRIC_TEXT"
    html = generate_tear_sheet(
        "momentum_20d",
        ic_result,
        bt_result,
        to_result,
        llm_explanation={
            "factor_intuition": "LLM_ONLY_INTUITION_TEXT",
            "cross_metric_analysis": llm_text,
        },
    )

    summary = html.split("<h2>综合评估</h2>", 1)[1].split(
        "<h2>大模型补充解读</h2>", 1
    )[0]

    assert "LLM_ONLY_INTUITION_TEXT" not in summary
    assert llm_text not in summary
    assert html.count(llm_text) == 1
```

- [ ] **Step 3: Update the section-order test**

In `test_summary_and_llm_explanation_appear_before_analysis`, use the two-field fixture and replace the heading lookup with:

```python
llm_pos = html.index("<h2>大模型补充解读</h2>")
```

Keep the assertions that both summary and LLM supplement appear before `收益表现`.

- [ ] **Step 4: Keep the LLM supplement out of the first-screen overview**

In `test_overview_panel_is_focused_executive_summary`, replace the old heading assertion with:

```python
assert "大模型补充解读" not in overview_html
```

The supplement remains in the separate `research-summary` panel and must not alter the first-screen deterministic decision.

- [ ] **Step 5: Add malformed-shape degradation coverage**

Append near the other LLM report tests:

```python
def test_report_ignores_legacy_llm_judgment_shape(
    self, ic_result, bt_result, to_result
):
    html = generate_tear_sheet(
        "momentum_20d",
        ic_result,
        bt_result,
        to_result,
        llm_explanation={
            "rating": "weak",
            "confidence": "low",
            "factor_intuition": "legacy",
            "evidence_assessment": "legacy",
            "risk_flags": [],
            "usage_suggestion": "legacy",
            "next_steps": [],
        },
    )

    assert "大模型补充解读" not in html
```

- [ ] **Step 6: Run focused report tests and verify failure**

Run:

```bash
pixi run pytest tests/test_reporting.py -k "llm or summary_and_llm" -v
```

Expected: failures because the old report embeds LLM text in the summary and still renders rating, confidence, risks, usage suggestions, and next steps.

- [ ] **Step 7: Remove LLM input from deterministic summary generation**

In `src/factorzen/reports/tear_sheet.py`:

1. Change the signature to:

```python
def _generate_summary_text(
    factor_name: str,
    metrics: dict[str, Any],
) -> str:
```

2. Delete the block that reads `evidence_assessment`, `usage_suggestion`, `rating`, and `confidence`.
3. Change the call site to:

```python
summary_html = _generate_summary_text(factor_name, metrics)
```

Do not alter any rule-based scorecard, IC, IR, return, cap, or warning logic in this function.

- [ ] **Step 8: Make the report view accept only the new fields**

Replace `_prepare_llm_explanation_view` with:

```python
def _prepare_llm_explanation_view(
    llm_explanation: dict[str, Any] | None,
) -> dict[str, str] | None:
    if not isinstance(llm_explanation, dict):
        return None

    factor_intuition = str(llm_explanation.get("factor_intuition", "")).strip()
    cross_metric_analysis = str(
        llm_explanation.get("cross_metric_analysis", "")
    ).strip()
    if not factor_intuition or not cross_metric_analysis:
        return None
    return {
        "factor_intuition": factor_intuition,
        "cross_metric_analysis": cross_metric_analysis,
    }
```

Remove `_format_label_with_code` only if `rg "_format_label_with_code" src/factorzen` confirms that the helper has no remaining callers. This removal is allowed because the new LLM view makes it unused.

- [ ] **Step 9: Replace the template section**

In `src/factorzen/reports/templates/tear_sheet.html`, replace the old LLM block with:

```html
  <!-- 大模型补充解读 -->
  {% if llm_explanation is not none %}
  <div style="margin-top:16px">
    <h2>大模型补充解读</h2>
    <p><strong>因子直觉：</strong>{{ llm_explanation.factor_intuition }}</p>
    <p><strong>跨指标观察：</strong>{{ llm_explanation.cross_metric_analysis }}</p>
    <p style="color:#7f8c8d;font-size:13px;margin-top:8px">
      该内容仅补充解释，不参与本报告的评分、评级、风险判断或研究决策。
    </p>
  </div>
  {% endif %}
```

- [ ] **Step 10: Run focused and full report tests**

Run:

```bash
pixi run pytest tests/test_reporting.py -k "llm or summary_and_llm" -v
pixi run pytest tests/test_reporting.py -v
```

Expected: both commands pass.

- [ ] **Step 11: Commit report de-duplication**

```bash
git add src/factorzen/reports/tear_sheet.py src/factorzen/reports/templates/tear_sheet.html tests/test_reporting.py
git -c user.name='rookiewu417' -c user.email='1007372080@qq.com' commit -m "refactor: remove duplicate llm judgments"
git log -1 --format='%an <%ae>'
```

Expected identity: `rookiewu417 <1007372080@qq.com>`.

### Task 4: Require Explicit LLM Opt-In Everywhere

**Files:**
- Modify: `tests/test_run_daily_single_config.py`
- Modify: `tests/test_generate_report_persistence.py`
- Modify: `src/factorzen/pipelines/daily_single.py`
- Modify: `src/factorzen/pipelines/_report_config.py`

- [ ] **Step 1: Change daily preset expectations**

In `tests/test_run_daily_single_config.py`:

1. Rename `test_merge_run_config_args_all_enables_single_factor_defaults` to `test_merge_run_config_args_all_keeps_llm_opt_in`.
2. Change its assertion to:

```python
assert merged.llm_explain is False
```

3. Rename `test_merge_run_config_args_without_yaml_enables_comprehensive_defaults` to `test_merge_run_config_args_without_yaml_keeps_llm_opt_in`.
4. Change its final assertion to:

```python
assert merged.llm_explain is False
```

5. Add:

```python
def test_merge_run_config_args_preserves_explicit_llm_opt_in():
    from factorzen.pipelines.daily_single import _merge_run_config_args

    args = Namespace(
        factor="momentum_20d",
        start="20240101",
        end="20240131",
        universe=None,
        benchmark=None,
        seed=None,
        ic_method=None,
        neutralized_ic=None,
        event_study=None,
        all=True,
        llm_explain=True,
        llm_refresh=False,
    )

    merged = _merge_run_config_args(args, None)

    assert merged.llm_explain is True
```

- [ ] **Step 2: Change report preset expectations**

In `tests/test_generate_report_persistence.py`:

1. Rename `test_merge_report_config_args_all_enables_report_defaults` to `test_merge_report_config_args_all_keeps_llm_opt_in`.
2. Change its LLM assertion to:

```python
assert merged.llm_explain is False
```

3. In `test_merge_report_config_args_all_overrides_yaml_benchmark`, change the LLM assertion to:

```python
assert merged.llm_explain is False
```

4. Add:

```python
def test_merge_report_config_args_preserves_explicit_llm_opt_in():
    from argparse import Namespace

    from factorzen.pipelines import generate_report as mod

    args = Namespace(
        factor="momentum_20d",
        start="20240101",
        end="20240131",
        universe=None,
        benchmark=None,
        frequency="daily",
        reuse=False,
        config=None,
        all=True,
        ic_method=None,
        neutralized_ic=None,
        event_study=None,
        llm_explain=True,
        llm_refresh=False,
    )

    merged = mod._merge_report_config_args(args, None)

    assert merged.llm_explain is True
```

- [ ] **Step 3: Run merge tests and verify failure**

Run:

```bash
pixi run pytest tests/test_run_daily_single_config.py tests/test_generate_report_persistence.py -k "llm or all_enables or all_keeps or without_yaml" -v
```

Expected: false assertions fail because `--all` and the no-YAML path still force `llm_explain = True`.

- [ ] **Step 4: Remove implicit enablement from the daily pipeline**

In `_merge_run_config_args()` in `src/factorzen/pipelines/daily_single.py`:

- Delete `args.llm_explain = True` from the `if getattr(args, "all", False):` block.
- Delete:

```python
if using_builtin_default:
    args.llm_explain = True
```

- If `using_builtin_default` then has no remaining references, remove its assignment.
- Keep all benchmark, IC method, neutralized IC, event study, seed, preprocessing, strategy, and walk-forward defaults unchanged.

Update `--llm-explain` help to:

```python
help="显式启用大模型补充解读；默认关闭，缺少 FACTORZEN_LLM_* 配置时跳过",
```

Update the `--all` help so its feature list no longer mentions LLM.

- [ ] **Step 5: Remove implicit enablement from report config merging**

In `src/factorzen/pipelines/_report_config.py`, delete:

```python
args.llm_explain = True
```

from the `--all` block. Do not change other deep-report defaults.

In `src/factorzen/pipelines/generate_report.py`, change `--llm-explain` help to use `大模型补充解读`, and remove LLM from the `--all` help text.

- [ ] **Step 6: Run pipeline config and CLI tests**

Run:

```bash
pixi run pytest tests/test_run_daily_single_config.py tests/test_generate_report_persistence.py tests/test_cli.py -v
```

Expected: all tests pass. CLI forwarding still includes `--llm-explain` when the user explicitly provides it.

- [ ] **Step 7: Commit explicit opt-in behavior**

```bash
git add src/factorzen/pipelines/daily_single.py src/factorzen/pipelines/_report_config.py src/factorzen/pipelines/generate_report.py tests/test_run_daily_single_config.py tests/test_generate_report_persistence.py
git -c user.name='rookiewu417' -c user.email='1007372080@qq.com' commit -m "change: require explicit llm report opt in"
git log -1 --format='%an <%ae>'
```

Expected identity: `rookiewu417 <1007372080@qq.com>`.

### Task 5: Align Documentation And Verify The Whole Change

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `docs/runbook.md`
- Modify: `docs/project-explanation.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update environment documentation**

In `.env.example`, change:

```dotenv
FACTORZEN_LLM_MAX_TOKENS=400
```

Keep the existing statement that CLI calls must explicitly enable the feature.

- [ ] **Step 2: Correct README default behavior**

Update `README.md` so it states:

- LLM supplement is disabled by default.
- `--all` does not enable LLM.
- `--llm-explain` is required for an API call.
- The LLM does not participate in score, rating, risk, gap, or next-step decisions.
- The no-YAML research preset still includes the existing deterministic research modules, but not LLM.

Use this concise wording where the current default claims appear:

```markdown
LLM 补充解读默认关闭；只有显式传入 `--llm-explain` 时才会读取
`FACTORZEN_LLM_*` 配置。LLM 仅解释因子直觉与跨指标关系，不参与评分、评级或研究决策。
```

- [ ] **Step 3: Make runbook guidance internally consistent**

In `docs/runbook.md`:

- Keep the existing line that says LLM is explicitly enabled.
- Remove LLM from the no-YAML preset list.
- Add one example:

```bash
pixi run fz factor run my_alpha --start 20230101 --end 20241231 --llm-explain
```

- State that `--llm-refresh` is meaningful only together with `--llm-explain`.

- [ ] **Step 4: Clarify architecture documentation**

In `docs/project-explanation.md`, change the `llm/` description to:

```text
llm/            可选 LLM 补充解释：因子直觉与跨指标关系，不参与确定性评分
```

Add one sentence in the report section:

```markdown
评分、评级、风险、证据缺口和下一步均由确定性报告规则生成；显式启用的 LLM 只提供补充解释。
```

- [ ] **Step 5: Record the user-visible behavior change**

Under `## [Unreleased]` → `### Changed` in `CHANGELOG.md`, add:

```markdown
- **LLM 报告解读：** 收窄为因子直觉与跨指标补充解释，移除重复的评级、风险、使用建议和下一步；`--all` 与无 YAML 默认运行不再自动启用 LLM，必须显式传入 `--llm-explain`。
```

- [ ] **Step 6: Run focused regression tests**

Run:

```bash
pixi run pytest tests/test_llm_explain_schema.py tests/test_llm_explain_prompt.py tests/test_llm_explain_snapshot.py tests/test_llm_explain_cache.py tests/test_llm_explain_service.py tests/test_llm_explain_config.py tests/test_llm_explain_client.py tests/test_run_daily_single_config.py tests/test_generate_report_persistence.py tests/test_cli.py -v
pixi run pytest tests/test_reporting.py -v
```

Expected: all focused tests pass.

- [ ] **Step 7: Run repository verification**

Run:

```bash
pixi run lint
pixi run typecheck
pixi run test
pixi run coverage
git diff --check
```

Expected:

- Ruff exits 0.
- mypy exits 0.
- full pytest suite exits 0.
- coverage meets the repository threshold.
- `git diff --check` prints no output.

- [ ] **Step 8: Verify no legacy report fields remain in production code**

Run:

```bash
rg -n '"rating"|"confidence"|evidence_assessment|risk_flags|usage_suggestion|next_steps' src/factorzen/llm src/factorzen/reports
```

Expected:

- No matches in `src/factorzen/llm`.
- Report-side `next_steps` may still appear only in deterministic research-decision code.
- No LLM template references to rating, confidence, risk flags, usage suggestion, or LLM next steps.

- [ ] **Step 9: Verify explicit enablement behavior without making an API call**

Run:

```bash
pixi run fz factor run volume_return_corr_20d --start 20240101 --end 20240331 --dry-run
```

Expected dry-run JSON:

```json
"execution": {
  "llm_explain": false,
  "llm_refresh": false
}
```

Then run:

```bash
pixi run fz factor run volume_return_corr_20d --start 20240101 --end 20240331 --llm-explain --dry-run
```

Expected dry-run JSON:

```json
"execution": {
  "llm_explain": true,
  "llm_refresh": false
}
```

- [ ] **Step 10: Commit documentation**

```bash
git add .env.example README.md docs/runbook.md docs/project-explanation.md CHANGELOG.md
git -c user.name='rookiewu417' -c user.email='1007372080@qq.com' commit -m "docs: clarify optional llm report role"
git log -1 --format='%an <%ae>'
```

Expected identity: `rookiewu417 <1007372080@qq.com>`.

## Final Acceptance Criteria

- A normal report without `--llm-explain` contains no LLM section and makes no LLM request.
- `--all` and no-YAML runs leave `llm_explain` false.
- Explicit `--llm-explain` produces at most one request for a new semantic snapshot.
- A repeated explicit run with the same factor, period, model, prompt version, metrics, and actual direction loads the existing cache.
- Changing only direction reason wording does not invalidate the cache.
- Changing actual direction, metrics, model, period, or prompt version invalidates the cache.
- LLM JSON contains only `factor_intuition` and `cross_metric_analysis`.
- The report displays both LLM fields exactly once.
- LLM text never appears inside `综合评估` or `最终研究决策`.
- Rule-based score, stars, evidence strength, gaps, and next actions are unchanged by the presence or absence of LLM content.
- Existing v1 cache files remain untouched and are naturally bypassed by `PROMPT_VERSION = "v2"`.
- Focused tests, full tests, lint, typecheck, coverage, and `git diff --check` all pass.
- Every implementation commit reports `rookiewu417 <1007372080@qq.com>` as author.

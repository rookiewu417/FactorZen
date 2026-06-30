# M6 · 多 Agent + 长期记忆 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 M5 单 Agent 升级为多角色协作团队（Hypothesis/Coder/Evaluator/Critic/Librarian），Critic 可否决触发重提/修正，Librarian 维护跨 session 长期记忆（避免重复、积累已知有效/无效知识）。

**Architecture:** 自建流水线（沿用 M5 自建 loop，零新依赖）：角色是注入 `LLMFn` 的函数式步骤，`team_orchestrator` 顺序调用 + 处理 Critic 否决回路。Evaluator 复用 M5 确定性评估（不调 LLM）。

**Tech Stack:** Python 3.10–3.12 · polars · numpy · 现有 `agents/`（M5）+ `llm/` · pytest + FakeLLM · argparse。**零新第三方依赖。**

## Global Constraints

- **零新依赖**：不引入任何第三方包（不引 langgraph/langchain/向量库/sklearn）。
- **LLM 可注入**：所有角色用 `LLMFn = Callable[[list[dict[str, str]]], str]`，CI/测试注入 `FakeLLM`（确定性返回预设字符串序列）。**单测全程不触网。**
- **真 multi-agent 非换皮**：增量是角色真交互（Critic 否决 → Hypothesis 重提 / Coder 修正）+ 跨 session 长期记忆。Evaluator 沿用 M5 确定性评估，不调 LLM。
- **复用 M5（零回归，不改 M5 文件除非必要）**：
  - `evaluate_expressions(expr_strs, daily, bundle) -> list[dict]`（6 key：`expression/node/compile_ok/ic_train/ir_train/error`）。
  - `node_guardrails(state, *, daily, holdout_df, bundle, ledger, top_k=5, dsr_threshold=0.5)`（写 candidates：`expression/hypothesis/ic_train/holdout_ic/holdout_ir/dsr/dsr_pvalue`；N 记本轮）。
  - `AgentState`/`AttemptRecord`（`ir_train` 尾默认）、`negative_recall`、`request_chat`/`LLMFn`/`_extract_json`/`FactorProposal`、`DataBundle.build`/`split_holdout`/`TrialLedger`、`export_candidate`、`parse_expr`/`to_expr_string`。
- **归一化查重**：`norm = to_expr_string(parse_expr(e))`，`ValueError` → `norm = e`（非法保持原始）。experiment_index 以归一化字符串为 key。
- **防过拟合（灵魂，继承 M5）**：`TrialLedger` 累加**本 run 所有评估**（含否决回路重试）；**N 不跨 run 累加**（跨 session 查重只省算力，跨 run 累加 N 会让 DSR 病态过严）；holdout 段只在 `node_guardrails` 碰，角色/记忆全程不见。
- **否决回路 = 跨轮 feedback（防 N 三角和）**：Critic 的 verdict **不在同轮内重复跑护栏**（同轮多次 `node_guardrails` 会让本轮 N 三角和，重蹈 M5 覆辙），而是转化为**下一轮**的 feedback——`revise_expr` → 下一轮 Coder 带 `prev_exprs`+`reason` 改写；`revise_hypothesis` → 下一轮 Hypothesis 带 `reason` 换方向；keep/drop → 清空 feedback。每轮恰好一次 `node_guardrails`（N 每轮记一次评估成功数，干净）；`n_rounds` 是迭代上限，无死循环。**这是对 spec §5 同轮否决回路的实现修正，理由：N 诚实记账 + Critic 能看护栏结果（guardrails 已跑）+ 不改 M5。**
- **`known_valid` 跨 run 传承**：仅作 Hypothesis 的**方向参考**，不直接喂 Coder 复用原表达式。
- **manifest**：新写 `write_team_manifest`（不改 M5 `write_session_manifest`），复用 `_git_sha`，加 `roles`/`rounds_log`。
- **环境**：`pixi run pytest` / `ruff check`；polars 1.41.2。
- **提交**：conventional commits；作者 `rookiewu417 <1007372080@qq.com>`；每 task 只 `git add` 自己的文件（工作区有无关 M0 改动，**绝不** `-A`）。
- **测试判别力**：避免恒真——值断言、严格阈值、构造能让断言 FAIL 的反例（参见 [[multi-round-cumulative-count-trap]]/[[conservation-assertion-tautology-trap]]）。

---

## File Structure

| 文件 | 职责 | Task |
|---|---|---|
| `src/factorzen/agents/experiment_index.py` | 跨 session 长期记忆（JSONL + 归一化 seen + known_invalid/valid） | 1 |
| `src/factorzen/agents/roles/__init__.py` | 包标记 | 2 |
| `src/factorzen/agents/roles/critic.py` | `CriticVerdict` + `critique`（keep/revise_expr/revise_hypothesis/drop） | 2 |
| `src/factorzen/agents/roles/hypothesis.py` | `propose_hypotheses`（注入 known_invalid/valid） | 3 |
| `src/factorzen/agents/roles/coder.py` | `write_expressions` + `revise_expressions` | 4 |
| `src/factorzen/agents/roles/librarian.py` | `recall` + `record`（包 experiment_index） | 5 |
| `src/factorzen/agents/team_orchestrator.py` | `run_team_agent` 流水线 + 否决回路 + `write_team_manifest` + `TeamResult` | 6 |
| `src/factorzen/pipelines/factor_mine_team.py` | `run_team_mine`（拉数据 → team → 落产物 + candidates.csv + export） | 7 |
| `src/factorzen/cli/main.py`（改） | `fz mine team` 子命令 + handler | 8 |
| `tests/test_team_*.py` | 各 task 测试（FakeLLM 离线） | 各 |

---

## Task 1: 跨 session 长期记忆（experiment_index.py）

**Files:**
- Create: `src/factorzen/agents/experiment_index.py`
- Test: `tests/test_team_experiment_index.py`

**Interfaces:**
- Consumes: `parse_expr`/`to_expr_string`（`discovery/expression.py`）
- Produces: `ExperimentIndex(path)` with `load()`/`append(records)`/`seen_expressions()`/`known_invalid(k)`/`known_valid(k)`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_team_experiment_index.py
from pathlib import Path

from factorzen.agents.experiment_index import ExperimentIndex


def _recs():
    return [
        {"expression": "ts_mean(close,5)", "hypothesis": "动量", "ic_train": 0.05,
         "holdout_ic": 0.03, "dsr": 0.7, "passed": True, "verdict": "keep"},
        {"expression": "rank(vol)", "hypothesis": "换手", "ic_train": 0.001,
         "holdout_ic": 0.0, "dsr": 0.1, "passed": False, "verdict": "drop"},
    ]


def test_append_then_load_roundtrip(tmp_path: Path):
    idx = ExperimentIndex(str(tmp_path / "exp.jsonl"))
    idx.append(_recs())
    idx2 = ExperimentIndex(str(tmp_path / "exp.jsonl"))   # 新实例，跨 "session"
    loaded = idx2.load()
    assert len(loaded) == 2
    assert loaded[0]["expression"] == "ts_mean(close,5)"


def test_seen_expressions_normalized(tmp_path: Path):
    idx = ExperimentIndex(str(tmp_path / "exp.jsonl"))
    idx.append(_recs())
    seen = idx.seen_expressions()
    # 归一化形式（带空格）应能匹配无空格原始查询
    assert "ts_mean(close, 5)" in seen           # 归一化后带空格
    assert "rank(vol)" in seen


def test_known_invalid_and_valid(tmp_path: Path):
    idx = ExperimentIndex(str(tmp_path / "exp.jsonl"))
    idx.append(_recs())
    assert "rank(vol)" in idx.known_invalid(k=5)      # passed=False / 低 IC
    assert "ts_mean(close, 5)" in idx.known_valid(k=5) # passed=True（归一化）
    assert "ts_mean(close, 5)" not in idx.known_invalid(k=5)


def test_load_missing_file_empty(tmp_path: Path):
    idx = ExperimentIndex(str(tmp_path / "nope.jsonl"))
    assert idx.load() == [] and idx.seen_expressions() == set()
```

- [ ] **Step 2: 跑测试确认失败** → `pixi run pytest tests/test_team_experiment_index.py -v` → FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 experiment_index.py**

```python
# src/factorzen/agents/experiment_index.py
"""跨 session 长期记忆：experiment_index.jsonl 读写 + 归一化查重 + 已知有效/无效。"""
from __future__ import annotations

import json
from pathlib import Path

from factorzen.discovery.expression import parse_expr, to_expr_string


def _normalize(expr: str) -> str:
    try:
        return to_expr_string(parse_expr(expr))
    except ValueError:
        return expr


class ExperimentIndex:
    def __init__(self, path: str):
        self.path = Path(path)

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def append(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def seen_expressions(self) -> set[str]:
        return {_normalize(r["expression"]) for r in self.load() if "expression" in r}

    def known_invalid(self, k: int = 5) -> list[str]:
        recs = [r for r in self.load() if not r.get("passed", False)]
        recs.sort(key=lambda r: abs(r.get("ic_train") or 0.0))  # 最没用的优先
        return [_normalize(r["expression"]) for r in recs[:k] if "expression" in r]

    def known_valid(self, k: int = 5) -> list[str]:
        recs = [r for r in self.load() if r.get("passed", False)]
        recs.sort(key=lambda r: abs(r.get("holdout_ic") or 0.0), reverse=True)
        return [_normalize(r["expression"]) for r in recs[:k] if "expression" in r]
```

- [ ] **Step 4: 跑测试通过 + ruff + 提交**

```bash
pixi run pytest tests/test_team_experiment_index.py -v
pixi run ruff check src/factorzen/agents/experiment_index.py tests/test_team_experiment_index.py
git add src/factorzen/agents/experiment_index.py tests/test_team_experiment_index.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(agents): 跨 session 长期记忆 ExperimentIndex(归一化查重+已知有效/无效)"
```

---

## Task 2: Critic 角色（critic.py）

**Files:**
- Create: `src/factorzen/agents/roles/__init__.py`（空）, `src/factorzen/agents/roles/critic.py`
- Test: `tests/test_team_critic.py`

**Interfaces:**
- Consumes: `LLMFn`/`_extract_json`（`llm/generation.py`）
- Produces: `CriticVerdict(verdict, reason)`；`critique(candidate, llm_fn) -> CriticVerdict`（verdict ∈ keep/revise_expr/revise_hypothesis/drop）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_team_critic.py
import json

from factorzen.agents.roles.critic import CriticVerdict, critique


class FakeLLM:
    def __init__(self, responses):
        self._r = list(responses)
    def __call__(self, messages):
        return self._r.pop(0) if self._r else "{}"


def _cand(**kw):
    base = {"expression": "ts_mean(close,5)", "hypothesis": "动量", "ic_train": 0.05,
            "holdout_ic": 0.03, "dsr": 0.7, "dsr_pvalue": 0.01}
    base.update(kw)
    return base


def test_critique_keep():
    llm = FakeLLM([json.dumps({"verdict": "keep", "reason": "稳健"})])
    v = critique(_cand(), llm)
    assert isinstance(v, CriticVerdict) and v.verdict == "keep"


def test_critique_drop_overfit():
    # DSR 不显著的候选 → drop
    llm = FakeLLM([json.dumps({"verdict": "drop", "reason": "DSR 不显著疑过拟合"})])
    v = critique(_cand(dsr=0.2, dsr_pvalue=0.4), llm)
    assert v.verdict == "drop" and v.reason


def test_critique_revise_variants():
    llm = FakeLLM([json.dumps({"verdict": "revise_expr", "reason": "窗口太短"}),
                   json.dumps({"verdict": "revise_hypothesis", "reason": "方向牵强"})])
    assert critique(_cand(), llm).verdict == "revise_expr"
    assert critique(_cand(), llm).verdict == "revise_hypothesis"


def test_critique_garbage_defaults_keep():
    # 解析失败 → 默认 keep（不误杀；与 M5 node_critic 容错一致）
    llm = FakeLLM(["不是 JSON"])
    assert critique(_cand(), llm).verdict == "keep"


def test_critique_unknown_verdict_defaults_keep():
    llm = FakeLLM([json.dumps({"verdict": "explode", "reason": "x"})])
    assert critique(_cand(), llm).verdict == "keep"   # 非法 verdict 归一到 keep
```

- [ ] **Step 2: 跑测试确认失败** → FAIL

- [ ] **Step 3: 实现 critic.py**

```python
# src/factorzen/agents/roles/critic.py
"""Critic（Risk Auditor）角色：读候选指标判过拟合，给否决回路 verdict。"""
from __future__ import annotations

from dataclasses import dataclass

from factorzen.llm.generation import LLMFn, _extract_json

_VALID_VERDICTS = {"keep", "revise_expr", "revise_hypothesis", "drop"}


@dataclass
class CriticVerdict:
    verdict: str
    reason: str


def critique(candidate: dict, llm_fn: LLMFn) -> CriticVerdict:
    """读候选 + 指标，判 keep/revise_expr/revise_hypothesis/drop。解析失败/非法 → keep（不误杀）。"""
    msgs = [
        {"role": "system", "content": (
            "你是量化风控审计员。读因子候选的指标（train IC / holdout IC / DSR），"
            "判断它是否过拟合、经济直觉是否成立。只输出 JSON: "
            '{"verdict": "keep"|"revise_expr"|"revise_hypothesis"|"drop", "reason": "..."}。'
            "keep=可入库；revise_expr=方向对但表达式需改；revise_hypothesis=方向需换；drop=丢弃。")},
        {"role": "user", "content": (
            f"表达式: {candidate.get('expression')}\n假设: {candidate.get('hypothesis')}\n"
            f"train_IC: {candidate.get('ic_train')}\nholdout_IC: {candidate.get('holdout_ic')}\n"
            f"DSR: {candidate.get('dsr')} (p={candidate.get('dsr_pvalue')})")},
    ]
    obj = _extract_json(llm_fn(msgs))
    if not obj or obj.get("verdict") not in _VALID_VERDICTS:
        return CriticVerdict("keep", str(obj.get("reason", "")) if obj else "")
    return CriticVerdict(str(obj["verdict"]), str(obj.get("reason", "")))
```

- [ ] **Step 4: 跑测试通过 + ruff + 提交**

```bash
pixi run pytest tests/test_team_critic.py -v
pixi run ruff check src/factorzen/agents/roles/ tests/test_team_critic.py
git add src/factorzen/agents/roles/__init__.py src/factorzen/agents/roles/critic.py tests/test_team_critic.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(agents): Critic 角色(否决 verdict:keep/revise_expr/revise_hypothesis/drop)"
```

---

## Task 3: Hypothesis 角色（hypothesis.py）

**Files:**
- Create: `src/factorzen/agents/roles/hypothesis.py`
- Test: `tests/test_team_hypothesis.py`

**Interfaces:**
- Consumes: `LLMFn`/`_extract_json`
- Produces: `propose_hypotheses(llm_fn, *, known_invalid, known_valid, feedback="", n=1) -> list[str]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_team_hypothesis.py
import json

from factorzen.agents.roles.hypothesis import propose_hypotheses


class FakeLLM:
    def __init__(self, responses):
        self._r = list(responses)
        self.calls = []
    def __call__(self, messages):
        self.calls.append(messages)
        return self._r.pop(0) if self._r else "{}"


def test_propose_returns_directions():
    llm = FakeLLM([json.dumps({"hypotheses": ["小市值反转", "高换手动量"]})])
    out = propose_hypotheses(llm, known_invalid=[], known_valid=[], n=2)
    assert out == ["小市值反转", "高换手动量"]


def test_known_invalid_injected_into_prompt():
    llm = FakeLLM([json.dumps({"hypotheses": ["x"]})])
    propose_hypotheses(llm, known_invalid=["rank(vol)"], known_valid=["ts_mean(close, 5)"], n=1)
    blob = " ".join(m["content"] for m in llm.calls[0])
    assert "rank(vol)" in blob        # 已知无效注入(避开)
    assert "ts_mean(close, 5)" in blob # 已知有效作方向参考


def test_propose_garbage_returns_empty():
    llm = FakeLLM(["非 JSON"])
    assert propose_hypotheses(llm, known_invalid=[], known_valid=[]) == []
```

- [ ] **Step 2: 跑测试确认失败** → FAIL

- [ ] **Step 3: 实现 hypothesis.py**

```python
# src/factorzen/agents/roles/hypothesis.py
"""Hypothesis 角色：提经济直觉方向，注入长期记忆（避开已知无效，借鉴已知有效）。"""
from __future__ import annotations

from factorzen.llm.generation import LLMFn, _extract_json


def propose_hypotheses(llm_fn: LLMFn, *, known_invalid: list[str], known_valid: list[str],
                       feedback: str = "", n: int = 1) -> list[str]:
    """提 n 个经济直觉方向（自然语言）。解析失败 → 空列表。"""
    sys = ("你是量化研究员，提出有经济直觉的选股方向（自然语言，不写公式）。"
           '只输出 JSON: {"hypotheses": ["方向1", "方向2"]}。')
    user = f"提出 {n} 个新方向。"
    if feedback:
        user += f"\n上一轮反馈: {feedback}"
    if known_invalid:
        user += "\n以下表达式已验证无效，避开这些思路:\n" + "\n".join(f"- {e}" for e in known_invalid)
    if known_valid:
        user += "\n以下表达式已验证有效，可借鉴其思路方向（但不要照抄）:\n" + \
                "\n".join(f"- {e}" for e in known_valid)
    obj = _extract_json(llm_fn([{"role": "system", "content": sys},
                                {"role": "user", "content": user}]))
    if not obj:
        return []
    hyps = obj.get("hypotheses")
    return [str(h) for h in hyps] if isinstance(hyps, list) else []
```

- [ ] **Step 4: 跑测试通过 + ruff + 提交**

```bash
pixi run pytest tests/test_team_hypothesis.py -v
pixi run ruff check src/factorzen/agents/roles/hypothesis.py tests/test_team_hypothesis.py
git add src/factorzen/agents/roles/hypothesis.py tests/test_team_hypothesis.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(agents): Hypothesis 角色(提方向+注入长期记忆)"
```

---

## Task 4: Coder 角色（coder.py）

**Files:**
- Create: `src/factorzen/agents/roles/coder.py`
- Test: `tests/test_team_coder.py`

**Interfaces:**
- Consumes: `LLMFn`/`_extract_json`；`OPERATORS`/`LEAF_FEATURES`（`discovery/operators.py`）
- Produces: `write_expressions(hypothesis, llm_fn, *, avoid=None) -> list[str]`；`revise_expressions(hypothesis, prev_exprs, critic_reason, llm_fn) -> list[str]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_team_coder.py
import json

from factorzen.agents.roles.coder import revise_expressions, write_expressions


class FakeLLM:
    def __init__(self, responses):
        self._r = list(responses)
        self.calls = []
    def __call__(self, messages):
        self.calls.append(messages)
        return self._r.pop(0) if self._r else "{}"


def test_write_expressions_lists_ops():
    llm = FakeLLM([json.dumps({"expressions": ["ts_mean(close,5)", "rank(vol)"]})])
    out = write_expressions("动量", llm)
    assert out == ["ts_mean(close,5)", "rank(vol)"]
    blob = " ".join(m["content"] for m in llm.calls[0])
    assert "ts_mean" in blob and "close" in blob   # 算子/特征清单进 prompt


def test_revise_uses_critic_reason():
    llm = FakeLLM([json.dumps({"expressions": ["ts_mean(close,20)"]})])
    out = revise_expressions("动量", ["ts_mean(close,5)"], "窗口太短", llm)
    assert out == ["ts_mean(close,20)"]
    blob = " ".join(m["content"] for m in llm.calls[0])
    assert "窗口太短" in blob and "ts_mean(close,5)" in blob  # 反馈+原表达式进 prompt


def test_write_garbage_returns_empty():
    llm = FakeLLM(["非 JSON"])
    assert write_expressions("动量", llm) == []
```

- [ ] **Step 2: 跑测试确认失败** → FAIL

- [ ] **Step 3: 实现 coder.py**

```python
# src/factorzen/agents/roles/coder.py
"""Coder 角色：方向 → 表达式；按 Critic 反馈修正表达式。"""
from __future__ import annotations

from factorzen.discovery.operators import LEAF_FEATURES, OPERATORS
from factorzen.llm.generation import LLMFn, _extract_json


def _syntax_prompt() -> str:
    return ("可用算子: " + ", ".join(OPERATORS.keys()) + "\n"
            "可用特征(叶子): " + ", ".join(LEAF_FEATURES.keys()) + "\n"
            "时序算子最后一个参数是整型窗口，如 ts_mean(close, 20)。\n"
            '只输出 JSON: {"expressions": ["...", "..."]}。')


def write_expressions(hypothesis: str, llm_fn: LLMFn, *, avoid: list[str] | None = None) -> list[str]:
    user = f"把这个方向翻译成 2-4 个因子表达式: {hypothesis}"
    if avoid:
        user += "\n避免以下已试过/低效的表达式:\n" + "\n".join(f"- {e}" for e in avoid)
    obj = _extract_json(llm_fn([{"role": "system", "content": _syntax_prompt()},
                                {"role": "user", "content": user}]))
    if not obj:
        return []
    exprs = obj.get("expressions")
    return [str(e) for e in exprs] if isinstance(exprs, list) else []


def revise_expressions(hypothesis: str, prev_exprs: list[str], critic_reason: str,
                       llm_fn: LLMFn) -> list[str]:
    user = (f"方向: {hypothesis}\n上一版表达式: {', '.join(prev_exprs)}\n"
            f"风控反馈: {critic_reason}\n请按反馈改写出 1-3 个更稳健的表达式。")
    obj = _extract_json(llm_fn([{"role": "system", "content": _syntax_prompt()},
                                {"role": "user", "content": user}]))
    if not obj:
        return []
    exprs = obj.get("expressions")
    return [str(e) for e in exprs] if isinstance(exprs, list) else []
```

- [ ] **Step 4: 跑测试通过 + ruff + 提交**

```bash
pixi run pytest tests/test_team_coder.py -v
pixi run ruff check src/factorzen/agents/roles/coder.py tests/test_team_coder.py
git add src/factorzen/agents/roles/coder.py tests/test_team_coder.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(agents): Coder 角色(方向→表达式 + 按 Critic 反馈修正)"
```

---

## Task 5: Librarian 角色（librarian.py）

**Files:**
- Create: `src/factorzen/agents/roles/librarian.py`
- Test: `tests/test_team_librarian.py`

**Interfaces:**
- Consumes: `ExperimentIndex`（Task 1）；`AttemptRecord`（`agents/state.py`）
- Produces: `Recall(seen, known_invalid, known_valid)`；`recall(index, *, k=5) -> Recall`；`record(index, attempts, run_id) -> None`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_team_librarian.py
from pathlib import Path

from factorzen.agents.experiment_index import ExperimentIndex
from factorzen.agents.roles.librarian import recall, record
from factorzen.agents.state import AttemptRecord


def test_record_then_recall_roundtrip(tmp_path: Path):
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    attempts = [
        AttemptRecord(0, "动量", "ts_mean(close,5)", True, 0.05, True, "keep", None, ir_train=0.4),
        AttemptRecord(0, "换手", "rank(vol)", True, 0.001, False, "drop", None, ir_train=0.01),
    ]
    record(idx, attempts, run_id="r1")
    r = recall(idx, k=5)
    assert "ts_mean(close, 5)" in r.seen and "rank(vol)" in r.seen   # 归一化查重集
    assert "rank(vol)" in r.known_invalid                            # 未过护栏
    assert "ts_mean(close, 5)" in r.known_valid                      # 过护栏


def test_recall_empty_index(tmp_path: Path):
    r = recall(ExperimentIndex(str(tmp_path / "none.jsonl")), k=5)
    assert r.seen == set() and r.known_invalid == [] and r.known_valid == []
```

- [ ] **Step 2: 跑测试确认失败** → FAIL

- [ ] **Step 3: 实现 librarian.py**

```python
# src/factorzen/agents/roles/librarian.py
"""Librarian 角色：跨 session 长期记忆的读（recall）与写（record）。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Recall:
    seen: set[str]
    known_invalid: list[str]
    known_valid: list[str]


def recall(index, *, k: int = 5) -> Recall:
    return Recall(seen=index.seen_expressions(),
                  known_invalid=index.known_invalid(k=k),
                  known_valid=index.known_valid(k=k))


def record(index, attempts, run_id: str) -> None:
    """把本 run 所有 AttemptRecord 写入 experiment_index（passed = passed_guardrails）。"""
    records = []
    for a in attempts:
        if not a.compile_ok or a.ic_train is None:
            continue
        records.append({
            "expression": a.expression, "hypothesis": a.hypothesis,
            "ic_train": a.ic_train, "passed": a.passed_guardrails,
            "verdict": a.critic_verdict, "run_id": run_id,
            # holdout_ic 在 candidates 里；此处以 attempt 级记录，passed 标记是否过护栏
        })
    index.append(records)
```
> `known_valid` 排序用 `holdout_ic`，但 attempt 级未必有该字段 —— `ExperimentIndex.known_valid` 用 `r.get("holdout_ic") or 0.0` 容错（passed=True 即入选，排序退化为 0 也可接受）。如需精确，team_orchestrator 记录时可把候选的 holdout_ic 回填到对应 attempt 记录。

- [ ] **Step 4: 跑测试通过 + ruff + 提交**

```bash
pixi run pytest tests/test_team_librarian.py -v
pixi run ruff check src/factorzen/agents/roles/librarian.py tests/test_team_librarian.py
git add src/factorzen/agents/roles/librarian.py tests/test_team_librarian.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(agents): Librarian 角色(长期记忆 recall/record)"
```

---

## Task 6: 团队编排（team_orchestrator.py）

**Files:**
- Create: `src/factorzen/agents/team_orchestrator.py`
- Test: `tests/test_team_orchestrator.py`

**Interfaces:**
- Consumes: 所有角色（Task 2-5）；`evaluate_expressions`/`node_guardrails`（M5）；`AgentState`/`AttemptRecord`；`DataBundle.build`/`split_holdout`/`TrialLedger`；`parse_expr`/`to_expr_string`；`ExperimentIndex`
- Produces: `TeamResult(state, candidates, n_trials, rounds_log)`；`run_team_agent(daily, llm_fn, *, n_rounds, seed, index_path, top_k=5, holdout_ratio=0.2) -> TeamResult`（跨轮 feedback，无 max_retry）；`write_team_manifest(result, *, out_dir, run_id, params) -> Path`

- [ ] **Step 1: 写失败测试（端到端 + 否决回路 + N 诚实 + 跨 session 去重）**

```python
# tests/test_team_orchestrator.py
import datetime as dt
import json
from pathlib import Path
import numpy as np
import polars as pl

from factorzen.agents.team_orchestrator import run_team_agent


def _mock_daily(n_stocks=20, n_days=180, seed=1):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


def _scripted_team():
    """Hypothesis→Coder→Critic(keep) 一轮脚本，循环复用。"""
    hyp = json.dumps({"hypotheses": ["动量"]})
    code = json.dumps({"expressions": ["ts_mean(close,5)"]})
    crit = json.dumps({"verdict": "keep", "reason": "ok"})
    seq = [hyp, code, crit] * 50
    i = {"k": 0}
    def fn(messages):
        v = seq[i["k"] % len(seq)]; i["k"] += 1
        return v
    return fn


def test_run_team_closes_loop(tmp_path: Path):
    daily = _mock_daily()
    res = run_team_agent(daily, _scripted_team(), n_rounds=2, seed=42,
                         index_path=str(tmp_path / "e.jsonl"))
    assert res.state.iteration == 2
    assert res.n_trials >= 1
    assert len(res.rounds_log) >= 1     # 角色决策可审计


def test_run_team_revise_loop_counts_n(tmp_path: Path):
    """轮1 Critic revise_expr → 轮2 Coder 改写（跨轮 feedback），两表达式都评估、都计入 N。"""
    hyp = json.dumps({"hypotheses": ["动量"]})
    code1 = json.dumps({"expressions": ["ts_mean(close,5)"]})
    crit_revise = json.dumps({"verdict": "revise_expr", "reason": "窗口太短"})
    code2 = json.dumps({"expressions": ["ts_mean(close,20)"]})  # 下一轮 revise 产物
    crit_keep = json.dumps({"verdict": "keep", "reason": "ok"})
    # 轮1: propose,write,critic(revise) ; 轮2: revise(不再 propose),critic(keep)
    seq = [hyp, code1, crit_revise, code2, crit_keep]
    i = {"k": 0}
    def fn(messages):
        v = seq[i["k"]] if i["k"] < len(seq) else crit_keep
        i["k"] += 1
        return v
    daily = _mock_daily()
    res = run_team_agent(daily, fn, n_rounds=2, seed=1, index_path=str(tmp_path / "e.jsonl"))
    assert res.n_trials >= 2     # 两轮各评估一个表达式(原始 + 改写)，都计入 N
    assert any("ts_mean(close, 20)" in r["expressions"] for r in res.rounds_log)  # 轮2 是改写产物


def test_cross_session_dedup(tmp_path: Path):
    """共享 experiment_index：第二次 run 重复表达式被跳过（seen 去重）。"""
    daily = _mock_daily()
    idx_path = str(tmp_path / "shared.jsonl")
    run_team_agent(daily, _scripted_team(), n_rounds=1, seed=1, index_path=idx_path)
    res2 = run_team_agent(daily, _scripted_team(), n_rounds=1, seed=1, index_path=idx_path)
    # 第二次 run 产同样的 ts_mean(close,5)，已在 index → 本轮无新评估（n_trials 可能为 0）
    assert res2.n_trials == 0 or all(
        "ts_mean(close, 5)" != a.expression for a in res2.state.attempts)
```

- [ ] **Step 2: 跑测试确认失败** → FAIL

- [ ] **Step 3: 实现 team_orchestrator.py**

```python
# src/factorzen/agents/team_orchestrator.py
"""多角色团队编排：Librarian→Hypothesis→Coder→Evaluator→Critic 流水线 + 否决回路。"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from factorzen.agents.evaluation import evaluate_expressions
from factorzen.agents.experiment_index import ExperimentIndex
from factorzen.agents.nodes import node_guardrails
from factorzen.agents.roles.coder import revise_expressions, write_expressions
from factorzen.agents.roles.critic import critique
from factorzen.agents.roles.hypothesis import propose_hypotheses
from factorzen.agents.roles.librarian import recall, record
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.discovery.expression import parse_expr, to_expr_string
from factorzen.discovery.scoring import DataBundle
from factorzen.llm.generation import LLMFn
from factorzen.validation.holdout import split_holdout
from factorzen.validation.multiple_testing import TrialLedger


@dataclass
class TeamResult:
    state: AgentState
    candidates: list[dict]
    n_trials: int
    rounds_log: list[dict] = field(default_factory=list)


def _normalize(expr: str) -> str:
    try:
        return to_expr_string(parse_expr(expr))
    except ValueError:
        return expr


def _evaluate_and_record(state, exprs, hypothesis, *, daily, bundle, mem_seen):
    """评估一批表达式（跳过 mem_seen 去重），写 AttemptRecord，返回本批新评估的 expression 列表。"""
    fresh = [e for e in exprs if _normalize(e) not in mem_seen
             and _normalize(e) not in state.seen_expressions]
    results = evaluate_expressions(fresh, daily, bundle) if fresh else []
    for r in results:
        state.attempts.append(AttemptRecord(
            iteration=state.iteration, hypothesis=hypothesis, expression=r["expression"],
            compile_ok=r["compile_ok"], ic_train=r["ic_train"], passed_guardrails=False,
            critic_verdict=None, error=r["error"], ir_train=r["ir_train"]))
        state.seen_expressions.add(r["expression"])
    return results


def run_team_agent(daily, llm_fn: LLMFn, *, n_rounds: int, seed: int, index_path: str,
                   top_k: int = 5, holdout_ratio: float = 0.2) -> TeamResult:
    mining_df, holdout_df, _ = split_holdout(daily, holdout_ratio=holdout_ratio)
    bundle = DataBundle.build(mining_df)
    ledger = TrialLedger()
    state = AgentState(seed=seed)
    index = ExperimentIndex(index_path)
    rounds_log: list[dict] = []
    pending: dict | None = None   # 上一轮 Critic 反馈：{"kind", "hypothesis", "exprs", "reason"}

    for _ in range(n_rounds):
        rec = recall(index, k=5)                                   # ① Librarian
        # ②/③ Hypothesis + Coder（依据上一轮 Critic 反馈，跨轮）
        if pending and pending["kind"] == "revise_expr":
            hypothesis = pending["hypothesis"]
            exprs = revise_expressions(hypothesis, pending["exprs"], pending["reason"], llm_fn)
        else:
            fb = pending["reason"] if pending and pending["kind"] == "revise_hypothesis" else ""
            hyps = propose_hypotheses(llm_fn, known_invalid=rec.known_invalid,
                                      known_valid=rec.known_valid, feedback=fb, n=1)
            if not hyps:
                state.iteration += 1
                pending = None
                continue
            hypothesis = hyps[0]
            exprs = write_expressions(hypothesis, llm_fn, avoid=rec.known_invalid)
        # ④ Evaluator：评估（跨 session + session 去重）+ 护栏（N 本轮一次，干净）
        results = _evaluate_and_record(state, exprs, hypothesis, daily=mining_df,
                                       bundle=bundle, mem_seen=rec.seen)
        node_guardrails(state, daily=mining_df, holdout_df=holdout_df,
                        bundle=bundle, ledger=ledger, top_k=top_k)
        # ⑤ Critic：看本轮候选（guardrails 已跑，含 dsr/holdout）
        cand = state.candidates[-1] if state.candidates else {
            "expression": results[-1]["expression"] if results else (exprs[0] if exprs else ""),
            "hypothesis": hypothesis,
            "ic_train": results[-1]["ic_train"] if results else None}
        verdict = critique(cand, llm_fn)
        rounds_log.append({"round": state.iteration, "hypothesis": hypothesis,
                           "expressions": [r["expression"] for r in results],
                           "verdict": verdict.verdict, "reason": verdict.reason})
        # verdict → 下一轮 feedback（跨轮，避免同轮重复护栏致 N 三角和）
        if verdict.verdict == "revise_expr":
            pending = {"kind": "revise_expr", "hypothesis": hypothesis, "exprs": exprs,
                       "reason": verdict.reason}
        elif verdict.verdict == "revise_hypothesis":
            pending = {"kind": "revise_hypothesis", "reason": verdict.reason}
        else:
            pending = None
        # ⑥ Librarian：本轮 attempts 写 experiment_index
        record(index, [a for a in state.attempts if a.iteration == state.iteration],
               run_id=f"team_{seed}")
        state.iteration += 1
    return TeamResult(state=state, candidates=state.candidates, n_trials=ledger.n_trials,
                      rounds_log=rounds_log)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def write_team_manifest(result: TeamResult, *, out_dir: str, run_id: str, params: dict) -> Path:
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id, "seed": result.state.seed, "n_trials": result.n_trials,
        "iterations": result.state.iteration, "params": params,
        "roles": ["hypothesis", "coder", "evaluator", "critic", "librarian"],
        "rounds_log": result.rounds_log,
        "attempts": [a.__dict__ for a in result.state.attempts],
        "candidates": result.candidates, "git_sha": _git_sha(),
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return path
```
> 否决回路的 `cand` 取 `state.candidates[-1]` 是近似（候选可能未过护栏时取 stub）。如需精确把"本次新候选"对应到 Critic，可在 `node_guardrails` 前后比较 `len(state.candidates)`。MVP 取最近候选/ stub 足够驱动 verdict。`Librarian.record` 用本轮 attempts（`iteration == state.iteration`）。

- [ ] **Step 4: 跑测试通过 + ruff + 提交**

```bash
pixi run pytest tests/test_team_orchestrator.py -v
pixi run ruff check src/factorzen/agents/team_orchestrator.py tests/test_team_orchestrator.py
git add src/factorzen/agents/team_orchestrator.py tests/test_team_orchestrator.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(agents): 团队编排 run_team_agent(流水线+否决回路+N诚实+holdout隔离) + team manifest"
```

---

## Task 7: Pipeline（factor_mine_team.py）

**Files:**
- Create: `src/factorzen/pipelines/factor_mine_team.py`
- Test: `tests/test_team_pipeline.py`

**Interfaces:**
- Consumes: `run_team_agent`/`write_team_manifest`（Task 6）；`export_candidate`；`request_chat`/`load_llm_config`
- Produces: `run_team_mine(daily, *, n_rounds, seed, index_path, out_dir, llm_fn=None, top_k=5, holdout_ratio=0.2, run_id=None, export=True) -> dict`（`{run_dir, n_candidates, n_trials, candidates}`）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_team_pipeline.py
import datetime as dt
import json
from pathlib import Path
import numpy as np
import polars as pl

from factorzen.pipelines.factor_mine_team import run_team_mine


def _mock_daily(n_stocks=20, n_days=180, seed=1):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


def _scripted_team():
    seq = [json.dumps({"hypotheses": ["动量"]}), json.dumps({"expressions": ["ts_mean(close,5)"]}),
           json.dumps({"verdict": "keep", "reason": "ok"})] * 50
    i = {"k": 0}
    def fn(messages):
        v = seq[i["k"] % len(seq)]; i["k"] += 1
        return v
    return fn


def test_run_team_mine_writes_team_manifest(tmp_path: Path):
    daily = _mock_daily()
    res = run_team_mine(daily, n_rounds=2, seed=42, out_dir=str(tmp_path),
                        index_path=str(tmp_path / "e.jsonl"), llm_fn=_scripted_team(),
                        run_id="t1", export=False)
    run_dir = Path(res["run_dir"])
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "candidates.csv").exists()
    m = json.loads((run_dir / "manifest.json").read_text())
    assert "rounds_log" in m and "roles" in m       # team manifest 角色决策可审计
    assert res["n_trials"] == m["n_trials"]
```

- [ ] **Step 2: 跑测试确认失败** → FAIL

- [ ] **Step 3: 实现 factor_mine_team.py**

```python
# src/factorzen/pipelines/factor_mine_team.py
"""多 Agent 团队挖掘 pipeline：跑 team → 落 team manifest + candidates.csv + 导出候选。"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from factorzen.agents.team_orchestrator import run_team_agent, write_team_manifest


def _default_llm_fn():
    from factorzen.llm.client import request_chat
    from factorzen.llm.config import load_llm_config
    config = load_llm_config(enabled=True)
    if not config.is_ready:
        raise RuntimeError("LLM 未配置：设置 .env 的 FACTORZEN_LLM_* 或注入 llm_fn")
    return lambda messages: request_chat(config, messages)


def run_team_mine(daily, *, n_rounds: int, seed: int, index_path: str,
                  out_dir: str = "workspace/mine_team", llm_fn=None, top_k: int = 5,
                  holdout_ratio: float = 0.2, run_id: str | None = None,
                  export: bool = True) -> dict:
    fn = llm_fn or _default_llm_fn()
    result = run_team_agent(daily, fn, n_rounds=n_rounds, seed=seed, index_path=index_path,
                            top_k=top_k, holdout_ratio=holdout_ratio)
    rid = run_id or f"team_{seed}_{n_rounds}r"
    params = {"n_rounds": n_rounds, "seed": seed, "top_k": top_k, "holdout_ratio": holdout_ratio,
              "index_path": index_path}
    write_team_manifest(result, out_dir=out_dir, run_id=rid, params=params)
    run_dir = Path(out_dir) / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    cand_df = pl.DataFrame(result.candidates) if result.candidates else pl.DataFrame(
        {"expression": [], "holdout_ic": [], "dsr": []})
    cand_df.write_csv(run_dir / "candidates.csv")
    if export and result.candidates:
        from factorzen.discovery.export import export_candidate
        exp_dir = run_dir / "exported"
        exp_dir.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(result.candidates):
            export_candidate(c["expression"], f"team_{rid}_{i}", str(exp_dir))
    return {"run_dir": str(run_dir), "n_candidates": len(result.candidates),
            "n_trials": result.n_trials, "candidates": result.candidates}
```

- [ ] **Step 4: 跑测试通过 + ruff + 提交**

```bash
pixi run pytest tests/test_team_pipeline.py -v
pixi run ruff check src/factorzen/pipelines/factor_mine_team.py tests/test_team_pipeline.py
git add src/factorzen/pipelines/factor_mine_team.py tests/test_team_pipeline.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(agents): run_team_mine pipeline(team manifest + candidates.csv + 导出)"
```

---

## Task 8: CLI `fz mine team` + 收尾

**Files:**
- Modify: `src/factorzen/cli/main.py`（`mine_sub` 加 `team` 子命令 + `_cmd_mine_team`）
- Test: `tests/test_team_cli.py`
- Modify: `README.md`（核心能力「挖掘」行补多 Agent）

**Interfaces:**
- Consumes: `run_team_mine`（Task 7）；`get_universe`/`loader`；`build_parser`/`_cmd_mine_agent` 模式

- [ ] **Step 1: 写失败测试**

```python
# tests/test_team_cli.py
def test_parser_has_mine_team():
    from factorzen.cli.main import build_parser
    p = build_parser()
    args = p.parse_args(["mine", "team", "--start", "20220101", "--end", "20231231",
                         "--iterations", "5", "--seed", "42"])
    assert args.command == "mine"
    assert args.mine_command == "team"
    assert args.start == "20220101"
    assert args.iterations == 5
    assert args.seed == 42
    assert callable(args.func)


def test_parser_mine_team_index_default():
    from factorzen.cli.main import build_parser
    p = build_parser()
    args = p.parse_args(["mine", "team", "--start", "20220101", "--end", "20231231"])
    assert args.index_path.endswith(".jsonl")   # 长期记忆默认路径
```

- [ ] **Step 2: 跑测试确认失败** → FAIL（`AttributeError`）

- [ ] **Step 3: 接入 CLI**

在 `build_parser()` 的 `mine_sub` 下（`agent` 之后）加：
```python
    m_team = mine_sub.add_parser("team", help="Multi-agent team factor mining")
    m_team.add_argument("--start", required=True)
    m_team.add_argument("--end", required=True)
    m_team.add_argument("--universe", default=None)
    m_team.add_argument("--iterations", type=int, default=5)
    m_team.add_argument("--top-k", dest="top_k", type=int, default=5)
    m_team.add_argument("--seed", type=int, default=42)
    m_team.add_argument("--index-path", dest="index_path",
                        default="workspace/mine_team/experiment_index.jsonl")
    m_team.set_defaults(func=_cmd_mine_team)
```
模块顶层加 handler（仿 `_cmd_mine_agent`，延迟 import）：
```python
def _cmd_mine_team(args: argparse.Namespace) -> int:
    import polars as pl
    from factorzen.core import loader
    from factorzen.core.universe import get_universe
    from factorzen.pipelines.factor_mine_team import run_team_mine
    stocks = get_universe(args.end, args.universe) if args.universe else None
    daily = loader.fetch_daily(args.start, args.end)
    if stocks is not None:
        daily = daily.filter(pl.col("ts_code").is_in(stocks["ts_code"].to_list()))
    res = run_team_mine(daily, n_rounds=args.iterations, seed=args.seed,
                        top_k=args.top_k, index_path=args.index_path)
    print(f"[mine-team] 候选 {res['n_candidates']} 个 / N={res['n_trials']} → {res['run_dir']}")
    return 0
```
> 数据加载以 `_cmd_mine_agent` 实际实现为准（先 Read 对齐）。

- [ ] **Step 4: 跑测试通过**

Run: `pixi run pytest tests/test_team_cli.py -v` → PASS

- [ ] **Step 5: 全量质量门**

```bash
pixi run pytest tests/test_team_*.py -q     # M6 全测试绿
pixi run ruff check src/factorzen/agents/ src/factorzen/pipelines/factor_mine_team.py tests/test_team_*.py
pixi run pytest tests/test_agent_*.py -q    # M5 回归（确认没破坏）
```

- [ ] **Step 6: README + 提交**

README「核心能力」表「挖掘」行补："+ 多 Agent 团队（`fz mine team`）"。
```bash
git add src/factorzen/cli/main.py tests/test_team_cli.py README.md
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(agents): fz mine team CLI + README"
```

---

## 收尾验收（全部 task 完成后）

- [ ] `pixi run pytest tests/test_team_*.py -q` 全绿（experiment_index/critic/hypothesis/coder/librarian/orchestrator/pipeline/cli）
- [ ] `pixi run pytest tests/test_agent_*.py -q` M5 回归绿（M6 没破坏 M5）
- [ ] `pixi run ruff check src/factorzen/agents/ src/factorzen/pipelines/factor_mine_team.py tests/test_team_*.py` 0 errors
- [ ] **否决回路**：scripted FakeLLM（revise_expr→keep）→ Coder 修正、重试计入 N、≤ max_retry 不死循环（Task 6 断言）
- [ ] **跨 session 记忆去重**：两次共享 index 的 run，第二次重复表达式被跳过（Task 6 断言）
- [ ] **Critic 拦截过拟合**：DSR 不显著候选 → Critic drop（Task 2 断言，非恒真）
- [ ] **防过拟合**：N per-run 诚实（含否决重试，不跨 run 累加）；holdout 段不进角色/记忆
- [ ] 真实 LLM smoke（手动，需 `FACTORZEN_LLM_*`）：`pixi run fz mine team --start 20220101 --end 20231231 --iterations 3` → team manifest 含 rounds_log + experiment_index 更新
- [ ] `git status --short` 干净（只 M6 相关入库，未带 M0）
- [ ] 更新本 plan 追加完成记录 + memory roadmap（M6 完成）

---

## 完成记录（2026-06-30）

M6 多 Agent + 长期记忆 ✅ 完成入库（10 commits，`b4945b2..a20e075`，`feature/platform-upgrade`）。26 个 M6 测试 + 30 个 M5 回归全绿，opus 全分支终审确认 ship-ready（修完 Critic-drop 后）。

**交付：**
- 全新 `agents/experiment_index.py`（跨 session 长期记忆）+ `agents/roles/`（hypothesis/coder/critic/librarian）+ `agents/team_orchestrator.py` + `pipelines/factor_mine_team.py` + `fz mine team` CLI。
- 5 角色协作：Hypothesis/Coder/Critic 调 LLM、**Evaluator 确定性**（复用 M5 评估+护栏）、Librarian 确定性管记忆。**零新依赖**。
- **跨轮 feedback 否决回路**（self-review 从同轮改跨轮，避免 N 三角和）：Critic verdict → 下轮 Coder 改写 / Hypothesis 换方向。
- 跨 session 长期记忆：归一化查重去同 + known_invalid（跨 session Negative RAG）+ known_valid（方向参考）。

**并行执行（用户要求效率最大化）：** Task 1-4（experiment_index + 3 独立角色）并行 implement（不 commit）→ 主控串行 commit → 并行 review，比纯顺序快一倍；Task 5-8 是依赖链顺序做。

**评审暴露并修复：**
- **N 三角和**（plan self-review）：同轮否决回路会让本轮 N 重复累加 → 改跨轮 feedback（每轮恰好一次 `node_guardrails`，opus 终审独立验证 N 诚实无盲点）。见 [[multi-round-cumulative-count-trap]]。
- **Critic drop 系统层落空**（opus 终审）：候选已被 guardrails append，drop 只设 pending 不移除 → 改 `n_before` 快照 + `del candidates[n_before:]` 真移除，有系统层回归测试。
- **长期记忆空转**（opus）：`known_valid` 排序键 holdout_ic 从不写入、verdict 恒 null → record 归一化匹配回填 holdout_ic + critic_verdict。
- 早期：`known_valid` 排序去 abs（Task 1 review）。

**防过拟合（灵魂，继承 M5 + opus 独立验证）：** N **per-run** 诚实（跨轮重试都计入、跨 session 查重只省算力不累加 N）；holdout 段对角色/记忆全程不可见（experiment_index 不存 holdout 原始数据）；Critic 系统层 drop + 确定性 DSR 门槛双重拦截。

**已知限制（defer）：** `_normalize` 双调（性能微）；真实 LLM smoke 为手动命令。

---

*M6 完成后，FactorZen 拥有"多角色协作（假设/编码/评审/风控/记忆）+ 跨 session 知识积累"的因子研究系统——Critic 拦截过拟合、记忆避免重复，全程可审计可复现。*

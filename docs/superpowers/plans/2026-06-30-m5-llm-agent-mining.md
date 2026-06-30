# M5 · LLM 单 Agent 闭环挖掘 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 M1 挖掘 + M2 护栏之上加一个 LLM 驱动的闭环因子挖掘器：假设 → 生成表达式 → 编译/评估 → 护栏 → 反思迭代，全程可审计、可复现、CI 可测。

**Architecture:** 自建极简 agent loop——纯 Python `for` 循环 + `dataclass` State + 函数式节点 `node(State)→State` + 注入 `LLMFn`（不引 LangGraph/LangChain）。新建 `agents/` 模块 + 扩展 `llm/`，复用 `discovery/`（表达式编译/评估）与 `validation/`（防过拟合护栏）。

**Tech Stack:** Python 3.10–3.12 · polars · numpy · 现有 `llm/`（urllib，OpenAI 兼容）· pytest + FakeLLM · argparse。**零新第三方依赖。**

## Global Constraints

- **零新依赖**：不引入任何第三方包（不引 langgraph/langchain/pydantic-ai/dspy/faiss/embedding/sklearn）。
- **LLM 可注入**：`LLMFn = Callable[[list[dict[str, str]]], str]`。生产注入真实 `request_chat`，CI/测试注入 `FakeLLM`（确定性返回预设字符串序列）。**单测全程不触网。**
- **表达式 DSL**：LLM 产出表达式字符串 → `parse_expr(s)` 验证，`ValueError` = 非法直接拒绝（不需 sandbox）。
- **零回归**：`evaluate_expressions` 用**公开** `evaluate(node, df)` + `score_candidate`/`quick_fitness`，**不重构 `run_session`**（M1 已入库，36 测试保护）。
- **防过拟合（灵魂）**：`TrialLedger` 的 N **累加 Agent 所有轮、所有评估过的表达式**；holdout 段对 Agent 全程不可见（只见 mining 段）；候选报 holdout_ic/PBO/DSR/相关性/family。
- **可复现**：`numpy.random.default_rng(seed)`；同 seed + FakeLLM 逐字节复跑。
- **LLM 扩展**：新建 `request_chat(config, messages) -> str`（复用 `_build_payload` 逻辑，去 `response_format` 强制，返回原始 content）；**不改** `request_llm_explanation`。
- **文件命名**：pipeline 用 `pipelines/factor_mine_agent.py`（与现有 `factor_mine.py` 一致）。
- **环境**：`pixi run pytest` / `pixi run ruff check`；polars 1.41.2（`pl.len()` 非 `pl.count()`）。
- **提交**：conventional commits；作者 `rookiewu417 <1007372080@qq.com>`；每 task 只 `git add` 自己的文件（工作区有无关 M0 改动，**绝不** `-A`）。
- **测试判别力**：避免恒真——值断言、严格阈值、构造能让断言 FAIL 的反例（参见 M3 教训：守恒类断言当被验证量由分量构造时恒真，需跨函数/ground-truth 验证）。

---

## File Structure

| 文件 | 职责 | Task |
|---|---|---|
| `src/factorzen/agents/__init__.py` | 包标记 | 1 |
| `src/factorzen/agents/state.py` | `AgentState` / `AttemptRecord`（JSON 可序列化 dataclass） | 1 |
| `src/factorzen/llm/client.py`（改） | 新增 `request_chat(config, messages) -> str` | 2 |
| `src/factorzen/llm/generation.py` | `FactorProposal` + `generate_factor_proposal` + `semantic_check` + `build_agent_messages` | 2 |
| `src/factorzen/agents/memory.py` | session 记忆 + Negative RAG 平面召回 + family 并查集分组 | 3 |
| `src/factorzen/agents/evaluation.py` | `evaluate_expressions`（用公开 evaluate+score_candidate） | 4 |
| `src/factorzen/agents/nodes.py` | 七个 `node_*` 函数式节点 | 5（生成侧）+6（验收侧） |
| `src/factorzen/agents/orchestrator.py` | `run_llm_agent` 主循环 + N 累加 + holdout 隔离 + human-review | 7 |
| `src/factorzen/agents/manifest.py` | session manifest 落盘 | 8 |
| `src/factorzen/pipelines/factor_mine_agent.py` | `run_agent_mine`（拉数据 → Agent → 落产物 + export） | 9 |
| `src/factorzen/cli/main.py`（改） | `fz mine agent` 子命令 + handler | 10 |
| `tests/test_agent_*.py` | 各 task 测试（FakeLLM 离线） | 各 |

**共用测试夹具**：`FakeLLM` 在 Task 2 首次定义于 `tests/test_agent_generation.py`，后续测试各自构造（或从该文件 import）。每个测试文件自带所需 mock helper。

---

## Task 1: Agent 状态数据结构

**Files:**
- Create: `src/factorzen/agents/__init__.py`（空）, `src/factorzen/agents/state.py`
- Test: `tests/test_agent_state.py`

**Interfaces:**
- Produces: `AttemptRecord`（dataclass）；`AgentState`（dataclass，含 `to_dict()` JSON 可序列化）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_agent_state.py
from factorzen.agents.state import AgentState, AttemptRecord


def test_attempt_record_fields():
    r = AttemptRecord(iteration=1, hypothesis="低换手反转", expression="rank(close)",
                      compile_ok=True, ic_train=0.03, passed_guardrails=False,
                      critic_verdict=None, error=None)
    assert r.iteration == 1 and r.compile_ok is True


def test_agent_state_defaults_and_serializable():
    import json
    s = AgentState(seed=42)
    assert s.iteration == 0 and s.attempts == [] and s.candidates == []
    assert s.seen_expressions == set() and s.negative_examples == []
    s.attempts.append(AttemptRecord(iteration=0, hypothesis="h", expression="rank(close)",
                                    compile_ok=True, ic_train=0.05, passed_guardrails=True,
                                    critic_verdict="keep", error=None))
    s.seen_expressions.add("rank(close)")
    d = s.to_dict()  # set 转 list，dataclass 转 dict
    assert json.dumps(d)  # 不抛 = JSON 可序列化
    assert d["attempts"][0]["expression"] == "rank(close)"
    assert "rank(close)" in d["seen_expressions"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_agent_state.py -v`
Expected: FAIL（`ModuleNotFoundError: factorzen.agents.state`）

- [ ] **Step 3: 实现 state.py**

```python
# src/factorzen/agents/state.py
"""Agent 闭环的显式状态（JSON 可序列化 dataclass）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class AttemptRecord:
    iteration: int
    hypothesis: str
    expression: str
    compile_ok: bool
    ic_train: float | None
    passed_guardrails: bool
    critic_verdict: str | None
    error: str | None


@dataclass
class AgentState:
    seed: int
    iteration: int = 0
    attempts: list[AttemptRecord] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    seen_expressions: set[str] = field(default_factory=set)   # session 记忆
    negative_examples: list[str] = field(default_factory=list)  # Negative RAG

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "iteration": self.iteration,
            "attempts": [asdict(a) for a in self.attempts],
            "candidates": self.candidates,
            "seen_expressions": sorted(self.seen_expressions),
            "negative_examples": self.negative_examples,
        }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_agent_state.py -v`
Expected: PASS

- [ ] **Step 5: ruff + 提交**

```bash
pixi run ruff check src/factorzen/agents/ tests/test_agent_state.py
git add src/factorzen/agents/__init__.py src/factorzen/agents/state.py tests/test_agent_state.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(agents): Agent 闭环状态数据结构 AgentState/AttemptRecord"
```

---

## Task 2: LLM 生成层（request_chat + generation）

**Files:**
- Modify: `src/factorzen/llm/client.py`（新增 `request_chat`）
- Create: `src/factorzen/llm/generation.py`
- Test: `tests/test_agent_generation.py`

**Interfaces:**
- Consumes: `LLMConfig`（`llm/config.py`，字段 enabled/base_url/api_key/model/temperature/max_tokens；`is_ready`/`chat_completions_url`）；`parse_llm_explanation` 的容错风格
- Produces: `request_chat(config, messages) -> str`；`FactorProposal(hypothesis, expressions, rationale)`；`generate_factor_proposal(messages, llm_fn) -> list[FactorProposal]`；`semantic_check(hypothesis, expression, llm_fn) -> tuple[bool, str]`；`build_agent_messages(op_names, leaf_names, feedback, negatives) -> list[dict]`；`LLMFn = Callable[[list[dict[str,str]]], str]`

- [ ] **Step 1: 写失败测试（FakeLLM + 解析容错）**

```python
# tests/test_agent_generation.py
import json

from factorzen.llm.generation import (
    FactorProposal, build_agent_messages, generate_factor_proposal, semantic_check,
)


class FakeLLM:
    """确定性 LLMFn：按调用顺序返回预设字符串。"""
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self._responses.pop(0) if self._responses else "{}"


def test_generate_factor_proposal_parses_json():
    raw = json.dumps({"hypothesis": "低换手反转",
                      "expressions": ["rank(close)", "ts_mean(vol,5)"],
                      "rationale": "..."})
    llm = FakeLLM([raw])
    props = generate_factor_proposal([{"role": "user", "content": "x"}], llm, n_hypotheses=1)
    assert len(props) == 1
    assert props[0].hypothesis == "低换手反转"
    assert props[0].expressions == ["rank(close)", "ts_mean(vol,5)"]


def test_generate_factor_proposal_tolerates_garbage():
    # 非 JSON → 返回空列表（降级，不抛）
    llm = FakeLLM(["这不是 JSON"])
    props = generate_factor_proposal([{"role": "user", "content": "x"}], llm)
    assert props == []


def test_generate_extracts_json_substring():
    # JSON 嵌在自然语言里 → 提取首个 {...}
    raw = '好的，这是我的提议：{"hypothesis":"h","expressions":["rank(close)"],"rationale":"r"} 完毕'
    llm = FakeLLM([raw])
    props = generate_factor_proposal([{"role": "user", "content": "x"}], llm)
    assert props and props[0].expressions == ["rank(close)"]


def test_semantic_check_yes_no():
    llm = FakeLLM([json.dumps({"consistent": True, "reason": "对齐"}),
                  json.dumps({"consistent": False, "reason": "表达式与假设无关"})])
    ok1, _ = semantic_check("动量", "ts_mean(close,20)", llm)
    ok2, reason2 = semantic_check("动量", "rank(pb)", llm)
    assert ok1 is True and ok2 is False and reason2


def test_build_agent_messages_lists_ops_and_leaves():
    msgs = build_agent_messages(op_names=["ts_mean", "rank", "div"],
                                leaf_names=["close", "vol", "pb"],
                                feedback="上轮 IC 偏低", negatives=["rank(close)"])
    blob = " ".join(m["content"] for m in msgs)
    assert "ts_mean" in blob and "close" in blob       # 算子/特征清单进 prompt
    assert "rank(close)" in blob                         # Negative RAG 负例进 prompt
    assert any(m["role"] == "system" for m in msgs)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_agent_generation.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 新增 `request_chat` 到 client.py**

在 `src/factorzen/llm/client.py` 末尾加（复用现有 `_build_payload` 思路，但返回原始 content）：
```python
def request_chat(config: LLMConfig, messages: list[dict[str, str]]) -> str:
    """通用 chat 请求：返回 choices[0].message.content 原始字符串。
    与 request_llm_explanation 的区别：不强制 response_format、不绑定 schema。"""
    import json
    import urllib.request

    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.thinking:
        payload["thinking"] = config.thinking
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        config.chat_completions_url, data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {config.api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=config.timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise LLMClientError(f"chat 请求失败: {exc}") from exc
```
> 注：若现有 `_build_payload` 已可参数化 `response_format`，优先复用它构造 payload，避免重复。以实际 client.py 结构为准（先 Read）。

- [ ] **Step 4: 实现 generation.py**

```python
# src/factorzen/llm/generation.py
"""LLM 因子生成层：假设 + 表达式提议 + 语义对齐自检 + prompt 模板。"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

LLMFn = Callable[[list[dict[str, str]]], str]


@dataclass
class FactorProposal:
    hypothesis: str
    expressions: list[str]
    rationale: str


def _extract_json(raw: str) -> dict | None:
    """容错解析：直接 json.loads；失败找首个 {...} 子串；再失败返回 None。"""
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:  # noqa: BLE001
            return None
    return None


def generate_factor_proposal(messages: list[dict[str, str]], llm_fn: LLMFn,
                             *, n_hypotheses: int = 1) -> list[FactorProposal]:
    """调用 LLM 生成 1+ 个 (假设, 表达式集)。解析失败的丢弃（降级不抛）。"""
    proposals: list[FactorProposal] = []
    for _ in range(max(1, n_hypotheses)):
        obj = _extract_json(llm_fn(messages))
        if not obj:
            continue
        exprs = obj.get("expressions")
        if not isinstance(exprs, list) or not exprs:
            continue
        proposals.append(FactorProposal(
            hypothesis=str(obj.get("hypothesis", "")),
            expressions=[str(e) for e in exprs],
            rationale=str(obj.get("rationale", "")),
        ))
    return proposals


def semantic_check(hypothesis: str, expression: str, llm_fn: LLMFn) -> tuple[bool, str]:
    """LLM 自查表达式是否实现假设。返回 (一致?, 理由)。解析失败 → (True, '') 放行（避免误杀）。"""
    msgs = [
        {"role": "system", "content": "你判断量化因子表达式是否实现了给定假设，只输出 JSON: "
                                       '{"consistent": true/false, "reason": "..."}'},
        {"role": "user", "content": f"假设: {hypothesis}\n表达式: {expression}"},
    ]
    obj = _extract_json(llm_fn(msgs))
    if not obj or "consistent" not in obj:
        return True, ""  # 解析失败放行，不误杀
    return bool(obj["consistent"]), str(obj.get("reason", ""))


def build_agent_messages(op_names: list[str], leaf_names: list[str],
                         feedback: str = "", negatives: list[str] | None = None) -> list[dict[str, str]]:
    """构造生成 prompt：算子/特征清单 + 上轮反馈 + Negative RAG 负例。"""
    neg = negatives or []
    system = (
        "你是量化研究员，提出有经济直觉的假设并翻译成因子表达式。\n"
        f"可用算子: {', '.join(op_names)}\n"
        f"可用特征(叶子): {', '.join(leaf_names)}\n"
        "时序算子最后一个参数是整型窗口，如 ts_mean(close, 20)。\n"
        '只输出 JSON: {"hypothesis": "...", "expressions": ["...", "..."], "rationale": "..."}'
    )
    user = "提出一个新假设并给出 2-4 个候选表达式。"
    if feedback:
        user += f"\n上一轮反馈: {feedback}"
    if neg:
        user += "\n避免以下已探索过/低效的模式:\n" + "\n".join(f"- {n}" for n in neg)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
```

- [ ] **Step 5: 跑测试确认通过 + ruff + 提交**

```bash
pixi run pytest tests/test_agent_generation.py -v
pixi run ruff check src/factorzen/llm/ tests/test_agent_generation.py
git add src/factorzen/llm/client.py src/factorzen/llm/generation.py tests/test_agent_generation.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(llm): request_chat + 因子生成层(假设/表达式/语义对齐/prompt)"
```

---

## Task 3: 记忆与多样性（memory.py）

**Files:**
- Create: `src/factorzen/agents/memory.py`
- Test: `tests/test_agent_memory.py`

**Interfaces:**
- Produces: `negative_recall(seen: list[tuple[str,float]], k=3, ic_threshold=0.0) -> list[str]`（按低 IC 召回负例）；`family_groups(corr_pairs: dict[tuple[str,str], float], names: list[str], threshold=0.7) -> list[set[str]]`（并查集分组）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_agent_memory.py
from factorzen.agents.memory import family_groups, negative_recall


def test_negative_recall_picks_low_ic():
    seen = [("rank(close)", 0.001), ("ts_mean(vol,5)", 0.08), ("div(close,open)", -0.002)]
    neg = negative_recall(seen, k=2, ic_threshold=0.01)
    # 只召回 IC < 阈值的，按 |IC| 升序（最没用的优先），不含高 IC 的
    assert "ts_mean(vol,5)" not in neg
    assert "rank(close)" in neg
    assert len(neg) <= 2


def test_negative_recall_empty_when_all_good():
    seen = [("a", 0.1), ("b", 0.2)]
    assert negative_recall(seen, ic_threshold=0.01) == []


def test_family_groups_union_find():
    names = ["f1", "f2", "f3", "f4"]
    # f1-f2 高相关, f3-f4 高相关, 两组互不相关
    pairs = {("f1", "f2"): 0.9, ("f1", "f3"): 0.1, ("f3", "f4"): 0.85, ("f2", "f3"): 0.2}
    groups = family_groups(pairs, names, threshold=0.7)
    # 应分成两族 {f1,f2} 和 {f3,f4}
    assert {frozenset(g) for g in groups} == {frozenset({"f1", "f2"}), frozenset({"f3", "f4"})}


def test_family_groups_all_singletons_when_low_corr():
    names = ["a", "b", "c"]
    pairs = {("a", "b"): 0.1, ("b", "c"): 0.2, ("a", "c"): 0.05}
    groups = family_groups(pairs, names, threshold=0.7)
    assert len(groups) == 3  # 全独立
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_agent_memory.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 memory.py**

```python
# src/factorzen/agents/memory.py
"""session 记忆：Negative RAG 平面召回 + family-aware 并查集分组（无向量库/无 sklearn）。"""
from __future__ import annotations


def negative_recall(seen: list[tuple[str, float]], *, k: int = 3,
                    ic_threshold: float = 0.0) -> list[str]:
    """从 (表达式, IC) 历史里召回低 IC 负例，供 Negative RAG 注入 prompt。
    只取 |IC| < threshold 的，按 |IC| 升序（最没用优先），最多 k 个。"""
    low = [(e, ic) for e, ic in seen if abs(ic) < ic_threshold]
    low.sort(key=lambda t: abs(t[1]))
    return [e for e, _ in low[:k]]


def family_groups(corr_pairs: dict[tuple[str, str], float], names: list[str],
                  *, threshold: float = 0.7) -> list[set[str]]:
    """按两两相关 > threshold 并查集分组（family-aware 多样性）。"""
    parent = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for (a, b), c in corr_pairs.items():
        if a in parent and b in parent and abs(c) > threshold:
            union(a, b)
    groups: dict[str, set[str]] = {}
    for n in names:
        groups.setdefault(find(n), set()).add(n)
    return list(groups.values())
```

- [ ] **Step 4: 跑测试确认通过 + ruff + 提交**

```bash
pixi run pytest tests/test_agent_memory.py -v
pixi run ruff check src/factorzen/agents/memory.py tests/test_agent_memory.py
git add src/factorzen/agents/memory.py tests/test_agent_memory.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(agents): Negative RAG 平面召回 + family-aware 并查集分组"
```

---

## Task 4: 表达式评估（evaluation.py，零回归复用）

**Files:**
- Create: `src/factorzen/agents/evaluation.py`
- Test: `tests/test_agent_evaluation.py`

**Interfaces:**
- Consumes: `parse_expr`/`to_expr_string`/`evaluate`（`discovery/expression.py`）；`DataBundle.build`/`quick_fitness`（`discovery/scoring.py`）
- Produces: `evaluate_expressions(expr_strs: list[str], daily: pl.DataFrame, bundle) -> list[dict]`（每项 `{expression, node, compile_ok, ic_train, ir_train, error}`）

- [ ] **Step 1: 写失败测试（mock daily）**

```python
# tests/test_agent_evaluation.py
import datetime as dt
import numpy as np
import polars as pl

from factorzen.agents.evaluation import evaluate_expressions
from factorzen.discovery.scoring import DataBundle


def _mock_daily(n_stocks=20, n_days=120, seed=1):
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
            rows.append({"trade_date": dd, "ts_code": c, "close": px,
                         "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


def test_evaluate_valid_expressions():
    daily = _mock_daily()
    bundle = DataBundle.build(daily)
    out = evaluate_expressions(["ts_mean(close,5)", "rank(vol)"], daily, bundle)
    assert len(out) == 2
    for r in out:
        assert r["compile_ok"] is True
        assert r["ic_train"] is not None        # 真算出了 IC（非 None）
        assert isinstance(r["ic_train"], float)


def test_evaluate_rejects_illegal_expression():
    daily = _mock_daily()
    bundle = DataBundle.build(daily)
    out = evaluate_expressions(["this_is_not_an_operator(close)", "ts_mean(close,5)"], daily, bundle)
    assert out[0]["compile_ok"] is False and out[0]["error"]   # 非法被拒，记错误
    assert out[0]["ic_train"] is None
    assert out[1]["compile_ok"] is True                         # 合法的照常评估
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_agent_evaluation.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 evaluation.py**

```python
# src/factorzen/agents/evaluation.py
"""把 LLM 产出的表达式字符串批量评估为 Rank IC/IR。
全部用 discovery 的公开接口，不重构 run_session（零回归）。"""
from __future__ import annotations

import polars as pl

from factorzen.discovery.expression import evaluate as eval_node
from factorzen.discovery.expression import parse_expr, to_expr_string
from factorzen.discovery.scoring import quick_fitness


def _node_to_factor_df(node, daily: pl.DataFrame) -> pl.DataFrame:
    """用公开 evaluate(node, df) 算因子值，组装成 [trade_date, ts_code, factor_value]。"""
    series = eval_node(node, daily.sort(["ts_code", "trade_date"]))
    return daily.sort(["ts_code", "trade_date"]).select(["trade_date", "ts_code"]).with_columns(
        series.alias("factor_value"))


def evaluate_expressions(expr_strs: list[str], daily: pl.DataFrame, bundle) -> list[dict]:
    """批量评估表达式集。非法表达式（parse_expr 抛 ValueError）记 compile_ok=False。"""
    results: list[dict] = []
    for s in expr_strs:
        try:
            node = parse_expr(s)
        except ValueError as exc:
            results.append({"expression": s, "node": None, "compile_ok": False,
                            "ic_train": None, "ir_train": None, "error": str(exc)})
            continue
        try:
            fdf = _node_to_factor_df(node, daily)
            fit = quick_fitness(fdf, bundle, segment="train")
            results.append({"expression": to_expr_string(node), "node": node,
                            "compile_ok": True, "ic_train": float(fit["ic_mean"]),
                            "ir_train": float(fit["ir"]), "error": None})
        except Exception as exc:  # noqa: BLE001
            results.append({"expression": s, "node": node, "compile_ok": False,
                            "ic_train": None, "ir_train": None, "error": str(exc)})
    return results
```
> 注：`_node_to_factor_df` 的列组装需与 `quick_fitness` 期望的 `[trade_date, ts_code, factor_value]` 一致；以 `scoring.py` 实际期望为准（先 Read `quick_fitness` 与 `_factor_values`）。

- [ ] **Step 4: 跑测试确认通过 + ruff + 提交**

```bash
pixi run pytest tests/test_agent_evaluation.py -v
pixi run ruff check src/factorzen/agents/evaluation.py tests/test_agent_evaluation.py
git add src/factorzen/agents/evaluation.py tests/test_agent_evaluation.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(agents): evaluate_expressions 批量评估(复用公开接口,零回归)"
```

---

## Task 5: 闭环节点 · 生成侧（nodes.py 前四步）

**Files:**
- Create: `src/factorzen/agents/nodes.py`
- Test: `tests/test_agent_nodes_gen.py`

**Interfaces:**
- Consumes: `AgentState`/`AttemptRecord`（Task 1）；`generate_factor_proposal`/`semantic_check`/`build_agent_messages`（Task 2）；`evaluate_expressions`（Task 4）；`OPERATORS`/`LEAF_FEATURES`（`discovery/operators.py`）
- Produces: `node_generate(state, llm_fn, *, daily, bundle) -> AgentState`；`node_evaluate(state, daily, bundle) -> AgentState`（compile + semantic 已融入 generate→evaluate 流程）；`AgentContext`（dataclass，持 daily/bundle/op_names/leaf_names）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_agent_nodes_gen.py
import datetime as dt
import json
import numpy as np
import polars as pl

from factorzen.agents.nodes import node_evaluate, node_generate
from factorzen.agents.state import AgentState
from factorzen.discovery.scoring import DataBundle


class FakeLLM:
    def __init__(self, responses):
        self._r = list(responses)
    def __call__(self, messages):
        return self._r.pop(0) if self._r else "{}"


def _mock_daily(n_stocks=20, n_days=120, seed=1):
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


def test_node_generate_then_evaluate_populates_attempts():
    daily = _mock_daily()
    bundle = DataBundle.build(daily)
    raw = json.dumps({"hypothesis": "动量", "expressions": ["ts_mean(close,5)", "rank(vol)"],
                      "rationale": "r"})
    # semantic_check 也走 llm：两次 consistent=true
    sem = json.dumps({"consistent": True, "reason": "ok"})
    llm = FakeLLM([raw, sem, sem])
    state = AgentState(seed=42)
    state = node_generate(state, llm, daily=daily, bundle=bundle)
    state = node_evaluate(state, daily=daily, bundle=bundle)
    assert len(state.attempts) == 2
    assert all(a.compile_ok for a in state.attempts)
    assert all(a.ic_train is not None for a in state.attempts)
    assert "ts_mean(close, 5)" in state.seen_expressions or "ts_mean(close,5)" in state.seen_expressions


def test_node_generate_rejects_illegal_and_records_error():
    daily = _mock_daily()
    bundle = DataBundle.build(daily)
    raw = json.dumps({"hypothesis": "h", "expressions": ["bogus_op(close)"], "rationale": "r"})
    sem = json.dumps({"consistent": True, "reason": "ok"})
    llm = FakeLLM([raw, sem])
    state = AgentState(seed=1)
    state = node_generate(state, llm, daily=daily, bundle=bundle)
    state = node_evaluate(state, daily=daily, bundle=bundle)
    assert state.attempts[0].compile_ok is False and state.attempts[0].error
```

- [ ] **Step 2: 跑测试确认失败** → `pixi run pytest tests/test_agent_nodes_gen.py -v` → FAIL

- [ ] **Step 3: 实现 nodes.py 生成侧**

```python
# src/factorzen/agents/nodes.py
"""Agent 闭环的函数式节点：node(State) -> State。"""
from __future__ import annotations

from dataclasses import dataclass, field

from factorzen.agents.evaluation import evaluate_expressions
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.discovery.operators import LEAF_FEATURES, OPERATORS
from factorzen.llm.generation import (
    LLMFn, build_agent_messages, generate_factor_proposal, semantic_check,
)


@dataclass
class _PendingExpr:
    hypothesis: str
    expression: str


@dataclass
class AgentContext:
    op_names: list[str] = field(default_factory=lambda: list(OPERATORS.keys()))
    leaf_names: list[str] = field(default_factory=lambda: list(LEAF_FEATURES.keys()))


def node_generate(state: AgentState, llm_fn: LLMFn, *, daily, bundle,
                  n_hypotheses: int = 1, feedback: str = "") -> AgentState:
    """生成假设+表达式 → 语义对齐自检 → 暂存待评估（compile/eval 在 node_evaluate）。"""
    ctx = AgentContext()
    msgs = build_agent_messages(ctx.op_names, ctx.leaf_names, feedback, state.negative_examples)
    proposals = generate_factor_proposal(msgs, llm_fn, n_hypotheses=n_hypotheses)
    pending: list[_PendingExpr] = []
    for p in proposals:
        for expr in p.expressions:
            if expr in state.seen_expressions:
                continue  # session 记忆去重
            ok, _reason = semantic_check(p.hypothesis, expr, llm_fn)
            if ok:
                pending.append(_PendingExpr(p.hypothesis, expr))
    state.__dict__.setdefault("_pending", [])
    state._pending = pending  # type: ignore[attr-defined]
    return state


def node_evaluate(state: AgentState, *, daily, bundle) -> AgentState:
    """对暂存表达式批量评估，写 AttemptRecord + 更新 seen。"""
    pending = getattr(state, "_pending", [])
    exprs = [p.expression for p in pending]
    results = evaluate_expressions(exprs, daily, bundle) if exprs else []
    for p, r in zip(pending, results):
        state.attempts.append(AttemptRecord(
            iteration=state.iteration, hypothesis=p.hypothesis, expression=r["expression"],
            compile_ok=r["compile_ok"], ic_train=r["ic_train"], passed_guardrails=False,
            critic_verdict=None, error=r["error"]))
        state.seen_expressions.add(r["expression"])
    state._pending = []  # type: ignore[attr-defined]
    return state
```
> `_pending` 用临时属性传递；如更喜欢显式，可在 `AgentState` 加 `pending: list = field(default_factory=list)` 字段（更干净，推荐）。

- [ ] **Step 4: 跑测试确认通过 + ruff + 提交**

```bash
pixi run pytest tests/test_agent_nodes_gen.py -v
pixi run ruff check src/factorzen/agents/nodes.py tests/test_agent_nodes_gen.py
git add src/factorzen/agents/nodes.py tests/test_agent_nodes_gen.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(agents): 闭环生成侧节点(生成/语义对齐/评估)"
```

---

## Task 6: 闭环节点 · 验收侧（guardrails + critic + reflect）

**Files:**
- Modify: `src/factorzen/agents/nodes.py`（新增 3 节点）
- Test: `tests/test_agent_nodes_eval.py`

**Interfaces:**
- Consumes: `split_holdout`/`holdout_ic`/`deflated_sharpe`/`compute_pbo`/`TrialLedger`（`validation/`）；`family_groups`/`negative_recall`（Task 3）；`max_correlation`（`scoring.py`）
- Produces: `node_guardrails(state, *, daily, holdout_df, ledger) -> AgentState`；`node_critic(state, llm_fn) -> AgentState`；`node_reflect(state) -> AgentState`

- [ ] **Step 1: 写失败测试**（聚焦判别力：护栏真过滤、N 累加、critic 改判、reflect 更新负例）

```python
# tests/test_agent_nodes_eval.py
from factorzen.agents.nodes import node_critic, node_reflect
from factorzen.agents.state import AgentState, AttemptRecord


class FakeLLM:
    def __init__(self, responses):
        self._r = list(responses)
    def __call__(self, messages):
        return self._r.pop(0) if self._r else '{"verdict":"keep","reason":"ok"}'


def _state_with_attempts():
    s = AgentState(seed=1)
    s.attempts = [
        AttemptRecord(0, "h1", "ts_mean(close,5)", True, 0.05, True, None, None),
        AttemptRecord(0, "h2", "rank(vol)", True, 0.001, False, None, None),  # 低IC未过护栏
    ]
    return s


def test_node_critic_marks_verdict():
    import json
    s = _state_with_attempts()
    llm = FakeLLM([json.dumps({"verdict": "keep", "reason": "经济直觉成立"}),
                  json.dumps({"verdict": "drop", "reason": "疑似数据窥探"})])
    s = node_critic(s, llm)
    verdicts = [a.critic_verdict for a in s.attempts]
    assert "keep" in verdicts and "drop" in verdicts


def test_node_reflect_feeds_low_ic_to_negatives():
    s = _state_with_attempts()
    s = node_reflect(s, ic_threshold=0.01)
    # 低 IC 的 rank(vol) 进负例库，高 IC 的不进
    assert "rank(vol)" in s.negative_examples
    assert "ts_mean(close,5)" not in s.negative_examples
```

> 护栏节点 `node_guardrails` 的测试需构造带 holdout 的 mock daily（仿 Task 4 的 `_mock_daily`，n_days≥150 留出 holdout 段），断言：① 过护栏的候选进 `state.candidates` 且带 holdout_ic/pbo/dsr 字段；② `ledger.n_trials` 累加了本轮评估数（N 诚实记账）；③ holdout 段未参与 train 评估（隔离）。实现 Step 3 后补此测试（用真实 validation 接口）。

- [ ] **Step 2: 跑测试确认失败** → FAIL

- [ ] **Step 3: 实现验收侧节点**（追加到 nodes.py）

```python
import json

from factorzen.agents.memory import negative_recall
from factorzen.llm.generation import LLMFn


def node_guardrails(state: AgentState, *, daily, holdout_df, ledger, top_k: int = 5) -> AgentState:
    """对过编译的候选记账 N、跑 holdout_ic/DSR/PBO，过关者进 candidates。
    复用 validation/ 接线（split_holdout 已在 pipeline 完成，这里收 holdout_df）。"""
    from factorzen.agents.evaluation import _node_to_factor_df
    from factorzen.discovery.expression import parse_expr
    from factorzen.validation.deflated_sharpe import deflated_sharpe
    from factorzen.validation.holdout import holdout_ic

    passed = [a for a in state.attempts if a.compile_ok and a.ic_train is not None]
    ledger.record(len(passed))  # N 累加本轮所有评估过的表达式（诚实多重检验）
    passed.sort(key=lambda a: abs(a.ic_train), reverse=True)
    for a in passed[:top_k]:
        try:
            node = parse_expr(a.expression)
            fdf_hold = _node_to_factor_df(node, holdout_df)
            ic_h, ir_h, ci = holdout_ic(fdf_hold, holdout_df)
            dsr, pval = deflated_sharpe(a.ir_train if a.ir_train else abs(a.ic_train),
                                        ledger.n_trials, n_obs=max(len(holdout_df), 20))
            if ic_h is not None and dsr > 0.5:  # 过关门槛（以实际 DSR 口径为准）
                a.passed_guardrails = True
                state.candidates.append({"expression": a.expression, "hypothesis": a.hypothesis,
                                         "ic_train": a.ic_train, "holdout_ic": ic_h,
                                         "holdout_ir": ir_h, "dsr": dsr, "dsr_pvalue": pval})
        except Exception:  # noqa: BLE001
            continue
    return state


def node_critic(state: AgentState, llm_fn: LLMFn) -> AgentState:
    """LLM 以风控审计员身份批判每个候选：keep/drop/mutate。"""
    for a in state.attempts:
        if a.critic_verdict is not None:
            continue
        msgs = [{"role": "system", "content": "你是风控审计员，判断因子是否过拟合/经济直觉是否成立，"
                                              '只输出 JSON: {"verdict":"keep"|"drop"|"mutate","reason":"..."}'},
                {"role": "user", "content": f"假设:{a.hypothesis} 表达式:{a.expression} "
                                            f"train_IC:{a.ic_train} 过护栏:{a.passed_guardrails}"}]
        try:
            obj = json.loads(llm_fn(msgs))
            a.critic_verdict = str(obj.get("verdict", "keep"))
        except Exception:  # noqa: BLE001
            a.critic_verdict = "keep"
    return state


def node_reflect(state: AgentState, *, ic_threshold: float = 0.01) -> AgentState:
    """更新 Negative RAG 负例库 + 推进迭代计数。"""
    seen = [(a.expression, a.ic_train) for a in state.attempts if a.ic_train is not None]
    state.negative_examples = negative_recall(seen, k=5, ic_threshold=ic_threshold)
    state.iteration += 1
    return state
```

- [ ] **Step 4: 补 `node_guardrails` 测试**（追加到 `tests/test_agent_nodes_eval.py`）

```python
def test_node_guardrails_n_accounting_and_holdout_isolation():
    import datetime as dt
    import numpy as np
    import polars as pl
    from factorzen.agents.evaluation import evaluate_expressions
    from factorzen.agents.nodes import node_guardrails
    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import split_holdout
    from factorzen.validation.multiple_testing import TrialLedger

    rng = np.random.default_rng(3)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < 180:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(20)]
    rows = []
    for c in codes:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    daily = pl.DataFrame(rows)
    mining_df, holdout_df, _ = split_holdout(daily, holdout_ratio=0.2)
    bundle = DataBundle.build(mining_df)

    s = AgentState(seed=1)
    for r in evaluate_expressions(["ts_mean(close,5)", "rank(vol)"], mining_df, bundle):
        s.attempts.append(AttemptRecord(0, "h", r["expression"], r["compile_ok"],
                                        r["ic_train"], False, None, r["error"]))
    ledger = TrialLedger()
    s = node_guardrails(s, daily=mining_df, holdout_df=holdout_df, ledger=ledger, top_k=5)

    assert ledger.n_trials >= 1                              # ① N 诚实累加本轮评估数
    for c in s.candidates:
        assert "holdout_ic" in c and "dsr" in c             # ② 候选带 holdout 证据
    assert mining_df["trade_date"].max() < holdout_df["trade_date"].min()  # ③ holdout 隔离
```
> 过护栏门槛（DSR > 0.5）在随机 mock 下可能无候选通过——此时 `s.candidates` 为空，①③ 仍有判别力；如需保证有候选，可放宽门槛或用真实数据手动验证。断言聚焦 N 记账 + holdout 隔离这两条灵魂约束。

- [ ] **Step 5: 跑测试 + ruff + 提交**

```bash
pixi run pytest tests/test_agent_nodes_eval.py -v
pixi run ruff check src/factorzen/agents/nodes.py tests/test_agent_nodes_eval.py
git add src/factorzen/agents/nodes.py tests/test_agent_nodes_eval.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(agents): 闭环验收侧节点(护栏/N记账/critic/reflect)"
```

---

## Task 7: 主循环编排（orchestrator.py）

**Files:**
- Create: `src/factorzen/agents/orchestrator.py`
- Test: `tests/test_agent_orchestrator.py`

**Interfaces:**
- Consumes: 所有 `node_*`（Task 5/6）；`split_holdout`/`TrialLedger`（validation）；`DataBundle.build`
- Produces: `AgentResult`（dataclass：state + candidates + n_trials）；`run_llm_agent(daily, llm_fn, *, n_rounds, seed, top_k=5, holdout_ratio=0.2, human_review=False) -> AgentResult`

- [ ] **Step 1: 写失败测试（端到端闭环 + 可复现 + holdout 隔离）**

```python
# tests/test_agent_orchestrator.py
import datetime as dt
import json
import numpy as np
import polars as pl

from factorzen.agents.orchestrator import run_llm_agent


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


def _scripted_llm():
    """每轮：1 个 proposal + semantic(pass) + critic(keep)。无限循环复用。"""
    prop = json.dumps({"hypothesis": "动量", "expressions": ["ts_mean(close,5)"], "rationale": "r"})
    sem = json.dumps({"consistent": True, "reason": "ok"})
    crit = json.dumps({"verdict": "keep", "reason": "ok"})
    seq = [prop, sem, crit] * 50
    i = {"k": 0}
    def fn(messages):
        v = seq[i["k"] % len(seq)]; i["k"] += 1
        return v
    return fn


def test_run_llm_agent_closes_loop():
    daily = _mock_daily()
    res = run_llm_agent(daily, _scripted_llm(), n_rounds=3, seed=42)
    assert res.state.iteration == 3
    assert res.n_trials >= 1            # N 累加了
    assert len(res.state.attempts) >= 1


def test_run_llm_agent_reproducible():
    daily = _mock_daily()
    r1 = run_llm_agent(daily, _scripted_llm(), n_rounds=2, seed=7)
    r2 = run_llm_agent(daily, _scripted_llm(), n_rounds=2, seed=7)
    # 同 seed + 同 scripted LLM → 尝试序列逐字节一致
    assert [a.expression for a in r1.state.attempts] == [a.expression for a in r2.state.attempts]
    assert r1.n_trials == r2.n_trials
```

- [ ] **Step 2: 跑测试确认失败** → FAIL

- [ ] **Step 3: 实现 orchestrator.py**

```python
# src/factorzen/agents/orchestrator.py
"""Agent 闭环主循环：只调度，业务逻辑在 nodes。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from factorzen.agents.nodes import (
    node_critic, node_evaluate, node_generate, node_guardrails, node_reflect,
)
from factorzen.agents.state import AgentState
from factorzen.discovery.scoring import DataBundle
from factorzen.llm.generation import LLMFn
from factorzen.validation.holdout import split_holdout
from factorzen.validation.multiple_testing import TrialLedger


@dataclass
class AgentResult:
    state: AgentState
    candidates: list[dict]
    n_trials: int


def run_llm_agent(daily, llm_fn: LLMFn, *, n_rounds: int, seed: int, top_k: int = 5,
                  holdout_ratio: float = 0.2, human_review: bool = False) -> AgentResult:
    rng = np.random.default_rng(seed)  # noqa: F841 预留给未来随机选择，保证可复现入口
    mining_df, holdout_df, _hstart = split_holdout(daily, holdout_ratio=holdout_ratio)
    bundle = DataBundle.build(mining_df)        # Agent 只见 mining 段
    ledger = TrialLedger()
    state = AgentState(seed=seed)
    feedback = ""
    for _ in range(n_rounds):
        state = node_generate(state, llm_fn, daily=mining_df, bundle=bundle, feedback=feedback)
        state = node_evaluate(state, daily=mining_df, bundle=bundle)
        state = node_guardrails(state, daily=mining_df, holdout_df=holdout_df,
                                ledger=ledger, top_k=top_k)
        state = node_critic(state, llm_fn)
        if human_review:
            _human_gate(state)  # 打印候选 + 等输入（非交互/CI 跳过）
        state = node_reflect(state)
        feedback = _summarize_feedback(state)
    return AgentResult(state=state, candidates=state.candidates, n_trials=ledger.n_trials)


def _summarize_feedback(state: AgentState) -> str:
    if not state.attempts:
        return ""
    last = state.attempts[-1]
    return f"上轮最佳 train_IC={last.ic_train}; 已试 {len(state.seen_expressions)} 个表达式。"


def _human_gate(state: AgentState) -> None:
    import sys
    if not sys.stdin.isatty():   # 非交互（CI/管道）跳过
        return
    print(f"[agent] 本轮候选 {len(state.candidates)} 个，回车继续...")
    try:
        input()
    except EOFError:
        pass
```
> 关键：`split_holdout` 切出 `mining_df` 后，`bundle` 只用 mining 段，holdout 段仅在 `node_guardrails` 验收时碰——**holdout 对生成/反思完全不可见**。

- [ ] **Step 4: 跑测试确认通过 + ruff + 提交**

```bash
pixi run pytest tests/test_agent_orchestrator.py -v
pixi run ruff check src/factorzen/agents/orchestrator.py tests/test_agent_orchestrator.py
git add src/factorzen/agents/orchestrator.py tests/test_agent_orchestrator.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(agents): run_llm_agent 主循环(N累加/holdout隔离/human-review)"
```

---

## Task 8: Session manifest 落盘

**Files:**
- Create: `src/factorzen/agents/manifest.py`
- Test: `tests/test_agent_manifest.py`

**Interfaces:**
- Consumes: `AgentResult`/`AgentState`（Task 7）
- Produces: `write_session_manifest(result, *, out_dir, run_id, params) -> Path`（落 `manifest.json` 含 seed/每轮 attempts/candidates/参数/git_sha）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_agent_manifest.py
import json
from pathlib import Path

from factorzen.agents.manifest import write_session_manifest
from factorzen.agents.orchestrator import AgentResult
from factorzen.agents.state import AgentState, AttemptRecord


def test_manifest_written_with_audit_trail(tmp_path: Path):
    s = AgentState(seed=42)
    s.attempts = [AttemptRecord(0, "h", "rank(close)", True, 0.04, True, "keep", None)]
    s.candidates = [{"expression": "rank(close)", "holdout_ic": 0.03, "dsr": 0.8}]
    res = AgentResult(state=s, candidates=s.candidates, n_trials=5)
    p = write_session_manifest(res, out_dir=str(tmp_path), run_id="t1",
                               params={"n_rounds": 3, "seed": 42})
    m = json.loads(Path(p).read_text())
    assert m["seed"] == 42 and m["n_trials"] == 5
    assert m["params"]["n_rounds"] == 3
    assert m["attempts"][0]["expression"] == "rank(close)"   # 全程尝试可审计
    assert m["candidates"][0]["dsr"] == 0.8
    assert "git_sha" in m
```

- [ ] **Step 2: 跑测试确认失败** → FAIL

- [ ] **Step 3: 实现 manifest.py**

```python
# src/factorzen/agents/manifest.py
"""Agent session manifest：把假设/表达式/分数/候选/参数全程落盘（可审计、可复现）。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def write_session_manifest(result, *, out_dir: str, run_id: str, params: dict) -> Path:
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state = result.state
    manifest = {
        "run_id": run_id, "seed": state.seed, "n_trials": result.n_trials,
        "iterations": state.iteration, "params": params,
        "attempts": [a.__dict__ for a in state.attempts],
        "candidates": result.candidates,
        "git_sha": _git_sha(),
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return path
```

- [ ] **Step 4: 跑测试通过 + ruff + 提交**

```bash
pixi run pytest tests/test_agent_manifest.py -v
pixi run ruff check src/factorzen/agents/manifest.py tests/test_agent_manifest.py
git add src/factorzen/agents/manifest.py tests/test_agent_manifest.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(agents): session manifest 落盘(可审计可复现)"
```

---

## Task 9: Pipeline（factor_mine_agent.py）

**Files:**
- Create: `src/factorzen/pipelines/factor_mine_agent.py`
- Test: `tests/test_agent_pipeline.py`

**Interfaces:**
- Consumes: `run_llm_agent`（Task 7）；`write_session_manifest`（Task 8）；`export_candidate`（`discovery/export.py`）；`request_chat`/`load_llm_config`（生产 LLMFn）
- Produces: `run_agent_mine(daily, *, n_rounds, seed, out_dir, llm_fn=None, top_k=5, run_id=None, export=True) -> dict`（`{run_dir, n_candidates, n_trials, candidates}`）

- [ ] **Step 1: 写失败测试（注入 FakeLLM，验证产物 + 导出）**

```python
# tests/test_agent_pipeline.py
import datetime as dt
import json
from pathlib import Path
import numpy as np
import polars as pl

from factorzen.pipelines.factor_mine_agent import run_agent_mine


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


def _scripted_llm():
    prop = json.dumps({"hypothesis": "动量", "expressions": ["ts_mean(close,5)"], "rationale": "r"})
    sem = json.dumps({"consistent": True, "reason": "ok"})
    crit = json.dumps({"verdict": "keep", "reason": "ok"})
    seq = [prop, sem, crit] * 50
    i = {"k": 0}
    def fn(messages):
        v = seq[i["k"] % len(seq)]; i["k"] += 1
        return v
    return fn


def test_run_agent_mine_writes_manifest(tmp_path: Path):
    daily = _mock_daily()
    res = run_agent_mine(daily, n_rounds=2, seed=42, out_dir=str(tmp_path),
                         llm_fn=_scripted_llm(), run_id="t1", export=False)
    run_dir = Path(res["run_dir"])
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "candidates.csv").exists()   # 兼容 fz mine leaderboard
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["n_trials"] >= 1
    assert res["n_trials"] == m["n_trials"]
```

- [ ] **Step 2: 跑测试确认失败** → FAIL

- [ ] **Step 3: 实现 factor_mine_agent.py**

```python
# src/factorzen/pipelines/factor_mine_agent.py
"""LLM Agent 闭环挖掘 pipeline：跑 Agent → 落 manifest + 导出候选。"""
from __future__ import annotations

from pathlib import Path

from factorzen.agents.manifest import write_session_manifest
from factorzen.agents.orchestrator import run_llm_agent


def _default_llm_fn():
    """生产 LLMFn：包 request_chat + load_llm_config。"""
    from factorzen.llm.client import request_chat
    from factorzen.llm.config import load_llm_config
    config = load_llm_config(enabled=True)
    if not config.is_ready:
        raise RuntimeError("LLM 未配置：设置 .env 的 FACTORZEN_LLM_* 或注入 llm_fn")
    return lambda messages: request_chat(config, messages)


def run_agent_mine(daily, *, n_rounds: int, seed: int, out_dir: str = "workspace/mine_agent",
                   llm_fn=None, top_k: int = 5, holdout_ratio: float = 0.2,
                   human_review: bool = False, run_id: str | None = None,
                   export: bool = True) -> dict:
    fn = llm_fn or _default_llm_fn()
    result = run_llm_agent(daily, fn, n_rounds=n_rounds, seed=seed, top_k=top_k,
                           holdout_ratio=holdout_ratio, human_review=human_review)
    rid = run_id or f"agent_{seed}_{n_rounds}r"
    params = {"n_rounds": n_rounds, "seed": seed, "top_k": top_k, "holdout_ratio": holdout_ratio}
    write_session_manifest(result, out_dir=out_dir, run_id=rid, params=params)
    run_dir = Path(out_dir) / rid
    # candidates.csv —— 兼容 fz mine leaderboard（M1 读取格式）
    import polars as pl
    cand_df = pl.DataFrame(result.candidates) if result.candidates else pl.DataFrame(
        {"expression": [], "holdout_ic": [], "dsr": []})
    cand_df.write_csv(run_dir / "candidates.csv")
    if export and result.candidates:
        from factorzen.discovery.export import export_candidate
        exp_dir = run_dir / "exported"
        exp_dir.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(result.candidates):
            export_candidate(c["expression"], f"agent_{rid}_{i}", str(exp_dir))
    return {"run_dir": str(run_dir), "n_candidates": len(result.candidates),
            "n_trials": result.n_trials, "candidates": result.candidates}
```

- [ ] **Step 4: 跑测试通过 + ruff + 提交**

```bash
pixi run pytest tests/test_agent_pipeline.py -v
pixi run ruff check src/factorzen/pipelines/factor_mine_agent.py tests/test_agent_pipeline.py
git add src/factorzen/pipelines/factor_mine_agent.py tests/test_agent_pipeline.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(agents): run_agent_mine pipeline(manifest + 导出候选)"
```

---

## Task 10: CLI `fz mine agent` + 收尾

**Files:**
- Modify: `src/factorzen/cli/main.py`（`mine_sub` 加 `agent` 子命令 + `_cmd_mine_agent`）
- Test: `tests/test_agent_cli.py`
- Modify: `README.md`（核心能力「挖掘」行补 Agent）；plan 完成记录

**Interfaces:**
- Consumes: `run_agent_mine`（Task 9）；`get_universe`/`loader`（拉数据，仿 `factor_mine.run_mine`）；`build_parser`/`_cmd_mine_search` 模式

- [ ] **Step 1: 写失败测试**

```python
# tests/test_agent_cli.py
def test_parser_has_mine_agent():
    from factorzen.cli.main import build_parser
    p = build_parser()
    args = p.parse_args(["mine", "agent", "--start", "20220101", "--end", "20231231",
                         "--iterations", "5", "--seed", "42"])
    assert args.command == "mine"
    assert args.mine_command == "agent"
    assert args.start == "20220101"
    assert args.iterations == 5
    assert args.seed == 42
    assert callable(args.func)
```

- [ ] **Step 2: 跑测试确认失败** → FAIL（`AttributeError`）

- [ ] **Step 3: 接入 CLI**

在 `build_parser()` 的 `mine_sub` 下（`leaderboard` 之后）加：
```python
    m_agent = mine_sub.add_parser("agent", help="LLM-guided agent factor mining")
    m_agent.add_argument("--start", required=True)
    m_agent.add_argument("--end", required=True)
    m_agent.add_argument("--universe", default=None)
    m_agent.add_argument("--iterations", type=int, default=5)
    m_agent.add_argument("--top-k", dest="top_k", type=int, default=5)
    m_agent.add_argument("--seed", type=int, default=42)
    m_agent.add_argument("--human-review", action="store_true", dest="human_review")
    m_agent.set_defaults(func=_cmd_mine_agent)
```
模块顶层加 handler（仿 `_cmd_mine_search`，延迟 import + 仿 `run_mine` 的数据加载）：
```python
def _cmd_mine_agent(args: argparse.Namespace) -> int:
    from factorzen.core import loader
    from factorzen.core.universe import get_universe
    from factorzen.pipelines.factor_mine_agent import run_agent_mine
    # 仿 factor_mine.run_mine 的数据加载（universe + daily）
    stocks = get_universe(args.end, args.universe) if args.universe else None
    daily = loader.fetch_daily(args.start, args.end)
    if stocks is not None:
        import polars as pl
        daily = daily.filter(pl.col("ts_code").is_in(stocks["ts_code"].to_list()))
    res = run_agent_mine(daily, n_rounds=args.iterations, seed=args.seed,
                         top_k=args.top_k, human_review=args.human_review)
    print(f"[mine-agent] 候选 {res['n_candidates']} 个 / N={res['n_trials']} → {res['run_dir']}")
    return 0
```
> 数据加载以 `factor_mine.run_mine` 实际实现为准（可能用 `FactorDataContext` 带 lookback）。先 Read `factor_mine.py` 对齐。

- [ ] **Step 4: 跑测试通过**

Run: `pixi run pytest tests/test_agent_cli.py -v` → PASS

- [ ] **Step 5: 全量质量门**

```bash
pixi run pytest tests/test_agent_*.py -q     # M5 全测试绿
pixi run ruff check src/factorzen/agents/ src/factorzen/llm/generation.py src/factorzen/pipelines/factor_mine_agent.py tests/test_agent_*.py
```

- [ ] **Step 6: README + 提交**

README「核心能力」表「挖掘」行补："+ LLM Agent 闭环（`fz mine agent`）"。
```bash
git add src/factorzen/cli/main.py tests/test_agent_cli.py README.md
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(agents): fz mine agent CLI + README"
```

---

## 收尾验收（全部 task 完成后）

- [ ] `pixi run pytest tests/test_agent_*.py -q` 全绿（state/generation/memory/evaluation/nodes×2/orchestrator/manifest/pipeline/cli）
- [ ] `pixi run ruff check src/factorzen/agents/ src/factorzen/llm/generation.py src/factorzen/pipelines/factor_mine_agent.py tests/test_agent_*.py` 0 errors
- [ ] **可复现**：同 seed + FakeLLM，`run_llm_agent` 两次尝试序列逐字节一致（Task 7 已断言）
- [ ] **防过拟合**：N 累加所有评估表达式（Task 6 断言）；holdout 段不进 mining 评估（Task 7 隔离）
- [ ] **增强机制有判别力**：语义对齐拒不一致表达式、Negative RAG 负例进 prompt、family 分组识别同族——均非恒真
- [ ] **Leaderboard 兼容**：`run_agent_mine` 产 `candidates.csv`，`fz mine leaderboard <agent_run>/candidates.csv` 可读（手动核对）
- [ ] 真实 LLM smoke（手动，需 `FACTORZEN_LLM_*`）：`pixi run fz mine agent --start 20220101 --end 20231231 --iterations 3` → 产 manifest + 候选，人工核对假设/表达式合理
- [ ] `git status --short` 干净（只 M5 相关入库，未带 M0）
- [ ] 更新本 plan 追加完成记录 + memory roadmap（M5 完成）

---

*M5 完成后，FactorZen 拥有"假设 → 生成 → 评估 → 反思"的 LLM 闭环因子挖掘，全程可审计、可复现、CI 可测，严守 M2 护栏——为 M6 多 Agent 协作铺路。*

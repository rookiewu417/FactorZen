# M6 · 多 Agent + 长期记忆 — 设计文档

> 状态：设计讨论完成（2026-06-30），待用户复核 → 转实现计划。
> 上游：[FactorZen 升级计划](../../FactorZen-升级计划.md) 的里程碑 **M6**，依赖 M5（单 Agent 闭环）。
> 定位：把 M5 单 Agent 升级为**多角色协作系统**——Hypothesis / Coder / Evaluator / Critic / Librarian 流水线协作，Critic 可否决并触发重提/修正，Librarian 维护跨 session 长期记忆（避免重复、积累"已知有效/无效"知识）。

---

## 1. 目标与定位

把 M5 的单 Agent 七步闭环，拆成**显式的多角色协作团队**，并加上 M5 没有的**跨 session 长期记忆**。

**核心立场：真 multi-agent，不是换皮。** M5 的节点已经隐含角色，M6 的增量必须是真能力——① 角色间**真交互**（Critic 否决 → Hypothesis 重提 / Coder 修正）；② **跨 session 长期记忆**（experiment_index 查重 + 知识反馈）。给确定性计算（Evaluator）套 Agent 壳没有价值，故 Evaluator 沿用 M5 的确定性评估，不调 LLM。

### 1.1 已拍板决策（讨论结论）

| 决策 | 选择 | 理由 |
|---|---|---|
| 范围 | **A 务实端到端** | 完整兑现升级计划 M6 核心，不做多轮辩论/残差记忆（defer） |
| 协作形态 | **流水线 + Critic 否决回路** | 各司其职 + 关键交互（Critic 否决→重提/修正），卖点真、可测 |
| 长期记忆 | **查重 + 跨 session 知识库** | experiment_index 跨 run 去重 + 已知有效/无效反馈，复用 M5 归一化/Negative RAG 扩到跨 session |
| 角色拆分 | **5 角色边界，Hypothesis/Coder 分开** | Evaluator 确定性（复用 M5）；H/C 分开支持 Critic 细粒度修正（换方向 vs 换表达式） |
| Critic 判据 | **读 manifest 数值**（过拟合/IC/holdout/DSR） | tear sheet 文本判定 defer（范围 B） |
| 编排技术 | **自建流水线**（沿用 M5 自建 loop，零新依赖） | 与 M5 一脉相承；多角色仅是更多函数式步骤 + 否决回路 |
| 可测 | **角色 LLM 全可注入 FakeLLM** | CI 离线；否决回路用 scripted FakeLLM 测确定性 |

---

## 2. 现状（基线）

M6 站在已入库的 M5 之上：

| 复用对象 | 提供什么 | 位置 |
|---|---|---|
| 表达式评估 | `evaluate_expressions(expr_strs, daily, bundle) -> list[dict]` | `agents/evaluation.py` |
| 护栏（Evaluator） | `node_guardrails(state, *, daily, holdout_df, ledger, bundle, top_k)` | `agents/nodes.py` |
| 状态 | `AgentState` / `AttemptRecord`（含 ir_train） | `agents/state.py` |
| session 记忆 | `negative_recall` / `family_groups` | `agents/memory.py` |
| manifest | `write_session_manifest` | `agents/manifest.py` |
| LLM 生成 | `request_chat` / `generate_factor_proposal` / `semantic_check` / `build_agent_messages` / `LLMFn` | `llm/generation.py`、`llm/client.py` |
| 数据/护栏 | `DataBundle.build` / `split_holdout` / `TrialLedger` / `deflated_sharpe` / `holdout_ic` | `discovery/scoring.py`、`validation/` |
| 导出 | `export_candidate` | `discovery/export.py` |
| CLI | `fz mine agent`、`build_parser`、`_cmd_mine_agent` | `cli/main.py` |

**缺口**：无显式角色（隐含在节点里）；无跨 session 长期记忆（M5 memory 仅 session 内）。

---

## 3. 架构与模块边界

**自建流水线**（沿用 M5 自建 loop 哲学，零新依赖）：角色是注入 `LLMFn` 的函数式步骤，`team_orchestrator` 顺序调用 + 处理 Critic 否决回路。

### 3.1 模块分解

| 层 | 模块 | 新建/复用 | 职责 |
|---|---|---|---|
| 角色 | `agents/roles/__init__.py` | 🆕 | 包标记 |
| 角色 | `agents/roles/hypothesis.py` | 🆕 | `propose_hypotheses(ctx, llm_fn, *, known_invalid, known_valid, n) -> list[str]` |
| 角色 | `agents/roles/coder.py` | 🆕 | `write_expressions(hypothesis, llm_fn, *, avoid) -> list[str]`；`revise_expressions(hypothesis, prev_exprs, critic_reason, llm_fn) -> list[str]` |
| 角色 | `agents/roles/critic.py` | 🆕 | `critique(candidate, llm_fn) -> CriticVerdict` |
| 角色 | `agents/roles/librarian.py` | 🆕 | `recall(index) -> Recall`；`record(index, attempts, run_id)` |
| 记忆 | `agents/experiment_index.py` | 🆕 | `ExperimentIndex`（jsonl 跨 session 持久化 + 归一化查重 + known_invalid/valid） |
| 编排 | `agents/team_orchestrator.py` | 🆕 | `run_team_mine(...)` 流水线 + 否决回路 |
| 评估 | `evaluate_expressions` + `node_guardrails`（Evaluator） | ♻️ M5 | 确定性评估 + 护栏 |
| 复用 | session 记忆/归一化、manifest、state、llm/generation | ♻️ M5 | |
| 入口 | `pipelines/factor_mine_team.py` | 🆕 | `run_team_mine` 拉数据 + 落产物 |
| CLI | `cli/main.py` `fz mine team` | 🆕（扩展） | |

---

## 4. 角色职责

| 角色 | LLM? | 职责 |
|---|---|---|
| **Hypothesis** | ✅ | 提经济直觉方向（自然语言，如"小市值×低换手反转"）；收到长期记忆，避开已知无效方向 |
| **Coder** | ✅ | 把方向翻译成 1-N 个表达式；按 Critic 反馈**修正表达式**（同假设换写法） |
| **Evaluator** | ❌ 确定性 | `evaluate_expressions`（Rank IC）+ `node_guardrails`（holdout/DSR/N 记账）。**复用 M5，不调 LLM** |
| **Critic** | ✅ | 读候选 + manifest 指标（IC/holdout_ic/DSR/N），判过拟合/经济直觉 → `CriticVerdict`（keep / revise_expr / revise_hypothesis / drop） |
| **Librarian** | ❌（确定性 + 可选 LLM 摘要） | 跨 session 长期记忆：查重集、已知无效（负例）、已知有效（正例）；本轮尝试写回 experiment_index |

---

## 5. 数据流（一轮 = 流水线 + 否决回路）

```
① Librarian.recall   从 experiment_index 取 known_invalid（负例）+ known_valid（正例）+ seen（查重集）
② Hypothesis         提方向（注入 known_invalid 避开已知无效；known_valid 作正例参考）
③ Coder              方向 → 表达式（跨 session seen + 本 session 已试 去同，跳过重复）
④ Evaluator          evaluate_expressions + node_guardrails（holdout/DSR/N）[确定性]
⑤ Critic.critique    读候选 + 指标 → CriticVerdict:
                        keep              → 入候选
                        revise_expr       → 回 ③ Coder.revise（同假设改写，retry ≤ MAX_CODER_RETRY）
                        revise_hypothesis → 回 ② Hypothesis（换方向，retry ≤ MAX_HYP_RETRY）
                        drop              → 丢弃
⑥ Librarian.record   本轮所有尝试（归一化表达式/ic/holdout_ic/dsr/verdict）写入 experiment_index
   迭代 N 轮 → 候选 + team manifest（角色决策全程可审计）
```

**否决回路防死循环**：每个假设最多 `MAX_CODER_RETRY`（如 2）次 Coder 修正；每轮最多 `MAX_HYP_RETRY`（如 1）次 Hypothesis 重提；超限即放弃本方向，进下一轮。所有重试产生的表达式**都经 Evaluator 评估并计入 N**（诚实记账）。

---

## 6. 跨 session 长期记忆（experiment_index.jsonl）

**格式**（每行一个尝试，JSONL）：
```json
{"expression": "ts_mean(close, 5)", "hypothesis": "动量", "ic_train": 0.03,
 "holdout_ic": 0.02, "dsr": 0.6, "passed": true, "verdict": "keep",
 "run_id": "team_42_5r", "ts": "20260630T..."}
```

**`ExperimentIndex` 接口**：
- `load() -> list[dict]`：读全量历史。
- `append(records: list[dict])`：本 run 尾追加。
- `seen_expressions() -> set[str]`：归一化表达式集（跨 session 查重，Coder 用 `to_expr_string(parse_expr(e))` 对齐）。
- `known_invalid(k) -> list[str]`：低 IC / 未过护栏的表达式（跨 session Negative RAG，注入 Hypothesis/Coder prompt）。
- `known_valid(k) -> list[str]`：过护栏的表达式（正例参考）。

**用法**：① Coder 生成的表达式若归一化形式 ∈ `seen_expressions()` → 跳过（不重复评估，省算力）；② Hypothesis/Coder prompt 注入 `known_invalid`（"这些方向已验证无效，避开"）+ `known_valid`（"这些已验证有效，可借鉴"）。复用 M5 的 `negative_recall`/归一化思路，扩到跨 run。

---

## 7. 防过拟合接入（灵魂，继承 M5）

| 约束 | 做法 |
|---|---|
| N 诚实记账 | `TrialLedger` 累加**本 run 所有评估过的表达式**（含否决回路重试），沿用 M5 的本轮语义修复。**N 不跨 run 累加**——跨 session 查重只为省算力，若把历史所有 run 的 N 累加会让 DSR 病态过严 |
| holdout 隔离 | holdout 段只在 Evaluator 的 `node_guardrails` 碰；Hypothesis/Coder/Critic/Librarian/长期记忆全程只见 mining 段指标 |
| 跨 session 查重不泄漏 | experiment_index 存的是 mining 段 IC + holdout 段验收结果（已是终值），不让 Hypothesis 看到 holdout 原始数据 |
| `known_valid` 跨 run 传承的权衡 | "已验证有效"的表达式跨 run 传给 Hypothesis 是**长期记忆的目的**（积累知识），不是同 run 内 holdout 泄漏。但反复把有效因子当种子复用会过拟合到 holdout，故 `known_valid` 仅作 Hypothesis 的**方向参考**（"这类思路有效，可借鉴变体"），**不直接喂给 Coder 复用原表达式**；每个 run 的 holdout 验收仍独立、N 仍 per-run 诚实 |
| Critic 拦截 | Critic 对 DSR 不显著 / N 过大 / holdout_ic 与 train 背离的候选判 drop（M6 验收：Critic 能拦截过拟合候选） |

---

## 8. 接口契约

**新建：**
```python
# agents/roles/critic.py
@dataclass
class CriticVerdict:
    verdict: str   # "keep" | "revise_expr" | "revise_hypothesis" | "drop"
    reason: str
def critique(candidate: dict, llm_fn: LLMFn) -> CriticVerdict

# agents/roles/hypothesis.py
def propose_hypotheses(llm_fn: LLMFn, *, known_invalid: list[str], known_valid: list[str],
                       feedback: str = "", n: int = 1) -> list[str]

# agents/roles/coder.py
def write_expressions(hypothesis: str, llm_fn: LLMFn, *, avoid: list[str]) -> list[str]
def revise_expressions(hypothesis: str, prev_exprs: list[str], critic_reason: str,
                       llm_fn: LLMFn) -> list[str]

# agents/roles/librarian.py
@dataclass
class Recall:
    seen: set[str]; known_invalid: list[str]; known_valid: list[str]
def recall(index: "ExperimentIndex", *, k: int = 5) -> Recall
def record(index: "ExperimentIndex", attempts: list, run_id: str) -> None

# agents/experiment_index.py
class ExperimentIndex:
    def __init__(self, path: str): ...
    def load(self) -> list[dict]: ...
    def append(self, records: list[dict]) -> None: ...
    def seen_expressions(self) -> set[str]: ...
    def known_invalid(self, k: int = 5) -> list[str]: ...
    def known_valid(self, k: int = 5) -> list[str]: ...

# agents/team_orchestrator.py
@dataclass
class TeamResult:
    state: AgentState; candidates: list[dict]; n_trials: int; rounds_log: list[dict]
def run_team_mine(daily, llm_fn: LLMFn, *, n_rounds: int, seed: int, index_path: str,
                  top_k: int = 5, holdout_ratio: float = 0.2,
                  max_coder_retry: int = 2, max_hyp_retry: int = 1) -> TeamResult

# pipelines/factor_mine_team.py
def run_team_mine(daily, *, n_rounds, seed, out_dir, index_path, llm_fn=None, ...) -> dict
```

**复用**（writing-plans 用 interface agent 精确化）：`evaluate_expressions`、`node_guardrails`、`AgentState`/`AttemptRecord`、`write_session_manifest`、`negative_recall`、`request_chat`/`LLMFn`/`build_agent_messages`、`DataBundle`/`split_holdout`/`TrialLedger`、`export_candidate`、`build_parser`。

---

## 9. 测试策略 + 验收（DoD）

- [ ] **角色单测**（FakeLLM 离线）：Hypothesis 注入 known_invalid 避开；Coder 写表达式 + revise 按 reason 改；Critic 各 verdict 解析；Librarian recall/record。
- [ ] **否决回路确定性**：scripted FakeLLM（Critic 先 revise_expr 再 keep）→ Coder 被调用修正、最终入候选、重试计入 N、≤ max_retry 不死循环（跨轮回归断言）。
- [ ] **跨 session 记忆去重**：两次 run 共享 experiment_index → 第二次 run 的重复表达式被 `seen_expressions` 跳过（不重复评估）。**M6 验收：记忆能去重。**
- [ ] **Critic 拦截过拟合**：构造一个 DSR 不显著 / holdout 背离的候选 → Critic 判 drop（非恒真）。**M6 验收：Critic 能拦截过拟合候选。**
- [ ] **防过拟合**：N per-run 诚实累加（含否决重试）；holdout 段不进生成/记忆。
- [ ] **CLI**：`fz mine team` parser + handler smoke。
- [ ] **真实 LLM smoke**（手动）：跑通多角色一轮，team manifest 记录角色决策 + experiment_index 更新。
- [ ] 零新依赖；ruff/test 绿；`git add` 只含 M6 不带 M0；提交 `rookiewu417`。

CI 离线：全程 FakeLLM 不触网；experiment_index 用 tmp_path。

---

## 10. 建议实现顺序（为 writing-plans 铺垫）

1. **`agents/experiment_index.py`** + 测试：jsonl 读写 + 归一化 seen + known_invalid/valid。
2. **`agents/roles/critic.py`**：`CriticVerdict` + `critique`（FakeLLM 测各 verdict）。
3. **`agents/roles/hypothesis.py`**：`propose_hypotheses`（注入 known_invalid/valid）。
4. **`agents/roles/coder.py`**：`write_expressions` + `revise_expressions`（含归一化去重对齐）。
5. **`agents/roles/librarian.py`**：`recall` + `record`（包 experiment_index）。
6. **`agents/team_orchestrator.py`**：`run_team_mine` 流水线 + 否决回路 + N 诚实 + holdout 隔离 + max_retry 防死循环。
7. **`pipelines/factor_mine_team.py`**：`run_team_mine` + team manifest + candidates.csv + export。
8. **CLI `fz mine team`** + README。
9. **真实 LLM smoke** + plan/memory 完成记录。

---

## 11. 范围外

- Critic 读 tear sheet / 报告文本判 OOS（范围 B）→ 后续。
- 角色多轮辩论（Hypothesis↔Critic 来回辩护）→ 后续。
- 残差边级记忆 / AlphaMemo（M5 已 defer）。
- 知识库可视化 / 导出页（M7 展示）。
- 跨 run 累加 N（会让 DSR 病态过严，明确不做）。
- LangGraph/多 Agent 框架（M5 已论证不引；M6 仍自建，节点形态预留迁移口）。

---

*M6 完成后，FactorZen 拥有"多角色协作（假设/编码/评审/风控/记忆）+ 跨 session 知识积累"的因子研究系统——Critic 拦截过拟合、记忆避免重复，全程可审计可复现，是简历级的 multi-agent 差异化成果。*

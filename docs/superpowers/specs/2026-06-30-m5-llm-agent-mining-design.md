# M5 · LLM 单 Agent 闭环挖掘 — 设计文档

> 状态：设计讨论完成（2026-06-30），待用户复核 → 转实现计划。
> 上游：[FactorZen 升级计划](../../FactorZen-升级计划.md) 的里程碑 **M5**，依赖 M1（挖掘）+ M2（护栏）。
> 定位：把"研究员思维"自动化——LLM 提出**带经济直觉的假设** → 生成因子表达式 → 复用 M1 评估 + M2 护栏 → 反思迭代，全过程可审计、可复现、CI 可测。

---

## 1. 目标与定位

在 M1（程序化随机/遗传搜索）+ M2（防过拟合护栏）之上，加一个 **LLM 驱动的闭环挖掘器**：用 LLM 的研究员直觉（"小市值 × 低换手 × 反转"这类有经济逻辑的假设）替代盲目的算子组合，让挖掘**有方向**。

**核心立场：M5 是"研究员思维加速器"，但严守 M2 护栏——这是它与"危险的 LLM 挖矿机"的根本区别。** Agent 会快速产出大量表达式（= 大规模多重检验，升级计划点名的头号风险），所以防过拟合是 M5 的**灵魂约束**，不是附加项。

### 1.1 已拍板决策（讨论结论）

| 决策 | 选择 | 理由 |
|---|---|---|
| 范围 | **C 完整 M5** | 单 Agent 闭环 + 三个业界增强 + human-review + manifest + leaderboard |
| 生成形态 | **表达式 DSL**（非 Python 代码） | 复用 `expression.py` 编译器，DSL 受限 = 天然 sandbox，无需 `sandbox.py`；业界共识（AlphaAgent/Hubble） |
| 闭环结构 | **假设驱动 + LLM 直接写表达式** | 纯 LLM 生成（不掺 genetic）；假设文本落 manifest（"研究员思维可审计"卖点） |
| 每轮反馈 | **指标 + 诊断 + session 记忆** | 紧凑可测，足够反思 |
| 编排技术 | **自建极简 loop**（不引 LangGraph/LangChain） | 单 Agent + N≤20 轮 + 已有评估设施；框架伤害轻依赖/可复现/可测试/已有 llm 四约束 |
| 业界增强（纳入） | **语义对齐自检 + family-aware 多样性 + Negative RAG** | 三者低成本、与已有设施天然衔接、高 ROI |
| 业界增强（推迟） | 残差边级记忆（AlphaMemo）→ M6；MCTS / 多 Agent → M6 | YAGNI，避免过早复杂 |
| 新依赖 | **零** | 全部复用项目已有栈 |

### 1.2 调研依据（业界 2024-2026）

- **DSL > 自由代码**：AlphaAgent (KDD'25)、Hubble、QuantaAlpha 均选受限 DSL，规避代码安全/兼容问题。
- **防过拟合是业界普遍短板**：多数方案只有 OOS 窗口；我们已有的 PBO + DSR + holdout 永久隔离属业界前列——是护城河，须复用并接 Agent。
- **三个高 ROI 机制**：语义对齐自检（AlphaAgent，省 ~30% 无效回测）、family-aware 家族多样性（Hubble，防假多样性）、Negative RAG（Hubble，低成本防重复探索）。
- **单 Agent + FakeLLM 的 CI 可测性**是被业界低估的工程优势（多 Agent 越多 mock 点越多越难测）。

---

## 2. 现状（基线）

M5 站在已入库的 M1 + M2 + `llm/` 之上：

| 复用对象 | 提供什么 | 位置 |
|---|---|---|
| 表达式 DSL | AST ↔ 字符串 ↔ polars 编译、算子库、叶子特征 | `discovery/expression.py`、`operators.py` |
| 评估 | Rank IC/IR 打分（挖掘段） | `discovery/scoring.py`、`mining_session.run_session` |
| 导出 | 表达式 → workspace 因子文件 | `discovery/export.export_candidate` |
| 护栏 | TrialLedger(N 记账) / bootstrap(IC CI) / Deflated Sharpe / PBO(CSCV) / holdout 永久隔离 | `validation/` |
| LLM 客户端 | OpenAI 兼容 `chat/completions`（纯 `urllib`，零依赖）+ config/prompt/schema/cache | `llm/client.py` 等 |
| 数据切分 | train(mining) / holdout 永久隔离 | M2 的 `DataBundle` |
| CLI | `fz mine search/leaderboard`，`build_parser()` | `cli/main.py` |

**缺口**：`agents/` 模块全新；`llm/` 当前只为"因子解释"服务（`generate_llm_explanation` + 单一 `LLMExplanation` schema），需扩展出"因子**生成**"能力。

---

## 3. 架构与模块边界

**自建极简 loop**：纯 Python `for` 循环 + 显式 `dataclass` State + 函数式节点 `node(State) → State` + 注入 `LLMFn`。借鉴 LangGraph 设计哲学（显式状态、函数式节点、主循环只调度），但**不引入框架包**。与现有 `mining_session.py` 的"for 循环 → 评分 → 护栏 → manifest"一脉相承。

### 3.1 模块分解

| 层 | 模块 | 新建/复用 | 职责 |
|---|---|---|---|
| 智能 | `llm/generation.py` | 🆕 扩展 `llm/` | `generate_factor_proposal(ctx) → FactorProposal`；`semantic_check(hypothesis, expr) → bool` |
| 状态 | `agents/state.py` | 🆕 | `AgentState` / `AttemptRecord`（JSON 可序列化 dataclass） |
| 节点 | `agents/nodes.py` | 🆕 | `node_generate / compile / semantic_check / evaluate / guardrails / critic / reflect` |
| 记忆 | `agents/memory.py` | 🆕 | session 记忆 + Negative RAG 负例库 + family 多样性分组 |
| 编排 | `agents/orchestrator.py` | 🆕 | `run_llm_agent(...)` 主循环（只调度） |
| 落盘 | `agents/manifest.py` | 🆕 | session manifest（假设/表达式/分数/护栏/批判/LLM 元数据/seed） |
| 入口 | `pipelines/agent_mine.py` | 🆕 仿 `factor_mine` | `run_agent_mine(...)` 拉数据 → 跑 Agent → 落产物 |
| 评估 | `expression.py` / `scoring.py` / `run_session` 评估段 | ♻️ M1 | 编译 + Rank IC/IR |
| 验收 | `validation/` | ♻️ M2 | holdout/PBO/DSR/N 记账 |
| 导出 | `export.export_candidate` + leaderboard 汇入 | ♻️ M1 | |
| CLI | `cli/main.py` `fz mine agent` | 🆕（扩展） | |
| 测试 | `tests/.../fake_llm.py` 或夹具 | 🆕 | 确定性 `FakeLLM` |

---

## 4. 闭环数据流（一轮 = 七步）

每轮（round）按序执行七个函数式节点（方括号为业界增强）：

```
① node_generate     LLM 收到 {算子/特征清单 + 上轮反馈(指标+诊断)
                     + session 记忆 + [Negative RAG 负例模板]}
                     → 输出 JSON: {hypothesis, expressions[], rationale}
② node_compile      expression.py 编译；非法表达式记错误 → 喂回下轮(不浪费评估)
③ [semantic_check]  LLM 自查"表达式实现假设了吗"；不一致 → 丢弃/要求修正(省无效回测)
④ node_evaluate     复用 M1 Rank IC/IR(挖掘段)打分
⑤ node_guardrails   top 候选过 M2(holdout_ic/PBO/DSR) + [family-aware 家族多样性检查]
⑥ node_critic       LLM 自我批判(过拟合嫌疑? 经济直觉牵强?) → 保留/丢弃/变体
⑦ node_reflect      组装反馈 + 更新 session 记忆 + [Negative RAG 负例库]
                     → 进入下一轮
```

迭代 N 轮后 → 导出穿过护栏的候选 + session manifest。

**多假设并行**：① 一次产多个假设，各自走 ②–⑥（同一轮内并行评估，复用 M1 批量评估）。

**节点签名（统一形态）**：

```python
def node_generate(state: AgentState, llm_fn: LLMFn, ctx: AgentContext) -> AgentState: ...
def node_compile(state: AgentState) -> AgentState: ...
def node_semantic_check(state: AgentState, llm_fn: LLMFn) -> AgentState: ...
def node_evaluate(state: AgentState, bundle: DataBundle) -> AgentState: ...
def node_guardrails(state: AgentState, bundle: DataBundle) -> AgentState: ...
def node_critic(state: AgentState, llm_fn: LLMFn) -> AgentState: ...
def node_reflect(state: AgentState) -> AgentState: ...
```

---

## 5. 增强机制（业界吸收）

### 5.1 假设-因子语义对齐自检（来源 AlphaAgent）
编译通过后、评估前，LLM 轻量自查"表达式 E 实现了假设 H 吗？"。不一致的当场丢弃或要求修正——在昂贵的 IC 评估之前过滤"随机但语法对"的因子，省无效回测、提命中率。失败的对齐记入诊断，反馈给下一轮。

### 5.2 family-aware 家族多样性（来源 Hubble）
不仅控单因子 IC，还控**因子池的家族多样性**：避免 N 个 momentum 细微变体造成的假多样性。实现 = 复用 M1 贪心去相关的**相关矩阵** + numpy 阈值并查集分组，对同族聚集的候选施加惩罚（降权或要求 Agent 换族）。与 M1 的 top-K 去相关天然衔接。

### 5.3 Negative RAG 防重复探索（来源 Hubble）
**不是向量 RAG**：规模小（N≤20 轮、每轮几个表达式），负例召回 = 从 session 记忆按"AST 结构相似 + 低 IC"做**平面 Python 召回**，拼成"避免这些模式"注入下一轮 system prompt。无需 embedding/Faiss/向量库。

### 5.4 自我批判 critic（C 核心）
每轮候选过护栏后，LLM 以"挑剔的风控审计员"身份审视：这个因子是数据窥探吗？经济直觉站得住吗？样本够吗？→ 决定保留 / 丢弃 / 提变体。批判文本落 manifest。

### 5.5 多假设并行（C 核心）
每轮生成多个假设拓宽搜索面，各自独立评估，提高每轮 LLM 调用的产出。**"并行"指逻辑批量评估（一轮内多个假设的表达式批量过 IC/护栏），非线程并发——单 Agent 串行调度，保证可复现。**

---

## 6. 技术架构 / 技术栈

**核心：零新第三方依赖**，全部复用项目已有栈 + 自建轻量编排。

| 关注点 | 技术选型 | 新依赖 |
|---|---|---|
| Agent 编排 | 自建 `for` 循环 + `dataclass` State + 函数式节点 | 无 |
| LLM 客户端 | 复用 `llm/client.py`（`urllib`，OpenAI 兼容） | 无 |
| LLM 抽象 | `LLMFn = Callable[[list[dict[str,str]]], str]` 协议，可注入 | 无 |
| 结构化输出 | JSON（`response_format` + prompt 约束）+ `dataclass` 手写解析（仿 `parse_llm_explanation`），解析失败 = 该轮判无效并重试 | 无 |
| 表达式编译 | 复用 `discovery/expression.py` | 无 |
| IC 评估 | 复用 `discovery/scoring.py` + `run_session` 评估段 | 无 |
| 防过拟合 | 复用 `validation/`（DSR/PBO/holdout/TrialLedger） | 无 |
| Negative RAG | 平面字符串/AST 召回（结构相似 + 低 IC），非向量库 | 无 |
| family-aware | 复用相关矩阵 + numpy 阈值并查集分组 | 无 |
| 数据 | polars + `DataBundle`（M2 train/holdout 切分） | 无 |
| 持久化 | `manifest.json` + `candidates.csv` + parquet | 无 |
| 可复现 | `numpy.random.default_rng(seed)` | 无 |
| 测试 | pytest + `FakeLLM`（确定性 `Callable`） | 无 |
| CLI | argparse（扩展 `build_parser`） | 无 |

**LLM 接入细节**：OpenAI 兼容 `chat/completions`，温度/seed/max_tokens/base_url/api_key 全走现有 `.env` 的 `FACTORZEN_LLM_*`；模型可配（默认走用户配置的网关，推荐最新 Claude）。CI 默认 `FACTORZEN_LLM_ENABLED=false`，单测全程注入 `FakeLLM`，绝不触网。

**为何不用 LangGraph/LangChain**：单 Agent + 固定流程 + 已有评估设施，框架的多 Agent/动态路由/checkpoint/可观测能力一个都用不上，反而新增 15+ 传递依赖、要把 `urllib` 客户端包装成 `BaseChatModel`、FakeLLM 要实现框架接口。**升级退路**：节点已用 `node(State)→State` 形态（= LangGraph 节点签名），M6 若真要多 Agent 可平滑迁移，不锁死。

---

## 7. 防过拟合接入（灵魂约束）

| 约束 | 做法 |
|---|---|
| 诚实多重检验记账 | `TrialLedger` 的 N **累加 Agent 所有轮、所有表达式**（含被语义对齐/编译淘汰的尝试都计入"看过的假设空间"，DSR/PBO 据此收紧） |
| holdout 永久隔离 | Agent 全程**只见 mining 段**；holdout 段对 LLM、对反思、对 Negative RAG 完全不可见；仅最终候选验收用一次 |
| 候选证据链 | 每个穿过护栏的候选报告：holdout_ic、PBO、DSR、IC CI、与现有池相关性、family 归属 |
| family 多样性 | §5.2，防假多样性 |

> Agent 越能产，N 越大，门槛越严——这套机制保证"Agent 挖出来的东西可信"，是 M5 与 naive LLM 挖矿的分水岭。

---

## 8. 可复现 + manifest + human-in-the-loop

- **可复现**：`seed → numpy rng 序列 → FakeLLM 确定性返回`。同 seed + FakeLLM **逐字节复跑**。真实 LLM 温度 > 0 不保证逐字复现，但 manifest 全程可审计。
- **session manifest**（`agents/manifest.py`）：run_id、model、prompt 版本、温度、seed、universe、start/end、每轮 {假设, 表达式, 编译结果, 语义对齐, IC, 护栏结果, 批判}、最终候选、git SHA、耗时。
- **human-in-the-loop**：`--human-review` 每轮暂停，打印假设 + 候选 + 护栏结果，等人输入（保留哪些 / 继续 / 停）；默认 / CI 自动模式跳过（非交互）。

---

## 9. CLI + Leaderboard

```bash
fz mine agent --hypothesis "小市值低换手反转" --max-rounds 10 \
   --universe csi500 --start 20200101 --end 20231231 --seed 42 [--human-review]
```

- `_cmd_mine_agent(args) -> int`（仿 `_cmd_mine_search`，延迟 import）：拉数据 → `run_agent_mine(...)` → 打印候选数 + session 目录。
- 产物：`workspace/mine_agent/{run_id}/`（manifest.json + candidates.csv + exported/*.py）。
- **Leaderboard**：Agent 候选汇入现有 `fz mine leaderboard`（复用 M1 的 leaderboard 读取/排序）。

---

## 10. 接口契约

**新建：**

```python
# llm/generation.py
@dataclass
class FactorProposal:
    hypothesis: str
    expressions: list[str]
    rationale: str
def generate_factor_proposal(ctx: AgentContext, llm_fn: LLMFn, *, n_hypotheses: int = 1) -> list[FactorProposal]
def semantic_check(hypothesis: str, expression: str, llm_fn: LLMFn) -> tuple[bool, str]  # (一致?, 理由/修正)

# agents/state.py
@dataclass
class AttemptRecord:
    iteration: int; hypothesis: str; expression: str
    compile_ok: bool; ic_train: float | None; passed_guardrails: bool
    critic_verdict: str | None; error: str | None
@dataclass
class AgentState:
    seed: int; iteration: int = 0
    attempts: list[AttemptRecord] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    seen_expressions: set[str] = field(default_factory=set)   # session 记忆
    negative_examples: list[str] = field(default_factory=list)  # Negative RAG

# agents/orchestrator.py
LLMFn = Callable[[list[dict[str, str]]], str]
def run_llm_agent(bundle: DataBundle, llm_fn: LLMFn, *, hypothesis_seed: str,
                  n_rounds: int, seed: int, human_review: bool = False) -> AgentResult

# pipelines/agent_mine.py
def run_agent_mine(daily, *, hypothesis: str, n_rounds: int, seed: int,
                   universe: str, start: str, end: str, out_dir: str,
                   llm_fn: LLMFn | None = None, human_review: bool = False) -> dict
```

**复用**（writing-plans 阶段用 interface agent 精确化签名）：`expression.py` 编译、`scoring.py` Rank IC、`mining_session` 评估段（很可能需从 `run_session` 抽出一个 `evaluate_expressions(exprs, bundle) → scored` 复用函数，供 Agent 与原搜索共用）、`validation/`（TrialLedger/deflated_sharpe/pbo/holdout）、`export.export_candidate`、`llm/client.py` 请求（需扩展出通用 `request_chat(messages) → str`，现有 `request_llm_explanation` 是其特例）、`llm/config.LLMConfig`、`DataBundle`、`build_parser`。

---

## 11. 测试策略 + 验收（DoD）

- [ ] **节点单测**：每个 `node_*` 用 `FakeLLM` + 构造 `AgentState` 测确定性转移（生成/编译/语义对齐/评估/护栏/批判/反思），纯 mock 离线。
- [ ] **闭环端到端**：`run_llm_agent` 用 `FakeLLM`（预设假设+表达式序列）跑 N 轮 → 候选可复现（同 seed 逐字节一致）、manifest 字段齐、N 正确累加、holdout 未泄漏。
- [ ] **防过拟合断言**：Agent 所有尝试计入 N；holdout 段不进 mining 评估（反向验证：泄漏会让断言失败）。
- [ ] **增强机制有判别力**：语义对齐能拒不一致表达式；Negative RAG 负例确实进 prompt；family 分组能识别同族聚集——均非恒真。
- [ ] **CLI**：`fz mine agent` parser + handler smoke。
- [ ] **真实 LLM smoke**（手动，需 `FACTORZEN_LLM_*`）：跑通一次闭环，产出 manifest + 候选，人工核对假设/表达式合理。
- [ ] ruff/test 绿；`git add` 只含 M5 相关，不带 M0 未提交改动；提交 `rookiewu417`。

CI 离线：单测全程 `FakeLLM` 不触网；真实 LLM 为手动命令。

---

## 12. 建议实现顺序（为 writing-plans 铺垫）

> 先建状态与 LLM 生成地基，再逐节点拼闭环，最后接 pipeline/CLI。每步 TDD + FakeLLM。

1. **`agents/state.py`** + **`tests`**：`AgentState`/`AttemptRecord` dataclass + 序列化。
2. **`llm/generation.py`**：`FactorProposal` + `generate_factor_proposal` + `semantic_check`（FakeLLM 测 JSON 解析 + 容错）。
3. **`agents/nodes.py` 评估侧**：`node_generate / compile / semantic_check / evaluate`（复用 expression+scoring）。
4. **`agents/nodes.py` 验收侧**：`node_guardrails`（复用 validation + family-aware）+ `node_critic` + `node_reflect`（含 Negative RAG/记忆更新）。
5. **`agents/memory.py`**：session 记忆 + Negative RAG 平面召回 + family 并查集分组。
6. **`agents/orchestrator.py`**：`run_llm_agent` 主循环 + **N 累加 / holdout 隔离断言** + human-review。
7. **`agents/manifest.py`**：session 落盘。
8. **`pipelines/agent_mine.py`**：`run_agent_mine` 编排 + 真实 client 接线。
9. **CLI `fz mine agent`** + leaderboard 汇入。
10. **真实 LLM smoke** + README/plan 完成记录。

---

## 13. 范围外

- **多 Agent 角色协作**（Hypothesis/Coder/Critic/Risk Auditor/Librarian）→ M6。
- **残差边级记忆 / SSPM**（AlphaMemo）→ M6（MVP 用简单 session 记忆）。
- **MCTS 树搜索**（内存爆炸、复杂）。
- **Python 自由代码生成 + sandbox**（DSL 已够，避免安全面）。
- **LangGraph/LangChain 引入**（M6 多 Agent 时再评估；节点形态已预留迁移口）。
- **向量 RAG / embedding 知识库**（规模小，平面召回够）。
- **跨 session 长期记忆库**（M6）。

---

*M5 完成后，FactorZen 拥有"假设 → 生成 → 评估 → 反思"的 LLM 闭环因子挖掘，全程可审计、可复现、CI 可测，且严守 M2 防过拟合护栏——这是简历级的差异化成果，并为 M6 多 Agent 协作铺路。*

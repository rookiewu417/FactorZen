# M2 · 防过拟合护栏 — 设计文档

> 状态：设计已评审通过（2026-06-30），待转实现计划。
> 上游：[FactorZen 升级计划](../../FactorZen-升级计划.md) 的里程碑 **M2**；建立在 [M1 因子挖掘引擎](../plans/2026-06-29-m1-factor-mining-engine.md) 之上。
> 定位：给 M1「会量产好看因子」的挖掘流水线套上针对数据窥探的统计护栏，让挖出的因子**可信**。这是计划所说「高手与业余的分水岭」，也是后续 Agent 挖掘（M5）可信的前提。

---

## 1. 目标与定位

给 M1 挖掘流水线套上**针对多重检验/数据窥探的统计护栏**：永久隔离的 OOS holdout（挖掘全程不可见，仅验收用一次）+ 多重检验记账 + Deflated Sharpe Ratio + PBO + bootstrap 置信区间，让挖出的因子附带可信度证据；并顺带对齐 M1 deferred 的 IC parity 与 genetic 多重检验 N 记账。

### 1.1 已拍板决策（评审结论）

| 决策 | 选择 | 理由 |
|---|---|---|
| M2 范围 | 地基 + 高 ROI 检验 | OOS holdout 隔离 + 记账 + DSR + PBO + bootstrap，接回 M1；Reality Check/SPA、regime、参数高原留增量 |
| 护栏统计基础 | **IC 序列**（非每候选跑回测） | 挖掘原生产出 IC 序列，快；IC 即因子日度表现，做 PBO/DSR 统计成立 |
| OOS holdout 隔离 | **软隔离 + 纪律 + 测试保证** | 硬隔离 overkill；mining 段/holdout 段切分，holdout 不传入挖掘循环，测试断言不可见 |
| holdout 比例 | 默认 **最后 20%**（可配置） | 时间序最后一段做永久 holdout |
| DSR / PBO 层级 | DSR 单因子级、PBO 候选池级 | 标准做法：DSR 评单因子显著性，PBO 评「从池中选最优」的过拟合概率 |
| 独立命令 | 含轻量 `fz validate overfit <factor>` | 对单个已有因子算 DSR + bootstrap CI（PBO 不适用单因子，跳过） |

---

## 2. 非目标（留增量）

- White Reality Check / Hansen SPA（与 PBO 的多重比较控制有重叠）。
- regime（牛熊）稳定性、参数高原可视化。
- 把护栏接入非挖掘的常规 `fz factor run` 主线（M2 只接挖掘流程 + 独立 validate 命令）。
- 硬隔离 holdout（物理分库）。

---

## 3. 模块结构

```text
src/factorzen/validation/              （新增，独立可复用的防过拟合库）
├── __init__.py
├── deflated_sharpe.py    DSR：Sharpe/IR + 试验数 N + 样本长度 + 偏度峰度 → (dsr, pvalue)
├── pbo.py                PBO(CSCV)：候选 × 日度 IC 矩阵 → 回测过拟合概率
├── bootstrap.py          block bootstrap：IC 序列 → IC 均值 95% 置信区间
├── multiple_testing.py   多重检验记账：尝试总数 N，喂给 DSR、报告「从 N 选出」
└── holdout.py            时间切分（mining/holdout）+ 隔离校验 helper

接回 M1（修改）：
├── discovery/mining_session.py   mining/holdout 切分 + top-K 在 holdout 验收 + 落护栏指标 + 修正 genetic N 记账
└── pipelines/factor_mine.py      run_mine 增加 holdout_ratio 参数

CLI（修改 cli/main.py）：
├── fz validate overfit <factor> --start --end   对单因子算 DSR + bootstrap CI
└── fz mine search                                leaderboard/manifest 增列护栏指标

tests/
├── test_validation_deflated_sharpe.py
├── test_validation_pbo.py
├── test_validation_bootstrap.py
├── test_validation_multiple_testing.py
├── test_validation_holdout.py
├── test_discovery_session.py（扩展：holdout 隔离 + 护栏指标）
└── test_validation_cli.py
```

---

## 4. OOS holdout 隔离机制（M2 的灵魂）

**软隔离 + 纪律 + 测试保证**：

- `holdout.py` 的 `split_holdout(daily, holdout_ratio=0.2) -> (mining_df, holdout_start_date)`：按交易日时间序，最后 `holdout_ratio` 比例的日期为 holdout，其余为 mining 段。
- `run_session` **只把 mining 段传入挖掘循环**（DataBundle、搜索、去相关全部只见 mining 段）。holdout 段数据**不进入任何挖掘逻辑**。
- 仅当 top-K 选定后，调 `holdout_evaluate(top_k, holdout_df)` 算一次 holdout IC/DSR/bootstrap —— **holdout 只用一次**。
- **隔离测试**（`test_validation_holdout.py` + session 测试）：断言挖掘期任何 DataBundle 的最大 `trade_date` < holdout_start（holdout 真的不可见），且 holdout 评估只被调用一次。

---

## 5. 各护栏组件（接口契约）

### 5.1 `deflated_sharpe.py`
```python
def deflated_sharpe(
    sharpe: float,        # 观测 Sharpe 或 annualized IR
    n_trials: int,        # 多重检验记账提供的真实尝试数
    n_obs: int,           # 样本期数（IC 序列长度）
    skew: float = 0.0,    # IC 序列偏度
    kurt: float = 3.0,    # IC 序列峰度
) -> tuple[float, float]:  # (deflated_sharpe_ratio, p_value)
```
用 Bailey & López de Prado 公式：先由 `n_trials` 估计「期望最大 Sharpe」作为 deflation 基准，再算观测 Sharpe 超越它的概率。`p_value < 0.05` 视为扣除多重检验后仍显著。

### 5.2 `pbo.py`（CSCV）
```python
def compute_pbo(
    perf_matrix: np.ndarray,  # shape (n_candidates, n_periods)，每行一个候选的日度 IC
    n_splits: int = 10,       # CSCV 块数 S（偶数）
) -> float:                   # PBO ∈ [0,1]，越高越过拟合
```
把时间分 S 块，枚举 C(S, S/2) 种 IS/OOS 划分；每种里在 IS 找最优候选（最高 IS 平均 IC），记其 OOS 相对秩 → logit；`PBO = P(logit ≤ 0)`（IS 最优在 OOS 落后半区的频率）。S 默认 10（C(10,5)=252 组合，可控）。

### 5.3 `bootstrap.py`
```python
def block_bootstrap_ic_ci(
    ic_series: np.ndarray,   # 日度 IC
    block_size: int = 10,    # 块长（保留时序自相关）
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:    # (ci_low, ci_high) of mean IC
```
moving block bootstrap，保留 IC 的时序相关；`ci_low ≤ 0` 触发警示。

### 5.4 `multiple_testing.py`
```python
@dataclass
class TrialLedger:
    n_trials: int = 0
    def record(self, k: int = 1) -> None: ...   # 累加真实评估候选数
```
挖掘 session 用它记录**真实评估的候选数**（修正 M1 deferred：genetic 用 `pop_size*generations` 实际评估数，而非 CLI `n_trials`）；该 N 喂给 DSR，并在 leaderboard/manifest 报告「从 N 个候选选出 top-K」。

---

## 6. 数据流（接回 M1）

```text
run_mine(start, end, holdout_ratio=0.2, ...):
  1. split_holdout(daily, holdout_ratio) → mining_df, holdout_start
  2. 挖掘(M1：random/GP + train/valid + 去相关)只在 mining_df → top-K + TrialLedger.n_trials
     ★ holdout 数据不传入，挖掘循环不可见
  3. 护栏评估（holdout 只用一次）—— 三者各司其职：
       - DSR（挖掘表现的多重检验显著性）：每个 top-K 用其 **mining/train 段 IR** + n_trials N → deflated_sharpe
       - PBO（选优过拟合概率）：候选池 **mining 段** IC 矩阵 → compute_pbo
       - holdout OOS 验证：每个 top-K 在 **holdout 段** 算 holdout_ic + block_bootstrap_ic_ci(holdout IC 序列)
  4. leaderboard/manifest 增列：n_trials(N) / pbo / 每候选 holdout_ic / dsr_pvalue / ic_ci_low
     报告显式警示：holdout IC 大幅衰减 / PBO 高(>0.5) / DSR 不显著(p≥0.05) / ic_ci_low≤0
```

**IC parity 对齐**（M1 deferred）：mining 段 IC 与 holdout 段 IC 用**同一预处理口径**（plain cross_sectional_zscore），且都从 `start` 之后取值（排除 lookback 预热）；leaderboard 注明该口径，复现指引保持 `--set preprocessing.neutralize=false`。

---

## 7. 接口契约（复用现有）

| 用途 | 复用接口 | 位置 |
|---|---|---|
| IC 序列 | `compute_rank_ic(...).ic_series`（列 trade_date, ic） | `daily/evaluation/ic_analysis.py` |
| 快速 IC/IR | M1 `quick_fitness` / `DataBundle` | `discovery/scoring.py` |
| top-K 候选 | M1 `run_session` 的 scored/top | `discovery/mining_session.py` |
| 因子值求值 | M1 `_factor_values` / `compile_expr` | `discovery/`、`expression.py` |
| CLI 接入 | argparse `build_parser`，仿 `fz mine` | `cli/main.py` |

新增 `validation/` 模块不依赖 M1（纯统计函数 + holdout helper），M1 的 `mining_session` 反向依赖 `validation/`。

---

## 8. 测试策略

- **构造验证（关键）**：
  - 纯噪声因子（随机，与未来收益无关）→ DSR 不显著（p≥0.05）、PBO ≈ 0.5、bootstrap CI 跨 0。
  - 强信号因子（= 次日收益）→ DSR 显著、PBO 低（<0.2）、CI 全正。
- **holdout 隔离测试**：断言挖掘期 DataBundle 最大日期 < holdout_start；holdout 评估只调用一次。
- **PBO/DSR/bootstrap 数值正确性**：对小型已知矩阵对拍（PBO 对称性、DSR 随 n_trials 单调收紧、CI 覆盖）。
- 端到端：`fz mine search` 产出带护栏指标的 leaderboard；同 seed 可复现。
- 全离线 mock，无磁盘/网络；ruff + typecheck 绿（提交前自查 `ruff check src/factorzen/validation/ src/factorzen/discovery/`）。

---

## 9. 验收标准（DoD）

- [ ] `validation/` 提供 DSR / PBO / bootstrap / 记账 / holdout，均有构造验证测试。
- [ ] 挖掘流程 holdout **永久隔离**有测试保证（挖掘期不可见 holdout）。
- [ ] `fz mine search` 的 leaderboard/manifest 含 N / pbo / holdout_ic / dsr_pvalue / ic_ci，并对过拟合风险显式警示。
- [ ] `fz validate overfit <factor>` 对单因子输出 DSR + bootstrap CI。
- [ ] 多重检验 N 对 genetic 路径准确（修正 M1 deferred）。
- [ ] IC parity 口径对齐并在 leaderboard 注明。
- [ ] 同 seed 可复现；ruff/typecheck/test 绿；`git status` 干净（只提交 M2 文件）。

---

## 10. 建议实现顺序（为 writing-plans 铺垫）

> 先做无依赖的纯统计函数（各自独立可测），再做 holdout 切分，最后接回 M1 与 CLI。

1. **`multiple_testing.py`**（TrialLedger，最简单，无依赖）。
2. **`bootstrap.py`**（block bootstrap IC CI）。
3. **`deflated_sharpe.py`**（DSR + p 值，构造验证）。
4. **`pbo.py`**（CSCV，构造验证 + 对称性）。
5. **`holdout.py`**（split_holdout + 隔离校验 helper + holdout_evaluate）。
6. **接回 `mining_session` / `run_mine`**：holdout 切分 + top-K 验收 + 落护栏指标 + 修正 genetic N + IC parity（端到端 + 隔离测试）。
7. **CLI**：`fz validate overfit` + leaderboard 增列护栏指标。

---

*本设计建立在 M1 已验证的接口之上；护栏基于 IC 序列、holdout 软隔离均为评审拍板决策。M2 完成后，挖掘流水线即具备「可信」属性，可支撑 M5 Agent 挖掘。*

# 风险与组合优化

> [FactorZen](../../README.md) · [文档](../README.md) · **风险与组合优化**

因子入库之后的下一步：用风险模型刻画暴露，用凸优化把 α 信号翻译成可执行的目标权重，再把结果拆回因子与行业。

本页覆盖 `fz risk build`、`fz portfolio build` 与随组合一起落盘的归因产物。参数全表见 [CLI 参考](../reference/cli.md#fz-risk)；因子从哪来见[因子库与增量准入](../concepts/factor-library.md)。

> ⚠️ **先读这一段再决定要不要用。** 组合优化与归因是本平台**当前最轻的能力**——`portfolio/` 与 `attribution/` 合计只有 216 行，配对的 `daily/optimization/` 也只有 348 行，而挖掘侧的 `discovery/` 是 12,645 行。这个落差是真实的能力权重分布，不是「重实现藏在别处」。风险模型本身（`risk/`，1,737 行）成熟度明显更高，但**只服务 A 股**。把本页的能力当作「可用的 MVP」而非「生产级组合管理系统」。

---

## 能力边界速查

| 能力 | 状态 |
|---|---|
| Barra 风险模型（8 风格 + 行业） | 完整，**仅 A 股** |
| 风险模型接入多市场 Port | **未接入**（`markets/ashare/profile.py` 的 `risk=None`） |
| crypto 风险 | 独立实现 `markets/crypto/risk.py`，与 `risk/` 不共用 |
| futures / us 风险 | **无风险模型** |
| 组合优化 `fz portfolio build` | 可用，**仅 `ashare` / `crypto`** 两个市场 |
| 行业中性 | 相对**等权**行业基准，不是市值加权中性 |
| 收益归因 | Brinson-Fachler **两项法**，交互项并入选股；不提供 BHB 三项法 |
| 建仓时点的收益归因 | **占位 0**，见[归因的可用性边界](#归因的可用性边界) |
| 换手约束 | 需要上一期权重，单次 `fz portfolio build` **不生效**（会告警） |

---

## 风险模型：`fz risk build`

### 模型形态

标准的 Barra 式截面多因子风险模型。每个交易日跑一次截面回归：

```text
ret_i = X_i · f + eps_i
```

- `X_i` —— 股票 i 的因子暴露行（8 个风格因子 + 行业哑变量）
- `f` —— 该日的因子收益（回归系数）
- `eps_i` —— 特质收益（残差）

收集 `f_t` 的时间序列后估协方差，收集残差矩阵后估特质风险。

### 8 个风格因子

单一真源是 `src/factorzen/risk/style_factors.py` 的 `STYLE_FACTOR_NAMES`：

| 因子 | 定义 | 实现 |
|---|---|---|
| `size` | `ln(total_mv)` | `style_factors.py` |
| `value` | `1/pb`（Book-to-Price） | `style_factors.py` |
| `momentum` | 252 日累计收益，**跳过最近 21 日** | `style_factors.py` |
| `volatility` | 60 日收益率标准差 | `style_factors.py` |
| `liquidity` | `ln(20 日换手率均值)` | `style_factors.py` |
| `quality` | ROE 近似 = `pb / pe_ttm` | `style_factors.py` |
| `growth` | 盈利增长近似 = `pe_ttm` 倒数的变化率 | `style_factors.py` |
| `leverage` | 杠杆近似 = `pb − 1`（净资产乘数代理） | `style_factors.py` |

> ℹ️ `quality` / `growth` / `leverage` 三个是**估值指标的代理近似**，不是从财报三表直接构造的 Barra 原版定义。它们只依赖 `daily_basic`，好处是无需财务数据即可构建，代价是经济含义比原版弱。

每个因子在截面上先 Winsorize 再 Z-score 标准化（`cs_standardize`，`style_factors.py`，默认 MAD 法）。

行业侧用 one-hot 哑变量列，列名统一加 `ind_` 前缀（`risk/model.py` 的 `_normalize_ind_cols`），并在整个窗口上取**行业并集**、缺列补 0——避免因子集在不同交易日之间漂移导致丢日。

### 协方差与特质风险

**因子协方差**（`risk/covariance.py`）：

1. 指数加权去均值，衰减 `lam = 0.5 ** (1 / half_life)`
2. 加权样本协方差
3. **Newey-West 自相关修正**，Bartlett 核权重 `1 − lag/(nw_lags+1)`
4. 对称化 + 特征值截断到 ≥0，保证半正定

**特质风险**（`risk/covariance.py`）：

1. 每只股票独立算指数加权残差方差（时序估计）
2. 取所有股票 `ts_var` 的截面均值（截面估计）
3. **贝叶斯收缩**：`blended = (1 − shrinkage) · ts_var + shrinkage · cs_mean_var`
4. 开方得标准差向量

收缩的作用是把样本量不足、残差方差估计极端的个股拉回截面中枢。默认 `shrinkage=0.3`。

模块里还有一个 Monte Carlo 特征值调整 `eigenvector_adjustment`（`risk/covariance.py`），当前**不在 `fz risk build` 的默认链路上**，属于可选校正工具。

### 风险预测与 MCR 分解

`RiskModel.predict_risk`（`risk/model.py`）给年化总波动：

```text
σ² = wᵀ X F Xᵀ w + wᵀ D² w
σ_annual = sqrt(σ²) × sqrt(periods_per_year)
```

`RiskModel.decompose_risk`（`risk/model.py`）在此基础上做边际风险贡献分解，返回一个字典：

| 键 | 口径 |
|---|---|
| `total_risk` / `factor_risk` / `specific_risk` | **标准差**口径，年化 |
| 各因子名（`size` / `ind_银行` / …） | **MCR 份额**口径：`(Xw)_i · (F·Xw)_i / total_var × total_std × ann` |

> ⚠️ **两种口径不能相加。** `total_risk` / `factor_risk` / `specific_risk` 是标准差，各因子名下的值是 MCR 份额。把 `size` 的值和 `specific_risk` 加起来没有意义。`decompose_risk` 的 docstring 就是这么标注的。

年化周期数 `periods_per_year` 默认 252（A 股日频），crypto 侧传 365。

### 运行

```bash
pixi run -- fz risk build --start 20200101 --end 20241231 --universe all_a \
  --cov-half-life 90 --nw-lags 2 --spec-half-life 90 --spec-shrinkage 0.3
```

本命令**没有 `--market`**，是 A 股专属链路。

**产物**落 `workspace/risk_models/risk_<start>_<end>/`：

| 文件 | 内容 |
|---|---|
| `exposures.parquet` | 最新一期暴露矩阵，`ts_code` + 各因子列 |
| `factor_covariance.parquet` | 因子协方差矩阵 |
| `specific_risk.parquet` | `ts_code` + `specific_risk`（标准差） |
| `factor_returns.parquet` | 因子收益时间序列 |
| `risk_summary.csv` | 长表 `section / metric / value`：因子波动、特质风险分布、R²、风格暴露统计、等权组合风险分解 |
| `manifest.json` | 窗口、参数、`r_squared`、`factor_names`、`n_valid_dates`、`n_factor_mismatch`、`equal_weight_decomp` |

### 别踩的坑

> ⚠️ **`fz risk build` 的产物当前没有下游消费方。**
> `RISK_MODELS_DIR` 在全仓只有一个生产者（`pipelines/risk_build.py`），`src/` 内**零读取方**。
> `fz portfolio build` **不会**去读 `workspace/risk_models/`，而是在进程内自建一个 `RiskModel()` 重算（`cli/main.py` 的 `_cmd_portfolio_build()`）。
>
> 也就是说这两条命令看起来像流水线的上下游，实际是**各自独立**的。`fz risk build` 的用途是产出可审计的风险模型快照供人查看与分析，不是 `portfolio build` 的前置步骤——跳过它直接 `portfolio build` 完全可行，结果不受影响。想复用磁盘上的风险模型需要自己写代码加载。

> ⚠️ **回归窗口需要预热。** `momentum` 用 252 日滚动窗、`volatility` 用 60 日窗。如果只拉 `[start, end]` 的行情，窗口早期这些因子全空，模型会静默退化成少数几个非滚动因子。CLI 已经自动补足：`load_risk_inputs`（`pipelines/risk_build.py`）会往前多拉 **420 个日历日**（`risk_lookback_start`，约覆盖 252 个交易日 + 春节余量），回归区间本身不变。**自己调 `RiskModel.build` 时必须自己补这段历史。**

> ⚠️ **看 `n_factor_mismatch`。** manifest 里这个字段 > 0 表示有交易日因为因子集与全局固定集不一致被跳过，日志会打 `[DEGRADED]` 告警。正常应当 ≈ 0，非零说明行业并集或 reindex 有问题，此时协方差是在残缺样本上估的。

---

## 组合优化：`fz portfolio build`

### 目标函数

`portfolio/optimizer.py` 的 `optimize_portfolio`，因子风险模型形式的 mean-variance QP，用 **cvxpy + CLARABEL** 求解：

```text
max   αᵀw − risk_aversion · ( (Xᵀw)ᵀ F (Xᵀw) + Σ (D_i w_i)² )
                              └── 因子风险 ──┘   └─ 特质风险 ─┘
```

用因子形式而不是全 Σ，是因为 `n × n` 的股票协方差矩阵在全 A 规模下既难估准也难求解；`X F Xᵀ + D²` 把维度压到因子数。

求解前对 `F` 做对称化 + 特征值 clip 到 ≥0（`_psd`，`optimizer.py`），满足 cvxpy `quad_form` 的 PSD 要求。

**状态处理**：`optimal` 与 `optimal_inaccurate`（CLARABEL 的 AlmostSolved）都视为可用解。`SolverError` / `DCPError` 被捕获后返回 `status="error"` 而不是抛出——这组输入解不出来不是程序错误，不应该炸掉整条 pipeline。

> ⚠️ **`risk_aversion` 的缩放约定与 `daily/optimization/mean_variance.py` 差 2 倍。** 那边的目标是 `wᵀμ − (λ/2)·wᵀΣw`（含 1/2），这边没有。同一个数值在两处对应的实际风险惩罚强度差一倍，**调参经验不能跨模块套用**。这条写在 `optimizer.py` 的 docstring 里。

### 约束体系

`portfolio/constraints.py` 的 `build_constraints`，由 `ConstraintConfig` 驱动。模块 docstring 归的是四类（box / budget / 中性 / 换手），实现里另有一条为 crypto 做空组合准备的杠杆约束：

| 类别 | 约束 | 配置字段 | CLI |
|---|---|---|---|
| **budget** | `Σw == budget` | `budget`（A 股 = 1；crypto 市场中性 = 0；`None` 不约束） | 由 `--market` 决定 |
| **box** | `w ≤ w_max`；long-only 时 `w ≥ 0`，否则 `w ≥ −w_max` | `w_max` (0.05)、`long_only` (True) | `--w-max` |
| **杠杆** | `Σ\|w\| ≤ gross_limit` | `gross_limit` | `--gross-limit`（crypto） |
| **中性** | `X_sᵀ w == X_sᵀ w_bench`（或 `== 0`） | `neutral_factors`、`benchmark_weights` | `--industry-neutral` |
| **换手** | `‖w − w_prev‖₁ ≤ turnover_budget` | `turnover_budget`、`prev_weights` | `--turnover` |

### 行业中性为什么是「相对等权基准」

这是本平台反复踩过的一个坑，代码里有 `.. warning::` 专门标注（`constraints.py`）：

**绝对中性到 0 + long-only + `Σw = 1` 必然无解。** 行业哑变量是 one-hot 的 0/1 列，「所有行业暴露为 0」意味着组合在每个行业的总权重都是 0，与 `Σw = 1` 直接矛盾，求解器只会返回 infeasible。

所以 `--industry-neutral` 的实现是**中性到基准暴露**：

```text
X_sᵀ w == X_sᵀ w_bench
```

而 `w_bench` 当前取的是 **universe 等权**（`cli/main.py` 的 `_cmd_portfolio_build()`：`np.full(len(codes), 1.0/len(codes))`）。

> ⚠️ **这不等同于市值加权指数的行业中性。** 等权基准的行业分布与 CSI300/CSI500 的市值加权行业分布可以差很多——小市值股票多的行业在等权基准下权重被放大。真实指数基准权重是后续扩展项，当前是 MVP。任何把 `--industry-neutral` 解读成「对标指数的行业中性」的分析都会偏。

### 换手约束的静默失效

换手约束需要**同时**有 `turnover_budget` 和 `prev_weights` 才会加进 cvxpy 问题。单次 `fz portfolio build` 没有上一期权重，只传 `--turnover` 会被静默丢弃。

平台对此做了显式告警（`pipelines/portfolio_build.py`）：

```text
[portfolio] ⚠ 设了 turnover_budget=... 但无 prev_weights，换手约束不生效
```

> ✅ 要真正生效，走多期链路：`fz research run` 会按调仓日循环、把上一期权重串进下一期（`pipelines/research_run.py`）。或者自己调 `run_portfolio` 时显式传 `prev_weights`。

### 运行

```bash
pixi run -- fz portfolio build --start 20200101 --end 20241231 --universe all_a \
  --alpha-file workspace/alpha/20241231.parquet \
  --lam 1.0 --w-max 0.05 --industry-neutral --run-id 20241231
```

α 信号文件是 `ts_code` + `alpha` 两列的 parquet/csv，通常来自 `fz mine export-alpha`。对齐时以风险模型的 `codes` 顺序为准，**缺失的股票填 0**。

> ⚠️ **`--run-id` 不传会用 `--end` 的日期串做目录名。** 做多期构建时忘了区分，后一期会静默覆盖前一期。多期循环务必显式传不同的 `--run-id`。

> ⚠️ `--market` 在这里只有 `{ashare, crypto}` 两个取值，与 `fz mine` / `fz factor-library` 的四值域不同。crypto 走的是另一条实现（`markets/crypto/portfolio.py`，市场中性做空 + `gross_limit`），不经过 `risk/` 的 Barra 模型。

**产物**落 `workspace/portfolios/<run_id>/`：

| 文件 | 内容 |
|---|---|
| `weights.parquet` | `ts_code` / `target_weight` / `prev_weight` |
| `attribution.csv` | 长表 `type / key / value`：`factor_return` · `specific_return` · `brinson_allocation` · `brinson_selection` |
| `risk_summary.csv` | `metric / value`，即 `decompose_risk` 的输出 |
| `manifest.json` | `status`、`objective`、`n_holdings`、约束参数、`turnover`、`return_attribution_available` |

求解失败（`status` 非 optimal）时权重落全 0，`attribution.csv` / `risk_summary.csv` 写空表——**先看 manifest 的 `status` 再看权重**，否则会把「解不出来」误读成「清仓信号」。

---

## 归因

组合构建时同步产出两套归因，都写进 `attribution.csv`。

### 收益归因：Brinson-Fachler 两项法

`attribution/brinson.py`。按行业分组，每个行业：

```text
配置效应 = (w_p − w_b) · (r_b_sector − r_b_total)
选股效应 = w_p · (r_p_sector − r_b_sector)
```

守恒关系：`Σ(配置 + 选股) = 组合收益 − 基准收益`。

> ⚠️ **这是 Brinson-Fachler（BF）两项法，不是 Brinson-Hood-Beebower（BHB）三项法。** 两点区别必须写清楚，否则跨平台对比会得出错误结论：
> 1. **配置效应的基准不同**——BF 用 `r_b_sector − r_b_total`（行业基准收益减去总基准收益），BHB 用 `r_b_sector` 本身。
> 2. **没有独立的交互项**——BF 把交互效应并入选股（选股用 `w_p` 而非 `w_b` 加权）。
>
> 平台历史上曾在单因子评估里有过独立的 BHB 三项法实现，已随单因子评估精简一并摘除。**需要三项法必须另行实现，不要混进 `attribution/brinson.py`**（该模块 docstring 明确要求）。

### 风险因子归因

`attribution/risk_attribution.py`。把组合收益与风险都拆到风格/行业因子上：

- **收益贡献**：`factor_return_contrib[j] = (Xᵀw)_j × f_j`，即组合在因子 j 上的暴露乘该因子当期收益
- **特异收益**：`specific_return = 组合实际收益 − Σ 因子贡献`（残差口径，不是独立估的）
- **风险贡献**：直接复用风险模型的 `decompose_risk`，即上文的 MCR 份额

### 归因的可用性边界

> ⚠️ **`fz portfolio build` 产出的收益归因在建仓时点是占位 0。** 命令构建的是**未来**要持有的目标权重，此刻还没有持仓期收益可用——`cli/main.py` 的 `_cmd_portfolio_build()` 传的就是 `stock_returns=np.zeros(...)` 和 `factor_returns_latest={}`。因此：
>
> - `brinson_allocation` / `brinson_selection` / `factor_return` 三类行全是 0
> - **`risk_summary.csv`（风险归因）仍然有效**，它不依赖收益
> - manifest 里 `return_attribution_available: false`，并附 `return_attribution_note` 说明原因
>
> **判读前先看 manifest 这两个字段。** 要拿到有意义的收益归因，必须提供持仓期的股票收益与因子收益——即在模拟/执行跑完之后回过头做，而不是在建仓瞬间。

其他限制：

- 归因是**单期**的，多期需要自行按期累加，模块不提供跨期链接（geometric linking）。
- 不支持日内高频归因。行业分类取自 universe 快照的 `industry` 列，缺失记为空串 `""` 归为一组。

---

## 两条优化路径：故意分离，不要合并

平台里有两套组合优化实现，这是**有意为之**：

| | `portfolio/`（121 行） | `daily/optimization/`（348 行） |
|---|---|---|
| 风险刻画 | **因子形式** `X F Xᵀ + D²` | **全 Σ** 股票协方差矩阵 |
| 目标 | `αᵀw − λ · wᵀΣw` | `wᵀμ − (λ/2) · wᵀΣw` |
| 方法 | 单一 mean-variance QP | `MeanVarianceOptimizer` / `MaxSharpeOptimizer` / `RiskParityOptimizer` |
| 协方差估计 | 来自 Barra 风险模型 | 自带 `sample` / `ewma` / `ledoit_wolf` 收缩 |
| 约束 | budget / box / 中性 / 换手 / 杠杆 | min/max weight / net / gross / turnover |
| 求解失败 | 返回 `status="error"`，权重 `None` | **回退**上期权重或等权 |
| 入口 | `fz portfolio build`、`fz research run` | 日频回测策略内部 |

> ⚠️ **不要合并这两条路径。** 它们服务不同场景（组合级建仓 vs 回测内的逐期建仓），风险刻画与失败语义都不同。**唯一需要保持一致的是 optimizer status 的口径**——两侧都必须把 `optimal_inaccurate` 当作可用解接受。历史上有一次事故正是因为一侧拒收 `optimal_inaccurate`、退化成全零权重，被下游 `sim/` 当作真实清仓信号执行了。

---

## 典型流程

从因子库到组合的一条完整路径：

```bash
# 1) 构建风险模型（可选：portfolio build 内部会自己建一次，这步是为了单独审阅 risk_summary）
pixi run -- fz risk build --start 20200101 --end 20241231 --universe csi500

# 2) 从挖掘 session 导出某个截面日的 α
pixi run -- fz mine export-alpha \
  --session workspace/mine_team/20260718_120000 --rank 1 \
  --date 20241231 --universe csi500 --out workspace/alpha/20241231.parquet

# 3) 求解目标权重
pixi run -- fz portfolio build --start 20200101 --end 20241231 --universe csi500 \
  --alpha-file workspace/alpha/20241231.parquet \
  --lam 1.0 --w-max 0.05 --industry-neutral --run-id 20241231

# 4) 用权重跑模拟交易（--portfolio-dir 传的是根目录）
pixi run -- fz sim run --portfolio-dir workspace/portfolios \
  --start 20200101 --end 20241231

# 5) 出组合 Dashboard（--portfolio-dir 传的是单个 run 目录）
pixi run -- fz report portfolio \
  --sim-dir workspace/sim/<run_id> \
  --portfolio-dir workspace/portfolios/20241231
```

> ⚠️ 第 4 步与第 5 步的 `--portfolio-dir` **同名异义**：`fz sim run` 要的是**根目录**（其下各 `{run_id}/` 组成调仓日程），`fz report portfolio` 要的是**单个 run 目录**。传错读不到文件。

> ✅ 想要多期调仓 + 换手约束真正生效，用 `fz research run` 一条命令跑完挖掘 → 逐调仓日建仓 → 模拟 → 报告。注意它目前是**单因子 + in-sample** 的编排，不是多因子组合链路。

---

## 相关阅读

- [因子库与增量准入](../concepts/factor-library.md) —— 进入组合的因子从哪来
- [多因子组合](combination.md) —— 四方法样本外对比，与本页的单 α 建仓是两条不同的路
- [模拟与向前执行](execution.md) —— 目标权重之后怎么落地成交易
- [CLI 参考](../reference/cli.md#fz-portfolio) —— `fz risk build` / `fz portfolio build` 参数全表
- [产物布局](../reference/artifacts.md) —— `workspace/risk_models/` 与 `workspace/portfolios/` 字段

# 配置参考

> [FactorZen](../../README.md) · [文档](../README.md) · **配置参考**

FactorZen 的配置分三层，越靠后优先级越高：

| 层 | 载体 | 作用域 | 谁定义 |
|---|---|---|---|
| 路径常量 | `src/factorzen/config/settings.py` | 全仓所有产物/数据路径 | 代码，不可配置 |
| 全局常量 | `src/factorzen/config/constants.py` | 交易日历、成本率、涨跌停等魔法数字 | 代码，不可配置 |
| 研究运行配置 | `src/factorzen/config/research.py`（pydantic v2） | 单因子评估的一次 run | YAML + CLI 旗标 + `--set` |

本页只讲**第三层**（用户真正会改的部分）以及它与前两层的交界。产物落盘位置见 [产物布局](artifacts.md)，环境变量见 [环境变量](environment.md)。

---

## 1. 配置模型总览

顶层模型是 `RunConfig`（`src/factorzen/config/research.py:92`），四个嵌套节：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `factor` | `str` | **必填** | 因子名，对应内置因子或 `workspace/factors/` 下的 python 因子 |
| `universe` | `str` | `"csi500"` | 股票池名 |
| `start` / `end` | `str` | **必填** | 日期，格式 `YYYYMMDD`（YAML 里务必加引号，否则被解析成整数） |
| `benchmark` | `str \| None` | `None` | 留空则按 `universe` 推导，见下 |
| `seed` | `int \| None` | `None` | 复现种子 |
| `preprocessing` | `PreprocessingConfig` | 见 §1.1 | 去极值 / 标准化 / 中性化 |
| `backtest` | `BacktestConfig` | 见 §1.2 | 策略组、成本模型、权重约束 |
| `walk_forward` | `WalkForwardConfig` | 见 §1.3 | 滚动向前验证 |

> ℹ️ `RunConfig` 沿用 pydantic 默认的 `extra="ignore"`。旧 YAML 里的 `ic_method`、`event_study`、`neutralized_ic` 等字段会被**静默忽略**，不会报错——升级配置时别指望校验器提醒你字段已废弃（`research.py:93`）。

`benchmark` 留空时的推导表（`research.py:12-20`）：

| universe | 推导出的 benchmark |
|---|---|
| `csi300` | `000300.SH` |
| `csi500` | `000905.SH` |
| `csi800` | `000906.SH` |
| 其它（含 `all_a`） | 兜底 `000300.SH` |

### 1.1 `preprocessing`

`PreprocessingConfig`（`research.py:23`）：

| 字段 | 取值域 | 类默认 |
|---|---|---|
| `outlier` | `mad` / `winsorize` / `sigma` | `mad` |
| `normalizer` | `zscore` / `rank_uniform` / `rank_normal` / `quantile_normal` | `zscore` |
| `neutralize` | `bool` | `false` |
| `neutralize_by` | `industry` / `size` / `industry+size` | `industry+size` |

> ⚠️ **`neutralize` 有两个不同的"默认"，这是本项目最容易踩的配置坑**，详见 §4。

> ℹ️ 行业中性是相对**等权**行业基准的中性化，不等同于市值加权行业中性。这是当前的已知简化。

### 1.2 `backtest`

`BacktestConfig`（`research.py:41`）：

| 字段 | 取值域 | 默认 | 说明 |
|---|---|---|---|
| `top_n` | `int` | `50` | legacy 单策略字段，见下方自动生成规则 |
| `quantiles` | `int` | `5` | 分位统计用 |
| `max_abs_weight` | `float` | `0.1` | 个股绝对权重上限 |
| `cost_model` | `linear` / `square_root_impact` | `linear` | |
| `rebalance_threshold` | `float \| None` | `None` | 换手低于该值则跳过调仓；`None` = 每期都调 |
| `alpha` | `float` | `0.1` | 仅 `square_root_impact` 使用的冲击系数 |
| `fallback_adv` | `float` | `1e7` | ADV 缺失时的参考值，**单位是元** |
| `primary` | `str \| None` | `None` | 主策略名，用于报告与准入 |
| `strategies` | `list[StrategySpec]` | `[]` | 多策略列表 |

单条策略 `StrategySpec`（`research.py:30`）：`name`（必填）、`type`（必填，内置类型名或自定义 Strategy 子类的 dotted path）、`params`（dict）、以及可选的逐策略覆盖 `max_abs_weight` / `rebalance_threshold` / `cost_model` / `alpha` / `fallback_adv`。

**自动补全规则**（`model_validator(mode="after")`，`research.py:52-65`）：

1. `strategies` 为空 → 自动生成**一条** `topn_long_only` 策略，名为 `topn_{top_n}`，`params={"top_n": top_n}`。
2. `primary` 为空 → 取 `strategies` 第一条的 `name`。

这就是 legacy 单策略写法（只写 `top_n: 50`，不写 `strategies`）仍然能跑的原因。

### 1.3 `walk_forward`

`WalkForwardConfig`（`research.py:83`）：

| 字段 | 默认 | 说明 |
|---|---|---|
| `enabled` | `false` | **默认关闭**，模板注释理由是「computationally expensive」 |
| `train_days` | `504` | IS 观察期（交易日） |
| `test_days` | `63` | OOS 验证期（交易日） |
| `step_days` | `63` | 滚动步长 |
| `embargo_days` | `5` | IS 末与 OOS 首之间的隔离带，降低泄漏 |
| `n_trials` | `50` | IS 阶段试的 `top_n` 候选数 |

候选生成是**确定性**的（`build_top_n_candidate_params`，`research.py:263`）：由 `max_abs_weight` 反推下界 `min_top_n = ceil(1 / max_abs_weight)`，在 `[min_top_n, top_n]` 区间上取 `min(span, n_trials)` 个候选。所以 `max_abs_weight=0.1` 时 `top_n` 不会试到小于 10 的值——权重上限本身就让更小的组合无解。

---

## 2. YAML 模板

模板放在 `workspace/configs/`：

| 路径 | 内容 |
|---|---|
| `workspace/configs/daily/daily_factor_template.yaml` | 官方全字段模板，含四条示例策略与逐字段注释 |
| `workspace/configs/daily/volume_return_corr_20d.yaml` | 真实在用的极简配置 |
| `workspace/configs/intraday/` | 目前只有 `.gitkeep`，**空** |

模板头部注释即用法：`Copy this file to {factor}.yaml or {factor}_{purpose}.yaml before running.`

### 2.1 最小 YAML

`volume_return_corr_20d.yaml` 是文档推荐的最小形态——只写必填项和真正要改的节，其余全吃 pydantic 默认：

```yaml
factor: volume_return_corr_20d
start: "20160606"
end: "20260606"

walk_forward:
  enabled: false
  train_days: 504
  test_days: 252
  step_days: 252
  embargo_days: 5
  n_trials: 1
```

跑它：

```bash
pixi run -- fz factor run --config workspace/configs/daily/volume_return_corr_20d.yaml
```

### 2.2 多策略写法

完整模板里 `backtest.strategies` 给了四种策略类型的真实参数形态：

```yaml
backtest:
  primary: topn_50
  strategies:
    - name: topn_50
      type: topn_long_only
      params: { top_n: 50 }
      max_abs_weight: 0.1
      cost_model: linear

    - name: quantile_ls_5
      type: quantile_long_short
      params: { quantiles: 5 }
      max_abs_weight: 0.1
      cost_model: linear

    - name: factor_weighted_ls
      type: factor_weighted
      params: { long_only: false, gross_exposure: 2.0 }
      max_abs_weight: 0.05
      cost_model: linear

    - name: optimizer_mv_long_only
      type: optimizer_strategy
      params:
        optimizer: mean_variance
        risk_aversion: 1.0
        lookback_days: 60
        cov_estimator: ledoit_wolf
        long_only: true
        top_n: 100
        max_weight: 0.08
        gross_exposure: 1.0
        net_exposure: 1.0
      max_abs_weight: 0.08
      cost_model: linear
```

> ✅ `type` 接受自定义 Strategy 子类的 dotted path。新增策略只要写一个子类并在 `type` 里引用它，**无需改动任何 pipeline 代码**（模板原注释）。

### 2.3 校验 YAML

不跑评估、只验证配置是否合法：

```bash
pixi run -- fz config validate workspace/configs/daily/volume_return_corr_20d.yaml
```

---

## 3. `--set key=value` 覆盖

### 3.1 挂载点（不是全局旗标）

> ⚠️ **`--set` 不是全局旗标。** 它只在下面这几处存在，写在别的命令上会直接报 unrecognized arguments：
>
> | 挂载点 | 来源 |
> |---|---|
> | `fz factor run` | `cli/parser.py:105` → `_add_factor_run_arguments`（`parser.py:38-45`） |
> | `fz factor test` | `cli/parser.py:109`（已标注 Deprecated alias for `factor run`） |
> | `fz factor sweep` | `cli/parser.py:122-128`，语义是「应用到每个组合的固定覆盖」 |
> | `pixi run daily`（`python -m factorzen.pipelines.daily_single`） | `pipelines/daily_single.py:791/805` |

签名 `--set KEY=VALUE`，`action="append"` 因此**可重复**：

```bash
pixi run -- fz factor run momentum_20d \
  --start 20230101 --end 20241231 \
  --set backtest.top_n=30 \
  --set preprocessing.normalizer=rank_normal \
  --set walk_forward.enabled=true \
  --dry-run
```

> ✅ 学 `--set` 时永远配 `--dry-run`（`parser.py:47`，"Print effective config without running"）——它打印生效后的完整配置却不跑评估，是验证覆盖是否命中的最快方式。

### 3.2 值的类型推断

值经 **`yaml.safe_load`** 解析，与 YAML 同源（`research.py:170`）：

| 写法 | 解析结果 |
|---|---|
| `--set backtest.top_n=30` | `int` 30 |
| `--set walk_forward.enabled=true` | `bool` True |
| `--set backtest.max_abs_weight=0.1` | `float` 0.1 |
| `--set preprocessing.normalizer=rank_normal` | `str` `"rank_normal"` |
| `--set backtest.rebalance_threshold=null` | `None` |

### 3.3 注入时机与错误

覆盖在 **pydantic 校验之前**注入到原始 dict（`research.py:145-181`）。这个顺序有两个好处：

1. 非法取值仍然由 pydantic 报错，`--set` 不绕过任何校验。
2. `backtest.top_n` 这类靠 `model_validator` 自动生成策略的 legacy 字段能用**新值**正确生成策略，不需要特判。

dotted key 走嵌套 dict，中间缺失的键按需创建。三种报错：

| 情形 | 异常信息 |
|---|---|
| 没有 `=` | `--set 需要 key=value 形式，收到: ...` |
| 键名为空（如 `--set .top_n=30`） | `--set 键名非法: ...` |
| 中间键存在但不是映射 | `--set 路径冲突：... 不是映射` |

### 3.4 `backtest.top_n` 的特殊同步

覆盖 `backtest.top_n` 时，`build_run_config_from_dict`（`research.py:222-232`）会额外调 `_sync_default_top_n_strategy`：把自动生成的 `topn_50` 策略重命名成 `topn_{新值}`，同步它的 `params.top_n`，并在 `primary` 恰好指向旧名时一并改掉。否则 `--set backtest.top_n=30` 会得到一个「名叫 topn_50、实际只买 30 只」的策略。

> ⚠️ 这个同步**只在无 YAML 路径生效**。`load_run_config`（`research.py:235`，即 `--config` 走的入口）不调用 `_sync_default_top_n_strategy`——因为带 YAML 时策略通常是显式写死的。若你的 YAML 只写了 legacy `top_n` 又用 `--set backtest.top_n=` 覆盖，策略名会停留在旧值。

---

## 4. 两套「默认」：`neutralize` 的坑

同一个字段在两条路径上默认值相反：

| 路径 | `neutralize` | 来源 |
|---|---|---|
| pydantic 类默认 `PreprocessingConfig` | **`false`** | `research.py:26` |
| YAML 模板 `daily_factor_template.yaml` | **`false`** | 模板与类默认一致 |
| **无 YAML 的内置预设** `build_default_daily_research_config()` | **`true`**（`industry+size`） | `research.py:127` |

也就是说：

```bash
# 不给 --config → 走内置预设 → 中性化是【开】的
pixi run -- fz factor run momentum_20d --start 20230101 --end 20241231

# 给了 --config 且 YAML 没写 neutralize → 吃类默认 → 中性化是【关】的
pixi run -- fz factor run --config workspace/configs/daily/my_factor.yaml
```

> ⚠️ **同一个因子、同样的窗口，加不加 `--config` 会得到不同的中性化处理，因而 IC 数值不同。** 对比两次 run 的结果前，先用 `--dry-run` 或翻 `manifest.json` 里的 `config.preprocessing.neutralize` 确认两侧口径一致。

### 内置预设全表

无 YAML 时 `build_default_daily_research_config()`（`research.py:106-142`）产出的完整配置：

| 节 | 值 |
|---|---|
| `universe` | `csi500`（未显式给时） |
| `benchmark` | 按 universe 推导，`csi500` → `000905.SH` |
| `seed` | **`42`**（未显式给时） |
| `preprocessing` | `outlier=mad`、`normalizer=zscore`、**`neutralize=true`**、`neutralize_by=industry+size` |
| `backtest` | `primary=quantile_ls_5`，单策略 `quantile_ls_5` = `quantile_long_short`，`params={"quantiles":5}`、`max_abs_weight=0.1`、`cost_model=linear` |
| `walk_forward` | `enabled=false`、train 504 / test 63 / step 63 / embargo 5 / n_trials 50 |

> ℹ️ 内置预设的主策略是**5 分位多空** `quantile_ls_5`，而 YAML 模板的 `primary` 是 **`topn_50` 多头**。这是第二处「加不加 `--config` 结果不同」的地方。

生效配置最终会被完整写进该次 run 的 `manifest.json`（字段 `config` = `RunConfig.model_dump()`），事后可对账——详见 [产物布局](artifacts.md)。

---

## 5. 不可配置的全局常量

`src/factorzen/config/constants.py` 存的是全局魔法数字，**没有配置入口**，改它们要改代码：

| 常量 | 值 | 用途 |
|---|---|---|
| `COMMISSION_RATE` | `0.00025` | 单边佣金（万 2.5） |
| `STAMP_TAX_RATE` | `0.001` | 卖出印花税（千 1） |
| `SLIPPAGE_RATE` | `0.0005` | 单边冲击/滑点（万 5） |
| `BORROW_RATE_ANNUAL` | `0.085` | 融券年化利率（做空用） |

这四个值会被原样写进 `fz sim run` 的 manifest `cost_model` 字段，可用于事后核对。

> ⚠️ `constants.py:44` 定义了 `MIN_MARKET_CAP_CNY = 3e8`，但它在 `src/` 里**没有任何消费方**——常量已声明、未接线，不要当成生效的默认过滤。真正生效的流动性门槛是 `min_amount = 10_000_000`（`core/universe.py:724/1160`），见 [数据源与口径](data-sources.md)。

路径常量见 `src/factorzen/config/settings.py`，全部路径的实际落盘形态见 [产物布局](artifacts.md)。

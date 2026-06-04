# FactorZen 项目说明

面向维护者、研究使用者与自动化代理,解释 FactorZen 的目标、结构、运行方式、核心数据流、质量门与边界。布局总览见 [README](../README.md),架构见 [architecture](architecture.md),因子编写见 [factor-authoring](factor-authoring.md)。

当前版本:`0.2.0` · 许可:MIT

---

## 1. 项目定位

FactorZen 是面向 A 股单因子的**可信研究框架**,目标不是实盘交易系统,而是把一个因子从数据 → 计算 → 预处理 → IC / 分层回测 → walk-forward 样本外 → 数据质量 → 实验 manifest → HTML 报告串成**可复现**链路。当前聚焦**日频评估与报告**。

**明确不覆盖:** 实盘 OMS/EMS、盘口撮合/逐笔成交、生产组合执行闭环、Tick 数据接入、intraday 主线(已冻结)。

## 2. 目录结构

```text
src/factorzen/
  config/         路径/常量/Tushare 配置(settings, constants, tushare_config)
  core/           日历、universe、存储、加载、数据质量、实验 manifest、校验、计时、日志
  daily/          日频主线:data(PIT)、preprocessing、factors、evaluation、optimization
  intraday/       分钟线(冻结)
  llm/            LLM 研究解读(可选)
  pipelines/      daily_single、generate_report 端到端流程
  reports/        Tear Sheet 报告引擎(tear_sheet + _formatting/_scoring/_charts/_strategy/_summaries + templates)
  research/       实验性多因子合成(research/combination)
  cli/            统一 CLI 入口(fz)
workspace/factors/            用户日常新增因子({daily,weekly,monthly,intraday})
workspace/configs/            实验 YAML 配置
workspace/factor_evaluations/ 每次运行的 report.html / manifest.json / parquet 产物 + experiment_index.jsonl
data/                         本地数据缓存(parquet,不入库)
```

## 3. 快速开始

```bash
pixi install
cp .env.example .env          # 配置 TUSHARE_TOKEN(不入库)
pixi run smoke                # import 自检

pixi run fz factor list
pixi run fz factor new my_alpha --frequency daily
pixi run fz factor run my_alpha --start 20250101 --end 20260513 --universe csi500
pixi run fz report path <run_id>
pixi run fz config validate workspace/configs/daily/my_run.yaml
```

`scripts`/`daily`/`report`/`factor test`/`report open` 仅作兼容别名保留。

## 4. 配置体系

- `factorzen/config/settings.py` —— 集中路径与调度默认值(`ROOT`/`DATA_*`/`WORKSPACE_*`/`SCHEDULER_*`),业务代码一律从这里取路径。
- `factorzen/config/constants.py` —— 研究常量(年/月/周交易日数、MAD 去极值参数、IC 最小样本、默认分位数、涨跌停阈值、回测有效性阈值、默认成本、基准映射)。
- `factorzen/config/tushare_config.py` —— 读取 `.env`,暴露 `TUSHARE_TOKEN` 与 `ensure_token()`,token 首次真正调用时才校验(离线测试不因 import 失败)。

**YAML 运行配置**(`factorzen/core/config_loader.py`,Pydantic v2 校验),示例:

```yaml
factor: momentum_20d
universe: csi500
start: "20230101"
end: "20241231"
benchmark: "000300.SH"
seed: 42
preprocessing: { outlier: mad, normalizer: zscore, neutralize: false, neutralize_by: industry+size }
backtest: { top_n: 50, quantiles: 5, max_abs_weight: 0.1, cost_model: linear, rebalance_threshold: null }
walk_forward: { train_days: 504, test_days: 63, step_days: 63, embargo_days: 5, n_trials: 50 }
ic_method: rank
event_study: false
neutralized_ic: false
```

`neutralize=true` 时按 `neutralize_by` 传入数据:`industry`(股票池行业)、`size`(`daily_basic.total_mv`)、`industry+size`(两者)。配置样例在 `workspace/configs/{daily,...}/`。

## 5. 数据流(日频主线)

```text
本地 parquet 缓存 (data/)
  → PIT 数据上下文 (daily/data)
  → 因子计算 (daily/factors) + 预处理 (daily/preprocessing:去极值/标准化/中性化)
  → 前向收益 (compute_fwd_returns) + IC 分析 (compute_rank_ic / Pearson / 中性化 IC)
  → 分层回测 (run_strategy_backtest) + 换手 (compute_turnover) + walk-forward OOS
  → 报告引擎 (reports/tear_sheet) → Tear Sheet HTML
```

`pipelines/daily_single.py` 与 `pipelines/generate_report.py` 把上述串成端到端流程,由 `fz factor run` / `fz report build` 调用。原始数据按 Hive 风格 `year=YYYY/month=MM` 分区落 `data/raw/`,缓存落 `data/cache/`。

## 6. 回测口径

- t 日因子生成目标权重;t+1 开盘执行调仓。
- 旧持仓承担 overnight return,新持仓承担 open-to-close return(`ret_definition = open_to_close_with_overnight_carry`)。
- **无未来函数**:前向收益用复权价;`adv_20d` 仅取执行日之前最多 20 期均成交额(用于平方根冲击成本)。

**策略接口**:继承 `Strategy` 实现 `generate_weights(context) -> DataFrame[ts_code, target_weight]`。内置 `QuantileLongShortStrategy` / `TopNLongOnlyStrategy` / `FactorWeightedStrategy` / `OptimizerStrategy`。

**成交约束**:停牌不成交、涨停不买入、跌停不卖出、`max_participation_rate` 容量、`max_abs_weight`、`max_gross_exposure`、`rebalance_threshold`。

## 7. 评估与报告

- **IC**:Rank IC(Spearman)/ Pearson IC / 中性化 IC、多持有期一致性、样本外分割、HAC t 统计。
- **分层回测**:分组/多空 NAV、月度收益、分位价差。
- **稳健性**:单调性、Rank 自相关、因子相关性、市值/行业/市场状态分层、事件研究、walk-forward。
- **报告**:`reports/tear_sheet.generate_tear_sheet` 输出综合评级评分卡 + 各分析面板 + 附录(复现摘要 + 模块状态);报告引擎按职责拆为 `_formatting/_scoring/_charts/_strategy/_summaries` 五个模块 + 编排主体。

## 8. 可复现与可观测

- **实验 manifest**(`core/experiment.run_experiment`):每次运行落 `workspace/factor_evaluations/{run_id}/manifest.json`,记录 `git_sha` / `git_dirty` / `pixi_lock_sha256` / `command` / `config` / `duration_seconds` / `stage_timings` / `outputs` / 状态;并 append 到 `experiment_index.jsonl` 供跨运行检索。工作树 dirty 时运行会 `WARNING` 提示。
- **按阶段计时**(`core/timing.StageTimer`):两条日频管线对 IC / 回测 / 换手 / 报告四阶段计时并写入 `stage_timings`。
- **数据契约**(`core/validation.require_columns`):`compute_fwd_returns` / `compute_turnover` / 回测入口对必需列 fail-fast 校验,畸形输入早失败给清晰错误。

## 9. 质量门

```bash
pixi run lint        # ruff
pixi run typecheck   # mypy(src/factorzen,0 错)
pixi run test        # pytest
pixi run coverage    # 覆盖率(--fail-under 门槛)
```

CI 在 push / PR 到 `master` 时运行上述门;提交前建议 `pre-commit install`(经 pixi 统一 ruff + mypy)。详见 [CONTRIBUTING](../CONTRIBUTING.md)。

## 10. 扩展

新增因子:`fz factor new <name> --frequency daily` 生成模板到 `workspace/factors/daily/`,实现因子逻辑后由注册表自动发现。详见 [factor-authoring](factor-authoring.md)。

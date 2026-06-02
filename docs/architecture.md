# FactorZen 架构

FactorZen 分为框架包(`src/factorzen/`)和工作区(`workspace/`)两层。

```text
src/factorzen/      框架代码
workspace/factors/  用户自定义因子
workspace/configs/  实验 YAML 配置
workspace/factor_evaluations/     每次运行的自包含输出
data/               本地市场数据缓存
tests/              pytest 测试
tests/benchmarks/   性能基准脚本
docs/               项目文档
```

## 分层职责

- **框架包(`src/factorzen/`)** 负责稳定接口:数据读取(`core` / `daily/data`)、因子注册(`daily/factors`)、预处理(`daily/preprocessing`)、评估(`daily/evaluation`)、报告(`reports`)、调度(`automation`)和 CLI 编排(`cli`)。
- **工作区(`workspace/`)** 负责日常研究:新因子放在 `workspace/factors/{daily,weekly,monthly,intraday}`,实验配置放在 `workspace/configs/`。

## 数据流(日频主线)

```text
本地 parquet 缓存 (data/)
   → PIT 数据上下文 (daily/data)
   → 因子计算 (daily/factors) + 预处理 (daily/preprocessing)
   → 评估 (daily/evaluation: IC / 分层回测 / 换手 / walk-forward / 归因 / 基准)
   → 报告引擎 (reports/tear_sheet) → Tear Sheet HTML
```

`pipelines/daily_single.py` 与 `pipelines/generate_report.py` 把上述步骤串成端到端流程,由 `fz factor run` / `fz report build` 调用。

## 产物与可复现

每次运行把 manifest 和标准产物写到 `workspace/factor_evaluations/{run_id}/`(`report.html`、`manifest.json`、`universe.parquet`、IC/回测 parquet),便于直接定位报告与复现实验。

## 范围

当前迭代聚焦日频。`intraday/` 分钟线主线与报告 UI 已冻结,Tick 级研究、实盘执行不纳入框架包。

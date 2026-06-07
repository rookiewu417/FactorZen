# 运行手册

> [FactorZen](../README.md) · [文档](README.md) · [架构](architecture.md) · **运行手册** · [路线图](evolution-plan-2026.md)

所有命令从仓库根目录执行，并通过 `pixi run` 进入项目环境。

## 命令速查

| 场景 | 命令 |
|------|------|
| 环境自检 | `pixi run smoke` |
| 数据 smoke | `pixi run smoke-data --start … --end …` |
| 列出因子 | `pixi run fz factor list` |
| 新建因子 | `pixi run fz factor new <name> --frequency daily` |
| 运行因子 | `pixi run fz factor run <name> --start … --end … --universe csi500` |
| 校验配置 | `pixi run fz config validate <path>` |
| 生成报告 | `pixi run fz report build <name> …` |
| 报告路径 | `pixi run fz report path <run_id>` |
| 拉取数据 | `pixi run fz data fetch daily --start … --end …` |
| 运行历史 | `pixi run fz runs list` / `pixi run fz runs show <run_id>` |
| 质量门 | `pixi run lint && pixi run typecheck && pixi run test && pixi run coverage` |

## 环境自检

```bash
pixi install
cp .env.example .env
pixi run smoke
```

真实数据拉取需要在 `.env` 配置 `TUSHARE_TOKEN`。LLM 研究解读默认关闭；无 YAML 默认配置与 `--all` 模式会自动尝试，缺少 `FACTORZEN_LLM_*` 配置时自动跳过。普通/自定义运行需显式传入 `--llm-explain` 才会尝试读取相关配置。

需要验证真实环境时，用数据 smoke 检查 Tushare 连通性与本地原始数据分区完整性（不入默认 CI）：

```bash
pixi run smoke-data --start 20230101 --end 20231231   # 连通性 + daily/daily_basic/finance 审计
pixi run smoke-data --skip-tushare                     # 仅离线审计本地分区
```

退出码 0=全部正常，1=出现 error，2=仅 warning。

## 因子工作流

```bash
pixi run fz factor list
pixi run fz factor new my_alpha --frequency daily
pixi run fz factor run my_alpha --start 20230101 --end 20241231
```

无 `--config` 时会使用内置研究级默认配置：`csi500`、匹配 benchmark、`seed=42`、行业+市值中性化、内置 4 策略套件、both IC、neutralized IC、event study 与 LLM 解读。缺少 `FACTORZEN_LLM_*` 配置时，LLM 解读会自动跳过。walk-forward 默认关闭，按需通过 YAML `walk_forward.enabled: true` 或 `--set walk_forward.enabled=true` 开启。

仍可显式覆盖默认配置：

```bash
pixi run fz factor run momentum_20d --start 20230101 --end 20241231 \
  --set backtest.top_n=30 --set preprocessing.normalizer=rank_normal
```

## YAML 配置

```bash
pixi run fz config validate workspace/configs/daily/daily_factor_template.yaml
pixi run fz factor run --config workspace/configs/daily/daily_factor_template.yaml
```

`config validate` 会打印生效后的配置与标准输出目录，不会启动完整回测。

walk-forward 样本外评估默认关闭，需要时在 YAML 打开：

```yaml
walk_forward:
  enabled: true
  train_days: 504
  test_days: 252
  step_days: 252
  embargo_days: 5
```

也可用 `--set walk_forward.enabled=true` 临时开启。

## 命令行调参

`--set key=value` 在校验前覆盖任意字段（含 `preprocessing` / `backtest` / `walk_forward`），可重复。
值类型与 YAML 同源推断（`30→int`、`true→bool`、`rank_normal→str`），并写入 manifest 保持可复现：

```bash
pixi run fz factor run momentum_20d --start 20230101 --end 20241231 \
  --set backtest.top_n=30 --set preprocessing.neutralize=true --set walk_forward.train_days=252
```

无 YAML 默认配置下，`--set backtest.top_n=N` 会同步更新默认主策略为 `topn_N`；
若 YAML 已显式定义多策略 `strategies:`，`backtest.top_n` 只改 vestigial 字段——多策略维度请用 sweep 或编辑 YAML。
先用 `--dry-run` 打印生效配置确认后再跑。

`factor sweep` 在 `--set` 之上做网格扫描：每个组合串行跑一次完整评估，按指标排序出对比表并落
`workspace/factor_evaluations/sweep_{ts}/sweep_results.csv`。单组失败不中断全局，会在表中标注 error。

```bash
pixi run fz factor sweep --config workspace/configs/daily/daily_factor_template.yaml \
  --grid backtest.top_n=30,50,100 --grid preprocessing.normalizer=zscore,rank_normal --sort-by ir
# --grid 可多个维度（笛卡尔积）；--set 施加到每个组合的固定覆盖；--sort-by 取 ir/ic_mean/ic_pos/t
```

## 报告

```bash
pixi run fz report build momentum_20d --start 20230101 --end 20241231 --universe csi500
pixi run fz report path <run_id>
```

已有产物可复用时，加 `--reuse`：

```bash
pixi run fz report build momentum_20d --start 20230101 --end 20241231 --universe csi500 --reuse
```

报告与 manifest 的标准位置：

```text
workspace/factor_evaluations/{run_id}/
```

## 数据

```bash
pixi run fz data fetch daily --start 20230101 --end 20241231
pixi run fz data fetch daily-basic --start 20230101 --end 20241231
```

管线会在运行时审计并补齐所需缓存。真实网络请求不进入默认 CI。

## 运行历史

```bash
pixi run fz runs list
pixi run fz runs list --limit 50
pixi run fz runs show <run_id>
```

`runs list` 读取 `workspace/factor_evaluations/experiment_index.jsonl`，`runs show` 读取单次运行的 `manifest.json`。

## 质量门

```bash
pixi run lint
pixi run typecheck
pixi run test
pixi run coverage
```

CI 在 push / PR 到 `main` 或 `master` 时运行同一套检查。

## 兼容入口

`pixi run daily`、`pixi run report`、`fz factor test` 与 `fz report open` 仍保留为兼容入口。新增脚本与文档优先使用 `fz factor run`、`fz report build`、`fz report path`。

## 故障处理

| 现象 | 处理 |
|------|------|
| `TUSHARE_TOKEN` 缺失 | 只影响真实数据拉取；离线测试不应依赖真实 token |
| `report path` 找不到报告 | 先用 `fz runs list` 确认 `run_id`，再用 `fz runs show <run_id>` 查看 manifest |
| qlib 因子运行失败 | 确认 `QLIB_PROVIDER_URI` 指向的数据包覆盖运行日期，详见 [`src/factorzen/builtin_factors/qlib/README.md`](../src/factorzen/builtin_factors/qlib/README.md) |
| manifest 标记 `git_dirty=true` | 工作树存在未提交改动，该 run 不能只凭 git SHA 完全复现 |

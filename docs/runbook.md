# 运行手册

常用命令:

```bash
pixi run fz factor list
pixi run fz factor new my_alpha --frequency daily
pixi run fz factor run my_alpha --start 20250101 --end 20260513 --universe csi500
pixi run fz report path <run_id>
```

YAML 配置运行:

```bash
pixi run fz config validate workspace/configs/daily/daily_factor_template.yaml
pixi run fz factor run --config workspace/configs/daily/daily_factor_template.yaml
```

报告生成:

```bash
pixi run fz report build momentum_20d --start 20250101 --end 20260513
pixi run fz report path <run_id>
```

数据拉取(需在 `.env` 配置 `TUSHARE_TOKEN`):

```bash
pixi run fz data fetch daily --start 20250101 --end 20260513
pixi run fz data fetch daily-basic --start 20250101 --end 20260513
```

运行历史:

```bash
pixi run fz runs list
```

## 质量门(提交前 / CI)

```bash
pixi run lint
pixi run typecheck
pixi run test
pixi run coverage
```

## 说明

- 关键输出在 `workspace/factor_evaluations/{run_id}/`(报告 + manifest + parquet)。
- 新增流程统一优先使用 `fz`;`daily`、`report`、`factor test`、`report open` 仅作为兼容别名保留。
- 长任务(批量回测 / 数据拉取)建议放入 tmux 并记录命令、日志与输出目录。

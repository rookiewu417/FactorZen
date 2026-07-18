# 运维工具

这里仅放受版本控制、可从项目根通过 `pixi run` 执行的运维入口。生成数据始终写入
`data/`，研究产物写入 `workspace/`；脚本自身不得放在这两个 gitignored 产物根下。

> ℹ️ 本目录是**一次性数据搬运与修复工具**，与日常研究链路无关。日常取数用
> `pixi run fz data fetch`，见 [CLI 参考](../docs/reference/cli.md)；数据源与单位口径见
> [数据源与口径](../docs/reference/data-sources.md)。

## 分钟数据导入

`ingest_minute.py` 通过 `factorzen.dataio.minute_ingest` 的统一接口接收两种历史布局：

- 按日全市场 parquet（代码列为 `code` 或 `ts_code`）；
- 按股票 parquet（代码列为 `ts_code`）。

两者都会规范成
`ts_code, trade_time, open, high, low, close, vol, amount`，写入
`data/raw/minute_1min/year=YYYY/month=MM/data.parquet`。现有分区会按
`(trade_time, ts_code)` 合并，绝不因“分区文件已存在”而跳过补缺。

```bash
# Tushare 按股票补缺目录 → 生产分钟湖
pixi run python tools/ingest_minute.py data/_gapfill/minute/1min

# 外部按日源，只导入一个月做验证
pixi run python tools/ingest_minute.py /mnt/e/BaiduNetdiskDownload --month 202001
```

长任务启动前按项目约定检查 `nvidia-smi` / `tmux ls`，在 tmux 中运行并将日志、输出目录、
命令和完成 sentinel 写入唯一时间戳目录。

## Tushare 批量下载

`download_tushare_lake.py` 是原先误放在 gitignored `data/_tools/` 的可续跑下载器。现从
项目 `.env` 读取 `TUSHARE_TOKEN`，可选变量为 `TUSHARE_API_URL`、
`TUSHARE_LAKE_MAX_PER_MIN` 和 `TUSHARE_LAKE_WORKERS`；凭据不会写进 manifest。

```bash
nvidia-smi
tmux ls
tmux new-session -d -s tushare_gapfill \
  'ROOT=data/_gapfill TIMEOUT_SECONDS=86400 tools/run_tushare_lake.sh --phases 1 --min-start 20260413'
```

轮询 `data/_gapfill/download.done`，完成后用上节的 `ingest_minute.py` 直接把按股票分钟源
合并进生产湖，无需再经过中间按日副本。

## Raw 快照补缺

旧备份不能按目录大小判定“冗余”。`repair_raw_partition.py` 先按键做 anti-join，只把目标
缺失行对齐到当前 schema 后追加，目标已有值永远优先；每次运行会在
`workspace/data_maintenance/<run_id>/` 写 manifest 与完成 sentinel。

```bash
pixi run python tools/repair_raw_partition.py data/raw/<type>.bak.<timestamp> \
  --target-data-type daily_basic
```

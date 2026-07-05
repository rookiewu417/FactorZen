# Windows 任务计划兜底(WSL2 未常驻时)

当 WSL2 未启用 systemd(或不常开)时,用 Windows 任务计划程序在宿主机触发 WSL 内的 `fz ops daily`。

## 创建任务(PowerShell,管理员)

```powershell
# 工作日 18:35 触发(比 systemd 晚 5 分钟,作为兜底)
schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 18:35 /TN FactorZenOps /TR ^
  "wsl.exe -d Ubuntu -- bash -lc 'cd ~/projects/FactorZen && ~/.pixi/bin/pixi run fz ops daily --config workspace/configs/ops.yaml'"
```

- `-d Ubuntu`:按 `wsl -l -v` 的实际发行版名调整。
- `bash -lc`:走登录 shell 以加载 pixi 的 PATH。

## 验证

```powershell
schtasks /Run /TN FactorZenOps      # 立即触发一次
schtasks /Query /TN FactorZenOps /V /FO LIST   # 查看上次运行结果码(0=成功)
```

## 删除

```powershell
schtasks /Delete /TN FactorZenOps /F
```

## 说明

- 幂等:同日多次触发(systemd + 本任务都跑)无害,已完成阶段自动跳过。
- 真正的"电脑关了也在跑"需迁移到 VPS(见 `docs/runbook.md` 的 VPS 章节),Windows/WSL 方案仅作过渡。

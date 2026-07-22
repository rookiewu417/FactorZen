# 部署

> [FactorZen](../../README.md) · [文档](../README.md) · **部署**

本文覆盖两件事：把**只读展示服务**跑起来，以及把**无人值守日链路**挂上定时触发。研究链路本身是一次性命令，产物落 `workspace/`；需要「长期运行」的只有这两块。

读完能：在本机或容器里起 Dashboard、复制并校验 `ops.yaml`、用 systemd timer / Windows 任务计划 / Docker 触发每日链路。

| 部署对象 | 是什么 | 入口 |
|---|---|---|
| **只读 Web 服务** | 把 `workspace/` 里已有的产物读出来展示，不触发任何计算 | `pixi run serve` |
| **无人值守日链路** | 每交易日收盘后跑一遍 8 阶段链路 | `fz ops daily` + systemd / 任务计划 / cron |

---

## 只读 Web 服务

### 依赖前提

服务层依赖 `fastapi` 与 `uvicorn`，这两个包在 `pyproject.toml` 里属于 **`dev` optional-dependencies，不在运行时依赖集内**。

- 用 **pixi**（推荐）：默认环境已合入 `dev` feature，`pixi install` 之后开箱可用。
- 用 **pip 安装运行时依赖**：`pip install factorzen` 装不到 fastapi/uvicorn，服务起不来，需要额外 `pip install "factorzen[dev]"`。

验证依赖是否就位：

```bash
pixi run -- python -c "import fastapi, uvicorn; print(fastapi.__version__, uvicorn.__version__)"
```

### 启动

```bash
pixi run serve
```

这是 `pixi.toml` 里定义的任务，展开等于：

```bash
uvicorn factorzen.server.api:app --host 0.0.0.0 --port 8000
```

> ⚠️ `pixi run serve` **不是 `fz` 的子命令**，`fz serve` 不存在。它直接调 uvicorn。

> ⚠️ 默认绑定 `0.0.0.0`，即**监听所有网络接口**，同一局域网内其他机器可直接访问。服务**无鉴权**（见下方[边界](#边界与已知限制)）。只想本机访问就别用 `pixi run serve`，改为显式指定回环地址：
>
> ```bash
> pixi run -- uvicorn factorzen.server.api:app --host 127.0.0.1 --port 8000
> ```
>
> 换端口同理——`pixi run serve` 的 host/port 是写死在任务里的，要改就绕过任务直接调 uvicorn。

启动后可访问：

| 地址 | 内容 |
|---|---|
| `http://localhost:8000/` | 单页 Dashboard（服务端渲染） |
| `http://localhost:8000/docs` | FastAPI 自带的 OpenAPI 交互文档 |
| `http://localhost:8000/api/...` | REST 端点，见下表 |

### REST 端点

一共 **4 个**：

| 端点 | 说明 |
|---|---|
| `GET /api/health` | 健康检查，返回 `status` 与可用 `domains` 列表 |
| `GET /api/runs?domain=<d>` | 某个域下的全部 run（`run_id` / `git_sha` / `status` / 完整 `manifest`） |
| `GET /api/runs/{domain}/{run_id}` | 单个 run 的 manifest 与指标 |
| `GET /api/nav/{domain}/{run_id}` | 该 run 的净值序列 |

`domain` 取值固定为 6 个：

```text
factor_evaluations · mining_sessions · portfolios · sim · execution · combinations
```

对应 `workspace/<domain>/<run_id>/manifest.json` 的目录布局（见[产物参考](../reference/artifacts.md)）。传入不在列表里的 `domain` 返回 404。

```bash
curl -s http://localhost:8000/api/health
curl -s "http://localhost:8000/api/runs?domain=sim"
```

### 单页 Dashboard

根路径 `/` 是服务端渲染的单页总览：各域产物计数 + 最近 run 列表 + `execution` 域最新一个 run 的净值曲线。没有 `execution` 产物时净值区为空，页面照常渲染。

### 产物根目录

服务读取的 `workspace/` 路径是**代码里固定的仓库根下 `workspace/`**，没有环境变量可以改写。要让服务展示别处的产物，只能在容器里把目标目录**挂载到 `/app/workspace`**（下节）。

### 容器部署

仓库内 `deploy/docker/` 已有无人值守运营用的镜像定义（`Dockerfile` + `compose.yaml`）。Dockerfile 的 `ENTRYPOINT` 固定为 `pixi run fz`，所以 Web 服务需要覆盖 entrypoint。先构建镜像，再追加服务：

```bash
docker compose -f deploy/docker/compose.yaml build ops   # 产出 factorzen:latest
```

```yaml
# 追加到 deploy/docker/compose.yaml 的 services: 下
  web:
    image: factorzen:latest          # 需先构建；本服务不自带 build:
    entrypoint: ["pixi", "run", "serve"]
    volumes:
      - ../../workspace:/app/workspace:ro   # 只读挂载，服务动不了产物
    ports:
      - "8000:8000"
```

> ℹ️ 想只展示一份脱敏的示例产物，**把那份目录挂成 `/app/workspace`**，而不是挂到别的路径：
> `- ../../workspace-demo:/app/workspace:ro`。因为产物根在代码里写死，挂到其他路径服务读不到。

公网暴露前必须前置反向代理（Caddy / nginx）补上 **HTTPS + basic auth**，并且只挂脱敏产物——真实 `workspace/` 含研究细节与因子库，不建议出内网。

### 边界与已知限制

- **纯读**：不触发任何计算，不 import 研究模块。服务崩不影响研究，研究改不崩服务。
- **无鉴权**：服务层没有任何认证/授权，任何能连上端口的人都能读全部产物。
- **无分页**：`/api/runs` 一次返回该域全部 run，且每条内嵌完整 `manifest`。产物累积多了响应体会很大。
- **坏产物容错**：manifest 解析失败会记 warning 并跳过该 run，不会让整个接口报错。

---

## 无人值守日链路

`fz ops daily` 按 `ops.yaml` 声明的顺序推进 8 个阶段：

```text
guard → data → audit → intraday_features → signal → live_step → report → publish
```

`guard` 判定非交易日会直接短路，后续阶段不执行。每个阶段的完成状态写入 `workspace/ops/state/<YYYY-MM-DD>.json`——**同日重复触发是幂等的**，已完成阶段自动跳过，失败处续跑。这是把它挂到定时器上的前提。

命令参数与产物见 [CLI 参考 · fz ops](../reference/cli.md#fz-ops)，配置字段见[配置参考](../reference/configuration.md)。阶段语义与排查见[无人值守运营](operations.md)。

### 准备配置

仓库只提供模板 `deploy/ops.example.yaml`，**不提供开箱即用的 `workspace/configs/ops.yaml`**。下面的 systemd / 任务计划示例都指向后者，所以先复制：

```bash
mkdir -p workspace/configs
cp deploy/ops.example.yaml workspace/configs/ops.yaml
```

按需修改 `session_dir` / `portfolio_run_dirs_glob` / `benchmark` 等字段。配置用 `extra=forbid` 校验，**字段名拼错会直接被拒**而不是静默忽略。

先手工跑一次确认链路通：

```bash
pixi run -- fz ops daily --config workspace/configs/ops.yaml --date 20241231
pixi run -- fz ops status --config workspace/configs/ops.yaml --date 20241231
```

### Linux / WSL2：systemd timer

仓库内已有两个单元文件：`deploy/systemd/factorzen-ops.service` 与 `deploy/systemd/factorzen-ops.timer`。

service 以 `Type=oneshot` 跑一次日链路，timer 设为 `OnCalendar=Mon..Fri 18:30 Asia/Shanghai`（A 股收盘后留足数据落地时间），并开了 `Persistent=true`，机器关机错过触发时开机补跑。

安装前必须按本机情况改两处：

- `User=` 与 `WorkingDirectory=` —— 改成你的用户名与仓库绝对路径。
- `ExecStart=` 里的 pixi 绝对路径 —— 用 `which pixi` 查（常见 `~/.pixi/bin/pixi`）。systemd 不走登录 shell，不能依赖 PATH。

```bash
sudo cp deploy/systemd/factorzen-ops.service /etc/systemd/system/
sudo cp deploy/systemd/factorzen-ops.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now factorzen-ops.timer

systemctl list-timers factorzen-ops.timer     # 确认下次触发时间
sudo systemctl start factorzen-ops.service    # 立即手工触发一次
journalctl -u factorzen-ops.service -n 100    # 查看日志
```

> ⚠️ WSL2 默认**不启用 systemd**。启用需要在发行版内 `/etc/wsl.conf` 写 `[boot]\nsystemd=true` 再 `wsl --shutdown` 重启。不想启用、或 WSL 不常开机，用下面的 Windows 任务计划方案。

### Windows 宿主机：任务计划程序

WSL2 未常驻 systemd 时，可以从 Windows 侧触发 WSL 内的 `fz ops daily`。

以**管理员身份的 cmd.exe** 执行（`schtasks` 的 `/TR` 参数在 PowerShell 下引号转义规则不同，用 cmd 更省事）：

```bat
schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 18:35 /TN FactorZenOps /TR "wsl.exe -d Ubuntu -- bash -lc \"cd ~/projects/FactorZen && ~/.pixi/bin/pixi run fz ops daily --config workspace/configs/ops.yaml\""
```

- `-d Ubuntu` —— 按 `wsl -l -v` 的实际发行版名调整。
- `bash -lc` —— 走登录 shell 以加载 pixi 的 PATH；即便如此仍建议 pixi 写绝对路径。
- `18:35` —— 比 systemd 的 18:30 晚 5 分钟。两者同时启用无害：链路幂等，先跑完的那次会让后跑的直接跳过全部阶段。

管理与验证：

```bat
schtasks /Run   /TN FactorZenOps                  :: 立即触发一次
schtasks /Query /TN FactorZenOps /V /FO LIST      :: 查看上次运行结果码(0=成功)
schtasks /Delete /TN FactorZenOps /F              :: 删除
```

> ⚠️ Windows 任务计划只是 WSL 场景的**过渡方案**：宿主机关机就不跑。真正的「电脑关了也在跑」需要迁到常驻主机。

### VPS / 容器

`deploy/docker/compose.yaml` 定义了 `ops` 服务，把 `data/` 与 `workspace/` 卷挂载到宿主机做持久化，`.env` 经 `env_file` 注入（需含 `TUSHARE_TOKEN`，webhook 通知另需 `FACTORZEN_NOTIFY_WEBHOOK`）：

```bash
docker compose -f deploy/docker/compose.yaml run --rm ops
```

在 VPS 上用 cron 触发这条命令即可获得常驻的日链路。镜像用 `pixi install --locked` 按锁文件精确安装，与本地/CI 环境一致。

### 通知

`ops.yaml` 的 `notify_kind` 支持 `stdout`（默认）与 `webhook`。`webhook` 模式从 `notify_url_env` 指定的环境变量读 URL（默认 `FACTORZEN_NOTIFY_WEBHOOK`），可对接企业微信机器人等。**URL 只放环境变量，不要写进配置文件提交上去。**

---

## 相关文档

- [无人值守运营](operations.md) —— 日链路各阶段做了什么、失败如何排查
- [CLI 参考 · fz ops](../reference/cli.md#fz-ops) —— 命令参数与产物
- [配置参考](../reference/configuration.md) —— `ops.yaml` 字段说明
- [环境变量](../reference/environment.md) —— `TUSHARE_TOKEN` / 通知 / LLM 配置
- [性能与资源](performance.md) —— 长任务的内存与并行调优

# 只读服务层部署

FastAPI 只读 API + Web Dashboard,暴露 workspace 产物(runs/manifest/NAV)。纯读、零侵入现有 pipeline。

## 本地启动

```bash
pixi run serve                       # uvicorn :8000
# 打开 http://localhost:8000/         → Dashboard 总览
#      http://localhost:8000/docs     → OpenAPI 交互文档
#      http://localhost:8000/api/runs?domain=sim → REST
```

## API

| 端点 | 说明 |
|------|------|
| `GET /api/health` | 健康检查 + 可用 domains |
| `GET /api/runs?domain=<d>` | 某域 run 列表(run_id/status/git_sha) |
| `GET /api/runs/{domain}/{run_id}` | 单 run manifest + metrics |
| `GET /api/nav/{domain}/{run_id}` | NAV 序列 [(date, nav)] |

domain ∈ factor_evaluations / mining_sessions / portfolios / sim / execution / combinations。

## 容器 / VPS

复用无人值守运营的 pixi 镜像(见 `deploy/docker/`),追加 server 服务:

```yaml
# 追加到 deploy/docker/compose.yaml
  web:
    image: factorzen:latest
    entrypoint: ["pixi", "run", "uvicorn", "factorzen.server.api:app", "--host", "0.0.0.0", "--port", "8000"]
    volumes:
      - ../../workspace:/app/workspace:ro   # 只读挂载,服务动不了产物
    ports:
      - "8000:8000"
```

公网部署:前置 Caddy/nginx 自动 HTTPS + basic auth;demo 实例建议只挂 `workspace-demo/`(脱敏示例产物),
真实 workspace 不出内网。

## 边界

- 纯读:不触发任何计算,不 import 研究模块——服务崩不了研究,研究改不崩服务。
- 认证:MVP 未内置;公网暴露前务必加反代层 basic auth + HTTPS。

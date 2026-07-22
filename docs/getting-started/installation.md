# 安装与环境

> [FactorZen](../../README.md) · [文档](../README.md) · **安装与环境**

装完这一页，你会有一个可直接跑 `fz` 的环境，以及一份填好的 `.env`。全程约 10 分钟，大头是依赖下载。本页默认 A 股路径（Tushare token）；多市场见[多市场适配](../concepts/multi-market.md)。

---

## 1. 前置条件

| 项 | 要求 | 说明 |
|---|---|---|
| [pixi](https://pixi.sh/) | 任意近版本 | 唯一的环境管理器，负责拉 Python 与全部依赖 |
| git | — | 克隆仓库 |
| 磁盘 | 环境约 3 GB | 数据另算：全 A 日线约 300 MB，分钟线可达数十 GB |
| 内存 | 建议 ≥ 16 GB | 全 A 长窗口挖掘与日内特征构建吃内存，见[性能与资源](../guides/performance.md) |

> ℹ️ **本项目不使用全局 Python。** 不需要预装 Python、不需要 `venv`、不需要 `pip install`。Python 3.10–3.12 由 pixi 按 `pixi.toml` 声明装进项目本地环境（本机实测 3.12.13）。

安装 pixi（Linux / macOS）：

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

---

## 2. 安装

```bash
git clone https://github.com/rookiewu417/FactorZen.git
cd FactorZen
pixi install
```

`pixi install` 按仓库内已提交的 `pixi.lock` 精确复现依赖版本，不做重新求解。

**此后所有命令都从仓库根目录执行，都带 `pixi run` 前缀。** 核心入口是 `fz`：

```bash
pixi run fz --help
```

`fz` 是 `pixi.toml` 里声明的一个 task，等价于 `python -m factorzen.cli.main`。全部命令与参数见 [CLI 参考](../reference/cli.md)。

### 环境里装了什么

`pixi.toml` 的默认环境**已经包含 dev feature**，所以一次 `pixi install` 同时给你运行时依赖和开发/测试依赖：

| 组 | 内容 |
|---|---|
| 运行时（23 个） | polars · numpy · scipy · pandas · statsmodels · lightgbm · scikit-learn · optuna · cvxpy · tushare · ccxt · pyarrow · openai · matplotlib · jinja2 · pydantic 等 |
| 开发（8 个） | pytest · pytest-xdist · pytest-cov · coverage · ruff · mypy · jupyterlab · hypothesis |
| 展示服务 | fastapi · uvicorn · httpx |
| 可解释性 | shap（多因子组合的 LightGBM 重要度） |

> ⚠️ **`fastapi` / `uvicorn` / `shap` 在 `pyproject.toml` 里属 dev extras，不是运行时依赖。** 走 pixi 安装不受影响（默认环境含 dev）；但如果你绕开 pixi、只按 `[project] dependencies` 装运行时依赖，`pixi run serve` 起的只读展示 server 会因缺 fastapi/uvicorn 直接起不来，多因子组合的 SHAP 重要度也会缺失。这种情况下需要显式装 `factorzen[dev]`。

> ℹ️ `pyproject.toml` 的 `dev` extra 与 `pixi.toml` 的 dev feature 略有出入：`mypy` / `pytest-xdist` / `pytest-cov` 只在 pixi 侧声明。**要跑完整的 lint/typecheck/test 三件套，请用 pixi 环境**，别自行拼 pip 环境。

---

## 3. 配置 `.env`

凭据一律从仓库根目录的 `.env` 读取，仓库里带了一份模板：

```bash
cp .env.example .env
```

> ⚠️ **`.env` 已被 gitignore，绝不能提交。** 也不要把 token 写进任何文档、计划文件或提交信息里。

### 3.1 `TUSHARE_TOKEN`（A 股取数必填）

```bash
TUSHARE_TOKEN=<你的 tushare token>
```

- 去 [tushare.pro](https://tushare.pro/) 注册取 token。日线、每日指标、财务等接口有积分门槛，具体见[数据源与口径](../reference/data-sources.md)。
- **缺 token 不会让 CLI 崩溃**：token 校验是延迟的，不在 import 阶段执行。`fz factor list`、`fz ops validate-config`、`fz runs list` 这类离线命令照常可用；只有真正要联网取数时才抛错。
- 多市场（crypto/期货/美股）的取数与凭据见[多市场适配](../concepts/multi-market.md)。

### 3.2 `FACTORZEN_LLM_*`（LLM 挖掘必填，其余可选）

`.env.example` 里 LLM 段的默认值是 `FACTORZEN_LLM_ENABLED=false`。**这个默认值只适合「报告里的 LLM 解读」这一类附加功能**（缺配置就静默跳过，报告照常出）。

> ⚠️ **`fz mine agent` / `fz mine team` 要求一个就绪的 LLM，缺配置时直接报错退出，不会静默降级。** 而且模板里的 `FACTORZEN_LLM_ENABLED=false` 会**强制**关掉 LLM，即使调用方内部传了 `enabled=True`。要跑 LLM 挖掘，这一项必须显式改成 `true`：

```bash
FACTORZEN_LLM_ENABLED=true
FACTORZEN_LLM_BASE_URL=<OpenAI 兼容网关地址>
FACTORZEN_LLM_API_KEY=<你的 key>
FACTORZEN_LLM_MODEL=<模型名>
```

四项（`ENABLED` / `BASE_URL` / `API_KEY` / `MODEL`）**全真才算就绪**，缺任何一项都视作未配置。

不跑 LLM 挖掘的话，这段可以完全不管——表达式搜索（`fz mine search`）、单因子评估、因子库准入、风险与组合、模拟交易全都不依赖 LLM。

**完整变量表**（含多 profile 切换、`FLAVOR` / `STREAM` 语义、`.env` 的两套加载器与容错细节）见[环境变量参考](../reference/environment.md)。

---

## 4. 验证安装

按下面三级验证，出问题时也便于定位是环境、数据还是代码的事。

### 4.1 依赖自检（不联网，秒级）

```bash
pixi run smoke
```

预期输出 `ok`。这一步只验证关键依赖能 import，不碰网络也不碰数据。

### 4.2 CLI 可用

```bash
pixi run fz --help
```

应打印 15 个顶层命令：`factor` `report` `data` `runs` `mine` `factor-library` `research` `validate` `risk` `portfolio` `sim` `strategies` `live` `combine` `ops`。

### 4.3 数据链路 smoke

```bash
# 完整版：先查 Tushare 连通性，再审计本地 data/raw/ 分区
pixi run smoke-data

# 离线版：跳过连通性检查，只审计本地已有数据
pixi run smoke-data --skip-tushare
```

连通性检查用一次轻量真实调用（5 天 `trade_cal`）验证 token 与网络；分区审计只读本地 `data/raw/`，可完全离线跑。退出码：`0` = 全部 ok，`1` = 出现 error，`2` = 只有 warning。

刚装完、本地还没有任何数据时，审计报 warning 是正常的——去跑[快速上手](quickstart.md)拉一批数据回来即可。

### 4.4 （可选）跑测试

```bash
pixi run lint        # ruff check，扫全仓
pixi run typecheck   # mypy，扫全 src/factorzen
timeout 900 pixi run test   # pytest -n auto，952 个测试
```

> ⚠️ **绝不要跑 `pixi run format`。** 全仓 `ruff format` 会一次改动数百个文件、污染 diff。格式问题请按 lint 报错逐处修。

---

## 5. 常见问题

**`pixi: command not found`**
pixi 装在 `~/.pixi/bin`，需要把它加进 `PATH`（安装脚本通常会写进 shell rc，重开终端即可）。

**`python: command not found` / 想直接跑 `python xxx.py`**
本项目没有全局 Python，这是预期行为。一切经 `pixi run python <args>` 或 `pixi run <task>`。

**`RuntimeError: TUSHARE_TOKEN ...`**
`.env` 没建、没填 token，或者不在仓库根目录执行命令。错误信息里同时给了 Windows `set` 与 `.env` 两种写法。

**`.env` 填了 token 但仍然报缺 token**
最常见的原因是 `.env` 带了 UTF-8 BOM，让首行的键名变成 `﻿TUSHARE_TOKEN` 而静默失效。加载器已用 `utf-8-sig` 打开来兼容这种情况，若仍失败，检查该行有没有被行内注释误剥——只有「空格 + `#`」才算注释起点。

**跑 `fz mine team` 报「LLM 未配置」**
见 §3.2：`.env.example` 模板里的 `FACTORZEN_LLM_ENABLED=false` 必须改成 `true`，并把 `BASE_URL` / `API_KEY` / `MODEL` 三项填全。

**`pixi run serve` 起不来，报缺 fastapi/uvicorn**
说明当前环境不是 pixi 默认环境（默认环境含 dev feature）。回到仓库根重新 `pixi install`；若你自建了 pip 环境，装 `factorzen[dev]`。

**依赖装到一半失败 / 想重来**
删掉项目内的 `.pixi/` 目录后重新 `pixi install`。这只会重装环境，不动 `data/` 与 `workspace/` 里的数据与产物。

**报告 HTML 里中文显示成方框**
matplotlib 缺 CJK 字体。字体优先级回退代码里已配好，装任一中文字体（如 `fonts-noto-cjk`）即可。这不影响任何数值结果。

---

## 下一步

- [快速上手](quickstart.md) —— 5 分钟跑通核心闭环：挖掘 → 增量准入 → 组合
- [端到端教程](end-to-end-tutorial.md) —— 完整链路：拉数据 → 挖掘 → 准入 → 风险 → 组合 → 模拟 → 报告
- [环境变量参考](../reference/environment.md) —— 全部变量、多 profile 切换、缺失行为
- [数据源与口径](../reference/data-sources.md) —— A 股数据源与单位口径；多市场见[多市场适配](../concepts/multi-market.md)

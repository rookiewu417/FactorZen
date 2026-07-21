# 环境变量参考

> [FactorZen](../../README.md) · [文档](../README.md) · **环境变量**

所有凭据从项目根的 `.env` 读取。仓库里有一份 `.env.example` 作模板：

```bash
cp .env.example .env
# 然后填入本地凭据
```

> ⚠️ **`.env` 已被 gitignore，绝不能提交。** 本文档以及任何记忆/计划文件里都不写真实 token 值，只写变量名。

---

## 1. 变量总表

### 1.1 `.env.example` 已列出的变量

| 变量 | 默认 / 示例 | 必需性 | 消费点 |
|---|---|---|---|
| `TUSHARE_TOKEN` | `your_tushare_token_here` | 真实取数**必需** | `config/tushare_config.py:42,48` |
| `TUSHARE_POINTS` | `2000` | 可选（示例中被注释） | `config/tushare_config.py:69` |
| `TUSHARE_MAX_RPS` | `5` | 可选（示例中被注释） | `config/tushare_config.py:70` |
| `FACTORZEN_LLM_ENABLED` | `false` | 见 §3 | `llm/config.py:194` |
| `FACTORZEN_LLM_BASE_URL` | `https://www.aiping.cn/api/v1` | LLM 功能必需 | `llm/config.py:225` |
| `FACTORZEN_LLM_API_KEY` | `your_llm_api_key_here` | LLM 功能必需 | `llm/config.py:226` |
| `FACTORZEN_LLM_MODEL` | `DeepSeek-V4-Pro` | LLM 功能必需 | `llm/config.py:227` |
| `FACTORZEN_LLM_TIMEOUT_SECONDS` | `30` | 可选 | `llm/config.py:203` |
| `FACTORZEN_LLM_MAX_TOKENS` | `700` | 可选 | `llm/config.py:204` |
| `FACTORZEN_LLM_THINKING` | `false`（示例中被注释） | 可选 | `llm/config.py:229` |
| `FACTORZEN_LLM_PROVIDER` | `DeepSeek` | 可选 | `llm/config.py:230` |

### 1.2 代码真实读取、但 `.env.example` 未列的变量

| 变量 | 默认 | 消费点 | 用途 |
|---|---|---|---|
| `FACTORZEN_LLM_PROFILE` | 未设 | `llm/config.py:77` | profile 切换，见 §2 |
| `FACTORZEN_LLM_FLAVOR` | `aiping` | `llm/config.py:219` | 上游适配风格，见 §2.2 |
| `FACTORZEN_LLM_STREAM` | 按 flavor 推导 | `llm/config.py:220` | 流式开关，见 §2.2 |
| `FACTORZEN_LLM_MAX_RETRIES` | `3` | `llm/config.py:205` | 交给 OpenAI SDK 的重试次数 |
| `FACTORZEN_POOL_SUBPROC` | 未设 | `cli/main.py:849` | `=1` 时池构建放子进程，退出全额归还内存；等效同名 CLI 旗标（`cli/parser.py:460` help 明说） |
| `FACTORZEN_NOTIFY_WEBHOOK` | — | `ops/config.py:49` | 无人值守运营通知 webhook。注意配置里的字段 `notify_url_env` **存的是变量名**，不是值 |
| `QLIB_PROVIDER_URI` | `~/.qlib/qlib_data/cn_data` | `builtin_factors/qlib/handler.py:91,145`、`core/data_ensure.py:253` | qlib 数据目录 |
| `QLIB_INSTRUMENTS` | `csi500` | `builtin_factors/qlib/handler.py:95,146` | qlib universe |
| `QLIB_KERNELS` | `1` | `builtin_factors/qlib/handler.py:109` | qlib 并行核数 |
| `QLIB_JOBLIB_BACKEND` | `threading` | `builtin_factors/qlib/handler.py:110` | qlib joblib 后端 |
| `TUSHARE_API_URL` | `http://api.tushare.pro` | `tools/download_tushare_lake.py:34` | 湖下载脚本的 API 端点 |
| `TUSHARE_LAKE_MAX_PER_MIN` | `140` | `tools/download_tushare_lake.py:36` | 每分钟请求上限 |
| `TUSHARE_LAKE_WORKERS` | `6` | `tools/download_tushare_lake.py:38` | 并发 worker 数 |

> ℹ️ `LLMConfig` 有一个 `temperature=0.2` 字段，但它**不从环境变量读取**——`load_llm_config` 从不设置它。要改温度需在代码里构造 `LLMConfig`。

---

## 2. LLM 配置详解

### 2.1 `FACTORZEN_LLM_PROFILE` —— profile 切换

FactorZen 支持在两套（或更多）上游之间运行时切换，避免为了换模型而反复编辑 `.env`。

**机制**（`llm/config.py:54-82`）：

- **未设 `FACTORZEN_LLM_PROFILE`** → 平铺模式，每个字段直接读 `FACTORZEN_LLM_<FIELD>`。这是历史默认，零回归。
- **设了 `FACTORZEN_LLM_PROFILE=foo`** → 每个字段**优先**读 `FACTORZEN_LLM_FOO_<FIELD>`（profile 名转大写），读不到再**回落**到平铺的 `FACTORZEN_LLM_<FIELD>`。

> ℹ️ **profile 名是任意字符串，不是枚举。** 代码里没有硬编码任何 profile 名——`FACTORZEN_LLM_SUB2API_*` 只是 docstring 与测试用的举例。你可以叫它任何名字，只要 `.env` 里的键名前缀对得上。

解析优先级：显式传入的 `profile=` 实参 > 环境/文件里的 `FACTORZEN_LLM_PROFILE`；空串视为未设。

可加 profile 前缀的字段全集：

```text
ENABLED  BASE_URL  API_KEY  MODEL  TIMEOUT_SECONDS  MAX_TOKENS
THINKING  PROVIDER  MAX_RETRIES  FLAVOR  STREAM
```

`.env` 写法示例（平铺默认 + 一个名为 `cockpit` 的第二 profile）：

```bash
# 平铺默认
FACTORZEN_LLM_ENABLED=true
FACTORZEN_LLM_BASE_URL=https://www.aiping.cn/api/v1
FACTORZEN_LLM_API_KEY=<your-key>
FACTORZEN_LLM_MODEL=DeepSeek-V4-Pro
FACTORZEN_LLM_PROVIDER=DeepSeek

# 第二 profile
FACTORZEN_LLM_COCKPIT_BASE_URL=http://localhost:8080/v1
FACTORZEN_LLM_COCKPIT_API_KEY=<your-key>
FACTORZEN_LLM_COCKPIT_MODEL=<model-name>
FACTORZEN_LLM_COCKPIT_FLAVOR=openai

# 切到第二 profile（注释掉这行就回平铺默认）
FACTORZEN_LLM_PROFILE=cockpit
```

生效的 profile 名会记进 `LLMConfig.profile` 供审计。

### 2.2 `FLAVOR` 与 `STREAM`

| 变量 | 取值 | 缺省 | 非法值行为 |
|---|---|---|---|
| `FACTORZEN_LLM_FLAVOR` | `aiping` / `openai` | `aiping` | 抛 `ValueError("非法 LLM flavor=...")` |
| `FACTORZEN_LLM_STREAM` | `1/true/yes/on` 或 `0/false/no/off` | 按 flavor 推导 | 抛 `ValueError("非法 FACTORZEN_LLM_STREAM=...")` |

`STREAM` 未显式设置时按 flavor 取缺省（`llm/config.py:164-168`）：

| flavor | stream 缺省 | 理由 |
|---|---|---|
| `aiping` | `True` | 沿现状 |
| `openai` | `False` | 本地 OpenAI 兼容网关实测长流式会中途断流 |

> ⚠️ 用本地/自建的 OpenAI 兼容网关时，把 `FLAVOR` 设成 `openai` 就能自动关掉流式。若你手动开了 `STREAM=true` 又碰到长回复中途截断，那多半就是网关的流式不稳，改回非流式即可。

### 2.3 `is_ready` 与 `ENABLED` 的两级判定

`LLMConfig.is_ready` = `enabled` ∧ `base_url` ∧ `api_key` ∧ `model` **四者全真**。缺任何一项都算未就绪。

`enabled` 本身是两级与运算（`llm/config.py:194-201`）：

```text
final_enabled = 调用方传入的 enabled  AND  (FACTORZEN_LLM_ENABLED 不是 0/false/no/off)
```

两个方向都成立：

1. **调用方不显式传 `enabled=True` 就是关的。**「LLM 解读」这类附加功能默认不传，所以默认不开。
2. **`FACTORZEN_LLM_ENABLED=false` 能强制关闭**，即使调用方传了 `enabled=True`。

### 2.4 缺失 LLM 配置时的行为

两类消费方，行为**不同**：

| 场景 | 缺配置时的行为 |
|---|---|
| `fz mine agent`、`fz mine team` | **直接报错退出**：`RuntimeError("LLM 未配置：设置 .env 的 FACTORZEN_LLM_* 或注入 llm_fn")`（`pipelines/factor_mine_agent.py:19`、`pipelines/factor_mine_team.py:25`） |
| 报告里的「LLM 解读」附加功能 | **静默跳过**，报告照常生成，只是少一段解读文字 |

> ⚠️ **这是 `.env.example` 里 `FACTORZEN_LLM_ENABLED=false` 最容易误导人的地方。** 模板给的默认值是给「解读附加功能」用的（默认关），但 `fz mine agent` / `fz mine team` 要求就绪的 LLM——**必须把它改成 `true`**，否则 `load_llm_config(enabled=True)` 会被这个 `false` 强制关掉，agent 带着「LLM 未配置」中止。这一点 `.env.example` 的注释里也专门标了。

---

## 3. `.env` 的加载机制（两套，各自独立）

FactorZen 里有**两个互不相干**的 `.env` 读取器，行为不同：

| 读取器 | 时机 | 范围 | 优先级 |
|---|---|---|---|
| `config/tushare_config.py::_load_dotenv` | **import 时**执行 | 全部键，填充 `os.environ` | **不覆盖**已存在的环境变量 |
| `llm/config.py::_read_env_file` | `load_llm_config` 调用时 | **只收集 `FACTORZEN_LLM_` 前缀键** | 真实环境变量优先（`os.getenv(name) or file_values.get(name)`，`llm/config.py:51`） |

两者都是「已存在的真实环境变量赢」，所以临时覆盖一次运行可以直接在命令前加变量：

```bash
FACTORZEN_LLM_PROFILE=cockpit pixi run -- fz mine team --market ashare
```

### 3.1 Tushare 加载器的容错

`_load_dotenv` 做了不少防御，都是踩过坑补的：

- **用 `utf-8-sig` 打开以容忍 BOM**。历史 bug：用 `utf-8` 会把 BOM 读进首行，让 `TUSHARE_TOKEN` 变成 `﻿TUSHARE_TOKEN` 而**静默失效**（`config/tushare_config.py:14`）。
- 容忍 CRLF 行尾、首尾空白、成对引号。
- 剥行内注释，但**只有「空格 + `#`」才算注释起点**。所以 URL 里的 fragment（`...#anchor`，`#` 前无空格）不会被误剥。

### 3.2 延迟校验

Token 校验是**延迟**的，不在 import 阶段执行：

- `TUSHARE_TOKEN` 缺失时 import 不崩。
- 真正需要取数时 `ensure_token()` 才抛 `RuntimeError`，错误信息同时给出 Windows `set` 与 `.env` 两种写法（`config/tushare_config.py:51-53`）。

`_int_env`（读 `TUSHARE_POINTS` / `TUSHARE_MAX_RPS`）同样 import 期安全：剥行内注释、非数字回退默认值。

> ✅ 这个设计的意义：**`.env` 写错不会让整个 CLI 在 import 阶段崩掉**，连不需要联网的离线命令（`fz factor list`、`fz ops validate-config`、`fz runs list`）都还能正常用。

---

## 4. 自检

确认依赖与数据链路：

```bash
# 依赖 import 自检（不联网）
pixi run smoke

# 数据链路 smoke：连通性检查需要 TUSHARE_TOKEN；本地 data/raw/ 审计可离线跑
pixi run smoke-data
```

数据源本身的接口清单与单位口径见 [数据源与口径](data-sources.md)。

# FactorZen 下一步进化计划（专注低频）

> **注:** 本文从最近的干净历史版本恢复(原 UTF-8 版本在一次合并中被编码损坏)。部分路径示例反映包合并前的布局;最新布局以 [README](../README.md) 与 [architecture](architecture.md) 为准。最新企业级升级计划见 [docs/superpowers/plans/2026-06-03-enterprise-grade-daily-platform.md](superpowers/plans/2026-06-03-enterprise-grade-daily-platform.md)。

## Context

FactorZen 的低频（daily/周/月）研究闭环已经完整：因子 → 预处理 → IC → 分层回测 → walk-forward OOS → 归因 → 实验 manifest → HTML 报告，并有 lint/mypy/coverage(70%) 质量门。日内（intraday）只是分钟级 IC 验证管线、未纳入质量门；Tick 级研究当前不保留正式代码包。

用户当前**专注低频**，并已决策：
- **Phase 1 重心 = 数据完整性与可靠性**（"数据进来是对的、拉取逻辑被测过"是一切下游结论的信任根基）。
- **intraday 与 Tick 级研究本版 roadmap 全部冻结**，不投入新功能，待低频稳固后再单独立项。

本计划据此把用户原"维护顺序"重排为以低频可信度为轴的四个阶段，Phase 1 给出可直接执行的细节。

## 重排后的优先级原则

数据正确性 → 可复现性 → 研究结论可信度 → 工程门统一。越靠下游的结论，越依赖上游数据可信，因此先做数据层。

---

## Phase 1（现在执行）：数据完整性与可靠性

目标：让"原始数据是否完整"可被一条命令审计，让 `common/loader.py` 的拉取/重试/缓存逻辑被 mock 测试覆盖。

### WS1 — 原始数据完整性审计报告（新能力）

现有 `common/data_quality.py:build_daily_quality_report` 是**单 run 级**检查，**不覆盖** `data/raw/` 分区的横向完整性。新增一个**原始数据层**审计，与之互补、不替换。

- 新增 `common/data_audit.py`（与 `data_quality.py` 区分：前者审 raw 分区，后者审单次 run）。核心函数 `build_raw_data_audit(*, data_type, start, end, universe_codes) -> dict`，针对 `daily` / `daily_basic` / `finance` 三类分区，对照交易日历（复用 `common/calendar.py`）与目标股票池（复用 `common/universe.py`）报告：
  - **日期缺口**：相对 `get_trade_dates` 期望集合，缺失的 trade_date 列表/计数。
  - **股票覆盖**：每个 trade_date 实际覆盖的 ts_code 数 vs universe 规模（复用 `_universe_stats` 的覆盖率口径思路）。
  - **字段空值率**：`daily_basic` 的 pe/pb/total_mv/circ_mv、`finance` 的关键财务字段逐列 null 占比（复用 `data_quality._value_stats` 的统计风格）。
  - **finance PIT 缺口**：每只票最近一期财报的 `ann_date` 是否过期（结合 `daily/data/pit.py` 的 PIT 对齐口径），标记长期未更新的票。
- 分区读取走 `common/storage.py`（与 loader 写入一致），不直连 Tushare —— 审计的是**本地已落盘数据**，可在无网络/无 token 时运行。
- 报告 JSON 结构对齐 `build_daily_quality_report` 的 `{status, checks, warnings, errors}` 形态，便于后续统一报告元数据。
- CLI 入口 `scripts/audit_raw_data.py`，参数 `--data-type --universe --start --end`，输出 JSON + 人类可读摘要到 `output/logs/` 或 stdout。

### WS2 — `common/loader.py` 的 mock Tushare 测试（新增 `tests/test_loader.py`）

当前**无任何测试直接覆盖 loader 函数**（universe 集成测试 skipif token 不算）。参照 `tests/test_benchmark.py` / `test_calendar.py` 的 monkeypatch / mock 模式，mock 模块级 `_pro` 与 `init_tushare`，构造合成 pandas DataFrame，覆盖：

- **分段拉取**：`fetch_daily`/`fetch_daily_basic` 按年分段、`fetch_finance` 按季度 + 50 只批次（`_FINANCE_BATCH_SIZE`）、`fetch_minute` 按月分段 —— 断言调用次数与区间切分正确。
- **重试分类**（`_retry`，loader.py:69）：网络/超时类 → 重试；参数/权限/积分类 → 立即抛出不重试；空结果 → 触发重试；`stk_mins` 频率超限 → 等待路径（mock `time.sleep` 断言被调用，不真正 sleep 62s）。
- **缓存跳过**：`partition_exists` 命中时跳过该分段拉取（断言 `_pro` 对应方法**未被调用**）；`fetch_stock_basic` 的 mtime + `CACHE_EXPIRE_DAYS` 7 天逻辑。
- **pandas → polars 转换**：mock 返回 pandas，断言输出为 polars 且 schema/列名正确。
- **限流** `_rate_limit`：mock 时钟，断言 ≤ `MAX_RPS`。

### WS3 — 把 loader/storage/universe 纳入质量门（小改）

数据层是可靠性核心却不在 mypy/coverage 范围内（当前范围：common, daily/evaluation, daily/preprocessing, daily/factors, reporting, automation —— `common` 已含 loader/universe/storage，**确认它们已在 source 内**）。复核 `pyproject.toml` `[tool.coverage.run].source` 与 `[tool.mypy].files`：

- 若 `common` 整体在内则无需改范围，只需 WS1/WS2 新增代码自然计入；
- 新增 `common/data_audit.py` 自动落入 `common` 范围，需保证 mypy 通过、被测试覆盖以不拖低 70% 门槛。

**Phase 1 关键文件**：新增 `common/data_audit.py`、`scripts/audit_raw_data.py`、`tests/test_loader.py`、`tests/test_data_audit.py`；复用 `common/{calendar,universe,storage,data_quality}.py`、`daily/data/pit.py`。

---

## Phase 2：可复现性基础设施

目标：每个实验的股票池、参数、结果可追溯复现。

- **universe 快照落盘**：当前 `common/universe.py` 仅缓存指数成分（`_load_index_members`），无"组合后实际使用的 universe"快照，且降级（指数加载失败→全 A）是静默 `logger.warning`。新增：run 开始时把 `get_universe` 最终明细 + 是否降级标志落盘到实验目录，供复现与"严肃研究前检查 universe 明细"。
- **实验结果索引/数据库**：现有 `common/experiment.py` 写 per-run manifest，但无跨 run 检索。新增轻量索引（SQLite 或 append-only parquet/jsonl），登记 manifest 路径、因子、universe 快照、关键指标，支持按因子/日期/universe 查询历史实验。

---

## Phase 3：研究结论可信度

目标：防止研究结论被高估或误读。

- **combination 样本内边界更醒目**：`research/combination/methods.py` 的 `ic_weighted`/`max_ir` 用**样本内 IC** 估权重。在 HTML 报告（`reporting/tear_sheet.py` + 模板）与 manifest 里加显著 in-sample 警示标识，避免被当作 OOS 组合表现。
- **更真实的成交/冲击模型 + 收口 adv TODO**：
  - `common/config_loader.py:103-106` 选 `square_root_impact` 时**未传参**，全用默认 `alpha=0.1`/`fallback_adv=1e7`。让 config 可配 `alpha`/`fallback_adv` 并透传到 `SquareRootImpactCostModel`。
  - 收口 `daily/evaluation/backtest.py:99-100` 的 `adv_20d` 配置层 TODO（回测循环已用 `_compute_adv_20d` 填充，但配置入口仍是空 dict —— 统一来源、删除误导性 TODO 或补全注释）。
  - 为 cost model 增加更多场景测试（adv 缺失、极端换手、零成交额）。

---

## Phase 4：质量门与报告统一收口（低频范围内）

- **冻结模块不纳入**：按冻结决定，**不**把 intraday 或 Tick 级研究纳入 mypy/coverage。
- 把 Phase 1–3 新增模块（data_audit、experiment 索引、cost model 参数化）确保都在门控内。
- 统一日频报告元数据格式（为将来与日内对齐留接口，但本版只做日频侧）。
- **真实数据 smoke 手动命令**：现有 `pixi run smoke` 仅验证依赖可导入。新增一个**手动触发、不入默认 CI** 的真实数据 smoke（如 `pixi run smoke-data`），跑一次最小真实拉取 + WS1 审计，需 token，文档标注手动运行。

---

## 冻结项（本版不投入）

- **intraday**：分钟级 IC 管线维持现状，不纳入质量门、不加新因子、不接真实分钟 smoke。待低频稳固后单独立项。
- **Tick 级研究**：不保留正式代码包。需独立数据供应商 adapter（CTP/Wind）与订单簿/逐笔存储设计，远期单独立项。

---

## Verification

- **WS2/单测**：`pixi run test`（新增 `test_loader.py`、`test_data_audit.py` 全绿）、`pixi run typecheck`、`pixi run coverage`（≥70% 不被新代码拖低）、`pixi run lint`。
- **WS1 端到端**：在已有本地 `data/raw/` 数据上跑 `python scripts/audit_raw_data.py --data-type daily_basic --universe csi300 --start 2023-01-01 --end 2023-12-31`，确认能正确报出日期缺口/字段空值率/覆盖率，且无网络/无 token 也能运行。
- **WS2 离线性**：断网/清空 `TUSHARE_TOKEN` 跑 `test_loader.py` 仍全绿（证明完全 mock、无真实网络依赖）。
- **回归**：完整 `pixi run lint && pixi run typecheck && pixi run test && pixi run coverage`，确保现有 ~60 个测试不回归。

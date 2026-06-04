# FactorZen 2026 演进计划

本文记录当前公开版本的演进方向。最新目录结构以 [README](../README.md) 和 [architecture](architecture.md) 为准。

## 当前定位

FactorZen 的主线是低频单因子研究闭环：

```text
本地数据缓存
  -> PIT 数据上下文
  -> 因子计算
  -> 预处理
  -> IC / 分层回测 / walk-forward
  -> 数据质量与实验 manifest
  -> Tear Sheet HTML 报告
```

项目暂不扩展实盘交易、OMS/EMS、撮合、Tick 数据接入或生产组合执行闭环。`intraday/` 保留为分钟线研究代码，但不作为当前路线图主线。

## 优先级原则

1. 数据正确性优先于下游结论。
2. 可复现性优先于展示效果。
3. 报告结论必须暴露样本不足、覆盖率不足和模块缺失。
4. 新能力进入主线前必须有测试和质量门保护。

## Phase 1：数据完整性与可靠性

目标：让本地数据是否完整、是否可用于研究，在无网络环境下也能被审计。

- 完善 `src/factorzen/core/data_audit.py`，覆盖 `daily`、`daily_basic`、`finance` 等本地分区的数据缺口、股票覆盖率和关键字段空值率。
- 保持 `tests/test_data_audit.py` 和 `tests/test_loader.py` 全量 mock，不依赖真实 Tushare token 或本地 `data/`。
- 将真实数据 smoke 作为手动命令，而不是默认 CI 步骤；CI 保持离线可重复。

建议验证：

```bash
pixi run pytest tests/test_data_audit.py tests/test_loader.py -q
pixi run lint
pixi run typecheck
pixi run coverage
```

## Phase 2：实验可复现性

目标：任何一次研究运行都能追溯输入、配置、代码版本和输出。

- 继续强化 `src/factorzen/core/experiment.py` 的 manifest 元数据，确保失败运行也记录状态和错误。
- 保持 `workspace/factor_evaluations/{run_id}/` 为单次运行的标准输出位置。
- 维护 `experiment_index.jsonl`，便于跨 run 检索因子、universe、状态和报告路径。
- 对工作树 dirty、lockfile 变化、配置变更给出明确提示。

## Phase 3：研究结论可信度

目标：降低过拟合、误读和交易可行性高估。

- 对 `research/combination/` 的样本内权重估计保持醒目标注，避免把样本内组合结果误读为 OOS 结果。
- 持续完善成本模型和容量约束，尤其是 `square_root_impact` 的参数、ADV 缺失和极端换手场景。
- 对收益对齐、涨跌停、停牌、容量和 rebalance threshold 的回归测试保持高优先级。

## Phase 4：报告和用户体验

目标：让报告更适合研究复核，而不是只展示漂亮图表。

- 保持 Tear Sheet 的结论、证据、限制和复现信息并列呈现。
- 对缺失模块显示明确状态和下一步建议。
- 避免在报告中隐藏样本不足、覆盖率不足或 OOS 不成立的问题。

## 非目标

- 不内置商业行情数据。
- 不承诺生产交易或实盘执行能力。
- 不把真实 Tushare 网络请求放入默认 CI。
- 不把本地 `data/`、`workspace/runs/`、`workspace/factor_evaluations/` 的运行产物提交到仓库。

## 发布前检查

```bash
pixi run lint
pixi run typecheck
pixi run test
pixi run coverage
git status --short
```

同时确认 `.env`、本地行情数据、运行日志、代理状态目录和任何 token 都没有被跟踪。

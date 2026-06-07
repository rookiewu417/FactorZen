# 示例报告

> [FactorZen](../../README.md) · [文档](../README.md) · **示例报告**

本目录收录一份由 FactorZen 真实评估流程生成的 Tear Sheet，供你在不自己跑数据的情况下先看产出长什么样、怎么读。

## 文件

| 文件 | 说明 |
|------|------|
| [`volume_return_corr_20d-tear-sheet.html`](volume_return_corr_20d-tear-sheet.html) | 示例因子的完整 HTML 报告（自包含，约 1.1MB）|

> GitHub 不会内联渲染 HTML。请点开后下载用浏览器打开，或借助 htmlpreview 类工具在线预览。

## 示例因子

`volume_return_corr_20d` —— 20 日「1 日收益 × 对数成交量」滚动 Pearson 相关，衡量量价同步 / 背离程度。实现见 [`workspace/factors/daily/volume_return_corr_20d.py`](../../workspace/factors/daily/volume_return_corr_20d.py)。

## 运行配置

| 项 | 值 |
|----|----|
| universe | csi500 |
| 区间 | 2016-06-06 ~ 2026-06-06 |
| benchmark | 000905.SH |
| seed | 42 |
| 预处理 | MAD 去极值 · zscore 标准化 · industry+size 中性化 |
| 策略套件 | topn_50 · quantile_ls_5 · factor_weighted_ls · optimizer_mv_long_only |
| IC | both（Rank + Pearson）· 中性化 IC · event study |
| walk-forward | 关闭（默认）|
| LLM 解读 | 开启 |

复现命令：

```bash
pixi run fz factor run --config workspace/configs/daily/volume_return_corr_20d.yaml
pixi run fz report path <run_id>
```

实跑产物落在 `workspace/factor_evaluations/{run_id}/`（默认不入库）；本页 HTML 是其中一次运行的快照。

## 这是一个「弱因子」示例

示例特意选了一个预测能力不强的因子，用来展示 FactorZen 如何**诚实暴露问题**而非用漂亮图表掩盖（设计原则 #3）：

| 指标 | 值 | 解读 |
|------|----|----|
| IC 均值 | ≈ -0.016 | 方向稳定但绝对值很小 |
| IC t 统计 | ≈ -11.5 | 统计显著（长样本放大了显著性）|
| IR | ≈ -0.23 | 信息比率低 |
| 样本外 IC | ≈ -0.014 | 略有衰减 |
| 多空年化 | ≈ 0.3% | 经济意义弱 |
| 夏普 | ≈ 0.08 | 收益不稳定 |
| 换手率 | ≈ 48% | 偏高，成本侵蚀收益 |

结论：统计上存在、经济上微弱、交易上不划算——不建议单独使用。这正是报告该说清楚的事。

## 报告分区导读

报告按「结论 → 证据 → 限制 → 复现」组织，自上而下：

| 分区 | 看什么 |
|------|--------|
| 研究仪表盘 | 第一屏速览：评分、星级、关键指标与一句话结论 |
| 综合评估 | 规则引擎给出的结论、评级、证据强度、风险与下一步（确定性，不依赖 LLM）|
| 大模型研究解读 | 可选的 LLM 解读（本快照为当前版本，呈现评级 / 风险 / 建议等字段）|
| 跨策略对比 | 4 个策略并排对比收益、夏普、回撤、换手 |
| 收益表现 | 主策略 NAV、多空 NAV、月度收益与分位价差 |
| 预测能力 | Rank / Pearson IC、中性化 IC、多持有期一致性、HAC t 统计 |
| 结构检验 | 单调性、Rank 自相关、因子相关性、分组分层 |
| 交易可行性 | 换手、成本、容量约束 |
| 稳健性验证 | walk-forward 等样本外检验（本例 walk-forward 关闭，会标注为未运行）|
| 风险归因 | 市值 / 行业 / 市场状态分层归因 |
| 附录 | 复现摘要：配置、命令、git SHA、lockfile hash、阶段耗时与模块状态 |

读报告的顺序建议：先看仪表盘与综合评估拿结论，再用预测能力 / 交易可行性 / 稳健性核对证据，最后看附录确认可复现。

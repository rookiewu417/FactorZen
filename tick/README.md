# tick/ — Tick 数据因子（预留位）

## 当前状态

**未实现** — 此目录为预留占位，当前不包含任何可运行代码。

## 为什么预留

本项目当前阶段专注于**单因子研究**（日频/周频/月频 + 分钟级 IC 验证），
Tick 数据因子属于更高复杂度的下一阶段工作。

主要障碍：
- Tushare Pro 不提供 Tick 行情数据
- 需要接入 CTP、Wind 或其他行情终端
- 订单簿重建、逐笔成交分析的存储方案需要单独设计

## 接口草稿（设计规划，代码未实现）

未来实现时，`tick/factors/base.py` 将定义 `TickFactor` 抽象基类：

```python
class TickFactor(BaseFactor):
    frequency: str = "tick"
    required_data: list[str] = ["tick"]

    @abstractmethod
    def compute(self, ctx: "TickDataContext") -> pl.DataFrame:
        """Returns: trade_time, ts_code, factor_value"""
        ...
```

## 未来接入路径

1. 对接 CTP 实时行情或 Wind Tick 历史数据
2. 实现 `TickDataContext` 加载逐笔成交 / 订单簿快照
3. 在此目录下实现具体因子（买卖压力、高频反转、订单失衡等）
4. 在 `intraday/evaluation/` 的 IC 框架上扩展 Tick 级评估

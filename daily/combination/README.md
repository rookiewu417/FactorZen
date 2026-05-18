# daily/combination/ — 多因子合成

> 状态：实验性研究工具。`ic_weighted` 和 `max_ir` 当前使用样本内 IC 估计权重，适合方法对比和候选因子筛选，不应用作无偏的样本外组合表现。

## 实现的合成方法

| 方法 | 模块 | 说明 |
|------|------|------|
| 等权平均 | `methods.equal_weight` | 各因子截面 z-score 后取均值 |
| IC 加权 | `methods.ic_weighted` | 历史 IC 均值为权重（仅取正向 IC 因子）|
| 最大化 IR | `methods.max_ir` | 闭式解 w = Σ^{-1}·μ，Ledoit-Wolf 协方差收缩 |

## 快速使用

```python
from daily.combination.methods import equal_weight, ic_weighted, max_ir

# factor_dfs: dict[str, pl.DataFrame]，每个 df 含 trade_date, ts_code, factor_value
# ret_df: 含 trade_date, ts_code, ret 的前向收益

combined = equal_weight(factor_dfs)
combined = ic_weighted(factor_dfs, ret_df, ic_window=60)
combined = max_ir(factor_dfs, ret_df, lookback=120)
```

## 一体化评估

```python
from daily.combination.pipeline import combine_and_evaluate

combined_df, ic_result, bt_result = combine_and_evaluate(
    factor_dfs, price_df, method="ic_weighted"
)
print(ic_result.summary())
```

## CLI

```bash
pixi run python scripts/run_combination.py \
    --factors momentum_20d reversal_5d volatility_20d \
    --method ic_weighted \
    --start 20240101 --end 20250101
```

## 参考文献

- Barra USE4 Risk Model Handbook
- Lopez de Prado, *Advances in Financial Machine Learning* (2018)
- Fama & MacBeth (1973), *Risk, Return, and Equilibrium*

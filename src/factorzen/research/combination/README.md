# research/combination/ —— 多因子合成

> **样本外(OOS)口径。** 通过 purged & embargoed walk-forward CV 逐折「train 估权 / test 应用」,
> 消除线性方法早期版本的样本内估权偏差。`methods.*` 的直接调用仍是**样本内**研究口径
> (方法对比/候选筛选用),无偏的样本外组合表现请走 `oos.combine_oos` / `models.combine_lgbm`
> 或 `experiment.run_combination_experiment`。

## 合成方法

| 方法 | 模块 | 说明 |
|------|------|------|
| 等权平均 | `methods.equal_weight` | 各因子截面 z-score 后取均值 |
| IC 加权 | `methods.ic_weighted` | 历史 IC 均值为权重（仅取正向 IC 因子）|
| 最大化 IR | `methods.max_ir` | 闭式解 w = Σ⁻¹·μ，Ledoit-Wolf 协方差收缩 |
| **LightGBM** | `models.combine_lgbm` | 树模型学非线性/交互,截面 rank 标签,滚动训练 |

估权与应用已拆分(`estimate_*_weights` / `apply_weights`),供 OOS 协议逐折调用。

## 样本外(OOS)对比 —— 推荐入口

```python
from factorzen.research.combination.cv import PurgedWalkForwardCV
from factorzen.research.combination.experiment import run_combination_experiment

# factor_dfs: {name: DataFrame(trade_date, ts_code, factor_value)}
# ret_df:     DataFrame(trade_date, ts_code, ret) — 对齐因子日的前向收益
cv = PurgedWalkForwardCV(train_days=120, test_days=20, purge_days=5, embargo_days=0)
res = run_combination_experiment(
    factor_dfs, ret_df, cv=cv,
    methods=["equal_weight", "ic_weighted", "max_ir", "lgbm"], seed=0,
)
print(res["comparison"])  # method × {rank_ic_mean, icir, top_bottom_spread, max_drawdown, ...}
```

产物落 `workspace/combinations/<run_id>/`:`combined_<method>.parquet`、`comparison.csv`、
`importance.csv`(lgbm SHAP/gain)、`report.md`、`manifest.json`(cv 参数/seed/git_sha)。

## 命令行

```bash
fz combine run --factor fa.parquet --factor fb.parquet --ret ret.parquet \
  --train-days 120 --test-days 20 --purge-days 5 --methods all --seed 42 --run-id exp1
```

因子 parquet 可来自因子评估产物或 `fz mine export-alpha` 导出的 α 截面。

## 防泄漏保证

`oos.for_each_fold` 是线性与树模型共用的逐折骨架:估权/训练只用 train 段(因子+收益),
应用/预测只用 test 段(因子,不碰收益),配合 CV 的 purge(剔 train 末尾与 test 标签重叠段)
与 embargo。**泄漏探针测试**(扰动 cutoff 后收益、断言 cutoff 前 OOS 值逐行不变)常驻覆盖全部方法。

## 参考文献

- Barra USE4 Risk Model Handbook
- López de Prado, *Advances in Financial Machine Learning* (2018) — purged CV / embargo
- Fama & MacBeth (1973), *Risk, Return, and Equilibrium*

# daily/combination/ — 多因子合成（预留位）

## 当前状态

**未实现** — 此目录为预留占位。

## 为什么预留

本项目当前阶段**专注于单因子研究**：
- 理解单个因子的 IC、稳健性、IC 衰减、换手率特征
- 在日频和分钟频率上交叉验证同一因子的有效性

多因子合成是**下一阶段**的产物，要求先有充分经过检验的单因子库。

## 接口草稿

```python
class FactorCombiner(ABC):
    """多因子合成器基类。"""

    @abstractmethod
    def combine(
        self,
        factor_dict: dict[str, pl.DataFrame],  # {factor_name: factor_df}
        ret_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """返回合成因子 DataFrame: trade_date, ts_code, factor_value"""
        ...
```

## 合成方法参考

| 方法 | 说明 | 适用场景 |
|------|------|---------|
| IC 加权 | 以历史滚动 IC 均值为权重 | 因子 IC 差异较大时 |
| 等权平均 | 直接平均 z-score 化后的因子值 | 快速基线 |
| PCA 第一主成分 | 捕捉因子共同信息 | 因子高度相关时降维 |
| Fama-MacBeth | 截面回归系数加权 | 学术验证 |

## 参考文献

- Barra USE4 Risk Model Handbook
- Lopez de Prado, *Advances in Financial Machine Learning* (2018)
- Fama & MacBeth (1973), *Risk, Return, and Equilibrium*

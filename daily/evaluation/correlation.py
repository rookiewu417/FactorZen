"""因子截面相关性分析。"""

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class CorrelationResult:
    factor_names: list[str]
    corr_matrix: np.ndarray      # 相关性矩阵

    def summary(self) -> str:
        lines = ["Factor Correlation:"]
        for i, name in enumerate(self.factor_names):
            corrs = ", ".join(
                f"{self.factor_names[j]}={self.corr_matrix[i][j]:.3f}"
                for j in range(len(self.factor_names)) if i != j
            )
            lines.append(f"  {name}: {corrs}")
        return "\n".join(lines)


def compute_factor_correlation(
    factor_dict: dict[str, pl.DataFrame],
    factor_col: str = "factor_clean",
) -> CorrelationResult:
    """计算多个因子的截面相关性。

    Args:
        factor_dict: {factor_name: DataFrame(trade_date, ts_code, {factor_col})}
        factor_col: 因子列名

    Returns:
        CorrelationResult，含相关性矩阵
    """
    names = list(factor_dict.keys())
    if len(names) < 2:
        return CorrelationResult(factor_names=names, corr_matrix=np.eye(len(names)))

    # 合并所有因子到一个 DataFrame
    merged = None
    for name, df in factor_dict.items():
        renamed = df.select(["trade_date", "ts_code", pl.col(factor_col).alias(name)])
        if merged is None:
            merged = renamed
        else:
            merged = merged.join(renamed, on=["trade_date", "ts_code"], how="inner")

    if merged is None or merged.is_empty():
        return CorrelationResult(factor_names=names, corr_matrix=np.eye(len(names)))

    # 对每个日期算截面相关性，然后平均
    dates = merged["trade_date"].unique().sort().to_list()
    n = len(names)
    cum_corr = np.zeros((n, n))
    count = 0

    for d in dates:
        cross = merged.filter(pl.col("trade_date") == d).drop_nulls()
        if len(cross) < 30:
            continue
        arr = np.column_stack([cross[name].to_numpy() for name in names])
        corr = np.corrcoef(arr.T)
        cum_corr += corr
        count += 1

    if count > 0:
        cum_corr /= count

    np.fill_diagonal(cum_corr, 1.0)
    return CorrelationResult(factor_names=names, corr_matrix=cum_corr)

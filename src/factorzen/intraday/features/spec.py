"""日内特征规格与电池入口。"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from factorzen.intraday.sessions import ASHARE_BAR_FREQS, normalize_freq


@dataclass(frozen=True)
class IntradayFeatureSpec:
    """单条日内日频特征的声明性规格。

    Attributes:
        name: 叶子名，统一 ``i_`` 前缀。
        freq: 计算频率（由 ``battery(freq=...)`` 参数化）。
        pre: bar 级辅助列（``with_columns`` 阶段）。
        agg: ``group_by(["ts_code","trade_date"]).agg`` 的标量表达式。
        formula: 精确数学定义（写入 manifest）。
        description: 中文一句话语义。
        source: 预留 ``"expression"``（二期 LLM 表达式特征）；v1 为 ``"builtin"``。
        expression: ``source="expression"`` 时的日内表达式串；v1 恒 ``None``。
    """

    name: str
    freq: str
    pre: tuple[pl.Expr, ...]
    agg: pl.Expr
    formula: str
    description: str
    source: str = "builtin"
    expression: str | None = None


def battery(version: str = "v1", freq: str = "5min") -> list[IntradayFeatureSpec]:
    """按版本与频率返回特征电池规格列表。

    Args:
        version: 电池版本，目前仅 ``"v1"``。
        freq: bar 频率，经 ``normalize_freq``；``minutes > 30`` 时拒绝
            （``k30 = 30 // minutes`` 必须 ≥ 1）。

    Returns:
        特征规格列表（v1 共 17 个）。

    Raises:
        ValueError: 未知 version，或频率不支持 v1 电池。
    """
    freq_n = normalize_freq(freq)
    minutes = ASHARE_BAR_FREQS[freq_n].minutes
    if minutes > 30:
        raise ValueError(
            f"v1 电池不支持 freq={freq_n!r}（minutes={minutes}>30，k30 必须≥1）"
        )
    if version == "v1":
        from factorzen.intraday.features.battery_v1 import battery_v1

        return battery_v1(freq_n)
    raise ValueError(f"未知电池版本: {version!r}，支持 ['v1']")


def battery_hash(specs: Sequence[IntradayFeatureSpec]) -> str:
    """电池内容哈希：``sha256("|".join(name:freq:formula))`` 前 16 位十六进制。"""
    payload = "|".join(f"{s.name}:{s.freq}:{s.formula}" for s in specs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

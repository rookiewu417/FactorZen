"""Purged & embargoed walk-forward 时间序列交叉验证切分。

面向多因子组合的样本外(OOS)估权协议(López de Prado, AFML):
- purge:剔除 train 末尾与 test 标签窗口重叠的样本(前向收益 horizon 天)防泄漏;
- embargo:test 前额外隔离带,进一步降低序列自相关导致的泄漏;
- expanding/rolling:训练集展开或定长滚动。

split 返回逐折 (train_dates, test_dates),test 段首尾相接覆盖 train_days 之后全部日期。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PurgedWalkForwardCV:
    """时间序列 walk-forward 切分,带 purge/embargo 防泄漏。"""

    train_days: int
    test_days: int
    purge_days: int
    embargo_days: int = 0
    expanding: bool = True

    def split(self, dates: list[str]) -> list[tuple[list[str], list[str]]]:
        """把升序唯一交易日切成逐折 (train, test)。

        Raises:
            ValueError: 日期数不足以构成任何一折。
        """
        ds = list(dates)
        n = len(ds)
        folds: list[tuple[list[str], list[str]]] = []
        i = 0
        while True:
            test_start = self.train_days + i * self.test_days
            if test_start >= n:
                break
            test = ds[test_start : test_start + self.test_days]
            if not test:
                break
            train_end = test_start - self.purge_days - self.embargo_days
            train_start = 0 if self.expanding else max(0, test_start - self.train_days)
            train = ds[train_start:train_end] if train_end > train_start else []
            if train:
                folds.append((train, test))
            i += 1
        if not folds:
            raise ValueError(
                f"日期数({n})不足以构成一折"
                f"(train_days={self.train_days}, test_days={self.test_days})"
            )
        return folds

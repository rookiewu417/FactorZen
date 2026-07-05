"""Purged & embargoed walk-forward CV 切分协议的测试。"""
from __future__ import annotations

import pytest

from factorzen.research.combination.cv import PurgedWalkForwardCV


def _dates(n: int) -> list[str]:
    """n 个唯一升序交易日串(等长 8 字符,可字典序/整数比较)。"""
    return [f"2025{1 + i // 28:02d}{1 + i % 28:02d}" for i in range(n)]


def test_purge_gap_holds():
    dates = _dates(100)
    cv = PurgedWalkForwardCV(train_days=40, test_days=20, purge_days=5)
    folds = cv.split(dates)
    assert len(folds) == 3  # (40|20)(60|20)(80|20)
    for tr, te in folds:
        # train 末尾与 test 首之间至少隔 purge_days+1(防前向标签重叠泄漏)
        assert dates.index(te[0]) - dates.index(tr[-1]) >= 5 + 1


def test_test_segments_contiguous_non_overlapping():
    dates = _dates(100)
    folds = PurgedWalkForwardCV(40, 20, 5).split(dates)
    flat = [d for _, te in folds for d in te]
    assert flat == dates[40:100]  # test 并集连续覆盖 train_days..end,无重叠


def test_max_train_before_min_test():
    dates = _dates(100)
    for tr, te in PurgedWalkForwardCV(40, 20, 5).split(dates):
        assert max(int(d) for d in tr) < min(int(d) for d in te)


def test_expanding_train_grows_and_nests():
    dates = _dates(100)
    folds = PurgedWalkForwardCV(40, 20, 5, expanding=True).split(dates)
    sizes = [len(tr) for tr, _ in folds]
    assert sizes == sorted(sizes)  # 单调不减
    assert set(folds[0][0]).issubset(set(folds[1][0]))  # 后折 train 含前折


def test_rolling_window_fixed_size():
    dates = _dates(100)
    folds = PurgedWalkForwardCV(40, 20, 5, expanding=False).split(dates)
    sizes = [len(tr) for tr, _ in folds]
    assert max(sizes) - min(sizes) <= 1  # 滚动窗训练集基本定长


def test_embargo_widens_gap():
    dates = _dates(100)
    f0 = PurgedWalkForwardCV(40, 20, 5, embargo_days=0).split(dates)
    f5 = PurgedWalkForwardCV(40, 20, 5, embargo_days=5).split(dates)
    # embargo 使 train 末尾更早,间隔更大
    assert dates.index(f5[0][0][-1]) < dates.index(f0[0][0][-1])


def test_insufficient_dates_raises():
    with pytest.raises(ValueError):
        PurgedWalkForwardCV(40, 20, 5).split(_dates(30))

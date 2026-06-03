"""按阶段计时:把流水线各阶段耗时记录下来并打日志,便于观测运行性能瓶颈。"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from factorzen.core.logger import get_logger

logger = get_logger(__name__)


class StageTimer:
    """累计各命名阶段的耗时。

    用法::

        timer = StageTimer()
        with timer.stage("IC 分析"):
            ...
        timer.timings  # {"IC 分析": 1.23}

    每个阶段结束(含异常退出)都会记录耗时并在 INFO 级别打一条日志。
    """

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = round(time.perf_counter() - start, 3)
            # 同名阶段重复进入则累加
            self.timings[name] = round(self.timings.get(name, 0.0) + elapsed, 3)
            logger.info("[stage] %s 耗时 %.3fs", name, elapsed)

"""性能基准工具：记录流水线各步骤耗时与峰值内存，用于优化瓶颈定位。

提供三个核心组件：

- ``@benchmark_step`` — 装饰器，自动记录函数耗时与峰值内存到 ``BenchmarkReport``
- ``BenchmarkReport`` — 数据类，汇聚多步骤的基准信息
- ``format_benchmark_report()`` — 将报告序列化为 JSON 友好字典

使用示例::

    report = BenchmarkReport()

    @benchmark_step(report, "因子计算")
    def compute_factors():
        ...

    compute_factors()
    print(format_benchmark_report(report))
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from factorzen.core.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class StepTiming:
    """单步骤基准记录。"""

    name: str
    elapsed_seconds: float
    peak_memory_mb: float | None = None


@dataclass
class BenchmarkReport:
    """汇聚多步骤的性能基准报告。

    Attributes
    ----------
    steps : list[StepTiming]
        各步骤的耗时与峰值内存记录（按追加顺序排列）。
    """

    steps: list[StepTiming] = field(default_factory=list)

    def add_step(
        self,
        name: str,
        elapsed_seconds: float,
        peak_memory_mb: float | None = None,
    ) -> None:
        """追加一条步骤记录。"""
        self.steps.append(
            StepTiming(
                name=name,
                elapsed_seconds=round(elapsed_seconds, 6),
                peak_memory_mb=round(peak_memory_mb, 2) if peak_memory_mb is not None else None,
            )
        )

    @property
    def total_elapsed(self) -> float:
        """所有步骤的累计耗时（秒）。"""
        return round(sum(s.elapsed_seconds for s in self.steps), 6)


def _get_peak_memory_mb() -> float | None:
    """获取当前进程的峰值内存（MB）。仅 Windows / Linux 可用。"""
    try:
        import platform

        if platform.system() == "Windows":
            import ctypes
            import ctypes.wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.wintypes.DWORD),
                    ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
            handle = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return pmc.PeakWorkingSetSize / (1024 * 1024)
            return None
        else:
            # Unix/Linux/macOS
            import resource

            ru = resource.getrusage(resource.RUSAGE_SELF)
            # macOS reports bytes, Linux reports KB
            peak_kb = ru.ru_maxrss
            if platform.system() == "Darwin":
                return peak_kb / (1024 * 1024)
            return peak_kb / 1024
    except Exception:
        return None


def benchmark_step(report: BenchmarkReport, name: str) -> Callable[[F], F]:
    """装饰器：记录被装饰函数的耗时与峰值内存到 ``report``。

    Parameters
    ----------
    report : BenchmarkReport
        结果写入的报告对象。
    name : str
        步骤名称。

    Returns
    -------
    Callable
        装饰后的函数，行为不变，仅追加基准记录。
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                peak_mb = _get_peak_memory_mb()
                report.add_step(name, elapsed, peak_mb)
                logger.info(
                    "[benchmark] %s 耗时 %.3fs%s",
                    name,
                    elapsed,
                    f" 峰值内存 {peak_mb:.1f}MB" if peak_mb is not None else "",
                )

        return wrapper  # type: ignore[return-value]

    return decorator


def format_benchmark_report(report: BenchmarkReport) -> dict[str, Any]:
    """将 ``BenchmarkReport`` 序列化为 JSON 友好字典。

    Parameters
    ----------
    report : BenchmarkReport
        待序列化的报告。

    Returns
    -------
    dict[str, Any]
        包含 ``steps`` 列表与 ``total_elapsed`` 的字典。

    Example
    -------
    >>> report = BenchmarkReport()
    >>> report.add_step("step_a", 1.234, 128.5)
    >>> result = format_benchmark_report(report)
    >>> result["total_elapsed"]
    1.234
    """
    return {
        "steps": [
            {
                "name": s.name,
                "elapsed_seconds": s.elapsed_seconds,
                "peak_memory_mb": s.peak_memory_mb,
            }
            for s in report.steps
        ],
        "total_elapsed": report.total_elapsed,
    }

"""reporting/ — 因子报告生成。

提供信号轨报告与交易轨报告（各自独立设计，不复用旧 tear sheet）。

Usage:
    >>> from factorzen.reports.signal_report import generate_signal_report
    >>> from factorzen.reports.trading_report import generate_trading_report
"""

from factorzen.reports.signal_report import generate_signal_report
from factorzen.reports.trading_report import generate_trading_report

__all__ = ["generate_signal_report", "generate_trading_report"]

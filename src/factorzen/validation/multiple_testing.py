"""多重检验记账：记录挖掘过程真实评估的候选数 N。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrialLedger:
    """累加真实评估候选数；该 N 喂给 DSR 并在报告中标注「从 N 个候选选出」。"""

    n_trials: int = 0

    def record(self, k: int = 1) -> None:
        self.n_trials += k

"""全局随机种子控制。"""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> dict[str, int]:
    """设置全局随机种子。返回种子字典供 manifest 记录。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return {"seed": seed, "pythonhashseed": seed}


def get_optuna_sampler(seed: int | None):
    """返回固定种子的 Optuna TPE 采样器。"""
    import optuna

    return optuna.samplers.TPESampler(seed=seed)

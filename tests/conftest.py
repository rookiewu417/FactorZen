"""全局测试隔离。

日志文件 handler 隔离：``setup_logging()`` 默认把 DEBUG 全量日志写
``workspace/runs/logs/factor_research.log``。测试里任何调用 ``main()`` 入口
（manifest 失败注入等）都会触发它——重定向到 tmp，绝不写真实 workspace。
``_initialized`` 是进程级 once 守卫，首个触发者绑定的目录会贯穿整个
session，因此必须在**所有**测试外层兜住，而不是逐个测试打补丁。
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_log_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "factorzen.core.logger.default_log_dir", lambda: tmp_path / "_logs"
    )

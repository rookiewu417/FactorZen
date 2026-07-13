"""Run pytest coverage with an isolated temporary data file.

强制最低覆盖率门槛(`--cov-fail-under`),防止覆盖率随改动悄悄回退。
当前基线约 82%;门槛设 74% 留出波动缓冲(实测显著高于门槛)。提高覆盖率后可上调此值。

2026-07 提速:改用 pytest-cov + pytest-xdist(`-n auto`)——CI 从「Test 跑一遍 +
coverage 再跑一遍」合并为本脚本单遍并行(测试失败即非零退出,coverage 同时产出);
pytest-cov 自动合并 xdist worker 的覆盖数据,source/omit 口径仍读 pyproject.toml
`[tool.coverage.*]`(与旧 `coverage run` 同源,fail-under 百分比可比)。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid

MIN_COVERAGE = 74


def main() -> int:
    coverage_file = os.path.join(
        tempfile.gettempdir(),
        f"factorzen-coverage-{uuid.uuid4().hex}.coverage",
    )
    env = {**os.environ, "COVERAGE_FILE": coverage_file}
    # --cov 不带值 → 用 pyproject [tool.coverage.run] 的 source(src/factorzen)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-n", "auto",
         "--cov", "--cov-report=term",
         f"--cov-fail-under={MIN_COVERAGE}"],
        env=env,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())

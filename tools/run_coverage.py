"""Run pytest coverage with an isolated temporary data file.

强制最低覆盖率门槛(`--fail-under`),防止覆盖率随改动悄悄回退。
当前基线约 76%;门槛设 74% 留出小幅波动缓冲。提高覆盖率后可上调此值。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid

MIN_COVERAGE = 74


def _run(args: list[str], env: dict[str, str]) -> int:
    return subprocess.run([sys.executable, "-m", *args], env=env).returncode


def main() -> int:
    coverage_file = os.path.join(
        tempfile.gettempdir(),
        f"factorzen-coverage-{uuid.uuid4().hex}.coverage",
    )
    env = {**os.environ, "COVERAGE_FILE": coverage_file}

    test_status = _run(["coverage", "run", "-m", "pytest", "tests"], env)
    if test_status != 0:
        return test_status
    return _run(["coverage", "report", f"--fail-under={MIN_COVERAGE}"], env)


if __name__ == "__main__":
    raise SystemExit(main())

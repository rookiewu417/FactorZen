"""Run pytest coverage with a unique temporary data file."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid


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
    return _run(["coverage", "report", "--fail-under=70"], env)


if __name__ == "__main__":
    raise SystemExit(main())

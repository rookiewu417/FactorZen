"""Small terminal progress helpers for CLI pipelines."""

from __future__ import annotations

import sys
from atexit import register


class OverallProgress:
    """Render a coarse overall progress bar for stage-based CLI workflows."""

    def __init__(self, total: int, *, label: str = "Overall") -> None:
        self.total = max(total, 1)
        self.label = label
        self.current = 0
        self._enabled = sys.stderr.isatty()
        self._closed = False

    def start(self) -> "OverallProgress":
        if self._enabled:
            register(self.close)
            self._render("starting")
        return self

    def advance(self, step: str) -> None:
        self.current = min(self.current + 1, self.total)
        if self._enabled:
            self._render(step)

    def close(self) -> None:
        if self._enabled and not self._closed:
            sys.stderr.write("\n")
            sys.stderr.flush()
        self._closed = True

    def _render(self, step: str) -> None:
        width = 28
        filled = round(width * self.current / self.total)
        bar = "#" * filled + "-" * (width - filled)
        percent = round(100 * self.current / self.total)
        sys.stderr.write(
            f"\r{self.label} [{bar}] {self.current}/{self.total} {percent:3d}% {step}"
        )
        sys.stderr.flush()

"""HFT framework — high-frequency / tick-level factors (RESERVED, not yet implemented)."""

from abc import ABC
from typing import Any

import polars as pl


class HFTFactor(ABC):
    """Abstract base class for HFT (tick-level) factors.

    All methods raise NotImplementedError — HFT framework is not yet implemented.
    """

    name: str = ""
    frequency: str = "tick"

    def compute(self, ctx: Any) -> pl.DataFrame:
        """Compute HFT factor values."""
        raise NotImplementedError("HFT framework not yet implemented")

    def validate(self, result: Any) -> dict[str, Any]:
        """Validate HFT factor computation result."""
        raise NotImplementedError("HFT framework not yet implemented")


__all__ = ["HFTFactor"]

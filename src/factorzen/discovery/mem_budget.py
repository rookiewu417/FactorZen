"""Mining memory budget: measure free RAM, estimate peaks, choose isolation actions.

Used by ``fz mine agent`` (and optionally team) so all_a × multi-year windows can
run on ~20 GiB hosts without requiring the user to remember ``--pool-subproc``.

Policy is pure + injectable (``avail_bytes=``) for unit tests. Process isolation
and lifecycle only — evaluation formulas / holdout cuts are unchanged.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

# Safety cushion left for OS / filesystem cache / LLM client / misc.
DEFAULT_SAFETY_MARGIN_BYTES = 2 * (1024**3)  # 2 GiB

# Rough bytes per (row × f64 leaf) for a wide prepped mining frame.
# Empirically all_a long windows land multi-GB; 8 B × leaves is the lower bound
# for value columns alone (keys/Categorical add more, so we apply a fudge).
_FRAME_BYTES_PER_CELL = 8
_FRAME_FUDGE = 1.35  # keys + null masks + polars overhead

# In-process pool build peaks near "prepped frame + compact pool values + sort copy".
# Subprocess isolation returns that peak to the OS on exit; parent only loads cache.
_POOL_BUILD_MULT = 1.6  # relative to estimated value-column bytes during build
_POOL_VALUE_BYTES_PER_CELL = 4  # compact path uses f32

# When free RAM is below this, force serial lift even if formula would allow more.
_LIFT_TIGHT_BYTES = 12 * (1024**3)
_LIFT_MODERATE_BYTES = 18 * (1024**3)

# Row-count heuristic triggers: above this, prefer pool subprocess on ≤24 GiB hosts.
_LARGE_FRAME_ROWS = 2_000_000
_HOST_TIGHT_TOTAL_BYTES = 24 * (1024**3)


@dataclass(frozen=True)
class PeakEstimate:
    n_rows: int
    n_leaf_cols: int
    n_lib_factors: int
    frame_bytes: int
    pool_value_bytes: int
    pool_build_peak_bytes: int
    simultaneous_inproc_bytes: int  # frame held + in-process pool build

    def as_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_leaf_cols": self.n_leaf_cols,
            "n_lib_factors": self.n_lib_factors,
            "frame_bytes": self.frame_bytes,
            "pool_value_bytes": self.pool_value_bytes,
            "pool_build_peak_bytes": self.pool_build_peak_bytes,
            "simultaneous_inproc_bytes": self.simultaneous_inproc_bytes,
        }


@dataclass(frozen=True)
class MemActions:
    use_pool_subproc: bool
    lift_workers: int | None  # None = leave caller's adaptive default alone
    reason: str
    avail_bytes: int
    estimated_peak_bytes: int
    headroom_bytes: int
    peaks: PeakEstimate | None = None

    def as_dict(self) -> dict:
        return {
            "use_pool_subproc": self.use_pool_subproc,
            "lift_workers": self.lift_workers,
            "reason": self.reason,
            "avail_bytes": self.avail_bytes,
            "estimated_peak_bytes": self.estimated_peak_bytes,
            "headroom_bytes": self.headroom_bytes,
            "peaks": None if self.peaks is None else self.peaks.as_dict(),
        }


def available_memory_bytes() -> int:
    """Best-effort free RAM (Linux: MemAvailable via sysconf AVPHYS).

    Falls back to a conservative 8 GiB if probing fails (forces isolation rather
    than assuming infinite RAM).
    """
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError, TypeError):
        pass
    # /proc/meminfo MemAvailable (kB)
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 8 * (1024**3)


def total_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError, TypeError):
        return None


def estimate_mining_peaks(
    *,
    n_rows: int,
    n_leaf_cols: int,
    n_lib_factors: int,
) -> PeakEstimate:
    """Size estimates for the agent/team mining hot path (order-of-magnitude)."""
    n_rows = max(0, int(n_rows))
    n_leaf_cols = max(0, int(n_leaf_cols))
    n_lib_factors = max(0, int(n_lib_factors))
    frame = int(n_rows * n_leaf_cols * _FRAME_BYTES_PER_CELL * _FRAME_FUDGE)
    pool_vals = int(n_rows * n_lib_factors * _POOL_VALUE_BYTES_PER_CELL)
    pool_build = int(pool_vals * _POOL_BUILD_MULT) + int(frame * 0.25)  # sort/copy share
    # Parent holding full prepped + building pool in-process.
    simultaneous = frame + pool_build
    return PeakEstimate(
        n_rows=n_rows,
        n_leaf_cols=n_leaf_cols,
        n_lib_factors=n_lib_factors,
        frame_bytes=frame,
        pool_value_bytes=pool_vals,
        pool_build_peak_bytes=pool_build,
        simultaneous_inproc_bytes=simultaneous,
    )


def estimate_rows_heuristic(
    *,
    start: str | None,
    end: str | None,
    universe: str | None,
    market: str = "ashare",
) -> int:
    """Cheap row estimate without loading the mining frame.

    Trading days ≈ 243/year × calendar span; universe sizes are order-of-magnitude
    (all_a ~5000 listed historically over a decade union is larger — use 5500).
    """
    if market != "ashare":
        # Non-ashare agent path is not the primary 20 GiB target; small stub.
        return 500_000
    uni = (universe or "all_a").strip().lower()
    n_names = {
        "all_a": 5500,
        "csi800": 800,
        "csi500": 500,
        "csi300": 300,
        "csi1000": 1000,
    }.get(uni, 800)
    years = 10.0
    if start and end and len(start) >= 8 and len(end) >= 8:
        try:
            from datetime import datetime

            d0 = datetime.strptime(start[:8], "%Y%m%d")
            d1 = datetime.strptime(end[:8], "%Y%m%d")
            years = max(0.25, (d1 - d0).days / 365.25)
        except ValueError:
            pass
    # + warmup ~2y of bars for AGENT_WARMUP_LOOKBACK=504
    years_with_warm = years + 2.0
    return int(n_names * 243 * years_with_warm)


def plan_mem_actions(
    *,
    avail_bytes: int | None = None,
    n_rows: int = 0,
    n_leaf_cols: int = 64,
    n_lib_factors: int = 0,
    library_orthogonal: bool = True,
    force_pool_subproc: bool | None = None,
    safety_margin_bytes: int = DEFAULT_SAFETY_MARGIN_BYTES,
    total_bytes: int | None = None,
    measure_fn: Callable[[], int] | None = None,
) -> MemActions:
    """Decide pool subprocess isolation and lift concurrency from budget.

    Parameters
    ----------
    avail_bytes
        Free RAM; ``None`` → measure via ``measure_fn`` or ``available_memory_bytes``.
    force_pool_subproc
        ``True`` / ``False`` override auto; ``None`` = auto from estimates.
    library_orthogonal
        When False, pool is unused → never require subprocess for pool.
    """
    if avail_bytes is None:
        avail_bytes = (measure_fn or available_memory_bytes)()
    avail_bytes = max(0, int(avail_bytes))
    if total_bytes is None:
        total_bytes = total_memory_bytes()

    peaks = estimate_mining_peaks(
        n_rows=n_rows, n_leaf_cols=n_leaf_cols, n_lib_factors=n_lib_factors,
    )
    budget = max(0, avail_bytes - int(safety_margin_bytes))
    est_peak = peaks.simultaneous_inproc_bytes if library_orthogonal else peaks.frame_bytes

    reasons: list[str] = []
    use_sub = False

    if not library_orthogonal:
        use_sub = False
        reasons.append("library_orthogonal=off → no pool")
    elif force_pool_subproc is True:
        use_sub = True
        reasons.append("force_pool_subproc=True")
    elif force_pool_subproc is False:
        use_sub = False
        reasons.append("force_pool_subproc=False")
    else:
        # Auto
        if library_orthogonal and peaks.simultaneous_inproc_bytes > budget:
            use_sub = True
            reasons.append(
                f"inproc_peak≈{peaks.simultaneous_inproc_bytes / (1024**3):.1f}G "
                f"> budget≈{budget / (1024**3):.1f}G "
                f"(avail={avail_bytes / (1024**3):.1f}G − margin)"
            )
        elif (
            library_orthogonal
            and n_rows >= _LARGE_FRAME_ROWS
            and (total_bytes is None or total_bytes <= _HOST_TIGHT_TOTAL_BYTES)
            and avail_bytes <= 22 * (1024**3)
        ):
            # 20–24 GiB class host + multi-million-row frame: always isolate pool.
            use_sub = True
            reasons.append(
                f"large_frame rows={n_rows} on tight host "
                f"(avail={avail_bytes / (1024**3):.1f}G)"
            )
        else:
            reasons.append(
                f"inproc_peak≈{peaks.simultaneous_inproc_bytes / (1024**3):.1f}G "
                f"fits budget≈{budget / (1024**3):.1f}G"
            )

    # Lift concurrency cap under tight free RAM (agent rarely runs lift; team reuses).
    lift_w: int | None
    if avail_bytes < _LIFT_TIGHT_BYTES:
        lift_w = 1
        reasons.append("lift_workers→1 (avail<12G)")
    elif avail_bytes < _LIFT_MODERATE_BYTES:
        lift_w = 2
        reasons.append("lift_workers→2 (avail<18G)")
    else:
        lift_w = None  # leave adaptive default

    return MemActions(
        use_pool_subproc=use_sub,
        lift_workers=lift_w,
        reason="; ".join(reasons),
        avail_bytes=avail_bytes,
        estimated_peak_bytes=est_peak,
        headroom_bytes=budget - est_peak if not use_sub else budget,
        peaks=peaks,
    )


def format_mem_actions(actions: MemActions) -> str:
    """One-line human summary for CLI banners."""
    a = actions
    peak_g = a.estimated_peak_bytes / (1024**3)
    avail_g = a.avail_bytes / (1024**3)
    return (
        f"pool_subproc={a.use_pool_subproc} lift_workers={a.lift_workers} "
        f"avail≈{avail_g:.1f}G est_peak≈{peak_g:.1f}G | {a.reason}"
    )

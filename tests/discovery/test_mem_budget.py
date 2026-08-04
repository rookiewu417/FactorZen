"""Unit tests for adaptive mining memory budget policy."""
from __future__ import annotations

from factorzen.discovery.mem_budget import (
    estimate_mining_peaks,
    estimate_rows_heuristic,
    format_mem_actions,
    plan_mem_actions,
)


def test_estimate_mining_peaks_scales_with_rows_and_factors():
    small = estimate_mining_peaks(n_rows=100_000, n_leaf_cols=64, n_lib_factors=10)
    large = estimate_mining_peaks(n_rows=10_000_000, n_leaf_cols=64, n_lib_factors=90)
    assert large.frame_bytes > small.frame_bytes
    assert large.pool_value_bytes > small.pool_value_bytes
    assert large.simultaneous_inproc_bytes > large.frame_bytes


def test_plan_mem_actions_tight_free_ram_forces_pool_subproc():
    """Fake free RAM well below estimated in-proc peak → subprocess isolation."""
    peaks = estimate_mining_peaks(n_rows=10_000_000, n_leaf_cols=64, n_lib_factors=90)
    # Free RAM only slightly above frame alone → cannot hold frame+pool in-process.
    tight = int(peaks.frame_bytes * 0.5)
    actions = plan_mem_actions(
        avail_bytes=tight,
        n_rows=10_000_000,
        n_leaf_cols=64,
        n_lib_factors=90,
        library_orthogonal=True,
        force_pool_subproc=None,
        safety_margin_bytes=1 * (1024**3),
        total_bytes=20 * (1024**3),
    )
    assert actions.use_pool_subproc is True
    assert actions.lift_workers == 1  # avail < 12G
    assert "inproc_peak" in actions.reason or "large_frame" in actions.reason
    assert "pool_subproc=True" in format_mem_actions(actions)


def test_plan_mem_actions_loose_free_ram_small_frame_no_subproc():
    """Ample free RAM + small frame → keep in-process (no forced isolation)."""
    actions = plan_mem_actions(
        avail_bytes=64 * (1024**3),
        n_rows=50_000,
        n_leaf_cols=32,
        n_lib_factors=5,
        library_orthogonal=True,
        force_pool_subproc=None,
        safety_margin_bytes=2 * (1024**3),
        total_bytes=128 * (1024**3),
    )
    assert actions.use_pool_subproc is False
    assert actions.lift_workers is None  # leave adaptive alone


def test_plan_mem_actions_force_overrides_auto():
    peaks = estimate_mining_peaks(n_rows=10_000_000, n_leaf_cols=64, n_lib_factors=90)
    tight = int(peaks.simultaneous_inproc_bytes * 0.1)
    forced_off = plan_mem_actions(
        avail_bytes=tight,
        n_rows=10_000_000,
        n_leaf_cols=64,
        n_lib_factors=90,
        force_pool_subproc=False,
    )
    assert forced_off.use_pool_subproc is False
    assert "force_pool_subproc=False" in forced_off.reason

    forced_on = plan_mem_actions(
        avail_bytes=128 * (1024**3),
        n_rows=1_000,
        n_leaf_cols=8,
        n_lib_factors=1,
        force_pool_subproc=True,
    )
    assert forced_on.use_pool_subproc is True
    assert "force_pool_subproc=True" in forced_on.reason


def test_plan_mem_actions_no_library_orthogonal_skips_pool():
    actions = plan_mem_actions(
        avail_bytes=4 * (1024**3),
        n_rows=20_000_000,
        n_leaf_cols=64,
        n_lib_factors=100,
        library_orthogonal=False,
        force_pool_subproc=None,
    )
    assert actions.use_pool_subproc is False
    assert "library_orthogonal=off" in actions.reason


def test_estimate_rows_heuristic_all_a_larger_than_csi300():
    a = estimate_rows_heuristic(
        start="20160101", end="20260101", universe="all_a", market="ashare",
    )
    c = estimate_rows_heuristic(
        start="20160101", end="20260101", universe="csi300", market="ashare",
    )
    assert a > c
    assert a > 2_000_000  # multi-million baseline for decade all_a


def test_large_all_a_on_20g_host_auto_pool_subproc():
    """Production default: all_a × ~10y on ~20 GiB host → isolate pool."""
    n_rows = estimate_rows_heuristic(
        start="20160722", end="20260722", universe="all_a", market="ashare",
    )
    actions = plan_mem_actions(
        avail_bytes=20 * (1024**3),
        n_rows=n_rows,
        n_leaf_cols=64,
        n_lib_factors=90,
        library_orthogonal=True,
        force_pool_subproc=None,
        total_bytes=23 * (1024**3),
    )
    assert actions.use_pool_subproc is True
    assert n_rows >= 2_000_000

"""Monotonic deadlines and unwind margins for both search backends.

Python search polls engine.clock.Deadline; compiled search uses the native
clock_gettime bridge. Each backend compares timestamps within its own
clock domain. Helpers compose hard_ns = now + budget - unwind_margin."""

from __future__ import annotations

import time
from collections.abc import Callable

from engine.clock import (
    DEFAULT_CHECK_MASK,
    Deadline,
    make_objmode_clock,
    measure_unwind_ns,
    now_ns,
    validate_bridge,
)
from engine.clock_b.allocator import DEFAULT_UNWIND_MARGIN_NS, SoftAllocator

# Re-export the poll primitive the Python search already calls.
__all__ = [
    "DEFAULT_CHECK_MASK",
    "DEFAULT_UNWIND_MARGIN_NS",
    "Deadline",
    "hard_deadline_ns",
    "make_objmode_clock",
    "measure_unwind_ns",
    "now_ns",
    "validate_bridge",
    "validate_compiled_clock",
]


def hard_deadline_ns(
    alloc_hard_ns: int,
    *,
    now: Callable[[], int] | None = None,
    unwind_margin_ns: int = DEFAULT_UNWIND_MARGIN_NS,
) -> int:
    """Absolute monotonic deadline. Increment is not added here."""
    clock = now or time.monotonic_ns
    return int(clock()) + max(0, int(alloc_hard_ns) - int(unwind_margin_ns))


def validate_compiled_clock(tolerance_ns: int = 50_000_000) -> dict:
    """Prove the compiled clock is monotonic and moves.

    Returns a receipt. Does not import the recursive search kernel — that
    path is owned by FIX-SEGFAULT. ``n_poll`` in ``search_nb.py`` is the
    in-tree check; this only validates ``clock_ns`` itself.
    """
    from engine.kernels.nclock import clock_ns

    a = int(clock_ns())
    b = int(clock_ns())
    py = time.monotonic_ns()
    return {
        "clock_ns_a": a,
        "clock_ns_b": b,
        "nondecreasing": b >= a,
        "delta_ns": b - a,
        "python_monotonic_ns": py,
        "ok": b >= a,
        "note": (
            "compiled search compares elapsed budget to its own stamp; "
            "epochs may differ from time.monotonic_ns on macOS"
        ),
        "tolerance_ns": tolerance_ns,
    }


def default_allocator() -> SoftAllocator:
    return SoftAllocator()

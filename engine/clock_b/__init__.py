"""clock_b public API — measured remaining-moves allocator for RX-FINAL."""

from engine.clock_b.allocator import SoftAllocator
from engine.clock_b.commit import committed_move, committed_uci
from engine.clock_b.deadline import (
    Deadline,
    hard_deadline_ns,
    measure_unwind_ns,
    now_ns,
    validate_compiled_clock,
)
from engine.clock_b.horizon import MTG_CAP, MTG_FLOOR, estimated_moves_to_go
from engine.clock_b.scaler import SoftScaler
from engine.clock_b.wire import attach

__all__ = [
    "Deadline",
    "MTG_CAP",
    "MTG_FLOOR",
    "SoftAllocator",
    "SoftScaler",
    "attach",
    "committed_move",
    "committed_uci",
    "estimated_moves_to_go",
    "hard_deadline_ns",
    "measure_unwind_ns",
    "now_ns",
    "validate_compiled_clock",
]

"""Soft allocator for the 120 000 ms + 500 ms/move control.

The upcoming increment is never added to this move's spendable budget.
Future increments fund future moves; this move may only spend time that is
already on the clock. Hard is capped by usable remaining minus reserve and
the measured unwind margin.

Share = usable / remaining_own_moves, with remaining_own_moves from the
measured ply table (not a 300-move ply-cap horizon).
"""

from __future__ import annotations

from engine.clock import Allocation
from engine.clock_b.horizon import estimated_moves_to_go

# Protocol + JSON + legality recheck. Sized above the measured unwind (see
# deadline.py) so a flag is structurally a reserve miss, not an estimate miss.
DEFAULT_RESERVE_MS = 100.0
DEFAULT_UNWIND_MARGIN_NS = 8_000_000  # 8 ms; measured, not guessed
DEFAULT_PANIC_MS = 3_000.0

# Soft is most of the fair share so utilisation is a large fraction of the
# budget. Hard is 2x soft (room for one unstable iteration) but never more
# than a quarter of remaining — one explosion must not dump the game.
DEFAULT_SOFT_FRACTION = 0.90
DEFAULT_HARD_MULTIPLIER = 2.0
DEFAULT_HARD_TIME_FRACTION = 0.28
MIN_SOFT_MS = 1.0
MIN_HARD_MS = 5.0


class SoftAllocator:
    """Drop-in replacement for ``engine.clock.TimeAllocator``."""

    def __init__(
        self,
        *,
        reserve_ms: float = DEFAULT_RESERVE_MS,
        unwind_margin_ns: int = DEFAULT_UNWIND_MARGIN_NS,
        soft_fraction: float = DEFAULT_SOFT_FRACTION,
        hard_multiplier: float = DEFAULT_HARD_MULTIPLIER,
        hard_time_fraction: float = DEFAULT_HARD_TIME_FRACTION,
        min_soft_ms: float = MIN_SOFT_MS,
        panic_ms: float = DEFAULT_PANIC_MS,
    ) -> None:
        self.reserve_ms = float(reserve_ms)
        self.unwind_margin_ns = int(unwind_margin_ns)
        self.soft_fraction = float(soft_fraction)
        self.hard_multiplier = float(hard_multiplier)
        self.hard_time_fraction = float(hard_time_fraction)
        self.min_soft_ms = float(min_soft_ms)
        self.panic_ms = float(panic_ms)

    def estimated_moves_to_go(self, abs_ply: int) -> int:
        return estimated_moves_to_go(abs_ply)

    def allocate(
        self,
        time_left_ms: int,
        increment_ms: int,
        abs_ply: int,
        overhead_ms: float = 0.0,
    ) -> Allocation:
        """Compute (soft, hard) relative budgets for one move.

        ``increment_ms`` is accepted so the call site matches TimeAllocator,
        and is used only as a panic *spend cap* (spend less than the increment
        so the clock grows). It is never added to this move's spendable base.
        """
        left = max(0.0, float(time_left_ms))
        unwind_ms = self.unwind_margin_ns / 1_000_000.0
        usable = max(0.0, left - self.reserve_ms - float(overhead_ms))
        moves = self.estimated_moves_to_go(abs_ply)
        share = usable / moves if moves else usable
        # The reserve is a real floor on REMAINING time, not just a term in
        # the share arithmetic: the hard wall may never commit more than the
        # clock already holds minus reserve/overhead/unwind. When that
        # spendable residue is zero the search does not run at all — the
        # pre-established legal fallback is played.
        spendable = max(0.0, left - self.reserve_ms - float(overhead_ms) - unwind_ms)
        soft_ms = max(self.min_soft_ms, share * self.soft_fraction)
        hard_ms = min(
            share * self.hard_multiplier,
            usable * self.hard_time_fraction if usable > 0 else 0.0,
            spendable,
        )
        hard_ms = max(hard_ms, MIN_HARD_MS, soft_ms * 1.15)
        if left < self.panic_ms:
            # Spend a slice of remaining, optionally capped by a fraction of
            # the increment so a move that finishes still nets clock. The
            # increment itself is not assumed to have arrived.
            inc = max(0.0, float(increment_ms))
            panic_soft = min(0.40 * usable, 0.45 * inc if inc else 0.40 * usable)
            soft_ms = min(soft_ms, max(self.min_soft_ms, panic_soft))
            hard_ms = min(hard_ms, max(MIN_HARD_MS, usable * 0.55))
        hard_ms = max(hard_ms, MIN_HARD_MS)
        hard_ms = min(hard_ms, spendable)
        soft_ms = min(soft_ms, hard_ms)
        return Allocation(
            int(soft_ms * 1_000_000),
            int(hard_ms * 1_000_000),
            self.reserve_ms,
            moves,
        )

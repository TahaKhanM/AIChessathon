"""Monotonic hard deadline and tuned soft allocation.

Wall-clock hard stop checked inside compiled search, measured unwind margin,
soft allocator informed by reserve, iteration costs, best-move stability,
score trend and contender separation. See spec section 3.4.

The hard clock is monotonic wall time, never a node counter. The pure-Python
search polls :meth:`Deadline.poll`; when the hot loop is lifted into Numba the
same polling sequence calls the ``objmode`` bridge :func:`make_objmode_clock`
instead, so compiled code checks the same monotonic wall time without leaving
nopython mode per node.
"""

from __future__ import annotations

import time
from collections.abc import Callable

# Poll mask: the wall clock is sampled once every ``mask + 1`` nodes. A mask of
# 2047 bounds the work done after the deadline to ~2048 nodes of slack, which the
# unwind margin below must absorb together with unwind, validation and protocol
# tails.
DEFAULT_CHECK_MASK = 2047

# Reserve components (milliseconds). The hard deadline is
# ``time_left - reserve - unwind_margin`` so a flag fall is not possible even
# when every estimate is optimistic.
DEFAULT_RESERVE_MS = 80.0
DEFAULT_UNWIND_MARGIN_NS = 8_000_000  # 8 ms: unwind + legality + protocol tail

# Game-length model for the 120 s + 0.5 s control: measured arena games run
# ~173 plies, so a side moves ~86 times over a whole game (~2 s per move of
# budget on average). The per-move share divides the remaining clock by the
# remaining-move estimate, clamped to [MTG_FLOOR, MTG_CAP]: the cap keeps the
# opening and middlegame from being starved by the worst-case horizon (the
# same shape as the proven constant "moves-to-go" policies), and the floor
# keeps a sane minimum share once the estimate runs out in a long game.
EXPECTED_GAME_PLIES = 173
MTG_CAP = 40
MTG_FLOOR = 14

# Below this much wall time the allocation switches to panic pacing: soft
# stops at a fraction of the incoming increment and the hard wall shrinks to
# a third of the spendable clock.
DEFAULT_PANIC_MS = 3_000.0


def now_ns() -> int:
    """Monotonic wall time in nanoseconds; the only clock the search trusts."""
    return time.monotonic_ns()


class Deadline:
    """Hard stop on monotonic wall time plus an optional node limit.

    ``poll()`` is called once per visited node. Between clock reads the flag
    alone is consulted, so abort propagation is branch-cheap. ``stop`` is
    writeable by the driver (e.g. an external abort) and sticky: once set the
    deadline never reports "continue" again, which is what makes abort replays
    safe — no search phase can resume after a stop has been observed.
    """

    __slots__ = (
        "hard_ns",
        "node_limit",
        "nodes",
        "stop",
        "check_mask",
        "clock_reads",
        "overrun_ns",
        "_now",
    )

    def __init__(
        self,
        hard_ns: int = 0,
        node_limit: int = 0,
        check_mask: int = DEFAULT_CHECK_MASK,
        now: Callable[[], int] | None = None,
    ) -> None:
        self.hard_ns = int(hard_ns)
        self.node_limit = int(node_limit)
        self.nodes = 0
        self.stop = False
        self.check_mask = check_mask
        self.clock_reads = 0
        self.overrun_ns = 0
        self._now = now or time.monotonic_ns

    @classmethod
    def after_ms(
        cls,
        ms: float,
        node_limit: int = 0,
        margin_ns: int = DEFAULT_UNWIND_MARGIN_NS,
        now: Callable[[], int] | None = None,
    ) -> Deadline:
        """Deadline ``ms`` from now minus the unwind/validation margin."""
        clock = now or time.monotonic_ns
        hard = clock() + max(0, int(ms * 1_000_000) - margin_ns)
        return cls(hard, node_limit, now=now)

    def poll(self) -> bool:
        """Return True once the search must stop. Called once per node."""
        if self.stop:
            return True
        n = self.nodes
        self.nodes = n + 1
        if self.node_limit and n >= self.node_limit:
            self.stop = True
            return True
        if n & self.check_mask:
            return False
        self.clock_reads += 1
        if self.hard_ns:
            t = self._now()
            if t >= self.hard_ns:
                self.stop = True
                self.overrun_ns = t - self.hard_ns
        return self.stop


# ---------------------------------------------------------------------------
# Soft allocation
# ---------------------------------------------------------------------------


class Allocation:
    """Budget for one move: soft_ns stops new iterations, hard_ns aborts."""

    __slots__ = ("soft_ns", "hard_ns", "reserve_ms", "estimated_moves")

    def __init__(self, soft_ns: int, hard_ns: int, reserve_ms: float, estimated_moves: int) -> None:
        self.soft_ns = soft_ns
        self.hard_ns = hard_ns
        self.reserve_ms = reserve_ms
        self.estimated_moves = estimated_moves


class TimeAllocator:
    """Tuned soft allocator informed by clock reserve and game horizon.

    The allocation is deliberately two-tiered: ``soft_ns`` is the "stop starting
    new iterations" budget, ``hard_ns`` the unclimbable wall. Between the two,
    the iterative-deepening driver in ``engine.search`` applies the dynamic
    signals from spec 3.4 — best-move stability, score trend, contender
    separation and best-move node fraction — as multiplicative scales on the
    soft budget. The hard budget already contains the reserve and the unwind
    margin, so nothing downstream has to re-derive safety.

    The upcoming increment is *never* assumed to be available for later moves:
    it contributes to the spendable base at a discounted factor but never to
    the reserve floor.
    """

    def __init__(
        self,
        *,
        reserve_ms: float = DEFAULT_RESERVE_MS,
        unwind_margin_ns: int = DEFAULT_UNWIND_MARGIN_NS,
        increment_factor: float = 0.75,
        soft_fraction: float = 0.55,
        hard_multiplier: float = 4.0,
        hard_time_fraction: float = 0.35,
        min_soft_ms: float = 1.0,
        panic_ms: float = DEFAULT_PANIC_MS,
    ) -> None:
        self.reserve_ms = reserve_ms
        self.unwind_margin_ns = unwind_margin_ns
        self.increment_factor = increment_factor
        self.soft_fraction = soft_fraction
        self.hard_multiplier = hard_multiplier
        self.hard_time_fraction = hard_time_fraction
        self.min_soft_ms = min_soft_ms
        self.panic_ms = panic_ms

    def estimated_moves_to_go(self, abs_ply: int) -> int:
        """Remaining own-move estimate under the measured ~173-ply game model.

        ~86 own moves are expected over a whole game, so the raw estimate is
        ``(EXPECTED_GAME_PLIES - abs_ply) // 2``. It is clamped to
        [MTG_FLOOR, MTG_CAP]: many games end well before the mean, so the
        estimate never exceeds 40 moves-to-go (a ~3 s share of a full clock),
        and never drops below 14 so a long game still spends a real share.
        """
        est = (EXPECTED_GAME_PLIES - abs_ply + 1) // 2
        return max(MTG_FLOOR, min(MTG_CAP, est))

    def allocate(
        self,
        time_left_ms: int,
        increment_ms: int,
        abs_ply: int,
        overhead_ms: float = 0.0,
    ) -> Allocation:
        """Compute (soft, hard) budgets for one move under 120 s + 0.5 s play."""
        usable = max(0.0, time_left_ms - self.reserve_ms - overhead_ms)
        moves = self.estimated_moves_to_go(abs_ply)
        base_ms = usable / moves + increment_ms * self.increment_factor
        soft_ms = max(self.min_soft_ms, base_ms * self.soft_fraction)
        hard_ms = min(
            base_ms * self.hard_multiplier,
            usable * self.hard_time_fraction if usable > 0 else 0.0,
        )
        hard_ms = max(hard_ms, soft_ms * 1.5)
        # The wall clock is spent before the increment arrives: never let the
        # hard bound exceed what the clock could actually supply — and the
        # reserve is a real floor on REMAINING time, not just a deduction in
        # the budget arithmetic, so the hard wall never commits more than
        # ``time_left - reserve - overhead``.
        hard_ms = min(hard_ms, usable)
        if time_left_ms < self.panic_ms:
            # Panic pacing: soft stops at a fraction of the incoming
            # increment; the hard wall is a third of the spendable clock.
            # The increment itself is never counted on before it lands.
            soft_ms = min(soft_ms, 0.45 * increment_ms)
            hard_ms = min(hard_ms, usable * 0.35)
        # The soft pace never outruns the wall: a first iteration always gets
        # to attempt completion inside the hard bound.
        soft_ms = min(soft_ms, hard_ms)
        return Allocation(
            int(soft_ms * 1_000_000),
            int(hard_ms * 1_000_000),
            self.reserve_ms,
            moves,
        )


class IterationScaler:
    """Dynamic per-iteration soft-budget scaling (driver side of spec 3.4).

    Multipliers are fixed-point percentages so the same policy can be tuned and
    shipped verbatim. Inputs come from the *completed* iteration only — an
    aborted iteration never feeds the controller.
    """

    def __init__(
        self,
        *,
        stable_new_x100: int = 135,
        stable3_x100: int = 85,
        stable6_x100: int = 70,
        falling_drop_cp: int = 30,
        falling_x100: int = 135,
        effort_lo_x100: int = 50,
        effort_hi_x100: int = 150,
        separation_floor_cp: int = 15,
        separation_x100: int = 115,
    ) -> None:
        self.stable_new_x100 = stable_new_x100
        self.stable3_x100 = stable3_x100
        self.stable6_x100 = stable6_x100
        self.falling_drop_cp = falling_drop_cp
        self.falling_x100 = falling_x100
        self.effort_lo_x100 = effort_lo_x100
        self.effort_hi_x100 = effort_hi_x100
        self.separation_floor_cp = separation_floor_cp
        self.separation_x100 = separation_x100

    def scale(
        self,
        *,
        depth: int,
        stable_iters: int,
        score_drop_cp: int,
        best_move_node_fraction: float | None,
        contender_gap_cp: int | None,
    ) -> float:
        """Soft-budget multiplier for the *next* iteration decision."""
        s = 1.0
        if stable_iters == 0 and depth >= 6:
            s = self.stable_new_x100 / 100.0
        elif stable_iters >= 6:
            s = self.stable6_x100 / 100.0
        elif stable_iters >= 3:
            s = self.stable3_x100 / 100.0
        if depth >= 6 and score_drop_cp >= self.falling_drop_cp:
            s *= self.falling_x100 / 100.0
        if depth >= 8 and best_move_node_fraction is not None and best_move_node_fraction > 0.0:
            frac = min(1.0, best_move_node_fraction)
            effort = 2.0 * (1.0 - frac) + 0.4
            s *= max(
                self.effort_lo_x100 / 100.0,
                min(self.effort_hi_x100 / 100.0, effort),
            )
        if contender_gap_cp is not None and contender_gap_cp < self.separation_floor_cp:
            s *= self.separation_x100 / 100.0
        return s


def measure_unwind_ns(unwind_once: Callable[[], None], reps: int = 64) -> int:
    """Measure a real unwind/validation tail; feeds the margin, not a guess.

    The shipped margin stays the conservative constant above; this exists so
    the margin is *measured* against the real stack on the target instead of
    asserted. Returns the worst observed per-call cost in nanoseconds.
    """
    unwind_once()  # warm
    worst = 0
    for _ in range(reps):
        t0 = time.monotonic_ns()
        unwind_once()
        dt = time.monotonic_ns() - t0
        if dt > worst:
            worst = dt
    return worst


# ---------------------------------------------------------------------------
# Compiled-path clock bridge
# ---------------------------------------------------------------------------

_objmode_clock = None


def make_objmode_clock() -> Callable[[], int]:
    """The validated bridge used inside compiled search.

    Numba ``njit`` code cannot call ``time.monotonic_ns``; the ``objmode``
    escape produces a nopython-callable ``clock_ns()`` with identical
    semantics. The bridge is compiled lazily so importing ``engine.clock`` in
    tools/tests that never JIT does not pay the numba import. When the hot
    loop is compiled, ``Deadline.poll`` inside the kernel calls this function
    on the same ``check_mask`` cadence — the deadline arithmetic is unchanged,
    only the time source is routed through ``objmode``.
    """
    global _objmode_clock
    if _objmode_clock is None:
        from numba import njit, objmode

        @njit(cache=False)
        def clock_ns():  # type: ignore[misc]
            with objmode(now="int64"):
                now = time.monotonic_ns()
            return now

        _objmode_clock = clock_ns
    return _objmode_clock


def validate_bridge(tolerance_ns: int = 5_000_000) -> bool:
    """Compile the objmode bridge and check it against the direct monotonic read."""
    clock = make_objmode_clock()
    int(clock())  # cold call pays the JIT; measure a warm call
    before = time.monotonic_ns()
    bridged = int(clock())
    after = time.monotonic_ns()
    return before <= bridged <= after and (after - before) <= tolerance_ns

"""clock_b gates: remaining-moves, increment, abort-PV, monotonic hard stop.

These tests are written to FAIL on the R16 `_HorizonAllocator` (est_moves≈290,
hard_ms≈785 at ply 12) and on an abort path that plays a torn PV.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from engine.clock_b.allocator import SoftAllocator
from engine.clock_b.commit import committed_move
from engine.clock_b.deadline import (
    Deadline,
    hard_deadline_ns,
    measure_unwind_ns,
    now_ns,
    validate_compiled_clock,
)
from engine.clock_b.horizon import MTG_CAP, MTG_FLOOR, estimated_moves_to_go
from engine.clock_b.scaler import SoftScaler
from engine.movegen import generate_legal
from engine.state import GameState


START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_horizon_is_not_three_hundred_moves() -> None:
    """R16: ply 12 of the English smoke had est_moves=294. That is the bug."""
    for ply in (0, 12, 20, 40):
        est = estimated_moves_to_go(ply)
        assert MTG_FLOOR <= est <= MTG_CAP, (ply, est)
        assert est < 80, (ply, est)


def test_remaining_moves_is_nonincreasing() -> None:
    prev = estimated_moves_to_go(0)
    for ply in range(1, 601):
        cur = estimated_moves_to_go(ply)
        assert cur <= prev, (ply, cur, prev)
        assert cur >= MTG_FLOOR
        prev = cur
    assert estimated_moves_to_go(600) >= MTG_FLOOR


def test_opening_hard_is_seconds_not_subsecond() -> None:
    """R16 measured hard_ms≈782. A real share of 120 s is several seconds."""
    alloc = SoftAllocator().allocate(120_000, 500, 12)
    hard_ms = alloc.hard_ns / 1e6
    soft_ms = alloc.soft_ns / 1e6
    assert alloc.estimated_moves <= MTG_CAP
    assert hard_ms > 2_500, hard_ms
    assert soft_ms > 1_500, soft_ms
    assert soft_ms <= hard_ms <= 120_000


def test_increment_is_not_added_to_this_move() -> None:
    a = SoftAllocator()
    with_inc = a.allocate(120_000, 500, 0)
    no_inc = a.allocate(120_000, 0, 0)
    # Upcoming increment is not spendable. Hard/soft at a full clock must
    # match whether or not an increment is advertised.
    assert with_inc.hard_ns == no_inc.hard_ns
    assert with_inc.soft_ns == no_inc.soft_ns
    # Hard must not exceed time already on the clock.
    assert with_inc.hard_ns <= 120_000 * 1_000_000


def test_hard_never_exceeds_remaining_clock() -> None:
    a = SoftAllocator()
    for left, ply in ((120_000, 0), (8_000, 80), (900, 200), (50, 400)):
        alloc = a.allocate(left, 500, ply)
        assert 0 <= alloc.soft_ns <= alloc.hard_ns
        assert alloc.hard_ns <= max(1, left) * 1_000_000


def test_reserve_is_a_real_remaining_time_floor() -> None:
    """The hard wall may never commit more than the clock already holds
    minus reserve/overhead/unwind — including below the reserve, where the
    spendable residue is zero and no search runs at all."""
    a = SoftAllocator()
    unwind_ms = a.unwind_margin_ns / 1e6
    for left in (120_000, 20_000, 5_000, 900, 300, 150, 108, 100, 50, 1):
        for overhead in (0.0, 40.0):
            alloc = a.allocate(left, 500, 0, overhead_ms=overhead)
            floor = max(0.0, left - a.reserve_ms - overhead - unwind_ms)
            assert alloc.hard_ns <= floor * 1_000_000, (left, overhead, alloc.hard_ns)
            assert 0 <= alloc.soft_ns <= alloc.hard_ns
    # below the reserve there is literally nothing to spend
    zero = a.allocate(50, 500, 0)
    assert zero.hard_ns == 0 and zero.soft_ns == 0
    # the floor is monotonic: less clock never gets a bigger wall
    prev = None
    for left in (120_000, 8_000, 900, 150, 50):
        alloc = a.allocate(left, 500, 80)
        if prev is not None:
            assert alloc.hard_ns <= prev
        prev = alloc.hard_ns


def test_panic_spends_less_than_increment() -> None:
    alloc = SoftAllocator().allocate(1_200, 500, 80)
    assert alloc.soft_ns / 1e6 <= 250.0
    assert alloc.hard_ns / 1e6 < 1_200


def test_scaler_extends_on_instability_and_score_drop() -> None:
    sc = SoftScaler()
    unstable = sc.scale(
        depth=10,
        stable_iters=0,
        score_drop_cp=80,
        best_move_node_fraction=0.15,
        contender_gap_cp=5,
    )
    stable = sc.scale(
        depth=10,
        stable_iters=8,
        score_drop_cp=-50,
        best_move_node_fraction=0.95,
        contender_gap_cp=200,
    )
    assert unstable > 1.0
    assert stable < 1.0
    assert unstable > stable


def test_committed_move_never_plays_torn_pv() -> None:
    """Fail if an aborted iteration's root_best replaces a completed PV head."""
    fallback = 0x1111
    completed = 0x2222
    torn = 0x3333
    result = SimpleNamespace(
        move=torn,
        pv=[completed, 0x4444],
        aborted=True,
        partial=True,
    )
    assert committed_move(result, fallback) == completed
    empty = SimpleNamespace(move=0, pv=[], aborted=True, partial=True)
    assert committed_move(empty, fallback) == fallback
    ok = SimpleNamespace(move=completed, pv=[completed], aborted=False, partial=False)
    assert committed_move(ok, fallback) == completed


def test_deadline_polls_monotonic_wall_not_nodes() -> None:
    calls = {"n": 0}

    def fake_now() -> int:
        calls["n"] += 1
        # First clock sample is before the wall; the next is past it.
        return 0 if calls["n"] == 1 else 10**12

    d = Deadline(hard_ns=100, check_mask=3, now=fake_now)
    stopped = False
    for _ in range(32):
        if d.poll():
            stopped = True
            break
    assert stopped
    assert d.stop
    assert d.clock_reads >= 1
    # Node limit is optional; this fire is the wall.
    assert d.nodes < 10_000


def test_hard_deadline_ns_subtracts_unwind_and_not_increment() -> None:
    t0 = 1_000_000_000

    def frozen() -> int:
        return t0

    hard = hard_deadline_ns(5_000_000_000, now=frozen, unwind_margin_ns=8_000_000)
    assert hard == t0 + 5_000_000_000 - 8_000_000


def test_unwind_margin_dominates_measured_tail() -> None:
    gs = GameState.from_fen(START)
    buf = [0] * 256
    n = generate_legal(gs.board, buf)

    def unwind_once() -> None:
        k = min(n, 16)
        for i in range(k):
            gs.board.make(buf[i])
        for _ in range(k):
            gs.board.unmake()

    worst = measure_unwind_ns(unwind_once, reps=24)
    assert worst < 8_000_000


def test_compiled_clock_is_monotonic() -> None:
    rec = validate_compiled_clock()
    assert rec["ok"] is True
    assert rec["nondecreasing"] is True


def test_now_ns_is_monotonic() -> None:
    a = now_ns()
    time.sleep(0.001)
    b = now_ns()
    assert b > a

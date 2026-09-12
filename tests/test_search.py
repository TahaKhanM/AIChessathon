"""Search tests and the W03 gates: abort replay, terminal adjudication,
TT sweep, fallback. Spec §§3.2-3.4, gate list per work order.

Gate tests are deliberately sized for evidence: ABORT_TRIALS >= 10,000
randomized abort points and TERMINAL_CASES >= 50,000 referee comparisons.
"""

from __future__ import annotations

import random
import time

import chess

from engine.board import MAX_MOVES, decode_move, move_to_uci
from engine.clock import (
    DEFAULT_UNWIND_MARGIN_NS,
    IterationScaler,
    TimeAllocator,
    measure_unwind_ns,
    validate_bridge,
)
from engine.movegen import generate_legal
from engine.search import (
    INF,
    MATE,
    MATE_IN_MAX,
    Searcher,
    simple_eval,
)
from engine.state import PLY_CAP, GameState, referee_terminal
from engine.tt import TranspositionTable
from tests.test_search_abort import _CommitTap

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWI = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
ENDGAME = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
TACTIC = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"

ABORT_TRIALS = 10_000
TERMINAL_POSITIONS = 50_000


def _searcher(mib: int = 4, **kw) -> Searcher:
    return Searcher(tt=TranspositionTable(mib=mib), **kw)


def _far() -> int:
    return time.monotonic_ns() + 600_000_000_000


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


def test_search_returns_legal_move() -> None:
    s = _searcher()
    for fen in (START, KIWI, ENDGAME, TACTIC):
        gs = GameState.from_fen(fen)
        res = s.search(gs, hard_ns=_far(), max_depth=4)
        buf = [0] * MAX_MOVES
        n = generate_legal(gs.board, buf)
        assert res.move != 0
        assert any(buf[i] == res.move for i in range(n)), (fen, res.uci)
        assert res.depth >= 1


def test_mate_in_one_found() -> None:
    s = _searcher()
    gs = GameState.from_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    res = s.search(gs, hard_ns=_far(), max_depth=4)
    assert res.uci == "a1a8"
    assert res.score > MATE_IN_MAX  # mate score band


def test_mate_in_two_found() -> None:
    """K+R mate-in-2 (verified against python-chess): every black reply to
    Kc7/Kb6 is mate next move. The score must be the mate-in-2 band
    (MATE-3), not the mate-in-3 Rh8 line — under default parameters a
    checking quiet was once LMP-pruned at the mate-critical node and the
    poisoned bound was re-served from the TT every later iteration."""
    s = _searcher()
    gs = GameState.from_fen("k7/8/2K5/8/8/8/8/7R w - - 0 1")
    res = s.search(gs, hard_ns=_far(), max_depth=7)
    assert res.score == MATE - 3  # forced mate-in-2 under DEFAULT params
    assert res.uci in ("c6b6", "c6c7")
    buf = [0] * MAX_MOVES
    n = generate_legal(gs.board, buf)
    assert any(buf[i] == res.move for i in range(n))


def test_terminal_root_returns_no_move() -> None:
    s = _searcher()
    mated = GameState.from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert referee_terminal(mated) == ("black", "checkmate")
    res = s.search(mated, hard_ns=_far())
    assert res.move == 0 and res.uci == "0000"


def test_qsearch_no_standpat_in_check() -> None:
    """In check there is no stand-pat: the qsearch value is decided by the
    legal evasions, so a mated node returns -MATE+ply and a checked node
    never returns the raw static eval unverified."""
    s = _searcher()
    # side to move is checkmated on the spot
    gs = GameState.from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    s._begin(gs, _far(), 0)
    v = s._qs(-INF, INF, 1)
    assert v == -MATE + 1
    # in check with legal evasions (no captures): qsearch must search them
    gs2 = GameState.from_fen("8/8/8/8/8/8/8/r3K2k w - - 0 1")
    s._begin(gs2, _far(), 0)
    v2 = s._qs(-INF, INF, 1)
    assert -MATE_IN_MAX < v2 < MATE_IN_MAX  # real searched value, not a stand-pat


def test_threefold_in_game_history_is_terminal() -> None:
    gs = GameState.from_fen(START)
    for _ in range(2):
        for u in ("g1f3", "g8f6", "f3g1", "f6g8"):
            gs.apply_own_uci(u)
    assert referee_terminal(gs) == ("draw", "threefold_repetition")


def test_null_move_never_enters_history_or_ply_budget() -> None:
    s = _searcher()
    gs = GameState.from_fen(KIWI)
    s._begin(gs, _far(), 0)
    b = gs.board
    ply0 = b.absolute_ply()
    half0 = b.halfmove
    key0 = int(b.key)
    s._make_null(0)
    assert b.absolute_ply() == ply0  # official budget untouched
    assert b.side == 1
    # path floor bounds the repetition scan: the pre-null position is off-limits
    ph, gh = s._rep_scan(1)
    assert ph is False
    s._unmake_null(0)
    assert int(b.key) == key0 and b.halfmove == half0 and b.absolute_ply() == ply0


def test_in_path_repetition_detected() -> None:
    """A reversible line that revisits the root inside the search path is a
    draw heuristic; null moves cannot fabricate it."""
    s = _searcher()
    gs = GameState.from_fen(START)
    s._begin(gs, _far(), 0)
    b = gs.board
    moves = {"g1f3": 0, "g8f6": 0, "f3g1": 0, "f6g8": 0}
    buf = [0] * MAX_MOVES
    for uci, dest in (("g1f3", 1), ("g8f6", 2), ("f3g1", 3), ("f6g8", 4)):
        n = generate_legal(b, buf)
        for i in range(n):
            if move_to_uci(buf[i]) == uci:
                moves[uci] = buf[i]
        s._make(dest - 1, moves[uci])
    # position at ply 4 == root position (in-path repetition)
    ph, gh = s._rep_scan(4)
    assert ph is True
    for uci in ("f6g8", "f3g1", "g8f6", "g1f3"):
        s._unmake(0, moves[uci])  # order irrelevant to unmake correctness


def test_reasons_are_recorded() -> None:
    s = _searcher()
    gs = GameState.from_fen(KIWI)
    s.search(gs, hard_ns=_far(), max_depth=6)
    # at least the common pruning machinery must have fired and been counted
    assert s.stats["lmp"] + s.stats["futility"] + s.stats["see_quiet"] > 0
    assert s.stats["tt_cut"] + s.stats["nmp"] > 0


def test_lmrs_and_researches_happen() -> None:
    s = _searcher()
    gs = GameState.from_fen(KIWI)
    s.search(gs, hard_ns=_far(), max_depth=7)
    assert s.stats["lmr"] > 0
    # The LMR re-search itself must fire — a pvs_research alone cannot
    # satisfy this claim (the unreduced re-search is the verification the
    # reduction owes).
    assert s.stats["lmr_research"] > 0


def test_nmp_verification_is_live_at_search_depths() -> None:
    """Every NMP fail-high at every depth is verified (nmp_verify_depth=0);
    a position where the verification disproves the null must produce
    nmp_verify_fail > 0 — the counter is reachable, not dead code."""
    s = _searcher()
    gs = GameState.from_fen("8/3k4/8/8/2p5/8/8/2K1N3 w - - 0 1")
    s.search(gs, hard_ns=_far(), max_depth=8)
    assert s.stats["nmp"] > 0
    assert s.stats["nmp_verify_fail"] > 0


def test_white_pawn_previous_move_scores_history() -> None:
    """Piece code 0 (a white pawn) is a real previous move, not the
    no-previous-move sentinel (-1): a pawn move as prev1 must update
    continuation and counter history and order through them."""
    from engine.board import PAWN
    from engine.history import HistoryTables

    hist = HistoryTables()
    gs = GameState.from_fen(START)
    board = gs.board
    buf = [0] * MAX_MOVES
    n = generate_legal(board, buf)
    assert n > 0
    m = buf[0]
    piece = board._sq[m & 63]
    hist.update_quiets(
        board,
        m,
        [m],
        1,
        depth=8,
        prev1=(PAWN, 28),  # a white pawn moved to e4 last ply
        prev2=(-1, 0),  # no two-plies-up move
    )
    assert int(hist.cont[0, PAWN, 28, piece, (m >> 6) & 63]) != 0
    assert int(hist.counter[PAWN, 28]) == m
    # and the ordering path reads both back
    assert hist.counter_move((PAWN, 28)) == m
    assert hist.counter_move((-1, 0)) == 0
    s = hist.quiet_score(board, m & 63, (m >> 6) & 63, piece, (PAWN, 28), (-1, 0))
    assert s != 0
    # a null-move sentinel is still "no previous move"
    s2 = hist.quiet_score(board, m & 63, (m >> 6) & 63, piece, (-1, 0), (-1, 0))
    assert s != s2


def test_no_research_starts_after_stop() -> None:
    """An abort landing inside the reduced LMR search must unwind through
    unmake — never let the parent start the unreduced re-search (or any
    further _ab call) with the deadline already stopped."""
    s = _searcher()
    gs = GameState.from_fen(KIWI)
    stopped_entries = 0
    armed = False
    orig = s._ab

    prev_lmr = 0

    def tap(d, a, b, p, pv, cn):
        nonlocal stopped_entries, armed, prev_lmr
        if s.deadline.stop:
            stopped_entries += 1
        # The "lmr" counter bumps immediately before the reduced call —
        # a call entered right after a bump IS that reduced search.
        bumped = s.stats["lmr"] > prev_lmr
        prev_lmr = s.stats["lmr"]
        # beta > 0 means the parent's alpha < 0, so the torn score 0 > alpha
        # would force the re-search — exactly the window the defect lives in.
        if not armed and bumped and p >= 1 and pv == 0 and cn == 1 and b == a + 1 and b > 0:
            armed = True
            s.deadline.stop = True  # the deadline lands inside the reduced call
        return orig(d, a, b, p, pv, cn)

    s._ab = tap
    try:
        res = s.search(gs, hard_ns=_far(), max_depth=8)
    finally:
        s._ab = orig
    assert armed  # the stop-inside-a-reduced-search scenario really ran
    assert res.aborted
    assert stopped_entries == 0


def test_search_stack_sentinel_is_not_a_piece() -> None:
    """ss_piece uses -1 for "no previous move": after a null move the child
    frame must read the sentinel, never a phantom white pawn."""
    s = _searcher()
    gs = GameState.from_fen(KIWI)
    s._begin(gs, _far(), 0)
    assert s.ss_piece[3] == -1 and s.ss_piece[2] == -1
    # what _ab writes around a null: an empty (move, piece) pair
    s.ss_null[4] = 1
    s.ss_move[4] = 0
    s.ss_piece[4] = -1
    s._make_null(0)
    # the child reads "no previous move" — no counter/continuation ordering
    assert s.hist.counter_move((s.ss_piece[4], (s.ss_move[4] >> 6) & 63)) == 0
    s._unmake_null(0)
    assert s.ss_piece[4] == -1 and s.ss_move[4] == 0


def test_aborted_iteration_never_overwrites_completed_pv() -> None:
    s = _searcher()
    gs = GameState.from_fen(KIWI)
    full = s.search(gs, hard_ns=_far(), max_depth=4)
    assert full.pv
    # Abort mid-iteration: the returned move/PV is the last COMPLETED
    # iteration's commit, never the torn in-flight root candidate.
    res = None
    last = None
    for limit in (60, 150, 400, 1200, 4000):
        tap = _CommitTap(s)
        try:
            res = s.search(gs, node_limit=limit, hard_ns=_far(), max_depth=12)
        finally:
            tap.close()
        if res.aborted and tap.last is not None:
            last = tap.last
            break
    assert res is not None and res.aborted
    assert last is not None, "no limit produced an abort after a completed iteration"
    assert res.move == last["move"]
    assert res.pv == last["pv"]
    assert res.depth == last["depth"]
    # whatever is returned is legal and sane
    assert res.move != 0


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------


def test_hard_deadline_is_monotonic() -> None:
    s = _searcher()
    gs = GameState.from_fen(KIWI)
    t0 = time.monotonic()
    res = s.search(gs, hard_ns=time.monotonic_ns() + 80_000_000, max_depth=12)
    elapsed = time.monotonic() - t0
    assert res.aborted
    # stop + unwind must land well inside the margin scale
    assert elapsed < 0.4
    buf = [0] * MAX_MOVES
    n = generate_legal(gs.board, buf)
    assert any(buf[i] == res.move for i in range(n))


def test_objmode_clock_bridge_validates() -> None:
    assert validate_bridge()


def test_deadline_node_limit_fires_everywhere() -> None:
    s = _searcher()
    for limit in (1, 2, 17, 333):
        gs = GameState.from_fen(KIWI)
        res = s.search(gs, node_limit=limit, hard_ns=_far(), max_depth=8)
        assert res.aborted and s.info.nodes <= limit + 1
        buf = [0] * MAX_MOVES
        n = generate_legal(gs.board, buf)
        assert any(buf[i] == res.move for i in range(n))


def test_allocator_bounds() -> None:
    alloc = TimeAllocator()
    for left in (120_000, 60_000, 5_000, 900):
        a = alloc.allocate(left, 500, 0)
        assert 0 < a.soft_ns <= a.hard_ns
        assert a.hard_ns <= left * 1_000_000


def test_allocator_reserve_is_a_remaining_time_floor() -> None:
    """The reserve is a floor on REMAINING time: the hard wall never commits
    more than ``time_left - reserve - overhead`` — and below the reserve the
    allocation is zero, so no search runs (the fallback plays instantly)."""
    alloc = TimeAllocator()
    for left in (120_000, 20_000, 5_000, 900, 300, 150, 90, 80, 50, 1):
        for overhead in (0.0, 40.0):
            a = alloc.allocate(left, 500, 0, overhead_ms=overhead)
            floor = max(0.0, left - alloc.reserve_ms - overhead) * 1_000_000
            assert a.hard_ns <= floor, (left, overhead, a.hard_ns)
            assert 0 <= a.soft_ns <= a.hard_ns
    a = alloc.allocate(50, 500, 0)
    assert a.hard_ns == 0 and a.soft_ns == 0


def test_scaler_directions() -> None:
    sc = IterationScaler()
    rising = sc.scale(
        depth=8,
        stable_iters=0,
        score_drop_cp=60,
        best_move_node_fraction=0.1,
        contender_gap_cp=5,
    )
    falling = sc.scale(
        depth=8,
        stable_iters=8,
        score_drop_cp=-20,
        best_move_node_fraction=0.9,
        contender_gap_cp=200,
    )
    assert rising > 1.0 and falling < 1.0


def test_unwind_margin_is_measured() -> None:
    """The shipped margin must dominate the real unwind tail."""
    gs = GameState.from_fen(KIWI)
    buf = [0] * MAX_MOVES
    n = generate_legal(gs.board, buf)

    def unwind_once() -> None:
        for i in range(min(n, 24)):
            gs.board.make(buf[i])
        for _ in range(min(n, 24)):
            gs.board.unmake()

    worst = measure_unwind_ns(unwind_once, reps=32)
    assert worst < DEFAULT_UNWIND_MARGIN_NS  # 8ms dwarfs a full-root unwind


# ---------------------------------------------------------------------------
# GATE 1 — abort replay, every search phase, >= 10,000 randomized abort points
# ---------------------------------------------------------------------------


ABORT_POSITIONS = [
    START,
    KIWI,
    ENDGAME,
    TACTIC,
    "r1bqk2r/pp1ppppp/2n2n2/8/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 1",
    "8/6k1/8/8/8/2K5/8/4R3 w - - 0 1",
    "rnbqk2r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
]


def test_abort_replay_gate() -> None:
    """Every aborted search leaves the board, the accumulator-sync counters
    and the TT interpretation in a state fit for the next turn."""
    rng = random.Random(0xA10)
    s = _searcher(mib=4)
    # Accumulator-sync surrogate: a W05 accumulator subscribes to these
    # callbacks; net balance must return to zero after ANY abort.
    sync = {"depth": 0, "imbalance_seen": 0}

    def mk(_p: int, _m: int) -> None:
        sync["depth"] += 1

    def umk(_p: int, _m: int) -> None:
        sync["depth"] -= 1
        if sync["depth"] < 0:
            sync["imbalance_seen"] += 1

    s.on_make.append(mk)
    s.on_unmake.append(umk)
    s.on_null.append(lambda _p: mk(_p, 0))
    s.on_unnull.append(lambda _p: umk(_p, 0))

    phases = {0: 0, 1: 0, 2: 0}
    aborted_trials = 0
    trials = 0
    buf = [0] * MAX_MOVES
    while aborted_trials < ABORT_TRIALS:
        gs = GameState.from_fen(rng.choice(ABORT_POSITIONS))
        fen0 = gs.board.to_fen()
        key0 = int(gs.board.key)
        un0 = gs.board._un
        half0 = gs.board.halfmove
        limit = rng.randint(1, 160)
        res = s.search(gs, node_limit=limit, hard_ns=_far(), max_depth=8)
        trials += 1
        if not res.aborted:
            continue
        aborted_trials += 1
        # (a) board restored to the exact pre-search state
        assert int(gs.board.key) == key0
        assert gs.board.to_fen() == fen0
        assert gs.board._un == un0
        assert gs.board.halfmove == half0
        # (b) accumulator unwind balanced — nothing left on the path
        assert sync["depth"] == 0
        # (c) TT interpretation: a follow-up bounded search must still produce
        # a legal move — no torn entry may steer the next turn
        res2 = s.search(gs, node_limit=48, hard_ns=_far(), max_depth=8)
        n = generate_legal(gs.board, buf)
        assert any(buf[i] == res2.move for i in range(n)), (
            f"abort at {limit} corrupted next turn: {res2.uci}"
        )
        phases[s.info.phase] += 1
        assert s.stats is not None
    assert sync["imbalance_seen"] == 0
    assert aborted_trials >= ABORT_TRIALS
    # aborts must land inside both compiled-boundary kernels
    assert phases[1] > 0 and phases[2] > 0
    print(
        f"\nabort-replay gate: {aborted_trials} aborts over {trials} trials, "
        f"phase-at-abort histogram={phases}"
    )


# ---------------------------------------------------------------------------
# GATE 2 — terminal adjudication vs external referee, >= 50,000 positions
# ---------------------------------------------------------------------------


def _oracle(cb: chess.Board) -> tuple[str, str] | None:
    """The referee's adjudication (harness/referee.py lines 58-66 verbatim)."""
    finish = cb.outcome()
    if finish is not None:
        d = "draw" if finish.winner is None else ("white" if finish.winner else "black")
        return d, finish.termination.name.lower()
    if cb.is_repetition(3):
        return "draw", "threefold_repetition"
    if cb.is_fifty_moves():
        return "draw", "fifty_moves"
    if cb.ply() >= PLY_CAP:
        return "draw", "ply_cap"
    return None


def _compare(cb: chess.Board, gs: GameState, reasons: dict) -> int:
    want = _oracle(cb)
    got = referee_terminal(gs)
    if want is not None:
        reasons[want[1]] = reasons.get(want[1], 0) + 1
    else:
        reasons["nonterminal"] = reasons.get("nonterminal", 0) + 1
    assert got == want, f"disagree: ours={got} oracle={want} fen={cb.fen()}"
    return 1


def test_terminal_adjudication_gate() -> None:
    """Zero disagreements over >= 50,000 randomized positions including
    ordinary outcomes, threefold, fifty-move and the 600-ply cap (with a mate
    delivered exactly at the cap)."""
    rng = random.Random(0x7EF)
    reasons: dict[str, int] = {}
    compared = 0

    # (1) bulk: random playouts — every position compared, terminal included
    while compared < 34_000:
        cb = chess.Board()
        gs = GameState.from_fen(START)
        while compared < 34_000:
            compared += _compare(cb, gs, reasons)
            if cb.is_game_over():
                break
            mv = rng.choice(list(cb.legal_moves))
            gs.apply_own_uci(mv.uci())
            cb.push(mv)

    # (2) threefold terminals via shuffle games
    for _ in range(200):
        cb = chess.Board()
        gs = GameState.from_fen(START)
        seq = ("g1f3", "g8f6", "f3g1", "f6g8")
        for _rep in range(3):
            for u in seq:
                compared += _compare(cb, gs, reasons)
                gs.apply_own_uci(u)
                cb.push_uci(u)
        assert reasons.get("threefold_repetition", 0) > 0

    # (3) fifty-move: halfmove near 100 with material, one reversible move
    fifty_fens = [
        "8/8/8/3k4/8/3K4/3R4/8 w - - 99 40",
        "8/8/8/3k4/8/3K4/8/3N4 w - - 100 40",
        "6k1/8/8/8/8/8/1B6/K7 w - - 100 60",
        "8/8/8/8/8/8/1k6/K6R w - - 99 55",
    ]
    for fen in fifty_fens:
        cb = chess.Board(fen)
        gs = GameState.from_fen(fen)
        compared += _compare(cb, gs, reasons)
        for mv in cb.legal_moves:
            cb.push(mv)
            gs2 = GameState.from_fen(fen)
            gs2.apply_own_uci(mv.uci())
            compared += _compare(cb, gs2, reasons)
            cb.pop()

    # (4) the 600-ply cap, including a mate delivered exactly at the cap
    cap_pre = "K7/8/2k5/8/8/8/8/1q6 b - - 0 300"  # black mates at ply 600
    cb = chess.Board(cap_pre)
    gs = GameState.from_fen(cap_pre)
    compared += _compare(cb, gs, reasons)
    gs.apply_own_uci("b1b7")
    cb.push_uci("b1b7")
    compared += _compare(cb, gs, reasons)
    assert reasons.get("checkmate", 0) >= 1
    # non-mate move at ply 600 -> cap draw
    cap_draw_pre = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b - - 0 300"
    cb = chess.Board(cap_draw_pre)
    gs = GameState.from_fen(cap_draw_pre)
    compared += _compare(cb, gs, reasons)
    for mv in cb.legal_moves:
        cb.push(mv)
        gs2 = GameState.from_fen(cap_draw_pre)
        gs2.apply_own_uci(mv.uci())
        compared += _compare(cb, gs2, reasons)
        cb.pop()

    # (5) material-poor terminals + mates + stalemates
    extra = [
        "8/8/8/8/8/8/8/K6k w - - 0 1",  # bare kings
        "8/8/8/8/8/8/6B1/K6k w - - 0 1",
        "k7/8/1Q6/8/8/8/8/8 b - - 0 1",  # mate/stalemate shapes
        "k7/8/1Q6/K7/8/8/8/8 b - - 0 1",
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        "k7/2Q5/8/8/8/8/8/K7 b - - 0 1",
        "K7/8/8/8/4q3/8/8/6k1 w - - 0 1",
        "8/8/8/8/8/2k5/1q6/K7 w - - 0 1",
    ]
    for fen in extra:
        cb = chess.Board(fen)
        gs = GameState.from_fen(fen)
        compared += _compare(cb, gs, reasons)
        # and every legal move's landing position
        for mv in list(cb.legal_moves):
            cb.push(mv)
            gs2 = GameState.from_fen(fen)
            gs2.apply_own_uci(mv.uci())
            compared += _compare(cb, gs2, reasons)
            cb.pop()

    # (6) pad to >= 50k with more playouts from varied openings
    while compared < TERMINAL_POSITIONS:
        cb = chess.Board()
        gs = GameState.from_fen(START)
        for _ in range(rng.randint(2, 30)):  # random opening depth
            if cb.is_game_over():
                break
            mv = rng.choice(list(cb.legal_moves))
            gs.apply_own_uci(mv.uci())
            cb.push(mv)
        while compared < TERMINAL_POSITIONS:
            compared += _compare(cb, gs, reasons)
            if cb.is_game_over():
                break
            mv = rng.choice(list(cb.legal_moves))
            gs.apply_own_uci(mv.uci())
            cb.push(mv)

    assert compared >= TERMINAL_POSITIONS
    for k in (
        "checkmate",
        "threefold_repetition",
        "fifty_moves",
        "ply_cap",
        "insufficient_material",
        "stalemate",
    ):
        assert reasons.get(k, 0) > 0, f"no coverage for {k}: {reasons}"
    print(
        f"\nterminal-adjudication gate: {compared} positions, "
        f"0 disagreements, reasons={dict(sorted(reasons.items()))}"
    )


# ---------------------------------------------------------------------------
# GATE 4 — TT sweep 64/128/256/512 MiB with real long-game reuse
# ---------------------------------------------------------------------------

# A real long game (40 plies, verified-legal playout) played move-by-move so
# later searches reuse entries written during earlier searches.
LONG_GAME = (
    "b1a3 d7d6 a3b1 c8e6 a2a3 a7a6 b2b3 f7f6 g2g3 c7c5 g1h3 e8d7 "
    "c1b2 d8a5 b2e5 d6e5 f2f3 b7b5 h3f2 g7g5 b1c3 d7d6 b3b4 d6c7 "
    "c3a4 e6d7 d1c1 c7c8 f2h3 e5e4 a1a2 e4f3 c1a1 e7e5 a1c3 a5b6 "
    "c3a1 a8a7 a1d4 h7h5"
).split()


def test_tt_sweep_gate() -> None:
    """Same move sequence, four table sizes; reuse is measured over the real
    game, not a fixed-node microbenchmark."""
    results = {}
    for mib in (64, 128, 256, 512):
        s = _searcher(mib=mib)
        s.new_game()
        gs = GameState.from_fen(START)
        hits = 0
        nodes = 0
        for uci in LONG_GAME:
            res = s.search(gs, hard_ns=_far(), max_depth=4)
            buf = [0] * MAX_MOVES
            n = generate_legal(gs.board, buf)
            assert any(buf[i] == res.move for i in range(n)), (mib, uci)
            hits += s.info.tt_hits
            nodes += s.info.nodes
            gs.apply_own_uci(uci)  # fixed line: same positions for every size
        results[mib] = (hits, nodes, s.tt.hashfull())
    print(f"\ntt-sweep gate: {results}")
    # larger tables must retain enough for at least as much reuse on the game
    assert results[512][0] >= results[64][0]
    assert all(v[0] > 0 for v in results.values())


# ---------------------------------------------------------------------------
# GATE 5 — fallback before expensive work, book/TB miss returns to search
# ---------------------------------------------------------------------------


def test_fallback_legal_before_expensive_work() -> None:
    """With an already-expired deadline the fallback must still be a legal
    move — it is established before any expensive work runs."""
    s = _searcher()
    gs = GameState.from_fen(KIWI)
    res = s.search(gs, hard_ns=time.monotonic_ns() - 1, max_depth=8)
    buf = [0] * MAX_MOVES
    n = generate_legal(gs.board, buf)
    assert any(buf[i] == res.move for i in range(n))


def test_book_answer_used_when_legal() -> None:
    s = _searcher()
    gs = GameState.from_fen(START)
    mv = s.choose_move(gs, 5_000, root_answer=lambda _s: "e2e4")
    assert mv == "e2e4"


def test_book_miss_returns_to_search() -> None:
    s = _searcher()
    gs = GameState.from_fen(START)
    # miss: returns None
    mv = s.choose_move(gs, 1_500, root_answer=lambda _s: None, max_depth=4)
    buf = [0] * MAX_MOVES
    n = generate_legal(gs.board, buf)
    assert any(move_to_uci(buf[i]) == mv for i in range(n))


def test_book_exception_returns_to_search() -> None:
    s = _searcher()
    gs = GameState.from_fen(START)

    def boom(_s: GameState) -> None:
        raise RuntimeError("tablebase read failed")

    mv = s.choose_move(gs, 1_500, root_answer=boom, max_depth=4)
    buf = [0] * MAX_MOVES
    n = generate_legal(gs.board, buf)
    assert any(move_to_uci(buf[i]) == mv for i in range(n))


def test_book_illegal_answer_rejected() -> None:
    s = _searcher()
    gs = GameState.from_fen(START)
    # a legal-looking string that is not legal in this position
    mv = s.choose_move(gs, 1_500, root_answer=lambda _s: "e2e5", max_depth=4)
    assert mv != "e2e5"
    buf = [0] * MAX_MOVES
    n = generate_legal(gs.board, buf)
    assert any(move_to_uci(buf[i]) == mv for i in range(n))


def test_terminal_position_returns_0000() -> None:
    s = _searcher()
    gs = GameState.from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert s.choose_move(gs, 5_000) == "0000"


def test_legality_recheck_of_final_move() -> None:
    """Every returned move passes the generated-legal membership check."""
    s = _searcher()
    rng = random.Random(3)
    for _ in range(12):
        cb = chess.Board()
        for _ in range(rng.randint(0, 40)):
            if cb.is_game_over():
                break
            cb.push(rng.choice(list(cb.legal_moves)))
        if cb.is_game_over():
            continue
        gs = GameState.from_fen(cb.fen())
        res = s.search(gs, hard_ns=_far(), max_depth=4)
        buf = [0] * MAX_MOVES
        n = generate_legal(gs.board, buf)
        assert any(buf[i] == res.move for i in range(n)), gs.board.to_fen()


def test_correction_history_learns_residual() -> None:
    """Correction updates move the corrected score toward searched evidence
    relative to the raw static value (never recursively on itself)."""
    from engine.history import HistoryTables

    h = HistoryTables()
    gs = GameState.from_fen(KIWI)
    b = gs.board
    raw = simple_eval(b)
    before = h.correction_cp(b, b.side)
    for _ in range(6):
        h.update_correction(b, b.side, depth=8, raw_static=raw, best=raw + 60)
    after = h.correction_cp(b, b.side)
    assert after > before  # learned the +60 residual direction
    assert after <= 60


def test_move_decode_consistency() -> None:
    gs = GameState.from_fen(KIWI)
    buf = [0] * MAX_MOVES
    n = generate_legal(gs.board, buf)
    for i in range(n):
        frm, to, promo, flag, piece, captured = decode_move(buf[i])
        assert 0 <= frm < 64 and 0 <= to < 64
        assert gs.board._sq[frm] == piece

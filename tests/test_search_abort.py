"""Abort-commit and root-terminal regressions.

An aborted iteration must not overwrite the last completed principal
variation. When no iteration completed, the pre-search legal fallback
remains authoritative. Node-limit and monotonic-clock aborts exercise
restoration at different points in the tree.

Root adjudication checks ordinary outcomes before threefold repetition,
fifty-move draws and the absolute 600-ply cap. Mate at the cap must be
resolved before the cap's draw rule."""

from __future__ import annotations

import random
import time

import chess
import pytest

from engine.board import MAX_MOVES, move_to_uci
from engine.movegen import generate_legal
from engine.search import DRAW, INF, MATE, Searcher
from engine.state import PLY_CAP, GameState, referee_terminal
from engine.tt import TranspositionTable

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWI = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
ENDGAME = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"  # node_limit=251 case
TACTIC = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
MATE1 = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
KRK = "k7/8/2K5/8/8/8/8/7R w - - 0 1"

# Root-terminal edge cases (verified against python-chess + the referee).
# Referee draw: fifty-move root even though a1a8 mates next ply.
FIFTY_WITH_MATE = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 100 50"
# Referee mate: checkmate beats the fifty-move claim (outcome() first).
MATED_AT_FIFTY = "R5k1/5ppp/8/8/8/8/8/6K1 b - - 101 50"
# Referee draw: absolute ply >= 600 (startpos pushed to fullmove 301).
CAP_ROOT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 301"
# Referee mate: mated at exactly ply 600 — outcome() beats the cap.
MATED_AT_CAP = "K7/1q6/2k5/8/8/8/8/8 w - - 1 301"
# Non-terminal root at ply 599 with b1b7 delivering mate exactly at 600.
PRE_CAP_MATE = "K7/8/2k5/8/8/8/8/1q6 b - - 0 300"

ABORT_FENS = (
    START,
    KIWI,
    ENDGAME,
    TACTIC,
    MATE1,
    KRK,
    "r1bqk2r/pp1ppppp/2n2n2/8/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 1",
    "8/6k1/8/8/8/2K5/8/4R3 w - - 0 1",
    "rnbqk2r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/1P6/8/4k3/8/4K3 w - - 0 1",
    "rnbq1bnr/pppkpppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1",
    "8/8/8/8/8/5k2/8/3K3q b - - 0 1",
)

ABORTS_TARGET = 20_000
TERMINAL_POSITIONS = 50_000


def _far() -> int:
    return time.monotonic_ns() + 10**12


def _searcher(mib: int = 1, **kw) -> Searcher:
    return Searcher(tt=TranspositionTable(mib=mib), **kw)


class _CommitTap:
    """Record every COMPLETED iteration's committed (move, score, depth, pv).

    ``search()`` commits an iteration by assigning ``info.root_best`` to
    ``result.move`` and then calling ``self._extract_pv(depth)``. Tapping
    that call captures ``root_best``/``root_score``/``pv`` at exactly the
    values the driver committed for that iteration. An aborted iteration
    never reaches the call, so ``last`` is always the last completed
    commit — the provenance reference.
    """

    def __init__(self, s: Searcher) -> None:
        self.s = s
        self.calls: list[dict] = []
        self._orig = s._extract_pv
        s._extract_pv = self._wrapped  # type: ignore[method-assign]

    def _wrapped(self, depth: int) -> list[int]:
        pv = self._orig(depth)
        self.calls.append(
            {
                "move": int(self.s.info.root_best),
                "score": int(self.s.info.root_score),
                "depth": int(depth),
                "pv": list(pv),
            }
        )
        return pv

    @property
    def last(self) -> dict | None:
        return self.calls[-1] if self.calls else None

    def close(self) -> None:
        self.s._extract_pv = self._orig  # type: ignore[method-assign]


def _expected_fallback(gs: GameState, root_hint: int = 0) -> int:
    """The move ``search()`` pre-establishes as the legal fallback."""
    buf = [0] * MAX_MOVES
    n = generate_legal(gs.board, buf)
    fb = buf[0]
    if root_hint:
        for i in range(n):
            if (buf[i] & 0x7FFF) == (root_hint & 0x7FFF):
                fb = buf[i]
                break
    return fb


def _assert_committed(res, tap: _CommitTap, gs: GameState, root_hint: int = 0) -> None:
    """The provenance claim: ``res.move`` is the last completed iteration's
    commit; with zero completed iterations it is the legal fallback."""
    last = tap.last
    if last is None:
        assert res.move == _expected_fallback(gs, root_hint), (
            f"no completed iteration but played {move_to_uci(res.move)}; "
            f"expected the legal fallback "
            f"{move_to_uci(_expected_fallback(gs, root_hint))}"
        )
        assert res.depth == 0 and res.pv == []
        return
    assert res.move == last["move"], (
        f"torn-PV: played {move_to_uci(res.move)}, committed "
        f"{move_to_uci(last['move'])} at depth {last['depth']}"
    )
    assert res.pv == last["pv"], "aborted iteration overwrote the committed PV"
    assert res.depth == last["depth"]
    assert res.score == last["score"]


# ---------------------------------------------------------------------------
# Gate 2 — deterministic regressions (FAIL on the pre-repair driver)
# ---------------------------------------------------------------------------


def test_regression_abort_returns_committed_move_known_case() -> None:
    """The reviewer's measured case: ENDGAME with node_limit=251.

    Pre-repair driver played the torn iteration's root_best (b4f4) over
    the committed completed-iteration move (e2e4)."""
    s = _searcher(4)
    gs = GameState.from_fen(ENDGAME)
    tap = _CommitTap(s)
    try:
        res = s.search(gs, node_limit=251, hard_ns=_far(), max_depth=10)
    finally:
        tap.close()
    assert res.aborted
    assert tap.last is not None  # at least one iteration completed
    _assert_committed(res, tap, gs)


def test_regression_fifty_move_root_with_mate_available_is_draw() -> None:
    """Fifty-move root adjudicated before the mate lands: R6/R7's case.

    The referee returns draw(fifty_moves); the broken driver searched and
    answered a1a8 with score MATE-1 = 29999."""
    gs = GameState.from_fen(FIFTY_WITH_MATE)
    assert referee_terminal(gs) == ("draw", "fifty_moves")
    s = _searcher(1)
    s._begin(gs, _far(), 0)
    nodes0 = s.info.nodes
    assert s._ab(4, -INF, INF, 0, 1, 0) == DRAW
    assert s.info.nodes - nodes0 == 1  # decided at the root, no move searched
    res = _searcher(1).search(gs, hard_ns=_far(), max_depth=4)
    assert res.score == DRAW


def test_regression_threefold_root_is_draw() -> None:
    """A threefold root is terminal: the referee draws it before search."""
    gs = GameState.from_fen(START)
    for _ in range(2):
        for u in ("g1f3", "g8f6", "f3g1", "f6g8"):
            gs.apply_own_uci(u)
    assert referee_terminal(gs) == ("draw", "threefold_repetition")
    s = _searcher(1)
    s._begin(gs, _far(), 0)
    nodes0 = s.info.nodes
    assert s._ab(4, -INF, INF, 0, 1, 0) == DRAW
    assert s.info.nodes - nodes0 == 1
    res = _searcher(1).search(gs, hard_ns=_far(), max_depth=4)
    assert res.score == DRAW


def test_regression_ply_cap_root_is_draw() -> None:
    """abs_ply >= 600 with legal moves is a draw, not a searched position.

    On the broken driver this still scores 0 — every child is over the cap —
    so the single-node delta is what makes the fix observable."""
    gs = GameState.from_fen(CAP_ROOT)
    assert gs.board.absolute_ply() == PLY_CAP
    assert referee_terminal(gs) == ("draw", "ply_cap")
    s = _searcher(1)
    s._begin(gs, _far(), 0)
    nodes0 = s.info.nodes
    assert s._ab(3, -INF, INF, 0, 1, 0) == DRAW
    assert s.info.nodes - nodes0 == 1  # decided at the root, no move searched
    res = _searcher(1).search(gs, hard_ns=_far(), max_depth=3)
    assert res.score == DRAW


def test_regression_insufficient_material_root_is_draw() -> None:
    """K-vs-KB has legal moves but is an immediate insufficient-material draw.

    halfmove is 0 so the searched score would be the bishop's material —
    the draw decision, not a coincidence of all children being draws."""
    fen = "6k1/8/8/8/8/8/1B6/K7 w - - 0 60"
    gs = GameState.from_fen(fen)
    assert referee_terminal(gs) == ("draw", "insufficient_material")
    s = _searcher(1)
    s._begin(gs, _far(), 0)
    nodes0 = s.info.nodes
    assert s._ab(3, -INF, INF, 0, 1, 0) == DRAW
    assert s.info.nodes - nodes0 == 1


def test_mate_at_ply_cap_is_mate_not_draw() -> None:
    """Mate delivered exactly at ply 600: outcome() beats the cap.
    referee_terminal reports the WINNER — black mates the white king."""
    gs = GameState.from_fen(MATED_AT_CAP)
    assert gs.board.absolute_ply() == PLY_CAP
    assert referee_terminal(gs) == ("black", "checkmate")
    s = _searcher(1)
    s._begin(gs, _far(), 0)
    assert s._ab(3, -INF, INF, 0, 1, 0) == -MATE


def test_fifty_move_claim_on_mated_ply_is_mate() -> None:
    """halfmove >= 100 AND mated on the same ply: outcome() beats fifty."""
    gs = GameState.from_fen(MATED_AT_FIFTY)
    assert gs.board.halfmove >= 100
    assert referee_terminal(gs) == ("white", "checkmate")
    s = _searcher(1)
    s._begin(gs, _far(), 0)
    assert s._ab(3, -INF, INF, 0, 1, 0) == -MATE


def test_mate_in_one_at_the_cap_is_found() -> None:
    """Ply-599 root with b1b7 mate at ply 600: near-terminal, not terminal —
    the real search runs and scores the cap-edge mate correctly."""
    gs = GameState.from_fen(PRE_CAP_MATE)
    assert gs.board.absolute_ply() == PLY_CAP - 1
    assert referee_terminal(gs) is None
    s = _searcher(1)
    s._begin(gs, _far(), 0)
    nodes0 = s.info.nodes
    v = s._ab(3, -INF, INF, 0, 1, 0)
    assert s.info.nodes - nodes0 > 1  # a real search ran
    assert v == MATE - 1


def test_twofold_root_is_not_terminal() -> None:
    """A second occurrence is an in-tree draw heuristic, NOT a root terminal —
    the referee requires threefold. Guards against repairing the hole by
    routing the root through ``_draw_score``."""
    gs = GameState.from_fen(START)
    for u in ("g1f3", "g8f6", "f3g1", "f6g8"):
        gs.apply_own_uci(u)
    assert referee_terminal(gs) is None  # twofold only
    s = _searcher(1)
    s._begin(gs, _far(), 0)
    nodes0 = s.info.nodes
    v = s._ab(3, -INF, INF, 0, 1, 0)
    assert s.info.nodes - nodes0 > 1  # real search, not a false draw claim
    assert -MATE < v < MATE


def test_terminal_root_search_reports_draw() -> None:
    """The shipped ``search()`` path on a terminal-draw root: no iteration
    runs, score is the draw, and the pre-established legal move stands."""
    gs = GameState.from_fen(FIFTY_WITH_MATE)
    res = _searcher(1).search(gs, hard_ns=_far(), max_depth=4)
    assert res.depth == 0 and res.iterations == []
    assert res.score == DRAW
    buf = [0] * MAX_MOVES
    n = generate_legal(gs.board, buf)
    assert any(buf[i] == res.move for i in range(n))


def test_fifty_move_play_path_returns_legal_without_search() -> None:
    """The shipped move-CHOICE path on a fifty-move root: referee ordering
    draws it before any iteration — the returned move is legal and zero
    nodes are searched (the mate-in-1 is not played)."""
    gs = GameState.from_fen(FIFTY_WITH_MATE)
    s = _searcher(1)
    mv = s.choose_move(gs, 5000)
    buf = [0] * MAX_MOVES
    n = generate_legal(gs.board, buf)
    assert any(move_to_uci(buf[i]) == mv for i in range(n))
    assert s.info.nodes == 0  # decided by the referee: no search ran


def test_pre_fifty_mate_in_one_play_path_still_searches() -> None:
    """One halfmove earlier the same position is NOT terminal: the play
    path runs a real search and plays the mate-in-1."""
    gs = GameState.from_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 99 50")
    assert referee_terminal(gs) is None
    s = _searcher(1)
    assert s.choose_move(gs, 5000) == "a1a8"
    assert s.info.nodes > 0


# ---------------------------------------------------------------------------
# Gate 1 — >= 20,000 randomized abort points, committed-PV provenance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [256, pytest.param(ABORTS_TARGET, marks=pytest.mark.slow)])
def test_abort_committed_move_gate(target: int) -> None:
    """Aborting mid-iteration never returns a torn root move.

    For every aborted search with at least one completed iteration the
    returned (move, score, depth, pv) must equal the LAST completed
    commit; with zero completed iterations the move is the pre-search
    legal fallback. Both abort mechanisms are exercised — the node limit
    and the monotonic hard clock — and the abort point's poll-site phase
    (``info.phase``: 1 = _ab, 2 = _qs, 0 = driver) is histogrammed.
    """
    rng = random.Random(0xB03F1)  # independent of prior gates' seeds
    s = _searcher(4)
    aborts = 0
    with_committed = 0
    overwrites = 0
    pv_tears = 0
    exceptions = 0
    trials = 0
    phases_all: dict[int, int] = {}
    phases_ow: dict[int, int] = {}
    clock_aborts = 0
    examples = []
    while aborts < target:
        fen = rng.choice(ABORT_FENS)
        gs = GameState.from_fen(fen)
        buf = [0] * MAX_MOVES
        n = generate_legal(gs.board, buf)
        if n == 0:
            continue
        # ~1 trial in 9 uses an injected monotonic clock that jumps past the
        # deadline mid-search — the same path a real 120+0.5 flag fall hits.
        use_clock = rng.random() < 0.11
        hint = buf[rng.randrange(n)] if rng.random() < 0.12 else 0
        if use_clock:
            expire_after = rng.randint(6, 300)
            base = time.monotonic_ns()
            hard = base + 50_000_000
            reads = {"n": 0}

            def clock(
                reads: dict = reads,
                base: int = base,
                hard: int = hard,
                k: int = expire_after,
            ) -> int:
                reads["n"] += 1
                return hard + 1 if reads["n"] >= k else base

            srch = _searcher(4, now=clock)
            kw = {"hard_ns": hard}
        else:
            limit = rng.choice(
                (
                    rng.randint(1, 60),
                    rng.randint(30, 300),
                    rng.randint(200, 900),
                )
            )
            srch = s
            kw = {"node_limit": limit, "hard_ns": _far()}
        tap = _CommitTap(srch)
        try:
            res = srch.search(gs, max_depth=10, root_hint=hint, **kw)
        except Exception:
            tap.close()
            raise
        tap.close()
        trials += 1
        if not res.aborted:
            continue
        aborts += 1
        if use_clock:
            clock_aborts += 1
        phases_all[srch.info.phase] = phases_all.get(srch.info.phase, 0) + 1
        try:
            _assert_committed(res, tap, gs, hint)
            if tap.last is not None:
                with_committed += 1
        except AssertionError:
            exceptions += 1
            overwrites += 1
            phases_ow[srch.info.phase] = phases_ow.get(srch.info.phase, 0) + 1
            if res.pv and res.pv[0] != res.move:
                pv_tears += 1
            if len(examples) < 8:
                examples.append(
                    dict(
                        fen=fen.split()[0],
                        kw={k: v for k, v in kw.items() if k != "hard_ns"},
                        played=move_to_uci(res.move),
                        committed=(move_to_uci(tap.last["move"]) if tap.last else None),
                        pv=res.pv_uci,
                        depth=res.depth,
                        phase=srch.info.phase,
                        clock=use_clock,
                    )
                )
    print(
        f"\nabort-PV gate: trials={trials} aborts={aborts} "
        f"(clock={clock_aborts}) with_committed={with_committed} "
        f"overwrites={overwrites} pv_tears={pv_tears} exceptions={exceptions}"
    )
    print(f"phase_at_abort={dict(sorted(phases_all.items()))} phase_at_overwrite={phases_ow}")
    for ex in examples:
        print(" example", ex)
    assert exceptions == 0, f"{exceptions} torn-PV returns"
    # Coverage: both poll-site phases and a strong completed-iteration body.
    assert phases_all.get(1, 0) > 0 and phases_all.get(2, 0) > 0
    assert with_committed >= target // 2


# ---------------------------------------------------------------------------
# Gate 3 — root-terminal agreement with the official referee, >= 50,000
# ---------------------------------------------------------------------------

# referee_terminal() termination name -> REASONS bucket for the root note.
_ROOT_REASON = {
    "checkmate": "mate",
    "stalemate": "stalemate",
    "insufficient_material": "insufficient",
    "seventyfive_moves": "seventyfive",
    "fivefold_repetition": "fivefold",
    "threefold_repetition": "rep",
    "fifty_moves": "fifty",
    "ply_cap": "cap",
}
_TERMINAL_STATS = set(_ROOT_REASON.values())


def _oracle(cb: chess.Board) -> tuple[str, str] | None:
    """The referee's adjudication (harness/referee.py verbatim ordering)."""
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


def _root_ab_compare(cb: chess.Board, gs: GameState, s: Searcher, reasons: dict, bad: list) -> int:
    """Drive the patched ``_ab`` root branch on one position.

    A terminal claim at the root returns before any move is generated, so
    ``info.nodes`` grows by exactly 1 — that single-node delta is the
    airtight discriminator between "the root adjudicated terminal" and "a
    real search ran". Only positions the referee could actually see are
    compared — an invalid FEN here is a test bug and fails loudly.
    """
    assert cb.is_valid(), f"test supplied invalid position: {cb.fen()}"
    want = _oracle(cb)
    stats0 = {r: s.stats[r] for r in _TERMINAL_STATS}
    s._begin(gs, _far(), 0)
    nodes0 = s.info.nodes
    got = s._ab(1, -INF, INF, 0, 1, 0)
    n_used = s.info.nodes - nodes0
    deltas = {r: s.stats[r] - stats0[r] for r in _TERMINAL_STATS}
    tag = want[1] if want is not None else "nonterminal"
    reasons[tag] = reasons.get(tag, 0) + 1
    if want is None:
        ok = n_used > 1 and -MATE < got < MATE
    elif want[0] == "draw":
        ok = (
            got == DRAW
            and n_used == 1
            and sum(deltas.values()) == 1
            and deltas[_ROOT_REASON[want[1]]] == 1
        )
    else:
        ok = got == -MATE and n_used == 1 and sum(deltas.values()) == 1 and deltas["mate"] == 1
    if not ok:
        bad.append((cb.fen(), want, got, n_used, {k: v for k, v in deltas.items() if v}))
    return 1


def _root_search_compare(cb: chess.Board, gs: GameState, s: Searcher, bad: list) -> None:
    """The shipped ``search()`` path on the same position."""
    want = _oracle(cb)
    res = s.search(gs, hard_ns=_far(), max_depth=1)
    buf = [0] * MAX_MOVES
    n = generate_legal(gs.board, buf)
    if want is None:
        ok = res.depth >= 1
    elif want[0] == "draw":
        # Either nothing legal exists (stalemate -> "0000") or the
        # pre-established legal fallback stands; score is the draw either way.
        ok = (
            res.score == DRAW
            and res.depth == 0
            and (res.move == 0 or any(buf[i] == res.move for i in range(n)))
        )
    else:
        ok = res.move == 0  # mated root: nothing legal exists
    if not ok:
        bad.append((cb.fen(), want, res.score, res.depth, res.uci))


@pytest.mark.parametrize("target", [1000, pytest.param(TERMINAL_POSITIONS, marks=pytest.mark.slow)])
def test_root_terminal_agreement_gate(target: int) -> None:
    """``_ab`` at ply 0 and ``search()`` both agree with the referee on
    a deterministic sample of terminal and near-terminal positions.
    The extended profile uses at least 50,000 positions.

    Terminal roots return exactly the referee's decision — DRAW (0) for
    every draw rule, -MATE for a mated root — and never run a live search
    (the single-node delta proves it). Non-terminal roots always run a
    real search (nodes > 1). The referee's ordering is exercised at every
    contested boundary: mate vs cap, mate vs fifty, threefold vs fifty,
    fivefold vs seventyfive, twofold (in-tree draw heuristic) vs the
    root's real threefold requirement.
    """
    rng = random.Random(0x7E3D)
    s = _searcher(1)
    reasons: dict[str, int] = {}
    bad_ab: list = []
    bad_search: list = []
    compared = 0
    search_checked = 0

    # (1) bulk random playouts: every position through _ab at the root,
    # every 10th also through the shipped search() path.
    while compared < target * 4 // 5:
        cb = chess.Board()
        gs = GameState.from_fen(START)
        while compared < target * 4 // 5:
            compared += _root_ab_compare(cb, gs, s, reasons, bad_ab)
            if compared % 10 == 1:
                _root_search_compare(cb, gs, s, bad_search)
                search_checked += 1
            if cb.is_game_over():
                break
            mv = rng.choice(list(cb.legal_moves))
            gs.apply_own_uci(mv.uci())
            cb.push(mv)

    # (2) targeted terminals and near-terminals at every contested boundary.
    # ``_do`` compares the current (cb, gs) through both surfaces inline, so
    # no snapshotting is needed while the generators mutate them in place.
    def _do(cb: chess.Board, gs: GameState) -> None:
        nonlocal compared, search_checked
        compared += _root_ab_compare(cb, gs, s, reasons, bad_ab)
        _root_search_compare(cb, gs, s, bad_search)
        search_checked += 1

    # fifty-move / cap / insufficient / mate boundaries
    edge_fens = [
        FIFTY_WITH_MATE,  # fifty root with a mate-in-1 available -> draw
        MATED_AT_FIFTY,  # mated AND halfmove>=100 -> mate
        CAP_ROOT,  # abs_ply 600 with legal moves -> draw
        MATED_AT_CAP,  # mated at exactly ply 600 -> mate
        PRE_CAP_MATE,  # ply 599, mate-in-1 at the cap -> search finds it
        "8/8/8/3k4/8/3K4/3R4/8 w - - 99 40",  # halfmove 99: near-fifty
        "8/8/8/3k4/8/3K4/8/3N4 w - - 100 40",  # fifty
        "6k1/8/8/8/8/8/1B6/K7 w - - 100 60",  # fifty AND insufficient
        "k7/8/8/8/8/8/8/K6R w - - 99 55",  # near-fifty
        "k7/8/8/8/8/8/8/K6R w - - 100 55",  # fifty
        "k7/8/8/8/8/8/8/K6R w - - 149 55",  # near-seventyfive
        "k7/8/8/8/8/8/8/K6R w - - 150 55",  # seventyfive
        "8/8/8/8/8/8/8/K6k w - - 0 1",  # bare kings, legal moves
        "8/8/8/8/8/8/2B5/K6k w - - 0 1",  # KBvK
        "8/8/8/8/8/8/3N4/K6k w - - 0 1",  # KNvK
        "6k1/8/8/8/8/5b2/6B1/K7 w - - 0 1",  # KBvKB same-colour bishops
        "k7/8/1Q6/8/8/8/K7/8 b - - 0 1",  # stalemate shapes
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        "k7/2Q5/8/8/8/8/8/K7 b - - 0 1",
        "K7/8/8/8/4q3/8/8/6k1 w - - 0 1",  # mated / near-mate
        "8/8/8/8/8/2k5/1q6/K7 w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b - - 0 300",  # ply 599
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b - - 0 301",  # ply 601
        "k7/8/8/8/8/8/8/K6R w - - 100 300",  # fifty AND ply cap
        "k7/8/8/8/8/8/8/K6R w - - 99 300",  # near-fifty at ply 599
    ]
    for fen in edge_fens:
        cb = chess.Board(fen)
        gs = GameState.from_fen(fen)
        _do(cb, gs)
        for mv in list(cb.legal_moves):
            cb2 = chess.Board(fen)
            cb2.push(mv)
            gs2 = GameState.from_fen(fen)
            gs2.apply_own_uci(mv.uci())
            _do(cb2, gs2)

    # repetition families: twofold (non-terminal!), threefold, fivefold
    for cycles in (1, 2, 4):
        cb = chess.Board()
        gs = GameState.from_fen(START)
        seq = ("g1f3", "g8f6", "f3g1", "f6g8")
        for _rep in range(cycles):
            for u in seq:
                _do(cb, gs)
                gs.apply_own_uci(u)
                cb.push_uci(u)

    # threefold claimed while halfmove >= 100: referee orders threefold
    # BEFORE fifty — a long reversible shuffle game supplies it.
    for _ in range(40):
        cb = chess.Board("8/8/8/8/3k4/8/3K4/3r3R w - - 0 1")
        gs = GameState.from_fen(cb.fen())
        while not cb.is_game_over() and cb.halfmove_clock < 160:
            _do(cb, gs)
            if cb.is_repetition(3) and cb.halfmove_clock >= 100:
                break
            mv = rng.choice(list(cb.legal_moves))
            gs.apply_own_uci(mv.uci())
            cb.push(mv)

    # fivefold reached past seventyfive: outcome() orders seventyfive first.
    cb = chess.Board()
    gs = GameState.from_fen(START)
    seq = ("g1f3", "g8f6", "f3g1", "f6g8")
    for _rep in range(38):
        for u in seq:
            _do(cb, gs)
            gs.apply_own_uci(u)
            cb.push_uci(u)

    # (3) pad to >= 50,000 with playouts from varied opening depths
    while compared < target:
        cb = chess.Board()
        gs = GameState.from_fen(START)
        for _ in range(rng.randint(2, 30)):
            if cb.is_game_over():
                break
            mv = rng.choice(list(cb.legal_moves))
            gs.apply_own_uci(mv.uci())
            cb.push(mv)
        while compared < target:
            compared += _root_ab_compare(cb, gs, s, reasons, bad_ab)
            if compared % 10 == 1:
                _root_search_compare(cb, gs, s, bad_search)
                search_checked += 1
            if cb.is_game_over():
                break
            mv = rng.choice(list(cb.legal_moves))
            gs.apply_own_uci(mv.uci())
            cb.push(mv)

    print(
        f"\nroot-terminal gate: {compared} positions through _ab, "
        f"{search_checked} through search(), "
        f"_ab_disagreements={len(bad_ab)} search_disagreements={len(bad_search)}"
    )
    print(f"reasons={dict(sorted(reasons.items()))}")
    for row in bad_ab[:8]:
        print(" _ab disagree", row)
    for row in bad_search[:8]:
        print(" search disagree", row)
    assert not bad_ab and not bad_search
    assert compared >= target
    for k in (
        "checkmate",
        "threefold_repetition",
        "fifty_moves",
        "ply_cap",
        "insufficient_material",
        "stalemate",
        "nonterminal",
    ):
        assert reasons.get(k, 0) > 0, f"no coverage for {k}: {reasons}"

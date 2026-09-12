"""End-to-end integration tests for the wired RX-FINAL agent (spec 3.1).

Covers the full path ``get_move(fen, time_left_ms) -> UCI``: FEN parse,
legal fallback before expensive work, observed-history reconciliation with
the unknown-prefix marker, the book/TB eligibility gate, bounded ID-PVS +
qsearch over the real F512-EF evaluator, abort-safe iteration commit, and
the final legality recheck. Terminal cases are cross-checked against python-chess.

Random weights are generated deterministically; nothing here asserts move
quality — only legality, safety and contract behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import agent_rx, model_io  # noqa: E402
from engine.board import Board, move_to_uci  # noqa: E402
from engine.evaluate import EvalWeights, evaluate_fresh  # noqa: E402
from engine.movegen import generate_legal, legal_uci  # noqa: E402
from engine.search import SearchResult  # noqa: E402
from engine.state import GameState, referee_terminal  # noqa: E402

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
EP_FEN = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
PROMO_FEN = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
MATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
STALEMATE_FEN = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
INSUFFICIENT_FEN = "8/8/8/8/8/8/8/K6k w - - 0 1"
FIFTY_FEN = "4k3/8/8/8/8/8/8/R3K3 w - - 100 60"
CAP_FEN = "4k3/8/8/8/8/8/7P/4K3 w - - 50 301"  # ply 600
MATE_AT_CAP_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 301"

_buf = [0] * 256


def _legal_set(board: Board) -> set[str]:
    return set(legal_uci(board))


def _diag_sink(monkeypatch) -> list[dict]:
    rows: list[dict] = []

    def sink(payload: dict) -> None:
        rows.append(payload)

    monkeypatch.setattr(agent_rx, "_emit", sink)
    return rows


@pytest.fixture(scope="module")
def weights() -> EvalWeights:
    return EvalWeights.random(0x5EED)


@pytest.fixture()
def agent(weights, monkeypatch):
    """A fresh wired Agent (no kernel warm; first eval compiles lazily)."""
    diags = _diag_sink(monkeypatch)
    ag = agent_rx.Agent(weights, {"weights": "random", "seed": 0x5EED}, tt_mib=8)
    yield ag, diags
    ag.state = None


# ---------------------------------------------------------------------------
# Model <-> weights round-trip
# ---------------------------------------------------------------------------


def test_model_roundtrip_parity(weights):
    meta = {"weights": "random", "seed": 0x5EED, "scale_shift": int(weights.scale_shift)}
    blob = model_io.write_model(agent_rx.model_dict_from_weights(weights, meta))
    model = model_io.read_model(blob)
    w2 = agent_rx.weights_from_model_dict(model)
    names = (
        "psq_w",
        "thr_w",
        "pp_w",
        "bias",
        "w1",
        "b1",
        "w2",
        "b2",
        "w3",
        "b3",
        "psqt_w",
        "psqt_b",
    )
    for name in names:
        a, b = getattr(weights, name), getattr(w2, name)
        assert a.shape == b.shape and np.array_equal(a, b), name
    for fen in (STARTPOS, EP_FEN, PROMO_FEN):
        board = Board.from_fen(fen)
        assert evaluate_fresh(weights, board) == evaluate_fresh(w2, board)


def test_frugal_decode_matches_read_model(weights):
    """The chunked container decode is byte-identical to read_model."""
    blob = model_io.write_model(agent_rx.model_dict_from_weights(weights, {"t": 1}))
    ref = model_io.read_model(blob)
    got = agent_rx.read_model_frugal(blob)
    assert set(ref) == set(got)
    for name in ref:
        if name == "_meta":
            assert ref[name] == got[name]
        else:
            assert np.array_equal(ref[name], got[name]), name
    # and it rejects what read_model rejects
    bad = bytearray(blob)
    bad[-1] ^= 1
    with pytest.raises(model_io.ModelFormatError):
        agent_rx.read_model_frugal(bytes(bad))


def test_evaluator_is_the_value_fn(agent):
    ag, _ = agent
    # The searcher must call OUR wrapped F512-EF evaluator, not simple_eval.
    assert ag.searcher.eval_fn.__self__ is ag
    assert isinstance(ag.evaluator, agent_rx.Evaluator)


# ---------------------------------------------------------------------------
# get_move: legality battery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fen",
    [
        STARTPOS,
        EP_FEN,
        PROMO_FEN,
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "8/8/8/8/8/5k2/8/6K1 w - - 0 1",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    ],
)
def test_get_move_returns_legal_uci(agent, fen):
    ag, diags = agent
    uci = ag.get_move(fen, 5000)
    assert uci in _legal_set(Board.from_fen(fen))
    assert ag.fallback_events == 0
    assert not any(str(d.get("reason", "")).startswith("fallback:") for d in diags)


def test_promotion_suffix_formation(agent):
    """Promotion UCI suffixes form correctly end-to-end (not just legality)."""
    board = Board.from_fen(PROMO_FEN)
    promos = {m for m in legal_uci(board) if len(m) == 5}
    assert promos == {"a7a8q", "a7a8r", "a7a8b", "a7a8n"}
    ag, _ = agent
    # Force the returned move to be a promotion by capping the choice set at
    # the promotion square via a stored-answer hit.
    for promo in ("a7a8q", "a7a8n"):
        ag.root_table = lambda _state, p=promo: p
        assert ag.get_move(PROMO_FEN, 200) == promo
    ag.root_table = None


def test_get_move_terminal_returns_0000(agent):
    ag, _ = agent
    assert ag.get_move(MATE_FEN, 5000) == "0000"
    assert ag.get_move(STALEMATE_FEN, 5000) == "0000"


def test_get_move_malformed_fen(agent):
    ag, diags = agent
    assert ag.get_move("not a fen", 5000) == "0000"
    assert ag.get_move("8/8/8/8/8/8/8/8 w - - 0 1", 5000) == "0000"  # no kings
    reasons = [d.get("reason") for d in diags]
    assert "fallback:fen_parse" in reasons


# ---------------------------------------------------------------------------
# History reconciliation
# ---------------------------------------------------------------------------


def _opp_reply(board: Board) -> int:
    """Deterministic 'opponent' move: first legal in generator order."""
    generate_legal(board, _buf)
    return _buf[0]


def test_history_reconciles_across_calls(agent):
    ag, diags = agent
    ref = chess.Board()
    u1 = ag.get_move(STARTPOS, 5000)
    ref.push(chess.Move.from_uci(u1))
    # opponent plays the first legal reply by our own generator
    our_board = Board.from_fen(ref.fen())
    opp = _opp_reply(our_board)
    ref.push(chess.Move.from_uci(move_to_uci(opp)))
    u2 = ag.get_move(ref.fen(), 5000)
    assert u2 in {m.uci() for m in ref.legal_moves}
    # root + our move + inferred opponent move + our next move == 4 records,
    # and the opponent's reply was reconciled without a reset.
    assert ag.state._game_n == 4
    assert ag.history_resets == 0
    assert not any(d.get("event") == "history_discontinuity" for d in diags)


def test_unknown_prefix_is_retained(agent):
    ag, _ = agent
    u1 = ag.get_move(STARTPOS, 200)
    assert ag.state.unknown_prefix is True
    # Continue the real game: our returned move + a deterministic reply.
    ref = chess.Board()
    ref.push(chess.Move.from_uci(u1))
    our = Board.from_fen(ref.fen())
    ref.push(chess.Move.from_uci(move_to_uci(_opp_reply(our))))
    u2 = ag.get_move(ref.fen(), 200)
    assert u2 in {m.uci() for m in ref.legal_moves}
    # The reconciled history grew, but the pre-start prefix stays unknown.
    assert ag.state.unknown_prefix is True
    assert ag.state._game_n == 4
    assert ag.history_resets == 0


def test_unmatched_observation_resets_and_reports(agent):
    ag, diags = agent
    ag.get_move(STARTPOS, 200)
    # A position unrelated to the game history -> conservative reset.
    ag.get_move("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", 200)
    assert ag.history_resets == 1
    assert ag.state.unknown_prefix is True
    assert any(d.get("event") == "history_discontinuity" for d in diags)


# ---------------------------------------------------------------------------
# Root eligibility gate (book/TB contract)
# ---------------------------------------------------------------------------


def test_root_gate_hit_returns_stored_move(agent):
    ag, diags = agent
    ag.root_table = lambda state: "e2e4"
    uci = ag.get_move(STARTPOS, 5000)
    assert uci == "e2e4"
    assert any(d.get("book") == "hit" for d in diags)


def test_root_gate_illegal_answer_falls_to_search(agent):
    ag, diags = agent
    ag.root_table = lambda state: "a1a8"  # never legal in the start position
    uci = ag.get_move(STARTPOS, 5000)
    assert uci != "a1a8"
    assert uci in _legal_set(Board.from_fen(STARTPOS))


def test_root_gate_exception_falls_to_search(agent):
    ag, _ = agent

    def boom(state):
        raise RuntimeError("book corrupted")

    ag.root_table = boom
    uci = ag.get_move(STARTPOS, 5000)
    assert uci in _legal_set(Board.from_fen(STARTPOS))
    assert ag.fallback_events == 0  # a gate error is a miss, not a fallback


def test_root_gate_miss_searches(agent):
    ag, diags = agent
    ag.root_table = lambda state: None
    uci = ag.get_move(STARTPOS, 5000)
    assert uci in _legal_set(Board.from_fen(STARTPOS))
    assert any(d.get("book") == "miss" for d in diags)


def test_root_gate_ineligible_after_move_20(agent):
    ag, diags = agent
    called = []

    def probe(state):
        called.append(1)
        return "e2e4"

    ag.root_table = probe
    # fullmove 25, pieces present -> ineligible under both rules
    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 25"
    uci = ag.get_move(fen, 200)
    assert not called
    assert uci in _legal_set(Board.from_fen(fen))
    assert any(d.get("book") == "ineligible" for d in diags)


# ---------------------------------------------------------------------------
# Abort safety / clocks
# ---------------------------------------------------------------------------


def test_zero_clock_returns_legal_instantly(agent):
    ag, _ = agent
    uci = ag.get_move(STARTPOS, 0)
    assert uci in _legal_set(Board.from_fen(STARTPOS))


def test_tiny_clock_returns_legal(agent):
    ag, _ = agent
    uci = ag.get_move(STARTPOS, 1)
    assert uci in _legal_set(Board.from_fen(STARTPOS))


def test_aborted_iteration_keeps_completed_pv(agent):
    """An aborted iteration never overwrites the best completed PV."""
    from tests.test_search_abort import _CommitTap

    ag, _ = agent
    state = GameState.from_fen(STARTPOS)
    ag.evaluator.set_root(state.board)
    full = ag.searcher.search(state, max_depth=4)
    assert full.move and full.depth >= 3 and not full.fallback_used
    # Now abort mid-iteration: the returned move is the last completed
    # commit (or the legal fallback if nothing completed) — never the torn
    # in-flight root candidate.
    res = None
    last = None
    for limit in (60, 150, 400, 1200, 4000):
        tap = _CommitTap(ag.searcher)
        try:
            res = ag.searcher.search(state, node_limit=limit, max_depth=10)
        finally:
            tap.close()
        if res.aborted and tap.last is not None:
            last = tap.last
            break
    assert res is not None and res.aborted
    if last is not None:
        assert res.move == last["move"]
        assert res.pv == last["pv"]
    else:
        # zero completed iterations: the pre-search legal fallback stands
        assert generate_legal(state.board, _buf) > 0
        assert res.move == _buf[0]
    assert res.move == 0 or move_to_uci(res.move) in _legal_set(state.board)
    # and the completed-iteration commit is intact either way
    assert full.pv and all(m != 0 for m in full.pv)


def test_get_move_never_plays_torn_pv(agent, monkeypatch):
    """The packaged path plays the completed PV head, never a torn root move.

    Agent-level guarantee independent of the search driver (R16 blocker 1):
    if a SearchResult carries partial=True with a completed PV, get_move
    returns pv[0] even when result.move holds a different legal root move.
    """
    ag, diags = agent
    state = GameState.from_fen(STARTPOS)
    ag.evaluator.set_root(state.board)
    full = ag.searcher.search(state, max_depth=4)
    assert full.pv and all(m != 0 for m in full.pv)
    committed = full.pv[0]
    n = generate_legal(state.board, _buf)
    torn = next(_buf[i] for i in range(n) if _buf[i] != committed)

    def torn_search(*a, **k):
        res = SearchResult()
        res.move = torn  # an aborted iteration's torn candidate
        res.pv = list(full.pv)
        res.partial = True
        res.aborted = True
        res.depth = full.depth
        res.score = full.score
        return res

    monkeypatch.setattr(ag.searcher, "search", torn_search)
    uci = ag.get_move(STARTPOS, 5000)
    assert uci == move_to_uci(committed), (
        f"played torn-PV {uci}; committed PV head is {move_to_uci(committed)}"
    )
    assert any(d.get("event") == "pv_guard" for d in diags)

    # With zero completed iterations the resolved partial root move is
    # still a legal, playable answer (no completed PV exists to keep).
    def all_aborted(*a, **k):
        res = SearchResult()
        res.move = torn
        res.partial = True
        res.aborted = True
        return res

    monkeypatch.setattr(ag.searcher, "search", all_aborted)
    ag.state = None
    assert ag.get_move(STARTPOS, 5000) in _legal_set(Board.from_fen(STARTPOS))


def test_get_move_on_terminal_draw_root_returns_legal(agent):
    """A root the referee already drew (fifty-move w/ mate-in-1) is not
    searched as a win: get_move returns a legal move, no search ran."""
    ag, diags = agent
    fen = "6k1/8/6K1/8/8/8/8/R7 w - - 100 60"  # Ra8# exists; referee: draw
    uci = ag.get_move(fen, 2000)
    assert uci in _legal_set(Board.from_fen(fen))
    assert ag.last_result is None  # search never ran on a decided root
    assert any(d.get("event") == "root_terminal" for d in diags)


def test_search_exception_falls_back(agent, monkeypatch):
    ag, diags = agent

    def boom(*a, **k):
        raise RuntimeError("injected search failure")

    monkeypatch.setattr(ag.searcher, "search", boom)
    uci = ag.get_move(STARTPOS, 5000)
    assert uci in _legal_set(Board.from_fen(STARTPOS))
    assert ag.fallback_events == 1
    assert any(d.get("reason") == "fallback:search_exception" for d in diags)


def test_board_desync_detected_and_rebuilt(agent, monkeypatch):
    ag, diags = agent

    real_search = ag.searcher.search

    def corrupt(state, **kw):
        # simulate an escape that left a move made
        generate_legal(state.board, _buf)
        state.board.make(_buf[0])
        return real_search(state, **kw)

    monkeypatch.setattr(ag.searcher, "search", corrupt)
    uci = ag.get_move(STARTPOS, 200)
    assert uci in _legal_set(Board.from_fen(STARTPOS))
    assert ag.fallback_events == 1
    assert any(d.get("reason") == "fallback:board_desync" for d in diags)


def test_evaluator_fault_degrades_to_simple_eval(agent, monkeypatch):
    ag, diags = agent

    def boom(board):
        raise RuntimeError("eval fault")

    monkeypatch.setattr(ag.evaluator, "evaluate", boom)
    uci = ag.get_move(STARTPOS, 5000)
    assert uci in _legal_set(Board.from_fen(STARTPOS))
    assert ag.eval_faults >= 1
    assert any(d.get("reason") == "fallback:evaluator_fault" for d in diags)


# ---------------------------------------------------------------------------
# Terminal agreement with the referee ordering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fen,expected",
    [
        (MATE_FEN, ("black", "checkmate")),
        (STALEMATE_FEN, ("draw", "stalemate")),
        (INSUFFICIENT_FEN, ("draw", "insufficient_material")),
        (FIFTY_FEN, ("draw", "fifty_moves")),
        (CAP_FEN, ("draw", "ply_cap")),
        # mate beats the ply cap (referee checks outcome first)
        (MATE_AT_CAP_FEN, ("black", "checkmate")),
    ],
)
def test_referee_terminal_ordering(fen, expected):
    state = GameState.from_fen(fen)
    assert referee_terminal(state) == expected
    # cross-check with python-chess where it has a concept
    board = chess.Board(fen)
    if expected[1] == "checkmate":
        assert board.is_checkmate()
    elif expected[1] == "stalemate":
        assert board.is_stalemate()
    elif expected[1] == "insufficient_material":
        assert board.is_insufficient_material()
    elif expected[1] == "fifty_moves":
        assert board.is_fifty_moves()
    elif expected[1] == "ply_cap":
        assert board.ply() >= 600
        assert board.outcome() is None  # python-chess has no cap; referee adds it


def test_threefold_agrees_with_referee():
    state = GameState.from_fen(STARTPOS)
    b = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        move = None
        n = generate_legal(state.board, _buf)
        for i in range(n):
            if move_to_uci(_buf[i]) == uci:
                move = _buf[i]
                break
        assert move is not None
        state._push_game(move)
        b.push(chess.Move.from_uci(uci))
    assert referee_terminal(state) == ("draw", "threefold_repetition")
    assert b.is_repetition(3)

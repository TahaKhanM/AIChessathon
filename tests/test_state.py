"""W01 state: three identities, observed history, referee terminal order."""

from __future__ import annotations

import chess
import pytest

from engine.movegen import legal_uci
from engine.state import (
    PLY_CAP,
    GameState,
    RepetitionContext,
    ValueContext,
    referee_terminal,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _pc_referee(pc: chess.Board) -> tuple[str, str] | None:
    finish = pc.outcome()
    if finish is not None:
        winner = "draw" if finish.winner is None else ("white" if finish.winner else "black")
        return winner, finish.termination.name.lower()
    if pc.is_repetition(3):
        return "draw", "threefold_repetition"
    if pc.is_fifty_moves():
        return "draw", "fifty_moves"
    if pc.ply() >= 600:
        return "draw", "ply_cap"
    return None


def test_three_identities_are_distinct_objects() -> None:
    state = GameState.from_fen(START_FEN, model_version=3, utility_version=7)
    geo = state.geometric_identity()
    value = state.value_context()
    rep = state.repetition_context()
    assert isinstance(geo, int)
    assert isinstance(value, ValueContext)
    assert isinstance(rep, RepetitionContext)
    assert value.halfmove == 0
    assert value.remaining_horizon == PLY_CAP
    assert value.model_version == 3
    assert value.utility_version == 7
    assert rep.unknown_prefix is True
    assert rep.history_complete is False
    assert len(rep.known_reversible_keys) >= 1

    # Same geometry, different value context (halfmove / horizon / model).
    twin = GameState.from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 12 1",
        model_version=9,
        utility_version=1,
    )
    assert twin.geometric_identity() == geo
    assert twin.value_context() != value
    assert twin.value_context().halfmove == 12
    assert twin.value_context().remaining_horizon == PLY_CAP - twin.board.absolute_ply()

    # Repetition context is not folded into the geometric key.
    assert state.repetition_context() != geo
    assert state.value_context() != geo


def test_starting_fen_counters_drive_absolute_ply() -> None:
    fen = "6k1/pbp1r1p1/6qp/2p5/P1N1P3/1P2R2P/1BP3PK/3R4 w - - 2 25"
    state = GameState.from_fen(fen)
    pc = chess.Board(fen)
    assert state.board.absolute_ply() == pc.ply() == 48
    assert state.board.halfmove == 2
    assert state.value_context().remaining_horizon == PLY_CAP - 48


def test_unknown_prefix_and_observed_history_reconciliation() -> None:
    state = GameState.from_fen(START_FEN)
    assert state.repetition_context().unknown_prefix is True
    state.apply_own_uci("e2e4")
    assert state.board.to_fen() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    # Opponent replies; engine sees only the new FEN.
    opp = chess.Board()
    opp.push_uci("e2e4")
    opp.push_uci("e7e5")
    state.observe_fen(opp.fen())
    assert state.board.to_fen() == opp.fen()
    assert state.repetition_context().unknown_prefix is True
    assert len(state.repetition_context().known_reversible_keys) >= 3
    # Search path is empty at the root.
    assert state.repetition_context().search_path_len == 0


def test_null_move_does_not_fabricate_threefold_or_consume_horizon() -> None:
    state = GameState.from_fen(START_FEN)
    before = state.value_context()
    geo_before = state.geometric_identity()
    keys_before = list(state.repetition_context().known_reversible_keys)
    state.make_null()
    assert state.geometric_identity() != geo_before
    assert state.value_context().remaining_horizon == before.remaining_horizon
    assert list(state.repetition_context().known_reversible_keys) == keys_before
    assert referee_terminal(state) is None
    state.unmake_null()
    assert state.geometric_identity() == geo_before


def test_null_moves_live_on_the_search_stack() -> None:
    """make_null records on the search path so pop_search unwinds it:
    the undo stack and _search_n can never desynchronize."""
    state = GameState.from_fen(START_FEN)
    state.push_search_uci("e2e4")
    fen_after = state.board.to_fen()
    rep_after = state.repetition_context()
    state.make_null()
    # the post-null position is on the path, marked as a hard boundary
    mid = state.repetition_context()
    assert mid.search_path_len == rep_after.search_path_len + 1
    assert state.repetition_count() == 1  # the pre-null position is hidden
    state.pop_search()  # pops the NULL — board and stack stay in lock-step
    assert state.board.to_fen() == fen_after
    assert state.repetition_context().search_path_len == rep_after.search_path_len
    state.pop_search()
    assert state.repetition_context().search_path_len == 0
    assert state.board.to_fen() == START_FEN
    # an over-pop is a desync bug, not a silent drift
    with pytest.raises(RuntimeError):
        state.pop_search()


def test_unmake_null_round_trips_through_pop_search() -> None:
    state = GameState.from_fen(START_FEN)
    geo0 = state.geometric_identity()
    state.make_null()
    assert state.geometric_identity() != geo0
    state.unmake_null()
    assert state.geometric_identity() == geo0
    assert state.repetition_context().search_path_len == 0
    with pytest.raises(RuntimeError):
        state.unmake_null()  # nothing to unmake — must raise, not drift


def test_checkmate_beats_ply_cap() -> None:
    # Fool's mate geometry with fullmove set so ply is already 600, black mated white.
    # Position after 1.f3 e5 2.g4 Qh4# is mate, ply=4 normally.
    mate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 3"
    pc = chess.Board(mate)
    assert pc.is_checkmate()
    # Same geometry with fullmove 301, white to move → ply = 2*(301-1)+0 = 600.
    capped = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 301"
    pc2 = chess.Board(capped)
    assert pc2.ply() == 600
    assert pc2.outcome() is not None
    state = GameState.from_fen(capped)
    result, reason = referee_terminal(state)
    assert (result, reason) == _pc_referee(pc2)
    assert reason == "checkmate"
    assert result == "black"


def test_ordinary_outcome_before_fifty_and_ply_cap() -> None:
    # Stalemate with a high halfmove clock and ply already at the cap.
    fen = "7k/5K2/6Q1/8/8/8/8/8 b - - 120 301"
    pc = chess.Board(fen)
    assert pc.ply() >= 600
    assert pc.halfmove_clock >= 100
    assert pc.is_stalemate()
    state = GameState.from_fen(fen)
    result, reason = referee_terminal(state)
    assert (result, reason) == _pc_referee(pc)
    assert reason == "stalemate"


def test_threefold_then_fifty_then_ply_cap_order() -> None:
    # Build a threefold by repeating a quiet manoeuvre; referee must report threefold
    # (not fifty-move) while the halfmove clock is still below 100.
    pc = chess.Board()
    for uci in ["b1c3", "b8c6", "c3b1", "c6b8", "b1c3", "b8c6", "c3b1", "c6b8"]:
        pc.push_uci(uci)
    assert pc.is_repetition(3)
    assert not pc.is_fifty_moves()
    state = GameState.from_fen(START_FEN)
    for uci in ["b1c3", "b8c6", "c3b1", "c6b8", "b1c3", "b8c6", "c3b1", "c6b8"]:
        state.apply_own_uci(uci)
    got = referee_terminal(state)
    assert got == _pc_referee(pc)
    assert got == ("draw", "threefold_repetition")

    # Fifty-move with no repetition: K+R vs K, shuffled until halfmove >= 100.
    fen = "7k/8/8/8/8/8/8/R3K3 w Q - 99 200"
    pc = chess.Board(fen)
    assert not pc.is_fifty_moves()
    pc.push_uci("a1a2")
    assert pc.is_fifty_moves()
    assert pc.outcome() is None
    state = GameState.from_fen(fen)
    state.apply_own_uci("a1a2")
    got = referee_terminal(state)
    assert got == _pc_referee(pc)
    assert got == ("draw", "fifty_moves")

    # Ply cap with counters that skip mate/stalemate/fifty/threefold.
    fen = "4k3/8/8/8/8/8/8/R3K3 w Q - 0 301"
    pc = chess.Board(fen)
    assert pc.ply() == 600
    assert pc.outcome() is None
    assert not pc.is_repetition(3)
    assert not pc.is_fifty_moves()
    state = GameState.from_fen(fen)
    got = referee_terminal(state)
    assert got == _pc_referee(pc)
    assert got == ("draw", "ply_cap")


def test_insufficient_material_via_outcome() -> None:
    state = GameState.from_fen("8/8/8/8/8/8/8/4K2k w - - 0 1")
    pc = chess.Board("8/8/8/8/8/8/8/4K2k w - - 0 1")
    got = referee_terminal(state)
    assert got == _pc_referee(pc)
    assert got == ("draw", "insufficient_material")


def test_search_path_repetition_is_separate_from_game_history() -> None:
    state = GameState.from_fen(START_FEN)
    state.apply_own_uci("b1c3")
    root_rep = state.repetition_context()
    assert root_rep.search_path_len == 0
    state.push_search_uci("b8c6")
    state.push_search_uci("c3b1")
    mid = state.repetition_context()
    assert mid.search_path_len == 2
    assert mid.unknown_prefix is True
    assert list(mid.known_reversible_keys) == list(root_rep.known_reversible_keys)
    assert state.repetition_count() >= 1
    state.pop_search()
    state.pop_search()
    assert state.repetition_context().search_path_len == 0
    assert list(state.repetition_context().known_reversible_keys) == list(
        root_rep.known_reversible_keys
    )


def test_referee_terminal_matches_python_chess_on_a_short_game() -> None:
    pc = chess.Board()
    state = GameState.from_fen(START_FEN)
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        assert legal_uci(state.board)
        pc.push_uci(uci)
        state.apply_own_uci(uci)
        assert referee_terminal(state) == _pc_referee(pc)
    assert referee_terminal(state) == ("black", "checkmate")

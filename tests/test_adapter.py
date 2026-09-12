"""Public entry-point contracts, including interruption and history changes."""

import importlib

import chess
import pytest

import agent


@pytest.fixture
def adapter():
    return importlib.reload(agent)


def test_no_weights_needed_and_zero_clock_is_legal(adapter):
    board = chess.Board()
    move = chess.Move.from_uci(adapter.get_move(board.fen(), 0))
    assert move in board.legal_moves


def test_observed_history_across_calls(adapter):
    board = chess.Board()
    board.push_uci(adapter.get_move(board.fen(), 0))
    board.push(next(iter(board.legal_moves)))
    move = chess.Move.from_uci(adapter.get_move(board.fen(), 0))
    assert move in board.legal_moves
    board.push(move)
    assert adapter._state.board.to_fen() == board.fen(en_passant="fen")


def test_unrelated_observation_resets_state(adapter):
    adapter.get_move(chess.STARTING_FEN, 0)
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    assert chess.Move.from_uci(adapter.get_move(board.fen(), 0)) in board.legal_moves


def test_terminal_and_invalid_input(adapter):
    assert adapter.get_move("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", 1000) == "0000"
    with pytest.raises(ValueError):
        adapter.get_move("not a fen", 1000)

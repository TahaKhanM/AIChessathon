"""W01 board: FEN, make/unmake, Zobrist, promotions, EP occupancy."""

from __future__ import annotations

import random

import numpy as np
import pytest

from engine.board import (
    BB_COUNT,
    EMPTY,
    Board,
    move_to_uci,
    snapshot,
)
from engine.movegen import generate_legal, legal_uci

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _apply_uci(board: Board, uci: str) -> None:
    moves = np.empty(256, dtype=np.uint32)
    n = generate_legal(board, moves)
    target = None
    for i in range(n):
        m = int(moves[i])
        if move_to_uci(m) == uci:
            target = m
            break
    assert target is not None, f"{uci} not legal in {board.to_fen()}"
    board.make(target)


def test_start_fen_roundtrip() -> None:
    board = Board.from_fen(START_FEN)
    assert board.to_fen() == START_FEN
    assert board.side == 0
    assert int(board.mailbox[0]) != EMPTY
    assert board.bitboards.shape == (BB_COUNT,)
    assert board.bitboards.dtype == np.uint64


def test_zobrist_matches_full_recompute_after_e4() -> None:
    board = Board.from_fen(START_FEN)
    _apply_uci(board, "e2e4")
    assert board.key == board.compute_key()
    assert board.to_fen() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def test_make_unmake_restores_start() -> None:
    board = Board.from_fen(START_FEN)
    before = snapshot(board)
    _apply_uci(board, "e2e4")
    board.unmake()
    assert snapshot(board) == before
    assert board.key == before.key


def test_all_four_promotion_types_and_capture_promotions() -> None:
    board = Board.from_fen("k7/4P3/8/8/8/8/8/4K3 w - - 0 1")
    ucis = set(legal_uci(board))
    assert {"e7e8q", "e7e8r", "e7e8b", "e7e8n"} <= ucis
    for uci in sorted(ucis):
        before = snapshot(board)
        _apply_uci(board, uci)
        assert board.key == board.compute_key()
        board.unmake()
        assert snapshot(board) == before

    board = Board.from_fen("k5n1/5P2/8/8/8/8/8/4K3 w - - 0 1")
    ucis = set(legal_uci(board))
    assert {"f7f8q", "f7g8q", "f7g8n", "f7g8r", "f7g8b"} <= ucis
    _apply_uci(board, "f7g8q")
    assert board.to_fen().startswith("k5Q1/8/8/8/8/8/8/4K3 b - -")


def test_ep_updates_three_squares_and_zobrist() -> None:
    # White pawn e5 captures d6 EP: from e5, to d6, captured pawn on d5.
    board = Board.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    before = snapshot(board)
    d5, e5, d6 = 35, 36, 43
    assert int(board.mailbox[e5]) != EMPTY
    assert int(board.mailbox[d5]) != EMPTY
    assert int(board.mailbox[d6]) == EMPTY
    _apply_uci(board, "e5d6")
    assert int(board.mailbox[e5]) == EMPTY
    assert int(board.mailbox[d5]) == EMPTY
    assert int(board.mailbox[d6]) != EMPTY
    assert board.key == board.compute_key()
    occ = int(board.occupied)
    assert (occ & (1 << d5)) == 0
    assert (occ & (1 << e5)) == 0
    assert occ & (1 << d6)
    board.unmake()
    assert snapshot(board) == before


def test_null_move_roundtrip_clears_ep_without_consuming_game_ply() -> None:
    board = Board.from_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 2")
    ply_before = board.absolute_ply()
    before = snapshot(board)
    board.make_null()
    assert board.side == 0
    assert board.ep_square < 0
    assert board.absolute_ply() == ply_before
    assert board.key == board.compute_key()
    board.unmake_null()
    assert snapshot(board) == before


def test_make_unmake_roundtrip_one_million_moves() -> None:
    rng = random.Random(20260911)
    moves_buf = np.empty(256, dtype=np.uint32)
    restored = 0
    sequences = 0
    while restored < 1_000_000:
        board = Board.from_fen(START_FEN)
        before = snapshot(board)
        ply = 0
        while ply < 180:
            n = generate_legal(board, moves_buf)
            if n == 0:
                break
            board.make(int(moves_buf[rng.randrange(n)]))
            ply += 1
        sequences += 1
        while ply:
            board.unmake()
            ply -= 1
            restored += 1
        assert snapshot(board) == before
        assert board.key == before.key
        np.testing.assert_array_equal(board.bitboards, before.bitboards)
        np.testing.assert_array_equal(board.mailbox, before.mailbox)
    assert restored >= 1_000_000
    assert sequences >= 1


@pytest.mark.parametrize(
    "fen,uci",
    [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4"),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", "e1g1"),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", "e1c1"),
        ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", "d7c8q"),
        ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", "d7c8n"),
    ],
)
def test_zobrist_incremental_equals_recompute(fen: str, uci: str) -> None:
    board = Board.from_fen(fen)
    _apply_uci(board, uci)
    assert int(board.key) == int(board.compute_key())
    board.unmake()
    assert int(board.key) == int(board.compute_key())

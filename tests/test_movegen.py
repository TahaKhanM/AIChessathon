"""W01 movegen: perft, python-chess differential, pins, castling, EP."""

from __future__ import annotations

import random

import chess
import numpy as np
import pytest

from engine.board import Board, move_to_uci
from engine.movegen import generate_legal, in_check, legal_uci, perft

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Chess Programming Wiki perft suite (positions 1–6).
PERFT_CASES = [
    (START_FEN, 6, 119_060_324),
    # Position 2: Kiwipete
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193_690_690),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674_624),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15_833_292),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 5, 89_941_194),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 5, 164_075_551),
]

SHALLOW_PERFT = [
    (START_FEN, 1, 20),
    (START_FEN, 2, 400),
    (START_FEN, 3, 8_902),
    (START_FEN, 4, 197_281),
    (PERFT_CASES[1][0], 1, 48),
    (PERFT_CASES[1][0], 2, 2_039),
    (PERFT_CASES[1][0], 3, 97_862),
    (PERFT_CASES[2][0], 1, 14),
    (PERFT_CASES[3][0], 1, 6),
    (PERFT_CASES[4][0], 1, 44),
    (PERFT_CASES[5][0], 1, 46),
]


@pytest.mark.parametrize("fen,depth,expected", SHALLOW_PERFT)
def test_shallow_perft(fen: str, depth: int, expected: int) -> None:
    board = Board.from_fen(fen)
    assert perft(board, depth) == expected
    assert board.to_fen().split(" ")[:4] == fen.split(" ")[:4]


@pytest.mark.slow
@pytest.mark.parametrize("fen,depth,expected", PERFT_CASES)
def test_standard_perft_suite(fen: str, depth: int, expected: int) -> None:
    board = Board.from_fen(fen)
    assert perft(board, depth) == expected


def test_castling_through_check_and_out_of_check() -> None:
    # Rook on f3 attacks f1: white cannot castle kingside through check.
    board = Board.from_fen("4k3/8/8/8/8/5r2/8/R3K2R w KQ - 0 1")
    ucis = set(legal_uci(board))
    assert "e1g1" not in ucis
    assert "e1c1" in ucis  # queenside path is free and not attacked

    # Destination g1 attacked:
    board = Board.from_fen("4k3/8/8/8/8/6r1/8/R3K2R w KQ - 0 1")
    ucis = set(legal_uci(board))
    assert "e1g1" not in ucis

    # King in check: no castling.
    board = Board.from_fen("4k3/8/8/8/8/4r3/8/R3K2R w KQ - 0 1")
    ucis = set(legal_uci(board))
    assert "e1g1" not in ucis
    assert "e1c1" not in ucis

    # Occupied b1 blocks queenside.
    board = Board.from_fen("4k3/8/8/8/8/8/8/RN2K2R w KQ - 0 1")
    ucis = set(legal_uci(board))
    assert "e1c1" not in ucis
    assert "e1g1" in ucis


def test_legal_vs_pseudo_legal_en_passant() -> None:
    # Horizontal discovery: capturing EP would leave king in check.
    fen = "8/8/8/K2pP2r/8/8/8/4k3 w - d6 0 1"
    pc = chess.Board(fen)
    board = Board.from_fen(fen)
    our = set(legal_uci(board))
    theirs = {m.uci() for m in pc.legal_moves}
    assert our == theirs
    assert "e5d6" not in our
    assert pc.has_legal_en_passant() is False
    assert pc.ep_square == chess.D6


def test_pinned_en_passant() -> None:
    # The capturing pawn is pinned on the d-file; EP is illegal, the push is not.
    fen = "3r4/8/8/2pP4/8/8/8/3K3k w - c6 0 1"
    pc = chess.Board(fen)
    board = Board.from_fen(fen)
    our = set(legal_uci(board))
    theirs = {m.uci() for m in pc.legal_moves}
    assert our == theirs
    assert "d5c6" not in our
    assert "d5d6" in our


def test_discovered_attacks_from_pinned_slider() -> None:
    # Queen on e4 pinned to white king by black rook on e8; can move on the e-file only.
    board = Board.from_fen("4r2k/8/8/8/4Q3/8/8/4K3 w - - 0 1")
    ucis = set(legal_uci(board))
    assert "e4d5" not in ucis
    assert "e4f5" not in ucis
    assert "e4e5" in ucis
    assert "e4e6" in ucis
    assert "e4e7" in ucis
    assert "e4e8" in ucis  # capture the pinning rook
    # Knight on e4, bishop on e2: moving the knight is a discovered attack on e8.
    board = Board.from_fen("4k3/8/8/8/4N3/8/4B3/4K3 w - - 0 1")
    ucis = set(legal_uci(board))
    assert "e4d6" in ucis
    assert "e4f6" in ucis


def test_all_promotion_types_including_capture() -> None:
    board = Board.from_fen("r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1")
    ucis = set(legal_uci(board))
    assert "b7a8q" in ucis
    assert "b7a8r" in ucis
    assert "b7a8b" in ucis
    assert "b7a8n" in ucis
    assert "b7b8q" in ucis


def test_random_legal_differential_against_python_chess() -> None:
    rng = random.Random(20260911)
    moves_buf = np.empty(256, dtype=np.uint32)
    compared = 0
    mismatches = 0
    samples: list[str] = []
    while compared < 200_000:
        pc = chess.Board()
        board = Board.from_fen(pc.fen())
        for _ in range(256):
            ours = set(legal_uci(board))
            theirs = {m.uci() for m in pc.legal_moves}
            if ours != theirs:
                mismatches += 1
                samples.append(f"{pc.fen()} extra={ours - theirs} missing={theirs - ours}")
                break
            assert in_check(board) == pc.is_check()
            compared += 1
            if compared >= 200_000:
                break
            n = generate_legal(board, moves_buf)
            if n == 0:
                break
            choice = int(moves_buf[rng.randrange(n)])
            uci = move_to_uci(choice)
            board.make(choice)
            pc.push_uci(uci)
        if mismatches:
            break
    assert mismatches == 0, "\n".join(samples[:8])
    assert compared >= 200_000

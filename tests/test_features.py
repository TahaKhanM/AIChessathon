"""W02 canonical feature schema: row counts, pinned FullThreats parity,
compact pawn-pair map, early-fused accumulator and integer contract."""

from __future__ import annotations

import json
import random
from pathlib import Path

import chess
import numpy as np
import pytest

from engine import features as F
from engine.board import (
    BISHOP,
    BLACK,
    KING,
    KNIGHT,
    PAWN,
    QUEEN,
    ROOK,
    WHITE,
    Board,
)
from engine.movegen import generate_legal

ARCH_PATH = Path(__file__).resolve().parents[1] / "spec" / "RX_FINAL_PLAN" / "architecture.json"
ARCH = json.loads(ARCH_PATH.read_text())
SPEC = ARCH["features"]

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
_MBUF = [0] * 256


def _random_boards(n: int, seed: int = 20260911) -> list[Board]:
    """Random legal-playout positions plus pathological dense boards."""
    rng = random.Random(seed)
    boards: list[Board] = []
    board = Board.from_fen(START_FEN)
    ply = 0
    while len(boards) < n:
        cnt = generate_legal(board, _MBUF)
        if cnt == 0 or ply >= 160:
            board = Board.from_fen(START_FEN)
            ply = 0
            continue
        board.make(_MBUF[rng.randrange(cnt)])
        ply += 1
        boards.append(Board.from_fen(board.to_fen()))
    # Dense, promotion-heavy stress positions (legal placement, both kings).
    for fen in (
        "qq2k2q/8/8/3q4/q2q3q/8/8/QQ2K2Q w - - 0 1",
        "rnb1k2r/pp1p1ppp/4pn2/2b5/2B1P3/2N2N2/PP3PPP/R1B1K2R w KQkq - 0 8",
        "3k4/PPPPPPPP/8/8/8/8/pppppppp/3K4 w - - 0 1",
        "r1b1kb1r/1ppnpppp/p1n2n2/6B1/2B1P3/2N2N2/PPP2PPP/R2K3R w kq - 2 8",
        "4k3/8/1b2b3/8/3Q4/8/1B2B3/4K3 w - - 0 1",
    ):
        boards.append(Board.from_fen(fen))
    return boards


# ---------------------------------------------------------------------------
# Gate 1: exact row counts, asserted from the generated tables
# ---------------------------------------------------------------------------


def test_contract_constants_match_architecture_json() -> None:
    assert F.PSQ_ROWS == SPEC["psq"]["rows"] == 9216 == 12 * 2 * 6 * 64
    assert F.THREAT_ROWS == SPEC["threats"]["rows"] == 59808
    assert F.PP_ROWS == SPEC["pawn_pairs"]["rows"] == 1488
    assert F.CHANNELS == SPEC["channels"] == 512
    assert F.PERSPECTIVES == SPEC["perspectives"] == 2
    assert SPEC["shared_transform_tables"] is True
    assert F.PSQ_COEF_ABS_LIMIT == SPEC["psq"]["coefficient_abs_limit"] == 255
    assert F.THREAT_COEF_ABS_LIMIT == SPEC["threats"]["coefficient_abs_limit"] == 63
    assert F.PP_COEF_ABS_LIMIT == SPEC["pawn_pairs"]["coefficient_abs_limit"] == 31
    assert F.BIAS_ABS_LIMIT == SPEC["bias"]["coefficient_abs_limit"] == 2040
    assert F.PSQ_STORAGE_BITS == SPEC["psq"]["lossless_storage_bits"] == 9
    assert F.THREAT_STORAGE_BITS == SPEC["threats"]["lossless_storage_bits"] == 7
    assert F.PP_STORAGE_BITS == SPEC["pawn_pairs"]["lossless_storage_bits"] == 6
    assert F.BIAS_STORAGE_BITS == SPEC["bias"]["lossless_storage_bits"] == 16
    assert F.THREAT_MAX_ACTIVE == SPEC["threats"]["max_active_bound"] == 256
    assert F.PP_MAX_ACTIVE == SPEC["pawn_pairs"]["max_active_bound"] == 120
    assert F.ACCUMULATOR_ABS_BOUND == SPEC["accumulator"]["proven_abs_bound"] == 30048


def test_king_bucket_map_matches_json() -> None:
    want = SPEC["psq"]["king_bucket_map_by_rank_from_perspective_home_rank"]
    assert [list(r) for r in F.KING_BUCKET_MAP] == want
    # PlentyChess getKingBucket: bucket = LAYOUT[ksq ^ (56*color)].
    assert F.KING_BUCKET_LAYOUT[0] == 0 and F.KING_BUCKET_LAYOUT[7] == 0
    assert F.KING_BUCKET_LAYOUT[8] == 4
    assert F.KING_BUCKET_LAYOUT[11] == 7
    assert F.KING_BUCKET_LAYOUT[28] == 10
    assert F.king_bucket(WHITE, 4) == 3  # e1
    assert F.king_bucket(BLACK, 60) == 3  # e8
    assert F.king_bucket(WHITE, 27) == 10  # d4
    assert F.king_bucket(BLACK, 36) == 10  # e5 -> oriented e4 -> row 4
    for ksq in range(64):
        assert 0 <= F.king_bucket(WHITE, ksq) < 12
        assert F.king_bucket(BLACK, ksq) == F.king_bucket(WHITE, ksq ^ 56)


def test_psq_index_space_is_exactly_9216() -> None:
    # For a fixed (persp, ksq) the 12*64 inputs occupy exactly one 768 block.
    for persp, ksq in ((WHITE, 4), (BLACK, 51), (WHITE, 63), (BLACK, 7)):
        idxs = {F.psq_index(persp, ksq, p, s) for p in range(12) for s in range(64)}
        assert len(idxs) == 768
        base = F.king_bucket(persp, ksq) * 768
        assert idxs == set(range(base, base + 768))
        # all rows in bounds for every square/piece at this frame
    # Over all king squares the blocks tile the whole 9,216-row space.
    seen = set()
    for ksq in range(64):
        for p in range(12):
            for s in range(64):
                seen.add(F.psq_index(WHITE, ksq, p, s))
    assert seen == set(range(F.PSQ_ROWS))


# ---------------------------------------------------------------------------
# FullThreats parity: literal port of the pinned C++ on python-chess tables
# ---------------------------------------------------------------------------

_SF_NVT = (0, 4, 10, 8, 8, 10, 0, 0, 0, 4, 10, 8, 8, 10, 0, 0)
_SF_MAP = (
    (-1, 0, -1, 1, -1, -1),
    (0, 1, 2, 3, 4, -1),
    (0, 1, 2, 3, -1, -1),
    (0, 1, 2, 3, -1, -1),
    (0, 1, 2, 3, 4, -1),
    (-1, -1, -1, -1, -1, -1),
)


def _lit_pseudo(sf_piece: int, sq: int) -> int:
    """PseudoAttacks of the pinned source, rebuilt on python-chess tables."""
    pt = sf_piece & 7
    if pt == 1:
        # SF piece>>3: 0 = white, 1 = black; chess.WHITE is True.
        return int(chess.BB_PAWN_ATTACKS[chess.WHITE if sf_piece >> 3 == 0 else chess.BLACK][sq])
    if pt == 2:
        return int(chess.BB_KNIGHT_ATTACKS[sq])
    if pt == 6:
        return int(chess.BB_KING_ATTACKS[sq])
    dirs = ()
    if pt in (3, 5):
        dirs += ((1, 1), (1, -1), (-1, 1), (-1, -1))
    if pt in (4, 5):
        dirs += ((0, 1), (0, -1), (1, 0), (-1, 0))
    f, r = sq & 7, sq >> 3
    out = 0
    for df, dr in dirs:
        nf, nr = f + df, r + dr
        while 0 <= nf < 8 and 0 <= nr < 8:
            out |= 1 << (nr * 8 + nf)
            nf += df
            nr += dr
    return out


def _lit_tables():
    """Re-derive index_lut1/offsets/index_lut2 as plain dicts (fresh code path)."""
    offsets, cum_piece, cum = {}, {}, {}
    running = 0
    for p in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14):
        cpo = 0
        for frm in range(64):
            offsets[(p, frm)] = cpo
            if (p & 7) != 1 or 8 <= frm <= 55:
                cpo += bin(_lit_pseudo(p, frm)).count("1")
        cum_piece[p], cum[p] = cpo, running
        running += _SF_NVT[p] * cpo
    assert running == 59808

    lut1 = {}
    for a in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14):
        for d in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14):
            at, dt = a & 7, d & 7
            m = _SF_MAP[at - 1][dt - 1]
            enemy = (a ^ d) == 8
            semi = at == dt and (enemy or at != 1)
            base = cum[a] + ((d >> 3) * (_SF_NVT[a] // 2) + m) * cum_piece[a]
            lut1[(a, d, 0)] = 59808 if m < 0 else base
            lut1[(a, d, 1)] = 59808 if (m < 0 or semi) else base

    lut2 = {}
    for p in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14):
        for frm in range(64):
            atk = _lit_pseudo(p, frm)
            for to in range(64):
                lut2[(p, frm, to)] = bin(((1 << to) - 1) & atk).count("1")
    return lut1, offsets, lut2


_LIT_LUT1, _LIT_OFF, _LIT_LUT2 = _lit_tables()


def _lit_index(persp: int, ksq: int, attacker_sf: int, frm: int, to: int, attacked_sf: int) -> int:
    orient = (56 * persp) ^ (7 if (ksq & 4) else 0)
    fo, to_o = frm ^ orient, to ^ orient
    a = attacker_sf ^ (8 * persp)
    d = attacked_sf ^ (8 * persp)
    return _LIT_LUT1[(a, d, 1 if fo < to_o else 0)] + _LIT_OFF[(a, fo)] + _LIT_LUT2[(a, fo, to_o)]


def _chess_relations(cb: chess.Board) -> set[tuple[int, int, int, int]]:
    """All (attacker, from, to, attacked) relations the pinned code generates,
    enumerated independently via python-chess (SF piece numbering)."""
    out: set[tuple[int, int, int, int]] = set()
    for frm, piece in cb.piece_map().items():
        pt = piece.piece_type
        if pt == chess.KING:
            continue
        a_sf = pt + (0 if piece.color else 8)
        for to in cb.attacks(frm):
            target = cb.piece_at(to)
            if target is None:
                continue
            if _SF_MAP[pt - 1][target.piece_type - 1] < 0:
                continue  # excluded target class never reaches make_index
            d_sf = target.piece_type + (0 if target.color else 8)
            out.add((a_sf, frm, to, d_sf))
    return out


def _lit_active(cb: chess.Board, persp: int) -> set[int]:
    king_sq = cb.king(bool(1 - persp))
    return {
        idx
        for (a, frm, to, d) in _chess_relations(cb)
        if (idx := _lit_index(persp, king_sq, a, frm, to, d)) < 59808
    }


@pytest.mark.parametrize("persp", [WHITE, BLACK])
def test_threat_index_space_bounds_and_injectivity(persp: int) -> None:
    """Every valid oriented relation indexes inside [0, 59808), distinctly."""
    # Six king squares covering both mirror sides and several buckets.
    for ksq in (0, 4, 27, 33, 56, 63):
        seen: set[int] = set()
        for a in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14):
            # Pawn from-squares are restricted to ranks 2-7: the pinned code
            # only accumulates offsets there, and illegal origins alias live
            # slots (a1->b2 collides with a2->b3 by construction).
            for frm in range(64):
                if (a & 7) == 1 and not 8 <= frm <= 55:
                    continue
                atk = _lit_pseudo(a, frm)
                for to in range(64):
                    if not (atk >> to) & 1:
                        continue
                    for d in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14):
                        idx = _lit_index(persp, ksq, a, frm, to, d)
                        if idx < 59808:
                            assert idx not in seen  # the (a, class, from, to) code is injective
                            seen.add(idx)
        # no index may exceed the declared space
        assert max(seen) < 59808


def test_threat_parity_with_literal_port() -> None:
    """Our tabled encoder must emit exactly the literal port's index sets."""
    boards = _random_boards(60)
    boards.insert(0, Board.from_fen(START_FEN))
    for board in boards:
        cb = chess.Board(board.to_fen())
        for persp in (WHITE, BLACK):
            ours = set(int(i) for i in F.active_threat_indices(board, persp))
            assert ours == _lit_active(cb, persp), board.to_fen()


def test_threat_semantics_exclusions() -> None:
    """Pinned exclusions: pawn->pawn dead, kings never appear, mutual
    same-type relations emitted exactly once, same-side defence present."""
    # WP e4 "attacks" BP d5 (pawn->pawn excluded) and BN f5 (pawn->N kept).
    b = Board.from_fen("4k3/8/8/3p1n2/4P3/8/8/4K3 w - - 0 1")
    thr = F.active_threat_indices(b, WHITE)
    assert len(thr) == 1
    idx = F.threat_index(WHITE, 4, PAWN, 28, 37, 6 + KNIGHT)  # Pe4 x Nf5
    assert idx == thr[0] and idx < F.THREAT_ROWS
    assert F.threat_index(WHITE, 4, PAWN, 28, 35, 6 + PAWN) >= F.THREAT_ROWS  # Pe4 x Pd5

    # Target-class exclusions at index level: pawns only target N/R, kings
    # are never attackers or targets.
    for dead in (PAWN, BISHOP, QUEEN, KING):
        assert F.threat_index(WHITE, 4, PAWN, 28, 35, 6 + dead) >= F.THREAT_ROWS
    for tgt in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING):
        assert F.threat_index(WHITE, 4, KING, 4, 11, 6 + tgt) >= F.THREAT_ROWS
    assert F.threat_index(WHITE, 4, QUEEN, 27, 60, 6 + KING) >= F.THREAT_ROWS

    # No relation at all: Nc3 and Nh1 attack nothing occupied.
    b2 = Board.from_fen("4k3/8/8/8/8/2N5/8/4K2N w - - 0 1")
    assert len(F.active_threat_indices(b2, WHITE)) == 0
    # Mutual same-type pairs (enemy and friendly) are emitted exactly once,
    # the oriented from>to direction only.
    for b3 in (
        Board.from_fen("4k3/8/8/8/4n3/2N5/8/4K3 w - - 0 1"),  # Nc3 x ne4
        Board.from_fen("4k3/8/8/8/4N3/2N5/8/4K3 w - - 0 1"),  # Nc3 x Ne4
    ):
        thr3 = F.active_threat_indices(b3, WHITE)
        assert len(thr3) == 1
    fwd = F.threat_index(WHITE, 4, KNIGHT, 18, 28, KNIGHT)  # c3->e4: fo<to -> excluded
    rev = F.threat_index(WHITE, 4, KNIGHT, 28, 18, KNIGHT)
    assert fwd >= F.THREAT_ROWS
    assert (
        rev
        == F.active_threat_indices(Board.from_fen("4k3/8/8/8/4N3/2N5/8/4K3 w - - 0 1"), WHITE)[0]
    )

    # Friendly different-type defence is a live feature: Nc3 defends Bb5.
    b4 = Board.from_fen("4k3/8/8/1B6/8/2N5/8/4K3 w - - 0 1")
    assert len(F.active_threat_indices(b4, WHITE)) == 1
    assert F.threat_index(WHITE, 4, KNIGHT, 18, 33, BISHOP) == F.active_threat_indices(b4, WHITE)[0]

    # Pawn attacks a king: dead (king is never a target), and the king
    # cannot generate features either -> zero rows.
    b5 = Board.from_fen("4k3/8/8/2k5/1P6/8/8/4K3 w - - 0 1")  # Pb4 "attacks" kc5
    assert len(F.active_threat_indices(b5, WHITE)) == 0
    # Pawn attacks a rook: live; the rook cannot attack the pawn back.
    b6 = Board.from_fen("4k3/8/8/8/1r6/2P5/8/4K3 w - - 0 1")  # c3 x b4 rook
    assert len(F.active_threat_indices(b6, WHITE)) == 1


def test_threat_active_bounds() -> None:
    for board in _random_boards(200):
        for persp in (WHITE, BLACK):
            t = F.active_threat_indices(board, persp)
            assert len(t) <= F.THREAT_MAX_ACTIVE
            assert len(set(t.tolist())) == len(t)  # injective
            if len(t):
                assert t.min() >= 0 and t.max() < F.THREAT_ROWS


# ---------------------------------------------------------------------------
# Compact pawn pairs
# ---------------------------------------------------------------------------


def test_pp_map_matches_spec_enumeration() -> None:
    identities = [(c, rank, file) for c in range(2) for rank in range(1, 7) for file in range(8)]
    valid = []
    for b, (_, rb, fb) in enumerate(identities):
        for a in range(b):
            _, ra, fa = identities[a]
            if (ra, fa) != (rb, fb) and abs(fa - fb) <= 1:
                valid.append(b * (b - 1) // 2 + a)
    assert len(valid) == len(set(valid)) == 1488 == F.PP_ROWS
    assert valid == sorted(valid)
    dense = {int(F.PP_OLD_TO_NEW[t]) for t in valid}
    assert dense == set(range(F.PP_ROWS))
    assert set(int(x) for x in F.PP_NEW_TO_OLD) == set(valid)


def test_pp_semantics() -> None:
    # Same-file and adjacent-file pairs live; distance-2 dead.
    b = Board.from_fen("4k3/8/8/8/8/1P6/1P6/4K3 w - - 0 1")  # Pb2, Pb3
    assert len(F.active_pawn_pair_indices(b, WHITE)) == 1
    b2 = Board.from_fen("4k3/8/8/8/8/8/1P1P4/4K3 w - - 0 1")  # Pb2, Pd2
    assert len(F.active_pawn_pair_indices(b2, WHITE)) == 0
    # Identity is relative-colour: same physical pair differs by perspective.
    b3 = Board.from_fen("4k3/8/8/8/8/1p6/1P6/4K3 w - - 0 1")  # Pb2 own, pb3 enemy (white)
    w = F.active_pawn_pair_indices(b3, WHITE)
    bl = F.active_pawn_pair_indices(b3, BLACK)
    assert len(w) == len(bl) == 1 and w[0] != bl[0]
    for board in _random_boards(120):
        for persp in (WHITE, BLACK):
            pp = F.active_pawn_pair_indices(board, persp)
            assert len(pp) <= F.PP_MAX_ACTIVE
            assert len(set(pp.tolist())) == len(pp)


# ---------------------------------------------------------------------------
# Perspective symmetry: rotated + colour-swapped twin board
# ---------------------------------------------------------------------------


def _rotated_twin(board: Board) -> Board:
    twin = Board()
    for s in range(64):
        pc = board._sq[s]
        if pc >= 0:
            twin._place(s ^ 63, (pc + 6) % 12)
    twin._king[WHITE] = board._king[BLACK] ^ 63
    twin._king[BLACK] = board._king[WHITE] ^ 63
    twin.side = board.side ^ 1
    twin.key = twin.compute_key()
    return twin


def test_mirror_consistency() -> None:
    for board in _random_boards(120):
        twin = _rotated_twin(board)
        for a, b in ((WHITE, BLACK), (BLACK, WHITE)):
            fa = F.active_features(board, a)
            fb = F.active_features(twin, b)
            assert sorted(fa.psq.tolist()) == sorted(fb.psq.tolist())
            assert sorted(fa.thr.tolist()) == sorted(fb.thr.tolist())
            assert sorted(fa.pp.tolist()) == sorted(fb.pp.tolist())


# ---------------------------------------------------------------------------
# Accumulator + integer contract (gate 4)
# ---------------------------------------------------------------------------


def _bounded_tables(seed: int = 1, scale: float = 1.0) -> F.FeatureTables:
    rng = np.random.default_rng(seed)
    return F.FeatureTables(
        bias=rng.integers(-F.BIAS_ABS_LIMIT, F.BIAS_ABS_LIMIT + 1, F.CHANNELS).astype(np.int16),
        psq=rng.integers(-255, 256, (F.PSQ_ROWS, F.CHANNELS)).astype(np.int16),
        thr=rng.integers(-63, 64, (F.THREAT_ROWS, F.CHANNELS)).astype(np.int8),
        pp=rng.integers(-31, 32, (F.PP_ROWS, F.CHANNELS)).astype(np.int8),
    )


def test_accumulator_bound_proof() -> None:
    # Gate 4 literal arithmetic.
    assert 32 * 255 + 256 * 63 + 120 * 31 + 2040 == 30048 < 32767
    assert F.ACCUMULATOR_ABS_BOUND == 30048
    # Saturating tables at the declared limits: a maximal legal position's
    # accumulator stays inside the proven envelope and inside int16.
    tables = F.FeatureTables(
        bias=np.full(F.CHANNELS, 2040, np.int16),
        psq=np.full((F.PSQ_ROWS, F.CHANNELS), 255, np.int16),
        thr=np.full((F.THREAT_ROWS, F.CHANNELS), 63, np.int8),
        pp=np.full((F.PP_ROWS, F.CHANNELS), 31, np.int8),
    )
    dense = Board.from_fen("qq2k2q/8/8/3q4/q2q3q/8/8/QQ2K2Q w - - 0 1")
    acc = F.Accumulator.refresh(tables, dense)
    peak = int(np.abs(acc.state.astype(np.int64)).max())
    assert peak <= 30048


def test_accumulator_delta_parity_and_intermediate_safety() -> None:
    """Delta path must equal full refresh, with every intermediate state
    inside the int16 envelope (checked inside apply) and boundary states
    inside the proven 30,048 bound."""
    tables = _bounded_tables()
    rng = random.Random(3)
    for board in _random_boards(80):
        frame = {p: F.frame_signature(p, board._king[p]) for p in (WHITE, BLACK)}
        before = {p: F.active_features(board, p) for p in (WHITE, BLACK)}
        acc = F.Accumulator.refresh(tables, board)
        cnt = generate_legal(board, _MBUF)
        if cnt == 0:
            continue
        board.make(_MBUF[rng.randrange(cnt)])
        for p in (WHITE, BLACK):
            if frame[p] != F.frame_signature(p, board._king[p]):
                continue  # king frame changed: refresh path, not delta
            after = F.active_features(board, p)
            removed, added = F.feature_delta(before[p], after)
            acc.apply(tables, p, removed, added)
            expected = F.Accumulator._compute(tables, after)
            np.testing.assert_array_equal(acc.state[p], expected)
            assert np.abs(acc.state[p].astype(np.int64)).max() <= 30048


def test_accumulator_intermediate_remove_before_add() -> None:
    """An add-first ordering could transiently leave the proven envelope;
    removes-first plus int32 intermediates is the canonical safe protocol."""
    tables = F.FeatureTables(
        bias=np.full(F.CHANNELS, 2040, np.int16),
        psq=np.full((F.PSQ_ROWS, F.CHANNELS), 255, np.int16),
        thr=np.full((F.THREAT_ROWS, F.CHANNELS), 63, np.int8),
        pp=np.full((F.PP_ROWS, F.CHANNELS), 31, np.int8),
    )
    dense = Board.from_fen("qq2k2q/8/8/3q4/q2q3q/8/8/QQ2K2Q w - - 0 1")
    acc = F.Accumulator.refresh(tables, dense)
    start = int(acc.state[WHITE][0])
    # Swap one active PSQ row for another: remove-then-add returns to the
    # envelope at every step; add-then-remove would need +255 headroom.
    feats = F.active_psq_indices(dense, WHITE)
    rem = F.FeatureTriple(np.array([feats[0]]), np.array([], np.int32), np.array([], np.int32))
    add = F.FeatureTriple(np.array([feats[0] ^ 1]), np.array([], np.int32), np.array([], np.int32))
    acc.apply(tables, WHITE, rem, add)
    assert abs(int(acc.state[WHITE][0])) <= 30048
    assert int(acc.state[WHITE][0]) == start - 255 + 255


def test_dirty_capacity_and_bound_enforcement() -> None:
    tables = _bounded_tables()
    acc = F.Accumulator.refresh(tables, Board.from_fen(START_FEN))
    too_many = F.FeatureTriple(
        np.arange(33, dtype=np.int32), np.array([], np.int32), np.array([], np.int32)
    )
    with pytest.raises(F.FeatureContractError):
        acc.apply(tables, WHITE, too_many, F.FeatureTriple(*[np.array([], np.int32)] * 3))
    # Post-fold enforcement: folded factors may exceed limits only if the
    # folded result does not.
    base = np.full((4, 4), 200, np.int16)
    ok = F.fold_factorized([base, np.full((4, 4), 55, np.int16)], 255, np.int16)
    assert ok.max() == 255
    with pytest.raises(F.FeatureContractError):
        F.fold_factorized([base, np.full((4, 4), 56, np.int16)], 255, np.int16)
    # And loader-side validation rejects out-of-bound decoded tables.
    bad = F.FeatureTables(
        bias=np.zeros(F.CHANNELS, np.int16),
        psq=np.full((F.PSQ_ROWS, F.CHANNELS), 256, np.int16),
        thr=np.zeros((F.THREAT_ROWS, F.CHANNELS), np.int8),
        pp=np.zeros((F.PP_ROWS, F.CHANNELS), np.int8),
    )
    with pytest.raises(F.FeatureContractError):
        F.validate_feature_tables(bad)


def test_early_fusion_shared_channels() -> None:
    """All three row sets land in the SAME 512 channels before activation."""
    tables = _bounded_tables()
    board = Board.from_fen(START_FEN)
    feats = F.active_features(board, WHITE)
    acc = F.Accumulator._compute(tables, feats)
    manual = tables.bias.astype(np.int64).copy()
    manual += tables.psq[feats.psq].sum(axis=0, dtype=np.int64)
    manual += tables.thr[feats.thr].sum(axis=0, dtype=np.int64)
    manual += tables.pp[feats.pp].sum(axis=0, dtype=np.int64)
    np.testing.assert_array_equal(acc.astype(np.int64), manual)


def test_head_forward_mechanics() -> None:
    """Scalar reference: activation range, stack selection, PSQT, rescale."""
    rng = np.random.default_rng(9)
    model = {
        "bias": np.zeros(F.CHANNELS, np.int16),
        "psq": np.zeros((F.PSQ_ROWS, F.CHANNELS), np.int16),
        "thr": np.zeros((F.THREAT_ROWS, F.CHANNELS), np.int8),
        "pp": np.zeros((F.PP_ROWS, F.CHANNELS), np.int8),
        "head_w1": rng.integers(-127, 128, (8, 512, 16)).astype(np.int8),
        "head_b1": rng.integers(-4096, 4096, (8, 16)).astype(np.int32),
        "head_w2": rng.integers(-127, 128, (8, 32, 32)).astype(np.int8),
        "head_b2": rng.integers(-4096, 4096, (8, 32)).astype(np.int32),
        "head_w3": rng.integers(-127, 128, (8, 96, 4)).astype(np.int8),
        "head_b3": rng.integers(-4096, 4096, (8, 4)).astype(np.int32),
        "psqt_w": rng.integers(-512, 512, (9216, 8)).astype(np.int16),
        "psqt_b": rng.integers(-4096, 4096, (8,)).astype(np.int32),
        "_meta": {"scale_num": 3, "scale_shift": 1, "neural_bound": 10_000_000},
    }
    board = Board.from_fen("r1bqk2r/pp1n1pbp/2p1p1p1/4P3/2pP3N/2N1B1P1/PP2QPBP/R3K2R w KQkq - 4 12")
    tables = F.FeatureTables(model["bias"], model["psq"], model["thr"], model["pp"])
    acc = F.Accumulator.refresh(tables, board)
    v = F.paired_activations(acc.state, WHITE)
    assert v.shape == (512,) and v.min() >= 0 and v.max() <= 127
    pc = bin(board._occ_all).count("1")
    psq_rows = (F.active_psq_indices(board, WHITE), F.active_psq_indices(board, BLACK))
    res = F.head_forward(model, acc.state, WHITE, pc, psq_rows)
    assert res["stack"] == min(7, max(0, (pc - 2) // 4))
    assert res["scalar"] == int(((int(res["raw"][0]) + res["psqt"]) * 3) >> 1)
    # Deployed leaf clamp: absent neural_bound defaults to LEAF_NEURAL_BOUND.
    sat = {**model, "_meta": {"scale_num": 10_000, "scale_shift": 0}}
    res_sat = F.head_forward(sat, acc.state, WHITE, pc, psq_rows)
    assert abs(res_sat["scalar"]) == F.LEAF_NEURAL_BOUND
    assert res["wdl"].shape == (3,)
    # Dense-head MAC count = the contract's 9312.
    macs = ARCH["reference_raw_accounting"]["scalar_dense_head_macs"]
    assert 512 * 16 + 32 * 32 + 96 == 9312 == macs
    # Zero model -> zero scalar.
    zero = {k: (np.zeros_like(vv) if isinstance(vv, np.ndarray) else vv) for k, vv in model.items()}
    zero["_meta"] = {}
    assert F.evaluate_position(zero, board) == 0

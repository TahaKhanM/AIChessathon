"""Canonical PSQ/threat/pawn-pair encoders and deltas.

F512-EF-K12 reference: 9,216 PSQ rows, 59,808 FullThreats rows, 1,488
compact pawn-pair rows, non-uniform K12 king buckets, two perspectives.
Demand-driven materialization; delta replay / refresh-cache / full refresh
chosen by measured cost. See docs/architecture.md section 4.

This module is the ONE canonical feature schema shared by data extraction,
training, the scalar integer reference and the Numba export (spec 10.1).
All index maps are generated here from the pinned definitions:

- PSQ rows: 12 non-uniform king buckets x 2 relative colours x 6 piece
  types x 64 oriented squares. Bucket map adopted from the inspected
  PlentyChess@04e07a98 KING_BUCKET_LAYOUT; index layout follows its
  ``768*bucket + 384*relativeColour + 64*pieceType + square`` order.
- Threat rows: Stockfish@59aae690 ``src/nnue/features/full_threats.{h,cpp}``
  exactly -- selected occupied-target relations including same-side
  protection, its target exclusions, semi-exclusion of mutual same-type
  relations, and orientation. Pawns only target N/R; kings are never
  attackers or targets; mutual N-N/B-B/R-R/Q-Q and enemy same-type
  relations are emitted once (oriented ``from > to`` only).
- Pawn pairs: 96 relative-colour pawn identities on ranks 2-7, unordered
  pairs on distinct physical squares with file distance <= 1 (the live
  inputs of upstream PP_3Wide), densely remapped to 1,488 rows ordered by
  ``b*(b-1)//2 + a``. Not claimed identical to every upstream PP encoding.

Perspective normalisation, identical for all three sets: squares are
rank-flipped for the black perspective (``sq ^ 56``), then file-mirrored
when the perspective's own king sits on files e-h (``sq ^ 7``). Piece
colours are XORed by the perspective, so relative colour 0 is always "own".

The int16 accumulator contract (spec 4.4) is enforced here: coefficient
bounds are checked on *folded* tables, updates remove before adding with
widened int32 intermediates, and every intermediate state is asserted
against the proven 30,048 envelope / int16 storage bound.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from engine.board import (
    BISHOP,
    BLACK,
    KING_ATK,
    KNIGHT,
    KNIGHT_ATK,
    MASK64,
    PAWN,
    PAWN_ATK,
    QUEEN,
    ROOK,
    WHITE,
    Board,
)
from engine.movegen import BISHOP_NEG, BISHOP_POS, ROOK_NEG, ROOK_POS, bishop_attacks, rook_attacks

# ---------------------------------------------------------------------------
# Canonical contract constants (spec/RX_FINAL_PLAN/architecture.json is
# authoritative; tests assert these literals against the JSON).
# ---------------------------------------------------------------------------

PERSPECTIVES = 2
CHANNELS = 512

KING_BUCKETS = 12
PSQ_ROWS = 12 * 2 * 6 * 64  # 9216
THREAT_ROWS = 59808
PP_ROWS = 1488

PSQ_RUNTIME_DTYPE = np.int16
THREAT_RUNTIME_DTYPE = np.int8
PP_RUNTIME_DTYPE = np.int8
BIAS_RUNTIME_DTYPE = np.int16

PSQ_COEF_ABS_LIMIT = 255
THREAT_COEF_ABS_LIMIT = 63
PP_COEF_ABS_LIMIT = 31
BIAS_ABS_LIMIT = 2040

PSQ_STORAGE_BITS = 9
THREAT_STORAGE_BITS = 7
PP_STORAGE_BITS = 6
BIAS_STORAGE_BITS = 16

PSQ_MAX_ACTIVE = 32  # standard legal chess: at most 32 pieces on the board
THREAT_MAX_ACTIVE = 256  # conservative bound; <= 8 occupied targets per attacker
PP_MAX_ACTIVE = 120  # C(16, 2) pairs of at most 16 pawns

ACCUMULATOR_ABS_BOUND = (
    32 * PSQ_COEF_ABS_LIMIT
    + THREAT_MAX_ACTIVE * THREAT_COEF_ABS_LIMIT
    + PP_MAX_ACTIVE * PP_COEF_ABS_LIMIT
    + BIAS_ABS_LIMIT
)
INT16_MAX = 32767

# Paired transform / head numeric contract (spec 4.2, numeric_contract).
FT_OPERAND_CLIP = (0, 255)
PAIRED_PRODUCT_SHIFT = 9
PAIRED_PRODUCT_MAX = 127
HIDDEN_CLIP = (0, 127)
HIDDEN_AFFINE_SHIFT = 6
SQUARE_ACTIVATION_SHIFT = 7

# Deployed leaf transform (numeric_contract.leaf_transform /
# leaf_transform_export_defaults): cp = clip(raw * LEAF_SCALE_NUM >>
# LEAF_SCALE_SHIFT, -LEAF_NEURAL_BOUND, +LEAF_NEURAL_BOUND) in int64.
# LEAF_NEURAL_BOUND = MATE_IN_MAX - 1 = 27951, strictly inside the mate
# band [27952, 30000).  These are the canonical export defaults; a trained
# export may override them only via RXF1 _meta, versioned with the weights.
LEAF_SCALE_NUM = 275
LEAF_SCALE_SHIFT = 8
LEAF_NEURAL_BOUND = 27951
HEAD_STACKS = 8
HEAD_L1 = 16
HEAD_L2 = 32
HEAD_OUT = 4
HEAD_INPUTS = 512

# K12 non-uniform king buckets, indexed by rank from the perspective's home
# rank then file. Rows are file-symmetric, so the horizontal mirror does not
# change the bucket. Equals architecture.json
# features.psq.king_bucket_map_by_rank_from_perspective_home_rank.
KING_BUCKET_MAP = (
    (0, 1, 2, 3, 3, 2, 1, 0),
    (4, 5, 6, 7, 7, 6, 5, 4),
    (8, 8, 9, 9, 9, 9, 8, 8),
    (10, 10, 10, 10, 10, 10, 10, 10),
    (11, 11, 11, 11, 11, 11, 11, 11),
    (11, 11, 11, 11, 11, 11, 11, 11),
    (11, 11, 11, 11, 11, 11, 11, 11),
    (11, 11, 11, 11, 11, 11, 11, 11),
)

# Flat 64-entry layout indexed by ``ksq ^ (56 * perspective)``, matching the
# inspected PlentyChess getKingBucket.
KING_BUCKET_LAYOUT = tuple(KING_BUCKET_MAP[r][f] for r in range(8) for f in range(8))


class FeatureContractError(ValueError):
    """Raised when the canonical integer/index contract is violated."""


# ---------------------------------------------------------------------------
# Perspective normalisation and piece-square features
# ---------------------------------------------------------------------------


def orientation(persp: int, ksq: int) -> int:
    """Square XOR mask for `persp` whose own king sits on `ksq`.

    Rank flip for black (56) plus a file mirror (7) when the king is on
    files e-h, i.e. OrientTBL[ksq] ^ (56 * perspective) from the pinned
    FullThreats / PlentyChess sources.
    """
    return (56 * persp) ^ (7 if ksq & 4 else 0)


def king_bucket(persp: int, ksq: int) -> int:
    """K12 bucket of the perspective's own king (0..11)."""
    return KING_BUCKET_LAYOUT[ksq ^ (56 * persp)]


def frame_signature(persp: int, ksq: int) -> tuple[int, int]:
    """(bucket, orientation): delta updates are only valid while this is fixed."""
    return king_bucket(persp, ksq), orientation(persp, ksq)


def psq_index(persp: int, ksq: int, piece: int, sq: int) -> int:
    """PSQ row for `piece` (0-11, colour*6+type) on `sq`, seen from `persp`."""
    osq = sq ^ orientation(persp, ksq)
    rel = (piece // 6) ^ persp
    return king_bucket(persp, ksq) * 768 + rel * 384 + (piece % 6) * 64 + osq


# ---------------------------------------------------------------------------
# FullThreats: pinned Stockfish@59aae690 definition
# ---------------------------------------------------------------------------


# Stockfish piece codes: W_PAWN=1..W_KING=6, B_PAWN=9..B_KING=14.
def _to_sf(piece: int) -> int:
    return (piece % 6) + 1 + (piece // 6) * 8


def _from_sf(sf: int) -> int:
    return (sf & 7) - 1 + (sf >> 3) * 6


_SF_PIECES = (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14)  # AllPieces order

# numValidTargets[PIECE_NB] from full_threats.h.
_NUM_VALID_TARGETS = (0, 4, 10, 8, 8, 10, 0, 0, 0, 4, 10, 8, 8, 10, 0, 0)

# FullThreats::map[attackerType-1][attackedType-1] from full_threats.h.
_INTERACTION_MAP = (
    (-1, 0, -1, 1, -1, -1),
    (0, 1, 2, 3, 4, -1),
    (0, 1, 2, 3, -1, -1),
    (0, 1, 2, 3, -1, -1),
    (0, 1, 2, 3, 4, -1),
    (-1, -1, -1, -1, -1, -1),
)


def _pseudo_attacks(sf_piece: int, sq: int) -> int:
    """Unblocked attack set, i.e. Attacks::PseudoAttacks of the pinned source."""
    pt = sf_piece & 7
    if pt == 1:
        return PAWN_ATK[sf_piece >> 3][sq]
    if pt == 2:
        return KNIGHT_ATK[sq]
    if pt == 3:
        return BISHOP_POS[sq][0] | BISHOP_POS[sq][1] | BISHOP_NEG[sq][0] | BISHOP_NEG[sq][1]
    if pt == 4:
        return ROOK_POS[sq][0] | ROOK_POS[sq][1] | ROOK_NEG[sq][0] | ROOK_NEG[sq][1]
    if pt == 5:
        return _pseudo_attacks(3, sq) | _pseudo_attacks(4, sq)
    if pt == 6:
        return KING_ATK[sq]
    return 0


def _build_threat_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (index_lut1, offsets, index_lut2) exactly as the pinned C++."""
    lut1 = np.zeros((16, 16, 2), dtype=np.int32)
    offsets = np.zeros((16, 64), dtype=np.int32)
    lut2 = np.zeros((16, 64, 64), dtype=np.int32)

    cum_piece = [0] * 16
    cum = [0] * 16
    cumulative = 0
    for p in _SF_PIECES:
        cpo = 0
        for frm in range(64):
            offsets[p][frm] = cpo
            if (p & 7) != 1 or 8 <= frm <= 55:
                cpo += bin(_pseudo_attacks(p, frm)).count("1")
        cum_piece[p] = cpo
        cum[p] = cumulative
        cumulative += _NUM_VALID_TARGETS[p] * cpo
    if cumulative != THREAT_ROWS:
        raise FeatureContractError(f"threat space is {cumulative}, expected {THREAT_ROWS}")

    for a in _SF_PIECES:
        for d in _SF_PIECES:
            at, dt = a & 7, d & 7
            m = _INTERACTION_MAP[at - 1][dt - 1]
            enemy = (a ^ d) == 8  # same type, opposite colour
            semi = at == dt and (enemy or at != 1)
            base = cum[a] + ((d >> 3) * (_NUM_VALID_TARGETS[a] // 2) + m) * cum_piece[a]
            lut1[a][d][0] = THREAT_ROWS if m < 0 else base
            lut1[a][d][1] = THREAT_ROWS if (m < 0 or semi) else base

    for p in _SF_PIECES:
        for frm in range(64):
            attacks = _pseudo_attacks(p, frm)
            for to in range(64):
                lut2[p][frm][to] = bin(((1 << to) - 1) & attacks).count("1")

    return lut1, offsets, lut2


_THREAT_LUT1, _THREAT_OFFSETS, _THREAT_LUT2 = _build_threat_tables()


def threat_index(persp: int, ksq: int, attacker: int, frm: int, to: int, attacked: int) -> int:
    """FullThreats row for (attacker@frm) -> (attacked@to) from `persp`.

    `attacker`/`attacked` use this engine's 0-11 piece encoding. Returns a
    value >= THREAT_ROWS when the relation is excluded by the pinned
    definition (invalid target type or semi-excluded orientation); callers
    must filter with ``index < THREAT_ROWS``. `to` must be a genuine attack
    target of `attacker` on `frm` -- the index is only meaningful then.
    """
    o = orientation(persp, ksq)
    fo = frm ^ o
    too = to ^ o
    swap = 8 * persp
    a = _to_sf(attacker) ^ swap
    d = _to_sf(attacked) ^ swap
    base = _THREAT_LUT1[a, d, 1 if fo < too else 0]
    return int(base + _THREAT_OFFSETS[a, fo] + _THREAT_LUT2[a, fo, too])


def _pop_lsb_iter(bb: int):
    while bb:
        sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        yield sq


def active_threat_indices(board: Board, persp: int) -> np.ndarray:
    """Active FullThreats rows for `persp`, in the pinned generation order."""
    ksq = board._king[persp]
    occ = board._occ_all
    bb = board._bb
    sq = board._sq
    pawn_t = bb[KNIGHT] | bb[ROOK] | bb[6 + KNIGHT] | bb[6 + ROOK]
    minor_t = pawn_t | bb[PAWN] | bb[6 + PAWN] | bb[BISHOP] | bb[6 + BISHOP]
    queen_t = minor_t | bb[QUEEN] | bb[6 + QUEEN]

    out: list[int] = []
    file_a = 0x0101010101010101
    file_h = file_a << 7
    # Pawn attacks are enumerated by shifted occupancy, pinned order:
    # WHITE +9 (NE, mask ~FileH), WHITE +7 (NW, ~FileA),
    # BLACK -9 (SW, ~FileA), BLACK -7 (SE, ~FileH).
    for attacker, delta, mask in (
        (WHITE * 6 + PAWN, 9, ~file_h & MASK64),
        (WHITE * 6 + PAWN, 7, ~file_a & MASK64),
        (BLACK * 6 + PAWN, -9, ~file_a & MASK64),
        (BLACK * 6 + PAWN, -7, ~file_h & MASK64),
    ):
        src = bb[attacker] & mask
        attacks = ((src << delta) if delta > 0 else (src >> -delta)) & MASK64 & pawn_t
        for to in _pop_lsb_iter(attacks):
            frm = to - delta
            idx = threat_index(persp, ksq, attacker, frm, to, sq[to])
            if idx < THREAT_ROWS:
                out.append(idx)

    for c in (WHITE, BLACK):
        for pt, targets in (
            (KNIGHT, queen_t),
            (BISHOP, minor_t),
            (ROOK, minor_t),
            (QUEEN, queen_t),
        ):
            pieces = bb[c * 6 + pt]
            for frm in _pop_lsb_iter(pieces):
                if pt == KNIGHT:
                    attacks = KNIGHT_ATK[frm]
                elif pt == BISHOP:
                    attacks = bishop_attacks(frm, occ)
                elif pt == ROOK:
                    attacks = rook_attacks(frm, occ)
                else:
                    attacks = bishop_attacks(frm, occ) | rook_attacks(frm, occ)
                for to in _pop_lsb_iter(attacks & targets):
                    idx = threat_index(persp, ksq, c * 6 + pt, frm, to, sq[to])
                    if idx < THREAT_ROWS:
                        out.append(idx)
    return np.asarray(out, dtype=np.int32)


# ---------------------------------------------------------------------------
# Compact pawn pairs: 96 identities, 1,488 dense rows
# ---------------------------------------------------------------------------

PP_ALL_PAIRS = 96 * 95 // 2  # 4560 triangular slots before the dense remap

# Dense map ordered by b*(b-1)//2 + a over identity pairs on distinct
# physical squares with file distance <= 1. Identity = relcolour*48 +
# (oriented square - 8); the square is ranks 2-7 by construction.
PP_OLD_TO_NEW = np.full(PP_ALL_PAIRS, -1, dtype=np.int32)
_pp_new: list[int] = []
for _b in range(1, 96):
    _sb = _b % 48 + 8
    for _a in range(_b):
        _sa = _a % 48 + 8
        if _sa != _sb and abs((_sa & 7) - (_sb & 7)) <= 1:
            PP_OLD_TO_NEW[_b * (_b - 1) // 2 + _a] = len(_pp_new)
            _pp_new.append(_b * (_b - 1) // 2 + _a)
PP_NEW_TO_OLD = np.asarray(_pp_new, dtype=np.int32)
if len(_pp_new) != PP_ROWS:
    raise FeatureContractError(f"pawn-pair space is {len(_pp_new)}, expected {PP_ROWS}")


def pawn_identity(relcolor: int, oriented_sq: int) -> int:
    """Pawn identity 0..95; `oriented_sq` must be on ranks 2-7 (8..55)."""
    return relcolor * 48 + oriented_sq - 8


def pawn_pair_index(persp: int, ksq: int, pc1: int, sq1: int, pc2: int, sq2: int) -> int:
    """Dense pawn-pair row for the unordered pair {(pc1,sq1),(pc2,sq2)}.

    Pieces use the 0-11 encoding; colour is taken relative to `persp`.
    Returns -1 when the oriented pair is not a live input (off the pawn
    ranks, same square, or file distance > 1).
    """
    o = orientation(persp, ksq)
    a = sq1 ^ o
    b = sq2 ^ o
    if not (8 <= a < 56 and 8 <= b < 56) or a == b or abs((a & 7) - (b & 7)) > 1:
        return -1
    ida = pawn_identity((pc1 // 6) ^ persp, a)
    idb = pawn_identity((pc2 // 6) ^ persp, b)
    hi, lo = (ida, idb) if ida > idb else (idb, ida)
    return int(PP_OLD_TO_NEW[hi * (hi - 1) // 2 + lo])


def active_pawn_pair_indices(board: Board, persp: int) -> np.ndarray:
    """Dense pawn-pair rows over all unordered pairs of board pawns."""
    ksq = board._king[persp]
    sq = board._sq
    pawns = [(pc, s) for s in range(64) if (pc := sq[s]) >= 0 and pc % 6 == PAWN]
    out: list[int] = []
    for i in range(len(pawns)):
        pc_i, sq_i = pawns[i]
        for j in range(i + 1, len(pawns)):
            pc_j, sq_j = pawns[j]
            idx = pawn_pair_index(persp, ksq, pc_i, sq_i, pc_j, sq_j)
            if idx >= 0:
                out.append(idx)
    return np.asarray(out, dtype=np.int32)


# ---------------------------------------------------------------------------
# Combined active features / accumulator
# ---------------------------------------------------------------------------


def active_psq_indices(board: Board, persp: int) -> np.ndarray:
    """One PSQ row per on-board piece, including both kings."""
    ksq = board._king[persp]
    sq = board._sq
    out = [psq_index(persp, ksq, pc, s) for s in range(64) if (pc := sq[s]) >= 0]
    return np.asarray(out, dtype=np.int32)


class FeatureTriple(NamedTuple):
    """Active feature rows of one perspective: (psq, threat, pawn_pair)."""

    psq: np.ndarray
    thr: np.ndarray
    pp: np.ndarray


def active_features(board: Board, persp: int) -> FeatureTriple:
    return FeatureTriple(
        active_psq_indices(board, persp),
        active_threat_indices(board, persp),
        active_pawn_pair_indices(board, persp),
    )


def encode(board: Board) -> tuple[FeatureTriple, FeatureTriple]:
    """Both perspectives' active features: index 0 = white, 1 = black."""
    return active_features(board, WHITE), active_features(board, BLACK)


def feature_delta(
    before: FeatureTriple, after: FeatureTriple
) -> tuple[FeatureTriple, FeatureTriple]:
    """(removed, added) rows between two snapshots of the same frame."""
    return (
        FeatureTriple(
            np.setdiff1d(before.psq, after.psq),
            np.setdiff1d(before.thr, after.thr),
            np.setdiff1d(before.pp, after.pp),
        ),
        FeatureTriple(
            np.setdiff1d(after.psq, before.psq),
            np.setdiff1d(after.thr, before.thr),
            np.setdiff1d(after.pp, before.pp),
        ),
    )


# ---------------------------------------------------------------------------
# Integer contract
# ---------------------------------------------------------------------------


def check_abs_bound(values: np.ndarray, limit: int, what: str) -> None:
    peak = int(np.abs(values.astype(np.int64)).max()) if values.size else 0
    if peak > limit:
        raise FeatureContractError(f"{what} exceeds declared bound: |{peak}| > {limit}")


def check_accumulator_bound(values: np.ndarray, what: str = "accumulator") -> None:
    check_abs_bound(values, ACCUMULATOR_ABS_BOUND, what)


def check_int16_safe(values: np.ndarray, what: str = "intermediate accumulator") -> None:
    check_abs_bound(values, INT16_MAX, what)


def fold_factorized(terms: list[np.ndarray], abs_limit: int, dtype) -> np.ndarray:
    """Sum factorized weight terms and enforce the bound AFTER folding.

    Coefficient limits apply to the folded result, never to unfactorized
    factors (spec 4.4: 'Apply bounds after factorized weights are folded').
    """
    if not terms:
        raise FeatureContractError("no factor terms to fold")
    acc = terms[0].astype(np.int64)
    for t in terms[1:]:
        acc = acc + t.astype(np.int64)
    check_abs_bound(acc, abs_limit, "folded factorized weights")
    return acc.astype(dtype)


class FeatureTables(NamedTuple):
    """Runtime transform tables: rows x CHANNELS, plus the int16 bias."""

    bias: np.ndarray  # int16 [512]
    psq: np.ndarray  # int16 [9216, 512]
    thr: np.ndarray  # int8 [59808, 512]
    pp: np.ndarray  # int8 [1488, 512]


def validate_feature_tables(t: FeatureTables) -> None:
    """Post-fold/post-decode coefficient enforcement."""
    check_abs_bound(t.bias, BIAS_ABS_LIMIT, "transform bias")
    check_abs_bound(t.psq, PSQ_COEF_ABS_LIMIT, "PSQ rows")
    check_abs_bound(t.thr, THREAT_COEF_ABS_LIMIT, "threat rows")
    check_abs_bound(t.pp, PP_COEF_ABS_LIMIT, "pawn-pair rows")


# Dirty-feature capacities equal the active bounds: a delta can only remove
# a subset of a legal position's features and add a subset of the next
# position's. They are checked, never inferred from average churn.
PSQ_MAX_DIRTY = PSQ_MAX_ACTIVE
THREAT_MAX_DIRTY = THREAT_MAX_ACTIVE
PP_MAX_DIRTY = PP_MAX_ACTIVE


class Accumulator:
    """int16 [2, 512] early-fused accumulator with the contract update path.

    ``apply`` removes obsolete rows before adding new ones and keeps all
    intermediates in widened int32, asserting the proven 30,048 envelope at
    every state boundary and the int16 storage bound on every step. Any
    violation raises FeatureContractError so callers can fall back to a
    full refresh rather than silently wrapping an int16 lane.
    """

    __slots__ = ("state",)

    def __init__(self) -> None:
        self.state = np.zeros((PERSPECTIVES, CHANNELS), dtype=np.int16)

    @classmethod
    def refresh(cls, tables: FeatureTables, board: Board) -> Accumulator:
        acc = cls()
        for p in (WHITE, BLACK):
            feats = active_features(board, p)
            acc.state[p] = acc._compute(tables, feats)
        return acc

    @staticmethod
    def _compute(tables: FeatureTables, feats: FeatureTriple) -> np.ndarray:
        tmp = tables.bias.astype(np.int32).copy()
        if feats.psq.size:
            tmp += tables.psq[feats.psq].sum(axis=0, dtype=np.int32)
        if feats.thr.size:
            tmp += tables.thr[feats.thr].sum(axis=0, dtype=np.int32)
        if feats.pp.size:
            tmp += tables.pp[feats.pp].sum(axis=0, dtype=np.int32)
        check_accumulator_bound(tmp)
        return tmp.astype(np.int16)

    def apply(
        self,
        tables: FeatureTables,
        persp: int,
        removed: FeatureTriple,
        added: FeatureTriple,
    ) -> None:
        """Delta-update one perspective; removes before adds, int32 inside."""
        for name, rows, cap in (
            ("removed PSQ", removed.psq, PSQ_MAX_DIRTY),
            ("removed threat", removed.thr, THREAT_MAX_DIRTY),
            ("removed pawn-pair", removed.pp, PP_MAX_DIRTY),
            ("added PSQ", added.psq, PSQ_MAX_DIRTY),
            ("added threat", added.thr, THREAT_MAX_DIRTY),
            ("added pawn-pair", added.pp, PP_MAX_DIRTY),
        ):
            if len(rows) > cap:
                raise FeatureContractError(f"{name} dirty count {len(rows)} exceeds capacity {cap}")

        tmp = self.state[persp].astype(np.int32)
        for rows, table in (
            (removed.psq, tables.psq),
            (removed.thr, tables.thr),
            (removed.pp, tables.pp),
        ):
            for r in rows:
                tmp -= table[r]
            check_accumulator_bound(tmp, "post-remove accumulator")
        for rows, table in (
            (added.psq, tables.psq),
            (added.thr, tables.thr),
            (added.pp, tables.pp),
        ):
            for r in rows:
                tmp += table[r]
                check_int16_safe(tmp)
        check_accumulator_bound(tmp, "post-add accumulator")
        self.state[persp] = tmp


# ---------------------------------------------------------------------------
# Scalar integer head (reference arithmetic; the Numba kernels mirror this)
# ---------------------------------------------------------------------------


def material_stack(piece_count: int) -> int:
    """Head bucket: min(7, max(0, (piece_count - 2) // 4))."""
    return min(7, max(0, (piece_count - 2) // 4))


def paired_activations(acc_state: np.ndarray, stm: int) -> np.ndarray:
    """512 clipped paired products: [stm perspective, ntm perspective]."""
    out = np.empty(HEAD_INPUTS, dtype=np.int32)
    for i, p in enumerate((stm, stm ^ 1)):
        a = np.clip(acc_state[p][: CHANNELS // 2], *FT_OPERAND_CLIP).astype(np.int32)
        b = np.clip(acc_state[p][CHANNELS // 2 :], *FT_OPERAND_CLIP).astype(np.int32)
        out[i * 256 : (i + 1) * 256] = (a * b) >> PAIRED_PRODUCT_SHIFT
    return out


def _hidden_activation(affine: np.ndarray) -> np.ndarray:
    """clip(x,0,127) concat clip((x*x)>>7,0,127) where x = affine>>6 is the
    UNCLIPPED shifted affine — the square is sign-blind (negative x yields
    a positive square channel).  This is the pinned SqrClippedReLU /
    pair-activation semantic; see architecture.json
    numeric_contract.hidden_activation_square_operand."""
    x = (affine >> HIDDEN_AFFINE_SHIFT).astype(np.int64)
    lin = np.clip(x, *HIDDEN_CLIP)
    sq = np.clip((x * x) >> SQUARE_ACTIVATION_SHIFT, *HIDDEN_CLIP)
    return np.concatenate([lin, sq])


def _trunc_div2(v: int) -> int:
    """C-style integer division by 2 (truncation toward zero)."""
    return -((-v) // 2) if v < 0 else v // 2


def head_forward(
    model: dict,
    acc_state: np.ndarray,
    stm: int,
    piece_count: int,
    psq_rows: tuple[np.ndarray, np.ndarray],
) -> dict:
    """Exact integer forward pass: scalar + W/D/L logits + PSQT skip.

    Layout: paired products -> affine 512->16 -> act(32) -> affine 32->32 ->
    act(64) -> concat 96 -> affine 96->4. The PSQT skip is a piece-square
    linear path over the same 9,216 rows: bias + trunc((stm - ntm)/2) for
    the selected material stack. The leaf applies the deployed rescale
    ``cp = clip(raw * scale_num >> scale_shift, ±neural_bound)`` in int64;
    absent meta keys default to the export-fixed constants
    (LEAF_SCALE_NUM/LEAF_SCALE_SHIFT/LEAF_NEURAL_BOUND).
    """
    v = paired_activations(acc_state, stm)
    s = material_stack(piece_count)

    x1 = model["head_b1"][s].astype(np.int64) + v @ model["head_w1"][s].astype(np.int64)
    a1 = _hidden_activation(x1)
    x2 = model["head_b2"][s].astype(np.int64) + a1 @ model["head_w2"][s].astype(np.int64)
    a2 = _hidden_activation(x2)
    z = np.concatenate([a1, a2])
    out = model["head_b3"][s].astype(np.int64) + z @ model["head_w3"][s].astype(np.int64)

    w_stm = model["psqt_w"][psq_rows[0]]
    w_ntm = model["psqt_w"][psq_rows[1]]
    d = w_stm.sum(axis=0, dtype=np.int64) - w_ntm.sum(axis=0, dtype=np.int64)
    psqt = int(model["psqt_b"][s]) + _trunc_div2(int(d[s]))

    meta = model.get("_meta", {})
    scale_num = int(meta.get("scale_num", LEAF_SCALE_NUM))
    scale_shift = int(meta.get("scale_shift", LEAF_SCALE_SHIFT))
    neural_bound = int(meta.get("neural_bound", LEAF_NEURAL_BOUND))
    scalar = int(
        np.clip(
            (int(out[0]) + psqt) * scale_num >> scale_shift,
            -neural_bound,
            neural_bound,
        )
    )
    return {"scalar": scalar, "wdl": out[1:4], "raw": out, "psqt": psqt, "stack": s}


def evaluate_position(model: dict, board: Board) -> int:
    """Scalar integer reference evaluation of `board` (side to move)."""
    tables = FeatureTables(model["bias"], model["psq"], model["thr"], model["pp"])
    acc = Accumulator.refresh(tables, board)
    feats_w = active_psq_indices(board, WHITE)
    feats_b = active_psq_indices(board, BLACK)
    stm = board.side
    res = head_forward(
        model,
        acc.state,
        stm,
        bin(board._occ_all).count("1"),
        (feats_w, feats_b) if stm == WHITE else (feats_b, feats_w),
    )
    return int(res["scalar"])


# ---------------------------------------------------------------------------
# Schema manifest: the single source of truth other components pin to.
# ---------------------------------------------------------------------------


def schema_manifest() -> dict:
    """Canonical feature-schema description for data/training/export pinning."""
    import hashlib

    return {
        "name": "RX-FINAL F512-EF-K12-16/32",
        "channels": CHANNELS,
        "perspectives": PERSPECTIVES,
        "king_buckets": KING_BUCKETS,
        "king_bucket_map": [list(r) for r in KING_BUCKET_MAP],
        "rows": {"psq": PSQ_ROWS, "threat": THREAT_ROWS, "pawn_pair": PP_ROWS},
        "runtime_dtypes": {"psq": "int16", "threat": "int8", "pawn_pair": "int8", "bias": "int16"},
        "coefficient_abs_limits": {
            "psq": PSQ_COEF_ABS_LIMIT,
            "threat": THREAT_COEF_ABS_LIMIT,
            "pawn_pair": PP_COEF_ABS_LIMIT,
            "bias": BIAS_ABS_LIMIT,
        },
        "storage_bits": {
            "psq": PSQ_STORAGE_BITS,
            "threat": THREAT_STORAGE_BITS,
            "pawn_pair": PP_STORAGE_BITS,
            "bias": BIAS_STORAGE_BITS,
        },
        "max_active": {
            "psq": PSQ_MAX_ACTIVE,
            "threat": THREAT_MAX_ACTIVE,
            "pawn_pair": PP_MAX_ACTIVE,
        },
        "accumulator_abs_bound": ACCUMULATOR_ABS_BOUND,
        "numeric_contract": {
            "ft_operand_clip": list(FT_OPERAND_CLIP),
            "paired_product_shift": PAIRED_PRODUCT_SHIFT,
            "paired_product_max": PAIRED_PRODUCT_MAX,
            "hidden_clip": list(HIDDEN_CLIP),
            "hidden_affine_shift": HIDDEN_AFFINE_SHIFT,
            "square_activation_shift": SQUARE_ACTIVATION_SHIFT,
            "rounding": "floor right shifts (Python/C arithmetic >>)",
            "material_stack": "min(7,max(0,(piece_count-2)//4))",
            "psqt": "bias[stack] + trunc((stm_rows - ntm_rows) / 2)",
        },
        "fusion": "psq + threat + pawn_pair rows share the same 512 channels",
        "threat_definition": "Stockfish@59aae690 src/nnue/features/full_threats.{h,cpp}",
        "pawn_pair_definition": (
            "96 relative-colour identities, ranks 2-7, file distance <= 1, dense by b*(b-1)//2+a"
        ),
        "table_sha256": {
            "threat_lut1": hashlib.sha256(_THREAT_LUT1.tobytes()).hexdigest(),
            "threat_offsets": hashlib.sha256(_THREAT_OFFSETS.tobytes()).hexdigest(),
            "threat_lut2": hashlib.sha256(_THREAT_LUT2.tobytes()).hexdigest(),
            "pp_old_to_new": hashlib.sha256(PP_OLD_TO_NEW.tobytes()).hexdigest(),
            "pp_new_to_old": hashlib.sha256(PP_NEW_TO_OLD.tobytes()).hexdigest(),
        },
    }

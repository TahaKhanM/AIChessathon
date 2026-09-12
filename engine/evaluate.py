"""Integer kernels, refresh caches, scalar outputs.

W05 — full-fusion incremental accumulator and integer kernels for the
F512-EF-K12-16/32 reference evaluator (docs/architecture.md
sections 4.2, 4.4, 4.5).

The canonical feature encoders live in ``engine/features.py`` (work package
W02).  All index LUTs here are *derived from* that module's canonical tables
(imported at module load) so the two implementations cannot drift:

- PSQ rows: ``768 * king_bucket + 384 * relative_color + 64 * piece_type +
  oriented_square`` — ``features.psq_index``.
- Threat rows: ``features.threat_index``, a faithful port of Stockfish
  @59aae690 ``src/nnue/features/full_threats.{h,cpp}`` ``make_index``
  (59,808 rows, excluded pairs land on the dimension bound and are
  dropped).  This module adds the ``update_piece_threats`` dirty-threat
  port (``_update_piece_threats`` / ``_proc_sliders``) that features.py
  intentionally does not provide.
- Pawn pairs: ``features.pawn_pair_index`` — 96 relative-colour identities
  on ranks 2-7, file distance <= 1, dense rows ordered by
  ``b*(b-1)//2 + a``.

Layer structure per spec section 4.2:

    acc[persp][512] (int16, proven |.| <= 30048)
        -> paired transform: clip halves to [0,255], widen, a*b >> 9
           -> 256 activations in [0,127] per perspective
        -> concat side-to-move then opponent -> x[512]
    head = material stack min(7, max(0, (piece_count - 2) // 4)):
        affine 512->16; clip(z>>6,0,127) ++ clip((z>>6)^2>>7,0,127) -> 32
        affine 32->32;  clip(z>>6,0,127) ++ clip((z>>6)^2>>7,0,127) -> 64
        concat 32+64 = 96; affine 96->4 (scalar + W/D/L logits)
    scalar path computes only output 0: 8192 + 1024 + 96 = 9,312 MACs.
    PSQT skip: int16[9216][8] + int32[8] biases, accumulated in int32.

Hidden activation (architecture.json numeric_contract.hidden_activation):
x = z>>6 is the UNCLIPPED shifted affine; the square branch computes
clip((x*x)>>7, 0, 127) — sign-blind on negatives (pinned SqrClippedReLU /
pair-activation semantic).

Rounding is deterministic floor shift (two's-complement arithmetic shift,
identical in Python, NumPy and Numba).  The leaf transform is
cp = clip((dot + psqt) * SCALE_NUM >> SCALE_SHIFT, +-NEURAL_BOUND) in
int64; the constants are export-fixed fields on EvalWeights, never
runtime-edited.
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit

from engine.board import (
    BISHOP,
    BLACK,
    EMPTY,
    FLAG_CASTLE,
    FLAG_EP,
    FLAG_NULL,
    FLAG_PROMO,
    FLAG_PROMO_CAP,
    KING,
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
    decode_move,
)
from engine.movegen import (
    BISHOP_NEG,
    BISHOP_POS,
    ROOK_NEG,
    ROOK_POS,
    bishop_attacks,
    rook_attacks,
)
import engine.movegen as _mg
import engine.features as _features
from engine.tt import MATE_IN_MAX

# ============================================================================
# Contract constants — every value is asserted against
# spec/RX_FINAL_PLAN/architecture.json by tests/test_evaluate.py; the JSON is
# authoritative and this table exists so the runtime does not read spec/ files.
# ============================================================================

CHANNELS = 512
HALF = CHANNELS // 2
FT_CLIP_HI = 255
PRODUCT_SHIFT = 9
PRODUCT_MAX = 127

KING_BUCKETS = 12
PSQ_ROWS = 9216
THREAT_ROWS = 59808
PP_ROWS = 1488
PP_WIDE_ROWS = 96 * 95 // 2  # 4560 triangular-index space

PSQ_COEF_MAX = 255
THREAT_COEF_MAX = 63
PP_COEF_MAX = 31
BIAS_COEF_MAX = 2040
ACC_BOUND = 30048  # 32*255 + 256*63 + 120*31 + 2040

HEAD_STACKS = 8
HEAD_IN = 512
L1_OUT = 16
L1_ACT = 32
L2_OUT = 32
L2_ACT = 64
SKIP = 96
HEAD_OUT = 4
HIDDEN_SHIFT = 6
HIDDEN_CLIP = 127
SQUARE_SHIFT = 7
SCALAR_DENSE_MACS = 512 * 16 + 32 * 32 + 96  # 9,312 (scalar column only)

PSQT_BUCKETS = 8

THREAT_OP_CAP = 96  # SF bound: non-castling <= 80, castling <= 36, +16 pad
PP_OP_CAP = 64  # <=2 removed + 1 added pawn, <=17 partners each (SF uses 64)
PSQ_OP_CAP = 8  # mover + capture + promo swap + castle rook, generous
MAX_ACTIVE_THREATS = 256
MAX_ACTIVE_PP = 120
MAX_PIECES = 32
STACK_PLY = 2048

# Deployed leaf transform (architecture.json numeric_contract.leaf_transform
# and leaf_transform_export_defaults):
#   cp = clip((dot + psqt) * SCALE_NUM >> SCALE_SHIFT, -NEURAL_BOUND, +NEURAL_BOUND)
# in int64.  SCALE_NUM/SCALE_SHIFT are the canonical export defaults recorded
# in RXF1 _meta; they are versioned with the weights, never edited beneath an
# existing export.
SCALE_NUM = _features.LEAF_SCALE_NUM
SCALE_SHIFT = _features.LEAF_SCALE_SHIFT
# Strictly below the mate band: mate-in-N scores occupy [MATE_IN_MAX, MATE).
# A clipped neural score must never equal a mate sentinel, so the clamp is
# MATE_IN_MAX - 1 = 27951 (spec: "bounded strictly inside mate score band;
# enforce after exact exported transforms"; spec/RX_FINAL_PLAN/architecture.json).
NEURAL_BOUND = MATE_IN_MAX - 1

U64 = np.uint64
U0 = np.uint64(0)
U1 = np.uint64(1)


# ============================================================================
# Geometry tables
# ============================================================================


def _pc(x: int) -> int:
    return bin(x).count("1")


def _build_luts() -> dict[str, np.ndarray]:
    t: dict[str, np.ndarray] = {}
    t["PAWN_ATK"] = np.asarray(PAWN_ATK, dtype=np.uint64)  # [2][64]
    t["KNIGHT_ATK"] = np.asarray(KNIGHT_ATK, dtype=np.uint64)
    t["KING_ATK"] = np.asarray(KING_ATK, dtype=np.uint64)
    t["RPOS"] = np.asarray(ROOK_POS, dtype=np.uint64)
    t["RNEG"] = np.asarray(ROOK_NEG, dtype=np.uint64)
    t["BPOS"] = np.asarray(BISHOP_POS, dtype=np.uint64)
    t["BNEG"] = np.asarray(BISHOP_NEG, dtype=np.uint64)

    # pseudo-attacks on empty board, indexed by engine piece type 0..5
    pseudo = np.zeros((6, 64), dtype=np.uint64)
    for sq in range(64):
        pseudo[KNIGHT, sq] = KNIGHT_ATK[sq]
        pseudo[BISHOP, sq] = np.uint64(bishop_attacks(sq, 0))
        pseudo[ROOK, sq] = np.uint64(rook_attacks(sq, 0))
        pseudo[QUEEN, sq] = np.uint64(bishop_attacks(sq, 0) | rook_attacks(sq, 0))
        pseudo[KING, sq] = KING_ATK[sq]
    t["PSEUDO"] = pseudo

    # ray_pass(a,b): squares on the ray from a through b, excluding a,
    # including b and everything beyond (Stockfish RayPassBB semantics).
    ray_pass = np.zeros((64, 64), dtype=np.uint64)
    for a in range(64):
        for b in range(64):
            if a == b:
                continue
            if pseudo[BISHOP, a] & (1 << b):
                atk_a = bishop_attacks(a, 0)
                atk_b = bishop_attacks(b, 1 << a)
            elif pseudo[ROOK, a] & (1 << b):
                atk_a = rook_attacks(a, 0)
                atk_b = rook_attacks(b, 1 << a)
            else:
                continue
            ray_pass[a, b] = np.uint64(atk_a & (atk_b | (1 << b)))
    t["RAY_PASS"] = ray_pass

    # pawn-pair partner mask: own file +/- 1, ranks 2..7, excluding s
    pp_mask = np.zeros(64, dtype=np.uint64)
    file_a = 0x0101010101010101
    rank27 = 0x00FFFFFFFFFFFF00
    for sq in range(64):
        f = sq & 7
        files = file_a << f
        if f > 0:
            files |= file_a << (f - 1)
        if f < 7:
            files |= file_a << (f + 1)
        pp_mask[sq] = np.uint64(files & rank27 & ~(1 << sq))
    t["PP_MASK"] = pp_mask

    # dense pawn-pair row map — the canonical table from features.py
    # (identity = relcolor*48 + (oriented_sq-8); rows ordered by
    # hi*(hi-1)//2+lo).  int32 -> int32 for numba.
    t["PP_DENSE"] = np.ascontiguousarray(_features.PP_OLD_TO_NEW, dtype=np.int32)

    t["K12"] = np.asarray(_features.KING_BUCKET_LAYOUT, dtype=np.uint8)

    t["POP16"] = np.asarray([_pc(i) for i in range(1 << 16)], dtype=np.uint8)
    t["MSB8"] = np.asarray([0] + [i.bit_length() - 1 for i in range(1, 256)], dtype=np.uint8)
    return t


_T = _build_luts()
PAWN_ATK_A = _T["PAWN_ATK"]
KNIGHT_ATK_A = _T["KNIGHT_ATK"]
RPOS_A = _T["RPOS"]
RNEG_A = _T["RNEG"]
BPOS_A = _T["BPOS"]
BNEG_A = _T["BNEG"]
RAY_PASS_A = _T["RAY_PASS"]
PP_MASK_A = _T["PP_MASK"]
PP_DENSE_A = _T["PP_DENSE"]
K12_A = _T["K12"]
POP16_A = _T["POP16"]
MSB8_A = _T["MSB8"]

_BETWEEN_A = np.asarray(_mg.BETWEEN, dtype=np.uint64)
_LINE_A = np.asarray(_mg.LINE, dtype=np.uint64)


# ============================================================================
# Threat index LUTs — canonical tables are built by features.py exactly as
# the pinned Stockfish full_threats.{h,cpp}; here they are re-indexed from
# SF piece codes ((colour<<3)|(type+1)) into engine codes (colour*6+type)
# so the kernels never pay the conversion.
# ============================================================================


def _build_threat_luts() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    off = np.zeros((12, 64), dtype=np.int32)
    sub = np.zeros((12, 64, 64), dtype=np.int16)
    base = np.full((12, 12, 2), THREAT_ROWS, dtype=np.int32)
    for ea in range(12):
        sf_a = (ea % 6) + 1 + (ea // 6) * 8
        off[ea] = _features._THREAT_OFFSETS[sf_a]
        sub[ea] = _features._THREAT_LUT2[sf_a].astype(np.int16)
        for ed in range(12):
            sf_d = (ed % 6) + 1 + (ed // 6) * 8
            base[ea, ed, 0] = _features._THREAT_LUT1[sf_a, sf_d, 0]
            base[ea, ed, 1] = _features._THREAT_LUT1[sf_a, sf_d, 1]
    return off, sub, base


_T_OFF, _T_SUB, _T_BASE = _build_threat_luts()


def _orientation(ksq: int, persp: int) -> int:
    """Square xor mask: file mirror from the perspective king's side + rank
    flip for black.  Identical to features.orientation / OrientTBL usage."""
    return _features.orientation(persp, ksq)


def _frame(ksq: int, persp: int) -> int:
    """Packed king-normalization frame: bucket<<1 | mirror."""
    return (int(_features.king_bucket(persp, ksq)) << 1) | (
        1 if _features.orientation(persp, ksq) & 7 else 0
    )


def _swap_pc(pc: int, persp: int) -> int:
    """Perspective colour swap on engine codes (colour*6+type): not xor."""
    if persp == 0:
        return pc
    return pc + 6 if pc < 6 else pc - 6


def threat_row_py(persp: int, pc: int, frm: int, to: int, dpc: int, ksq: int) -> int:
    """Canonical FullThreats row via features.threat_index. -1 excluded."""
    r = _features.threat_index(persp, ksq, pc, frm, to, dpc)
    return r if r < THREAT_ROWS else -1


def psq_row_py(persp: int, pc: int, sq: int, ksq: int) -> int:
    return _features.psq_index(persp, ksq, pc, sq)


def pp_row_py(persp: int, c_a: int, sq_a: int, c_b: int, sq_b: int, ksq: int) -> int:
    """Canonical dense pawn-pair row. c_a/c_b are absolute colours (0/1)."""
    return _features.pawn_pair_index(persp, ksq, c_a * 6 + PAWN, sq_a, c_b * 6 + PAWN, sq_b)


# ============================================================================
# Numba bit helpers
# ============================================================================


@njit(cache=True)
def _pop64(b):
    return (
        POP16_A[np.int64(b & U64(0xFFFF))]
        + POP16_A[np.int64((b >> 16) & U64(0xFFFF))]
        + POP16_A[np.int64((b >> 32) & U64(0xFFFF))]
        + POP16_A[np.int64(b >> 48)]
    )


@njit(cache=True)
def _lsb(b):
    return _pop64((b & (U0 - b)) - U1)


@njit(cache=True)
def _msb(b):
    r = np.int64(0)
    if b >= U64(0x100000000):
        r += 32
        b >>= 32
    if b >= U64(0x10000):
        r += 16
        b >>= 16
    if b >= U64(0x100):
        r += 8
        b >>= 8
    return r + MSB8_A[np.int64(b)]


@njit(cache=True)
def _ray_attacks(sq, occ, pos, neg):
    atk = U0
    for i in range(2):
        ray = pos[sq, i]
        hits = ray & occ
        if hits:
            b = _lsb(hits)
            atk |= _BETWEEN_A[sq, b] | (U1 << b)
        else:
            atk |= ray
    for i in range(2):
        ray = neg[sq, i]
        hits = ray & occ
        if hits:
            b = _msb(hits)
            atk |= _BETWEEN_A[sq, b] | (U1 << b)
        else:
            atk |= ray
    return atk


# ============================================================================
# Dirty-threat op emitter — the incremental update_piece_threats port.
# Row ids resolve through the canonical feature tables (engine/features.py
# via _threat_row_nb/_psq_row_nb/_pp_row_from_ids); nothing below re-encodes
# the feature set.
# ============================================================================


@njit(cache=True)
def _b_att(sq, occ):
    return _ray_attacks(sq, occ, BPOS_A, BNEG_A)


@njit(cache=True)
def _r_att(sq, occ):
    return _ray_attacks(sq, occ, RPOS_A, RNEG_A)


@njit(cache=True)
def _emit(dts, n, add, pc, tpc, s, tsq):
    """Pack a DirtyThreat: add<<31 | pc<<20 | tpc<<16 | tsq<<8 | s."""
    if n < dts.shape[0]:
        dts[n] = (
            (np.uint32(0x80000000) if add else np.uint32(0))
            | (np.uint32(pc) << 20)
            | (np.uint32(tpc) << 16)
            | (np.uint32(tsq) << 8)
            | np.uint32(s)
        )
    return n + 1


@njit(cache=True)
def _can_slider_threat(pc, slider):
    return (pc % 6) != QUEEN or (slider % 6) == QUEEN


@njit(cache=True)
def _proc_sliders(mb, s, slider_attacks, occ_nok, sliders, no_rays, put, pc, add_direct, dts, n):
    b = sliders
    while b:
        sq = _lsb(b)
        b &= b - U1
        slider = mb[sq]
        ray = RAY_PASS_A[sq, s]
        discovered = ray & slider_attacks & occ_nok
        if discovered and (ray & no_rays) != no_rays:
            tsq = _lsb(discovered)
            tpc = mb[tsq]
            if _can_slider_threat(tpc, slider):
                n = _emit(dts, n, not put, slider, tpc, sq, tsq)
        if add_direct and _can_slider_threat(pc, slider):
            n = _emit(dts, n, put, slider, pc, sq, s)
    return n


@njit(cache=True)
def _update_piece_threats(mb, bb, occ, pc, put, s, no_rays, compute_ray, dts, n):
    """Port of Stockfish update_piece_threats on a mutable scratch state."""
    b_a = _b_att(s, occ)
    r_a = _r_att(s, occ)
    slider_attacks = b_a | r_a
    occ_nok = occ ^ (bb[5] | bb[11])  # occ minus both kings
    sliders = (
        (bb[2] | bb[4] | bb[8] | bb[10]) & b_a  # WB|WQ|BB|BQ on bishop rays
    ) | (
        (bb[3] | bb[4] | bb[9] | bb[10]) & r_a  # WR|WQ|BR|BQ on rook rays
    )
    pt = pc % 6
    if pt == KING:
        if compute_ray:
            n = _proc_sliders(
                mb, s, slider_attacks, occ_nok, sliders, no_rays, put, pc, False, dts, n
            )
        return n
    ttargets = U0
    patk = U0
    if pt == PAWN:
        ttargets = bb[1] | bb[7] | bb[3] | bb[9]  # N,R both colours
        patk = PAWN_ATK_A[pc // 6, s]
    elif pt == BISHOP or pt == ROOK:
        ttargets = bb[0] | bb[6] | bb[1] | bb[7] | bb[2] | bb[8] | bb[3] | bb[9]
        patk = b_a if pt == BISHOP else r_a
    elif pt == QUEEN:
        ttargets = occ_nok
        patk = slider_attacks
    else:  # KNIGHT
        ttargets = occ_nok
        patk = KNIGHT_ATK_A[s]
    threatened = patk & ttargets
    incoming = KNIGHT_ATK_A[s] & (bb[1] | bb[7])
    if pt == KNIGHT or pt == ROOK:
        incoming |= (PAWN_ATK_A[0, s] & bb[6]) | (PAWN_ATK_A[1, s] & bb[0])
    t = threatened
    while t:
        tsq = _lsb(t)
        t &= t - U1
        n = _emit(dts, n, put, pc, mb[tsq], s, tsq)
    if compute_ray:
        n = _proc_sliders(mb, s, slider_attacks, occ_nok, sliders, no_rays, put, pc, True, dts, n)
    else:
        incoming |= (sliders & (bb[4] | bb[10])) if pt == QUEEN else sliders
    while incoming:
        src = _lsb(incoming)
        incoming &= incoming - U1
        n = _emit(dts, n, put, mb[src], pc, src, s)
    return n


# --- scratch piece ops ----------------------------------------------------


@njit(cache=True)
def _sq_clear(mb, bb, occ_c, s):
    pc = mb[s]
    bit = U1 << s
    mb[s] = EMPTY
    bb[pc] &= MASK64 ^ bit
    occ_c[pc // 6] &= MASK64 ^ bit
    return pc


@njit(cache=True)
def _sq_set(mb, bb, occ_c, s, pc):
    bit = U1 << s
    mb[s] = pc
    bb[pc] |= bit
    occ_c[pc // 6] |= bit


@njit(cache=True)
def _psq_op(ops, n, pc, frm, to):
    if n < ops.shape[0]:
        ops[n, 0] = pc
        ops[n, 1] = frm
        ops[n, 2] = to
    return n + 1


@njit(cache=True)
def _do_remove(mb, bb, occ_c, occ, s, dts, nt, pops, np_):
    pc = mb[s]
    nt = _update_piece_threats(mb, bb, occ, pc, False, s, MASK64, True, dts, nt)
    _sq_clear(mb, bb, occ_c, s)
    occ &= MASK64 ^ (U1 << s)
    np_ = _psq_op(pops, np_, pc, s, -1)
    return occ, nt, np_


@njit(cache=True)
def _do_put(mb, bb, occ_c, occ, s, pc, dts, nt, pops, np_):
    _sq_set(mb, bb, occ_c, s, pc)
    occ |= U1 << s
    nt = _update_piece_threats(mb, bb, occ, pc, True, s, MASK64, True, dts, nt)
    np_ = _psq_op(pops, np_, pc, -1, s)
    return occ, nt, np_


@njit(cache=True)
def _do_move(mb, bb, occ_c, occ, frm, to, dts, nt, pops, np_):
    pc = mb[frm]
    ft = (U1 << frm) | (U1 << to)
    nt = _update_piece_threats(mb, bb, occ, pc, False, frm, ft, True, dts, nt)
    bit_f = U1 << frm
    bit_t = U1 << to
    mb[frm] = EMPTY
    mb[to] = pc
    bb[pc] = (bb[pc] & (MASK64 ^ bit_f)) | bit_t
    occ_c[pc // 6] = (occ_c[pc // 6] & (MASK64 ^ bit_f)) | bit_t
    occ = (occ & (MASK64 ^ bit_f)) | bit_t
    nt = _update_piece_threats(mb, bb, occ, pc, True, to, ft, True, dts, nt)
    np_ = _psq_op(pops, np_, pc, frm, to)
    return occ, nt, np_


@njit(cache=True)
def _do_swap(mb, bb, occ_c, occ, s, newpc, dts, nt, pops, np_):
    old = mb[s]
    _sq_clear(mb, bb, occ_c, s)
    occ &= MASK64 ^ (U1 << s)
    nt = _update_piece_threats(mb, bb, occ, old, False, s, MASK64, False, dts, nt)
    _sq_set(mb, bb, occ_c, s, newpc)
    occ |= U1 << s
    nt = _update_piece_threats(mb, bb, occ, newpc, True, s, MASK64, False, dts, nt)
    np_ = _psq_op(pops, np_, old, s, -1)
    np_ = _psq_op(pops, np_, newpc, -1, s)
    return occ, nt, np_


@njit(cache=True)
def _castle_rook_squares(to):
    if to == 6:
        return 7, 5
    if to == 2:
        return 0, 3
    if to == 62:
        return 63, 61
    return 56, 59


@njit(cache=True)
def _compute_dirties(mb, bb, occ_c, occ, frm, to, promo, flag, piece, captured, us, dts, pops):
    """Fill dts/pops replaying Stockfish do_move's op order on the scratch
    board arrays (mutated to the after-state). Returns counts and pawn bbs."""
    pw_b, pb_b = bb[0], bb[6]
    nt = np_ = 0
    if flag == FLAG_CASTLE:
        rfrom, rto = _castle_rook_squares(to)
        occ, nt, np_ = _do_remove(mb, bb, occ_c, occ, frm, dts, nt, pops, np_)
        occ, nt, np_ = _do_remove(mb, bb, occ_c, occ, rfrom, dts, nt, pops, np_)
        occ, nt, np_ = _do_put(mb, bb, occ_c, occ, to, us * 6 + KING, dts, nt, pops, np_)
        occ, nt, np_ = _do_put(mb, bb, occ_c, occ, rto, us * 6 + ROOK, dts, nt, pops, np_)
    elif flag == FLAG_EP:
        cap = to - 8 if us == WHITE else to + 8
        occ, nt, np_ = _do_remove(mb, bb, occ_c, occ, cap, dts, nt, pops, np_)
        occ, nt, np_ = _do_move(mb, bb, occ_c, occ, frm, to, dts, nt, pops, np_)
    elif flag == FLAG_PROMO:
        occ, nt, np_ = _do_remove(mb, bb, occ_c, occ, frm, dts, nt, pops, np_)
        occ, nt, np_ = _do_put(mb, bb, occ_c, occ, to, us * 6 + promo, dts, nt, pops, np_)
    elif captured != 15:  # FLAG_CAPTURE / FLAG_PROMO_CAP
        dest = us * 6 + promo if flag == FLAG_PROMO_CAP else piece
        occ, nt, np_ = _do_remove(mb, bb, occ_c, occ, frm, dts, nt, pops, np_)
        occ, nt, np_ = _do_swap(mb, bb, occ_c, occ, to, dest, dts, nt, pops, np_)
    elif flag != FLAG_NULL:
        occ, nt, np_ = _do_move(mb, bb, occ_c, occ, frm, to, dts, nt, pops, np_)
    return nt, np_, pw_b, pb_b, bb[0], bb[6], occ


@njit(cache=True)
def _reverse_ops(mb, bb, occ_c, occ, ops, n):
    """Undo the recorded piece ops on the shadow state (pop path)."""
    for i in range(n - 1, -1, -1):
        pc = np.int64(ops[i, 0])
        frm = np.int64(ops[i, 1])
        to = np.int64(ops[i, 2])
        if to >= 0:
            bit = U1 << to
            mb[to] = EMPTY
            bb[pc] &= MASK64 ^ bit
            occ_c[pc // 6] &= MASK64 ^ bit
            occ &= MASK64 ^ bit
        if frm >= 0:
            bit = U1 << frm
            mb[frm] = pc
            bb[pc] |= bit
            occ_c[pc // 6] |= bit
            occ |= bit
    return occ


# --- row resolution (ops -> per-perspective row lists) ---------------------


@njit(cache=True)
def _threat_row_nb(a, d, fo, to_):
    idx = _T_BASE[a, d, 1 if fo < to_ else 0] + _T_OFF[a, fo] + _T_SUB[a, fo, to_]
    return idx if idx < THREAT_ROWS else -1


@njit(cache=True)
def _swap_pc_nb(pc, persp):
    if persp == 0:
        return pc
    return pc + 6 if pc < 6 else pc - 6


@njit(cache=True)
def _threat_rows(ops, n, persp, orient, out_add, out_rem):
    """Returns (n_add, n_rem); counts run past output capacity on overflow."""
    na = nr = 0
    for i in range(n):
        op = ops[i]
        add = (op >> 31) != 0
        pc = np.int32((op >> 20) & 15)
        tpc = np.int32((op >> 16) & 15)
        tsq = np.int32((op >> 8) & 63)
        s = np.int32(op & 63)
        row = _threat_row_nb(
            _swap_pc_nb(pc, persp),
            _swap_pc_nb(tpc, persp),
            s ^ orient,
            tsq ^ orient,
        )
        if row < 0:
            continue
        if add:
            if na < out_add.shape[0]:
                out_add[na] = row
            na += 1
        else:
            if nr < out_rem.shape[0]:
                out_rem[nr] = row
            nr += 1
    return na, nr


@njit(cache=True)
def _psq_rows(ops, n, persp, bucket, orient, out_add, out_rem):
    na = nr = 0
    for i in range(n):
        pc = np.int32(ops[i, 0])
        frm = np.int32(ops[i, 1])
        to = np.int32(ops[i, 2])
        rel = np.int32(1 if (pc // 6) != persp else 0)
        base = 768 * bucket + 384 * rel + 64 * (pc % 6)
        if frm >= 0:
            if nr < out_rem.shape[0]:
                out_rem[nr] = base + (frm ^ orient)
            nr += 1
        if to >= 0:
            if na < out_add.shape[0]:
                out_add[na] = base + (to ^ orient)
            na += 1
    return na, nr


@njit(cache=True)
def _pp_row_from_ids(ida, idb):
    # ids outside [0,96) mean a pawn on a back rank reached the shadow
    # bitboards; the triangular index would read past PP_DENSE_A and the
    # garbage "row" passes the caller's >=0 check before indexing pp_w.
    if ida < 0 or ida >= 96 or idb < 0 or idb >= 96:
        return np.int32(-1)
    lo = ida if ida < idb else idb
    hi = idb if ida < idb else ida
    return PP_DENSE_A[hi * (hi - 1) // 2 + lo]


@njit(cache=True)
def _pp_rows(pw_b, pb_b, pw_a, pb_a, persp, orient, out_add, out_rem):
    """Pawn-pair diffs from before/after pawn bitboards.  Both directions are
    emitted and deduplicated by pawn identity so every pair lands once."""
    na = nr = 0
    rem_w = pw_b & ~pw_a
    rem_b = pb_b & ~pb_a
    rem_all = rem_w | rem_b
    for c in range(2):
        ch = rem_w if c == 0 else rem_b
        relc = np.int32(1 if c != persp else 0)
        allb = pw_b | pb_b
        u = ch
        while u:
            a = _lsb(u)
            u &= u - U1
            ida = relc * 48 + (a ^ orient) - 8
            part = allb & PP_MASK_A[a]
            while part:
                bsq = _lsb(part)
                part &= part - U1
                bc = np.int32(0 if (pw_b >> bsq) & U1 else 1)
                relb = np.int32(1 if bc != persp else 0)
                idb = relb * 48 + (bsq ^ orient) - 8
                if (rem_all >> bsq) & U1 and idb < ida:
                    continue  # both removed: emit under the higher id only
                row = _pp_row_from_ids(ida, idb)
                if row >= 0:
                    if nr < out_rem.shape[0]:
                        out_rem[nr] = row
                    nr += 1
    add_w = pw_a & ~pw_b
    add_b = pb_a & ~pb_b
    add_all = add_w | add_b
    for c in range(2):
        ch = add_w if c == 0 else add_b
        relc = np.int32(1 if c != persp else 0)
        alla = pw_a | pb_a
        u = ch
        while u:
            a = _lsb(u)
            u &= u - U1
            ida = relc * 48 + (a ^ orient) - 8
            part = alla & PP_MASK_A[a]
            while part:
                bsq = _lsb(part)
                part &= part - U1
                bc = np.int32(0 if (pw_a >> bsq) & U1 else 1)
                relb = np.int32(1 if bc != persp else 0)
                idb = relb * 48 + (bsq ^ orient) - 8
                if (add_all >> bsq) & U1 and idb < ida:
                    continue
                row = _pp_row_from_ids(ida, idb)
                if row >= 0:
                    if na < out_add.shape[0]:
                        out_add[na] = row
                    na += 1
    return na, nr


# --- full enumeration (refresh) --------------------------------------------


@njit(cache=True)
def _enum_psq(mb, persp, bucket, orient, out):
    n = 0
    for sq in range(64):
        pc = mb[sq]
        if pc < 0:
            continue
        rel = 1 if (pc // 6) != persp else 0
        if n < out.shape[0]:
            out[n] = 768 * bucket + 384 * rel + 64 * (pc % 6) + (sq ^ orient)
        n += 1
    return n


@njit(cache=True)
def _enum_threats(mb, bb, occ, persp, orient, out):
    n = 0
    pawn_t = bb[1] | bb[7] | bb[3] | bb[9]  # N,R
    minor_t = pawn_t | bb[0] | bb[6] | bb[2] | bb[8]  # +P,B
    queen_t = minor_t | bb[4] | bb[10]  # +Q
    for c in range(2):
        b = bb[c * 6]
        while b:
            frm = _lsb(b)
            b &= b - U1
            t = PAWN_ATK_A[c, frm] & pawn_t
            while t:
                to = _lsb(t)
                t &= t - U1
                row = _threat_row_nb(
                    _swap_pc_nb(c * 6, persp),
                    _swap_pc_nb(mb[to], persp),
                    frm ^ orient,
                    to ^ orient,
                )
                if row >= 0:
                    if n < out.shape[0]:
                        out[n] = row
                    n += 1
    for c in range(2):
        for pt in range(1, 5):  # N,B,R,Q — kings emit none
            piece = c * 6 + pt
            targets = minor_t if (pt == BISHOP or pt == ROOK) else queen_t
            b = bb[piece]
            while b:
                frm = _lsb(b)
                b &= b - U1
                if pt == BISHOP:
                    atk = _b_att(frm, occ)
                elif pt == ROOK:
                    atk = _r_att(frm, occ)
                elif pt == QUEEN:
                    atk = _b_att(frm, occ) | _r_att(frm, occ)
                else:
                    atk = KNIGHT_ATK_A[frm]
                t = atk & targets
                while t:
                    to = _lsb(t)
                    t &= t - U1
                    row = _threat_row_nb(
                        _swap_pc_nb(piece, persp),
                        _swap_pc_nb(mb[to], persp),
                        frm ^ orient,
                        to ^ orient,
                    )
                    if row >= 0:
                        if n < out.shape[0]:
                            out[n] = row
                        n += 1
    return n


@njit(cache=True)
def _enum_apply_persp(
    acc,
    psqt_acc,
    psq_part,
    mb,
    bb,
    occ,
    persp,
    bucket,
    orient,
    bias,
    psq_w,
    thr_w,
    pp_w,
    psqt_w,
    psq_buf,
    thr_buf,
    pp_buf,
):
    """Full refresh of one perspective in a single kernel: enumerate PSQ,
    threat and pawn-pair rows and accumulate.  ``psq_part`` (int32[512])
    receives bias+PSQ rows alone for the Finny cache; the PSQ row list is
    left in ``psq_buf``.  Accumulation runs on an int32 scratch and the
    caller checks ``peak`` before the int16 store is committed.  Returns
    (n_psq, n_thr, n_pp, peak)."""
    n_psq = _enum_psq(mb, persp, bucket, orient, psq_buf)
    n_thr = _enum_threats(mb, bb, occ, persp, orient, thr_buf)
    n_pp = _enum_pp(bb[0], bb[6], persp, orient, pp_buf)
    n_psq_c = min(n_psq, psq_buf.shape[0])
    n_thr_c = min(n_thr, thr_buf.shape[0])
    n_pp_c = min(n_pp, pp_buf.shape[0])
    tmp = np.empty(CHANNELS, np.int32)
    for c in range(CHANNELS):
        tmp[c] = bias[c]
        psq_part[c] = np.int32(bias[c])
    for c in range(PSQT_BUCKETS):
        psqt_acc[c] = 0
    for r in range(n_psq_c):
        row = psq_w[psq_buf[r]]
        for c in range(CHANNELS):
            psq_part[c] += row[c]
            tmp[c] += row[c]
    peak = _acc_peak(tmp)
    _acc_add_rows(tmp, thr_w, thr_buf, n_thr_c)
    peak = max(peak, _acc_peak(tmp))
    _acc_add_rows(tmp, pp_w, pp_buf, n_pp_c)
    peak = max(peak, _acc_peak(tmp))
    if peak <= ACC_BOUND:
        for c in range(CHANNELS):
            acc[c] = np.int16(tmp[c])
        _psqt_add(psqt_acc, psqt_w, psq_buf, n_psq_c)
    return n_psq, n_thr, n_pp, peak


@njit(cache=True)
def _enum_pp(pw, pb, persp, orient, out):
    n = 0
    b = pw
    while b:
        a = _lsb(b)
        b &= b - U1
        inner = b & PP_MASK_A[a]
        while inner:
            bsq = _lsb(inner)
            inner &= inner - U1
            ida = (0 != persp) * 48 + (a ^ orient) - 8
            idb = (0 != persp) * 48 + (bsq ^ orient) - 8
            row = _pp_row_from_ids(ida, idb)
            if row >= 0:
                if n < out.shape[0]:
                    out[n] = row
                n += 1
        inner = pb & PP_MASK_A[a]
        while inner:
            bsq = _lsb(inner)
            inner &= inner - U1
            ida = (0 != persp) * 48 + (a ^ orient) - 8
            idb = (1 != persp) * 48 + (bsq ^ orient) - 8
            row = _pp_row_from_ids(ida, idb)
            if row >= 0:
                if n < out.shape[0]:
                    out[n] = row
                n += 1
    b = pb
    while b:
        a = _lsb(b)
        b &= b - U1
        inner = b & PP_MASK_A[a]
        while inner:
            bsq = _lsb(inner)
            inner &= inner - U1
            ida = (1 != persp) * 48 + (a ^ orient) - 8
            idb = (1 != persp) * 48 + (bsq ^ orient) - 8
            row = _pp_row_from_ids(ida, idb)
            if row >= 0:
                if n < out.shape[0]:
                    out[n] = row
                n += 1
    return n


# ============================================================================
# Weights
# ============================================================================


class EvalWeights:
    """Bounded integer model.  Layout mirrors the section 4.3 accounting:

    psq_w  int16[9216,512]  |.|<=255        bias  int16[512] |.|<=2040
    thr_w  int8 [59808,512] |.|<=63         pp_w  int8[1488,512] |.|<=31
    w1 int8[8,16,512] b1 int32[8,16]   w2 int8[8,32,32] b2 int32[8,32]
    w3 int8[8,4,96]   b3 int32[8,4]    psqt int16[9216,8] psqt_b int32[8]
    """

    __slots__ = (
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
        "scale_num",
        "scale_shift",
        "neural_bound",
    )

    def __init__(self) -> None:
        # Defaults are the DEPLOYED leaf constants (architecture.json
        # numeric_contract.leaf_transform_export_defaults); a versioned
        # export may override them via RXF1 _meta only.
        self.scale_num = np.int64(SCALE_NUM)
        self.scale_shift = SCALE_SHIFT
        self.neural_bound = NEURAL_BOUND

    @classmethod
    def random(cls, seed: int, *, sparse: bool = False) -> EvalWeights:
        """Deterministic bounded-random weights for qualification runs."""
        rng = np.random.default_rng(seed)
        w = cls()

        def i16(shape, lim):
            return rng.integers(-lim, lim + 1, shape, dtype=np.int16)

        def i8(shape, lim):
            return rng.integers(-lim, lim + 1, shape, dtype=np.int8)

        w.psq_w = i16((PSQ_ROWS, CHANNELS), PSQ_COEF_MAX)
        w.thr_w = i8((THREAT_ROWS, CHANNELS), THREAT_COEF_MAX)
        w.pp_w = i8((PP_ROWS, CHANNELS), PP_COEF_MAX)
        w.bias = i16(CHANNELS, BIAS_COEF_MAX)
        w.w1 = i8((HEAD_STACKS, L1_OUT, HEAD_IN), 127)
        w.w2 = i8((HEAD_STACKS, L2_OUT, L1_ACT), 127)
        # FR1/FR4: the head's OUTPUT layer is drawn small so the deployed
        # leaf transform (raw*275 >> 8, clamp ±27951) lands in-band for a
        # large fraction of random-weight positions.  Full-range w3/b3
        # saturate the clamp on ~every position — ref-vs-opt scalar parity
        # would then compare "both clamped" and could not detect a head
        # divergence.  The accumulator lane keeps its full-range stress.
        w.w3 = i8((HEAD_STACKS, HEAD_OUT, SKIP), 8)
        w.b1 = rng.integers(-(1 << 20), 1 << 20, (HEAD_STACKS, L1_OUT), np.int32)
        w.b2 = rng.integers(-(1 << 20), 1 << 20, (HEAD_STACKS, L2_OUT), np.int32)
        w.b3 = rng.integers(-(1 << 14), 1 << 14, (HEAD_STACKS, HEAD_OUT), np.int32)
        w.psqt_w = i16((PSQ_ROWS, PSQT_BUCKETS), 32767)
        w.psqt_b = rng.integers(-(1 << 20), 1 << 20, PSQT_BUCKETS, np.int32)
        if sparse:
            for arr, frac in ((w.psq_w, 0.7), (w.thr_w, 0.5), (w.pp_w, 0.5)):
                mask = rng.random(arr.shape) < frac
                arr[mask] = 0
        return w

    @classmethod
    def from_model(cls, model: dict) -> EvalWeights:
        """EvalWeights from a ``model_io.read_model`` dict.

        The packed model stores head matrices [in][out]; the runtime layout
        is [out][in] — every head matrix is transposed here (including the
        square-shaped w2, whose orientation is not self-evident).
        """
        w = cls()
        w.psq_w = np.ascontiguousarray(model["psq"], dtype=np.int16)
        w.thr_w = np.ascontiguousarray(model["thr"], dtype=np.int8)
        w.pp_w = np.ascontiguousarray(model["pp"], dtype=np.int8)
        w.bias = np.ascontiguousarray(model["bias"], dtype=np.int16)
        w.w1 = np.ascontiguousarray(model["head_w1"].transpose(0, 2, 1), dtype=np.int8)
        w.b1 = np.ascontiguousarray(model["head_b1"], dtype=np.int32)
        w.w2 = np.ascontiguousarray(model["head_w2"].transpose(0, 2, 1), dtype=np.int8)
        w.b2 = np.ascontiguousarray(model["head_b2"], dtype=np.int32)
        w.w3 = np.ascontiguousarray(model["head_w3"].transpose(0, 2, 1), dtype=np.int8)
        w.b3 = np.ascontiguousarray(model["head_b3"], dtype=np.int32)
        w.psqt_w = np.ascontiguousarray(model["psqt_w"], dtype=np.int16)
        w.psqt_b = np.ascontiguousarray(model["psqt_b"], dtype=np.int32)
        meta = model.get("_meta", {})
        w.scale_num = np.int64(meta.get("scale_num", w.scale_num))
        w.scale_shift = int(meta.get("scale_shift", w.scale_shift))
        w.neural_bound = int(meta.get("neural_bound", w.neural_bound))
        w.check_bounds()
        return w

    def check_bounds(self) -> None:
        assert np.abs(self.psq_w.astype(np.int32)).max() <= PSQ_COEF_MAX
        assert np.abs(self.thr_w.astype(np.int32)).max() <= THREAT_COEF_MAX
        assert np.abs(self.pp_w.astype(np.int32)).max() <= PP_COEF_MAX
        assert np.abs(self.bias.astype(np.int32)).max() <= BIAS_COEF_MAX
        for a in (self.w1, self.w2, self.w3):
            assert a.dtype == np.int8


# ============================================================================
# Accumulator application kernels (removals before additions, int16 safe)
# ============================================================================
#
# FR1-F3: the accumulator lane is int32 scratch, not int16.  These kernels
# are signature-pinned to ``int32[:]`` accumulators so a caller can NEVER
# pass the int16 storage array: numba raises TypeError at dispatch instead
# of silently wrapping past 32,767.  Callers own the int16 narrow, gated
# on ``_acc_peak``/``_check_acc`` against the proven 30,048 bound.
_ACC_ROW_SIGS = [
    "void(int32[:], int8[:, :], int32[:], int64)",
    "void(int32[:], int16[:, :], int32[:], int64)",
]


@njit(_ACC_ROW_SIGS, cache=True)
def _acc_sub_rows(acc, table, rows, n):
    for r in range(n):
        row = table[rows[r]]
        for c in range(CHANNELS):
            acc[c] -= row[c]


@njit(_ACC_ROW_SIGS, cache=True)
def _acc_add_rows(acc, table, rows, n):
    for r in range(n):
        row = table[rows[r]]
        for c in range(CHANNELS):
            acc[c] += row[c]


@njit(cache=True)
def _psqt_sub(psqt_acc, table, rows, n):
    for r in range(n):
        for b in range(PSQT_BUCKETS):
            psqt_acc[b] -= table[rows[r], b]


@njit(cache=True)
def _psqt_add(psqt_acc, table, rows, n):
    for r in range(n):
        for b in range(PSQT_BUCKETS):
            psqt_acc[b] += table[rows[r], b]


@njit(cache=True)
def _acc_peak(acc32):
    m = np.int32(0)
    for c in range(CHANNELS):
        v = acc32[c]
        if v < 0:
            v = -v
        if v > m:
            m = v
    return m


@njit(cache=True)
def _apply_ply(
    acc,
    psqt_acc,
    psq_w,
    thr_w,
    pp_w,
    psqt_w,
    tops,
    tn,
    pops,
    pn,
    pawnbb,
    persp,
    bucket,
    orient,
):
    """Whole per-ply delta application as one kernel: resolve rows from the
    recorded op lists, then removals before additions so every intermediate
    state is a subset of a legal position.  Arithmetic runs on an int32
    scratch (``tmp``): a corrupt dirty list (e.g. a duplicated removal) can
    transiently leave the int16 domain, and the per-group ``peak`` lets the
    caller enforce the proven bound BEFORE the int16 store — no silent wrap.
    Returns (rows_touched, n_pp_max, n_thr_max, n_psq_rows, peak)."""
    psq_add = np.empty(16, np.int32)
    psq_rem = np.empty(16, np.int32)
    thr_add = np.empty(THREAT_OP_CAP + 8, np.int32)
    thr_rem = np.empty(THREAT_OP_CAP + 8, np.int32)
    pp_add = np.empty(PP_OP_CAP + 8, np.int32)
    pp_rem = np.empty(PP_OP_CAP + 8, np.int32)
    na_s, nr_s = _psq_rows(pops, pn, persp, bucket, orient, psq_add, psq_rem)
    na_t, nr_t = _threat_rows(tops, tn, persp, orient, thr_add, thr_rem)
    na_p, nr_p = _pp_rows(pawnbb[0], pawnbb[1], pawnbb[2], pawnbb[3], persp, orient, pp_add, pp_rem)
    tmp = np.empty(CHANNELS, np.int32)
    for c in range(CHANNELS):
        tmp[c] = acc[c]
    _acc_sub_rows(tmp, psq_w, psq_rem, nr_s)
    peak = _acc_peak(tmp)
    _acc_sub_rows(tmp, thr_w, thr_rem, nr_t)
    peak = max(peak, _acc_peak(tmp))
    _acc_sub_rows(tmp, pp_w, pp_rem, nr_p)
    peak = max(peak, _acc_peak(tmp))
    _acc_add_rows(tmp, psq_w, psq_add, na_s)
    peak = max(peak, _acc_peak(tmp))
    _acc_add_rows(tmp, thr_w, thr_add, na_t)
    peak = max(peak, _acc_peak(tmp))
    _acc_add_rows(tmp, pp_w, pp_add, na_p)
    peak = max(peak, _acc_peak(tmp))
    if peak <= ACC_BOUND:
        for c in range(CHANNELS):
            acc[c] = np.int16(tmp[c])
        _psqt_sub(psqt_acc, psqt_w, psq_rem, nr_s)
        _psqt_add(psqt_acc, psqt_w, psq_add, na_s)
    return (
        na_s + nr_s + na_t + nr_t + na_p + nr_p,
        max(na_p, nr_p),
        max(na_t, nr_t),
        na_s + nr_s,
        peak,
    )


class BoundViolation(AssertionError):
    pass


def _check_acc(name: str, acc: np.ndarray) -> None:
    peak = int(np.abs(acc.astype(np.int32)).max())
    if peak > ACC_BOUND:
        raise BoundViolation(f"{name}: |acc| {peak} > {ACC_BOUND}")


# ============================================================================
# Paired transform + heads — reference (NumPy) and Numba kernels
# ============================================================================


def paired_transform_ref(acc_p: np.ndarray) -> np.ndarray:
    """acc_p int16[512] -> int32[256] activations in [0,127]."""
    a = np.clip(acc_p[:HALF].astype(np.int32), 0, FT_CLIP_HI)
    b = np.clip(acc_p[HALF:].astype(np.int32), 0, FT_CLIP_HI)
    return (a * b) >> PRODUCT_SHIFT


@njit(cache=True)
def _paired_transform_nb(acc_p, out):
    for i in range(HALF):
        a = np.int32(acc_p[i])
        b = np.int32(acc_p[i + HALF])
        if a < 0:
            a = np.int32(0)
        elif a > FT_CLIP_HI:
            a = np.int32(FT_CLIP_HI)
        if b < 0:
            b = np.int32(0)
        elif b > FT_CLIP_HI:
            b = np.int32(FT_CLIP_HI)
        out[i] = (a * b) >> PRODUCT_SHIFT


def _act32_ref(z: np.ndarray) -> np.ndarray:
    """Canonical hidden activation (features._hidden_activation): the square
    term is computed on the UNCLIPPED shifted affine (x = z>>6), then
    clipped to [0,127] — not the square of the clipped value.  This is the
    pinned SqrClippedReLU semantic: sign-blind, saturating for |x| >= 128."""
    x = z >> HIDDEN_SHIFT
    lin = np.clip(x, 0, HIDDEN_CLIP)
    sq = np.clip((x * x) >> SQUARE_SHIFT, 0, HIDDEN_CLIP)
    return np.concatenate([lin, sq])


def head_ref(w: EvalWeights, x: np.ndarray, bucket: int, want_wdl: bool = False):
    """Reference head. x in [0,127]. Returns int64 scalar or int64[4].
    Weight layout is [out][in]; dots computed in int64 for exact parity."""
    xi = x.astype(np.int64)
    z1 = w.b1[bucket].astype(np.int64) + w.w1[bucket].astype(np.int64) @ xi
    a1 = _act32_ref(z1).astype(np.int64)
    z2 = w.b2[bucket].astype(np.int64) + w.w2[bucket].astype(np.int64) @ a1
    a2 = _act32_ref(z2).astype(np.int64)
    skip = np.concatenate([a1, a2])
    n_out = HEAD_OUT if want_wdl else 1
    out = w.b3[bucket, :n_out].astype(np.int64) + w.w3[bucket, :n_out].astype(np.int64) @ skip
    return out if want_wdl else out[0]


@njit(cache=True)
def _head_scalar_nb(x, w1, b1, w2, b2, w3, b3, bucket):
    a1 = np.empty(L1_ACT, np.int32)
    for o in range(L1_OUT):
        s = b1[bucket, o]
        for i in range(HEAD_IN):
            s += np.int32(w1[bucket, o, i]) * x[i]
        c = s >> HIDDEN_SHIFT
        sq64 = (np.int64(c) * np.int64(c)) >> SQUARE_SHIFT
        sq = np.int32(HIDDEN_CLIP if sq64 > HIDDEN_CLIP else sq64)
        if c < 0:
            c = 0
        elif c > HIDDEN_CLIP:
            c = HIDDEN_CLIP
        a1[o] = c
        a1[o + L1_OUT] = sq
    a2 = np.empty(L2_ACT, np.int32)
    for o in range(L2_OUT):
        s = b2[bucket, o]
        for i in range(L1_ACT):
            s += np.int32(w2[bucket, o, i]) * a1[i]
        c = s >> HIDDEN_SHIFT
        sq64 = (np.int64(c) * np.int64(c)) >> SQUARE_SHIFT
        sq = np.int32(HIDDEN_CLIP if sq64 > HIDDEN_CLIP else sq64)
        if c < 0:
            c = 0
        elif c > HIDDEN_CLIP:
            c = HIDDEN_CLIP
        a2[o] = c
        a2[o + L2_OUT] = sq
    s64 = np.int64(b3[bucket, 0])
    for i in range(SKIP):
        v = a1[i] if i < L1_ACT else a2[i - L1_ACT]
        s64 += np.int64(w3[bucket, 0, i]) * np.int64(v)
    return s64


@njit(cache=True)
def _head_wdl_nb(x, w1, b1, w2, b2, w3, b3, bucket, out):
    a1 = np.empty(L1_ACT, np.int32)
    for o in range(L1_OUT):
        s = b1[bucket, o]
        for i in range(HEAD_IN):
            s += np.int32(w1[bucket, o, i]) * x[i]
        c = s >> HIDDEN_SHIFT
        sq64 = (np.int64(c) * np.int64(c)) >> SQUARE_SHIFT
        sq = np.int32(HIDDEN_CLIP if sq64 > HIDDEN_CLIP else sq64)
        if c < 0:
            c = 0
        elif c > HIDDEN_CLIP:
            c = HIDDEN_CLIP
        a1[o] = c
        a1[o + L1_OUT] = sq
    a2 = np.empty(L2_ACT, np.int32)
    for o in range(L2_OUT):
        s = b2[bucket, o]
        for i in range(L1_ACT):
            s += np.int32(w2[bucket, o, i]) * a1[i]
        c = s >> HIDDEN_SHIFT
        sq64 = (np.int64(c) * np.int64(c)) >> SQUARE_SHIFT
        sq = np.int32(HIDDEN_CLIP if sq64 > HIDDEN_CLIP else sq64)
        if c < 0:
            c = 0
        elif c > HIDDEN_CLIP:
            c = HIDDEN_CLIP
        a2[o] = c
        a2[o + L2_OUT] = sq
    for k in range(HEAD_OUT):
        s64 = np.int64(b3[bucket, k])
        for i in range(L1_ACT):
            s64 += np.int64(w3[bucket, k, i]) * np.int64(a1[i])
        for i in range(L2_ACT):
            s64 += np.int64(w3[bucket, k, L1_ACT + i]) * np.int64(a2[i])
        out[k] = s64


# ============================================================================
# Incremental accumulator stack
# ============================================================================


class AccumulatorOverflow(RuntimeError):
    """A dirty-feature list exceeded its compile-time capacity."""


class Stack:
    """Lazy per-ply accumulator state with measured-cost update selection.

    push() records narrow per-move dirties (SF do_move ordering).  Wide state
    is materialized only by materialize(): valid-ancestor delta replay,
    king-bucket refresh-cache (Finny) differences, or full refresh, chosen by
    measured cost.  A changed king-normalization frame (bucket or mirror of
    that perspective's own king) forces refresh for that perspective only.
    """

    # Historical measured-cost policy, in ns; remeasure before retuning.
    # One replay ply = one _apply_ply kernel: ~5.5us fixed + ~0.35us/row.
    # Full refresh = ~20us cold; a Finny entry for the frame cuts it to ~7us
    # (signature hit ~2us, piece-diff ~10us).  Replay therefore wins short
    # spans; refresh wins once the span or a cold frame makes it cheaper.
    COST_REPLAY_PLY_NS = 5_500.0
    COST_REPLAY_ROW_NS = 350.0
    COST_REFRESH_COLD_NS = 20_000.0
    COST_REFRESH_WARM_NS = 7_000.0

    def __init__(
        self,
        w: EvalWeights,
        max_ply: int = STACK_PLY,
        *,
        check_bounds: bool = False,
        threat_cap: int = THREAT_OP_CAP,
        pp_cap: int = PP_OP_CAP,
        psq_cap: int = PSQ_OP_CAP,
        force_replay: bool = False,
    ) -> None:
        self.w = w
        self.check = check_bounds
        self.force_replay = force_replay
        self.max_ply = max_ply
        self.threat_cap = threat_cap
        self.pp_cap = pp_cap
        self.acc = np.zeros((max_ply, 2, CHANNELS), np.int16)
        self.psqt = np.zeros((max_ply, 2, PSQT_BUCKETS), np.int32)
        self.valid = np.zeros((max_ply, 2), np.bool_)
        self.frame = np.zeros((max_ply, 2), np.uint8)
        self.refresh = np.zeros((max_ply, 2), np.bool_)
        self.tops = np.zeros((max_ply, threat_cap), np.uint32)
        self.tn = np.zeros(max_ply, np.int32)
        self.pops = np.zeros((max_ply, psq_cap, 3), np.int8)
        self.pn = np.zeros(max_ply, np.int32)
        self.pawnbb = np.zeros((max_ply, 4), np.uint64)
        self.depth = 0
        self.finny: list[dict[int, tuple]] = [{}, {}]
        self._psq_part = np.zeros(CHANNELS, np.int32)  # refresh scratch
        self._acc32 = np.zeros(CHANNELS, np.int32)  # widened acc scratch
        # instrumentation
        self.stat_geometry_ns = 0
        self.stat_update_ns = 0
        self.stat_head_ns = 0
        self.stat_rows = 0
        self.stat_bytes = 0
        self.stat_evals = 0
        self.stat_replays = 0
        self.stat_refresh_full = 0
        self.stat_refresh_cache_hit = 0
        self.stat_refresh_cache_diff = 0
        self.stat_refresh_miss = 0
        self.max_threat_ops = 0
        self.max_pp_rows = 0
        self.max_psq_ops = 0
        self._rb_add = np.empty(512, np.int32)
        self._rb_rem = np.empty(512, np.int32)
        self._thr_enum = np.empty(MAX_ACTIVE_THREATS, np.int32)
        self._pp_enum = np.empty(160, np.int32)
        self._pp_buf_a = np.empty(PP_OP_CAP, np.int32)
        self._pp_buf_r = np.empty(PP_OP_CAP, np.int32)
        # shadow board state (the feature view's own copy — Stockfish style)
        self._smb = np.full(64, EMPTY, np.int8)
        self._sbb = np.zeros(12, np.uint64)
        self._socc = np.zeros(2, np.uint64)
        self._socc_all = U0
        self._sking = np.zeros(2, np.int64)

    # -- lifecycle ---------------------------------------------------------

    def set_root(self, board: Board) -> None:
        self.depth = 0
        self.finny[0].clear()
        self.finny[1].clear()
        self.valid[0] = False
        self._smb = np.asarray(board._sq, dtype=np.int8).copy()
        self._sbb = np.asarray(board._bb, dtype=np.uint64).copy()
        self._socc = np.asarray(board._occ, dtype=np.uint64).copy()
        self._socc_all = np.uint64(board._occ_all)
        self._sking[0] = board._king[0]
        self._sking[1] = board._king[1]
        for p in (0, 1):
            self.frame[0, p] = _frame(board._king[p], p)
            self.refresh[0, p] = True

    def assert_shadow(self, board: Board) -> None:
        """Debug check: shadow state must mirror the real board."""
        assert np.array_equal(self._smb, np.asarray(board._sq, dtype=np.int8))
        assert np.array_equal(self._sbb, np.asarray(board._bb, dtype=np.uint64))
        assert self._socc_all == np.uint64(board._occ_all)
        assert int(self._sking[0]) == board._king[0]
        assert int(self._sking[1]) == board._king[1]

    def push(self, board: Board, move: int) -> None:
        """Record narrow dirties for ``move``. Call BEFORE board.make()."""
        t0 = time.perf_counter_ns()
        frm, to, promo, flag, piece, captured = decode_move(move)
        us = board.side
        k = self.depth + 1
        nt, npo, pwb, pbb, pwa, pba, occ = _compute_dirties(
            self._smb,
            self._sbb,
            self._socc,
            self._socc_all,
            frm,
            to,
            promo,
            flag,
            piece,
            captured,
            us,
            self.tops[k],
            self.pops[k],
        )
        self._socc_all = np.uint64(occ)
        if nt > self.threat_cap:
            raise AccumulatorOverflow(f"threat ops {nt} > {self.threat_cap}")
        if npo > self.pops.shape[1]:
            raise AccumulatorOverflow(f"psq ops {npo} > {self.pops.shape[1]}")
        self.tn[k] = nt
        self.pn[k] = npo
        self.pawnbb[k] = (pwb, pbb, pwa, pba)
        self.max_threat_ops = max(self.max_threat_ops, nt)
        self.max_psq_ops = max(self.max_psq_ops, npo)
        if piece % 6 == KING:
            self._sking[us] = to
        self.frame[k, WHITE] = _frame(int(self._sking[WHITE]), WHITE)
        self.frame[k, BLACK] = _frame(int(self._sking[BLACK]), BLACK)
        self.refresh[k, WHITE] = self.frame[k, WHITE] != self.frame[self.depth, WHITE]
        self.refresh[k, BLACK] = self.frame[k, BLACK] != self.frame[self.depth, BLACK]
        self.valid[k] = False
        self.depth = k
        self.stat_geometry_ns += time.perf_counter_ns() - t0

    def pop(self) -> None:
        k = self.depth
        if self.pn[k] > 0:
            self._socc_all = np.uint64(
                _reverse_ops(
                    self._smb,
                    self._sbb,
                    self._socc,
                    self._socc_all,
                    self.pops[k],
                    self.pn[k],
                )
            )
            for i in range(self.pn[k]):
                if (
                    self.pops[k, i, 0] >= 0
                    and self.pops[k, i, 0] % 6 == KING
                    and self.pops[k, i, 1] >= 0
                ):
                    pc = int(self.pops[k, i, 0])
                    self._sking[pc // 6] = int(self.pops[k, i, 1])
        self.depth -= 1

    # -- materialization ----------------------------------------------------

    def _diff_rows(self, k: int, p: int):
        f = int(self.frame[k, p])
        orient = ((f & 1) * 7) ^ (56 * p)
        bucket = f >> 1
        ra, rr = self._rb_add, self._rb_rem
        na_psq, nr_psq = _psq_rows(self.pops[k], self.pn[k], p, bucket, orient, ra, rr)
        psq_add, psq_rem = ra[:na_psq].copy(), rr[:nr_psq].copy()
        na_t, nr_t = _threat_rows(self.tops[k], self.tn[k], p, orient, ra, rr)
        if na_t > MAX_ACTIVE_THREATS or nr_t > MAX_ACTIVE_THREATS:
            raise AccumulatorOverflow(f"threat rows {na_t}+{nr_t}")
        thr_add, thr_rem = ra[:na_t].copy(), rr[:nr_t].copy()
        pwb, pbb, pwa, pba = self.pawnbb[k]
        na_p, nr_p = _pp_rows(pwb, pbb, pwa, pba, p, orient, self._pp_buf_a, self._pp_buf_r)
        if na_p > self.pp_cap or nr_p > self.pp_cap:
            raise AccumulatorOverflow(f"pp rows +{na_p}/-{nr_p} > {self.pp_cap}")
        self.max_pp_rows = max(self.max_pp_rows, na_p, nr_p)
        pp_add = self._pp_buf_a[:na_p].copy()
        pp_rem = self._pp_buf_r[:nr_p].copy()
        return psq_add, psq_rem, thr_add, thr_rem, pp_add, pp_rem

    def _apply_diff(self, k: int, p: int) -> None:
        f = int(self.frame[k, p])
        orient = ((f & 1) * 7) ^ (56 * p)
        bucket = f >> 1
        acc = self.acc[self.depth, p]
        w = self.w
        rows_touched, pp_max, thr_max, psq_rows, peak = _apply_ply(
            acc,
            self.psqt[self.depth, p],
            w.psq_w,
            w.thr_w,
            w.pp_w,
            w.psqt_w,
            self.tops[k],
            self.tn[k],
            self.pops[k],
            self.pn[k],
            self.pawnbb[k],
            p,
            bucket,
            orient,
        )
        if pp_max > self.pp_cap:
            raise AccumulatorOverflow(f"pp rows {pp_max} > {self.pp_cap}")
        if thr_max > MAX_ACTIVE_THREATS:
            raise AccumulatorOverflow(f"threat rows {thr_max}")
        if peak > ACC_BOUND:
            raise BoundViolation(f"diff ply {k} persp {p}: |acc| {peak} > {ACC_BOUND}")
        self.max_pp_rows = max(self.max_pp_rows, pp_max)
        self.stat_rows += int(rows_touched)
        # psq rows are int16[512] (1KB); threat/pp rows are int8[512] (0.5KB)
        self.stat_bytes += psq_rows * 1024 + (int(rows_touched) - psq_rows) * 512
        if self.check:
            _check_acc(f"diff ply {k} persp {p}", acc)

    def _board_arrays(self, board: Board):
        return (
            np.asarray(board._sq, dtype=np.int8),
            np.asarray(board._bb, dtype=np.uint64),
            np.uint64(board._occ_all),
        )

    def _refresh(self, k: int, p: int, board: Board) -> None:
        """Recompute perspective p at ply k through the king-bucket refresh
        cache.  Relation rows are always enumerated fresh in the current
        frame — a changed king-normalization frame invalidates every
        relation index, so no relation cache survives it.  The cache only
        carries the PSQ part (piece-local rows) plus the last full state for
        exact piece-set hits."""
        mb, bb, occ = self._board_arrays(board)
        f = int(self.frame[k, p])
        orient = ((f & 1) * 7) ^ (56 * p)
        bucket = f >> 1
        w = self.w
        sig = bb.tobytes()
        ent = self.finny[p].get(f)
        acc = self.acc[k, p]
        psqt = self.psqt[k, p]
        if ent is not None and ent[0] == sig:
            acc[:] = ent[2]
            psqt[:] = ent[4]
            self.stat_refresh_cache_hit += 1
            self.stat_refresh_full += 1
            return
        n_thr = n_pp = 0
        if ent is not None:
            # Finny difference path needs the relation rows enumerated.
            n_thr = _enum_threats(mb, bb, occ, p, orient, self._thr_enum)
            if n_thr > MAX_ACTIVE_THREATS:
                raise AccumulatorOverflow(f"active threats {n_thr} > {MAX_ACTIVE_THREATS}")
            n_pp = _enum_pp(bb[0], bb[6], p, orient, self._pp_enum)
            if n_pp > MAX_ACTIVE_PP:
                raise AccumulatorOverflow(f"active pawn pairs {n_pp} > {MAX_ACTIVE_PP}")
            # Finny difference: keep the cached PSQ part, subtract rows for
            # pieces that left and add rows for pieces that arrived, all in
            # this frame's row space.
            old_bb = ent[1]
            psq_part = ent[3].astype(np.int32).copy()
            psqt_part = ent[4].astype(np.int64)
            _check_acc(f"refresh finny-diff ply {k} persp {p} (cached part)", psq_part)
            n_changed = 0
            for pc in range(12):
                rem_b = int(old_bb[pc] & ~bb[pc])
                add_b = int(bb[pc] & ~old_bb[pc])
                if not (rem_b or add_b):
                    continue
                relc = 1 if (pc // 6) != p else 0
                base = 768 * bucket + 384 * relc + 64 * (pc % 6)
                while rem_b:
                    s = (rem_b & -rem_b).bit_length() - 1
                    rem_b &= rem_b - 1
                    r = base + (s ^ orient)
                    psq_part -= w.psq_w[r]
                    psqt_part -= w.psqt_w[r]
                    n_changed += 1
                while add_b:
                    s = (add_b & -add_b).bit_length() - 1
                    add_b &= add_b - 1
                    r = base + (s ^ orient)
                    psq_part += w.psq_w[r]
                    psqt_part += w.psqt_w[r]
                    n_changed += 1
            self.stat_refresh_cache_diff += 1
            self.stat_rows += n_changed + n_thr + n_pp
            # Bound-checked int32 path: the cached part or the piece diff
            # can leave the int16 domain on corrupt input; check before
            # every narrow so no out-of-range value silently wraps.
            _check_acc(f"refresh finny-diff ply {k} persp {p} (psq part)", psq_part)
            fused = self._acc32
            fused[:] = psq_part
            _acc_add_rows(fused, w.thr_w, self._thr_enum, n_thr)
            _check_acc(f"refresh finny-diff ply {k} persp {p} (+thr)", fused)
            _acc_add_rows(fused, w.pp_w, self._pp_enum, n_pp)
            _check_acc(f"refresh finny-diff ply {k} persp {p} (+pp)", fused)
            acc[:] = fused.astype(np.int16)
            psqt[:] = psqt_part.astype(np.int32)
        else:
            self.stat_refresh_miss += 1
            psq_part = self._psq_part  # int32[512] scratch, stored in finny
            n_psq, n_thr, n_pp, peak = _enum_apply_persp(
                acc,
                psqt,
                psq_part,
                mb,
                bb,
                occ,
                p,
                bucket,
                orient,
                w.bias,
                w.psq_w,
                w.thr_w,
                w.pp_w,
                w.psqt_w,
                self._rb_add,
                self._thr_enum,
                self._pp_enum,
            )
            if n_thr > MAX_ACTIVE_THREATS:
                raise AccumulatorOverflow(f"active threats {n_thr} > {MAX_ACTIVE_THREATS}")
            if n_pp > MAX_ACTIVE_PP:
                raise AccumulatorOverflow(f"active pawn pairs {n_pp} > {MAX_ACTIVE_PP}")
            if peak > ACC_BOUND:
                raise BoundViolation(f"refresh ply {k} persp {p}: |acc| {peak} > {ACC_BOUND}")
            self.stat_rows += n_psq + n_thr + n_pp
            self.stat_bytes += n_psq * 1024
        self.finny[p][f] = (sig, bb.copy(), acc.copy(), psq_part.copy(), psqt.copy())
        self.stat_refresh_full += 1
        self.stat_bytes += (n_thr + n_pp) * 512
        if self.check:
            _check_acc(f"refresh ply {k} persp {p}", acc)

    def _ensure(self, p: int, board: Board) -> None:
        n = self.depth
        if self.valid[n, p]:
            return
        last_refresh = 0
        for j in range(n, -1, -1):
            if self.refresh[j, p]:
                last_refresh = j
                break
        anc = -1
        for j in range(n, last_refresh - 1, -1):
            if self.valid[j, p]:
                anc = j
                break
        if anc >= 0:
            span_plies = n - anc
            span_rows = int(self.tn[anc + 1 : n + 1].sum() + self.pn[anc + 1 : n + 1].sum())
            replay_ns = span_plies * self.COST_REPLAY_PLY_NS + span_rows * self.COST_REPLAY_ROW_NS
            f = int(self.frame[n, p])
            refresh_ns = (
                self.COST_REFRESH_WARM_NS if f in self.finny[p] else self.COST_REFRESH_COLD_NS
            )
            if self.force_replay or replay_ns <= refresh_ns:
                self.acc[n, p] = self.acc[anc, p]
                self.psqt[n, p] = self.psqt[anc, p]
                for k in range(anc + 1, n + 1):
                    self._apply_diff(k, p)
                self.stat_replays += 1
                self.valid[n, p] = True
                return
        self._refresh(n, p, board)
        self.valid[n, p] = True

    def materialize(self, board: Board) -> tuple[np.ndarray, np.ndarray]:
        """(acc int16[2,512], psqt int32[2,8]) for the current position.
        The PSQT pair is per-perspective; callers combine it canonically as
        psqt_b[s] + trunc((psqt[stm] - psqt[ntm])[s] / 2) (features.py)."""
        t0 = time.perf_counter_ns()
        for p in (0, 1):
            self._ensure(p, board)
        self.stat_update_ns += time.perf_counter_ns() - t0
        return self.acc[self.depth], self.psqt[self.depth]


# ============================================================================
# Evaluator facade
# ============================================================================


def material_bucket(board: Board) -> int:
    return _features.material_stack(board._occ_all.bit_count())


def _trunc_div2(v: int) -> int:
    """C-style integer division by 2 (truncation toward zero)."""
    return -((-v) // 2) if v < 0 else v // 2


def _psqt_term(w: EvalWeights, psqt_pair: np.ndarray, bucket: int, stm: int) -> int:
    """Canonical PSQT skip: bias[s] + trunc((stm - ntm)[s] / 2)."""
    d = int(psqt_pair[stm, bucket]) - int(psqt_pair[stm ^ 1, bucket])
    return int(w.psqt_b[bucket]) + _trunc_div2(d)


def _final_scalar(w: EvalWeights, dot: int, psqt_term: int) -> int:
    v = (np.int64(dot) + np.int64(psqt_term)) * w.scale_num >> np.int64(w.scale_shift)
    b = w.neural_bound
    return int(max(-b, min(b, v)))


class Evaluator:
    """Lazy incremental evaluator (the production path)."""

    def __init__(
        self, w: EvalWeights, max_ply: int = STACK_PLY, *, check_bounds: bool = False
    ) -> None:
        self.w = w
        self.stack = Stack(w, max_ply, check_bounds=check_bounds)
        self._x = np.empty(HEAD_IN, np.int32)
        self._wdl = np.empty(HEAD_OUT, np.int64)

    def set_root(self, board: Board) -> None:
        self.stack.set_root(board)

    def push(self, board: Board, move: int) -> None:
        self.stack.push(board, move)

    def pop(self) -> None:
        self.stack.pop()

    def evaluate(self, board: Board) -> int:
        st = self.stack
        acc2, psqt_pair = st.materialize(board)
        t0 = time.perf_counter_ns()
        x = self._x
        _paired_transform_nb(acc2[board.side], x[:HALF])
        _paired_transform_nb(acc2[board.side ^ 1], x[HALF:])
        bucket = material_bucket(board)
        dot = _head_scalar_nb(
            x, self.w.w1, self.w.b1, self.w.w2, self.w.b2, self.w.w3, self.w.b3, bucket
        )
        st.stat_head_ns += time.perf_counter_ns() - t0
        st.stat_bytes += 16 * 512 + 32 * 32 + 96 + 512 * 2
        st.stat_evals += 1
        return _final_scalar(self.w, dot, _psqt_term(self.w, psqt_pair, bucket, board.side))

    def evaluate_wdl(self, board: Board) -> np.ndarray:
        """Auxiliary path: full 4-logit head. NOT on the scalar hot path."""
        acc2, psqt_pair = self.stack.materialize(board)
        x = self._x
        _paired_transform_nb(acc2[board.side], x[:HALF])
        _paired_transform_nb(acc2[board.side ^ 1], x[HALF:])
        bucket = material_bucket(board)
        _head_wdl_nb(
            x,
            self.w.w1,
            self.w.b1,
            self.w.w2,
            self.w.b2,
            self.w.w3,
            self.w.b3,
            bucket,
            self._wdl,
        )
        out = self._wdl.copy()
        out[0] = _final_scalar(
            self.w, int(out[0]), _psqt_term(self.w, psqt_pair, bucket, board.side)
        )
        return out


# ============================================================================
# Fast full-refresh path (Numba) — used by the differential gates; an
# independent pure-Python reference follows below.
# ============================================================================

_ENUM_PSQ_BUF = np.empty(40, np.int32)
_ENUM_THR_BUF = np.empty(MAX_ACTIVE_THREATS + 8, np.int32)
_ENUM_PP_BUF = np.empty(160, np.int32)


def refresh_acc_nb(w: EvalWeights, board: Board, persp: int) -> np.ndarray:
    """From-scratch accumulator via the Numba enumerators. Shares nothing
    with the incremental stack."""
    mb = np.asarray(board._sq, dtype=np.int8)
    bb = np.asarray(board._bb, dtype=np.uint64)
    occ = np.uint64(board._occ_all)
    f = _frame(board._king[persp], persp)
    orient = ((f & 1) * 7) ^ (56 * persp)
    bucket = f >> 1
    n_psq = _enum_psq(mb, persp, bucket, orient, _ENUM_PSQ_BUF)
    n_thr = _enum_threats(mb, bb, occ, persp, orient, _ENUM_THR_BUF)
    if n_thr > MAX_ACTIVE_THREATS:
        raise AccumulatorOverflow(f"active threats {n_thr} > {MAX_ACTIVE_THREATS}")
    n_pp = _enum_pp(bb[0], bb[6], persp, orient, _ENUM_PP_BUF)
    if n_pp > MAX_ACTIVE_PP:
        raise AccumulatorOverflow(f"active pawn pairs {n_pp} > {MAX_ACTIVE_PP}")
    acc32 = w.bias.astype(np.int32).copy()
    _acc_add_rows(acc32, w.psq_w, _ENUM_PSQ_BUF, n_psq)
    _check_acc(f"refresh_nb persp {persp} (+psq)", acc32)
    _acc_add_rows(acc32, w.thr_w, _ENUM_THR_BUF, n_thr)
    _check_acc(f"refresh_nb persp {persp} (+thr)", acc32)
    _acc_add_rows(acc32, w.pp_w, _ENUM_PP_BUF, n_pp)
    _check_acc(f"refresh_nb persp {persp} (+pp)", acc32)
    return acc32.astype(np.int16)


def psqt_nb(w: EvalWeights, board: Board) -> np.ndarray:
    """Per-perspective PSQT row sums, int32[2,8] (index = perspective)."""
    out = np.zeros((2, PSQT_BUCKETS), np.int32)
    for persp in (0, 1):
        mb = np.asarray(board._sq, dtype=np.int8)
        f = _frame(board._king[persp], persp)
        orient = ((f & 1) * 7) ^ (56 * persp)
        n_psq = _enum_psq(mb, persp, f >> 1, orient, _ENUM_PSQ_BUF)
        _psqt_add(out[persp], w.psqt_w, _ENUM_PSQ_BUF, n_psq)
    return out


def evaluate_fresh(w: EvalWeights, board: Board) -> int:
    """Scalar via full Numba refresh (no incremental state)."""
    accs = np.empty((2, CHANNELS), np.int16)
    accs[0] = refresh_acc_nb(w, board, 0)
    accs[1] = refresh_acc_nb(w, board, 1)
    x = np.empty(HEAD_IN, np.int32)
    _paired_transform_nb(accs[board.side], x[:HALF])
    _paired_transform_nb(accs[board.side ^ 1], x[HALF:])
    bucket = material_bucket(board)
    dot = _head_scalar_nb(x, w.w1, w.b1, w.w2, w.b2, w.w3, w.b3, bucket)
    return _final_scalar(w, int(dot), _psqt_term(w, psqt_nb(w, board), bucket, board.side))


# ============================================================================
# Reference evaluator — independent NumPy path used for kernel parity gates.
# ============================================================================


def refresh_acc_ref(w: EvalWeights, board: Board, persp: int) -> np.ndarray:
    """Full-refresh accumulator via the pure-Python encoders."""
    acc = np.zeros(CHANNELS, np.int64)
    acc += w.bias
    ksq = board._king[persp]
    mb = board._sq
    for sq in range(64):
        pc = mb[sq]
        if pc < 0:
            continue
        acc += w.psq_w[psq_row_py(persp, pc, sq, ksq)]
    occ = board._occ_all
    pawn_t = board._bb[1] | board._bb[7] | board._bb[3] | board._bb[9]
    minor_t = pawn_t | board._bb[0] | board._bb[6] | board._bb[2] | board._bb[8]
    queen_t = minor_t | board._bb[4] | board._bb[10]
    for sq in range(64):
        pc = mb[sq]
        if pc < 0:
            continue
        pt, c = pc % 6, pc // 6
        if pt == PAWN:
            t = PAWN_ATK[c][sq] & pawn_t
        elif pt == KNIGHT:
            t = KNIGHT_ATK[sq] & queen_t
        elif pt == BISHOP:
            t = bishop_attacks(sq, occ) & minor_t
        elif pt == ROOK:
            t = rook_attacks(sq, occ) & minor_t
        elif pt == QUEEN:
            t = (bishop_attacks(sq, occ) | rook_attacks(sq, occ)) & queen_t
        else:
            continue
        while t:
            to = (t & -t).bit_length() - 1
            t &= t - 1
            row = threat_row_py(persp, pc, sq, to, mb[to], ksq)
            if row >= 0:
                acc += w.thr_w[row]
    pws = [s for s in range(64) if mb[s] == 0]
    pbs = [s for s in range(64) if mb[s] == 6]
    for i, a in enumerate(pws):
        for bsq in pws[i + 1 :]:
            if PP_MASK_A[a] & (1 << bsq):
                r = pp_row_py(persp, 0, a, 0, bsq, ksq)
                if r >= 0:
                    acc += w.pp_w[r]
        for bsq in pbs:
            if PP_MASK_A[a] & (1 << bsq):
                r = pp_row_py(persp, 0, a, 1, bsq, ksq)
                if r >= 0:
                    acc += w.pp_w[r]
    for i, a in enumerate(pbs):
        for bsq in pbs[i + 1 :]:
            if PP_MASK_A[a] & (1 << bsq):
                r = pp_row_py(persp, 1, a, 1, bsq, ksq)
                if r >= 0:
                    acc += w.pp_w[r]
    peak = int(np.abs(acc).max())
    if peak > ACC_BOUND:
        raise BoundViolation(f"refresh persp {persp}: |acc| {peak} > {ACC_BOUND}")
    return acc.astype(np.int16)


def psqt_ref(w: EvalWeights, board: Board) -> np.ndarray:
    """Per-perspective PSQT row sums, int32[2,8] (index = perspective)."""
    out = np.zeros((2, PSQT_BUCKETS), np.int64)
    for persp in (0, 1):
        ksq = board._king[persp]
        for sq in range(64):
            pc = board._sq[sq]
            if pc < 0:
                continue
            out[persp] += w.psqt_w[psq_row_py(persp, pc, sq, ksq)]
    return out.astype(np.int32)


def evaluate_ref(w: EvalWeights, board: Board) -> int:
    """Scalar reference: full refresh + NumPy head. No WDL computed."""
    accs = (refresh_acc_ref(w, board, WHITE), refresh_acc_ref(w, board, BLACK))
    stm, opp = board.side, board.side ^ 1
    x = np.empty(HEAD_IN, np.int64)
    x[:HALF] = paired_transform_ref(accs[stm])
    x[HALF:] = paired_transform_ref(accs[opp])
    bucket = material_bucket(board)
    dot = head_ref(w, x, bucket)
    return _final_scalar(w, int(dot), _psqt_term(w, psqt_ref(w, board), bucket, board.side))


def evaluate_wdl_ref(w: EvalWeights, board: Board) -> np.ndarray:
    accs = (refresh_acc_ref(w, board, WHITE), refresh_acc_ref(w, board, BLACK))
    stm, opp = board.side, board.side ^ 1
    x = np.empty(HEAD_IN, np.int64)
    x[:HALF] = paired_transform_ref(accs[stm])
    x[HALF:] = paired_transform_ref(accs[opp])
    bucket = material_bucket(board)
    out = head_ref(w, x, bucket, want_wdl=True).astype(np.int64)
    out[0] = _final_scalar(w, int(out[0]), _psqt_term(w, psqt_ref(w, board), bucket, board.side))
    return out

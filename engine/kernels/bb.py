"""Compiled board + movegen kernels (nopython).

Transliterated operation-for-operation from ``engine/board.py`` and
``engine/movegen.py`` — same emission order, same Zobrist updates, same
pin/EP/castling semantics — so the compiled path is bit-identical to the
pure-Python oracle.  All state lives in the flat arenas from
``engine.kernels.layout``; functions take ``ctx`` (the 8-tuple) and index
regions with the ``X_*``/``I_*``/``J_*`` constants.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from engine.kernels.layout import (
    A32,
    A8,
    AI,
    AU,
    I_ABSPLY,
    I_CASTLE,
    I_EP,
    I_FULL,
    I_HALF,
    I_SIDE,
    I_UN,
    J_EPKEY,
    J_KEY,
    TABLES,
    U_CASTLE,
    U_EP,
    U_FULL,
    U_HALF,
    U_KING,
    U_PLY,
    X_BB,
    X_KING,
    X_MB,
    X_OCC,
    X_PARAMS,
    X_ST,
    X_U64,
    X_UKEY,
    X_U_MISC,
    X_U_MOVE,
)

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5
WP, WN, WB, WR, WQ, WK = 0, 1, 2, 3, 4, 5
BP, BN, BB_, BR, BQ, BK = 6, 7, 8, 9, 10, 11
WHITE, BLACK = 0, 1
WKC, WQC, BKC, BQC = 1, 2, 4, 8
FLAG_NORMAL = 0
FLAG_CAPTURE = 1
FLAG_DOUBLE = 2
FLAG_EP = 3
FLAG_CASTLE = 4
FLAG_PROMO = 5
FLAG_PROMO_CAP = 6
FLAG_NULL = 7
EMPTY = -1

U0 = np.uint64(0)
U1 = np.uint64(1)
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)

_PAWN_ATK = TABLES["PAWN_ATK"]
_KNIGHT_ATK = TABLES["KNIGHT_ATK"]
_KING_ATK = TABLES["KING_ATK"]
_BETWEEN = TABLES["BETWEEN"]
_LINE = TABLES["LINE"]
_RPOS = TABLES["RPOS"]
_RNEG = TABLES["RNEG"]
_BPOS = TABLES["BPOS"]
_BNEG = TABLES["BNEG"]
_Z_PIECE = TABLES["Z_PIECE"]
_Z_CASTLE = TABLES["Z_CASTLE"]
_Z_EP = TABLES["Z_EP"]
_Z_SIDE = TABLES["Z_SIDE"]
_CCLEAR = TABLES["CASTLE_CLEAR"]
_POP16 = TABLES["POP16"]
_MSB8 = TABLES["MSB8"]

# Full (occupancy-independent) ray sets — engine.search.ROOK_RAYS/BISHOP_RAYS.
_ROOK_RAYS = np.zeros(64, np.uint64)
_BISHOP_RAYS = np.zeros(64, np.uint64)
for _sq in range(64):
    _ROOK_RAYS[_sq] = _RPOS[_sq, 0] | _RPOS[_sq, 1] | _RNEG[_sq, 0] | _RNEG[_sq, 1]
    _BISHOP_RAYS[_sq] = _BPOS[_sq, 0] | _BPOS[_sq, 1] | _BNEG[_sq, 0] | _BNEG[_sq, 1]


# ---------------------------------------------------------------------------
# bit helpers (int64 results — keeps int arithmetic uniform downstream)
# ---------------------------------------------------------------------------


@njit(cache=True)
def k_pop64(b):
    return np.int64(
        _POP16[np.int64(b & np.uint64(0xFFFF))]
        + _POP16[np.int64((b >> 16) & np.uint64(0xFFFF))]
        + _POP16[np.int64((b >> 32) & np.uint64(0xFFFF))]
        + _POP16[np.int64(b >> 48)]
    )


@njit(cache=True)
def k_lsb(b):
    return k_pop64((b & (U0 - b)) - U1)


@njit(cache=True)
def k_msb(b):
    r = np.int64(0)
    if b >= np.uint64(0x100000000):
        r += 32
        b >>= 32
    if b >= np.uint64(0x10000):
        r += 16
        b >>= 16
    if b >= np.uint64(0x100):
        r += 8
        b >>= 8
    return np.int64(r + _MSB8[np.int64(b)])


@njit(cache=True)
def k_ray_attacks(sq, occ, pos, neg):
    atk = U0
    for i in range(2):
        ray = pos[sq, i]
        hits = ray & occ
        if hits:
            b = k_lsb(hits)
            atk |= _BETWEEN[sq, b] | (U1 << b)
        else:
            atk |= ray
    for i in range(2):
        ray = neg[sq, i]
        hits = ray & occ
        if hits:
            b = k_msb(hits)
            atk |= _BETWEEN[sq, b] | (U1 << b)
        else:
            atk |= ray
    return atk


@njit(cache=True)
def k_bishop_attacks(sq, occ):
    return k_ray_attacks(sq, occ, _BPOS, _BNEG)


@njit(cache=True)
def k_rook_attacks(sq, occ):
    return k_ray_attacks(sq, occ, _RPOS, _RNEG)


@njit(cache=True)
def k_queen_attacks(sq, occ):
    return k_bishop_attacks(sq, occ) | k_rook_attacks(sq, occ)


# ---------------------------------------------------------------------------
# attack queries — mirror movegen.square_attacked / _pin_mask / _ep_legal
# ---------------------------------------------------------------------------


@njit(cache=True)
def k_square_attacked(ctx, sq, by, occ):
    U = ctx[AU]
    if _PAWN_ATK[by ^ 1, sq] & U[X_BB + by * 6 + PAWN]:
        return True
    if _KNIGHT_ATK[sq] & U[X_BB + by * 6 + KNIGHT]:
        return True
    if _KING_ATK[sq] & U[X_BB + by * 6 + KING]:
        return True
    if k_bishop_attacks(sq, occ) & (U[X_BB + by * 6 + BISHOP] | U[X_BB + by * 6 + QUEEN]):
        return True
    if k_rook_attacks(sq, occ) & (U[X_BB + by * 6 + ROOK] | U[X_BB + by * 6 + QUEEN]):
        return True
    return False


@njit(cache=True)
def k_in_check(ctx):
    us = ctx[AI][X_ST + I_SIDE]
    return k_square_attacked(ctx, ctx[AI][X_KING + us], us ^ 1, ctx[AU][X_OCC + 2])


@njit(cache=True)
def k_pin_mask(ctx):
    U = ctx[AU]
    L = ctx[AI]
    us = L[X_ST + I_SIDE]
    them = us ^ 1
    king = L[X_KING + us]
    occ = U[X_OCC + 2]
    us_occ = U[X_OCC + us]
    pinned = U0
    snipers = U[X_BB + them * 6 + ROOK] | U[X_BB + them * 6 + QUEEN]
    while snipers:
        s = k_lsb(snipers)
        snipers &= snipers - U1
        df = (s & 7) - (king & 7)
        dr = (s >> 3) - (king >> 3)
        if df != 0 and dr != 0:
            continue
        blockers = occ & _BETWEEN[king, s]
        if blockers and k_pop64(blockers) == 1 and blockers & us_occ:
            pinned |= blockers
    snipers = U[X_BB + them * 6 + BISHOP] | U[X_BB + them * 6 + QUEEN]
    while snipers:
        s = k_lsb(snipers)
        snipers &= snipers - U1
        df = (s & 7) - (king & 7)
        dr = (s >> 3) - (king >> 3)
        if df == 0 or (df if df >= 0 else -df) != (dr if dr >= 0 else -dr):
            continue
        blockers = occ & _BETWEEN[king, s]
        if blockers and k_pop64(blockers) == 1 and blockers & us_occ:
            pinned |= blockers
    return pinned


@njit(cache=True)
def k_ep_legal(ctx, frm, ep):
    L = ctx[AI]
    U = ctx[AU]
    us = L[X_ST + I_SIDE]
    them = us ^ 1
    cap = ep - 8 if us == WHITE else ep + 8
    occ = U[X_OCC + 2] ^ (U1 << frm) ^ (U1 << ep) ^ (U1 << cap)
    king = L[X_KING + us]
    pawn = U[X_BB + them * 6 + PAWN] & ~(U1 << cap)
    if _PAWN_ATK[us, king] & pawn:
        return False
    if _KNIGHT_ATK[king] & U[X_BB + them * 6 + KNIGHT]:
        return False
    if _KING_ATK[king] & U[X_BB + them * 6 + KING]:
        return False
    if k_bishop_attacks(king, occ) & (U[X_BB + them * 6 + BISHOP] | U[X_BB + them * 6 + QUEEN]):
        return False
    if k_rook_attacks(king, occ) & (U[X_BB + them * 6 + ROOK] | U[X_BB + them * 6 + QUEEN]):
        return False
    return True


@njit(cache=True)
def k_has_legal_ep(ctx):
    L = ctx[AI]
    ep = L[X_ST + I_EP]
    if ep < 0:
        return False
    us = L[X_ST + I_SIDE]
    capturers = _PAWN_ATK[us ^ 1, ep] & ctx[AU][X_BB + us * 6 + PAWN]
    if us == WHITE:
        capturers &= np.uint64(0x000000FF00000000)
    else:
        capturers &= np.uint64(0x00000000FF000000)
    while capturers:
        frm = k_lsb(capturers)
        capturers &= capturers - U1
        if k_ep_legal(ctx, frm, ep):
            return True
    return False


@njit(cache=True)
def k_ep_zobrist(ctx):
    if k_has_legal_ep(ctx):
        return _Z_EP[ctx[AI][X_ST + I_EP] & 7]
    return U0


# ---------------------------------------------------------------------------
# piece place/remove on the kernel arenas
# ---------------------------------------------------------------------------


@njit(cache=True)
def _k_place(ctx, sq, piece):
    U = ctx[AU]
    ctx[A8][X_MB + sq] = np.int8(piece)
    bit = U1 << sq
    U[X_BB + piece] |= bit
    U[X_OCC + piece // 6] |= bit
    U[X_OCC + 2] |= bit


@njit(cache=True)
def _k_remove(ctx, sq, piece):
    U = ctx[AU]
    ctx[A8][X_MB + sq] = np.int8(EMPTY)
    bit = U1 << sq
    U[X_BB + piece] &= M64 ^ bit
    U[X_OCC + piece // 6] &= M64 ^ bit
    U[X_OCC + 2] &= M64 ^ bit


# ---------------------------------------------------------------------------
# make / unmake — mirrors Board.make / Board.unmake including the
# ep-square Zobrist (which requires the legal-EP test each make).
# ---------------------------------------------------------------------------


@njit(cache=True)
def k_make(ctx, move):
    U = ctx[AU]
    L = ctx[AI]
    N = ctx[A32]
    u = L[X_ST + I_UN]
    N[X_U_MOVE + u] = np.int32(move)
    um = u * 6 + X_U_MISC
    N[um + U_CASTLE] = np.int32(L[X_ST + I_CASTLE])
    N[um + U_EP] = np.int32(L[X_ST + I_EP])
    N[um + U_HALF] = np.int32(L[X_ST + I_HALF])
    N[um + U_FULL] = np.int32(L[X_ST + I_FULL])
    N[um + U_KING] = np.int32(L[X_KING + L[X_ST + I_SIDE]])
    N[um + U_PLY] = np.int32(L[X_ST + I_ABSPLY])
    U[X_UKEY + u * 2] = U[X_U64 + J_KEY]
    U[X_UKEY + u * 2 + 1] = U[X_U64 + J_EPKEY]
    L[X_ST + I_UN] = u + 1

    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    flag = (move >> 15) & 7
    captured = (move >> 18) & 15
    piece = (move >> 22) & 15
    us = L[X_ST + I_SIDE]
    key = U[X_U64 + J_KEY] ^ U[X_U64 + J_EPKEY]
    key ^= _Z_CASTLE[L[X_ST + I_CASTLE]]
    U[X_U64 + J_EPKEY] = U0

    if flag != FLAG_NULL:
        _k_remove(ctx, frm, piece)
        key ^= _Z_PIECE[piece, frm]
        if flag == FLAG_EP:
            cap_sq = to - 8 if us == WHITE else to + 8
            cap_piece = (us ^ 1) * 6 + PAWN
            _k_remove(ctx, cap_sq, cap_piece)
            key ^= _Z_PIECE[cap_piece, cap_sq]
        elif captured != 15:
            _k_remove(ctx, to, captured)
            key ^= _Z_PIECE[captured, to]
        dest = us * 6 + promo if promo else piece
        _k_place(ctx, to, dest)
        key ^= _Z_PIECE[dest, to]
        if flag == FLAG_CASTLE:
            if to == 6:
                _k_remove(ctx, 7, WR)
                _k_place(ctx, 5, WR)
                key ^= _Z_PIECE[WR, 7] ^ _Z_PIECE[WR, 5]
            elif to == 2:
                _k_remove(ctx, 0, WR)
                _k_place(ctx, 3, WR)
                key ^= _Z_PIECE[WR, 0] ^ _Z_PIECE[WR, 3]
            elif to == 62:
                _k_remove(ctx, 63, BR)
                _k_place(ctx, 61, BR)
                key ^= _Z_PIECE[BR, 63] ^ _Z_PIECE[BR, 61]
            else:
                _k_remove(ctx, 56, BR)
                _k_place(ctx, 59, BR)
                key ^= _Z_PIECE[BR, 56] ^ _Z_PIECE[BR, 59]
        if piece % 6 == KING:
            L[X_KING + us] = to
        L[X_ST + I_CASTLE] = L[X_ST + I_CASTLE] & _CCLEAR[frm] & _CCLEAR[to]
        if piece % 6 == PAWN or captured != 15 or flag == FLAG_EP:
            L[X_ST + I_HALF] = 0
        else:
            L[X_ST + I_HALF] += 1
        L[X_ST + I_EP] = (frm + to) // 2 if flag == FLAG_DOUBLE else -1
        if us == BLACK:
            L[X_ST + I_FULL] += 1
        L[X_ST + I_ABSPLY] += 1
    else:
        L[X_ST + I_EP] = -1

    key ^= _Z_CASTLE[L[X_ST + I_CASTLE]]
    key ^= _Z_SIDE
    L[X_ST + I_SIDE] = us ^ 1
    U[X_U64 + J_EPKEY] = k_ep_zobrist(ctx)
    U[X_U64 + J_KEY] = key ^ U[X_U64 + J_EPKEY]


@njit(cache=True)
def k_unmake(ctx):
    U = ctx[AU]
    L = ctx[AI]
    N = ctx[A32]
    u = L[X_ST + I_UN] - 1
    L[X_ST + I_UN] = u
    move = np.int64(N[X_U_MOVE + u])
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    flag = (move >> 15) & 7
    captured = (move >> 18) & 15
    piece = (move >> 22) & 15
    L[X_ST + I_SIDE] ^= 1
    us = L[X_ST + I_SIDE]
    um = u * 6 + X_U_MISC
    L[X_ST + I_CASTLE] = N[um + U_CASTLE]
    L[X_ST + I_EP] = N[um + U_EP]
    L[X_ST + I_HALF] = N[um + U_HALF]
    L[X_ST + I_FULL] = N[um + U_FULL]
    L[X_KING + us] = N[um + U_KING]
    L[X_ST + I_ABSPLY] = N[um + U_PLY]
    U[X_U64 + J_KEY] = U[X_UKEY + u * 2]
    U[X_U64 + J_EPKEY] = U[X_UKEY + u * 2 + 1]

    if flag == FLAG_NULL:
        return
    dest = us * 6 + promo if promo else piece
    _k_remove(ctx, to, dest)
    _k_place(ctx, frm, piece)
    if flag == FLAG_EP:
        cap_sq = to - 8 if us == WHITE else to + 8
        _k_place(ctx, cap_sq, (us ^ 1) * 6 + PAWN)
    elif captured != 15:
        _k_place(ctx, to, captured)
    if flag == FLAG_CASTLE:
        if to == 6:
            _k_remove(ctx, 5, WR)
            _k_place(ctx, 7, WR)
        elif to == 2:
            _k_remove(ctx, 3, WR)
            _k_place(ctx, 0, WR)
        elif to == 62:
            _k_remove(ctx, 61, BR)
            _k_place(ctx, 63, BR)
        else:
            _k_remove(ctx, 59, BR)
            _k_place(ctx, 56, BR)


@njit(cache=True)
def k_make_null(ctx):
    k_make(ctx, FLAG_NULL << 15)


@njit(cache=True)
def k_unmake_null(ctx):
    k_unmake(ctx)


# ---------------------------------------------------------------------------
# move emission helpers (same order as movegen._gen_legal)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _emit_promos(buf, n, frm, to, piece, captured, flag_quiet, flag_cap):
    flag = flag_cap if captured != 15 else flag_quiet
    for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
        buf[n] = np.int32(
            frm | (to << 6) | (promo << 12) | (flag << 15) | (captured << 18) | (piece << 22)
        )
        n += 1
    return n


@njit(cache=True)
def _emit_pawn_to(ctx, buf, n, dests, delta, cap, promo, king, pinned, target):
    B = ctx[A8]
    pawn_piece = ctx[AI][X_ST + I_SIDE] * 6 + PAWN
    t = dests & target
    while t:
        to = k_lsb(t)
        t &= t - U1
        frm = to - delta
        if pinned & (U1 << frm) and not (_LINE[king, frm] & (U1 << to)):
            continue
        captured = np.int64(B[X_MB + to]) if cap else np.int64(15)
        if captured < 0:
            captured = np.int64(15)
        if promo:
            n = _emit_promos(
                buf,
                n,
                frm,
                to,
                pawn_piece,
                captured,
                FLAG_PROMO,
                FLAG_PROMO_CAP,
            )
        elif cap:
            buf[n] = np.int32(
                frm | (to << 6) | (FLAG_CAPTURE << 15) | (captured << 18) | (pawn_piece << 22)
            )
            n += 1
        else:
            flag = FLAG_DOUBLE if (delta if delta >= 0 else -delta) == 16 else FLAG_NORMAL
            buf[n] = np.int32(frm | (to << 6) | (flag << 15) | (pawn_piece << 22) | (15 << 18))
            n += 1
    return n


@njit(cache=True)
def _emit_leaper(ctx, buf, n, pieces, attacks, ptype, target, king, pinned):
    B = ctx[A8]
    piece = ctx[AI][X_ST + I_SIDE] * 6 + ptype
    b = pieces
    while b:
        frm = k_lsb(b)
        b &= b - U1
        dests = attacks[frm] & target
        if pinned & (U1 << frm):
            dests &= _LINE[king, frm]
        while dests:
            to = k_lsb(dests)
            dests &= dests - U1
            captured = np.int64(B[X_MB + to])
            if captured < 0:
                buf[n] = np.int32(
                    frm | (to << 6) | (FLAG_NORMAL << 15) | (piece << 22) | (15 << 18)
                )
            else:
                buf[n] = np.int32(
                    frm | (to << 6) | (FLAG_CAPTURE << 15) | (piece << 22) | (captured << 18)
                )
            n += 1
    return n


@njit(cache=True)
def _emit_slider(ctx, buf, n, pieces, bishop, ptype, target, king, pinned):
    B = ctx[A8]
    occ = ctx[AU][X_OCC + 2]
    piece = ctx[AI][X_ST + I_SIDE] * 6 + ptype
    b = pieces
    while b:
        frm = k_lsb(b)
        b &= b - U1
        if ptype == QUEEN:
            dests = k_queen_attacks(frm, occ) & target
        elif bishop:
            dests = k_bishop_attacks(frm, occ) & target
        else:
            dests = k_rook_attacks(frm, occ) & target
        if pinned & (U1 << frm):
            dests &= _LINE[king, frm]
        while dests:
            to = k_lsb(dests)
            dests &= dests - U1
            captured = np.int64(B[X_MB + to])
            if captured < 0:
                buf[n] = np.int32(
                    frm | (to << 6) | (FLAG_NORMAL << 15) | (piece << 22) | (15 << 18)
                )
            else:
                buf[n] = np.int32(
                    frm | (to << 6) | (FLAG_CAPTURE << 15) | (piece << 22) | (captured << 18)
                )
            n += 1
    return n


# ---------------------------------------------------------------------------
# legal move generation — mirrors movegen._gen_legal emission order exactly
# ---------------------------------------------------------------------------


@njit(cache=True)
def k_gen_legal(ctx, buf):
    U = ctx[AU]
    L = ctx[AI]
    us = L[X_ST + I_SIDE]
    them = us ^ 1
    occ = U[X_OCC + 2]
    occ_us = U[X_OCC + us]
    occ_them = U[X_OCC + them]
    king = L[X_KING + us]
    checkers = _PAWN_ATK[us, king] & U[X_BB + them * 6 + PAWN]
    checkers |= _KNIGHT_ATK[king] & U[X_BB + them * 6 + KNIGHT]
    bq = U[X_BB + them * 6 + BISHOP] | U[X_BB + them * 6 + QUEEN]
    rq = U[X_BB + them * 6 + ROOK] | U[X_BB + them * 6 + QUEEN]
    if bq:
        checkers |= k_bishop_attacks(king, occ) & bq
    if rq:
        checkers |= k_rook_attacks(king, occ) & rq
    ncheck = k_pop64(checkers)
    pinned = k_pin_mask(ctx)
    n = 0
    target = M64 ^ occ_us
    if ncheck == 1:
        cs = k_lsb(checkers)
        target &= checkers | _BETWEEN[king, cs]
    elif ncheck >= 2:
        target = U0

    if ncheck < 2:
        pawns = U[X_BB + us * 6 + PAWN]
        if us == WHITE:
            single = ((pawns << 8) & ~occ) & M64
            double = ((single & np.uint64(0x0000000000FF0000)) << 8) & ~occ
            promo_push = single & np.uint64(0xFF00000000000000)
            single &= np.uint64(0x00FFFFFFFFFFFFFF)
            left = ((pawns & np.uint64(0xFEFEFEFEFEFEFEFE)) << 7) & occ_them
            right = ((pawns & np.uint64(0x7F7F7F7F7F7F7F7F)) << 9) & occ_them
            promo_left = left & np.uint64(0xFF00000000000000)
            promo_right = right & np.uint64(0xFF00000000000000)
            left &= np.uint64(0x00FFFFFFFFFFFFFF)
            right &= np.uint64(0x00FFFFFFFFFFFFFF)
        else:
            single = (pawns >> 8) & ~occ
            double = ((single & np.uint64(0x0000FF0000000000)) >> 8) & ~occ
            promo_push = single & np.uint64(0x00000000000000FF)
            single &= np.uint64(0xFFFFFFFFFFFFFF00)
            left = ((pawns & np.uint64(0x7F7F7F7F7F7F7F7F)) >> 7) & occ_them
            right = ((pawns & np.uint64(0xFEFEFEFEFEFEFEFE)) >> 9) & occ_them
            promo_left = left & np.uint64(0x00000000000000FF)
            promo_right = right & np.uint64(0x00000000000000FF)
            left &= np.uint64(0xFFFFFFFFFFFFFF00)
            right &= np.uint64(0xFFFFFFFFFFFFFF00)

        if us == WHITE:
            n = _emit_pawn_to(ctx, buf, n, single, 8, False, False, king, pinned, target)
            n = _emit_pawn_to(ctx, buf, n, double, 16, False, False, king, pinned, target)
            n = _emit_pawn_to(ctx, buf, n, left, 7, True, False, king, pinned, target)
            n = _emit_pawn_to(ctx, buf, n, right, 9, True, False, king, pinned, target)
            n = _emit_pawn_to(ctx, buf, n, promo_push, 8, False, True, king, pinned, target)
            n = _emit_pawn_to(ctx, buf, n, promo_left, 7, True, True, king, pinned, target)
            n = _emit_pawn_to(ctx, buf, n, promo_right, 9, True, True, king, pinned, target)
        else:
            n = _emit_pawn_to(ctx, buf, n, single, -8, False, False, king, pinned, target)
            n = _emit_pawn_to(ctx, buf, n, double, -16, False, False, king, pinned, target)
            n = _emit_pawn_to(ctx, buf, n, left, -7, True, False, king, pinned, target)
            n = _emit_pawn_to(ctx, buf, n, right, -9, True, False, king, pinned, target)
            n = _emit_pawn_to(ctx, buf, n, promo_push, -8, False, True, king, pinned, target)
            n = _emit_pawn_to(ctx, buf, n, promo_left, -7, True, True, king, pinned, target)
            n = _emit_pawn_to(ctx, buf, n, promo_right, -9, True, True, king, pinned, target)

        ep = L[X_ST + I_EP]
        if ep >= 0:
            capturers = _PAWN_ATK[us ^ 1, ep] & pawns
            if us == WHITE:
                capturers &= np.uint64(0x000000FF00000000)
            else:
                capturers &= np.uint64(0x00000000FF000000)
            cap_sq = ep - 8 if us == WHITE else ep + 8
            while capturers:
                frm = k_lsb(capturers)
                capturers &= capturers - U1
                if ncheck and not ((U1 << cap_sq) & checkers):
                    continue
                if pinned & (U1 << frm) and not (_LINE[king, frm] & (U1 << ep)):
                    continue
                if k_ep_legal(ctx, frm, ep):
                    buf[n] = np.int32(
                        frm
                        | (ep << 6)
                        | (FLAG_EP << 15)
                        | ((them * 6 + PAWN) << 18)
                        | ((us * 6 + PAWN) << 22)
                    )
                    n += 1

        n = _emit_leaper(
            ctx, buf, n, U[X_BB + us * 6 + KNIGHT], _KNIGHT_ATK, KNIGHT, target, king, pinned
        )
        n = _emit_slider(ctx, buf, n, U[X_BB + us * 6 + BISHOP], True, BISHOP, target, king, pinned)
        n = _emit_slider(ctx, buf, n, U[X_BB + us * 6 + ROOK], False, ROOK, target, king, pinned)
        n = _emit_slider(ctx, buf, n, U[X_BB + us * 6 + QUEEN], False, QUEEN, target, king, pinned)

        if ncheck == 0:
            castle = L[X_ST + I_CASTLE]
            if us == WHITE:
                if (
                    (castle & WKC)
                    and not (occ & np.uint64(0x60))
                    and not (
                        k_square_attacked(ctx, 4, them, occ)
                        or k_square_attacked(ctx, 5, them, occ)
                        or k_square_attacked(ctx, 6, them, occ)
                    )
                ):
                    buf[n] = np.int32(
                        4 | (6 << 6) | (FLAG_CASTLE << 15) | ((us * 6 + KING) << 22) | (15 << 18)
                    )
                    n += 1
                if (
                    (castle & WQC)
                    and not (occ & np.uint64(0x0E))
                    and not (
                        k_square_attacked(ctx, 4, them, occ)
                        or k_square_attacked(ctx, 3, them, occ)
                        or k_square_attacked(ctx, 2, them, occ)
                    )
                ):
                    buf[n] = np.int32(
                        4 | (2 << 6) | (FLAG_CASTLE << 15) | ((us * 6 + KING) << 22) | (15 << 18)
                    )
                    n += 1
            else:
                if (
                    (castle & BKC)
                    and not (occ & np.uint64(0x6000000000000000))
                    and not (
                        k_square_attacked(ctx, 60, them, occ)
                        or k_square_attacked(ctx, 61, them, occ)
                        or k_square_attacked(ctx, 62, them, occ)
                    )
                ):
                    buf[n] = np.int32(
                        60 | (62 << 6) | (FLAG_CASTLE << 15) | ((us * 6 + KING) << 22) | (15 << 18)
                    )
                    n += 1
                if (
                    (castle & BQC)
                    and not (occ & np.uint64(0x0E00000000000000))
                    and not (
                        k_square_attacked(ctx, 60, them, occ)
                        or k_square_attacked(ctx, 59, them, occ)
                        or k_square_attacked(ctx, 58, them, occ)
                    )
                ):
                    buf[n] = np.int32(
                        60 | (58 << 6) | (FLAG_CASTLE << 15) | ((us * 6 + KING) << 22) | (15 << 18)
                    )
                    n += 1

    B = ctx[A8]
    occ_nk = occ ^ (U1 << king)
    dests = _KING_ATK[king] & ~occ_us
    piece = us * 6 + KING
    while dests:
        to = k_lsb(dests)
        dests &= dests - U1
        if k_square_attacked(ctx, to, them, occ_nk):
            continue
        captured = np.int64(B[X_MB + to])
        if captured < 0:
            buf[n] = np.int32(king | (to << 6) | (FLAG_NORMAL << 15) | (piece << 22) | (15 << 18))
        else:
            buf[n] = np.int32(
                king | (to << 6) | (FLAG_CAPTURE << 15) | (piece << 22) | (captured << 18)
            )
        n += 1
    return n


# ---------------------------------------------------------------------------
# misc position queries used by search
# ---------------------------------------------------------------------------


@njit(cache=True)
def k_attackers_to(ctx, sq, occ):
    """All pieces of both colours attacking ``sq`` under occupancy ``occ``."""
    U = ctx[AU]
    att = (_PAWN_ATK[1, sq] & U[X_BB + 0]) | (_PAWN_ATK[0, sq] & U[X_BB + 6])
    att |= _KNIGHT_ATK[sq] & (U[X_BB + 1] | U[X_BB + 7])
    att |= _KING_ATK[sq] & (U[X_BB + 5] | U[X_BB + 11])
    att |= k_bishop_attacks(sq, occ) & (U[X_BB + 2] | U[X_BB + 8] | U[X_BB + 4] | U[X_BB + 10])
    att |= k_rook_attacks(sq, occ) & (U[X_BB + 3] | U[X_BB + 9] | U[X_BB + 4] | U[X_BB + 10])
    return att


@njit(cache=True)
def k_absolute_pins(ctx, color, occ):
    U = ctx[AU]
    them = color ^ 1
    ksq = ctx[AI][X_KING + color]
    own = U[X_OCC + color]
    tb = them * 6
    snipers = (_ROOK_RAYS[ksq] & (U[X_BB + tb + ROOK] | U[X_BB + tb + QUEEN])) | (
        _BISHOP_RAYS[ksq] & (U[X_BB + tb + BISHOP] | U[X_BB + tb + QUEEN])
    )
    snipers &= occ
    pinned = U0
    while snipers:
        s = k_lsb(snipers)
        snipers &= snipers - U1
        blockers = _BETWEEN[ksq, s] & occ
        if blockers and k_pop64(blockers) == 1 and blockers & own:
            pinned |= blockers
    return pinned


# SEE value vectors live in the params arena (Searcher.pval / see_value):
# pval by piece code = (v0..v4, 0, v0..v4, 0); see_value by type =
# (v0..v4, val_king_see).
PVAL_OFF = (37, 38, 39, 40, 41)  # P_ indices of val_pawn..val_queen
KING_SEE_OFF = 42  # P_ index of val_king_see


@njit(cache=True)
def k_pval(ctx, code):
    """pval[code] — piece-value by engine piece code (king code -> 0)."""
    t = code % 6
    if t == KING:
        return np.int64(0)
    return np.int64(ctx[A32][X_PARAMS + 37 + t])


@njit(cache=True)
def k_sval(ctx, ptype):
    """see_value[ptype] — by piece type 0..5 (king -> val_king_see)."""
    if ptype == KING:
        return np.int64(ctx[A32][X_PARAMS + 42])
    return np.int64(ctx[A32][X_PARAMS + 37 + ptype])


@njit(cache=True)
def k_see_ge(ctx, m, threshold):
    """Mirror of search.see_ge — pin-aware static exchange >= threshold."""
    frm = m & 63
    to = (m >> 6) & 63
    promo = (m >> 12) & 7
    flag = (m >> 15) & 7
    if flag == FLAG_EP or flag == FLAG_CASTLE or promo:
        return threshold <= 0
    U = ctx[AU]
    L = ctx[AI]
    B = ctx[A8]
    victim = np.int64(B[X_MB + to])
    swap = (np.int64(0) if victim < 0 else k_pval(ctx, victim)) - threshold
    if swap < 0:
        return False
    attacker = np.int64(B[X_MB + frm])
    swap = k_pval(ctx, attacker) - swap
    if swap <= 0:
        return True
    occ = (U[X_OCC + 2] ^ (U1 << frm) ^ (U1 << to)) & M64
    stm = (attacker // 6) ^ 1
    attackers = k_attackers_to(ctx, to, occ) & occ
    res = np.int64(1)
    while True:
        stm_att = attackers & U[X_OCC + stm]
        if stm_att == 0:
            break
        pins = k_absolute_pins(ctx, stm, occ)
        ksq = L[X_KING + stm]
        res ^= 1
        found = np.int64(-1)
        csq = np.int64(-1)
        base = stm * 6
        for t in range(PAWN, KING + 1):
            pcs = stm_att & U[X_BB + base + t]
            while pcs:
                cand = k_lsb(pcs)
                pcs &= pcs - U1
                cb = U1 << cand
                if (pins & cb) and not (_LINE[ksq, cand] & (U1 << to)):
                    continue
                csq = cand
                found = np.int64(t)
                break
            if found >= 0:
                break
        if found < 0:
            res ^= 1
            break
        occ ^= U1 << csq
        if found == KING:
            if attackers & U[X_OCC + (stm ^ 1)] & occ:
                res ^= 1
            break
        swap = k_sval(ctx, found) - swap
        if swap < res:
            break
        if found == PAWN or found == BISHOP or found == QUEEN:
            attackers |= k_bishop_attacks(to, occ) & (
                U[X_BB + 2] | U[X_BB + 8] | U[X_BB + 4] | U[X_BB + 10]
            )
        if found == ROOK or found == QUEEN:
            attackers |= k_rook_attacks(to, occ) & (
                U[X_BB + 3] | U[X_BB + 9] | U[X_BB + 4] | U[X_BB + 10]
            )
        attackers &= occ
        stm ^= 1
    return res != 0


@njit(cache=True)
def k_gives_check_fast(ctx, m):
    """Mirror of search._gives_check_fast (direct + discovered check only)."""
    frm = m & 63
    to = (m >> 6) & 63
    promo = (m >> 12) & 7
    flag = (m >> 15) & 7
    piece = (m >> 22) & 15
    if flag == FLAG_EP or flag == FLAG_CASTLE:
        return False
    U = ctx[AU]
    L = ctx[AI]
    us = L[X_ST + I_SIDE]
    them = us ^ 1
    ksq = L[X_KING + them]
    kb = U1 << ksq
    occ = ((U[X_OCC + 2] ^ (U1 << frm)) | (U1 << to)) & M64
    ptype = promo if promo else piece % 6
    if ptype == PAWN:
        if _PAWN_ATK[us, to] & kb:
            return True
    elif ptype == KNIGHT:
        if _KNIGHT_ATK[to] & kb:
            return True
    elif ptype == BISHOP:
        if k_bishop_attacks(to, occ) & kb:
            return True
    elif ptype == ROOK:
        if k_rook_attacks(to, occ) & kb:
            return True
    elif ptype == QUEEN:
        if (k_bishop_attacks(to, occ) | k_rook_attacks(to, occ)) & kb:
            return True
    base = us * 6
    not_frm = M64 ^ (U1 << frm)
    if _BISHOP_RAYS[ksq] & (U1 << frm):
        if (
            k_bishop_attacks(ksq, occ)
            & (U[X_BB + base + BISHOP] | U[X_BB + base + QUEEN])
            & not_frm
        ):
            return True
    if _ROOK_RAYS[ksq] & (U1 << frm):
        if k_rook_attacks(ksq, occ) & (U[X_BB + base + ROOK] | U[X_BB + base + QUEEN]) & not_frm:
            return True
    return False


@njit(cache=True)
def k_is_irreversible(ctx, m):
    """Pre-move irreversibility test (mirror of state._irreversible)."""
    if k_has_legal_ep(ctx):
        return np.int64(1)
    frm = m & 63
    to = (m >> 6) & 63
    promo = (m >> 12) & 7
    flag = (m >> 15) & 7
    captured = (m >> 18) & 15
    piece = (m >> 22) & 15
    if piece % 6 == PAWN or captured != 15 or flag == FLAG_EP or promo:
        return np.int64(1)
    L = ctx[AI]
    castle = L[X_ST + I_CASTLE]
    return np.int64(1 if castle != (castle & _CCLEAR[frm] & _CCLEAR[to]) else 0)


@njit(cache=True)
def k_king_has_legal_move(ctx):
    L = ctx[AI]
    U = ctx[AU]
    us = L[X_ST + I_SIDE]
    them = us ^ 1
    ksq = L[X_KING + us]
    occ_nk = U[X_OCC + 2] ^ (U1 << ksq)
    targets = _KING_ATK[ksq] & ~U[X_OCC + us]
    while targets:
        to = k_lsb(targets)
        targets &= targets - U1
        if not k_square_attacked(ctx, to, them, occ_nk):
            return True
    return False


@njit(cache=True)
def _k_insufficient_color(ctx, color):
    U = ctx[AU]
    occ_c = U[X_OCC + color]
    pawns = U[X_BB + color * 6 + PAWN]
    knights = U[X_BB + color * 6 + KNIGHT]
    bishops = U[X_BB + color * 6 + BISHOP]
    rooks = U[X_BB + color * 6 + ROOK]
    queens = U[X_BB + color * 6 + QUEEN]
    if occ_c & (pawns | rooks | queens):
        return False
    them = color ^ 1
    kings = U[X_BB + KING] | U[X_BB + 6 + KING]
    all_queens = U[X_BB + QUEEN] | U[X_BB + 6 + QUEEN]
    if occ_c & knights:
        return bool(k_pop64(occ_c) <= 2 and not (U[X_OCC + them] & ~kings & ~all_queens))
    if occ_c & bishops:
        all_bishops = U[X_BB + BISHOP] | U[X_BB + 6 + BISHOP]
        all_pawns = U[X_BB + PAWN] | U[X_BB + 6 + PAWN]
        all_knights = U[X_BB + KNIGHT] | U[X_BB + 6 + KNIGHT]
        dark = np.uint64(0xAA55AA55AA55AA55)
        light = M64 ^ dark
        same_color = (not (all_bishops & dark)) or (not (all_bishops & light))
        return bool(same_color and not all_pawns and not all_knights)
    return True


@njit(cache=True)
def k_insufficient(ctx):
    return _k_insufficient_color(ctx, 0) and _k_insufficient_color(ctx, 1)


@njit(cache=True)
def k_pawn_key(ctx):
    U = ctx[AU]
    x = (U[X_BB + 0] * np.uint64(0x9E3779B97F4A7C15)) & M64
    x ^= (U[X_BB + 6] + np.uint64(0x632BE59BD9B4E019)) & M64
    x ^= x >> 31
    x = (x * np.uint64(0xBF58476D1CE4E5B9)) & M64
    x ^= x >> 29
    return x


@njit(cache=True)
def k_nonpawn_key(ctx, color):
    U = ctx[AU]
    base = color * 6
    x = (U[X_BB + base + KNIGHT] * np.uint64(0x9E3779B97F4A7C15)) & M64
    x ^= (U[X_BB + base + BISHOP] * np.uint64(0x632BE59BD9B4E019)) & M64
    x ^= (U[X_BB + base + ROOK] * np.uint64(0xB7E151628AED2A6B)) & M64
    x ^= (U[X_BB + base + QUEEN] * np.uint64(0x94D049BB133111EB)) & M64
    x ^= x >> 31
    x = (x * np.uint64(0xBF58476D1CE4E5B9)) & M64
    x ^= x >> 29
    return x


@njit(cache=True)
def k_compute_key(ctx):
    key = U0
    B = ctx[A8]
    for sq in range(64):
        piece = np.int64(B[X_MB + sq])
        if piece >= 0:
            key ^= _Z_PIECE[piece, sq]
    L = ctx[AI]
    key ^= _Z_CASTLE[L[X_ST + I_CASTLE]]
    if L[X_ST + I_SIDE] == BLACK:
        key ^= _Z_SIDE
    return key ^ k_ep_zobrist(ctx)

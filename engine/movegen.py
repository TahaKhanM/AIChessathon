"""Legal move generation, attacks, pins, castling and en passant.

Differential tests cover pinned en passant, castling through check and
all promotion types. Sliding attacks use ray bitboards and BETWEEN/LINE
tables with LSB/MSB blocker isolation, without requiring PEXT."""

from __future__ import annotations

from engine.board import (
    BISHOP,
    FLAG_CAPTURE,
    FLAG_CASTLE,
    FLAG_DOUBLE,
    FLAG_EP,
    FLAG_NORMAL,
    FLAG_PROMO,
    FLAG_PROMO_CAP,
    KING,
    KING_ATK,
    KNIGHT,
    KNIGHT_ATK,
    MASK64,
    MAX_MOVES,
    PAWN,
    PAWN_ATK,
    QUEEN,
    ROOK,
    WHITE,
    BKC,
    BQC,
    WKC,
    WQC,
    Board,
    Z_EP,
    _STEP,
    encode_move,
    move_to_uci,
)

_BUF = [0] * MAX_MOVES
_STACK = [[0] * MAX_MOVES for _ in range(128)]

BETWEEN: list[list[int]] = [[0] * 64 for _ in range(64)]
LINE: list[list[int]] = [[0] * 64 for _ in range(64)]
ROOK_POS = [(0, 0)] * 64
ROOK_NEG = [(0, 0)] * 64
BISHOP_POS = [(0, 0)] * 64
BISHOP_NEG = [(0, 0)] * 64


def _ray(sq: int, delta: int) -> int:
    bits = 0
    s = sq
    while True:
        n = _STEP[delta + 9][s]
        if n < 0:
            break
        bits |= 1 << n
        s = n
    return bits


def _init_lines() -> None:
    deltas = (1, -1, 8, -8, 9, 7, -7, -9)
    for a in range(64):
        north = _ray(a, 8)
        east = _ray(a, 1)
        south = _ray(a, -8)
        west = _ray(a, -1)
        ne = _ray(a, 9)
        nw = _ray(a, 7)
        se = _ray(a, -7)
        sw = _ray(a, -9)
        ROOK_POS[a] = (north, east)
        ROOK_NEG[a] = (south, west)
        BISHOP_POS[a] = (ne, nw)
        BISHOP_NEG[a] = (se, sw)
        for d in deltas:
            line = 0
            s = a
            while True:
                n = _STEP[d + 9][s]
                if n < 0:
                    break
                line |= 1 << n
                s = n
            s = a
            while True:
                n = _STEP[(-d) + 9][s]
                if n < 0:
                    break
                line |= 1 << n
                s = n
            between_acc = 0
            s = a
            while True:
                n = _STEP[d + 9][s]
                if n < 0:
                    break
                LINE[a][n] = line | (1 << a)
                BETWEEN[a][n] = between_acc
                between_acc |= 1 << n
                s = n


_init_lines()


def _attacks_from_rays(sq: int, occ: int, pos: tuple[int, int], neg: tuple[int, int]) -> int:
    atk = 0
    for ray in pos:
        hits = ray & occ
        if hits:
            b = (hits & -hits).bit_length() - 1
            atk |= BETWEEN[sq][b] | (1 << b)
        else:
            atk |= ray
    for ray in neg:
        hits = ray & occ
        if hits:
            b = hits.bit_length() - 1
            atk |= BETWEEN[sq][b] | (1 << b)
        else:
            atk |= ray
    return atk


def rook_attacks(sq: int, occ: int) -> int:
    return _attacks_from_rays(sq, occ, ROOK_POS[sq], ROOK_NEG[sq])


def bishop_attacks(sq: int, occ: int) -> int:
    return _attacks_from_rays(sq, occ, BISHOP_POS[sq], BISHOP_NEG[sq])


def queen_attacks(sq: int, occ: int) -> int:
    return rook_attacks(sq, occ) | bishop_attacks(sq, occ)


def square_attacked(board: Board, sq: int, by: int, occ: int) -> bool:
    bb = board._bb
    if PAWN_ATK[by ^ 1][sq] & bb[by * 6 + PAWN]:
        return True
    if KNIGHT_ATK[sq] & bb[by * 6 + KNIGHT]:
        return True
    if KING_ATK[sq] & bb[by * 6 + KING]:
        return True
    if bishop_attacks(sq, occ) & (bb[by * 6 + BISHOP] | bb[by * 6 + QUEEN]):
        return True
    if rook_attacks(sq, occ) & (bb[by * 6 + ROOK] | bb[by * 6 + QUEEN]):
        return True
    return False


def in_check(board: Board) -> bool:
    us = board.side
    return square_attacked(board, board._king[us], us ^ 1, board._occ_all)


def _pin_mask(board: Board) -> int:
    us = board.side
    them = us ^ 1
    king = board._king[us]
    occ = board._occ_all
    us_occ = board._occ[us]
    pinned = 0
    rq = board._bb[them * 6 + ROOK] | board._bb[them * 6 + QUEEN]
    bq = board._bb[them * 6 + BISHOP] | board._bb[them * 6 + QUEEN]
    snipers = rq
    while snipers:
        s = (snipers & -snipers).bit_length() - 1
        snipers &= snipers - 1
        df = (s & 7) - (king & 7)
        dr = (s >> 3) - (king >> 3)
        if df != 0 and dr != 0:
            continue
        blockers = occ & BETWEEN[king][s]
        if blockers and blockers.bit_count() == 1 and blockers & us_occ:
            pinned |= blockers
    snipers = bq
    while snipers:
        s = (snipers & -snipers).bit_length() - 1
        snipers &= snipers - 1
        df = (s & 7) - (king & 7)
        dr = (s >> 3) - (king >> 3)
        if abs(df) != abs(dr) or df == 0:
            continue
        blockers = occ & BETWEEN[king][s]
        if blockers and blockers.bit_count() == 1 and blockers & us_occ:
            pinned |= blockers
    return pinned


def _ep_legal(board: Board, frm: int, ep: int) -> bool:
    us = board.side
    them = us ^ 1
    cap = ep - 8 if us == WHITE else ep + 8
    occ = board._occ_all ^ (1 << frm) ^ (1 << ep) ^ (1 << cap)
    king = board._king[us]
    bb = board._bb
    pawn = bb[them * 6 + PAWN] & ~(1 << cap)
    if PAWN_ATK[us][king] & pawn:
        return False
    if KNIGHT_ATK[king] & bb[them * 6 + KNIGHT]:
        return False
    if KING_ATK[king] & bb[them * 6 + KING]:
        return False
    if bishop_attacks(king, occ) & (bb[them * 6 + BISHOP] | bb[them * 6 + QUEEN]):
        return False
    if rook_attacks(king, occ) & (bb[them * 6 + ROOK] | bb[them * 6 + QUEEN]):
        return False
    return True


def has_legal_ep(board: Board) -> bool:
    ep = board.ep_square
    if ep < 0:
        return False
    us = board.side
    capturers = PAWN_ATK[us ^ 1][ep] & board._bb[us * 6 + PAWN]
    if us == WHITE:
        capturers &= 0x000000FF00000000
    else:
        capturers &= 0x00000000FF000000
    while capturers:
        frm = (capturers & -capturers).bit_length() - 1
        capturers &= capturers - 1
        if _ep_legal(board, frm, ep):
            return True
    return False


def ep_zobrist(board: Board) -> int:
    if has_legal_ep(board):
        return Z_EP[board.ep_square & 7]
    return 0


def _add(buf: list[int], n: int, move: int) -> int:
    buf[n] = move
    return n + 1


def _add_promos(
    buf: list[int],
    n: int,
    frm: int,
    to: int,
    piece: int,
    captured: int,
    flag_quiet: int,
    flag_cap: int,
) -> int:
    flag = flag_cap if captured != 15 else flag_quiet
    for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
        buf[n] = encode_move(frm, to, promo, flag, piece, captured)
        n += 1
    return n


def _gen_legal(board: Board, buf: list[int]) -> int:
    us = board.side
    them = us ^ 1
    bb = board._bb
    sq = board._sq
    occ = board._occ_all
    occ_us = board._occ[us]
    occ_them = board._occ[them]
    king = board._king[us]
    checkers = 0
    if PAWN_ATK[us][king] & bb[them * 6 + PAWN]:
        checkers |= PAWN_ATK[us][king] & bb[them * 6 + PAWN]
    if KNIGHT_ATK[king] & bb[them * 6 + KNIGHT]:
        checkers |= KNIGHT_ATK[king] & bb[them * 6 + KNIGHT]
    bq = bb[them * 6 + BISHOP] | bb[them * 6 + QUEEN]
    rq = bb[them * 6 + ROOK] | bb[them * 6 + QUEEN]
    if bq:
        checkers |= bishop_attacks(king, occ) & bq
    if rq:
        checkers |= rook_attacks(king, occ) & rq
    ncheck = checkers.bit_count()
    pinned = _pin_mask(board)
    n = 0
    target = MASK64 ^ occ_us
    if ncheck == 1:
        cs = (checkers & -checkers).bit_length() - 1
        target &= checkers | BETWEEN[king][cs]
    elif ncheck >= 2:
        target = 0

    if ncheck < 2:
        pawns = bb[us * 6 + PAWN]
        pawn_piece = us * 6 + PAWN
        if us == WHITE:
            single = ((pawns << 8) & ~occ) & MASK64
            double = ((single & 0x0000000000FF0000) << 8) & ~occ
            promo_push = single & 0xFF00000000000000
            single &= ~0xFF00000000000000
            left = ((pawns & ~0x0101010101010101) << 7) & occ_them
            right = ((pawns & ~0x8080808080808080) << 9) & occ_them
            promo_left = left & 0xFF00000000000000
            promo_right = right & 0xFF00000000000000
            left &= ~0xFF00000000000000
            right &= ~0xFF00000000000000
        else:
            single = (pawns >> 8) & ~occ
            double = ((single & 0x0000FF0000000000) >> 8) & ~occ
            promo_push = single & 0x00000000000000FF
            single &= ~0x00000000000000FF
            left = ((pawns & ~0x8080808080808080) >> 7) & occ_them
            right = ((pawns & ~0x0101010101010101) >> 9) & occ_them
            promo_left = left & 0x00000000000000FF
            promo_right = right & 0x00000000000000FF
            left &= ~0x00000000000000FF
            right &= ~0x00000000000000FF

        def emit_pawn_to(dests: int, delta: int, cap: bool, promo: bool) -> None:
            nonlocal n
            t = dests & target
            while t:
                to = (t & -t).bit_length() - 1
                t &= t - 1
                frm = to - delta
                if pinned & (1 << frm) and not (LINE[king][frm] & (1 << to)):
                    continue
                captured = sq[to] if cap else 15
                if captured < 0:
                    captured = 15
                if promo:
                    n = _add_promos(
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
                    n = _add(
                        buf,
                        n,
                        encode_move(frm, to, 0, FLAG_CAPTURE, pawn_piece, captured),
                    )
                else:
                    flag = FLAG_DOUBLE if abs(delta) == 16 else FLAG_NORMAL
                    n = _add(buf, n, encode_move(frm, to, 0, flag, pawn_piece, 15))

        if us == WHITE:
            emit_pawn_to(single, 8, False, False)
            emit_pawn_to(double, 16, False, False)
            emit_pawn_to(left, 7, True, False)
            emit_pawn_to(right, 9, True, False)
            emit_pawn_to(promo_push, 8, False, True)
            emit_pawn_to(promo_left, 7, True, True)
            emit_pawn_to(promo_right, 9, True, True)
        else:
            emit_pawn_to(single, -8, False, False)
            emit_pawn_to(double, -16, False, False)
            emit_pawn_to(left, -7, True, False)
            emit_pawn_to(right, -9, True, False)
            emit_pawn_to(promo_push, -8, False, True)
            emit_pawn_to(promo_left, -7, True, True)
            emit_pawn_to(promo_right, -9, True, True)

        ep = board.ep_square
        if ep >= 0:
            capturers = PAWN_ATK[us ^ 1][ep] & pawns
            if us == WHITE:
                capturers &= 0x000000FF00000000
            else:
                capturers &= 0x00000000FF000000
            cap_sq = ep - 8 if us == WHITE else ep + 8
            while capturers:
                frm = (capturers & -capturers).bit_length() - 1
                capturers &= capturers - 1
                if ncheck and not ((1 << cap_sq) & checkers):
                    continue
                if pinned & (1 << frm) and not (LINE[king][frm] & (1 << ep)):
                    continue
                if _ep_legal(board, frm, ep):
                    n = _add(
                        buf,
                        n,
                        encode_move(frm, ep, 0, FLAG_EP, pawn_piece, them * 6 + PAWN),
                    )

        def emit_leaper(pieces: int, attacks: tuple[int, ...], ptype: int) -> None:
            nonlocal n
            piece = us * 6 + ptype
            b = pieces
            while b:
                frm = (b & -b).bit_length() - 1
                b &= b - 1
                dests = attacks[frm] & target
                if pinned & (1 << frm):
                    dests &= LINE[king][frm]
                while dests:
                    to = (dests & -dests).bit_length() - 1
                    dests &= dests - 1
                    captured = sq[to]
                    if captured < 0:
                        n = _add(
                            buf,
                            n,
                            encode_move(frm, to, 0, FLAG_NORMAL, piece, 15),
                        )
                    else:
                        n = _add(
                            buf,
                            n,
                            encode_move(frm, to, 0, FLAG_CAPTURE, piece, captured),
                        )

        emit_leaper(bb[us * 6 + KNIGHT], KNIGHT_ATK, KNIGHT)

        def emit_slider(pieces: int, bishop: bool, ptype: int) -> None:
            nonlocal n
            piece = us * 6 + ptype
            b = pieces
            while b:
                frm = (b & -b).bit_length() - 1
                b &= b - 1
                dests = (bishop_attacks(frm, occ) if bishop else rook_attacks(frm, occ)) & target
                if ptype == QUEEN:
                    dests = queen_attacks(frm, occ) & target
                if pinned & (1 << frm):
                    dests &= LINE[king][frm]
                while dests:
                    to = (dests & -dests).bit_length() - 1
                    dests &= dests - 1
                    captured = sq[to]
                    if captured < 0:
                        n = _add(buf, n, encode_move(frm, to, 0, FLAG_NORMAL, piece, 15))
                    else:
                        n = _add(
                            buf,
                            n,
                            encode_move(frm, to, 0, FLAG_CAPTURE, piece, captured),
                        )

        emit_slider(bb[us * 6 + BISHOP], True, BISHOP)
        emit_slider(bb[us * 6 + ROOK], False, ROOK)
        emit_slider(bb[us * 6 + QUEEN], False, QUEEN)

        if ncheck == 0:
            castle = board.castling
            if us == WHITE:
                if (
                    (castle & WKC)
                    and not (occ & 0x60)
                    and not (
                        square_attacked(board, 4, them, occ)
                        or square_attacked(board, 5, them, occ)
                        or square_attacked(board, 6, them, occ)
                    )
                ):
                    n = _add(buf, n, encode_move(4, 6, 0, FLAG_CASTLE, us * 6 + KING, 15))
                if (
                    (castle & WQC)
                    and not (occ & 0x0E)
                    and not (
                        square_attacked(board, 4, them, occ)
                        or square_attacked(board, 3, them, occ)
                        or square_attacked(board, 2, them, occ)
                    )
                ):
                    n = _add(buf, n, encode_move(4, 2, 0, FLAG_CASTLE, us * 6 + KING, 15))
            else:
                if (
                    (castle & BKC)
                    and not (occ & 0x6000000000000000)
                    and not (
                        square_attacked(board, 60, them, occ)
                        or square_attacked(board, 61, them, occ)
                        or square_attacked(board, 62, them, occ)
                    )
                ):
                    n = _add(buf, n, encode_move(60, 62, 0, FLAG_CASTLE, us * 6 + KING, 15))
                if (
                    (castle & BQC)
                    and not (occ & 0x0E00000000000000)
                    and not (
                        square_attacked(board, 60, them, occ)
                        or square_attacked(board, 59, them, occ)
                        or square_attacked(board, 58, them, occ)
                    )
                ):
                    n = _add(buf, n, encode_move(60, 58, 0, FLAG_CASTLE, us * 6 + KING, 15))

    occ_nk = occ ^ (1 << king)
    dests = KING_ATK[king] & ~occ_us
    piece = us * 6 + KING
    while dests:
        to = (dests & -dests).bit_length() - 1
        dests &= dests - 1
        if square_attacked(board, to, them, occ_nk):
            continue
        captured = sq[to]
        if captured < 0:
            n = _add(buf, n, encode_move(king, to, 0, FLAG_NORMAL, piece, 15))
        else:
            n = _add(buf, n, encode_move(king, to, 0, FLAG_CAPTURE, piece, captured))
    return n


def generate_legal(board: Board, out: object) -> int:
    n = _gen_legal(board, _BUF)
    for i in range(n):
        out[i] = _BUF[i]  # type: ignore[index]
    return n


def legal_uci(board: Board) -> list[str]:
    n = _gen_legal(board, _BUF)
    return [move_to_uci(_BUF[i]) for i in range(n)]


def perft(board: Board, depth: int, ply: int = 0) -> int:
    if depth <= 0:
        return 1
    buf = _STACK[ply]
    n = _gen_legal(board, buf)
    if depth == 1:
        return n
    total = 0
    for i in range(n):
        board.make(buf[i])
        total += perft(board, depth - 1, ply + 1)
        board.unmake()
    return total

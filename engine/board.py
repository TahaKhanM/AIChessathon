"""Bitboards, mailbox, move encoding and reversible state.

Differential legal-move tests and make/unmake snapshots validate the board
against python-chess. The reference representation exposes parallel NumPy
views for compiled kernels."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

BB_COUNT = 12
EMPTY = -1
MAX_PLY = 2048
MAX_MOVES = 256

WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5
WP, WN, WB, WR, WQ, WK = 0, 1, 2, 3, 4, 5
BP, BN, BB, BR, BQ, BK = 6, 7, 8, 9, 10, 11

WKC, WQC, BKC, BQC = 1, 2, 4, 8

FLAG_NORMAL = 0
FLAG_CAPTURE = 1
FLAG_DOUBLE = 2
FLAG_EP = 3
FLAG_CASTLE = 4
FLAG_PROMO = 5
FLAG_PROMO_CAP = 6
FLAG_NULL = 7

MASK64 = 0xFFFFFFFFFFFFFFFF
FILE_A = 0x0101010101010101
FILE_H = 0x8080808080808080

PIECE_CHAR = "PNBRQKpnbrqk"
CHAR_TO_PIECE = {c: i for i, c in enumerate(PIECE_CHAR)}
SQUARE_NAMES = [f"{file}{rank}" for rank in "12345678" for file in "abcdefgh"]
NAME_TO_SQUARE = {name: i for i, name in enumerate(SQUARE_NAMES)}

# Squares whose from/to events clear a castling right.
_CASTLE_CLEAR = [15] * 64
_CASTLE_CLEAR[0] = 15 & ~WQC
_CASTLE_CLEAR[4] = 15 & ~(WKC | WQC)
_CASTLE_CLEAR[7] = 15 & ~WKC
_CASTLE_CLEAR[56] = 15 & ~BQC
_CASTLE_CLEAR[60] = 15 & ~(BKC | BQC)
_CASTLE_CLEAR[63] = 15 & ~BKC
CASTLE_CLEAR = tuple(_CASTLE_CLEAR)

_STEP = [[-1] * 64 for _ in range(19)]


def _build_step_tables() -> None:
    deltas = (-9, -8, -7, -1, 1, 7, 8, 9)
    for sq in range(64):
        f, r = sq & 7, sq >> 3
        for d in deltas:
            to = sq + d
            if to < 0 or to > 63:
                continue
            df = (to & 7) - f
            dr = (to >> 3) - r
            if abs(df) > 1 or abs(dr) > 1:
                continue
            _STEP[d + 9][sq] = to


_build_step_tables()


def _splitmix(seed: int) -> int:
    seed = (seed + 0x9E3779B97F4A7C15) & MASK64
    z = seed
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & MASK64
    return z ^ (z >> 31)


def _zobrist_tables() -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], tuple[int, ...], int]:
    seed = 0xA1C4E55A1C4E5501
    piece = []
    for _p in range(12):
        row = []
        for _sq in range(64):
            seed = _splitmix(seed)
            row.append(seed)
        piece.append(tuple(row))
    castle = []
    for _i in range(16):
        seed = _splitmix(seed)
        castle.append(seed)
    ep = []
    for _i in range(8):
        seed = _splitmix(seed)
        ep.append(seed)
    seed = _splitmix(seed)
    return tuple(piece), tuple(castle), tuple(ep), seed


Z_PIECE, Z_CASTLE, Z_EP, Z_SIDE = _zobrist_tables()


def encode_move(frm: int, to: int, promo: int, flag: int, piece: int, captured: int) -> int:
    return frm | (to << 6) | (promo << 12) | (flag << 15) | (captured << 18) | (piece << 22)


def decode_move(move: int) -> tuple[int, int, int, int, int, int]:
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    flag = (move >> 15) & 7
    captured = (move >> 18) & 15
    piece = (move >> 22) & 15
    return frm, to, promo, flag, piece, captured


def move_to_uci(move: int) -> str:
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    text = SQUARE_NAMES[frm] + SQUARE_NAMES[to]
    if promo:
        text += "nbrq"[promo - 1]
    return text


def move_from_uci(uci: str) -> tuple[int, int, int]:
    frm = NAME_TO_SQUARE[uci[0:2]]
    to = NAME_TO_SQUARE[uci[2:4]]
    promo = 0
    if len(uci) > 4:
        promo = "nbrq".index(uci[4]) + 1
    return frm, to, promo


class Snapshot(NamedTuple):
    bitboards: tuple[int, ...]
    mailbox: tuple[int, ...]
    occupied: int
    occupied_color: tuple[int, int]
    side: int
    castling: int
    ep_square: int
    halfmove: int
    fullmove: int
    key: int
    king_sq: tuple[int, int]


def snapshot(board: Board) -> Snapshot:
    return Snapshot(
        bitboards=tuple(board._bb),
        mailbox=tuple(board._sq),
        occupied=board._occ_all,
        occupied_color=(board._occ[0], board._occ[1]),
        side=board.side,
        castling=board.castling,
        ep_square=board.ep_square,
        halfmove=board.halfmove,
        fullmove=board.fullmove,
        key=int(board.key),
        king_sq=(board._king[0], board._king[1]),
    )


def _build_leaper_attacks() -> tuple[tuple[int, ...], tuple[int, ...]]:
    knight_d = (17, 15, 10, 6, -17, -15, -10, -6)
    king_d = (1, -1, 8, -8, 9, 7, -9, -7)
    knight = []
    king = []
    for sq in range(64):
        na = ka = 0
        f, r = sq & 7, sq >> 3
        for d in knight_d:
            to = sq + d
            if 0 <= to <= 63 and abs((to & 7) - f) <= 2 and abs((to >> 3) - r) <= 2:
                nf = to & 7
                if abs(nf - f) in (1, 2):
                    na |= 1 << to
        for d in king_d:
            to = _STEP[d + 9][sq]
            if to >= 0:
                ka |= 1 << to
        knight.append(na)
        king.append(ka)
    return tuple(knight), tuple(king)


_KNIGHT_ATK, _KING_ATK = _build_leaper_attacks()
KNIGHT_ATK = _KNIGHT_ATK
KING_ATK = _KING_ATK


def _pawn_attacks() -> tuple[tuple[int, ...], tuple[int, ...]]:
    white = []
    black = []
    for sq in range(64):
        f = sq & 7
        w = b = 0
        if sq <= 55:
            if f > 0:
                w |= 1 << (sq + 7)
            if f < 7:
                w |= 1 << (sq + 9)
        if sq >= 8:
            if f > 0:
                b |= 1 << (sq - 9)
            if f < 7:
                b |= 1 << (sq - 7)
        white.append(w)
        black.append(b)
    return tuple(white), tuple(black)


PAWN_ATK_WHITE, PAWN_ATK_BLACK = _pawn_attacks()
PAWN_ATK = (PAWN_ATK_WHITE, PAWN_ATK_BLACK)


class Board:
    """Mailbox + bitboard position with incremental Zobrist and undo."""

    __slots__ = (
        "_bb",
        "_sq",
        "_occ",
        "_occ_all",
        "_king",
        "side",
        "castling",
        "ep_square",
        "halfmove",
        "fullmove",
        "key",
        "_ep_key",
        "_abs_ply",
        "_un",
        "_u_move",
        "_u_castle",
        "_u_ep",
        "_u_half",
        "_u_full",
        "_u_key",
        "_u_epkey",
        "_u_king",
        "_u_ply",
    )

    def __init__(self) -> None:
        self._bb = [0] * 12
        self._sq = [EMPTY] * 64
        self._occ = [0, 0]
        self._occ_all = 0
        self._king = [0, 0]
        self.side = WHITE
        self.castling = 0
        self.ep_square = -1
        self.halfmove = 0
        self.fullmove = 1
        self.key = 0
        self._ep_key = 0
        self._abs_ply = 0
        self._un = 0
        self._u_move = [0] * MAX_PLY
        self._u_castle = [0] * MAX_PLY
        self._u_ep = [-1] * MAX_PLY
        self._u_half = [0] * MAX_PLY
        self._u_full = [1] * MAX_PLY
        self._u_key = [0] * MAX_PLY
        self._u_epkey = [0] * MAX_PLY
        self._u_king = [0] * MAX_PLY
        self._u_ply = [0] * MAX_PLY

    @classmethod
    def from_fen(cls, fen: str) -> Board:
        board = cls()
        parts = fen.split()
        placement = parts[0]
        rank, file = 7, 0
        for ch in placement:
            if ch == "/":
                rank -= 1
                file = 0
                continue
            if ch.isdigit():
                file += int(ch)
                continue
            piece = CHAR_TO_PIECE[ch]
            sq = rank * 8 + file
            board._place(sq, piece)
            if piece % 6 == KING:
                board._king[piece // 6] = sq
            file += 1
        board.side = WHITE if parts[1] == "w" else BLACK
        castle = 0
        if len(parts) > 2 and parts[2] != "-":
            if "K" in parts[2]:
                castle |= WKC
            if "Q" in parts[2]:
                castle |= WQC
            if "k" in parts[2]:
                castle |= BKC
            if "q" in parts[2]:
                castle |= BQC
        if board._sq[4] != WK:
            castle &= ~(WKC | WQC)
        if board._sq[7] != WR:
            castle &= ~WKC
        if board._sq[0] != WR:
            castle &= ~WQC
        if board._sq[60] != BK:
            castle &= ~(BKC | BQC)
        if board._sq[63] != BR:
            castle &= ~BKC
        if board._sq[56] != BR:
            castle &= ~BQC
        board.castling = castle
        board.ep_square = -1
        if len(parts) > 3 and parts[3] != "-":
            board.ep_square = NAME_TO_SQUARE[parts[3]]
        board.halfmove = int(parts[4]) if len(parts) > 4 else 0
        board.fullmove = int(parts[5]) if len(parts) > 5 else 1
        board._abs_ply = 2 * (board.fullmove - 1) + (board.side == BLACK)
        board._ep_key = board._relevant_ep_key()
        board.key = board.compute_key()
        return board

    def to_fen(self) -> str:
        ranks = []
        for rank in range(7, -1, -1):
            empty = 0
            row = []
            for file in range(8):
                piece = self._sq[rank * 8 + file]
                if piece < 0:
                    empty += 1
                    continue
                if empty:
                    row.append(str(empty))
                    empty = 0
                row.append(PIECE_CHAR[piece])
            if empty:
                row.append(str(empty))
            ranks.append("".join(row))
        side = "w" if self.side == WHITE else "b"
        castle = ""
        if self.castling & WKC:
            castle += "K"
        if self.castling & WQC:
            castle += "Q"
        if self.castling & BKC:
            castle += "k"
        if self.castling & BQC:
            castle += "q"
        if not castle:
            castle = "-"
        ep = "-"
        if self.ep_square >= 0:
            from engine.movegen import has_legal_ep

            if has_legal_ep(self):
                ep = SQUARE_NAMES[self.ep_square]
        return f"{'/'.join(ranks)} {side} {castle} {ep} {self.halfmove} {self.fullmove}"

    @property
    def bitboards(self) -> np.ndarray:
        return np.array(self._bb, dtype=np.uint64)

    @property
    def mailbox(self) -> np.ndarray:
        return np.array(self._sq, dtype=np.int8)

    @property
    def occupied(self) -> int:
        return self._occ_all

    def absolute_ply(self) -> int:
        return self._abs_ply

    def _place(self, sq: int, piece: int) -> None:
        self._sq[sq] = piece
        bit = 1 << sq
        self._bb[piece] |= bit
        self._occ[piece // 6] |= bit
        self._occ_all |= bit

    def _remove(self, sq: int, piece: int) -> None:
        self._sq[sq] = EMPTY
        bit = 1 << sq
        self._bb[piece] &= MASK64 ^ bit
        self._occ[piece // 6] &= MASK64 ^ bit
        self._occ_all &= MASK64 ^ bit

    def compute_key(self) -> int:
        key = 0
        sq_list = self._sq
        for sq in range(64):
            piece = sq_list[sq]
            if piece >= 0:
                key ^= Z_PIECE[piece][sq]
        key ^= Z_CASTLE[self.castling]
        if self.side == BLACK:
            key ^= Z_SIDE
        key ^= self._relevant_ep_key()
        return key & MASK64

    def _relevant_ep_key(self) -> int:
        ep = self.ep_square
        if ep < 0:
            return 0
        from engine.movegen import has_legal_ep

        if has_legal_ep(self):
            return Z_EP[ep & 7]
        return 0

    def make(self, move: int) -> None:
        u = self._un
        self._u_move[u] = move
        self._u_castle[u] = self.castling
        self._u_ep[u] = self.ep_square
        self._u_half[u] = self.halfmove
        self._u_full[u] = self.fullmove
        self._u_key[u] = self.key
        self._u_epkey[u] = self._ep_key
        self._u_king[u] = self._king[self.side]
        self._u_ply[u] = self._abs_ply
        self._un = u + 1

        frm, to, promo, flag, piece, captured = decode_move(move)
        us = self.side
        key = self.key ^ self._ep_key
        key ^= Z_CASTLE[self.castling]
        self._ep_key = 0

        if flag != FLAG_NULL:
            self._remove(frm, piece)
            key ^= Z_PIECE[piece][frm]
            if flag == FLAG_EP:
                cap_sq = to - 8 if us == WHITE else to + 8
                cap_piece = (us ^ 1) * 6 + PAWN
                self._remove(cap_sq, cap_piece)
                key ^= Z_PIECE[cap_piece][cap_sq]
            elif captured != 15:
                self._remove(to, captured)
                key ^= Z_PIECE[captured][to]
            dest = us * 6 + promo if promo else piece
            self._place(to, dest)
            key ^= Z_PIECE[dest][to]
            if flag == FLAG_CASTLE:
                if to == 6:
                    self._remove(7, WR)
                    self._place(5, WR)
                    key ^= Z_PIECE[WR][7] ^ Z_PIECE[WR][5]
                elif to == 2:
                    self._remove(0, WR)
                    self._place(3, WR)
                    key ^= Z_PIECE[WR][0] ^ Z_PIECE[WR][3]
                elif to == 62:
                    self._remove(63, BR)
                    self._place(61, BR)
                    key ^= Z_PIECE[BR][63] ^ Z_PIECE[BR][61]
                else:
                    self._remove(56, BR)
                    self._place(59, BR)
                    key ^= Z_PIECE[BR][56] ^ Z_PIECE[BR][59]
            if piece % 6 == KING:
                self._king[us] = to
            self.castling &= CASTLE_CLEAR[frm]
            self.castling &= CASTLE_CLEAR[to]
            if piece % 6 == PAWN or captured != 15 or flag == FLAG_EP:
                self.halfmove = 0
            else:
                self.halfmove += 1
            self.ep_square = (frm + to) // 2 if flag == FLAG_DOUBLE else -1
            if us == BLACK:
                self.fullmove += 1
            self._abs_ply += 1
        else:
            self.ep_square = -1

        key ^= Z_CASTLE[self.castling]
        key ^= Z_SIDE
        self.side = us ^ 1
        self._ep_key = self._relevant_ep_key()
        self.key = (key ^ self._ep_key) & MASK64

    def unmake(self) -> None:
        u = self._un - 1
        self._un = u
        move = self._u_move[u]
        frm, to, promo, flag, piece, captured = decode_move(move)
        self.side ^= 1
        us = self.side
        self.castling = self._u_castle[u]
        self.ep_square = self._u_ep[u]
        self.halfmove = self._u_half[u]
        self.fullmove = self._u_full[u]
        self.key = self._u_key[u]
        self._ep_key = self._u_epkey[u]
        self._king[us] = self._u_king[u]
        self._abs_ply = self._u_ply[u]

        if flag == FLAG_NULL:
            return

        dest = us * 6 + promo if promo else piece
        self._remove(to, dest)
        self._place(frm, piece)
        if flag == FLAG_EP:
            cap_sq = to - 8 if us == WHITE else to + 8
            self._place(cap_sq, (us ^ 1) * 6 + PAWN)
        elif captured != 15:
            self._place(to, captured)
        if flag == FLAG_CASTLE:
            if to == 6:
                self._remove(5, WR)
                self._place(7, WR)
            elif to == 2:
                self._remove(3, WR)
                self._place(0, WR)
            elif to == 62:
                self._remove(61, BR)
                self._place(63, BR)
            else:
                self._remove(59, BR)
                self._place(56, BR)

    def make_null(self) -> None:
        self.make(encode_move(0, 0, 0, FLAG_NULL, 0, 15))

    def unmake_null(self) -> None:
        self.unmake()

"""Move ordering and correction histories.

Capture, quiet and continuation histories improve ordering. Correction
updates use searched evidence relative to the raw static evaluation."""

from __future__ import annotations

import numpy as np

from engine.board import (
    BISHOP,
    KNIGHT,
    QUEEN,
    ROOK,
    Board,
)
from engine.movegen import square_attacked

HISTORY_MAX = 16384

# Correction tables store centipawns * CORR_GRAIN so updates stay integral.
CORR_SIZE = 1 << 14
CORR_GRAIN = 256
CORR_LIMIT = 64 * CORR_GRAIN
CORR_WEIGHT_CAP = 16

PAWN_HIST_SIZE = 512  # pawn-structure-conditioned ordering buckets


def stat_bonus(depth: int, cap: int = 1200, quad: int = 16, lin: int = 32, const: int = 16) -> int:
    """Depth-squared update magnitude shared by every ordering table."""
    b = quad * depth * depth + lin * depth + const
    return b if b < cap else cap


def hist_update(value: int, bonus: int) -> int:
    """Gravity update: bounded, self-damping history increment."""
    v = value + bonus - value * abs(bonus) // HISTORY_MAX
    if v > HISTORY_MAX:
        return HISTORY_MAX
    if v < -HISTORY_MAX:
        return -HISTORY_MAX
    return v


def _mix64(x: int) -> int:
    x &= 0xFFFFFFFFFFFFFFFF
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def pawn_key(board: Board) -> int:
    """Splitmix-style hash of both pawn bitboards (correction/pawn-hist index)."""
    bb = board._bb
    x = (bb[0] * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x ^= (bb[6] + 0x632BE59BD9B4E019) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 31
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 29
    return int(x)


def nonpawn_key(board: Board, color: int) -> int:
    """Hash of ``color``'s non-pawn, non-king occupancy (minors and majors)."""
    bb = board._bb
    base = color * 6
    x = (bb[base + KNIGHT] * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x ^= (bb[base + BISHOP] * 0x632BE59BD9B4E019) & 0xFFFFFFFFFFFFFFFF
    x ^= (bb[base + ROOK] * 0xB7E151628AED2A6B) & 0xFFFFFFFFFFFFFFFF
    x ^= (bb[base + QUEEN] * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 31
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 29
    return int(x)


class HistoryTables:
    """All per-game learned ordering/correction state.

    Ordering tables
        quiet[stm][frm][to]        main quiet ("butterfly") history
        capture[piece][to][victim] capture history, victim 0..5 (+6 = none kept
                                   for promotion ordering uniformity)
        cont[k][piece][to][pp][pt] continuation history: move (piece,to) scored
                                   against the move k plies up the path
        counter[piece][to]         refutation move for ordering bonus
        pawn[bucket][piece][to]    pawn-structure-conditioned quiet history
        threat[frm_th][to_th][frm][to]
                                   quiet history conditioned on whether the
                                   from/to squares are enemy-attacked

    ``prev`` tuples are ``(piece, to)`` where ``piece`` is the full piece
    code 0..11 of the move made k plies up the path, or ``piece == -1``
    (``engine.board.EMPTY``) when no such move exists (root margin, or the
    path move was a null). ``-1`` is the sentinel — never ``0``: a white
    pawn is piece code 0 and must not collide with "no previous move".

    Correction tables (value ``* CORR_GRAIN``)
        corr[0][stm][pawn_key]     pawn-structure correction
        corr[1][stm][nonpawn_w]    white non-pawn correction
        corr[2][stm][nonpawn_b]    black non-pawn correction
    """

    __slots__ = (
        "quiet",
        "capture",
        "cont",
        "counter",
        "pawn",
        "threat",
        "corr",
        "corr_weight_cap",
        "corr_pawn_w",
        "corr_np_w",
        "pawn_div",
        "threat_div",
        "bonus_quad",
        "bonus_lin",
        "bonus_const",
    )

    def __init__(
        self,
        *,
        corr_weight_cap: int = CORR_WEIGHT_CAP,
        corr_pawn_w: int = 2,
        corr_np_w: int = 1,
        pawn_div: int = 2,
        threat_div: int = 2,
        bonus_quad: int = 16,
        bonus_lin: int = 32,
        bonus_const: int = 16,
    ) -> None:
        self.quiet = np.zeros((2, 64, 64), dtype=np.int32)
        self.capture = np.zeros((13, 64, 7), dtype=np.int32)
        self.cont = np.zeros((2, 13, 64, 13, 64), dtype=np.int32)
        self.counter = np.zeros((13, 64), dtype=np.int32)
        self.pawn = np.zeros((PAWN_HIST_SIZE, 13, 64), dtype=np.int32)
        self.threat = np.zeros((2, 2, 64, 64), dtype=np.int32)
        self.corr = np.zeros((3, 2, CORR_SIZE), dtype=np.int32)
        self.corr_weight_cap = corr_weight_cap
        self.corr_pawn_w = corr_pawn_w
        self.corr_np_w = corr_np_w
        self.pawn_div = pawn_div
        self.threat_div = threat_div
        self.bonus_quad = bonus_quad
        self.bonus_lin = bonus_lin
        self.bonus_const = bonus_const

    def clear(self) -> None:
        """Reset all learned state (once per new game, never per move)."""
        self.quiet.fill(0)
        self.capture.fill(0)
        self.cont.fill(0)
        self.counter.fill(0)
        self.pawn.fill(0)
        self.threat.fill(0)
        self.corr.fill(0)

    # -- ordering scores ----------------------------------------------------

    def quiet_score(
        self,
        board: Board,
        frm: int,
        to: int,
        piece: int,
        prev1: tuple[int, int],
        prev2: tuple[int, int],
    ) -> int:
        """Composite quiet ordering score: main + continuation + pawn + threat."""
        stm = board.side
        s = int(self.quiet[stm, frm, to])
        p1, t1 = prev1
        p2, t2 = prev2
        if p1 >= 0:
            s += int(self.cont[0, p1, t1, piece, to])
        if p2 >= 0:
            s += int(self.cont[1, p2, t2, piece, to])
        s += int(self.pawn[pawn_key(board) % PAWN_HIST_SIZE, piece, to]) // self.pawn_div
        occ = board._occ_all
        them = stm ^ 1
        frm_th = 1 if square_attacked(board, frm, them, occ) else 0
        to_th = 1 if square_attacked(board, to, them, occ) else 0
        s += int(self.threat[frm_th, to_th, frm, to]) // self.threat_div
        return s

    def capture_score(self, piece: int, to: int, victim: int) -> int:
        return int(self.capture[piece, to, victim])

    def counter_move(self, prev1: tuple[int, int]) -> int:
        p1, t1 = prev1
        if p1 < 0:
            return 0
        return int(self.counter[p1, t1])

    # -- ordering updates ---------------------------------------------------

    def update_quiets(
        self,
        board: Board,
        best: int,
        quiets: list[int],
        n_quiets: int,
        depth: int,
        prev1: tuple[int, int],
        prev2: tuple[int, int],
        bonus_cap: int = 1200,
    ) -> None:
        """+bonus to the cutoff quiet, -bonus to every quiet tried before it.

        Must be called while ``board`` still shows the node position (i.e.
        after the cutoff move has been unmade) so piece lookups and threat
        tests see the same occupancy the ordering scores were computed from.
        """
        stm = board.side
        bonus = stat_bonus(depth, bonus_cap, self.bonus_quad, self.bonus_lin, self.bonus_const)
        p1, t1 = prev1
        p2, t2 = prev2
        pidx = pawn_key(board) % PAWN_HIST_SIZE
        occ = board._occ_all
        them = stm ^ 1
        mailbox = board._sq

        for i in range(n_quiets):
            m = quiets[i]
            frm = m & 63
            to = (m >> 6) & 63
            piece = mailbox[frm]
            b = bonus if m == best else -bonus
            self.quiet[stm, frm, to] = hist_update(int(self.quiet[stm, frm, to]), b)
            if p1 >= 0:
                self.cont[0, p1, t1, piece, to] = hist_update(
                    int(self.cont[0, p1, t1, piece, to]), b
                )
            if p2 >= 0:
                self.cont[1, p2, t2, piece, to] = hist_update(
                    int(self.cont[1, p2, t2, piece, to]), b
                )
            self.pawn[pidx, piece, to] = hist_update(int(self.pawn[pidx, piece, to]), b)
            frm_th = 1 if square_attacked(board, frm, them, occ) else 0
            to_th = 1 if square_attacked(board, to, them, occ) else 0
            self.threat[frm_th, to_th, frm, to] = hist_update(
                int(self.threat[frm_th, to_th, frm, to]), b
            )
        if p1 >= 0:
            self.counter[p1, t1] = best

    def update_capture(
        self, piece: int, to: int, victim: int, depth: int, bonus_cap: int = 1200
    ) -> None:
        b = stat_bonus(depth, bonus_cap, self.bonus_quad, self.bonus_lin, self.bonus_const)
        self.capture[piece, to, victim] = hist_update(int(self.capture[piece, to, victim]), b)

    # -- correction history -------------------------------------------------

    def correction_cp(self, board: Board, stm: int) -> int:
        """Centipawn correction to add to the RAW static eval for ``stm``."""
        c = self.corr
        total = (
            self.corr_pawn_w * int(c[0, stm, pawn_key(board) & (CORR_SIZE - 1)])
            + self.corr_np_w * int(c[1, stm, nonpawn_key(board, 0) & (CORR_SIZE - 1)])
            + self.corr_np_w * int(c[2, stm, nonpawn_key(board, 1) & (CORR_SIZE - 1)])
        )
        return total // ((self.corr_pawn_w + 2 * self.corr_np_w) * CORR_GRAIN)

    def update_correction(
        self, board: Board, stm: int, depth: int, raw_static: int, best: int
    ) -> None:
        """Move corrections toward ``best - raw_static`` (the correct residual).

        The target is measured against the *raw* static evaluation, never the
        already-corrected score: feeding the corrected value back solves
        C = (S - R) - C and converges to half the intended offset. Callers in
        the search are responsible for the bound/tactical safeguards (no
        in-check nodes, no tactical best moves, bound direction must support
        the correction).
        """
        target = (best - raw_static) * CORR_GRAIN
        if target > CORR_LIMIT:
            target = CORR_LIMIT
        elif target < -CORR_LIMIT:
            target = -CORR_LIMIT
        weight = depth + 1
        if weight > self.corr_weight_cap:
            weight = self.corr_weight_cap
        c = self.corr
        for table, idx in (
            (0, pawn_key(board) & (CORR_SIZE - 1)),
            (1, nonpawn_key(board, 0) & (CORR_SIZE - 1)),
            (2, nonpawn_key(board, 1) & (CORR_SIZE - 1)),
        ):
            old = int(c[table, stm, idx])
            new = ((256 - weight) * old + weight * target) // 256
            if new > CORR_LIMIT:
                new = CORR_LIMIT
            elif new < -CORR_LIMIT:
                new = -CORR_LIMIT
            c[table, stm, idx] = new

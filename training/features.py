"""Canonical sparse encoders for F512-EF-K12-16/32, generated from the spec.

Implements, for both perspectives:

* PSQ rows (9,216): ``((bucket*2 + relcolour)*6 + pt)*64 + norm_sq`` where the
  king bucket comes from the non-uniform K12 map and ``norm_sq`` is the
  rank-flipped (black perspective), king-side-mirrored square.
* FullThreats rows (59,808): the pinned Stockfish definition
  (``training/reference/full_threats_59aae690.{h,cpp}``, blob sha1
  ``ac4f79da58c983e1a46c989697428e4edf1abbdf`` per ``resources.json``):
  occupied-square threat relations, same-side protection included, kings
  never attacker or target, same-piece-type pairs encoded once
  (``from < to`` orientation suppressed).
* Compact pawn pairs (1,488): 96 rel-colour pawn identities on normalised
  ranks 2..7; unordered pairs on distinct physical squares with file
  distance <= 1; dense map ordered by ``b*(b-1)//2 + a``.

Each block also emits a factorizer row (coarse equivalence class folded at
export): PSQ drops bucket+mirror conditioning (768 rows), threats use the
identity-orientation index (59,808 rows), pawn pairs drop rel-colour
(372 rows).

Only full materialization is implemented here; incremental deltas are
W05/engine scope.  Everything is derived from ``training/feature_spec.SPEC``
constants; the LUT constructions assert the authoritative row counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from training.feature_spec import SPEC, FeatureSpec

# --- square / bitboard helpers -------------------------------------------------
# Square convention matches both Stockfish and python-chess: A1=0 .. H8=63,
# sq = rank*8 + file.

FILES = "abcdefgh"


def _sq(file_: int, rank: int) -> int:
    return rank * 8 + file_


def _pidx(color: chess.Color) -> int:
    """Stockfish colour index: WHITE=0, BLACK=1.

    python-chess uses WHITE=True/BLACK=False, the opposite of Stockfish's
    0/1; every orientation XOR below goes through this.
    """
    return 0 if color == chess.WHITE else 1


def _pseudo_attacks(pt: int, sq: int) -> int:
    """Pseudo-legal attack bitboard on an empty board.

    pt uses python-chess piece types: PAWN=1 .. KING=6.  Pawn attacks here are
    WHITE-oriented; colour is applied by the caller through square flips, so
    this function is only used for the non-pawn types and the king.
    """
    f, r = sq % 8, sq // 8
    bb = 0
    if pt == chess.KNIGHT:
        for df, dr in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)):
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                bb |= 1 << _sq(nf, nr)
        return bb
    if pt == chess.KING:
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df == 0 and dr == 0:
                    continue
                nf, nr = f + df, r + dr
                if 0 <= nf < 8 and 0 <= nr < 8:
                    bb |= 1 << _sq(nf, nr)
        return bb
    dirs: tuple[tuple[int, int], ...]
    if pt == chess.BISHOP:
        dirs = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    elif pt == chess.ROOK:
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    elif pt == chess.QUEEN:
        dirs = ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        raise ValueError(f"no pseudo attacks for piece type {pt}")
    for df, dr in dirs:
        nf, nr = f + df, r + dr
        while 0 <= nf < 8 and 0 <= nr < 8:
            bb |= 1 << _sq(nf, nr)
            nf += df
            nr += dr
    return bb


def _pawn_attacks(color: chess.Color, sq: int) -> int:
    f, r = sq % 8, sq // 8
    dr = 1 if color == chess.WHITE else -1
    bb = 0
    for df in (-1, 1):
        nf, nr = f + df, r + dr
        if 0 <= nf < 8 and 0 <= nr < 8:
            bb |= 1 << _sq(nf, nr)
    return bb


def _slider_attacks(pt: int, sq: int, occupied: int) -> int:
    """Occupancy-aware attacks for B/R/Q; pseudo attacks for N/K."""
    if pt in (chess.KNIGHT, chess.KING):
        return _pseudo_attacks(pt, sq)
    f, r = sq % 8, sq // 8
    dirs: tuple[tuple[int, int], ...]
    if pt == chess.BISHOP:
        dirs = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    elif pt == chess.ROOK:
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    elif pt == chess.QUEEN:
        dirs = ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        raise ValueError(pt)
    bb = 0
    for df, dr in dirs:
        nf, nr = f + df, r + dr
        while 0 <= nf < 8 and 0 <= nr < 8:
            t = _sq(nf, nr)
            bb |= 1 << t
            if (occupied >> t) & 1:
                break
            nf += df
            nr += dr
    return bb


# --- FullThreats lookup tables (pinned source port) ---------------------------
# Stockfish piece numbering: W_PAWN=1..W_KING=6, B_PAWN=9..B_KING=14.
def _sf_piece(color: chess.Color, pt: int) -> int:
    return pt if color == chess.WHITE else pt + 8


_NUM_VALID_TARGETS = {1: 4, 2: 10, 3: 8, 4: 8, 5: 10, 6: 0}

# map[attacker_pt-1][attacked_pt-1] from the pinned header.
_THREAT_MAP = (
    (-1, 0, -1, 1, -1, -1),  # pawn  -> N,R only
    (0, 1, 2, 3, 4, -1),  # knight-> P,N,B,R,Q
    (0, 1, 2, 3, -1, -1),  # bishop-> P,N,B,R
    (0, 1, 2, 3, -1, -1),  # rook  -> P,N,B,R
    (0, 1, 2, 3, 4, -1),  # queen -> P,N,B,R,Q
    (-1, -1, -1, -1, -1, -1),  # king  -> none
)

# OrientTBL from the pinned header: file a-d -> 0, file e-h -> 7.
_ORIENT_TBL = tuple(0 if sq % 8 < 4 else 7 for sq in range(64))


class _ThreatTables:
    """LUTs exactly mirroring init_threat_offsets / init_index_luts /
    index_lut2_array of the pinned full_threats.cpp."""

    def __init__(self, rows: int) -> None:
        self.rows = rows
        all_pieces = list(range(1, 7)) + list(range(9, 15))

        def pseudo(piece: int, sq: int) -> int:
            pt = piece % 8  # strip colour bit
            if pt == chess.PAWN:
                return _pawn_attacks(piece < 8, sq)
            return _pseudo_attacks(pt, sq)

        # index_lut2[piece][from][to]: rank of `to` among pseudo-attack targets
        lut2 = np.zeros((16, 64, 64), dtype=np.int32)
        for p in all_pieces:
            for frm in range(64):
                att = pseudo(p, frm)
                for to in range(64):
                    lut2[p, frm, to] = bin(((1 << to) - 1) & att).count("1")

        # offsets[piece][from] + helper_offsets[piece]
        offsets = np.zeros((16, 64), dtype=np.int64)
        helper_piece_offset = np.zeros(16, dtype=np.int64)
        helper_cum_offset = np.zeros(16, dtype=np.int64)
        cumulative = 0
        for p in all_pieces:
            piece_offset = 0
            for frm in range(64):
                offsets[p, frm] = piece_offset
                pt = p % 8
                if pt != chess.PAWN:
                    piece_offset += bin(pseudo(p, frm)).count("1")
                elif 8 <= frm <= 55:  # A2..H7 only
                    piece_offset += bin(pseudo(p, frm)).count("1")
            helper_piece_offset[p] = piece_offset
            helper_cum_offset[p] = cumulative
            cumulative += _NUM_VALID_TARGETS[pt_of(p)] * piece_offset

        self.total_dimensions = cumulative
        self.offsets = offsets
        self.helper_piece_offset = helper_piece_offset
        self.helper_cum_offset = helper_cum_offset

        # index_lut1[attacker][attacked][from<to]
        lut1 = np.full((16, 16, 2), rows, dtype=np.int64)
        for attacker in all_pieces:
            at_pt = pt_of(attacker)
            for attacked in all_pieces:
                ad_pt = pt_of(attacked)
                enemy = (attacker ^ attacked) == 8
                m = _THREAT_MAP[at_pt - 1][ad_pt - 1]
                semi_excluded = at_pt == ad_pt and (enemy or at_pt != chess.PAWN)
                feature = (
                    helper_cum_offset[attacker]
                    + ((1 if attacked >= 8 else 0) * (_NUM_VALID_TARGETS[at_pt] // 2) + m)
                    * helper_piece_offset[attacker]
                )
                excluded = m < 0
                lut1[attacker, attacked, 0] = rows if excluded else feature
                lut1[attacker, attacked, 1] = rows if (excluded or semi_excluded) else feature
        self.lut1 = lut1
        self.lut2 = lut2

    def make_index(
        self,
        perspective: chess.Color,
        attacker: int,
        frm: int,
        to: int,
        attacked: int,
        ksq: int,
    ) -> int:
        p = _pidx(perspective)
        orientation = _ORIENT_TBL[ksq] ^ (56 * p)
        fo = frm ^ orientation
        to_o = to ^ orientation
        swap = 8 * p
        a = attacker ^ swap
        d = attacked ^ swap
        return int(
            self.lut1[a, d, 1 if fo < to_o else 0] + self.offsets[a, fo] + self.lut2[a, fo, to_o]
        )

    def make_index_unoriented(self, attacker: int, frm: int, to: int, attacked: int) -> int:
        """Same LUT arithmetic under identity orientation (no mirror/flip)."""
        return int(
            self.lut1[attacker, attacked, 1 if frm < to else 0]
            + self.offsets[attacker, frm]
            + self.lut2[attacker, frm, to]
        )

    def enumerate_relations(self) -> dict[int, tuple[int, int, int, int]]:
        """index -> (attacker, attacked, from, to) for every valid row.

        Inverts the LUT once at init so factorization can be a pure function
        of the stored row id (required for exact export folding).
        """
        inv: dict[int, tuple[int, int, int, int]] = {}
        all_pieces = list(range(1, 7)) + list(range(9, 15))
        for a in all_pieces:
            for frm in range(64):
                pt = pt_of(a)
                if pt == chess.PAWN and not (8 <= frm <= 55):
                    continue
                att = _pawn_attacks(a < 8, frm) if pt == chess.PAWN else _pseudo_attacks(pt, frm)
                while att:
                    to = (att & -att).bit_length() - 1
                    att &= att - 1
                    for d in all_pieces:
                        for parity in (0, 1):
                            idx = int(
                                self.lut1[a, d, parity]
                                + self.offsets[a, frm]
                                + self.lut2[a, frm, to]
                            )
                            want = 1 if frm < to else 0
                            if parity == want and idx < self.rows:
                                prev = inv.get(idx)
                                if prev is not None and prev != (a, d, frm, to):
                                    raise AssertionError(
                                        f"threat index collision {idx}: {prev} vs {(a, d, frm, to)}"
                                    )
                                inv[idx] = (a, d, frm, to)
        return inv

    def build_row_to_fac(self) -> np.ndarray:
        """row -> canonical-orientation factor row.

        Factor = minimum unoriented index of the same relative relation under
        the orientation group {identity, file-mirror, rank-flip, 180-rot},
        considering only transforms under which the relation is encodable
        (respects the same-type parity exclusion).
        """
        inv = self.enumerate_relations()
        out = np.arange(self.rows, dtype=np.int64)
        for idx, (a, d, frm, to) in inv.items():
            best = self.rows
            for t in (0, 7, 56, 63):
                cand = self.make_index_unoriented(a, frm ^ t, to ^ t, d)
                if cand < best:
                    best = cand
            out[idx] = best
        return out


def pt_of(sf_piece: int) -> int:
    return sf_piece % 8


# --- pawn-pair dense map --------------------------------------------------------
def _build_pp_maps(spec: FeatureSpec) -> tuple[np.ndarray, dict[tuple[int, int], int], int]:
    """identity id = relcolour*48 + (norm_rank-1)*8 + file, norm_rank in 1..6.

    Returns (pair_map[b,a] -> dense index or -1, factor_map[(sq_lo,sq_hi)] ->
    dense index, factor_rows).
    """
    pair_map = np.full((96, 96), -1, dtype=np.int32)
    factor_map: dict[tuple[int, int], int] = {}
    next_idx = 0
    for b in range(96):
        sb = b % 48
        for a in range(b):
            sa = a % 48
            if sa == sb:
                continue  # distinct physical squares required
            if abs(sa % 8 - sb % 8) > 1:
                continue  # file distance <= 1
            # iteration order is exactly the triangular order b*(b-1)//2 + a
            pair_map[b, a] = next_idx
            next_idx += 1
            lo, hi = (sa, sb) if sa < sb else (sb, sa)
            factor_map.setdefault((lo, hi), len(factor_map))
    assert next_idx == spec.pawn_pairs.rows == 1488
    assert len(factor_map) == spec.pawn_pairs.factorizer_rows == 372
    return pair_map, factor_map, len(factor_map)


def _build_psq_row_to_fac(spec: FeatureSpec) -> np.ndarray:
    """PSQ row -> factor row: drop the king bucket, keep (rel, pt, nsq)."""
    out = np.zeros(spec.psq.rows, dtype=np.int64)
    for r in range(spec.psq.rows):
        bucket, rem = divmod(r, 768)
        rel_pt, sq = divmod(rem, 64)
        rel, pt = divmod(rel_pt, 6)
        out[r] = (rel * 6 + pt) * 64 + sq
    return out


@dataclass
class Encoded:
    """Sparse rows for one position, both perspectives.

    Indices are int32 row ids within each row space; ``*_fac`` are the
    factorizer row ids within their own factorizer spaces.
    """

    psq: tuple[np.ndarray, np.ndarray]
    threats: tuple[np.ndarray, np.ndarray]
    pawn_pairs: tuple[np.ndarray, np.ndarray]
    psq_fac: tuple[np.ndarray, np.ndarray]
    threats_fac: tuple[np.ndarray, np.ndarray]
    pawn_pairs_fac: tuple[np.ndarray, np.ndarray]
    head: int
    piece_count: int

    def flat(self, space: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rows_cat, fac_cat, segments) for space in
        {'psq','threats','pawn_pairs'}: concatenated row ids over both
        perspectives and segment starts [p0_end, total]."""
        rows = getattr(self, space)
        fac = getattr(self, space + "_fac")
        rows_cat = np.concatenate([rows[0], rows[1]]).astype(np.int64)
        fac_cat = np.concatenate([fac[0], fac[1]]).astype(np.int64)
        seg = np.array([len(rows[0]), len(rows[0]) + len(rows[1])], np.int64)
        return rows_cat, fac_cat, seg


class FeatureEncoder:
    """Board -> sparse row indices, generated from the FeatureSpec."""

    def __init__(self, spec: FeatureSpec = SPEC) -> None:
        self.spec = spec
        self.threats = _ThreatTables(spec.threats.rows)
        assert self.threats.total_dimensions == spec.threats.rows == 59808
        self.pp_map, self.pp_factor_map, _ = _build_pp_maps(spec)
        self.king_map = np.asarray(spec.king_bucket_map, dtype=np.int32)
        # row -> factorizer-row maps (pure projections of the stored row id;
        # export folds fac[g(r)] into row r under the same maps)
        self.psq_row_to_fac = _build_psq_row_to_fac(spec)
        self.thr_row_to_fac = self.threats.build_row_to_fac()
        self.pp_row_to_fac = np.zeros(spec.pawn_pairs.rows, dtype=np.int64)
        for b in range(96):
            for a in range(b):
                idx = self.pp_map[b, a]
                if idx >= 0:
                    sa, sb = a % 48, b % 48
                    lo, hi = (sa, sb) if sa < sb else (sb, sa)
                    self.pp_row_to_fac[idx] = self.pp_factor_map[(lo, hi)]

    # -- helpers ---------------------------------------------------------------
    def material_head(self, board: chess.Board) -> int:
        n = bin(board.occupied).count("1")
        return min(7, max(0, (n - 2) // 4))

    def _norm(self, sq: int, perspective: chess.Color, mirror: int) -> int:
        return sq ^ (56 * _pidx(perspective)) ^ mirror

    def _king_frame(self, board: chess.Board, perspective: chess.Color) -> tuple[int, int, int]:
        """Return (mirror_bit, normalized king square, bucket)."""
        ksq = board.king(perspective)
        mirror = 7 if (ksq % 8) >= 4 else 0
        nksq = self._norm(ksq, perspective, mirror)
        bucket = int(self.king_map[nksq // 8, nksq % 8])
        return mirror, nksq, bucket

    # -- encoders ---------------------------------------------------------------
    def encode(self, board: chess.Board) -> Encoded:
        out = []
        for perspective in (board.turn, not board.turn):
            out.append(self._encode_perspective(board, perspective))
        (psq, thr, pp, psq_f, thr_f, pp_f) = zip(*out, strict=True)
        return Encoded(
            psq=(psq[0], psq[1]),
            threats=(thr[0], thr[1]),
            pawn_pairs=(pp[0], pp[1]),
            psq_fac=(psq_f[0], psq_f[1]),
            threats_fac=(thr_f[0], thr_f[1]),
            pawn_pairs_fac=(pp_f[0], pp_f[1]),
            head=self.material_head(board),
            piece_count=bin(board.occupied).count("1"),
        )

    def _encode_perspective(
        self, board: chess.Board, perspective: chess.Color
    ) -> tuple[np.ndarray, ...]:
        mirror, nksq, bucket = self._king_frame(board, perspective)
        ksq = board.king(perspective)  # raw square, for threat orientation

        psq_rows: list[int] = []
        pawn_ids: list[int] = []
        occupied = int(board.occupied)

        for sq, piece in board.piece_map().items():
            pt = piece.piece_type
            rel = 0 if piece.color == perspective else 1
            nsq = self._norm(sq, perspective, mirror)
            psq_rows.append(((bucket * 2 + rel) * 6 + (pt - 1)) * 64 + nsq)
            if pt == chess.PAWN:
                nrank = nsq // 8
                if 1 <= nrank <= 6:  # normalised ranks 2..7 only
                    pawn_ids.append(rel * 48 + (nrank - 1) * 8 + nsq % 8)

        # threats ---------------------------------------------------------------
        thr_rows: list[int] = []
        pawn_t = int(board.knights | board.rooks)
        minor_t = int(board.pawns | board.knights | board.bishops | board.rooks)
        queen_t = minor_t | int(board.queens)

        for c in (chess.WHITE, chess.BLACK):
            for frm in chess.SquareSet(board.pawns & board.occupied_co[c]):
                for to in chess.SquareSet(_pawn_attacks(c, frm) & pawn_t):
                    attacked = board.piece_at(to)
                    assert attacked is not None
                    idx = self.threats.make_index(
                        perspective,
                        _sf_piece(c, chess.PAWN),
                        frm,
                        to,
                        _sf_piece(attacked.color, attacked.piece_type),
                        ksq,
                    )
                    if idx < self.spec.threats.rows:
                        thr_rows.append(idx)

        for c in (chess.WHITE, chess.BLACK):
            for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                bb = int(board.pieces(pt, c))
                targets = queen_t if pt in (chess.KNIGHT, chess.QUEEN) else minor_t
                while bb:
                    frm = (bb & -bb).bit_length() - 1
                    bb &= bb - 1
                    att = _slider_attacks(pt, frm, occupied) & targets
                    while att:
                        to = (att & -att).bit_length() - 1
                        att &= att - 1
                        attacked = board.piece_at(to)
                        assert attacked is not None
                        idx = self.threats.make_index(
                            perspective,
                            _sf_piece(c, pt),
                            frm,
                            to,
                            _sf_piece(attacked.color, attacked.piece_type),
                            ksq,
                        )
                        if idx < self.spec.threats.rows:
                            thr_rows.append(idx)

        # pawn pairs --------------------------------------------------------------
        pp_rows: list[int] = []
        pawn_ids.sort()
        for j in range(len(pawn_ids)):
            b = pawn_ids[j]
            for i in range(j):
                idx = self.pp_map[b, pawn_ids[i]]
                if idx >= 0:
                    pp_rows.append(int(idx))

        psq_a = np.asarray(sorted(set(psq_rows)), dtype=np.int32)
        thr_a = np.asarray(sorted(set(thr_rows)), dtype=np.int32)
        pp_a = np.asarray(sorted(set(pp_rows)), dtype=np.int32)
        return (
            psq_a,
            thr_a,
            pp_a,
            self.psq_row_to_fac[psq_a].astype(np.int32),
            self.thr_row_to_fac[thr_a].astype(np.int32),
            self.pp_row_to_fac[pp_a].astype(np.int32),
        )


def canonical_board_key(board: chess.Board) -> str:
    """Canonical legal-EP board identity: pieces, side, castling, EP.

    EP is emitted only when a legal EP capture exists (``en_passant='legal'``),
    so positions differing only in an unusable EP flag share a key.
    """
    parts = board.fen(en_passant="legal").split(" ")
    return " ".join(parts[:4])


def legal_context_key(board: chess.Board, history_complete: bool, unknown_prefix: bool) -> str:
    """Value-context identity: counters/history state, kept separate from the
    geometric board key per spec section 9."""
    return "|".join(
        (
            canonical_board_key(board),
            f"hm={board.halfmove_clock}",
            f"fm={board.fullmove_number}",
            f"hc={int(history_complete)}",
            f"up={int(unknown_prefix)}",
        )
    )

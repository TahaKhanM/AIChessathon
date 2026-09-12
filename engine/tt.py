"""Packed move/static/score transposition table with context rules.

Default 128 MiB, packed clustered entries, measured replacement policy;
sweep 64/128/256/512 with real long-game reuse. Rule50/history qualification
per spec sections 3.3; strict audit mode versus declared-heuristic reuse.

Three distinct identities are maintained (spec 3.3):

1. **Geometric identity** — the Zobrist key (pieces, turn, castling, relevant
   EP). It selects the cluster and matches the 32-bit tag.
2. **Value context** — halfmove counter, remaining absolute horizon, model and
   utility version. Score *cutoffs* are gated on the value context; a legal
   move hint remains usable even when the score is not.
3. **Repetition context** — a coarse band of the position's known
   repetition count plus the unknown-prefix flag. Entries stored under a
   different repetition context may carry draw-flavoured scores, so score
   cutoffs require the band to match; move hints never consult it.

Declared production heuristics (per spec 3.3, "coarse counter bands or less
strict reuse as *declared heuristics*"):

- ``RULE50_CUTOFF_MAX``: when the *current* halfmove clock is at or above this
  value, no stored score produces a cutoff — the subtree that produced it may
  have priced fifty-move draws differently. Stored entries written *at* high
  halfmove are likewise ineligible for cutoff under a low current halfmove.
- ``HORIZON_SLACK``: a stored score is horizon-safe iff the storing subtree
  could not have seen the 600-ply cap (``entry.horizon > entry.depth +
  HORIZON_SLACK``) *and* crediting it now cannot reach the cap either
  (``current horizon > entry.depth + HORIZON_SLACK``). Otherwise only an
  exact horizon-band match qualifies the score.
- ``rep_band`` is ``min(3, repetition_count)`` at the storing node; score
  cutoffs require equality, and the ``unknown_prefix`` flag must match.
- The ``eval`` slot stores the RAW neural/static evaluation only. Online
  correction is applied by the consumer; a corrected score is never stored as
  the immutable raw value.

``StrictTable`` is the audit oracle: a dict keyed by the *complete* score
identity (geometric key, exact halfmove, exact remaining horizon, model and
utility version, unknown-prefix flag and a fingerprint of the full known
reversible history). ``audit`` compares production reuse against it on
counterfactual histories and near-draw states.

Entry layout — 16 bytes, 4 entries per 64-byte cluster:

word0 ``[0:32)`` key tag  ``[32:47)`` move15  ``[47:55)`` depth8
      ``[55:57)`` bound2  ``[57:63)`` age6
word1 ``[0:16)`` score16  ``[16:32)`` eval16  ``[32:39)`` halfmove7
      ``[39:44)`` horizon-band5  ``[44:52)`` model8  ``[52:56)`` util4
      ``[56:58)`` rep-band2  ``[58)`` unknown-prefix1
"""

from __future__ import annotations

import numpy as np

MATE = 30000
INF = 32000
# Mate-in-N scores occupy [MATE_IN_MAX, MATE); MATE_IN_MAX must stay below
# any real eval bound and above the ply at which mate is delivered.
MATE_IN_MAX = MATE - 2048

BOUND_EMPTY, BOUND_UPPER, BOUND_LOWER, BOUND_EXACT = 0, 1, 2, 3

CLUSTER = 4
DEFAULT_MIB = 128
SWEEP_MIB = (64, 128, 256, 512)

# Declared heuristic constants (see module docstring).
RULE50_CUTOFF_MAX = 90
HORIZON_SLACK = 96
HORIZON_BAND = 20

AGE_MASK = 63
MASK64 = 0xFFFFFFFFFFFFFFFF


def _s64(x: int) -> int:
    """Python int -> signed int64 for numpy assignment (np.uint64() of a
    value >= 2**63 raises OverflowError; int64 storage avoids it)."""
    return x - 0x10000000000000000 if x >> 63 else x


def _u64(x: int) -> int:
    return x & MASK64


def score_to_tt(score: int, ply: int, mate_min: int = MATE_IN_MAX) -> int:
    """Normalize a mate score to "plies to mate from this node" on store."""
    if score >= mate_min:
        return score + ply
    if score <= -mate_min:
        return score - ply
    return score


def score_from_tt(score: int, ply: int, mate_min: int = MATE_IN_MAX) -> int:
    """Denormalize a stored mate score to "plies to mate from root" on load."""
    if score >= mate_min:
        return score - ply
    if score <= -mate_min:
        return score + ply
    return score


def horizon_band(remaining_horizon: int) -> int:
    b = remaining_horizon // HORIZON_BAND
    return 31 if b > 31 else b


def rep_band(count: int) -> int:
    return count if count < 3 else 3


class Probe:
    """Result of a TT probe. ``score``/``cutoff_ok`` are only meaningful on hit."""

    __slots__ = (
        "hit",
        "move16",
        "score",
        "raw_eval",
        "depth",
        "bound",
        "cutoff_ok",
        "eval_ok",
        "slot",
    )

    def __init__(self) -> None:
        self.hit = False
        self.move16 = 0
        self.score = 0
        self.raw_eval = -INF
        self.depth = 0
        self.bound = BOUND_EMPTY
        self.cutoff_ok = False
        self.eval_ok = False
        self.slot = -1

    def __repr__(self) -> str:
        return (
            f"Probe(hit={self.hit}, move16={self.move16}, score={self.score}, "
            f"eval={self.raw_eval}, depth={self.depth}, bound={self.bound}, "
            f"cutoff_ok={self.cutoff_ok})"
        )


class ValueContext:
    """The non-geometric identity a stored score was produced under."""

    __slots__ = ("halfmove", "horizon", "model", "util", "rep", "unknown")

    def __init__(
        self,
        halfmove: int,
        remaining_horizon: int,
        model_version: int,
        utility_version: int,
        rep_count: int,
        unknown_prefix: bool,
    ) -> None:
        self.halfmove = halfmove
        self.horizon = remaining_horizon
        self.model = model_version
        self.util = utility_version
        self.rep = rep_count
        self.unknown = unknown_prefix

    def score_cutoff_ok(
        self,
        e_halfmove: int,
        e_horizon: int,
        e_depth: int,
        e_model: int,
        e_util: int,
        e_rep: int,
        e_unknown: int,
    ) -> bool:
        """Production heuristic gate for reusing a stored *score*.

        Move hints are never gated by this; only cutoffs/exact returns are.
        """
        if e_model != (self.model & 0xFF) or e_util != (self.util & 0xF):
            return False
        if e_unknown != (1 if self.unknown else 0):
            return False
        if e_rep != rep_band(self.rep):
            return False
        # rule50 / fifty-move proximity
        if self.halfmove >= RULE50_CUTOFF_MAX or e_halfmove >= RULE50_CUTOFF_MAX:
            return False
        # 600-ply-cap proximity. ``e_horizon`` is a band (lower bound =
        # e_horizon * HORIZON_BAND plies); the stored subtree provably could
        # not see the cap only when that lower bound exceeds depth + slack.
        stored_safe = e_horizon * HORIZON_BAND > e_depth + HORIZON_SLACK
        current_safe = self.horizon > e_depth + HORIZON_SLACK
        if not (stored_safe and current_safe):
            if horizon_band(self.horizon) != e_horizon:
                return False
        return True


def _pack(
    key: int,
    move16: int,
    score: int,
    raw_eval: int,
    depth: int,
    bound: int,
    age: int,
    ctx: ValueContext,
) -> tuple[int, int]:
    w0 = (key >> 32) & 0xFFFFFFFF
    w0 |= (move16 & 0x7FFF) << 32
    w0 |= (depth & 0xFF) << 47
    w0 |= (bound & 3) << 55
    w0 |= (age & AGE_MASK) << 57
    w1 = score & 0xFFFF
    w1 |= (raw_eval & 0xFFFF) << 16
    w1 |= (min(ctx.halfmove, 127) & 0x7F) << 32
    w1 |= (horizon_band(ctx.horizon) & 0x1F) << 39
    w1 |= (ctx.model & 0xFF) << 44
    w1 |= (ctx.util & 0xF) << 52
    w1 |= (rep_band(ctx.rep) & 3) << 56
    if ctx.unknown:
        w1 |= 1 << 58
    # Top 4 bits of w1 echo the top 4 key bits: a store writes payload (w1)
    # before meta (w0), so a torn pair fails this tag check instead of
    # pairing new data with a stale bound.
    w1 |= (key >> 60) << 60
    return w0, w1


def _unpack(w0: int, w1: int) -> tuple[int, int, int, int, int, int]:
    """Return (move16, score, raw_eval, depth, bound, age)."""
    move16 = (w0 >> 32) & 0x7FFF
    depth = (w0 >> 47) & 0xFF
    bound = (w0 >> 55) & 3
    age = (w0 >> 57) & AGE_MASK
    score = w1 & 0xFFFF
    if score >= 0x8000:
        score -= 0x10000
    raw_eval = (w1 >> 16) & 0xFFFF
    if raw_eval >= 0x8000:
        raw_eval -= 0x10000
    return move16, score, raw_eval, depth, bound, age


def _unpack_ctx(w1: int) -> tuple[int, int, int, int, int, int]:
    """Return (halfmove, horizon_band, model, util, rep_band, unknown)."""
    return (
        (w1 >> 32) & 0x7F,
        (w1 >> 39) & 0x1F,
        (w1 >> 44) & 0xFF,
        (w1 >> 52) & 0xF,
        (w1 >> 56) & 3,
        (w1 >> 58) & 1,
    )


class TranspositionTable:
    """Clustered packed TT, ``mib`` mebibytes, 4 entries per 64-byte cluster."""

    def __init__(self, mib: int = DEFAULT_MIB) -> None:
        total_bytes = mib << 20
        n_clusters = max(1, total_bytes // (CLUSTER * 16))
        # power-of-two cluster count for mask indexing
        p = 1
        while p * 2 <= n_clusters:
            p <<= 1
        self.clusters = np.zeros((p, CLUSTER, 2), dtype=np.int64)
        self.n_clusters = p
        self.mask = p - 1
        self.age = 0
        self.writes = 0
        self.replaces = 0

    @property
    def bytes(self) -> int:
        return int(self.clusters.nbytes)

    def clear(self) -> None:
        self.clusters.fill(0)
        self.age = 0
        self.writes = 0
        self.replaces = 0

    def new_search(self) -> None:
        self.age = (self.age + 1) & AGE_MASK

    def probe_into(self, out: Probe, key: int, ply: int, ctx: ValueContext | None = None) -> Probe:
        """Fill ``out`` with the probe result (no allocation in the hot loop)."""
        out.hit = False
        out.move16 = 0
        out.raw_eval = -INF
        out.depth = 0
        out.bound = BOUND_EMPTY
        out.cutoff_ok = False
        out.eval_ok = False
        out.slot = -1
        key &= MASK64
        idx = key & self.mask
        tag = (key >> 32) & 0xFFFFFFFF
        tag_hi = (key >> 60) & 0xF
        cluster = self.clusters[idx]
        for slot in range(CLUSTER):
            w0 = int(cluster[slot, 0]) & MASK64
            if (w0 & 0xFFFFFFFF) != tag or ((w0 >> 55) & 3) == BOUND_EMPTY:
                continue
            w1 = int(cluster[slot, 1]) & MASK64
            if ((w1 >> 60) & 0xF) != tag_hi:
                continue  # torn or colliding pair
            move16, score, raw_eval, depth, bound, age = _unpack(w0, w1)
            score = score_from_tt(score, ply)
            out.hit = True
            out.move16 = move16
            out.score = score
            out.raw_eval = raw_eval
            out.depth = depth
            out.bound = bound
            out.slot = idx * CLUSTER + slot
            if ctx is not None:
                e_half, e_hband, e_model, e_util, e_rep, e_unk = _unpack_ctx(w1)
                out.cutoff_ok = ctx.score_cutoff_ok(
                    e_half, e_hband, depth, e_model, e_util, e_rep, e_unk
                )
                # Raw-eval reuse is model-identity only — a raw eval is the
                # immutable network output for this position; halfmove /
                # horizon / repetition context cannot change it.
                out.eval_ok = e_model == (ctx.model & 0xFF) and e_util == (ctx.util & 0xF)
            return out
        return out

    def probe(self, key: int, ply: int, ctx: ValueContext | None = None) -> Probe:
        """Allocating probe for tools/tests; hot loop uses :meth:`probe_into`."""
        return self.probe_into(Probe(), key, ply, ctx)

    def store(
        self,
        key: int,
        move16: int,
        score: int,
        raw_eval: int,
        depth: int,
        bound: int,
        ctx: ValueContext,
        ply: int,
    ) -> None:
        """Store under geometric key; mate scores normalized by ``ply``.

        ``raw_eval`` is the RAW static/neural evaluation — callers must never
        pass an online-corrected score into this slot.
        """
        score = score_to_tt(score, ply)
        if score > 32767:
            score = 32767
        elif score < -32768:
            score = -32768
        if raw_eval > 32767:
            raw_eval = 32767
        elif raw_eval < -32768:
            raw_eval = -32768
        if depth < 0:
            depth = 0
        key &= MASK64
        idx = key & self.mask
        tag = (key >> 32) & 0xFFFFFFFF
        cluster = self.clusters[idx]
        target = -1
        best_q = 1 << 30
        for slot in range(CLUSTER):
            w0 = int(cluster[slot, 0]) & MASK64
            obound = (w0 >> 55) & 3
            if obound == BOUND_EMPTY:
                target = slot
                break
            if (w0 & 0xFFFFFFFF) == tag:
                target = slot
                omove, _s, _e, odepth, obound, oage = _unpack(w0, int(cluster[slot, 1]) & MASK64)
                if move16 == 0:
                    move16 = omove  # never lose a stored hint for the same key
                # Deliberate same-key policy (mirrored in kernels/tthist.py):
                # worth = depth (+2 when EXACT) with each generation of age
                # costing 4 plies — the same scale cross-key eviction uses
                # below. Replace only on a strict win; a tie replaces iff the
                # new entry is not shallower. A deeper EXACT therefore
                # survives a shallower same-key entry of ANY bound, a
                # same-depth-or-deeper EXACT always supersedes a bound, and
                # staleness erodes the incumbent 4 plies per generation.
                q_new = depth + (2 if bound == BOUND_EXACT else 0)
                q_old = (
                    odepth
                    + (2 if obound == BOUND_EXACT else 0)
                    - 4 * ((self.age - oage) & AGE_MASK)
                )
                if q_new < q_old or (q_new == q_old and depth < odepth):
                    return
                break
            _m, _s, _e, odepth, _b, oage = _unpack(w0, int(cluster[slot, 1]) & MASK64)
            q = odepth - 4 * ((self.age - oage) & AGE_MASK)
            if q < best_q:
                best_q = q
                target = slot
        w0, w1 = _pack(key, move16, score, raw_eval, depth, bound, self.age, ctx)
        if int(cluster[target, 0]) != 0:
            self.replaces += 1
        # Payload before meta: a pair torn between the writes fails the
        # tag_hi check at probe time instead of yielding a half-new entry.
        cluster[target, 1] = _s64(w1)
        cluster[target, 0] = _s64(w0)
        self.writes += 1

    def hashfull(self) -> int:
        """Occupancy in per-mille over the first 1000 clusters (diagnostic)."""
        n = min(1000, self.n_clusters)
        used = 0
        for c in range(n):
            for slot in range(CLUSTER):
                w0 = int(self.clusters[c, slot, 0]) & MASK64
                if ((w0 >> 55) & 3) != BOUND_EMPTY:
                    used += 1
        return used * 1000 // (n * CLUSTER)

    def audit_entries(self):
        """Yield (key_tag, move16, score, raw_eval, depth, bound, age, ctx_tuple).

        Debug/audit iterator over every occupied entry — gate tooling scans
        this to assert no entry was corrupted by an aborted search.
        """
        for c in range(self.n_clusters):
            for slot in range(CLUSTER):
                w0 = int(self.clusters[c, slot, 0]) & MASK64
                bound = (w0 >> 55) & 3
                if bound == BOUND_EMPTY:
                    continue
                w1 = int(self.clusters[c, slot, 1]) & MASK64
                move16, score, raw_eval, depth, _b, age = _unpack(w0, w1)
                yield (
                    w0 & 0xFFFFFFFF,
                    move16,
                    score,
                    raw_eval,
                    depth,
                    bound,
                    age,
                    _unpack_ctx(w1),
                )


class StrictTable:
    """Audit oracle: score identity = COMPLETE known history + exact counters.

    A hit is returned only when every component of the score identity matches:
    geometric key, exact halfmove, exact remaining horizon, model/utility
    version, unknown-prefix flag and a fingerprint of the whole known
    reversible key sequence. Production reuse is compared against this on
    counterfactual histories; any production cutoff on an entry the strict
    table does not confirm is a declared-heuristic event, not a silent reuse.
    """

    def __init__(self) -> None:
        self.map: dict[tuple, tuple] = {}

    @staticmethod
    def identity(
        key: int,
        halfmove: int,
        remaining_horizon: int,
        model_version: int,
        utility_version: int,
        unknown_prefix: bool,
        reversible_keys: tuple[int, ...],
        rep_count: int,
    ) -> tuple:
        fp = hash(reversible_keys) & 0xFFFFFFFFFFFFFFFF
        return (
            key,
            halfmove,
            remaining_horizon,
            model_version,
            utility_version,
            unknown_prefix,
            rep_count,
            fp,
        )

    def store(
        self, ident: tuple, move16: int, score: int, raw_eval: int, depth: int, bound: int
    ) -> None:
        self.map[ident] = (move16, score, raw_eval, depth, bound)

    def probe(self, ident: tuple) -> tuple | None:
        return self.map.get(ident)

    def clear(self) -> None:
        self.map.clear()

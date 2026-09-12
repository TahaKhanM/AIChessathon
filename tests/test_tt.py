"""Transposition-table tests: packing, three-identity rules, mate
normalization, sweep capacities, strict-audit comparison. Spec §3.3."""

from __future__ import annotations

import random

import pytest

from engine.tt import (
    BOUND_EXACT,
    BOUND_LOWER,
    BOUND_UPPER,
    CLUSTER,
    INF,
    MATE,
    MATE_IN_MAX,
    Probe,
    StrictTable,
    SWEEP_MIB,
    TranspositionTable,
    ValueContext,
    score_from_tt,
    score_to_tt,
)


def ctx(
    halfmove=0,
    horizon=600,
    model=0,
    util=0,
    rep=1,
    unknown=False,
) -> ValueContext:
    return ValueContext(halfmove, horizon, model, util, rep, unknown)


def test_default_size_is_128mib() -> None:
    tt = TranspositionTable()
    assert tt.bytes == (128 << 20)
    assert tt.n_clusters == (128 << 20) // (CLUSTER * 16)


def test_sweep_capacities() -> None:
    """Gate-4 sizes: 64/128/256/512 MiB all construct at the requested budget."""
    for mib in SWEEP_MIB:
        tt = TranspositionTable(mib=mib)
        assert tt.bytes == mib << 20
        assert tt.n_clusters & (tt.n_clusters - 1) == 0  # power of two


def test_store_probe_roundtrip() -> None:
    tt = TranspositionTable(mib=1)
    c = ctx()
    tt.store(0xDEADBEEFCAFEF00D, 0x1234, 137, 42, 8, BOUND_EXACT, c, ply=0)
    pr = tt.probe(0xDEADBEEFCAFEF00D, 0, c)
    assert pr.hit
    assert pr.move16 == 0x1234
    assert pr.score == 137
    assert pr.raw_eval == 42
    assert pr.depth == 8
    assert pr.bound == BOUND_EXACT
    assert pr.cutoff_ok


def test_mate_ply_normalization_every_depth() -> None:
    """Gate 3: mate scores survive store/load at every ply.

    A node-local "mate in k" at ply p is stored ply-independently and
    denormalized back relative to the probing node's ply.
    """
    tt = TranspositionTable(mib=1)
    c = ctx()
    key = 0x0123456789ABCDEF  # gitleaks:allow -- deterministic position-key fixture
    for ply in range(0, 200):
        for k in range(1, 64):
            local_win = MATE - ply - k  # win: mate k plies from this node
            tt.clear()
            tt.store(key, 0x111, local_win, 0, 10, BOUND_EXACT, c, ply)
            pr = tt.probe(key, ply, c)
            assert pr.hit and pr.score == local_win
            # Denormalized to a different ply: mate is still k plies out.
            for q in (0, ply, ply + 3, 150):
                pr2 = tt.probe(key, q, c)
                assert pr2.score == MATE - k - q
            local_loss = -(MATE - ply - k)  # being mated k plies out
            tt.clear()
            tt.store(key, 0x111, local_loss, 0, 10, BOUND_EXACT, c, ply)
            pr = tt.probe(key, ply, c)
            assert pr.hit and pr.score == local_loss
            for q in (0, ply, ply + 3, 150):
                pr2 = tt.probe(key, q, c)
                assert pr2.score == -MATE + k + q


def test_score_codec_unit() -> None:
    for ply in range(0, 64):
        for s in (-MATE + ply, -500, -1, 0, 1, 500, MATE - ply - 1):
            stored = score_to_tt(s, ply)
            assert score_from_tt(stored, ply) == s


def test_value_context_gates_score_not_hint() -> None:
    """A stored score is not reusable under a mismatched value context, but
    the legal move hint still is (spec 3.3 declared heuristic)."""
    tt = TranspositionTable(mib=1)
    key = 0xAAAA5555FFFF0000
    tt.store(key, 0x2222, 300, 50, 12, BOUND_LOWER, ctx(halfmove=10), ply=0)
    # high current halfmove -> score gated off, hint survives
    pr = tt.probe(key, 0, ctx(halfmove=95))
    assert pr.hit and not pr.cutoff_ok
    assert pr.move16 == 0x2222
    # entry stored at high halfmove is not reusable low either
    tt2 = TranspositionTable(mib=1)
    tt2.store(key, 0x2222, 300, 50, 12, BOUND_LOWER, ctx(halfmove=95), ply=0)
    pr = tt2.probe(key, 0, ctx(halfmove=10))
    assert pr.hit and not pr.cutoff_ok and pr.move16 == 0x2222


def test_horizon_gating_near_cap() -> None:
    tt = TranspositionTable(mib=1)
    key = 0x1111222233334444
    # stored far from the cap: reusable anywhere
    tt.store(key, 1, 200, 0, 10, BOUND_LOWER, ctx(horizon=600), ply=0)
    assert tt.probe(key, 0, ctx(horizon=590)).cutoff_ok
    # stored near the cap: only same-band reuse cuts off
    tt2 = TranspositionTable(mib=1)
    tt2.store(key, 1, 200, 0, 10, BOUND_LOWER, ctx(horizon=30), ply=0)
    assert not tt2.probe(key, 0, ctx(horizon=600)).cutoff_ok
    assert tt2.probe(key, 0, ctx(horizon=31)).cutoff_ok
    assert not tt2.probe(key, 0, ctx(horizon=10)).cutoff_ok


def test_repetition_context_gating() -> None:
    tt = TranspositionTable(mib=1)
    key = 0x99990000EEEE1111
    tt.store(key, 7, 150, 0, 9, BOUND_EXACT, ctx(rep=1), ply=0)
    assert tt.probe(key, 0, ctx(rep=1)).cutoff_ok
    assert not tt.probe(key, 0, ctx(rep=2)).cutoff_ok
    assert not tt.probe(key, 0, ctx(rep=1, unknown=True)).cutoff_ok


def test_model_utility_identity() -> None:
    tt = TranspositionTable(mib=1)
    key = 0x77778888CCCCDDDD
    tt.store(key, 5, 120, 33, 7, BOUND_EXACT, ctx(model=3, util=1), ply=0)
    assert tt.probe(key, 0, ctx(model=3, util=1)).cutoff_ok
    assert tt.probe(key, 0, ctx(model=3, util=1)).eval_ok
    wrong_model = tt.probe(key, 0, ctx(model=4, util=1))
    assert not wrong_model.cutoff_ok and not wrong_model.eval_ok
    # move hint survives a model change
    assert wrong_model.move16 == 5


def test_upper_bound_never_cutoffs_below_alpha_mismatch() -> None:
    tt = TranspositionTable(mib=1)
    key = 0x12340000ABCD0000
    c = ctx()
    tt.store(key, 9, -40, -30, 6, BOUND_UPPER, c, ply=0)
    pr = tt.probe(key, 0, c)
    assert pr.hit and pr.bound == BOUND_UPPER and pr.score == -40


def test_replacement_prefers_deeper_same_key() -> None:
    tt = TranspositionTable(mib=1)
    key = 0xF0F0F0F0F0F0F0F0
    c = ctx()
    tt.store(key, 1, 10, 0, 3, BOUND_UPPER, c, ply=0)
    tt.store(key, 2, 40, 0, 9, BOUND_EXACT, c, ply=0)
    pr = tt.probe(key, 0, c)
    assert pr.depth == 9 and pr.move16 == 2


def test_same_key_shallow_does_not_evict_deep() -> None:
    tt = TranspositionTable(mib=1)
    key = 0x13572468ACE02468  # gitleaks:allow -- deterministic position-key fixture
    c = ctx()
    tt.store(key, 1, 200, 0, 12, BOUND_EXACT, c, ply=0)
    tt.store(key, 2, -50, 0, 4, BOUND_UPPER, c, ply=0)  # shallow same-age
    pr = tt.probe(key, 0, c)
    assert pr.depth == 12 and pr.score == 200


# ---------------------------------------------------------------------------
# Same-key replacement policy (R6/R7 audit): worth = depth (+2 for EXACT)
# minus 4 plies per generation of age — the same scale the cross-key cluster
# scan uses. The incumbent is replaced only on a strict win, or on a tie when
# the new entry is not shallower. Every decision edge is probed below.
# ---------------------------------------------------------------------------

SAME_KEY = 0xA5A5C3C30F0F5A5A


def test_same_key_exact_survives_shallower_bound() -> None:
    """The defect case: a bound 1-3 plies shallower must not evict a
    deeper EXACT (pre-repair it did — only d < odepth-3 was refused)."""
    for gap in (1, 2, 3, 5):
        for bnd in (BOUND_UPPER, BOUND_LOWER):
            tt = TranspositionTable(mib=1)
            c = ctx()
            tt.store(SAME_KEY, 1, 200, 0, 12, BOUND_EXACT, c, ply=0)
            tt.store(SAME_KEY, 2, -50, 0, 12 - gap, bnd, c, ply=0)
            pr = tt.probe(SAME_KEY, 0, c)
            assert (pr.depth, pr.bound, pr.score, pr.move16) == (
                12,
                BOUND_EXACT,
                200,
                1,
            ), (gap, bnd)


def test_same_key_exact_survives_shallower_exact() -> None:
    """A shallower EXACT must not evict a deeper one either."""
    tt = TranspositionTable(mib=1)
    c = ctx()
    tt.store(SAME_KEY, 1, 200, 0, 12, BOUND_EXACT, c, ply=0)
    tt.store(SAME_KEY, 2, -50, 0, 10, BOUND_EXACT, c, ply=0)
    pr = tt.probe(SAME_KEY, 0, c)
    assert (pr.depth, pr.score, pr.move16) == (12, 200, 1)


def test_same_key_deeper_entries_replace() -> None:
    """Strictly deeper new entries win regardless of bound class."""
    tt = TranspositionTable(mib=1)
    c = ctx()
    tt.store(SAME_KEY, 1, 50, 0, 8, BOUND_EXACT, c, ply=0)
    tt.store(SAME_KEY, 2, 60, 0, 10, BOUND_EXACT, c, ply=0)
    pr = tt.probe(SAME_KEY, 0, c)
    assert (pr.depth, pr.score, pr.move16) == (10, 60, 2)
    tt.store(SAME_KEY, 3, 70, 0, 14, BOUND_LOWER, c, ply=0)
    pr = tt.probe(SAME_KEY, 0, c)
    assert (pr.depth, pr.bound, pr.score) == (14, BOUND_LOWER, 70)


def test_same_key_exact_supersedes_bound_at_same_depth() -> None:
    """Exact's +2 bonus makes it strictly outrank a bound at equal depth."""
    tt = TranspositionTable(mib=1)
    c = ctx()
    tt.store(SAME_KEY, 1, 50, 0, 10, BOUND_LOWER, c, ply=0)
    tt.store(SAME_KEY, 2, 60, 0, 10, BOUND_EXACT, c, ply=0)
    pr = tt.probe(SAME_KEY, 0, c)
    assert (pr.bound, pr.score, pr.move16) == (BOUND_EXACT, 60, 2)


def test_same_key_bound_vs_exact_boundary() -> None:
    """A bound one ply deeper than an EXACT still loses (worth 13 < 14);
    two plies deeper it ties and wins on depth (14 vs 12)."""
    tt = TranspositionTable(mib=1)
    c = ctx()
    tt.store(SAME_KEY, 1, 200, 0, 12, BOUND_EXACT, c, ply=0)
    tt.store(SAME_KEY, 2, -50, 0, 13, BOUND_LOWER, c, ply=0)
    pr = tt.probe(SAME_KEY, 0, c)
    assert (pr.depth, pr.bound) == (12, BOUND_EXACT)
    tt.store(SAME_KEY, 3, -60, 0, 14, BOUND_LOWER, c, ply=0)
    pr = tt.probe(SAME_KEY, 0, c)
    assert (pr.depth, pr.bound, pr.score) == (14, BOUND_LOWER, -60)


def test_same_key_bound_survives_shallower_bound() -> None:
    tt = TranspositionTable(mib=1)
    c = ctx()
    tt.store(SAME_KEY, 1, 50, 0, 12, BOUND_LOWER, c, ply=0)
    tt.store(SAME_KEY, 2, 60, 0, 10, BOUND_LOWER, c, ply=0)
    pr = tt.probe(SAME_KEY, 0, c)
    assert (pr.depth, pr.score, pr.move16) == (12, 50, 1)


def test_same_key_exact_replaces_bound_one_ply_shallower() -> None:
    """A fresh EXACT outranks a bound within a ply of its depth; two plies
    shallower it does not (worth 12 < 12)."""
    tt = TranspositionTable(mib=1)
    c = ctx()
    tt.store(SAME_KEY, 1, 50, 0, 12, BOUND_LOWER, c, ply=0)
    tt.store(SAME_KEY, 2, 60, 0, 11, BOUND_EXACT, c, ply=0)
    pr = tt.probe(SAME_KEY, 0, c)
    assert (pr.bound, pr.depth) == (BOUND_EXACT, 11)
    tt2 = TranspositionTable(mib=1)
    tt2.store(SAME_KEY, 1, 50, 0, 12, BOUND_LOWER, c, ply=0)
    tt2.store(SAME_KEY, 2, 60, 0, 10, BOUND_EXACT, c, ply=0)
    pr2 = tt2.probe(SAME_KEY, 0, c)
    assert (pr2.bound, pr2.depth, pr2.score) == (BOUND_LOWER, 12, 50)


def test_same_key_age_erodes_incumbent() -> None:
    """Each generation of age costs the incumbent 4 plies of worth."""
    tt = TranspositionTable(mib=1)
    c = ctx()
    tt.store(SAME_KEY, 1, 50, 0, 12, BOUND_LOWER, c, ply=0)
    tt.new_search()  # incumbent is now one generation old: worth 12-4 = 8
    tt.store(SAME_KEY, 2, 60, 0, 7, BOUND_LOWER, c, ply=0)  # 7 < 8: keeps
    pr = tt.probe(SAME_KEY, 0, c)
    assert (pr.depth, pr.score, pr.move16) == (12, 50, 1)
    tt.store(SAME_KEY, 3, 70, 0, 9, BOUND_LOWER, c, ply=0)  # 9 > 8: wins
    pr = tt.probe(SAME_KEY, 0, c)
    assert (pr.depth, pr.score, pr.move16) == (9, 70, 3)


def test_same_key_stale_exact_loses_to_fresh_exact() -> None:
    """Three generations of age drain an EXACT to worth 2 — a fresh
    depth-1 EXACT (worth 3) replaces it."""
    tt = TranspositionTable(mib=1)
    c = ctx()
    tt.store(SAME_KEY, 1, 50, 0, 12, BOUND_EXACT, c, ply=0)
    for _ in range(3):
        tt.new_search()
    tt.store(SAME_KEY, 2, 60, 0, 1, BOUND_EXACT, c, ply=0)
    pr = tt.probe(SAME_KEY, 0, c)
    assert (pr.depth, pr.score, pr.move16) == (1, 60, 2)


def test_same_key_hint_inherited_when_moveless() -> None:
    """A score-only store inherits the incumbent's move hint."""
    tt = TranspositionTable(mib=1)
    c = ctx()
    tt.store(SAME_KEY, 0x123, 50, 0, 5, BOUND_LOWER, c, ply=0)
    tt.store(SAME_KEY, 0, 60, 0, 9, BOUND_EXACT, c, ply=0)
    pr = tt.probe(SAME_KEY, 0, c)
    assert (pr.move16, pr.bound) == (0x123, BOUND_EXACT)


def test_age_replacement() -> None:
    tt = TranspositionTable(mib=1)
    c = ctx()
    keys = [0x1000000000000000 + i for i in range(40)]
    for k in keys:
        tt.store(k, 1, 50, 0, 6, BOUND_EXACT, c, ply=0)
    tt.new_search()
    for k in keys:
        tt.store(k, 2, 60, 0, 6, BOUND_EXACT, c, ply=0)
    assert all(tt.probe(k, 0, c).score == 60 for k in keys)


def test_raw_eval_is_immutable_not_corrected() -> None:
    """Storing a corrected score into the raw slot would corrupt the cache;
    the table keeps them in distinct fields so the raw value stays raw."""
    tt = TranspositionTable(mib=1)
    c = ctx()
    key = 0x55AA55AA00FF00FF
    tt.store(key, 0, 250, 180, 8, BOUND_EXACT, c, ply=0)  # score!=eval: corrected 250 vs raw 180
    pr = tt.probe(key, 0, c)
    assert pr.raw_eval == 180 and pr.score == 250


def test_strict_table_identity() -> None:
    """Strict audit: complete identity required; counterfactuals miss."""
    st = StrictTable()
    hist_a = (111, 222, 333)
    hist_b = (111, 999, 333)
    ident_a = StrictTable.identity(42, 10, 600, 1, 0, False, hist_a, 1)
    st.store(ident_a, 5, 100, 40, 8, BOUND_EXACT)
    assert st.probe(ident_a) == (5, 100, 40, 8, BOUND_EXACT)
    # any identity component change -> miss
    assert st.probe(StrictTable.identity(42, 10, 600, 1, 0, False, hist_b, 1)) is None
    assert st.probe(StrictTable.identity(42, 11, 600, 1, 0, False, hist_a, 1)) is None
    assert st.probe(StrictTable.identity(42, 10, 599, 1, 0, False, hist_a, 1)) is None
    assert st.probe(StrictTable.identity(42, 10, 600, 2, 0, False, hist_a, 1)) is None
    assert st.probe(StrictTable.identity(42, 10, 600, 1, 0, True, hist_a, 1)) is None


def test_production_vs_strict_audit() -> None:
    """Compare production heuristic reuse against strict-audit behaviour.

    Production is *stricter* than the oracle on halfmove/horizon proximity
    (score reuse refused near rule50/cap) and *coarser* on repetition count
    (band equality, not exact). A cutoff is a declared-heuristic event;
    a strict-confirmed probe must always at least hit geometrically, and a
    production cutoff may never come from a different-position tag.
    """
    tt = TranspositionTable(mib=1)
    st = StrictTable()
    rng = random.Random(7)
    keys = [rng.getrandbits(64) for _ in range(200)]
    hist = tuple(keys[:20])
    stored = {}
    for i, k in enumerate(keys):
        half = i % 97
        horizon = 600 - (i % 50)
        rep = 1 + (i % 4)
        unk = bool(i & 1)
        c = ctx(halfmove=half, horizon=horizon, rep=rep, unknown=unk)
        tt.store(k, i & 0x7FFF, i % 500 - 250, i % 300 - 150, 5, BOUND_EXACT, c, 0)
        ident = StrictTable.identity(k, half, horizon, 0, 0, unk, hist, rep)
        st.store(ident, i & 0x7FFF, i % 500 - 250, i % 300 - 150, 5, BOUND_EXACT)
        stored[k] = (half, horizon, rep, unk)
    prod_cutoff_confirmed = 0
    prod_cutoff_heuristic = 0
    prod_hint_only = 0
    strict_confirmed = 0
    strict_evicted = 0
    for i, k in enumerate(keys):
        half, horizon, rep, unk = stored[k]
        for h2, hz2, r2, u2 in (
            (half, horizon, rep, unk),  # identical context
            ((half + 31) % 97, horizon, rep, unk),  # counterfactual halfmove
            (half, horizon, (rep % 4) + 1, unk),  # counterfactual rep
        ):
            c = ctx(halfmove=h2, horizon=hz2, rep=r2, unknown=u2)
            pr = tt.probe(k, 0, c)
            ident = StrictTable.identity(k, h2, hz2, 0, 0, u2, hist, r2)
            strict_hit = st.probe(ident) is not None
            if strict_hit:
                strict_confirmed += 1
                if not pr.hit:
                    strict_evicted += 1  # same-cluster eviction: legal miss
            if pr.hit and not pr.cutoff_ok:
                prod_hint_only += 1
            if pr.cutoff_ok:
                if strict_hit:
                    prod_cutoff_confirmed += 1
                else:
                    prod_cutoff_heuristic += 1
                assert pr.score == i % 500 - 250  # same-position score
    assert strict_confirmed > 0
    assert strict_evicted <= strict_confirmed // 20  # evictions are rare
    assert prod_cutoff_confirmed > 0
    assert prod_hint_only > 0  # the gates actually fire
    print(
        f"\nproduction-vs-strict audit: strict_confirmed={strict_confirmed} "
        f"evicted={strict_evicted} cutoffs_confirmed={prod_cutoff_confirmed} "
        f"cutoffs_heuristic={prod_cutoff_heuristic} hint_only={prod_hint_only}"
    )


def test_no_bogus_entry_after_clear() -> None:
    tt = TranspositionTable(mib=1)
    for i in range(300):
        tt.store(i * 0x9E3779B97F4A7C15, i, i - 150, i, 4, BOUND_EXACT, ctx(), 0)
    tt.clear()
    p = Probe()
    for i in range(300):
        tt.probe_into(p, i * 0x9E3779B97F4A7C15, 0, ctx())
        assert not p.hit


def test_hashfull_grows() -> None:
    tt = TranspositionTable(mib=1)
    for i in range(2000):
        tt.store(
            i * 0x1234567 + 0xABCDEF0000000000,
            i & 0x7FFF,
            i % 400 - 200,
            i % 200 - 100,
            6,
            BOUND_EXACT,
            ctx(),
            0,
        )
    assert tt.hashfull() > 0


def test_entry_fits_cluster() -> None:
    """Packing oracle: two uint64 words per entry, 4 entries = 64B cluster."""
    tt = TranspositionTable(mib=1)
    assert tt.clusters.dtype.itemsize == 8
    assert tt.clusters.shape[1] == CLUSTER
    assert tt.clusters.shape[2] == 2


@pytest.mark.parametrize("ply", [0, 1, 7, 33, 100])
def test_mate_sentinel_bounds(ply: int) -> None:
    # MATE_IN_MAX boundary scores normalize like mates.
    for score in (MATE_IN_MAX, -MATE_IN_MAX):
        assert score_from_tt(score_to_tt(score, ply), ply) == score
    for score in (MATE_IN_MAX - 1, -(MATE_IN_MAX - 1)):
        assert score_from_tt(score_to_tt(score, ply), ply) == score
    assert -INF < -MATE_IN_MAX and MATE_IN_MAX < MATE < INF

"""Compiled search kernels (nopython): quiescence, PVS alpha-beta, path
and repetition bookkeeping, draw adjudication, move ordering, the
transposition table interface and the monotonic hard-clock deadline.

Every function is a bit-exact transliteration of the corresponding
``engine.search.Searcher`` method (same pruning order, same tie-breaks,
same bound logic, same TT store cadence) — the pure-Python search remains
the scalar oracle the parity gates compare against.

Numeric notes:
  * All score/depth bookkeeping uses int64; move/score buffers are int32.
  * ``np.empty`` scratch inside njit is native allocation (no Python
    objects); per-ply buffers live in the arenas.
  * The hard clock is ``clock_ns`` — a direct clock_gettime(CLOCK_MONOTONIC)
    call via engine.kernels.nclock (monotonic wall time, no objmode).
"""

from __future__ import annotations

import numpy as np
from numba import njit, types

from engine.kernels.layout import (
    A32,
    A8,
    AI,
    AU,
    I_ABSPLY,
    I_AGE,
    I_BESTNODES,
    I_CLKREADS,
    I_DL_HARD,
    I_DL_MASK,
    I_DL_NLIM,
    I_DL_NODES,
    I_DL_START,
    I_DL_STOP,
    I_EVALKIND,
    I_HALF,
    I_HISTN,
    I_MODELVER,
    I_NODES,
    I_NULLMIN,
    I_OVERRUN,
    I_PHASE,
    I_QNODES,
    I_ROOTBEST,
    I_ROOTDEPTH,
    I_ROOTHINT,
    I_ROOTSCORE,
    I_SELDEPTH,
    I_SIDE,
    I_TTHITS,
    I_UNKNOWN,
    I_UTILVER,
    J_KEY,
    MCAP,
    N_ST,
    PATH,
    X_BB,
    X_HISTIRR,
    X_HISTKEY,
    X_KILLERS,
    X_LMR,
    X_MB,
    X_MOVES,
    X_NODEREP,
    X_PARAMS,
    X_PATHKEY,
    X_PIF,
    X_PNF,
    X_PVBUF,
    X_QUIETS,
    X_ROOTSC,
    X_SCORES,
    X_SS_EVAL,
    X_SS_EXCL,
    X_SS_MOVE,
    X_SS_NULL,
    X_SS_PIECE,
    X_ST,
    X_STATS,
    X_U64,
    # parameter indices (compile-time constants — see layout.PARAM_ORDER)
    P_check_ext,
    P_check_ext_depth,
    P_check_ply_cap_mult,
    P_eval_clamp,
    P_fut_base,
    P_fut_depth,
    P_fut_per_depth,
    P_iir_depth,
    P_lmp_base,
    P_lmp_improving_div,
    P_lmp_quad,
    P_lmr_cut_node,
    P_lmr_gate_nonpv,
    P_lmr_gate_pv,
    P_lmr_hist_div,
    P_lmr_killer,
    P_lmr_max_sub,
    P_lmr_min_depth,
    P_lmr_min_r,
    P_lmr_not_improving,
    P_lmr_pv,
    P_nmp_base,
    P_nmp_depth_div,
    P_nmp_eval_div,
    P_nmp_eval_margin,
    P_nmp_eval_max,
    P_nmp_min_depth,
    P_nmp_no_consec,
    P_nmp_tt_guard,
    P_nmp_verify_den,
    P_nmp_verify_depth,
    P_nmp_verify_min,
    P_nmp_verify_num,
    P_ord_cap_mvv_mult,
    P_ord_goodcap_see_div,
    P_ord_promo_queen,
    P_ord_promo_under,
    P_pc_depth,
    P_pc_margin,
    P_pc_see_cap,
    P_pc_tt_slack,
    P_q_fut_see_gate,
    P_q_futility,
    P_q_futility_on,
    P_q_see,
    P_q_see_on,
    P_razor_depth,
    P_razor_margin,
    P_rfp_depth,
    P_rfp_improving,
    P_rfp_margin,
    P_see_noisy_depth,
    P_see_noisy_mult,
    P_see_quiet_depth,
    P_see_quiet_min_ld,
    P_see_quiet_mult,
    P_sing_depth,
    P_sing_ext,
    P_sing_half_div,
    P_sing_margin,
    P_sing_ply_cap_mult,
    P_sing_tt_slack,
)
from engine.kernels.bb import (
    BISHOP,
    FLAG_NULL,
    KNIGHT,
    QUEEN,
    ROOK,
    k_gen_legal,
    k_gives_check_fast,
    k_in_check,
    k_insufficient,
    k_is_irreversible,
    k_king_has_legal_move,
    k_make,
    k_make_null,
    k_pval,
    k_see_ge,
    k_unmake,
    k_unmake_null,
)
from engine.kernels.tthist import (
    k_capture_score,
    k_correction_cp,
    k_counter_move,
    k_quiet_score,
    k_tt_probe,
    k_tt_store,
    k_update_capture,
    k_update_correction,
    k_update_quiets,
)
from engine.kernels.ev import n_ev_pop, n_ev_push, n_ev_set_root, n_static_eval
from engine.search import (
    DRAW,
    INF,
    MATE,
    MATE_IN_MAX,
    PLY_CAP,
    SEARCH_PATH,
    S_BAD_CAPTURE,
    S_GOOD_CAPTURE,
    S_KILLER,
    S_TT,
)
from engine.tt import (
    BOUND_EXACT,
    BOUND_LOWER,
    BOUND_UPPER,
)
from engine.search import REASONS as _REASONS

# compile-time reason indices (nopython cannot index dicts).  Real module
# constants — NOT globals() injection — so linters see them and a REASONS
# reorder fails the assert below at import instead of silently mis-indexing
# (numba bakes globals into cached code; an invisible constant is a stale-
# cache hazard).
(
    R_tt_cut,
    R_mate_dist,
    R_rfp,
    R_razor,
    R_razor_verify,
    R_nmp,
    R_nmp_verify_fail,
    R_probcut,
    R_lmp,
    R_futility,
    R_see_quiet,
    R_see_noisy,
    R_lmr,
    R_lmr_research,
    R_pvs_research,
    R_iir,
    R_sing_ext,
    R_check_ext,
    R_q_futility,
    R_q_see,
    R_rep,
    R_fifty,
    R_cap,
    R_insufficient,
    R_mate,
    R_stalemate,
    R_corr_update,
    R_seventyfive,
    R_fivefold,
) = range(len(_REASONS))
for _i, _r in enumerate(_REASONS):
    assert globals()["R_" + _r] == _i, f"R_{_r} constant drifted"
del _i, _r


# ---------------------------------------------------------------------------
# hard clock — direct clock_gettime(CLOCK_MONOTONIC) via engine.kernels.nclock.
# The kernel stores a *budget* (ns duration) + the clock's own start stamp so
# the clock epoch never has to match time.monotonic_ns (different on macOS).
# ---------------------------------------------------------------------------

from engine.kernels.nclock import clock_ns  # noqa: E402


@njit(cache=True)
def n_poll(ctx):
    """Deadline.poll mirror — sticky stop, node limit, masked clock reads."""
    L = ctx[AI]
    if L[X_ST + I_DL_STOP]:
        return True
    n = L[X_ST + I_DL_NODES]
    L[X_ST + I_DL_NODES] = n + 1
    if L[X_ST + I_DL_NLIM] and n >= L[X_ST + I_DL_NLIM]:
        L[X_ST + I_DL_STOP] = 1
        return True
    if n & L[X_ST + I_DL_MASK]:
        return False
    L[X_ST + I_CLKREADS] += 1
    if L[X_ST + I_DL_HARD]:
        t = clock_ns()
        if t - L[X_ST + I_DL_START] >= L[X_ST + I_DL_HARD]:
            L[X_ST + I_DL_STOP] = 1
            L[X_ST + I_OVERRUN] = t - L[X_ST + I_DL_START] - L[X_ST + I_DL_HARD]
    return L[X_ST + I_DL_STOP] != 0


# ---------------------------------------------------------------------------
# stats / path / repetition — mirror of _note, _push_path, _rep_scan
# ---------------------------------------------------------------------------


@njit(cache=True)
def n_note(ctx, reason, ply, move):
    ctx[AI][X_STATS + reason] += 1


@njit(cache=True)
def n_push_path(ctx, ply, null_move, irr):
    U = ctx[AU]
    N = ctx[A32]
    U[X_PATHKEY + ply] = U[X_U64 + J_KEY]
    if null_move:
        N[X_PNF + ply] = ply
        N[X_PIF + ply] = N[X_PIF + ply - 1]
    else:
        N[X_PNF + ply] = N[X_PNF + ply - 1]
        N[X_PIF + ply] = ply if irr else N[X_PIF + ply - 1]


@njit(cache=True)
def n_rep_scan(ctx, ply):
    """(in-path hit, game-history hits) — mirror of _rep_scan."""
    U = ctx[AU]
    L = ctx[AI]
    N = ctx[A32]
    key = U[X_U64 + J_KEY]
    null_floor = np.int64(N[X_PNF + ply])
    irr_floor = np.int64(N[X_PIF + ply])
    stop = null_floor + 1 if null_floor > irr_floor else irr_floor
    i = ply - 2
    while i >= stop and i >= 0:
        if U[X_PATHKEY + i] == key:
            return True, np.int64(0)
        i -= 2
    hits = np.int64(0)
    if null_floor == 0 and irr_floor == 0:
        n = L[X_ST + I_HISTN]
        i = n - 2
        while i >= 0:
            if ctx[A8][X_HISTIRR + i]:
                break
            if U[X_HISTKEY + i] == key:
                hits += 1
            i -= 1
    return False, hits


@njit(cache=True)
def n_draw_score(ctx, ply, checked):
    """(has_result, value) — mirror of _draw_score. value 0 = DRAW."""
    L = ctx[AI]
    N = ctx[A32]
    path_hit, game_hits = n_rep_scan(ctx, ply)
    N[X_NODEREP + ply] = np.int32(game_hits + 1 + (1 if path_hit else 0))
    reason = -1
    if path_hit or game_hits >= 1:
        reason = R_rep
    elif L[X_ST + I_HALF] >= 100:
        reason = R_fifty
    elif L[X_ST + I_ABSPLY] >= PLY_CAP:
        reason = R_cap
    elif k_insufficient(ctx):
        reason = R_insufficient
    if reason < 0:
        return False, np.int64(0)
    if checked:
        buf = N[X_MOVES + ply * MCAP : X_MOVES + ply * MCAP + MCAP]
        n = k_gen_legal(ctx, buf)
        if n == 0:
            n_note(ctx, R_mate, ply, 0)
            return True, -MATE + ply
    n_note(ctx, reason, ply, 0)
    return True, np.int64(DRAW)


# ---------------------------------------------------------------------------
# value context + corrected eval
# ---------------------------------------------------------------------------


@njit(cache=True)
def n_vctx(ctx, ply):
    """(half, horizon, model, util, rep, unknown) — mirror of _vctx_at."""
    L = ctx[AI]
    N = ctx[A32]
    return (
        L[X_ST + I_HALF],
        PLY_CAP - L[X_ST + I_ABSPLY],
        L[X_ST + I_MODELVER],
        L[X_ST + I_UTILVER],
        np.int64(N[X_NODEREP + ply]),
        L[X_ST + I_UNKNOWN],
    )


@njit(cache=True)
def n_corrected(ctx, raw):
    clamp = np.int64(ctx[A32][X_PARAMS + P_eval_clamp])
    v = raw + k_correction_cp(ctx, ctx[AI][X_ST + I_SIDE])
    if v > clamp:
        return np.int64(clamp)
    if v < -clamp:
        return np.int64(-clamp)
    return v


# ---------------------------------------------------------------------------
# ordering — mirror of _score_moves / _pick / _gen_noisy / _move16_at
# ---------------------------------------------------------------------------


@njit(cache=True)
def n_score_moves(ctx, n, ply, tt_move16, in_qsearch):
    L = ctx[AI]
    N = ctx[A32]
    B = ctx[A8]
    moves = N[X_MOVES + ply * MCAP : X_MOVES + ply * MCAP + MCAP]
    scores = N[X_SCORES + ply * MCAP : X_SCORES + ply * MCAP + MCAP]
    hint = L[X_ST + I_ROOTHINT]
    p1 = np.int64(N[X_SS_PIECE + ply + 3])
    t1 = (np.int64(N[X_SS_MOVE + ply + 3]) >> 6) & 63
    p2 = np.int64(N[X_SS_PIECE + ply + 2])
    t2 = (np.int64(N[X_SS_MOVE + ply + 2]) >> 6) & 63
    cm = k_counter_move(ctx, p1, t1)
    for i in range(n):
        m = np.int64(moves[i])
        frm = m & 63
        to = (m >> 6) & 63
        promo = (m >> 12) & 7
        captured = (m >> 18) & 15
        if (m & 0x7FFF) == tt_move16 or (ply == 0 and hint and (m & 0x7FFF) == (hint & 0x7FFF)):
            scores[i] = np.int32(S_TT)
            continue
        piece = np.int64(B[X_MB + frm])
        if captured != 15 or promo:
            vt = np.int64(0 if captured == 15 else captured % 6 + 1)
            s = np.int64(N[X_PARAMS + P_ord_cap_mvv_mult]) * k_pval(
                ctx, captured if captured != 15 else 0
            ) + k_capture_score(ctx, piece, to, vt)
            if promo:
                s += (
                    np.int64(N[X_PARAMS + P_ord_promo_queen])
                    if promo == QUEEN
                    else np.int64(N[X_PARAMS + P_ord_promo_under])
                )
            if (
                in_qsearch
                or promo
                or k_see_ge(
                    ctx,
                    m,
                    -k_pval(ctx, piece) // np.int64(N[X_PARAMS + P_ord_goodcap_see_div]),
                )
            ):
                scores[i] = np.int32(S_GOOD_CAPTURE + s)
            else:
                scores[i] = np.int32(S_BAD_CAPTURE + s)
        else:
            k0 = np.int64(N[X_KILLERS + ply * 2])
            k1 = np.int64(N[X_KILLERS + ply * 2 + 1])
            if m == k0:
                scores[i] = np.int32(S_KILLER + 2)
            elif m == k1:
                scores[i] = np.int32(S_KILLER + 1)
            elif m == cm:
                scores[i] = np.int32(S_KILLER)
            else:
                scores[i] = np.int32(k_quiet_score(ctx, frm, to, piece, p1, t1, p2, t2))


@njit(cache=True)
def n_pick(ctx, i, n, ply):
    N = ctx[A32]
    moves = N[X_MOVES + ply * MCAP : X_MOVES + ply * MCAP + MCAP]
    scores = N[X_SCORES + ply * MCAP : X_SCORES + ply * MCAP + MCAP]
    best = i
    for j in range(i + 1, n):
        if np.int64(scores[j]) > np.int64(scores[best]):
            best = j
    if best != i:
        tmp = moves[i]
        moves[i] = moves[best]
        moves[best] = tmp
        ts = scores[i]
        scores[i] = scores[best]
        scores[best] = ts
    return np.int64(moves[i])


@njit(cache=True)
def n_gen_noisy(ctx, ply):
    """Legal captures + promotions — mirror of _gen_noisy (scores as tmp)."""
    N = ctx[A32]
    buf = N[X_MOVES + ply * MCAP : X_MOVES + ply * MCAP + MCAP]
    tmp = N[X_SCORES + ply * MCAP : X_SCORES + ply * MCAP + MCAP]
    n = k_gen_legal(ctx, tmp)
    k = 0
    for i in range(n):
        m = np.int64(tmp[i])
        if (m >> 18) & 15 != 15 or (m >> 12) & 7:
            buf[k] = np.int32(m)
            k += 1
    return k


@njit(cache=True)
def n_move16_at(ctx, ply, move16):
    """Resolve a stored move15 to a full legal move (mirror _move16_at)."""
    N = ctx[A32]
    buf = N[X_SCORES + ply * MCAP : X_SCORES + ply * MCAP + MCAP]
    n = k_gen_legal(ctx, buf)
    for i in range(n):
        if (np.int64(buf[i]) & 0x7FFF) == move16:
            return np.int64(buf[i])
    return np.int64(0)


# ---------------------------------------------------------------------------
# transactional make/unmake with eval push + path bookkeeping
# ---------------------------------------------------------------------------


@njit(cache=True)
def n_make(ctx, ply, m):
    irr = k_is_irreversible(ctx, m)
    if ctx[AI][X_ST + I_EVALKIND] == 1:
        n_ev_push(ctx, m)
    k_make(ctx, m)
    n_push_path(ctx, ply + 1, 0, irr)


@njit(cache=True)
def n_unmake(ctx, ply, m):
    k_unmake(ctx)
    if ctx[AI][X_ST + I_EVALKIND] == 1:
        n_ev_pop(ctx)


@njit(cache=True)
def n_make_null(ctx, ply):
    if ctx[AI][X_ST + I_EVALKIND] == 1:
        n_ev_push(ctx, FLAG_NULL << 15)
    k_make_null(ctx)
    n_push_path(ctx, ply + 1, 1, 0)


@njit(cache=True)
def n_unmake_null(ctx, ply):
    k_unmake_null(ctx)
    if ctx[AI][X_ST + I_EVALKIND] == 1:
        n_ev_pop(ctx)


# ---------------------------------------------------------------------------
# single-dispatcher search — quiescence (depth <= 0) + PVS alpha-beta.
#
# The two are merged into ONE njit function deliberately: every edge in the
# call graph becomes self-recursion, which avoids a flaky aarch64/numba
# cross-dispatcher crash observed with separate n_qs/n_ab dispatchers
# (call_cfunc null-deref, numba#9857/#8738 family).
# ---------------------------------------------------------------------------


@njit(cache=True)
def U_key(ctx):
    return ctx[AU][X_U64 + J_KEY]


# ``n_search`` must compile to EXACTLY ONE specialization.  Its recursive
# call sites pass literal ``0``/``1``/``1 - cut_node`` for ``is_pv`` /
# ``cut_node``, which numba types as ``Literal[int]`` — that spawned six
# specializations whose recursion edges cross module boundaries.  A cached
# specialization stores such an edge as a ``.numba.unresolved$`` symbol that
# is NOT rebindable on cache load (numba issues #6061/#6713/#9129): the
# loaded machine code calls a NULL pointer and the process SIGSEGVs at the
# first ``n_search`` call (reproduced 3/3 on macOS arm64 and on EPYC x86_64,
# eval_kind 0 and 1).  Pinning the signature collapses every call site onto
# one specialization, so the recursive call is intra-module — the shape the
# numba cache round-trips correctly.
CTX_T = types.Tuple(
    (
        types.Array(types.uint64, 1, "C"),  # AU
        types.Array(types.int64, 1, "C"),  # AI
        types.Array(types.int32, 1, "C"),  # A32
        types.Array(types.int16, 1, "C"),  # A16
        types.Array(types.int8, 1, "C"),  # A8
        types.Array(types.uint8, 1, "C"),  # AU8
        types.Array(types.uint32, 1, "C"),  # AU32
        types.Array(types.int64, 3, "C"),  # CTT — TranspositionTable.clusters
    )
)
N_SEARCH_SIG = types.int64(CTX_T, *([types.int64] * 6))


@njit(N_SEARCH_SIG, cache=True)
def n_search(ctx, depth, alpha, beta, ply, is_pv, cut_node):
    L = ctx[AI]
    N = ctx[A32]
    B = ctx[A8]
    U = ctx[AU]
    # Every recursive call site below must pass ONLY int64-typed values —
    # a literal `0`/`1` in the call-site signature makes numba compile a
    # second specialization, and a cached specialization's cross-module
    # recursion edge never resolves on load (see the pinned-signature note).
    Z = np.int64(0)
    ONE = np.int64(1)
    if depth <= 0:
        # ==== quiescence body — mirror of Searcher._qs ====
        L[X_ST + I_NODES] += 1
        L[X_ST + I_QNODES] += 1
        if ply > L[X_ST + I_SELDEPTH]:
            L[X_ST + I_SELDEPTH] = ply
        L[X_ST + I_PHASE] = 2
        if n_poll(ctx):
            return np.int64(0)
        checked = k_in_check(ctx)
        has_draw, dv = n_draw_score(ctx, ply, checked)
        if has_draw:
            return dv
        if ply >= SEARCH_PATH - 8:
            return n_static_eval(ctx, ply)

        key = U[X_U64 + J_KEY]
        half, horizon, model, util, rep, unknown = n_vctx(ctx, ply)
        hit, tt_move16, tt_score, tt_eval, _td, tt_bound, tt_cutoff, eval_ok, _s = k_tt_probe(
            ctx, key, ply, half, horizon, model, util, rep, unknown
        )
        if not eval_ok:
            tt_eval = -INF
        if (
            hit
            and tt_cutoff
            and (
                tt_bound == BOUND_EXACT
                or (tt_bound == BOUND_LOWER and tt_score >= beta)
                or (tt_bound == BOUND_UPPER and tt_score <= alpha)
            )
        ):
            n_note(ctx, R_tt_cut, ply, 0)
            return tt_score

        moves = N[X_MOVES + ply * MCAP : X_MOVES + ply * MCAP + MCAP]
        best_move = np.int64(0)
        static = -INF
        if checked:
            best = -INF
            n = k_gen_legal(ctx, moves)
            if n == 0:
                n_note(ctx, R_mate, ply, 0)
                return -MATE + ply
        else:
            if not k_king_has_legal_move(ctx):
                n_all = k_gen_legal(ctx, moves)
                if n_all == 0:
                    n_note(ctx, R_stalemate, ply, 0)
                    return np.int64(DRAW)
            if hit and tt_eval > -MATE_IN_MAX:
                static = tt_eval
            else:
                static = n_static_eval(ctx, ply)
            best = n_corrected(ctx, static)
            if (
                hit
                and tt_cutoff
                and (
                    (tt_bound == BOUND_LOWER and tt_score > best)
                    or (tt_bound == BOUND_UPPER and tt_score < best)
                )
            ):
                best = tt_score
            if best >= beta:
                if not hit:
                    k_tt_store(
                        ctx,
                        key,
                        np.int64(0),
                        best,
                        static,
                        np.int64(0),
                        BOUND_LOWER,
                        half,
                        horizon,
                        model,
                        util,
                        rep,
                        unknown,
                        ply,
                    )
                return best
            if best > alpha:
                alpha = best
            n = n_gen_noisy(ctx, ply)

        n_score_moves(ctx, n, ply, tt_move16 if hit else np.int64(0), True)
        futility = static + np.int64(N[X_PARAMS + P_q_futility]) if static > -INF else -INF
        for i in range(n):
            m = n_pick(ctx, i, n, ply)
            promo = (m >> 12) & 7
            piece = (m >> 22) & 15
            captured = (m >> 18) & 15
            if not checked and best > -MATE_IN_MAX:
                if captured != 15 and not promo and N[X_PARAMS + P_q_futility_on]:
                    gain = k_pval(ctx, captured)
                    if futility + gain <= alpha and not k_see_ge(
                        ctx, m, np.int64(N[X_PARAMS + P_q_fut_see_gate])
                    ):
                        if best < futility + gain:
                            best = futility + gain
                        n_note(ctx, R_q_futility, ply, m)
                        continue
                if N[X_PARAMS + P_q_see_on] and not k_see_ge(
                    ctx, m, -np.int64(N[X_PARAMS + P_q_see])
                ):
                    n_note(ctx, R_q_see, ply, m)
                    continue
            N[X_SS_MOVE + ply + 4] = np.int32(m)
            N[X_SS_PIECE + ply + 4] = np.int32(piece)
            n_make(ctx, ply, m)
            score = -n_search(ctx, Z, -beta, -alpha, ply + 1, Z, Z)
            n_unmake(ctx, ply, m)
            if L[X_ST + I_DL_STOP]:
                return np.int64(0)
            if score > best:
                best = score
                if score > alpha:
                    best_move = m
                    if score >= beta:
                        break
                    alpha = score
        if checked and best == -INF:
            best = -MATE + ply
        bound = BOUND_LOWER if best >= beta else BOUND_UPPER
        half2, horizon2, model2, util2, rep2, unknown2 = n_vctx(ctx, ply)
        k_tt_store(
            ctx,
            key,
            best_move & 0x7FFF,
            best,
            static,
            np.int64(0),
            bound,
            half2,
            horizon2,
            model2,
            util2,
            rep2,
            unknown2,
            ply,
        )
        return best

    # ==== main alpha-beta body — mirror of Searcher._ab ====
    U = ctx[AU]
    L = ctx[AI]
    N = ctx[A32]
    B = ctx[A8]
    L[X_ST + I_NODES] += 1
    if ply > L[X_ST + I_SELDEPTH]:
        L[X_ST + I_SELDEPTH] = ply
    L[X_ST + I_PHASE] = 1
    root = ply == 0
    if n_poll(ctx):
        return np.int64(0)
    checked = k_in_check(ctx)
    if not root:
        has_draw, dv = n_draw_score(ctx, ply, checked)
        if has_draw:
            return dv
        if ply >= SEARCH_PATH - 8:
            return n_static_eval(ctx, ply)
        a = -MATE + ply
        if a > alpha:
            alpha = a
        b_ = MATE - ply - 1
        if b_ < beta:
            beta = b_
        if alpha >= beta:
            n_note(ctx, R_mate_dist, ply, 0)
            return alpha

    key = U[X_U64 + J_KEY]
    excluded = np.int64(N[X_SS_EXCL + ply + 4])
    half, horizon, model, util, rep, unknown = n_vctx(ctx, ply)
    hit, tt_move16, tt_score, tt_eval, tt_depth, tt_bound, tt_cutoff, eval_ok, _slot = k_tt_probe(
        ctx, key, ply, half, horizon, model, util, rep, unknown
    )
    if not eval_ok:
        tt_eval = -INF
    if hit:
        L[X_ST + I_TTHITS] += 1
    if (
        not is_pv
        and hit
        and tt_cutoff
        and tt_depth >= depth
        and excluded == 0
        and (
            tt_bound == BOUND_EXACT
            or (tt_bound == BOUND_LOWER and tt_score >= beta)
            or (tt_bound == BOUND_UPPER and tt_score <= alpha)
        )
    ):
        n_note(ctx, R_tt_cut, ply, 0)
        return tt_score

    N[X_SS_NULL + ply + 4] = np.int32(0)
    raw_static = -INF
    if checked:
        static = -INF
        improving = False
    else:
        if hit and tt_eval > -MATE_IN_MAX:
            raw_static = tt_eval
        else:
            raw_static = n_static_eval(ctx, ply)
        static = n_corrected(ctx, raw_static)
        prev2 = np.int64(N[X_SS_EVAL + ply + 2])
        improving = static > prev2 if prev2 != -INF else True
    N[X_SS_EVAL + ply + 4] = np.int32(static)
    N[X_KILLERS + (ply + 2) * 2] = np.int32(0)
    N[X_KILLERS + (ply + 2) * 2 + 1] = np.int32(0)

    if not is_pv and not checked and beta < MATE_IN_MAX and beta > -MATE_IN_MAX and excluded == 0:
        # Reverse futility
        if (
            depth <= N[X_PARAMS + P_rfp_depth]
            and static
            - (
                np.int64(N[X_PARAMS + P_rfp_margin]) * depth
                - (np.int64(N[X_PARAMS + P_rfp_improving]) if improving else np.int64(0))
            )
            >= beta
        ):
            n_note(ctx, R_rfp, ply, 0)
            return static
        # Razoring
        if (
            depth <= N[X_PARAMS + P_razor_depth]
            and static + np.int64(N[X_PARAMS + P_razor_margin]) * depth <= alpha
        ):
            v = n_search(ctx, Z, alpha, beta, ply, Z, Z)
            if L[X_ST + I_DL_STOP]:
                return np.int64(0)
            n_note(ctx, R_razor_verify, ply, 0)
            if v <= alpha:
                n_note(ctx, R_razor, ply, 0)
                return v
        # Null-move pruning
        stm = L[X_ST + I_SIDE]
        base = stm * 6
        non_pawn = (
            U[X_BB + base + KNIGHT]
            | U[X_BB + base + BISHOP]
            | U[X_BB + base + ROOK]
            | U[X_BB + base + QUEEN]
        )
        if (
            depth >= N[X_PARAMS + P_nmp_min_depth]
            and static - beta >= N[X_PARAMS + P_nmp_eval_margin]
            and (not N[X_PARAMS + P_nmp_no_consec] or N[X_SS_NULL + ply + 3] == 0)
            and non_pawn != 0
            and ply >= L[X_ST + I_NULLMIN]
            and (
                not N[X_PARAMS + P_nmp_tt_guard]
                or not hit
                or tt_bound != BOUND_UPPER
                or tt_score >= beta
            )
        ):
            r = (
                np.int64(N[X_PARAMS + P_nmp_base])
                + depth // np.int64(N[X_PARAMS + P_nmp_depth_div])
                + min(
                    (static - beta) // np.int64(N[X_PARAMS + P_nmp_eval_div]),
                    np.int64(N[X_PARAMS + P_nmp_eval_max]),
                )
            )
            N[X_SS_NULL + ply + 4] = np.int32(1)
            N[X_SS_MOVE + ply + 4] = np.int32(0)
            N[X_SS_PIECE + ply + 4] = np.int32(-1)  # -1 = no prev move (WP==0)
            n_make_null(ctx, ply)
            score = -n_search(ctx, depth - r, -beta, -beta + 1, ply + 1, Z, ONE - cut_node)
            n_unmake_null(ctx, ply)
            N[X_SS_NULL + ply + 4] = np.int32(0)
            if L[X_ST + I_DL_STOP]:
                return np.int64(0)
            if score >= beta:
                if score >= MATE_IN_MAX:
                    score = beta
                if depth < N[X_PARAMS + P_nmp_verify_depth]:
                    n_note(ctx, R_nmp, ply, 0)
                    return score
                saved = L[X_ST + I_NULLMIN]
                L[X_ST + I_NULLMIN] = ply + max(
                    np.int64(N[X_PARAMS + P_nmp_verify_min]),
                    np.int64(N[X_PARAMS + P_nmp_verify_num])
                    * max(np.int64(1), depth - r)
                    // np.int64(N[X_PARAMS + P_nmp_verify_den]),
                )
                v = n_search(ctx, depth - r, beta - 1, beta, ply, Z, Z)
                L[X_ST + I_NULLMIN] = saved
                if L[X_ST + I_DL_STOP]:
                    return np.int64(0)
                if v >= beta:
                    n_note(ctx, R_nmp, ply, 0)
                    return score
                n_note(ctx, R_nmp_verify_fail, ply, 0)
        # ProbCut
        if (
            depth >= N[X_PARAMS + P_pc_depth]
            and beta > -MATE_IN_MAX
            and not (
                hit
                and tt_bound == BOUND_LOWER
                and tt_depth >= depth - N[X_PARAMS + P_pc_tt_slack]
                and tt_score < beta + N[X_PARAMS + P_pc_margin]
            )
        ):
            pc_beta = beta + np.int64(N[X_PARAMS + P_pc_margin])
            n_pc = n_gen_noisy(ctx, ply)
            pc_moves = N[X_MOVES + ply * MCAP : X_MOVES + ply * MCAP + MCAP]
            for j in range(n_pc):
                m2 = np.int64(pc_moves[j])
                if not k_see_ge(
                    ctx,
                    m2,
                    min(pc_beta - static, np.int64(N[X_PARAMS + P_pc_see_cap])),
                ):
                    continue
                N[X_SS_MOVE + ply + 4] = np.int32(m2)
                N[X_SS_PIECE + ply + 4] = np.int32(B[X_MB + (m2 & 63)])
                n_make(ctx, ply, m2)
                v = -n_search(ctx, Z, -pc_beta, -pc_beta + 1, ply + 1, Z, Z)
                if v >= pc_beta and depth - N[X_PARAMS + P_pc_depth] > 0:
                    v = -n_search(
                        ctx,
                        depth - N[X_PARAMS + P_pc_depth],
                        -pc_beta,
                        -pc_beta + 1,
                        ply + 1,
                        Z,
                        ONE,
                    )
                n_unmake(ctx, ply, m2)
                if L[X_ST + I_DL_STOP]:
                    return np.int64(0)
                if v >= pc_beta:
                    N[X_SS_MOVE + ply + 4] = np.int32(0)
                    N[X_SS_PIECE + ply + 4] = np.int32(-1)
                    n_note(ctx, R_probcut, ply, m2)
                    h2, ho2, mo2, uu2, rp2, un2 = n_vctx(ctx, ply)
                    k_tt_store(
                        ctx,
                        key,
                        m2 & 0x7FFF,
                        v,
                        raw_static,
                        depth - np.int64(N[X_PARAMS + P_pc_depth]),
                        BOUND_LOWER,
                        h2,
                        ho2,
                        mo2,
                        uu2,
                        rp2,
                        un2,
                        ply,
                    )
                    return v
            N[X_SS_MOVE + ply + 4] = np.int32(0)
            N[X_SS_PIECE + ply + 4] = np.int32(-1)

    # Singular extension
    singular = np.int64(0)
    if (
        not root
        and excluded == 0
        and depth >= N[X_PARAMS + P_sing_depth]
        and hit
        and tt_move16 != 0
        and tt_bound != BOUND_UPPER
        and tt_depth >= depth - N[X_PARAMS + P_sing_tt_slack]
        and (tt_score if tt_score >= 0 else -tt_score) < MATE_IN_MAX
        and ply < np.int64(N[X_PARAMS + P_sing_ply_cap_mult]) * L[X_ST + I_ROOTDEPTH]
    ):
        s_beta = tt_score - np.int64(N[X_PARAMS + P_sing_margin]) * depth
        N[X_SS_EXCL + ply + 4] = np.int32(n_move16_at(ctx, ply, tt_move16))
        if N[X_SS_EXCL + ply + 4]:
            v = n_search(
                ctx,
                (depth - 1) // np.int64(N[X_PARAMS + P_sing_half_div]),
                s_beta - 1,
                s_beta,
                ply,
                Z,
                cut_node,
            )
            N[X_SS_EXCL + ply + 4] = np.int32(0)
            if L[X_ST + I_DL_STOP]:
                return np.int64(0)
            if v < s_beta:
                singular = np.int64(N[X_PARAMS + P_sing_ext])
                n_note(ctx, R_sing_ext, ply, 0)
        else:
            N[X_SS_EXCL + ply + 4] = np.int32(0)
    # Internal iterative reduction
    if depth >= N[X_PARAMS + P_iir_depth] and (not hit or tt_move16 == 0) and (is_pv or cut_node):
        depth -= 1
        n_note(ctx, R_iir, ply, 0)

    moves = N[X_MOVES + ply * MCAP : X_MOVES + ply * MCAP + MCAP]
    scores = N[X_SCORES + ply * MCAP : X_SCORES + ply * MCAP + MCAP]
    quiets = N[X_QUIETS + ply * MCAP : X_QUIETS + ply * MCAP + MCAP]
    n = k_gen_legal(ctx, moves)
    if n == 0:
        if excluded != 0:
            return alpha
        if checked:
            n_note(ctx, R_mate, ply, 0)
            return -MATE + ply
        n_note(ctx, R_stalemate, ply, 0)
        return np.int64(DRAW)
    n_score_moves(ctx, n, ply, tt_move16 if hit else np.int64(0), False)
    best = -INF
    best_move = np.int64(0)
    moves_seen = np.int64(0)
    quiet_count = np.int64(0)
    lmp_limit = (
        np.int64(N[X_PARAMS + P_lmp_base]) + np.int64(N[X_PARAMS + P_lmp_quad]) * depth * depth
    ) // max(
        np.int64(1),
        np.int64(N[X_PARAMS + P_lmp_improving_div]) - (np.int64(1) if improving else np.int64(0)),
    )
    for i in range(n):
        m = n_pick(ctx, i, n, ply)
        if excluded and (m & 0x7FFF) == (excluded & 0x7FFF):
            continue
        mscore = np.int64(scores[i])
        frm = m & 63
        to = (m >> 6) & 63
        promo = (m >> 12) & 7
        captured = (m >> 18) & 15
        piece = np.int64(B[X_MB + frm])
        is_quiet = captured == 15 and promo == 0
        moves_seen += 1
        lmr_r = (
            L[X_LMR + min(depth, 63) * 64 + min(moves_seen, 63)] if moves_seen > 1 else np.int64(0)
        )
        lmr_depth = depth - 1 - lmr_r
        if not root and best > -MATE_IN_MAX and not checked:
            if is_quiet:
                # A checking move is never prunable-quiet material (mirror
                # of _ab): checks are where forced mates hide.  -1 = the
                # check test has not run yet.
                checking = np.int64(-1)
                if quiet_count >= lmp_limit:
                    checking = np.int64(1) if k_gives_check_fast(ctx, m) else np.int64(0)
                    if not checking:
                        n_note(ctx, R_lmp, ply, m)
                        continue
                if (
                    lmr_depth <= N[X_PARAMS + P_fut_depth]
                    and static
                    + np.int64(N[X_PARAMS + P_fut_base])
                    + np.int64(N[X_PARAMS + P_fut_per_depth]) * lmr_depth
                    <= alpha
                ):
                    if checking < 0:
                        checking = np.int64(1) if k_gives_check_fast(ctx, m) else np.int64(0)
                    if not checking:
                        n_note(ctx, R_futility, ply, m)
                        continue
                if lmr_depth <= N[X_PARAMS + P_see_quiet_depth]:
                    ld = (
                        lmr_depth
                        if lmr_depth > N[X_PARAMS + P_see_quiet_min_ld]
                        else np.int64(N[X_PARAMS + P_see_quiet_min_ld])
                    )
                    if not k_see_ge(
                        ctx,
                        m,
                        -np.int64(N[X_PARAMS + P_see_quiet_mult]) * ld * ld,
                    ):
                        if checking < 0:
                            checking = np.int64(1) if k_gives_check_fast(ctx, m) else np.int64(0)
                        if not checking:
                            n_note(ctx, R_see_quiet, ply, m)
                            continue
            elif (
                depth <= N[X_PARAMS + P_see_noisy_depth]
                and mscore < S_KILLER
                and not k_see_ge(
                    ctx,
                    m,
                    -np.int64(N[X_PARAMS + P_see_noisy_mult]) * depth,
                )
            ):
                n_note(ctx, R_see_noisy, ply, m)
                continue
        N[X_SS_MOVE + ply + 4] = np.int32(m)
        N[X_SS_PIECE + ply + 4] = np.int32(piece)
        nodes_before = L[X_ST + I_NODES]
        n_make(ctx, ply, m)
        gives_check = k_in_check(ctx)
        ext = (
            np.int64(N[X_PARAMS + P_check_ext])
            if (
                gives_check
                and depth < N[X_PARAMS + P_check_ext_depth]
                and ply < np.int64(N[X_PARAMS + P_check_ply_cap_mult]) * L[X_ST + I_ROOTDEPTH]
            )
            else np.int64(0)
        )
        if ext:
            n_note(ctx, R_check_ext, ply, m)
        if singular and (m & 0x7FFF) == tt_move16:
            ext = singular
        new_depth = depth - 1 + ext
        score = -INF
        do_full = True
        if (
            depth >= N[X_PARAMS + P_lmr_min_depth]
            and moves_seen
            > (N[X_PARAMS + P_lmr_gate_pv] if is_pv else N[X_PARAMS + P_lmr_gate_nonpv])
            and (is_quiet or mscore < 0)
        ):
            r = lmr_r
            if not improving:
                r += N[X_PARAMS + P_lmr_not_improving]
            if cut_node:
                r += N[X_PARAMS + P_lmr_cut_node]
            if is_pv:
                r -= N[X_PARAMS + P_lmr_pv]
            if mscore >= S_KILLER:
                r -= N[X_PARAMS + P_lmr_killer]
            elif is_quiet:
                r -= mscore // np.int64(N[X_PARAMS + P_lmr_hist_div])
            lo_r = np.int64(N[X_PARAMS + P_lmr_min_r])
            hi_r = new_depth - np.int64(N[X_PARAMS + P_lmr_max_sub])
            if hi_r < lo_r:
                hi_r = lo_r
            if r < lo_r:
                r = lo_r
            elif r > hi_r:
                r = hi_r
            if r > 0:
                n_note(ctx, R_lmr, ply, m)
                score = -n_search(ctx, new_depth - r, -alpha - 1, -alpha, ply + 1, Z, ONE)
                if L[X_ST + I_DL_STOP]:
                    n_unmake(ctx, ply, m)
                    return np.int64(0)
                do_full = score > alpha
                if do_full:
                    n_note(ctx, R_lmr_research, ply, m)
        if do_full and (not is_pv or moves_seen > 1):
            if L[X_ST + I_DL_STOP]:
                n_unmake(ctx, ply, m)
                return np.int64(0)
            score = -n_search(ctx, new_depth, -alpha - 1, -alpha, ply + 1, Z, ONE - cut_node)
        if is_pv and (moves_seen == 1 or (alpha < score < beta)):
            if do_full or score > alpha:
                n_note(ctx, R_pvs_research, ply, m)
            if L[X_ST + I_DL_STOP]:
                n_unmake(ctx, ply, m)
                return np.int64(0)
            score = -n_search(ctx, new_depth, -beta, -alpha, ply + 1, ONE, Z)
        n_unmake(ctx, ply, m)
        if L[X_ST + I_DL_STOP]:
            return np.int64(0)
        if is_quiet:
            quiets[quiet_count] = np.int32(m)
            quiet_count += 1
        if root:
            # root per-move score table keyed by move identity
            N[X_ROOTSC + i] = np.int32(score)
        if score > best:
            best = score
            if score > alpha:
                best_move = m
                if root:
                    L[X_ST + I_ROOTBEST] = m
                    L[X_ST + I_ROOTSCORE] = score
                    L[X_ST + I_BESTNODES] = L[X_ST + I_NODES] - nodes_before
                if score >= beta:
                    if is_quiet:
                        if m != np.int64(N[X_KILLERS + ply * 2]):
                            N[X_KILLERS + ply * 2 + 1] = N[X_KILLERS + ply * 2]
                            N[X_KILLERS + ply * 2] = np.int32(m)
                        p1 = np.int64(N[X_SS_PIECE + ply + 3])
                        t1 = (np.int64(N[X_SS_MOVE + ply + 3]) >> 6) & 63
                        p2 = np.int64(N[X_SS_PIECE + ply + 2])
                        t2 = (np.int64(N[X_SS_MOVE + ply + 2]) >> 6) & 63
                        k_update_quiets(
                            ctx,
                            m,
                            X_QUIETS + ply * MCAP,
                            quiet_count,
                            depth,
                            p1,
                            t1,
                            p2,
                            t2,
                        )
                    else:
                        vt = np.int64(0 if captured == 15 else captured % 6 + 1)
                        k_update_capture(ctx, piece, to, vt, depth)
                    break
                alpha = score
    if excluded != 0:
        return alpha if best == -INF else best
    if best >= beta:
        bound = BOUND_LOWER
    elif is_pv and best_move != 0:
        bound = BOUND_EXACT
    else:
        bound = BOUND_UPPER
    if (
        not checked
        and (best if best >= 0 else -best) < MATE_IN_MAX
        and (best_move == 0 or (((best_move >> 18) & 15) == 15 and ((best_move >> 12) & 7) == 0))
        and not (bound == BOUND_LOWER and best <= static)
        and not (bound == BOUND_UPPER and best >= static)
    ):
        k_update_correction(ctx, L[X_ST + I_SIDE], depth, raw_static, best)
        n_note(ctx, R_corr_update, ply, best_move)
    h3, ho3, mo3, uu3, rp3, un3 = n_vctx(ctx, ply)
    k_tt_store(
        ctx,
        key,
        best_move & 0x7FFF,
        best,
        raw_static,
        depth,
        bound,
        h3,
        ho3,
        mo3,
        uu3,
        rp3,
        un3,
        ply,
    )
    return best


# ---------------------------------------------------------------------------
# per-search reset — mirror of Searcher._begin (+tt.new_search, eval set_root)
# ---------------------------------------------------------------------------


@njit(cache=True)
def k_begin(ctx, hard_budget_ns, node_limit, check_mask):
    """hard_budget_ns is a *duration* (ns), not an absolute deadline."""
    L = ctx[AI]
    U = ctx[AU]
    N = ctx[A32]
    st = L[X_ST : X_ST + N_ST]
    st[I_DL_START] = clock_ns()
    st[I_NODES] = 0
    st[I_QNODES] = 0
    st[I_SELDEPTH] = 0
    st[I_TTHITS] = 0
    st[I_NULLMIN] = 0
    st[I_ROOTBEST] = 0
    st[I_ROOTSCORE] = -INF
    st[I_BESTNODES] = 0
    st[I_PHASE] = 0
    st[I_AGE] = (st[I_AGE] + 1) & 63
    for i in range(PATH * 2):
        N[X_KILLERS + i] = np.int32(0)
    st[I_DL_HARD] = hard_budget_ns
    st[I_DL_NLIM] = node_limit
    st[I_DL_MASK] = check_mask
    st[I_DL_NODES] = 0
    st[I_DL_STOP] = 0
    st[I_CLKREADS] = 0
    st[I_OVERRUN] = 0
    U[X_PATHKEY] = U[X_U64 + J_KEY]
    N[X_PNF] = np.int32(0)
    N[X_PIF] = np.int32(0)
    for i in range(8):
        N[X_SS_MOVE + i] = np.int32(0)
        N[X_SS_PIECE + i] = np.int32(-1)  # -1 sentinel (a white pawn is 0)
    _ph, gh = n_rep_scan(ctx, 0)
    N[X_NODEREP] = np.int32(gh + 1)
    if st[I_EVALKIND] == 1:
        n_ev_set_root(ctx)


@njit(cache=True)
def n_extract_pv(ctx, depth):
    """TT-walk PV extraction into X_PVBUF — mirror of _extract_pv.

    Only called by the driver after a completed iteration.  Makes/unmakes
    restore the board exactly; returns the PV length."""
    U = ctx[AU]
    N = ctx[A32]
    n_pv = 0
    made = 0
    buf = N[X_SCORES + (SEARCH_PATH - 1) * MCAP : X_SCORES + (SEARCH_PATH - 1) * MCAP + MCAP]
    limit = depth + 4
    if limit > SEARCH_PATH // 2:
        limit = SEARCH_PATH // 2
    for p in range(limit):
        hit, m16, _s, _e, _d, _b, _co, _eo, _sl = k_tt_probe(
            ctx, U[X_U64 + J_KEY], p, 0, 0, 0, 0, 0, 0
        )
        if not hit or m16 == 0:
            break
        n = k_gen_legal(ctx, buf)
        found = np.int64(0)
        for i in range(n):
            if (np.int64(buf[i]) & 0x7FFF) == m16:
                found = np.int64(buf[i])
                break
        if not found:
            break
        N[X_PVBUF + n_pv] = np.int32(found)
        n_pv += 1
        k_make(ctx, found)
        made += 1
    for _ in range(made):
        k_unmake(ctx)
    return n_pv

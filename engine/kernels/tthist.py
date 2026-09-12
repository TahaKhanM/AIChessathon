"""Compiled transposition table + ordering/correction history kernels.

Bit-exact transliterations of ``engine/tt.py`` (packed 16-byte clustered
entries, value-context and repetition-band score gating) and
``engine/history.py`` (gravity updates, continuation/capture/pawn/threat
ordering, correction history).  The TT data array is the SAME
``TranspositionTable.clusters`` buffer the Python object owns; the history
tables are flat regions of the i32 arena initialised from ``HistoryTables``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from engine.kernels.layout import (
    A32,
    A8,
    AI,
    AU,
    CTT,
    I_AGE,
    I_REPLACES,
    I_SIDE,
    I_WRITES,
    J_TTMASK,
    P_corr_np_w,
    P_corr_pawn_w,
    P_corr_weight_cap,
    P_hist_bonus_cap,
    P_hist_bonus_const,
    P_hist_bonus_lin,
    P_hist_bonus_quad,
    P_hist_pawn_div,
    P_hist_threat_div,
    X_CORR,
    X_HCAP,
    X_HCOUNTER,
    X_HCONT,
    X_HPAWN,
    X_HTHREAT,
    X_HQUIET,
    X_MB,
    X_OCC,
    X_PARAMS,
    X_ST,
    X_U64,
)
from engine.kernels.bb import (
    M64,
    k_nonpawn_key,
    k_pawn_key,
    k_square_attacked,
)
from engine.tt import (
    BOUND_EMPTY,
    BOUND_EXACT,
    HORIZON_BAND,
    HORIZON_SLACK,
    RULE50_CUTOFF_MAX,
    MATE_IN_MAX,
)
from engine.history import (
    CORR_GRAIN,
    CORR_LIMIT,
    CORR_SIZE,
    HISTORY_MAX,
    PAWN_HIST_SIZE,
)

CLUSTER = 4
AGE_MASK = 63
INF = 32000


# ---------------------------------------------------------------------------
# TT word packing — mirrors tt._pack / _unpack / _unpack_ctx bit-for-bit
# ---------------------------------------------------------------------------


@njit(cache=True)
def _score_to_tt(score, ply, mate_min=MATE_IN_MAX):
    if score >= mate_min:
        return score + ply
    if score <= -mate_min:
        return score - ply
    return score


@njit(cache=True)
def _score_from_tt(score, ply, mate_min=MATE_IN_MAX):
    if score >= mate_min:
        return score - ply
    if score <= -mate_min:
        return score + ply
    return score


@njit(cache=True)
def _horizon_band(remaining_horizon):
    b = remaining_horizon // HORIZON_BAND
    return np.int64(31) if b > 31 else np.int64(b)


@njit(cache=True)
def _rep_band(count):
    return np.int64(count) if count < 3 else np.int64(3)


@njit(cache=True)
def _tt_pack(
    key, move16, score, raw_eval, depth, bound, age, halfmove, horizon, model, util, rep, unknown
):
    """Return (w0, w1) exactly as tt._pack."""
    w0 = (key >> 32) & np.uint64(0xFFFFFFFF)
    w0 |= np.uint64(move16 & 0x7FFF) << np.uint64(32)
    w0 |= np.uint64(depth & 0xFF) << np.uint64(47)
    w0 |= np.uint64(bound & 3) << np.uint64(55)
    w0 |= np.uint64(age & AGE_MASK) << np.uint64(57)
    w1 = np.uint64(score & 0xFFFF)
    w1 |= np.uint64(raw_eval & 0xFFFF) << np.uint64(16)
    hm = halfmove if halfmove < 127 else np.int64(127)
    w1 |= np.uint64(hm & 0x7F) << np.uint64(32)
    w1 |= np.uint64(_horizon_band(horizon) & 0x1F) << np.uint64(39)
    w1 |= np.uint64(model & 0xFF) << np.uint64(44)
    w1 |= np.uint64(util & 0xF) << np.uint64(52)
    w1 |= np.uint64(_rep_band(rep) & 3) << np.uint64(56)
    if unknown:
        w1 |= np.uint64(1) << np.uint64(58)
    w1 |= (key >> np.uint64(60)) << np.uint64(60)
    return w0, w1


@njit(cache=True)
def _tt_unpack(w0, w1):
    """(move16, score, raw_eval, depth, bound, age) — mirror of tt._unpack."""
    move16 = np.int64((w0 >> np.uint64(32)) & np.uint64(0x7FFF))
    depth = np.int64((w0 >> np.uint64(47)) & np.uint64(0xFF))
    bound = np.int64((w0 >> np.uint64(55)) & np.uint64(3))
    age = np.int64((w0 >> np.uint64(57)) & np.uint64(AGE_MASK))
    score = np.int64(w1 & np.uint64(0xFFFF))
    if score >= 0x8000:
        score -= 0x10000
    raw_eval = np.int64((w1 >> np.uint64(16)) & np.uint64(0xFFFF))
    if raw_eval >= 0x8000:
        raw_eval -= 0x10000
    return move16, score, raw_eval, depth, bound, age


@njit(cache=True)
def _score_cutoff_ok(half, horizon, rep, unknown, e_half, e_hband, e_depth, e_rep, e_unknown):
    """Mirror of ValueContext.score_cutoff_ok (model/util already compared)."""
    if e_unknown != (1 if unknown else 0):
        return False
    if e_rep != _rep_band(rep):
        return False
    if half >= RULE50_CUTOFF_MAX or e_half >= RULE50_CUTOFF_MAX:
        return False
    stored_safe = e_hband * HORIZON_BAND > e_depth + HORIZON_SLACK
    current_safe = horizon > e_depth + HORIZON_SLACK
    if not (stored_safe and current_safe):
        if _horizon_band(horizon) != e_hband:
            return False
    return True


# ---------------------------------------------------------------------------
# probe / store — mirrors TranspositionTable.probe_into / store
# ---------------------------------------------------------------------------


@njit(cache=True)
def k_tt_probe(ctx, key, ply, half, horizon, model, util, rep, unknown):
    """Probe the shared cluster array.

    Returns (hit, move16, score, raw_eval, depth, bound, cutoff_ok, eval_ok,
    slot).  Mirrors probe_into with ctx always applied (the caller supplies
    the live value context).
    """
    tt = ctx[CTT].ravel()
    mask = ctx[AU][X_U64 + J_TTMASK]
    idx = key & mask
    tag = (key >> np.uint64(32)) & np.uint64(0xFFFFFFFF)
    tag_hi = (key >> np.uint64(60)) & np.uint64(0xF)
    base = idx * np.uint64(CLUSTER * 2)
    for slot in range(CLUSTER):
        w0 = np.uint64(tt[base + slot * 2]) & M64
        if (w0 & np.uint64(0xFFFFFFFF)) != tag or (
            (w0 >> np.uint64(55)) & np.uint64(3)
        ) == BOUND_EMPTY:
            continue
        w1 = np.uint64(tt[base + slot * 2 + 1]) & M64
        if ((w1 >> np.uint64(60)) & np.uint64(0xF)) != tag_hi:
            continue
        move16, score, raw_eval, depth, bound, age = _tt_unpack(w0, w1)
        score = _score_from_tt(score, ply)
        e_half = np.int64((w1 >> np.uint64(32)) & np.uint64(0x7F))
        e_hband = np.int64((w1 >> np.uint64(39)) & np.uint64(0x1F))
        e_model = np.int64((w1 >> np.uint64(44)) & np.uint64(0xFF))
        e_util = np.int64((w1 >> np.uint64(52)) & np.uint64(0xF))
        e_rep = np.int64((w1 >> np.uint64(56)) & np.uint64(3))
        e_unk = np.int64((w1 >> np.uint64(58)) & np.uint64(1))
        cutoff_ok = False
        if e_model == (model & 0xFF) and e_util == (util & 0xF):
            cutoff_ok = _score_cutoff_ok(
                half,
                horizon,
                rep,
                unknown,
                e_half,
                e_hband,
                depth,
                e_rep,
                e_unk,
            )
        eval_ok = e_model == (model & 0xFF) and e_util == (util & 0xF)
        slot_id = np.int64(idx) * CLUSTER + slot
        return (True, move16, score, raw_eval, depth, bound, cutoff_ok, eval_ok, slot_id)
    return (
        False,
        np.int64(0),
        np.int64(0),
        np.int64(-INF),
        np.int64(0),
        np.int64(BOUND_EMPTY),
        False,
        False,
        np.int64(-1),
    )


@njit(cache=True)
def k_tt_store(
    ctx, key, move16, score, raw_eval, depth, bound, half, horizon, model, util, rep, unknown, ply
):
    """Mirror of TranspositionTable.store (payload-before-meta)."""
    tt = ctx[CTT].ravel()
    L = ctx[AI]
    score = _score_to_tt(score, ply)
    if score > 32767:
        score = np.int64(32767)
    elif score < -32768:
        score = np.int64(-32768)
    if raw_eval > 32767:
        raw_eval = np.int64(32767)
    elif raw_eval < -32768:
        raw_eval = np.int64(-32768)
    if depth < 0:
        depth = np.int64(0)
    mask = ctx[AU][X_U64 + J_TTMASK]
    idx = key & mask
    tag = (key >> np.uint64(32)) & np.uint64(0xFFFFFFFF)
    base = idx * np.uint64(CLUSTER * 2)
    age = L[X_ST + I_AGE]
    target = np.int64(-1)
    best_q = np.int64(1) << 30
    for slot in range(CLUSTER):
        w0 = np.uint64(tt[base + slot * 2]) & M64
        obound = (w0 >> np.uint64(55)) & np.uint64(3)
        if obound == BOUND_EMPTY:
            target = np.int64(slot)
            break
        w1 = np.uint64(tt[base + slot * 2 + 1]) & M64
        if (w0 & np.uint64(0xFFFFFFFF)) == tag:
            target = np.int64(slot)
            omove, _s, _e, odepth, obound, oage = _tt_unpack(w0, w1)
            if move16 == 0:
                move16 = omove
            # Same-key policy (mirror of TranspositionTable.store):
            # worth = depth (+2 EXACT) - 4 plies per generation of age;
            # replace on a strict win, or on a tie when not shallower.
            q_new = depth + (np.int64(2) if bound == BOUND_EXACT else np.int64(0))
            q_old = (
                odepth
                + (np.int64(2) if obound == BOUND_EXACT else np.int64(0))
                - np.int64(4) * ((age - oage) & AGE_MASK)
            )
            if q_new < q_old or (q_new == q_old and depth < odepth):
                return
            break
        _m, _s, _e, odepth, _b, oage = _tt_unpack(w0, w1)
        q = odepth - 4 * ((age - oage) & AGE_MASK)
        if q < best_q:
            best_q = q
            target = np.int64(slot)
    w0, w1 = _tt_pack(
        key, move16, score, raw_eval, depth, bound, age, half, horizon, model, util, rep, unknown
    )
    if np.int64(tt[base + target * 2]) != 0:
        L[X_ST + I_REPLACES] += 1
    tt[base + target * 2 + 1] = np.int64(w1)
    tt[base + target * 2] = np.int64(w0)
    L[X_ST + I_WRITES] += 1


# ---------------------------------------------------------------------------
# ordering + correction histories — mirrors history.HistoryTables
# ---------------------------------------------------------------------------


@njit(cache=True)
def k_hist_update(value, bonus):
    v = value + bonus - value * (bonus if bonus >= 0 else -bonus) // HISTORY_MAX
    if v > HISTORY_MAX:
        return HISTORY_MAX
    if v < -HISTORY_MAX:
        return -HISTORY_MAX
    return v


@njit(cache=True)
def k_stat_bonus(ctx, depth):
    """history.stat_bonus with the params-vector coefficients."""
    N = ctx[A32]
    b = (
        np.int64(N[X_PARAMS + P_hist_bonus_quad]) * depth * depth
        + np.int64(N[X_PARAMS + P_hist_bonus_lin]) * depth
        + np.int64(N[X_PARAMS + P_hist_bonus_const])
    )
    cap = np.int64(N[X_PARAMS + P_hist_bonus_cap])
    return b if b < cap else cap


@njit(cache=True)
def k_quiet_score(ctx, frm, to, piece, p1, t1, p2, t2):
    """Composite quiet ordering — mirror of HistoryTables.quiet_score."""
    N = ctx[A32]
    L = ctx[AI]
    stm = L[X_ST + I_SIDE]
    s = np.int64(N[X_HQUIET + stm * 64 * 64 + frm * 64 + to])
    # p1/p2 == -1 is the no-previous-move sentinel (a white pawn is 0).
    if p1 >= 0:
        s += N[X_HCONT + 0 * 13 * 64 * 13 * 64 + p1 * 64 * 13 * 64 + t1 * 13 * 64 + piece * 64 + to]
    if p2 >= 0:
        s += N[X_HCONT + 1 * 13 * 64 * 13 * 64 + p2 * 64 * 13 * 64 + t2 * 13 * 64 + piece * 64 + to]
    s += (
        N[X_HPAWN + (k_pawn_key(ctx) % PAWN_HIST_SIZE) * 13 * 64 + piece * 64 + to]
        // N[X_PARAMS + P_hist_pawn_div]
    )
    occ = ctx[AU][X_OCC + 2]
    them = stm ^ 1
    frm_th = 1 if k_square_attacked(ctx, frm, them, occ) else 0
    to_th = 1 if k_square_attacked(ctx, to, them, occ) else 0
    s += (
        N[X_HTHREAT + frm_th * 2 * 64 * 64 + to_th * 64 * 64 + frm * 64 + to]
        // N[X_PARAMS + P_hist_threat_div]
    )
    return s


@njit(cache=True)
def k_capture_score(ctx, piece, to, victim):
    return np.int64(ctx[A32][X_HCAP + piece * 64 * 7 + to * 7 + victim])


@njit(cache=True)
def k_counter_move(ctx, p1, t1):
    if p1 < 0:
        return np.int64(0)
    return np.int64(ctx[A32][X_HCOUNTER + p1 * 64 + t1])


@njit(cache=True)
def k_update_quiets(ctx, best, quiets_off, n_quiets, depth, p1, t1, p2, t2):
    """+bonus to the cutoff quiet, -bonus to the others (mirror)."""
    N = ctx[A32]
    L = ctx[AI]
    stm = L[X_ST + I_SIDE]
    bonus = k_stat_bonus(ctx, depth)
    pidx = k_pawn_key(ctx) % PAWN_HIST_SIZE
    occ = ctx[AU][X_OCC + 2]
    them = stm ^ 1
    B = ctx[A8]
    for i in range(n_quiets):
        m = np.int64(N[quiets_off + i])
        frm = m & 63
        to = (m >> 6) & 63
        piece = np.int64(B[X_MB + frm])
        b = bonus if m == best else -bonus
        N[X_HQUIET + stm * 64 * 64 + frm * 64 + to] = np.int32(
            k_hist_update(np.int64(N[X_HQUIET + stm * 64 * 64 + frm * 64 + to]), b)
        )
        if p1 >= 0:
            o = X_HCONT + p1 * 64 * 13 * 64 + t1 * 13 * 64 + piece * 64 + to
            N[o] = np.int32(k_hist_update(np.int64(N[o]), b))
        if p2 >= 0:
            o = X_HCONT + 13 * 64 * 13 * 64 + p2 * 64 * 13 * 64 + t2 * 13 * 64 + piece * 64 + to
            N[o] = np.int32(k_hist_update(np.int64(N[o]), b))
        o = X_HPAWN + pidx * 13 * 64 + piece * 64 + to
        N[o] = np.int32(k_hist_update(np.int64(N[o]), b))
        frm_th = 1 if k_square_attacked(ctx, frm, them, occ) else 0
        to_th = 1 if k_square_attacked(ctx, to, them, occ) else 0
        o = X_HTHREAT + frm_th * 2 * 64 * 64 + to_th * 64 * 64 + frm * 64 + to
        N[o] = np.int32(k_hist_update(np.int64(N[o]), b))
    if p1 >= 0:
        N[X_HCOUNTER + p1 * 64 + t1] = np.int32(best)


@njit(cache=True)
def k_update_capture(ctx, piece, to, victim, depth):
    N = ctx[A32]
    b = k_stat_bonus(ctx, depth)
    o = X_HCAP + piece * 64 * 7 + to * 7 + victim
    N[o] = np.int32(k_hist_update(np.int64(N[o]), b))


@njit(cache=True)
def k_correction_cp(ctx, stm):
    """Centipawn correction to add to RAW static eval — mirror."""
    N = ctx[A32]
    c = X_CORR
    pw_w = np.int64(N[X_PARAMS + P_corr_pawn_w])
    np_w = np.int64(N[X_PARAMS + P_corr_np_w])
    total = (
        pw_w
        * np.int64(
            N[
                c
                + 0 * 2 * CORR_SIZE
                + stm * CORR_SIZE
                + np.int64(k_pawn_key(ctx) & np.uint64(CORR_SIZE - 1))
            ]
        )
        + np_w
        * np.int64(
            N[
                c
                + 1 * 2 * CORR_SIZE
                + stm * CORR_SIZE
                + np.int64(k_nonpawn_key(ctx, 0) & np.uint64(CORR_SIZE - 1))
            ]
        )
        + np_w
        * np.int64(
            N[
                c
                + 2 * 2 * CORR_SIZE
                + stm * CORR_SIZE
                + np.int64(k_nonpawn_key(ctx, 1) & np.uint64(CORR_SIZE - 1))
            ]
        )
    )
    return total // ((pw_w + 2 * np_w) * CORR_GRAIN)


@njit(cache=True)
def k_update_correction(ctx, stm, depth, raw_static, best):
    N = ctx[A32]
    target = (best - raw_static) * CORR_GRAIN
    if target > CORR_LIMIT:
        target = np.int64(CORR_LIMIT)
    elif target < -CORR_LIMIT:
        target = np.int64(-CORR_LIMIT)
    weight = depth + 1
    cap = np.int64(N[X_PARAMS + P_corr_weight_cap])
    if weight > cap:
        weight = cap
    c = X_CORR + stm * CORR_SIZE
    for t in range(3):
        if t == 0:
            idx = k_pawn_key(ctx) & np.uint64(CORR_SIZE - 1)
            o = c + 0 * 2 * CORR_SIZE + idx
        elif t == 1:
            idx = k_nonpawn_key(ctx, 0) & np.uint64(CORR_SIZE - 1)
            o = c + 1 * 2 * CORR_SIZE + idx
        else:
            idx = k_nonpawn_key(ctx, 1) & np.uint64(CORR_SIZE - 1)
            o = c + 2 * 2 * CORR_SIZE + idx
        old = np.int64(N[o])
        new = ((256 - weight) * old + weight * target) // 256
        if new > CORR_LIMIT:
            new = np.int64(CORR_LIMIT)
        elif new < -CORR_LIMIT:
            new = np.int64(-CORR_LIMIT)
        N[o] = np.int32(new)

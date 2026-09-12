"""Compiled evaluation kernels (nopython).

Ports of ``engine.evaluate.Stack`` (lazy incremental F512 accumulator with
king-bucket Finny refresh cache) and ``simple_eval`` onto the arena layout.
Reuses the owner's njit row resolvers / accumulators / head kernels from
``engine.evaluate`` so the feature math is the same code, not a rewrite.

Mirror map (python -> kernel):
  Stack.set_root    -> n_ev_set_root
  Stack.push        -> n_ev_push      (call BEFORE k_make, like the oracle)
  Stack.pop         -> n_ev_pop
  Stack._ensure     -> n_ev_ensure    (same replay-vs-refresh chooser costs)
  Stack._apply_diff -> n_ev_apply_diff
  Stack._refresh    -> n_ev_refresh   (Finny cache = FSIG/FACC/FPSQP/FPSQT)
  Evaluator.evaluate-> n_ev_evaluate
  simple_eval       -> n_simple_eval
"""

from __future__ import annotations

import numpy as np
from numba import njit

import engine.evaluate as _ev
from engine.kernels.layout import (
    A16,
    A32,
    A8,
    AI,
    AU,
    AU8,
    AU32,
    CHANNELS,
    FINNY_FRAMES,
    TABLES,
    I_ADEPTH,
    I_EVALKIND,
    I_FORCEREPLAY,
    I_MAXPP,
    I_MAXPSQ,
    I_MAXT,
    I_NEURALBOUND,
    I_SCALENUM,
    I_SCALESHIFT,
    I_SIDE,
    X_ACC,
    X_AFRAME,
    X_AREFRESH,
    X_AVALID,
    X_B1,
    X_B2,
    X_B3,
    X_BB,
    X_BIAS,
    X_EVALX,
    X_FACC,
    X_FPSQP,
    X_FPSQT,
    X_FSIG,
    X_FVALID,
    X_KING,
    X_MB,
    X_OCC,
    X_PARAMS,
    X_PAWNBB,
    X_PN,
    X_POPS,
    X_PPENUM,
    X_PPW,
    X_PSQPART,
    X_PSQTB,
    X_PSQTA,
    X_PSQTW,
    X_PSQW,
    X_RBADD,
    X_SBB,
    X_SCR64,
    X_SKING,
    X_SMB,
    X_SOCC,
    X_ST,
    X_THRENUM,
    X_THRW,
    X_TN,
    X_TOPS,
    X_W1,
    X_W2,
    X_W3,
    P_eval_clamp,
)
from engine.kernels.bb import (
    KING,
    M64,
    U1,
    k_lsb,
    k_pop64,
)
from engine.evaluate import (
    _acc_add_rows,
    _apply_ply,
    _compute_dirties,
    _enum_apply_persp,
    _enum_pp,
    _enum_threats,
    _head_scalar_nb,
    _paired_transform_nb,
    _reverse_ops,
    ACC_BOUND,
    MAX_ACTIVE_PP,
    MAX_ACTIVE_THREATS,
    PP_OP_CAP,
    PSQT_BUCKETS,
    THREAT_OP_CAP,
)
from engine.search import EVAL_CLAMP, MATE, MATE_IN_MAX  # noqa: F401

_psqt_tab = TABLES["PSQT"]
_K12 = TABLES["K12"]

COST_REPLAY_PLY = 5500
COST_REPLAY_ROW = 350
COST_REFRESH_COLD = 20000
COST_REFRESH_WARM = 7000


# ---------------------------------------------------------------------------
# king-normalization frame — mirror of evaluate._frame
# ---------------------------------------------------------------------------


@njit(cache=True)
def n_frame(ksq, p):
    # frame = (bucket << 1) | mirror; mirror = 1 iff orientation term has the
    # file-flip bit set iff ksq & 4.  bucket = K12[ksq ^ (56*p)].
    return np.int64((_K12[ksq ^ (56 * p)] << 1) | ((ksq >> 2) & 1))


# ---------------------------------------------------------------------------
# shadow-board lifecycle — mirror of Stack.set_root / push / pop
# ---------------------------------------------------------------------------


@njit(cache=True)
def n_ev_set_root(ctx):
    U = ctx[AU]
    L = ctx[AI]
    B = ctx[A8]
    Y = ctx[AU8]
    L[X_ST + I_ADEPTH] = 0
    Y[X_FVALID : X_FVALID + 2 * FINNY_FRAMES] = np.uint8(0)
    Y[X_AVALID] = np.uint8(0)
    Y[X_AVALID + 1] = np.uint8(0)
    for i in range(64):
        B[X_SMB + i] = B[X_MB + i]
    for i in range(12):
        U[X_SBB + i] = U[X_BB + i]
    for i in range(3):
        U[X_SOCC + i] = U[X_OCC + i]
    L[X_SKING] = L[X_KING]
    L[X_SKING + 1] = L[X_KING + 1]
    for p in range(2):
        Y[X_AFRAME + p] = np.uint8(n_frame(L[X_KING + p], p))
        Y[X_AREFRESH + p] = np.uint8(1)


@njit(cache=True)
def n_ev_push(ctx, move):
    """Record narrow dirties for ``move`` (pre-move board state)."""
    U = ctx[AU]
    L = ctx[AI]
    N = ctx[A32]
    B = ctx[A8]
    Y = ctx[AU8]
    W = ctx[AU32]
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    flag = (move >> 15) & 7
    captured = (move >> 18) & 15
    piece = (move >> 22) & 15
    us = L[X_ST + I_SIDE]
    k = L[X_ST + I_ADEPTH] + 1
    smb = B[X_SMB : X_SMB + 64]
    sbb = U[X_SBB : X_SBB + 12]
    socc_c = U[X_SOCC : X_SOCC + 2]
    tops_k = W[X_TOPS + k * THREAT_OP_CAP : X_TOPS + k * THREAT_OP_CAP + THREAT_OP_CAP]
    pops_k = B[X_POPS + k * 24 : X_POPS + k * 24 + 24].reshape(8, 3)
    nt, npo, pwb, pbb, pwa, pba, occ = _compute_dirties(
        smb,
        sbb,
        socc_c,
        U[X_SOCC + 2],
        frm,
        to,
        promo,
        flag,
        piece,
        captured,
        us,
        tops_k,
        pops_k,
    )
    U[X_SOCC + 2] = occ
    assert nt <= THREAT_OP_CAP and npo <= 8
    N[X_TN + k] = np.int32(nt)
    N[X_PN + k] = np.int32(npo)
    U[X_PAWNBB + k * 4] = pwb
    U[X_PAWNBB + k * 4 + 1] = pbb
    U[X_PAWNBB + k * 4 + 2] = pwa
    U[X_PAWNBB + k * 4 + 3] = pba
    if nt > L[X_ST + I_MAXT]:
        L[X_ST + I_MAXT] = nt
    if npo > L[X_ST + I_MAXPSQ]:
        L[X_ST + I_MAXPSQ] = npo
    if piece % 6 == KING:
        L[X_SKING + us] = to
    d = k - 1
    for p in range(2):
        f = n_frame(L[X_SKING + p], p)
        Y[X_AFRAME + k * 2 + p] = np.uint8(f)
        Y[X_AREFRESH + k * 2 + p] = np.uint8(1 if f != np.int64(Y[X_AFRAME + d * 2 + p]) else 0)
    Y[X_AVALID + k * 2] = np.uint8(0)
    Y[X_AVALID + k * 2 + 1] = np.uint8(0)
    L[X_ST + I_ADEPTH] = k


@njit(cache=True)
def n_ev_pop(ctx):
    U = ctx[AU]
    L = ctx[AI]
    N = ctx[A32]
    B = ctx[A8]
    k = L[X_ST + I_ADEPTH]
    np_ = np.int64(N[X_PN + k])
    if np_ > 0:
        pops_k = B[X_POPS + k * 24 : X_POPS + k * 24 + 24].reshape(8, 3)
        U[X_SOCC + 2] = _reverse_ops(
            B[X_SMB : X_SMB + 64],
            U[X_SBB : X_SBB + 12],
            U[X_SOCC : X_SOCC + 2],
            U[X_SOCC + 2],
            pops_k,
            np_,
        )
        for i in range(np_):
            pc = np.int64(B[X_POPS + k * 24 + i * 3])
            frm = np.int64(B[X_POPS + k * 24 + i * 3 + 1])
            if pc >= 0 and pc % 6 == KING and frm >= 0:
                L[X_SKING + pc // 6] = frm
    L[X_ST + I_ADEPTH] = k - 1


# ---------------------------------------------------------------------------
# materialization — mirrors Stack._ensure / _apply_diff / _refresh
# ---------------------------------------------------------------------------


@njit(cache=True)
def n_ev_apply_diff(ctx, k, p):
    """Apply ply-k's recorded delta onto the working accumulator at the
    current depth, perspective p — mirror of Stack._apply_diff."""
    U = ctx[AU]
    L = ctx[AI]
    N = ctx[A32]
    S = ctx[A16]
    B = ctx[A8]
    Y = ctx[AU8]
    W = ctx[AU32]
    depth = L[X_ST + I_ADEPTH]
    f = np.int64(Y[X_AFRAME + k * 2 + p])
    orient = ((f & 1) * 7) ^ (56 * p)
    bucket = f >> 1
    acc = S[X_ACC + (depth * 2 + p) * CHANNELS : X_ACC + (depth * 2 + p) * CHANNELS + CHANNELS]
    psqt_acc = N[
        X_PSQTA + (depth * 2 + p) * PSQT_BUCKETS : X_PSQTA
        + (depth * 2 + p) * PSQT_BUCKETS
        + PSQT_BUCKETS
    ]
    psqw = S[X_PSQW : X_PSQW + 9216 * CHANNELS].reshape(9216, CHANNELS)
    thrw = B[X_THRW : X_THRW + 59808 * CHANNELS].reshape(59808, CHANNELS)
    ppw = B[X_PPW : X_PPW + 1488 * CHANNELS].reshape(1488, CHANNELS)
    psqtw = S[X_PSQTW : X_PSQTW + 9216 * PSQT_BUCKETS].reshape(9216, PSQT_BUCKETS)
    tops_k = W[X_TOPS + k * THREAT_OP_CAP : X_TOPS + k * THREAT_OP_CAP + THREAT_OP_CAP]
    pops_k = B[X_POPS + k * 24 : X_POPS + k * 24 + 24].reshape(8, 3)
    pawnbb_k = U[X_PAWNBB + k * 4 : X_PAWNBB + k * 4 + 4]
    rows_touched, pp_max, thr_max, _psq_rows_touched, peak = _apply_ply(
        acc,
        psqt_acc,
        psqw,
        thrw,
        ppw,
        psqtw,
        tops_k,
        np.int64(N[X_TN + k]),
        pops_k,
        np.int64(N[X_PN + k]),
        pawnbb_k,
        p,
        bucket,
        orient,
    )
    assert pp_max <= PP_OP_CAP and thr_max <= MAX_ACTIVE_THREATS
    assert peak <= ACC_BOUND
    if pp_max > L[X_ST + I_MAXPP]:
        L[X_ST + I_MAXPP] = pp_max


@njit(cache=True)
def n_ev_refresh(ctx, k, p):
    """Recompute perspective p at ply k through the Finny refresh cache —
    mirror of Stack._refresh (piece-set signature hit / diff / cold)."""
    U = ctx[AU]
    L = ctx[AI]
    N = ctx[A32]
    S = ctx[A16]
    B = ctx[A8]
    Y = ctx[AU8]
    f = np.int64(Y[X_AFRAME + k * 2 + p])
    orient = ((f & 1) * 7) ^ (56 * p)
    bucket = f >> 1
    mb = B[X_MB : X_MB + 64]
    bb = U[X_BB : X_BB + 12]
    occ = U[X_OCC + 2]
    acc = S[X_ACC + (k * 2 + p) * CHANNELS : X_ACC + (k * 2 + p) * CHANNELS + CHANNELS]
    psqt = N[
        X_PSQTA + (k * 2 + p) * PSQT_BUCKETS : X_PSQTA + (k * 2 + p) * PSQT_BUCKETS + PSQT_BUCKETS
    ]
    psqw = S[X_PSQW : X_PSQW + 9216 * CHANNELS].reshape(9216, CHANNELS)
    thrw = B[X_THRW : X_THRW + 59808 * CHANNELS].reshape(59808, CHANNELS)
    ppw = B[X_PPW : X_PPW + 1488 * CHANNELS].reshape(1488, CHANNELS)
    psqtw = S[X_PSQTW : X_PSQTW + 9216 * PSQT_BUCKETS].reshape(9216, PSQT_BUCKETS)
    fsig = U[X_FSIG + (p * FINNY_FRAMES + f) * 12 : X_FSIG + (p * FINNY_FRAMES + f) * 12 + 12]
    thr_enum = N[X_THRENUM : X_THRENUM + MAX_ACTIVE_THREATS + 8]
    pp_enum = N[X_PPENUM : X_PPENUM + 160]
    psq_part = N[X_PSQPART : X_PSQPART + CHANNELS]
    sig_match = True
    for i in range(12):
        if fsig[i] != bb[i]:
            sig_match = False
            break
    n_thr = 0
    n_pp = 0
    if Y[X_FVALID + p * FINNY_FRAMES + f] and sig_match:
        facc = S[
            X_FACC + (p * FINNY_FRAMES + f) * CHANNELS : X_FACC
            + (p * FINNY_FRAMES + f) * CHANNELS
            + CHANNELS
        ]
        fpsqt = N[
            X_FPSQT + (p * FINNY_FRAMES + f) * PSQT_BUCKETS : X_FPSQT
            + (p * FINNY_FRAMES + f) * PSQT_BUCKETS
            + PSQT_BUCKETS
        ]
        for i in range(CHANNELS):
            acc[i] = facc[i]
        for i in range(PSQT_BUCKETS):
            psqt[i] = fpsqt[i]
        return
    if Y[X_FVALID + p * FINNY_FRAMES + f]:
        # Finny difference: cached PSQ part +- changed piece rows.
        n_thr = _enum_threats(mb, bb, occ, p, orient, thr_enum)
        assert n_thr <= MAX_ACTIVE_THREATS
        n_pp = _enum_pp(bb[0], bb[6], p, orient, pp_enum)
        assert n_pp <= MAX_ACTIVE_PP
        fpsqp = N[
            X_FPSQP + (p * FINNY_FRAMES + f) * CHANNELS : X_FPSQP
            + (p * FINNY_FRAMES + f) * CHANNELS
            + CHANNELS
        ]
        for i in range(CHANNELS):
            psq_part[i] = fpsqp[i]
        psqt_part = L[X_SCR64 : X_SCR64 + 8]
        fpsqt = N[
            X_FPSQT + (p * FINNY_FRAMES + f) * PSQT_BUCKETS : X_FPSQT
            + (p * FINNY_FRAMES + f) * PSQT_BUCKETS
            + PSQT_BUCKETS
        ]
        for i in range(PSQT_BUCKETS):
            psqt_part[i] = np.int64(fpsqt[i])
        for pc in range(12):
            rem_b = fsig[pc] & (M64 ^ bb[pc])
            add_b = bb[pc] & (M64 ^ fsig[pc])
            if not (rem_b or add_b):
                continue
            relc = 1 if (pc // 6) != p else 0
            base = 768 * bucket + 384 * relc + 64 * (pc % 6)
            while rem_b:
                s = k_lsb(rem_b)
                rem_b &= rem_b - U1
                r = base + (s ^ orient)
                for i in range(CHANNELS):
                    psq_part[i] -= np.int32(psqw[r, i])
                for i in range(PSQT_BUCKETS):
                    psqt_part[i] -= np.int64(psqtw[r, i])
            while add_b:
                s = k_lsb(add_b)
                add_b &= add_b - U1
                r = base + (s ^ orient)
                for i in range(CHANNELS):
                    psq_part[i] += np.int32(psqw[r, i])
                for i in range(PSQT_BUCKETS):
                    psqt_part[i] += np.int64(psqtw[r, i])
        # widened bound check before the int16 store (mirror of _refresh)
        tmp = np.empty(CHANNELS, np.int32)
        for i in range(CHANNELS):
            tmp[i] = psq_part[i]
        assert _ev._acc_peak(tmp) <= ACC_BOUND
        _acc_add_rows(tmp, thrw, thr_enum, n_thr)
        assert _ev._acc_peak(tmp) <= ACC_BOUND
        _acc_add_rows(tmp, ppw, pp_enum, n_pp)
        assert _ev._acc_peak(tmp) <= ACC_BOUND
        for i in range(CHANNELS):
            acc[i] = np.int16(tmp[i])
        for i in range(PSQT_BUCKETS):
            psqt[i] = np.int32(psqt_part[i])
    else:
        psq_buf = N[X_RBADD : X_RBADD + 40]
        bias = S[X_BIAS : X_BIAS + CHANNELS]
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
            bias,
            psqw,
            thrw,
            ppw,
            psqtw,
            psq_buf,
            thr_enum,
            pp_enum,
        )
        assert n_thr <= MAX_ACTIVE_THREATS and n_pp <= MAX_ACTIVE_PP
        assert peak <= ACC_BOUND
    # store into finny slot
    for i in range(12):
        fsig[i] = bb[i]
    facc = S[
        X_FACC + (p * FINNY_FRAMES + f) * CHANNELS : X_FACC
        + (p * FINNY_FRAMES + f) * CHANNELS
        + CHANNELS
    ]
    fpsqp = N[
        X_FPSQP + (p * FINNY_FRAMES + f) * CHANNELS : X_FPSQP
        + (p * FINNY_FRAMES + f) * CHANNELS
        + CHANNELS
    ]
    fpsqt = N[
        X_FPSQT + (p * FINNY_FRAMES + f) * PSQT_BUCKETS : X_FPSQT
        + (p * FINNY_FRAMES + f) * PSQT_BUCKETS
        + PSQT_BUCKETS
    ]
    for i in range(CHANNELS):
        facc[i] = acc[i]
        fpsqp[i] = psq_part[i]
    for i in range(PSQT_BUCKETS):
        fpsqt[i] = psqt[i]
    Y[X_FVALID + p * FINNY_FRAMES + f] = np.uint8(1)


@njit(cache=True)
def n_ev_ensure(ctx, p):
    """Mirror of Stack._ensure: valid-ancestor replay vs refresh chooser."""
    L = ctx[AI]
    N = ctx[A32]
    S = ctx[A16]
    Y = ctx[AU8]
    n = L[X_ST + I_ADEPTH]
    if Y[X_AVALID + n * 2 + p]:
        return
    last_refresh = 0
    for j in range(n, -1, -1):
        if Y[X_AREFRESH + j * 2 + p]:
            last_refresh = j
            break
    anc = -1
    for j in range(n, last_refresh - 1, -1):
        if Y[X_AVALID + j * 2 + p]:
            anc = j
            break
    if anc >= 0:
        span_plies = n - anc
        span_rows = np.int64(0)
        for j in range(anc + 1, n + 1):
            span_rows += np.int64(N[X_TN + j]) + np.int64(N[X_PN + j])
        replay_ns = span_plies * COST_REPLAY_PLY + span_rows * COST_REPLAY_ROW
        f = np.int64(Y[X_AFRAME + n * 2 + p])
        refresh_ns = COST_REFRESH_WARM if Y[X_FVALID + p * FINNY_FRAMES + f] else COST_REFRESH_COLD
        if L[X_ST + I_FORCEREPLAY] or replay_ns <= refresh_ns:
            src = S[X_ACC + (anc * 2 + p) * CHANNELS : X_ACC + (anc * 2 + p) * CHANNELS + CHANNELS]
            dst = S[X_ACC + (n * 2 + p) * CHANNELS : X_ACC + (n * 2 + p) * CHANNELS + CHANNELS]
            for i in range(CHANNELS):
                dst[i] = src[i]
            sps = N[
                X_PSQTA + (anc * 2 + p) * PSQT_BUCKETS : X_PSQTA
                + (anc * 2 + p) * PSQT_BUCKETS
                + PSQT_BUCKETS
            ]
            dps = N[
                X_PSQTA + (n * 2 + p) * PSQT_BUCKETS : X_PSQTA
                + (n * 2 + p) * PSQT_BUCKETS
                + PSQT_BUCKETS
            ]
            for i in range(PSQT_BUCKETS):
                dps[i] = sps[i]
            for k in range(anc + 1, n + 1):
                n_ev_apply_diff(ctx, k, p)
            Y[X_AVALID + n * 2 + p] = np.uint8(1)
            return
    n_ev_refresh(ctx, n, p)
    Y[X_AVALID + n * 2 + p] = np.uint8(1)


@njit(cache=True)
def n_ev_materialize(ctx):
    """Materialize both perspectives at the current depth."""
    n_ev_ensure(ctx, 0)
    n_ev_ensure(ctx, 1)


# ---------------------------------------------------------------------------
# scalar evaluation — mirror of Evaluator.evaluate and simple_eval
# ---------------------------------------------------------------------------


@njit(cache=True)
def n_ev_evaluate(ctx):
    U = ctx[AU]
    L = ctx[AI]
    N = ctx[A32]
    S = ctx[A16]
    B = ctx[A8]
    n_ev_materialize(ctx)
    d = L[X_ST + I_ADEPTH]
    stm = L[X_ST + I_SIDE]
    x = N[X_EVALX : X_EVALX + 512]
    acc_s = S[X_ACC + (d * 2 + stm) * CHANNELS : X_ACC + (d * 2 + stm) * CHANNELS + CHANNELS]
    acc_n = S[
        X_ACC + (d * 2 + (stm ^ 1)) * CHANNELS : X_ACC + (d * 2 + (stm ^ 1)) * CHANNELS + CHANNELS
    ]
    _paired_transform_nb(acc_s, x[:256])
    _paired_transform_nb(acc_n, x[256:512])
    pc = k_pop64(U[X_OCC + 2])
    bucket = pc - 2
    if bucket < 0:
        bucket = np.int64(0)
    bucket = bucket // 4
    if bucket > 7:
        bucket = np.int64(7)
    w1 = B[X_W1 : X_W1 + 8 * 16 * 512].reshape(8, 16, 512)
    w2 = B[X_W2 : X_W2 + 8 * 32 * 32].reshape(8, 32, 32)
    w3 = B[X_W3 : X_W3 + 8 * 4 * 96].reshape(8, 4, 96)
    b1 = N[X_B1 : X_B1 + 8 * 16].reshape(8, 16)
    b2 = N[X_B2 : X_B2 + 8 * 32].reshape(8, 32)
    b3 = N[X_B3 : X_B3 + 8 * 4].reshape(8, 4)
    dot = _head_scalar_nb(x, w1, b1, w2, b2, w3, b3, bucket)
    ps0 = np.int64(N[X_PSQTA + (d * 2 + stm) * PSQT_BUCKETS + bucket])
    ps1 = np.int64(N[X_PSQTA + (d * 2 + (stm ^ 1)) * PSQT_BUCKETS + bucket])
    dterm = ps0 - ps1
    if dterm < 0:
        dterm = -((-dterm) // 2)
    else:
        dterm = dterm // 2
    pterm = np.int64(N[X_PSQTB + bucket]) + dterm
    v = (dot + pterm) * L[X_ST + I_SCALENUM] >> L[X_ST + I_SCALESHIFT]
    nb = L[X_ST + I_NEURALBOUND]
    if v > nb:
        v = nb
    elif v < -nb:
        v = -nb
    return v


@njit(cache=True)
def n_simple_eval(ctx):
    """Mirror of search.simple_eval — PVAL[t] + PSQT[t][s ^ (56 if black)],
    +10 tempo for the side to move."""
    B = ctx[A8]
    score = np.int64(0)
    for sq in range(64):
        piece = np.int64(B[X_MB + sq])
        if piece < 0:
            continue
        t = piece % 6
        s = sq if piece < 6 else (sq ^ 56)
        v = np.int64(_PVAL[t]) + np.int64(_psqt_tab[t, s])
        score += v if piece < 6 else -v
    if ctx[AI][X_ST + I_SIDE] == 0:
        return score + 10
    return -score + 10


_PVAL = np.asarray((100, 320, 330, 500, 950, 0), dtype=np.int64)


@njit(cache=True)
def n_static_eval(ctx, ply):
    """`_static_eval` mirror — returns corrected eval in (-MATE, MATE)."""
    L = ctx[AI]
    if L[X_ST + I_EVALKIND] == 1:
        raw = n_ev_evaluate(ctx)
    else:
        raw = n_simple_eval(ctx)
    clamp = np.int64(ctx[A32][X_PARAMS + P_eval_clamp])
    if raw > clamp:
        raw = np.int64(clamp)
    elif raw < -clamp:
        raw = np.int64(-clamp)
    return raw

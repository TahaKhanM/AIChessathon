"""Numba runtime for the exported RXF1 model.

Decodes the packed payload once into runtime dtypes (int16/int8/int32) and
evaluates with the deployed integer arithmetic: accumulator rows summed in
int32 (the stored accumulator is int16; intermediates widen so no silent
wrap is possible — the proven 30,048 bound is asserted by ``evaluate``),
widened int32 paired products, int64 rescale.  This is the reference
runtime the parity gate exercises; the engine hot path (W05) must agree
with it on golden vectors.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from training.export import read_export
from training.feature_spec import SPEC
from training.features import Encoded


@njit(cache=True)
def _unpack_packed(buf: np.ndarray, count: int, bits: int) -> np.ndarray:
    out = np.empty(count, dtype=np.int64)
    buffer = np.uint64(0)
    available = 0
    offset = 0
    mask = np.uint64((1 << bits) - 1)
    sign = np.uint64(1 << (bits - 1))
    for i in range(count):
        while available < bits:
            buffer |= np.uint64(buf[offset]) << np.uint64(available)
            available += 8
            offset += 1
        value = buffer & mask
        buffer >>= np.uint64(bits)
        available -= bits
        v = np.int64(value)
        if value & sign:
            v -= np.int64(1) << np.int64(bits)
        out[i] = v
    if buffer != 0:
        raise ValueError("nonzero terminal padding")
    return out


@njit(cache=True)
def evaluate_nb(
    psq_w: np.ndarray,
    thr_w: np.ndarray,
    pp_w: np.ndarray,
    ft_b: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
    psqt_w: np.ndarray,
    psqt_b: np.ndarray,
    psq0: np.ndarray,
    thr0: np.ndarray,
    pp0: np.ndarray,
    psq1: np.ndarray,
    thr1: np.ndarray,
    pp1: np.ndarray,
    head: int,
    channels: int,
    hc: int,
    prod_shift: int,
    hid_shift: int,
    sq_shift: int,
) -> np.ndarray:
    """Exact deployed-arithmetic evaluation.

    Returns int64 array [scalar, W, D, L, max_abs_acc].
    Accumulator arithmetic runs in int32 (the storage dtype is int16 per
    contract; intermediates widen so a malformed row list cannot wrap —
    the wrapper asserts the proven 30,048 bound on res[4]).
    """
    acc = np.zeros((2, channels), dtype=np.int32)
    for c in range(channels):
        acc[0, c] = np.int32(ft_b[c])
        acc[1, c] = np.int32(ft_b[c])
    for r in psq0:
        for c in range(channels):
            acc[0, c] += np.int32(psq_w[r, c])
    for r in psq1:
        for c in range(channels):
            acc[1, c] += np.int32(psq_w[r, c])
    for r in thr0:
        for c in range(channels):
            acc[0, c] += np.int32(thr_w[r, c])
    for r in thr1:
        for c in range(channels):
            acc[1, c] += np.int32(thr_w[r, c])
    for r in pp0:
        for c in range(channels):
            acc[0, c] += np.int32(pp_w[r, c])
    for r in pp1:
        for c in range(channels):
            acc[1, c] += np.int32(pp_w[r, c])

    max_abs = 0
    inp = np.empty(2 * hc, dtype=np.int32)
    for p in range(2):
        for i in range(hc):
            a = np.int32(acc[p, i])
            b = np.int32(acc[p, hc + i])
            if a < 0:
                a = 0
            if a > 255:
                a = 255
            if b < 0:
                b = 0
            if b > 255:
                b = 255
            inp[p * hc + i] = (a * b) >> prod_shift
            v = abs(int(acc[p, i]))
            if v > max_abs:
                max_abs = v
            v = abs(int(acc[p, hc + i]))
            if v > max_abs:
                max_abs = v

    # head
    n1 = w1.shape[1]
    n2 = w2.shape[1]
    o1 = np.empty(2 * n1, dtype=np.int32)
    z1 = np.empty(n1, dtype=np.int32)
    for i in range(n1):
        z = np.int64(b1[head, i])
        for j in range(2 * hc):
            z += np.int64(w1[head, i, j]) * np.int64(inp[j])
        z1[i] = np.int32(z)
    for i in range(n1):
        # deployed activation: square the UNCLIPPED shifted affine
        # (engine _head_scalar_nb / features._hidden_activation)
        c = z1[i] >> hid_shift
        sq64 = (np.int64(c) * np.int64(c)) >> sq_shift
        sq = np.int32(127 if sq64 > 127 else sq64)
        if c < 0:
            c = 0
        elif c > 127:
            c = 127
        o1[i] = c
        o1[n1 + i] = sq
    o2 = np.empty(2 * n2, dtype=np.int32)
    for i in range(n2):
        z = np.int64(b2[head, i])
        for j in range(2 * n1):
            z += np.int64(w2[head, i, j]) * np.int64(o1[j])
        c = np.int32(z) >> hid_shift
        sq64 = (np.int64(c) * np.int64(c)) >> sq_shift
        sq = np.int32(127 if sq64 > 127 else sq64)
        if c < 0:
            c = 0
        elif c > 127:
            c = 127
        o2[i] = c
        o2[n2 + i] = sq
    o = np.empty(2 * n1 + 2 * n2, dtype=np.int32)
    for i in range(2 * n1):
        o[i] = o1[i]
    for i in range(2 * n2):
        o[2 * n1 + i] = o2[i]
    res = np.empty(5, dtype=np.int64)
    for k in range(4):
        z = np.int64(b3[head, k])
        for j in range(o.shape[0]):
            z += np.int64(w3[head, k, j]) * np.int64(o[j])
        res[k] = z
    ps0 = np.int64(0)
    for r in psq0:
        ps0 += np.int64(psqt_w[r, head])
    ps1 = np.int64(0)
    for r in psq1:
        ps1 += np.int64(psqt_w[r, head])
    d = ps0 - ps1
    half = -((-d) // 2) if d < 0 else d // 2
    res[0] += np.int64(psqt_b[head]) + half
    res[4] = max_abs
    return res


class NumbaRuntime:
    """Loads an RXF1 export and evaluates Encoded features."""

    def __init__(self, path: str):
        self.header, self.model = read_export(path)
        s = SPEC
        self.channels = s.channels
        self.hc = s.half_channels
        self.prod_shift = s.numeric.paired_product_shift
        self.hid_shift = s.numeric.hidden_affine_shift
        self.sq_shift = s.numeric.square_activation_shift
        # warm the jit on a zero feature set
        empty = np.zeros(0, dtype=np.int32)
        evaluate_nb(
            self.model.psq_w,
            self.model.thr_w,
            self.model.pp_w,
            self.model.ft_b,
            self.model.w1,
            self.model.b1,
            self.model.w2,
            self.model.b2,
            self.model.w3,
            self.model.b3,
            self.model.psqt_w,
            self.model.psqt_b,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            0,
            self.channels,
            self.hc,
            self.prod_shift,
            self.hid_shift,
            self.sq_shift,
        )

    def evaluate(self, enc: Encoded) -> tuple[int, tuple[int, int, int], int]:
        m = self.model
        res = evaluate_nb(
            m.psq_w,
            m.thr_w,
            m.pp_w,
            m.ft_b,
            m.w1,
            m.b1,
            m.w2,
            m.b2,
            m.w3,
            m.b3,
            m.psqt_w,
            m.psqt_b,
            enc.psq[0],
            enc.threats[0],
            enc.pawn_pairs[0],
            enc.psq[1],
            enc.threats[1],
            enc.pawn_pairs[1],
            enc.head,
            self.channels,
            self.hc,
            self.prod_shift,
            self.hid_shift,
            self.sq_shift,
        )
        if res[4] > SPEC.numeric.accumulator_proven_abs_bound:
            raise ValueError(f"|acc| {res[4]} > {SPEC.numeric.accumulator_proven_abs_bound}")
        return int(res[0]), (int(res[1]), int(res[2]), int(res[3])), int(res[4])

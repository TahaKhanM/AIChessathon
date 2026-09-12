"""Scalar integer evaluator: the exact-arithmetic reference hop.

Operates on already-integer weight arrays (what the export stores), in
Python int64 arithmetic with explicit floor shifts.  This is an independent
implementation of the same contract as ``model.F512Model.forward`` — the
parity gate requires EXACT agreement between them — and the same arithmetic
as ``numba_rt.evaluate`` in deployed dtypes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from training.feature_spec import SPEC, FeatureSpec
from training.features import Encoded


@dataclass
class IntegerModel:
    """Folded integer weights; the exact content of an export."""

    psq_w: np.ndarray  # int16 [9216, 512]
    thr_w: np.ndarray  # int8  [59808, 512]
    pp_w: np.ndarray  # int8  [1488, 512]
    ft_b: np.ndarray  # int16 [512]
    w1: np.ndarray  # int8  [8, 16, 512]
    b1: np.ndarray  # int32 [8, 16]
    w2: np.ndarray  # int8  [8, 32, 32]
    b2: np.ndarray  # int32 [8, 32]
    w3: np.ndarray  # int8  [8, 4, 96]
    b3: np.ndarray  # int32 [8, 4]
    psqt_w: np.ndarray  # int16 [9216, 8]
    psqt_b: np.ndarray  # int32 [8]

    def array_dict(self) -> dict[str, np.ndarray]:
        return {
            "psq_w": self.psq_w,
            "thr_w": self.thr_w,
            "pp_w": self.pp_w,
            "ft_b": self.ft_b,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "w3": self.w3,
            "b3": self.b3,
            "psqt_w": self.psqt_w,
            "psqt_b": self.psqt_b,
        }


def evaluate_int(
    model: IntegerModel, enc: Encoded, head: int, spec: FeatureSpec = SPEC
) -> tuple[int, tuple[int, int, int], int]:
    """Return (scalar, (W,D,L) logits, max|acc|) in exact integer arithmetic."""
    hc = spec.half_channels
    acc = np.zeros((2, spec.channels), dtype=np.int64)
    acc += model.ft_b.astype(np.int64)
    for p in range(2):
        if len(enc.psq[p]):
            acc[p] += model.psq_w[enc.psq[p]].astype(np.int64).sum(0)
        if len(enc.threats[p]):
            acc[p] += model.thr_w[enc.threats[p]].astype(np.int64).sum(0)
        if len(enc.pawn_pairs[p]):
            acc[p] += model.pp_w[enc.pawn_pairs[p]].astype(np.int64).sum(0)
    max_acc = int(np.abs(acc).max())

    a = np.clip(acc[:, :hc], 0, 255)
    bb = np.clip(acc[:, hc:], 0, 255)
    act = (a * bb) >> spec.numeric.paired_product_shift  # (2, hc)
    inp = np.concatenate([act[0], act[1]]).astype(np.int64)

    w1 = model.w1[head].astype(np.int64)
    w2 = model.w2[head].astype(np.int64)
    w3 = model.w3[head].astype(np.int64)
    z1 = w1 @ inp + model.b1[head].astype(np.int64)
    x1 = z1 >> spec.numeric.hidden_affine_shift  # unclipped shifted
    c1 = np.clip(x1, 0, 127)
    s1 = np.clip((x1 * x1) >> spec.numeric.square_activation_shift, 0, 127)
    o1 = np.concatenate([c1, s1])
    z2 = w2 @ o1 + model.b2[head].astype(np.int64)
    x2 = z2 >> spec.numeric.hidden_affine_shift
    c2 = np.clip(x2, 0, 127)
    s2 = np.clip((x2 * x2) >> spec.numeric.square_activation_shift, 0, 127)
    o2 = np.concatenate([c2, s2])
    o = np.concatenate([o1, o2])
    out = w3 @ o + model.b3[head].astype(np.int64)

    # Deployed PSQT (engine/evaluate._psqt_term): bias + trunc((p0-p1)/2).
    p0 = int(model.psqt_w[enc.psq[0], head].astype(np.int64).sum()) if len(enc.psq[0]) else 0
    p1 = int(model.psqt_w[enc.psq[1], head].astype(np.int64).sum()) if len(enc.psq[1]) else 0
    d = p0 - p1
    half = -((-d) // 2) if d < 0 else d // 2
    psqt = int(model.psqt_b[head]) + half
    scalar = int(out[0]) + psqt
    return scalar, (int(out[1]), int(out[2]), int(out[3])), max_acc

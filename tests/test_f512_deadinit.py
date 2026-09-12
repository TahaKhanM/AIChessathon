"""Paired-product liveness at random init (F512-TRAIN2 dead-start).

TRAIN2 measured inp nonzero frac = 0 at step 0: every paired product was
dead, the head saw the zero vector, and a constant output was the MLE.
This test encodes real positions (not synthetic index noise) and requires
a healthy fraction of head inputs to be live, with accumulators inside the
int16 contract bound and not saturating the [0,255] clip.
"""

from __future__ import annotations

import chess
import numpy as np

from training.feature_spec import SPEC
from training.features import FeatureEncoder
from training.model import Batch, F512Model

# kiwipete, startpos, a sparse endgame, a queen-loss regression position
_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r1bqk2r/2ppbnpp/p1n5/8/1p1p4/1B6/PPP2PPP/RNBQR1K1 w kq - 0 12",
]


def _batch(encs) -> Batch:
    n = len(encs)
    return Batch(
        enc=encs,
        u_targets=[[] for _ in range(n)],
        u_bounds=[[] for _ in range(n)],
        wdl_targets=[[] for _ in range(n)],
        result_targets=[None] * n,
        rank_pairs=[],
        head=np.asarray([e.head for e in encs], np.int64),
    )


def test_paired_products_live_at_init_on_real_positions():
    enc = FeatureEncoder()
    encs = [enc.encode(chess.Board(fen)) for fen in _FENS]
    model = F512Model(SPEC, seed=0)
    fwd = model.forward(_batch(encs), need_grad=False)
    acc = fwd["acc"]
    inp = fwd["inp"]
    ops = acc.ravel()

    inp_nz = float((inp != 0).mean())
    clip0 = float((ops <= 0).mean())
    clip255 = float((ops >= 255).mean())
    max_abs = float(np.abs(acc).max())

    assert inp_nz >= 0.5, (
        f"dead paired products at init: inp nonzero frac={inp_nz:.4f} "
        f"(TRAIN2 failure mode; head sees the zero vector)"
    )
    assert clip0 < 0.1, f"too many operands at the clip floor: {clip0:.3f}"
    assert clip255 < 0.05, f"too many operands saturating 255: {clip255:.3f}"
    assert max_abs <= SPEC.numeric.accumulator_proven_abs_bound, (
        f"|acc| {max_abs} exceeds int16 bound {SPEC.numeric.accumulator_proven_abs_bound}"
    )


def test_ft_bias_starts_in_live_band():
    model = F512Model(SPEC, seed=0)
    from training.model import fq

    b = fq(model.params["ft_b"], SPEC.bias_limit)
    assert float(b.min()) >= 64, f"ft_b min {b.min()} is below the live band"
    assert float(b.max()) <= 192, f"ft_b max {b.max()} crowds the 255 clip"
    assert float(np.abs(b).max()) <= SPEC.bias_limit

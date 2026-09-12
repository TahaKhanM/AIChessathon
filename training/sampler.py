"""Deterministic source-mixture sampler with serializable state.

Each epoch is a seeded permutation over weighted record copies: a record's
epoch weight is ``mixture_weight[lineage_family] / family_count`` so family
weights behave as *source mixture* shares, not per-record inflation.  State
(epoch, offset, rng bit-generator state) is JSON-serializable and stored in
checkpoints so a resumed job continues the exact sequence.
"""

from __future__ import annotations

import numpy as np


class MixtureSampler:
    def __init__(
        self,
        record_families: list[str],
        mixture_weights: dict[str, float] | None = None,
        *,
        seed: int = 0,
        epoch: int = 0,
        offset: int = 0,
        rng_state: dict | None = None,
    ) -> None:
        self.families = record_families
        fams = sorted(set(record_families))
        mw = mixture_weights or {f: 1.0 for f in fams}
        counts = {f: record_families.count(f) for f in fams}
        w = np.array([mw.get(f, 0.0) / counts[f] for f in record_families], dtype=np.float64)
        if not np.isfinite(w).all() or (w <= 0).all():
            raise ValueError("mixture weights leave no positive-mass records")
        self.weights = w
        self.epoch = epoch
        self.offset = offset
        self.rng = np.random.default_rng(seed)
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
        else:
            # advance the rng to the declared epoch boundary so a restored
            # logical epoch reproduces the same permutations
            for _ in range(epoch):
                self._permutation()
        self._order: np.ndarray | None = None
        self._ensure_order()

    def _permutation(self) -> np.ndarray:
        n = len(self.families)
        keys = self.rng.random(n) / np.maximum(self.weights, 1e-12)
        return np.argsort(-keys, kind="stable")  # weighted reservoir order

    def _ensure_order(self) -> None:
        if self._order is None:
            # capture the pre-draw rng state so a checkpoint restores the
            # SAME order for this epoch
            self.epoch_rng_state = self.rng.bit_generator.state
            self._order = self._permutation()

    def next_batch(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.int64)
        i = 0
        while i < n:
            take = min(n - i, len(self._order) - self.offset)
            out[i : i + take] = self._order[self.offset : self.offset + take]
            self.offset += take
            i += take
            if self.offset >= len(self._order):
                self.epoch += 1
                self.offset = 0
                self._order = self._permutation()
        return out

    def get_state(self) -> dict:
        return {
            "epoch": self.epoch,
            "offset": self.offset,
            # rng state at the moment the current epoch's order was drawn
            "epoch_rng_state": self.epoch_rng_state,
        }

    def set_state(self, s: dict) -> None:
        self.epoch = s["epoch"]
        self.offset = s["offset"]
        self.rng.bit_generator.state = s["epoch_rng_state"]
        self._order = None
        self._ensure_order()

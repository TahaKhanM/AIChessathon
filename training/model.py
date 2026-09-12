"""Reference trainer for F512-EF-K12-16/32 with exact-arithmetic fake quant.

Design: every trainable tensor is stored float32, but the forward pass uses
``fq(w) = clip(floor(w + 1/2), -L, +L)`` on the *folded* coefficients
(base + factorizer row), so the effective weights are integer-valued and
bounded for the whole run — quantization is present from step 0, not bolted
on (spec 10.2 Stage D requirement, honoured early).  All activation
arithmetic uses the deployed floor-shift semantics on integer-valued
operands; feature/paired-product math is exact in float32 (< 2^24) and the
head runs in float64 because the deployed square activation takes the
*unclipped* shifted affine (|x| up to ~33M, x*x needs > float32 mantissa),
so the float model and the integer model agree *exactly* by construction.

Gradients use straight-through estimators: round/clip on weights pass
gradients, activation clips mask gradients outside the active range, and
the floor shifts keep their true linearized scale (1/2^s), so the training
objective is the linearization of the actual deployed function.

This reference implementation anchors export/runtime parity. It is retained
for numerical verification; no new release-model training is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from training.features import Encoded
from training.feature_spec import SPEC, FeatureSpec

CLIP = np.clip

# Mid-band of the deployed FT clip [0, 255]. Paired products are
# floor(a*b / 512); both operands must sit well above ~23 or the product
# is identically 0 and the head sees the zero vector. N(0, 4) bias plus
# O(1) feature rows land acc ~ N(0, 15) — half clip to 0, the rest
# multiply to < 512. This constant is the standard NNUE remedy.
FT_BIAS_INIT = 128.0

# --------------------------------------------------------------------------
# QAT learning rate — DERIVED from the quantization step (FR1-F5, FR2-B1).
#
# The fq grid step is one integer cell: QAT_GRID_STEP = 1.0.
#
# AdamW's update is lr * (m̂ / (√v̂ + eps)): the direction m̂/√v̂ is bounded in
# [-1, 1] and is INVARIANT to scaling all gradients by a constant — so the
# summed-over-batch gradient does NOT multiply the step by B, and no 1/B
# term belongs in the formula.  (The per-record sum matters upstream for
# the loss, not for the Adam step size.)  What remains is the grid:
#
#   * fake-quant is live every step — a persistent gradient direction must
#     move a weight by ~Δ_q within O(10) coherent steps or rounding pins it;
#   * stability — one step must stay well under a cell or the weight
#     oscillates across rounding boundaries and channels saturate.
#
# Both constraints land on the same scale: lr = Δ_q / S for S ≈ 10 steps
# to cross one cell.  The measured band (learn_probe, batch 512) confirms
# the derivation: lr ∈ [0.05, 0.15] learns, 0.4 saturates channels, and
# 4e-4 (the pre-fix default) was ~250x under the grid scale — dead.
QAT_GRID_STEP = 1.0  # the fq integer grid cell
QAT_STEPS_PER_CELL = 10.0  # coherent-gradient steps to cross one cell
LR_QAT = QAT_GRID_STEP / QAT_STEPS_PER_CELL  # = 0.1


def material_psqt_table(spec: FeatureSpec, pawn_cp: float = 100.0) -> np.ndarray:
    """stm-relative 1/3/3/5/9 PSQT. Not published NNUE weights.

    Deployed skip is bias + trunc((p0-p1)/2). +V on relcolour 0 and -V on
    relcolour 1 makes that equal stm material. ``pawn_cp`` is the deployed
    centipawn target for one extra pawn (``raw * cp_num >> cp_shift``).
    """
    pawn_raw = int(round((pawn_cp * (1 << spec.numeric.cp_shift)) / max(1, spec.numeric.cp_num)))
    values = (pawn_raw, 3 * pawn_raw, 3 * pawn_raw, 5 * pawn_raw, 9 * pawn_raw, 0)
    n_heads = spec.head.material_stacks
    w = np.zeros((spec.psqt_rows, n_heads), np.float32)
    for bucket in range(spec.king_buckets):
        for rel in (0, 1):
            sign = 1.0 if rel == 0 else -1.0
            for pt, val in enumerate(values):
                if val == 0:
                    continue
                base = ((bucket * 2 + rel) * 6 + pt) * 64
                w[base : base + 64, :] = sign * val
    return w


def fq(w: np.ndarray, limit: int) -> np.ndarray:
    """Fake-quant to the bounded integer grid (floor(x+1/2) rounding)."""
    return CLIP(np.floor(w + 0.5), -limit, limit)


def default_params(
    spec: FeatureSpec, rng: np.random.Generator, factorized: bool = True
) -> dict[str, np.ndarray]:
    """Random init whose fake-quant forward is live on real positions.

    Feature rows stay N(0, 1): fake-quant from step 0 maps |w|<0.5 to 0,
    so 1/sqrt(fan-in) scaling would zero the tables. Typical active counts
    (≤32 PSQ, ~25-40 threats, ~30 pawn pairs per perspective) give an
    accumulator std of ~15, which sits inside (0, 255) once the bias is
    mid-band. Head stays N(0, 2). PSQT is the 1/3/3/5/9 material prior
    (stream still consumes the old N(0, 2) draw so other tensors match).
    """
    del factorized  # tables are always allocated; fold is a forward switch
    C = spec.channels
    H = spec.head
    p: dict[str, np.ndarray] = {}
    p["psq_w"] = rng.normal(0, 1.0, (spec.psq.rows, C)).astype(np.float32)
    p["thr_w"] = rng.normal(0, 1.0, (spec.threats.rows, C)).astype(np.float32)
    p["pp_w"] = rng.normal(0, 1.0, (spec.pawn_pairs.rows, C)).astype(np.float32)
    p["psq_fac"] = rng.normal(0, 1.0, (spec.psq.factorizer_rows, C)).astype(np.float32)
    p["thr_fac"] = rng.normal(0, 1.0, (spec.threats.factorizer_rows, C)).astype(np.float32)
    p["pp_fac"] = rng.normal(0, 1.0, (spec.pawn_pairs.factorizer_rows, C)).astype(np.float32)
    p["ft_b"] = np.full(C, FT_BIAS_INIT, np.float32)
    p["w1"] = rng.normal(0, 2.0, (H.material_stacks, H.first_affine_outputs, H.input_count)).astype(
        np.float32
    )
    p["b1"] = np.zeros((H.material_stacks, H.first_affine_outputs), np.float32)
    p["w2"] = rng.normal(
        0, 2.0, (H.material_stacks, H.second_affine_outputs, H.first_activation_concat)
    ).astype(np.float32)
    p["b2"] = np.zeros((H.material_stacks, H.second_affine_outputs), np.float32)
    p["w3"] = rng.normal(0, 2.0, (H.material_stacks, len(H.outputs), H.skip_concat_outputs)).astype(
        np.float32
    )
    p["b3"] = np.zeros((H.material_stacks, len(H.outputs)), np.float32)
    _ = rng.normal(0, 2.0, (spec.psqt_rows, H.material_stacks))  # consume stream
    p["psqt_w"] = material_psqt_table(spec)
    p["psqt_b"] = np.zeros(H.material_stacks, np.float32)
    return p


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class LossWeights:
    value: float = 1.0
    wdl: float = 1.0
    result: float = 1.0
    rank: float = 0.0  # action-regret; off unless action bank data
    consistency: float = 0.1
    quant_reg: float = 0.0  # pull-to-grid penalty; 0 = pure STE
    weight_decay: float = 0.0
    bound_softplus_beta: float = 1.0
    rank_margin: float = 0.02
    rank_temperature: float = 0.05


@dataclass
class Batch:
    """Encoded features + resolved targets for one minibatch."""

    enc: list[Encoded]
    u_targets: list[list[float]]  # point value targets (stm POV)
    u_bounds: list[list[tuple[float | None, float | None]]]
    wdl_targets: list[list[list[float]]]
    result_targets: list[float | None]
    rank_pairs: list[tuple[int, int, float]]  # (i, j, teacher u_i - u_j)
    head: np.ndarray


class F512Model:
    """Fake-quant float32 model; effective weights are bounded integers."""

    PARAM_LIMITS = {
        "psq_w": 255,
        "thr_w": 63,
        "pp_w": 31,
        "psq_fac": 255,
        "thr_fac": 63,
        "pp_fac": 31,
        "ft_b": 2040,
        "w1": 127,
        "w2": 127,
        "w3": 127,
        "b1": 2**31 - 1,
        "b2": 2**31 - 1,
        "b3": 2**31 - 1,
        "psqt_w": 32767,
        "psqt_b": 2**31 - 1,
    }

    def __init__(self, spec: FeatureSpec = SPEC, seed: int = 0, factorized: bool = True):
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.factorized = factorized
        self.params = default_params(spec, self.rng, factorized)
        # AdamW state
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.step_count = 0

    # -- forward ---------------------------------------------------------------
    def _folded(
        self, base: np.ndarray, fac: np.ndarray, fac_rows: np.ndarray, limit: int
    ) -> np.ndarray:
        if self.factorized:
            return fq(base + fac[fac_rows], limit)
        return fq(base, limit)

    def forward(self, batch: Batch, need_grad: bool = False) -> dict:
        """Returns per-sample scalar (int-valued float) + wdl logits + caches."""
        spec = self.spec
        n = len(batch.enc)
        C = spec.channels
        hc = spec.half_channels
        acc = np.zeros((n, 2, C), dtype=np.float32)
        acc += fq(self.params["ft_b"], spec.bias_limit).astype(np.float32)[None, None, :]
        psqt = np.zeros(n, dtype=np.float64)
        psqt += fq(self.params["psqt_b"], self.PARAM_LIMITS["psqt_b"])[batch.head]

        # flat per-space index arrays: per-record flat() concatenates both
        # perspectives; segments are (i, pside) blocks in order.
        row_index: dict[str, dict] = {}
        psqt_rows_l, psqt_head_l, psqt_src_l, psqt_sign_l = [], [], [], []
        for i, e in enumerate(batch.enc):
            for key, space in (("psq", "psq"), ("thr", "threats"), ("pp", "pawn_pairs")):
                rows_cat, fac_cat, seg = e.flat(space)
                ri = row_index.setdefault(key, {"rows": [], "frows": [], "lens": []})
                ri["rows"].append(rows_cat)
                ri["frows"].append(fac_cat)
                ri["lens"].append((seg[0], seg[1] - seg[0]))
            if len(e.psq[0]) or len(e.psq[1]):
                for p, sign in ((0, 1), (1, -1)):
                    if len(e.psq[p]):
                        psqt_rows_l.append(e.psq[p])
                        psqt_head_l.append(np.full(len(e.psq[p]), batch.head[i], np.int64))
                        psqt_src_l.append(np.full(len(e.psq[p]), i, np.int64))
                        psqt_sign_l.append(np.full(len(e.psq[p]), sign, np.int64))
        accf = acc.reshape(n * 2, C)
        lim = {
            "psq": self.PARAM_LIMITS["psq_w"],
            "thr": self.PARAM_LIMITS["thr_w"],
            "pp": self.PARAM_LIMITS["pp_w"],
        }
        for key, wk, fk in (
            ("psq", "psq_w", "psq_fac"),
            ("thr", "thr_w", "thr_fac"),
            ("pp", "pp_w", "pp_fac"),
        ):
            ri = row_index[key]
            rows_cat = np.concatenate(ri["rows"]) if ri["rows"] else np.zeros(0, np.int64)
            fac_cat = np.concatenate(ri["frows"]) if ri["frows"] else np.zeros(0, np.int64)
            seg_lens = np.asarray([x for lp in ri["lens"] for x in lp], np.int64)
            seg_starts = np.zeros(n * 2, np.int64)
            seg_starts[1:] = np.cumsum(seg_lens)[:-1]
            src = np.repeat(np.arange(n * 2), seg_lens)
            ri.update(rows=rows_cat, frows=fac_cat, src=src)
            if len(rows_cat):
                eff = fq(
                    self.params[wk][rows_cat]
                    + (self.params[fk][fac_cat] if self.factorized else 0),
                    lim[key],
                )
                # segmented sum over (i,p) blocks; skip empty segments so
                # reduceat boundaries stay valid
                nonempty = seg_lens > 0
                sums = np.add.reduceat(eff, seg_starts[nonempty])
                accf[nonempty] += sums
        row_index["psqt"] = {
            "rows": np.concatenate(psqt_rows_l) if psqt_rows_l else np.zeros(0, np.int64),
            "head": np.concatenate(psqt_head_l) if psqt_head_l else np.zeros(0, np.int64),
            "src": np.concatenate(psqt_src_l) if psqt_src_l else np.zeros(0, np.int64),
            "sign": np.concatenate(psqt_sign_l) if psqt_sign_l else np.zeros(0, np.int64),
        }
        rp0 = row_index["psqt"]
        pair = np.zeros(n, dtype=np.float64)
        if len(rp0["rows"]):
            contrib = (
                fq(self.params["psqt_w"][rp0["rows"]], self.PARAM_LIMITS["psqt_w"])[
                    np.arange(len(rp0["rows"])), rp0["head"]
                ].astype(np.float64)
                * rp0["sign"]
            )
            np.add.at(pair, rp0["src"], contrib)
        psqt = psqt + np.trunc(pair * 0.5)

        # paired clipped-product activation
        x = acc[:, :, :hc]
        y = acc[:, :, hc:]
        a = CLIP(x, 0, 255)
        bb = CLIP(y, 0, 255)
        act = np.floor(a * bb / (1 << spec.numeric.paired_product_shift))
        inp = np.concatenate([act[:, 0], act[:, 1]], axis=1)  # (n,512) stm first

        # 8 material heads; process per-head groups.  Head math runs in
        # float64: the deployed square term uses the UNCLIPPED shifted
        # affine (x = z>>6, |x| can reach ~33M), so x*x needs more than the
        # float32 mantissa to stay exact.  All values are integer-valued so
        # float64 is exact for |v| < 2^53.
        hs_ = 1 << spec.numeric.hidden_affine_shift
        sqs_ = 1 << spec.numeric.square_activation_shift
        out = np.zeros((n, 4), dtype=np.float64)
        caches: dict[int, dict] = {}
        for h in np.unique(batch.head):
            idx = np.where(batch.head == h)[0]
            w1 = fq(self.params["w1"][h], 127).astype(np.float64)
            w2 = fq(self.params["w2"][h], 127).astype(np.float64)
            w3 = fq(self.params["w3"][h], 127).astype(np.float64)
            b1 = fq(self.params["b1"][h], self.PARAM_LIMITS["b1"]).astype(np.float64)
            b2 = fq(self.params["b2"][h], self.PARAM_LIMITS["b2"]).astype(np.float64)
            b3 = fq(self.params["b3"][h], self.PARAM_LIMITS["b3"]).astype(np.float64)
            z1 = inp[idx].astype(np.float64) @ w1.T + b1
            x1 = np.floor(z1 / hs_)  # unclipped shifted affine
            c1 = CLIP(x1, 0, 127)
            v1 = np.floor(x1 * x1 / sqs_)  # square of UNCLIPPED x
            s1 = CLIP(v1, 0, 127)
            o1 = np.concatenate([c1, s1], axis=1)
            z2 = o1 @ w2.T + b2
            x2 = np.floor(z2 / hs_)
            c2 = CLIP(x2, 0, 127)
            v2 = np.floor(x2 * x2 / sqs_)
            s2 = CLIP(v2, 0, 127)
            o2 = np.concatenate([c2, s2], axis=1)
            o = np.concatenate([o1, o2], axis=1)
            out[idx] = o @ w3.T + b3
            if need_grad:
                caches[h] = dict(
                    idx=idx,
                    z1=z1,
                    x1=x1,
                    v1=v1,
                    o1=o1,
                    z2=z2,
                    x2=x2,
                    v2=v2,
                    o=o,
                    w1=w1,
                    w2=w2,
                    w3=w3,
                )
        scalar = out[:, 0] + psqt
        res = {
            "scalar": scalar,
            "wdl_logits": out[:, 1:4],
            "acc": acc,
            "inp": inp,
            "caches": caches,
            "act": act,
            "a": a,
            "bb": bb,
            "row_index": row_index,
            "x": x,
            "y": y,
        }
        return res

    # -- losses -----------------------------------------------------------------
    def losses(self, fwd: dict, batch: Batch, lw: LossWeights) -> tuple[float, dict]:
        """Compute total loss + d(scalar)/d(wdl_logits) upstream grads."""
        n = len(batch.enc)
        spec = self.spec
        scale = 2.0**-spec.numeric.scalar_u_log2_divisor
        scalar = fwd["scalar"]
        u_pred = _sigmoid(scalar * scale)
        logits = fwd["wdl_logits"] * scale
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        probs = e / e.sum(axis=1, keepdims=True)
        u_wdl = probs[:, 0] + 0.5 * probs[:, 1]

        dscalar = np.zeros(n)
        dlogits = np.zeros((n, 3))
        # spec 9: every loss term needs a GENUINE target.  Consistency
        # couples the scalar and WDL heads — it is only legitimate where a
        # real WDL triple supervises that head; on scalar-only rows it
        # would fabricate WDL signal from the scalar itself.
        term_counts = {"value": 0, "bound": 0, "wdl": 0, "result": 0, "consistency": 0, "rank": 0}
        term_loss = {
            "value": 0.0,
            "bound": 0.0,
            "wdl": 0.0,
            "result": 0.0,
            "consistency": 0.0,
            "rank": 0.0,
        }

        for i in range(n):
            # point value targets (calibrated value); mean within record
            if batch.u_targets[i]:
                grads, ls = [], []
                for u_t in batch.u_targets[i]:
                    ls.append(
                        -(
                            u_t * math.log(u_pred[i] + 1e-12)
                            + (1 - u_t) * math.log(1 - u_pred[i] + 1e-12)
                        )
                    )
                    grads.append(scale * (u_pred[i] - u_t))
                term_loss["value"] += float(np.mean(ls))
                dscalar[i] += float(np.mean(grads)) * lw.value
                term_counts["value"] += 1
            # bound targets: one-sided softplus in logit space
            for lo, hi in batch.u_bounds[i]:
                z = scalar[i] * scale
                g = 0.0
                ls = 0.0
                beta = lw.bound_softplus_beta
                if lo is not None:
                    z_lo = math.log(lo + 1e-9) - math.log(1 - lo + 1e-9)
                    ls += beta * math.log1p(math.exp((z_lo - z) / beta))
                    g -= _sigmoid(np.float64((z_lo - z) / beta)) * scale
                if hi is not None:
                    z_hi = math.log(hi + 1e-9) - math.log(1 - hi + 1e-9)
                    ls += beta * math.log1p(math.exp((z - z_hi) / beta))
                    g += _sigmoid(np.float64((z - z_hi) / beta)) * scale
                term_loss["bound"] += ls
                dscalar[i] += g * lw.value
                term_counts["bound"] += 1
            for w in batch.wdl_targets[i]:
                p = probs[i]
                term_loss["wdl"] += float(
                    -(
                        w[0] * math.log(p[0] + 1e-12)
                        + w[1] * math.log(p[1] + 1e-12)
                        + w[2] * math.log(p[2] + 1e-12)
                    )
                )
                dlogits[i] += (p - np.asarray(w)) * scale * lw.wdl
                term_counts["wdl"] += 1
            if i < len(batch.result_targets) and batch.result_targets[i] is not None:
                u_r = batch.result_targets[i]
                term_loss["result"] += float(
                    -(
                        u_r * math.log(u_pred[i] + 1e-12)
                        + (1 - u_r) * math.log(1 - u_pred[i] + 1e-12)
                    )
                )
                dscalar[i] += scale * (u_pred[i] - u_r) * lw.result
                term_counts["result"] += 1

        # scalar/WDL consistency — only where a genuine WDL triple exists
        # (spec 9).  On scalar-only rows the WDL head is unsupervised and
        # this term would be the sole WDL signal, fabricated from the
        # scalar it claims to check.
        dlogits_consistency = np.zeros((n, 3))
        for i in range(n):
            if not batch.wdl_targets[i]:
                continue
            p = probs[i]
            # exact jacobian of u_wdl = p_W + 0.5 p_D: ju_j = p_j (c_j - u)
            cvec = np.array([1.0, 0.5, 0.0])
            ju = p * (cvec - u_wdl[i])
            d = u_pred[i] - u_wdl[i]
            term_loss["consistency"] += float(d * d)
            dscalar[i] += 2 * d * scale * u_pred[i] * (1 - u_pred[i]) * lw.consistency
            dlogits_consistency[i] = -2 * d * lw.consistency * ju * scale
            term_counts["consistency"] += 1
        dlogits += dlogits_consistency

        # action-regret pairwise ranking (parent-POV action values): the model
        # side compares 1-u_child on each sibling, teacher delta already in
        # parent POV (built by the batch assembler).
        tau = lw.rank_temperature
        for i, j, delta in batch.rank_pairs:
            diff = (1.0 - u_pred[i]) - (1.0 - u_pred[j])  # parent POV
            ls = math.log1p(math.exp(-delta * diff / tau)) * abs(delta)
            term_loss["rank"] += ls
            g = -delta * _sigmoid(np.float64(-delta * diff / tau)) / tau * abs(delta)
            # diff = (1-u_i)-(1-u_j): d/dscalar_i = -s*u(1-u), d/dscalar_j = +
            dscalar[i] -= g * scale * u_pred[i] * (1 - u_pred[i]) * lw.rank
            dscalar[j] += g * scale * u_pred[j] * (1 - u_pred[j]) * lw.rank
            term_counts["rank"] += 1

        # normalise per-term by presented count
        denom = {k: max(1, c) for k, c in term_counts.items()}
        norm_loss = sum(term_loss[k] / denom[k] for k in term_loss)
        grads = {
            "dscalar": dscalar,
            "dlogits": dlogits,
            "norm_loss": norm_loss,
            "term_loss": term_loss,
            "term_counts": term_counts,
        }
        return norm_loss, grads

    # -- backward ---------------------------------------------------------------
    def backward(self, fwd: dict, batch: Batch, grads: dict) -> dict[str, np.ndarray]:
        spec = self.spec
        n = len(batch.enc)
        hc = spec.half_channels
        dout = np.concatenate([grads["dscalar"][:, None], grads["dlogits"]], axis=1)
        dinp = np.zeros((n, spec.head.input_count))
        # dense grads only for small params; the big tables get sparse
        # (row_idx, grad_block) entries below
        g: dict = {
            k: np.zeros_like(v)
            for k, v in self.params.items()
            if k in ("ft_b", "w1", "b1", "w2", "b2", "w3", "b3", "psqt_b")
        }

        for h, c in fwd["caches"].items():
            idx = c["idx"]
            do = dout[idx] @ fq(self.params["w3"][h], 127)
            g["w3"][h] += dout[idx].T @ c["o"]
            g["b3"][h] += dout[idx].sum(0)
            do1 = do[:, : spec.head.first_activation_concat]
            do2 = do[:, spec.head.first_activation_concat :]
            m1 = spec.head.first_affine_outputs
            m2 = spec.head.second_affine_outputs
            hs = 1 << spec.numeric.hidden_affine_shift
            # Deployed activation (features._hidden_activation / W05):
            #   x = floor(z/64);  lin = clip(x,0,127);
            #   sq = clip(floor(x*x/128), 0,127) on the UNCLIPPED x.
            # STE masks: lin active on 0<x<127; sq active on 0<v<127 where
            # v=floor(x*x/128); linearized dsq/dx = x/64, dx/dz = 1/64.
            lin2m = (c["x2"] > 0) & (c["x2"] < 127)
            sq2m = (c["v2"] > 0) & (c["v2"] < 127)
            dz2 = (do2[:, :m2] * lin2m + do2[:, m2:] * sq2m * (c["x2"] / 64.0)) / hs
            g["w2"][h] += dz2.T @ c["o1"]
            g["b2"][h] += dz2.sum(0)
            do1 += dz2 @ fq(self.params["w2"][h], 127)
            lin1m = (c["x1"] > 0) & (c["x1"] < 127)
            sq1m = (c["v1"] > 0) & (c["v1"] < 127)
            dz1 = (do1[:, :m1] * lin1m + do1[:, m1:] * sq1m * (c["x1"] / 64.0)) / hs
            g["w1"][h] += dz1.T @ fwd["inp"][idx]
            g["b1"][h] += dz1.sum(0)
            dinp[idx] = dz1 @ fq(self.params["w1"][h], 127)

        # paired product -> accumulator
        dact = np.stack([dinp[:, :hc], dinp[:, hc:]], axis=1)  # (n,2,hc)
        x, y, a, bb = fwd["x"], fwd["y"], fwd["a"], fwd["bb"]
        sh = 1 << spec.numeric.paired_product_shift
        dx = dact * (bb / sh) * ((x > 0) & (x < 255))
        dy = dact * (a / sh) * ((y > 0) & (y < 255))
        dacc = np.concatenate([dx, dy], axis=2).reshape(n * 2, spec.channels)

        g["ft_b"] += dacc.sum(axis=0)
        # scatter into table rows: per-element grads -> sort -> reduceat over
        # equal-row runs (fast segmented collect, no atomics)
        ri = fwd["row_index"]
        for key, wk, fk in (
            ("psq", "psq_w", "psq_fac"),
            ("thr", "thr_w", "thr_fac"),
            ("pp", "pp_w", "pp_fac"),
        ):
            r = ri[key]
            if len(r["rows"]):
                d_elem = dacc[r["src"]]
                g[wk] = self._collect(r["rows"], d_elem, spec.channels)
                g[fk] = self._collect(r["frows"], d_elem, spec.channels)
            else:
                g[wk] = g[fk] = None
        rp = ri["psqt"]
        if len(rp["rows"]):
            flat = rp["rows"] * spec.head.material_stacks + rp["head"]
            gsrc = (grads["dscalar"][rp["src"]] * rp["sign"] * 0.5).reshape(-1, 1)
            g["psqt_w"] = self._collect(flat, gsrc, 1)
        else:
            g["psqt_w"] = None
        g["psqt_b"] += np.bincount(
            batch.head, weights=grads["dscalar"], minlength=spec.head.material_stacks
        )
        return g

    @staticmethod
    def _collect(rows: np.ndarray, d_elem: np.ndarray, width: int):
        """Sum d_elem over equal row ids -> (unique_rows, summed_grads)."""
        order = np.argsort(rows, kind="stable")
        rsorted = rows[order]
        starts = np.concatenate([[0], np.flatnonzero(np.diff(rsorted)) + 1])
        gb = np.add.reduceat(d_elem[order].reshape(-1, width), starts)
        return rsorted[starts], gb

    def apply_grads(self, g: dict, lr: float, lw: LossWeights) -> None:
        b1, b2 = 0.9, 0.999
        eps = 1e-8
        self.step_count += 1
        t = self.step_count
        for k, w in self.params.items():
            grad = g.get(k)
            if grad is None:
                continue
            if isinstance(grad, tuple):
                idx, gb = grad  # sparse row update
                if k == "psqt_w":
                    w2d = w.reshape(-1)
                    m2d = self.m[k].reshape(-1)
                    v2d = self.v[k].reshape(-1)
                else:
                    w2d, m2d, v2d = w, self.m[k], self.v[k]
                gb = gb.reshape(m2d[idx].shape)
                if lw.quant_reg:
                    gb = gb + lw.quant_reg * (w2d[idx] - np.floor(w2d[idx] + 0.5))
                m2d[idx] = b1 * m2d[idx] + (1 - b1) * gb
                v2d[idx] = b2 * v2d[idx] + (1 - b2) * gb * gb
                mh = m2d[idx] / (1 - b1**t)
                vh = v2d[idx] / (1 - b2**t)
                w2d[idx] -= lr * (mh / (np.sqrt(vh) + eps) + lw.weight_decay * w2d[idx])
            else:
                if lw.quant_reg:
                    grad = grad + lw.quant_reg * (w - np.floor(w + 0.5))
                self.m[k] = b1 * self.m[k] + (1 - b1) * grad
                self.v[k] = b2 * self.v[k] + (1 - b2) * grad * grad
                mh = self.m[k] / (1 - b1**t)
                vh = self.v[k] / (1 - b2**t)
                w -= lr * (mh / (np.sqrt(vh) + eps) + lw.weight_decay * w)


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 256
    lr: float = LR_QAT  # derived from the fq grid step (see above)
    seed: int = 0
    loss_weights: LossWeights = field(default_factory=LossWeights)

"""Training driver: records -> batches -> fake-quant training -> export.

Implements the W04 pipeline end to end:

    shards -> dedup/split -> encode (schema) -> train (fake-quant)
           -> fold/integerize -> RXF1 export -> Numba runtime parity

Stage A (spec 10.2) is the target here: reproduction and semantic pilot on
a small real shard — the gate is semantic correctness, not loss quality.
"""

from __future__ import annotations

import os
import time

import chess
import numpy as np

from training.checkpoint import save_model_checkpoint, load_latest
from training.export import integerize, write_export
from training.features import FeatureEncoder, Encoded
from training.feature_spec import SPEC
from training.labels import record_targets
from training.model import Batch, F512Model, LR_QAT, TrainConfig
from training.records import load_shard
from training.sampler import MixtureSampler
from training.splits import assign_splits


def board_of(rec: dict) -> chess.Board:
    pos = rec["position"]
    hm = pos.get("halfmove_clock")
    fm = pos.get("fullmove_number")
    fen = pos["fen4"] + f" {hm if hm is not None else 0} {fm if fm is not None else 1}"
    # variant comes from the record — a chess960 position parsed as
    # orthodox silently changes the position's legality/castling state.
    # "standard" and "chess960" are honoured; ANY other tag is rejected
    # loudly — a non-orthodox position must never parse as orthodox.
    v = pos.get("variant", "standard")
    if v == "chess960":
        return chess.Board(fen, chess960=True)
    if v == "standard":
        return chess.Board(fen, chess960=False)
    raise ValueError(
        f"position variant {v!r} cannot be reconstructed as orthodox "
        "chess — honour it upstream or drop the record"
    )


def build_batch(
    records: list[dict],
    encoder: FeatureEncoder,
    enc_cache: dict[str, Encoded] | None = None,
) -> Batch:
    enc_list: list[Encoded] = []
    u_targets: list[list[float]] = []
    u_bounds: list[list[tuple[float | None, float | None]]] = []
    wdl_targets: list[list[list[float]]] = []
    result_targets: list[float | None] = []
    heads = np.zeros(len(records), dtype=np.int64)
    parent_groups: dict[str, list[tuple[int, float]]] = {}
    for i, rec in enumerate(records):
        b = board_of(rec)
        if enc_cache is not None and rec["record_id"] in enc_cache:
            enc = enc_cache[rec["record_id"]]
        else:
            enc = encoder.encode(b)
            if enc_cache is not None:
                enc_cache[rec["record_id"]] = enc
        enc_list.append(enc)
        heads[i] = enc.head
        stm_white = b.turn == chess.WHITE
        tgts = record_targets(rec, stm_white)
        # value targets exclude results: a game outcome feeds L_result only
        u_targets.append([t.u for t in tgts if t.u is not None and not t.is_result])
        u_bounds.append([(t.u_lo, t.u_hi) for t in tgts if t.is_bound])
        wdl_targets.append([t.wdl for t in tgts if t.wdl is not None])
        res = [t.u for t in tgts if t.is_result and t.u is not None]
        result_targets.append(res[0] if res else None)
        # action bank: children of a shared parent form regret pairs
        for t in tgts:
            if t.obs_kind == "action_searched" and t.parent_record_id and t.u is not None:
                parent_groups.setdefault(t.parent_record_id, []).append((i, t.u))
    rank_pairs: list[tuple[int, int, float]] = []
    for members in parent_groups.values():
        members.sort(key=lambda x: x[1])
        for a in range(len(members)):
            for c in range(a + 1, len(members)):
                i_lo, u_lo = members[a]
                i_hi, u_hi = members[c]
                # parent-POV delta = (1-u_hi_child) - (1-u_lo_child) = u_lo-u_hi
                rank_pairs.append((i_hi, i_lo, u_lo - u_hi))
    return Batch(
        enc=enc_list,
        u_targets=u_targets,
        u_bounds=u_bounds,
        wdl_targets=wdl_targets,
        result_targets=result_targets,
        rank_pairs=rank_pairs,
        head=heads,
    )


def run_pilot(
    shard_dirs: list[str],
    *,
    out_dir: str,
    config: TrainConfig | None = None,
    checkpoint_every: int = 25,
    export_name: str = "pilot.rxf1",
    resume: bool = True,
) -> dict:
    """End-to-end Stage-A pilot. Returns the run report dict."""
    config = config or TrainConfig()
    os.makedirs(out_dir, exist_ok=True)
    ckpt_root = os.path.join(out_dir, "checkpoints")

    records: list[dict] = []
    shard_manifests = []
    for d in shard_dirs:
        shard = load_shard(d)
        shard_manifests.append(shard.manifest)
        records.extend(shard.records())
    split_res = assign_splits(records)
    train_recs = [r for r in split_res.kept if r["split"] == "train"]
    dev_recs = [r for r in split_res.kept if r["split"] == "development"]

    encoder = FeatureEncoder()
    enc_cache: dict[str, Encoded] = {}
    t0 = time.time()
    for r in train_recs + dev_recs:
        enc_cache[r["record_id"]] = encoder.encode(board_of(r))
    encode_s = time.time() - t0

    families = [r["source"]["lineage_family"] for r in train_recs]
    model = F512Model(SPEC, seed=config.seed)
    sampler = MixtureSampler(families, seed=config.seed)
    start_step = 0
    if resume:
        ck = load_latest(ckpt_root)
        if ck is not None:
            params = ck.npz("params.npz")
            optim = ck.npz("optim.npz")
            for k in model.params:
                model.params[k] = params[k]
                model.m[k] = optim[f"m::{k}"]
                model.v[k] = optim[f"v::{k}"]
            st = ck.state()
            model.step_count = st["step"]
            sampler.set_state(st["sampler"])
            start_step = st["step"]

    steps_per_epoch = max(1, len(train_recs) // config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    history: list[dict] = []
    losses_epoch: list[float] = []
    for step in range(start_step, total_steps):
        idx = sampler.next_batch(config.batch_size)
        batch = build_batch([train_recs[i] for i in idx], encoder, enc_cache)
        fwd = model.forward(batch, need_grad=True)
        loss, grads = model.losses(fwd, batch, config.loss_weights)
        g = model.backward(fwd, batch, grads)
        model.apply_grads(g, config.lr, config.loss_weights)
        losses_epoch.append(loss)
        if (step + 1) % steps_per_epoch == 0:
            ep = (step + 1) // steps_per_epoch
            history.append(
                {
                    "epoch": ep,
                    "loss": float(np.mean(losses_epoch)),
                    "terms": grads["term_loss"],
                    "counts": grads["term_counts"],
                }
            )
            losses_epoch = []
        if (step + 1) % checkpoint_every == 0 or step + 1 == total_steps:
            optim = {f"m::{k}": v for k, v in model.m.items()}
            optim.update({f"v::{k}": v for k, v in model.v.items()})
            save_model_checkpoint(
                ckpt_root,
                step=step + 1,
                params=model.params,
                optim=optim,
                state={
                    "step": step + 1,
                    "sampler": sampler.get_state(),
                    "feature_schema_id": SPEC.schema_id,
                    "feature_schema_version": SPEC.schema_version,
                    "record_schema_version": "rx-final-1",
                    "data_manifest": [m["payload_sha256"] for m in shard_manifests],
                    "export_revision": "RXF1/1",
                    "loss_weights": vars(config.loss_weights),
                    "seed": config.seed,
                },
            )

    int_model = integerize(model.params, encoder, SPEC, model.factorized)
    export_path = os.path.join(out_dir, export_name)
    export_info = write_export(
        int_model, export_path, SPEC, extra_meta={"stage": "A-pilot", "steps": total_steps}
    )

    return {
        "history": history,
        "export": export_info,
        "n_train": len(train_recs),
        "n_dev": len(dev_recs),
        "n_dropped_dups": len(split_res.dropped_duplicates),
        "n_clusters": len(split_res.clusters),
        "dedup_report": split_res.dedup_report,
        "encode_seconds": encode_s,
        "ckpt_root": ckpt_root,
        "encoder": encoder,
        "model": model,
        "train_recs": train_recs,
        "dev_recs": dev_recs,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument(
        "--lr",
        type=float,
        default=LR_QAT,
        help="default LR_QAT = QAT_GRID_STEP/10 = 0.1 — derived "
        "from the fq integer grid step (AdamW's m̂/√v̂ step "
        "is batch-scale invariant, so lr IS the per-step "
        "movement in grid units; see training.model)",
    )
    args = ap.parse_args()
    rep = run_pilot(
        args.shards,
        out_dir=args.out,
        config=TrainConfig(epochs=args.epochs, batch_size=args.batch, lr=args.lr),
    )
    for h in rep["history"]:
        print(f"epoch {h['epoch']:3d}  loss {h['loss']:.6f}")
    print("export:", rep["export"])

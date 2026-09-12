"""Typed sharded training records implementing the ``rx-final-1`` contract.

The minimum contract is ``spec/RX_FINAL_PLAN/training_schema.json``; this
module enforces the expressible parts (required fields, enums, ranges, WDL
normalisation, score-kind/field coherence) plus the semantic rules listed
under ``semantic_checks_not_expressible_here`` that can be checked locally.

Shard layout (immutable once committed)::

    <shard_dir>/records.jsonl        one JSON record per line
    <shard_dir>/manifest.json        counts, hashes, schema ids, split stats

Writes are transactional: payload lands in ``<shard>.tmp/`` and is renamed
into place only after the manifest (with payload sha256) is written.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any
from collections.abc import Iterator

RECORD_SCHEMA_VERSION = "rx-final-1"
SHARD_FORMAT_VERSION = "rx-final-shard-1"

OBS_KINDS = {
    "direct_neural",
    "searched",
    "action_searched",
    "game_outcome",
    "tablebase",
    "policy_visits",
    "counterfactual",
}
PERSPECTIVES = {"white", "black", "side_to_move", "parent_side_to_move"}
SCORE_KINDS = {
    "cp",
    "mate",
    "wdl",
    "expected_score",
    "interval",
    "visits",
    "outcome",
    "operation_result",
}
BOUND_KINDS = {"none", "search_exact", "lower", "upper", "interval", "unknown"}
SPLITS = {"train", "development", "sealed", "quarantine"}
VARIANTS = {"standard", "chess960", "other"}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------
# constructors


def make_position(
    *,
    fen4: str,
    variant: str,
    halfmove_clock: int | None,
    fullmove_number: int | None,
    history_complete: bool,
    unknown_prefix: bool,
    absolute_ply: int | None = None,
    observed_reversible_history_keys: list[str] | None = None,
    known_fields: list[str] | None = None,
    chosen_relabel_context: dict | None = None,
) -> dict:
    return {
        "fen4": fen4,
        "variant": variant,
        "halfmove_clock": halfmove_clock,
        "fullmove_number": fullmove_number,
        "absolute_ply": absolute_ply,
        "history_complete": history_complete,
        "unknown_prefix": unknown_prefix,
        "observed_reversible_history_keys": observed_reversible_history_keys or [],
        "known_fields": known_fields or [],
        "chosen_relabel_context": chosen_relabel_context,
    }


def make_source(
    *,
    corpus: str,
    object_sha256: str,
    decoder_revision: str,
    lineage_family: str,
    original_position_id: str,
    original_game_id: str | None = None,
    licence_record_id: str | None = None,
    derivation_chain: list[dict] | None = None,
    split_group: str | None = None,
) -> dict:
    return {
        "corpus": corpus,
        "object_sha256": object_sha256,
        "decoder_revision": decoder_revision,
        "original_game_id": original_game_id,
        "original_position_id": original_position_id,
        "lineage_family": lineage_family,
        "licence_record_id": licence_record_id,
        "derivation_chain": derivation_chain or [],
        "split_group": split_group,
    }


def make_observation(
    *,
    kind: str,
    perspective: str,
    score_kind: str,
    bound_kind: str = "none",
    context_known: bool = True,
    cp: float | None = None,
    mate_plies: int | None = None,
    wdl: list[float] | None = None,
    expected_score: float | None = None,
    raw_payload: Any = None,
    teacher_id: str | None = None,
    nodes: int | None = None,
    depth: int | None = None,
    uncached_neural_evaluations: int | None = None,
    elapsed_ms: float | None = None,
    parent_record_id: str | None = None,
    action_uci: str | None = None,
    termination: str | None = None,
    interrupted: bool = False,
    right_censored: bool = False,
    calibration_id: str | None = None,
) -> dict:
    return {
        "kind": kind,
        "perspective": perspective,
        "score_kind": score_kind,
        "bound_kind": bound_kind,
        "context_known": context_known,
        "cp": cp,
        "mate_plies": mate_plies,
        "wdl": wdl,
        "expected_score": expected_score,
        "raw_payload": raw_payload,
        "teacher_id": teacher_id,
        "nodes": nodes,
        "depth": depth,
        "uncached_neural_evaluations": uncached_neural_evaluations,
        "elapsed_ms": elapsed_ms,
        "parent_record_id": parent_record_id,
        "action_uci": action_uci,
        "termination": termination,
        "interrupted": interrupted,
        "right_censored": right_censored,
        "calibration_id": calibration_id,
    }


def make_record(
    *,
    record_id: str,
    position: dict,
    source: dict,
    observations: list[dict],
    legal_actions_enumerated: bool = False,
    split: str | None = None,
) -> dict:
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "record_id": record_id,
        "position": position,
        "source": source,
        "observations": observations,
        "legal_actions_enumerated": legal_actions_enumerated,
        "split": split,
    }


# --------------------------------------------------------------------------
# validation


def validate_record(rec: dict) -> list[str]:
    """Return a list of contract violations; empty means valid."""
    errs: list[str] = []
    for k in ("schema_version", "record_id", "position", "source", "observations"):
        if k not in rec:
            errs.append(f"missing top-level field {k}")
    if errs:
        return errs
    if rec["schema_version"] != RECORD_SCHEMA_VERSION:
        errs.append(f"schema_version {rec['schema_version']!r} != {RECORD_SCHEMA_VERSION!r}")
    if not isinstance(rec["record_id"], str) or not rec["record_id"]:
        errs.append("record_id must be a nonempty string")
    if rec.get("split") is not None and rec["split"] not in SPLITS:
        errs.append(f"bad split {rec['split']!r}")

    pos = rec["position"]
    for k in (
        "fen4",
        "variant",
        "halfmove_clock",
        "fullmove_number",
        "history_complete",
        "unknown_prefix",
    ):
        if k not in pos:
            errs.append(f"position missing {k}")
    if pos.get("variant") not in VARIANTS:
        errs.append(f"bad variant {pos.get('variant')!r}")
    hmc, fmn = pos.get("halfmove_clock"), pos.get("fullmove_number")
    if hmc is not None and not (isinstance(hmc, int) and hmc >= 0):
        errs.append("halfmove_clock must be int >= 0 or null")
    if fmn is not None and not (isinstance(fmn, int) and fmn >= 1):
        errs.append("fullmove_number must be int >= 1 or null")
    if len(str(pos.get("fen4", "")).split(" ")) != 4:
        errs.append("fen4 must have exactly 4 fields (pieces side castling ep)")

    src = rec["source"]
    for k in (
        "corpus",
        "object_sha256",
        "decoder_revision",
        "lineage_family",
        "original_position_id",
    ):
        if k not in src:
            errs.append(f"source missing {k}")
    if not _SHA256_RE.match(str(src.get("object_sha256", ""))):
        errs.append("source.object_sha256 must be 64 lowercase hex")

    obs = rec["observations"]
    if not isinstance(obs, list) or not obs:
        errs.append("observations must be a nonempty array")
        return errs
    for i, o in enumerate(obs):
        tag = f"observations[{i}]"
        for k in ("kind", "perspective", "score_kind", "bound_kind", "context_known"):
            if k not in o:
                errs.append(f"{tag} missing {k}")
        if o.get("kind") not in OBS_KINDS:
            errs.append(f"{tag}.kind {o.get('kind')!r} not in enum")
        if o.get("perspective") not in PERSPECTIVES:
            errs.append(f"{tag}.perspective {o.get('perspective')!r} not in enum")
        if o.get("score_kind") not in SCORE_KINDS:
            errs.append(f"{tag}.score_kind {o.get('score_kind')!r} not in enum")
        if o.get("bound_kind") not in BOUND_KINDS:
            errs.append(f"{tag}.bound_kind {o.get('bound_kind')!r} not in enum")
        w = o.get("wdl")
        if w is not None:
            if not (
                isinstance(w, list)
                and len(w) == 3
                and all(isinstance(x, (int, float)) and 0.0 <= x <= 1.0 for x in w)
            ):
                errs.append(f"{tag}.wdl must be [W,D,L] probabilities in [0,1]")
            elif abs(sum(w) - 1.0) > 1e-6:
                errs.append(f"{tag}.wdl does not sum to 1 ({sum(w)})")
        es = o.get("expected_score")
        if es is not None and not (0.0 <= es <= 1.0):
            errs.append(f"{tag}.expected_score out of [0,1]")
        # score-kind/field coherence: the declared kind must carry its payload
        # (an interrupted/right-censored observation legitimately has none)
        censored = bool(o.get("interrupted") or o.get("right_censored"))
        sk = o.get("score_kind")
        if not censored:
            if sk == "cp" and o.get("cp") is None:
                errs.append(f"{tag}: score_kind=cp but cp is null")
            if sk == "mate" and o.get("mate_plies") is None:
                errs.append(f"{tag}: score_kind=mate but mate_plies is null")
            if sk == "wdl" and w is None:
                errs.append(f"{tag}: score_kind=wdl but wdl is null")
            if sk in ("expected_score", "outcome") and es is None and w is None:
                errs.append(f"{tag}: score_kind={sk} needs expected_score or wdl")
        # bound kinds must not pose as point labels
        if o.get("bound_kind") in ("lower", "upper", "interval", "unknown") and sk in (
            "mate",
            "outcome",
        ):
            errs.append(f"{tag}: bound_kind {o['bound_kind']!r} incompatible with {sk}")
        if o.get("interrupted") and not o.get("right_censored"):
            errs.append(f"{tag}: interrupted trials must be marked right_censored")
    return errs


# --------------------------------------------------------------------------
# shard IO


def shard_manifest(
    *,
    shard_id: str,
    feature_schema_id: str,
    payload_sha256: str,
    record_count: int,
    records: list[dict],
) -> dict:
    splits: dict[str, int] = {}
    games: set[str] = set()
    families: set[str] = set()
    kinds: dict[str, int] = {}
    for r in records:
        splits[str(r.get("split"))] = splits.get(str(r.get("split")), 0) + 1
        if r["source"].get("original_game_id"):
            games.add(r["source"]["original_game_id"])
        families.add(r["source"]["lineage_family"])
        for o in r["observations"]:
            kinds[o["kind"]] = kinds.get(o["kind"], 0) + 1
    return {
        "shard_format": SHARD_FORMAT_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "feature_schema_id": feature_schema_id,
        "shard_id": shard_id,
        "record_count": record_count,
        "payload_sha256": payload_sha256,
        "split_counts": splits,
        "game_ids": sorted(games),
        "lineage_families": sorted(families),
        "observation_kinds": kinds,
    }


def write_shard(
    shard_dir: str, records: list[dict], *, shard_id: str, feature_schema_id: str
) -> dict:
    """Write an immutable shard transactionally; returns the manifest."""
    for r in records:
        errs = validate_record(r)
        if errs:
            raise ValueError(f"record {r.get('record_id')}: {errs}")
    tmp_dir = shard_dir + ".tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    payload = b"".join(
        (json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n").encode() for r in records
    )
    digest = hashlib.sha256(payload).hexdigest()
    payload_path = os.path.join(tmp_dir, "records.jsonl")
    with open(payload_path, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    manifest = shard_manifest(
        shard_id=shard_id,
        feature_schema_id=feature_schema_id,
        payload_sha256=digest,
        record_count=len(records),
        records=records,
    )
    with open(os.path.join(tmp_dir, "manifest.json"), "wb") as fh:
        fh.write(json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n")
        fh.flush()
        os.fsync(fh.fileno())
    if os.path.exists(shard_dir):
        raise FileExistsError(f"shard {shard_dir} already exists (shards are immutable)")
    os.rename(tmp_dir, shard_dir)
    return manifest


@dataclass
class Shard:
    manifest: dict
    path: str

    def records(self) -> Iterator[dict]:
        with open(os.path.join(self.path, "records.jsonl")) as fh:
            for line in fh:
                yield json.loads(line)


def load_shard(shard_dir: str, verify: bool = True) -> Shard:
    with open(os.path.join(shard_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    if verify:
        h = hashlib.sha256()
        with open(os.path.join(shard_dir, "records.jsonl"), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != manifest["payload_sha256"]:
            raise ValueError(f"shard {shard_dir} payload hash mismatch")
    return Shard(manifest=manifest, path=shard_dir)

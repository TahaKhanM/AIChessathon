"""Versioned model export: the canonical deployed RXF1 container.

As of the FIX-CONTRACT unification there is ONE RXF1 dialect: the
``engine.model_io`` container (magic "RXF1", versioned header, meta JSON,
packed sections, SHA-256).  ``write_export`` emits exactly that container,
so a produced artifact loads through ``model_io.read_model`` ->
``EvalWeights.from_model`` -> the production evaluator with no shim.

``read_export`` remains dialect-tolerant: it decodes the canonical
container first and falls back to the pre-unification trainer layout
(``"RXF1" + u16 version + u32 header-len + JSON header + payload +
sha256 trailer``, sections in ``psq|thr|pp|ft_b|...`` order) so the
already-written audit artifacts stay readable.  The legacy format is
READ-ONLY — nothing writes it any more.

Canonical payload (model_io SECTIONS order, 9/7/6-bit packed + raw):
    bias   int16  [512]
    psq    packed signed 9-bit  [9216*512]
    thr    packed signed 7-bit  [59808*512]
    pp     packed signed 6-bit  [1488*512]
    head_w1 int8  [8,512,16]  ([in][out]; transposed vs IntegerModel)
    head_b1 int32 [8,16]
    head_w2 int8  [8,32,32]
    head_b2 int32 [8,32]
    head_w3 int8  [8,96,4]
    head_b3 int32 [8,4]
    psqt_w int16  [9216,8]
    psqt_b int32  [8]

Total numeric payload for the reference config must equal the authoritative
``reference_packed_accounting.numeric_payload_bytes`` = 32,900,768.
"""

from __future__ import annotations

import hashlib
import json
import struct

import numpy as np

from typing import TYPE_CHECKING

from engine.model_io import (
    HEADER_BYTES,
    ModelFormatError,
    read_model,
    write_model,
)
from training.feature_spec import SPEC, FeatureSpec
from training.int_eval import IntegerModel
from training.packing import unpack_signed_array

if TYPE_CHECKING:
    from training.features import FeatureEncoder

MAGIC = b"RXF1"
EXPORT_VERSION = 1

# Numeric payload size — asserted by the gates; both dialects pack the same
# section bytes, only the order/head orientation differ.
PACKED_PAYLOAD_BYTES = 32_900_768

# (name, dtype, bits-or-None-for-packed) — LEGACY trainer order, kept only so
# read-side can describe old artifacts.
SECTION_ORDER = [
    ("psq", "packed", 9),
    ("thr", "packed", 7),
    ("pp", "packed", 6),
    ("ft_b", "int16", None),
    ("w1", "int8", None),
    ("b1", "int32", None),
    ("w2", "int8", None),
    ("b2", "int32", None),
    ("w3", "int8", None),
    ("b3", "int32", None),
    ("psqt_w", "int16", None),
    ("psqt_b", "int32", None),
]


def integerize(
    params: dict[str, np.ndarray],
    enc: FeatureEncoder,
    spec: FeatureSpec = SPEC,
    factorized: bool = True,
) -> IntegerModel:
    """Produce the IntegerModel from trainable float params.

    ``enc`` supplies the factorizer index maps (row -> factor row), so the
    fold is generated from the same schema-driven maps as extraction.
    """
    lim = {
        "psq": spec.psq.coefficient_abs_limit,
        "thr": spec.threats.coefficient_abs_limit,
        "pp": spec.pawn_pairs.coefficient_abs_limit,
    }

    def fq_i(x: np.ndarray, limit: int) -> np.ndarray:
        return np.clip(np.floor(x + 0.5), -limit, limit)

    def fold_rows(wkey: str, fkey: str, row_to_fac: np.ndarray, limit: int) -> np.ndarray:
        w = params[wkey].astype(np.float64)
        if factorized:
            w = w + params[fkey][row_to_fac].astype(np.float64)
        return fq_i(w, limit)

    psq = fold_rows("psq_w", "psq_fac", enc.psq_row_to_fac, lim["psq"]).astype(np.int16)
    thr = fold_rows("thr_w", "thr_fac", enc.thr_row_to_fac, lim["thr"]).astype(np.int8)
    pp = fold_rows("pp_w", "pp_fac", enc.pp_row_to_fac, lim["pp"]).astype(np.int8)
    out = IntegerModel(
        psq_w=psq,
        thr_w=thr,
        pp_w=pp,
        ft_b=fq_i(params["ft_b"], spec.bias_limit).astype(np.int16),
        w1=fq_i(params["w1"], 127).astype(np.int8),
        b1=fq_i(params["b1"], 2**31 - 1).astype(np.int32),
        w2=fq_i(params["w2"], 127).astype(np.int8),
        b2=fq_i(params["b2"], 2**31 - 1).astype(np.int32),
        w3=fq_i(params["w3"], 127).astype(np.int8),
        b3=fq_i(params["b3"], 2**31 - 1).astype(np.int32),
        psqt_w=fq_i(params["psqt_w"], spec.psqt_abs_limit).astype(np.int16),
        psqt_b=fq_i(params["psqt_b"], 2**31 - 1).astype(np.int32),
    )
    return out


def int_model_sections(model: IntegerModel) -> dict:
    """IntegerModel ([out][in] heads) -> model_io section dict ([in][out])."""
    return {
        "bias": np.asarray(model.ft_b, dtype=np.int16),
        "psq": np.asarray(model.psq_w, dtype=np.int16),
        "thr": np.asarray(model.thr_w, dtype=np.int8),
        "pp": np.asarray(model.pp_w, dtype=np.int8),
        "head_w1": np.ascontiguousarray(model.w1.transpose(0, 2, 1)),
        "head_b1": np.asarray(model.b1, dtype=np.int32),
        "head_w2": np.ascontiguousarray(model.w2.transpose(0, 2, 1)),
        "head_b2": np.asarray(model.b2, dtype=np.int32),
        "head_w3": np.ascontiguousarray(model.w3.transpose(0, 2, 1)),
        "head_b3": np.asarray(model.b3, dtype=np.int32),
        "psqt_w": np.asarray(model.psqt_w, dtype=np.int16),
        "psqt_b": np.asarray(model.psqt_b, dtype=np.int32),
    }


def sections_to_int_model(d: dict) -> IntegerModel:
    """model_io decoded sections ([in][out]) -> IntegerModel ([out][in])."""
    return IntegerModel(
        psq_w=np.asarray(d["psq"], dtype=np.int16),
        thr_w=np.asarray(d["thr"], dtype=np.int8),
        pp_w=np.asarray(d["pp"], dtype=np.int8),
        ft_b=np.asarray(d["bias"], dtype=np.int16),
        w1=np.ascontiguousarray(d["head_w1"].transpose(0, 2, 1)),
        b1=np.asarray(d["head_b1"], dtype=np.int32),
        w2=np.ascontiguousarray(d["head_w2"].transpose(0, 2, 1)),
        b2=np.asarray(d["head_b2"], dtype=np.int32),
        w3=np.ascontiguousarray(d["head_w3"].transpose(0, 2, 1)),
        b3=np.asarray(d["head_b3"], dtype=np.int32),
        psqt_w=np.asarray(d["psqt_w"], dtype=np.int16),
        psqt_b=np.asarray(d["psqt_b"], dtype=np.int32),
    )


def _numeric_meta(spec: FeatureSpec) -> dict:
    n = spec.numeric
    return {
        "ft_operand_clip": list(n.ft_operand_clip),
        "paired_product_shift": n.paired_product_shift,
        "hidden_clip": list(n.hidden_clip),
        "hidden_affine_shift": n.hidden_affine_shift,
        "square_activation_shift": n.square_activation_shift,
        "scalar_u_log2_divisor": n.scalar_u_log2_divisor,
        "cp_num": n.cp_num,
        "cp_shift": n.cp_shift,
        "score_bound": n.score_bound,
        "mate_band_start": n.mate_band_start,
    }


def _export_meta(spec: FeatureSpec, extra_meta: dict | None) -> dict:
    meta = {
        "format": "RXF1",
        "version": EXPORT_VERSION,
        "model_name": spec.name,
        "feature_schema_version": spec.schema_version,
        "feature_schema_id": spec.schema_id,
        "channels": spec.channels,
        "numeric": _numeric_meta(spec),
        # Deployed-rescale fields read by EvalWeights.from_model.
        "scale_num": spec.numeric.cp_num,
        "scale_shift": spec.numeric.cp_shift,
        "neural_bound": spec.numeric.score_bound,
        # Free-form trainer annotations (step, run id, ...).
        "meta": dict(extra_meta or {}),
    }
    return meta


def write_export(
    model: IntegerModel, path: str, spec: FeatureSpec = SPEC, extra_meta: dict | None = None
) -> dict:
    """Emit the canonical deployed container for ``model``."""
    meta = _export_meta(spec, extra_meta)
    blob = write_model(int_model_sections(model), packed=True, meta=meta)
    with open(path, "wb") as fh:
        fh.write(blob)
    # Payload region = everything after header+meta; meta_len is the exact
    # JSON write_model serialised (sort_keys).
    meta_len = len(json.dumps(meta, sort_keys=True).encode())
    payload = blob[HEADER_BYTES + meta_len :]
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "path": path,
        "payload_bytes": len(payload),
        "payload_sha256": digest,
        "total_bytes": len(blob),
    }


def read_export(path: str) -> tuple[dict, IntegerModel]:
    with open(path, "rb") as fh:
        blob = fh.read()
    return read_export_bytes(blob)


def read_export_bytes(blob: bytes) -> tuple[dict, IntegerModel]:
    """Parse an RXF1 blob of either dialect.

    Canonical containers decode through ``model_io.read_model`` (hash- and
    range-checked); anything else is tried as the legacy trainer layout.
    The returned header dict carries the keys consumers rely on —
    ``feature_schema_id``, ``model_name``, ``numeric``, ``payload_sha256``,
    ``meta`` — plus ``container`` ("canonical" | "legacy").
    """
    try:
        d = read_model(blob)
    except (ModelFormatError, ValueError, KeyError, IndexError, TypeError, struct.error):
        return _read_export_bytes_legacy(blob)
    meta = d.get("_meta", {})
    meta_len = len(json.dumps(meta, sort_keys=True).encode())
    payload = blob[HEADER_BYTES + meta_len :]
    header = {
        "format": "RXF1",
        "version": meta.get("version", EXPORT_VERSION),
        "model_name": meta.get("model_name"),
        "feature_schema_version": meta.get("feature_schema_version"),
        "feature_schema_id": meta.get("feature_schema_id"),
        "channels": meta.get("channels"),
        "numeric": meta.get("numeric", {}),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "meta": meta.get("meta", {}),
        "container": "canonical",
        "sections": [s for s in d if not s.startswith("_")],
    }
    return header, sections_to_int_model(d)


# ---------------------------------------------------------------------------
# Legacy trainer dialect — read-only support for pre-unification artifacts.
# ---------------------------------------------------------------------------


def _read_export_bytes_legacy(blob: bytes) -> tuple[dict, IntegerModel]:
    """Pre-unification container: "RXF1" + u16 + u32 hlen + JSON + payload
    + sha256 trailer, sections in ``SECTION_ORDER`` ([out][in] heads)."""
    if blob[:4] != MAGIC:
        raise ValueError("bad magic")
    version, hlen = struct.unpack("<HI", blob[4:10])
    if version != EXPORT_VERSION:
        raise ValueError(f"unsupported export version {version}")
    header = json.loads(blob[10 : 10 + hlen])
    payload = blob[10 + hlen : -32]
    trailer = blob[-32:].hex()
    if not (hashlib.sha256(payload).hexdigest() == trailer == header["payload_sha256"]):
        raise ValueError("export sha256 mismatch")
    arrs: dict[str, np.ndarray] = {}
    for s in header["sections"]:
        sec = payload[s["offset"] : s["offset"] + s["bytes"]]
        if s["kind"] == "packed":
            count = int(np.prod(s["shape"]))
            a = unpack_signed_array(sec, count, s["bits"])
            dtype = {"psq": np.int16, "thr": np.int8, "pp": np.int8}[s["name"]]
            arrs[s["name"]] = a.astype(dtype).reshape(s["shape"])
        else:
            arrs[s["name"]] = np.frombuffer(sec, dtype=s["kind"]).reshape(s["shape"]).copy()
    model = IntegerModel(
        psq_w=arrs["psq"],
        thr_w=arrs["thr"],
        pp_w=arrs["pp"],
        ft_b=arrs["ft_b"],
        w1=arrs["w1"],
        b1=arrs["b1"],
        w2=arrs["w2"],
        b2=arrs["b2"],
        w3=arrs["w3"],
        b3=arrs["b3"],
        psqt_w=arrs["psqt_w"],
        psqt_b=arrs["psqt_b"],
    )
    header = dict(header)
    header["container"] = "legacy"
    return header, model

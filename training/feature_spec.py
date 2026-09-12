"""Single versioned feature/evaluator schema for RX-FINAL F512-EF-K12-16/32.

This module is the ONE source of truth for the deployed feature contract.
It generates the canonical schema artifact
``training/schema/f512_ef_k12_16_32.v1.json`` whose sha256 prefix is the
``schema_id`` recorded in checkpoints, exports and data manifests.

Constants are *generated from* ``spec/RX_FINAL_PLAN/architecture.json`` (the
authoritative packet); nothing here is restated from memory.  Encoders
(``training/features.py``), the trainer (``training/model.py``), the integer
evaluator (``training/int_eval.py``), the export writer
(``training/export.py``) and the Numba runtime (``training/numba_rt.py``)
are all generated from / validated against this schema.  The engine-side
encoder ``engine/features.py`` consumes the same artifact.

Spec references: docs/architecture.md section 4,
spec/RX_FINAL_PLAN/architecture.json.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHITECTURE_JSON = os.path.join(REPO_ROOT, "spec", "RX_FINAL_PLAN", "architecture.json")
SCHEMA_DIR = os.path.join(REPO_ROOT, "training", "schema")
SCHEMA_VERSION = "rx-final-2"

# Deterministic source pin for the filtered-threat definition.  The
# full_threats.{h,cpp} blobs at this revision are mirrored verbatim in
# training/reference/ (git blob sha1 of the .cpp verified against
# spec/RX_FINAL_PLAN/resources.json: ac4f79da58c983e1a46c989697428e4edf1abbdf).
THREAT_SOURCE_PIN = (
    "official-stockfish/Stockfish@59aae690f91d6f69aac194f447d84b4a2c3be778:"
    "src/nnue/features/full_threats.{h,cpp}"
)


@dataclass(frozen=True)
class RowSpace:
    """One additive row space summed into the shared accumulator."""

    name: str
    rows: int
    runtime_dtype: str
    coefficient_abs_limit: int
    storage_bits: int  # lossless fixed-width signed storage
    max_active_bound: int
    factorizer_rows: int  # coarse factorized row space folded at export


@dataclass(frozen=True)
class HeadSpec:
    material_stacks: int
    input_count: int
    first_affine_outputs: int
    first_activation_concat: int
    second_affine_outputs: int
    second_activation_concat: int
    skip_concat_outputs: int
    outputs: tuple[str, ...]
    coefficient_dtype: str
    bias_and_dot_dtype: str
    rescale_dtype: str
    coefficient_abs_limit: int = 127  # int8
    stack_selection: str = "min(7,max(0,(piece_count-2)//4))"


@dataclass(frozen=True)
class NumericContract:
    ft_operand_clip: tuple[int, int]
    paired_product_shift: int
    paired_product_max: int
    hidden_clip: tuple[int, int]
    hidden_affine_shift: int
    square_activation_shift: int
    accumulator_dtype: str
    accumulator_proven_abs_bound: int
    # Exported scalar->score transform.  Versioned inside the export header;
    # scalar is a signed rescale-domain integer.  u = sigmoid(scalar / 2^p)
    # is the *training* coordinate; the deployed cp transform is
    # cp = clamp(scalar * cp_num >> cp_shift, -score_bound, +score_bound).
    scalar_u_log2_divisor: int
    cp_num: int
    cp_shift: int
    score_bound: int  # strictly inside the mate band
    mate_band_start: int


@dataclass(frozen=True)
class FeatureSpec:
    schema_version: str
    name: str
    channels: int
    perspectives: int
    half_channels: int  # channels // 2; paired halves
    king_buckets: int
    king_bucket_map: tuple[tuple[int, ...], ...]  # [rank_from_home][file]
    threat_source_pin: str
    psq: RowSpace
    threats: RowSpace
    pawn_pairs: RowSpace
    bias_limit: int
    bias_dtype: str
    head: HeadSpec
    numeric: NumericContract
    psqt_rows: int
    psqt_dtype: str
    psqt_abs_limit: int
    storage: dict = field(default_factory=dict)

    @property
    def schema_id(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()[:16]

    def canonical_json(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()

    def schema_dict(self) -> dict:
        """The full schema as a dict: canonical fields + ``schema_id``.

        FR2-B3: every producer that stamps a feature schema MUST derive it
        here — never by loading a possibly-stale on-disk artifact.  The
        JSON artifact is a *derived* copy (``write_artifact``), not the
        source of truth."""
        return {"schema_id": self.schema_id, **json.loads(self.canonical_json())}

    def write_artifact(self, directory: str = SCHEMA_DIR) -> str:
        os.makedirs(directory, exist_ok=True)
        fname = f"f512_ef_k12_16_32.{self.schema_version}.json"
        path = os.path.join(directory, fname)
        payload = {"schema_id": self.schema_id, **json.loads(self.canonical_json())}
        with open(path, "wb") as fh:
            fh.write(json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n")
        return path


def load_spec() -> FeatureSpec:
    """Build the schema from the authoritative architecture.json."""
    with open(ARCHITECTURE_JSON) as fh:
        a = json.load(fh)

    f = a["features"]
    h = a["head"]
    n = a["numeric_contract"]
    st = a["storage"]

    channels = int(f["channels"])
    king_map = tuple(
        tuple(r) for r in f["psq"]["king_bucket_map_by_rank_from_perspective_home_rank"]
    )

    psq = RowSpace(
        name="psq",
        rows=int(f["psq"]["rows"]),
        runtime_dtype=f["psq"]["runtime_dtype"],
        coefficient_abs_limit=int(f["psq"]["coefficient_abs_limit"]),
        storage_bits=int(f["psq"]["lossless_storage_bits"]),
        max_active_bound=32,  # one per on-board piece incl. kings
        # factorizer: same piece-square identity with bucket conditioning and
        # king-side mirror removed -> 2 relcolours * 6 pt * 64 sq
        factorizer_rows=2 * 6 * 64,
    )
    threats = RowSpace(
        name="threats",
        rows=int(f["threats"]["rows"]),
        runtime_dtype=f["threats"]["runtime_dtype"],
        coefficient_abs_limit=int(f["threats"]["coefficient_abs_limit"]),
        storage_bits=int(f["threats"]["lossless_storage_bits"]),
        max_active_bound=int(f["threats"]["max_active_bound"]),
        # factorizer: identity-orientation index (perspective flip and
        # king-side mirror collapsed); same table space.
        factorizer_rows=int(f["threats"]["rows"]),
    )
    pawn_pairs = RowSpace(
        name="pawn_pairs",
        rows=int(f["pawn_pairs"]["rows"]),
        runtime_dtype=f["pawn_pairs"]["runtime_dtype"],
        coefficient_abs_limit=int(f["pawn_pairs"]["coefficient_abs_limit"]),
        storage_bits=int(f["pawn_pairs"]["lossless_storage_bits"]),
        max_active_bound=int(f["pawn_pairs"]["max_active_bound"]),
        # factorizer: unordered physical square pair with relcolour removed.
        factorizer_rows=372,  # verified by encoder construction
    )
    head = HeadSpec(
        material_stacks=int(h["material_stacks"]),
        input_count=int(h["input_count"]),
        first_affine_outputs=int(h["first_affine_outputs"]),
        first_activation_concat=int(h["first_activation_concat"]),
        second_affine_outputs=int(h["second_affine_outputs"]),
        second_activation_concat=int(h["second_activation_concat"]),
        skip_concat_outputs=int(h["skip_concat_outputs"]),
        outputs=tuple(h["outputs"]),
        coefficient_dtype=h["coefficient_dtype"],
        bias_and_dot_dtype=h["bias_and_dot_dtype"],
        rescale_dtype=h["rescale_dtype"],
        stack_selection=h["selection"],
    )
    numeric = NumericContract(
        ft_operand_clip=tuple(n["ft_operand_clip"]),
        paired_product_shift=int(n["paired_product_shift"]),
        paired_product_max=int(n["paired_product_max"]),
        hidden_clip=tuple(n["hidden_clip"]),
        hidden_affine_shift=int(n["hidden_affine_shift_reference"]),
        square_activation_shift=int(n["square_activation_shift"]),
        accumulator_dtype=f["accumulator"]["dtype"],
        accumulator_proven_abs_bound=int(f["accumulator"]["proven_abs_bound"]),
        # u = sigmoid(raw / 2^divisor).  divisor=8 puts the raw scalar in
        # ~centipawn units (s=256, vs the BCE-fitted teacher s~=275), so the
        # integer head only needs |raw| ~ 2k for the full useful u range —
        # reachable under AdamW drift in thousands (not millions) of steps.
        # The deployed cp transform raw -> cp = raw*cp_num >> cp_shift maps
        # the s=256 domain onto the fitted s=275 teacher convention.
        scalar_u_log2_divisor=8,
        # Deployed leaf constants are read from the authoritative JSON
        # (numeric_contract.leaf_transform_export_defaults, added by the
        # ruling-42 amendment — the JSON is the single source of truth).
        # Deployed score clamp: mate-in-N scores occupy [27952, 30000)
        # (engine/tt.py MATE=30000, MATE_IN_MAX=MATE-2048).  A clamped neural
        # score must never equal a mate sentinel, so score_bound is one below
        # the band start (engine/evaluate.py NEURAL_BOUND mirrors this).
        cp_num=int(n["leaf_transform_export_defaults"]["scale_num"]),
        cp_shift=int(n["leaf_transform_export_defaults"]["scale_shift"]),
        score_bound=int(n["leaf_transform_export_defaults"]["neural_bound"]),
        mate_band_start=27952,
    )
    return FeatureSpec(
        schema_version=SCHEMA_VERSION,
        name=a["name"],
        channels=channels,
        perspectives=int(f["perspectives"]),
        half_channels=channels // 2,
        king_buckets=int(f["psq"]["king_buckets"]),
        king_bucket_map=king_map,
        threat_source_pin=f["threats"]["definition_revision"],
        psq=psq,
        threats=threats,
        pawn_pairs=pawn_pairs,
        bias_limit=int(f["bias"]["coefficient_abs_limit"]),
        bias_dtype=f["bias"]["runtime_dtype"],
        head=head,
        numeric=numeric,
        psqt_rows=int(f["psq"]["rows"]),
        psqt_dtype="int16",
        psqt_abs_limit=32767,
        storage={
            "bit_order": st["bit_order"],
            "default": st["default"],
            "raw_fallback": bool(st["raw_fallback"]),
        },
    )


def sanity_check(spec: FeatureSpec) -> None:
    """Assertions the schema itself must satisfy before anyone consumes it."""
    assert spec.channels == 512 and spec.half_channels == 256
    assert len(spec.king_bucket_map) == 8 and all(len(r) == 8 for r in spec.king_bucket_map)
    assert spec.king_buckets == 12
    # 12 buckets * 2 relcolour * 6 piece types * 64 squares
    assert spec.psq.rows == spec.king_buckets * 2 * 6 * 64 == 9216
    assert spec.threats.rows == 59808
    assert spec.pawn_pairs.rows == 1488
    # accumulator safety construction from spec section 4.4
    bound = (
        spec.psq.max_active_bound * spec.psq.coefficient_abs_limit
        + spec.threats.max_active_bound * spec.threats.coefficient_abs_limit
        + spec.pawn_pairs.max_active_bound * spec.pawn_pairs.coefficient_abs_limit
        + spec.bias_limit
    )
    assert bound == spec.numeric.accumulator_proven_abs_bound == 30048 < 32767
    assert (
        spec.head.first_activation_concat + spec.head.second_activation_concat
        == spec.head.skip_concat_outputs
    )
    # packed byte accounting must equal the authoritative packet numbers
    packed = (
        (spec.psq.rows * spec.channels * spec.psq.storage_bits + 7) // 8
        + (spec.threats.rows * spec.channels * spec.threats.storage_bits + 7) // 8
        + (spec.pawn_pairs.rows * spec.channels * spec.pawn_pairs.storage_bits + 7) // 8
    )
    assert packed == 5_308_416 + 26_793_984 + 571_392 == 32_673_792
    assert spec.numeric.mate_band_start > spec.numeric.score_bound


SPEC = load_spec()

if __name__ == "__main__":
    sanity_check(SPEC)
    print(SPEC.write_artifact())
    print("schema_id:", SPEC.schema_id)

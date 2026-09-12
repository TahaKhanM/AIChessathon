"""Versioned, hash-checked model export loading.

Validate header and hashes at import, decode packed 9/7/6-bit fields once
into runtime int16/int8 arrays, warm every reachable specialization. See
docs/architecture.md sections 3.4 and 4.3.

The codec is the signed fixed-width contract of
``spec/RX_FINAL_PLAN/signed_packing_reference.py``: little-endian within
the byte stream (first field's least-significant bit is byte 0 bit 0),
two's-complement signed fields, zero terminal padding. ``pack_signed`` /
``unpack_signed`` are the vectorized production path; ``pack_signed_slow`` /
``unpack_signed_slow`` are the scalar fallback sharing the oracle's exact
semantics. tests/test_packing.py round-trips all three bit-for-bit against
the oracle.

All packed decoding is chunked: ``unpack_signed`` and ``read_model`` never
materialise more than ``DECODE_CHUNK_FIELDS`` fields of intermediates at
once, so a 30M-field section cannot produce the ~3.9 GB transient that
exceeds the platform's 2 GB RSS cap (measured on EPYC-Milan; see
``docs/architecture.md``). Decoding bounds are explicit below.

File layout (all integers little-endian)::

    header  = magic "RXF1" | u16 version | u16 flags | u16 channels
            | u32 psq_rows | u32 threat_rows | u32 pp_rows
            | u8 head_stacks | u8 head_outputs | u32 meta_len
            | u64 payload_len | 32-byte sha256(meta || payload)
    meta    = utf-8 JSON (untrusted size-bounded; scale/export metadata)
    payload = concatenated sections in SECTION order

Section kinds: ``p9``/``p7``/``p6`` = signed packed fields when
flags & FLAG_PACKED else the matching int16/int8 raw array; ``i8``/``i16``/
``i32`` = raw little-endian arrays. The packed payload is lossless relative
to the same bounded integer model, per storage.quality_claim.
"""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Any

import numpy as np

from engine.features import (
    BIAS_ABS_LIMIT,
    CHANNELS,
    HEAD_INPUTS,
    HEAD_L1,
    HEAD_L2,
    HEAD_OUT,
    HEAD_STACKS,
    PP_COEF_ABS_LIMIT,
    PP_ROWS,
    PP_RUNTIME_DTYPE,
    PP_STORAGE_BITS,
    PSQ_COEF_ABS_LIMIT,
    PSQ_ROWS,
    PSQ_RUNTIME_DTYPE,
    PSQ_STORAGE_BITS,
    THREAT_COEF_ABS_LIMIT,
    THREAT_ROWS,
    THREAT_RUNTIME_DTYPE,
    THREAT_STORAGE_BITS,
)

MAGIC = b"RXF1"
FORMAT_VERSION = 1
FLAG_PACKED = 1

_HEADER = struct.Struct("<4sHHHIIIBBIQ32s")
HEADER_BYTES = _HEADER.size


class ModelFormatError(ValueError):
    """Raised when a model blob fails header, hash or range validation."""


# ---------------------------------------------------------------------------
# Signed fixed-width codec (little-endian bit stream, two's complement)
# ---------------------------------------------------------------------------


def _field_range(bits: int) -> tuple[int, int]:
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


# Fields per decode chunk. A single-shot vectorized decode materialises
# count*bits uint64 intermediates — ~3.9 GB transient for the 30.6M-field
# threat section, over the platform's 2 GB cap. The bitstream is
# field-contiguous, so decoding in whole-field chunks keeps the oracle's
# exact semantics while bounding the per-chunk working set at ~175 MB
# (worst contract width: 9 bits). ``2**20`` is a multiple of 8, so chunk
# boundaries are byte-aligned for every supported field width (2..16):
# only the final chunk of a section can carry terminal padding, exactly as
# in a single-shot decode.
DECODE_CHUNK_FIELDS = 1 << 20


def pack_signed(values: Any, bits: int) -> bytes:
    """Vectorized signed packing; byte-identical to the spec oracle."""
    if not isinstance(bits, int) or isinstance(bits, bool) or not 2 <= bits <= 16:
        raise ValueError("bits must be an integer in [2, 16]")
    v = np.asarray(values)
    if v.dtype == np.bool_ or not np.issubdtype(v.dtype, np.integer):
        raise TypeError("values must be integers")
    v = v.astype(np.int64).ravel()
    n = v.size
    if n == 0:
        return b""
    lo, hi = _field_range(bits)
    if v.min() < lo or v.max() > hi:
        raise ValueError(f"value outside the signed {bits}-bit range")
    u = (v & ((1 << bits) - 1)).astype(np.uint64)
    planes = ((u[:, None] >> np.arange(bits, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8)
    return np.packbits(planes.ravel(), bitorder="little").tobytes()


def _unpack_chunk(data: bytes, count: int, bits: int) -> np.ndarray:
    """Decode one byte-aligned run of at most ``DECODE_CHUNK_FIELDS`` fields.

    This is the entire vectorized decode applied to a bounded slice; every
    larger decode is a loop over calls to this function, so no call site can
    reintroduce a section-scale uint64 materialisation.
    """
    if len(data) != (count * bits + 7) // 8:
        raise ValueError("packed byte length does not match count and bits")
    raw = np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8), bitorder="little")
    if count * bits:
        if raw[count * bits :].any():
            raise ValueError("nonzero terminal padding")
        fields = raw[: count * bits].reshape(count, bits).astype(np.uint64)
        u = (fields << np.arange(bits, dtype=np.uint64)).sum(axis=1)
        sign = np.uint64(1 << (bits - 1))
        return np.where(u & sign, u.astype(np.int64) - (1 << bits), u.astype(np.int64))
    return np.zeros(0, dtype=np.int64)


def unpack_signed(data: bytes, count: int, bits: int) -> np.ndarray:
    """Vectorized signed unpacking; identical semantics to the spec oracle.

    Decodes in ``DECODE_CHUNK_FIELDS``-field chunks: mid-chunks end on field
    boundaries (byte-aligned, zero padding) and the final chunk carries —
    and validates — the terminal padding bits, exactly as the oracle does on
    the whole stream.
    """
    if not isinstance(bits, int) or isinstance(bits, bool) or not 2 <= bits <= 16:
        raise ValueError("bits must be an integer in [2, 16]")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    if len(data) != (count * bits + 7) // 8:
        raise ValueError("packed byte length does not match count and bits")
    out = np.empty(count, dtype=np.int64)
    for start in range(0, count, DECODE_CHUNK_FIELDS):
        c = min(DECODE_CHUNK_FIELDS, count - start)
        b0 = start * bits // 8
        b1 = ((start + c) * bits + 7) // 8
        out[start : start + c] = _unpack_chunk(data[b0:b1], c, bits)
    return out


def pack_signed_slow(values, bits: int) -> bytes:
    """Scalar fallback sharing the oracle's exact semantics."""
    if not isinstance(bits, int) or isinstance(bits, bool) or not 2 <= bits <= 16:
        raise ValueError("bits must be an integer in [2, 16]")
    lo, hi = _field_range(bits)
    mask = (1 << bits) - 1
    out = bytearray()
    buffer = available = 0
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("values must be Python integers")
        if not lo <= value <= hi:
            raise ValueError(f"{value} is outside the signed {bits}-bit range")
        buffer |= (value & mask) << available
        available += bits
        while available >= 8:
            out.append(buffer & 255)
            buffer >>= 8
            available -= 8
    if available:
        out.append(buffer)
    return bytes(out)


def unpack_signed_slow(data: bytes, count: int, bits: int) -> list[int]:
    """Scalar fallback sharing the oracle's exact semantics."""
    if not isinstance(bits, int) or isinstance(bits, bool) or not 2 <= bits <= 16:
        raise ValueError("bits must be an integer in [2, 16]")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if len(data) != (count * bits + 7) // 8:
        raise ValueError("packed byte length does not match count and bits")
    mask, sign, modulus = (1 << bits) - 1, 1 << (bits - 1), 1 << bits
    out: list[int] = []
    buffer = available = offset = 0
    for _ in range(count):
        while available < bits:
            buffer |= data[offset] << available
            available += 8
            offset += 1
        value = buffer & mask
        buffer >>= bits
        available -= bits
        out.append(value - modulus if value & sign else value)
    if buffer:
        raise ValueError("nonzero terminal padding")
    return out


# ---------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------

# (field name, kind, shape, coefficient bound). ``pK`` packs at K bits when
# FLAG_PACKED is set and stores the runtime dtype raw otherwise.
SECTIONS: tuple[tuple[str, str, tuple[int, ...], int | None], ...] = (
    ("bias", "i16", (CHANNELS,), BIAS_ABS_LIMIT),
    ("psq", "p9", (PSQ_ROWS, CHANNELS), PSQ_COEF_ABS_LIMIT),
    ("thr", "p7", (THREAT_ROWS, CHANNELS), THREAT_COEF_ABS_LIMIT),
    ("pp", "p6", (PP_ROWS, CHANNELS), PP_COEF_ABS_LIMIT),
    ("head_w1", "i8", (HEAD_STACKS, HEAD_INPUTS, HEAD_L1), 127),
    ("head_b1", "i32", (HEAD_STACKS, HEAD_L1), None),
    ("head_w2", "i8", (HEAD_STACKS, HEAD_L2, HEAD_L2), 127),
    ("head_b2", "i32", (HEAD_STACKS, HEAD_L2), None),
    ("head_w3", "i8", (HEAD_STACKS, HEAD_L1 * 2 + HEAD_L2 * 2, HEAD_OUT), 127),
    ("head_b3", "i32", (HEAD_STACKS, HEAD_OUT), None),
    ("psqt_w", "i16", (PSQ_ROWS, HEAD_STACKS), None),
    ("psqt_b", "i32", (HEAD_STACKS,), None),
)

_KIND_RUNTIME = {"p9": PSQ_RUNTIME_DTYPE, "p7": THREAT_RUNTIME_DTYPE, "p6": PP_RUNTIME_DTYPE}
_KIND_BITS = {"p9": PSQ_STORAGE_BITS, "p7": THREAT_STORAGE_BITS, "p6": PP_STORAGE_BITS}
_KIND_RAW = {"i8": np.int8, "i16": np.int16, "i32": np.int32}
_KIND_FALLBACK = {"p9": np.int16, "p7": np.int8, "p6": np.int8}


class Model(dict):
    """A dict of named runtime arrays plus a ``_meta`` export record."""


def _encode_section(arr: np.ndarray, kind: str, packed: bool) -> bytes:
    if kind in _KIND_BITS and packed:
        return pack_signed(arr.ravel(), _KIND_BITS[kind])
    dtype = _KIND_FALLBACK.get(kind, _KIND_RAW.get(kind))
    if dtype is None:
        raise ModelFormatError(f"unknown section kind {kind}")
    return np.ascontiguousarray(arr.astype(dtype)).tobytes()


def _decode_section(
    buf: memoryview,
    name: str,
    kind: str,
    count: int,
    packed: bool,
    limit: int | None,
) -> tuple[np.ndarray, int]:
    """Decode one section to its runtime dtype; returns (flat, nbytes).

    Packed sections stream through ``_unpack_chunk`` in
    ``DECODE_CHUNK_FIELDS``-field chunks directly into the runtime array —
    the full-section int64 frame never exists. Coefficient bounds are
    enforced on each decoded chunk *before* the narrowing cast.
    """
    if kind in _KIND_BITS and packed:
        bits = _KIND_BITS[kind]
        nbytes = (count * bits + 7) // 8
        out = np.empty(count, dtype=_KIND_RUNTIME[kind])
        for start in range(0, count, DECODE_CHUNK_FIELDS):
            c = min(DECODE_CHUNK_FIELDS, count - start)
            b0 = start * bits // 8
            b1 = ((start + c) * bits + 7) // 8
            flat = _unpack_chunk(bytes(buf[b0:b1]), c, bits)
            if limit is not None:
                peak = int(np.abs(flat).max()) if flat.size else 0
                if peak > limit:
                    raise ModelFormatError(f"section {name}: |{peak}| exceeds bound {limit}")
            out[start : start + c] = flat
        return out, nbytes
    dtype = _KIND_FALLBACK.get(kind, _KIND_RAW.get(kind))
    itemsize = np.dtype(dtype).itemsize
    nbytes = count * itemsize
    flat = np.frombuffer(bytes(buf[:nbytes]), dtype=dtype)
    if limit is not None and flat.size:
        peak = int(np.abs(flat.astype(np.int64)).max())
        if peak > limit:
            raise ModelFormatError(f"section {name}: |{peak}| exceeds bound {limit}")
    runtime = _KIND_RUNTIME.get(kind, dtype)
    return flat.astype(runtime), nbytes


def write_model(model: dict, *, packed: bool = True, meta: dict | None = None) -> bytes:
    """Serialize a model to the canonical container; validates ranges first."""
    payload = bytearray()
    for name, kind, shape, limit in SECTIONS:
        arr = np.asarray(model[name])
        if arr.shape != shape:
            raise ModelFormatError(f"section {name}: shape {arr.shape} != {shape}")
        if limit is not None:
            peak = int(np.abs(arr.astype(np.int64)).max()) if arr.size else 0
            if peak > limit:
                raise ModelFormatError(f"section {name}: |{peak}| exceeds bound {limit}")
        payload += _encode_section(arr, kind, packed)
    # FR4: ``meta or ...`` would silently drop an explicitly empty dict
    # (falsy) back to ``model["_meta"]``.  Explicit {} must write {} —
    # the loader-side leaf defaults are the deployed constants anyway.
    meta_json = json.dumps(
        model.get("_meta", {}) if meta is None else meta, sort_keys=True
    ).encode()
    flags = FLAG_PACKED if packed else 0
    digest = hashlib.sha256(meta_json + payload).digest()
    header = _HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        flags,
        CHANNELS,
        PSQ_ROWS,
        THREAT_ROWS,
        PP_ROWS,
        HEAD_STACKS,
        HEAD_OUT,
        len(meta_json),
        len(payload),
        digest,
    )
    return header + meta_json + payload


def read_model(blob: bytes, *, require_canonical: bool = True) -> Model:
    """Decode once at init: header validation, hash check, range checks."""
    if len(blob) < HEADER_BYTES:
        raise ModelFormatError("model shorter than header")
    (
        magic,
        version,
        flags,
        channels,
        psq_rows,
        threat_rows,
        pp_rows,
        stacks,
        outputs,
        meta_len,
        payload_len,
        digest,
    ) = _HEADER.unpack(blob[:HEADER_BYTES])
    if magic != MAGIC:
        raise ModelFormatError("bad magic")
    if version != FORMAT_VERSION:
        raise ModelFormatError(f"unsupported format version {version}")
    if flags & ~FLAG_PACKED:
        raise ModelFormatError(f"unknown flag bits {flags:#x}")
    if require_canonical and (channels, psq_rows, threat_rows, pp_rows, stacks, outputs) != (
        CHANNELS,
        PSQ_ROWS,
        THREAT_ROWS,
        PP_ROWS,
        HEAD_STACKS,
        HEAD_OUT,
    ):
        raise ModelFormatError("model dimensions do not match the canonical F512 schema")
    if len(blob) != HEADER_BYTES + meta_len + payload_len:
        raise ModelFormatError("model length does not match header")
    meta_bytes = blob[HEADER_BYTES : HEADER_BYTES + meta_len]
    payload = blob[HEADER_BYTES + meta_len :]
    if hashlib.sha256(meta_bytes + payload).digest() != digest:
        raise ModelFormatError("payload hash mismatch")
    meta = json.loads(meta_bytes or b"{}")

    packed = bool(flags & FLAG_PACKED)
    model = Model()
    view = memoryview(payload)
    offset = 0
    for name, kind, shape, limit in SECTIONS:
        count = int(np.prod(shape))
        flat, nbytes = _decode_section(view[offset:], name, kind, count, packed, limit)
        model[name] = flat.reshape(shape)
        offset += nbytes
    if offset != payload_len:
        raise ModelFormatError("trailing bytes after last section")
    model["_meta"] = meta
    return model


# ---------------------------------------------------------------------------
# Byte accounting (mirrors validate_spec.payload; tests diff it vs the JSON)
# ---------------------------------------------------------------------------


def payload_accounting(
    channels: int = CHANNELS,
    *,
    packed: bool = True,
    threat_width: int | None = None,
    threat_bits: int | None = None,
    routes: int = 0,
) -> dict:
    """Numeric payload accounting for width `channels` (F512 default)."""
    h = channels
    r = h if threat_width is None else threat_width
    tbits = (7 if packed else 8) if threat_bits is None else threat_bits
    psq = (9216 * h * (9 if packed else 16)) // 8
    thr = (59808 * r * tbits) // 8
    pp = (1488 * h * (6 if packed else 8)) // 8
    head_w = 8 * (h * 16 + 32 * 32 + 96 * 4)
    head_b = 8 * (16 + 32 + 4) * 4
    tbias = 2 * h
    psqt = 9216 * 8 * 2 + 8 * 4
    total = psq + thr + pp + head_w + head_b + tbias + psqt + routes
    return {
        "psq_bytes": psq,
        "threat_bytes": thr,
        "pawn_pair_bytes": pp,
        "routing_bytes": routes,
        "head_weight_bytes": head_w,
        "head_bias_bytes": head_b,
        "transform_bias_bytes": tbias,
        "psqt_bytes": psqt,
        "numeric_payload_bytes": total,
        "scalar_dense_head_macs": h * 16 + 32 * 32 + 96,
    }


if __name__ == "__main__":
    for p in (False, True):
        acc = payload_accounting(512, packed=p)
        print(f"packed={p}: {acc}")

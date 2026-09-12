"""W02 signed 9/7/6-bit codec vs the spec oracle, container validation and
byte accounting. Gate 2: bit-for-bit round-trips against
spec/RX_FINAL_PLAN/signed_packing_reference.py, exhaustive for 6 and 7
bits, >=10,000,000 random values for 9 bits. Gate 3: byte accounting equals
architecture.json for both the raw and packed payloads."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from engine import model_io as M
from engine import features as F

_SPEC_DIR = Path(__file__).resolve().parents[1] / "spec" / "RX_FINAL_PLAN"
sys.path.insert(0, str(_SPEC_DIR))
import signed_packing_reference as ORACLE  # noqa: E402

ARCH = json.loads((_SPEC_DIR / "architecture.json").read_text())


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.PCG64(seed))


def _numeric(d: dict) -> dict:
    return {k: v for k, v in d.items() if isinstance(v, int)}


def _diff(got: dict, ref: dict) -> dict:
    return {k: (got.get(k), ref.get(k)) for k in ref if got.get(k) != ref.get(k)}


def _oracle_pack(vals, bits):
    return ORACLE.pack_signed([int(v) for v in vals], bits)


def _oracle_unpack(data, count, bits):
    return ORACLE.unpack_signed(data, count, bits)


# ---------------------------------------------------------------------------
# Gate 2a: golden vectors + exhaustive 6/7-bit round-trips
# ---------------------------------------------------------------------------


def test_golden_vectors() -> None:
    assert M.pack_signed([-1], 6) == b"\x3f"
    assert M.pack_signed([-1], 7) == b"\x7f"
    assert M.pack_signed([-1], 9) == b"\xff\x01"
    assert M.pack_signed([0, 1], 9) == b"\x00\x02\x00"
    assert M.pack_signed_slow([-1], 6) == b"\x3f"
    assert M.unpack_signed(b"\x3f", 1, 6).tolist() == [-1]
    assert M.unpack_signed_slow(b"\x3f", 1, 6) == [-1]


@pytest.mark.parametrize("bits", [6, 7])
def test_exhaustive_small_widths(bits: int) -> None:
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    full = list(range(lo, hi + 1))
    # every value, single-field round-trips
    for v in full:
        blob = M.pack_signed([v], bits)
        assert blob == _oracle_pack([v], bits)
        assert M.unpack_signed(blob, 1, bits).tolist() == [v]
        assert M.pack_signed_slow([v], bits) == blob
        assert M.unpack_signed_slow(blob, 1, bits) == [v]
    # every ordered pair
    pairs = [v for a in full for b in full for v in (a, b)]
    blob = M.pack_signed(pairs, bits)
    assert blob == _oracle_pack(pairs, bits)
    assert M.unpack_signed(blob, len(pairs), bits).tolist() == pairs
    # every triple for 6-bit, the full range as one stream for 7-bit
    triples = [v for a in full for b in full for c in (lo, 0, hi) for v in (a, b, c)]
    blob = M.pack_signed(triples, bits)
    assert blob == _oracle_pack(triples, bits)
    assert M.unpack_signed(blob, len(triples), bits).tolist() == triples


def test_random_9bit_ten_million() -> None:
    """Gate 2b: >=10,000,000 random 9-bit values, bit-for-bit vs the oracle."""
    rng = _rng(20260911)
    total = 0
    for _ in range(20):  # 20 x 500,001 = 10,000,020 values
        vals = rng.integers(-256, 256, size=500_001)
        ours = M.pack_signed(vals, 9)
        theirs = _oracle_pack(vals, 9)
        assert ours == theirs
        assert M.unpack_signed(ours, vals.size, 9).tolist() == [int(v) for v in vals]
        assert _oracle_unpack(ours, vals.size, 9) == [int(v) for v in vals]
        total += vals.size
    print(f"\n9-bit random round-trips vs oracle: {total:,} values, bit-identical")
    assert total >= 10_000_000


def test_codec_rejects() -> None:
    for bits, bad in ((6, -33), (6, 32), (7, -65), (7, 64), (9, -257), (9, 256)):
        with pytest.raises(ValueError):
            M.pack_signed([bad], bits)
        with pytest.raises(ValueError):
            _oracle_pack([bad], bits)
    with pytest.raises(ValueError):
        M.unpack_signed(b"", 1, 6)  # too short
    with pytest.raises(ValueError):
        M.unpack_signed(b"\x00\x00", 1, 6)  # too long
    with pytest.raises(ValueError):
        M.unpack_signed(b"\xff\xff", 1, 6)  # nonzero terminal padding
    with pytest.raises(ValueError):
        _oracle_unpack(b"\xff\xff", 1, 6)
    with pytest.raises(TypeError):
        M.pack_signed([1.5], 9)
    with pytest.raises(ValueError):
        M.pack_signed([0], 1)  # bits out of [2, 16]


def test_codec_contract_limits_are_tighter_than_field() -> None:
    """9-bit fields hold [-256,255] but the contract allows only |v|<=255."""
    blob = M.pack_signed([-256], 9)  # representable
    assert M.unpack_signed(blob, 1, 9).tolist() == [-256]
    bad = _zeroed_model()
    bad["psq"] = np.full((F.PSQ_ROWS, F.CHANNELS), -256, np.int16)
    with pytest.raises(M.ModelFormatError):
        M.write_model(bad)


# ---------------------------------------------------------------------------
# Gate 3: byte accounting vs architecture.json, raw and packed
# ---------------------------------------------------------------------------


def test_byte_accounting_matches_architecture_json() -> None:
    raw = M.payload_accounting(512, packed=False)
    packed = M.payload_accounting(512, packed=True)
    raw_ref = _numeric(ARCH["reference_raw_accounting"])
    packed_ref = _numeric(ARCH["reference_packed_accounting"])
    print("\nraw   :", raw)
    print("ref   :", raw_ref)
    assert raw == raw_ref, _diff(raw, raw_ref)
    print("packed:", packed)
    print("ref   :", packed_ref)
    assert packed == packed_ref, _diff(packed, packed_ref)
    # Challenger widths also reproduce, proving the accounting is
    # schema-derived rather than hardcoded (same checks as validate_spec).
    ch = {c["name"]: c for c in ARCH["principal_challengers"]}
    for name, h in (
        ("F384 dense early", 384),
        ("F640 dense early", 640),
        ("F768 dense early", 768),
    ):
        for flag, key in ((False, "accounting_raw"), (True, "accounting_packed")):
            got = M.payload_accounting(h, packed=flag)
            ref = _numeric(ch[name][key])
            assert got == ref, (name, key, _diff(got, ref))
    int4 = M.payload_accounting(768, packed=True, threat_bits=4)
    ref_int4 = ch["F768 learned int4 threat, exact9bit PSQ and6bit PP"]["accounting"]
    assert int4["numeric_payload_bytes"] == ref_int4["numeric_payload_bytes"]
    b768 = M.payload_accounting(768, packed=True, threat_width=128, routes=59808)
    ref_b768 = ch["B768 blocked early"]["accounting_packed"]
    assert b768["numeric_payload_bytes"] == ref_b768["numeric_payload_bytes"]


def _zeroed_model(**overrides) -> dict:
    model = {
        "bias": np.zeros(F.CHANNELS, np.int16),
        "psq": np.zeros((F.PSQ_ROWS, F.CHANNELS), np.int16),
        "thr": np.zeros((F.THREAT_ROWS, F.CHANNELS), np.int8),
        "pp": np.zeros((F.PP_ROWS, F.CHANNELS), np.int8),
        "head_w1": np.zeros((8, 512, 16), np.int8),
        "head_b1": np.zeros((8, 16), np.int32),
        "head_w2": np.zeros((8, 32, 32), np.int8),
        "head_b2": np.zeros((8, 32), np.int32),
        "head_w3": np.zeros((8, 96, 4), np.int8),
        "head_b3": np.zeros((8, 4), np.int32),
        "psqt_w": np.zeros((9216, 8), np.int16),
        "psqt_b": np.zeros((8,), np.int32),
        "_meta": {"name": "w02-zero", "scale_num": 1, "scale_shift": 0},
    }
    for k, v in overrides.items():
        model[k] = v
    return model


def _edge_model() -> dict:
    """Every section at its declared limit values."""
    m = _zeroed_model()
    m["bias"][:] = -2040
    m["psq"][:] = 255
    m["thr"][:] = -63
    m["pp"][:] = 31
    return m


_META_JSON = b'{"name": "w02-zero", "scale_num": 1, "scale_shift": 0}'


def test_model_roundtrip_packed_sizes() -> None:
    blob = M.write_model(_edge_model(), packed=True)
    expected = ARCH["reference_packed_accounting"]["numeric_payload_bytes"]
    assert len(blob) == M.HEADER_BYTES + len(_META_JSON) + expected
    model = M.read_model(blob)
    assert model["psq"].dtype == np.int16
    assert model["thr"].dtype == np.int8
    assert model["pp"].dtype == np.int8
    for name in ("psq", "thr", "pp", "bias", "head_w3", "psqt_w"):
        np.testing.assert_array_equal(model[name], _edge_model()[name])
    assert model["_meta"]["name"] == "w02-zero"


def test_model_roundtrip_raw_sizes() -> None:
    blob = M.write_model(_edge_model(), packed=False)
    meta_len = len(b'{"name": "w02-zero", "scale_num": 1, "scale_shift": 0}')
    expected = ARCH["reference_raw_accounting"]["numeric_payload_bytes"]
    assert len(blob) == M.HEADER_BYTES + meta_len + expected
    model = M.read_model(blob)
    np.testing.assert_array_equal(model["psq"], _edge_model()["psq"])


def test_model_random_roundtrip() -> None:
    rng = _rng(11)
    m = _zeroed_model()
    m["bias"] = rng.integers(-2040, 2041, 512).astype(np.int16)
    m["psq"] = rng.integers(-255, 256, (9216, 512)).astype(np.int16)
    # only a slice randomized to keep the 30M-value pack quick
    m["thr"][:1024] = rng.integers(-63, 64, (1024, 512)).astype(np.int8)
    m["pp"] = rng.integers(-31, 32, (1488, 512)).astype(np.int8)
    blob = M.write_model(m, packed=True)
    back = M.read_model(blob)
    for name in ("bias", "psq", "thr", "pp", "head_w1", "head_b3", "psqt_w"):
        np.testing.assert_array_equal(back[name], m[name])


def test_model_header_and_hash_validation() -> None:
    blob = M.write_model(_zeroed_model(), packed=True)
    M.read_model(blob)
    # corrupt one payload byte -> hash mismatch
    bad = bytearray(blob)
    bad[-1] ^= 1
    with pytest.raises(M.ModelFormatError):
        M.read_model(bytes(bad))
    # bad magic
    bad = bytearray(blob)
    bad[0] = ord("X")
    with pytest.raises(M.ModelFormatError):
        M.read_model(bytes(bad))
    # truncated
    with pytest.raises(M.ModelFormatError):
        M.read_model(blob[:-5])
    with pytest.raises(M.ModelFormatError):
        M.read_model(blob[: M.HEADER_BYTES // 2])
    # version / dim corruption -> either hash or schema validation trips
    bad = bytearray(blob)
    bad[4] = 9  # version
    with pytest.raises(M.ModelFormatError):
        M.read_model(bytes(bad))
    bad = bytearray(blob)
    bad[16] ^= 0xFF  # psq_rows byte
    with pytest.raises(M.ModelFormatError):
        M.read_model(bytes(bad))
    # a header whose declared payload is shorter than the blob
    short = _zeroed_model()
    short["_meta"] = {}
    blob2 = M.write_model(short, packed=True)
    hacked = bytearray(blob2)
    hacked[8:12] = (256).to_bytes(4, "little")  # channels != 512
    with pytest.raises(M.ModelFormatError):
        M.read_model(bytes(hacked))


def test_model_hash_actually_verified() -> None:
    """Flip a bit inside a packed section and recompute nothing -> reject."""
    blob = M.write_model(_edge_model(), packed=True)
    mid = M.HEADER_BYTES + len(_META_JSON) + 1000
    bad = bytearray(blob)
    bad[mid] ^= 0x40
    with pytest.raises(M.ModelFormatError):
        M.read_model(bytes(bad))
    # and a forged hash is of course rejected too
    bad2 = bytearray(blob)
    bad2[mid] ^= 0x40
    bad2[M.HEADER_BYTES - 32 :] = hashlib.sha256(b"forgery").digest()
    with pytest.raises(M.ModelFormatError):
        M.read_model(bytes(bad2))

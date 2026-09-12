"""Signed fixed-width packing, bit-compatible with the packet oracle.

Semantics identical to ``spec/RX_FINAL_PLAN/signed_packing_reference.py``:
little-endian bit order within the byte stream, two's-complement signed
fields, strict range checks, nonzero terminal padding rejected.  The test
suite runs golden vectors against the oracle module directly.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def pack_signed(values: Iterable[int], bits: int) -> bytes:
    if not isinstance(bits, int) or isinstance(bits, bool) or not 2 <= bits <= 16:
        raise ValueError("bits must be an integer in [2, 16]")
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    mask = (1 << bits) - 1
    out = bytearray()
    buffer = available = 0
    for value in values:
        v = int(value)
        if not low <= v <= high:
            raise ValueError(f"{v} is outside the signed {bits}-bit range")
        buffer |= (v & mask) << available
        available += bits
        while available >= 8:
            out.append(buffer & 255)
            buffer >>= 8
            available -= 8
    if available:
        out.append(buffer)
    return bytes(out)


def pack_signed_array(values: np.ndarray, bits: int) -> bytes:
    """Vectorized equivalent of pack_signed over an int64 array."""
    vals = values.reshape(-1).astype(np.int64)
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if vals.size and (vals.min() < low or vals.max() > high):
        raise ValueError(f"values outside signed {bits}-bit range")
    mask = (1 << bits) - 1
    n = vals.size
    bitpos = np.arange(n, dtype=np.int64) * bits
    byte_idx = bitpos >> 3
    shift = bitpos & 7
    masked = (vals & mask).astype(np.uint64)
    nbytes = (n * bits + 7) // 8
    out = np.zeros(nbytes + 8, dtype=np.uint8)
    contrib = masked << shift.astype(np.uint64)
    # a <=16-bit field at a <=7-bit offset spans <=3 bytes
    for k in range(3):
        piece = ((contrib >> np.uint64(8 * k)) & np.uint64(255)).astype(np.uint8)
        np.bitwise_or.at(out, byte_idx + k, piece)
    return bytes(out[:nbytes])


def unpack_signed(data: bytes, count: int, bits: int) -> list[int]:
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


def unpack_signed_array(data: bytes, count: int, bits: int) -> np.ndarray:
    """Vectorized decoder; same contract as unpack_signed."""
    expected = (count * bits + 7) // 8
    if len(data) != expected:
        raise ValueError("packed byte length does not match count and bits")
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.uint64)
    bitpos = np.arange(count, dtype=np.int64) * bits
    byte_idx = bitpos >> 3
    shift = (bitpos & 7).astype(np.uint64)
    wide = np.zeros(count, dtype=np.uint64)
    for k in range(3):  # a <=16-bit field at a <=7-bit offset spans <=3 bytes
        idx = byte_idx + k
        valid = idx < len(arr)
        gathered = np.zeros(count, dtype=np.uint64)
        gathered[valid] = arr[idx[valid]]
        wide |= gathered << np.uint64(8 * k)
    mask = np.uint64((1 << bits) - 1)
    fields = (wide >> shift) & mask
    sign = np.uint64(1 << (bits - 1))
    modulus = np.int64(1 << bits)
    signed = fields.astype(np.int64)
    signed[fields & sign != 0] -= modulus
    # nonzero terminal padding check
    total_bits = count * bits
    if expected * 8 > total_bits:
        pad = int(data[-1]) >> (total_bits & 7)
        if pad:
            raise ValueError("nonzero terminal padding")
    return signed

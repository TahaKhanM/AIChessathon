"""Portable correctness reference for signed fixed-width model storage.

This is a small-vector oracle, NOT a benchmarked production loader. A release
loader should use a vectorized or JIT-compiled decoder, with the same golden
vectors and exact byte contract, and pass the 90-second cold-init limit.
"""

from __future__ import annotations

from collections.abc import Iterable


def pack_signed(values: Iterable[int], bits: int) -> bytes:
    """Pack two's-complement fields, least-significant bits/bytes first."""
    if not isinstance(bits, int) or isinstance(bits, bool) or not 2 <= bits <= 16:
        raise ValueError("bits must be an integer in [2, 16]")
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    mask = (1 << bits) - 1
    out = bytearray()
    buffer = available = 0
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("values must be Python integers")
        if not low <= value <= high:
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


def unpack_signed(data: bytes, count: int, bits: int) -> list[int]:
    """Decode exactly count fields; reject wrong length or nonzero padding."""
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

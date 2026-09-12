"""Native monotonic clock for compiled search.

The ``objmode`` clock bridge (``engine.clock.make_objmode_clock``) works but
drags the GIL and Python-object machinery into the recursive hot path, and on
aarch64 it sits behind an extra dispatcher boundary that proved flaky in
practice.  This module instead exposes ``clock_ns()`` as a direct
``clock_gettime(CLOCK_MONOTONIC)`` call via an ``llvmlite`` symbol binding —
a real C function call from JIT code, no objmode, no Python objects.

Epoch caveat: on Linux ``CLOCK_MONOTONIC`` is the same epoch as
``time.monotonic_ns``; on macOS it is not (``mach_absolute_time``).  The
compiled search therefore never compares against an absolute Python deadline —
``k_begin`` records ``clock_ns()`` as the iteration's start and the driver
passes a *budget* (``hard_ns - now``) — so the clock's epoch is internal and
self-consistent.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys

from llvmlite import binding, ir
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic

if sys.platform == "darwin":
    _CLOCK_MONOTONIC = 6
else:
    _CLOCK_MONOTONIC = 1


def _bind() -> None:
    path = ctypes.util.find_library("c")
    libc = ctypes.CDLL(path) if path else ctypes.CDLL(None)
    addr = ctypes.cast(libc.clock_gettime, ctypes.c_void_p).value
    binding.add_symbol("clock_gettime", addr)


_bind()


@intrinsic
def _clock_gettime_ns(typingctx):  # pragma: no cover - JIT internals
    sig = types.int64()

    def codegen(context, builder, sig, args):
        i64 = ir.IntType(64)
        i32 = ir.IntType(32)
        ts = builder.alloca(i64, size=2, name="ts")
        fnty = ir.FunctionType(i32, [i32, ir.PointerType(ir.IntType(8))])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "clock_gettime")
        builder.call(
            fn,
            [
                ir.Constant(i32, _CLOCK_MONOTONIC),
                builder.bitcast(ts, ir.PointerType(ir.IntType(8))),
            ],
        )
        sec = builder.load(builder.gep(ts, [ir.Constant(i32, 0)]))
        nsec = builder.load(builder.gep(ts, [ir.Constant(i32, 1)]))
        return builder.add(builder.mul(sec, ir.Constant(i64, 1_000_000_000)), nsec)

    return sig, codegen


@njit(cache=False)
def clock_ns() -> int:
    """Monotonic nanoseconds from ``clock_gettime`` — nopython, no GIL."""
    return _clock_gettime_ns()

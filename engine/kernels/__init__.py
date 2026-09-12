"""RX-FINAL compiled hot path (W08-numba).

Nopython kernels mirroring the pure-Python oracle in ``engine/``:

- ``layout``  — the flat-arena context contract (8-tuple of dtype arenas)
- ``bb``      — board representation, make/unmake/null, legal movegen, SEE
- ``tthist``  — transposition table + ordering/correction histories
- ``ev``      — F512-EF lazy accumulator stack + simple_eval + head
- ``search_nb`` — quiescence, PVS alpha-beta, deadline, rep/draw, ordering
- ``driver``  — ``CompiledSearcher``: python-side iterative-deepening driver

The pure-Python modules remain the scalar reference; every kernel is
validated against them.  See ``NUMBA_REPORT.md`` for gate results.
"""

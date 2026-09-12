"""Regression gate for the ``n_search`` cache-load SIGSEGV (numba#6061 class).

``engine.kernels.search_nb.n_search`` is self-recursive and is compiled with
``cache=True``.  When its recursive call sites produced *different* argument
type specialisations (literal ``0``/``1``/``1 - cut_node`` for
``is_pv``/``cut_node``), each specialisation's machine code referenced a
sibling specialisation's entry point through ``.numba.unresolved$`` symbols
that do not survive a cache round-trip.  On load those references resolve to
NULL and the very first ``n_search`` call in a fresh process segfaults
(``call_cfunc`` -> ``call NULL``) — reproduced 3/3 on macOS arm64 and on EPYC
Linux x86_64, for ``eval_kind`` 0 and 1.

This is the deployed-failure shape: a shipped package carries its numba cache
inside ``__pycache__``, so *every* game process would die at the first search
call.  A crash is an immediate loss under the competition contract.

Gate (two clean child processes, shared fresh ``NUMBA_CACHE_DIR``):

1. child A compiles ``n_search`` and runs real searches — must exit 0;
2. child B points at the same cache (load path) and runs searches for both
   ``eval_kind`` values plus node-limit aborts — must exit 0.

On the pre-fix tree child B exits with SIGSEGV (returncode -11) every time.
The test also asserts ``n_search`` has exactly one compiled specialisation —
the mechanism-level invariant that keeps the recursion intra-module.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

_CHILD_SRC = textwrap.dedent(
    """
    import faulthandler, json, sys, time
    faulthandler.enable()
    sys.path.insert(0, {repo!r})
    from engine.evaluate import EvalWeights
    from engine.kernels import search_nb as SN
    from engine.kernels.driver import CompiledSearcher
    from engine.state import GameState
    from engine.tt import TranspositionTable

    START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    KIWI = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    ABORT = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
    far = lambda: time.monotonic_ns() + 10**12

    out = {{"nsig": None, "searches": []}}
    for kind in (0, 1):
        w = EvalWeights.random(1234) if kind else None
        s = CompiledSearcher(tt=TranspositionTable(mib=4), check_mask=15,
                             weights=w, eval_kind=kind)
        for fen, d in ((START, 1), (START, 3), (KIWI, 3)):
            r = s.search(GameState.from_fen(fen), hard_ns=far(), max_depth=d)
            out["searches"].append(
                {{"kind": kind, "d": d, "move": int(r.move),
                  "score": int(r.score), "nodes": int(r.nodes),
                  "aborted": bool(r.aborted)}})
        # node-limited abort: unwinds the recursive stack mid-search
        r = s.search(GameState.from_fen(ABORT), hard_ns=far(),
                     max_depth=12, node_limit=251)
        out["searches"].append(
            {{"kind": kind, "abort": True, "aborted": bool(r.aborted),
              "move": int(r.move), "nodes": int(r.nodes)}})
    out["nsig"] = len(SN.n_search.signatures)
    print("CHILD-RESULT " + json.dumps(out), flush=True)
    """
)


def _run_child(cache_dir: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(
        [sys.executable, "-c", _CHILD_SRC.format(repo=str(_REPO))],
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )


def _child_payload(proc: subprocess.CompletedProcess) -> dict:
    for line in proc.stdout.splitlines():
        if line.startswith("CHILD-RESULT "):
            return json.loads(line[len("CHILD-RESULT ") :])
    return {}


@pytest.fixture(scope="module")
def warm_cache(tmp_path_factory) -> Path:
    """Child A: populate a fresh numba cache by compiling + searching."""
    cache = tmp_path_factory.mktemp("nbcache")
    proc = _run_child(cache)
    assert proc.returncode == 0, (
        "fresh-compile child crashed (not the cache-load bug — worse):\n" + proc.stderr[-3000:]
    )
    payload = _child_payload(proc)
    assert payload.get("searches"), "child produced no search results"
    return cache


def test_cache_load_search_does_not_segfault(warm_cache: Path) -> None:
    """Child B: load the compiled kernels from cache and search."""
    proc = _run_child(warm_cache)
    assert proc.returncode == 0, (
        f"cache-load child exited rc={proc.returncode} "
        "(SIGSEGV on cache load — the n_search recursion/cache bug):\n"
        + proc.stdout[-1500:]
        + "\n--- stderr ---\n"
        + proc.stderr[-3000:]
    )
    payload = _child_payload(proc)
    kinds = {s.get("kind") for s in payload["searches"]}
    assert kinds == {0, 1}, "must exercise eval_kind 0 and 1 from cache"
    assert all(s["move"] for s in payload["searches"]), "a cache-loaded search returned no move"


def test_n_search_single_specialization(warm_cache: Path) -> None:
    """The mechanism invariant: one compiled n_search signature, so every
    recursive edge is intra-module and cache-safe."""
    proc = _run_child(warm_cache)
    if proc.returncode != 0:
        pytest.skip("cache-load child crashed (covered by the load test)")
    payload = _child_payload(proc)
    assert payload["nsig"] == 1, (
        f"n_search compiled {payload['nsig']} specialisations — "
        "cross-specialisation recursion is what breaks the numba cache"
    )

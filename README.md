# Chessathon Engine

A Python and Numba chess engine exploring **decision quality under a fixed compute budget**. The project combines adversarial search, exact integer neural inference, and explicit treatment of latency, memory and uncertain state.

The public source includes a runnable classical baseline and the F512 neural evaluation implementation. Model checkpoints and private infrastructure are excluded. No new training is required to run the baseline, tests or benchmarks. This release does not claim a measured Elo or a tournament-ready neural model.

## Engineering highlights

| Problem | Implementation | Evidence to inspect |
|---|---|---|
| Search must return before the clock expires | Iterative deepening, monotonic deadlines, legal fallback, committed completed iterations | [Search](engine/search.py), [abort regressions](tests/test_search_abort.py) |
| A position's value depends on its history | Separate position identity, rule counters and repetition context; unknown history stays explicit | [State](engine/state.py), [transposition table](engine/tt.py) |
| Neural inference competes with search for CPU time | Incremental 512-channel integer accumulators, compiled kernels, scalar parity oracle | [Evaluator](engine/evaluate.py), [parity tests](tests/test_evaluate.py) |
| Model storage is bounded | Signed 9/7/6-bit packing, versioned containers, SHA-256 integrity checks | [Model I/O](engine/model_io.py), [codec tests](tests/test_packing.py) |
| Experimental results can leak across splits | Canonical position keys, source metadata, split checks and explicit label semantics | [Data contracts](training/records.py), [split logic](training/splits.py) |
| Faster kernels can fail only after restart | Independent-process compilation and cache-reload regression | [Cache test](tests/test_kernels_segload.py) |

These are also useful problems in quantitative systems: bounded decision latency, stateful computation, numerical reproducibility, and separating model quality from execution cost. Chess provides a concrete adversarial environment in which to test them.

## Run locally

Use CPython 3.12. The dependency versions match the development environment.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
make test
make lint
make benchmark
```

For an exact dependency resolution, `uv sync --locked --extra dev` uses the checked-in lockfile.

Get a move without downloading weights:

```python
from agent import get_move

fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
print(get_move(fen, time_left_ms=1_000))
```

The adapter owns state for one game. Use a fresh process per game and call it serially. It returns a UCI move; `0000` denotes a position with no legal move. Invalid FEN input raises an exception. The default evaluation is material plus piece-square terms, so the demo makes no neural-strength claim.

## Read the system

Start with [architecture and tradeoffs](docs/architecture.md), then [verification and measurement](docs/verification.md). The [benchmark runner](bench/run.py) emits structured results with environment metadata and individual samples. [Contributing](CONTRIBUTING.md) describes the local checks and review expectations.

```text
agent.py           runnable, single-game reference adapter
engine/            board, search, state, evaluation, model I/O
  kernels/         compiled search and evaluation kernels
  clock_b/         soft allocation and deadline utilities
training/          retained numerical and data-pipeline source; no datasets or checkpoints
spec/              machine-readable evaluator contracts and codec oracle
tests/             differential, numerical, corruption and failure-path tests
bench/             reproducible fixed-depth benchmark
```

The numerical contract is F512-EF-K12-16/32: 12 king buckets, 512 shared feature channels, eight material heads, and exact clipped integer activations. The numeric payload occupies 32,900,768 bytes before container metadata; tests verify the accounting. A compatible privately retained model can be loaded through `engine.agent_rx.init(model_path)`. The default adapter does not silently substitute random weights.

## Scope and limitations

The reference search executes in Python; the neural kernels and separate compiled search backend use Numba. Correctness tests are distinct from playing-strength evidence. Local benchmark timings are hardware-specific and exclude the competition sandbox. Historical calibration constants remain heuristics, not validated performance guarantees.

The original deployment target was one CPU core, 2 GB RAM, and a 50,000,000-byte unpacked submission. The [current competition documentation](https://aichessathon.com/docs), checked 12 September 2026, specifies 90-second qualifier initialization and 30-second final initialization. This source release has not been requalified for that final budget.

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md) for component attribution. No upstream chess network weights are distributed.

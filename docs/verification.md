# Verification and measurement

The public repository keeps the machinery for checking correctness and measuring changes, but it does not commit historical playing-strength or local benchmark results.

Run the default checks from a CPython 3.12 environment installed with `.[dev]`:

```sh
make lint
make test
```

The suite requires no cloud credentials, network downloads or pretrained weights.

## What the tests cover

| Layer | Checks |
|---|---|
| Board and move generation | Reference perft positions, legal move sets against python-chess, special moves and make/unmake restoration |
| State and cache | History reconciliation, repetition context, mate normalization, draw counters and context-sensitive reuse |
| Search | Completed-iteration commitment, forced aborts, legal fallback, null-move semantics and restoration |
| Evaluator | Scalar/optimized parity, incremental/full refresh agreement, clipping and overflow boundaries |
| Serialization | Signed packing, corruption handling, format rejection and export/runtime parity |
| Data | Label perspective, bound and censoring semantics, deterministic split assignment and leakage checks |
| Compiled runtime | Fresh-process compilation followed by a separate process loading the generated cache |

The extended profile adds deeper perft and randomized stress cases:

```sh
make test-deep
```

## Local benchmarking

```sh
make benchmark
.venv/bin/python -m bench.run --depth 3 --repeats 5 --output bench/results/local.json
```

The benchmark runner uses deterministic positions and records source revision, dependency versions, platform information and backend configuration alongside the samples.

These measurements are intended for controlled comparisons between changes. A fixed-depth timing or nodes-per-second number is not a playing-strength estimate, and a model with lower validation loss is not automatically a stronger complete engine.

For deployment work, measure the whole package rather than only the search loop. Initialization, model loading, maximum resident memory, package size, deadline behaviour and actual-clock games all matter under the competition contract.

## Review expectations

Search and numerical changes should include a regression that would fail under the old behaviour. Prefer deterministic seeds and independent reference paths where possible.

Avoid assertions about exact wall-clock timings inside unit tests. Keep timing measurements in explicit local benchmark outputs with enough metadata to reproduce the environment.

CI runs lint, formatting, the default suite, package builds and an installed-wheel smoke test on Python 3.12.

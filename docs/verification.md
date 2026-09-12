# Verification and measurement

Run `make test` from a CPython 3.12 environment installed with `.[dev]`. The suite requires no cloud credentials, network downloads, pretrained weights or external referee checkout. Small synthetic numerical training steps exercise the reference pipeline; they do not train a release model.

## What the tests establish

| Layer | Checks |
|---|---|
| Board and move generation | Reference perft counts, legal move sets against python-chess, special moves, make/unmake restoration |
| State and cache | History reconciliation, unknown prefix, mate normalization, draw counters and context-sensitive reuse |
| Search | Completed-iteration commitment, forced aborts, fallback legality, null-move semantics and restoration |
| Evaluator | Scalar/optimized parity, incremental/full refresh agreement, clipping and overflow boundaries |
| Serialization | Golden vectors, exhaustive narrow signed domains, corruption and format rejection |
| Data | Label perspective, bound/censoring semantics, sentinel rejection, deterministic splits and export parity |
| Compiled runtime | Fresh-process compilation followed by a separate process loading the same cache |

The default numerical and compiled suite can take several minutes. Eight extended cases are explicitly excluded from the default run: six exhaustive perft positions, 20,000 randomized aborts, and 50,000 terminal-state comparisons. The default suite retains 256 randomized aborts, a smaller terminal-state sample with the same explicit boundary families, and shallow reference perft counts. Run `make test-deep` for the extended profile; it can take hours in the Python reference backend. Local test fixtures include a small set of historical competition PGNs for parser and record-contract tests; these are test inputs, not a training corpus or a strength benchmark.

## Reproduce timing

```sh
make benchmark
.venv/bin/python -m bench.run --depth 3 --repeats 5 --output bench/results/local.json
```

The runner uses deterministic fixed positions, a fresh 8 MiB table for each sample and the classical reference evaluator. It warms the path once, then reports each sample and aggregate median/p95 elapsed time. It records interpreter, package versions, platform, source revision, source SHA-256 and dirty-tree state. Percentiles over a few samples are descriptive only.

These measurements exclude imports, sandbox startup, neural-model loading and compilation. They are not target-EPYC qualification results, and nodes per second are not an Elo estimate. For a deployed release, separately measure cold initialization, maximum RSS, package bytes, deadline overruns and actual-clock paired games on target hardware.

## Review expectations

A change to search or numerical semantics should include a regression that can fail under the old behaviour. Use deterministic seeds and compare against an independent oracle where possible. Avoid assertions about exact wall-clock timings in unit tests. Put measurements in explicit benchmark output with environment metadata.

CI runs lint, formatting and the default suite on Python 3.12. It also builds a wheel and exercises the installed package outside the checkout to catch missing package data. CI results are available only after the branch is pushed and workflows have run.

## Local publication checks

[Verification record](verification-results.json): 248 default cases covered across a full run and a focused rerun. The full run passed 247 cases and exposed one test tied to an excluded historical encoder. That test was replaced with a regression through the public batch builder; all six sentinel tests then passed. Eight extended stress cases were not run. Lint, formatting, wheel/sdist builds and an installed-wheel smoke test outside the checkout passed. This records local verification, not a GitHub CI or target-hardware result.

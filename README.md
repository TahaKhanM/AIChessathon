# Miskeen | AI Chessathon

[![Checks](https://github.com/TahaKhanM/AIChessathon/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/TahaKhanM/AIChessathon/actions/workflows/ci.yml)

**2nd out of ~500 teams in the Global AIChessathon Final Qualification**

I built Miskeen around a simple question: how strong a chess engine could I make when the deployed agent gets one CPU core, 2 GB of RAM, no network, no GPU and a 50 MB package limit?

By v4, the engine combined a custom Numba-compiled bitboard searcher with an incrementally updated NNUE-style evaluator that I trained from scratch. The hard part was not making either half impressive in isolation. It was getting the **search, model, memory layout and clock to work as one system**.

A better evaluator is not better if it makes search too slow. A pruning rule is not useful if it saves nodes by deleting the wrong branch. A model that looks better offline is not a stronger chess engine until the complete packaged agent holds up under the real runtime constraints.

This repository is a cleaned public release rather than a byte-for-byte tournament bundle. It keeps the search, runtime, training and verification work, while tournament weights, private data and historical match outputs are intentionally omitted. It also contains later evaluator research, which I label separately instead of presenting it as the v4 architecture.

## At a glance

| Area | v4 approach |
|---|---|
| Search | Iterative-deepening PVS / alpha-beta with aspiration, quiescence, transposition-table reuse and selective pruning |
| State | Custom bitboards, reversible make/unmake, repetition and draw context, legal root fallback |
| Evaluation | K16 × 768 sparse features per perspective → H1024 SCReLU → 8 piece-count / material output buckets |
| Runtime | Numba hot path, preallocated NumPy storage, incremental integer accumulators, explicit deadline handling |
| Training | Roughly billion-record mixed pipeline, own teacher-labelled positions + public Lichess evaluations, quantization-aware training |
| Validation | Paired full-engine games at the competition clock, target-CPU checks and package-level qualification |

## Why the constraints mattered

The competition runtime was deliberately tight:

- one AMD EPYC CPU core
- 2 GB RAM
- no network and no GPU during play
- Python 3.12 with NumPy and Numba available
- 50 MB maximum unpacked submission
- 120 seconds plus 0.5 seconds per move

That changes how you design the engine. Search, inference, memory layout and time management are not independent problems when they all spend the same CPU budget.

For me, the useful objective became **expected playing strength per unit of runtime**, not model accuracy, nodes per second or any other single benchmark by itself.

## v4 architecture

During the final development window, the stable v4 line used a K16 NNUE with a direct H1024 SCReLU head on top of the search engine. The exact locked tournament checkpoint is not distributed here, but the architecture and development process are described below.

```mermaid
flowchart LR
    A[FEN + remaining clock] --> B[Bitboard state + history]
    B --> C[Iterative deepening PVS]
    C <--> D[Transposition table]
    C <--> E[Make / unmake]
    E <--> F[Incremental NNUE accumulators]
    F --> G[Integer evaluation]
    G --> C
    H[Soft budget + hard deadline] --> C
    C --> I[Last completed iteration]
    I --> J[Legal UCI move]
```

### Search

The searcher is an iterative-deepening principal variation search built on alpha-beta. Quiescence handles unstable tactical leaves, while aspiration windows and move ordering try to make the useful branch fail high as early as possible.

Around that core, the competition-era search used transposition-table cutoffs and static-eval reuse, verified null-move pruning, reverse futility and razoring, internal iterative reduction, late-move reductions and pruning, static exchange evaluation, history-based ordering and a small number of targeted extensions.

I treated those as heuristics rather than correctness rules. Reduced searches can be wrong. Cached scores can become unsafe when draw context changes. Null moves belong to the search tree, not played history. The engine therefore separates move-ordering hints from score reuse and keeps enough game context to avoid treating every geometrically identical board as equivalent.

Cancellation was equally important. Before expensive work starts, the engine already has a legal fallback. It only publishes a principal variation from a fully completed iteration. If the hard deadline fires halfway through the next depth, board state and evaluator state unwind together and the previous result survives.

The public reference implementation lives in [`engine/search.py`](engine/search.py), [`engine/state.py`](engine/state.py), [`engine/tt.py`](engine/tt.py) and [`engine/board.py`](engine/board.py).

### NNUE evaluation

The v4 evaluator was designed for the workload a chess search actually creates: batch size one, millions of closely related positions and a very small latency budget per leaf.

For each perspective, it used **16 king buckets × 768 piece-square features**, giving a sparse 12,288-row feature space. Active rows were accumulated into a hidden layer and updated as moves were made and unmade rather than rebuilt from scratch at every node.

```text
sparse king-relative piece-square features
        ↓
white and black perspective accumulators
        ↓
H1024 SCReLU hidden transform
        ↓
8 piece-count / material output buckets
        ↓
integer scalar evaluation
```

Weights and accumulators were integer-valued at runtime, with widened reductions where needed. The point was not just to make the network small. It was to make it cheap enough that better positional information still translated into useful search depth.

One of the most useful performance lessons came from something much less glamorous than the network design. A multidimensional indexing pattern in the neural hot loop stopped LLVM from vectorising the dot product effectively on the target EPYC. Reworking the same computation around contiguous one-dimensional row views removed that bottleneck without changing the model. Profiling-driven changes like that became a recurring part of v4.

The public tree now contains a later experimental integer evaluator under [`engine/evaluate.py`](engine/evaluate.py) and [`spec/RX_FINAL_PLAN/`](spec/RX_FINAL_PLAN/). **That F512/K12 design is post-competition research and should not be confused with the K16 v4 network described above.**

### Training pipeline

I trained the v4 networks rather than starting from public pretrained chess weights.

The data pipeline mixed two useful distributions:

- positions generated offline and labelled by a strong teacher engine
- the public CC0 Lichess evaluation database, which added deeper evaluations from human-game positions

By the later v4 experiments, the pipeline was operating at roughly billion-record scale. Data generation, conversion, training and match evaluation ran as separate stages so I could change one variable without silently changing the rest of the experiment.

The training loop moved from floating-point optimisation into a quantization-aware phase before export. The important contract was not just low training loss. The quantized training forward pass, exported coefficients and integer runtime had to agree on clipping, widening, shifts and activation order.

I also kept provenance and split identity with the records. Related positions were grouped where needed so validation did not become a disguised duplicate of training.

The cleaned public pipeline is under [`training/`](training/). It is intentionally a reference implementation rather than a copy of the private competition training corpus.

### Time and memory management

A soft allocator estimated whether another iteration was worth starting, while a monotonic hard deadline guaranteed enough time to unwind and return a legal move.

The Numba backend stores recursive state in fixed-layout typed arrays instead of Python objects in the hot path. Move buffers, search stacks, histories, transposition-table storage and evaluator state are allocated ahead of time so runtime behaviour is much more predictable.

This was also why I tested the complete package rather than just functions in isolation. Import time, JIT compilation, model loading, resident memory and worst-case stopping latency all count when the referee can lose the game for a timeout, crash or illegal move.

See [`engine/kernels/`](engine/kernels/) and [`engine/clock_b/`](engine/clock_b/) for the public runtime work.

## How I got to v4

v4 was not built by adding every chess-engine technique I could find. It came from repeatedly freezing a working baseline, changing one thing and asking whether the whole engine was actually better.

### 1. Start with a correct engine

The early versions focused on bitboards, legal move generation, make/unmake, terminal rules and a classical evaluator. This gave me a trustworthy search harness before adding a neural model.

That order mattered. A faster evaluator cannot rescue broken repetition state, and a stronger search cannot rescue an engine that occasionally fails to restore its board after an interrupted branch.

### 2. Move the recursive hot path into Numba

Once the engine was correct enough to profile, Python overhead was the obvious constraint. I moved the search-critical data structures into typed NumPy arrays and compiled the recursive path with Numba.

I kept a slower reference path alongside it so optimised code could be checked against an independent implementation instead of debugging two moving targets at once.

### 3. Replace hand-written evaluation with a sparse neural model

The first neural versions showed that better representation could repay some of the lost node rate, but they also exposed how expensive the wrong head architecture could be.

The v4 family simplified the dense path and put more capacity into the sparse transform. The direct SCReLU head was easier to make fast, easier to quantize and better matched to the single-position inference pattern of alpha-beta search.

### 4. Scale the data, not just the width

I built an offline generation pipeline, added public Lichess evaluations and trained multiple widths and schedules. One useful lesson was that apparently obvious upgrades were not automatically upgrades. Making the network wider, training for longer or bolting on another standard search heuristic could improve a local metric without improving the complete engine.

The changes that survived were the ones that held up when search cost and data distribution were included in the experiment.

### 5. Make training and runtime agree exactly

Quantization initially created a gap between the model being optimised and the arithmetic being deployed. v4 added quantization-aware fine-tuning and explicit parity checks so I could reason about one model rather than a floating-point model and a slightly different integer one.

The same mindset applied to warm starts, clipping and export. If a resumed run changed the incumbent simply by loading it, that was a bug, not a new experiment.

### 6. Promote whole engines, not isolated metrics

Serious candidates eventually had to play through the real referee shape at the real clock, with colours paired and source/model identity frozen. Static validation, nodes per second and microbenchmarks were diagnostic tools, not promotion criteria by themselves.

That discipline also meant rejecting work. Several changes that looked attractive in isolation were neutral or worse once evaluator cost, the search tree and the clock interacted. Keeping the known-good build was often the right decision.

### 7. Freeze v4 and keep later research separate

Once v4 was stable, I kept it as a protected baseline while testing more experimental successors. That separation is reflected here too. The later F512 evaluator and richer feature work are useful research, but they are not retroactively presented as the architecture that produced the competition result.

## Public release

A fresh clone is deliberately usable without downloading a checkpoint. [`agent.py`](agent.py) runs the classical reference evaluator, while [`engine/agent_rx.py`](engine/agent_rx.py) exposes the neural runtime when given an explicit model artifact.

The public repository includes:

- board representation, move generation, search and time management
- integer neural inference and incremental accumulator logic
- training records, splits, checkpointing and export code
- differential correctness and failure-path tests
- local benchmark tooling with environment metadata

It does **not** include tournament weights, private datasets, opening assets or historical playing-strength result files.

## Quickstart

Use CPython 3.12.

```sh
uv sync --locked --extra dev
```

Or:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Choose a move with the weights-free reference adapter:

```sh
.venv/bin/python - <<'PY'
import chess
from agent import get_move

board = chess.Board()
move = get_move(board.fen(), time_left_ms=1_000)
assert chess.Move.from_uci(move) in board.legal_moves
print(move)
PY
```

## Training and verification

The bundled PGNs are small fixtures for exercising the public pipeline, not tournament training data. The typed-record and model contracts live under [`spec/RX_FINAL_PLAN/`](spec/RX_FINAL_PLAN/).

```sh
make lint
make test
make benchmark
make test-deep
```

The suite covers move-generation agreement, reversible state, draw and repetition semantics, abort-safe search, integer evaluator parity, serialization, data-split leakage and compiled-cache loading.

Benchmark outputs are local and include source and environment metadata. They are useful for comparing changes on a controlled machine, not for making public Elo claims.

See [`docs/architecture.md`](docs/architecture.md) and [`docs/verification.md`](docs/verification.md) for deeper engineering notes.

## Repository map

```text
agent.py                 weights-free reference adapter
engine/
  board.py, movegen.py   board representation and legal moves
  search.py, history.py  PVS, quiescence, pruning and ordering
  state.py, tt.py        history, counters and cache semantics
  evaluate.py            integer evaluation and accumulator lifecycle
  agent_rx.py            neural runtime integration
  kernels/               Numba-compiled search and evaluation
  clock_b/               allocation and deadline handling
training/                records, splits, trainer, checkpoints and export
spec/RX_FINAL_PLAN/      post-competition evaluator and data contracts
tests/                   correctness and failure-path regressions
bench/                   local benchmark tooling
docs/                    architecture and verification notes
```

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE), [third-party notices](THIRD_PARTY_NOTICES.md) and [contribution guidelines](CONTRIBUTING.md).

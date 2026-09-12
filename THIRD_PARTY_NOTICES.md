# Third-party notices

This project retains its GPL-3.0-or-later distribution terms. See [LICENSE](LICENSE). Upstream components retain their copyright and licence notices; no upstream network weights or complete engine executable are distributed.

Copyright (C) 2026 AI Chessathon team, for the Python/Numba implementation and integration. Copyright (C) 2004–2026 The Stockfish developers; their author list is preserved in [LICENSES/Stockfish-AUTHORS.txt](LICENSES/Stockfish-AUTHORS.txt). Additional component sources credited by the project include Terje Kirstihagen (Weiss) and Jay Honnold (Berserk).

| Source | Revision | Use in this source release |
|---|---|---|
| Stockfish | `59aae690f91d6f69aac194f447d84b4a2c3be778` | FullThreats feature definition used by runtime and training encoders; verbatim reference `.h`/`.cpp` files in `training/reference/` |
| PlentyChess | `04e07a98ee6ac104c30e7374450c94b96d94ef4d` | King-bucket geometry and incremental early-fusion design reference for `engine/features.py` and evaluation |
| Stockfish | `edb0d9db6731067ec50ce619ff372b463bc4dd5d` | Search, evaluation and static-exchange mechanisms inspected during component development |
| Weiss | `c735b8f3d2ddb0cdf42b135a5fb42e21c01f7a3d` | Selective search and history mechanisms inspected during component development |
| Berserk | `32628515050b83805bab4afa1026dd2bcaa93f55` | Static-exchange mechanisms inspected during component development |

The project's adaptations use its own board encoding, state/history contracts, evaluator layout, bounds, and deadline handling. Search constants are project configuration, not claims about upstream performance. Stockfish reference headers contain the original GPL-3.0-or-later notice. Reference sources are included for audit and are not compiled by this package.

The narrow signed-packing oracle and machine-readable contracts under `spec/` are retained project reference material. `training/testdata/` contains historical competition game records used solely for deterministic parser and schema tests. They are not evidence of the public baseline's playing strength.

Python, NumPy, Numba, llvmlite and python-chess are separately installed dependencies with their own licences. The source distribution contains editable Python, the frozen clock-table JSON and licence materials. Numba generates machine code at runtime from that source; precompiled caches and native chess engine binaries are excluded.

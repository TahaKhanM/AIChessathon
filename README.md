# Chessathon Engine

Python and Numba chess search with exact integer neural evaluation, explicit
history context and bounded move selection. This publication begins with a
curated source snapshot; it does not reconstruct the development chronology.

The root adapter is a runnable classical baseline. Neural implementation and
numerical pipeline source are retained, but no model checkpoints are included.
Use Python 3.12 and `uv sync --locked --extra dev`, then `make test` and
`make lint`. Exhaustive high-depth perft cases run separately with
`make test-deep`.

GPL-3.0-or-later. Component attribution is in THIRD_PARTY_NOTICES.md.

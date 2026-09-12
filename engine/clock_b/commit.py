"""Choose a committed principal variation or the pre-search legal fallback.

An aborted or partial iteration cannot replace a completed result. This
adapter guard complements the searcher's transactional commit protocol."""

from __future__ import annotations

from engine.board import move_to_uci


def committed_move(result, fallback: int) -> int:
    """Return the move that may be played. Never a torn abort candidate.

    Preference:
      1. completed PV head (``result.pv[0]``) when present
      2. ``result.move`` only when it agrees with that PV, or when no PV
         was extracted (depth-1 / fallback)
      3. ``fallback`` (legal, established before search)
    """
    if result is None:
        return int(fallback)
    pv = getattr(result, "pv", None) or []
    move = int(getattr(result, "move", 0) or 0)
    partial = bool(getattr(result, "partial", False))
    aborted = bool(getattr(result, "aborted", False))
    if pv:
        head = int(pv[0])
        if (partial or aborted) and move and move != head:
            return head
        return head
    if move:
        return move
    return int(fallback)


def committed_uci(result, fallback: int) -> str:
    move = committed_move(result, fallback)
    return move_to_uci(move) if move else "0000"

"""Frozen moves-to-go calibration used by the soft time allocator.

The table is a historical empirical estimate, bounded to 12–38 own moves.
It is a policy input, not a guarantee about remaining game length.
"""

from __future__ import annotations

import json
from pathlib import Path

PLY_CAP = 600
MTG_CAP = 38
MTG_FLOOR = 12
PCTL = 40.0

_RECEIPT = Path(__file__).with_name("horizon_table.json")


def _load_table() -> tuple[int, ...]:
    data = json.loads(_RECEIPT.read_text())
    table = tuple(int(x) for x in data["table"])
    if len(table) != PLY_CAP + 1:
        raise RuntimeError(f"horizon table length {len(table)} != {PLY_CAP + 1}")
    if table[0] != MTG_CAP:
        raise RuntimeError(f"ply-0 estimate {table[0]} != cap {MTG_CAP}")
    return table


REMAINING_OWN_MOVES: tuple[int, ...] = _load_table()


def estimated_moves_to_go(abs_ply: int) -> int:
    """Remaining own-move estimate, including the move about to be played."""
    p = int(abs_ply)
    if p < 0:
        p = 0
    if p >= len(REMAINING_OWN_MOVES):
        return MTG_FLOOR
    return max(MTG_FLOOR, min(MTG_CAP, int(REMAINING_OWN_MOVES[p])))

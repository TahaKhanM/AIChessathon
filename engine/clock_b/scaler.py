"""Soft-budget scaler: stability, score trend, effort, contender gap.

Inputs come from a *completed* iteration only. An aborted iteration must
never feed this controller (the ID driver already skips the scale call on
stop). Spec 3.4.
"""

from __future__ import annotations


class SoftScaler:
    """Multiplicative scale on the remaining soft budget after a completed ID step."""

    def __init__(
        self,
        *,
        stable_new_x100: int = 145,
        stable3_x100: int = 85,
        stable6_x100: int = 70,
        falling_drop_cp: int = 25,
        falling_x100: int = 130,
        rising_cp: int = 40,
        rising_x100: int = 92,
        effort_lo_x100: int = 70,
        effort_hi_x100: int = 150,
        separation_floor_cp: int = 15,
        separation_x100: int = 115,
        scale_cap_x100: int = 180,
        next_iter_cost_pct: int = 150,
    ) -> None:
        self.stable_new_x100 = stable_new_x100
        self.stable3_x100 = stable3_x100
        self.stable6_x100 = stable6_x100
        self.falling_drop_cp = falling_drop_cp
        self.falling_x100 = falling_x100
        self.rising_cp = rising_cp
        self.rising_x100 = rising_x100
        self.effort_lo_x100 = effort_lo_x100
        self.effort_hi_x100 = effort_hi_x100
        self.separation_floor_cp = separation_floor_cp
        self.separation_x100 = separation_x100
        self.scale_cap_x100 = scale_cap_x100
        self.next_iter_cost_pct = next_iter_cost_pct

    def scale(
        self,
        *,
        depth: int,
        stable_iters: int,
        score_drop_cp: int,
        best_move_node_fraction: float | None,
        contender_gap_cp: int | None,
    ) -> float:
        s = 1.0
        if stable_iters == 0 and depth >= 6:
            s = self.stable_new_x100 / 100.0
        elif stable_iters >= 6:
            s = self.stable6_x100 / 100.0
        elif stable_iters >= 3:
            s = self.stable3_x100 / 100.0
        if depth >= 6 and score_drop_cp >= self.falling_drop_cp:
            s *= self.falling_x100 / 100.0
        elif depth >= 8 and score_drop_cp <= -self.rising_cp:
            s *= self.rising_x100 / 100.0
        if depth >= 8 and best_move_node_fraction is not None and best_move_node_fraction > 0.0:
            frac = min(1.0, max(0.0, float(best_move_node_fraction)))
            effort = 2.0 * (1.0 - frac) + 0.4
            s *= max(
                self.effort_lo_x100 / 100.0,
                min(self.effort_hi_x100 / 100.0, effort),
            )
        if contender_gap_cp is not None and contender_gap_cp < self.separation_floor_cp:
            s *= self.separation_x100 / 100.0
        cap = self.scale_cap_x100 / 100.0
        if s > cap:
            s = cap
        return s

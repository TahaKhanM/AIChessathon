"""Label semantics for the rx-final-1 record contract (spec section 9).

Common coordinate: ``u = P(win) + 0.5 * P(draw)`` expressed in the
side-to-move perspective of the record's own position.

Rules enforced here:

* raw CP, mate, WDL, expected score, direct neural output, searched value,
  action value, visit distribution and game result stay DISTINCT; a cp value
  only becomes ``u`` through a named calibration (never a universal curve).
* child-to-parent action conversion is ``1 - u_child``; WDL conversion swaps
  W and L — never a bare negation of a probability.
* ``bound_kind`` lower/upper/interval produce interval targets only — they
  are one-sided supervision, not point labels; ``search_exact`` is a search
  result, not a game-theoretic proof.
* interrupted/censored observations and ``unknown`` bounds yield no target.
* a scalar relabel does not fabricate a WDL triple; WDL supervision only
  exists where a triple is genuinely present (wdl, mate, outcome).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# calibrations: cp -> u is per-teacher/per-calibration_id, versioned here.


@dataclass(frozen=True)
class CpCalibration:
    """sigmoid(cp / scale) mapping.  ``scale`` is in centipawns."""

    name: str
    scale: float

    def u(self, cp: float) -> float:
        return 1.0 / (1.0 + math.exp(-cp / self.scale))


# Default placeholder calibration for unversioned cp sources.  Real teachers
# must register their own entry (teacher manifest resolves calibration_id).
DEFAULT_CP_CALIBRATION = CpCalibration(name="default-sigmoid-400", scale=400.0)

CALIBRATIONS: dict[str, CpCalibration] = {
    DEFAULT_CP_CALIBRATION.name: DEFAULT_CP_CALIBRATION,
    # Fitted on data_v2 train by golden-section BCE vs stm-POV game
    # result (training/f512/build_datav2.py).  Recorded in eshard
    # manifests; F512 bakes u_teacher at encode time.  Registered here
    # so typed-record consumers do not fall through to sigmoid-400.
    "datav2-gen": CpCalibration(name="datav2-gen", scale=275.06),
    "datav2-self": CpCalibration(name="datav2-self", scale=274.88),
    # Floor-saturated: TB scores are near-decisive (±1400).  Not a
    # collapsed fit — u is ~{0, 0.5, 1}, which is the correct target.
    "datav2-tb6": CpCalibration(name="datav2-tb6", scale=40.0),
}


def register_calibration(cal: CpCalibration) -> None:
    CALIBRATIONS[cal.name] = cal


# ---------------------------------------------------------------------------
# SENTINEL SAFETY (rulings 15 and 29).
#
# ``kSkippedScore = 32002`` marks a record the relabeller SKIPPED.  It is not a
# score and must never become a training target.  `data/labels.py` already
# raises on it; this typed path did not, and `cp_to_u(32002, None)` returned
# **1.0** — a perfect "sure win" label on a record carrying no information.
# That is the exact poison ruling 15 quarantined the pipeline for, so the guard
# lives at every public decode entry point here too, and it RAISES rather than
# clipping: a silently clipped sentinel is indistinguishable from a real win.

SENTINEL_SKIP = 32002


class SentinelError(ValueError):
    """Raised when the relabeller skip sentinel reaches a decode path."""


def is_sentinel_skip(score: float) -> bool:
    return int(score) == SENTINEL_SKIP


def _reject_sentinel(cp: float) -> None:
    if is_sentinel_skip(cp):
        raise SentinelError(
            f"score {SENTINEL_SKIP} is the relabeller skip sentinel, not an "
            "evaluation; the record must be filtered before labelling "
            "(rulings 15, 29)"
        )


# Score-bearing field names — top-level and inside raw_payload.  A
# sentinel value in ANY of them poisons the observation — the check runs
# before score_kind dispatch so no kind/bound combination can slip it
# past, and a malformed record spelling bounds top-level is caught too.
_PAYLOAD_SCORE_KEYS = (
    "cp",
    "score_i16",
    "raw_score_i16",
    "score",
    "raw_score",
    "raw_cp",
    "bound_lo",
    "bound_hi",
)


def _reject_obs_sentinel(obs: dict) -> None:
    """Raise SentinelError when any score-bearing field carries 32002.

    The sentinel reaches the resolver via ``obs["cp"]`` or via a
    ``raw_payload`` score field.  A sentinel-marked observation carries
    NO supervision, on every declared score_kind/bound_kind — this check
    runs first, before the interrupted/censored early-return, because a
    poisoned value is never "merely provenance".
    """
    for k in _PAYLOAD_SCORE_KEYS:
        v = obs.get(k)
        if isinstance(v, (int, float)):
            _reject_sentinel(float(v))
    pl = obs.get("raw_payload")
    if isinstance(pl, dict):
        for k in _PAYLOAD_SCORE_KEYS:
            v = pl.get(k)
            if isinstance(v, (int, float)):
                _reject_sentinel(float(v))


def _check_u_bound(v: float, field: str) -> float:
    """u-space bounds are probabilities: an out-of-range value is producer
    corruption (e.g. a raw score leaked into a u field), not a bound."""
    v = float(v)
    if not (0.0 <= v <= 1.0):
        raise ValueError(
            f"{field}={v} is outside [0,1] — a u-space bound must be a "
            "probability; a raw score in a u field is producer corruption"
        )
    return v


def cp_to_u(cp: float, calibration_id: str | None) -> float:
    _reject_sentinel(cp)
    cal = CALIBRATIONS.get(calibration_id or "", DEFAULT_CP_CALIBRATION)
    return cal.u(cp)


def wdl_to_u(wdl: list[float]) -> float:
    w, d, ls = wdl
    return w + 0.5 * d


def u_to_wdl_point(u: float) -> list[float]:
    """NOT a WDL estimate — only used where the supervision itself is a bare
    scalar and no triple exists.  Kept explicit so it is never confused with
    a genuine wdl observation."""
    raise NotImplementedError("a scalar u does not specify three WDL probabilities (spec 9)")


def child_u_to_parent(u_child: float) -> float:
    return 1.0 - u_child


def child_wdl_to_parent(wdl: list[float]) -> list[float]:
    w, d, ls = wdl
    return [ls, d, w]


def mate_to_u(mate_plies: int) -> float:
    """Decisive mate supervision in plies, side-to-move POV.

    Positive = side to move mates.  Not arbitrarily clipped cp.
    """
    return 1.0 if mate_plies > 0 else 0.0


# ---------------------------------------------------------------------------
# target extraction


@dataclass
class ValueTarget:
    """One resolved supervision signal in record-stm u-space."""

    u: float | None = None  # point target
    u_lo: float | None = None  # one-sided lower bound
    u_hi: float | None = None  # one-sided upper bound
    wdl: list[float] | None = None  # genuine triple only
    is_result: bool = False  # came from a game_outcome observation
    is_bound: bool = False
    parent_record_id: str | None = None
    action_uci: str | None = None
    obs_kind: str = ""


def _to_stm(u: float, perspective: str, stm_is_white: bool) -> float:
    if perspective == "white":
        return u if stm_is_white else 1.0 - u
    if perspective == "black":
        return u if not stm_is_white else 1.0 - u
    if perspective == "side_to_move":
        return u
    if perspective == "parent_side_to_move":
        return child_u_to_parent(u)
    raise ValueError(perspective)


def _to_stm_wdl(wdl: list[float], perspective: str, stm_is_white: bool) -> list[float]:
    if perspective == "white":
        return wdl if stm_is_white else child_wdl_to_parent(wdl)
    if perspective == "black":
        return wdl if not stm_is_white else child_wdl_to_parent(wdl)
    if perspective == "side_to_move":
        return wdl
    if perspective == "parent_side_to_move":
        return child_wdl_to_parent(wdl)
    raise ValueError(perspective)


def resolve_observation(obs: dict, stm_is_white: bool) -> ValueTarget | None:
    """Map one observation to a ValueTarget in record-stm coordinates.

    Returns None when the observation carries no usable supervision
    (interrupted/censored, unknown bound, or a payload-only kind).
    Raises SentinelError when any score-bearing field carries the
    relabeller skip sentinel — on every score_kind and bound_kind.
    """
    _reject_obs_sentinel(obs)
    if obs.get("interrupted") or obs.get("right_censored"):
        return None
    if not obs.get("context_known", True):
        # context-blind labels train geometry-only targets; still usable,
        # but flagged by callers via position.known_fields.  The value itself
        # is still a valid target for the corresponding geometric target.
        pass
    sk = obs["score_kind"]
    bk = obs["bound_kind"]
    if bk == "unknown":
        return None
    tgt = ValueTarget(
        is_result=obs["kind"] == "game_outcome",
        parent_record_id=obs.get("parent_record_id"),
        action_uci=obs.get("action_uci"),
        obs_kind=obs["kind"],
    )

    def point(u_abs: float) -> None:
        tgt.u = _to_stm(u_abs, obs["perspective"], stm_is_white)

    if sk == "cp":
        u_abs = cp_to_u(float(obs["cp"]), obs.get("calibration_id"))
    elif sk == "mate":
        u_abs = mate_to_u(int(obs["mate_plies"]))
    elif sk == "wdl":
        w = _to_stm_wdl(list(obs["wdl"]), obs["perspective"], stm_is_white)
        tgt.wdl = w
        u_abs = wdl_to_u(w)
    elif sk in ("expected_score", "outcome"):
        if obs.get("expected_score") is not None:
            u_abs = _check_u_bound(obs["expected_score"], "expected_score")
        elif obs.get("wdl") is not None:
            w = _to_stm_wdl(list(obs["wdl"]), obs["perspective"], stm_is_white)
            tgt.wdl = w
            u_abs = wdl_to_u(w)
        else:
            return None
    elif sk == "interval":
        # raw_payload carries {"u_lo": .., "u_hi": ..} in observation POV.
        # A declared bound with no bound payload is a producer bug, not a
        # vacuous [0,1] target — fail loudly instead of teaching nothing.
        pl = obs.get("raw_payload") or {}
        lo, hi = pl.get("u_lo"), pl.get("u_hi")
        if lo is None and obs.get("cp") is not None:
            lo = cp_to_u(float(obs["cp"]), obs.get("calibration_id"))
        need_lo = bk in ("lower", "interval", "none", "search_exact")
        need_hi = bk in ("upper", "interval", "none", "search_exact")
        if (need_lo and lo is None) or (need_hi and hi is None):
            raise ValueError(
                f"score_kind=interval bound_kind={bk} requires "
                f"raw_payload u_lo/u_hi, got u_lo={lo} u_hi={hi}"
            )
        tgt.is_bound = True
        if lo is not None:
            tgt.u_lo = _to_stm(_check_u_bound(lo, "u_lo"), obs["perspective"], stm_is_white)
        if hi is not None:
            tgt.u_hi = _to_stm(_check_u_bound(hi, "u_hi"), obs["perspective"], stm_is_white)
        if bk == "lower":
            tgt.u_hi = None
        elif bk == "upper":
            tgt.u_lo = None
        return tgt
    else:
        # visits / operation_result: payload-only kinds, no scalar target
        return None

    if bk in ("lower", "upper", "interval"):
        tgt.is_bound = True
        tgt.u = None
        u_abs = _check_u_bound(u_abs, "bound")
        if bk in ("lower", "interval"):
            if bk == "lower":
                tgt.u_lo = u_abs
            else:
                pl = obs.get("raw_payload") or {}
                if pl.get("u_lo") is None or pl.get("u_hi") is None:
                    raise ValueError(
                        "bound_kind='interval' requires raw_payload u_lo/u_hi "
                        f"(score_kind={sk}); a missing bound is not a "
                        "vacuous [0,1] target"
                    )
                tgt.u_lo = _to_stm(
                    _check_u_bound(pl["u_lo"], "u_lo"), obs["perspective"], stm_is_white
                )
        if bk in ("upper", "interval"):
            if bk == "upper":
                tgt.u_hi = u_abs
            else:
                tgt.u_hi = _to_stm(
                    _check_u_bound((obs.get("raw_payload") or {})["u_hi"], "u_hi"),
                    obs["perspective"],
                    stm_is_white,
                )
        return tgt

    point(u_abs)
    if sk == "mate":
        tgt.wdl = [1.0, 0.0, 0.0] if u_abs >= 0.5 else [0.0, 0.0, 1.0]
    if sk == "outcome" and obs.get("wdl") is not None:
        tgt.wdl = _to_stm_wdl(list(obs["wdl"]), obs["perspective"], stm_is_white)
    return tgt


def record_targets(rec: dict, stm_is_white: bool) -> list[ValueTarget]:
    """All usable supervision on a record, distinct per observation."""
    out = []
    for o in rec["observations"]:
        t = resolve_observation(o, stm_is_white)
        if t is not None:
            out.append(t)
    return out

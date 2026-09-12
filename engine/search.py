"""Iterative-deepening PVS, quiescence and abort-safe search.

The reference search operates on Board and GameState. Ply-indexed state
supports transactional unwind and the parallel compiled implementation.
A legal fallback is established before expensive work; only completed
iterations publish their principal variation.

Technique provenance: selective reductions follow published Stockfish,
Ethereal and Weiss designs; see THIRD_PARTY_NOTICES.md. Constants are
heuristic configuration and require complete-engine measurement.

Draw estimates depend on history and counters. Null moves are search
devices and never enter played history or consume the legal-game horizon."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from engine.board import (
    BISHOP,
    CASTLE_CLEAR,
    FLAG_CASTLE,
    FLAG_EP,
    KING,
    KING_ATK,
    KNIGHT,
    KNIGHT_ATK,
    MASK64,
    MAX_MOVES,
    PAWN,
    PAWN_ATK,
    QUEEN,
    ROOK,
    Board,
    decode_move,
    move_to_uci,
)
from engine.clock import Deadline, IterationScaler, TimeAllocator
from engine.history import HistoryTables
from engine.movegen import (
    BETWEEN,
    LINE,
    bishop_attacks,
    generate_legal,
    has_legal_ep,
    in_check,
    rook_attacks,
    square_attacked,
)
from engine.state import PLY_CAP, GameState, _insufficient, referee_terminal
from engine.tt import (
    BOUND_EXACT,
    BOUND_LOWER,
    BOUND_UPPER,
    INF,
    MATE,
    MATE_IN_MAX,
    Probe,
    TranspositionTable,
    ValueContext,
)

DRAW = 0
EVAL_CLAMP = 20000
SEARCH_PATH = 832  # 600-ply cap + 192 depth + slack; undo capacity is 2048
MAX_DEPTH = 192

PIECE_VALUE = (100, 320, 330, 500, 950, 0)
# Indexed by full piece code 0..11.
PVAL = PIECE_VALUE + PIECE_VALUE
# Indexed by piece *type* 0..5, king last for SEE.
SEE_VALUE = (100, 320, 330, 500, 950, 20000)

# ordering score bands
S_TT = 1 << 30
S_GOOD_CAPTURE = 1 << 28
S_KILLER = 1 << 26
S_BAD_CAPTURE = -(1 << 26)

# Full ray sets (all squares along each ray, occupancy-independent) built on
# the shared movegen tables.
ROOK_RAYS = [0] * 64
BISHOP_RAYS = [0] * 64


def _init_rays() -> None:
    from engine.movegen import BISHOP_NEG, BISHOP_POS, ROOK_NEG, ROOK_POS

    for sq in range(64):
        ROOK_RAYS[sq] = ROOK_POS[sq][0] | ROOK_POS[sq][1] | ROOK_NEG[sq][0] | ROOK_NEG[sq][1]
        BISHOP_RAYS[sq] = (
            BISHOP_POS[sq][0] | BISHOP_POS[sq][1] | BISHOP_NEG[sq][0] | BISHOP_NEG[sq][1]
        )


_init_rays()


# ---------------------------------------------------------------------------
# Tunable parameters — every pruning/reduction/extension constant lives in one
# dict so a single code path serves normal play, offline tuning and the
# shipped build. Untuned seeds; spec 3.2 requires joint tuning in W08.
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "rfp_depth": 8,
    "rfp_margin": 80,
    "rfp_improving": 50,
    "razor_depth": 3,
    "razor_margin": 250,
    "nmp_min_depth": 3,
    "nmp_base": 3,
    "nmp_depth_div": 3,
    "nmp_eval_div": 200,
    "nmp_eval_max": 3,
    # Below this depth an NMP fail-high is trusted without verification.
    # Default 0 = every fail-high at every depth is verified: zugzwang risk
    # does not vanish at shallow depth and an unverified null cutoff is how
    # phantom scores enter the TT (R6/R7: verify was unreachable at d<=13,
    # i.e. the entire production range). W08 may re-raise the floor if
    # measurements justify skipping the re-search.
    "nmp_verify_depth": 0,
    "lmr_base_x100": 75,
    "lmr_div_x100": 225,
    "lmr_hist_div": 8192,
    "lmr_min_depth": 3,
    "lmp_base": 3,
    "lmp_quad": 1,
    "fut_depth": 8,
    "fut_base": 100,
    "fut_per_depth": 120,
    "see_quiet_depth": 6,
    "see_quiet_mult": 30,
    "see_noisy_depth": 6,
    "see_noisy_mult": 100,
    "sing_depth": 8,
    "sing_margin": 2,
    "sing_tt_slack": 3,
    "asp_delta": 18,
    "asp_min_depth": 5,
    "q_futility": 150,
    "q_see": 50,
    "check_ext_depth": 12,
    "iir_depth": 4,
    "hist_bonus_cap": 1200,
    "pc_depth": 5,
    "pc_margin": 190,
    "pc_see_cap": 2500,
}

# ---------------------------------------------------------------------------
# Extended tunables (W08). ``DEFAULT_PARAMS`` is frozen: the compiled kernel
# packer (engine/kernels/layout.py) asserts its key tuple equals PARAM_ORDER,
# so nothing may be added to it. Every *other* constant that was previously
# an embedded literal lives here with a default byte-identical to the literal
# it replaces. Kernels consume only PARAM_ORDER entries; these extras tune
# the Python reference path until the kernel packer extends PARAM_ORDER.
#
# Groups (see PARAM_GROUPS):
#   values   — piece-value vector shared by ordering, SEE, q-gain and ProbCut
#   ordering — move-ordering weights and capture banding
#   nmp      — null-move guards and verification span
#   ext      — singular/check extension shape
#   lmr      — reduction gates and signed modifiers
#   lmp      — late-move-count policy
#   driver   — iterative-deepening / aspiration / effort policy
#   hist     — history/correction update coefficients (applied to the tables)
# ---------------------------------------------------------------------------

TUNING_EXTRA = {
    # values (ordering/SEE/gain; the eval function's own scale is unaffected)
    "val_pawn": 100,
    "val_knight": 320,
    "val_bishop": 330,
    "val_rook": 500,
    "val_queen": 950,
    "val_king_see": 20000,
    # ordering
    "ord_cap_mvv_mult": 16,
    "ord_promo_queen": 2400,
    "ord_promo_under": -3000,
    "ord_goodcap_see_div": 12,
    # qsearch tactical gates
    "q_futility_on": 1,
    "q_see_on": 1,
    "q_fut_see_gate": 1,
    # null move
    "nmp_eval_margin": 0,
    "nmp_no_consec": 1,
    "nmp_tt_guard": 1,
    "nmp_verify_min": 2,
    "nmp_verify_num": 3,
    "nmp_verify_den": 4,
    # ProbCut
    "pc_tt_slack": 3,
    # extensions
    "sing_ext": 1,
    "sing_half_div": 2,
    "sing_ply_cap_mult": 2,
    "check_ext": 1,
    "check_ply_cap_mult": 2,
    # late move count
    "lmp_improving_div": 2,
    # SEE quiet depth floor
    "see_quiet_min_ld": 1,
    # LMR gates / signed modifiers / clamps
    "lmr_gate_pv": 3,
    "lmr_gate_nonpv": 2,
    "lmr_not_improving": 1,
    "lmr_cut_node": 1,
    "lmr_pv": 1,
    "lmr_killer": 1,
    "lmr_min_r": 0,
    "lmr_max_sub": 1,
    # driver / aspiration / effort
    "asp_widen_pct": 50,
    "asp_widen_add": 5,
    "asp_fail_low_blend": 1,
    "mate_break_depth": 6,
    "effort_min_depth": 8,
    "next_iter_min_depth": 8,
    "next_iter_cost_pct": 150,
    # history / correction coefficients
    "hist_bonus_quad": 16,
    "hist_bonus_lin": 32,
    "hist_bonus_const": 16,
    "hist_pawn_div": 2,
    "hist_threat_div": 2,
    "corr_pawn_w": 2,
    "corr_np_w": 1,
    "corr_weight_cap": 16,
    # evaluation bound applied by the search (must stay below MATE_IN_MAX)
    "eval_clamp": 20000,
}

# The complete tuning surface: every overridable constant. Defaults are the
# pre-W08 literals; ``Searcher(params=...)`` and ``load_search_params`` merge
# over this.
PARAM_DEFAULTS = {**DEFAULT_PARAMS, **TUNING_EXTRA}

# (lo, hi) bounds for every tunable: loader validation and the SPSA range.
# Bounds are sanity rails, not tuned ranges — the SPSA spec picks per-param
# steps inside them.
# Depth-gated mechanisms are disabled by an unreachable gate value: a *_depth
# of -1 (condition `depth <= d` / `depth < d`) or a min-depth above MAX_DEPTH
# (condition `depth >= d`) can never fire. Factorial "off" levels use those.
PARAM_BOUNDS = {
    "rfp_depth": (-1, 16),
    "rfp_margin": (10, 400),
    "rfp_improving": (0, 200),
    "razor_depth": (-1, 8),
    "razor_margin": (0, 800),
    "nmp_min_depth": (1, 200),
    "nmp_base": (0, 8),
    "nmp_depth_div": (1, 8),
    "nmp_eval_div": (20, 800),
    "nmp_eval_max": (0, 8),
    "nmp_verify_depth": (0, 24),
    "lmr_base_x100": (0, 300),
    "lmr_div_x100": (50, 600),
    "lmr_hist_div": (512, 65536),
    "lmr_min_depth": (1, 200),
    "lmp_base": (1, 100000),
    "lmp_quad": (0, 8),
    "fut_depth": (-1, 16),
    "fut_base": (0, 400),
    "fut_per_depth": (0, 400),
    "see_quiet_depth": (-1, 16),
    "see_quiet_mult": (0, 200),
    "see_noisy_depth": (-1, 16),
    "see_noisy_mult": (0, 300),
    "sing_depth": (2, 200),
    "sing_margin": (0, 12),
    "sing_tt_slack": (0, 8),
    "asp_delta": (4, 80),
    "asp_min_depth": (1, 12),
    "q_futility": (0, 600),
    "q_see": (0, 300),
    "check_ext_depth": (0, 32),
    "iir_depth": (1, 200),
    "hist_bonus_cap": (64, 8192),
    "pc_depth": (2, 200),
    "pc_margin": (20, 600),
    "pc_see_cap": (100, 20000),
    "val_pawn": (40, 240),
    "val_knight": (200, 500),
    "val_bishop": (200, 500),
    "val_rook": (350, 700),
    "val_queen": (700, 1400),
    "val_king_see": (9000, 30000),
    "ord_cap_mvv_mult": (1, 64),
    "ord_promo_queen": (0, 8000),
    "ord_promo_under": (-12000, 0),
    "ord_goodcap_see_div": (2, 64),
    "q_fut_see_gate": (0, 400),
    "q_futility_on": (0, 1),
    "q_see_on": (0, 1),
    "nmp_eval_margin": (0, 400),
    "nmp_no_consec": (0, 1),
    "nmp_tt_guard": (0, 1),
    "nmp_verify_min": (0, 10),
    "nmp_verify_num": (0, 8),
    "nmp_verify_den": (1, 8),
    "pc_tt_slack": (0, 8),
    "sing_ext": (0, 2),
    "sing_half_div": (2, 6),
    "sing_ply_cap_mult": (0, 8),
    "check_ext": (0, 2),
    "check_ply_cap_mult": (0, 8),
    "lmp_improving_div": (1, 4),
    "see_quiet_min_ld": (0, 6),
    "lmr_gate_pv": (1, 10),
    "lmr_gate_nonpv": (1, 10),
    "lmr_not_improving": (0, 4),
    "lmr_cut_node": (0, 4),
    "lmr_pv": (0, 4),
    "lmr_killer": (0, 4),
    "lmr_min_r": (0, 4),
    "lmr_max_sub": (1, 4),
    "asp_widen_pct": (0, 300),
    "asp_widen_add": (0, 60),
    "asp_fail_low_blend": (0, 1),
    "mate_break_depth": (1, 64),
    "effort_min_depth": (1, 32),
    "next_iter_min_depth": (1, 32),
    "next_iter_cost_pct": (50, 400),
    "hist_bonus_quad": (0, 64),
    "hist_bonus_lin": (0, 128),
    "hist_bonus_const": (0, 128),
    "hist_pawn_div": (1, 8),
    "hist_threat_div": (1, 8),
    "corr_pawn_w": (1, 8),
    "corr_np_w": (1, 8),
    "corr_weight_cap": (0, 64),
    "eval_clamp": (1000, 27000),
}

assert set(PARAM_BOUNDS) == set(PARAM_DEFAULTS), "bounds/defaults drifted"

PARAM_GROUPS = {
    "core": tuple(DEFAULT_PARAMS),
    "values": ("val_pawn", "val_knight", "val_bishop", "val_rook", "val_queen", "val_king_see"),
    "ordering": (
        "ord_cap_mvv_mult",
        "ord_promo_queen",
        "ord_promo_under",
        "ord_goodcap_see_div",
        "q_fut_see_gate",
        "q_futility_on",
        "q_see_on",
    ),
    "nmp": (
        "nmp_eval_margin",
        "nmp_no_consec",
        "nmp_tt_guard",
        "nmp_verify_min",
        "nmp_verify_num",
        "nmp_verify_den",
    ),
    "ext": ("sing_ext", "sing_half_div", "sing_ply_cap_mult", "check_ext", "check_ply_cap_mult"),
    "lmr": (
        "lmr_gate_pv",
        "lmr_gate_nonpv",
        "lmr_not_improving",
        "lmr_cut_node",
        "lmr_pv",
        "lmr_killer",
        "lmr_min_r",
        "lmr_max_sub",
    ),
    "lmp_see": ("lmp_improving_div", "see_quiet_min_ld", "pc_tt_slack"),
    "driver": (
        "asp_widen_pct",
        "asp_widen_add",
        "asp_fail_low_blend",
        "mate_break_depth",
        "effort_min_depth",
        "next_iter_min_depth",
        "next_iter_cost_pct",
    ),
    "hist": (
        "hist_bonus_quad",
        "hist_bonus_lin",
        "hist_bonus_const",
        "hist_pawn_div",
        "hist_threat_div",
        "corr_pawn_w",
        "corr_np_w",
        "corr_weight_cap",
    ),
    "eval": ("eval_clamp",),
}

assert {k for group in PARAM_GROUPS.values() for k in group} == set(PARAM_DEFAULTS), (
    "groups/defaults drifted"
)


class TuningError(ValueError):
    """Raised at load time by an invalid tuning document — never mid-game."""


def load_search_params(source: dict | str | Path | None) -> dict:
    """Validated, transactional tuning for the search.

    ``source`` may be a params dict, a JSON object (with an optional top-level
    ``"search"`` section), or a path to such a JSON file. ``None`` returns a
    copy of the defaults. The whole document is validated BEFORE anything is
    produced: an unknown key, a non-integer value or an out-of-bounds value is
    a load-time error, never a partially-applied override.
    """
    import json as _json
    from pathlib import Path as _Path

    if source is None:
        return dict(PARAM_DEFAULTS)
    if isinstance(source, (str, _Path)):
        try:
            doc = _json.loads(_Path(source).read_text())
        except OSError as e:
            raise TuningError(f"tuning file unreadable: {e}") from e
        except _json.JSONDecodeError as e:
            raise TuningError(f"tuning file is not valid JSON: {e}") from e
    elif isinstance(source, dict):
        doc = source
    else:
        raise TuningError(f"unsupported tuning source type {type(source)!r}")
    if not isinstance(doc, dict):
        raise TuningError("tuning document must be a JSON object")
    if "search" in doc:
        doc = doc["search"]
        if not isinstance(doc, dict):
            raise TuningError("'search' section must be an object")
    out = dict(PARAM_DEFAULTS)
    for k, v in doc.items():
        if k not in PARAM_DEFAULTS:
            raise TuningError(f"unknown tuning key {k!r}")
        if isinstance(v, bool) or not isinstance(v, int):
            raise TuningError(f"tuning key {k!r} must be an integer, got {v!r}")
        lo, hi = PARAM_BOUNDS[k]
        if not lo <= v <= hi:
            raise TuningError(f"tuning key {k!r}={v} outside [{lo}, {hi}]")
        out[k] = v
    return out


# Recorded reasons (spec 3.2: "record WHY a branch was reduced or pruned").
REASONS = (
    "tt_cut",
    "mate_dist",
    "rfp",
    "razor",
    "razor_verify",
    "nmp",
    "nmp_verify_fail",
    "probcut",
    "lmp",
    "futility",
    "see_quiet",
    "see_noisy",
    "lmr",
    "lmr_research",
    "pvs_research",
    "iir",
    "sing_ext",
    "check_ext",
    "q_futility",
    "q_see",
    "rep",
    "fifty",
    "cap",
    "insufficient",
    "mate",
    "stalemate",
    "corr_update",
    # Appended for the root adjudication (kept last: the compiled kernel's
    # R_* indices are enumerated in this order — appending preserves them).
    "seventyfive",
    "fivefold",
)

# referee_terminal() termination name -> REASONS bucket for the root note.
_ROOT_REASON = {
    "checkmate": "mate",
    "stalemate": "stalemate",
    "insufficient_material": "insufficient",
    "seventyfive_moves": "seventyfive",
    "fivefold_repetition": "fivefold",
    "threefold_repetition": "rep",
    "fifty_moves": "fifty",
    "ply_cap": "cap",
}


def build_lmr_table(base_x100: int, div_x100: int) -> np.ndarray:
    table = np.zeros((64, 64), dtype=np.int64)
    base = base_x100 / 100.0
    divisor = max(div_x100, 1) / 100.0
    for d in range(1, 64):
        for m in range(1, 64):
            table[d, m] = int(base + math.log(d) * math.log(m) / divisor)
    return table


# ---------------------------------------------------------------------------
# Stand-in static evaluation (W03 baseline). The real F512-EF evaluator lands
# in W02/W05; Searcher accepts ``eval_fn`` so this swaps out without touching
# the search. Material + compact piece-square tables, side-to-move centipawns.
# ---------------------------------------------------------------------------

_PSQT = {
    PAWN: (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        5,
        10,
        10,
        -20,
        -20,
        10,
        10,
        5,
        5,
        -5,
        -10,
        0,
        0,
        -10,
        -5,
        5,
        0,
        0,
        0,
        20,
        20,
        0,
        0,
        0,
        5,
        5,
        10,
        25,
        25,
        10,
        5,
        5,
        10,
        10,
        20,
        30,
        30,
        20,
        10,
        10,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ),
    KNIGHT: (
        -50,
        -40,
        -30,
        -30,
        -30,
        -30,
        -40,
        -50,
        -40,
        -20,
        0,
        5,
        5,
        0,
        -20,
        -40,
        -30,
        5,
        10,
        15,
        15,
        10,
        5,
        -30,
        -30,
        0,
        15,
        20,
        20,
        15,
        0,
        -30,
        -30,
        5,
        15,
        20,
        20,
        15,
        5,
        -30,
        -30,
        0,
        10,
        15,
        15,
        10,
        0,
        -30,
        -40,
        -20,
        0,
        0,
        0,
        0,
        -20,
        -40,
        -50,
        -40,
        -30,
        -30,
        -30,
        -30,
        -40,
        -50,
    ),
    BISHOP: (
        -20,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -20,
        -10,
        5,
        0,
        0,
        0,
        0,
        5,
        -10,
        -10,
        10,
        10,
        10,
        10,
        10,
        10,
        -10,
        -10,
        0,
        10,
        10,
        10,
        10,
        0,
        -10,
        -10,
        5,
        5,
        10,
        10,
        5,
        5,
        -10,
        -10,
        0,
        5,
        10,
        10,
        5,
        0,
        -10,
        -10,
        0,
        0,
        0,
        0,
        0,
        0,
        -10,
        -20,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -20,
    ),
    ROOK: (
        0,
        0,
        0,
        5,
        5,
        0,
        0,
        0,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        5,
        10,
        10,
        10,
        10,
        10,
        10,
        5,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ),
    QUEEN: (
        -20,
        -10,
        -10,
        -5,
        -5,
        -10,
        -10,
        -20,
        -10,
        0,
        5,
        0,
        0,
        0,
        0,
        -10,
        -10,
        5,
        5,
        5,
        5,
        5,
        0,
        -10,
        0,
        0,
        5,
        5,
        5,
        5,
        0,
        -5,
        -5,
        0,
        5,
        5,
        5,
        5,
        0,
        -5,
        -10,
        0,
        5,
        5,
        5,
        5,
        0,
        -10,
        -10,
        0,
        0,
        0,
        0,
        0,
        0,
        -10,
        -20,
        -10,
        -10,
        -5,
        -5,
        -10,
        -10,
        -20,
    ),
    KING: (
        20,
        30,
        10,
        0,
        0,
        10,
        30,
        20,
        20,
        20,
        0,
        0,
        0,
        0,
        20,
        20,
        -10,
        -20,
        -20,
        -20,
        -20,
        -20,
        -20,
        -10,
        -20,
        -30,
        -30,
        -40,
        -40,
        -30,
        -30,
        -20,
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
    ),
}


def simple_eval(board: Board) -> int:
    """Material + PSQT stand-in, centipawns from the side to move."""
    sq = board._sq
    score = 0
    for s in range(64):
        p = sq[s]
        if p < 0:
            continue
        t = p % 6
        v = PVAL[t] + _PSQT[t][s if p < 6 else s ^ 56]
        score += v if p < 6 else -v
    return score + 10 if board.side == 0 else -score + 10


# ---------------------------------------------------------------------------
# Helpers shared by the search
# ---------------------------------------------------------------------------


def _attackers_to(board: Board, sq: int, occ: int) -> int:
    """All pieces of both colours attacking ``sq`` given occupancy ``occ``."""
    bb = board._bb
    att = PAWN_ATK[1][sq] & bb[0] | PAWN_ATK[0][sq] & bb[6]
    att |= KNIGHT_ATK[sq] & (bb[1] | bb[7])
    att |= KING_ATK[sq] & (bb[5] | bb[11])
    att |= bishop_attacks(sq, occ) & (bb[2] | bb[8] | bb[4] | bb[10])
    att |= rook_attacks(sq, occ) & (bb[3] | bb[9] | bb[4] | bb[10])
    return att


def _absolute_pins(board: Board, color: int, occ: int) -> int:
    """Own pieces that are the single blocker between ``color``'s king and a
    sniper, under occupancy ``occ`` (SEE passes a reduced occ)."""
    bb = board._bb
    them = color ^ 1
    ksq = board._king[color]
    own = board._occ[color]
    tb = them * 6
    snipers = (ROOK_RAYS[ksq] & (bb[tb + ROOK] | bb[tb + QUEEN])) | (
        BISHOP_RAYS[ksq] & (bb[tb + BISHOP] | bb[tb + QUEEN])
    )
    snipers &= occ
    pinned = 0
    while snipers:
        s = (snipers & -snipers).bit_length() - 1
        snipers &= snipers - 1
        blockers = BETWEEN[ksq][s] & occ
        if blockers and blockers.bit_count() == 1 and blockers & own:
            pinned |= blockers
    return pinned


def see_ge(
    board: Board,
    m: int,
    threshold: int,
    pval: tuple[int, ...] | None = None,
    see_value: tuple[int, ...] | None = None,
) -> bool:
    """True if the exchange starting with move ``m`` is worth >= ``threshold``.

    Pin-aware (absolute pins under the evolving occupancy). Special moves
    (EP, castling, promotions) are not SEE-evaluable: they pass iff the
    threshold is non-positive — the caller decides, never a silent discard.

    ``pval``/``see_value`` default to the module value vector; a Searcher
    passes its tuned vectors (W08 params ``val_*``).
    """
    pv = PVAL if pval is None else pval
    sv = SEE_VALUE if see_value is None else see_value
    frm, to, promo, flag, _piece, _captured = decode_move(m)
    if flag == FLAG_EP or flag == FLAG_CASTLE or promo:
        return threshold <= 0
    sq = board._sq
    victim = sq[to]
    swap = (0 if victim < 0 else pv[victim]) - threshold
    if swap < 0:
        return False
    attacker = sq[frm]
    swap = pv[attacker] - swap
    if swap <= 0:
        return True
    occ = (board._occ_all ^ (1 << frm) ^ (1 << to)) & MASK64
    stm = (attacker // 6) ^ 1  # side to move after the initial capture
    bb = board._bb
    attackers = _attackers_to(board, to, occ) & occ
    res = 1
    while True:
        stm_att = attackers & board._occ[stm]
        if stm_att == 0:
            break
        pins = _absolute_pins(board, stm, occ)
        ksq = board._king[stm]
        res ^= 1
        found = -1
        csq = -1
        base = stm * 6
        for t in range(PAWN, KING + 1):
            pcs = stm_att & bb[base + t]
            while pcs:
                cand = (pcs & -pcs).bit_length() - 1
                pcs &= pcs - 1
                cb = 1 << cand
                if (pins & cb) and not (LINE[ksq][cand] & (1 << to)):
                    continue
                csq = cand
                found = t
                break
            if found >= 0:
                break
        if found < 0:
            res ^= 1
            break
        occ ^= 1 << csq
        if found == KING:
            if attackers & board._occ[stm ^ 1] & occ:
                res ^= 1
            break
        swap = sv[found] - swap
        if swap < res:
            break
        if found == PAWN or found == BISHOP or found == QUEEN:
            attackers |= bishop_attacks(to, occ) & (bb[2] | bb[8] | bb[4] | bb[10])
        if found == ROOK or found == QUEEN:
            attackers |= rook_attacks(to, occ) & (bb[3] | bb[9] | bb[4] | bb[10])
        attackers &= occ
        stm ^= 1
    return res != 0


def _gives_check_fast(board: Board, m: int) -> bool:
    """Direct + discovered check without making the move.

    Castling and en passant return False (their occupancy changes are not
    modelled here); the post-make ``in_check`` remains authoritative for
    extensions. Exists so quiet-move pruning knows a move checks *before*
    discarding it.
    """
    frm, to, promo, flag, piece, _captured = decode_move(m)
    if flag == FLAG_EP or flag == FLAG_CASTLE:
        return False
    them = board.side ^ 1
    ksq = board._king[them]
    kb = 1 << ksq
    occ = ((board._occ_all ^ (1 << frm)) | (1 << to)) & MASK64
    ptype = promo if promo else piece % 6
    if ptype == PAWN:
        if PAWN_ATK[board.side][to] & kb:
            return True
    elif ptype == KNIGHT:
        if KNIGHT_ATK[to] & kb:
            return True
    elif ptype == BISHOP:
        if bishop_attacks(to, occ) & kb:
            return True
    elif ptype == ROOK:
        if rook_attacks(to, occ) & kb:
            return True
    elif ptype == QUEEN:
        if (bishop_attacks(to, occ) | rook_attacks(to, occ)) & kb:
            return True
    # Discovered check: our slider sees their king through the vacated square.
    bb = board._bb
    base = board.side * 6
    not_frm = MASK64 ^ (1 << frm)
    if BISHOP_RAYS[ksq] & (1 << frm):
        if bishop_attacks(ksq, occ) & (bb[base + BISHOP] | bb[base + QUEEN]) & not_frm:
            return True
    if ROOK_RAYS[ksq] & (1 << frm):
        if rook_attacks(ksq, occ) & (bb[base + ROOK] | bb[base + QUEEN]) & not_frm:
            return True
    return False


def _is_irreversible(board: Board, m: int) -> int:
    """Pre-move irreversibility test (mirrors state._irreversible semantics)."""
    if has_legal_ep(board):
        return 1
    frm, to, promo, flag, piece, captured = decode_move(m)
    if piece % 6 == PAWN or captured != 15 or flag == FLAG_EP or promo:
        return 1
    return 1 if board.castling != (board.castling & CASTLE_CLEAR[frm] & CASTLE_CLEAR[to]) else 0


def _king_has_legal_move(board: Board) -> bool:
    """Exact: has the side to move any legal king move? (Same legality test as
    generation.) A stalemate implies False, so it is a *sound* gate for the
    full generation that confirms a quiet stalemate in qsearch."""
    us = board.side
    them = us ^ 1
    ksq = board._king[us]
    occ_nk = board._occ_all ^ (1 << ksq)
    targets = KING_ATK[ksq] & ~board._occ[us] & MASK64
    while targets:
        to = (targets & -targets).bit_length() - 1
        targets &= targets - 1
        if not square_attacked(board, to, them, occ_nk):
            return True
    return False


def _victim_type(captured: int) -> int:
    """Captured piece *type* 1..6 for capture-history indexing (0 = none)."""
    if captured == 15:
        return 0
    return captured % 6 + 1


class Info:
    """Per-search counters and control (the compiled-port ``info`` block)."""

    __slots__ = (
        "nodes",
        "qnodes",
        "seldepth",
        "tt_hits",
        "root_best",
        "root_score",
        "root_depth",
        "best_nodes",
        "null_min_ply",
        "root_hint",
        "phase",
    )

    def __init__(self) -> None:
        self.nodes = 0
        self.qnodes = 0
        self.seldepth = 0
        self.tt_hits = 0
        self.root_best = 0
        self.root_score = -INF
        self.root_depth = 0
        self.best_nodes = 0
        self.null_min_ply = 0
        self.root_hint = 0
        self.phase = 0  # abort-coverage marker: 0=driver, 1=_ab, 2=_qs


class SearchResult:
    __slots__ = (
        "move",
        "score",
        "depth",
        "seldepth",
        "nodes",
        "qnodes",
        "elapsed_ms",
        "aborted",
        "partial",
        "iterations",
        "pv",
        "fallback_used",
        "best_effort",
        "contender_gap",
    )

    def __init__(self) -> None:
        self.move = 0
        self.score = 0
        self.depth = 0
        self.seldepth = 0
        self.nodes = 0
        self.qnodes = 0
        self.elapsed_ms = 0.0
        self.aborted = False
        self.partial = False
        self.iterations: list[tuple[int, int, int, int]] = []
        self.pv: list[int] = []
        self.fallback_used = False
        self.best_effort = 0.0
        self.contender_gap = -1

    @property
    def uci(self) -> str:
        return move_to_uci(self.move) if self.move else "0000"

    @property
    def pv_uci(self) -> list[str]:
        return [move_to_uci(m) for m in self.pv]


class Searcher:
    """ID-PVS over a ``GameState``; abort-safe, legal-fallback-first.

    ``eval_fn(board) -> int`` returns the RAW static evaluation in centipawns
    from the side to move — never a corrected score. ``on_make``/``on_unmake``
    callbacks let a W05 accumulator subscribe to the same transactional
    unwind the abort gate exercises.
    """

    def __init__(
        self,
        tt: TranspositionTable | None = None,
        history: HistoryTables | None = None,
        eval_fn: Callable[[Board], int] | None = None,
        params: dict[str, int] | str | Path | None = None,
        trace: bool = False,
        now: Callable[[], int] | None = None,
        check_mask: int = 15,
    ) -> None:
        self.tt = tt if tt is not None else TranspositionTable()
        self.hist = history if history is not None else HistoryTables()
        self.eval_fn = eval_fn or simple_eval
        # Transactional: the whole document is validated BEFORE anything is
        # applied — a bad key can never leave a half-tuned searcher.
        self.params = load_search_params(params)
        p = self.params
        self.lmr = build_lmr_table(p["lmr_base_x100"], p["lmr_div_x100"])
        # Piece-value vector shared by capture ordering, SEE, qsearch gain and
        # ProbCut gates — the eval function's own scale is unaffected.
        self.pval = (
            p["val_pawn"],
            p["val_knight"],
            p["val_bishop"],
            p["val_rook"],
            p["val_queen"],
            0,
        ) * 2
        self.see_value = (
            p["val_pawn"],
            p["val_knight"],
            p["val_bishop"],
            p["val_rook"],
            p["val_queen"],
            p["val_king_see"],
        )
        self.eval_clamp = p["eval_clamp"]
        # History-side coefficients ride the same params dict.
        for attr, key in (
            ("corr_weight_cap", "corr_weight_cap"),
            ("corr_pawn_w", "corr_pawn_w"),
            ("corr_np_w", "corr_np_w"),
            ("pawn_div", "hist_pawn_div"),
            ("threat_div", "hist_threat_div"),
            ("bonus_quad", "hist_bonus_quad"),
            ("bonus_lin", "hist_bonus_lin"),
            ("bonus_const", "hist_bonus_const"),
        ):
            if hasattr(self.hist, attr):
                setattr(self.hist, attr, p[key])
        self.trace_enabled = trace
        self.trace: list[tuple[int, str, int]] = []
        # Structured why-records (trace-gated): (ply, reason, move, detail).
        self.events: list[tuple[int, str, int, dict]] = []
        self.stats = {name: 0 for name in REASONS}
        self.now = now or time.monotonic_ns

        # Poll mask sized for THIS interpreter: pure-Python nodes cost ~200us
        # each, so a 16-node quantum bounds the clock slack to ~3ms — inside
        # the 8ms unwind margin. The compiled kernel (W05/W08) restores the
        # large DEFAULT_CHECK_MASK where nodes cost ~0.2us.
        self._check_mask = check_mask

        self.info = Info()
        self.deadline = Deadline(now=self.now)
        self.board: Board = Board()
        self.state: GameState | None = None

        # ply-indexed buffers (no per-node allocation)
        self.moves = [[0] * MAX_MOVES for _ in range(SEARCH_PATH)]
        self.scores = [[0] * MAX_MOVES for _ in range(SEARCH_PATH)]
        self.quiets = [[0] * MAX_MOVES for _ in range(SEARCH_PATH)]
        # Search-stack "move that produced the position at ply+1" records.
        # ss_piece uses -1 (board.EMPTY) as the no-previous-move sentinel:
        # piece code 0 is a real white pawn and must never collide with it.
        # The slots are WRITE-once per make and read only on the live path
        # (a parent's slot is rewritten before each child search), so unmake
        # deliberately leaves them — callers must never read a slot for a ply
        # whose position is not currently made on the board.
        self.ss_move = [0] * (SEARCH_PATH + 8)
        self.ss_piece = [-1] * (SEARCH_PATH + 8)
        self.ss_eval = [-INF] * (SEARCH_PATH + 8)
        self.ss_null = [0] * (SEARCH_PATH + 8)
        self.ss_excl = [0] * (SEARCH_PATH + 8)
        self.killers = [[0, 0] for _ in range(SEARCH_PATH)]

        # search-path identity arrays (null-move aware)
        self.path_keys = [0] * (SEARCH_PATH + 8)
        self.path_null_floor = [0] * (SEARCH_PATH + 8)
        self.path_irr_floor = [0] * (SEARCH_PATH + 8)
        self.node_rep = [0] * (SEARCH_PATH + 8)
        self._pending_irr = 0

        # known-history snapshot, taken once per search()
        self._hist_keys: list[int] = []
        self._hist_irr: list[bool] = []
        self._hist_n = 0
        self._unknown = True
        self._model_ver = 0
        self._util_ver = 0

        self._probe = Probe()
        self._vctx = ValueContext(0, 0, 0, 0, 1, True)
        self._root_scores: dict[int, int] = {}
        self.on_make: list[Callable[[int, int], None]] = []
        self.on_unmake: list[Callable[[int, int], None]] = []
        self.on_null: list[Callable[[int], None]] = []
        self.on_unnull: list[Callable[[int], None]] = []

    # -- lifecycle -----------------------------------------------------------

    def new_game(self) -> None:
        """Clear per-game state (TT, histories). Call once per new game."""
        self.tt.clear()
        self.hist.clear()

    def _note(self, reason: str, ply: int, move: int = 0, detail: dict | None = None) -> None:
        self.stats[reason] += 1
        if self.trace_enabled:
            self.trace.append((ply, reason, move))
            d = dict(detail) if detail else {}
            # The node position the decision was taken at — enables
            # counterfactual replay of the prune/reduction offline.
            d["fen"] = self.board.to_fen()
            self.events.append((ply, reason, move, d))

    def _vctx_at(self, ply: int) -> ValueContext:
        b = self.board
        v = self._vctx
        v.halfmove = b.halfmove
        v.horizon = PLY_CAP - b.absolute_ply()
        v.model = self._model_ver
        v.util = self._util_ver
        v.rep = self.node_rep[ply]
        v.unknown = self._unknown
        return v

    # -- path/repetition ------------------------------------------------------

    def _rep_scan(self, ply: int) -> tuple[bool, int]:
        """(in-path hit, game-history hits) for the current key.

        The path scan never crosses a null-move boundary (null floors) or an
        irreversible path move; the game-history scan then covers only the
        reversible suffix of the known game, matching the referee's
        transposition semantics (EP counted only when a legal capture exists,
        per the board key).
        """
        key = int(self.board.key)
        null_floor = self.path_null_floor[ply]
        irr_floor = self.path_irr_floor[ply]
        if null_floor > irr_floor:
            # A null move is the binding boundary: the null-produced position
            # itself is excluded, and nothing before it may be compared —
            # that is what prevents a search device from fabricating a repeat.
            stop = null_floor + 1
        else:
            # The irreversible-boundary position is real path history (it can
            # never match by key anyway); indices below it cannot be reached.
            stop = irr_floor
        i = ply - 2
        while i >= stop and i >= 0:
            if self.path_keys[i] == key:
                return True, 0
            i -= 2
        hits = 0
        if null_floor == 0 and irr_floor == 0:
            # Continue into the reversible suffix of known game history.
            n = self._hist_n
            keys = self._hist_keys
            irr = self._hist_irr
            i = n - 2
            while i >= 0:
                if irr[i]:
                    break
                if keys[i] == key:
                    hits += 1
                i -= 1
        return False, hits

    def _push_path(self, ply: int, null_move: bool, irr: int) -> None:
        """Record the position *entered* at ``ply`` on the search path."""
        self.path_keys[ply] = int(self.board.key)
        if null_move:
            # A null move is a search device: the position it produces is not
            # real legal history, so it bounds the repetition scan and no real
            # ply is consumed (board.make leaves abs_ply untouched on FLAG_NULL).
            self.path_null_floor[ply] = ply
            self.path_irr_floor[ply] = self.path_irr_floor[ply - 1]
        else:
            self.path_null_floor[ply] = self.path_null_floor[ply - 1]
            self.path_irr_floor[ply] = ply if irr else self.path_irr_floor[ply - 1]

    # -- evaluation -----------------------------------------------------------

    def _static_eval(self, ply: int) -> int:
        v = self.eval_fn(self.board)
        if v > self.eval_clamp:
            v = self.eval_clamp
        elif v < -self.eval_clamp:
            v = -self.eval_clamp
        return v

    def _corrected(self, raw: int) -> int:
        v = raw + self.hist.correction_cp(self.board, self.board.side)
        if v > self.eval_clamp:
            return self.eval_clamp
        if v < -self.eval_clamp:
            return -self.eval_clamp
        return v

    # -- move ordering --------------------------------------------------------

    def _score_moves(self, n: int, ply: int, tt_move16: int, in_qsearch: bool) -> None:
        board = self.board
        moves = self.moves[ply]
        scores = self.scores[ply]
        sq = board._sq
        hist = self.hist
        hint = self.info.root_hint
        p1 = self.ss_piece[ply + 3]
        t1 = (self.ss_move[ply + 3] >> 6) & 63
        p2 = self.ss_piece[ply + 2]
        t2 = (self.ss_move[ply + 2] >> 6) & 63
        cm = hist.counter_move((p1, t1))
        for i in range(n):
            m = moves[i]
            frm = m & 63
            to = (m >> 6) & 63
            promo = (m >> 12) & 7
            captured = (m >> 18) & 15
            if (m & 0x7FFF) == tt_move16 or (ply == 0 and hint and (m & 0x7FFF) == (hint & 0x7FFF)):
                scores[i] = S_TT
                continue
            piece = sq[frm]
            if captured != 15 or promo:
                vt = _victim_type(captured)
                s = self.params["ord_cap_mvv_mult"] * self.pval[
                    captured if captured != 15 else 0
                ] + hist.capture_score(piece, to, vt)
                if promo:
                    s += (
                        self.params["ord_promo_queen"]
                        if promo == QUEEN
                        else self.params["ord_promo_under"]
                    )
                if (
                    in_qsearch
                    or promo
                    or see_ge(
                        board,
                        m,
                        -self.pval[piece] // self.params["ord_goodcap_see_div"],
                        self.pval,
                        self.see_value,
                    )
                ):
                    scores[i] = S_GOOD_CAPTURE + s
                else:
                    scores[i] = S_BAD_CAPTURE + s
            else:
                if m == self.killers[ply][0]:
                    scores[i] = S_KILLER + 2
                elif m == self.killers[ply][1]:
                    scores[i] = S_KILLER + 1
                elif m == cm:
                    scores[i] = S_KILLER
                else:
                    scores[i] = hist.quiet_score(board, frm, to, piece, (p1, t1), (p2, t2))

    def _pick(self, i: int, n: int, ply: int) -> int:
        moves = self.moves[ply]
        scores = self.scores[ply]
        best = i
        for j in range(i + 1, n):
            if scores[j] > scores[best]:
                best = j
        if best != i:
            moves[i], moves[best] = moves[best], moves[i]
            scores[i], scores[best] = scores[best], scores[i]
        return moves[i]

    # -- draw / terminal heuristics -------------------------------------------

    def _draw_score(self, ply: int, checked: bool) -> int | None:
        """Heuristic draw value, or None. Referee order: mate beats draws, so
        when a draw condition holds while in check we must confirm a legal
        evasion exists before returning DRAW."""
        b = self.board
        path_hit, game_hits = self._rep_scan(ply)
        self.node_rep[ply] = game_hits + 1 + (1 if path_hit else 0)
        reason = None
        if path_hit or game_hits >= 1:
            # Any prior occurrence in the reversible region is a draw inside
            # the tree (the opponent may force the repeat); the referee's
            # threefold/fivefold are real terminals handled by GameState.
            reason = "rep"
        elif b.halfmove >= 100:
            reason = "fifty"
        elif b.absolute_ply() >= PLY_CAP:
            reason = "cap"
        elif _insufficient(b):
            reason = "insufficient"
        if reason is None:
            return None
        if checked:
            n = generate_legal(b, self.moves[ply])
            if n == 0:
                self._note("mate", ply)
                return -MATE + ply
        self._note(reason, ply)
        return DRAW

    # -- qsearch ----------------------------------------------------------------

    def _qs(self, alpha: int, beta: int, ply: int) -> int:
        info = self.info
        board = self.board
        info.nodes += 1
        info.qnodes += 1
        if ply > info.seldepth:
            info.seldepth = ply
        info.phase = 2
        if self.deadline.poll():
            return 0
        checked = in_check(board)
        draw = self._draw_score(ply, checked)
        if draw is not None:
            return draw
        if ply >= SEARCH_PATH - 8:
            return self._static_eval(ply)

        key = int(board.key)
        ctx = self._vctx_at(ply)
        pr = self.tt.probe_into(self._probe, key, ply, ctx)
        # Snapshot: nested searches reuse self._probe and clobber it.
        hit = pr.hit
        tt_move16 = pr.move16
        tt_score = pr.score
        tt_eval = pr.raw_eval if pr.eval_ok else -INF
        tt_bound = pr.bound
        tt_cutoff = pr.cutoff_ok
        if (
            hit
            and tt_cutoff
            and (
                tt_bound == BOUND_EXACT
                or (tt_bound == BOUND_LOWER and tt_score >= beta)
                or (tt_bound == BOUND_UPPER and tt_score <= alpha)
            )
        ):
            self._note("tt_cut", ply)
            return tt_score

        params = self.params
        best_move = 0
        if checked:
            # NO stand-pat in check: every legal evasion is generated and
            # searched; failing low is only decided by the moves themselves.
            static = -INF
            best = -INF
            n = generate_legal(board, self.moves[ply])
            if n == 0:
                self._note("mate", ply)
                return -MATE + ply
        else:
            # Quiet stalemate: stand-pat would otherwise score a drawn
            # position as good. The king test is exact — full generation runs
            # only when a stalemate is actually possible.
            if not _king_has_legal_move(board):
                n_all = generate_legal(board, self.moves[ply])
                if n_all == 0:
                    self._note("stalemate", ply)
                    return DRAW
            static = tt_eval if (hit and tt_eval > -MATE_IN_MAX) else self._static_eval(ply)
            best = self._corrected(static)
            if (
                hit
                and tt_cutoff
                and (
                    (tt_bound == BOUND_LOWER and tt_score > best)
                    or (tt_bound == BOUND_UPPER and tt_score < best)
                )
            ):
                best = tt_score
            if best >= beta:
                if not hit:
                    self.tt.store(key, 0, best, static, 0, BOUND_LOWER, ctx, ply)
                return best
            if best > alpha:
                alpha = best
            n = self._gen_noisy(ply)

        self._score_moves(n, ply, tt_move16 if hit else 0, True)
        futility = (static + params["q_futility"]) if static > -INF else -INF
        for i in range(n):
            m = self._pick(i, n, ply)
            _f, _t, promo, flag, piece, captured = decode_move(m)
            if not checked and best > -MATE_IN_MAX:
                # Tactical pruning is a controlled heuristic: counted, and
                # never applied to a mated-in-few position.
                if captured != 15 and not promo and params["q_futility_on"]:
                    gain = self.pval[captured]
                    if futility + gain <= alpha and not see_ge(
                        board, m, params["q_fut_see_gate"], self.pval, self.see_value
                    ):
                        if best < futility + gain:
                            best = futility + gain
                        self._note(
                            "q_futility",
                            ply,
                            m,
                            detail={"a": alpha, "b": beta, "f": futility + gain},
                        )
                        continue
                if params["q_see_on"] and not see_ge(
                    board, m, -params["q_see"], self.pval, self.see_value
                ):
                    self._note(
                        "q_see",
                        ply,
                        m,
                        detail={"a": alpha, "b": beta},
                    )
                    continue
            self.ss_move[ply + 4] = m
            self.ss_piece[ply + 4] = piece
            self._make(ply, m)
            score = -self._qs(-beta, -alpha, ply + 1)
            self._unmake(ply, m)
            if self.deadline.stop:
                return 0
            if score > best:
                best = score
                if score > alpha:
                    best_move = m
                    if score >= beta:
                        break
                    alpha = score
        if checked and best == -INF:
            best = -MATE + ply
        bound = BOUND_LOWER if best >= beta else BOUND_UPPER
        self.tt.store(
            key,
            best_move & 0x7FFF,
            best,
            static,
            0,
            bound,
            self._vctx_at(ply),
            ply,
        )
        return best

    def _gen_noisy(self, ply: int) -> int:
        """Legal captures + promotions (queen promos and underpromotions)."""
        buf = self.moves[ply]
        tmp = self.scores[ply]  # scratch: reuse score row as a move buffer
        n = generate_legal(self.board, tmp)
        k = 0
        for i in range(n):
            m = tmp[i]
            if (m >> 18) & 15 != 15 or (m >> 12) & 7:
                buf[k] = m
                k += 1
        return k

    # -- transactional make/unmake --------------------------------------------

    def _make(self, ply: int, m: int) -> None:
        irr = _is_irreversible(self.board, m)
        for cb in self.on_make:
            cb(ply, m)
        self.board.make(m)
        self._push_path(ply + 1, False, irr)

    def _unmake(self, ply: int, m: int) -> None:
        self.board.unmake()
        for cb in self.on_unmake:
            cb(ply, m)

    def _make_null(self, ply: int) -> None:
        for cb in self.on_null:
            cb(ply)
        self.board.make_null()
        self._push_path(ply + 1, True, 0)

    def _unmake_null(self, ply: int) -> None:
        self.board.unmake_null()
        for cb in self.on_unnull:
            cb(ply)

    # -- main search ------------------------------------------------------------

    def _ab(self, depth: int, alpha: int, beta: int, ply: int, is_pv: int, cut_node: int) -> int:
        if depth <= 0:
            return self._qs(alpha, beta, ply)
        info = self.info
        board = self.board
        info.nodes += 1
        if ply > info.seldepth:
            info.seldepth = ply
        info.phase = 1
        root = ply == 0
        if self.deadline.poll():
            return 0
        checked = in_check(board)
        if not root:
            draw = self._draw_score(ply, checked)
            if draw is not None:
                return draw
            if ply >= SEARCH_PATH - 8:
                return self._static_eval(ply)
            # Mate-distance pruning: tighten the window to the fastest mate.
            a = -MATE + ply
            if a > alpha:
                alpha = a
            b_ = MATE - ply - 1
            if b_ < beta:
                beta = b_
            if alpha >= beta:
                self._note("mate_dist", ply)
                return alpha
        else:
            # Root adjudication must agree with the official referee
            # exactly: ordinary outcome() (mate > insufficient > stalemate >
            # seventyfive > fivefold), then threefold, then fifty-move, then
            # absolute ply >= PLY_CAP — a mate delivered at the cap is
            # decided by the earlier terminal check, never by the cap. The
            # in-tree _draw_score heuristic (any repeat = draw) is NOT the
            # referee's rule: at the root the real threefold/fivefold counts
            # apply. referee_terminal encodes that ordering; it is diffed
            # 1:1 against the referee over the 50k-position gate.
            if self.state is not None:
                term = referee_terminal(self.state)
                if term is not None:
                    self._note(_ROOT_REASON[term[1]], ply)
                    return -MATE + ply if term[0] != "draw" else DRAW

        key = int(board.key)
        excluded = self.ss_excl[ply + 4]
        ctx = self._vctx_at(ply)
        pr = self.tt.probe_into(self._probe, key, ply, ctx)
        # Snapshot into locals: nested searches reuse self._probe and clobber
        # it, so nothing below may read pr.* after a child call.
        hit = pr.hit
        tt_move16 = pr.move16
        tt_score = pr.score
        tt_eval = pr.raw_eval
        tt_depth = pr.depth
        tt_bound = pr.bound
        tt_cutoff = pr.cutoff_ok
        if not pr.eval_ok:
            tt_eval = -INF
        if hit:
            info.tt_hits += 1
        if (
            not is_pv
            and hit
            and tt_cutoff
            and tt_depth >= depth
            and excluded == 0
            and (
                tt_bound == BOUND_EXACT
                or (tt_bound == BOUND_LOWER and tt_score >= beta)
                or (tt_bound == BOUND_UPPER and tt_score <= alpha)
            )
        ):
            self._note("tt_cut", ply)
            return tt_score

        self.ss_null[ply + 4] = 0
        raw_static = -INF
        if checked:
            static = -INF
            improving = False
        else:
            if hit and tt_eval > -MATE_IN_MAX:
                # Stored RAW eval is model-identical by the value-context gate;
                # corrections are applied on top, never re-stored as raw.
                raw_static = tt_eval
            else:
                raw_static = self._static_eval(ply)
            static = self._corrected(raw_static)
            prev2 = self.ss_eval[ply + 2]
            improving = static > prev2 if prev2 != -INF else True
        self.ss_eval[ply + 4] = static
        self.killers[ply + 2][0] = 0
        self.killers[ply + 2][1] = 0

        params = self.params
        if (
            not is_pv
            and not checked
            and beta < MATE_IN_MAX
            and beta > -MATE_IN_MAX
            and excluded == 0
        ):
            # Reverse futility: a static eval well above beta at a non-PV node.
            if (
                depth <= params["rfp_depth"]
                and static
                - (params["rfp_margin"] * depth - (params["rfp_improving"] if improving else 0))
                >= beta
            ):
                self._note(
                    "rfp",
                    ply,
                    detail={
                        "d": depth,
                        "a": alpha,
                        "b": beta,
                        "s": static,
                        "pv": is_pv,
                        "cn": cut_node,
                    },
                )
                return static
            # Razoring: hopeless nodes verified through qsearch.
            if depth <= params["razor_depth"] and static + params["razor_margin"] * depth <= alpha:
                v = self._qs(alpha, beta, ply)
                if self.deadline.stop:
                    return 0
                self._note(
                    "razor_verify",
                    ply,
                    detail={
                        "d": depth,
                        "a": alpha,
                        "b": beta,
                        "s": static,
                        "v": v,
                        "pv": is_pv,
                        "cn": cut_node,
                    },
                )
                if v <= alpha:
                    self._note(
                        "razor",
                        ply,
                        detail={
                            "d": depth,
                            "a": alpha,
                            "b": beta,
                            "v": v,
                            "pv": is_pv,
                            "cn": cut_node,
                        },
                    )
                    return v
            # Null-move pruning with zugzwang guards + verification.
            stm = board.side
            base = stm * 6
            non_pawn = (
                board._bb[base + KNIGHT]
                | board._bb[base + BISHOP]
                | board._bb[base + ROOK]
                | board._bb[base + QUEEN]
            )
            if (
                depth >= params["nmp_min_depth"]
                and static - beta >= params["nmp_eval_margin"]
                and (not params["nmp_no_consec"] or self.ss_null[ply + 3] == 0)
                and non_pawn != 0
                and ply >= info.null_min_ply
                and (
                    not params["nmp_tt_guard"]
                    or not hit
                    or tt_bound != BOUND_UPPER
                    or tt_score >= beta
                )
            ):
                r = (
                    params["nmp_base"]
                    + depth // params["nmp_depth_div"]
                    + min(
                        (static - beta) // params["nmp_eval_div"],
                        params["nmp_eval_max"],
                    )
                )
                self.ss_null[ply + 4] = 1
                self.ss_move[ply + 4] = 0
                self.ss_piece[ply + 4] = -1
                self._make_null(ply)
                score = -self._ab(depth - r, -beta, -beta + 1, ply + 1, 0, 1 - cut_node)
                self._unmake_null(ply)
                self.ss_null[ply + 4] = 0
                if self.deadline.stop:
                    return 0
                if score >= beta:
                    if score >= MATE_IN_MAX:
                        score = beta
                    if depth < params["nmp_verify_depth"]:
                        self._note(
                            "nmp",
                            ply,
                            detail={
                                "d": depth,
                                "a": alpha,
                                "b": beta,
                                "r": r,
                                "sc": score,
                                "cn": cut_node,
                            },
                        )
                        return score
                    # Verification: re-search with null moves disabled over a
                    # span proportional to the remaining depth (zugzwang guard).
                    saved = info.null_min_ply
                    info.null_min_ply = ply + max(
                        params["nmp_verify_min"],
                        params["nmp_verify_num"] * max(1, depth - r) // params["nmp_verify_den"],
                    )
                    v = self._ab(depth - r, beta - 1, beta, ply, 0, 0)
                    info.null_min_ply = saved
                    if self.deadline.stop:
                        return 0
                    if v >= beta:
                        self._note(
                            "nmp",
                            ply,
                            detail={
                                "d": depth,
                                "a": alpha,
                                "b": beta,
                                "r": r,
                                "sc": score,
                                "vf": v,
                                "cn": cut_node,
                            },
                        )
                        return score
                    self._note(
                        "nmp_verify_fail",
                        ply,
                        detail={"d": depth, "a": alpha, "b": beta, "r": r, "v": v, "cn": cut_node},
                    )
            # ProbCut: a capture that still wins a reduced-depth search at a
            # raised beta prunes the whole node (non-PV only).
            if (
                depth >= params["pc_depth"]
                and beta > -MATE_IN_MAX
                and not (
                    hit
                    and tt_bound == BOUND_LOWER
                    and tt_depth >= depth - params["pc_tt_slack"]
                    and tt_score < beta + params["pc_margin"]
                )
            ):
                pc_beta = beta + params["pc_margin"]
                pc_buf = self.moves[ply]
                n_pc = self._gen_noisy(ply)
                for j in range(n_pc):
                    m2 = pc_buf[j]
                    if not see_ge(
                        board,
                        m2,
                        min(pc_beta - static, params["pc_see_cap"]),
                        self.pval,
                        self.see_value,
                    ):
                        continue
                    self.ss_move[ply + 4] = m2
                    self.ss_piece[ply + 4] = board._sq[m2 & 63]
                    self._make(ply, m2)
                    v = -self._qs(-pc_beta, -pc_beta + 1, ply + 1)
                    if v >= pc_beta and depth - params["pc_depth"] > 0:
                        v = -self._ab(
                            depth - params["pc_depth"],
                            -pc_beta,
                            -pc_beta + 1,
                            ply + 1,
                            0,
                            1,
                        )
                    self._unmake(ply, m2)
                    if self.deadline.stop:
                        return 0
                    if v >= pc_beta:
                        self.ss_move[ply + 4] = 0
                        self.ss_piece[ply + 4] = -1
                        self._note(
                            "probcut",
                            ply,
                            m2,
                            detail={
                                "d": depth,
                                "a": alpha,
                                "b": beta,
                                "pb": pc_beta,
                                "v": v,
                                "cn": cut_node,
                            },
                        )
                        self.tt.store(
                            key,
                            m2 & 0x7FFF,
                            v,
                            raw_static,
                            depth - params["pc_depth"],
                            BOUND_LOWER,
                            self._vctx_at(ply),
                            ply,
                        )
                        return v
                self.ss_move[ply + 4] = 0
                self.ss_piece[ply + 4] = -1

        # Singular extension (capped): if every move except the TT move fails
        # low against tt_score - margin*depth at half depth, the TT move is
        # singular and earns one extra ply. Runs before move generation so the
        # exclusion search cannot clobber this node's move buffers.
        singular = 0
        if (
            not root
            and excluded == 0
            and depth >= params["sing_depth"]
            and hit
            and tt_move16 != 0
            and tt_bound != BOUND_UPPER
            and tt_depth >= depth - params["sing_tt_slack"]
            and abs(tt_score) < MATE_IN_MAX
            and ply < params["sing_ply_cap_mult"] * info.root_depth
        ):
            s_beta = tt_score - params["sing_margin"] * depth
            self.ss_excl[ply + 4] = self._move16_at(ply, tt_move16)
            if self.ss_excl[ply + 4]:
                v = self._ab(
                    (depth - 1) // params["sing_half_div"],
                    s_beta - 1,
                    s_beta,
                    ply,
                    0,
                    cut_node,
                )
                self.ss_excl[ply + 4] = 0
                if self.deadline.stop:
                    return 0
                if v < s_beta:
                    singular = params["sing_ext"]
                    self._note("sing_ext", ply, detail={"d": depth, "sb": s_beta, "v": v})
            else:
                self.ss_excl[ply + 4] = 0
        # Internal iterative reduction.
        if depth >= params["iir_depth"] and (not hit or tt_move16 == 0) and (is_pv or cut_node):
            depth -= 1
            self._note("iir", ply)

        moves = self.moves[ply]
        scores = self.scores[ply]
        quiet_list = self.quiets[ply]
        n = generate_legal(board, moves)
        if n == 0:
            if excluded != 0:
                return alpha
            if checked:
                self._note("mate", ply)
                return -MATE + ply
            self._note("stalemate", ply)
            return DRAW
        self._score_moves(n, ply, tt_move16 if hit else 0, False)
        best = -INF
        best_move = 0
        moves_seen = 0
        quiet_count = 0
        lmp_limit = (params["lmp_base"] + params["lmp_quad"] * depth * depth) // max(
            1, params["lmp_improving_div"] - (1 if improving else 0)
        )
        for i in range(n):
            m = self._pick(i, n, ply)
            if excluded and (m & 0x7FFF) == (excluded & 0x7FFF):
                continue
            mscore = scores[i]
            frm = m & 63
            to = (m >> 6) & 63
            promo = (m >> 12) & 7
            captured = (m >> 18) & 15
            piece = board._sq[frm]
            is_quiet = captured == 15 and promo == 0
            moves_seen += 1
            lmr_r = int(self.lmr[min(depth, 63), min(moves_seen, 63)]) if moves_seen > 1 else 0
            lmr_depth = depth - 1 - lmr_r
            if not root and best > -MATE_IN_MAX and not checked:
                if is_quiet:
                    # A checking move is never prunable-quiet material:
                    # checks are the class of move forced mates hide in, and
                    # a check that mates but is never searched leaves a
                    # poisoned bound in the TT which every later iteration
                    # re-serves (the R6/R7 KRK mate-in-2 loss). `-1` = the
                    # (relatively costly) check test has not run yet.
                    checking = -1
                    if quiet_count >= lmp_limit:
                        checking = _gives_check_fast(board, m)
                        if not checking:
                            self._note(
                                "lmp",
                                ply,
                                m,
                                detail={
                                    "d": depth,
                                    "a": alpha,
                                    "b": beta,
                                    "lim": lmp_limit,
                                    "cn": cut_node,
                                },
                            )
                            continue
                    if (
                        lmr_depth <= params["fut_depth"]
                        and static + params["fut_base"] + params["fut_per_depth"] * lmr_depth
                        <= alpha
                    ):
                        if checking < 0:
                            checking = _gives_check_fast(board, m)
                        if not checking:
                            self._note(
                                "futility",
                                ply,
                                m,
                                detail={
                                    "ld": lmr_depth,
                                    "a": alpha,
                                    "b": beta,
                                    "s": static
                                    + params["fut_base"]
                                    + params["fut_per_depth"] * lmr_depth,
                                    "cn": cut_node,
                                },
                            )
                            continue
                    if lmr_depth <= params["see_quiet_depth"]:
                        ld = (
                            lmr_depth
                            if lmr_depth > params["see_quiet_min_ld"]
                            else params["see_quiet_min_ld"]
                        )
                        if not see_ge(
                            board,
                            m,
                            -params["see_quiet_mult"] * ld * ld,
                            self.pval,
                            self.see_value,
                        ):
                            if checking < 0:
                                checking = _gives_check_fast(board, m)
                            if not checking:
                                self._note(
                                    "see_quiet",
                                    ply,
                                    m,
                                    detail={
                                        "ld": ld,
                                        "a": alpha,
                                        "b": beta,
                                        "d": depth,
                                        "cn": cut_node,
                                    },
                                )
                                continue
                elif (
                    depth <= params["see_noisy_depth"]
                    and mscore < S_KILLER
                    and not see_ge(
                        board,
                        m,
                        -params["see_noisy_mult"] * depth,
                        self.pval,
                        self.see_value,
                    )
                ):
                    self._note(
                        "see_noisy",
                        ply,
                        m,
                        detail={"d": depth, "a": alpha, "b": beta, "cn": cut_node},
                    )
                    continue
            self.ss_move[ply + 4] = m
            self.ss_piece[ply + 4] = piece
            nodes_before = info.nodes
            self._make(ply, m)
            gives_check = in_check(board)
            ext = (
                params["check_ext"]
                if (
                    gives_check
                    and depth < params["check_ext_depth"]
                    and ply < params["check_ply_cap_mult"] * info.root_depth
                )
                else 0
            )
            if ext:
                self._note("check_ext", ply, m, detail={"d": depth})
            if singular and (m & 0x7FFF) == tt_move16:
                ext = singular
            new_depth = depth - 1 + ext
            score = -INF
            do_full = True
            if (
                depth >= params["lmr_min_depth"]
                and moves_seen > (params["lmr_gate_pv"] if is_pv else params["lmr_gate_nonpv"])
                and (is_quiet or mscore < 0)
            ):
                r = lmr_r
                if not improving:
                    r += params["lmr_not_improving"]
                if cut_node:
                    r += params["lmr_cut_node"]
                if is_pv:
                    r -= params["lmr_pv"]
                if mscore >= S_KILLER:
                    r -= params["lmr_killer"]
                elif is_quiet:
                    r -= mscore // params["lmr_hist_div"]
                lo_r = params["lmr_min_r"]
                hi_r = new_depth - params["lmr_max_sub"]
                if hi_r < lo_r:
                    hi_r = lo_r
                if r < lo_r:
                    r = lo_r
                elif r > hi_r:
                    r = hi_r
                if r > 0:
                    self._note(
                        "lmr",
                        ply,
                        m,
                        detail={
                            "d": depth,
                            "r": r,
                            "nd": new_depth,
                            "a": alpha,
                            "b": beta,
                            "cn": cut_node,
                        },
                    )
                    score = -self._ab(new_depth - r, -alpha - 1, -alpha, ply + 1, 0, 1)
                    if self.deadline.stop:
                        # Never start a re-search whose result the unwind
                        # discards — exit through unmake now.
                        self._unmake(ply, m)
                        return 0
                    do_full = score > alpha
                    if do_full:
                        # The reduced search exceeded its verification
                        # condition: mandatory unreduced re-search.
                        self._note(
                            "lmr_research",
                            ply,
                            m,
                            detail={
                                "d": depth,
                                "r": r,
                                "sc": score,
                                "a": alpha,
                                "b": beta,
                                "cn": cut_node,
                            },
                        )
            if do_full and (not is_pv or moves_seen > 1):
                if self.deadline.stop:
                    self._unmake(ply, m)
                    return 0
                score = -self._ab(new_depth, -alpha - 1, -alpha, ply + 1, 0, 1 - cut_node)
            if is_pv and (moves_seen == 1 or (alpha < score < beta)):
                if do_full or score > alpha:
                    self._note("pvs_research", ply, m)
                if self.deadline.stop:
                    self._unmake(ply, m)
                    return 0
                score = -self._ab(new_depth, -beta, -alpha, ply + 1, 1, 0)
            self._unmake(ply, m)
            if self.deadline.stop:
                return 0
            if is_quiet:
                quiet_list[quiet_count] = m
                quiet_count += 1
            if root:
                self._root_scores[m] = score
            if score > best:
                best = score
                if score > alpha:
                    best_move = m
                    if root:
                        info.root_best = m
                        info.root_score = score
                        info.best_nodes = info.nodes - nodes_before
                    if score >= beta:
                        if is_quiet:
                            if m != self.killers[ply][0]:
                                self.killers[ply][1] = self.killers[ply][0]
                                self.killers[ply][0] = m
                            self.hist.update_quiets(
                                board,
                                m,
                                quiet_list,
                                quiet_count,
                                depth,
                                (self.ss_piece[ply + 3], (self.ss_move[ply + 3] >> 6) & 63),
                                (self.ss_piece[ply + 2], (self.ss_move[ply + 2] >> 6) & 63),
                                params["hist_bonus_cap"],
                            )
                        else:
                            self.hist.update_capture(
                                piece,
                                to,
                                _victim_type(captured),
                                depth,
                                params["hist_bonus_cap"],
                            )
                        break
                    alpha = score
        if excluded != 0:
            # Exclusion search: no TT store, no correction update; "no
            # alternative move" means the bound, not a terminal result.
            return alpha if best == -INF else best
        if best >= beta:
            bound = BOUND_LOWER
        elif is_pv and best_move != 0:
            bound = BOUND_EXACT
        else:
            bound = BOUND_UPPER
        if (
            not checked
            and abs(best) < MATE_IN_MAX
            and (best_move == 0 or ((best_move >> 18) & 15) == 15 and ((best_move >> 12) & 7) == 0)
            and not (bound == BOUND_LOWER and best <= static)
            and not (bound == BOUND_UPPER and best >= static)
        ):
            self.hist.update_correction(board, board.side, depth, raw_static, best)
            self._note("corr_update", ply, best_move)
        self.tt.store(
            key,
            best_move & 0x7FFF,
            best,
            raw_static,
            depth,
            bound,
            self._vctx_at(ply),  # recompute: children mutate the shared ctx
            ply,
        )
        return best

    def _move16_at(self, ply: int, move16: int) -> int:
        """Resolve a stored move15 hint to a full legal move at this node."""
        buf = self.scores[ply]  # scratch row, not yet in use at this point
        n = generate_legal(self.board, buf)
        for i in range(n):
            if (buf[i] & 0x7FFF) == move16:
                return buf[i]
        return 0

    # -- iterative deepening driver -------------------------------------------

    def _begin(self, state: GameState, hard_ns: int, node_limit: int) -> None:
        info = self.info
        self.state = state
        self.board = state.board
        self._hist_keys = state._game_keys[: state._game_n]
        self._hist_irr = state._game_irr[: state._game_n - 1]
        self._hist_n = state._game_n
        self._unknown = state.unknown_prefix
        self._model_ver = state.model_version
        self._util_ver = state.utility_version
        info.nodes = 0
        info.qnodes = 0
        info.seldepth = 0
        info.tt_hits = 0
        info.null_min_ply = 0
        info.root_best = 0
        info.root_score = -INF
        info.best_nodes = 0
        self.tt.new_search()
        for k in self.killers:
            k[0] = 0
            k[1] = 0
        self.deadline.hard_ns = int(hard_ns)
        self.deadline.node_limit = int(node_limit)
        self.deadline.check_mask = self._check_mask
        self.deadline.nodes = 0
        self.deadline.stop = False
        self.deadline.clock_reads = 0
        self.deadline.overrun_ns = 0
        # Root path bookkeeping: ply 0 is the current game position.
        self.path_keys[0] = int(self.board.key)
        self.path_null_floor[0] = 0
        self.path_irr_floor[0] = 0
        # The "parent move" slots visible to plies 0-4 hold no move;
        # ss_piece's no-previous-move sentinel is -1 (a white pawn is 0).
        for i in range(8):
            self.ss_move[i] = 0
            self.ss_piece[i] = -1
        _ph, gh = self._rep_scan(0)
        self.node_rep[0] = gh + 1

    def _extract_pv(self, depth: int) -> list[int]:
        """Reconstruct the PV by TT-walking the completed iteration.

        Only called after an iteration completed without stop — an aborted
        iteration never disturbs the committed PV.
        """
        pv: list[int] = []
        board = self.board
        made = 0
        buf = self.scores[SEARCH_PATH - 1]
        probe = Probe()
        for _ply in range(min(depth + 4, SEARCH_PATH // 2)):
            pr = self.tt.probe_into(probe, int(board.key), _ply, None)
            if not pr.hit or pr.move16 == 0:
                break
            n = generate_legal(board, buf)
            found = 0
            for i in range(n):
                if (buf[i] & 0x7FFF) == pr.move16:
                    found = buf[i]
                    break
            if not found:
                break
            pv.append(found)
            board.make(found)
            made += 1
        for _ in range(made):
            board.unmake()
        return pv

    def search(
        self,
        state: GameState,
        soft_ns: int = 0,
        hard_ns: int = 0,
        max_depth: int = MAX_DEPTH,
        node_limit: int = 0,
        root_hint: int = 0,
        scaler: IterationScaler | None = None,
    ) -> SearchResult:
        """Iterative deepening with aspiration and abort-safe results.

        ``soft_ns`` stops new iterations (scaled by the IterationScaler);
        ``hard_ns`` is the monotonic abort deadline. A legal fallback is
        established before any expensive work; if every iteration aborts, the
        fallback is returned. An aborted iteration never overwrites the
        committed (move, score, depth, pv) of the last completed one.
        """
        started = self.now()
        result = SearchResult()
        board = state.board

        # Legal fallback BEFORE any expensive work (spec 3.4).
        n_root = generate_legal(board, self.moves[0])
        if n_root == 0:
            result.aborted = True
            return result
        fallback = self.moves[0][0]
        if root_hint:
            for i in range(n_root):
                if (self.moves[0][i] & 0x7FFF) == (root_hint & 0x7FFF):
                    fallback = self.moves[0][i]
                    break
        result.move = fallback

        self._begin(state, hard_ns, node_limit)
        self.info.root_hint = int(root_hint)
        self._root_scores = {}
        scaler = scaler or IterationScaler()

        # A root already terminal under the referee's ordering is decided
        # once, here — no iteration runs on it. ``_ab`` carries the same
        # check so the root node agrees even when driven directly.
        term = referee_terminal(state)
        if term is not None:
            self._note(_ROOT_REASON[term[1]], 0)
            result.score = DRAW if term[0] == "draw" else -MATE
            result.nodes = self.info.nodes
            result.elapsed_ms = (self.now() - started) / 1e6
            return result

        asp_delta = self.params["asp_delta"]
        asp_min = self.params["asp_min_depth"]
        stable = 0
        last_best = 0
        prev_score = 0
        iter_started = started

        for depth in range(1, max_depth + 1):
            self.info.root_score = -INF
            self.info.root_best = 0  # torn candidates may never leak out
            self.info.best_nodes = 0
            self.info.root_depth = depth
            self._root_scores = {}
            nodes_at_iter_start = self.info.nodes
            for row in range(SEARCH_PATH + 4):
                self.ss_eval[row] = -INF
                self.ss_null[row] = 0
                self.ss_excl[row] = 0
            delta = asp_delta
            if depth >= asp_min:
                alpha = max(prev_score - delta, -INF)
                beta = min(prev_score + delta, INF)
            else:
                alpha, beta = -INF, INF
            score = 0
            while True:
                score = self._ab(depth, alpha, beta, 0, 1, 0)
                if self.deadline.stop:
                    break
                if score <= alpha:
                    if self.params["asp_fail_low_blend"]:
                        beta = (alpha + beta) // 2
                    alpha = max(score - delta, -INF)
                    delta += (
                        delta * self.params["asp_widen_pct"] // 100 + self.params["asp_widen_add"]
                    )
                elif score >= beta:
                    beta = min(score + delta, INF)
                    delta += (
                        delta * self.params["asp_widen_pct"] // 100 + self.params["asp_widen_add"]
                    )
                else:
                    break
            now = self.now()
            if self.deadline.stop:
                # An aborted iteration never overwrites a completed PV:
                # ``result`` keeps the last completed iteration's committed
                # (move, score, depth, pv) — or the pre-search legal fallback
                # when no iteration has completed. ``partial`` records that
                # the in-flight iteration's root results were discarded; it
                # is telemetry, never a provenance claim on the move.
                result.aborted = True
                result.partial = True
                break
            result.move = self.info.root_best
            result.score = score
            result.depth = depth
            result.seldepth = self.info.seldepth
            result.pv = self._extract_pv(depth)
            iter_nodes = self.info.nodes - nodes_at_iter_start
            result.iterations.append((depth, score, self.info.nodes, (now - started) // 1_000_000))
            drop = prev_score - score if depth > 1 else 0
            prev_score = score
            if abs(score) >= MATE_IN_MAX and depth >= self.params["mate_break_depth"]:
                break
            if result.move == last_best:
                stable += 1
            else:
                stable = 0
                last_best = result.move
            iter_ns = now - iter_started
            iter_started = now
            elapsed = now - started
            if soft_ns:
                frac = (
                    self.info.best_nodes / iter_nodes
                    if depth >= self.params["effort_min_depth"] and iter_nodes > 0
                    else None
                )
                if frac is not None:
                    result.best_effort = frac
                gap = self._contender_gap()
                result.contender_gap = gap
                scale = scaler.scale(
                    depth=depth,
                    stable_iters=stable,
                    score_drop_cp=drop,
                    best_move_node_fraction=frac,
                    contender_gap_cp=gap if gap >= 0 else None,
                )
                if elapsed >= soft_ns * scale:
                    break
                if (
                    hard_ns
                    and depth >= self.params["next_iter_min_depth"]
                    and now + iter_ns * self.params["next_iter_cost_pct"] // 100 >= hard_ns
                ):
                    break
        result.nodes = self.info.nodes
        result.qnodes = self.info.qnodes
        result.elapsed_ms = (self.now() - started) / 1e6
        if not result.move:
            result.move = fallback
            result.fallback_used = True
        return result

    def _contender_gap(self) -> int:
        """Score separation between the two best root moves (-1 if unknown)."""
        vals = sorted(self._root_scores.values(), reverse=True)
        if len(vals) < 2:
            return -1
        return vals[0] - vals[1]

    # -- convenience one-shot entry -------------------------------------------

    def choose_move(
        self,
        state: GameState,
        time_left_ms: int,
        increment_ms: int = 500,
        allocator: TimeAllocator | None = None,
        root_answer: Callable[[GameState], int | str | None] | None = None,
        max_depth: int = MAX_DEPTH,
    ) -> str:
        """Full move-choice path: fallback → stored answer → bounded search.

        ``root_answer`` is the book/TB hook (engine.assets, W09): it may
        return a move int, a UCI string, or None. Any miss or exception falls
        back to search; any search failure falls back to the first legal move.
        """
        board = state.board
        n = generate_legal(board, self.moves[0])
        if n == 0:
            return "0000"
        fallback = self.moves[0][0]
        if root_answer is not None:
            try:
                ans = root_answer(state)
            except Exception:
                ans = None
            if ans is not None:
                cand = self._resolve_answer(ans, n)
                if cand:
                    return move_to_uci(cand)
                # stored answer existed but wasn't legal — fall through to
                # search rather than play an illegal move
        allocator = allocator or TimeAllocator()
        alloc = allocator.allocate(time_left_ms, increment_ms, board.absolute_ply())
        res = self.search(
            state,
            soft_ns=alloc.soft_ns,
            hard_ns=self.now() + alloc.hard_ns - allocator.unwind_margin_ns,
            max_depth=max_depth,
        )
        move = res.move or fallback
        # Legality recheck against a freshly generated list before returning
        # (spec 3.1): nothing inside the tree may produce an illegal move.
        n2 = generate_legal(board, self.moves[0])
        legal = False
        for i in range(n2):
            if self.moves[0][i] == move:
                legal = True
                break
        if not legal:
            move = self.moves[0][0] if n2 else fallback
        return move_to_uci(move)

    def _resolve_answer(self, ans: int | str, n: int) -> int:
        buf = self.moves[0]
        if isinstance(ans, str):
            for i in range(n):
                if move_to_uci(buf[i]) == ans:
                    return buf[i]
            return 0
        cand = int(ans)
        for i in range(n):
            if buf[i] == cand or (buf[i] & 0x7FFF) == (cand & 0x7FFF):
                return buf[i]
        return 0

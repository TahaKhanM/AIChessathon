"""Kernel context layout — the flat-array contract for the compiled hot path.

W08-numba work item: ``engine/kernels`` compiles the RX-FINAL hot path
(board, movegen, search, evaluation, TT, histories, hard clock) into Numba
nopython mode.  The pure-Python modules under ``engine/`` remain the scalar
reference oracle; this package never replaces them.

State model — flat "arena" arrays
---------------------------------
Every byte of mutable search/eval state lives in a small set of 1-D NumPy
arrays, one per element dtype, bundled into the 8-tuple ``ctx``::

    ctx = (AU, AI, A32, A16, A8, AU8, AU32, TT)

``AU`` uint64, ``AI`` int64, ``A32`` int32, ``A16`` int16, ``A8`` int8,
``AU8`` uint8, ``AU32`` uint32, ``TT`` the shared int64 transposition
cluster array (``TranspositionTable.clusters`` — the same object the Python
side owns, so ``hashfull``/``audit_entries`` keep working on it).

Logical arrays are fixed-offset regions inside an arena: e.g. the per-ply
move list is ``A32[X_MOVES + ply*MCAP : X_MOVES + ply*MCAP + MCAP]``.
Because Numba array types do not depend on size, a ctx built without model
weights (eval_kind=0) has the *same* type signature as one built with them —
one compiled specialization serves both.

Region order rule: weight-table regions always come LAST in their arena, so
every offset below is a compile-time constant regardless of configuration.

Spec contract honoured here (docs/architecture.md §§3.4/4.5):
explicit preallocated arrays for board state, move lists, search stack,
accumulators and time state; no Python object allocation inside the
recursive loop; monotonic hard clock via the validated objmode bridge.
"""

from __future__ import annotations

import numpy as np

import engine.features as _features
from engine.board import (
    KNIGHT_ATK,
    KING_ATK,
    MASK64,
    MAX_MOVES,
    MAX_PLY,
    PAWN_ATK,
    Board,
    Z_CASTLE,
    Z_EP,
    Z_PIECE,
    Z_SIDE,
)
from engine.movegen import (
    BETWEEN,
    LINE,
    BISHOP_NEG,
    BISHOP_POS,
    ROOK_NEG,
    ROOK_POS,
)
from engine.search import REASONS, SEARCH_PATH
from engine.state import MAX_GAME
from engine import evaluate as _ev

# ---------------------------------------------------------------------------
# Dimensions (all derived from the owning modules — never restated).
# ---------------------------------------------------------------------------

PATH = SEARCH_PATH  # 832 — max search ply index
SS = SEARCH_PATH + 8  # search-stack array length
UNDO = MAX_PLY  # 2048 undo capacity
GAME = MAX_GAME  # 2048 known-history capacity
MCAP = MAX_MOVES  # 256 moves per ply

CHANNELS = _features.CHANNELS
FINNY_FRAMES = 24  # bucket(12) << 1 | mirror(2)

# ---------------------------------------------------------------------------
# ctx tuple indices
# ---------------------------------------------------------------------------

AU = 0  # uint64 arena
AI = 1  # int64  arena
A32 = 2  # int32  arena
A16 = 3  # int16  arena
A8 = 4  # int8   arena
AU8 = 5  # uint8  arena
AU32 = 6  # uint32 arena
CTT = 7  # int64 transposition clusters (shared array)
N_CTX = 8

# ---------------------------------------------------------------------------
# Region offsets — AU (uint64)
# ---------------------------------------------------------------------------

X_BB = 0  # 12 piece bitboards
X_OCC = X_BB + 12  # 3: white occ, black occ, all occ
X_U64 = X_OCC + 3  # 4: key, ep_key, tt_mask, spare
X_UKEY = X_U64 + 4  # 2*UNDO undo keys (key, ep_key)
X_PATHKEY = X_UKEY + 2 * UNDO  # SS path keys
X_HISTKEY = X_PATHKEY + SS  # GAME known-history keys
X_PAWNBB = X_HISTKEY + GAME  # 4*SS pawn bitboards before/after
X_SBB = X_PAWNBB + 4 * SS  # 12 accumulator-shadow bitboards
X_SOCC = X_SBB + 12  # 3 accumulator-shadow occupancy
X_FSIG = X_SOCC + 3  # 12*2*24 finny signatures
AU_LEN = X_FSIG + 12 * 2 * FINNY_FRAMES

# U64 fields (inside X_U64)
J_KEY = 0
J_EPKEY = 1
J_TTMASK = 2

# ---------------------------------------------------------------------------
# Region offsets — AI (int64)
# ---------------------------------------------------------------------------

X_ST = 0  # N_ST scalar fields (I_* below)
X_KING = X_ST + 64  # 2 king squares
X_SKING = X_KING + 2  # 2 shadow king squares
X_LMR = X_SKING + 2  # 64*64 reduction table
X_STATS = X_LMR + 64 * 64  # len(REASONS) reason counters
X_SCR64 = X_STATS + 64  # 64 generic i64 scratch (finny psqt_part)
# X_LOG is last: its size is configurable (0 in production builds).
X_LOG = X_SCR64 + 64
AI_LEN = X_LOG  # + log_rows*5 appended at build

# C_ST scalar fields (int64) — indices relative to X_ST
I_SIDE = 0
I_CASTLE = 1
I_EP = 2
I_HALF = 3
I_FULL = 4
I_ABSPLY = 5
I_UN = 6  # undo depth
I_NODES = 8
I_QNODES = 9
I_SELDEPTH = 10
I_TTHITS = 11
I_ROOTBEST = 12
I_ROOTSCORE = 13
I_ROOTDEPTH = 14
I_BESTNODES = 15
I_NULLMIN = 16
I_PHASE = 17
I_ROOTHINT = 18
I_DL_HARD = 19
I_DL_NLIM = 20
I_DL_NODES = 21
I_DL_STOP = 22
I_DL_MASK = 23
I_CLKREADS = 24
I_OVERRUN = 25
I_UNKNOWN = 26
I_MODELVER = 27
I_UTILVER = 28
I_HISTN = 29
I_EVALKIND = 30  # 0 = simple_eval stand-in, 1 = F512-EF accumulator
I_ADEPTH = 31  # accumulator stack depth
I_SCALENUM = 32
I_SCALESHIFT = 33
I_NEURALBOUND = 34
I_DEBUG = 35  # event-log switch for the parity harness
I_LOGN = 36
I_AGE = 37  # TT age counter
I_WRITES = 38
I_REPLACES = 39
I_FORCEREPLAY = 40  # eval chooser override (test hook)
I_MAXT = 41  # capacity high-water marks
I_MAXPP = 42
I_MAXPSQ = 43
I_CORRWCAP = 44  # correction weight cap (HistoryTables.corr_weight_cap)
I_DL_START = 45  # clock_ns() stamp at k_begin (deadline budget anchor)
N_ST = 64

# Parameter count is needed by the A32 offsets below; the authoritative
# PARAM_ORDER tuple + its assert against PARAM_DEFAULTS lives further down.
N_PARAMS = 88

# ---------------------------------------------------------------------------
# Region offsets — A32 (int32)
# ---------------------------------------------------------------------------

X_MOVES = 0  # PATH*MCAP move lists
X_SCORES = X_MOVES + PATH * MCAP
X_QUIETS = X_SCORES + PATH * MCAP
X_SS_MOVE = X_QUIETS + PATH * MCAP
X_SS_PIECE = X_SS_MOVE + SS
X_SS_EVAL = X_SS_PIECE + SS
X_SS_NULL = X_SS_EVAL + SS
X_SS_EXCL = X_SS_NULL + SS
X_KILLERS = X_SS_EXCL + SS  # PATH*2
X_PNF = X_KILLERS + PATH * 2
X_PIF = X_PNF + SS
X_NODEREP = X_PIF + SS
X_ROOTSC = X_NODEREP + SS  # MCAP root per-move scores
X_PARAMS = X_ROOTSC + MCAP  # N_PARAMS
X_HQUIET = X_PARAMS + N_PARAMS
X_HCAP = X_HQUIET + 2 * 64 * 64  # 13*64*7
X_HCONT = X_HCAP + 13 * 64 * 7  # 2*13*64*13*64
X_HCOUNTER = X_HCONT + 2 * 13 * 64 * 13 * 64  # 13*64
X_HPAWN = X_HCOUNTER + 13 * 64  # 512*13*64
X_HTHREAT = X_HPAWN + 512 * 13 * 64  # 2*2*64*64
X_CORR = X_HTHREAT + 2 * 2 * 64 * 64  # 3*2*16384
X_PSQTA = X_CORR + 3 * 2 * 16384  # SS*2*8 per-perspective PSQT sums
X_TN = X_PSQTA + SS * 2 * 8  # SS
X_PN = X_TN + SS  # SS
X_RBADD = X_PN + SS  # 512 row-list scratch
X_RBREM = X_RBADD + 512  # 512
X_THRENUM = X_RBREM + 512  # MAX_ACTIVE_THREATS+8
X_PPENUM = X_THRENUM + _ev.MAX_ACTIVE_THREATS + 8  # 160
X_PPBUFA = X_PPENUM + 160  # PP_OP_CAP+8
X_PPBUFR = X_PPBUFA + _ev.PP_OP_CAP + 8  # PP_OP_CAP+8
X_PSQENUM = X_PPBUFR + _ev.PP_OP_CAP + 8  # 40
X_PSQPART = X_PSQENUM + 40  # 512 refresh PSQ partial
X_FPSQP = X_PSQPART + CHANNELS  # 2*24*512 finny psq partials
X_FPSQT = X_FPSQP + 2 * FINNY_FRAMES * CHANNELS  # 2*24*8
X_PVBUF = X_FPSQT + 2 * FINNY_FRAMES * 8  # 256 PV extraction
X_EVALX = X_PVBUF + MCAP  # 512 head input
# --- weight-table regions (LAST in arena; size 0 when eval_kind == 0) ---
X_B1 = X_EVALX + CHANNELS  # 8*16
X_B2 = X_B1 + 8 * 16  # 8*32
X_B3 = X_B2 + 8 * 32  # 8*4
X_PSQTB = X_B3 + 8 * 4  # 8
A32_LEN = X_PSQTB + 8

# ---------------------------------------------------------------------------
# Region offsets — A16 (int16)
# ---------------------------------------------------------------------------

X_ACC = 0  # SS*2*CHANNELS accumulator stack
X_FACC = X_ACC + SS * 2 * CHANNELS  # 2*24*512 finny accumulators
# --- weights LAST ---
X_BIAS = X_FACC + 2 * FINNY_FRAMES * CHANNELS  # 512
X_PSQW = X_BIAS + CHANNELS  # 9216*512
X_PSQTW = X_PSQW + 9216 * CHANNELS  # 9216*8
A16_LEN = X_PSQTW + 9216 * 8

# ---------------------------------------------------------------------------
# Region offsets — A8 (int8)
# ---------------------------------------------------------------------------

X_MB = 0  # 64 mailbox
X_SMB = X_MB + 64  # 64 accumulator-shadow mailbox
X_POPS = X_SMB + 64  # SS*8*3 dirty piece ops
X_HISTIRR = X_POPS + SS * 8 * 3  # GAME irreversibility marks
# --- weights LAST ---
X_THRW = X_HISTIRR + GAME  # 59808*512
X_PPW = X_THRW + 59808 * CHANNELS  # 1488*512
X_W1 = X_PPW + 1488 * CHANNELS  # 8*16*512
X_W2 = X_W1 + 8 * 16 * CHANNELS  # 8*32*32
X_W3 = X_W2 + 8 * 32 * 32  # 8*4*96
A8_LEN = X_W3 + 8 * 4 * 96

# ---------------------------------------------------------------------------
# Region offsets — AU8 (uint8), AU32 (uint32)
# ---------------------------------------------------------------------------

X_AVALID = 0  # SS*2
X_AFRAME = X_AVALID + SS * 2  # SS*2
X_AREFRESH = X_AFRAME + SS * 2  # SS*2
X_FVALID = X_AREFRESH + SS * 2  # 2*24
AU8_LEN = X_FVALID + 2 * FINNY_FRAMES

X_TOPS = 0  # SS*96 threat-op lists
AU32_LEN = X_TOPS + SS * _ev.THREAT_OP_CAP

# C_U_MISC-equivalent undo layout now lives in A32 (int32) — see below.
X_U_MOVE = A32_LEN  # placeholder marker (real region appended below)

# Undo regions in A32 — appended after the (fixed) layout above but BEFORE
# the weight regions would sit...  To keep every offset constant we place
# undo arrays right after X_PSQTB; weights are re-located after them.
X_U_MOVE = X_PSQTB + 8  # UNDO undo move
X_U_MISC = X_U_MOVE + UNDO  # UNDO*6 undo scalars
# shift the weight block start accordingly
X_B1 = X_U_MISC + UNDO * 6
X_B2 = X_B1 + 8 * 16
X_B3 = X_B2 + 8 * 32
X_PSQTB = X_B3 + 8 * 4
A32_LEN = X_PSQTB + 8

U_CASTLE, U_EP, U_HALF, U_FULL, U_KING, U_PLY = range(6)

LOG_STRIDE = 5
ROOT_SCORE_SENT = np.int32(0x7FFFFFFF)

# ---------------------------------------------------------------------------
# Shared read-only geometry tables (built once from the W01/W02 modules; the
# compiled kernels index them as globals — Numba freezes the array objects).
# ---------------------------------------------------------------------------

TABLES = {
    "PAWN_ATK": np.asarray(PAWN_ATK, dtype=np.uint64),  # [2,64]
    "KNIGHT_ATK": np.asarray(KNIGHT_ATK, dtype=np.uint64),  # [64]
    "KING_ATK": np.asarray(KING_ATK, dtype=np.uint64),  # [64]
    "BETWEEN": np.asarray(BETWEEN, dtype=np.uint64),  # [64,64]
    "LINE": np.asarray(LINE, dtype=np.uint64),  # [64,64]
    "RPOS": np.asarray(ROOK_POS, dtype=np.uint64),  # [64,2]
    "RNEG": np.asarray(ROOK_NEG, dtype=np.uint64),  # [64,2]
    "BPOS": np.asarray(BISHOP_POS, dtype=np.uint64),  # [64,2]
    "BNEG": np.asarray(BISHOP_NEG, dtype=np.uint64),  # [64,2]
    "Z_PIECE": np.asarray(Z_PIECE, dtype=np.uint64),  # [12,64]
    "Z_CASTLE": np.asarray(Z_CASTLE, dtype=np.uint64),  # [16]
    "Z_EP": np.asarray(Z_EP, dtype=np.uint64),  # [8]
    "Z_SIDE": np.uint64(Z_SIDE),
    "CASTLE_CLEAR": np.asarray(
        __import__("engine.board", fromlist=["CASTLE_CLEAR"]).CASTLE_CLEAR,
        dtype=np.int64,
    ),
    "T_OFF": _ev._T_OFF,  # int32[12,64]   threat offsets (engine codes)
    "T_SUB": _ev._T_SUB,  # int16[12,64,64] threat sub-indices
    "T_BASE": _ev._T_BASE,  # int32[12,12,2] threat pair bases
    "RAY_PASS": _ev.RAY_PASS_A,  # u64[64,64] ray-through-b
    "PP_MASK": _ev.PP_MASK_A,  # u64[64]    pawn-pair partner mask
    "PP_DENSE": _ev.PP_DENSE_A,  # i32[4560]  triangular->dense row
    "K12": _ev.K12_A,  # u8[64]     king bucket by oriented square
    "PSEUDO": _ev._T["PSEUDO"],  # u64[6,64]  pseudo attacks, empty board
    "POP16": _ev.POP16_A,  # u8[65536]  popcount
    "MSB8": _ev.MSB8_A,  # u8[256]    msb index
}
# simple_eval piece-square table as a dense [6,64] i32 — engine.search._PSQT
# is keyed by piece TYPE with black squares flipped (s ^ 56) at use site.
from engine.search import _PSQT as _PSQT_DICT  # noqa: E402

TABLES["PSQT"] = np.stack([np.asarray(_PSQT_DICT[t], dtype=np.int32) for t in range(6)])
MASK64_U = np.uint64(MASK64)

# ---------------------------------------------------------------------------
# Parameter packing — fixed order, asserted against DEFAULT_PARAMS.
# ---------------------------------------------------------------------------

PARAM_ORDER = (
    "rfp_depth",
    "rfp_margin",
    "rfp_improving",
    "razor_depth",
    "razor_margin",
    "nmp_min_depth",
    "nmp_base",
    "nmp_depth_div",
    "nmp_eval_div",
    "nmp_eval_max",
    "nmp_verify_depth",
    "lmr_base_x100",
    "lmr_div_x100",
    "lmr_hist_div",
    "lmr_min_depth",
    "lmp_base",
    "lmp_quad",
    "fut_depth",
    "fut_base",
    "fut_per_depth",
    "see_quiet_depth",
    "see_quiet_mult",
    "see_noisy_depth",
    "see_noisy_mult",
    "sing_depth",
    "sing_margin",
    "sing_tt_slack",
    "asp_delta",
    "asp_min_depth",
    "q_futility",
    "q_see",
    "check_ext_depth",
    "iir_depth",
    "hist_bonus_cap",
    "pc_depth",
    "pc_margin",
    "pc_see_cap",
    "val_pawn",
    "val_knight",
    "val_bishop",
    "val_rook",
    "val_queen",
    "val_king_see",
    "ord_cap_mvv_mult",
    "ord_promo_queen",
    "ord_promo_under",
    "ord_goodcap_see_div",
    "q_futility_on",
    "q_see_on",
    "q_fut_see_gate",
    "nmp_eval_margin",
    "nmp_no_consec",
    "nmp_tt_guard",
    "nmp_verify_min",
    "nmp_verify_num",
    "nmp_verify_den",
    "pc_tt_slack",
    "sing_ext",
    "sing_half_div",
    "sing_ply_cap_mult",
    "check_ext",
    "check_ply_cap_mult",
    "lmp_improving_div",
    "see_quiet_min_ld",
    "lmr_gate_pv",
    "lmr_gate_nonpv",
    "lmr_not_improving",
    "lmr_cut_node",
    "lmr_pv",
    "lmr_killer",
    "lmr_min_r",
    "lmr_max_sub",
    "asp_widen_pct",
    "asp_widen_add",
    "asp_fail_low_blend",
    "mate_break_depth",
    "effort_min_depth",
    "next_iter_min_depth",
    "next_iter_cost_pct",
    "hist_bonus_quad",
    "hist_bonus_lin",
    "hist_bonus_const",
    "hist_pawn_div",
    "hist_threat_div",
    "corr_pawn_w",
    "corr_np_w",
    "corr_weight_cap",
    "eval_clamp",
)
from engine.search import PARAM_DEFAULTS  # noqa: E402

assert tuple(PARAM_DEFAULTS.keys()) == PARAM_ORDER, (
    "PARAM_DEFAULTS changed; update PARAM_ORDER to match"
)

P_ = {name: i for i, name in enumerate(PARAM_ORDER)}
assert len(PARAM_ORDER) == N_PARAMS, "N_PARAMS constant drifted"

# Compile-time parameter indices for njit kernels — Numba cannot index a dict
# in nopython mode, so each parameter gets a module-level int constant named
# ``P_<name>``.  These are verified equal to the PARAM_ORDER positions above.
P_rfp_depth = 0
P_rfp_margin = 1
P_rfp_improving = 2
P_razor_depth = 3
P_razor_margin = 4
P_nmp_min_depth = 5
P_nmp_base = 6
P_nmp_depth_div = 7
P_nmp_eval_div = 8
P_nmp_eval_max = 9
P_nmp_verify_depth = 10
P_lmr_base_x100 = 11
P_lmr_div_x100 = 12
P_lmr_hist_div = 13
P_lmr_min_depth = 14
P_lmp_base = 15
P_lmp_quad = 16
P_fut_depth = 17
P_fut_base = 18
P_fut_per_depth = 19
P_see_quiet_depth = 20
P_see_quiet_mult = 21
P_see_noisy_depth = 22
P_see_noisy_mult = 23
P_sing_depth = 24
P_sing_margin = 25
P_sing_tt_slack = 26
P_asp_delta = 27
P_asp_min_depth = 28
P_q_futility = 29
P_q_see = 30
P_check_ext_depth = 31
P_iir_depth = 32
P_hist_bonus_cap = 33
P_pc_depth = 34
P_pc_margin = 35
P_pc_see_cap = 36
P_val_pawn = 37
P_val_knight = 38
P_val_bishop = 39
P_val_rook = 40
P_val_queen = 41
P_val_king_see = 42
P_ord_cap_mvv_mult = 43
P_ord_promo_queen = 44
P_ord_promo_under = 45
P_ord_goodcap_see_div = 46
P_q_futility_on = 47
P_q_see_on = 48
P_q_fut_see_gate = 49
P_nmp_eval_margin = 50
P_nmp_no_consec = 51
P_nmp_tt_guard = 52
P_nmp_verify_min = 53
P_nmp_verify_num = 54
P_nmp_verify_den = 55
P_pc_tt_slack = 56
P_sing_ext = 57
P_sing_half_div = 58
P_sing_ply_cap_mult = 59
P_check_ext = 60
P_check_ply_cap_mult = 61
P_lmp_improving_div = 62
P_see_quiet_min_ld = 63
P_lmr_gate_pv = 64
P_lmr_gate_nonpv = 65
P_lmr_not_improving = 66
P_lmr_cut_node = 67
P_lmr_pv = 68
P_lmr_killer = 69
P_lmr_min_r = 70
P_lmr_max_sub = 71
P_asp_widen_pct = 72
P_asp_widen_add = 73
P_asp_fail_low_blend = 74
P_mate_break_depth = 75
P_effort_min_depth = 76
P_next_iter_min_depth = 77
P_next_iter_cost_pct = 78
P_hist_bonus_quad = 79
P_hist_bonus_lin = 80
P_hist_bonus_const = 81
P_hist_pawn_div = 82
P_hist_threat_div = 83
P_corr_pawn_w = 84
P_corr_np_w = 85
P_corr_weight_cap = 86
P_eval_clamp = 87
for _name, _idx in P_.items():
    _g = globals()["P_" + _name]
    assert _g == _idx, f"P_{_name} constant {_g} != PARAM_ORDER index {_idx}"
del _name, _idx, _g

REASON_INDEX = {name: i for i, name in enumerate(REASONS)}
N_REASONS = len(REASONS)

LOG_ROWS = 1 << 22


def build_ctx(
    *,
    tt_clusters: np.ndarray,
    hist_arrays: tuple[np.ndarray, ...] | None = None,
    params: dict | None = None,
    lmr: np.ndarray | None = None,
    weights=None,
    eval_kind: int = 0,
    log_rows: int = 0,
) -> tuple:
    """Allocate the arena context.

    ``tt_clusters`` is the shared ``TranspositionTable.clusters`` array (the
    Python-side object stays authoritative for diagnostics).  ``hist_arrays``
    are the six ordering tables plus the correction table — passed *flat*
    views are created here so Python sees normal shaped arrays while the
    kernels index the same memory.
    """
    from engine.search import load_search_params

    merged = load_search_params(params)
    par = np.asarray([merged[k] for k in PARAM_ORDER], dtype=np.int64)
    if lmr is None:
        from engine.search import build_lmr_table

        lmr = build_lmr_table(merged["lmr_base_x100"], merged["lmr_div_x100"])
    if hist_arrays is None:
        from engine.history import HistoryTables

        h = HistoryTables()
        hist_arrays = (h.quiet, h.capture, h.cont, h.counter, h.pawn, h.threat, h.corr)
        corr_cap = merged["corr_weight_cap"]
    else:
        corr_cap = merged["corr_weight_cap"]

    has_w = weights is not None
    if has_w:
        eval_kind = 1

    au = np.zeros(AU_LEN, np.uint64)
    ai = np.zeros(AI_LEN + log_rows * LOG_STRIDE, np.int64)
    a32n = X_PSQTB + 8 if has_w else X_U_MISC + UNDO * 6
    a32 = np.zeros(a32n, np.int32)
    a16n = A16_LEN if has_w else X_BIAS
    a16 = np.zeros(a16n, np.int16)
    a8n = A8_LEN if has_w else X_THRW
    a8 = np.zeros(a8n, np.int8)
    au8 = np.zeros(AU8_LEN, np.uint8)
    au32 = np.zeros(AU32_LEN, np.uint32)

    st = ai[X_ST : X_ST + N_ST]
    st[I_EVALKIND] = eval_kind
    st[I_CORRWCAP] = corr_cap
    ai[X_ST + I_SCALENUM] = _ev.SCALE_NUM
    ai[X_ST + I_SCALESHIFT] = _ev.SCALE_SHIFT
    ai[X_ST + I_NEURALBOUND] = _ev.NEURAL_BOUND
    au[X_U64 + J_TTMASK] = np.uint64(tt_clusters.shape[0] - 1)

    a32[X_PARAMS : X_PARAMS + N_PARAMS] = par.astype(np.int32)
    ai[X_LMR : X_LMR + 64 * 64] = np.ascontiguousarray(lmr, np.int64).reshape(-1)
    # ss_piece no-previous-move sentinel is -1 (a white pawn is 0).
    a32[X_SS_PIECE : X_SS_PIECE + SS] = -1

    # history tables copied flat into the i32 arena; the shaped views are
    # handed back so callers can also mutate them through the normal names.
    flats = (
        (X_HQUIET, hist_arrays[0]),
        (X_HCAP, hist_arrays[1]),
        (X_HCONT, hist_arrays[2]),
        (X_HCOUNTER, hist_arrays[3]),
        (X_HPAWN, hist_arrays[4]),
        (X_HTHREAT, hist_arrays[5]),
        (X_CORR, hist_arrays[6]),
    )
    for off, arr in flats:
        a32[off : off + arr.size] = arr.reshape(-1)

    if has_w:
        w = weights
        a16[X_BIAS : X_BIAS + CHANNELS] = w.bias
        a16[X_PSQW : X_PSQW + w.psq_w.size] = w.psq_w.reshape(-1)
        a16[X_PSQTW : X_PSQTW + w.psqt_w.size] = w.psqt_w.reshape(-1)
        a8[X_THRW : X_THRW + w.thr_w.size] = w.thr_w.reshape(-1)
        a8[X_PPW : X_PPW + w.pp_w.size] = w.pp_w.reshape(-1)
        a8[X_W1 : X_W1 + w.w1.size] = w.w1.reshape(-1)
        a8[X_W2 : X_W2 + w.w2.size] = w.w2.reshape(-1)
        a8[X_W3 : X_W3 + w.w3.size] = w.w3.reshape(-1)
        a32[X_B1 : X_B1 + w.b1.size] = w.b1.reshape(-1)
        a32[X_B2 : X_B2 + w.b2.size] = w.b2.reshape(-1)
        a32[X_B3 : X_B3 + w.b3.size] = w.b3.reshape(-1)
        a32[X_PSQTB : X_PSQTB + w.psqt_b.size] = w.psqt_b.reshape(-1)
        st[I_SCALENUM] = int(w.scale_num)
        st[I_SCALESHIFT] = int(w.scale_shift)
        st[I_NEURALBOUND] = int(w.neural_bound)

    return (au, ai, a32, a16, a8, au8, au32, tt_clusters)


# -- python-side shaped views over the arenas (driver/tests only) ------------


def views(ctx: tuple) -> dict:
    """Shaped numpy views over the arena regions, for driver/test code."""
    au, ai, a32, a16, a8, au8, au32, tt = ctx
    return {
        "bb": au[X_BB : X_BB + 12],
        "mb": a8[X_MB : X_MB + 64],
        "occ": au[X_OCC : X_OCC + 3],
        "king": ai[X_KING : X_KING + 2],
        "st": ai[X_ST : X_ST + N_ST],
        "u64": au[X_U64 : X_U64 + 4],
        "moves": a32[X_MOVES : X_MOVES + PATH * MCAP].reshape(PATH, MCAP),
        "scores": a32[X_SCORES : X_SCORES + PATH * MCAP].reshape(PATH, MCAP),
        "quiets": a32[X_QUIETS : X_QUIETS + PATH * MCAP].reshape(PATH, MCAP),
        "ss_move": a32[X_SS_MOVE : X_SS_MOVE + SS],
        "ss_piece": a32[X_SS_PIECE : X_SS_PIECE + SS],
        "ss_eval": a32[X_SS_EVAL : X_SS_EVAL + SS],
        "ss_null": a32[X_SS_NULL : X_SS_NULL + SS],
        "ss_excl": a32[X_SS_EXCL : X_SS_EXCL + SS],
        "killers": a32[X_KILLERS : X_KILLERS + PATH * 2].reshape(PATH, 2),
        "pathkey": au[X_PATHKEY : X_PATHKEY + SS],
        "pnf": a32[X_PNF : X_PNF + SS],
        "pif": a32[X_PIF : X_PIF + SS],
        "noderep": a32[X_NODEREP : X_NODEREP + SS],
        "rootsc": a32[X_ROOTSC : X_ROOTSC + MCAP],
        "params": a32[X_PARAMS : X_PARAMS + N_PARAMS],
        "histkey": au[X_HISTKEY : X_HISTKEY + GAME],
        "histirr": a8[X_HISTIRR : X_HISTIRR + GAME],
        "u_move": a32[X_U_MOVE : X_U_MOVE + UNDO],
        "u_misc": a32[X_U_MISC : X_U_MISC + UNDO * 6].reshape(UNDO, 6),
        "u_key": au[X_UKEY : X_UKEY + UNDO * 2].reshape(UNDO, 2),
        "acc": a16[X_ACC : X_ACC + SS * 2 * CHANNELS].reshape(SS, 2, CHANNELS),
        "psqt": a32[X_PSQTA : X_PSQTA + SS * 2 * 8].reshape(SS, 2, 8),
        "avalid": au8[X_AVALID : X_AVALID + SS * 2].reshape(SS, 2),
        "aframe": au8[X_AFRAME : X_AFRAME + SS * 2].reshape(SS, 2),
        "arefresh": au8[X_AREFRESH : X_AREFRESH + SS * 2].reshape(SS, 2),
        "tops": au32[X_TOPS : X_TOPS + SS * _ev.THREAT_OP_CAP].reshape(SS, _ev.THREAT_OP_CAP),
        "tn": a32[X_TN : X_TN + SS],
        "pops": a8[X_POPS : X_POPS + SS * 8 * 3].reshape(SS, 8, 3),
        "pn": a32[X_PN : X_PN + SS],
        "pawnbb": au[X_PAWNBB : X_PAWNBB + SS * 4].reshape(SS, 4),
        "smb": a8[X_SMB : X_SMB + 64],
        "sbb": au[X_SBB : X_SBB + 12],
        "socc": au[X_SOCC : X_SOCC + 3],
        "sking": ai[X_SKING : X_SKING + 2],
        "tt": tt,
        "log": ai[X_LOG : X_LOG + (ai.size - X_LOG) // LOG_STRIDE * LOG_STRIDE].reshape(
            -1, LOG_STRIDE
        )
        if ai.size > X_LOG
        else np.zeros((0, LOG_STRIDE), np.int64),
    }


def load_board(ctx: tuple, board: Board) -> None:
    """Copy a ``Board`` into the kernel arenas (search-root sync)."""
    au, ai, a32, a16, a8, au8, au32, tt = ctx
    au[X_BB : X_BB + 12] = np.asarray(board._bb, dtype=np.uint64)
    a8[X_MB : X_MB + 64] = np.asarray(board._sq, dtype=np.int8)
    au[X_OCC] = np.uint64(board._occ[0])
    au[X_OCC + 1] = np.uint64(board._occ[1])
    au[X_OCC + 2] = np.uint64(board._occ_all)
    ai[X_KING] = board._king[0]
    ai[X_KING + 1] = board._king[1]
    st = ai[X_ST : X_ST + N_ST]
    st[I_SIDE] = board.side
    st[I_CASTLE] = board.castling
    st[I_EP] = board.ep_square
    st[I_HALF] = board.halfmove
    st[I_FULL] = board.fullmove
    st[I_ABSPLY] = board.absolute_ply()
    st[I_UN] = 0
    au[X_U64 + J_KEY] = np.uint64(int(board.key) & MASK64)
    au[X_U64 + J_EPKEY] = np.uint64(board._ep_key & MASK64)


def board_key(ctx: tuple) -> int:
    return int(ctx[AU][X_U64 + J_KEY])


def assert_board_matches(ctx: tuple, board: Board, un_base: int = 0) -> None:
    """Abort-gate check: kernel state mirrors the reference board exactly.

    ``un_base`` is the expected kernel undo depth — 0 after a balanced
    search, ``board._un - base_at_load`` mid-playout when checked against a
    board whose own undo stack includes game moves.
    """
    au, ai, a32, a16, a8, au8, au32, tt = ctx
    assert np.array_equal(au[X_BB : X_BB + 12], np.asarray(board._bb, np.uint64))
    assert np.array_equal(a8[X_MB : X_MB + 64], np.asarray(board._sq, np.int8))
    assert int(au[X_OCC]) == board._occ[0]
    assert int(au[X_OCC + 1]) == board._occ[1]
    assert int(au[X_OCC + 2]) == board._occ_all
    st = ai[X_ST : X_ST + N_ST]
    assert int(st[I_SIDE]) == board.side
    assert int(st[I_CASTLE]) == board.castling
    assert int(st[I_EP]) == board.ep_square
    assert int(st[I_HALF]) == board.halfmove
    assert int(st[I_FULL]) == board.fullmove
    assert int(st[I_ABSPLY]) == board.absolute_ply()
    assert int(st[I_UN]) == un_base, f"kernel undo depth {int(st[I_UN])} != expected {un_base}"
    assert board_key(ctx) == int(board.key)
    assert int(au[X_U64 + J_EPKEY]) == (board._ep_key & MASK64)
    assert int(ai[X_KING]) == board._king[0]
    assert int(ai[X_KING + 1]) == board._king[1]

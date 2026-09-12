"""Neural runtime adapter: observed FEN and clock to legal UCI move.

Connects model loading, history reconciliation, root answer eligibility,
bounded Python search, incremental integer evaluation and final legality
validation. The public root adapter uses the classical baseline instead.

Model loading and warmup belong in initialization. The evaluator tracks
transactional make/unmake callbacks. Unknown history remains explicit;
unmatched observations reset state and emit diagnostics. Evaluator faults
fall back to the classical evaluator for the remainder of the call.

Diagnostics are JSON lines on stderr. Random weights are available only
for deterministic contract tests and do not establish playing strength."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from engine.board import BK, WK, Board, move_to_uci, snapshot
from engine.clock import now_ns
from engine.clock_b import SoftAllocator, SoftScaler
from engine.evaluate import Evaluator, EvalWeights
from engine import model_io
from engine.movegen import generate_legal
from engine.search import Searcher, SearchResult, simple_eval
from engine.state import GameState, referee_terminal
from engine.tt import TranspositionTable
from engine.history import HistoryTables

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

INCREMENT_MS = 500
PLY_CAP = 600
OPENING_BOOK_MAX_FULLMOVE = 20
TB_MAX_PIECES = 7
DEFAULT_TT_MIB = 128
# Pure-Python nodes cost ~0.3-0.8 ms each once the Numba evaluator is on the
# path; mask 7 bounds post-deadline slack to ~4-6 ms inside the 8 ms unwind
# margin (search.check_mask default 15 was sized against ~0.2 ms nodes).
CHECK_MASK = 7
# Deterministic bounded-random weights for the qualification build. When the
# NPS-FALSIFIER artifact lands it is a byte-identical-format RXF1 file and
# only the seed/provenance change.
DEFAULT_SEED = 0x8F512E
MODEL_VERSION = 1
UTILITY_VERSION = 0

_MOVES = [0] * 256


# ---------------------------------------------------------------------------
# Model <-> EvalWeights adapters
# ---------------------------------------------------------------------------


def weights_from_model_dict(model: dict) -> EvalWeights:
    """RXF1 decoded sections ([in][out] head layout) -> EvalWeights ([out][in]).

    Delegates to ``EvalWeights.from_model`` — the single adapter site, so the
    [in][out]->[out][in] transpose and meta propagation cannot fork.
    """
    return EvalWeights.from_model(model)


def model_dict_from_weights(w: EvalWeights, meta: dict | None = None) -> dict:
    """EvalWeights ([out][in] heads) -> RXF1 section dict ([in][out] heads)."""
    merged = {
        "scale_num": int(w.scale_num),
        "scale_shift": int(w.scale_shift),
        "neural_bound": int(w.neural_bound),
    }
    if meta:
        merged.update(meta)
    return {
        "bias": np.asarray(w.bias, dtype=np.int16),
        "psq": np.asarray(w.psq_w, dtype=np.int16),
        "thr": np.asarray(w.thr_w, dtype=np.int8),
        "pp": np.asarray(w.pp_w, dtype=np.int8),
        "head_w1": np.ascontiguousarray(w.w1.transpose(0, 2, 1)),
        "head_b1": np.asarray(w.b1, dtype=np.int32),
        "head_w2": np.ascontiguousarray(w.w2.transpose(0, 2, 1)),
        "head_b2": np.asarray(w.b2, dtype=np.int32),
        "head_w3": np.ascontiguousarray(w.w3.transpose(0, 2, 1)),
        "head_b3": np.asarray(w.b3, dtype=np.int32),
        "psqt_w": np.asarray(w.psqt_w, dtype=np.int16),
        "psqt_b": np.asarray(w.psqt_b, dtype=np.int32),
        "_meta": merged,
    }


# ---------------------------------------------------------------------------
# Memory-bounded model decode
# ---------------------------------------------------------------------------


def read_model_frugal(blob: bytes) -> dict:
    """Backward-compatible alias for ``model_io.read_model``.

    The bounded chunked decode now lives in ``engine.model_io`` itself
    (``DECODE_CHUNK_FIELDS``-field chunks through the same oracle-exact
    kernel): the unbounded whole-section path no longer exists, so this
    alias simply forwards to the canonical loader. Kept for callers that
    predate the promotion.
    """
    return model_io.read_model(blob)


def load_weights(path: str | Path) -> tuple[EvalWeights, dict]:
    """Load an RXF1 container stored as a uint8 ``.npy`` (weights.npy)."""
    blob = np.load(Path(path), allow_pickle=False).tobytes()
    model = read_model_frugal(blob)
    return weights_from_model_dict(model), model.get("_meta", {})


# ---------------------------------------------------------------------------
# Diagnostics: one JSON object per line on stderr (never on the protocol pipe)
# ---------------------------------------------------------------------------


def _emit(payload: dict) -> None:
    try:
        line = json.dumps(payload, separators=(",", ":"), default=str)
        if len(line) > 1900:
            line = line[:1900]
        print(line, file=sys.stderr, flush=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# The wired agent
# ---------------------------------------------------------------------------


class Agent:
    """One process = one game. Holds every piece of mutable play state."""

    def __init__(
        self,
        weights: EvalWeights,
        meta: dict | None = None,
        *,
        tt_mib: int = DEFAULT_TT_MIB,
        model_version: int = MODEL_VERSION,
        utility_version: int = UTILITY_VERSION,
        increment_ms: int = INCREMENT_MS,
        reserve_ms: float = 80.0,
    ) -> None:
        self.weights = weights
        self.meta = meta or {}
        self.model_version = model_version
        self.utility_version = utility_version
        self.increment_ms = increment_ms

        self.evaluator = Evaluator(weights)
        self.history = HistoryTables()
        self.tt = TranspositionTable(tt_mib)
        # clock_b: remaining-own-moves from our own 120+0.5 games (p40,
        # capped at 38) so abort is not the default path. The no-flag
        # invariant is reserve + measured unwind, not a 300-move horizon.
        self.allocator = SoftAllocator(reserve_ms=reserve_ms)
        self.scaler = SoftScaler()
        self.searcher = Searcher(
            tt=self.tt,
            history=self.history,
            eval_fn=self._eval,
            check_mask=CHECK_MASK,
        )
        # Accumulator subscription: on_make fires pre-make, on_unmake post-
        # unmake. Both must never raise (a raised callback escapes the search
        # with ancestor moves still made); faults flip _stack_ok so eval_fn
        # degrades to simple_eval for the rest of the call.
        self.searcher.on_make.append(self._on_make)
        self.searcher.on_unmake.append(self._on_unmake)

        self.state: GameState | None = None
        self._stack_ok = True
        # Non-search per-call work (parse/observe/set_root/apply/emit) is
        # subtracted from the clock before allocating, so the bounded search
        # plus this overhead stays inside the per-move share.
        self.overhead_ms = 25.0
        self.eval_faults = 0
        self.history_resets = 0
        self.fallback_events = 0
        self.moves_played = 0
        self.last_result: SearchResult | None = None
        # Book/TB membership table (spec 6): v0 ships none; the gate still
        # runs and misses cleanly. Tests inject fakes through this slot.
        self.root_table: Callable[[GameState], int | str | None] | None = None

    # -- evaluator plumbing --------------------------------------------------

    def _on_make(self, ply: int, m: int) -> None:
        if not self._stack_ok:
            return
        try:
            self.evaluator.push(self.searcher.board, m)
        except Exception as exc:  # accumulator fault: degrade, never escape
            self._stack_ok = False
            self.eval_faults += 1
            _emit(
                {
                    "event": "fallback",
                    "reason": "fallback:evaluator_fault",
                    "where": "push",
                    "error": repr(exc)[:300],
                }
            )

    def _on_unmake(self, ply: int, m: int) -> None:
        if not self._stack_ok:
            return
        try:
            self.evaluator.pop()
        except Exception as exc:
            self._stack_ok = False
            self.eval_faults += 1
            _emit(
                {
                    "event": "fallback",
                    "reason": "fallback:evaluator_fault",
                    "where": "pop",
                    "error": repr(exc)[:300],
                }
            )

    def _eval(self, board: Board) -> int:
        """RAW static eval in centipawns (side to move); the only value fn."""
        if not self._stack_ok:
            return simple_eval(board)
        try:
            return self.evaluator.evaluate(board)
        except Exception as exc:
            self._stack_ok = False
            self.eval_faults += 1
            _emit(
                {
                    "event": "fallback",
                    "reason": "fallback:evaluator_fault",
                    "where": "evaluate",
                    "error": repr(exc)[:300],
                }
            )
            return simple_eval(board)

    # -- history reconciliation ----------------------------------------------

    def _reconcile(self, fen: str, incoming: Board) -> GameState:
        """Fold the observed FEN into game history; reset conservatively."""
        if self.state is None:
            self.state = GameState(
                incoming,
                model_version=self.model_version,
                utility_version=self.utility_version,
            )
            return self.state
        st = self.state
        prev_board = st.board
        try:
            st.observe_fen(fen)
        except Exception as exc:
            self.state = GameState(
                incoming,
                model_version=self.model_version,
                utility_version=self.utility_version,
            )
            self.history_resets += 1
            _emit(
                {
                    "event": "history_discontinuity",
                    "reason": "observe_exception",
                    "error": repr(exc)[:300],
                    "history_resets": self.history_resets,
                }
            )
            return self.state
        if st.board is not prev_board:
            self.history_resets += 1
            _emit(
                {
                    "event": "history_discontinuity",
                    "reason": "unmatched_observation",
                    "history_resets": self.history_resets,
                }
            )
        return st

    # -- root eligibility gate (spec 3.4 / assets contract) -------------------

    def _root_answer(self, state: GameState) -> tuple[str | None, str]:
        """Return (uci, "hit"|"miss"|"ineligible"); never raises."""
        b = state.board
        eligible = b.fullmove <= OPENING_BOOK_MAX_FULLMOVE or (
            b._occ_all.bit_count() <= TB_MAX_PIECES
        )
        if not eligible or self.root_table is None:
            return None, "ineligible" if not eligible else "miss"
        try:
            ans = self.root_table(state)
        except Exception as exc:
            _emit(
                {
                    "event": "root_gate",
                    "result": "error",
                    "error": repr(exc)[:300],
                }
            )
            return None, "miss"
        if ans is None:
            return None, "miss"
        # Membership check against a freshly generated legal list — an asset
        # answer that is not a legal move of the current root is discarded.
        n = generate_legal(b, _MOVES)
        if isinstance(ans, str):
            for i in range(n):
                if move_to_uci(_MOVES[i]) == ans:
                    return ans, "hit"
            return None, "miss"
        try:
            cand = int(ans)
        except (TypeError, ValueError):
            return None, "miss"
        for i in range(n):
            m = _MOVES[i]
            if m == cand or (m & 0x7FFF) == (cand & 0x7FFF):
                return move_to_uci(m), "hit"
        return None, "miss"

    # -- bounded search + authoritative result --------------------------------

    def _search(self, state: GameState, time_left_ms: int) -> tuple[SearchResult | None, Any]:
        alloc = self.allocator.allocate(
            max(0, int(time_left_ms)),
            self.increment_ms,
            state.board.absolute_ply(),
            overhead_ms=self.overhead_ms,
        )
        hard_ns = now_ns() + alloc.hard_ns - self.allocator.unwind_margin_ns
        try:
            result = self.searcher.search(
                state,
                soft_ns=alloc.soft_ns,
                hard_ns=hard_ns,
                scaler=self.scaler,
            )
        except Exception as exc:
            self.fallback_events += 1
            _emit(
                {
                    "event": "fallback",
                    "reason": "fallback:search_exception",
                    "error": repr(exc)[:300],
                }
            )
            return None, alloc
        return result, alloc

    # -- the one entry point --------------------------------------------------

    def get_move(self, fen: str, time_left_ms: int) -> str:
        t0 = time.perf_counter_ns()
        # 1. Parse/validate + establish a legal fallback before expensive work.
        try:
            incoming = Board.from_fen(fen)
            if incoming._bb[WK] == 0 or incoming._bb[BK] == 0:
                raise ValueError("position without both kings")
            n = generate_legal(incoming, _MOVES)
        except Exception as exc:
            _emit(
                {
                    "event": "fallback",
                    "reason": "fallback:fen_parse",
                    "fen": str(fen)[:160],
                    "error": repr(exc)[:300],
                }
            )
            return "0000"
        if n == 0:
            # No legal reply (terminal or invalid input): nothing legal exists.
            _emit(
                {
                    "event": "fallback",
                    "reason": "fallback:no_legal_move",
                    "fen": str(fen)[:160],
                }
            )
            return "0000"
        fallback = _MOVES[0]
        fallback_uci = move_to_uci(fallback)

        root_ply = incoming.absolute_ply()
        try:
            # 2. Reconcile observed history (unknown-prefix kept on reset).
            state = self._reconcile(fen, incoming)
            # 3. Sync the incremental accumulator to the current root.
            try:
                self.evaluator.set_root(state.board)
                self._stack_ok = True
            except Exception as exc:
                self._stack_ok = False
                self.eval_faults += 1
                _emit(
                    {
                        "event": "fallback",
                        "reason": "fallback:evaluator_fault",
                        "where": "set_root",
                        "error": repr(exc)[:300],
                    }
                )

            # 4. Root-terminal agreement with the referee (spec 3.2): a root
            # the referee has already adjudicated — fifty-move, threefold,
            # ply cap, insufficient material — is never searched as a live
            # position. Any legal move is a correct reply to a decided game.
            term = referee_terminal(state)
            result: SearchResult | None = None
            alloc = None
            self.last_result = None
            if term is not None:
                gate = f"terminal:{term[1]}"
                _emit(
                    {
                        "event": "root_terminal",
                        "result": term[0],
                        "termination": term[1],
                    }
                )
                uci = fallback_uci
            else:
                # 5. Root eligibility gate -> stored answer or search.
                answer, gate = self._root_answer(state)
                if answer is not None:
                    uci = answer
                else:
                    # 6-8. Bounded ID-PVS + qsearch over TT/histories/evaluator.
                    snap = snapshot(state.board)
                    result, alloc = self._search(state, time_left_ms)
                    if snapshot(state.board) != snap:
                        # A search escape left the root board desynced: rebuild
                        # from the observed FEN (history resets surface next call).
                        self.fallback_events += 1
                        _emit({"event": "fallback", "reason": "fallback:board_desync"})
                        self.state = GameState(
                            incoming,
                            model_version=self.model_version,
                            utility_version=self.utility_version,
                        )
                        state = self.state
                        result = None
                    self.last_result = result
                    if result is not None and result.fallback_used:
                        self.fallback_events += 1
                        _emit(
                            {
                                "event": "fallback",
                                "reason": "fallback:search_no_result",
                            }
                        )
                    # 9. Committed-move selection: the played move is the
                    # last COMPLETED iteration's PV head. Post-W03-repair,
                    # result.move is always that committed move — this guard
                    # is a second belt, kept in case result.move and the PV
                    # ever diverge again.
                    move = 0
                    if result is not None:
                        move = result.pv[0] if result.pv else result.move
                        if result.partial and result.pv and result.move != result.pv[0]:
                            _emit(
                                {
                                    "event": "pv_guard",
                                    "discarded": move_to_uci(result.move),
                                    "played": move_to_uci(result.pv[0]),
                                    "depth": result.depth,
                                }
                            )
                    # 10. Legality recheck against a freshly generated list.
                    n2 = generate_legal(state.board, _MOVES)
                    legal = any(_MOVES[i] == move for i in range(n2))
                    if not legal:
                        if move:
                            self.fallback_events += 1
                            _emit(
                                {
                                    "event": "fallback",
                                    "reason": "fallback:illegal_result",
                                }
                            )
                        move = fallback
                    uci = move_to_uci(move)
        except Exception as exc:
            self.fallback_events += 1
            _emit(
                {
                    "event": "fallback",
                    "reason": "fallback:agent_exception",
                    "error": repr(exc)[:300],
                }
            )
            # Best-effort: record the fallback we are about to return so the
            # next call can still reconcile the opponent's reply.
            try:
                if self.state is not None:
                    self.state.apply_own_uci(fallback_uci)
            except Exception:
                pass
            return fallback_uci

        # Record our own move so the next call can infer the opponent reply.
        try:
            state.apply_own_uci(uci)
        except Exception as exc:
            self.history_resets += 1
            _emit(
                {
                    "event": "history_discontinuity",
                    "reason": "own_move_apply_failed",
                    "uci": uci,
                    "error": repr(exc)[:300],
                    "history_resets": self.history_resets,
                }
            )
        self.moves_played += 1

        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        diag: dict[str, Any] = {
            "event": "move",
            "ply": root_ply,
            "fen": str(fen)[:120],
            "time_left_ms": int(time_left_ms),
            "uci": uci,
            "book": gate,
            "history_known": not state.unknown_prefix,
            "history_resets": self.history_resets,
            "eval_faults": self.eval_faults,
            "fallback_events": self.fallback_events,
            "elapsed_ms": round(elapsed_ms, 2),
        }
        if alloc is not None:
            diag["alloc"] = {
                "soft_ms": round(alloc.soft_ns / 1e6, 1),
                "hard_ms": round(alloc.hard_ns / 1e6, 1),
                "est_moves": alloc.estimated_moves,
            }
        if result is not None:
            diag.update(
                {
                    "depth": result.depth,
                    "seldepth": result.seldepth,
                    "score": result.score,
                    "nodes": result.nodes,
                    "qnodes": result.qnodes,
                    "search_ms": round(result.elapsed_ms, 1),
                    "partial": result.partial,
                    "aborted": result.aborted,
                    "fallback_used": result.fallback_used,
                    "pv": [move_to_uci(m) for m in result.pv[:12]],
                }
            )
        _emit(diag)
        return uci


# ---------------------------------------------------------------------------
# Process-level singleton + init/warm (the platform's 90 s import budget)
# ---------------------------------------------------------------------------

_AGENT: Agent | None = None


def init(
    model: str | Path | bytes | None = None,
    *,
    tt_mib: int = DEFAULT_TT_MIB,
    weights_seed: int = DEFAULT_SEED,
    warm: bool = True,
    emit: bool = True,
) -> Agent:
    """Build the wired agent. Idempotent; returns the process singleton.

    ``model`` may be a path to ``weights.npy`` (uint8 npy holding the RXF1
    container), raw RXF1 bytes, or ``None`` to generate the deterministic
    bounded-random qualification weights via ``EvalWeights.random``.
    """
    global _AGENT
    t0 = time.monotonic()
    if model is None:
        weights = EvalWeights.random(weights_seed)
        meta: dict = {
            "weights": "random",
            "seed": weights_seed,
            "scale_num": int(weights.scale_num),
            "scale_shift": int(weights.scale_shift),
        }
    elif isinstance(model, (bytes, bytearray)):
        weights, meta = _weights_from_blob(bytes(model))
    else:
        weights, meta = load_weights(model)
    _AGENT = Agent(weights, meta, tt_mib=tt_mib)
    init_s = time.monotonic() - t0
    warm_s = 0.0
    if warm:
        t1 = time.monotonic()
        try:
            _warm(_AGENT)
        except Exception as exc:
            # Warm is best-effort: an un-warmed kernel compiles lazily inside
            # the first move, slow but never illegal.
            _emit(
                {
                    "event": "warm_incomplete",
                    "error": repr(exc)[:300],
                }
            )
        warm_s = time.monotonic() - t1
    if emit:
        _emit(
            {
                "event": "init",
                "init_seconds": round(init_s + warm_s, 3),
                "load_seconds": round(init_s, 3),
                "warm_seconds": round(warm_s, 3),
                "tt_mib": tt_mib,
                "model": {k: meta.get(k) for k in ("weights", "seed", "name") if k in meta},
            }
        )
    return _AGENT


def _weights_from_blob(blob: bytes) -> tuple[EvalWeights, dict]:
    model = read_model_frugal(blob)
    return weights_from_model_dict(model), model.get("_meta", {})


def get_move(fen: str, time_left_ms: int) -> str:
    """``agent.py``-compatible entry: FEN + remaining clock -> UCI."""
    ag = _AGENT
    if ag is None:
        ag = init()
    try:
        return ag.get_move(fen, time_left_ms)
    except Exception as exc:
        # Last-resort legality: never let the call raise out of get_move.
        _emit(
            {
                "event": "fallback",
                "reason": "fallback:agent_crash",
                "error": repr(exc)[:300],
            }
        )
        try:
            incoming = Board.from_fen(fen)
            n = generate_legal(incoming, _MOVES)
            if n:
                return move_to_uci(_MOVES[0])
        except Exception:
            pass
        return "0000"


def configure_for_development(increment_ms: int = 0, reserve_ms: int = 10) -> None:
    """Arena development-clock hook: adjust increment/reserve assumptions."""
    ag = _AGENT or init()
    ag.increment_ms = int(increment_ms)
    ag.allocator.reserve_ms = float(reserve_ms)


def reset_for_tests() -> None:
    """Drop the singleton so a test can re-init with a different model."""
    global _AGENT
    _AGENT = None


# ---------------------------------------------------------------------------
# Kernel warm: compile every reachable specialization inside the init budget
# ---------------------------------------------------------------------------

# (fen, [uci moves]) — covers every move flag, a mirror-crossing king move
# (king-bucket refresh path) and ordinary incremental replay.
_WARM_SCRIPT: tuple[tuple[str, list[str]], ...] = (
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ["e2e4", "d7d5", "e4d5", "d8d5", "g1f3", "b8c6"],
    ),
    (
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        ["e1g1", "e8c8"],
    ),
    (
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        ["e5d6"],
    ),
    (
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        ["a7a8q"],
    ),
    (
        "4k3/8/8/8/8/8/p7/1K6 b - - 0 1",
        ["a2a1n"],
    ),
    (
        "r3k3/1P6/8/8/8/8/8/4K3 w q - 0 1",
        ["b7a8q"],
    ),
    (
        "4k3/8/8/8/8/8/3K4/8 w - - 0 1",
        ["d2e2"],
    ),
    (
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        ["f3g5", "e8e7", "g5f7"],
    ),
)


def _match_move(board: Board, uci: str) -> int:
    n = generate_legal(board, _MOVES)
    for i in range(n):
        if move_to_uci(_MOVES[i]) == uci:
            return _MOVES[i]
    raise ValueError(f"warm script move {uci} not legal in {board.to_fen()}")


def _warm(ag: Agent) -> None:
    """Exercise every flag/kernel path so JIT cost lands in the init budget."""
    ev = ag.evaluator
    for fen, ucis in _WARM_SCRIPT:
        board = Board.from_fen(fen)
        ev.set_root(board)
        made = 0
        try:
            for uci in ucis:
                m = _match_move(board, uci)
                ev.push(board, m)
                board.make(m)
                ev.evaluate(board)
                made += 1
        finally:
            for _ in range(made):
                board.unmake()
                ev.pop()
        ev.evaluate(board)
    # WDL auxiliary path is reachable; compile it too.
    board = Board.from_fen(_WARM_SCRIPT[0][0])
    ev.set_root(board)
    ev.evaluate_wdl(board)
    # One real bounded search: end-to-end path incl. TT/histories/qsearch.
    state = GameState.from_fen(
        _WARM_SCRIPT[0][0],
        model_version=ag.model_version,
        utility_version=ag.utility_version,
    )
    ev.set_root(state.board)
    ag._stack_ok = True
    ag.searcher.search(state, max_depth=3)

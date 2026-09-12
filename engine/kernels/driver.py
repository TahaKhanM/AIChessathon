"""Python-facing driver for the compiled search kernels.

``CompiledSearcher`` mirrors ``engine.search.Searcher`` — same ``search()``
control flow (aspiration, abort-safe root semantics, IterationScaler soft
budgeting), but the recursive interior is the nopython kernel
``engine.kernels.search_nb.n_search``.  The kernel and the oracle share the
transposition cluster array, so ``hashfull``/``audit_entries`` keep working.

Everything here is *outside* the recursive loop — Python is allowed at this
level (the spec only forbids Python object allocation inside the recursive
search).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from engine.board import Board, move_to_uci
from engine.clock import IterationScaler, TimeAllocator
from engine.movegen import generate_legal
from engine.search import (
    DRAW,
    INF,
    MAX_DEPTH,
    MATE,
    MATE_IN_MAX,
    SEARCH_PATH,
    SearchResult,
)
from engine.state import GameState, referee_terminal
from engine.tt import TranspositionTable

from engine.kernels import layout as L
from engine.kernels import search_nb as SN


class CompiledSearcher:
    """Drop-in compiled counterpart of ``engine.search.Searcher``.

    Args mirror the oracle: ``tt`` (shared clusters), ``params`` overrides,
    ``eval_kind`` selects the eval path (0 = simple_eval stand-in,
    1 = F512-EF accumulator; requires ``weights``), ``check_mask`` is the
    deadline poll mask (compiled nodes are ~0.2us so the default is the
    large clock mask, matching the search comment).
    """

    def __init__(
        self,
        tt: TranspositionTable | None = None,
        history=None,
        eval_fn=None,  # kept for signature parity; unused
        params: dict[str, int] | None = None,
        trace: bool = False,
        now: Callable[[], int] | None = None,
        check_mask: int = 2047,
        weights=None,
        eval_kind: int = 0,
    ) -> None:
        self.tt = tt if tt is not None else TranspositionTable()
        self.now = now or time.monotonic_ns
        self._check_mask = check_mask
        hist_arrays = None
        if history is not None:
            hist_arrays = (
                history.quiet,
                history.capture,
                history.cont,
                history.counter,
                history.pawn,
                history.threat,
                history.corr,
            )
        self.ctx = L.build_ctx(
            tt_clusters=self.tt.clusters,
            hist_arrays=hist_arrays,
            params=params,
            weights=weights,
            eval_kind=eval_kind,
            log_rows=0,
        )
        self._v = L.views(self.ctx)
        self.st = self._v["st"]
        # The kernel owns the history regions — expose shaped views so tests
        # and the abort gate can inspect/mirror the oracle's HistoryTables.
        self.hist = None
        self.state: GameState | None = None
        self.board: Board = Board()
        self._pending_pv: list[int] = []

    # -- state sync --------------------------------------------------------

    def _sync_state(self, state: GameState) -> None:
        """Copy game position + known history into the arenas."""
        self.state = state
        self.board = state.board
        L.load_board(self.ctx, state.board)
        st = self.st
        n = state._game_n
        au = self.ctx[L.AU]
        a8 = self.ctx[L.A8]
        au[L.X_HISTKEY : L.X_HISTKEY + n] = np.asarray(state._game_keys[:n], np.uint64)
        a8[L.X_HISTIRR : L.X_HISTIRR + max(0, n - 1)] = np.asarray(
            state._game_irr[: max(0, n - 1)], np.int8
        )
        st[L.I_HISTN] = n
        st[L.I_UNKNOWN] = 1 if state.unknown_prefix else 0
        st[L.I_MODELVER] = state.model_version
        st[L.I_UTILVER] = state.utility_version

    # -- info accessors (Info-compatible) -----------------------------------

    @property
    def nodes(self) -> int:
        return int(self.st[L.I_NODES])

    @property
    def qnodes(self) -> int:
        return int(self.st[L.I_QNODES])

    @property
    def seldepth(self) -> int:
        return int(self.st[L.I_SELDEPTH])

    @property
    def root_best(self) -> int:
        return int(self.st[L.I_ROOTBEST])

    @property
    def root_score(self) -> int:
        return int(self.st[L.I_ROOTSCORE])

    # -- the search driver — mirror of Searcher.search ----------------------

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
        started = self.now()
        result = SearchResult()
        board = state.board
        buf = [0] * 256
        n_root = generate_legal(board, buf)
        if n_root == 0:
            result.aborted = True
            return result
        fallback = buf[0]
        if root_hint:
            for i in range(n_root):
                if (buf[i] & 0x7FFF) == (root_hint & 0x7FFF):
                    fallback = buf[i]
                    break
        result.move = fallback

        self._sync_state(state)
        # A root already terminal under the referee's ordering is decided
        # once, here — no iteration runs on it. The kernel's n_ab still
        # skips draw adjudication at ply 0 (W08 parity gap; the kernel is
        # not on the shipped import closure), so the driver check is what
        # keeps the compiled path's root honest.
        term = referee_terminal(state)
        if term is not None:
            result.score = DRAW if term[0] == "draw" else -MATE
            result.nodes = int(self.st[L.I_NODES])
            result.elapsed_ms = (self.now() - started) / 1e6
            return result
        # The kernel clock's epoch differs from time.monotonic_ns on some
        # platforms, so pass the remaining *budget* (ns) — k_begin anchors
        # it against its own clock_ns() stamp.
        hard_budget = max(0, int(hard_ns) - self.now()) if hard_ns else 0
        SN.k_begin(self.ctx, hard_budget, int(node_limit), self._check_mask)
        st = self.st
        st[L.I_ROOTHINT] = int(root_hint)
        a32 = self.ctx[L.A32]
        v = self._v
        scaler = scaler or IterationScaler()

        p_asp_delta = int(v["params"][L.P_asp_delta])
        p_asp_min = int(v["params"][L.P_asp_min_depth])
        p_asp_wpct = int(v["params"][L.P_asp_widen_pct])
        p_asp_wadd = int(v["params"][L.P_asp_widen_add])
        p_asp_blend = int(v["params"][L.P_asp_fail_low_blend])
        p_mate_break = int(v["params"][L.P_mate_break_depth])
        stable = 0
        last_best = 0
        prev_score = 0
        iter_started = started
        ss_eval = v["ss_eval"]
        ss_null = v["ss_null"]
        ss_excl = v["ss_excl"]
        rootsc = v["rootsc"]

        for depth in range(1, max_depth + 1):
            st[L.I_ROOTSCORE] = -INF
            st[L.I_ROOTBEST] = 0  # torn candidates may never leak out
            st[L.I_BESTNODES] = 0
            st[L.I_ROOTDEPTH] = depth
            rootsc[:n_root] = L.ROOT_SCORE_SENT
            nodes_at_iter_start = st[L.I_NODES]
            ss_eval[: SEARCH_PATH + 4] = -INF
            ss_null[: SEARCH_PATH + 4] = 0
            ss_excl[: SEARCH_PATH + 4] = 0
            delta = p_asp_delta
            if depth >= p_asp_min:
                alpha = max(prev_score - delta, -INF)
                beta = min(prev_score + delta, INF)
            else:
                alpha, beta = -INF, INF
            score = 0
            while True:
                score = SN.n_search(self.ctx, depth, alpha, beta, 0, 1, 0)
                if st[L.I_DL_STOP]:
                    break
                if score <= alpha:
                    if p_asp_blend:
                        beta = (alpha + beta) // 2
                    alpha = max(score - delta, -INF)
                    delta += delta * p_asp_wpct // 100 + p_asp_wadd
                elif score >= beta:
                    beta = min(score + delta, INF)
                    delta += delta * p_asp_wpct // 100 + p_asp_wadd
                else:
                    break
            now = self.now()
            if st[L.I_DL_STOP]:
                # An aborted iteration never overwrites a completed PV:
                # ``result`` keeps the last completed iteration's committed
                # (move, score, depth, pv) — or the pre-search legal fallback
                # when no iteration has completed. Same abort-safe commit
                # rule as engine.search.Searcher.search.
                result.aborted = True
                result.partial = True
                break
            result.move = int(st[L.I_ROOTBEST])
            result.score = int(score)
            result.depth = depth
            result.seldepth = int(st[L.I_SELDEPTH])
            n_pv = SN.n_extract_pv(self.ctx, depth)
            result.pv = [int(a32[L.X_PVBUF + i]) for i in range(n_pv)]
            iter_nodes = st[L.I_NODES] - nodes_at_iter_start
            result.iterations.append(
                (depth, int(score), int(st[L.I_NODES]), (now - started) // 1_000_000)
            )
            drop = prev_score - score if depth > 1 else 0
            prev_score = score
            if abs(score) >= MATE_IN_MAX and depth >= p_mate_break:
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
                    st[L.I_BESTNODES] / iter_nodes
                    if depth >= int(v["params"][L.P_effort_min_depth]) and iter_nodes > 0
                    else None
                )
                if frac is not None:
                    result.best_effort = float(frac)
                gap = self._contender_gap(n_root)
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
                    and depth >= int(v["params"][L.P_next_iter_min_depth])
                    and now + iter_ns * int(v["params"][L.P_next_iter_cost_pct]) // 100 >= hard_ns
                ):
                    break
        result.nodes = int(st[L.I_NODES])
        result.qnodes = int(st[L.I_QNODES])
        result.elapsed_ms = (self.now() - started) / 1e6
        if not result.move:
            result.move = fallback
            result.fallback_used = True
        return result

    def _contender_gap(self, n_root: int) -> int:
        """Score separation between the two best root moves (mirror)."""
        rootsc = self._v["rootsc"][:n_root]
        vals = sorted((int(x) for x in rootsc if x != L.ROOT_SCORE_SENT), reverse=True)
        if len(vals) < 2:
            return -1
        return vals[0] - vals[1]

    # -- convenience one-shot entry — mirror of choose_move ------------------

    def choose_move(
        self,
        state: GameState,
        time_left_ms: int,
        increment_ms: int = 500,
        allocator: TimeAllocator | None = None,
        root_answer=None,
        max_depth: int = MAX_DEPTH,
    ) -> str:
        board = state.board
        buf = [0] * 256
        n = generate_legal(board, buf)
        if n == 0:
            return "0000"
        fallback = buf[0]
        if root_answer is not None:
            try:
                ans = root_answer(state)
            except Exception:
                ans = None
            if ans is not None:
                cand = self._resolve_answer(ans, buf, n)
                if cand:
                    return move_to_uci(cand)
        allocator = allocator or TimeAllocator()
        alloc = allocator.allocate(time_left_ms, increment_ms, board.absolute_ply())
        res = self.search(
            state,
            soft_ns=alloc.soft_ns,
            hard_ns=self.now() + alloc.hard_ns - allocator.unwind_margin_ns,
            max_depth=max_depth,
        )
        move = res.move or fallback
        # Legality recheck (spec 3.1) — mirror of the oracle.
        buf2 = [0] * 256
        n2 = generate_legal(board, buf2)
        legal = False
        for i in range(n2):
            if buf2[i] == move:
                legal = True
                break
        if not legal:
            move = buf2[0] if n2 else fallback
        return move_to_uci(move)

    def _resolve_answer(self, ans, buf, n) -> int:
        """Resolve a stored move int or UCI string to a legal move."""
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

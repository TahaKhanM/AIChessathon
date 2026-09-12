"""FEN counters, observed history, keys, terminal handling.

Maintains the three distinct identities from spec section 3.3:
geometric/legal position identity, value context (halfmove counter,
remaining absolute horizon, model/utility version) and repetition context
(known reversible history, unknown-prefix status, hypothetical search path).
"""

from __future__ import annotations

from typing import NamedTuple

from engine.board import (
    BISHOP,
    BLACK,
    FLAG_EP,
    KING,
    KNIGHT,
    MASK64,
    PAWN,
    QUEEN,
    ROOK,
    WHITE,
    Board,
    decode_move,
    move_to_uci,
)
from engine.movegen import generate_legal, has_legal_ep, in_check

PLY_CAP = 600
MAX_GAME = 2048
_DARK = 0xAA55AA55AA55AA55
_LIGHT = MASK64 ^ _DARK
_MOVES = [0] * 256


class ValueContext(NamedTuple):
    halfmove: int
    remaining_horizon: int
    model_version: int
    utility_version: int


class RepetitionContext(NamedTuple):
    known_reversible_keys: tuple[int, ...]
    unknown_prefix: bool
    history_complete: bool
    search_path_len: int


class GameState:
    """Observed game plus hypothetical search path, with separated keys."""

    __slots__ = (
        "board",
        "model_version",
        "utility_version",
        "unknown_prefix",
        "_game_keys",
        "_game_irr",
        "_game_n",
        "_search_keys",
        "_search_irr",
        "_search_n",
    )

    def __init__(
        self,
        board: Board,
        *,
        model_version: int = 0,
        utility_version: int = 0,
        unknown_prefix: bool = True,
    ) -> None:
        self.board = board
        self.model_version = model_version
        self.utility_version = utility_version
        self.unknown_prefix = unknown_prefix
        self._game_keys = [0] * MAX_GAME
        self._game_irr = [False] * MAX_GAME
        self._game_keys[0] = int(board.key)
        self._game_n = 1
        self._search_keys = [0] * MAX_GAME
        self._search_irr = [False] * MAX_GAME
        self._search_n = 0

    @classmethod
    def from_fen(
        cls,
        fen: str,
        model_version: int = 0,
        utility_version: int = 0,
    ) -> GameState:
        return cls(
            Board.from_fen(fen),
            model_version=model_version,
            utility_version=utility_version,
            unknown_prefix=True,
        )

    def geometric_identity(self) -> int:
        return int(self.board.key)

    def value_context(self) -> ValueContext:
        return ValueContext(
            halfmove=self.board.halfmove,
            remaining_horizon=PLY_CAP - self.board.absolute_ply(),
            model_version=self.model_version,
            utility_version=self.utility_version,
        )

    def repetition_context(self) -> RepetitionContext:
        return RepetitionContext(
            known_reversible_keys=tuple(self._game_keys[: self._game_n]),
            unknown_prefix=self.unknown_prefix,
            history_complete=not self.unknown_prefix,
            search_path_len=self._search_n,
        )

    def apply_own_uci(self, uci: str) -> None:
        move = _match_uci(self.board, uci)
        self._push_game(move)

    def observe_fen(self, fen: str) -> None:
        incoming = Board.from_fen(fen)
        if _same_observed(self.board, incoming):
            return
        n = generate_legal(self.board, _MOVES)
        matched = -1
        hits = 0
        for i in range(n):
            move = int(_MOVES[i])
            self.board.make(move)
            ok = _same_observed(self.board, incoming)
            self.board.unmake()
            if ok:
                matched = move
                hits += 1
        if hits == 1:
            self._push_game(matched)
            return
        self.board = incoming
        self.unknown_prefix = True
        self._game_keys[0] = int(incoming.key)
        self._game_n = 1
        self._search_n = 0

    def make_null(self) -> None:
        """Push a null move onto the search path.

        The post-null position is recorded like a real path move and marked
        irreversible — a null is a repetition boundary the scan may never
        look through — so ``pop_search`` unwinds it exactly like a real move
        and ``_search_n`` can never desynchronize from the board undo stack.
        """
        self.board.make_null()
        i = self._search_n
        self._search_keys[i] = int(self.board.key)
        self._search_irr[i] = True
        self._search_n = i + 1

    def unmake_null(self) -> None:
        """Pop the search-path null — the same unwind as ``pop_search``."""
        self.pop_search()

    def push_search_uci(self, uci: str) -> None:
        move = _match_uci(self.board, uci)
        irr = _irreversible(self.board, move)
        self.board.make(move)
        i = self._search_n
        self._search_keys[i] = int(self.board.key)
        self._search_irr[i] = irr
        self._search_n = i + 1

    def pop_search(self) -> None:
        """Pop the most recent search-path move (real or null).

        Every push must come from ``push_search_uci``/``make_null`` so the
        board undo stack and ``_search_n`` stay in lock-step — popping an
        empty stack is a desync bug and raises instead of silently drifting.
        """
        if self._search_n <= 0:
            raise RuntimeError("pop_search on an empty search stack")
        self.board.unmake()
        self._search_n -= 1

    def repetition_count(self) -> int:
        keys = self._game_keys[: self._game_n] + self._search_keys[: self._search_n]
        irrs = self._game_irr[: self._game_n - 1] + self._search_irr[: self._search_n]
        current = int(self.board.key)
        count = 1
        i = len(keys) - 1
        while i > 0:
            if irrs[i - 1]:
                break
            i -= 1
            if keys[i] == current:
                count += 1
        return count

    def _push_game(self, move: int) -> None:
        irr = _irreversible(self.board, move)
        self.board.make(move)
        i = self._game_n
        self._game_irr[i - 1] = irr
        self._game_keys[i] = int(self.board.key)
        self._game_n = i + 1


def _match_uci(board: Board, uci: str) -> int:
    n = generate_legal(board, _MOVES)
    for i in range(n):
        move = int(_MOVES[i])
        if move_to_uci(move) == uci:
            return move
    raise ValueError(f"not legal: {uci} in {board.to_fen()}")


def _same_observed(a: Board, b: Board) -> bool:
    return (
        int(a.key) == int(b.key)
        and a.halfmove == b.halfmove
        and a.fullmove == b.fullmove
        and a.side == b.side
        and a.castling == b.castling
    )


def _irreversible(board: Board, move: int) -> bool:
    if has_legal_ep(board):
        return True
    _frm, to, promo, flag, piece, captured = decode_move(move)
    if piece % 6 == PAWN or captured != 15 or flag == FLAG_EP or promo:
        return True
    from engine.board import CASTLE_CLEAR

    if board.castling != (board.castling & CASTLE_CLEAR[_frm] & CASTLE_CLEAR[to]):
        return True
    return False


def has_insufficient_material(board: Board, color: int) -> bool:
    occ_c = board._occ[color]
    pawns = board._bb[color * 6 + PAWN]
    knights = board._bb[color * 6 + KNIGHT]
    bishops = board._bb[color * 6 + BISHOP]
    rooks = board._bb[color * 6 + ROOK]
    queens = board._bb[color * 6 + QUEEN]
    if occ_c & (pawns | rooks | queens):
        return False
    them = color ^ 1
    kings = board._bb[WHITE * 6 + KING] | board._bb[BLACK * 6 + KING]
    all_queens = board._bb[WHITE * 6 + QUEEN] | board._bb[BLACK * 6 + QUEEN]
    if occ_c & knights:
        return occ_c.bit_count() <= 2 and not (board._occ[them] & ~kings & ~all_queens)
    if occ_c & bishops:
        all_bishops = board._bb[WHITE * 6 + BISHOP] | board._bb[BLACK * 6 + BISHOP]
        all_pawns = board._bb[WHITE * 6 + PAWN] | board._bb[BLACK * 6 + PAWN]
        all_knights = board._bb[WHITE * 6 + KNIGHT] | board._bb[BLACK * 6 + KNIGHT]
        same_color = (not (all_bishops & _DARK)) or (not (all_bishops & _LIGHT))
        return bool(same_color and not all_pawns and not all_knights)
    return True


def _insufficient(board: Board) -> bool:
    return has_insufficient_material(board, WHITE) and has_insufficient_material(board, BLACK)


def _game_repetition_count(state: GameState) -> int:
    current = int(state.board.key)
    count = 1
    for i in range(state._game_n - 2, -1, -1):
        if state._game_irr[i]:
            break
        if state._game_keys[i] == current:
            count += 1
    return count


def referee_terminal(state: GameState) -> tuple[str, str] | None:
    """Match official referee ordering: outcome, threefold, fifty, ply cap."""
    board = state.board
    n = generate_legal(board, _MOVES)
    checked = in_check(board)
    if n == 0 and checked:
        winner = "black" if board.side == WHITE else "white"
        return winner, "checkmate"
    if _insufficient(board):
        return "draw", "insufficient_material"
    if n == 0:
        return "draw", "stalemate"
    if board.halfmove >= 150:
        return "draw", "seventyfive_moves"
    reps = _game_repetition_count(state)
    if reps >= 5:
        return "draw", "fivefold_repetition"
    if reps >= 3:
        return "draw", "threefold_repetition"
    if board.halfmove >= 100:
        return "draw", "fifty_moves"
    if board.absolute_ply() >= PLY_CAP:
        return "draw", "ply_cap"
    return None

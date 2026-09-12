"""Single-game adapter using the material/position reference evaluator.

One module instance serves one game, serially. The neural runtime is exposed
separately through engine.agent_rx with an explicit model artifact.
"""

import chess

from engine.board import Board, move_to_uci
from engine.movegen import generate_legal
from engine.search import Searcher, simple_eval
from engine.state import GameState
from engine.tt import TranspositionTable

_searcher = Searcher(tt=TranspositionTable(8), eval_fn=simple_eval, check_mask=7)
_state: GameState | None = None


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal UCI move, or ``0000`` when no legal move exists.

    Invalid FEN input raises ValueError. The caller supplies a fresh process
    for each game; concurrent calls to a shared instance are unsupported.
    """
    global _state
    validated = chess.Board(fen)
    if not validated.is_valid():
        raise ValueError("FEN does not describe a valid chess position")
    fen = validated.fen(en_passant="fen")
    incoming = Board.from_fen(fen)
    moves = [0] * 256
    count = generate_legal(incoming, moves)
    if not count:
        return "0000"
    fallback = moves[0]
    if _state is None:
        _state = GameState.from_fen(fen)
    else:
        try:
            _state.observe_fen(fen)
        except ValueError:
            _state = GameState.from_fen(fen)
    uci = _searcher.choose_move(_state, max(0, int(time_left_ms)))
    if uci not in {move_to_uci(m) for m in moves[:count]}:
        uci = move_to_uci(fallback)
    _state.apply_own_uci(uci)
    return uci

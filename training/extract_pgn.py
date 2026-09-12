"""PGN source adapter: real games -> typed rx-final-1 records.

Each visited position becomes one record carrying a ``game_outcome``
observation (u, one-hot WDL and the PGN termination preserved as
provenance), plus parent/child trajectory links.  No searched/action labels
are fabricated — a PGN supplies outcome supervision only.

The adapter is the reference for W06 source adapters (binpack relabels,
teacher searches): those must emit the same record contract with their own
``decoder_revision``, teacher manifest ids and provenance.
"""

from __future__ import annotations

import hashlib
import io
import os

import chess
import chess.pgn

from training.features import canonical_board_key
from training.records import (
    make_observation,
    make_position,
    make_record,
    make_source,
)

DECODER_REVISION = "training.extract_pgn@1/python-chess-1.11.2"

_OUTCOME_U = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
_OUTCOME_WDL = {"1-0": [1.0, 0.0, 0.0], "0-1": [0.0, 0.0, 1.0], "1/2-1/2": [0.0, 1.0, 0.0]}


def _game_id(game: chess.pgn.Game, fallback: str) -> str:
    h = game.headers
    base = (
        f"{h.get('Event', '?')}|{h.get('Date', '?')}|{h.get('Round', '?')}"
        f"|{h.get('White', '?')}-vs-{h.get('Black', '?')}"
    )
    if base.count("?") == 5:
        return fallback
    return base


def extract_pgn_bytes(
    data: bytes,
    *,
    corpus: str,
    object_sha256: str | None = None,
    max_games: int | None = None,
    max_plies: int | None = None,
) -> list[dict]:
    """Parse all games in a PGN byte string into records."""
    if object_sha256 is None:
        object_sha256 = hashlib.sha256(data).hexdigest()
    records: list[dict] = []
    stream = io.StringIO(data.decode("utf-8", errors="replace"))
    n_games = 0
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        n_games += 1
        if max_games is not None and n_games > max_games:
            break
        gid = _game_id(game, f"{corpus}:game-{n_games}")
        # Variant discipline (FIX-DATA-2): every declared variant is either
        # honoured (parsed under its own rules) or rejected loudly — never
        # silently parsed as orthodox.  python-chess itself raises on
        # unknown variants (e.g. DFRC) inside read_game/board(); anything
        # else non-standard is rejected here.
        vraw = game.headers.get("Variant", "Standard")
        vlow = vraw.lower()
        if vlow in ("standard", "normal", "chess", "", "from position"):
            variant = "standard"
        elif "960" in vlow or vlow in ("fischerandom", "frc", "freestyle"):
            variant = "chess960"
        else:
            raise ValueError(
                f"{gid}: Variant={vraw!r} is unsupported — refusing to "
                "parse a non-standard game as orthodox"
            )
        try:
            start = game.board()
        except ValueError as exc:
            raise ValueError(f"{gid}: cannot build start board (Variant={vraw!r}): {exc}") from exc
        # Unlabelled DFRC/FRC can slip in via a SetUp FEN: a colour that
        # HOLDS castling rights while its king is off the home square
        # cannot exist under orthodox rules.
        if variant == "standard" and start.castling_rights:
            cr = start.castling_rights
            # a colour's rights bits live on its back rank (rook squares —
            # a1/h1 for KQ, file letters for DFRC-style FENs)
            for col, home, mask in (
                (chess.WHITE, chess.E1, chess.BB_RANK_1),
                (chess.BLACK, chess.E8, chess.BB_RANK_8),
            ):
                ksq = start.king(col)
                if cr & mask and ksq is not None and ksq != home:
                    raise ValueError(
                        f"{gid}: castling rights with a non-home king — the "
                        "start FEN is not orthodox chess (DFRC/FRC?), "
                        "refusing to parse it as standard"
                    )
        start_fen = start.fen()
        result = game.headers.get("Result", "*")
        termination = game.headers.get("Termination")
        lineage = f"{corpus}:start:{hashlib.sha256(start_fen.encode()).hexdigest()[:12]}"

        board = game.board()
        prev_record_id: str | None = None
        ply = 0
        positions: list[tuple[chess.Board, str | None]] = [(board.copy(stack=False), None)]
        for move in game.mainline_moves():
            board.push(move)
            positions.append((board.copy(stack=False), move.uci()))
        for ply, (b, played_uci) in enumerate(positions):
            if max_plies is not None and ply > max_plies:
                break
            rec_id = f"{gid}#ply{ply}"
            abs_ply = (b.fullmove_number - 1) * 2 + (0 if b.turn == chess.WHITE else 1)
            obs = make_observation(
                kind="game_outcome",
                perspective="white",
                score_kind="outcome",
                bound_kind="none",
                context_known=True,
                expected_score=_OUTCOME_U.get(result),
                wdl=_OUTCOME_WDL.get(result),
                termination=termination,
                interrupted=result == "*",
                right_censored=result == "*",
                parent_record_id=prev_record_id,
                action_uci=played_uci,
            )
            rec = make_record(
                record_id=rec_id,
                position=make_position(
                    fen4=canonical_board_key(b),
                    variant=variant,
                    halfmove_clock=b.halfmove_clock,
                    fullmove_number=b.fullmove_number,
                    absolute_ply=abs_ply,
                    history_complete=True,
                    unknown_prefix=False,
                    known_fields=["halfmove_clock", "fullmove_number", "history"],
                ),
                source=make_source(
                    corpus=corpus,
                    object_sha256=object_sha256,
                    decoder_revision=DECODER_REVISION,
                    lineage_family=lineage,
                    original_position_id=rec_id,
                    original_game_id=gid,
                    derivation_chain=[{"op": "pgn_start", "fen": start_fen}],
                    split_group=gid,
                ),
                observations=[obs],
                legal_actions_enumerated=False,
            )
            records.append(rec)
            prev_record_id = rec_id
    return records


def extract_pgn_file(path: str, *, corpus: str | None = None, **kw) -> list[dict]:
    with open(path, "rb") as fh:
        data = fh.read()
    return extract_pgn_bytes(
        data,
        corpus=corpus or os.path.splitext(os.path.basename(path))[0],
        **kw,
    )

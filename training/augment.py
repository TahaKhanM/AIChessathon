"""Label-preserving augmentation (spec section 9).

Implemented transforms:

* ``colour_swap`` — vertical mirror + colour exchange (``board.mirror()``).
  Always an exact chess symmetry: move graph, pawn direction, EP legality,
  castling geometry and counters are preserved.  Stored values are
  invariant: white/black perspectives are renamed, u/WDL/cp/mate are not
  flipped (flipping name and value is the classic double-flip).
* ``mirror_files`` — horizontal file mirror.  NOT applied when orthodox
  castling rights are live (a mirrored king on the d-file is not a legal
  standard-chess castling state, so the transform is not automatically
  label-preserving) or when an EP square is set, unless the caller opts in
  with ``force=True`` for variant data.  Callers get ``None`` back.

Augmented records are *derived* records: they carry a derivation_chain entry
and keep the source ``split_group``/cluster membership so they can never
leak across splits.
"""

from __future__ import annotations

import hashlib

import chess

from training.features import canonical_board_key


def colour_swap(board: chess.Board) -> chess.Board:
    """Exact colour-exchange symmetry (always legal)."""
    return board.mirror()


def mirror_files(board: chess.Board, *, force: bool = False) -> chess.Board | None:
    """Horizontal mirror; None when not provably label-preserving."""
    if not force:
        if board.castling_rights:
            return None  # live orthodox castling: not a safe symmetry
        if board.ep_square is not None:
            return None  # EP legality under mirror not assumed
    new = board.copy(stack=False)
    # mirror files: sq -> sq ^ 7
    pmap: dict[int, chess.Piece] = {}
    for sq, p in board.piece_map().items():
        pmap[sq ^ 7] = p
    new.set_piece_map(pmap)
    new.castling_rights = 0
    for flag, sq in (
        (chess.BB_H1, chess.H1),
        (chess.BB_A1, chess.A1),
        (chess.BB_H8, chess.H8),
        (chess.BB_A8, chess.A8),
    ):
        if board.castling_rights & flag:
            new.castling_rights |= chess.BB_SQUARES[sq ^ 7]
    new.ep_square = board.ep_square ^ 7 if board.ep_square is not None else None
    return new if new.is_valid() else None


def _map_observation_color_swap(obs: dict) -> dict:
    """Recolour an observation for ``board.mirror()``.

    Colour swap exchanges which *side* a fixed-colour perspective names,
    so the stored value is INVARIANT: perspective flips white<->black and
    u/cp/mate/wdl stay exactly as recorded.  (Flipping name AND value is
    the classic double-flip — it inverts every label.)  Colour-relative
    perspectives (side_to_move, parent_side_to_move) need no change at
    all.  Moves in ``action_uci`` map by vertical flip (sq ^ 56).
    """
    o = dict(obs)
    if o["perspective"] == "white":
        o["perspective"] = "black"
    elif o["perspective"] == "black":
        o["perspective"] = "white"
    # side_to_move / parent_side_to_move are colour-relative already
    if o.get("action_uci"):
        m = chess.Move.from_uci(o["action_uci"])
        o["action_uci"] = chess.Move(
            m.from_square ^ 56, m.to_square ^ 56, promotion=m.promotion
        ).uci()
    return o


def _map_observation_mirror(obs: dict) -> dict:
    o = dict(obs)
    if o.get("action_uci"):
        m = chess.Move.from_uci(o["action_uci"])
        o["action_uci"] = chess.Move(
            m.from_square ^ 7, m.to_square ^ 7, promotion=m.promotion
        ).uci()
    return o


def augment_record(rec: dict, board: chess.Board, transform: str) -> dict | None:
    """Return a derived record under ``transform`` or None if inapplicable.

    transform in {"colour_swap", "mirror_files"}.
    """
    if transform == "colour_swap":
        nb = colour_swap(board)
        obs_fn = _map_observation_color_swap
    elif transform == "mirror_files":
        nb = mirror_files(board)
        if nb is None:
            return None
        obs_fn = _map_observation_mirror
    else:
        raise ValueError(transform)

    new = {
        "schema_version": rec["schema_version"],
        "record_id": rec["record_id"] + f"#{transform}",
        "position": dict(rec["position"]),
        "source": dict(rec["source"]),
        "observations": [obs_fn(o) for o in rec["observations"]],
        "legal_actions_enumerated": rec.get("legal_actions_enumerated", False),
        "split": rec.get("split"),
    }
    new["position"]["fen4"] = canonical_board_key(nb)
    src = dict(rec["source"])
    chain = list(src.get("derivation_chain", []))
    chain.append(
        {
            "op": transform,
            "parent_record_id": rec["record_id"],
            "parent_fen4": rec["position"]["fen4"],
        }
    )
    src["derivation_chain"] = chain
    src["original_position_id"] = rec["source"]["original_position_id"] + f"#{transform}"
    new["source"] = src
    return new


def augmented_id(rec: dict, transform: str) -> str:
    return hashlib.sha256(f"{rec['record_id']}|{transform}".encode()).hexdigest()[:16]

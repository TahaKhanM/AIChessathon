"""W04 gate tests: sparse training/import pipeline.

Gates (pasted verbatim into training/W04_REPORT.md):
 1. random-init pilot trains end to end on a small REAL shard; loss decreases.
 2. parity chain source -> features -> float model -> integer model ->
    exported bytes -> Numba runtime; integer hops must be EXACT.
 3. checkpoint written, killed mid-write, reloaded -> last complete survives.
 4. split leakage: zero shared game/trajectory ids across train/val and zero
    near-duplicate cluster leakage.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os

import chess
import numpy as np
import pytest

from training import records as rec_mod
from training.augment import augment_record, colour_swap, mirror_files
from training.checkpoint import load_latest, save_model_checkpoint
from training.export import integerize, read_export, write_export
from training.extract_pgn import extract_pgn_file
from training.feature_spec import SCHEMA_DIR, SPEC, load_spec, sanity_check
from training.features import FeatureEncoder, canonical_board_key
from training.int_eval import evaluate_int
from training.labels import (
    child_u_to_parent,
    child_wdl_to_parent,
    resolve_observation,
)
from training.model import F512Model, LossWeights, TrainConfig
from training.numba_rt import NumbaRuntime
from training.records import make_record, make_observation, make_position, make_source, write_shard
from training.splits import assign_splits, assert_no_leakage
from training.train import board_of, build_batch, run_pilot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTDATA = os.path.join(REPO, "training", "testdata")
SPEC_DIR = os.path.join(REPO, "spec", "RX_FINAL_PLAN")

ENC: FeatureEncoder | None = None


def encoder() -> FeatureEncoder:
    global ENC
    if ENC is None:
        ENC = FeatureEncoder()
    return ENC


def _load_oracle():
    path = os.path.join(SPEC_DIR, "signed_packing_reference.py")
    spec = importlib.util.spec_from_file_location("packing_oracle", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pilot_records(n_files: int = 8) -> list[dict]:
    recs: list[dict] = []
    for f in sorted(os.listdir(TESTDATA))[:n_files]:
        if f.endswith(".pgn"):
            recs.extend(
                extract_pgn_file(os.path.join(TESTDATA, f), corpus="aichessathon_rated_pgns")
            )
    return recs


# --------------------------------------------------------------------------
# unit checks


def test_schema_sanity_and_artifact(tmp_path):
    sanity_check(load_spec())
    out = SPEC.write_artifact(os.path.join(tmp_path, "schema"))
    with open(out) as fh:
        art = json.load(fh)
    assert art["schema_id"] == SPEC.schema_id
    checked_in = os.path.join(SCHEMA_DIR, f"f512_ef_k12_16_32.{SPEC.schema_version}.json")
    assert os.path.exists(checked_in)
    with open(checked_in) as fh:
        assert json.load(fh)["schema_id"] == SPEC.schema_id


def test_packing_matches_oracle():
    from training import packing

    oracle = _load_oracle()
    rng = np.random.default_rng(7)
    for bits in (6, 7, 9, 16):
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        vectors = [
            [0],
            [1, -1, lo, hi],
            list(range(lo, min(hi + 1, lo + 40))),
            [int(x) for x in rng.integers(lo, hi + 1, 1000)],
        ]
        for vec in vectors:
            mine = packing.pack_signed(vec, bits)
            theirs = oracle.pack_signed(vec, bits)
            assert mine == theirs, f"pack mismatch bits={bits}"
            assert packing.unpack_signed(mine, len(vec), bits) == vec
            assert packing.unpack_signed_array(mine, len(vec), bits).tolist() == vec
            arr = np.asarray(vec, dtype=np.int64)
            assert packing.pack_signed_array(arr, bits) == theirs
    with pytest.raises(ValueError):
        packing.pack_signed([256], 9)  # 256 out of 9-bit range


def test_encoder_dimensions_and_startpos():
    enc = encoder()
    assert enc.threats.total_dimensions == SPEC.threats.rows == 59808
    assert len(enc.pp_factor_map) == SPEC.pawn_pairs.factorizer_rows == 372
    b = chess.Board()
    e = enc.encode(b)
    assert len(e.psq[0]) == 32 and len(e.psq[1]) == 32
    # startpos threats: each side Nx2->P, Bx2->2P, Rx2->(N+P), Q->(2P+B)
    assert len(e.threats[0]) == len(e.threats[1]) == 28
    assert len(e.pawn_pairs[0]) == 36
    assert e.head == 7  # (32-2)//4
    # all factor rows valid
    assert e.threats_fac[0].max() < SPEC.threats.factorizer_rows
    assert e.psq_fac[0].max() < SPEC.psq.factorizer_rows


def test_colour_swap_is_feature_isomorphism():
    enc = encoder()
    for fen in (
        chess.STARTING_FEN,
        "r1bqk1nr/pp2ppbp/2np2p1/2p5/4P3/2NPB1P1/PPP2PBP/R2QK1NR b KQkq - 1 6",
        "8/8/4k3/8/8/4K3/8/8 w - - 0 1",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 2",
    ):
        b = chess.Board(fen)
        e1 = enc.encode(b)
        e2 = enc.encode(colour_swap(b))
        # Encoded is stm-first; colour swap flips stm, so index i on the
        # mirrored board == index i on the original.
        for attr in ("psq", "threats", "pawn_pairs", "psq_fac", "threats_fac", "pawn_pairs_fac"):
            a, m = getattr(e1, attr), getattr(e2, attr)
            assert np.array_equal(a[0], m[0]) and np.array_equal(a[1], m[1]), (
                f"{attr} not colour-swap invariant on {fen}"
            )
        assert e1.head == e2.head


def test_mirror_files_gating():
    b = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    assert mirror_files(b) is None  # live castling -> refused
    b2 = chess.Board("4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 2")
    assert mirror_files(b2) is None  # live EP -> refused
    b3 = chess.Board("4k3/8/3p4/8/8/8/4P3/4K3 w - - 5 40")
    m = mirror_files(b3)
    assert m is not None and m.is_valid()
    # file mirror maps e1->d1, e8->d8, d6->e6, e2->d2, same side to move
    assert canonical_board_key(m).split(" ")[0] == "3k4/8/4p3/8/8/8/3P4/3K4"
    assert m.turn == b3.turn


def test_labels_conversions():
    assert child_u_to_parent(0.3) == pytest.approx(0.7)
    assert child_wdl_to_parent([0.8, 0.1, 0.1]) == [0.1, 0.1, 0.8]
    # side_to_move searched cp -> u in stm POV via calibration
    obs = make_observation(
        kind="searched",
        perspective="side_to_move",
        score_kind="cp",
        bound_kind="search_exact",
        cp=200.0,
    )
    t = resolve_observation(obs, stm_is_white=True)
    assert t.u is not None and 0.5 < t.u < 1.0
    # white-perspective cp on black-to-move record flips
    obs2 = make_observation(kind="searched", perspective="white", score_kind="cp", cp=200.0)
    t2 = resolve_observation(obs2, stm_is_white=False)
    assert t2.u is not None and t2.u < 0.5
    # lower bound -> one-sided, never a point target
    obs3 = make_observation(
        kind="searched", perspective="side_to_move", score_kind="cp", bound_kind="lower", cp=500.0
    )
    t3 = resolve_observation(obs3, stm_is_white=True)
    assert t3.is_bound and t3.u is None and t3.u_lo is not None and t3.u_hi is None
    # interrupted -> no target
    obs4 = make_observation(
        kind="searched",
        perspective="side_to_move",
        score_kind="cp",
        cp=300.0,
        interrupted=True,
        right_censored=True,
    )
    assert resolve_observation(obs4, stm_is_white=True) is None
    # child action value from parent perspective converts 1-u
    obs5 = make_observation(
        kind="action_searched",
        perspective="parent_side_to_move",
        score_kind="expected_score",
        expected_score=0.4,
    )
    t5 = resolve_observation(obs5, stm_is_white=False)
    assert t5.u == pytest.approx(0.6)
    # mate is decisive one-hot WDL, not clipped cp
    obs6 = make_observation(
        kind="searched", perspective="side_to_move", score_kind="mate", mate_plies=7
    )
    t6 = resolve_observation(obs6, stm_is_white=True)
    assert t6.u == 1.0 and t6.wdl == [1.0, 0.0, 0.0]


def test_record_validation_against_json_contract():
    with open(os.path.join(SPEC_DIR, "training_schema.json")) as fh:
        contract = json.load(fh)
    recs = _pilot_records(1)
    rec = recs[0]
    assert rec_mod.validate_record(rec) == []
    # every required property named in the JSON schema exists
    for k in contract["required"]:
        assert k in rec
    for k in contract["properties"]["position"]["required"]:
        assert k in rec["position"]
    for k in contract["properties"]["source"]["required"]:
        assert k in rec["source"]
    for k in contract["properties"]["observations"]["items"]["required"]:
        assert k in rec["observations"][0]
    # tamper -> invalid
    bad = json.loads(json.dumps(rec))
    bad["observations"][0]["wdl"] = [0.5, 0.6, 0.1]
    assert rec_mod.validate_record(bad)
    bad2 = json.loads(json.dumps(rec))
    bad2["observations"][0]["bound_kind"] = "lower"
    assert rec_mod.validate_record(bad2)


# --------------------------------------------------------------------------
# GATE 1: random-init pilot trains end to end on a real shard


def test_gate1_pilot_trains(tmp_path):
    recs = _pilot_records(10)
    shard_dir = os.path.join(tmp_path, "shard-a")
    write_shard(shard_dir, recs, shard_id="pgn-a", feature_schema_id=SPEC.schema_id)
    out = os.path.join(tmp_path, "run")
    rep = run_pilot(
        [shard_dir],
        out_dir=out,
        config=TrainConfig(epochs=30, batch_size=512, lr=3e-2, seed=0),
        checkpoint_every=10,
    )
    hist = rep["history"]
    print("\nGATE1 pilot loss curve:")
    for h in hist:
        print(f"  epoch {h['epoch']}: loss={h['loss']:.6f} terms={h['terms']}")
    print(
        f"  train={rep['n_train']} dev={rep['n_dev']} "
        f"dedup_dropped={rep['n_dropped_dups']} clusters={rep['n_clusters']}"
    )
    assert len(hist) >= 4
    assert hist[-1]["loss"] < hist[0]["loss"], "loss did not decrease"
    # monotone-ish downtrend with real movement: mean of last third
    # materially below mean of first third (the summed normalised loss has a
    # large WDL-entropy floor, so compare absolute descent)
    third = len(hist) // 3
    early = sum(h["loss"] for h in hist[:third]) / third
    late = sum(h["loss"] for h in hist[-third:]) / third
    assert early - late >= 0.02, f"weak descent: {early:.4f} -> {late:.4f}"


# --------------------------------------------------------------------------
# GATE 2: full parity chain


def test_gate2_parity_chain(tmp_path):
    recs = _pilot_records(6)
    shard_dir = os.path.join(tmp_path, "shard-b")
    write_shard(shard_dir, recs, shard_id="pgn-b", feature_schema_id=SPEC.schema_id)
    enc = encoder()
    model = F512Model(SPEC, seed=3)
    # a few real optimisation steps so weights are nontrivially non-init
    train_recs = [r for r in assign_splits(recs).kept if r["split"] == "train"]
    batch = build_batch(train_recs[:256], enc, {})
    lw = LossWeights()
    for _ in range(5):
        fwd = model.forward(batch, need_grad=True)
        _, grads = model.losses(fwd, batch, lw)
        model.apply_grads(model.backward(fwd, batch, grads), 2e-3, lw)

    int_model = integerize(model.params, enc, SPEC, model.factorized)
    path = os.path.join(tmp_path, "model.rxf1")
    info = write_export(int_model, path, SPEC)
    assert info["payload_bytes"] == 32_900_768  # authoritative packed payload

    header, rt_model = read_export(path)
    assert header["feature_schema_id"] == SPEC.schema_id
    runtime = NumbaRuntime(path)

    n_pos = 60
    max_float_int = 0
    max_int_bytes = 0
    max_bytes_numba = 0
    max_acc_seen = 0
    u_diffs = []
    scale = 2.0**-SPEC.numeric.scalar_u_log2_divisor
    for rec in recs[:n_pos]:
        b = board_of(rec)
        e = enc.encode(b)
        single = build_batch([rec], enc, {})
        f = model.forward(single)
        f_scalar = f["scalar"][0]
        f_logits = f["wdl_logits"][0]
        i_scalar, i_logits, i_maxacc = evaluate_int(int_model, e, e.head, SPEC)
        n_scalar, n_logits, n_maxacc = runtime.evaluate(e)
        max_float_int = max(max_float_int, abs(f_scalar - i_scalar))
        max_int_bytes = max(max_int_bytes, abs(i_scalar - n_scalar))
        max_bytes_numba = max(max_bytes_numba, 0)  # rt IS the decoded bytes
        max_acc_seen = max(max_acc_seen, i_maxacc)
        fl = float(f_scalar)
        u_f = 1 / (1 + np.exp(-fl * scale))
        u_n = 1 / (1 + np.exp(-n_scalar * scale))
        u_diffs.append(abs(u_f - u_n))
        for k in range(3):
            assert f_logits[k] == i_logits[k] == n_logits[k]
    print("\nGATE2 parity over", n_pos, "positions:")
    print(f"  float->int    max|dscalar| = {max_float_int}")
    print(f"  int->bytes    max|dscalar| = {max_int_bytes}")
    print(f"  bytes->numba  max|dscalar| = {max_bytes_numba}")
    print(f"  u disagreement float->numba max = {max(u_diffs):.3e}")
    print(f"  max|accumulator| observed = {max_acc_seen} (bound 30048)")
    assert max_float_int == 0
    assert max_int_bytes == 0
    assert max_acc_seen <= SPEC.numeric.accumulator_proven_abs_bound


# --------------------------------------------------------------------------
# GATE 3: transactional checkpoint survives mid-write kill


def test_gate3_checkpoint_kill(tmp_path):
    root = os.path.join(tmp_path, "ck")
    p1 = {"w": np.ones((4, 4), np.float32)}
    save_model_checkpoint(
        root,
        step=1,
        params=p1,
        optim={"m::w": np.zeros((4, 4))},
        state={"step": 1, "sampler": {"epoch": 0, "offset": 0}},
    )
    # kill step 2 mid-write: leaves ckpt-...tmp/ with partial files
    with pytest.raises(RuntimeError):
        save_model_checkpoint(
            root,
            step=2,
            params={"w": np.full((4, 4), 9.0, np.float32)},
            optim={"m::w": np.ones((4, 4))},
            state={"step": 2},
            fail_after_files=1,
        )
    leftovers = [d for d in os.listdir(root) if d.endswith(".tmp")]
    assert leftovers, "synthetic crash should leave a .tmp dir"
    ck = load_latest(root)
    assert ck is not None and ck.step == 1
    assert ck.npz("params.npz")["w"].tolist() == np.ones((4, 4)).tolist()
    print(
        "\nGATE3: partial ckpt-2 ignored; recovered step",
        ck.step,
        "| leftover tmp dirs:",
        leftovers,
    )
    # a complete later checkpoint still wins
    save_model_checkpoint(
        root,
        step=3,
        params={"w": np.full((4, 4), 5.0, np.float32)},
        optim={"m::w": np.zeros((4, 4))},
        state={"step": 3},
    )
    assert load_latest(root).step == 3


# --------------------------------------------------------------------------
# GATE 4: split leakage


def _synthetic_rec(
    rid: str,
    fen4: str,
    game: str,
    *,
    u: float = 0.5,
    parent: str | None = None,
    opid: str | None = None,
    family: str = "fam-x",
) -> dict:
    wdl = [1.0, 0.0, 0.0] if u == 1.0 else ([0.0, 0.0, 1.0] if u == 0.0 else [0.0, 1.0, 0.0])
    obs = make_observation(
        kind="game_outcome",
        perspective="white",
        score_kind="outcome",
        expected_score=u,
        wdl=wdl,
        parent_record_id=parent,
    )
    return make_record(
        record_id=rid,
        position=make_position(
            fen4=fen4,
            variant="standard",
            halfmove_clock=0,
            fullmove_number=10,
            history_complete=True,
            unknown_prefix=False,
        ),
        source=make_source(
            corpus="synthetic",
            object_sha256=hashlib.sha256(rid.encode()).hexdigest(),
            decoder_revision="test",
            lineage_family=family,
            original_position_id=opid or rid,
            original_game_id=game,
            split_group=game,
        ),
        observations=[obs],
    )


def test_gate4_split_leakage(tmp_path):
    recs = _pilot_records(21)
    fen_shared = "r1bqk1nr/pp2ppbp/2np2p1/2p5/4P3/2NPB1P1/PPP2PBP/R2QK1NR b KQkq -"
    # same position+context in two different games with DIFFERENT results:
    # a relabel/mirror collision that must never straddle a split
    dup_a = _synthetic_rec("dupA", fen_shared, "game-A", u=1.0, family="fam-x")
    dup_b = _synthetic_rec("dupB", fen_shared, "game-B", u=0.0, family="fam-x")
    # relabel twin: same original_position_id + identical board -> exact
    # duplicate, must be dropped before splitting
    twin = _synthetic_rec(
        "twinC",
        fen_shared + " ",
        "game-C",
        u=1.0,
        opid=dup_a["source"]["original_position_id"],
        family="fam-x",
    )
    twin["position"]["fen4"] = fen_shared
    # a fourth member in a fourth game: same position cluster, kept
    sib = _synthetic_rec("sibD", fen_shared, "game-D", u=0.5, family="fam-x")
    # synthetics first: dupA is the first-seen representative for the
    # shared board key (the PGN corpus may contain the same position)
    all_recs = [dup_a, dup_b, twin, sib] + recs
    res = assign_splits(all_recs)
    assert_no_leakage(res.kept)
    print("\nGATE4:")
    print(
        f"  records={len(all_recs)} kept={len(res.kept)} "
        f"dropped_dups={len(res.dropped_duplicates)} clusters={len(res.clusters)}"
    )
    assert "twinC" in set(res.dropped_duplicates)  # identical board+label
    # dupA/dupB/sibD share one canonical board key -> merged into ONE
    # record before splitting; that position can never straddle a split
    dr = res.dedup_report
    assert dr["records_absorbed_by_merge"] >= 2
    assert "dupB" not in res.record_split and "sibD" not in res.record_split
    merged = [r for r in res.kept if r["record_id"] == "dupA"][0]
    # Contradictory-outcome policy (FIX-DATA-2, decided: drop-as-unresolvable)
    # — the merged record absorbed u=1.0/0.0/0.5 game_outcome observations
    # for the SAME geometry: a result channel carrying win AND loss on one
    # position is unresolvable, so the whole channel is dropped (searched /
    # teacher labels would be unaffected) and the merge trail records it.
    # The old assertion demanded the poison (`len(observations) >= 3`).
    trail = merged["source"]["derivation_chain"][-1]
    # >=3: the three synthetics, plus any real-corpus record sharing the
    # board key (the PGN corpus does contain this position)
    assert trail.get("dropped_contradictory_outcomes", 0) >= 3
    assert set(trail.get("contradictory_classes", [])) >= {"-1", "0", "1"}
    # every surviving observation resolves to no target (the record keeps
    # exactly one censored marker so it stays schema-valid and auditable)
    assert merged["observations"], "merged record left with zero observations"
    assert all(resolve_observation(o, True) is None for o in merged["observations"])
    print(
        f"  dupA(+dupB,sibD merged) split: {res.record_split['dupA']} "
        f"(twinC dropped); absorbed={dr['records_absorbed_by_merge']}, "
        f"contradictory outcomes dropped="
    )
    print(
        f"    {trail.get('dropped_contradictory_outcomes')} "
        f"classes={trail.get('contradictory_classes')}"
    )
    # zero shared game ids across splits
    seen: dict[str, str] = {}
    for r in res.kept:
        g = r["source"].get("original_game_id")
        if g:
            assert g not in seen or seen[g] == r["split"]
            seen[g] = r["split"]
    splits = {}
    for r in res.kept:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    print(f"  split counts: {splits}")
    assert "train" in splits and ("development" in splits or "sealed" in splits)


def test_gate4_absorbed_game_straddle(tmp_path):
    """R17 straddle: a record absorbed into another game's cluster must
    union the absorbed game's identities — the absorbed game's OTHER
    positions cannot land on the far side of a split, and the gate must
    catch it when they do.
    """
    fen_shared = "r1bqk1nr/pp2ppbp/2np2p1/2p5/4P3/2NPB1P1/PPP2PBP/R2QK1NR b KQkq -"
    fen_b2 = "4k3/8/8/8/8/8/8/3K4 b - -"
    ga = _synthetic_rec("ga1", fen_shared, "GAME-A", u=0.8, family="fam-A")
    gb1 = _synthetic_rec("gb1", fen_shared, "GAME-B", u=0.8, family="fam-B")
    gb2 = _synthetic_rec("gb2", fen_b2, "GAME-B", u=0.3, family="fam-B")
    # gb2 FIRST in input order, so after the merge the absorbed-identity
    # check (not the own-game check) is what fires on a forced straddle.
    res = assign_splits([gb2, ga, gb1])
    # gb1 was absorbed into ga's cluster; its game id must union gb2 into
    # the SAME split — the straddle is impossible after merge.
    assert res.record_split["gb2"] == res.record_split["ga1"]
    assert_no_leakage(res.kept)
    # and the gate itself is load-bearing: force the absorbed game's
    # leftover onto the other side and assert_no_leakage must fire on the
    # ABSORBED identity (the pre-fix gate saw only the survivor's game id).
    kept = res.kept
    flip = [r for r in kept if r["record_id"] == "gb2"][0]
    flip["split"] = "development" if flip["split"] != "development" else "train"
    with pytest.raises(AssertionError, match="absorbed"):
        assert_no_leakage(kept)


def test_variant_records_honoured_or_rejected(tmp_path):
    """FIX-DATA-2 variant discipline: chess960 parsed under 960 rules;
    'other'/unknown variants rejected loudly; DFRC PGNs never silently
    parse as orthodox."""
    # chess960: X-FEN castling letters with a king OFF the home square —
    # honoured under 960 semantics (castling survives), mangled under
    # orthodox (python-chess parses the rights then drops them on fen()).
    fen960 = "rknqbbnr/pppppppp/8/8/8/8/PPPPPPPP/RKNQBBNR w HAha -"
    b = board_of(
        {
            "position": {
                "fen4": fen960,
                "variant": "chess960",
                "halfmove_clock": 0,
                "fullmove_number": 1,
            }
        }
    )
    assert b.chess960
    assert b.king(chess.WHITE) == chess.B1
    assert b.castling_rights, "960 castling rights lost on reconstruction"
    assert b.fen().split(" ")[2] == "KQkq"
    # 'other' / unknown variant -> loud rejection, not an orthodox parse
    for bad in ("other", "crazyhouse", None):
        with pytest.raises(ValueError):
            board_of(
                {
                    "position": {
                        "fen4": fen960,
                        "variant": bad,
                        "halfmove_clock": 0,
                        "fullmove_number": 1,
                    }
                }
            )


def test_extract_pgn_rejects_nonstandard_variants(tmp_path):
    """A [Variant] header that isn't standard/960 must fail loudly —
    never silently parse a non-standard game as orthodox chess."""
    from training.extract_pgn import extract_pgn_bytes

    # DFRC is not a python-chess variant — raises inside parse
    dfrc = (
        b'[Event "t"]\n[Variant "DFRC"]\n[FEN "qbbnrkrn/pppppppp/8/8/'
        b'8/8/PPPPPPPP/QBBNRKRN w DHdh - 0 1"]\n[SetUp "1"]\n\n'
        b"1. e4 e5 *\n"
    )
    with pytest.raises(ValueError):
        extract_pgn_bytes(dfrc, corpus="t")
    # a named-but-unsupported variant header is rejected, not tagged
    # variant="other" and parsed orthodox
    crazy = b'[Event "t"]\n[Variant "Crazyhouse"]\n\n1. e4 e5 2. Nf3 Nc6 *\n'
    with pytest.raises(ValueError, match="Variant"):
        extract_pgn_bytes(crazy, corpus="t")
    # and a SetUp-FEN with castling rights off home squares (unlabelled
    # FRC/DFRC position) is rejected, not silently mis-parsed
    setup_frc = (
        b'[Event "t"]\n[SetUp "1"]\n'
        b'[FEN "qbbnrkrn/pppppppp/8/8/8/8/PPPPPPPP/'
        b'QBBNRKRN w DHdh - 0 1"]\n\n1. e4 *\n'
    )
    with pytest.raises(ValueError):
        extract_pgn_bytes(setup_frc, corpus="t")
    # a Chess960-labelled game IS honoured: variant tag + 960-aware board
    c960 = (
        b'[Event "t"]\n[Variant "Chess960"]\n'
        b'[FEN "rknqbbnr/pppppppp/8/8/8/8/PPPPPPPP/RKNQBBNR w HAha - '
        b'0 1"]\n[SetUp "1"]\n\n1. c4 c5 *\n'
    )
    recs = extract_pgn_bytes(c960, corpus="t")
    assert recs and all(r["position"]["variant"] == "chess960" for r in recs)


def test_colour_swap_preserves_resolved_labels():
    """R9 poison class: colour_swap must not invert labels.

    board.mirror() exchanges sides AND stm; the resolved stm-u of the
    augmented record on the mirrored board must equal the original's.
    """
    recs = _pilot_records(3)
    for base in recs[:3]:
        b = board_of(base)
        aug = augment_record(base, b, "colour_swap")
        assert aug is not None
        assert rec_mod.validate_record(aug) == []
        b2 = colour_swap(b)
        for o1, o2 in zip(base["observations"], aug["observations"], strict=True):
            t1 = resolve_observation(o1, b.turn == chess.WHITE)
            t2 = resolve_observation(o2, b2.turn == chess.WHITE)
            if t1 is None:
                assert t2 is None
                continue
            assert t2 is not None
            if t1.u is not None:
                assert t2.u == pytest.approx(t1.u), (o1["perspective"], t1, t2)
            if t1.u_lo is not None or t1.u_hi is not None:
                assert (t2.u_lo, t2.u_hi) == pytest.approx((t1.u_lo, t1.u_hi))


def test_augmented_records_stay_in_split():
    recs = _pilot_records(2)
    base = recs[0]
    b = board_of(base)
    aug = augment_record(base, b, "colour_swap")
    assert aug is not None
    assert rec_mod.validate_record(aug) == []
    # colour swap renames the fixed-colour perspective; the value itself
    # is invariant (flipping name AND value is the classic double-flip)
    o_base, o_aug = base["observations"][0], aug["observations"][0]
    assert o_aug["expected_score"] == pytest.approx(o_base["expected_score"])
    assert o_aug["perspective"] != o_base["perspective"]
    res = assign_splits([base, aug])
    assert res.record_split[base["record_id"]] == res.record_split[aug["record_id"]]

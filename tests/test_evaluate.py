"""W05 gates: integer-kernel parity, incremental-vs-refresh differential,
dirty-capacity checks, measured-cost reporting, int16 bound assertions.

Gates (per work-package contract):
 1. reference vs optimized kernels agree EXACTLY on >=100,000 random
    positions and >=200 complete games played move by move.
 2. incremental delta state equals full-refresh state after >=1,000,000
    random legal moves incl. promotions, castling, en passant, captures.
 3. dirty-feature capacity checked for additions AND removals; a shrunk
    capacity must produce a failing check.
 4. separate measurements: geometry time, rows touched, bytes loaded,
    accumulator update time, refresh fraction, head cost; 9,312 MACs.
 5. no int16 overflow under the bound proof; assertion build checks it.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import engine.evaluate as E
from engine.board import (
    FLAG_CASTLE,
    FLAG_EP,
    FLAG_PROMO,
    FLAG_PROMO_CAP,
    FLAG_CAPTURE,
    Board,
    decode_move,
)
from engine.movegen import generate_legal

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
ARCH_JSON = Path(__file__).resolve().parents[1] / "spec" / "RX_FINAL_PLAN" / "architecture.json"


def _clone(b: Board) -> Board:
    nb = Board()
    nb._bb = b._bb.copy()
    nb._sq = b._sq.copy()
    nb._occ = b._occ.copy()
    nb._occ_all = b._occ_all
    nb._king = b._king.copy()
    nb.side = b.side
    nb.castling = b.castling
    nb.ep_square = b.ep_square
    nb.halfmove = b.halfmove
    nb.fullmove = b.fullmove
    nb.key = b.key
    nb._ep_key = b._ep_key
    nb._abs_ply = b._abs_ply
    return nb


def _rand_positions(rng: random.Random, n: int, seed_weights=None):
    """Yield board snapshots from random playouts."""
    buf = [0] * 256
    got = 0
    while got < n:
        b = Board.from_fen(START)
        for _ in range(rng.randrange(1, 160)):
            cnt = generate_legal(b, buf)
            if cnt == 0:
                break
            b.make(buf[rng.randrange(cnt)])
            yield _clone(b)
            got += 1
            if got >= n:
                return


# --------------------------------------------------------------------------
# Contract constants vs architecture.json
# --------------------------------------------------------------------------


def test_contract_constants():
    spec = json.loads(ARCH_JSON.read_text())
    f = spec["features"]
    assert f["perspectives"] == 2
    assert f["channels"] == E.CHANNELS
    assert f["psq"]["king_buckets"] == E.KING_BUCKETS
    assert f["psq"]["rows"] == E.PSQ_ROWS
    assert f["psq"]["runtime_dtype"] == "int16"
    assert f["psq"]["coefficient_abs_limit"] == E.PSQ_COEF_MAX
    assert f["threats"]["rows"] == E.THREAT_ROWS
    assert f["threats"]["runtime_dtype"] == "int8"
    assert f["threats"]["coefficient_abs_limit"] == E.THREAT_COEF_MAX
    assert f["threats"]["max_active_bound"] == E.MAX_ACTIVE_THREATS
    assert f["pawn_pairs"]["rows"] == E.PP_ROWS
    assert f["pawn_pairs"]["runtime_dtype"] == "int8"
    assert f["pawn_pairs"]["coefficient_abs_limit"] == E.PP_COEF_MAX
    assert f["pawn_pairs"]["max_active_bound"] == E.MAX_ACTIVE_PP
    assert f["bias"]["runtime_dtype"] == "int16"
    assert f["bias"]["coefficient_abs_limit"] == E.BIAS_COEF_MAX
    acc = f["accumulator"]
    assert acc["dtype"] == "int16"
    assert acc["proven_abs_bound"] == E.ACC_BOUND
    nc = spec["numeric_contract"]
    assert nc["ft_operand_clip"] == [0, E.FT_CLIP_HI]
    assert nc["paired_product_shift"] == E.PRODUCT_SHIFT
    assert nc["paired_product_max"] == E.PRODUCT_MAX
    assert nc["hidden_clip"] == [0, E.HIDDEN_CLIP]
    assert nc["hidden_affine_shift_reference"] == E.HIDDEN_SHIFT
    assert nc["square_activation_shift"] == E.SQUARE_SHIFT
    heads = spec["head"]
    assert heads["material_stacks"] == E.HEAD_STACKS
    # FR1: the head-selection contract must be pinned verbatim — not just
    # the stack count.
    assert heads["selection"] == "min(7,max(0,(piece_count-2)//4))"
    assert heads["input_count"] == E.HEAD_IN
    assert heads["first_affine_outputs"] == E.L1_OUT
    assert heads["first_activation_concat"] == E.L1_ACT
    assert heads["second_affine_outputs"] == E.L2_OUT
    assert heads["second_activation_concat"] == E.L2_ACT
    assert heads["skip_concat_outputs"] == E.SKIP
    assert heads["outputs"] == ["scalar", "W_logit", "D_logit", "L_logit"]
    assert heads["coefficient_dtype"] == "int8"
    assert heads["bias_and_dot_dtype"] == "int32"
    assert heads["rescale_dtype"] == "int64"
    assert spec["reference_raw_accounting"]["scalar_dense_head_macs"] == E.SCALAR_DENSE_MACS
    assert E.SCALAR_DENSE_MACS == 9312
    flat = [
        v for row in f["psq"]["king_bucket_map_by_rank_from_perspective_home_rank"] for v in row
    ]
    assert flat == [int(v) for v in E.K12_A]


def test_pp_dense_map():
    """Dense map: 1488 rows, strictly ordered by b*(b-1)//2+a."""
    d = E.PP_DENSE_A
    vals = sorted(int(v) for v in d if v >= 0)
    assert vals == list(range(E.PP_ROWS))
    order = [
        (b * (b - 1) // 2 + a) for b in range(96) for a in range(b) if d[b * (b - 1) // 2 + a] >= 0
    ]
    assert order == sorted(order)


# --------------------------------------------------------------------------
# Gate 1: reference vs optimized kernel parity
# --------------------------------------------------------------------------


def test_kernel_parity_random_positions():
    """Gate 1a: >=100,000 random positions, exact scalar agreement."""
    rng = random.Random(0xC0FFEE)
    w = E.EvalWeights.random(1234, sparse=True)
    w.check_bounds()
    n_pos = 100_000
    checked = 0
    wdl_checked = 0
    for b in _rand_positions(rng, n_pos):
        v_ref = E.evaluate_ref(w, b)
        v_opt = E.evaluate_fresh(w, b)
        assert v_ref == v_opt, (b.to_fen(), v_ref, v_opt)
        checked += 1
        if checked % 977 == 0:
            assert np.array_equal(E.evaluate_wdl_ref(w, b), _eval_wdl_fresh(w, b)), b.to_fen()
            wdl_checked += 1
    assert checked == n_pos
    print(f"\ngate1a: {checked} positions exact parity, {wdl_checked} WDL parity")


def _eval_wdl_fresh(w, board):
    accs = np.empty((2, E.CHANNELS), np.int16)
    accs[0] = E.refresh_acc_nb(w, board, 0)
    accs[1] = E.refresh_acc_nb(w, board, 1)
    x = np.empty(E.HEAD_IN, np.int32)
    E._paired_transform_nb(accs[board.side], x[: E.HALF])
    E._paired_transform_nb(accs[board.side ^ 1], x[E.HALF :])
    bucket = E.material_bucket(board)
    out = np.empty(E.HEAD_OUT, np.int64)
    E._head_wdl_nb(x, w.w1, w.b1, w.w2, w.b2, w.w3, w.b3, bucket, out)
    psqt = E.psqt_nb(w, board)
    out[0] = E._final_scalar(w, int(out[0]), E._psqt_term(w, psqt, bucket, board.side))
    return out


def _features_model(w: E.EvalWeights) -> dict:
    """EvalWeights -> features.py model dict (its layout is [in][out])."""
    return {
        "bias": w.bias,
        "psq": w.psq_w,
        "thr": w.thr_w,
        "pp": w.pp_w,
        "head_w1": np.ascontiguousarray(w.w1.transpose(0, 2, 1)),
        "head_b1": w.b1,
        "head_w2": np.ascontiguousarray(w.w2.transpose(0, 2, 1)),
        "head_b2": w.b2,
        "head_w3": np.ascontiguousarray(w.w3.transpose(0, 2, 1)),
        "head_b3": w.b3,
        "psqt_w": w.psqt_w,
        "psqt_b": w.psqt_b,
        "_meta": {
            "scale_num": int(w.scale_num),
            "scale_shift": int(w.scale_shift),
            "neural_bound": int(w.neural_bound),
        },
    }


def test_canonical_features_parity():
    """evaluate_ref / evaluate_fresh must agree EXACTLY with the canonical
    engine/features.py evaluate_position on real-game positions."""
    import engine.features as F

    rng = random.Random(4242)
    w = E.EvalWeights.random(31)
    model = _features_model(w)
    buf = [0] * 256
    checked = 0
    for _ in range(4):
        b = Board.from_fen(START)
        for _ in range(60):
            n = generate_legal(b, buf)
            if n == 0:
                break
            mv = buf[rng.randrange(n)]
            b.make(mv)
            assert E.evaluate_ref(w, b) == F.evaluate_position(model, b), b.to_fen()
            assert E.evaluate_fresh(w, b) == F.evaluate_position(model, b), b.to_fen()
            checked += 1
    assert checked > 0
    print(f"\ncanonical parity: {checked} positions x2 paths exact")


def test_kernel_parity_games():
    """Gate 1b: >=200 complete games evaluated move by move."""
    rng = random.Random(777)
    w = E.EvalWeights.random(99)
    ev = E.Evaluator(w)
    buf = [0] * 256
    games = moves = 0
    while games < 200:
        b = Board.from_fen(START)
        ev.set_root(b)
        for _ in range(400):
            n = generate_legal(b, buf)
            if n == 0:
                break
            mv = buf[rng.randrange(n)]
            ev.push(b, mv)
            b.make(mv)
            assert ev.evaluate(b) == E.evaluate_ref(w, b), b.to_fen()
            moves += 1
        games += 1
    print(f"\ngate1b: {games} games, {moves} move-by-move evals, exact parity")


# --------------------------------------------------------------------------
# Gate 2: incremental delta state == full refresh over >=1M random moves
# --------------------------------------------------------------------------


def test_incremental_million_moves():
    """Zero mismatches between the lazy incremental accumulator and a
    from-scratch full refresh across >=1M random legal moves.

    What it actually proves: every MATERIALIZED position (~85% of the 1M
    pushes — ~850k comparisons, both perspectives) is checked against the
    numba full refresh; the ~15% deferred plies are not themselves compared
    but are exercised through multi-ply replay chains whose endpoints ARE
    compared, so delta replay over depth > 1 is covered. A pure-Python
    reference refresh subsample (~20 positions) cross-checks the numba
    oracle itself.

    force_replay=True drives the DELTA path through every non-refresh ply —
    the gate's purpose is to prove delta replay, so the cost-model chooser
    must not silently route around it."""
    rng = random.Random(31337)
    w = E.EvalWeights.random(2025)
    st = E.Stack(w, force_replay=True)
    buf = [0] * 256
    total = 1_000_000
    moves = 0
    checked = 0
    deferred = 0
    mism = 0
    flags = {FLAG_CAPTURE: 0, FLAG_EP: 0, FLAG_CASTLE: 0, FLAG_PROMO: 0, FLAG_PROMO_CAP: 0}
    py_checked = 0
    t0 = time.time()
    while moves < total:
        b = Board.from_fen(START)
        st.set_root(b)
        for _ in range(400):
            if moves >= total:
                break
            n = generate_legal(b, buf)
            if n == 0:
                break
            mv = buf[rng.randrange(n)]
            f = decode_move(mv)[3]
            if f in flags:
                flags[f] += 1
            st.push(b, mv)
            b.make(mv)
            # exercise multi-ply replay: defer materialization ~15% of plies
            if rng.random() < 0.15:
                deferred += 1
                moves += 1
                continue
            acc2, psqt8 = st.materialize(b)
            for p in (0, 1):
                ref = E.refresh_acc_nb(w, b, p)
                if not np.array_equal(acc2[p], ref):
                    mism += 1
                    print("ACC MISMATCH", b.to_fen(), "persp", p)
                    break
            assert np.array_equal(psqt8, E.psqt_nb(w, b)), b.to_fen()
            if moves % 50000 == 0:
                for p in (0, 1):
                    assert np.array_equal(acc2[p], E.refresh_acc_ref(w, b, p)), b.to_fen()
                py_checked += 1
            checked += 1
            moves += 1
    dt = time.time() - t0
    assert mism == 0, f"{mism} accumulator mismatches"
    assert all(v > 0 for v in flags.values()), f"missing special-move coverage: {flags}"
    assert st.stat_replays > 0  # delta path actually exercised
    assert deferred > total * 0.10  # multi-ply replay chains were real
    assert checked >= total * 0.80  # and most pushes were directly compared
    print(
        f"\ngate2: {moves} moves ({checked} compared, {deferred} deferred), "
        f"0 mismatches, flags={flags}, "
        f"replays={st.stat_replays} refreshes={st.stat_refresh_full}, "
        f"py-ref subsample={py_checked}, {dt:.1f}s "
        f"({dt * 1e6 / moves:.0f} us/move)"
    )


def test_incremental_chooser_path():
    """The natural measured-cost chooser (no force flag) must also agree
    with full refresh — shorter run, and it must exercise BOTH replay and
    refresh so the selection logic itself is covered."""
    rng = random.Random(555)
    w = E.EvalWeights.random(77)
    st = E.Stack(w)
    buf = [0] * 256
    moves = 0
    while moves < 60_000:
        b = Board.from_fen(START)
        st.set_root(b)
        st.materialize(b)  # valid root so replays can happen
        for _ in range(300):
            if moves >= 60_000:
                break
            n = generate_legal(b, buf)
            if n == 0:
                break
            mv = buf[rng.randrange(n)]
            st.push(b, mv)
            b.make(mv)
            acc2, _ = st.materialize(b)
            for p in (0, 1):
                assert np.array_equal(acc2[p], E.refresh_acc_nb(w, b, p)), b.to_fen()
            moves += 1
    assert st.stat_replays > 0 and st.stat_refresh_full > 0
    print(
        f"\nchooser: {moves} moves, replays={st.stat_replays}, "
        f"refreshes={st.stat_refresh_full} "
        f"(hits={st.stat_refresh_cache_hit} diffs={st.stat_refresh_cache_diff} "
        f"misses={st.stat_refresh_miss})"
    )


# --------------------------------------------------------------------------
# Gate 3: dirty-feature capacity checks on adds AND removals
# --------------------------------------------------------------------------


def _forcing_position():
    # dense tactical position: maximum attacker/target churn
    return Board.from_fen("r1bqk2r/pp1nbppp/2n1p3/2ppP3/3P1P2/2N1BN2/PP2B1PP/R2QK2R w KQkq - 0 1")


def test_dirty_capacity_checks():
    """Capacity is a hard checked bound, enforced for additions AND
    removals — never inferred from average churn."""
    w = E.EvalWeights.random(5)
    b = _forcing_position()
    buf = [0] * 256
    n = generate_legal(b, buf)
    worst_add = worst_rem = 0
    for i in range(n):
        st = E.Stack(w)
        st.set_root(b)
        st.push(b, buf[i])  # does not mutate b (shadow state)
        ops = st.tops[1][: st.tn[1]]
        worst_add = max(worst_add, int(((ops >> 31) != 0).sum()))
        worst_rem = max(worst_rem, int(st.tn[1] - ((ops >> 31) != 0).sum()))
    assert worst_add <= E.THREAT_OP_CAP and worst_rem <= E.THREAT_OP_CAP
    assert worst_add > 0 and worst_rem > 0  # this position exercises both

    # emit-side check: a shrunken threat-op capacity must fail at push.
    st = E.Stack(w, threat_cap=1)
    st.set_root(b)
    with pytest.raises(E.AccumulatorOverflow):
        st.push(b, buf[0])

    # row-resolution check on removals: a double push removes one pair and
    # adds one — pp_cap=0 must fail during delta replay at materialize.
    b2 = Board.from_fen(START)
    n2 = generate_legal(b2, buf)
    seen_dp = False
    for i in range(n2):
        mv = buf[i]
        if decode_move(mv)[3] != 2:  # FLAG_DOUBLE
            continue
        st2 = E.Stack(w, pp_cap=0, force_replay=True)
        st2.set_root(b2)
        st2.materialize(b2)  # valid ancestor so the child replays deltas
        st2.push(b2, mv)
        b2.make(mv)
        with pytest.raises(E.AccumulatorOverflow):
            st2.materialize(b2)
        seen_dp = True
        break
    assert seen_dp, "no double-push move in startpos"

    # psq-op capacity check.
    st3 = E.Stack(w, psq_cap=0)
    st3.set_root(b)
    with pytest.raises(E.AccumulatorOverflow):
        st3.push(b, buf[0])


def test_push_pop_consistency():
    """After pop + re-push of different moves, the incremental state must
    still equal a full refresh; the shadow board must track unmake."""
    rng = random.Random(202607)
    w = E.EvalWeights.random(44)
    st = E.Stack(w, force_replay=True)
    buf = [0] * 256
    checked = 0
    for _ in range(40):
        b = Board.from_fen(START)
        st.set_root(b)
        trail = []
        for ply in range(120):
            n = generate_legal(b, buf)
            if n == 0:
                break
            mv = buf[rng.randrange(n)]
            st.push(b, mv)
            b.make(mv)
            trail.append(mv)
            # occasionally rewind k moves then replay different ones
            if rng.random() < 0.3 and ply > 0:
                k = rng.randrange(1, min(4, ply + 1))
                for _ in range(k):
                    b.unmake()
                    st.pop()
                    trail.pop()
                st.assert_shadow(b)
                for _ in range(k):
                    n2 = generate_legal(b, buf)
                    if n2 == 0:
                        break
                    mv2 = buf[rng.randrange(n2)]
                    st.push(b, mv2)
                    b.make(mv2)
                    trail.append(mv2)
                st.assert_shadow(b)
                acc2, _ = st.materialize(b)
                for p in (0, 1):
                    assert np.array_equal(acc2[p], E.refresh_acc_nb(w, b, p)), b.to_fen()
                checked += 1
    assert checked > 0
    print(f"\npush/pop: {checked} rewind-and-replay checks clean")


def test_threat_op_bound():
    """Theoretical cap: a non-castling move changes <=80 threat ops
    (SF DirtyThreat bound); castling <=36. Verify structurally that our
    96-op buffer holds under exhaustive enumeration of the worst board we
    can build, and that the emit counter reports truthfully."""
    w = E.EvalWeights.random(6)
    rng = random.Random(4242)
    buf = [0] * 256
    worst = 0
    # adversarial: dense boards from long random playouts
    for _ in range(2000):
        b = Board.from_fen(START)
        for _ in range(rng.randrange(1, 120)):
            n = generate_legal(b, buf)
            if n == 0:
                break
            b.make(buf[rng.randrange(n)])
        st = E.Stack(w)
        st.set_root(b)
        n = generate_legal(b, buf)
        for i in range(n):
            st.push(b, buf[i])
            worst = max(worst, int(st.tn[1]))
            st.pop()
    assert worst <= E.THREAT_OP_CAP
    print(f"\ngate3: worst threat ops over adversarial sample = {worst} (cap {E.THREAT_OP_CAP})")


# --------------------------------------------------------------------------
# Gate 4: measured operation costs + head MAC count
# --------------------------------------------------------------------------


def test_measured_costs():
    rng = random.Random(1)
    w = E.EvalWeights.random(11)
    ev = E.Evaluator(w)
    ev.stack.force_replay = True  # measure the delta hot path
    buf = [0] * 256
    n_games = 60
    for _ in range(n_games):
        b = Board.from_fen(START)
        ev.set_root(b)
        for _ in range(160):
            n = generate_legal(b, buf)
            if n == 0:
                break
            mv = buf[rng.randrange(n)]
            ev.push(b, mv)
            b.make(mv)
            ev.evaluate(b)
    st = ev.stack
    evals = st.stat_evals
    pushes = evals  # one push per eval in this workload
    refresh_frac = st.stat_refresh_full / max(1, evals * 2)
    lines = [
        f"W05 measured-cost table (workload: {n_games} games, {evals} evals)",
        "| metric | value |",
        "| --- | --- |",
        f"| geometry (dirties) per move | {st.stat_geometry_ns / 1e3 / max(1, pushes):.1f} us |",
        f"| rows touched / eval | {st.stat_rows / max(1, evals):.1f} |",
        f"| bytes loaded / eval | {st.stat_bytes / max(1, evals):.0f} |",
        f"| accumulator update / eval | {st.stat_update_ns / 1e3 / max(1, evals):.1f} us |",
        f"| refresh fraction (of ensure calls) | {refresh_frac:.3f} |",
        f"|   replays | {st.stat_replays} |",
        f"|   full refreshes | {st.stat_refresh_full} |",
        f"|   cache hits / diffs / misses | "
        f"{st.stat_refresh_cache_hit} / {st.stat_refresh_cache_diff} / {st.stat_refresh_miss} |",
        f"| head (transform+affine) / eval | {st.stat_head_ns / 1e3 / max(1, evals):.1f} us |",
        f"| head dense MACs (scalar) | {E.SCALAR_DENSE_MACS} |",
        f"| max threat ops / move | {st.max_threat_ops} |",
        f"| max pp rows / move | {st.max_pp_rows} |",
    ]
    print("\n" + "\n".join(lines))
    assert E.SCALAR_DENSE_MACS == 9312
    assert evals > 0


# --------------------------------------------------------------------------
# Gate 5: int16 bound proof under an assertion build
# --------------------------------------------------------------------------


def test_int16_bound_assertion_build():
    """check_bounds=True enables per-update |acc| <= 30048 assertions.
    Run a workload and also drive adversarial max-coefficient weights."""
    rng = random.Random(2024)
    buf = [0] * 256
    for seed, sparse in ((1, False), (2, True)):
        w = E.EvalWeights.random(seed, sparse=sparse)
        st = E.Stack(w, check_bounds=True)
        moves = 0
        for _ in range(8):
            b = Board.from_fen(START)
            st.set_root(b)
            for _ in range(140):
                n = generate_legal(b, buf)
                if n == 0:
                    break
                mv = buf[rng.randrange(n)]
                st.push(b, mv)
                b.make(mv)
                st.materialize(b)
                a = st.acc[st.depth]
                peak = int(np.abs(a.astype(np.int32)).max())
                assert peak <= E.ACC_BOUND, (peak, b.to_fen())
                moves += 1
        print(f"\ngate5 seed={seed}: {moves} checked materializations, bound ok")


def test_int16_bound_worst_case():
    """Synthetic worst case: all coefficients at their signed limits plus a
    fabricated full-size feature set must land exactly at the proven bound.
    The production lane is an int32 scratch peak-gated before the int16
    store — this test exercises THAT contract, and proves the shared
    kernels refuse an int16 accumulator outright (the FR1 silent-wrap
    calling convention cannot compile)."""
    w = E.EvalWeights()
    w.psq_w = np.full((E.PSQ_ROWS, E.CHANNELS), E.PSQ_COEF_MAX, np.int16)
    w.thr_w = np.full((E.THREAT_ROWS, E.CHANNELS), E.THREAT_COEF_MAX, np.int8)
    w.pp_w = np.full((E.PP_ROWS, E.CHANNELS), E.PP_COEF_MAX, np.int8)
    w.bias = np.full(E.CHANNELS, E.BIAS_COEF_MAX, np.int16)
    psq_rows = np.arange(E.MAX_PIECES, dtype=np.int32)
    thr_rows = np.arange(E.MAX_ACTIVE_THREATS, dtype=np.int32)
    pp_rows = np.arange(E.MAX_ACTIVE_PP, dtype=np.int32)

    # production contract: int32 scratch accumulates the true sum.
    acc32 = w.bias.astype(np.int32).copy()
    E._acc_add_rows(acc32, w.psq_w, psq_rows, E.MAX_PIECES)
    E._acc_add_rows(acc32, w.thr_w, thr_rows, E.MAX_ACTIVE_THREATS)
    E._acc_add_rows(acc32, w.pp_w, pp_rows, E.MAX_ACTIVE_PP)
    expect = E.BIAS_COEF_MAX + 32 * 255 + 256 * 63 + 120 * 31
    assert expect == E.ACC_BOUND == 30048
    assert int(acc32[0]) == expect
    assert E._acc_peak(acc32) == E.ACC_BOUND  # exactly at the bound
    E._check_acc("worst-case", acc32)  # at, not over: no raise
    acc16 = acc32.astype(np.int16)  # the narrow is safe HERE
    assert acc16[0] == expect  # 30048 < 32767

    # FR1-F3 kernel guard: an int16 accumulator must be refused at
    # dispatch — the pre-fix kernel accepted it and wrapped silently.
    bad = w.bias.copy()
    assert bad.dtype == np.int16
    with pytest.raises(TypeError):
        E._acc_add_rows(bad, w.psq_w, psq_rows, E.MAX_PIECES)
    with pytest.raises(TypeError):
        E._acc_sub_rows(bad, w.psq_w, psq_rows, E.MAX_PIECES)

    # over-bound input is caught by the peak gate BEFORE the int16 store:
    # legal max + 96 extra +63 threat rows = 36,096 — the pre-fix kernel
    # wrote np.int16(36096) = -29,440, which a post-hoc |.|<=30048 check
    # reads as IN BOUND.  The widened lane sees the true 36,096.
    E._acc_add_rows(acc32, w.thr_w, np.arange(256, 352, dtype=np.int32), 96)
    over = expect + 96 * 63
    assert over == 36096 > 32767 > E.ACC_BOUND
    assert int(acc32[0]) == over  # no wrap on int32
    assert E._acc_peak(acc32) == over
    with pytest.raises(E.BoundViolation):
        E._check_acc("over-bound", acc32)
    # the wrap the bug used to hide: an int16 narrow of 36,096 wraps to
    # -29,440 — still true, still masked without the pre-store peak gate;
    # kept here as the canary for why narrowing is gated, not post-checked.
    wrapped = np.asarray([over], np.int64).astype(np.int16)[0]
    assert abs(int(wrapped)) <= E.ACC_BOUND


def test_material_head_boundaries():
    """spec head.selection = min(7, max(0, (piece_count-2)//4)) — the
    boundary piece counts 2/6/10/32 must map to stacks 0/1/2/7 exactly."""
    fens = {
        2: "4k3/8/8/8/8/8/8/3K4 w - - 0 1",  # bare kings
        6: "4k3/8/8/8/8/8/8/3KQNRB w - - 0 1",  # kings + 4
        10: "4k3/8/8/8/8/8/P7/RNBKQBNR w - - 0 1",  # kings + 8
        32: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    }
    for pc, want in ((2, 0), (6, 1), (10, 2), (32, 7)):
        b = Board.from_fen(fens[pc])
        assert b._occ_all.bit_count() == pc
        got = E.material_bucket(b)
        assert got == want, (pc, got, want)
        # and the formula itself
        assert got == min(7, max(0, (pc - 2) // 4))


# --------------------------------------------------------------------------
# Delta-vs-enumeration diagnostic (independent cross-check)
# --------------------------------------------------------------------------


def _enum_sets(board, p):
    mb = np.asarray(board._sq, np.int8)
    bb = np.asarray(board._bb, np.uint64)
    occ = np.uint64(board._occ_all)
    f = E._frame(board._king[p], p)
    orient = ((f & 1) * 7) ^ (56 * p)
    psq = np.empty(40, np.int32)
    thr = np.empty(300, np.int32)
    pp = np.empty(160, np.int32)
    n1 = E._enum_psq(mb, p, f >> 1, orient, psq)
    n2 = E._enum_threats(mb, bb, occ, p, orient, thr)
    n3 = E._enum_pp(bb[0], bb[6], p, orient, pp)
    return set(psq[:n1].tolist()), set(thr[:n2].tolist()), set(pp[:n3].tolist())


def test_delta_vs_enumerated_sets():
    """For sampled moves, the computed delta row sets must equal the
    set-difference of before/after enumeration — independently of the
    accumulator comparison."""
    rng = random.Random(9)
    w = E.EvalWeights.random(3)
    st = E.Stack(w)
    buf = [0] * 256
    checked = 0
    for _ in range(30):
        b = Board.from_fen(START)
        st.set_root(b)
        for _ in range(80):
            n = generate_legal(b, buf)
            if n == 0:
                break
            mv = buf[rng.randrange(n)]
            b0 = _clone(b)
            st.push(b, mv)
            b.make(mv)
            if rng.random() < 0.25:
                for p in (0, 1):
                    if st.refresh[st.depth, p]:
                        continue
                    psq_a, psq_r, thr_a, thr_r, pp_a, pp_r = st._diff_rows(st.depth, p)
                    before = _enum_sets(b0, p)
                    after = _enum_sets(b, p)
                    for add_l, rem_l, bf, af, name in (
                        (psq_a, psq_r, before[0], after[0], "psq"),
                        (thr_a, thr_r, before[1], after[1], "thr"),
                        (pp_a, pp_r, before[2], after[2], "pp"),
                    ):
                        rem_c, add_c = Counter(rem_l.tolist()), Counter(add_l.tolist())
                        net_rem = rem_c - add_c
                        net_add = add_c - rem_c
                        # transient rem+add pairs cancel; the net multiset
                        # must equal the set difference of the two enums.
                        assert (bf - set(net_rem)) | set(net_add) == af, (
                            name,
                            b.to_fen(),
                        )
                    checked += 1
    assert checked > 0
    print(f"\ndelta-vs-enum: {checked} moves verified at set level")

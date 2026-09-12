"""Deduplication and grouped split assignment (spec section 9).

Split units are *clusters*, never bare records.  A cluster is the
union-find closure of:

* ``source.split_group`` (the split group the producer assigned — for
  trajectory data this is the chain/game id) AND
  ``source.original_game_id`` (both are unioned; neither may shadow the
  other),
* ``source.original_position_id`` equality (relabel twins),
* parent/child links (``observations[].parent_record_id`` and
  ``source.derivation_chain`` ``parent_record_id`` entries),
* the same (canonical board key, legal-context key) — the conservative
  near-duplicate merge.  Organic transpositions into an identical
  position+context are deliberately treated as one cluster: this is the
  conservative reading of "near-duplicate clusters ... get conservative
  grouped splits".

Deduplication happens BEFORE split assignment, in two passes:

1. exact duplicates (same board+context+label signature) are dropped to
   one representative;
2. ``dedup_records`` merges every record sharing a canonical board key
   (``position.fen4`` + ``position.variant``) into ONE record carrying the
   union of all observations.  This is the cross-source dedup the spec
   requires: e.g. the xushawn BT4 ``.q`` relabel and the linrock original
   are the same record stream with two label columns — they must appear
   once, not as two "independent" families.  ``DedupReport`` records the
   measured per-family and pairwise overlap before and after the merge.

Assignment is deterministic: ``sha256(cluster_key | split_salt)`` mapped
onto train/development/sealed/quarantine fractions, stratified per
``lineage_family`` so each family contributes proportionally.  With
``family_isolated=True`` whole families are assigned as units (for sealed
holdouts of entire opening families).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


def canonical_board_key(rec: dict) -> tuple[str, str]:
    """Canonical BOARD identity of a record: (fen4, variant).

    This is the dedup/overlap key — counters, history flags and labels
    are *context*, not position identity.
    """
    pos = rec["position"]
    return (pos["fen4"], pos["variant"])


def board_context_key(rec: dict) -> tuple[str, str]:
    pos = rec["position"]
    return (
        pos["fen4"],
        f"hm={pos.get('halfmove_clock')}|fm={pos.get('fullmove_number')}"
        f"|hc={int(pos['history_complete'])}|up={int(pos['unknown_prefix'])}",
    )


def label_signature(rec: dict) -> str:
    """Dedup signature over observation payloads (kind + score content)."""
    parts = []
    for o in rec["observations"]:
        parts.append(
            json.dumps(
                {
                    "k": o["kind"],
                    "p": o["perspective"],
                    "sk": o["score_kind"],
                    "bk": o["bound_kind"],
                    "cp": o.get("cp"),
                    "m": o.get("mate_plies"),
                    "w": o.get("wdl"),
                    "e": o.get("expected_score"),
                    "rp": "bytes"
                    if isinstance(o.get("raw_payload"), bytes)
                    else o.get("raw_payload"),
                },
                sort_keys=True,
                default=str,
            )
        )
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _obs_signature(o: dict) -> str:
    return json.dumps(o, sort_keys=True, default=str)


def family_board_overlap(records: list[dict]) -> dict:
    """Pairwise shared canonical board keys between declared families."""
    fam_keys: dict[str, set] = {}
    for r in records:
        fam_keys.setdefault(r["source"]["lineage_family"], set()).add(canonical_board_key(r))
    fams = sorted(fam_keys)
    out: dict[str, int] = {}
    for i, a in enumerate(fams):
        for b in fams[i + 1 :]:
            n = len(fam_keys[a] & fam_keys[b])
            if n:
                out[f"{a} ∩ {b}"] = n
    return out


def dedup_records(records: list[dict]) -> tuple[list[dict], dict, list[str]]:
    """Exact-dup drop + canonical-board-key merge, BEFORE split assignment.

    Returns ``(kept, report, dropped_exact_ids)``.

    Records sharing ``(fen4, variant)`` are merged into the first-seen
    representative: its observation list absorbs every distinct
    observation from the other members, and each absorbed member's
    provenance is appended to the representative's ``derivation_chain``.
    Returns ``(kept, report)`` where report carries per-family
    pre/post counts and the measured pairwise overlap matrix.
    """
    # ---- pass 0: exact duplicates -------------------------------------
    seen_sig: dict[tuple, dict] = {}
    uniq: list[dict] = []
    dropped_exact: list[str] = []
    for r in records:
        sig = (board_context_key(r), label_signature(r))
        if sig in seen_sig:
            # identical record dropped — but its SOURCE identity must ride
            # on the survivor's derivation_chain so game-level split
            # closure and assert_no_leakage still see it (the absorbed-
            # game straddle applies to exact-dup drops too).
            survivor = seen_sig[sig]
            survivor["source"].setdefault("derivation_chain", []).append(
                {
                    "step": "exact_duplicate_drop",
                    "absorbed": [
                        {
                            "record_id": r["record_id"],
                            "corpus": r["source"]["corpus"],
                            "lineage_family": r["source"]["lineage_family"],
                            "original_game_id": r["source"].get("original_game_id"),
                            "original_position_id": r["source"].get("original_position_id"),
                            "split_group": r["source"].get("split_group"),
                        }
                    ],
                }
            )
            dropped_exact.append(r["record_id"])
            continue
        seen_sig[sig] = r
        uniq.append(r)

    # ---- pass 1: canonical board-key merge -----------------------------
    fams_in: dict[str, int] = {}
    keys_in: dict[str, set] = {}
    for r in records:
        fams_in[r["source"]["lineage_family"]] = fams_in.get(r["source"]["lineage_family"], 0) + 1
    for r in uniq:
        keys_in.setdefault(r["source"]["lineage_family"], set()).add(canonical_board_key(r))
    overlap_pre = family_board_overlap(uniq)

    by_key: dict[tuple, dict] = {}
    order: list[tuple] = []
    absorbed: dict[tuple, list[dict]] = {}
    for r in uniq:
        k = canonical_board_key(r)
        if k not in by_key:
            by_key[k] = r
            order.append(k)
            absorbed[k] = []
            continue
        absorbed[k].append(r)

    kept: list[dict] = []
    for k in order:
        rep = by_key[k]
        members = absorbed[k]
        if members:
            rep_sigs = {_obs_signature(o) for o in rep["observations"]}
            trail = []
            for m in members:
                for o in m["observations"]:
                    s = _obs_signature(o)
                    if s not in rep_sigs:
                        rep["observations"].append(o)
                        rep_sigs.add(s)
                trail.append(
                    {
                        "record_id": m["record_id"],
                        "corpus": m["source"]["corpus"],
                        "lineage_family": m["source"]["lineage_family"],
                        "original_game_id": m["source"].get("original_game_id"),
                        "original_position_id": m["source"].get("original_position_id"),
                        "split_group": m["source"].get("split_group"),
                    }
                )
                # absorbed-of-absorbed: an exact-dup victim's identity
                # rides on this member's chain — flatten it onto the
                # survivor so kept-set audits see the full identity set
                for d in m["source"].get("derivation_chain", []):
                    for a in d.get("absorbed", []):
                        trail.append(dict(a))
            merge_step = {
                "step": "cross_source_boardkey_merge",
                "absorbed": trail,
            }
            rep["source"].setdefault("derivation_chain", []).append(merge_step)
            # Contradictory-outcome policy (FIX-DATA-2, decided):
            # a merged position reached in games with DIFFERENT outcome
            # classes (win/draw/loss in stm u-space) cannot honestly carry
            # a single result label — the whole game_outcome channel is
            # declared unresolvable and dropped; searched/teacher
            # observations are unaffected.  If that empties the record,
            # a censored marker keeps it schema-valid and auditable.
            out_i = [i for i, o in enumerate(rep["observations"]) if o["kind"] == "game_outcome"]
            if len(out_i) > 1:
                from training.labels import resolve_observation
                from training.records import make_observation

                stm_w = rep["position"]["fen4"].split(" ")[1] == "w"
                classes = set()
                for i in out_i:
                    try:
                        t = resolve_observation(rep["observations"][i], stm_w)
                    except Exception:
                        # a poisoned/malformed outcome obs can't attest a
                        # class — mark it "err" so the channel still drops
                        classes.add("err")
                        continue
                    classes.add(
                        None
                        if t is None or t.u is None
                        else (1 if t.u > 0.6 else -1 if t.u < 0.4 else 0)
                    )
                if len(classes) > 1:
                    rep["observations"] = [
                        o for i, o in enumerate(rep["observations"]) if i not in out_i
                    ]
                    merge_step["dropped_contradictory_outcomes"] = len(out_i)
                    merge_step["contradictory_classes"] = sorted(str(c) for c in classes)
                    if not rep["observations"]:
                        rep["observations"] = [
                            make_observation(
                                kind="game_outcome",
                                perspective="white",
                                score_kind="operation_result",
                                bound_kind="unknown",
                                context_known=True,
                                interrupted=True,
                                right_censored=True,
                                raw_payload={
                                    "status": "contradictory_outcomes_unresolvable",
                                    "absorbed_outcome_classes": sorted(str(c) for c in classes),
                                },
                            )
                        ]
            if rep.get("split") != "quarantine":
                rep["split"] = None
        kept.append(rep)

    overlap_post = family_board_overlap(kept)
    fams_out: dict[str, int] = {}
    for r in kept:
        fam = r["source"]["lineage_family"]
        fams_out[fam] = fams_out.get(fam, 0) + 1

    report = {
        "records_in": len(records),
        "exact_duplicates_dropped": len(dropped_exact),
        "records_after_exact_dedup": len(uniq),
        "unique_parent_positions": len(kept),
        "records_absorbed_by_merge": len(uniq) - len(kept),
        "multi_member_groups": sum(1 for v in absorbed.values() if v),
        "per_family": {
            fam: {
                "records_in": fams_in[fam],
                "unique_board_keys_in_source": len(keys_in.get(fam, ())),
                "records_after_merge": fams_out.get(fam, 0),
            }
            for fam in sorted(fams_in)
        },
        "pairwise_shared_board_keys_pre_dedup": overlap_pre,
        "pairwise_shared_board_keys_post_dedup": overlap_post,
    }
    if overlap_post:
        raise AssertionError(f"cross-family board-key overlap survived dedup: {overlap_post}")
    return kept, report, dropped_exact


class _DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        while self.parent.setdefault(x, x) != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass
class SplitResult:
    record_split: dict[str, str]  # record_id -> split name
    clusters: dict[str, list[str]]  # cluster_id -> member record_ids
    dropped_duplicates: list[str] = field(default_factory=list)
    kept: list[dict] = field(default_factory=list)
    dedup_report: dict = field(default_factory=dict)


def assign_splits(
    records: list[dict],
    *,
    fractions: dict[str, float] | None = None,
    salt: str = "rx-final-split-v1",
    family_isolated: bool = False,
) -> SplitResult:
    fractions = fractions or {"train": 0.85, "development": 0.10, "sealed": 0.05}
    names = list(fractions)

    # 1. dedup BEFORE split assignment: exact duplicates dropped, then all
    #    records sharing a canonical board key merged into one parent
    #    (cross-source relabel twins are one position, not two families).
    kept, dedup_report, dropped = dedup_records(records)

    # 2. cluster via union-find
    dsu = _DSU()
    by_id = {r["record_id"]: r for r in kept}
    for r in kept:
        rid = r["record_id"]
        dsu.find(rid)
        # every record is reachable via its own rec: alias so that
        # parent_record_id links below join the parent's cluster
        dsu.union(rid, f"rec:{rid}")
        # split_group AND original_game_id both constrain the split —
        # neither may shadow the other
        group = r["source"].get("split_group")
        if group:
            dsu.union(rid, f"grp:{group}")
        gid = r["source"].get("original_game_id")
        if gid:
            dsu.union(rid, f"game:{gid}")
        dsu.union(rid, f"opid:{r['source']['original_position_id']}")
        for o in r["observations"]:
            pr = o.get("parent_record_id")
            if pr:
                dsu.union(rid, f"rec:{pr}")
        for d in r["source"].get("derivation_chain", []):
            pr = d.get("parent_record_id")
            if pr:
                dsu.union(rid, f"rec:{pr}")
            # absorbed members' identities constrain this cluster exactly
            # as if the absorbed record were still present — otherwise the
            # absorbed game's other positions straddle the split (the R17
            # straddle: survivor holds GAME-A's key, GAME-B's leftovers
            # land on the other side).
            for a in d.get("absorbed", []):
                if a.get("record_id"):
                    dsu.union(rid, f"rec:{a['record_id']}")
                if a.get("split_group"):
                    dsu.union(rid, f"grp:{a['split_group']}")
                if a.get("original_game_id"):
                    dsu.union(rid, f"game:{a['original_game_id']}")
                if a.get("original_position_id"):
                    dsu.union(rid, f"opid:{a['original_position_id']}")
    by_posctx: dict[tuple, str] = {}
    for r in kept:
        key = board_context_key(r)
        if key in by_posctx:
            dsu.union(r["record_id"], by_posctx[key])
        else:
            by_posctx[key] = r["record_id"]

    clusters: dict[str, list[str]] = {}
    for r in kept:
        clusters.setdefault(dsu.find(r["record_id"]), []).append(r["record_id"])

    # 3. deterministic stratified assignment
    def cluster_split(cid: str, family: str) -> str:
        h = hashlib.sha256(f"{salt}|{family}|{cid}".encode()).hexdigest()
        x = int(h[:16], 16) / float(1 << 64)
        acc = 0.0
        for name in names:
            acc += fractions[name]
            if x < acc:
                return name
        return names[-1]

    record_split: dict[str, str] = {}
    if family_isolated:
        fam_cluster: dict[str, list[str]] = {}
        for cid, members in clusters.items():
            fam = by_id[members[0]]["source"]["lineage_family"]
            fam_cluster.setdefault(fam, []).append(cid)
        for fam, cids in fam_cluster.items():
            h = hashlib.sha256(f"{salt}|family|{fam}".encode()).hexdigest()
            x = int(h[:16], 16) / float(1 << 64)
            acc, split = 0.0, names[-1]
            for name in names:
                acc += fractions[name]
                if x < acc:
                    split = name
                    break
            for cid in cids:
                for rid in clusters[cid]:
                    record_split[rid] = split
    else:
        for cid, members in sorted(clusters.items()):
            fam = by_id[members[0]]["source"]["lineage_family"]
            s = cluster_split(cid, fam)
            for rid in members:
                record_split[rid] = s

    for r in kept:
        # a producer may have already condemned this record (invalid fen,
        # sentinel provenance, bank-level rejection) — a downstream split
        # must never resurrect it
        if r.get("split") == "quarantine":
            record_split[r["record_id"]] = "quarantine"
        else:
            r["split"] = record_split[r["record_id"]]
    return SplitResult(
        record_split=record_split,
        clusters=clusters,
        dropped_duplicates=dropped,
        kept=kept,
        dedup_report=dedup_report,
    )


def assert_no_leakage(records: list[dict]) -> None:
    """Hard assertions for the split-leakage gate.

    Checked invariants:
    * every record has an assigned split;
    * no ``split_group`` or ``original_game_id`` spans two splits
      (game/trajectory integrity — the train/dev/test gate) — including
      the ids of records ABSORBED by the cross-source merge (an absorbed
      game's leftovers must not land on the other side of the split);
    * no board+context or ``original_position_id`` spans two splits;
    * no canonical board key is shared between two declared families
      (cross-source dedup must have run first);
    * a record reachable via ``parent_record_id`` links must share its
      parent's split when both are present.
    """
    split_of_group: dict[str, str] = {}
    posctx_split: dict[tuple, str] = {}
    boardkey_seen: dict[tuple, tuple[str, str]] = {}
    opid_split: dict[str, str] = {}
    split_of_rec: dict[str, str] = {}
    for r in records:
        s = r.get("split")
        assert s in SPLITS_VALUES, f"{r['record_id']} has no assigned split"
        split_of_rec[r["record_id"]] = s
        for key in (
            r["source"].get("split_group"),
            r["source"].get("original_game_id"),
        ):
            if key:
                prev = split_of_group.setdefault(key, s)
                assert prev == s, f"game/trajectory {key} spans splits {prev}/{s}"
        pc = board_context_key(r)
        prev = posctx_split.setdefault(pc, s)
        assert prev == s, f"near-duplicate position/context spans splits {prev}/{s}"
        opid = r["source"]["original_position_id"]
        prev = opid_split.setdefault(opid, s)
        assert prev == s, f"relabel twin {opid} spans splits {prev}/{s}"
        bk = canonical_board_key(r)
        fam = r["source"]["lineage_family"]
        prev_bk = boardkey_seen.setdefault(bk, (s, fam))
        assert prev_bk[1] == fam, (
            f"canonical board key {bk} shared across families "
            f"{prev_bk[1]!r}/{fam!r} — run dedup_records before splitting"
        )
        assert prev_bk[0] == s, f"canonical board key {bk} spans splits {prev_bk[0]}/{s}"
        # absorbed members inherit the survivor's cluster: their game,
        # split-group and position ids must all agree with that split —
        # the straddle case is an absorbed record whose game has records
        # on BOTH sides of the split boundary.
        for d in r["source"].get("derivation_chain", []):
            for a in d.get("absorbed", []):
                for key in (a.get("split_group"), a.get("original_game_id")):
                    if key:
                        prev = split_of_group.setdefault(key, s)
                        assert prev == s, f"absorbed game/trajectory {key} spans splits {prev}/{s}"
                aopid = a.get("original_position_id")
                if aopid:
                    prev = opid_split.setdefault(aopid, s)
                    assert prev == s, f"absorbed relabel twin {aopid} spans splits {prev}/{s}"
                if a.get("record_id"):
                    split_of_rec[a["record_id"]] = s
    # parent/child links must not straddle splits
    for r in records:
        s = r["split"]
        links = [o.get("parent_record_id") for o in r["observations"]]
        links += [d.get("parent_record_id") for d in r["source"].get("derivation_chain", [])]
        for pr in links:
            if pr and pr in split_of_rec:
                assert split_of_rec[pr] == s, (
                    f"parent/child link {pr} -> {r['record_id']} spans splits "
                    f"{split_of_rec[pr]}/{s}"
                )


SPLITS_VALUES = {"train", "development", "sealed", "quarantine"}

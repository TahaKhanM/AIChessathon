"""Fixed-depth reference-search benchmark with per-sample provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import math
import json
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from engine.board import move_to_uci
from engine.search import Searcher, simple_eval
from engine.state import GameState
from engine.tt import TranspositionTable

POSITIONS = {
    "opening": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "middlegame": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "endgame": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
}


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def source_digest() -> str:
    """Fingerprint the measured code even when the working tree is dirty."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = sorted((root / "engine").rglob("*.py"))
    paths += [root / "engine/clock_b/horizon_table.json", Path(__file__).resolve()]
    for path in paths:
        data = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def measure(depth: int, repeats: int) -> dict:
    """Use a fresh searcher for each sample, excluding initialization time."""
    Searcher(tt=TranspositionTable(8), eval_fn=simple_eval).search(
        GameState.from_fen(POSITIONS["opening"]), max_depth=1
    )
    samples = []
    for name, fen in POSITIONS.items():
        for repeat in range(repeats):
            searcher = Searcher(tt=TranspositionTable(8), eval_fn=simple_eval)
            result = searcher.search(GameState.from_fen(fen), max_depth=depth)
            samples.append(
                {
                    "position": name,
                    "fen": fen,
                    "repeat": repeat,
                    "depth": result.depth,
                    "move": move_to_uci(result.move),
                    "score": result.score,
                    "nodes": result.nodes,
                    "qnodes": result.qnodes,
                    "elapsed_ms": result.elapsed_ms,
                    "aborted": result.aborted,
                }
            )
    times = sorted(sample["elapsed_ms"] for sample in samples)
    # Nearest-rank percentile; no interpolation suggesting unsupported precision.
    status = git_value("status", "--porcelain")
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "revision": git_value("rev-parse", "HEAD"),
        "source_sha256": source_digest(),
        "dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": {
            name: importlib.metadata.version(name) for name in ("numpy", "numba", "chess")
        },
        "configuration": {
            "backend": "python",
            "evaluator": "classical",
            "tt_mib": 8,
            "depth": depth,
            "repeats": repeats,
            "initialization_included": False,
        },
        "summary": {
            "samples": len(samples),
            "median_ms": statistics.median(times),
            "p95_ms": times[math.ceil(0.95 * len(times)) - 1],
        },
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.depth <= 8 or not 1 <= args.repeats <= 100:
        parser.error("depth must be 1–8 and repeats must be 1–100")
    report = json.dumps(measure(args.depth, args.repeats), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()

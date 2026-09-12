"""Transactional checkpoints (spec 10.1).

Layout::

    <ckpt_root>/ckpt-<step>-<uuid>.tmp/     # files land here first
        params.npz        model params (float32)
        optim.npz         AdamW m/v + step
        state.json        rng, sampler, schemas, data manifest, revision
        MANIFEST.json     sha256 + bytes of every file; written LAST
    <ckpt_root>/ckpt-<step>-<uuid>/        # atomic rename on commit

A checkpoint only becomes visible after MANIFEST.json verifies every listed
file's sha256.  ``load_latest`` picks the highest-step *complete* checkpoint
and ignores partial .tmp dirs or manifests that fail verification, so an
interrupted job never replaces the last complete checkpoint.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
import uuid
from dataclasses import dataclass

import numpy as np


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CheckpointWriter:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def commit(
        self,
        *,
        step: int,
        files: dict[str, bytes],
        fail_after_files: int | None = None,
    ) -> str:
        """Write a checkpoint transactionally.

        ``fail_after_files`` exists ONLY for tests: it raises a synthetic
        mid-write crash after N files are durably written, leaving a partial
        .tmp directory behind.
        """
        name = f"ckpt-{step:08d}-{uuid.uuid4().hex[:8]}"
        tmp = os.path.join(self.root, name + ".tmp")
        os.makedirs(tmp, exist_ok=False)
        manifest: dict[str, dict] = {}
        written = 0
        for fname, blob in files.items():
            p = os.path.join(tmp, fname)
            with open(p, "wb") as fh:
                fh.write(blob)
                fh.flush()
                os.fsync(fh.fileno())
            manifest[fname] = {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
            written += 1
            if fail_after_files is not None and written >= fail_after_files:
                # synthetic crash: tmp dir left behind, no MANIFEST
                raise RuntimeError(f"synthetic mid-write crash after {written} files")
        mpath = os.path.join(tmp, "MANIFEST.json")
        with open(mpath, "wb") as fh:
            fh.write(
                json.dumps(
                    {"step": step, "files": manifest, "created_utc": time.time()}, sort_keys=True
                ).encode()
            )
            fh.flush()
            os.fsync(fh.fileno())
        final = os.path.join(self.root, name)
        os.rename(tmp, final)
        return final


@dataclass
class Checkpoint:
    step: int
    path: str
    manifest: dict

    def read_file(self, name: str) -> bytes:
        with open(os.path.join(self.path, name), "rb") as fh:
            data = fh.read()
        want = self.manifest["files"][name]
        if hashlib.sha256(data).hexdigest() != want["sha256"]:
            raise ValueError(f"checkpoint file {name} hash mismatch")
        return data

    def npz(self, name: str) -> dict[str, np.ndarray]:
        with np.load(io.BytesIO(self.read_file(name))) as z:
            return {k: z[k] for k in z.files}

    def state(self) -> dict:
        return json.loads(self.read_file("state.json"))


def _verify_dir(path: str) -> Checkpoint | None:
    mpath = os.path.join(path, "MANIFEST.json")
    if not os.path.exists(mpath):
        return None
    try:
        with open(mpath) as fh:
            manifest = json.load(fh)
        for fname, meta in manifest["files"].items():
            fp = os.path.join(path, fname)
            if not os.path.exists(fp):
                return None
            if os.path.getsize(fp) != meta["bytes"]:
                return None
            if _sha256_file(fp) != meta["sha256"]:
                return None
        return Checkpoint(step=manifest["step"], path=path, manifest=manifest)
    except Exception:
        return None


def load_latest(root: str) -> Checkpoint | None:
    """Return the highest-step *verified complete* checkpoint, or None."""
    if not os.path.isdir(root):
        return None
    best: Checkpoint | None = None
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if not os.path.isdir(full) or entry.endswith(".tmp"):
            continue
        ck = _verify_dir(full)
        if ck is not None and (best is None or ck.step > best.step):
            best = ck
    return best


def save_model_checkpoint(
    root: str,
    *,
    step: int,
    params: dict[str, np.ndarray],
    optim: dict[str, np.ndarray],
    state: dict,
    fail_after_files: int | None = None,
) -> str:
    """Convenience wrapper: params/optim npz + state.json + manifest."""
    pb = io.BytesIO()
    np.savez(pb, **params)
    ob = io.BytesIO()
    np.savez(ob, **optim, _step=np.int64(step))
    files = {
        "params.npz": pb.getvalue(),
        "optim.npz": ob.getvalue(),
        "state.json": json.dumps(state, sort_keys=True).encode(),
    }
    return CheckpointWriter(root).commit(step=step, files=files, fail_after_files=fail_after_files)

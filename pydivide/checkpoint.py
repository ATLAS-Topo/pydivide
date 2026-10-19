# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Stage checkpointing.

Every pipeline stage saves its outputs to disk when it completes, together
with a fingerprint of the parameters it ran with. On restart, a stage whose
checkpoint exists with a matching fingerprint is loaded instead of
recomputed. Changing a parameter invalidates that stage and everything after
it (each stage should include its inputs' fingerprints in its own params).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np


def _fingerprint(params: dict | None) -> str:
    payload = json.dumps(params or {}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def array_fingerprint(*arrays, chunk: int = 1 << 26) -> str:
    """The identity of large arrays: shape, dtype and a hash of every
    byte, computed in chunks without copying the array. blake2b runs at
    about a gigabyte per second, so the 80 GB of flow edges of a
    1.2 x 10^10-cell domain cost a minute or two once per run, which is
    the price of never confusing two routings; a sampled hash could be
    defeated by a change outside the sample."""
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        a = np.asarray(a)
        h.update(f"{a.shape}|{a.dtype}|".encode())
        flat = np.ascontiguousarray(a).reshape(-1).view(np.uint8)
        for s in range(0, flat.size, chunk):
            h.update(memoryview(flat[s:s + chunk]))
    return h.hexdigest()


_FP_KEY = "__fingerprint__"


class StageStore:
    """Directory of per-stage .npz checkpoints with a JSON manifest."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / "manifest.json"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
        else:
            self.manifest = {}

    def _write_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2))
        tmp.replace(self.manifest_path)

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.npz"

    def has(self, name: str, params: dict | None = None) -> bool:
        """True only when the manifest entry AND the fingerprint stored
        inside the checkpoint file both match ``params``. The file carries
        its own fingerprint so that a save interrupted between writing the
        payload and updating the manifest can never pass another run's
        data off as this one's."""
        entry = self.manifest.get(name)
        if entry is None or not self.path(name).exists():
            return False
        fp = _fingerprint(params)
        if entry["fingerprint"] != fp:
            return False
        try:
            with np.load(self.path(name)) as z:
                return _FP_KEY in z.files and str(z[_FP_KEY]) == fp
        except Exception:
            return False

    def save(self, name: str, arrays: dict[str, np.ndarray],
             params: dict | None = None) -> None:
        """Atomically write a stage checkpoint (tmp file then rename), with
        the fingerprint of ``params`` inside the file, then the manifest."""
        fp = _fingerprint(params)
        tmp = self.path(name).with_suffix(".npz.tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays, **{_FP_KEY: np.array(fp)})
        tmp.replace(self.path(name))
        self.manifest[name] = {
            "fingerprint": fp,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "arrays": {k: [list(v.shape), str(v.dtype)]
                       for k, v in arrays.items()},
        }
        self._write_manifest()

    def load(self, name: str) -> dict[str, np.ndarray]:
        with np.load(self.path(name)) as z:
            return {k: z[k] for k in z.files if k != _FP_KEY}

    def stage(self, name: str, compute: Callable[[], dict[str, np.ndarray]],
              params: dict | None = None,
              verbose: bool = True) -> dict[str, np.ndarray]:
        """Load ``name`` if checkpointed with matching params, else compute,
        save, and return."""
        if self.has(name, params):
            if verbose:
                print(f"[checkpoint] {name}: loading from "
                      f"{self.path(name).name}", flush=True)
            return self.load(name)
        if verbose:
            print(f"[checkpoint] {name}: computing", flush=True)
        t0 = time.time()
        arrays = compute()
        self.save(name, arrays, params)
        if verbose:
            print(f"[checkpoint] {name}: done in {time.time()-t0:.1f} s, "
                  f"saved to {self.path(name).name}", flush=True)
        return arrays

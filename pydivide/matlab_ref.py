# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Loader for MATLAB reference exports (matlab_scripts/export_reference.m).

Reads the flat binary files and converts everything to Python conventions:
zero-based row-major cell indices, zero-based row-major node ids in the
(nr+1) x (nc+1) node grid (see pydivide.gridspec). NaN segment separators
in the nan-lists are preserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .gridspec import GridSpec, matlab_node_to_py


class MatlabReference:
    """All reference products for one DEM/threshold combination."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        man = json.loads((self.dir / "manifest.json").read_text())
        self.manifest = man
        self.nr = int(man["nr"])
        self.nc = int(man["nc"])
        self.cellsize = float(man["cellsize"])
        wf_cell = np.asarray(man["wf_cell"], dtype=np.float64)
        self.grid = GridSpec.from_matlab_wf(self.nr, self.nc,
                                            self.cellsize, wf_cell)
        self.wf_node = np.asarray(man["wf_node"], dtype=np.float64)

    # ---- raw readers ------------------------------------------------------

    def _bin(self, name: str, dtype, cols: int | None = None) -> np.ndarray:
        a = np.fromfile(self.dir / name, dtype=dtype)
        if cols is not None:
            a = a.reshape(-1, cols)
        return a

    # ---- cell-indexed products -------------------------------------------

    def fd_edges_rc(self) -> np.ndarray:
        """Full flow network as int64 (n, 4): r1, c1, r2, c2 zero-based."""
        return self._bin("fd_edges.bin", np.int64, 4) - 1

    def st_edges_rc(self) -> np.ndarray:
        """Stream flow edges as int64 (n, 4): zero-based rc pairs."""
        return self._bin("st_edges.bin", np.int64, 4) - 1

    def st_nodes_rc(self) -> np.ndarray:
        """Stream node cells as int64 (n, 2), zero-based."""
        return self._bin("st_nodes.bin", np.int64, 2) - 1

    def outlets(self) -> tuple[np.ndarray, np.ndarray]:
        """(outlet rc int64 (n,2) zero-based, reach label of each outlet)."""
        a = self._bin("outlets.bin", np.int64, 3)
        return a[:, :2] - 1, a[:, 2]

    def labelreach_nal(self) -> np.ndarray:
        return self._bin("labelreach_nal.bin", np.int64)

    def L_grid(self) -> np.ndarray:
        """Basin label raster, float64 (nr, nc), NaN where unlabelled."""
        return self._bin("L_grid.bin", np.float64).reshape(self.nr, self.nc)

    def FX_grid(self) -> np.ndarray:
        """Diagonal-crossing grid, int8 (nr+1, nc+1)."""
        return self._bin("FX_grid.bin", np.int8).reshape(self.nr + 1,
                                                         self.nc + 1)

    # ---- node-indexed products (converted to Python node ids) -------------

    def _nanlist(self, name: str) -> np.ndarray:
        raw = self._bin(name, np.float64)
        return matlab_node_to_py(raw, self.nr, self.nc)

    def D0_IX(self) -> np.ndarray:
        """Raw divide geometry (constructor, network=false), nan-list of
        Python node ids as float64."""
        return self._nanlist("D0_IX.bin")

    def D_IX(self) -> np.ndarray:
        """Final sorted network nan-list of Python node ids (float64)."""
        return self._nanlist("D_IX.bin")

    def D_ep(self) -> np.ndarray:
        raw = self._bin("D_ep.bin", np.int64)
        return matlab_node_to_py(raw, self.nr, self.nc)

    def D_jct(self) -> np.ndarray:
        raw = self._bin("D_jct.bin", np.int64)
        return matlab_node_to_py(raw, self.nr, self.nc)

    def D_jctedg(self) -> np.ndarray:
        return self._bin("D_jctedg.bin", np.int64)

    def D_distance(self) -> np.ndarray:
        """Distance aligned with D_IX (float64, NaN separators)."""
        return self._bin("D_distance.bin", np.float64)

    def order(self, scheme: str) -> np.ndarray:
        """Order nan-list for scheme in {'strahler','shreve','topo'}."""
        return self._bin(f"order_{scheme}.bin", np.float64)

    def coordcheck(self) -> np.ndarray:
        """(n, 3): MATLAB node linear index (1-based), x, y."""
        return self._bin("coordcheck.bin", np.float64, 3)

    def T0_edges(self) -> np.ndarray:
        """Pre-drop constructor edge list (matlab_scripts/export_T0_nodrop.m):
        the MATLAB edge set *before* the order-dependent single-edge-run
        drop. Unordered unique int64 (n, 2) in Python node ids."""
        raw = self._bin("T0_nodrop.bin", np.float64, 2)
        conv = matlab_node_to_py(raw.ravel(), self.nr, self.nc).reshape(-1, 2)
        ok = ~np.isnan(conv).any(axis=1)
        e = conv[ok].astype(np.int64)
        e.sort(axis=1)
        return np.unique(e, axis=0)

    # ---- derived: edge set of a nan-list ----------------------------------

    @staticmethod
    def nanlist_edges(ix: np.ndarray) -> np.ndarray:
        """Unordered unique edge set of a nan-list: int64 (n, 2), each row
        sorted (a < b), rows unique and lexicographically sorted."""
        a = ix[:-1]
        b = ix[1:]
        ok = ~(np.isnan(a) | np.isnan(b))
        e = np.stack([a[ok], b[ok]], axis=1).astype(np.int64)
        e.sort(axis=1)
        e = np.unique(e, axis=0)
        return e

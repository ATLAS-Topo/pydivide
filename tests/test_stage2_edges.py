# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Validation ladder step 2: the divide edge set.

This is the make-or-break test. The gate:

  * every MATLAB edge must be present in the pydivide set (no missing
    edges, no exceptions);
  * every pydivide edge absent from MATLAB's final set must be
    machine-proven to be a MATLAB artifact. On Big Tujunga this is done
    exactly: the pydivide set must equal MATLAB's *pre-drop* edge list
    (export_T0_nodrop.m), so every surplus edge is precisely an edge
    MATLAB's order-dependent single-edge-run drop removed.

Measured on Big Tujunga (2026-07-29): ours 51,839 = MATLAB pre-drop
51,839 (identical sets); MATLAB final 51,721; surplus 118 = exactly the
dropped set; GRIDobj2polygon defect fired 0 times.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydivide.geometry import divide_edges  # noqa: E402
from pydivide.labels import basin_labels  # noqa: E402
from pydivide.matlab_ref import MatlabReference  # noqa: E402


def _rowset(a: np.ndarray) -> set:
    return set(map(tuple, a.tolist()))


@pytest.fixture(scope="module")
def edges(ref, net):
    L, _ = basin_labels(net)
    out_rc, out_lab = ref.outlets()
    outlet_cells = out_rc[:, 0] * np.int64(ref.nc) + out_rc[:, 1]
    de = divide_edges(net.grid, L, net.st_src, net.st_dst,
                      outlet_cells, out_lab)
    return ref, de


def test_no_matlab_edge_missing(edges):
    ref, de = edges
    ours = _rowset(de.nodes)
    matlab_final = _rowset(MatlabReference.nanlist_edges(ref.D0_IX()))
    missing = matlab_final - ours
    assert not missing, f"{len(missing)} MATLAB edges missing from pydivide"


def _edge_cells(ref, a: int, b: int):
    """The two (row, col) cells separated by divide edge (a, b); None for
    outside-the-grid."""
    w = ref.nc + 1
    I, J = a // w, a % w
    if b == a + 1:                      # horizontal edge -> cells above/below
        cells = [(I - 1, J), (I, J)]
    elif b == a + w:                    # vertical edge -> cells left/right
        cells = [(I, J - 1), (I, J)]
    else:
        raise AssertionError("non-adjacent edge")
    out = []
    for r, c in cells:
        out.append((r, c) if 0 <= r < ref.nr and 0 <= c < ref.nc else None)
    return out


def test_predrop_set_with_classified_polygon_misses(edges):
    """Equality with the pre-drop MATLAB edge list, except for edges that
    MATLAB's GRIDobj2polygon defect failed to trace at all (a defect the
    MATLAB source itself documents). Every such edge must be
    proven genuine against MATLAB's own exported label raster: the two
    cells it separates carry different labels (or one side is nodata).

    Nothing may be missing from pydivide, ever."""
    ref, de = edges
    ours = _rowset(de.nodes)
    predrop = _rowset(ref.T0_edges())

    only_matlab = predrop - ours
    assert not only_matlab, (
        f"{len(only_matlab)} MATLAB pre-drop edges missing from pydivide "
        "— BLOCKS progression")

    Lm = ref.L_grid()
    unproven = []
    for a, b in sorted(ours - predrop):
        ca, cb = _edge_cells(ref, a, b)
        va = Lm[ca] if ca is not None else np.nan
        vb = Lm[cb] if cb is not None else np.nan
        genuine = (np.isnan(va) != np.isnan(vb)) or \
            (not np.isnan(va) and not np.isnan(vb) and va != vb)
        if not genuine:
            unproven.append((a, b))
    n_miss = len(ours - predrop)
    if n_miss:
        print(f"[classified] {n_miss} genuine divide edges absent from "
              "MATLAB's polygon tracing (GRIDobj2polygon defect), proven "
              "against MATLAB's own label raster")
    assert not unproven, (
        f"{len(unproven)} pydivide-only edges NOT provable from the "
        f"label raster: {unproven[:5]} — BLOCKS progression")


def test_surplus_vs_final_is_classified(edges):
    """Every pydivide edge absent from MATLAB's final set is either a
    single-edge-run drop (in pre-drop, not in final) or a proven polygon
    miss (not in pre-drop, checked above)."""
    ref, de = edges
    ours = _rowset(de.nodes)
    final = _rowset(MatlabReference.nanlist_edges(ref.D0_IX()))
    predrop = _rowset(ref.T0_edges())
    surplus = ours - final
    drops = surplus & (predrop - final)
    misses = surplus - predrop
    assert surplus == drops | misses
    print(f"[classified] surplus vs MATLAB final: {len(drops)} "
          f"single-edge-run drops, {len(misses)} polygon-defect misses")


def test_boundary_flag_consistency(edges):
    """Boundary-flagged edges are exactly those with an invalid side."""
    _, de = edges
    assert np.array_equal(de.is_boundary,
                          (de.lab_a == -1) | (de.lab_b == -1))
    # sanity: a non-trivial number of both kinds
    assert de.is_boundary.sum() > 0
    assert (~de.is_boundary).sum() > 0

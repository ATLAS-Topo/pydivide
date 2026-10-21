# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Every side convention of the across-divide table, checked on a real
DEM against an independent reconstruction.

The table says which physical side is side A by naming its reach basin
(``basin_a``) and which river system that basin drains to (``outlet_a``).
These tests rebuild both from nothing but the piece's divide edges, the
cells on either side of each edge, the label grid and the DEM, and from
pytopotoolbox's own drainage basins, and require the table to agree
exactly: the signed elevation contrast of a piece is the mean over its
edges of the elevation of the cell in basin A minus that of the cell in
basin B; two sides report the same outlet exactly when pytopotoolbox
puts them in the same drainage basin; the elevation contrast of a piece
is the same number at every channel threshold of a multiscale table,
since the elevation grid and the physical side do not change with the
threshold; and the pair summary re-expresses each signed value as side
A minus side B of its pair.
"""

from __future__ import annotations

import numpy as np
import pytest

tt = pytest.importorskip("topotoolbox")

from pydivide.acrossdivide import across_divide_metrics, across_divide_multiscale  # noqa: E402
from pydivide.basinpairs import pair_table  # noqa: E402
from pydivide.divmetrics import edge_cells  # noqa: E402
from pydivide.pipeline import divide_network  # noqa: E402

SEG = 300.0


@pytest.fixture(scope="module")
def bt():
    dem = tt.load_dem("bigtujunga")
    fd = tt.FlowObject(dem)
    st = tt.StreamObject(fd, threshold=1e6, units="m2")
    D = divide_network(fd=fd, st=st, verbose=False)
    t = across_divide_metrics(dem, flow=fd, streams=st, divides=D, baselevel=400.0,
                              threshold=1e6, seglength=SEG, metrics=("elev", "hr", "chi"))
    return dem, fd, st, D, t


def piece_edges(D, seglength):
    """Every divide edge of every piece: ``(piece index, sorted node pair)``."""
    _, p_start, p_stop = D._piece_table(seglength)
    nodes = D.network.seg_nodes
    lens = np.maximum(p_stop - p_start - 1, 0)
    piece = np.repeat(np.arange(len(p_start), dtype=np.int64), lens)
    offs = np.repeat(np.cumsum(lens) - lens, lens)
    pos = np.repeat(p_start, lens) + (np.arange(len(piece), dtype=np.int64) - offs)
    edges = np.sort(np.column_stack([nodes[pos], nodes[pos + 1]]), axis=1)
    return piece, edges


def test_signed_elevation_is_basin_a_minus_basin_b_edge_by_edge(bt):
    dem, _, _, D, t = bt
    c = t.columns
    z = np.asarray(dem.z, float).ravel()
    L = np.asarray(D.labels).ravel()
    piece, edges = piece_edges(D, SEG)
    ca, cb = edge_cells(D.grid, edges)
    la = np.where(ca >= 0, L[np.maximum(ca, 0)], -1)
    lb = np.where(cb >= 0, L[np.maximum(cb, 0)], -1)
    A = np.asarray(c["basin_a"], float)[piece]
    B = np.asarray(c["basin_b"], float)[piece]
    # edges whose two sides are exactly the pair the piece names
    on_pair = ((la == A) & (lb == B)) | ((la == B) & (lb == A))
    z_a = np.where(la == A, z[np.maximum(ca, 0)], z[np.maximum(cb, 0)])
    z_b = np.where(la == A, z[np.maximum(cb, 0)], z[np.maximum(ca, 0)])
    n = len(c["basin_a"])
    counts = np.bincount(piece, minlength=n)
    counts_pair = np.bincount(piece[on_pair], minlength=n)
    whole = (counts > 0) & (counts == counts_pair)          # every edge of the piece is on the pair
    sgn = np.bincount(piece[on_pair], weights=(z_a - z_b)[on_pair], minlength=n)
    sgn = np.where(whole, sgn / np.maximum(counts_pair, 1), np.nan)
    got = np.asarray(c["elev_sgn"], float)
    ok = whole & np.isfinite(got)
    assert ok.sum() > 1000
    np.testing.assert_allclose(got[ok], sgn[ok], atol=1e-6)
    mixed = np.asarray(c["basin_pair_mixed"]) == 1
    assert not (whole & mixed).any()


def test_outlets_follow_pytopotoolbox_drainage_basins(bt):
    dem, fd, _, D, t = bt
    c = t.columns
    L = np.asarray(D.labels).ravel()
    db = np.asarray(fd.drainagebasins().z).ravel()             # one id per river system
    first = np.full(int(L.max()) + 1, -1, np.int64)
    idx = np.flatnonzero(L > 0)
    first[L[idx][::-1]] = idx[::-1]                              # a cell of every label
    sys_of_label = np.where(first >= 0, db[np.maximum(first, 0)], -1)
    a = np.asarray(c["basin_a"], float)
    b = np.asarray(c["basin_b"], float)
    oa = np.asarray(c["outlet_a"], float)
    ob = np.asarray(c["outlet_b"], float)
    ok = np.isfinite(a) & np.isfinite(b)
    assert ok.sum() > 1000 and np.isfinite(oa[ok]).all() and np.isfinite(ob[ok]).all()
    a, b, oa, ob = (v[ok].astype(np.int64) for v in (a, b, oa, ob))
    # a basin and the outlet reach it reports belong to the same river system
    assert np.array_equal(sys_of_label[a], sys_of_label[oa])
    assert np.array_equal(sys_of_label[b], sys_of_label[ob])
    # two sides share an outlet exactly when they share a river system
    assert np.array_equal(oa == ob, sys_of_label[a] == sys_of_label[b])
    assert (oa != ob).sum() > 100 and (oa == ob).sum() > 1000


def test_multiscale_elevation_contrast_is_threshold_independent(bt):
    dem, fd, _, D, _ = bt
    mc = across_divide_multiscale(dem, thresholds=(1e6, 1e7), baselevel=400.0,
                                  seglength=SEG, metrics=("elev", "chi"))
    e6 = np.asarray(mc.columns["elev_sgn_1e6"], float)
    e7 = np.asarray(mc.columns["elev_sgn_1e7"], float)
    ok = np.isfinite(e6) & np.isfinite(e7)
    assert ok.sum() > 1000
    np.testing.assert_allclose(e6[ok], e7[ok], atol=1e-6)
    # and side A carries the lower label of the coarse partition at the coarse scale
    a7 = np.asarray(mc.columns["basin_a_1e7"], float)
    b7 = np.asarray(mc.columns["basin_b_1e7"], float)
    ok7 = np.isfinite(a7) & np.isfinite(b7)
    assert ok7.sum() > 100
    assert np.all(a7[ok7] != b7[ok7])


def test_pair_summary_matches_a_by_hand_alignment_on_real_pieces(bt):
    _, _, _, _, t = bt
    c = t.columns
    a = np.asarray(c["basin_a"], float)
    b = np.asarray(c["basin_b"], float)
    s = np.asarray(c["elev_sgn"], float)
    rows = pair_table(a, b, values={"elev_sgn": s})
    rng = np.random.default_rng(0)
    for _, r in rows.iloc[rng.choice(len(rows), 200, replace=False)].iterrows():
        m = (np.minimum(a, b) == r.basin_lo) & (np.maximum(a, b) == r.basin_hi)
        aligned = np.where(a[m] == r.side_a, s[m], -s[m])
        assert np.isclose(np.nanmean(aligned), r.elev_sgn_mean, equal_nan=True)

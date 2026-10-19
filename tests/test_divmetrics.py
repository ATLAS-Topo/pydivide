# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Across-divide sampling, side orientation and asymmetry.

The synthetic network is a single divide segment that steps diagonally
from the north-west to the south-east of a 6 x 6 grid, separating basin 1
(cells with column > row) from basin 2. That orientation is the one on
which the geometric side order (west cell of a vertical face, north cell
of a horizontal one) switches basin at every corner; the tests pin the
flaw, the repair, and the MATLAB `asymmetry` semantics by hand-computed
numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydivide import divmetrics as DM  # noqa: E402
from pydivide.gridspec import GridSpec  # noqa: E402
from pydivide.orderdist import SortedNetwork  # noqa: E402

NR = NC = 6
W = NC + 1


def _grid() -> GridSpec:
    return GridSpec(nr=NR, nc=NC, cellsize=1.0, x0=0.0, y0=float(NR))


def _network(node_rc: list[tuple[int, int]]) -> SortedNetwork:
    ids = np.array([W * i + j for i, j in node_rc], dtype=np.int64)
    return SortedNetwork(seg_nodes=ids,
                         seg_ptr=np.array([0, len(ids)], dtype=np.int64),
                         is_ring=np.zeros(1, dtype=bool),
                         endpoints=np.array([ids[0], ids[-1]]),
                         junctions=np.zeros(0, dtype=np.int64))


def _diagonal():
    """NW-SE staircase between basin 1 (J > I) and basin 2 (J <= I)."""
    nodes = []
    for I in range(5):
        nodes.append((I, I + 1))
        nodes.append((I + 1, I + 1))
    nodes.append((5, 6))
    sn = _network(nodes)
    L = np.where(np.arange(NC)[None, :] > np.arange(NR)[:, None], 1, 2)
    return sn, L.astype(np.int64)


def _values(L: np.ndarray) -> np.ndarray:
    return np.where(L == 1, 10.0, 2.0).ravel()


# ---------------------------------------------------------------------------
# the flaw and the repair

def test_geometric_side_switches_basin_along_a_nw_se_staircase():
    sn, L = _diagonal()
    pos, edges = DM.edge_positions(sn)
    assert len(pos) == 10
    ca, cb = DM.edge_cells(_grid(), edges)
    assert np.all(ca >= 0) and np.all(cb >= 0)
    lab_a = L.ravel()[ca]
    # west cell of a vertical face is basin 2, north cell of the next
    # horizontal face is basin 1: the "first" side alternates
    assert lab_a.tolist() == [2, 1] * 5


def test_orient_by_basin_gives_one_side_per_segment():
    sn, L = _diagonal()
    _, edges = DM.edge_positions(sn)
    ca, cb = DM.edge_cells(_grid(), edges)
    ca2, cb2, la, lb = DM.orient_by_basin(ca, cb, L.ravel())
    assert np.all(la == 1) and np.all(lb == 2)
    assert np.all(L.ravel()[ca2] == 1) and np.all(L.ravel()[cb2] == 2)
    # every cell pair is the same pair, just ordered
    assert set(map(tuple, np.sort(np.stack([ca, cb], 1), 1))) == \
        set(map(tuple, np.sort(np.stack([ca2, cb2], 1), 1)))


def test_signed_pair_field_cancels_geometrically_and_holds_when_oriented():
    sn, L = _diagonal()
    _, edges = DM.edge_positions(sn)
    ca, cb = DM.edge_cells(_grid(), edges)
    v = _values(L)
    p, q = DM.sample_pair(v, ca, cb)
    sgn_geo = DM.pair_metric_fields(p, q)["sgn"]
    assert sgn_geo.tolist() == [-8.0, 8.0] * 5
    assert abs(np.mean(sgn_geo)) < 1e-12          # the segment mean cancels

    ca2, cb2, _, _ = DM.orient_by_basin(ca, cb, L.ravel())
    p2, q2 = DM.sample_pair(v, ca2, cb2)
    f = DM.pair_metric_fields(p2, q2)
    assert np.all(f["sgn"] == 8.0)                # lower label (1) minus 2
    for k in ("mean", "diff", "min", "max"):      # unsigned fields unchanged
        assert np.allclose(f[k], DM.pair_metric_fields(p, q)[k])


def test_orientation_leaves_invalid_sides_alone():
    ca = np.array([5, -1, 7], dtype=np.int64)
    cb = np.array([6, 3, -1], dtype=np.int64)
    L = np.arange(10, dtype=np.int64)[::-1]       # label decreases with id
    ca2, cb2, la, lb = DM.orient_by_basin(ca, cb, L)
    assert (ca2[0], cb2[0]) == (6, 5)             # swapped: label 3 < 4
    assert (ca2[1], cb2[1]) == (-1, 3) and (la[1], lb[1]) == (-1, 6)
    assert (ca2[2], cb2[2]) == (7, -1)


# ---------------------------------------------------------------------------
# getvalue

def test_getvalue_pairs_and_aggregations():
    sn, L = _diagonal()
    v = _values(L)
    p, q = DM.getvalue(sn, _grid(), v, label_flat=L.ravel())
    assert np.all(p == 10.0) and np.all(q == 2.0)
    assert np.all(DM.getvalue(sn, _grid(), v, "diff") == 8.0)
    assert np.allclose(DM.getvalue(sn, _grid(), v, "normdiff"), 8.0 / 6.0)
    assert np.all(DM.getvalue(sn, _grid(), v, "mean") == 6.0)
    assert np.all(DM.getvalue(sn, _grid(), v, "max") == 10.0)
    assert np.all(DM.getvalue(sn, _grid(), v, "min") == 2.0)


def test_getvalue_without_labels_uses_matlab_top_right_side():
    sn, L = _diagonal()
    v = _values(L)
    p, q = DM.getvalue(sn, _grid(), v)
    # edge 0 is vertical: right (east) cell is basin 1; edge 1 is
    # horizontal: top (north) cell is basin 1 as well
    assert p[0] == 10.0 and q[0] == 2.0
    assert p[1] == 10.0 and q[1] == 2.0


def test_getvalue_mean_tolerates_one_missing_side():
    sn, L = _diagonal()
    v = _values(L)
    v[L.ravel() == 2] = np.nan
    assert np.all(DM.getvalue(sn, _grid(), v, "mean") == 10.0)
    assert np.all(np.isnan(DM.getvalue(sn, _grid(), v, "diff")))


def test_getvalue_rejects_unknown_fct():
    sn, L = _diagonal()
    with pytest.raises(ValueError):
        DM.getvalue(sn, _grid(), _values(L), "median")


# ---------------------------------------------------------------------------
# asymmetry, basin method (hand-computed)

def test_asymmetry_basin_hand_computation():
    sn, L = _diagonal()
    a = DM.asymmetry(sn, _grid(), _values(L), label_flat=L.ravel())
    assert a["n_basins"][0] == 2
    assert (a["side_lo"][0], a["side_hi"][0]) == (1, 2)
    assert (a["value_lo"][0], a["value_hi"][0]) == (10.0, 2.0)
    assert a["sgn"][0] == 8.0
    assert a["rho"][0] == 8.0
    assert np.isclose(a["dai"][0], 8.0 / 12.0)
    # side-1 centroid (3.5, 3.5), side-2 centroid (3.0, 3.0): the vector
    # points NE (45 deg); side 1 is the higher side, so MATLAB turns it
    # around to point towards the lower values: 225 deg
    assert np.isclose(a["theta"][0], 225.0)
    # midpoint: total length 10, half-way is node (3, 3) -> (3, 3)
    assert np.isclose(a["x"][0], 3.0) and np.isclose(a["y"][0], 3.0)


def test_asymmetry_basin_points_towards_lower_side_when_reversed():
    sn, L = _diagonal()
    v = np.where(L == 1, 2.0, 10.0).ravel()       # basin 2 now higher
    a = DM.asymmetry(sn, _grid(), v, label_flat=L.ravel())
    assert a["sgn"][0] == -8.0                    # still lower label minus higher
    assert np.isclose(a["theta"][0], 45.0)        # towards basin 1, the lower side
    assert np.isclose(a["dai"][0], 8.0 / 12.0)


def test_asymmetry_skips_segments_bordering_three_basins():
    sn, L = _diagonal()
    L = L.copy()
    L[0, 0] = 3
    a = DM.asymmetry(sn, _grid(), _values(L), label_flat=L.ravel())
    assert a["n_basins"][0] == 3
    assert np.isnan(a["rho"][0]) and np.isnan(a["theta"][0])
    assert np.isnan(a["dai"][0]) and a["side_lo"][0] == -1


def test_asymmetry_clips_negative_and_infinite_values_like_matlab():
    sn, L = _diagonal()
    v = np.where(L == 1, 10.0, -5.0).ravel()      # negatives become 0
    a = DM.asymmetry(sn, _grid(), v, label_flat=L.ravel())
    assert a["value_hi"][0] == 0.0 and a["dai"][0] == 1.0
    v[L.ravel() == 2] = np.inf                    # infinities become NaN
    a = DM.asymmetry(sn, _grid(), v, label_flat=L.ravel())
    assert np.isnan(a["value_hi"][0]) and np.isnan(a["theta"][0])


def test_asymmetry_aggregation_functions_agree_on_constant_sides():
    sn, L = _diagonal()
    v = _values(L)
    for agg in ("mean", "median", np.max):
        a = DM.asymmetry(sn, _grid(), v, label_flat=L.ravel(), aggfun=agg)
        assert a["sgn"][0] == 8.0, agg


def test_asymmetry_median_is_a_true_median():
    sn, L = _diagonal()
    v = _values(L).copy()
    v[L.ravel() == 1] = np.arange(15, dtype=float).repeat(1)[:np.sum(L == 1)]
    a_mean = DM.asymmetry(sn, _grid(), v, label_flat=L.ravel())
    a_med = DM.asymmetry(sn, _grid(), v, label_flat=L.ravel(),
                         aggfun="median")
    # the pooled side-1 samples (with multiplicity) differ in mean and median
    assert a_mean["value_lo"][0] != a_med["value_lo"][0]
    a_call = DM.asymmetry(sn, _grid(), v, label_flat=L.ravel(),
                          aggfun=np.median)
    assert a_call["value_lo"][0] == a_med["value_lo"][0]


def test_asymmetry_basin_requires_labels():
    sn, L = _diagonal()
    with pytest.raises(ValueError):
        DM.asymmetry(sn, _grid(), _values(L))


# ---------------------------------------------------------------------------
# asymmetry, pixel_pairs method (points towards the HIGHER side, as MATLAB)

def test_asymmetry_pixel_pairs_hand_computation():
    sn, L = _diagonal()
    a = DM.asymmetry(sn, _grid(), _values(L), method="pixel_pairs")
    # vertical faces: (left - right)/(sum) = -8/12 in u, 0 in v; horizontal
    # faces: (bottom - top)/(sum) = -8/12 in v, 0 in u; the segment mean
    # is (-1/3, -1/3): towards the south-west, where basin 2 (the lower
    # side) lies. MATLAB's diff([hr1,hr2],1,2) is hr2 - hr1; verified
    # against MATLAB output on Big Tujunga.
    assert np.isclose(a["u"][0], -1.0 / 3.0) and np.isclose(a["v"][0], -1.0 / 3.0)
    assert np.isclose(a["rho"][0], np.sqrt(2.0) / 3.0)
    assert np.isclose(a["theta"][0], 225.0)       # towards basin 2 = lower
    assert "dai" not in a


def test_the_two_matlab_methods_point_the_same_way():
    sn, L = _diagonal()
    v = _values(L)
    basin = DM.asymmetry(sn, _grid(), v, label_flat=L.ravel())["theta"][0]
    pairs = DM.asymmetry(sn, _grid(), v, method="pixel_pairs")["theta"][0]
    assert np.isclose(basin, pairs)


# ---------------------------------------------------------------------------
# helpers

def test_azimuth_convention():
    assert DM.azimuth_from_north(np.array([0.0]), np.array([1.0]))[0] == 90 - 90  # north
    assert np.isclose(DM.azimuth_from_north(np.array([1.0]), np.array([0.0]))[0], 90.0)
    assert np.isclose(DM.azimuth_from_north(np.array([0.0]), np.array([-1.0]))[0], 180.0)
    assert np.isclose(DM.azimuth_from_north(np.array([-1.0]), np.array([0.0]))[0], 270.0)
    assert np.isclose(DM.azimuth_from_north(np.array([0.0]), np.array([0.0]))[0], 90.0)


def test_segment_midpoint_of_a_straight_segment_and_a_single_node():
    sn = _network([(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)])   # along a row
    x, y = DM.segment_midpoints(sn, _grid())
    assert np.isclose(x[0], 2.0) and np.isclose(y[0], NR - 2.0)
    ids = np.array([W * 1 + 1], dtype=np.int64)
    one = SortedNetwork(seg_nodes=ids, seg_ptr=np.array([0, 1]),
                        is_ring=np.zeros(1, bool), endpoints=ids,
                        junctions=np.zeros(0, np.int64))
    x, y = DM.segment_midpoints(one, _grid())
    assert (x[0], y[0]) == (1.0, NR - 1.0)


def test_segment_of_position_covers_every_slot():
    sn = _network([(0, 0), (0, 1), (0, 2)])
    two = SortedNetwork(seg_nodes=np.concatenate([sn.seg_nodes, sn.seg_nodes + 7]),
                        seg_ptr=np.array([0, 3, 6]), is_ring=np.zeros(2, bool),
                        endpoints=np.zeros(0, np.int64), junctions=np.zeros(0, np.int64))
    s = DM.segment_of_position(two)
    assert s.tolist() == [0, 0, 0, 1, 1]           # the joint slot goes to seg 0
    pos, _ = DM.edge_positions(two)
    assert pos.tolist() == [0, 1, 3, 4]


def test_orient_by_segment_keeps_one_side_across_a_label_change():
    """Basin 1 (J > I) is split into reach basins 1 (rows 0-1) and 3
    (rows 2-5): the label on that side changes mid-segment with no
    junction. The per-edge lower-label rule swaps sides there; the
    segment rule does not."""
    sn, L = _diagonal()
    L = L.copy()
    L[(np.arange(NC)[None, :] > np.arange(NR)[:, None]) & (np.arange(NR)[:, None] >= 2)] = 3
    pos, edges = DM.edge_positions(sn)
    ca, cb = DM.edge_cells(_grid(), edges)
    # per-edge rule: side A = label 1 on the first edges, then label 2
    # (2 < 3) once the label on the other side becomes 3
    _, _, la_edge, _ = DM.orient_by_basin(ca, cb, L.ravel())
    assert la_edge.tolist()[:2] == [1, 1] and 2 in la_edge.tolist()[4:]
    # segment rule: A stays on the (1 -> 3) side throughout
    forward = sn.seg_nodes[pos] == edges[:, 0]
    seg = DM.segment_of_position(sn)[pos]
    ca2, cb2, la, lb = DM.orient_by_segment(_grid(), edges, ca, cb, L.ravel(),
                                            forward, seg)
    assert set(la.tolist()) == {1, 3} and set(lb.tolist()) == {2}
    v = np.where(L == 2, 2.0, 10.0).ravel()
    p, q = DM.sample_pair(v, ca2, cb2)
    assert np.all(DM.pair_metric_fields(p, q)["sgn"] == 8.0)


def test_orient_by_segment_equals_lower_label_rule_when_pair_is_constant():
    sn, L = _diagonal()
    pos, edges = DM.edge_positions(sn)
    ca, cb = DM.edge_cells(_grid(), edges)
    forward = sn.seg_nodes[pos] == edges[:, 0]
    seg = DM.segment_of_position(sn)[pos]
    a1, b1, _, _ = DM.orient_by_basin(ca, cb, L.ravel())
    a2, b2, _, _ = DM.orient_by_segment(_grid(), edges, ca, cb, L.ravel(), forward, seg)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)


def test_reach_outlet_labels_follows_chains_to_the_outlet():
    # cells 0..7 carry reach labels; stream edges 0->1 (1->2), 2->3 (2->3),
    # 4->5 (4->3): reaches 1, 2, 4 all drain to outlet reach 3; reach 6
    # has no stream edge and maps to itself; label 0 is untouched
    L = np.array([1, 2, 2, 3, 4, 3, 6, 0], dtype=np.int64)
    src = np.array([0, 2, 4], dtype=np.int64)
    dst = np.array([1, 3, 5], dtype=np.int64)
    out = DM.reach_outlet_labels(L, src, dst)
    assert out.tolist() == [0, 3, 3, 3, 3, 5, 6]


def test_reach_outlet_labels_long_chain_needs_pointer_jumping():
    n = 1000
    L = np.arange(1, n + 1, dtype=np.int64)          # cell i has label i+1
    src = np.arange(0, n - 1, dtype=np.int64)
    dst = src + 1                                     # label k -> k+1
    out = DM.reach_outlet_labels(L, src, dst)
    assert np.all(out[1:] == n)


# ---------------------------------------------------------------------------
# the real window, if its checkpoint is present

_CK = Path(__file__).resolve().parents[1] / "checkpoints" / "signflip_pilot_crop4000"


@pytest.mark.skipif(not (_CK / "labels.npz").exists(),
                    reason="Brazil 4000^2 pilot checkpoint not present")
def test_real_window_every_edge_has_the_lower_label_on_side_a():
    from pydivide import divmetrics as M
    net = np.load(_CK / "network.npz")
    sn = SortedNetwork(seg_nodes=net["seg_nodes"], seg_ptr=net["seg_ptr"],
                       is_ring=net["is_ring"], endpoints=net["endpoints"],
                       junctions=net["junctions"])
    L = np.load(_CK / "labels.npz")["L"].ravel()
    grid = GridSpec(nr=4000, nc=4000, cellsize=30.0, x0=0.0, y0=0.0)
    pos, edges = M.edge_positions(sn)
    tab = M.build_edge_table(grid, edges, np.zeros(L.shape, np.float64), L)
    ok = tab.has_pair | ((tab.pair_lo > 0) & (tab.pair_hi > 0)
                         & (tab.pair_lo != tab.pair_hi))
    assert np.all(L[tab.cell_a[ok]] == tab.pair_lo[ok])
    assert np.all(L[tab.cell_b[ok]] == tab.pair_hi[ok])

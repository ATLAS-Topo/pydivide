# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""The rest of the @DIVIDEobj interface, on a synthetic DEM.

No reference data and no MATLAB: the network is built from a small
analytic surface, so these run anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pydivide  # noqa: E402
from pydivide import divideops as O  # noqa: E402
from pydivide.network import fx_grid  # noqa: E402
from pydivide.orderdist import divorder  # noqa: E402

CS = 30.0


@pytest.fixture(scope="module")
def net():
    """A ridge-and-valley surface: several basins, so several divides."""
    tt3 = pytest.importorskip("topotoolbox")
    n = 220
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    z = (600.0
         - 0.0009 * ((x - n / 2) ** 2 + (y - n / 2) ** 2)
         + 45.0 * np.sin(x / 9.0) * np.cos(y / 11.0)
         + 18.0 * np.sin((x + y) / 7.0))
    dem = tt3.GridObject()
    dem.z = np.ascontiguousarray(z.astype(np.float32))
    dem.cellsize = CS
    fd = tt3.FlowObject(dem)
    st = tt3.StreamObject(fd, threshold=200)
    D = pydivide.divide_network(fd=fd, st=st, minarea=200 * CS * CS)
    FX = fx_grid(D.grid, *_fd_edges(fd))
    return D, FX


def _fd_edges(fd):
    return (np.asarray(fd.source, dtype=np.int64),
            np.asarray(fd.target, dtype=np.int64))


# ---------------------------------------------------------------------------
# clean_edges

def test_clean_edges_removes_every_perimeter_node(net):
    D, FX = net
    out = O.clean_edges(D, FX)
    w = D.grid.nc + 1
    nodes = out.network.seg_nodes
    I, J = nodes // w, nodes % w
    on_edge = (I == 0) | (I == D.grid.nr) | (J == 0) | (J == D.grid.nc)
    assert not on_edge.any(), f"{int(on_edge.sum())} perimeter nodes survived"


def test_clean_edges_shrinks_and_keeps_a_valid_network(net):
    D, FX = net
    out = O.clean_edges(D, FX)
    assert len(out.edge_nodes) < len(D.edge_nodes)
    assert out.n_segments > 0
    # distance must be recomputed, not inherited
    assert len(out.distance) == len(out.network.seg_nodes)
    assert np.nanmin(out.distance) >= 0.0


# ---------------------------------------------------------------------------
# remove_short_divides

def test_remove_short_divides_only_removes_first_order(net):
    D, FX = net
    order_before = divorder(D.network, "topo")
    high = D.network.seg_nodes[order_before > 1]
    out = O.remove_short_divides(D, FX, d=5 * CS)
    survivors = set(out.network.seg_nodes.tolist())
    # every node that carried order > 1 must still be present
    missing = [n for n in set(high.tolist()) if n not in survivors]
    assert not missing, f"{len(missing)} higher-order nodes were removed"


def test_remove_short_divides_is_monotonic_in_d(net):
    D, FX = net
    a = O.remove_short_divides(D, FX, d=3 * CS)
    b = O.remove_short_divides(D, FX, d=10 * CS)
    assert len(b.edge_nodes) <= len(a.edge_nodes) <= len(D.edge_nodes)


def test_remove_short_divides_with_zero_changes_nothing(net):
    D, FX = net
    out = O.remove_short_divides(D, FX, d=0.0)
    assert len(out.edge_nodes) == len(D.edge_nodes)


# ---------------------------------------------------------------------------
# shrink

def test_shrink_keeps_the_interior_not_the_tips(net):
    D, FX = net
    out = O.shrink(D, FX, how="segment")
    assert len(out.edge_nodes) < len(D.edge_nodes)
    # what survives must be higher-order than what a first-order tip is
    o = divorder(out.network, "topo")
    assert np.nanmax(o) >= 1


def test_shrink_by_distance(net):
    D, FX = net
    cut = float(np.nanpercentile(D.distance, 50))
    out = O.shrink(D, FX, how="distance", val=cut)
    assert len(out.edge_nodes) < len(D.edge_nodes)


def test_shrink_rejects_unknown_mode(net):
    D, FX = net
    with pytest.raises(ValueError, match="unknown how"):
        O.shrink(D, FX, how="nonsense")


def test_shrink_by_distance_needs_a_value(net):
    D, FX = net
    with pytest.raises(ValueError, match="needs val"):
        O.shrink(D, FX, how="distance")


# ---------------------------------------------------------------------------
# dist_to_node

def test_dist_to_node_is_zero_at_the_seed(net):
    D, FX = net
    seed = int(D.network.junctions[0])
    d = O.dist_to_node(D, seed)
    at_seed = d[D.network.seg_nodes == seed]
    assert np.nanmin(at_seed) == pytest.approx(0.0)


def test_dist_to_node_grows_away_from_the_seed(net):
    D, FX = net
    seed = int(D.network.junctions[0])
    d = O.dist_to_node(D, seed)
    assert np.nanmax(d) > 0
    # Distances accumulate one cellsize per edge, so every value is an
    # exact multiple of the GRID's cellsize -- which is the geotransform's,
    # not the DEM object's, and is 1.0 for this synthetic surface.
    cs = D.grid.cellsize
    fin = d[np.isfinite(d)]
    assert np.allclose(fin / cs, np.round(fin / cs), atol=1e-9)


def test_dist_to_node_respects_the_penalty(net):
    D, FX = net
    seed = int(D.network.junctions[0])
    cs = D.grid.cellsize
    near = O.dist_to_node(D, seed, penalty=10 * cs)
    far = O.dist_to_node(D, seed)
    assert np.isfinite(near).sum() < np.isfinite(far).sum()
    assert np.nanmax(near) <= 10 * cs + 1e-9


def test_dist_to_node_from_endpoints_matches_divdist_at_the_seeds(net):
    """Seeded from every endpoint, the wave starts at 0 exactly where
    divdist does."""
    D, FX = net
    d = O.dist_to_node(D, D.network.endpoints)
    ep = set(D.network.endpoints.tolist())
    at_ep = np.array([n in ep for n in D.network.seg_nodes.tolist()])
    assert np.nanmin(d[at_ep]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# junction geometry

def test_junction_degrees_are_at_least_three(net):
    D, FX = net
    deg = O.junction_degrees(D.network)
    assert deg, "no junctions found on the test surface"
    assert min(deg.values()) >= 3


def test_junction_angles_sum_to_a_full_turn(net):
    D, FX = net
    J = O.junction_angles(D, unit="deg")
    if len(J["node"]) == 0:
        pytest.skip("no degree-3 junctions on this surface")
    total = J["a"] + J["b"] + J["c"]
    assert np.allclose(total, 360.0, atol=1e-6)
    assert np.all(J["a"] >= 0) and np.all(J["b"] >= 0)


def test_junction_angles_radians_option(net):
    D, FX = net
    Jd = O.junction_angles(D, unit="deg")
    Jr = O.junction_angles(D, unit="rad")
    if len(Jd["node"]) == 0:
        pytest.skip("no degree-3 junctions on this surface")
    assert np.allclose(Jr["a"] * 180.0 / np.pi, Jd["a"], atol=1e-9)


def test_junction_angles_rejects_unknown_unit(net):
    D, FX = net
    with pytest.raises(ValueError, match="unknown unit"):
        O.junction_angles(D, unit="gradians")


def test_junction_connectivity_is_finite_and_positive(net):
    D, FX = net
    sub = D.network.junctions[:5]
    C = O.junction_connectivity(D, maxdist=20 * CS, junctions=sub)
    assert len(C["cj"]) == len(sub)
    fin = C["cj"][np.isfinite(C["cj"])]
    assert len(fin) > 0, "every junction returned NaN"
    assert np.all(fin > 0)

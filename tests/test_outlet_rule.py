# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Outlet-corner blanking uses DIVIDEobj's outlets.

DIVIDEobj takes its outlets from ``streampoi(S, 'outlets')``: stream nodes
with no outgoing and at least one incoming stream edge. A one-cell stream
network, a stream node with no edge at all, is not among them, so the divide
edges around it are kept. Such a node can only occur where a cell drains
straight out of the DEM; the surface below makes one on the top edge.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pydivide  # noqa: E402
from pydivide.flownet import FlowNet  # noqa: E402
from pydivide.geometry import divide_edges  # noqa: E402
from pydivide.gridspec import GridSpec  # noqa: E402
from pydivide.labels import basin_labels  # noqa: E402

tt3 = pytest.importorskip("topotoolbox")
CS = 30.0
NR, NC = 20, 9
THRESHOLD = 5                                   # pixels


def _surface():
    """Rows 0-2 drain north over a ridge at row 3, the rest drains south.

    Cell (0, 2) lies 5 m below the top edge, so (1, 1), (1, 2) and (1, 3)
    all drain into it: it gathers 10 cells while each donor gathers 3,
    which makes it a stream node with no stream edge at the threshold of
    5 pixels. The southern columns form ordinary stream networks that
    leave the DEM at the bottom edge.
    """
    z = np.zeros((NR, NC))
    for r in range(NR):
        z[r] = 10.0 * r if r <= 3 else 40.0 - 5.0 * (r - 3)
    z[0, 2] = -5.0
    dem = tt3.GridObject()
    dem.z = np.ascontiguousarray(z.astype(np.float32))
    dem.cellsize = CS
    return dem


@pytest.fixture(scope="module")
def routed():
    dem = _surface()
    fd = tt3.FlowObject(dem)
    st = tt3.StreamObject(fd, threshold=THRESHOLD)
    return fd, st


def test_blanking_outlets_on_a_hand_made_network():
    grid = GridSpec(nr=1, nc=4, cellsize=1.0, x0=0.0, y0=0.0)
    e = np.empty(0, np.int64)
    net = FlowNet(grid=grid, fd_src=e, fd_dst=e,
                  st_nodes=np.array([0, 1, 3], np.int64),
                  st_src=np.array([0], np.int64), st_dst=np.array([1], np.int64),
                  outlets=np.array([1, 3], np.int64))
    # node 1 ends the edge 0 -> 1; node 3 has no edge at all
    assert net.blanking_outlets().tolist() == [1]


def test_one_cell_network_exists(routed):
    fd, st = routed
    net = FlowNet.from_pytopotoolbox(fd, st)
    one_cell = np.setdiff1d(net.st_nodes, np.concatenate([net.st_src, net.st_dst]))
    assert one_cell.tolist() == [2]                       # cell (0, 2)
    assert 2 in net.outlets
    assert 2 not in net.blanking_outlets()
    assert len(net.blanking_outlets()) == len(net.outlets) - 1


def test_divide_edges_around_a_one_cell_network_are_kept(routed):
    fd, st = routed
    net = FlowNet.from_pytopotoolbox(fd, st)
    top_face = (2, 3)                                     # corner nodes (0, 2) and (0, 3)
    D = pydivide.divide_network(fd=fd, st=st, verbose=False)
    kept = set(map(tuple, np.sort(D.edge_nodes, axis=1).tolist()))
    assert top_face in kept

    # the same edge set as divide_edges with DIVIDEobj's outlets, and the
    # face would be lost if the one-cell network were blanked as an outlet
    L, nal = basin_labels(net)
    order = np.argsort(net.st_nodes, kind="stable")

    def edges_with(outlets):
        lab = nal[order][np.searchsorted(net.st_nodes[order], outlets)]
        de = divide_edges(net.grid, L, net.st_src, net.st_dst, outlets, lab)
        return set(map(tuple, np.sort(de.nodes, axis=1).tolist()))

    assert edges_with(net.blanking_outlets()) == kept
    assert top_face not in edges_with(net.outlets)

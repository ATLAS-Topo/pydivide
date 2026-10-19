# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Divide-network operations: the rest of the ``@DIVIDEobj`` interface.

Six methods of MATLAB TopoToolbox 3's ``@DIVIDEobj`` that the core
pipeline did not cover. Semantics follow the MATLAB sources
(``cleanedges.m``, ``removeshortdivides.m``, ``shrink.m``,
``dist2node.m``, ``jctangle.m``, ``jctcon.m``, all Scherler, 2019-2020),
with the deviations documented per function.

The two methods that read a grid across the divide, ``getvalue`` and
``asymmetry``, live in :mod:`pydivide.divmetrics`. Not implemented,
deliberately:

``plot``, ``plotc``
    MATLAB plotting methods. In Python the network is exported as a
    GeoDataFrame and drawn by matplotlib or geopandas, so wrapping a
    plotting call inside the library would add a dependency and take a
    choice away from the user.

Like the MATLAB originals, the three filtering operations take the flow
information as an argument (``fx``, the diagonal-crossing grid, which is
what ``divnet`` actually needs from ``FD``), because rebuilding the
network after a filter requires re-running the segmentation.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .geometry import pair_isin
from .gridspec import GridSpec
from .network import segment_network
from .orderdist import SortedNetwork, divdist, divorder, sort_network

__all__ = [
    "clean_edges",
    "remove_short_divides",
    "shrink",
    "dist_to_node",
    "junction_angles",
    "junction_connectivity",
    "junction_degrees",
]


# ---------------------------------------------------------------------------
# helpers


def _positions(sn: SortedNetwork):
    """(edge_index_of_position, is_within_segment) over seg_nodes[:-1].

    Position i spans ``seg_nodes[i] -> seg_nodes[i+1]``. Positions that
    straddle a segment boundary are not edges.
    """
    n = len(sn.seg_nodes)
    within = np.ones(n - 1, dtype=bool)
    within[sn.seg_ptr[1:-1] - 1] = False       # last node of each segment
    return within


def _edges_from_mask(sn: SortedNetwork, keep_node: np.ndarray) -> np.ndarray:
    """Edges surviving a per-node-instance keep mask.

    An edge survives when both of its node instances survive and the two
    are adjacent inside one segment. This is the array equivalent of
    MATLAB's "set the node to NaN and re-split the run".
    """
    within = _positions(sn)
    a = sn.seg_nodes[:-1]
    b = sn.seg_nodes[1:]
    ok = within & keep_node[:-1] & keep_node[1:]
    e = np.stack([a[ok], b[ok]], axis=1)
    if len(e) == 0:
        return e.astype(np.int64).reshape(0, 2)
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def _rebuild(result, edges: np.ndarray, fx: np.ndarray):
    """Re-segment, re-sort and recompute distance and any cached orders.

    Mirrors the tail of every MATLAB filter: ``divnet``, then ``sort``,
    then ``divdist`` and ``divorder`` if the input carried them.
    """
    grid: GridSpec = result.grid
    if len(edges) == 0:
        raise ValueError("filter removed every divide edge")
    nw = segment_network(grid, edges, fx)
    sn = sort_network(nw, nw.jct_deg)
    dist = divdist(sn, grid.cellsize)
    keep = pair_isin(np.sort(edges, axis=1),
                     result.edge_nodes[result.edge_is_boundary])
    out = replace(result, network=sn, distance=dist, edge_nodes=edges,
                  edge_is_boundary=keep, _orders={})
    for scheme in result._orders:
        out._orders[scheme] = divorder(sn, scheme)
    return out


def junction_degrees(sn: SortedNetwork) -> dict[int, int]:
    """Number of segments terminating at each junction node."""
    st = sn.st()
    jct = set(sn.junctions.tolist())
    deg: dict[int, int] = {}
    for a, b in st.tolist():
        for nid in (a, b):
            if nid in jct:
                deg[nid] = deg.get(nid, 0) + 1
    return deg


# ---------------------------------------------------------------------------
# filters


def clean_edges(result, fx: np.ndarray):
    """Delete divides lying on the perimeter of the grid (``cleanedges``).

    A divide network derived from a DEM contains segments that follow the
    edge of the grid itself. These are artifacts of where the data stops,
    not landforms, and Scherler & Schwanghart (2020) section 5.1 warns
    that they are potentially spurious truncation divides.

    MATLAB sets every perimeter *node* to NaN and re-splits the runs;
    here every edge with a perimeter endpoint is dropped, which is the
    same thing expressed on edges. The network is then re-segmented,
    re-sorted, and its distance and orders recomputed.

    ``pydivide`` already flags these as ``isbnd`` on export. This removes
    them and renumbers, which flagging does not.
    """
    grid: GridSpec = result.grid
    w = grid.nc + 1
    nodes = result.network.seg_nodes
    I, J = nodes // w, nodes % w
    on_edge = (I == 0) | (I == grid.nr) | (J == 0) | (J == grid.nc)
    return _rebuild(result, _edges_from_mask(result.network, ~on_edge), fx)


def remove_short_divides(result, fx: np.ndarray, d: float):
    """Remove first-order divides shorter than ``d`` (``removeshortdivides``).

    Divide networks include short first-order stubs that are usually
    excluded before analysis. A segment is removed when it is BOTH
    shorter than ``d`` map units AND of topological order 1 — the
    conjunction matters, because a short high-order segment is a real
    part of the trunk network.

    What "shorter than d" means is taken from the MATLAB code, not from
    its help text: the run length it compares with ``d / cellsize`` is
    the spacing of the NaN separators, i.e. the number of nodes plus
    one, so a segment is removed when ``(edges + 2) * cellsize < d``.
    The same measure is used here so that the two agree edge for edge
    (verified against MATLAB output on Big Tujunga).
    One deviation, deliberate: MATLAB's ``diff`` of separator positions
    skips the first run in its list, so that segment is never removed
    whatever its length; every segment is measured here.
    """
    sn: SortedNetwork = result.network
    order_topo = divorder(sn, "topo")
    cs = result.grid.cellsize
    n_min = d / cs

    keep = np.ones(len(sn.seg_nodes), dtype=bool)
    for k in range(sn.n_segments):
        lo, hi = int(sn.seg_ptr[k]), int(sn.seg_ptr[k + 1])
        if (hi - lo) + 1 < n_min and np.nanmax(order_topo[lo:hi]) <= 1:
            keep[lo:hi] = False
    return _rebuild(result, _edges_from_mask(sn, keep), fx)


def shrink(result, fx: np.ndarray, how: str = "segment",
           val: float | None = None):
    """Shrink the network by peeling its periphery (``shrink``).

    ``how``:

    ``"segment"``
        keep node instances whose *topological* order exceeds ``val``
        (default 1) — i.e. strip the outermost generation of segments;
    ``"order"``
        the same against whichever order scheme is already cached on the
        result (default: strip the minimum order present);
    ``"distance"``
        keep node instances whose divide distance exceeds ``val``.

    Note the direction: shrinking keeps the *interior*, high-order,
    high-distance part of the network and discards the tips. Runs left
    with a single node are dropped, as in MATLAB.
    """
    sn: SortedNetwork = result.network
    if how == "segment":
        v = 1.0 if val is None else float(val)
        keep = divorder(sn, "topo") > v
    elif how == "order":
        scheme = next(iter(result._orders), "topo")
        o = result.order(scheme)
        v = float(np.nanmin(o)) if val is None else float(val)
        keep = o > v
    elif how == "distance":
        if val is None:
            raise ValueError("shrink(how='distance') needs val")
        keep = result.distance > float(val)
    else:
        raise ValueError(f"unknown how={how!r}: "
                         "expected 'segment', 'order' or 'distance'")
    return _rebuild(result, _edges_from_mask(sn, np.asarray(keep)), fx)


# ---------------------------------------------------------------------------
# distance from arbitrary nodes


def dist_to_node(result, nodes, penalty: float = np.inf) -> np.ndarray:
    """Along-network distance from given nodes (``dist2node``).

    ``divdist`` measures distance from the network's own endpoints.
    This measures it from any chosen set of nodes — a junction, a
    coastline crossing, a point of interest — by expanding a wavefront
    over the segment graph and accumulating segment lengths.

    Returns an array aligned with ``result.network.seg_nodes``, NaN where
    a node was never reached (a disconnected component, or beyond
    ``penalty``).

    Semantics are MATLAB's, which are not a shortest path. ``dist2node``
    first re-sorts the network from the given nodes outward (its
    ``resortdivide``: a wavefront that consumes each segment the first
    time it is touched, orients it away from the arriving node, and
    drops it if the running distance exceeds ``penalty``), and then
    runs ``divdist`` on the re-sorted network with the given nodes as
    its endpoints. The result is therefore the *hierarchical* divide
    distance re-rooted at those nodes: at a junction the outgoing
    segment continues from the largest incoming distance, exactly as
    ordinary divide distance does from the network's own endpoints.
    The same two steps are taken here, reusing :func:`divdist`, which is
    validated node for node against MATLAB. Checked against MATLAB's
    ``dist2node`` output on Big Tujunga.
    """
    sn: SortedNetwork = result.network
    cs = result.grid.cellsize
    seed_list = sorted({int(n) for n in
                        np.atleast_1d(np.asarray(nodes, dtype=np.int64))})

    st = sn.st()
    at_node: dict[int, list[int]] = {}
    for k, (a, b) in enumerate(st.tolist()):
        at_node.setdefault(a, []).append(k)
        at_node.setdefault(b, []).append(k)

    consumed = np.zeros(sn.n_segments, dtype=bool)
    order: list[tuple[int, bool]] = []            # (segment, flipped)
    frontier: list[tuple[int, float]] = [(n, 0.0) for n in seed_list]
    while frontier:
        # MATLAB: every unconsumed segment touching a frontier node gets
        # dist = maxdist(node) + edges * cellsize (a segment touched at
        # both ends keeps the assignment made last, for the node at its
        # original end); then, in segment order, those under the
        # penalty are consumed and their far ends become the next
        # frontier (first assignment per node wins).
        first: dict[int, float] = {}
        for n, d in frontier:
            first.setdefault(n, d)
        # MATLAB's find() over the (segments x 2) membership matrix is
        # column-major: segments touched at their start node, ascending,
        # then segments touched at their end node, ascending. That is
        # both the assignment order (last wins) and the consumption
        # order (first occurrence consumes), and divdist afterwards
        # depends on the consumption order.
        touched: dict[int, tuple[bool, float]] = {}
        visit: list[int] = []
        for col in (0, 1):
            for k in range(sn.n_segments):
                if consumed[k]:
                    continue
                n = int(st[k, col])
                if n in first:
                    n_edges = int(sn.seg_ptr[k + 1] - sn.seg_ptr[k]) - 1
                    touched[k] = (col == 1, first[n] + n_edges * cs)
                    visit.append(k)
        nxt: list[tuple[int, float]] = []
        for k in visit:
            if consumed[k]:
                continue
            flipped, d = touched[k]
            if d < penalty:
                consumed[k] = True
                order.append((k, flipped))
                far = int(st[k, 0]) if flipped else int(st[k, 1])
                nxt.append((far, d))
        frontier = nxt

    out = np.full(len(sn.seg_nodes), np.nan)
    if not order:
        return out
    parts = []
    ptr = [0]
    for k, flipped in order:
        seg = sn.segment(k)
        parts.append(seg[::-1] if flipped else seg)
        ptr.append(ptr[-1] + len(seg))
    sn2 = SortedNetwork(seg_nodes=np.concatenate(parts).astype(np.int64),
                        seg_ptr=np.asarray(ptr, dtype=np.int64),
                        is_ring=sn.is_ring[[k for k, _ in order]],
                        endpoints=np.asarray(seed_list, dtype=np.int64),
                        junctions=sn.junctions)
    d2 = divdist(sn2, cs)
    for i, (k, flipped) in enumerate(order):
        vals = d2[ptr[i]:ptr[i + 1]]
        lo, hi = int(sn.seg_ptr[k]), int(sn.seg_ptr[k + 1])
        out[lo:hi] = vals[::-1] if flipped else vals
    return out


# ---------------------------------------------------------------------------
# junction geometry


def junction_angles(result, unit: str = "deg") -> dict:
    """Angles between the three segments at each degree-3 junction.

    Only junctions where exactly three segments meet have a well-defined
    triple of angles, so ``jctangle`` restricts itself to those, and so
    does this.

    For each incident segment the mean displacement of its nodes from the
    junction gives an average orientation; the three orientations are
    sorted by azimuth and differenced, so the three angles sum to a full
    turn by construction.

    Returns a dict of arrays: ``node`` (junction node id), ``x``, ``y``,
    and ``a``, ``b``, ``c`` (the three angles, ascending order of the
    sorted directions).
    """
    u = unit.lower()
    if u in ("d", "deg", "degree", "degrees"):
        scale = 180.0 / np.pi
    elif u in ("r", "rad", "radian", "radians"):
        scale = 1.0
    else:
        raise ValueError(f"unknown unit {unit!r}")

    sn: SortedNetwork = result.network
    deg = junction_degrees(sn)
    st = sn.st()
    at_node: dict[int, list[int]] = {}
    for k, (a, b) in enumerate(st.tolist()):
        at_node.setdefault(a, []).append(k)
        at_node.setdefault(b, []).append(k)

    nodes, ax, ay, aa, ab, ac = [], [], [], [], [], []
    for node, d in sorted(deg.items()):
        if d != 3:
            continue
        segs = at_node.get(node, [])
        if len(segs) != 3:
            continue
        x0, y0 = result.grid.node_xy(np.array([node], dtype=np.int64))
        thetas = []
        for k in segs:
            lo, hi = int(sn.seg_ptr[k]), int(sn.seg_ptr[k + 1])
            seg = sn.seg_nodes[lo:hi]
            if int(seg[0]) != node:
                seg = seg[::-1]
            sx, sy = result.grid.node_xy(seg)
            thetas.append(np.arctan2(np.mean(sy - y0[0]),
                                     np.mean(sx - x0[0])))
        t = np.sort(np.asarray(thetas))
        d1, d2 = t[1] - t[0], t[2] - t[1]
        nodes.append(node)
        ax.append(float(x0[0]))
        ay.append(float(y0[0]))
        aa.append(d1 * scale)
        ab.append(d2 * scale)
        ac.append((2 * np.pi - d1 - d2) * scale)

    return {"node": np.array(nodes, dtype=np.int64),
            "x": np.array(ax), "y": np.array(ay),
            "a": np.array(aa), "b": np.array(ab), "c": np.array(ac)}


def junction_connectivity(result, maxdist: float = 5000.0,
                          junctions=None) -> dict:
    """Junction connectivity (``jctcon``).

    For each junction, the network within ``maxdist`` along the divides
    is compared against straight-line distance from the junction:

        CJ = sum(euclidean / along-network) * cellsize / maxdist

    A network that runs straight away from the junction gives a ratio
    near one for every node; a network that doubles back gives less. It
    is a measure of how directly the divides radiate from the junction.

    Cost note, which the MATLAB documentation does not give: this is one
    wavefront expansion per junction. The full SE Brazil network has
    3,688,930 junctions, so computing it everywhere is not sensible.
    Pass ``junctions`` to restrict it to the ones of interest.
    """
    sn: SortedNetwork = result.network
    cs = result.grid.cellsize
    jl = (sn.junctions if junctions is None
          else np.atleast_1d(np.asarray(junctions, dtype=np.int64)))

    cj, jx, jy = [], [], []
    for node in jl.tolist():
        dd = dist_to_node(result, node, penalty=2.0 * maxdist)
        ok = np.isfinite(dd) & (dd > 0) & (dd < maxdist)
        if not ok.any():
            cj.append(np.nan)
        else:
            x0, y0 = result.grid.node_xy(np.array([node], dtype=np.int64))
            sx, sy = result.grid.node_xy(sn.seg_nodes[ok])
            euc = np.hypot(sx - x0[0], sy - y0[0])
            keep = euc > 0
            cj.append(float(np.sum(euc[keep] / dd[ok][keep]) * cs / maxdist)
                      if keep.any() else np.nan)
        x0, y0 = result.grid.node_xy(np.array([node], dtype=np.int64))
        jx.append(float(x0[0]))
        jy.append(float(y0[0]))

    return {"node": jl, "x": np.array(jx), "y": np.array(jy),
            "cj": np.array(cj)}

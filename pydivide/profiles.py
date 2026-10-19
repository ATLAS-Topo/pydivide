# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Profiles of every competing basin pair, along and across its divide.

A signed contrast says that one named basin stands higher, or is further
from equilibrium, than another. A table can state that but it cannot
show it. This module draws the divide that a pair of basins shares, in
full, and the ground on either side of it, so the sign can be looked at.

**The pair.** Two reach basins that share a divide at one partition of
the stream network (a channel threshold: 10⁶, 10⁷, 10⁸ or 10⁹ m² of
drainage area). The divide between them is every native divide edge
whose two flanking cells carry those two labels. It can be one line,
several separate stretches (components), or a line that forks (paths
inside a component). All of it is the pair, and all of it is drawn.
Between two large basins at a coarse partition that divide can be
hundreds of kilometres of crest; a fine partition splits it into the
divides of their headwaters.

**The along-divide profile.** Elevation against distance along each
path of the divide, over its full finite length. The elevation of a
native edge is the mean of its two flanking cells and every edge counts
one cell size of distance, the staircase length of the crest on the grid.

**The normal profiles.** Straight sections across the divide at
stations spaced along every path (default every 5 km, the first at half
a spacing from the path's start, never on a path end where the direction
of the divide is undefined). Each section has a chosen length centred
on the divide (default 5 km) and is sampled bilinearly every cell size,
or it is a swath of a chosen width whose statistics across the width
(mean, median, quartiles, minimum, maximum) are reported at every
distance. The direction of the divide at a station is the chord between
the points half a spacing before and after it along the path; a divide
on a grid is a staircase of cell faces and has a direction only over a
length. Every normal is turned so that negative distances lie in the
basin with the lower label (``label_a``) and positive distances in the
other (``label_b``), decided by the labels of the two cells of the edge
under the station: the side convention of the signed metrics. A normal
is a line in map space and can leave both basins within its length.

**The pair key.** ``pair_key(lo, hi, threshold)`` in
:mod:`pydivide.basinpairs` names a pair uniquely across partitions
(``700034600003461`` reads as partition 10⁷, basins 3460 and 3461).

    from pydivide import basin_pair_profiles, basin_pairs
    p = basin_pair_profiles(D, dem, pair=(3460, 3461), spacing=5000, half_length=2500)
    p.plot_along(stations=True)            # figure 2
    p.plot_normals(layout="grid")          # figure 3, stations a, b, c, ...
    p.save("out/pair_3460_3461")           # the four figures, tables, GeoPackage
    for r in basin_pairs(D, dem, labels=L7, threshold=1e7).itertuples():
        ...                                # every pair of the 10⁷ partition

Nothing in the reference implementation draws a pair; this is new.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .basinpairs import NO_PAIR, pair_key, readable_pair_key

#: colours: the main path (navy), the other paths, and the stations
MAIN_COLOUR = "#20374c"
OTHER_COLOURS = ("#d1603d", "#2e8b57", "#8a5aa8", "#c49a1a", "#2a9d9f",
                 "#b23a6e", "#6b8e23", "#5c6b73")
STATION_COLOUR = "#c85139"

STOP_ENDPOINT = "degree-1 tip"
STOP_BRANCHING = "fork (node of degree 3 or more)"
STOP_CLOSED_LOOP = "closed loop"
STOP_VISITED = "no unvisited edge"


# ----------------------------------------------------------------------------
# geometry: stations, directions, sampling
# ----------------------------------------------------------------------------

def arclength(xy) -> np.ndarray:
    """Cumulative length along a polyline of points ``(n, 2)``."""
    d = np.hypot(*np.diff(np.asarray(xy, float), axis=0).T)
    return np.r_[0.0, np.cumsum(d)]


def point_at(xy, s_cum, s) -> np.ndarray:
    """Points at arc-lengths ``s`` along the polyline (clipped to its ends)."""
    xy = np.asarray(xy, float)
    s = np.clip(np.asarray(s, float), 0.0, s_cum[-1])
    return np.column_stack([np.interp(s, s_cum, xy[:, 0]), np.interp(s, s_cum, xy[:, 1])])


def station_positions(length: float, spacing: float) -> np.ndarray:
    """``spacing/2 + k*spacing`` inside the path; one mid station if shorter."""
    if length <= 0:
        return np.zeros(0)
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    if length < spacing:
        return np.array([length / 2.0])
    return np.arange(spacing / 2.0, length, spacing)


def station_frames(xy, spacing: float, tangent_window: float | None = None,
                   closed: bool | None = None):
    """Station positions, points, unit tangents and unit normals of a path.

    The tangent at a station is the chord between the points half a
    ``tangent_window`` (default: the spacing) before and after it. On an
    open path the chord is clipped to the ends. On a closed path (first
    point equal to the last; ``closed=None`` detects it) the chord wraps
    around the loop and the window is capped at half the loop, so that a
    loop shorter than the spacing still has a direction (clipping both
    ends of a loop to its single end point would give a zero chord). A
    chord that is still degenerate falls back to the edge under the
    station, and a station with no direction at all is an error, never a
    zero vector.

    The normal is the tangent turned 90 degrees anticlockwise; which
    basin it points into is applied afterwards by :func:`orient_normals`.
    """
    xy = np.asarray(xy, float)
    sc = arclength(xy)
    length = float(sc[-1])
    s = station_positions(length, spacing)
    w = float(tangent_window if tangent_window else spacing)
    if closed is None:
        # exact equality: a grid path closes on the very node it started
        # from, and a relative tolerance on map coordinates would call a
        # 30 m gap closed at a northing of 8,000,000 m
        closed = len(xy) > 2 and bool(np.array_equal(xy[0], xy[-1]))
    p = point_at(xy, sc, s)
    if closed and length > 0:
        w = min(w, length / 2.0)
        t = (point_at(xy, sc, (s + w / 2.0) % length)
             - point_at(xy, sc, (s - w / 2.0) % length))
    else:
        t = point_at(xy, sc, s + w / 2.0) - point_at(xy, sc, s - w / 2.0)
    norm = np.hypot(t[:, 0], t[:, 1])
    flat = norm == 0
    if flat.any():
        k = np.clip(np.searchsorted(sc, s[flat], side="right") - 1, 0, len(xy) - 2)
        t[flat] = xy[k + 1] - xy[k]
        norm = np.hypot(t[:, 0], t[:, 1])
        if (norm == 0).any():
            raise ValueError("a station has no direction: the path repeats a point")
    t = t / norm[:, None]
    return s, p, t, np.column_stack([-t[:, 1], t[:, 0]])


def orient_normals(nrm, p, hi_side_xy) -> np.ndarray:
    """Flip each normal so it points from the station towards ``hi_side_xy``;
    a NaN target leaves the normal as it is."""
    nrm = np.array(nrm, float)
    d = np.asarray(hi_side_xy, float) - np.asarray(p, float)
    dot = nrm[:, 0] * d[:, 0] + nrm[:, 1] * d[:, 1]
    nrm[np.isfinite(dot) & (dot < 0)] *= -1.0
    return nrm


def bilinear(z, x0: float, y0: float, cs: float, xs, ys, nodata=None) -> np.ndarray:
    """Bilinear samples of a north-up array whose top-left corner is
    ``(x0, y0)``; NaN off the grid or next to a NoData cell. The array is
    read at the four corner cells only and never copied or converted."""
    z = np.asarray(z)
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    col = (xs - x0) / cs - 0.5
    row = (y0 - ys) / cs - 0.5
    with np.errstate(invalid="ignore"):
        c0 = np.floor(col).astype(np.int64)
        r0 = np.floor(row).astype(np.int64)
    nr, nc = z.shape
    ok = np.isfinite(col) & np.isfinite(row) & (c0 >= 0) & (r0 >= 0) & \
        (c0 + 1 < nc) & (r0 + 1 < nr)
    out = np.full(np.shape(xs), np.nan)
    if ok.any():
        c, r = c0[ok], r0[ok]
        a, b = col[ok] - c, row[ok] - r
        q = [z[r, c], z[r, c + 1], z[r + 1, c], z[r + 1, c + 1]]
        q = [np.asarray(v, dtype=np.float64) for v in q]
        if nodata is not None and np.isfinite(nodata):
            bad = np.zeros(len(c), bool)
            for v in q:
                bad |= v == float(nodata)
            for v in q:
                v[bad] = np.nan
        out[ok] = ((1 - a) * (1 - b) * q[0] + a * (1 - b) * q[1]
                   + (1 - a) * b * q[2] + a * b * q[3])
    return out


def normal_offsets(length: float, step: float) -> np.ndarray:
    """Distances along a normal, centred on the divide, every ``step``."""
    half = int(math.floor(length / 2.0 / step))
    return np.arange(-half, half + 1, dtype=float) * step


def sample_normals(sampler, p, t, nrm, length: float, step: float,
                   swath_width: float | None = None) -> dict:
    """Profiles (or swaths) along every normal.

    ``sampler(xs, ys)`` returns elevations. A line profile gives ``z`` of
    shape (stations, offsets). A swath also gives ``mean``, ``median``,
    ``q25``, ``q75``, ``min``, ``max`` and ``n`` across the width at every
    offset, from lines parallel to the normal ``step`` apart across
    ``swath_width``.
    """
    import warnings
    off = normal_offsets(length, step)
    p, t, nrm = (np.asarray(v, float) for v in (p, t, nrm))
    if not swath_width:
        xs = p[:, 0][:, None] + nrm[:, 0][:, None] * off[None, :]
        ys = p[:, 1][:, None] + nrm[:, 1][:, None] * off[None, :]
        return {"offset": off, "z": sampler(xs, ys), "swath": False}
    across = normal_offsets(float(swath_width), step)
    xs = (p[:, 0][:, None, None] + t[:, 0][:, None, None] * across[None, :, None]
          + nrm[:, 0][:, None, None] * off[None, None, :])
    ys = (p[:, 1][:, None, None] + t[:, 1][:, None, None] * across[None, :, None]
          + nrm[:, 1][:, None, None] * off[None, None, :])
    z = sampler(xs, ys)                                  # (station, across, offset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {"offset": off, "swath": True, "across": across,
                "mean": np.nanmean(z, axis=1), "median": np.nanmedian(z, axis=1),
                "q25": np.nanpercentile(z, 25, axis=1),
                "q75": np.nanpercentile(z, 75, axis=1),
                "min": np.nanmin(z, axis=1), "max": np.nanmax(z, axis=1),
                "n": np.isfinite(z).sum(axis=1)}


def station_label(k: int) -> str:
    """``a`` to ``z``, then ``aa``, ``ab``, ... for station ``k`` (0-based)."""
    k = int(k)
    s = ""
    while True:
        s = chr(ord("a") + k % 26) + s
        k = k // 26 - 1
        if k < 0:
            return s


# ----------------------------------------------------------------------------
# the paths of a pair's divide
# ----------------------------------------------------------------------------

@dataclass
class PairPath:
    """One ordered path of a pair's divide, with its profile."""

    pair: tuple
    key: int
    component: int
    path: int
    nodes: np.ndarray                    # (n + 1,) node ids along the path
    edges: np.ndarray                    # (n,) indices into the network's edge list
    start_stop: str
    end_stop: str
    is_cycle: bool
    xy: np.ndarray = field(default=None, repr=False)      # (n + 1, 2) node coordinates
    s: np.ndarray = field(default=None, repr=False)       # (n,) edge-centre distance
    z: np.ndarray = field(default=None, repr=False)       # (n,) edge-centre elevation
    ex: np.ndarray = field(default=None, repr=False)      # (n,) edge-centre x
    ey: np.ndarray = field(default=None, repr=False)
    values: dict = field(default_factory=dict, repr=False)
    normals: dict | None = field(default=None, repr=False)

    @property
    def n_edges(self) -> int:
        return int(len(self.edges))

    @property
    def cellsize(self) -> float:
        return float(2.0 * self.s[0]) if self.s is not None and len(self.s) else float("nan")

    @property
    def length(self) -> float:
        """Length in map units: one cell size per edge."""
        return self.n_edges * self.cellsize

    @property
    def n_stations(self) -> int:
        return 0 if self.normals is None else int(len(self.normals["station_s"]))


def order_paths(edge_nodes, edge_index=None) -> list:
    """Order one pair's edges into paths, stopping at forks and tips.

    A path runs from a degree-1 tip (or, on a closed loop, from any node)
    through nodes of degree 2 until a tip, a fork or a node already
    passed. Every edge belongs to exactly one path. Returns tuples
    ``(nodes, edge_indices, component, start_stop, end_stop, is_cycle)``.
    """
    e = np.asarray(edge_nodes, dtype=np.int64)
    if e.ndim != 2 or e.shape[1] != 2:
        raise ValueError("edge_nodes must be (m, 2) node ids")
    m = len(e)
    if m == 0:
        return []
    idx = np.arange(m) if edge_index is None else np.asarray(edge_index)
    nodes, inv = np.unique(e, return_inverse=True)
    inv = inv.reshape(e.shape)
    deg = np.bincount(inv.ravel(), minlength=len(nodes))
    order = np.argsort(inv.ravel(), kind="stable")
    nbr_edge = order // 2
    nbr_node = inv.ravel()[order ^ 1]
    ptr = np.r_[0, np.cumsum(deg)]
    comp = np.full(len(nodes), -1, np.int64)
    n_comp = 0
    for s0 in range(len(nodes)):
        if comp[s0] >= 0:
            continue
        stack = [s0]
        comp[s0] = n_comp
        while stack:
            u = stack.pop()
            for k in range(ptr[u], ptr[u + 1]):
                v = nbr_node[k]
                if comp[v] < 0:
                    comp[v] = n_comp
                    stack.append(v)
        n_comp += 1
    used = np.zeros(m, bool)
    out = []

    def walk(start_u, first_edge):
        chain_nodes, chain_edges = [start_u], []
        u, ek = start_u, first_edge
        while True:
            used[ek] = True
            chain_edges.append(ek)
            v = inv[ek, 0] if inv[ek, 1] == u else inv[ek, 1]
            chain_nodes.append(v)
            if v == start_u:
                return chain_nodes, chain_edges, STOP_CLOSED_LOOP
            if deg[v] == 1:
                return chain_nodes, chain_edges, STOP_ENDPOINT
            if deg[v] > 2:
                return chain_nodes, chain_edges, STOP_BRANCHING
            nxt = [k for k in range(ptr[v], ptr[v + 1]) if not used[nbr_edge[k]]]
            if not nxt:
                return chain_nodes, chain_edges, STOP_VISITED
            u, ek = v, nbr_edge[nxt[0]]

    def emit(chain_nodes, chain_edges, start_stop, end_stop):
        cyc = chain_nodes[0] == chain_nodes[-1]
        out.append((nodes[np.asarray(chain_nodes)], idx[np.asarray(chain_edges)],
                    int(comp[chain_nodes[0]]), start_stop, end_stop, bool(cyc)))

    for u in np.flatnonzero(deg == 1):
        for k in range(ptr[u], ptr[u + 1]):
            if not used[nbr_edge[k]]:
                cn, ce, stop = walk(u, nbr_edge[k])
                emit(cn, ce, STOP_ENDPOINT, stop)
    for u in np.flatnonzero(deg > 2):
        for k in range(ptr[u], ptr[u + 1]):
            if not used[nbr_edge[k]]:
                cn, ce, stop = walk(u, nbr_edge[k])
                emit(cn, ce, STOP_BRANCHING, stop)
    for ek in range(m):                       # what is left lies on closed loops
        if not used[ek]:
            cn, ce, stop = walk(inv[ek, 0], ek)
            emit(cn, ce, STOP_CLOSED_LOOP, stop)
    return out


# ----------------------------------------------------------------------------
# the network's edges with labels and elevations, computed once
# ----------------------------------------------------------------------------

class _PairContext:
    """Every divide edge of a network with its two cells, labels and
    elevations, so one pair or every pair can be assembled from it."""

    def __init__(self, D, dem, labels, threshold):
        from .divmetrics import edge_cells, edge_positions, segment_of_position
        from .gridspec import check_same_grid
        check_same_grid(D, "the divide network", dem, "the DEM")
        self.D = D
        self.grid = D.grid
        self.cs = float(D.grid.cellsize)
        self.crs = getattr(D, "crs", None)
        self.threshold = None if threshold is None else float(threshold)
        z2 = np.asarray(dem.z)
        if z2.ndim != 2:
            raise ValueError("the DEM must be a 2-D array")
        self.z2 = z2
        nd = getattr(dem, "nodata", None)
        self.nodata = float(nd) if isinstance(nd, (int, float)) and np.isfinite(nd) else None
        if labels is None:
            labels = getattr(D, "labels", None)
        if labels is None:
            raise ValueError(
                "labels is required: the result carries no basin-label raster; pass the "
                "one the network was built with, from pydivide.labels.basin_labels(net)[0], "
                "or the labels of another partition from reach_basin_labels().")
        L = np.asarray(labels).ravel()
        if L.size != self.grid.nr * self.grid.nc:
            raise ValueError(f"labels has {L.size:,} cells; the network's grid has "
                             f"{self.grid.nr * self.grid.nc:,}")
        pos, edges = edge_positions(D.network)
        self.pos = pos
        self.edges = np.sort(np.asarray(edges, np.int64), axis=1)
        self.seg = segment_of_position(D.network)[pos]
        self.ca, self.cb = edge_cells(self.grid, self.edges)
        self.la = np.where(self.ca >= 0, L[np.maximum(self.ca, 0)], -1)
        self.lb = np.where(self.cb >= 0, L[np.maximum(self.cb, 0)], -1)
        self.lo = np.minimum(self.la, self.lb)
        self.hi = np.maximum(self.la, self.lb)
        self.valid = (self.lo > 0) & (self.lo != self.hi)
        self._cache = {}

    def cell_values(self, flat, cells) -> np.ndarray:
        out = np.full(len(cells), np.nan)
        ok = cells >= 0
        v = np.asarray(flat[cells[ok]], dtype=np.float64)
        if self.nodata is not None:
            v = np.where(v == self.nodata, np.nan, v)
        out[ok] = v
        return out

    def edge_mean(self, flat, e) -> np.ndarray:
        """The mean of the two flanking cells of edges ``e`` (NaN if either
        side is outside the grid or NoData)."""
        return (self.cell_values(flat, self.ca[e]) + self.cell_values(flat, self.cb[e])) / 2.0

    def pairs(self):
        """Every pair present: ``(lo, hi, n_edges)`` arrays, longest first."""
        if "pairs" not in self._cache:
            sel = self.valid
            key = self.lo[sel].astype(np.int64) * np.int64(1 << 32) + self.hi[sel]
            uniq, counts = np.unique(key, return_counts=True)
            order = np.argsort(-counts, kind="stable")
            uniq, counts = uniq[order], counts[order]
            self._cache["pairs"] = (uniq >> np.int64(32), uniq & np.int64(0xFFFFFFFF), counts)
        return self._cache["pairs"]

    def edges_of(self, lo: int, hi: int) -> np.ndarray:
        return np.flatnonzero(self.valid & (self.lo == lo) & (self.hi == hi))

    def pair_of_segment(self, segid: int) -> tuple:
        """The pair on most of a segment's edges, ties to the smallest
        labels: a pair one of its edges borders."""
        sel = (self.seg == (int(segid) - 1)) & self.valid
        if not sel.any():
            raise ValueError(f"segment {segid} has no edge with two labelled sides")
        pairs = np.column_stack([self.lo[sel], self.hi[sel]])
        uniq, counts = np.unique(pairs, axis=0, return_counts=True)
        best = np.lexsort((uniq[:, 1], uniq[:, 0], -counts))[0]
        return int(uniq[best, 0]), int(uniq[best, 1])

    def sampler(self, xs, ys):
        g = self.grid
        return bilinear(self.z2, g.x0, g.y0, self.cs, xs, ys, self.nodata)

    def hi_side_points(self, p: PairPath, s, hi: int) -> np.ndarray:
        """Per station, the centre of the flanking cell in basin ``hi``."""
        from .divmetrics import cell_xy
        j = np.clip((np.asarray(s, float) / self.cs).astype(np.int64), 0, p.n_edges - 1)
        e = p.edges[j]
        cell = np.where(self.la[e] == hi, self.ca[e], np.where(self.lb[e] == hi, self.cb[e], -1))
        x, y = cell_xy(self.grid, cell)
        out = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
        out[cell < 0] = np.nan
        return out

    def build(self, lo: int, hi: int, *, spacing, half_length, step, swath_width,
              tangent_window, values, min_path, normals) -> "BasinPairProfiles":
        lo, hi = int(lo), int(hi)
        gi = self.edges_of(lo, hi)
        if not len(gi):
            raise ValueError(f"no divide edges separate basins {lo} and {hi}")
        key = pair_key(lo, hi, self.threshold) if self.threshold is not None else NO_PAIR
        zflat = self.z2.ravel()
        step = float(step if step is not None else self.cs)
        paths = []
        for pi, (nodes, ge, comp, a, b, cyc) in enumerate(order_paths(self.edges[gi],
                                                                       edge_index=gi)):
            p = PairPath(pair=(lo, hi), key=int(key), component=int(comp), path=pi,
                         nodes=np.asarray(nodes, np.int64), edges=np.asarray(ge, np.int64),
                         start_stop=a, end_stop=b, is_cycle=bool(cyc))
            x, y = self.grid.node_xy(p.nodes)
            p.xy = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
            p.ex, p.ey = (p.xy[:-1, 0] + p.xy[1:, 0]) / 2.0, (p.xy[:-1, 1] + p.xy[1:, 1]) / 2.0
            p.s = (np.arange(p.n_edges) + 0.5) * self.cs
            p.z = self.edge_mean(zflat, p.edges)
            for name, g in (values or {}).items():
                p.values[name] = self.edge_mean(np.asarray(g).ravel(), p.edges)
            if normals and p.length >= float(min_path):
                s, pt, t, nrm = station_frames(p.xy, spacing, tangent_window,
                                               closed=p.is_cycle)
                nrm = orient_normals(nrm, pt, self.hi_side_points(p, s, hi))
                res = sample_normals(self.sampler, pt, t, nrm, 2.0 * half_length, step,
                                     swath_width)
                res.update({"station_s": s, "station_xy": pt, "tangent": t, "normal": nrm,
                            "station_label": [station_label(k) for k in range(len(s))],
                            "z_divide": np.interp(s, p.s, p.z) if p.n_edges else s * np.nan})
                p.normals = res
            paths.append(p)
        main = int(np.argmax([p.n_edges for p in paths]))
        settings = {"spacing_m": float(spacing), "half_length_m": float(half_length),
                    "length_m": 2.0 * float(half_length), "step_m": step,
                    "swath_width_m": float(swath_width) if swath_width else None,
                    "tangent_window_m": float(tangent_window or spacing),
                    "min_path_m": float(min_path), "threshold": self.threshold,
                    "pair_key": int(key), "n_edges": int(len(gi)), "n_paths": len(paths),
                    "n_components": len({p.component for p in paths}),
                    "cellsize": self.cs,
                    "along_rule": "elevation of a native edge is the mean of its two "
                                  "flanking cells; every edge counts one cell size",
                    "side_rule": "negative offsets lie in basin label_a (the lower label), "
                                 "positive offsets in basin label_b, decided by the labels "
                                 "of the two cells of the edge under the station"}
        return BasinPairProfiles(pair=(lo, hi), key=int(key), threshold=self.threshold,
                                 paths=paths, main=main, label_a=lo, label_b=hi,
                                 settings=settings, crs=self.crs)


# ----------------------------------------------------------------------------
# the result
# ----------------------------------------------------------------------------

@dataclass
class BasinPairProfiles:
    """The profiles of one basin pair: every path of its divide, with the
    along-divide profile of each and the normal profiles at its stations.

    ``paths[main]`` is the longest path. ``along_s``, ``along_z``,
    ``along_xy`` and ``across`` are that path's profile and sections, kept
    for callers that expect one line; the others are in ``paths``.
    """

    pair: tuple
    key: int
    threshold: float | None
    paths: list
    main: int
    label_a: int
    label_b: int
    settings: dict = field(default_factory=dict)
    crs: object = None

    # ---- the main path, for callers that expect one line -----------------

    @property
    def along_s(self) -> np.ndarray:
        return self.paths[self.main].s

    @property
    def along_z(self) -> np.ndarray:
        return self.paths[self.main].z

    @property
    def along_xy(self) -> np.ndarray:
        return self.paths[self.main].xy

    @property
    def along_values(self) -> dict:
        return self.paths[self.main].values

    @property
    def across(self) -> list:
        """One dict per station of the main path: ``s``, ``d`` (offsets),
        ``z`` (the profile, or the swath mean), ``xy`` (the line) and
        ``z_divide``."""
        p = self.paths[self.main]
        r = p.normals
        if r is None:
            return []
        half = self.settings["length_m"] / 2.0
        out = []
        for k in range(len(r["station_s"])):
            pt, n = r["station_xy"][k], r["normal"][k]
            off = r["offset"]
            xy = np.column_stack([pt[0] + n[0] * off, pt[1] + n[1] * off])
            out.append({"s": float(r["station_s"][k]), "d": off.copy(),
                        "z": (r["mean"] if r["swath"] else r["z"])[k].copy(), "xy": xy,
                        "z_divide": float(r["z_divide"][k]), "label": r["station_label"][k],
                        "half_length": half})
        return out

    def __len__(self) -> int:
        return len(self.across)

    @property
    def length(self) -> float:
        """The whole divide of the pair, every path, in map units."""
        return float(sum(p.length for p in self.paths))

    @property
    def n_stations(self) -> int:
        return int(sum(p.n_stations for p in self.paths))

    @property
    def readable(self) -> str:
        if self.key != NO_PAIR:
            return readable_pair_key(self.key)
        return f"basins {self.label_a} | {self.label_b}"

    # ---- tables and GIS ----------------------------------------------------

    def along_table(self):
        import pandas as pd
        rows = [pd.DataFrame({"pair_key": p.key, "basin_lo": p.pair[0], "basin_hi": p.pair[1],
                              "component": p.component, "path": p.path,
                              "sample": np.arange(p.n_edges), "s_m": p.s, "x": p.ex,
                              "y": p.ey, "z_m": p.z, **{k: v for k, v in p.values.items()}})
                for p in self.paths]
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def normals_table(self):
        import pandas as pd
        out = []
        for p in self.paths:
            r = p.normals
            if r is None or not len(r["station_s"]):
                continue
            off, n_st = r["offset"], len(r["station_s"])
            base = {"pair_key": p.key, "basin_lo": p.pair[0], "basin_hi": p.pair[1],
                    "component": p.component, "path": p.path,
                    "station": np.repeat(np.arange(n_st), len(off)),
                    "station_label": np.repeat(np.asarray(r["station_label"], object), len(off)),
                    "station_s_m": np.repeat(r["station_s"], len(off)),
                    "offset_m": np.tile(off, n_st)}
            if r["swath"]:
                for k in ("mean", "median", "q25", "q75", "min", "max"):
                    base[f"z_{k}"] = np.asarray(r[k]).ravel()
                base["n_samples"] = np.asarray(r["n"]).ravel()
            else:
                base["z_m"] = np.asarray(r["z"]).ravel()
            out.append(pd.DataFrame(base))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    def gis_layers(self, crs=None) -> dict:
        """``pair_paths``, ``stations`` and ``normals`` (lines, or swath
        rectangles) as GeoDataFrames."""
        import geopandas as gpd
        import shapely
        crs = crs if crs is not None else self.crs
        lines = gpd.GeoDataFrame(
            {"pair_key": [p.key for p in self.paths], "pair": [self.readable] * len(self.paths),
             "basin_lo": [p.pair[0] for p in self.paths],
             "basin_hi": [p.pair[1] for p in self.paths],
             "component": [p.component for p in self.paths], "path": [p.path for p in self.paths],
             "length_m": [p.length for p in self.paths],
             "start_stop": [p.start_stop for p in self.paths],
             "end_stop": [p.end_stop for p in self.paths]},
            geometry=[shapely.LineString(p.xy) if len(p.xy) > 1 else shapely.Point(p.xy[0])
                      for p in self.paths], crs=crs)
        st_rows, st_geom, nm_rows, nm_geom = [], [], [], []
        half = self.settings["length_m"] / 2.0
        for p in self.paths:
            r = p.normals
            if r is None:
                continue
            for k, (s, pt, t, n) in enumerate(zip(r["station_s"], r["station_xy"],
                                                  r["tangent"], r["normal"])):
                st_rows.append({"pair_key": p.key, "component": p.component, "path": p.path,
                                "station": k, "station_label": r["station_label"][k],
                                "station_s_m": float(s)})
                st_geom.append(shapely.Point(*pt))
                a, b = pt - n * half, pt + n * half
                if r["swath"]:
                    w = self.settings["swath_width_m"] / 2.0
                    g = shapely.Polygon([a - t * w, b - t * w, b + t * w, a + t * w])
                else:
                    g = shapely.LineString([a, b])
                nm_rows.append({**st_rows[-1], "kind": "swath" if r["swath"] else "profile",
                                "note": f"negative end in basin {self.label_a}, positive end "
                                        f"in basin {self.label_b}"})
                nm_geom.append(g)
        out = {"pair_paths": lines}
        if st_rows:
            out["stations"] = gpd.GeoDataFrame(st_rows, geometry=st_geom, crs=crs)
            out["normals"] = gpd.GeoDataFrame(nm_rows, geometry=nm_geom, crs=crs)
        return out

    def to_geodataframe(self, crs=None):
        """The paths and the sections in one layer (``kind`` says which)."""
        import pandas as pd
        layers = self.gis_layers(crs)
        parts = [layers["pair_paths"].assign(kind="along_divide")]
        if "normals" in layers:
            parts.append(layers["normals"].assign(kind="across_divide"))
        return pd.concat(parts, ignore_index=True)

    # ---- figures -----------------------------------------------------------

    def plot_along(self, stations: bool = False, ylim=None, max_panels: int = 8,
                   min_path: float | None = None, title: str | None = None, width_mm=260):
        """Figure 1 (and, with ``stations``, figure 2): the along-divide
        profile of every path, the main path on top."""
        return plot_along(self.paths, self.main, stations=stations, ylim=ylim,
                          max_panels=max_panels,
                          min_path=self.settings["min_path_m"] if min_path is None else min_path,
                          title=title or f"pair {self.readable}", width_mm=width_mm)

    def plot_normals(self, path: int | None = None, layout: str = "grid", same_y=False,
                     ylim=None, stations=None, max_panels: int = 24, ncols: int = 6):
        """Figure 3 (``layout="grid"``: one panel per station, lettered) or
        figure 4 (``"stacked"``: every station on one axis) of one path,
        the main path by default; ``"side"`` draws them side by side along
        the divide."""
        p = self.paths[self.main if path is None else int(path)]
        if p.normals is None or not p.n_stations:
            return None
        head = f"pair {self.readable}, component {p.component}, path {p.path}"
        if layout == "grid":
            return plot_normals_grid(p, self.label_a, self.label_b, head, stations=stations,
                                     max_panels=max_panels, ncols=ncols, same_y=same_y,
                                     ylim=ylim)
        if layout == "stacked":
            return plot_normals_stacked(p, self.label_a, self.label_b, head, ylim=ylim)
        if layout == "side":
            return plot_normals_side(p, self.label_a, self.label_b, head, ylim=ylim)
        raise ValueError("layout must be 'grid', 'stacked' or 'side'")

    def plot(self, value: str | None = None, ax=None):
        """Two panels: the main path's crest, and its sections stacked."""
        import matplotlib.pyplot as plt
        p = self.paths[self.main]
        fig, (a0, a1) = plt.subplots(2, 1, figsize=(10, 8), dpi=140)
        a0.plot(p.s / 1000.0, p.z, color=MAIN_COLOUR, lw=1.0)
        a0.set_xlabel("distance along the shared divide (km)")
        a0.set_ylabel("crest elevation (m)")
        a0.set_title(f"divide shared by basin {self.label_a} and basin {self.label_b}"
                     f" ({self.readable}); main path of {len(self.paths)}", fontsize=9)
        if value and value in p.values:
            t = a0.twinx()
            t.plot(p.s / 1000.0, p.values[value], color="tab:red", lw=0.9, alpha=0.8)
            t.set_ylabel(value, color="tab:red")
        for sec in self.across:
            a1.plot(sec["d"], sec["z"], lw=0.7, alpha=0.6)
        a1.axvline(0.0, color="k", lw=0.8)
        a1.set_xlabel(f"distance from the divide (m), negative into basin {self.label_a}, "
                      f"positive into basin {self.label_b}")
        a1.set_ylabel("elevation (m)")
        a1.set_title(f"{len(self.across)} sections across the divide", fontsize=9)
        fig.tight_layout()
        return fig

    def save(self, out_dir, *, formats=("png", "pdf"), dpi: int = 200, same_y: bool = False,
             ylim=None, layouts=("grid", "stacked"), gis: bool = True, tables: bool = True,
             all_stations_pdf: bool = True) -> dict:
        """Write the four figures, the tables and the GeoPackage of this pair.

        ``1_along``, ``2_along_stations``, ``3_normals_grouped_<c>_<p>``
        (an overview of up to 24 stations; with ``all_stations_pdf`` every
        station in a PDF) and ``4_normals_stacked_<c>_<p>`` per path with
        stations; ``along.csv``, ``normals.csv``; ``pair_<key>.gpkg``.
        """
        import matplotlib.pyplot as plt
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written = []

        def emit(fig, stem):
            if fig is None:
                return
            for f in formats:
                fig.savefig(out / f"{stem}.{f}", dpi=dpi)
                written.append(f"{stem}.{f}")
            plt.close(fig)

        emit(self.plot_along(stations=False, ylim=ylim), "1_along")
        if self.n_stations:
            emit(self.plot_along(stations=True, ylim=ylim), "2_along_stations")
            for i, p in enumerate(self.paths):
                if p.normals is None or not p.n_stations:
                    continue
                stem = f"{p.component}_{p.path}"
                if "grid" in layouts:
                    emit(self.plot_normals(i, "grid", same_y=same_y, ylim=ylim),
                         f"3_normals_grouped_{stem}")
                    if all_stations_pdf:
                        normals_pdf(self, i, out / f"3_normals_grouped_{stem}_all_stations.pdf",
                                    same_y=same_y, ylim=ylim)
                        written.append(f"3_normals_grouped_{stem}_all_stations.pdf")
                if "stacked" in layouts:
                    emit(self.plot_normals(i, "stacked", ylim=ylim), f"4_normals_stacked_{stem}")
                if "side" in layouts:
                    emit(self.plot_normals(i, "side", ylim=ylim), f"5_normals_side_{stem}")
        if tables:
            self.along_table().to_csv(out / "along.csv", index=False)
            written.append("along.csv")
            if self.n_stations:
                self.normals_table().to_csv(out / "normals.csv", index=False)
                written.append("normals.csv")
        if gis:
            from .vectorio import repair_geopackage_crs
            name = f"pair_{self.key}.gpkg" if self.key != NO_PAIR else \
                f"pair_{self.label_a}_{self.label_b}.gpkg"
            gp = out / name
            if gp.exists():
                gp.unlink()
            for lname, g in self.gis_layers().items():
                g.to_file(gp, layer=lname, driver="GPKG")
            repair_geopackage_crs(gp)
            written.append(name)
        return {"folder": str(out), "files": written}


def basin_pair_profiles(D, dem, pair=None, *, segid=None, labels=None, threshold=None,
                        spacing: float = 5000.0, half_length: float = 2500.0,
                        step: float | None = None, swath_width: float | None = None,
                        tangent_window: float | None = None, values: dict | None = None,
                        min_path: float = 0.0, normals: bool = True) -> BasinPairProfiles:
    """Along-divide and normal profiles of one basin pair.

    Parameters
    ----------
    D
        A :class:`~pydivide.pipeline.DivideNetworkResult`.
    dem
        The DEM the network came from (same grid, checked).
    pair
        ``(label_a, label_b)`` reach-basin labels in either order, at the
        partition of ``labels``. Give this or ``segid``.
    segid
        One divide segment, 1-based as exported; its own pair is used
        (the pair on most of its edges, ties to the smallest labels).
    labels
        The reach-basin raster of the partition the pair is named in;
        the network's own labels (``D.labels``) when omitted. Labels of a
        coarser partition come from :func:`pydivide.reach_basin_labels`.
    threshold
        The partition's channel threshold in m², used to form the pair
        key (``pair_key``); the key is -1 when it is not given.
    spacing, half_length
        Distance between stations along each path, and the reach of a
        normal to either side of the divide, in map units.
    step
        Sample interval along a normal (and across a swath); the cell size
        when omitted.
    swath_width
        A width makes every normal a swath with statistics across it.
    tangent_window
        The length over which the divide's direction is taken at a
        station; the spacing when omitted.
    values
        Extra grids to sample along every path, ``{name: 2-D array}``.
    min_path
        Paths shorter than this get no stations (they are still drawn);
        every path by default; 1 km skips the few edges of a fork stub.
    normals
        ``False`` computes the along-divide profiles only.
    """
    if pair is None and segid is None:
        raise ValueError("give either pair=(label_a, label_b) or segid=")
    ctx = _PairContext(D, dem, labels, threshold)
    if segid is not None:
        lo, hi = ctx.pair_of_segment(segid)
    else:
        lo, hi = int(min(pair)), int(max(pair))
    return ctx.build(lo, hi, spacing=spacing, half_length=half_length, step=step,
                     swath_width=swath_width, tangent_window=tangent_window, values=values,
                     min_path=min_path, normals=normals)


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------

def _relations(paths, main):
    """For every path: None (the main path) or the distance along the main
    path where it leaves it, if it starts or ends on a node of it."""
    mp = paths[main]
    pos = {int(n): k * mp.cellsize for k, n in enumerate(mp.nodes)}
    return [None if i == main else pos.get(int(p.nodes[0]), pos.get(int(p.nodes[-1])))
            for i, p in enumerate(paths)]


def _relation_title(paths, main, fork, i):
    p = paths[i]
    km = p.length / 1000.0
    if i == main:
        return f"main path: component {p.component}, path {p.path}, {km:.1f} km"
    if fork[i] is not None:
        return (f"branch: component {p.component}, path {p.path}, {km:.1f} km, leaves the "
                f"main path at {fork[i] / 1000:.1f} km (triangle on the main path)")
    if p.component == paths[main].component:
        return (f"branch of a branch: component {p.component}, path {p.path}, {km:.1f} km, "
                "joined to the main path through another branch")
    return (f"separate stretch: component {p.component}, path {p.path}, {km:.1f} km, part of "
            "this pair's divide not connected to the main path")


def _zrange(paths, shown, ylim):
    if ylim is not None:
        return tuple(ylim)
    zs = [paths[i].z[np.isfinite(paths[i].z)] for i in shown]
    zs = np.concatenate([z for z in zs if z.size] or [np.zeros(1)])
    pad = max((zs.max() - zs.min()) * 0.06, 1.0)
    return (float(zs.min() - pad), float(zs.max() + pad))


def draw_along(axes, paths, main, stations=False, ylim=None, max_panels: int = 8,
               min_path: float = 1000.0):
    """Draw the along-divide panels on given axes (one per shown path);
    returns ``(shown, hidden, ylim)``."""
    import matplotlib.transforms as mtrans
    fork = _relations(paths, main)
    rest = sorted((i for i in range(len(paths)) if i != main), key=lambda i: -paths[i].n_edges)
    long_enough = [i for i in rest if paths[i].length >= min_path]
    shown = [main] + long_enough[:max(0, max_panels - 1)]
    hidden = [i for i in rest if i not in shown]
    ylim = _zrange(paths, shown, ylim)
    cs0 = paths[main].cellsize
    for row, (ax, i) in enumerate(zip(axes, shown)):
        p = paths[i]
        c = MAIN_COLOUR if i == main else OTHER_COLOURS[(row - 1) % len(OTHER_COLOURS)]
        for q in np.linspace(ylim[0], ylim[1], 5)[1:-1]:
            ax.axhline(q, color="#e5e9ed", lw=0.4, zorder=0)
        ax.plot(p.s / 1000.0, p.z, lw=0.9, color=c, zorder=3)
        ax.set_ylim(*ylim)
        ax.set_xlim(0, max(p.length, cs0) / 1000.0)
        ax.tick_params(labelsize=7, colors="#53636c")
        with_st = stations and p.normals is not None and p.n_stations > 0
        ax.set_title(_relation_title(paths, main, fork, i), fontsize=7.5, loc="left",
                     color=c, pad=13 if stations else 4)
        tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
        if i == main:
            for j, f in enumerate(fork):
                if f is not None and j in shown:
                    col = OTHER_COLOURS[(shown.index(j) - 1) % len(OTHER_COLOURS)]
                    ax.plot([f / 1000.0], [1.0], marker="v", ms=6, color=col, transform=tr,
                            clip_on=False, zorder=5)
        if with_st:
            st = p.normals["station_s"] / 1000.0
            for x in st:
                ax.axvline(x, color=STATION_COLOUR, lw=0.5, alpha=0.45, zorder=1)
            every = max(1, int(np.ceil(len(st) / 40)))
            for k in range(0, len(st), every):
                ax.text(st[k], 1.01, p.normals["station_label"][k], transform=tr, fontsize=5.5,
                        ha="center", va="bottom", color=STATION_COLOUR)
    return shown, hidden, ylim


def plot_along(paths, main, stations=False, ylim=None, max_panels: int = 8,
               min_path: float = 1000.0, title: str = "", width_mm: float = 260):
    """The along-divide profile of a pair: the main path (the longest) on
    top and every other path in its own panel on its own distance, named
    as a branch (with the kilometre where it leaves the main path), a
    branch of a branch, or a separate stretch; panels share the elevation
    axis, which ``ylim`` fixes so that pairs compare by eye. With
    ``stations`` the normal-profile stations are drawn and lettered."""
    import matplotlib.pyplot as plt
    rest = sorted((i for i in range(len(paths)) if i != main), key=lambda i: -paths[i].n_edges)
    n_shown = 1 + min(len([i for i in rest if paths[i].length >= min_path]), max(0, max_panels - 1))
    heights = [2.4] + [1.4] * (n_shown - 1)
    fig, axes = plt.subplots(n_shown, 1, squeeze=False, sharey=True,
                             figsize=(width_mm / 25.4, sum(heights) + 1.0),
                             gridspec_kw={"height_ratios": heights})
    shown, hidden, _ = draw_along(axes[:, 0], paths, main, stations, ylim, max_panels, min_path)
    axes[-1, 0].set_xlabel("km along the divide from the start of the path"
                           + ("   (red lines: normal-profile stations, lettered)"
                              if stations else ""), fontsize=8, color="#53636c")
    fig.supylabel("elevation, m", fontsize=8)
    note = (f"; {len(hidden)} short path(s) not drawn (component/path): "
            + ", ".join(f"{paths[i].component}/{paths[i].path}" for i in hidden[:12])
            + ("..." if len(hidden) > 12 else "")) if hidden else ""
    fig.suptitle(f"{title}, {len(paths)} path(s), {sum(p.length for p in paths) / 1000:.1f} km"
                 f"{note}", fontsize=9)
    fig.tight_layout()
    return fig


def normals_yrange(r: dict):
    """The elevation range over every station of a path."""
    import warnings
    keys = ("min", "max") if r["swath"] else ("z",)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lo = min(float(np.nanmin(r[k])) for k in keys)
        hi = max(float(np.nanmax(r[k])) for k in keys)
    pad = max((hi - lo) * 0.04, 1.0)
    return lo - pad, hi + pad


def _draw_section(ax, r, k):
    off = r["offset"] / 1000.0
    if r["swath"]:
        ax.fill_between(off, r["min"][k], r["max"][k], color="#c9d6df", lw=0)
        ax.fill_between(off, r["q25"][k], r["q75"][k], color="#8fa8ba", lw=0)
        ax.plot(off, r["mean"][k], color=MAIN_COLOUR, lw=0.8)
    else:
        ax.plot(off, r["z"][k], color=MAIN_COLOUR, lw=0.8)
    ax.axvline(0, color=STATION_COLOUR, lw=0.6, ls="--")


def _kind(r, settings_width=None):
    if r["swath"]:
        w = settings_width if settings_width is not None else \
            float(r["across"][-1] - r["across"][0])
        return f"swaths {w:g} m wide (mean; quartiles dark; min to max light)"
    return "profiles"


def plot_normals_grid(p: PairPath, label_a, label_b, head: str, stations=None,
                      max_panels: int = 24, ncols: int = 6, same_y=False, ylim=None):
    """Figure 3: the normal profiles (or swaths) of one path, one panel per
    station lettered a, b, c, ...; by default up to ``max_panels`` spread
    along the path. ``same_y`` shares the path's elevation range."""
    import matplotlib.pyplot as plt
    r = p.normals
    n = len(r["station_s"])
    pick = (np.asarray(stations, int) if stations is not None else
            np.unique(np.linspace(0, n - 1, min(n, max_panels)).round().astype(int)))
    rows = int(math.ceil(len(pick) / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(ncols * 2.4, rows * 1.9 + 0.8),
                             squeeze=False, sharex=True)
    fixed = ylim if ylim is not None else (normals_yrange(r) if same_y else None)
    for ax, k in zip(axes.ravel(), pick):
        _draw_section(ax, r, k)
        ax.set_title(f"{r['station_label'][k]}: {r['station_s'][k] / 1000:.1f} km", fontsize=7)
        ax.tick_params(labelsize=6)
        if fixed is not None:
            ax.set_ylim(*fixed)
    for ax in axes.ravel()[len(pick):]:
        ax.set_visible(False)
    sub = (f"; stations {r['station_label'][pick[0]]} to {r['station_label'][pick[-1]]} of {n}"
           if len(pick) < n else "")
    fig.suptitle(f"{head}: normal {_kind(r)}, every {p.normals['station_s'][0] * 2 / 1000:g} km, "
                 f"{(r['offset'][-1] - r['offset'][0]) / 1000:g} km long; x = km from the divide "
                 f"(negative in basin {label_a}, positive in basin {label_b}){sub}", fontsize=8)
    fig.supxlabel("distance from the divide, km", fontsize=8)
    fig.supylabel("elevation, m", fontsize=8)
    fig.tight_layout()
    return fig


def plot_normals_stacked(p: PairPath, label_a, label_b, head: str, ylim=None):
    """Figure 4: every normal profile of a path on one axis, coloured by its
    station along the divide, so a shared shape and departures from it
    show at a glance."""
    import matplotlib.pyplot as plt
    from matplotlib import colors
    r = p.normals
    s = r["station_s"]
    n = len(s)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    norm = colors.Normalize(vmin=float(s.min()) / 1000.0, vmax=float(s.max()) / 1000.0)
    cmap = plt.get_cmap("viridis")
    off = r["offset"] / 1000.0
    key = "mean" if r["swath"] else "z"
    for k in range(n):
        ax.plot(off, r[key][k], color=cmap(norm(s[k] / 1000.0)), lw=0.7, alpha=0.85)
    ax.axvline(0, color=STATION_COLOUR, lw=0.7, ls="--")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, label="station, km along the divide")
    ticks = np.unique(np.linspace(0, n - 1, min(n, 8)).round().astype(int))
    cb.set_ticks(s[ticks] / 1000.0)
    cb.set_ticklabels([f"{r['station_label'][k]} ({s[k] / 1000:.1f})" for k in ticks])
    ax.set_xlabel(f"distance from the divide, km (negative in basin {label_a}, positive in "
                  f"basin {label_b})")
    ax.set_ylabel("elevation, m" + (" (swath mean)" if r["swath"] else ""))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(f"{head}: all {n} normal {'swaths' if r['swath'] else 'profiles'} stacked",
                 fontsize=9)
    fig.tight_layout()
    return fig


def plot_normals_side(p: PairPath, label_a, label_b, head: str, ylim=None, fill: float = 0.9,
                      mm_per_station: float = 16.0):
    """Every normal profile of a path side by side along the divide, each
    centred on its station at its true elevation and squeezed to ``fill``
    of the spacing, with the along-divide profile faint behind them."""
    import matplotlib.pyplot as plt
    r = p.normals
    s = r["station_s"]
    n = len(s)
    spacing = float(s[0] * 2) if n else 1.0
    length = float(r["offset"][-1] - r["offset"][0]) or 1.0
    fig, ax = plt.subplots(figsize=(min(60.0, max(260.0, n * mm_per_station) / 25.4), 4.2))
    squeeze = fill * spacing / length
    ax.plot(p.s / 1000.0, p.z, color="#b8c3cc", lw=0.7, zorder=1, label="along-divide profile")
    off = r["offset"]
    for k in range(n):
        x = (s[k] + off * squeeze) / 1000.0
        if r["swath"]:
            ax.fill_between(x, r["q25"][k], r["q75"][k], color="#8fa8ba", lw=0, zorder=2)
            ax.plot(x, r["mean"][k], color=MAIN_COLOUR, lw=0.7, zorder=3)
        else:
            ax.plot(x, r["z"][k], color=MAIN_COLOUR, lw=0.7, zorder=3)
        ax.axvline(s[k] / 1000.0, color=STATION_COLOUR, lw=0.3, ls=":", zorder=0)
    ax.set_xlabel("distance along the divide, km (each section centred on its station)")
    ax.set_ylabel("elevation, m")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(f"{head}: {n} normal {'swaths' if r['swath'] else 'profiles'} drawn at "
                 f"{squeeze:.2f} of their length; basin {label_a} left of each dotted line, "
                 f"basin {label_b} right", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def normals_pdf(prof: BasinPairProfiles, path: int, out_path, per_page: int = 24,
                same_y: bool = False, ylim=None) -> int:
    """Every station of one path, ``per_page`` to a page, in one PDF."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    p = prof.paths[path]
    n = p.n_stations
    fixed = ylim if ylim is not None else (normals_yrange(p.normals) if same_y else None)
    pages = 0
    with PdfPages(str(out_path)) as pdf:
        for a in range(0, n, per_page):
            fig = prof.plot_normals(path, "grid", stations=np.arange(a, min(n, a + per_page)),
                                    ylim=fixed)
            pdf.savefig(fig)
            plt.close(fig)
            pages += 1
    return pages


# ----------------------------------------------------------------------------
# every pair of a partition
# ----------------------------------------------------------------------------

def basin_pairs(D, dem, *, labels=None, threshold=None, min_length: float = 0.0):
    """Every competing basin pair of one partition, longest divide first.

    Returns a DataFrame with ``basin_lo``, ``basin_hi``, ``n_edges``,
    ``divide_length_m`` (one cell size per edge, every path of the pair)
    and ``pair_key`` (-1 without a ``threshold``), for the pairs whose
    divide is at least ``min_length`` long. Loop over it and call
    :func:`basin_pair_profiles` and :meth:`BasinPairProfiles.save` to
    draw every pair of the partition.
    """
    import pandas as pd
    ctx = _PairContext(D, dem, labels, threshold)
    lo, hi, counts = ctx.pairs()
    keep = counts * ctx.cs >= float(min_length)
    lo, hi, counts = lo[keep], hi[keep], counts[keep]
    if ctx.threshold is not None and len(lo):
        keys = pair_key(lo, hi, ctx.threshold)
    else:
        keys = np.full(len(lo), NO_PAIR, np.int64)
    return pd.DataFrame({"basin_lo": lo.astype(np.int64), "basin_hi": hi.astype(np.int64),
                         "n_edges": counts.astype(np.int64),
                         "divide_length_m": counts * ctx.cs,
                         "pair_key": np.asarray(keys, np.int64)})

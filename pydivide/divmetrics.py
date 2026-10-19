# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Across-divide sampling and divide asymmetry.

The two ``@DIVIDEobj`` methods that read a grid on both sides of every
divide edge, ``getvalue`` and ``asymmetry``, together with the primitives
they share with the across-divide metrics.

Sides of a divide edge
----------------------
A divide edge is a cell face. :func:`edge_cells` returns the two cells it
separates in a fixed *geometric* order: the west cell of a vertical face
and the north cell of a horizontal face come first. That order is what the
grid gives; it says nothing about which basin a cell belongs to. Along a
segment that steps diagonally towards the north-west or the south-east,
the west cell of one face and the north cell of the next lie in
*opposite* basins, so any quantity defined as "first cell minus second
cell" changes sign from one edge to the next along the same segment.
Measured on a 4000 x 4000 window of a 30 m DEM of south-eastern Brazil:
the geometric first cell switches basin along 77 % of the segments,
covering 87 % of all edges. A segment mean of such a quantity mixes
+d and -d, is shrunk towards zero, and has an arbitrary sign for nearly
balanced segments.

The remedy is the one ``asymmetry.m`` itself uses: identify the two sides
by their reach-basin label (the labelled stream network mapped onto its
hillslopes, ``labelreach`` + ``mapfromnal``). The pair of labels is
constant along a segment (99.0 % of the segments on the same window; the
rest border a third basin and are skipped, exactly as MATLAB does), so
"lower label minus higher label" has one orientation per segment, and the
labels themselves can be exported so the reader knows which basin is
which. :func:`orient_by_basin` applies that rule to a cell pair; the
signed pair field of :func:`pair_metric_fields` and the ``sgn`` output of
:func:`asymmetry` are defined with it.

Direction in ``asymmetry.m``
----------------------------
Both of its methods point towards the side with the *lower* value: the
``basin`` method turns its centroid-to-centroid vector explicitly, and
the ``pixel_pairs`` method sums per-edge vectors whose components are
(bottom - top) and (left - right) over their sums (its source comment
says the opposite; the code, and MATLAB's output on Big Tujunga, say
this). Both are reproduced as the code behaves.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from pydivide.gridspec import GridSpec, assert_int64

__all__ = [
    "edge_positions", "scatter", "segment_of_position", "segment_midpoints",
    "edge_cells", "cell_xy", "sample_pair", "strict_pair_mean",
    "orient_by_basin", "orient_by_segment", "PAIR_FIELD_SUFFIXES", "pair_metric_fields",
    "getvalue", "asymmetry", "azimuth_from_north", "reach_outlet_labels",
]


# ---------------------------------------------------------------------------
# positions and segments of a SortedNetwork

def edge_positions(network) -> tuple[np.ndarray, np.ndarray]:
    """(positions, edges) for the real divide edges of a SortedNetwork.

    Position k denotes the edge between ``seg_nodes[k]`` and
    ``seg_nodes[k+1]``, mirroring MATLAB's D.IX convention where an edge
    belongs to the later of its two positions. Segment joints are not
    edges and are excluded; per-position arrays are still built at full
    length so NaN marks the joints.
    """
    a = network.seg_nodes[:-1]
    b = network.seg_nodes[1:]
    joint = np.zeros(len(network.seg_nodes), dtype=bool)
    joint[network.seg_ptr[1:] - 1] = True
    pos = np.nonzero(~joint[:-1])[0]
    e = np.stack([a[pos], b[pos]], axis=1)
    e.sort(axis=1)
    return pos, e


def scatter(values: np.ndarray, pos: np.ndarray, n_positions: int,
            n_cols: int | None = None) -> np.ndarray:
    """Place per-edge values back onto full position-length arrays."""
    if n_cols is None:
        out = np.full(n_positions, np.nan, dtype=np.float64)
        out[pos] = values
    else:
        out = np.full((n_positions, n_cols), np.nan, dtype=np.float32)
        out[pos, :] = values
    return out


def segment_of_position(network) -> np.ndarray:
    """Segment index of every position (edge slot); joints go to the
    segment they close."""
    n_pos = len(network.seg_nodes) - 1
    lengths = np.diff(network.seg_ptr)
    return np.repeat(np.arange(network.n_segments, dtype=np.int64),
                     lengths)[:n_pos]


def segment_midpoints(network, grid: GridSpec) -> tuple[np.ndarray,
                                                        np.ndarray]:
    """The point half-way along each segment (``MS.X``/``MS.Y`` of
    ``asymmetry.m``: ``interp1`` of the node coordinates at half the
    cumulative length). Single-node segments return that node."""
    x, y = grid.node_xy(network.seg_nodes)
    n_pos = len(network.seg_nodes) - 1
    d = np.hypot(np.diff(x), np.diff(y))
    joint = np.zeros(n_pos, dtype=bool)
    joint[network.seg_ptr[1:-1] - 1] = True
    d[joint] = 0.0
    cum = np.concatenate([[0.0], np.cumsum(d)])
    s0 = network.seg_ptr[:-1]
    s1 = network.seg_ptr[1:] - 1
    total = cum[s1] - cum[s0]
    target = cum[s0] + total / 2.0
    k = np.searchsorted(cum, target, side="left")
    k = np.clip(k, s0, s1)
    single = total <= 0
    k = np.where(single, s0, np.maximum(k, s0 + 1))
    # a single-node segment is its own node, not the node before it in
    # the concatenated list (which belongs to the previous segment)
    km1 = np.where(single, k, np.maximum(k - 1, 0))
    span = cum[k] - cum[km1]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(single | (span <= 0), 0.0, (target - cum[km1]) / span)
    xm = x[km1] + f * (x[k] - x[km1])
    ym = y[km1] + f * (y[k] - y[km1])
    return xm, ym


# ---------------------------------------------------------------------------
# edge geometry: the two cells each divide edge separates

def edge_cells(grid: GridSpec, edges: np.ndarray) -> tuple[np.ndarray,
                                                           np.ndarray]:
    """Cell ids on either side of each divide edge, -1 outside the grid.

    Uses the edge convention of the node grid: a node pair differing by
    ``nc+1`` separates two cells in the same row; differing by 1
    separates two cells in the same column. The first cell returned is
    the **west** cell of a vertical face and the **north** cell of a
    horizontal one. This is a geometric order only; see the module
    docstring and :func:`orient_by_basin` before subtracting anything.
    """
    assert_int64(edges)
    w = np.int64(grid.nc + 1)
    a = edges[:, 0]
    b = edges[:, 1]
    I = a // w
    J = a % w
    d = b - a

    r1 = np.empty(len(a), dtype=np.int64)
    c1 = np.empty(len(a), dtype=np.int64)
    r2 = np.empty(len(a), dtype=np.int64)
    c2 = np.empty(len(a), dtype=np.int64)

    vert = d == w          # separates (I, J-1) | (I, J)
    r1[vert] = I[vert]
    c1[vert] = J[vert] - 1
    r2[vert] = I[vert]
    c2[vert] = J[vert]

    horz = d == 1          # separates (I-1, J) / (I, J)
    r1[horz] = I[horz] - 1
    c1[horz] = J[horz]
    r2[horz] = I[horz]
    c2[horz] = J[horz]

    if not np.all(vert | horz):
        raise ValueError("edge list contains non-adjacent node pairs")

    def to_cell(r, c):
        ok = (r >= 0) & (r < grid.nr) & (c >= 0) & (c < grid.nc)
        out = np.full(len(r), -1, dtype=np.int64)
        out[ok] = r[ok] * np.int64(grid.nc) + c[ok]
        return out

    return to_cell(r1, c1), to_cell(r2, c2)


def _is_vertical(grid: GridSpec, edges: np.ndarray) -> np.ndarray:
    return (edges[:, 1] - edges[:, 0]) == np.int64(grid.nc + 1)


def cell_xy(grid: GridSpec, cells: np.ndarray) -> tuple[np.ndarray,
                                                        np.ndarray]:
    """World coordinates of cell centres (NaN for cell id -1)."""
    cells = np.asarray(cells, dtype=np.int64)
    I = cells // np.int64(grid.nc)
    J = cells % np.int64(grid.nc)
    x = grid.x0 + (J.astype(np.float64) + 0.5) * grid.cellsize
    y = grid.y0 - (I.astype(np.float64) + 0.5) * grid.cellsize
    bad = cells < 0
    x[bad] = np.nan
    y[bad] = np.nan
    return x, y


def sample_pair(values_flat: np.ndarray, cell_a: np.ndarray,
                cell_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample a flattened raster on both sides of each edge; NaN outside."""
    out = []
    for cells in (cell_a, cell_b):
        v = np.full(len(cells), np.nan, dtype=np.float64)
        ok = cells >= 0
        v[ok] = values_flat[cells[ok]]
        out.append(v)
    return out[0], out[1]


def strict_pair_mean(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Mean of two sides, NaN unless both are finite (the strict
    convention; ``getvalue`` itself is one-sided-tolerant, see there)."""
    out = np.full(p.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(q)
    out[ok] = (p[ok] + q[ok]) / 2.0
    return out


def orient_by_basin(cell_a: np.ndarray, cell_b: np.ndarray,
                    label_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray]:
    """Reorder each cell pair so that the first cell carries the **lower
    reach-basin label**.

    Returns ``(cell_a, cell_b, label_a, label_b)`` with ``label_a <=
    label_b`` wherever both sides are valid (labels ``-1`` outside the
    grid or in unlabelled cells are left in place). After this call
    "side A minus side B" has a single orientation along every segment
    that separates exactly two basins, which is the only case in which
    a signed across-divide quantity means anything.
    """
    la = np.full(len(cell_a), -1, dtype=np.int64)
    lb = np.full(len(cell_b), -1, dtype=np.int64)
    ok = cell_a >= 0
    la[ok] = label_flat[cell_a[ok]]
    ok = cell_b >= 0
    lb[ok] = label_flat[cell_b[ok]]
    swap = (la >= 0) & (lb >= 0) & (la > lb)
    ca = np.where(swap, cell_b, cell_a)
    cb = np.where(swap, cell_a, cell_b)
    la2 = np.where(swap, lb, la)
    lb2 = np.where(swap, la, lb)
    return ca, cb, la2, lb2


def orient_by_segment(grid: GridSpec, edges: np.ndarray, cell_a: np.ndarray,
                      cell_b: np.ndarray, label_flat: np.ndarray,
                      forward: np.ndarray, seg: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Segment-consistent side rule: side A is the same physical side of
    the divide along the whole segment.

    :func:`orient_by_basin` orders each edge's two cells by reach-basin
    label. Reach basins are finer than drainage basins, so along a
    segment between basins X and Y the label on the X side can change at
    a confluence inside X without any junction on the divide (1 % of the
    segments of the Brazil 4000² window), and a per-edge label order can
    then swap sides mid-segment. Here the side is fixed by the walking
    direction instead. Walking a segment in its stored direction, the
    left-hand cell of every face is on the same side of the divide,
    through corners and label changes alike. Side A is the left side if
    the label there is lower than the label on the right at the
    segment's first valid edge, and the right side otherwise, for every
    edge of that segment. On segments whose label pair is constant this
    gives exactly the lower-label rule.

    Parameters: ``edges`` sorted node pairs (a < b) aligned with
    ``cell_a``/``cell_b`` from :func:`edge_cells`; ``forward`` True where
    the segment traverses the edge from node a to node b; ``seg`` the
    segment index of each edge. Returns ``(cell_a, cell_b, label_a,
    label_b)`` like :func:`orient_by_basin`.
    """
    w = np.int64(grid.nc + 1)
    vertical = (edges[:, 1] - edges[:, 0]) == w
    # walking a vertical face from its upper node (a) to its lower node (b)
    # is southward and the left-hand cell is the east one (cell_b);
    # walking a horizontal face from a to b is eastward and the left-hand
    # cell is the north one (cell_a)
    left_is_a = np.where(vertical, ~forward, forward)
    left = np.where(left_is_a, cell_a, cell_b)
    right = np.where(left_is_a, cell_b, cell_a)
    ll = np.full(len(left), -1, dtype=np.int64)
    lr = np.full(len(right), -1, dtype=np.int64)
    ok = left >= 0
    ll[ok] = label_flat[left[ok]]
    ok = right >= 0
    lr[ok] = label_flat[right[ok]]
    valid = (ll >= 0) & (lr >= 0) & (ll != lr)
    n_seg = int(seg.max()) + 1 if len(seg) else 0
    # first valid edge of each segment, in edge order
    order = np.arange(len(seg))
    first = np.full(n_seg, -1, dtype=np.int64)
    idx_valid = order[valid]
    seg_valid = seg[valid]
    # np.unique returns the first occurrence per segment
    u, i0 = np.unique(seg_valid, return_index=True)
    first[u] = idx_valid[i0]
    flip_seg = np.zeros(n_seg, dtype=bool)
    has = first >= 0
    flip_seg[has] = ll[first[has]] > lr[first[has]]
    flip = flip_seg[seg]
    ca = np.where(flip, right, left)
    cb = np.where(flip, left, right)
    la = np.where(flip, lr, ll)
    lb = np.where(flip, ll, lr)
    return ca, cb, la, lb


# ---------------------------------------------------------------------------
# pair fields

PAIR_FIELD_SUFFIXES = ("mean", "diff", "min", "max", "sgn")


def pair_metric_fields(p: np.ndarray, q: np.ndarray) -> dict:
    """The per-edge pair fields (mean, |difference|, min, max) as in
    ``DIVIDEobj2mapstruct``'s aggregation of ``getvalue``, plus the
    **signed** difference ``p - q``.

    ``p`` must be side A and ``q`` side B of every edge as fixed by
    :func:`build_edge_table` (one physical side per segment, chosen so
    that side A carries the lower reach-basin label at the segment's
    first valid edge; :func:`orient_by_segment`). With a geometric side
    order the sign alternates along most segments (module docstring).
    All four unsigned fields are symmetric in ``p`` and ``q``.
    """
    ok = np.isfinite(p) & np.isfinite(q)
    mean = np.full(p.shape, np.nan)
    diff = np.full(p.shape, np.nan)
    lo = np.full(p.shape, np.nan)
    hi = np.full(p.shape, np.nan)
    sgn = np.full(p.shape, np.nan)
    mean[ok] = (p[ok] + q[ok]) / 2.0
    diff[ok] = np.abs(p[ok] - q[ok])
    lo[ok] = np.minimum(p[ok], q[ok])
    hi[ok] = np.maximum(p[ok], q[ok])
    sgn[ok] = p[ok] - q[ok]
    out = {"mean": mean, "diff": diff, "min": lo, "max": hi, "sgn": sgn}
    assert tuple(out) == PAIR_FIELD_SUFFIXES, "PAIR_FIELD_SUFFIXES is stale"
    return out


# ---------------------------------------------------------------------------
# getvalue

def getvalue(network, grid: GridSpec, values_flat: np.ndarray,
             fct: str | None = None, *, label_flat: np.ndarray | None = None):
    """``getvalue(D, GRID[, fct])``: the grid values of the two cells
    adjacent to every divide edge, per position (NaN at segment joints).

    Without ``fct`` returns ``(p, q)``. With ``label_flat`` given, ``p``
    is the side with the lower reach-basin label (recommended; see the
    module docstring). Without it, ``p`` is MATLAB's ``pix`` side, the
    top cell of a horizontal face and the right cell of a vertical one,
    and ``q`` its ``qix`` side.

    With ``fct`` in ``{'min','max','mean','diff','normdiff'}`` returns one
    value per edge, with MATLAB's ``omitmissing`` semantics for
    ``min``/``max``/``mean`` (a one-sided value survives) and NaN for
    ``diff``/``normdiff`` when either side is missing.
    """
    pos, edges = edge_positions(network)
    n_pos = len(network.seg_nodes) - 1
    ca, cb = edge_cells(grid, edges)
    if label_flat is not None:
        ca, cb, _, _ = orient_by_basin(ca, cb, label_flat)
    else:
        vert = _is_vertical(grid, edges)
        ca, cb = np.where(vert, cb, ca), np.where(vert, ca, cb)
    p, q = sample_pair(values_flat, ca, cb)
    if fct is None:
        return scatter(p, pos, n_pos), scatter(q, pos, n_pos)
    both = np.stack([p, q], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows
        if fct == "mean":
            r = np.nanmean(both, axis=1)
        elif fct == "max":
            r = np.nanmax(both, axis=1)
        elif fct == "min":
            r = np.nanmin(both, axis=1)
        elif fct == "diff":
            r = np.abs(p - q)
        elif fct == "normdiff":
            r = np.abs(p - q) / np.nanmean(both, axis=1)
        else:
            raise ValueError("fct must be one of min, max, mean, diff, "
                             f"normdiff; got {fct!r}")
    return scatter(r, pos, n_pos)


# ---------------------------------------------------------------------------
# asymmetry

def azimuth_from_north(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Clockwise angle from north, in degrees in [0, 360), of the vector
    (u east, v north): MATLAB's ``cart2pol`` followed by ``-theta+90`` and
    the wrap. ``(0, 0)`` gives 90, as in MATLAB."""
    with np.errstate(invalid="ignore"):
        theta = 90.0 - np.degrees(np.arctan2(v, u))
    theta = np.where(theta < 0, theta + 360.0, theta)
    return theta


def _grouped(values: np.ndarray, group: np.ndarray, n_groups: int,
             aggfun) -> np.ndarray:
    """Aggregate ``values`` per group with NaNs omitted.

    ``aggfun`` is ``"mean"``, ``"median"`` or a callable taking a 1-D
    array of finite values (the callable path loops over groups)."""
    out = np.full(n_groups, np.nan, dtype=np.float64)
    ok = np.isfinite(values) & (group >= 0)
    v = values[ok]
    g = group[ok]
    if len(v) == 0:
        return out
    if aggfun == "mean":
        s = np.bincount(g, weights=v, minlength=n_groups)
        c = np.bincount(g, minlength=n_groups)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s / c
        out[c > 0] = m[c > 0]
        return out
    order = np.lexsort((v, g))
    g = g[order]
    v = v[order]
    starts = np.searchsorted(g, np.arange(n_groups), side="left")
    ends = np.searchsorted(g, np.arange(n_groups), side="right")
    if aggfun == "median":
        n = ends - starts
        has = n > 0
        lo = starts + (n - 1) // 2
        hi = starts + n // 2
        lo = np.where(has, lo, 0)
        hi = np.where(has, hi, 0)
        med = (v[lo] + v[hi]) / 2.0
        out[has] = med[has]
        return out
    if not callable(aggfun):
        raise ValueError("aggfun must be 'mean', 'median' or a callable")
    for k in np.nonzero(ends > starts)[0]:
        out[k] = float(aggfun(v[starts[k]:ends[k]]))
    return out


def asymmetry(network, grid: GridSpec, values_flat: np.ndarray, *,
              label_flat: np.ndarray | None = None,
              method: str = "basin", aggfun="mean",
              clip_negative: bool = True,
              order: np.ndarray | None = None,
              distance: np.ndarray | None = None) -> dict:
    """``asymmetry(D, GRID, FD, S, method=..., aggfun=...)``: across-divide
    asymmetry of a grid, one record per segment.

    Parameters
    ----------
    network : SortedNetwork
    grid : GridSpec
    values_flat : flattened grid (e.g. hillslope relief). As in MATLAB,
        negative values are set to 0 and infinities to NaN before use
        (``clip_negative=False`` disables the first).
    label_flat : flattened reach-basin labels; required for ``"basin"``.
    method : ``"basin"`` (side identity by basin label; skips segments
        bordering fewer or more than two basins) or ``"pixel_pairs"``
        (per-edge vectors summed). MATLAB's default is ``"pixel_pairs"``;
        the default here is ``"basin"`` because it is the method whose
        side identity is well defined (module docstring).
    aggfun : ``"mean"`` (MATLAB default), ``"median"``, or a callable.
    order, distance : optional per-node arrays (``len(seg_nodes)``)
        whose segment means are returned as in ``MS.order``/``MS.dist``.

    Returns
    -------
    dict of per-segment arrays: ``x``, ``y`` (midpoint along the
    segment), ``rho`` (magnitude), ``theta`` (azimuth from north, degrees,
    towards the *lower* side under both methods, as in MATLAB), ``dai`` (divide asymmetry
    index, ``"basin"`` only), ``u``, ``v`` (vector components, east and
    north; ``"pixel_pairs"`` only), and for ``"basin"`` the extension
    that makes the sign usable: ``side_lo``/``side_hi`` (the two
    reach-basin labels), ``value_lo``/``value_hi`` (their aggregated
    values) and ``sgn = value_lo - value_hi``; ``n_basins`` counts the
    basins a segment borders. ``order``/``dist`` when given.
    """
    n_seg = network.n_segments
    pos, edges = edge_positions(network)
    seg = segment_of_position(network)[pos]
    ca, cb = edge_cells(grid, edges)
    va, vb = sample_pair(values_flat, ca, cb)
    for v in (va, vb):
        v[np.isinf(v)] = np.nan
        if clip_negative:
            v[v < 0] = 0.0

    xm, ym = segment_midpoints(network, grid)
    out = {"x": xm, "y": ym,
           "rho": np.full(n_seg, np.nan), "theta": np.full(n_seg, np.nan)}
    if order is not None:
        out["order"] = _node_mean(network, np.asarray(order,
                                                     dtype=np.float64))
    if distance is not None:
        out["dist"] = _node_mean(network, np.asarray(distance,
                                                    dtype=np.float64))

    if method == "pixel_pairs":
        vert = _is_vertical(grid, edges)
        p = np.where(vert, vb, va)      # right | top   (MATLAB pix)
        q = np.where(vert, va, vb)      # left  | bottom (MATLAB qix)
        # MATLAB: dhr = diff([hr1, hr2], 1, 2) = hr2 - hr1 = q - p, i.e.
        # bottom minus top and left minus right (its comment says the
        # opposite). The vector therefore points towards the LOWER side,
        # like the basin method. Verified against MATLAB output on Big
        # Tujunga (parity comparison against the MATLAB export).
        with np.errstate(invalid="ignore", divide="ignore"):
            dn = (q - p) / (p + q)
        u = np.where(vert, 1.0, 0.0) * dn       # NaN propagates, as in MATLAB
        v = np.where(vert, 0.0, 1.0) * dn
        us = _grouped(u, seg, n_seg, aggfun)
        vs = _grouped(v, seg, n_seg, aggfun)
        out["u"] = us
        out["v"] = vs
        out["rho"] = np.hypot(us, vs)
        out["theta"] = azimuth_from_north(us, vs)
        return out

    if method != "basin":
        raise ValueError("method must be 'basin' or 'pixel_pairs'")
    if label_flat is None:
        raise ValueError("method='basin' needs label_flat")

    # pool both sides of every edge, keeping multiplicity as MATLAB does
    cells = np.concatenate([ca, cb])
    vals = np.concatenate([va, vb])
    sg = np.concatenate([seg, seg])
    lab = np.full(len(cells), -1, dtype=np.int64)
    okc = cells >= 0
    lab[okc] = label_flat[cells[okc]]
    valid = lab >= 0

    # distinct labels per segment
    base = np.int64(int(lab.max()) + 2) if valid.any() else np.int64(2)
    key = np.unique(sg[valid] * base + lab[valid])
    seg_u = key // base
    lab_u = key % base
    n_basins = np.bincount(seg_u, minlength=n_seg)
    two = n_basins == 2
    starts = np.searchsorted(seg_u, np.arange(n_seg), side="left")
    side_lo = np.full(n_seg, -1, dtype=np.int64)
    side_hi = np.full(n_seg, -1, dtype=np.int64)
    idx = starts[two]
    side_lo[two] = lab_u[idx]
    side_hi[two] = lab_u[idx + 1]

    # side membership of every pooled sample
    grp = np.full(len(lab), -1, dtype=np.int64)
    m = valid & two[sg]
    is_lo = m & (lab == side_lo[sg])
    is_hi = m & (lab == side_hi[sg])
    grp[is_lo] = 2 * sg[is_lo]
    grp[is_hi] = 2 * sg[is_hi] + 1

    agg = _grouped(vals, grp, 2 * n_seg, aggfun)
    v1 = agg[0::2]
    v2 = agg[1::2]
    cx, cy = cell_xy(grid, cells)
    mx = _grouped(cx, grp, 2 * n_seg, "mean")
    my = _grouped(cy, grp, 2 * n_seg, "mean")
    x_conn = mx[0::2] - mx[1::2]
    y_conn = my[0::2] - my[1::2]

    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.abs(v1 - v2)
        dai = np.abs(v1 - v2) / (v1 + v2)
    dai[~np.isfinite(dai)] = np.nan
    theta = azimuth_from_north(x_conn, y_conn)
    flip = (v1 - v2) > 0
    theta = np.where(flip, np.where(theta > 180, theta - 180, theta + 180),
                     theta)
    theta[~two | ~np.isfinite(v1) | ~np.isfinite(v2)] = np.nan
    rho[~two] = np.nan
    dai[~two] = np.nan

    out.update({"rho": rho, "theta": theta, "dai": dai,
                "side_lo": side_lo, "side_hi": side_hi,
                "value_lo": v1, "value_hi": v2, "sgn": v1 - v2,
                "n_basins": n_basins})
    return out


def _node_mean(network, per_node: np.ndarray) -> np.ndarray:
    """Segment mean of a per-node array (NaNs omitted)."""
    seg_of_node = np.repeat(np.arange(network.n_segments, dtype=np.int64),
                            np.diff(network.seg_ptr))
    return _grouped(per_node, seg_of_node, network.n_segments, "mean")


# ---------------------------------------------------------------------------
# which outlet does each reach-basin drain to?

def reach_outlet_labels(label_flat: np.ndarray, st_src: np.ndarray,
                        st_dst: np.ndarray,
                        n_labels: int | None = None) -> np.ndarray:
    """For every reach-basin label, the label of the **outlet reach** its
    water leaves the network through.

    Reach-basin labels partition the landscape finely (one label per
    stream reach). Two divide sides with different reach labels may still
    drain to the same outlet (a divide between tributaries of one river)
    or to different outlets (a divide between river systems); that
    distinction is what a signed across-divide contrast is usually read
    for, so the export carries both the reach labels and their outlets.

    The reach graph is read off the stream edges: an edge whose two cells
    carry different labels links the upstream reach to the downstream
    one. Every reach then follows its chain to a reach with no downstream
    reach, by pointer jumping (O(log depth) passes). Returns an array
    indexed by label; label 0 (unlabelled) and reaches without stream
    edges map to themselves. Memory is one int64 per label, not per cell.
    ``n_labels`` (default: one more than the largest label in
    ``label_flat``, a single pass over the grid) sizes the result.

    The reach graph of a routed DEM is a forest, on which the pointer
    jumping settles within log2(number of reaches) + 1 passes onto the
    reaches that have no downstream reach. A graph with a cycle either
    never settles or, when the cycle's length is a power of two,
    settles onto members of the cycle; so the passes are bounded and,
    after they settle, every resolved target must be a reach that had
    no downstream reach to begin with. Anything else is rejected with
    ``ValueError`` rather than looped on or collapsed onto a false
    outlet.
    """
    ls = label_flat[st_src]
    ld = label_flat[st_dst]
    if n_labels is None:
        n_labels = int(label_flat.max(initial=0)) + 1
    n = max(int(n_labels), int(max(ls.max(initial=0), ld.max(initial=0))) + 1)
    down = np.arange(n, dtype=np.int64)
    cross = (ls != ld) & (ls > 0) & (ld > 0)
    down[ls[cross]] = ld[cross]
    is_outlet = np.ones(n, dtype=bool)          # a reach with no downstream reach
    is_outlet[ls[cross]] = False
    max_passes = int(np.ceil(np.log2(max(n, 2)))) + 2
    for _ in range(max_passes):
        nxt = down[down]
        if np.array_equal(nxt, down):
            if not np.all(is_outlet[down]):
                raise ValueError(
                    "the reach graph contains a cycle: the outlet propagation settled on "
                    f"{int((~is_outlet[down]).sum())} labels that are not outlet reaches; "
                    "the stream edges do not describe a tree")
            return down
        down = nxt
    raise ValueError(
        f"the reach graph contains a cycle: the outlet propagation did not settle in "
        f"{max_passes} passes over {n} labels, which a forest always does; the stream "
        "edges do not describe a tree")


# ---------------------------------------------------------------------------
# the per-edge table and the piece aggregation, which the across-divide
# call and the basin-pair profiles need

@dataclass
class EdgeTable:
    """Per-edge quantities, aligned with a divide edge list."""
    x: np.ndarray            # edge midpoint
    y: np.ndarray
    z: np.ndarray            # mean adjacent elevation (NaN if either side)
    cell_a: np.ndarray       # int64 cell ids, -1 outside
    cell_b: np.ndarray
    pair_lo: np.ndarray      # int64 reach-basin labels, -1 invalid
    pair_hi: np.ndarray
    distance: np.ndarray     # divide distance at the edge
    lab_a: np.ndarray | None = None   # label on side A (cell_a); == pair_lo
    lab_b: np.ndarray | None = None   # except on segments whose pair changes

    @property
    def has_pair(self) -> np.ndarray:
        return ((self.pair_lo > 0) & (self.pair_hi > 0)
                & (self.pair_lo != self.pair_hi) & np.isfinite(self.z))


def build_edge_table(grid: GridSpec, edges: np.ndarray, dem_flat: np.ndarray,
                     label_flat: np.ndarray,
                     distance: np.ndarray | None = None, *,
                     forward: np.ndarray | None = None,
                     seg: np.ndarray | None = None) -> EdgeTable:
    """Assemble the per-edge table used by every metric.

    ``cell_a`` is side A and ``cell_b`` side B of every edge. With
    ``forward`` (True where the segment traverses the edge from its
    lower to its higher node id) and ``seg`` (segment index per edge)
    given, side A is fixed per segment by walking direction and the
    labels at the first edge (``orient_by_segment``), so it is the same
    physical side along the whole segment even where the reach-basin
    label on one side changes at a confluence. Without them, side A is
    the lower reach-basin label edge by edge (``orient_by_basin``),
    which is the same thing on the 99 % of segments whose label pair is
    constant. ``lab_a``/``lab_b`` carry the labels of the two sides;
    ``pair_lo``/``pair_hi`` remain min/max, as the basin-pair metrics
    expect. Where a side is outside the grid the geometric order
    (west/north first) is kept; such edges have no valid pair anyway.
    """
    ca, cb = edge_cells(grid, edges)
    if forward is not None and seg is not None:
        ca, cb, lp, lq = orient_by_segment(grid, edges, ca, cb, label_flat,
                                           np.asarray(forward, dtype=bool),
                                           np.asarray(seg, dtype=np.int64))
    else:
        ca, cb, lp, lq = orient_by_basin(ca, cb, label_flat)
    zp, zq = sample_pair(dem_flat, ca, cb)
    z = strict_pair_mean(zp, zq)

    pair_lo = np.minimum(lp, lq)
    pair_hi = np.maximum(lp, lq)
    # an invalid side makes the pair invalid
    bad = (lp < 0) | (lq < 0)
    pair_lo[bad] = -1
    pair_hi[bad] = -1

    # midpoint of the two divide nodes = midpoint of the shared cell face
    xa, ya = grid.node_xy(edges[:, 0])
    xb, yb = grid.node_xy(edges[:, 1])
    if distance is None:
        distance = np.full(len(ca), np.nan)

    return EdgeTable(x=(xa + xb) / 2.0, y=(ya + yb) / 2.0, z=z,
                     cell_a=ca, cell_b=cb, pair_lo=pair_lo,
                     pair_hi=pair_hi, distance=distance,
                     lab_a=np.where(bad, -1, lp), lab_b=np.where(bad, -1, lq))


def aggregate_to_pieces(values: np.ndarray, p_start: np.ndarray,
                        p_stop: np.ndarray, how: str = "mean"
                        ) -> np.ndarray:
    """Aggregate per-edge-position values onto export pieces.

    ``values`` is aligned with edge positions (length len(seg_nodes)-1,
    NaN at segment joints); a piece spanning nodes [p_start, p_stop)
    covers edge positions [p_start, p_stop-1). NaNs are ignored, and a
    piece with no finite value yields NaN.
    """
    n = len(p_start)
    out = np.full(n, np.nan, dtype=np.float64)
    lens = p_stop - p_start - 1
    if n == 0:
        return out
    valid = lens > 0
    idx = np.nonzero(valid)[0]
    if idx.size == 0:
        return out

    starts = p_start[idx]
    stops = p_stop[idx] - 1
    counts = stops - starts
    total = int(counts.sum())
    offs = np.zeros(len(idx), dtype=np.int64)
    np.cumsum(counts[:-1], out=offs[1:])
    pos = (np.repeat(starts, counts)
           + np.arange(total, dtype=np.int64)
           - np.repeat(offs, counts))
    v = values[pos]
    fin = np.isfinite(v)

    cnt = np.add.reduceat(fin.astype(np.int64), offs)
    if how == "mean":
        s = np.add.reduceat(np.where(fin, v, 0.0), offs)
        with np.errstate(invalid="ignore", divide="ignore"):
            res = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    elif how == "max":
        res = np.maximum.reduceat(np.where(fin, v, -np.inf), offs)
        res = np.where(cnt > 0, res, np.nan)
    elif how == "min":
        res = np.minimum.reduceat(np.where(fin, v, np.inf), offs)
        res = np.where(cnt > 0, res, np.nan)
    else:
        raise ValueError(f"unknown aggregation {how!r}")
    out[idx] = res
    return out

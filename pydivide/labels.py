# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Stage 1 — basin labels.

Reproduces MATLAB `labelreach(ST)` + `mapfromnal(FD, ST, labels)`:
every cell is assigned the label of the *first* stream node its flowpath
meets; stream nodes are labelled per reach (section between confluences,
numbered exactly as MATLAB does so the comparison can be exact rather
than permutation-based).

Both propagations use pointer doubling (path halving) instead of MATLAB's
sequential reverse-topological sweep: stream cells are absorbing fixed
points, so ``ptr`` converges to the first stream cell downstream of every
cell in O(log(longest hillslope path)) fully vectorised passes. This is
O(N log N) worst case, and is
order-equivalent to the MATLAB sweep (the sweep also delivers the final
downstream value because givers precede receivers in topological order).
"""

from __future__ import annotations

import numpy as np

from .flownet import FlowNet
from .gridspec import assert_int64

_CHUNK = 64_000_000     # elements per in-place jump chunk (~0.5 GB temp)


def _absorbing_pointer_fixpoint(ptr: np.ndarray,
                                absorbing: np.ndarray | None = None) -> np.ndarray:
    """Iterate pointer jumping (ptr[k] <- ptr[ptr[k]]) until every chain
    reaches an absorbing (self-pointing) element. ``ptr`` must contain
    only valid indices into itself; absorbing states point to themselves.
    ``absorbing``, a boolean mask of the states that are allowed to be
    absorbing, turns a cycle (which pointer jumping collapses onto a
    self-pointing element that is not absorbing) into a ``ValueError``.

    Updates are performed *in place* in chunks: reading partially updated
    values is safe (any ptr value only ever moves further along its
    chain) and accelerates convergence; peak extra memory is one chunk,
    not a full copy — essential at full-domain size (a full int64 copy of
    the cell grid is ~97 GB).
    """
    assert_int64(ptr)
    n = len(ptr)
    # pointer doubling halves the remaining chain length every pass, so an
    # acyclic graph converges in at most log2(n) + 1 passes; more passes
    # means a cycle, which a routed DEM cannot contain but a custom graph can
    max_passes = int(np.ceil(np.log2(max(n, 2)))) + 2
    for _ in range(max_passes):
        changed = False
        for s in range(0, n, _CHUNK):
            e = min(s + _CHUNK, n)
            nxt = ptr[ptr[s:e]]
            if not changed and not np.array_equal(nxt, ptr[s:e]):
                changed = True
            ptr[s:e] = nxt
        if not changed:
            if absorbing is not None:
                for s in range(0, n, _CHUNK):
                    e = min(s + _CHUNK, n)
                    if not np.all(absorbing[ptr[s:e]]):
                        raise ValueError("flow graph contains a cycle: a chain "
                                         "ends on a state that is not absorbing")
            return ptr
    raise ValueError("flow graph contains a cycle: pointer jumping did not converge")


def labelreach(net: FlowNet) -> np.ndarray:
    """Reach labels for stream nodes, in NAL order (int64, 1-based labels).

    Mirrors MATLAB ``labelreach(S)`` (default options): reaches split at
    confluences; the marker nodes are b-confluences (nodes immediately
    upstream of a confluence) and outlets, numbered by NAL position;
    every node inherits the marker label of the first marker at-or-
    downstream of it; nodes with no edges get fresh labels.
    """
    n = net.st_nodes.size
    # stream edge endpoints as NAL positions
    order = np.argsort(net.st_nodes, kind="stable")
    sorted_nodes = net.st_nodes[order]

    def to_pos(cells: np.ndarray) -> np.ndarray:
        loc = np.searchsorted(sorted_nodes, cells)
        bad = loc >= sorted_nodes.size
        if np.any(bad) or np.any(sorted_nodes[np.minimum(loc, sorted_nodes.size - 1)] != cells):
            raise ValueError("stream edge endpoint not in stream node list")
        return order[loc]

    src = to_pos(net.st_src)
    dst = to_pos(net.st_dst)

    indeg = np.bincount(dst, minlength=n)
    has_out = np.zeros(n, dtype=bool)
    has_out[src] = True
    has_in = indeg > 0

    confluence = indeg > 1
    bconfl = np.zeros(n, dtype=bool)
    bconfl[src[confluence[dst]]] = True
    outlet = ~has_out & has_in
    marked = bconfl | outlet

    label = np.where(marked, np.cumsum(marked), 0).astype(np.int64)

    # propagate: unmarked nodes take the label of the first marked node
    # at-or-downstream; marked nodes are absorbing
    ptr = np.arange(n, dtype=np.int64)
    unmarked_src = ~marked[src]
    ptr[src[unmarked_src]] = dst[unmarked_src]
    # an unmarked node with no downstream edge is its own end point
    absorbing = marked.copy()
    absorbing[~has_out] = True
    ptr = _absorbing_pointer_fixpoint(ptr, absorbing)
    label = label[ptr]

    # nodes never labelled (isolated): fresh labels in NAL order
    iszero = label == 0
    label[iszero] = np.arange(1, int(iszero.sum()) + 1,
                              dtype=np.int64) + label.max(initial=0)
    return label


def first_stream_cell_downstream(net: FlowNet) -> np.ndarray:
    """For every cell: the cell id of the first stream cell on its
    downstream flowpath (itself if it is a stream cell), or -1 if the
    flowpath never meets a stream. int64, shape (n_cells,).

    This is the propagation kernel of MATLAB ``mapfromnal`` /
    ``propagatevaluesupstream(..., 'overwrite', false)``.
    """
    n_cells = int(net.grid.n_cells)
    sentinel = np.int64(n_cells)
    ptr = np.full(n_cells + 1, sentinel, dtype=np.int64)
    ptr[net.fd_src] = net.fd_dst
    ptr[net.st_nodes] = net.st_nodes   # absorbing
    ptr[sentinel] = sentinel           # absorbing "nowhere"
    is_stream = np.zeros(n_cells + 1, dtype=bool)
    is_stream[net.st_nodes] = True
    absorbing = is_stream.copy()
    absorbing[sentinel] = True
    ptr = _absorbing_pointer_fixpoint(ptr, absorbing)
    del absorbing

    result = ptr[:n_cells]
    # in-place: cells whose absorbing state is not a stream cell get -1
    for s in range(0, n_cells, _CHUNK):
        e = min(s + _CHUNK, n_cells)
        blk = result[s:e]
        blk[~is_stream[blk]] = -1
    return result


def check_stream_threshold(value, name: str = "threshold") -> float:
    """A channel-initiation threshold as a float, refused unless it is a
    finite positive number: NaN or infinity would give an empty stream
    network and zero or a negative value a stream through every cell,
    both silently."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}") from None
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return v


def reach_basin_labels(dem, threshold: float, *, fd=None, units: str = "m2") -> np.ndarray:
    """The reach-basin label raster of one partition of the stream network.

    Routes ``dem`` with pytopotoolbox (or takes the ``FlowObject`` ``fd``
    already built from it), delineates the streams above ``threshold``
    (in ``units``: ``"m2"`` of drainage area or ``"pixels"``) and labels
    every cell with the reach its flow path first meets, -1 where it meets
    none. This is the labelling that names the basin pairs of that
    partition: pass it to :func:`pydivide.basin_pair_profiles` or
    :func:`pydivide.basin_pairs` as ``labels`` with the same ``threshold``.
    A network built at 10⁶ m² keeps every edge at 10⁷, 10⁸ and 10⁹;
    only the pairs it separates change.
    """
    import topotoolbox as tt
    threshold = check_stream_threshold(threshold)
    if fd is None:
        fd = tt.FlowObject(dem)
    else:
        # a flow object built from another grid would label the wrong
        # place, and a label raster carries no transform to reveal it
        from .gridspec import check_same_grid
        check_same_grid(dem, "dem", fd, "fd")
    st = tt.StreamObject(fd, threshold=threshold, units=units)
    net = FlowNet.from_pytopotoolbox(fd, st)
    return basin_labels(net)[0]


def basin_labels(net: FlowNet) -> tuple[np.ndarray, np.ndarray]:
    """Stage 1 output: (L, nal).

    L is an int32 (nr, nc) raster of reach-basin labels, -1 where the
    cell's flowpath never meets a stream — exactly the semantics of
    MATLAB ``mapfromnal(FD, ST, labelreach(ST))`` with -1 in place of
    NaN. nal is the labelreach node-attribute list (int64). Labels are
    counts of reaches, far below 2^31 even at full-domain scale
    so the int32 raster halves the largest allocation
    of the stage. Cell/node *indices* remain int64 everywhere.
    """
    nal = labelreach(net)
    if nal.max(initial=0) >= np.iinfo(np.int32).max:
        raise OverflowError("reach label exceeds int32 — investigate")
    tgt = first_stream_cell_downstream(net)

    n_cells = int(net.grid.n_cells)
    # NAL value of each stream cell, via sorted lookup (no full-grid
    # position array: stream cells are ~1% of the grid)
    order = np.argsort(net.st_nodes, kind="stable")
    sorted_nodes = net.st_nodes[order]
    nal_sorted = nal[order].astype(np.int32)

    L = np.full(n_cells, -1, dtype=np.int32)
    for s in range(0, n_cells, _CHUNK):
        e = min(s + _CHUNK, n_cells)
        blk = tgt[s:e]
        ok = blk >= 0
        loc = np.searchsorted(sorted_nodes, blk[ok])
        L[s:e][ok] = nal_sorted[loc]
    return L.reshape(net.grid.nr, net.grid.nc), nal

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Supporting rasters the divide metrics need but pytopotoolbox lacks.

Several across-divide metrics require grids that ``FlowObject`` does
not provide:

``flowdistance``  distance along the flow path to the first stream cell
                  downstream (MATLAB ``flowdistance(FD, S)``)
``arcslope``      slope in degrees by the ArcGIS/Horn 3x3 method
                  (MATLAB ``arcslope(DEM, 'degree')``)
``upslopestats``  upslope mean of a grid (MATLAB
                  ``upslopestats(FD, SLP, 'mean', S)``)
``chi on hillslopes``
                  a stream node-attribute list propagated to every cell
                  (MATLAB ``mapfromnal(FD, S, chi)``)

Hillslope relief (``vertdistance2stream``) *is* available in
pytopotoolbox and is used directly.

Everything here is vectorised. The flow-distance accumulation uses the
same pointer-doubling scheme as the Stage 1 labels, carrying a path
weight, so it is O(N log L) rather than a per-cell walk.
"""

from __future__ import annotations

import numpy as np

from pydivide.flownet import FlowNet
from pydivide.gridspec import assert_int64
from pydivide.labels import _CHUNK

#: Rows per block in arcslope. ~1024 rows of a 116k-wide DEM is ~0.5 GB
#: per temporary, so the whole kernel stays a few GB whatever the DEM.
_SLOPE_BLOCK = 1024

#: Edges per chunk in the low-memory flow kernels below. 2e8 edges is
#: 0.8 GB per float32 temporary regardless of DEM size.
_EDGE_CHUNK = 200_000_000


def _stream_kernels():
    """The C kernels pytopotoolbox drives its flow traversals with."""
    from topotoolbox import _stream
    return _stream


def _check_c_order(fd) -> None:
    """The low-memory paths index the grid flat, which is only the same
    thing pytopotoolbox does when the grid is C-ordered."""
    if getattr(fd, "order", "C") != "C":
        raise NotImplementedError(
            "low-memory flow kernels assume a C-ordered grid; this "
            "FlowObject is Fortran-ordered")


def flow_accumulation_inplace(fd, w: np.ndarray) -> np.ndarray:
    """``fd.flow_accumulation(weights=w).z`` without the two full-grid
    copies pytopotoolbox makes.

    pytopotoolbox 0.0.12 opens ``flow_accumulation`` with::

        acc = np.array(self.ezgetnal(weights, dtype=np.float32),
                       copy=True, order=self.order)

    ``ezgetnal`` ends in ``k.astype(dtype)``, which copies even when the
    dtype already matches, and ``np.array(..., copy=True)`` copies that
    again. Two extra full-grid arrays -- 97.4 GB on the Brazil domain --
    ahead of a kernel that works in place, which is more than a 512 GB
    machine holds together with the routing.

    ``w`` is CONSUMED: it is accumulated in place and returned, so pass
    an array nothing else holds. Everything else is identical to the
    library call, including the C kernel, so results are bit-identical
    (``tests/test_lowmem_flow.py``).
    """
    _check_c_order(fd)
    w = np.asarray(w)
    if w.dtype != np.float32:
        raise TypeError(f"weights must be float32, got {w.dtype}")
    if w.shape != tuple(fd.shape):
        raise ValueError(f"weights shape {w.shape} != flow grid {fd.shape}")
    fraction = np.ones_like(fd.source, dtype=np.float32)
    _stream_kernels().traverse_down_f32_add_mul(w, fraction,
                                                fd.source, fd.target)
    del fraction
    return w


def flow_accumulation_cells(fd) -> np.ndarray:
    """Unweighted flow accumulation (cell counts), low memory."""
    return flow_accumulation_inplace(
        fd, np.ones(tuple(fd.shape), dtype=np.float32, order="C"))


def vertical_distance_to_stream(fd, stream, dem) -> np.ndarray:
    """``fd.vertdistance2stream(stream, dem).z`` at a fraction of the
    memory: height above the nearest drainage, along flow paths.

    The library version reads ``grid.z[self.source_indices]``, and
    ``source_indices`` is a *property* returning
    ``np.unravel_index(self.source, self.shape, self.order)`` -- a tuple
    of two int64 arrays, one entry per flow edge. On the Brazil domain
    that is 195 GB, rebuilt on every attribute access, and the function
    touches it twice and then holds three more full-length float32
    arrays (``zsources``, ``ztarget``, ``dz``): well over 340 GB.

    Indexing the grid flat gives exactly the same elements for a
    C-ordered grid without materialising any index array, and the
    difference is accumulated chunk-wise, so the peak is the two arrays
    the kernel genuinely needs: ``dz`` (per edge) and ``up_z`` (per
    cell), ~97 GB on Brazil.
    """
    _check_c_order(fd)
    zflat = np.asarray(dem.z, dtype=np.float32).reshape(-1)
    mflat = np.asarray(stream.gridmask).reshape(-1)
    src, tgt = fd.source, fd.target
    n_edges = int(src.size)

    dz = np.empty(n_edges, dtype=np.float32)
    for s in range(0, n_edges, _EDGE_CHUNK):
        e = min(s + _EDGE_CHUNK, n_edges)
        si = src[s:e]
        blk = dz[s:e]                     # view: writes reach dz
        blk[:] = zflat[si]
        blk -= zflat[tgt[s:e]]
        blk[mflat[si]] = np.float32(0.0)  # stream cells sit at zero
        del si, blk

    up_z = np.zeros(tuple(fd.shape), dtype=np.float32, order="C")
    _stream_kernels().traverse_up_f32_max_add(up_z, dz, src, tgt)
    del dz
    return up_z


def _absorbing_pointer_with_weight(ptr: np.ndarray, w: np.ndarray):
    """Pointer doubling that also accumulates the path weight.

    ``ptr`` maps each element to the next one, with absorbing states
    pointing to themselves and carrying weight 0. On return ``w`` holds
    the total weight from each element to its absorbing state.
    """
    assert_int64(ptr)
    n = len(ptr)
    max_passes = int(np.ceil(np.log2(max(n, 2)))) + 2   # cycle guard
    for _ in range(max_passes):
        changed = False
        for s in range(0, n, _CHUNK):
            e = min(s + _CHUNK, n)
            p = ptr[s:e]
            nxt = ptr[p]
            w[s:e] += w[p]
            if not changed and not np.array_equal(nxt, p):
                changed = True
            ptr[s:e] = nxt
        if not changed:
            return ptr, w
    raise ValueError("flow graph contains a cycle: pointer jumping did not converge")


def distance_to_stream(net: FlowNet) -> np.ndarray:
    """Flow distance (map units) from every cell to the first stream cell
    downstream; NaN where no stream is reached. Shape (nr, nc), float32.

    Equivalent to MATLAB ``flowdistance(FD, S)`` with ``inf`` mapped to
    NaN, as MATLAB does.
    """
    grid = net.grid
    n_cells = int(grid.n_cells)
    nc = np.int64(grid.nc)
    cs = float(grid.cellsize)

    sentinel = np.int64(n_cells)
    ptr = np.full(n_cells + 1, sentinel, dtype=np.int64)
    # float32: 48.7 GB instead of 97.4 GB on the full domain. Pointer
    # doubling performs only O(log path) additions (~25), not one per
    # cell, so float32 accumulation error stays far below a cell size.
    w = np.zeros(n_cells + 1, dtype=np.float32)

    # The step length is diagonal where the edge changes both row and
    # column. Written chunk-wise: a one-shot formulation builds r1, c1,
    # r2 and c2 as full int64 arrays with one entry per flow edge --
    # 4 x 97.4 GB on the Brazil domain, plus a float64 np.where result
    # and two bool masks, roughly 700 GB to scatter two distinct numbers.
    # Chunked, the temporaries are a few GB whatever the DEM.
    src, dst = net.fd_src, net.fd_dst
    n_edges = int(src.size)
    diag_len = np.float32(cs * np.sqrt(2.0))
    card_len = np.float32(cs)
    for s in range(0, n_edges, _EDGE_CHUNK):
        e = min(s + _EDGE_CHUNK, n_edges)
        a, b = src[s:e], dst[s:e]
        is_diag = ((a // nc) != (b // nc)) & ((a % nc) != (b % nc))
        ptr[a] = b
        w[a] = np.where(is_diag, diag_len, card_len)
        del a, b, is_diag

    # stream cells are absorbing with zero remaining distance
    ptr[net.st_nodes] = net.st_nodes
    w[net.st_nodes] = 0.0
    ptr[sentinel] = sentinel
    w[sentinel] = 0.0

    ptr, w = _absorbing_pointer_with_weight(ptr, w)

    is_stream = np.zeros(n_cells + 1, dtype=bool)
    is_stream[net.st_nodes] = True
    # a chain that ends anywhere but on a stream cell or the sentinel is a
    # cycle collapsed onto itself
    for s in range(0, n_cells, _CHUNK):
        e = min(s + _CHUNK, n_cells)
        end = ptr[s:e]
        if not np.all(is_stream[end] | (end == sentinel)):
            raise ValueError("flow graph contains a cycle: a chain ends on a "
                             "cell that is neither a stream cell nor the edge")
    # a view, not .astype(float32): w is already float32, and a cast
    # would be a silent full-grid copy of the result
    out = w[:n_cells]
    for s in range(0, n_cells, _CHUNK):
        e = min(s + _CHUNK, n_cells)
        blk = out[s:e]
        blk[~is_stream[ptr[s:e]]] = np.nan
        del blk
    return out.reshape(grid.nr, grid.nc)


def nal_to_hillslopes(net: FlowNet, nal: np.ndarray) -> np.ndarray:
    """Propagate a stream node-attribute list to every cell, following
    flow paths (MATLAB ``mapfromnal(FD, S, nal)``). float32 (nr, nc).

    Used for chi: the chi value of the first stream cell downstream is
    assigned to the whole contributing hillslope.
    """
    from pydivide.labels import first_stream_cell_downstream

    grid = net.grid
    tgt = first_stream_cell_downstream(net)
    order = np.argsort(net.st_nodes, kind="stable")
    sorted_nodes = net.st_nodes[order]
    vals = np.asarray(nal, dtype=np.float64)[order]

    out = np.full(int(grid.n_cells), np.nan, dtype=np.float32)
    n_cells = int(grid.n_cells)
    for s in range(0, n_cells, _CHUNK):
        e = min(s + _CHUNK, n_cells)
        blk = tgt[s:e]
        ok = blk >= 0
        if ok.any():
            loc = np.searchsorted(sorted_nodes, blk[ok])
            out[s:e][ok] = vals[loc]
    return out.reshape(grid.nr, grid.nc)


def arcslope_degrees(z: np.ndarray, cellsize: float) -> np.ndarray:
    """Slope in degrees, ArcGIS/Horn 3x3 method — the exact kernel of
    MATLAB ``arcslope``:

        kernel = [-1 -2 -1; 0 0 0; 1 2 1]

    dz/dx and dz/dy are the two convolutions divided by 8*cellsize, and
    the slope is ``atand(hypot(dzdx, dzdy))``. Border cells are
    replicate-padded, matching TopoToolbox. NaN propagates.
    """
    # Row-blocked, float32.
    #
    # np.pad(z.astype(np.float64), 1) would be a float64 copy of the WHOLE
    # DEM plus a border, 97 GB on a 1.2 x 10^10-cell domain. The kernel
    # needs only a one-row halo, so blocking makes
    # peak memory a fixed budget. float32 is ample: the result is an angle
    # in degrees, and the inputs are float32 elevations to begin with.
    zz = np.asarray(z)
    nr, nc = zz.shape
    out = np.empty((nr, nc), dtype=np.float32)
    denom = np.float32(8.0 * float(cellsize))

    for r0 in range(0, nr, _SLOPE_BLOCK):
        r1 = min(r0 + _SLOPE_BLOCK, nr)
        a0 = max(0, r0 - 1)
        a1 = min(nr, r1 + 1)
        blk = np.asarray(zz[a0:a1], dtype=np.float32)
        # replicate only the edges the DEM itself is missing
        pt = 1 if a0 == r0 else 0
        pb = 1 if a1 == r1 else 0
        zp = np.pad(blk, ((pt, pb), (1, 1)), mode="edge")
        a = zp[:-2, :-2]
        b = zp[:-2, 1:-1]
        c = zp[:-2, 2:]
        d = zp[1:-1, :-2]
        f = zp[1:-1, 2:]
        g = zp[2:, :-2]
        h = zp[2:, 1:-1]
        i = zp[2:, 2:]
        dzdx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / denom
        dzdy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / denom
        out[r0:r1] = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
        del blk, zp, dzdx, dzdy
    return out


def upslope_mean(fd, values: np.ndarray) -> np.ndarray:
    """Upslope mean of ``values`` (MATLAB ``upslopestats(...,'mean',S)``).

    mean = flowacc(values) / flowacc(1), both accumulated over the flow
    network.

    Note on the stream argument: MATLAB additionally removes flow edges
    whose *both* endpoints are stream cells, so accumulation does not run
    along the channel. That pruning cannot change the result at divide
    locations, which is the only place this grid is ever sampled: a
    divide cell is a flow-divergence line, so its upslope area contains
    no channel cells at all (a channel upslope of a divide would have to
    flow up over it).

    ``values`` is CONSUMED -- it is turned into the numerator in place.
    Delete the caller's name for it as soon as this returns.

    Memory on the full domain (1.217e10 cells): a direct transcription
    holds two float64 copies of a full-domain accumulation, 97.4 GB
    each, and ``fd.flow_accumulation`` adds two more full-grid float32
    copies of its own (see ``flow_accumulation_inplace``). Working in
    place throughout, the peak is the numerator, the denominator and
    one per-edge fraction array: ~146 GB. float32 is adequate here --
    the result is a mean slope in degrees and both operands are float32
    accumulations.
    """
    v = np.asarray(values, dtype=np.float32)
    if v.base is not None or v.dtype != np.float32:
        v = np.array(values, dtype=np.float32)        # own it before mutating
    finite = np.isfinite(v)
    # in place, no temporary: NaN and +/-inf both become 0
    np.nan_to_num(v, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    ssum = flow_accumulation_inplace(fd, v)
    del v, values
    cnt = flow_accumulation_inplace(fd, finite.astype(np.float32))
    del finite
    with np.errstate(invalid="ignore", divide="ignore"):
        np.divide(ssum, cnt, out=ssum)
    ssum[cnt == 0] = np.nan
    del cnt
    return ssum

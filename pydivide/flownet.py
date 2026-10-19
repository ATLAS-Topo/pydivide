# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Input contract of the pydivide pipeline.

The divide stages consume only:
  - grid geometry (GridSpec),
  - the full flow network as topologically ordered edge lists of int64
    cell ids (upstream-to-downstream: givers appear before their receivers),
  - the stream network: node cells in node-attribute-list (NAL) order and
    stream edges in the same topological convention,
  - outlet cells.

They do not care where these came from. Two adapters exist:
  - :meth:`FlowNet.from_matlab_reference` — validation runs, driven by the
    exact networks MATLAB used;
  - :meth:`FlowNet.from_pytopotoolbox` — production runs.

Cell id convention: zero-based row-major, ``cell = i * nc + j`` (int64).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gridspec import GridSpec, _crs_of, assert_int64, same_crs


@dataclass
class FlowNet:
    grid: GridSpec
    fd_src: np.ndarray   # (n_fd_edges,) int64 cell ids, topological order
    fd_dst: np.ndarray
    st_nodes: np.ndarray  # (n_st_nodes,) int64 cell ids, NAL order
    st_src: np.ndarray    # (n_st_edges,) int64 cell ids, topological order
    st_dst: np.ndarray
    outlets: np.ndarray   # (n_outlets,) int64 cell ids
    crs: object = None    # coordinate reference system (for exports)

    def __post_init__(self) -> None:
        assert_int64(self.fd_src, self.fd_dst, self.st_nodes,
                     self.st_src, self.st_dst, self.outlets)

    # ------------------------------------------------------------------

    @classmethod
    def from_matlab_reference(cls, ref) -> "FlowNet":
        """Build from a :class:`pydivide.matlab_ref.MatlabReference`."""
        g = ref.grid
        nc = np.int64(g.nc)

        def rc_to_cell(rc: np.ndarray) -> np.ndarray:
            return rc[:, 0] * nc + rc[:, 1]

        fd = ref.fd_edges_rc()
        st = ref.st_edges_rc()
        nodes = ref.st_nodes_rc()
        out_rc, _ = ref.outlets()
        return cls(
            grid=g,
            fd_src=rc_to_cell(fd[:, :2]),
            fd_dst=rc_to_cell(fd[:, 2:]),
            st_nodes=rc_to_cell(nodes),
            st_src=rc_to_cell(st[:, :2]),
            st_dst=rc_to_cell(st[:, 2:]),
            outlets=rc_to_cell(out_rc),
        )

    @classmethod
    def from_pytopotoolbox(cls, fd, st) -> "FlowNet":
        """Build from pytopotoolbox FlowObject + StreamObject.

        Relies on: fd.source/fd.target being int64 linear cell indices in
        topological (upstream-to-downstream) order, in the memory order
        ``fd.order`` of the grid; st.node_indices being the stream cells
        in NAL order as (row, column) pairs; st.source/st.target being
        positions into the NAL (converted here to cell ids).

        The divide stages read the grid flat in row-major order, which
        is the layout pytopotoolbox uses for every raster it reads from
        a file. A FlowObject built from a Fortran-ordered array carries
        ``order == "F"`` and stores its linear indices column-major; it
        is refused here, and by every other entry point, rather than
        supported in part. Every index is checked against the grid, and
        a StreamObject that describes another grid than the FlowObject
        is refused.
        """
        nr, nc = (int(v) for v in fd.shape)
        grid = GridSpec.from_rasterio_transform(nr, nc, fd.transform)
        order = getattr(fd, "order", "C")
        if order != "C":
            raise ValueError(
                "pydivide works on C-ordered (row-major) grids, the layout pytopotoolbox uses "
                f"for every raster it reads; this FlowObject has order {order!r}. Rebuild it "
                "from a C-ordered GridObject (numpy.ascontiguousarray) before calling pydivide.")
        st_shape = getattr(st, "shape", None)
        if st_shape is not None and tuple(int(v) for v in st_shape) != (nr, nc):
            raise ValueError(f"the stream network describes a {tuple(st_shape)} grid, the flow "
                             f"object a {(nr, nc)} grid")
        st_tr = getattr(st, "transform", None)
        if st_tr is not None and st_tr != fd.transform:
            raise ValueError("the stream network and the flow object carry different "
                             "transforms; they do not describe the same grid")
        crs_fd, crs_st = _crs_of(fd), _crs_of(st)
        if crs_fd is not None and crs_st is not None and not same_crs(crs_fd, crs_st):
            raise ValueError("the stream network and the flow object carry different "
                             "coordinate reference systems; they do not describe the same grid")

        def cells(ids):
            ids = np.asarray(ids, dtype=np.int64)
            if ids.size and (ids.min() < 0 or ids.max() >= nr * nc):
                raise ValueError("flow edge indices fall outside the grid")
            return ids

        fd_src = cells(fd.source)
        fd_dst = cells(fd.target)

        rows, cols = st.node_indices
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        if rows.size and (rows.min() < 0 or cols.min() < 0
                          or rows.max() >= nr or cols.max() >= nc):
            raise ValueError("stream node indices fall outside the grid")
        st_nodes = rows * np.int64(nc) + cols
        s = np.asarray(st.source, dtype=np.int64)
        t = np.asarray(st.target, dtype=np.int64)
        n_nodes = st_nodes.size
        if s.size and (s.max(initial=0) >= n_nodes
                       or t.max(initial=0) >= n_nodes):
            raise ValueError(
                "StreamObject.source/target look like raster indices, not "
                "NAL positions; adapter update needed for this "
                "pytopotoolbox version")
        st_src = st_nodes[s]
        st_dst = st_nodes[t]

        # outlets: stream nodes with no outgoing stream edge. A stream node
        # with no edge at all (a one-cell network) is its own outlet too.
        has_out = np.zeros(n_nodes, dtype=bool)
        has_out[s] = True
        outlets = st_nodes[~has_out]
        return cls(grid=grid, fd_src=fd_src, fd_dst=fd_dst,
                   st_nodes=st_nodes, st_src=st_src, st_dst=st_dst,
                   outlets=outlets, crs=getattr(fd, "georef", None))

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Stages 2-3 — divide edge geometry.

The divide edge set is computed directly from the basin label raster:
a divide edge is a face between two cells with different labels. This
replaces MATLAB's GRIDobj2polygon + per-basin coordinate matching.

To be *edge-set equal* to MATLAB DIVIDEobj (validation ladder step 2),
three additional MATLAB behaviours are reproduced exactly:

1. **Perimeter and nodata faces are included.** MATLAB polygons trace each
   basin's full outline, so faces between a labelled cell and the outside
   world (DEM edge) or an unlabelled (NaN) cell are divide edges too.

2. **Cardinal stream crossings are deleted.** A face whose two cells are
   connected by a cardinal stream flow edge is removed (both adjacent
   basin walks lose it symmetrically). Diagonal stream edges do not
   remove faces; they split the network at nodes (Stage 4).

3. **Outlet-corner blanking.** For every basin that contains a true
   stream-network outlet, MATLAB deletes from *that basin's walk* all
   walk nodes that coincide with corners of the outlet pixel (only
   corners that lie on the basin's outline). An edge disappears from the
   final network only if every walk containing it removed it: an edge
   between basins A and B survives via B's walk unless B removed it too;
   an edge between A and nodata/outside exists only in A's walk.

What is *not* reproduced (order-dependent MATLAB artifact, classified by
the validator instead): the constructor drops isolated single-edge runs
("avoid single nodes"), where isolation depends on polygon traversal
order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gridspec import GridSpec, assert_int64, node_id

# ---------------------------------------------------------------------------
# pair matching without 128-bit keys (safe for the full domain)

def _rows_view(a: np.ndarray) -> np.ndarray:
    """1-D structured view of an int64 (n, 2) array; sorts and searches
    lexicographically by (col0, col1)."""
    a = np.ascontiguousarray(a)
    return a.view([("x", np.int64), ("y", np.int64)]).ravel()


def pair_isin(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-membership test: for each row of int64 (n,2) array ``a``, is it
    a row of (m,2) array ``b``? Structured-view binary search; no scalar
    key is formed, so there is no overflow risk at full-domain node
    counts."""
    assert_int64(a, b)
    if b.size == 0 or a.size == 0:
        return np.zeros(len(a), dtype=bool)
    av = _rows_view(a)
    bv = np.sort(_rows_view(b))
    idx = np.searchsorted(bv, av)
    idx = np.minimum(idx, len(bv) - 1)
    return bv[idx] == av


# ---------------------------------------------------------------------------

@dataclass
class DivideEdges:
    """Result of Stages 2-3.

    Attributes
    ----------
    nodes : (n, 2) int64
        Divide edges as node-id pairs, each row sorted (a < b), rows
        unique. Python node convention (gridspec).
    lab_a, lab_b : (n,) int64
        Basin labels of the two cells each edge separates; -1 for
        outside-the-DEM or NaN-labelled cells. lab_a belongs to the
        left/upper cell.
    is_boundary : (n,) bool
        True where one side is outside/nodata (lab == -1). These are the
        potentially spurious truncation divides of Scherler & Schwanghart
        section 5.1; kept in the network but flagged.
    """
    nodes: np.ndarray
    lab_a: np.ndarray
    lab_b: np.ndarray
    is_boundary: np.ndarray


_ROW_BLOCK = 4096   # rows per block: caps boolean temporaries


def _faces_from_labels(grid: GridSpec, L: np.ndarray):
    """All divide faces of the label raster, before cuts and blanking.

    ``L`` is int32 with -1 for unlabelled (Stage 1 output); a float
    raster with NaN is accepted and converted for convenience at
    validation scale. Returns (nodes (n,2) int64 sorted-per-row, lab_a,
    lab_b int64). Faces with both sides invalid do not exist.

    Interior faces are extracted in row blocks so the boolean masks and
    index temporaries stay bounded regardless of raster size.
    """
    nr, nc = grid.nr, grid.nc
    if np.issubdtype(L.dtype, np.floating):
        Lm = np.where(np.isnan(L), -1, L).astype(np.int32)
    else:
        Lm = L

    parts_nodes = []
    parts_a = []
    parts_b = []

    for r0 in range(0, nr, _ROW_BLOCK):
        r1 = min(r0 + _ROW_BLOCK, nr)

        # vertical faces in rows [r0, r1):
        # between (i, j) and (i, j+1) -> nodes (i,j+1),(i+1,j+1)
        a = Lm[r0:r1, :-1]
        b = Lm[r0:r1, 1:]
        m = (a != b) & ~((a == -1) & (b == -1))
        i, j = np.nonzero(m)
        i = i.astype(np.int64) + r0
        j = j.astype(np.int64)
        parts_nodes.append(np.stack([node_id(i, j + 1, nc),
                                     node_id(i + 1, j + 1, nc)], axis=1))
        parts_a.append(a[m].astype(np.int64))
        parts_b.append(b[m].astype(np.int64))

        # horizontal faces with upper row in [r0, min(r1, nr-1)):
        # between (i, j) and (i+1, j) -> nodes (i+1,j),(i+1,j+1)
        h1 = min(r1, nr - 1)
        if h1 > r0:
            a = Lm[r0:h1, :]
            b = Lm[r0 + 1:h1 + 1, :]
            m = (a != b) & ~((a == -1) & (b == -1))
            i, j = np.nonzero(m)
            i = i.astype(np.int64) + r0
            j = j.astype(np.int64)
            parts_nodes.append(np.stack([node_id(i + 1, j, nc),
                                         node_id(i + 1, j + 1, nc)],
                                        axis=1))
            parts_a.append(a[m].astype(np.int64))
            parts_b.append(b[m].astype(np.int64))

    # perimeter faces: labelled cell against the outside world
    # top row: face above cell (0, j) -> nodes (0,j),(0,j+1)
    j = np.nonzero(Lm[0, :] != -1)[0].astype(np.int64)
    parts_nodes.append(np.stack([node_id(np.int64(0), j, nc),
                                 node_id(np.int64(0), j + 1, nc)], axis=1))
    parts_a.append(np.full(j.size, -1, dtype=np.int64))
    parts_b.append(Lm[0, j].astype(np.int64))
    # bottom row: face below cell (nr-1, j) -> nodes (nr,j),(nr,j+1)
    j = np.nonzero(Lm[nr - 1, :] != -1)[0].astype(np.int64)
    parts_nodes.append(np.stack([node_id(np.int64(nr), j, nc),
                                 node_id(np.int64(nr), j + 1, nc)], axis=1))
    parts_a.append(Lm[nr - 1, j].astype(np.int64))
    parts_b.append(np.full(j.size, -1, dtype=np.int64))
    # left column: face left of cell (i, 0) -> nodes (i,0),(i+1,0)
    i = np.nonzero(Lm[:, 0] != -1)[0].astype(np.int64)
    parts_nodes.append(np.stack([node_id(i, np.int64(0), nc),
                                 node_id(i + 1, np.int64(0), nc)], axis=1))
    parts_a.append(np.full(i.size, -1, dtype=np.int64))
    parts_b.append(Lm[i, 0].astype(np.int64))
    # right column: face right of cell (i, nc-1) -> nodes (i,nc),(i+1,nc)
    i = np.nonzero(Lm[:, nc - 1] != -1)[0].astype(np.int64)
    parts_nodes.append(np.stack([node_id(i, np.int64(nc), nc),
                                 node_id(i + 1, np.int64(nc), nc)], axis=1))
    parts_a.append(Lm[i, nc - 1].astype(np.int64))
    parts_b.append(np.full(i.size, -1, dtype=np.int64))

    nodes = np.concatenate(parts_nodes, axis=0)
    lab_a = np.concatenate(parts_a)
    lab_b = np.concatenate(parts_b)
    nodes.sort(axis=1)   # per-row a < b (rows are already distinct faces)
    return nodes, lab_a, lab_b


def _stream_cut_faces(grid: GridSpec, st_src: np.ndarray,
                      st_dst: np.ndarray) -> np.ndarray:
    """Node pairs of faces crossed by cardinal stream flow edges."""
    nc = np.int64(grid.nc)
    r1, c1 = st_src // nc, st_src % nc
    r2, c2 = st_dst // nc, st_dst % nc
    dr = r2 - r1
    dc = c2 - c1
    cardinal = (np.abs(dr) + np.abs(dc)) == 1

    out = []
    # horizontal step (same row, adjacent cols): vertical face between them
    m = cardinal & (dr == 0)
    if m.any():
        cmax = np.maximum(c1[m], c2[m])
        out.append(np.stack([node_id(r1[m], cmax, grid.nc),
                             node_id(r1[m] + 1, cmax, grid.nc)], axis=1))
    # vertical step (same col, adjacent rows): horizontal face between them
    m = cardinal & (dc == 0)
    if m.any():
        rmax = np.maximum(r1[m], r2[m])
        out.append(np.stack([node_id(rmax, c1[m], grid.nc),
                             node_id(rmax, c1[m] + 1, grid.nc)], axis=1))
    if not out:
        return np.empty((0, 2), dtype=np.int64)
    cuts = np.concatenate(out, axis=0)
    cuts.sort(axis=1)
    return np.unique(cuts, axis=0)


def divide_edges(grid: GridSpec, L: np.ndarray, st_src: np.ndarray,
                 st_dst: np.ndarray, outlet_cells: np.ndarray,
                 outlet_labels: np.ndarray) -> DivideEdges:
    """Stages 2-3: the divide edge set.

    Parameters
    ----------
    L : (nr, nc) int32
        Basin label raster from Stage 1 (-1 = unlabelled; float64 with
        NaN also accepted).
    st_src, st_dst : int64 cell ids
        Stream flow edges.
    outlet_cells : int64 cell ids of true stream-network outlets.
    outlet_labels : int64 reach/basin label containing each outlet.
    """
    assert_int64(st_src, st_dst, outlet_cells, outlet_labels)
    nc = np.int64(grid.nc)

    nodes, lab_a, lab_b = _faces_from_labels(grid, L)

    # ---- Stage 3: remove faces crossed by cardinal stream edges ----------
    cuts = _stream_cut_faces(grid, st_src, st_dst)
    cut_mask = pair_isin(nodes, cuts)

    # ---- outlet-corner blanking ------------------------------------------
    # candidate corner nodes per outlet basin
    r = outlet_cells // nc
    c = outlet_cells % nc
    corner_nodes = np.concatenate([
        node_id(r, c, grid.nc), node_id(r, c + 1, grid.nc),
        node_id(r + 1, c, grid.nc), node_id(r + 1, c + 1, grid.nc),
    ])
    corner_labels = np.concatenate([outlet_labels] * 4)

    # (label, node) pairs describing each basin's boundary endpoints
    face_lab_node = np.concatenate([
        np.stack([lab_a, nodes[:, 0]], axis=1),
        np.stack([lab_a, nodes[:, 1]], axis=1),
        np.stack([lab_b, nodes[:, 0]], axis=1),
        np.stack([lab_b, nodes[:, 1]], axis=1),
    ], axis=0)
    corner_pairs = np.stack([corner_labels, corner_nodes], axis=1)
    # a corner is blanked only if it lies on its basin's outline
    on_outline = pair_isin(corner_pairs, np.unique(face_lab_node, axis=0))
    blank_pairs = np.unique(corner_pairs[on_outline], axis=0)

    # per-face, per-side: was this face removed from that side's walk?
    def blanked_on_side(lab: np.ndarray) -> np.ndarray:
        p1 = np.stack([lab, nodes[:, 0]], axis=1)
        p2 = np.stack([lab, nodes[:, 1]], axis=1)
        return pair_isin(p1, blank_pairs) | pair_isin(p2, blank_pairs)

    blank_a = blanked_on_side(lab_a)
    blank_b = blanked_on_side(lab_b)

    # a face survives if at least one adjacent *labelled* basin's walk
    # kept it; walks exist only for labelled sides
    side_a_removed = blank_a | (lab_a == -1)
    side_b_removed = blank_b | (lab_b == -1)
    blanked_everywhere = side_a_removed & side_b_removed

    keep = ~cut_mask & ~blanked_everywhere

    return DivideEdges(
        nodes=nodes[keep],
        lab_a=lab_a[keep],
        lab_b=lab_b[keep],
        is_boundary=(lab_a[keep] == -1) | (lab_b[keep] == -1),
    )

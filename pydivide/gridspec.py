# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Grid and index conventions.

Cell grid: ``nr x nc``, cell (i, j), zero-based, row-major.
Divide-node grid: ``(nr+1) x (nc+1)``. Node (I, J) sits at the top-left
corner of cell (I, J), i.e. at world coordinates (x0 + J*cs, y0 - I*cs)
where (x0, y0) is the top-left corner of cell (0, 0).

Node linear index (int64, row-major):  nid = I * (nc + 1) + J

MATLAB DIVIDEobj uses 1-based, column-major linear indices into the node
grid of size (nr+1, nc+1); MATLAB FLOWobj/GRIDobj use 1-based column-major
indices into the (nr, nc) cell grid. The conversion helpers below are the
single place where the two conventions meet. Everything else in pydivide
speaks Python conventions only.

Divide edges (validated against MATLAB coord2ind by
tests/test_grid_conventions.py):

    vertical   edge separates cells (i, j) | (i, j+1); its endpoints are
               nodes (i, j+1) and (i+1, j+1)
    horizontal edge separates cells (i, j) / (i+1, j); its endpoints are
               nodes (i+1, j) and (i+1, j+1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def assert_int64(*arrays: np.ndarray) -> None:
    """Guard against silent integer downcasts.

    Every index array crossing a function boundary is asserted int64. A
    silent int32 downcast would produce plausible-looking wrong answers on
    the full domain rather than an error.
    """
    for a in arrays:
        if a.dtype != np.int64:
            raise TypeError(
                f"index array has dtype {a.dtype}, expected int64 "
                "(narrower index types are never allowed)"
            )


@dataclass(frozen=True)
class GridSpec:
    """Geometry of the cell grid and its divide-node grid.

    Parameters
    ----------
    nr, nc : int
        Cell grid dimensions (rows, columns).
    cellsize : float
        Cell size in map units (square cells only, as in TopoToolbox).
    x0, y0 : float
        World coordinates of the top-left corner of cell (0, 0) —
        equivalently of divide node (0, 0).
    """

    nr: int
    nc: int
    cellsize: float
    x0: float
    y0: float

    @property
    def node_shape(self) -> tuple[int, int]:
        return (self.nr + 1, self.nc + 1)

    @property
    def n_nodes(self) -> np.int64:
        return np.int64(self.nr + 1) * np.int64(self.nc + 1)

    @property
    def n_cells(self) -> np.int64:
        return np.int64(self.nr) * np.int64(self.nc)

    @classmethod
    def from_matlab_wf(cls, nr: int, nc: int, cellsize: float,
                       wf_cell: np.ndarray) -> "GridSpec":
        """Build from a MATLAB GRIDobj/FLOWobj world-file matrix.

        ``wf_cell`` is the 2x3 matrix of the *cell* grid: it maps 0-based
        (col, row) to the *center* of cell (row, col). The top-left corner
        of cell (0, 0) is therefore the center minus half a cell in x and
        plus half a cell in y.
        """
        wf_cell = np.asarray(wf_cell, dtype=np.float64)
        cx = wf_cell[0, 2]  # center of cell (0,0)
        cy = wf_cell[1, 2]
        if abs(wf_cell[0, 1]) > 1e-9 or abs(wf_cell[1, 0]) > 1e-9:
            raise ValueError("rotated grids are not supported")
        hcs = cellsize / 2.0
        return cls(nr=nr, nc=nc, cellsize=float(cellsize),
                   x0=float(cx - hcs), y0=float(cy + hcs))

    @classmethod
    def from_rasterio_transform(cls, nr: int, nc: int,
                                transform) -> "GridSpec":
        """Build from a rasterio ``Affine`` (maps (col,row) to pixel corner)."""
        if abs(transform.b) > 1e-9 or abs(transform.d) > 1e-9:
            raise ValueError("rotated grids are not supported")
        if not np.isclose(abs(transform.a), abs(transform.e)):
            raise ValueError("non-square cells are not supported")
        identity = (transform.a == 1 and transform.e == 1
                    and transform.c == 0 and transform.f == 0)   # no georeferencing
        if not identity and (transform.a <= 0 or transform.e >= 0):
            raise ValueError(
                "only north-up grids are supported (x increasing eastward, y "
                "decreasing southward); reproject or flip the raster first")
        return cls(nr=nr, nc=nc, cellsize=float(abs(transform.a)),
                   x0=float(transform.c), y0=float(transform.f))

    # ---- node coordinates -------------------------------------------------

    def node_xy(self, nid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World coordinates of node linear indices (int64 in, float64 out)."""
        nid = np.asarray(nid)
        assert_int64(nid)
        I, J = node_rc(nid, self.nc)
        x = self.x0 + J.astype(np.float64) * self.cellsize
        y = self.y0 - I.astype(np.float64) * self.cellsize
        return x, y


# ---- node index arithmetic (module-level so they work without a GridSpec) --

def node_id(I: np.ndarray, J: np.ndarray, nc: int) -> np.ndarray:
    """Row-major int64 linear index of node (I, J) in the (nr+1, nc+1) grid."""
    return np.int64(nc + 1) * np.asarray(I, dtype=np.int64) \
        + np.asarray(J, dtype=np.int64)


def node_rc(nid: np.ndarray, nc: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`node_id`."""
    nid = np.asarray(nid, dtype=np.int64)
    w = np.int64(nc + 1)
    return nid // w, nid % w


# ---- MATLAB <-> Python conversions ----------------------------------------
# MATLAB linear indices are 1-based and column-major. NaNs (used as segment
# separators in DIVIDEobj.IX) are preserved by the float variants.

def matlab_node_to_py(ix: np.ndarray, nr: int, nc: int) -> np.ndarray:
    """MATLAB DIVIDEobj node index -> Python node id. NaN passes through.

    Input may be float (to carry NaN separators); output is float64 with
    NaN preserved if input has NaNs, otherwise int64.
    """
    ix = np.asarray(ix)
    H = np.int64(nr + 1)         # MATLAB node grid height
    if np.issubdtype(ix.dtype, np.floating):
        out = np.full(ix.shape, np.nan, dtype=np.float64)
        ok = ~np.isnan(ix)
        ix0 = ix[ok].astype(np.int64) - 1
        I = ix0 % H
        J = ix0 // H
        out[ok] = (node_id(I, J, nc)).astype(np.float64)
        return out
    ix0 = ix.astype(np.int64) - 1
    I = ix0 % H
    J = ix0 // H
    return node_id(I, J, nc)


def py_node_to_matlab(nid: np.ndarray, nr: int, nc: int) -> np.ndarray:
    """Python node id -> MATLAB 1-based column-major node index (int64)."""
    nid = np.asarray(nid)
    assert_int64(nid)
    I, J = node_rc(nid, nc)
    return J * np.int64(nr + 1) + I + 1


def matlab_cell_to_py(ix: np.ndarray, nr: int, nc: int) -> np.ndarray:
    """MATLAB 1-based column-major cell index -> Python row-major int64."""
    ix = np.asarray(ix)
    ix0 = ix.astype(np.int64) - 1
    i = ix0 % np.int64(nr)
    j = ix0 // np.int64(nr)
    return i * np.int64(nc) + j


def py_cell_to_matlab(idx: np.ndarray, nr: int, nc: int) -> np.ndarray:
    """Python row-major cell index -> MATLAB 1-based column-major (int64)."""
    idx = np.asarray(idx)
    assert_int64(idx)
    i = idx // np.int64(nc)
    j = idx % np.int64(nc)
    return j * np.int64(nr) + i + 1


# ---------------------------------------------------------------------------
# do two objects describe the same grid?

def _crs_of(obj):
    """The coordinate reference system an object carries, or None."""
    for attr in ("georef", "crs"):
        v = getattr(obj, attr, None)
        if callable(v):
            try:
                v = v()
            except Exception:
                v = None
        if v is not None and str(v) != "":
            return v
    return None


def same_crs(a, b) -> bool:
    """Semantic equality of two CRS definitions, whatever their encoding
    (an EPSG code, the WKT and a proj string of one system are equal)."""
    try:
        from pyproj import CRS
        return CRS.from_user_input(a) == CRS.from_user_input(b)
    except Exception:
        return str(a) == str(b)


def _grid_of(obj, what: str):
    """``(shape, GridSpec or None, cellsize or None, crs or None)`` of a
    pytopotoolbox object or of a divide network result: the shape it
    carries, its geometry when it carries a transform or a grid, its cell
    size when it states one without a transform, and its CRS."""
    if hasattr(obj, "grid") and hasattr(obj.grid, "cellsize"):
        g = obj.grid
        return (int(g.nr), int(g.nc)), g, float(g.cellsize), _crs_of(obj)
    shape = getattr(obj, "shape", None)
    if shape is None and hasattr(obj, "z"):
        shape = np.asarray(obj.z).shape
    if shape is None:
        raise ValueError(f"{what} carries no grid shape")
    nr, nc = (int(v) for v in shape)
    cs = getattr(obj, "cellsize", None)
    cs = None if cs is None else float(cs)
    tr = getattr(obj, "transform", None)
    if tr is None:
        return (nr, nc), None, cs, _crs_of(obj)
    g = GridSpec.from_rasterio_transform(nr, nc, tr)
    return (nr, nc), g, float(g.cellsize), _crs_of(obj)


def check_same_grid(reference, ref_name: str, other, other_name: str,
                    *, rtol: float = 1e-9) -> None:
    """Refuse two objects that do not describe the same grid.

    Compared: shape, cell size, origin (the north-up transform) and the
    coordinate reference system, the last semantically. Missing
    information is not taken as agreement: what both objects state is
    compared (a cell size stated without a transform included), and a
    warning names what could not be checked because one of the two does
    not state it. Used at every public boundary where separately built
    objects meet: the across-divide call, the profiles and the flow
    adapter.
    """
    import warnings

    sa, ga, ca_cs, ca = _grid_of(reference, ref_name)
    sb, gb, cb_cs, cb = _grid_of(other, other_name)
    if sa != sb:
        raise ValueError(f"{other_name} shape {sb} differs from {ref_name} shape {sa}")
    if ca_cs is not None and cb_cs is not None:
        if not np.isclose(ca_cs, cb_cs, rtol=rtol, atol=0):
            raise ValueError(f"{other_name} cell size {cb_cs:g} differs from {ref_name} "
                             f"cell size {ca_cs:g}")
    elif (ca_cs is None) != (cb_cs is None):
        warnings.warn(f"only one of {ref_name} and {other_name} states a cell size; the "
                      "two are compared on shape alone", RuntimeWarning, stacklevel=3)
    if ga is not None and gb is not None:
        scale = max(abs(ga.cellsize), 1.0)
        if not (np.isclose(ga.x0, gb.x0, rtol=0, atol=rtol * scale * 1e3)
                and np.isclose(ga.y0, gb.y0, rtol=0, atol=rtol * scale * 1e3)):
            raise ValueError(f"{other_name} origin ({gb.x0:g}, {gb.y0:g}) differs from "
                             f"{ref_name} origin ({ga.x0:g}, {ga.y0:g}); the two do not cover "
                             "the same ground")
    else:
        missing = " and ".join(n for n, g in ((ref_name, ga), (other_name, gb)) if g is None)
        warnings.warn(f"{missing} carries no transform; the origins of {ref_name} and "
                      f"{other_name} cannot be compared", RuntimeWarning, stacklevel=3)
    if ca is not None and cb is not None:
        if not same_crs(ca, cb):
            raise ValueError(f"{other_name} coordinate reference system ({str(cb)[:60]}) differs "
                             f"from {ref_name} ({str(ca)[:60]})")
    elif (ca is None) != (cb is None):
        warnings.warn(f"only one of {ref_name} and {other_name} carries a coordinate reference "
                      "system; their alignment is taken on shape, cell size and origin alone",
                      RuntimeWarning, stacklevel=3)

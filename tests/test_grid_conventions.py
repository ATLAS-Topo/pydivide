# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Grid and index conventions, asserted against MATLAB.

The spec mandates these tests exist before any algorithm code: sign
conventions (y decreasing with row) cause silent, plausible-looking errors.

Tests that need the MATLAB export are skipped when reference_data/bigtujunga
has not been generated yet (run matlab_scripts/export_reference.m first).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydivide.gridspec import (  # noqa: E402
    assert_int64,
    matlab_cell_to_py,
    matlab_node_to_py,
    node_id,
    node_rc,
    py_cell_to_matlab,
    py_node_to_matlab,
)

# reference-backed tests use the parametrized `ref` fixture (conftest.py)


# ---- pure convention tests (no reference data needed) ----------------------

def test_node_id_roundtrip():
    rng = np.random.default_rng(0)
    nr, nc = 643, 1197
    I = rng.integers(0, nr + 1, size=1000).astype(np.int64)
    J = rng.integers(0, nc + 1, size=1000).astype(np.int64)
    nid = node_id(I, J, nc)
    assert nid.dtype == np.int64
    I2, J2 = node_rc(nid, nc)
    np.testing.assert_array_equal(I, I2)
    np.testing.assert_array_equal(J, J2)


def test_node_id_int64_no_overflow():
    # Target domain: 105008 x 115896 cells -> node grid 105009 x 115897
    nr, nc = 105008, 115896
    nid = node_id(np.array([nr], dtype=np.int64),
                  np.array([nc], dtype=np.int64), nc)
    assert nid.dtype == np.int64
    assert nid[0] == np.int64(nr) * (nc + 1) + nc
    assert nid[0] > np.iinfo(np.int32).max  # would overflow int32


def test_matlab_node_conversion_roundtrip():
    rng = np.random.default_rng(1)
    nr, nc = 643, 1197
    nid = rng.integers(0, (nr + 1) * (nc + 1), size=1000).astype(np.int64)
    ml = py_node_to_matlab(nid, nr, nc)
    back = matlab_node_to_py(ml, nr, nc)
    np.testing.assert_array_equal(nid, back)


def test_matlab_node_conversion_preserves_nan():
    nr, nc = 10, 20
    ix = np.array([1.0, np.nan, 5.0])
    out = matlab_node_to_py(ix, nr, nc)
    assert np.isnan(out[1])
    assert not np.isnan(out[0]) and not np.isnan(out[2])
    # MATLAB index 1 = node (0,0) column-major -> Python node id 0
    assert out[0] == 0.0


def test_matlab_cell_conversion_roundtrip():
    rng = np.random.default_rng(2)
    nr, nc = 643, 1197
    idx = rng.integers(0, nr * nc, size=1000).astype(np.int64)
    ml = py_cell_to_matlab(idx, nr, nc)
    back = matlab_cell_to_py(ml, nr, nc)
    np.testing.assert_array_equal(idx, back)


def test_assert_int64_rejects_narrow():
    with pytest.raises(TypeError):
        assert_int64(np.zeros(3, dtype=np.int32))
    assert_int64(np.zeros(3, dtype=np.int64))  # no raise


def test_edge_endpoint_tables():
    """The edge table, written out longhand.

    Vertical edge between cells (i,j) and (i,j+1): endpoints are nodes
    (i, j+1) and (i+1, j+1). Horizontal edge between (i,j) and (i+1,j):
    endpoints are nodes (i+1, j) and (i+1, j+1).
    """
    nc = 7
    i, j = 3, 2
    # vertical edge
    a = node_id(np.int64(i), np.int64(j + 1), nc)
    b = node_id(np.int64(i + 1), np.int64(j + 1), nc)
    assert (a, b) == (3 * 8 + 3, 4 * 8 + 3)
    # horizontal edge
    a = node_id(np.int64(i + 1), np.int64(j), nc)
    b = node_id(np.int64(i + 1), np.int64(j + 1), nc)
    assert (a, b) == (4 * 8 + 2, 4 * 8 + 3)


# ---- tests against the MATLAB export ---------------------------------------

def test_node_coordinates_match_matlab(ref):
    """Node world coordinates from GridSpec must equal MATLAB ind2coord(D,.)
    for the divide nodes MATLAB actually produced."""
    chk = ref.coordcheck()  # matlab 1-based node index, x, y
    ml_ix = chk[:, 0]
    nid = matlab_node_to_py(ml_ix, ref.nr, ref.nc).astype(np.int64)
    x, y = ref.grid.node_xy(nid)
    np.testing.assert_allclose(x, chk[:, 1], rtol=0, atol=1e-6)
    np.testing.assert_allclose(y, chk[:, 2], rtol=0, atol=1e-6)


def test_wf_node_consistency(ref):
    """MATLAB's D.wf (node grid) must agree with GridSpec (x0, y0):
    node (0,0) sits at the top-left corner of cell (0,0)."""
    # wf_node maps 0-based (col;row;1) -> node position; node (0,0):
    x00 = ref.wf_node[0, 2]
    y00 = ref.wf_node[1, 2]
    assert abs(x00 - ref.grid.x0) < 1e-6
    assert abs(y00 - ref.grid.y0) < 1e-6


def test_node_grid_shape(ref):
    fx = ref.FX_grid()
    assert fx.shape == (ref.nr + 1, ref.nc + 1)
    assert set(np.unique(fx)).issubset({0, 1, 2})

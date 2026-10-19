# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""The low-memory flow kernels must equal pytopotoolbox exactly.

``pydivide.grids`` reimplements three pytopotoolbox calls, not because
the results were wrong but because the library allocates full-grid or
per-edge temporaries the Brazil domain cannot hold (see the docstrings
in ``grids.py``). Reimplementing a library call is only safe if it can
be shown to give the *same numbers*, so every one of them is checked
against the original here on a real DEM.

Equality is exact, not approximate: the same C kernel does the same
additions in the same order. A tolerance here would hide a real
difference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
tt3 = pytest.importorskip("topotoolbox")

from pydivide.flownet import FlowNet  # noqa: E402
from pydivide.grids import (  # noqa: E402
    distance_to_stream,
    flow_accumulation_cells,
    flow_accumulation_inplace,
    upslope_mean,
    vertical_distance_to_stream,
)

# Resolved against the repository root rather than the working directory:
# a relative path made these tests depend on where pytest was launched from.
DEM = Path(__file__).resolve().parents[1] / "test_data" / "brazil_crop_4000.tif"
CROP = 1200          # keep the library versions affordable in a test


@pytest.fixture(scope="module")
def flow():
    # Without this guard the whole module ERRORS rather than skipping when
    # the DEM is absent, which is what a stranger cloning the repository
    # sees. These tests need a real routed network, so they cannot be made
    # synthetic; skipping is the honest outcome.
    if not DEM.exists():
        pytest.skip(f"needs {DEM.name}, which is not distributed with the "
                    "repository (see README, 'Running the tests')")
    with rasterio.open(DEM) as src:
        z = src.read(1).astype(np.float32)[:CROP, :CROP]
        cs = abs(src.transform.a)
    nodata = ~np.isfinite(z)
    if nodata.any():
        z[nodata] = np.nanmin(z[~nodata])
    dem = tt3.GridObject()
    dem.z = np.ascontiguousarray(z)
    dem.cellsize = cs
    fd = tt3.FlowObject(dem)
    st = tt3.StreamObject(fd, threshold=1000)
    return dem, fd, st


def test_order_is_c(flow):
    """The flat-indexing shortcut is only valid for a C-ordered grid."""
    _, fd, _ = flow
    assert fd.order == "C"


def test_unweighted_accumulation_matches(flow):
    _, fd, _ = flow
    ours = flow_accumulation_cells(fd)
    theirs = np.asarray(fd.flow_accumulation().z)
    assert ours.shape == theirs.shape
    assert np.array_equal(ours, theirs)


def test_weighted_accumulation_matches(flow):
    dem, fd, _ = flow
    rng = np.random.default_rng(0)
    w = rng.random(dem.z.shape).astype(np.float32)
    theirs = np.asarray(fd.flow_accumulation(weights=w.copy()).z)
    ours = flow_accumulation_inplace(fd, w.copy())
    assert np.array_equal(ours, theirs)


def test_accumulation_consumes_its_input(flow):
    """The caller must know the array is destroyed -- that is the whole
    point of the in-place path, and a caller that reused it would get
    silently wrong numbers."""
    dem, fd, _ = flow
    w = np.ones(dem.z.shape, dtype=np.float32)
    out = flow_accumulation_inplace(fd, w)
    assert out is w


def test_accumulation_rejects_wrong_dtype_and_shape(flow):
    dem, fd, _ = flow
    with pytest.raises(TypeError):
        flow_accumulation_inplace(fd, np.ones(dem.z.shape, dtype=np.float64))
    with pytest.raises(ValueError):
        flow_accumulation_inplace(fd, np.ones((3, 3), dtype=np.float32))


def test_vertical_distance_matches(flow):
    dem, fd, st = flow
    theirs = np.asarray(fd.vertdistance2stream(st, dem).z)
    ours = vertical_distance_to_stream(fd, st, dem)
    assert ours.shape == theirs.shape
    assert np.array_equal(ours, theirs)


def test_vertical_distance_matches_when_chunked(flow, monkeypatch):
    """Chunking is a memory device and must not touch the result."""
    import pydivide.grids as G
    dem, fd, st = flow
    whole = vertical_distance_to_stream(fd, st, dem)
    for chunk in (1, 7, 1000, 999_983):
        monkeypatch.setattr(G, "_EDGE_CHUNK", chunk)
        assert np.array_equal(vertical_distance_to_stream(fd, st, dem), whole)


def test_vertical_distance_is_zero_on_the_stream(flow):
    dem, fd, st = flow
    hand = vertical_distance_to_stream(fd, st, dem)
    mask = np.asarray(st.gridmask)
    assert np.all(hand[mask] == 0.0)
    assert np.all(hand >= 0.0)


def test_upslope_mean_matches_the_library_formulation(flow):
    """The in-place formulation must agree with the library-call
    formulation: flowacc(values) / flowacc(finite)."""
    dem, fd, _ = flow
    rng = np.random.default_rng(1)
    v = rng.random(dem.z.shape).astype(np.float32) * 30.0
    v[::17, ::13] = np.nan                       # exercise the finite mask

    finite = np.isfinite(v)
    ref_sum = np.asarray(fd.flow_accumulation(
        weights=np.where(finite, v, np.float32(0.0))).z)
    ref_cnt = np.asarray(fd.flow_accumulation(
        weights=finite.astype(np.float32)).z)
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = ref_sum / ref_cnt
    ref[ref_cnt == 0] = np.nan

    ours = upslope_mean(fd, v.copy())
    both_nan = np.isnan(ours) & np.isnan(ref)
    assert np.array_equal(ours[~both_nan], ref[~both_nan])
    assert np.array_equal(np.isnan(ours), np.isnan(ref))


@pytest.fixture(scope="module")
def net(flow):
    _, fd, st = flow
    return FlowNet.from_pytopotoolbox(fd, st)


def test_distance_to_stream_is_chunk_invariant(net, monkeypatch):
    """The per-edge step length is scattered chunk-wise, because a
    one-shot formulation builds four full int64 index arrays (~390 GB on
    the Brazil domain). Chunking is a memory device and must not move a
    single number."""
    import pydivide.grids as G
    whole = distance_to_stream(net)
    for chunk in (1, 5, 997, 999_983):
        monkeypatch.setattr(G, "_EDGE_CHUNK", chunk)
        got = distance_to_stream(net)
        assert np.array_equal(np.isnan(got), np.isnan(whole))
        m = ~np.isnan(whole)
        assert np.array_equal(got[m], whole[m])


def test_distance_to_stream_is_zero_on_the_stream(net, flow):
    _, _, st = flow
    d = distance_to_stream(net)
    mask = np.asarray(st.gridmask)
    assert np.all(d[mask] == 0.0)
    assert np.all(d[~np.isnan(d)] >= 0.0)


def test_distance_to_stream_steps_are_cell_or_diagonal(net):
    """Every finite distance is a sum of cardinal and diagonal steps, so
    it must be a non-negative multiple-of-nothing but never smaller than
    one cell unless it is exactly zero."""
    d = distance_to_stream(net)
    cs = net.grid.cellsize
    finite = d[~np.isnan(d)]
    nonzero = finite[finite > 0]
    assert nonzero.min() >= cs - 1e-3

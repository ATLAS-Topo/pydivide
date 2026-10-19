# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Contracts that hold without a DEM, a routing library or reference
data: signed pair summaries, the multiscale table's own interface, the
settings document beside every export, the pair a mixed piece reports,
the arrow destination, checked reprojection, the flow adapter's index
conventions, cycle rejection, checkpoint identity and the Shapefile
limits. Everything here runs on a fresh clone with the package's
dependencies alone."""

from __future__ import annotations

import json
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import LineString, MultiLineString, Point, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydivide import vectorio as V  # noqa: E402
from pydivide.acrossdivide import (  # noqa: E402
    AcrossDivideTable,
    across_divide_metrics,
    coherence,
    metric_column,
    read_export_settings,
    threshold_tag,
)
from pydivide.basinpairs import is_signed_column, pair_table  # noqa: E402
from pydivide.checkpoint import StageStore, array_fingerprint  # noqa: E402
from pydivide.divmetrics import aggregate_to_pieces, reach_outlet_labels  # noqa: E402
from pydivide.flownet import FlowNet  # noqa: E402
from pydivide.grids import distance_to_stream  # noqa: E402
from pydivide.gridspec import GridSpec  # noqa: E402
from pydivide.labels import basin_labels, first_stream_cell_downstream  # noqa: E402
from pydivide.orderdist import SortedNetwork  # noqa: E402
from pydivide.pipeline import DivideNetworkResult  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# ---- signed pair summaries ----------------------------------------------------

def test_pair_summary_expresses_every_signed_value_as_side_a_minus_side_b():
    """Two rows describing the same contrast with opposite orientations
    must not cancel: +10 as (1 minus 2) and -10 as (2 minus 1) are the
    same +10 once both are read as side A minus side B."""
    rows = pair_table([1., 2.], [2., 1.], values={"chi_sgn": np.array([10., -10.])})
    assert rows.side_a.iloc[0] == 1 and rows.n_flipped.iloc[0] == 1
    assert rows.chi_sgn_mean.iloc[0] == 10.
    assert rows.chi_sgn_absmax.iloc[0] == 10.


def test_pair_summary_is_invariant_to_reversing_rows():
    rng = np.random.default_rng(3)
    n = 400
    lo = rng.integers(1, 40, n)
    hi = lo + rng.integers(1, 40, n)
    sgn = rng.normal(size=n)
    dai = rng.random(n)                       # unsigned: must be averaged as it is
    sgn[::7] = np.nan
    a, b, s = lo.astype(float), hi.astype(float), sgn.copy()
    base = pair_table(a, b, values={"chi_sgn": s, "chi_sgn_1e5": s * 2, "dai": dai})
    flip = rng.random(n) < 0.6               # a majority of rows reversed
    a2 = np.where(flip, b, a)
    b2 = np.where(flip, a, b)
    s2 = np.where(flip, -s, s)
    other = pair_table(a2, b2, values={"chi_sgn": s2, "chi_sgn_1e5": s2 * 2, "dai": dai})
    # the reported side A follows the majority orientation, so it may swap;
    # the physical summary must not: expressed in one orientation the
    # signed means agree, and the unsigned ones do not move at all
    assert np.array_equal(base.pair_id, other.pair_id)
    same = (base.side_a.to_numpy() == other.side_a.to_numpy())
    flipsign = np.where(same, 1.0, -1.0)
    assert (~same).any() and same.any()
    for col in ("chi_sgn_mean", "chi_sgn_1e5_mean"):
        np.testing.assert_allclose(base[col].to_numpy(float),
                                   other[col].to_numpy(float) * flipsign,
                                   equal_nan=True, err_msg=col)
    for col in ("dai_mean", "chi_sgn_absmax", "n_pieces"):
        np.testing.assert_allclose(base[col].to_numpy(float), other[col].to_numpy(float),
                                   equal_nan=True, err_msg=col)
    # the mean of the values re-expressed as side A minus side B, by hand
    for _, r in other.iterrows():
        m = (np.minimum(a2, b2) == r.basin_lo) & (np.maximum(a2, b2) == r.basin_hi)
        aligned = np.where(a2[m] == r.side_a, s2[m], -s2[m])
        assert np.isclose(np.nanmean(aligned), r.chi_sgn_mean, equal_nan=True)
        assert np.isclose(np.nanmean(dai[m]), r.dai_mean)


def test_signed_columns_are_named_or_detected_explicitly():
    assert is_signed_column("chi_sgn") and is_signed_column("hr_sgn_1e5")
    assert not is_signed_column("chi_diff") and not is_signed_column("dai")
    v = {"x": np.array([10., -10.])}
    assert pair_table([1., 2.], [2., 1.], values=v).x_mean.iloc[0] == 0.
    assert pair_table([1., 2.], [2., 1.], values=v, signed=("x",)).x_mean.iloc[0] == 10.
    with pytest.raises(ValueError):
        pair_table([1., 2.], [2., 1.], values=v, signed=("y",))


# ---- the multiscale table reads its own columns -------------------------------

def multiscale_table():
    return AcrossDivideTable(
        {metric_column("chi", "sgn", "_5"): np.array([2., 3.]),
         metric_column("chi", "diff", "_5"): np.array([2., 6.]),
         metric_column("hr", "sgn", "_5"): np.array([-1., 4.]),
         metric_column("hr", "diff", "_5"): np.array([1., 4.])},
        {"thresholds": [5., 15.], "tags": ["_5", "_15"], "network_threshold": 15.,
         "mn": .45, "a0": 1., "baselevel_m": 5., "seglength_m": 300.,
         "metrics": ["chi", "hr"], "n_segments": 2}, 2)


def test_coherence_and_disagreement_read_the_columns_the_producer_writes():
    t = multiscale_table()
    np.testing.assert_array_equal(coherence(t.columns, "chi", "_5"), [1., .5])
    d = t.disagreement(tag="_5")
    np.testing.assert_array_equal(d["chi_agrees_with_hr"], [True, False])
    np.testing.assert_array_equal(d["chi_coherence"], [1., .5])


def test_describe_handles_a_multiscale_table_and_missing_optional_keys():
    s = multiscale_table().describe()
    assert isinstance(s, str) and "5" in s and "15" in s and "base level" in s
    single = AcrossDivideTable({"ID": np.array([1])}, {"threshold": 1e6}, 1)
    assert "1e+06" in single.describe() and "chi was not computed" in single.describe()


def test_threshold_tags_are_distinct_for_distinct_floats_and_refuse_bad_values():
    a = 1.23456789012345
    b = float(np.nextafter(a, np.inf))
    assert threshold_tag(a) != threshold_tag(b)
    assert threshold_tag(1e6) == "_1e6" and threshold_tag(110000) == "_110000"
    assert threshold_tag(100000.4) != threshold_tag(100000.5)
    assert threshold_tag(1e-5) == "_1em5"
    assert len({threshold_tag(t) for t in (1e5, 1.1e5, 1.4e5, 1.49e5, 1e6, 2.5e6)}) == 6
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            threshold_tag(bad)


# ---- exports carry their settings ---------------------------------------------

def _table():
    return AcrossDivideTable(
        {"ID": np.array([1, 2]), "chi_sgn": np.array([1.5, -2.0])},
        {"crs": "EPSG:32623", "threshold": 123456., "threshold_units": "m2",
         "baselevel_m": 5., "mn": .45, "a0": 1., "seglength_m": 300., "metrics": ["chi"],
         "provenance": {"tag": "SETTINGS_MARKER_9034", "pydivide_version": "0.1.0"}}, 2,
        geometry=[LineString([(0, 0), (30, 30)]), LineString([(30, 30), (60, 30)])])


def test_export_writes_a_settings_sidecar_and_embeds_it_in_the_geopackage(tmp_path):
    written = _table().to_file(tmp_path / "table.gpkg")
    side = tmp_path / "table_settings.json"
    assert side in written and side.exists()
    doc = json.loads(side.read_text(encoding="utf-8"))
    assert doc["settings"]["provenance"]["tag"] == "SETTINGS_MARKER_9034"
    assert doc["settings"]["threshold"] == 123456. and doc["n_pieces"] == 2
    assert "chi_sgn" in doc["columns"]
    # the same document inside the GeoPackage, through the metadata extension
    embedded = V.read_geopackage_metadata(tmp_path / "table.gpkg", layer="divides")
    assert embedded and json.loads(embedded[-1])["settings"]["baselevel_m"] == 5.
    with sqlite3.connect(tmp_path / "table.gpkg") as con:
        ext = {r[0] for r in con.execute("SELECT table_name FROM gpkg_extensions "
                                         "WHERE extension_name = 'gpkg_metadata'")}
    assert {"gpkg_metadata", "gpkg_metadata_reference"} <= ext


def test_settings_are_recoverable_in_a_fresh_process_from_the_file_alone(tmp_path):
    _table().to_file(tmp_path / "table.gpkg")
    code = ("import sys, json; sys.path.insert(0, sys.argv[1]); "
            "from pydivide.acrossdivide import read_export_settings; "
            "print(json.dumps(read_export_settings(sys.argv[2])['settings']))")
    for target in (tmp_path / "table.gpkg", tmp_path / "table_settings.json"):
        r = subprocess.run([sys.executable, "-c", code, str(ROOT), str(target)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert json.loads(r.stdout)["provenance"]["tag"] == "SETTINGS_MARKER_9034"
    # without the sidecar the embedded copy still answers
    (tmp_path / "table_settings.json").unlink()
    assert read_export_settings(tmp_path / "table.gpkg")["settings"]["mn"] == .45


def test_settings_sidecar_follows_the_overwrite_rule(tmp_path):
    t = _table()
    t.to_file(tmp_path / "t.gpkg")
    with pytest.raises(FileExistsError):
        t.to_file(tmp_path / "t.gpkg")
    (tmp_path / "t.gpkg").unlink()
    with pytest.raises(FileExistsError):             # the sidecar alone refuses too
        t.to_file(tmp_path / "t.gpkg")
    assert not (tmp_path / "t.gpkg").exists()        # nothing was written
    t.to_file(tmp_path / "t.gpkg", overwrite=True)
    parts = t.to_file(tmp_path / "s.shp", format="shp", max_features=1)
    assert len([p for p in parts if p.suffix == ".shp"]) == 2
    assert read_export_settings(parts[0])["settings"]["threshold"] == 123456.


# ---- the pair a piece reports occurs on one of its edges -----------------------

def small_network(labels):
    sn = SortedNetwork(np.array([1, 4, 7], np.int64), np.array([0, 3], np.int64),
                       np.zeros(1, bool), np.array([1, 7], np.int64), np.array([], np.int64))
    return DivideNetworkResult(GridSpec(2, 2, 30., 0., 60.), sn,
                               np.array([0., 30., 0.]), np.array([[1, 4], [4, 7]], np.int64),
                               np.zeros(2, bool), "EPSG:32623", np.asarray(labels, np.int32))


def controlled_metrics(labels, **kw):
    """The public call on a two-cell-by-two-cell grid whose labels are
    set by hand; only the routing objects are stand-ins and only the
    label production is replaced."""
    D = small_network(labels)
    fd = SimpleNamespace(shape=(2, 2), source=np.array([], np.int64),
                         target=np.array([], np.int64), transform=from_origin(0, 60, 30, 30),
                         georef="EPSG:32623", order="C")
    st = SimpleNamespace(node_indices=np.divmod(np.arange(4), 2),
                         source=np.array([], np.int64), target=np.array([], np.int64))
    dem = SimpleNamespace(z=np.array([[30, 10], [20, 5]], np.float32), name="controlled")
    with patch.dict(sys.modules, {"topotoolbox": SimpleNamespace()}), \
            patch("pydivide.labels.basin_labels", return_value=(D.labels, np.arange(4))):
        return across_divide_metrics(dem, divides=D, flow=fd, streams=st, threshold=1,
                                     units="pixels", seglength=None, metrics=("elev",), **kw)


def test_mixed_piece_reports_a_pair_that_occurs_on_an_edge():
    tab = controlled_metrics([[1, 100], [2, 50]])
    pair = (tab.columns["basin_a"][0], tab.columns["basin_b"][0])
    assert tab.columns["basin_pair_mixed"][0] == 1
    assert pair in {(1., 100.), (2., 50.)}
    assert tab.columns["basin_pair_share"][0] == 0.5
    assert "basin_pair_share" in tab.columns and tab.settings["provenance"]["flow_supplied"]


def test_uniform_piece_reports_its_pair_with_full_share():
    tab = controlled_metrics([[3, 9], [3, 9]])
    assert (tab.columns["basin_a"][0], tab.columns["basin_b"][0]) == (3., 9.)
    assert tab.columns["basin_pair_mixed"][0] == 0 and tab.columns["basin_pair_share"][0] == 1.
    assert "computed by this call" not in tab.settings["provenance"]["routing"]


def test_objects_on_different_grids_are_refused():
    D = small_network([[1, 2], [1, 2]])
    fd = SimpleNamespace(shape=(3, 3), source=np.array([], np.int64),
                         target=np.array([], np.int64), transform=from_origin(0, 90, 30, 30),
                         georef=None, order="C")
    st = SimpleNamespace(node_indices=np.divmod(np.arange(4), 2),
                         source=np.array([], np.int64), target=np.array([], np.int64))
    dem = SimpleNamespace(z=np.zeros((2, 2), np.float32), name="x")
    with patch.dict(sys.modules, {"topotoolbox": SimpleNamespace()}), \
            pytest.raises(ValueError, match="differs from the DEM shape"):
        across_divide_metrics(dem, divides=D, flow=fd, streams=st, threshold=1,
                              units="pixels", seglength=None, metrics=("elev",))


# ---- the arrow starts on an edge that borders the pair it names ----------------

def test_direction_arrow_never_points_into_nodata():
    plt = pytest.importorskip("matplotlib.pyplot")
    from pydivide.plotting import plot_signed_divides, side_a_direction
    sn = SortedNetwork(np.array([3, 4, 7], np.int64), np.array([0, 3], np.int64),
                       np.zeros(1, bool), np.array([3, 7], np.int64), np.array([], np.int64))
    # edge 3-4 borders no-data and basin 16; edge 4-7 borders 16 and 23
    D = DivideNetworkResult(GridSpec(2, 2, 30., 0., 60.), sn, np.array([0., 30., 0.]),
                            np.array([[3, 4], [4, 7]], np.int64), np.array([True, False]),
                            "EPSG:32623", np.array([[-1, -1], [16, 23]], np.int32))
    mid, unit, ok = side_a_direction(D, 60, basin_a=np.array([16.]))
    assert ok[0] and np.allclose(mid[0], (30., 15.)) and np.allclose(unit[0], (-1., 0.))
    ax = plot_signed_divides(D, np.array([-10.]), seglength=60, tick=12,
                             basin_a=np.array([16.]), basin_b=np.array([23.]))
    try:
        assert len(ax.texts) == 1
        x, y = ax.texts[0].xy
        r, c = int(np.floor((60 - y) / 30)), int(np.floor(x / 30))
        assert D.labels[r, c] == 23
    finally:
        plt.close(ax.figure)
    # a piece with no edge bordering the named pair is not drawn
    _, _, ok = side_a_direction(D, 60, basin_a=np.array([16.]), basin_b=np.array([99.]))
    assert not ok[0]


# ---- a failed reprojection is an error, not an empty selection -----------------

def test_invalid_reprojection_raises_instead_of_selecting_nothing(tmp_path):
    D = small_network([[1, 2], [1, 2]])
    D.grid = GridSpec(2, 2, 1., 0., 100.)           # latitude 100 does not exist
    D.crs = "EPSG:4326"
    p = tmp_path / "mask.tif"
    import rasterio
    with rasterio.open(p, "w", driver="GTiff", height=2, width=2, count=1, dtype="uint8",
                       transform=from_origin(-1e7, 1e7, 1e7, 1e7), crs="EPSG:3857") as f:
        f.write(np.ones((1, 2, 2), np.uint8))
    with pytest.raises(Exception):
        D.piece_selection(None, p)


def test_one_sided_crs_is_a_warning_not_a_silent_comparison(tmp_path):
    D = small_network([[1, 2], [1, 2]])
    D.crs = None
    g = gpd.GeoDataFrame(geometry=[Polygon([(-1, -1), (100, -1), (100, 100), (-1, 100)])],
                         crs="EPSG:32623")
    g.to_file(tmp_path / "m.gpkg", driver="GPKG")
    with pytest.warns(RuntimeWarning, match="only one"):
        sel = D.piece_selection(None, tmp_path / "m.gpkg")
    assert len(sel) == 1


# ---- the flow adapter's index conventions ---------------------------------------

def test_only_c_ordered_flow_objects_are_accepted_and_indices_are_checked():
    fd = SimpleNamespace(shape=(2, 3), source=np.array([0, 2, 4], np.int64),
                         target=np.array([1, 3, 5], np.int64), transform=from_origin(0, 2, 1, 1),
                         georef=None, order="F")
    st = SimpleNamespace(node_indices=(np.array([0, 0, 0, 1, 1, 1]),
                                       np.array([0, 1, 2, 0, 1, 2])),
                         source=np.array([0, 1, 2], np.int64), target=np.array([3, 4, 5], np.int64))
    with pytest.raises(ValueError, match="C-ordered"):          # the one supported layout
        FlowNet.from_pytopotoolbox(fd, st)
    fd.order = "C"
    net_c = FlowNet.from_pytopotoolbox(fd, st)
    np.testing.assert_array_equal(net_c.fd_src, [0, 2, 4])
    np.testing.assert_array_equal(net_c.st_src, [0, 1, 2])
    fd.order = "K"
    with pytest.raises(ValueError, match="C-ordered"):
        FlowNet.from_pytopotoolbox(fd, st)
    fd.order = "C"
    fd.source = np.array([0, 2, 9], np.int64)
    with pytest.raises(ValueError, match="outside the grid"):
        FlowNet.from_pytopotoolbox(fd, st)


# ---- cycles are rejected, never looped on or collapsed -------------------------

@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("operation", [first_stream_cell_downstream, distance_to_stream])
def test_flow_operations_refuse_cycles(n, operation):
    ids = np.array([0, 1, 3, 2][:n], np.int64)
    empty = np.array([], np.int64)
    net = FlowNet(GridSpec(2, 2, 30., 0., 60.), ids, np.roll(ids, -1), empty, empty, empty, empty)
    with pytest.raises(ValueError, match="cycle"):
        operation(net)


def test_reach_outlet_propagation_rejects_a_cycle_and_keeps_a_forest():
    with pytest.raises(ValueError, match="cycle"):
        reach_outlet_labels(np.array([1, 2, 3]), np.array([0, 1, 2]), np.array([1, 2, 0]))
    L = np.array([1, 2, 2, 3, 4, 3, 6, 0], dtype=np.int64)
    out = reach_outlet_labels(L, np.array([0, 2, 4], np.int64), np.array([1, 3, 5], np.int64))
    assert out.tolist() == [0, 3, 3, 3, 3, 5, 6]
    n = 5000
    chain = reach_outlet_labels(np.arange(1, n + 1, dtype=np.int64),
                                np.arange(0, n - 1, dtype=np.int64),
                                np.arange(1, n, dtype=np.int64))
    assert np.all(chain[1:] == n)


def test_malformed_stream_endpoint_raises_value_error():
    grid = GridSpec(nr=4, nc=4, cellsize=30.0, x0=0.0, y0=0.0)
    net = FlowNet(grid=grid, fd_src=np.array([1, 5], np.int64), fd_dst=np.array([5, 6], np.int64),
                  st_nodes=np.array([5, 6], np.int64), st_src=np.array([5], np.int64),
                  st_dst=np.array([99], np.int64), outlets=np.array([6], np.int64), crs=None)
    with pytest.raises(ValueError, match="not in stream node list"):
        basin_labels(net)


# ---- checkpoints bind to every byte ---------------------------------------------

def test_fingerprint_hashes_every_byte():
    a = np.arange(5_000_000, dtype=np.int64)
    b = a.copy()
    b[1_234_567] += 1
    b[3_333_333] -= 1                      # same sum, one changed pair
    assert array_fingerprint(a) != array_fingerprint(b)
    assert array_fingerprint(a) == array_fingerprint(a.copy())


def test_interrupted_save_cannot_pass_off_another_runs_payload(tmp_path):
    import shutil
    sa = StageStore(tmp_path / "a")
    sa.save("labels", {"L": np.arange(10)}, {"run": "A"})
    assert sa.has("labels", {"run": "A"})
    sb = StageStore(tmp_path / "b")
    sb.save("labels", {"L": np.arange(20)}, {"run": "B"})
    shutil.copy2(sb.path("labels"), sa.path("labels"))
    sa2 = StageStore(tmp_path / "a")
    assert not sa2.has("labels", {"run": "A"})
    assert not sa2.has("labels", {"run": "B"})


def test_south_up_transform_is_rejected():
    from affine import Affine
    with pytest.raises(ValueError, match="north-up"):
        GridSpec.from_rasterio_transform(10, 10, Affine(30, 0, 0, 0, 30, 0))
    g = GridSpec.from_rasterio_transform(10, 10, Affine(30, 0, 100, 0, -30, 900))
    assert g.cellsize == 30 and g.x0 == 100 and g.y0 == 900


def test_mixed_pair_flag_is_exact_for_large_labels():
    lo = np.array([7_000_001.0, 7_000_001.0, 7_000_001.0])
    hi = np.array([7_000_002.0, 7_000_002.0, 7_000_003.0])
    ps, pe = np.array([0]), np.array([4])
    lo_mixed = aggregate_to_pieces(lo, ps, pe, "min") != aggregate_to_pieces(lo, ps, pe, "max")
    hi_mixed = aggregate_to_pieces(hi, ps, pe, "min") != aggregate_to_pieces(hi, ps, pe, "max")
    assert bool(lo_mixed | hi_mixed)


# ---- Shapefile: names and sizes match what the driver writes -------------------

def test_shapefile_names_never_collide_with_existing_short_names():
    names = ["abcdefghijk", "abcdefghij", "ABCDEFGHI1", "abcdefghijz", "geometry"]
    mapping = V.truncate_field_names(names)
    renamed = [mapping.get(n, n) for n in names]
    assert len({r.lower() for r in renamed}) == len(renamed), renamed
    assert all(len(r) <= V.SHAPEFILE_NAME_LIMIT for r in renamed)
    assert "abcdefghij" not in mapping and "geometry" not in mapping


def _dbf_widths(path):
    with open(Path(path).with_suffix(".dbf"), "rb") as fh:
        hdr = fh.read(32)
        n_rec, hlen, rlen = struct.unpack("<IHH", hdr[4:12])
        out = {}
        for _ in range((hlen - 33) // 32):
            f = fh.read(32)
            out[f[:11].split(b"\0")[0].decode()] = f[16]
        return out, rlen, hlen


def test_dbf_widths_equal_the_widths_the_driver_writes(tmp_path):
    g = gpd.GeoDataFrame({"s3": ["abc"] * 3, "s254": ["x" * 254] * 3, "f": [1.5] * 3,
                          "i64": np.array([1, 2, 3], np.int64),
                          "i32": np.array([1, 2, 3], np.int32),
                          "isbnd": [True, False, True]},
                         geometry=[LineString([(0, 0), (1, 1)])] * 3, crs="EPSG:32623")
    out = V.write_shapefile(g, tmp_path / "w.shp")
    written, rlen, hlen = _dbf_widths(out[0])
    expect = V.dbf_field_widths(V._shapefile_safe(g))     # isbnd is cast to int32 by the writer
    assert written == expect, (written, expect)
    assert rlen == V.dbf_record_bytes(expect)
    assert hlen + 1 == V.dbf_header_bytes(len(expect))
    assert Path(out[0]).with_suffix(".dbf").stat().st_size == hlen + 3 * rlen + 1


def test_shapefile_parts_respect_the_budget_with_wide_strings(tmp_path):
    g = gpd.GeoDataFrame({"text": ["x" * 254] * 20},
                         geometry=[LineString([(0, 0), (1, 1)])] * 20, crs="EPSG:32623")
    with patch.object(V, "FILE_BUDGET_BYTES", 2000):
        paths = V.write_shapefile(g, tmp_path / "wide.shp")
    sizes = [p.with_suffix(".dbf").stat().st_size for p in paths]
    assert max(sizes) <= 2000 and sum(len(gpd.read_file(p)) for p in paths) == 20
    with patch.object(V, "FILE_BUDGET_BYTES", 200), pytest.raises(ValueError, match="single"):
        V.write_shapefile(g, tmp_path / "toowide.shp")


def test_shp_record_bytes_cover_every_geometry_type():
    geoms = [Point(0, 0), LineString([(0, 0), (1, 1), (2, 0)]),
             MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
             Polygon([(0, 0), (1, 0), (1, 1)], [[(0.2, 0.2), (0.4, 0.2), (0.4, 0.4)]])]
    rec = V.shp_record_bytes(np.array(geoms, dtype=object))
    assert rec.tolist() == [28, 52 + 4 + 16 * 3, 52 + 8 + 16 * 4, 52 + 8 + 16 * 8]
    assert V.shp_bytes_for_vertices([3])[0] == rec[1]


def test_crs_repair_keeps_an_existing_wkt2(tmp_path):
    p = tmp_path / "r.gpkg"
    gpd.GeoDataFrame({"a": [1]}, geometry=[LineString([(0, 0), (1, 1)])],
                     crs="EPSG:32632").to_file(p, driver="GPKG")
    V.repair_geopackage_crs(p)
    kept = 'PROJCRS["kept"]'
    con = sqlite3.connect(p)
    con.execute("UPDATE gpkg_spatial_ref_sys SET definition_12_063 = ? WHERE srs_id = 32632",
                (kept,))
    con.commit()
    con.close()
    V.repair_geopackage_crs(p)
    con = sqlite3.connect(p)
    vals = [r[0] for r in con.execute("SELECT definition_12_063 FROM gpkg_spatial_ref_sys "
                                      "WHERE srs_id = 32632")]
    con.close()
    assert vals == [kept]

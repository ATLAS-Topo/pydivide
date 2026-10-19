# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Contracts on the inputs and files of the across-divide workflow that
hold without a DEM or a routing library: objects that must describe one
grid, the memory order pydivide accepts, the upstream areas handed to
the chi integration, which settings document a file answers with, what
the settings record, profiles of short divides, the reach-outlet cycle
guard, and the Shapefile file set, widths and names."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydivide import vectorio as V  # noqa: E402
from pydivide.acrossdivide import (  # noqa: E402
    AcrossDivideTable,
    across_divide_metrics,
    check_same_grid,
    read_export_settings,
)
from pydivide.divmetrics import reach_outlet_labels  # noqa: E402
from pydivide.flownet import FlowNet  # noqa: E402
from pydivide.gridspec import GridSpec  # noqa: E402
from pydivide.orderdist import SortedNetwork  # noqa: E402
from pydivide.pipeline import DivideNetworkResult  # noqa: E402
from pydivide.profiles import basin_pair_profiles  # noqa: E402

TR = from_origin(0, 60, 30, 30)


def network(labels, crs="EPSG:32623", grid=None):
    sn = SortedNetwork(np.array([1, 4, 7], np.int64), np.array([0, 3], np.int64),
                       np.zeros(1, bool), np.array([1, 7], np.int64), np.array([], np.int64))
    return DivideNetworkResult(grid or GridSpec(2, 2, 30., 0., 60.), sn,
                               np.array([0., 30., 0.]), np.array([[1, 4], [4, 7]], np.int64),
                               np.zeros(2, bool), crs, np.asarray(labels, np.int32))


def objects(labels=((1, 100), (2, 50)), *, dem_tr=TR, dem_crs="EPSG:32623", fd_tr=TR,
            fd_crs="EPSG:32623", order="C", divides=None):
    D = divides if divides is not None else network(labels)
    fd = SimpleNamespace(shape=(2, 2), source=np.array([], np.int64),
                         target=np.array([], np.int64), transform=fd_tr, georef=fd_crs,
                         order=order, cellsize=30.0)
    st = SimpleNamespace(node_indices=np.divmod(np.arange(4), 2),
                         source=np.array([], np.int64), target=np.array([], np.int64),
                         shape=(2, 2), transform=fd_tr, georef=fd_crs)
    dem = SimpleNamespace(z=np.array([[30, 10], [20, 5]], np.float32), name="controlled",
                          transform=dem_tr, georef=dem_crs, shape=(2, 2), cellsize=30.0)
    return dem, fd, st, D


def run(dem, fd, st, D, **kw):
    with patch.dict(sys.modules, {"topotoolbox": SimpleNamespace()}), \
            patch("pydivide.labels.basin_labels", return_value=(D.labels, np.arange(4))):
        return across_divide_metrics(dem, divides=D, flow=fd, streams=st, threshold=1,
                                     units="pixels", seglength=None,
                                     metrics=kw.pop("metrics", ("elev",)), **kw)


# ---- objects must describe one grid ------------------------------------------

def test_matching_objects_are_accepted():
    tab = run(*objects())
    assert np.isfinite(tab.columns["elev_sgn"][0])


@pytest.mark.parametrize("case", ["dem_shifted", "dem_crs", "divides_shifted",
                                  "divides_cellsize", "divides_crs"])
def test_same_shape_different_grid_is_refused(case):
    kw = {}
    if case == "dem_shifted":
        kw["dem_tr"] = from_origin(1000, 60, 30, 30)
    elif case == "dem_crs":
        kw["dem_crs"] = "EPSG:32624"
    elif case == "divides_shifted":
        kw["divides"] = network([[1, 100], [2, 50]], grid=GridSpec(2, 2, 30., 1000., 60.))
    elif case == "divides_cellsize":
        kw["divides"] = network([[1, 100], [2, 50]], grid=GridSpec(2, 2, 60., 0., 60.))
    elif case == "divides_crs":
        kw["divides"] = network([[1, 100], [2, 50]], crs="EPSG:32624")
    with pytest.raises(ValueError, match="differs"):
        run(*objects(**kw))


def test_equivalent_crs_encodings_are_the_same_grid():
    from pyproj import CRS
    dem, fd, st, D = objects(dem_crs=CRS.from_epsg(32623).to_wkt(), fd_crs="EPSG:32623")
    D.crs = "+proj=utm +zone=23 +datum=WGS84 +units=m +no_defs +type=crs"
    assert np.isfinite(run(dem, fd, st, D).columns["elev_sgn"][0])
    check_same_grid(dem, "a", fd, "b")


def test_missing_georeferencing_is_a_warning_not_agreement():
    dem, fd, st, D = objects()
    del dem.transform
    with pytest.warns(RuntimeWarning, match="transform"):
        check_same_grid(dem, "the DEM", fd, "the flow object")
    dem, fd, st, D = objects(dem_crs=None)
    with pytest.warns(RuntimeWarning, match="coordinate reference"):
        check_same_grid(dem, "the DEM", fd, "the flow object")


def test_stream_network_on_another_grid_is_refused():
    dem, fd, st, D = objects()
    st.shape = (3, 3)
    with pytest.raises(ValueError, match="grid"):
        FlowNet.from_pytopotoolbox(fd, st)


# ---- memory order: one policy at every entry ----------------------------------

def test_fortran_ordered_objects_are_refused_everywhere():
    dem, fd, st, D = objects(order="F")
    with pytest.raises(ValueError, match="C-ordered"):
        FlowNet.from_pytopotoolbox(fd, st)
    with pytest.raises(ValueError, match="C-ordered"):
        run(dem, fd, st, D)
    from pydivide.streams import stream_segments_gdf
    with pytest.raises(ValueError, match="C-ordered"):
        stream_segments_gdf(fd, st, dem, seglength=30, chi_field="chi", baselevel=0)


def test_chi_upstream_areas_are_gathered_by_row_and_column():
    """The areas handed to the chi integration are read by the nodes'
    rows and columns, so they do not depend on how the grid is stored."""
    dem, fd, st, D = objects()
    acc = np.array([[1, 1], [2, 4]], np.float32)
    fd.flow_accumulation = lambda: acc
    got = {}

    class Trimmed:
        stream = np.array([0, 2, 1, 3])                # deliberately not row-major
        node_indices = (np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]))
        source = np.array([], np.int64)
        target = np.array([], np.int64)
        shape = (2, 2)
        transform = TR

        def chitransform(self, a, **kw):
            got["areas"] = np.asarray(a).tolist()
            return np.zeros(4, np.float32)

    st.upstreamto = lambda mask: SimpleNamespace(clean=lambda: Trimmed())
    with patch("pydivide.grids.nal_to_hillslopes", return_value=np.zeros(4, np.float32)):
        tab = run(dem, fd, st, D, metrics=("chi",), baselevel=1.0)
    assert got["areas"] == [1.0, 2.0, 1.0, 4.0]
    assert tab.settings["chi_computed"] and "chi_sgn" in tab.columns


# ---- which settings document a file answers with -------------------------------

def _table(threshold):
    return AcrossDivideTable({"ID": np.array([1])},
                             {"crs": "EPSG:32623", "threshold": float(threshold)}, 1,
                             geometry=[LineString([(0, 0), (30, 30)])])


def test_a_file_answers_with_its_own_settings_only(tmp_path):
    _table(100).to_file(tmp_path / "metrics.gpkg")
    _table(200).to_file(tmp_path / "metrics_part001.gpkg")
    assert read_export_settings(tmp_path / "metrics.gpkg")["settings"]["threshold"] == 100.
    assert read_export_settings(tmp_path / "metrics_part001.gpkg")["settings"]["threshold"] == 200.
    (tmp_path / "metrics_part001_settings.json").unlink()      # embedded copy answers
    assert read_export_settings(tmp_path / "metrics_part001.gpkg")["settings"]["threshold"] == 200.
    # a neighbouring sidecar never answers for a GeoPackage
    (tmp_path / "metrics_part001.gpkg").unlink()
    with pytest.raises(FileNotFoundError):
        read_export_settings(tmp_path / "metrics_part001.gpkg")


def test_split_shapefile_parts_answer_from_their_own_export(tmp_path):
    t = AcrossDivideTable({"ID": np.array([1, 2]), "v": np.array([1., 2.])},
                          {"crs": "EPSG:32623", "threshold": 300.}, 2,
                          geometry=[LineString([(0, 0), (30, 30)]), LineString([(0, 0), (1, 1)])])
    parts = t.to_file(tmp_path / "s.shp", format="shp", max_features=1)
    _table(999).to_file(tmp_path / "s_part001.gpkg")
    for p in parts:
        if p.suffix == ".shp":
            assert read_export_settings(p)["settings"]["threshold"] == 300.
    assert read_export_settings(tmp_path / "s_part001.gpkg")["settings"]["threshold"] == 999.
    # a part not listed by the export's sidecar is not answered by it
    stray = tmp_path / "s_part009.shp"
    stray.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="no settings document lists"):
        read_export_settings(stray)


def test_disagreeing_sidecar_and_embedded_settings_are_reported(tmp_path):
    _table(100).to_file(tmp_path / "t.gpkg")
    side = tmp_path / "t_settings.json"
    doc = json.loads(side.read_text(encoding="utf-8"))
    doc["settings"]["threshold"] = 5.
    side.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        read_export_settings(tmp_path / "t.gpkg")


# ---- what the settings record, and what the description claims ------------------

def test_side_rule_is_recorded_with_and_without_a_reference():
    from pydivide.checkpoint import array_fingerprint
    dem, fd, st, D = objects()
    tab = run(dem, fd, st, D)
    assert "lower reach-basin label" in tab.settings["side_rule"]
    assert "side_reference_fingerprint" not in tab.settings
    ref = np.array([[9, 1], [9, 1]], np.int32)
    tab2 = run(dem, fd, st, D, side_reference=ref)
    assert "caller-supplied" in tab2.settings["side_rule"]
    assert tab2.settings["side_reference_fingerprint"] == array_fingerprint(ref.ravel())
    assert tab2.settings["metrics_computed"] == ["elev"] and not tab2.settings["chi_computed"]


def test_description_does_not_claim_chi_that_was_not_computed():
    dem, fd, st, D = objects()
    tab = run(dem, fd, st, D, baselevel=1.0)            # elevation only, base level given
    text = tab.describe()
    assert "chi was not computed" in text and "elev" in text
    hand = AcrossDivideTable({"ID": np.array([1])}, {"baselevel_m": 5., "metrics": ["hr"]}, 1)
    assert "chi was not computed" in hand.describe()
    hand2 = AcrossDivideTable({"ID": np.array([1])}, {"baselevel_m": 5., "metrics": ["chi"]}, 1)
    assert "integrated" in hand2.describe()


# ---- profiles of short divides -----------------------------------------------------

def test_single_edge_profile_spans_its_section_and_samples_both_sides():
    D = network([[3, 9], [1, 2]])                # the pair (3, 9) has exactly one edge
    dem = SimpleNamespace(z=np.array([[30, 10], [20, 5]], np.float32))
    p = basin_pair_profiles(D, dem, pair=(3, 9), spacing=30, half_length=30, step=30,
                            labels=D.labels)
    assert len(p.across) == 1 and p.settings["n_edges"] == 1
    xy = p.across[0]["xy"]
    length = float(np.hypot(*(xy[-1] - xy[0])))
    assert np.isclose(length, 60.0), xy
    assert len(np.unique(xy, axis=0)) == 3
    # the section crosses the edge: one end in basin 3, the other in basin 9
    L = D.labels
    ends = [L[int((60 - y) // 30), int(x // 30)] for x, y in xy if 0 <= x < 60 and 0 <= y < 60]
    assert {3, 9} <= set(int(v) for v in ends)
    assert (p.label_a, p.label_b) in {(3, 9), (9, 3)}
    # a divide of several edges keeps its smoothed tangent
    D2 = network([[3, 9], [3, 9]])
    p2 = basin_pair_profiles(D2, dem, pair=(3, 9), spacing=30, half_length=30, step=30,
                             labels=D2.labels)
    assert p2.settings["n_edges"] == 2 and all(
        np.hypot(*(s["xy"][-1] - s["xy"][0])) > 0 for s in p2.across)


def test_profile_by_segment_uses_an_observed_pair():
    D = network([[1, 100], [2, 50]])
    dem = SimpleNamespace(z=np.array([[30, 10], [20, 5]], np.float32))
    p = basin_pair_profiles(D, dem, segid=1, spacing=30, half_length=30, labels=D.labels)
    assert tuple(sorted(p.pair)) in {(1, 100), (2, 50)}


# ---- reach-outlet cycles of every length are rejected ----------------------------

@pytest.mark.parametrize("n", [2, 3, 4, 5, 8])
def test_reach_cycles_of_any_length_are_rejected(n):
    labels = np.arange(1, n + 1)
    src = np.arange(n)
    dst = (np.arange(n) + 1) % n
    with pytest.raises(ValueError, match="cycle"):
        reach_outlet_labels(labels, src, dst)
    # a tree feeding the cycle is rejected too
    labels2 = np.r_[labels, n + 1, n + 2]
    src2 = np.r_[src, n, n + 1]
    dst2 = np.r_[dst, n + 1, 0]
    with pytest.raises(ValueError, match="cycle"):
        reach_outlet_labels(labels2, src2, dst2)


def test_reach_forests_still_resolve():
    L = np.array([1, 2, 2, 3, 4, 3, 6, 0], dtype=np.int64)
    out = reach_outlet_labels(L, np.array([0, 2, 4], np.int64), np.array([1, 3, 5], np.int64))
    assert out.tolist() == [0, 3, 3, 3, 3, 5, 6]
    n = 3000
    chain = reach_outlet_labels(np.arange(1, n + 1, dtype=np.int64),
                                np.arange(0, n - 1, dtype=np.int64),
                                np.arange(1, n, dtype=np.int64))
    assert np.all(chain[1:] == n)


# ---- Shapefile: widths, dimensions, the file set and the names ------------------

def _dbf_widths(path):
    with open(Path(path).with_suffix(".dbf"), "rb") as fh:
        hdr = fh.read(32)
        _, hlen, rlen = struct.unpack("<IHH", hdr[4:12])
        out = {}
        for _ in range((hlen - 33) // 32):
            f = fh.read(32)
            out[f[:11].split(b"\0")[0].decode()] = f[16]
        return out, rlen


def test_integer_widths_cover_the_widest_values(tmp_path):
    g = gpd.GeoDataFrame({"i32lo": np.array([-2**31 + 1, 5], np.int32),
                          "i32hi": np.array([2**31 - 1, 5], np.int32),
                          "i64lo": np.array([-2**63 + 1, 5], np.int64),
                          "i64hi": np.array([2**63 - 1, 5], np.int64),
                          "small": np.array([1, 2], np.int64), "f": [1e300, -1e300]},
                         geometry=[LineString([(0, 0), (1, 1)])] * 2, crs="EPSG:32623")
    out = V.write_shapefile(g, tmp_path / "w.shp")
    written, rlen = _dbf_widths(out[0])
    predicted = V.dbf_field_widths(V._shapefile_safe(g))
    for name, w in written.items():
        assert w <= predicted[name], (name, w, predicted[name])
    assert written["i32lo"] == 11 and written["i64lo"] == 20 and written["small"] == 18
    assert rlen <= V.dbf_record_bytes(predicted)


def test_ten_integer_columns_fit_a_small_budget(tmp_path):
    data = {f"c{k}": np.array([-2**31 + 1] * 30, np.int32) for k in range(10)}
    g = gpd.GeoDataFrame(data, geometry=[LineString([(0, 0), (1, 1)])] * 30, crs="EPSG:32623")
    with patch.object(V, "FILE_BUDGET_BYTES", 5000):
        paths = V.write_shapefile(g, tmp_path / "ints.shp")
    assert all(p.with_suffix(".dbf").stat().st_size <= 5000 for p in paths)


def test_z_geometries_are_sized_and_m_refused(tmp_path):
    gz = gpd.GeoDataFrame({"a": [1, 2, 3]},
                          geometry=[Point(0, 0, 5), Point(1, 1, 6), Point(2, 2, 7)],
                          crs="EPSG:32623")
    assert V.shp_record_bytes(gz.geometry.values).tolist() == [36, 36, 36]
    with patch.object(V, "FILE_BUDGET_BYTES", 190):
        paths = V.write_shapefile(gz, tmp_path / "z.shp", max_features=100)
    assert all(p.stat().st_size <= 190 for p in paths)
    gl = gpd.GeoDataFrame({"a": [1]}, geometry=[LineString([(0, 0, 1), (1, 1, 2), (2, 0, 3)])],
                          crs="EPSG:32623")
    out = V.write_shapefile(gl, tmp_path / "lz.shp")
    assert out[0].stat().st_size == 100 + int(V.shp_record_bytes(gl.geometry.values)[0])
    import shapely
    from shapely import from_wkt
    if hasattr(shapely, "has_m"):              # below Shapely 2.1 a measure cannot exist
        gm = gpd.GeoDataFrame({"a": [1]}, geometry=[from_wkt("POINT M (1 2 3)")],
                              crs="EPSG:32623")
        with pytest.raises(ValueError, match="measure"):
            V.write_shapefile(gm, tmp_path / "m.shp")


def _hashes(paths):
    import hashlib
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}


def test_shapefile_refuses_when_any_companion_exists(tmp_path):
    g = gpd.GeoDataFrame({"a": [5]}, geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:32623")
    # an existing dataset whose .shp is gone: the .dbf must survive untouched
    V.write_shapefile(gpd.GeoDataFrame({"a": [99]}, geometry=[LineString([(0, 0), (1, 1)])],
                                       crs="EPSG:32623"), tmp_path / "r.shp")
    (tmp_path / "r.shp").unlink()
    before = _hashes(V.shapefile_targets(tmp_path / "r.shp"))
    with pytest.raises(FileExistsError):
        V.write_shapefile(g, tmp_path / "r.shp")
    assert _hashes(V.shapefile_targets(tmp_path / "r.shp")) == before
    # an existing field-name sidecar alone is enough to refuse
    side = tmp_path / "q_fields.csv"
    side.write_text("KEEP\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        V.write_shapefile(g, tmp_path / "q.shp")
    assert side.read_text(encoding="utf-8") == "KEEP\n"
    # with overwrite the whole set is replaced, stale companions included
    (tmp_path / "r.qix").write_bytes(b"stale")
    V.write_shapefile(g, tmp_path / "r.shp", overwrite=True)
    assert not (tmp_path / "r.qix").exists()
    assert gpd.read_file(tmp_path / "r.shp")["a"].iloc[0] == 5


def test_case_only_short_names_are_made_unique_and_recorded(tmp_path):
    names = ["Value", "value", "VALUE", "abcdefghijk", "abcdefghij", "geometry"]
    m = V.truncate_field_names(names)
    final = [m.get(n, n) for n in names if n != "geometry"]
    assert len({f.lower() for f in final}) == len(final), final
    assert "value" in m and "VALUE" in m and "Value" not in m
    g = gpd.GeoDataFrame({"Value": [1], "value": [2], "VALUE": [3]},
                         geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:32623")
    out = V.write_shapefile(g, tmp_path / "c.shp")
    back = gpd.read_file(out[0])
    assert sorted(back[c].iloc[0] for c in back.columns if c != "geometry") == [1, 2, 3]
    side = (tmp_path / "c_fields.csv").read_text(encoding="utf-8")
    assert "value" in side and "VALUE" in side


# ---- registration at every public boundary ---------------------------------------

@pytest.mark.parametrize("case", ["shifted", "cellsize", "crs"])
def test_profiles_refuse_a_dem_on_another_grid(case):
    D = network([[3, 9], [1, 2]])
    dem = SimpleNamespace(z=np.array([[30, 10], [20, 5]], np.float32), shape=(2, 2),
                          transform=TR, georef="EPSG:32623", cellsize=30.0)
    if case == "shifted":
        dem.transform = from_origin(1000, 60, 30, 30)
    elif case == "cellsize":
        dem.transform = from_origin(0, 120, 60, 60)
        dem.cellsize = 60.0
    else:
        dem.georef = "EPSG:32624"
    with pytest.raises(ValueError, match="differs"):
        basin_pair_profiles(D, dem, pair=(3, 9), spacing=30, half_length=30, labels=D.labels)
    good = SimpleNamespace(z=dem.z, shape=(2, 2), transform=TR, georef="EPSG:32623", cellsize=30.0)
    p = basin_pair_profiles(D, good, pair=(3, 9), spacing=30, half_length=30, step=30,
                            labels=D.labels)
    assert len(p.across) == 1 and np.hypot(*(p.across[0]["xy"][-1] - p.across[0]["xy"][0])) > 0


def test_flow_and_stream_objects_in_different_systems_are_refused():
    dem, fd, st, D = objects()
    st.georef = "EPSG:32624"
    with pytest.raises(ValueError, match="coordinate reference"):
        FlowNet.from_pytopotoolbox(fd, st)
    st.georef = "+proj=utm +zone=23 +datum=WGS84 +units=m +no_defs +type=crs"
    FlowNet.from_pytopotoolbox(fd, st)                 # the same system, another encoding


def test_stated_cell_sizes_are_compared_even_without_transforms():
    a = SimpleNamespace(shape=(2, 2), cellsize=30.0)
    b = SimpleNamespace(shape=(2, 2), cellsize=60.0)
    with pytest.raises(ValueError, match="cell size"):
        check_same_grid(a, "a", b, "b")
    c = SimpleNamespace(shape=(2, 2), cellsize=30.0)
    with pytest.warns(RuntimeWarning, match="transform"):
        check_same_grid(a, "a", c, "c")


# ---- Shapefile writer at the declared Shapely floor, and layer dimension ------------

def test_shapefile_export_works_without_the_measure_detector(tmp_path, monkeypatch):
    """Shapely below 2.1 has no ``has_m`` and cannot carry a measure, so
    an ordinary export must still work there."""
    import shapely
    monkeypatch.delattr(shapely, "has_m", raising=False)
    g = gpd.GeoDataFrame({"a": [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:32623")
    out = V.write_shapefile(g, tmp_path / "p.shp")
    assert len(gpd.read_file(out[0])) == 2
    assert not V.has_measure(g.geometry.values).any()


@pytest.mark.parametrize("nv", [5, 10])
def test_mixed_dimension_layers_are_sized_as_the_driver_writes_them(tmp_path, nv):
    xyz = LineString([(i, i, 1.0) for i in range(nv)])
    xy = LineString([(i, -i) for i in range(nv)])
    g = gpd.GeoDataFrame({"a": [1, 2]}, geometry=[xyz, xy], crs="EPSG:32623")
    rec = V.shp_record_bytes(g.geometry.values, force_z=True)
    budget = 100 + int(rec[0])                      # one record fits, two do not
    with patch.object(V, "FILE_BUDGET_BYTES", budget):
        paths = V.write_shapefile(g, tmp_path / "mixed.shp", max_features=100)
    assert len(paths) == 2
    for p in paths:
        assert p.stat().st_size <= budget, (p, p.stat().st_size, budget)
    single = V.write_shapefile(g, tmp_path / "one.shp")
    assert single[0].stat().st_size == 100 + int(rec.sum())


def test_a_shorter_export_does_not_leave_earlier_parts(tmp_path):
    g3 = gpd.GeoDataFrame({"a": [1, 2, 3]}, geometry=[Point(i, i) for i in range(3)],
                          crs="EPSG:32623")
    parts = V.write_shapefile(g3, tmp_path / "out.shp", max_features=1)
    assert len(parts) == 3
    g1 = g3.iloc[:1]
    with pytest.raises(FileExistsError):              # earlier parts occupy the name
        V.write_shapefile(g1, tmp_path / "out.shp")
    out = V.write_shapefile(g1, tmp_path / "out.shp", overwrite=True)
    assert out == [tmp_path / "out.shp"]
    assert not list(tmp_path.glob("out_part*")), list(tmp_path.glob("out_part*"))
    # an unrelated dataset whose name merely starts the same way is untouched
    V.write_shapefile(g1, tmp_path / "out_other.shp")
    V.write_shapefile(g1, tmp_path / "out.shp", overwrite=True)
    assert (tmp_path / "out_other.shp").exists()


def test_empty_file_list_answers_for_no_file(tmp_path):
    side = tmp_path / "missing_settings.json"
    side.write_text(json.dumps({"files": [], "settings": {"threshold": 42}}), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        read_export_settings(tmp_path / "missing.gpkg")
    assert read_export_settings(side)["settings"]["threshold"] == 42   # by its own path

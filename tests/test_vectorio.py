# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Vector export: the two size limits, the CRS repair, and the .gdb trap.

Every test here is synthetic — no reference data, no DEM — so a stranger
who clones the repository can run them.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydivide import vectorio as V  # noqa: E402

# A custom projected CRS with no EPSG authority — this is what triggers the
# "coordinate system description is invalid" complaint in ArcGIS, because
# GDAL writes it with organization='NONE' and a WKT1 definition only. The
# real exports use exactly this projection.
ALBERS_SA = ("+proj=aea +lat_0=-32 +lon_0=-60 +lat_1=-5 +lat_2=-42 "
             "+x_0=0 +y_0=0 +ellps=aust_SA +units=m +no_defs")


def _gdf(n=50, extra_fields=0, crs=ALBERS_SA):
    rng = np.random.default_rng(0)
    geoms = [LineString([(i * 30.0, 0.0), (i * 30.0, 30.0)]) for i in range(n)]
    data = {
        "ID": np.arange(1, n + 1, dtype=np.int64),
        "segid": np.arange(1, n + 1, dtype=np.int64) * 7,
        "order": np.ones(n, dtype=np.int64),
        "distance": rng.random(n) * 1000.0,
        "isbnd": np.zeros(n, dtype=bool),
        "isring": np.zeros(n, dtype=bool),
    }
    for k in range(extra_fields):
        data[f"metric_with_a_long_name_{k:03d}"] = rng.random(n)
    return gpd.GeoDataFrame(data, geometry=geoms, crs=crs)


# ---------------------------------------------------------------------------
# size arithmetic: the two limits this module exists to respect

def test_capacity_is_geometry_bound_when_narrow():
    """With few fields the .shp binds, so capacity is field-independent."""
    assert V.shapefile_capacity(4) == V.shapefile_capacity(0)


def test_capacity_becomes_attribute_bound_as_fields_grow():
    """Splitting on feature count alone would exceed the .dbf limit: with
    ~106 metric columns the .dbf is the limit that binds first, and by a
    wide margin."""
    narrow = V.shapefile_capacity(4)
    wide = V.shapefile_capacity(106)
    assert wide < narrow
    # 106 float64 fields at 24 bytes is ~2.5 kB per row, after the header
    assert wide == (V.FILE_BUDGET_BYTES - V.dbf_header_bytes(106)) // (1 + 106 * V.BYTES_PER_FIELD)
    # the exact row width of a real table narrows or widens the estimate
    assert V.shapefile_capacity(106, row_bytes=1 + 106 * 9) > wide


def test_capacity_decreases_monotonically_with_fields():
    caps = [V.shapefile_capacity(k) for k in (0, 1, 10, 50, 106, 250)]
    assert caps == sorted(caps, reverse=True)
    assert all(c >= 1 for c in caps)


def test_parts_needed():
    cap = V.shapefile_capacity(106)
    assert V.shapefile_parts(cap, 106) == 1
    assert V.shapefile_parts(cap + 1, 106) == 2
    assert V.shapefile_parts(0, 106) == 0


# ---------------------------------------------------------------------------
# field names

def test_short_names_are_left_completely_alone():
    """Existing exports already use <=10-character names; they must not
    change, or every downstream ArcGIS project breaks."""
    assert V.truncate_field_names(
        ["ID", "segid", "order", "distance", "geometry"]) == {}


def test_long_names_are_truncated_uniquely():
    m = V.truncate_field_names(
        ["a_very_long_name_one", "a_very_long_name_two", "geometry"])
    assert len(set(m.values())) == 2, "collisions must be resolved"
    assert all(len(v) <= V.SHAPEFILE_NAME_LIMIT for v in m.values())


def test_geometry_column_is_never_renamed():
    assert "geometry" not in V.truncate_field_names(["geometry"])


# ---------------------------------------------------------------------------
# GeoPackage + the CRS repair

def test_geopackage_roundtrip_and_crs_repair(tmp_path):
    g = _gdf(40)
    p = V.write_geopackage(g, tmp_path / "divides.gpkg")
    assert p.exists()

    back = gpd.read_file(p)
    assert len(back) == 40
    assert list(back["ID"]) == list(g["ID"])

    con = sqlite3.connect(str(p))
    try:
        cols = [r[1] for r in
                con.execute("PRAGMA table_info(gpkg_spatial_ref_sys)")]
        assert "definition_12_063" in cols, "WKT2 column missing"
        wkt2 = [r[0] for r in con.execute(
            "SELECT definition_12_063 FROM gpkg_spatial_ref_sys "
            "WHERE definition_12_063 IS NOT NULL "
            "AND definition_12_063 != 'undefined'")]
        assert wkt2, "no CRS received a WKT2 definition"
        assert any("PROJCRS" in w or "BOUNDCRS" in w for w in wkt2)
    finally:
        con.close()


def test_crs_repair_is_idempotent(tmp_path):
    p = V.write_geopackage(_gdf(10), tmp_path / "a.gpkg")
    assert V.repair_geopackage_crs(p) is True
    assert V.repair_geopackage_crs(p) is True       # twice must be safe
    assert len(gpd.read_file(p)) == 10


def test_geopackage_accepts_a_stream_of_chunks(tmp_path):
    """A network too large to hold in memory is written chunk by chunk."""
    whole = _gdf(60)
    p = V.write_geopackage((whole.iloc[i:i + 20] for i in range(0, 60, 20)),
                           tmp_path / "chunked.gpkg")
    assert len(gpd.read_file(p)) == 60


def test_geopackage_replaces_rather_than_appends(tmp_path):
    """A leftover partial from an interrupted run must not gain rows."""
    p = tmp_path / "x.gpkg"
    V.write_geopackage(_gdf(10), p)
    with pytest.raises(FileExistsError):          # never silently replaced
        V.write_geopackage(_gdf(10), p)
    V.write_geopackage(_gdf(10), p, overwrite=True)
    assert len(gpd.read_file(p)) == 10


# ---------------------------------------------------------------------------
# Shapefile

def test_shapefile_roundtrip_and_bools_survive(tmp_path):
    g = _gdf(30)
    out = V.write_shapefile(g, tmp_path / "d.shp")
    assert len(out) == 1
    back = gpd.read_file(out[0])
    assert len(back) == 30
    # shapefile has no boolean type; they are written as int32
    assert set(np.unique(back["isbnd"])) <= {0, 1}


def test_shapefile_splits_into_parts_and_writes_a_name_map(tmp_path):
    g = _gdf(20, extra_fields=3)
    out = V.write_shapefile(g, tmp_path / "wide.shp", max_features=7)
    assert len(out) == 3
    assert sum(len(gpd.read_file(p)) for p in out) == 20
    side = tmp_path / "wide_fields.csv"
    assert side.exists(), "truncated names must be recorded"
    assert "metric_with_a_long_name_000" in side.read_text(encoding="utf-8")


def test_empty_input_raises_rather_than_writing_nothing(tmp_path):
    with pytest.raises(ValueError):
        V.write_shapefile(_gdf(0), tmp_path / "e.shp")


# ---------------------------------------------------------------------------
# File Geodatabase — including the trap

def test_filegdb_roundtrip_preserves_int64_and_crs(tmp_path):
    g = _gdf(25)
    p = V.write_filegdb(g, tmp_path / "divides.gdb")
    assert p.is_dir(), "a geodatabase is a directory, not a file"

    back = gpd.read_file(p, layer="divides")
    assert len(back) == 25
    # TARGET_ARCGIS_VERSION is what keeps these from being demoted to float
    assert np.issubdtype(back["ID"].dtype, np.integer), back["ID"].dtype
    assert np.issubdtype(back["segid"].dtype, np.integer)
    assert back.crs is not None


def test_filegdb_renames_the_reserved_order_field(tmp_path):
    p = V.write_filegdb(_gdf(10), tmp_path / "r.gdb")
    cols = set(gpd.read_file(p, layer="r").columns)
    assert "divorder" in cols, "'order' is reserved and must be renamed"
    assert "order_" not in cols, "GDAL's silent laundering must not happen"


def test_filegdb_replaces_an_existing_directory(tmp_path):
    p = tmp_path / "z.gdb"
    V.write_filegdb(_gdf(10), p)
    with pytest.raises(FileExistsError):          # never silently replaced
        V.write_filegdb(_gdf(10), p)
    V.write_filegdb(_gdf(10), p, overwrite=True)
    assert len(gpd.read_file(p, layer="z")) == 10


def test_the_gdb_trap_is_real_and_our_writer_avoids_it(tmp_path):
    """Pinning the measured trap.

    ``to_file("x.gdb")`` with no driver does NOT write a geodatabase and
    does NOT raise: GeoPandas finds no ``.gdb`` entry in its extension map
    and falls back to ESRI Shapefile, leaving .shp/.dbf/.shx in a directory
    that merely happens to be named .gdb. If a future GeoPandas fixes this,
    the first assertion fails and we can drop the workaround knowingly.
    """
    trap = tmp_path / "trap.gdb"
    _gdf(5).to_file(trap)                       # deliberately no driver=
    assert any(p.suffix == ".shp" for p in trap.iterdir()), (
        "expected the documented Shapefile fallback")

    good = V.write_filegdb(_gdf(5), tmp_path / "good.gdb")
    assert not any(p.suffix == ".shp" for p in good.iterdir())
    assert any(p.name.endswith(".gdbtable") for p in good.iterdir())


# ---------------------------------------------------------------------------
# the policy

def test_policy_defaults_to_geopackage_at_every_size():
    """GeoPackage is the default: open, QGIS-native, one file at any size.
    Shapefile viability is reported by export_plan, not chosen silently."""
    cap = V.shapefile_capacity(106)
    for n in (1_000, cap * 3, cap * 50):
        assert V.choose_format(n, 106) == "gpkg"
    assert V.export_plan(1_000, 10)["shapefile_fits_single_file"]
    assert not V.export_plan(cap * 50, 106)["shapefile_fits_single_file"]


def test_write_vector_substitutes_the_right_extension(tmp_path):
    for fmt, suffix in (("shp", ".shp"), ("gpkg", ".gpkg"), ("gdb", ".gdb")):
        out = V.write_vector(_gdf(8), tmp_path / f"n_{fmt}.dat", format=fmt)
        assert out[0].suffix == suffix, (fmt, out)
        assert out[0].exists()


def test_write_vector_rejects_an_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="unknown format"):
        V.write_vector(_gdf(5), tmp_path / "x", format="kml")


# ---------------------------------------------------------------------------
# export feasibility reporting

def test_export_plan_flags_a_single_file_case():
    p = V.export_plan(50_000, 10)
    assert p["shapefile_fits_single_file"]
    assert p["shapefile_parts"] == 1
    assert "single file" in p["recommendation"]


def test_export_plan_rejects_a_flat_shapefile_for_a_wide_metric_table():
    """The real case: 70M divide pieces with 106 metric fields."""
    p = V.export_plan(70_008_972, 106)
    assert not p["shapefile_fits_single_file"]
    assert p["shapefile_parts"] > 50
    # the attribute table alone would be well over a hundred gigabytes
    assert p["estimated_dbf_bytes"] > 100e9
    assert "Geodatabase" in p["recommendation"]


def test_export_plan_prefers_themes_when_they_fit_in_one_part():
    p = V.export_plan(3_000_000, 106)
    assert p["themed_shapefile_parts"] == 1
    assert p["shapefile_parts"] > 1
    assert "themed" in p["recommendation"]


def test_export_plan_always_allows_a_geodatabase():
    for n in (10, 10_000_000, 500_000_000):
        assert V.export_plan(n, 106)["geodatabase_fits"]


def test_describe_export_mentions_every_format():
    s = V.describe_export(1_000_000, 106)
    for token in ("flat Shapefile", "themed Shapefiles", "File Geodatabase"):
        assert token in s

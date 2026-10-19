# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Every GeoPackage writer in the package must apply the WKT2 CRS repair.

ArcGIS Pro refuses every feature class of a GeoPackage whose custom
projection carries only GDAL's WKT1 text (custom Albers systems, for
example). The one repair is ``pydivide.vectorio.repair_geopackage_crs``.
This test scans the package and
the scripts for GeoPackage writes and fails when a file writes a
GeoPackage without naming the repair, so a new writer cannot forget it.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN = ["pydivide", "scripts", "examples"]
WRITE = re.compile(r"driver=[\"']GPKG[\"']")
REPAIR = re.compile(r"repair_geopackage_crs")


def test_every_geopackage_writer_repairs_the_crs():
    missing = []
    for d in SCAN:
        if not (ROOT / d).is_dir():
            continue
        for f in sorted((ROOT / d).rglob("*.py")):
            text = f.read_text(encoding="utf-8", errors="replace")
            if WRITE.search(text) and not REPAIR.search(text):
                missing.append(f.relative_to(ROOT).as_posix())
    assert not missing, ("GeoPackage written without the WKT2 CRS repair "
                         "(call pydivide.vectorio.repair_geopackage_crs after the write): "
                         f"{missing}")


def test_repair_is_idempotent_and_marks_the_extension(tmp_path):
    import sqlite3

    import geopandas as gpd
    import shapely

    from pydivide.vectorio import repair_geopackage_crs
    crs = ('PROJCS["South America Albers Equal Area Conic_1_4",GEOGCS["SIRGAS 2000",'
           'DATUM["Sistema_de_Referencia_Geocentrico_para_las_AmericaS_2000",'
           'SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],'
           'UNIT["degree",0.0174532925199433]],PROJECTION["Albers_Conic_Equal_Area"],'
           'PARAMETER["latitude_of_center",-21.1210125],PARAMETER["longitude_of_center",-51.6610875],'
           'PARAMETER["standard_parallel_1",-14.14862825],PARAMETER["standard_parallel_2",-28.09339675],'
           'PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]')
    g = gpd.GeoDataFrame({"a": [1]}, geometry=[shapely.LineString([(0, 0), (30, 30)])], crs=crs)
    p = tmp_path / "t.gpkg"
    g.to_file(p, layer="x", driver="GPKG")
    assert repair_geopackage_crs(p) and repair_geopackage_crs(p)
    con = sqlite3.connect(p)
    wkt2 = con.execute("select definition_12_063 from gpkg_spatial_ref_sys "
                       "where srs_id >= 100000").fetchall()
    assert wkt2 and all(w[0].startswith("PROJCRS") for w in wkt2)
    n_ext = con.execute("select count(*) from gpkg_extensions "
                        "where extension_name='gpkg_crs_wkt'").fetchone()[0]
    assert n_ext == 1
    con.close()
    assert len(gpd.read_file(p, layer="x")) == 1
